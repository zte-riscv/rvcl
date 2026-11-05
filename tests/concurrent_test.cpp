/*******************************************************************************
 * Copyright 2026 ZTE Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * tests/concurrent_test.cpp — concurrent stress for the host-parallel stack
 *
 * Permanent smoke driver (PR-5): four concurrent scenarios, each gated so
 * all threads race into the shared path at once, all small enough to stay
 * smoke-fast on x86 and QEMU and cheap enough for the TSan job to amplify.
 *
 * 1. capability concurrent first-call — the first rvcl_get_capability() of
 *    the process is issued by T threads at once; the magic-static once
 *    converges them (same cached pointer + identical content, TSan clean).
 * 2. dispatch concurrent first-call — the first rvcl_dispatch_matmul() of
 *    the process races through the cold cache under the cache mutex; both
 *    a same-desc group (one key) and a different-desc group (varied shapes
 *    that collide into shared cache slots) must all succeed (cache-lock
 *    regression, including the concurrent-eviction path).
 * 3. kernel run() under exclusive M slices — a host scheduler slices M and
 *    calls run() per slice, so one kernel instance and one tensor pack
 *    (including the packed-B workspace configure() reports) are shared by
 *    every concurrent run().
 * 4. semantic concurrent — T threads each run rvcl_matmul() end-to-end on
 *    their own independent tensors at once; every result is bit-exact vs
 *    its own golden (the pack-once end-to-end concurrency endorsement).
 *
 * The kernel-run scenario fetches the kernel through the registry by name
 * (factory bypass, like differential_test), configures it manually with a
 * self-allocated workspace; B is pre-packed once via rvcl_kernel_pack_b()
 * and endorsed with pack.src_b_packed before any thread starts, then T
 * pthreads each run() one exclusive M slice — every run consumes the packed
 * panels read-only (the pack-once contract; the serial in-run packing path
 * stays covered by differential_test).
 *
 * Functional contract: slices own disjoint C rows and the packed workspace
 * is only read after the pre-pack, so the output must stay bit-exact vs the
 * golden on every host, and ThreadSanitizer must report nothing: concurrent
 * run() holds no cross-slice writes into the shared buffer.
 ******************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "golden.hpp"
#include "rvcl/rvcl.h"

static int g_failures;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  ok: %s\n", msg); \
        } else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

/* The generic rvv name exists only on vector builds; non-vector builds
 * register the portable _rvv_c entry instead. Exactly one of the two
 * exists in any build, so resolve the target at runtime (the probe that
 * differential_test uses). */
static const char *probe_rvv_f32_name(void) {
    return rvcl_registry_find_factory("rvcl_matmul_f32_rvv")
            ? "rvcl_matmul_f32_rvv"
            : "rvcl_matmul_f32_rvv_c";
}

/* The registry exposes descriptors by index; find the one whose name
 * matches the factory target. */
static const rvcl_kernel_desc_t *find_desc(const char *name) {
    for (int i = 0, n = rvcl_registry_count(); i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (kd && strcmp(kd->name, name) == 0) return kd;
    }
    return NULL;
}

/* --------------------------------------------------------- slice workers */

/* Ragged M so balanced slicing carries a remainder; small enough to stay
 * smoke-fast, big enough for packed B to span several panels. */
#define CONCURRENT_M 50
#define CONCURRENT_N 40
#define CONCURRENT_K 24
#define CONCURRENT_THREADS 4
#define CONCURRENT_ROUNDS 3

typedef struct {
    pthread_barrier_t *start; /* line every thread up before the burst */
    rvcl_kernel_t *kernel; /* shared instance: state is read-only now */
    const rvcl_tensor_pack_t *pack; /* shared pack incl. packed-B workspace */
    rvcl_work_range_t slice; /* this thread's exclusive M slice */
    rvcl_status_t status; /* run() result, checked after join */
} slice_worker_arg_t;

static void *slice_worker(void *p) {
    slice_worker_arg_t *arg = (slice_worker_arg_t *)p;
    pthread_barrier_wait(arg->start);
    arg->status = rvcl_kernel_run(arg->kernel, arg->pack, &arg->slice);
    return NULL;
}

/* Diff-1 balanced slice [idx] of [parts] over [total]'s M axis; the n/k
 * intervals pass through unchanged. */
static rvcl_work_range_t balanced_slice(
        const rvcl_work_range_t *total, int idx, int parts) {
    const int64_t rows = total->m_end - total->m_begin;
    const int64_t step = rows / parts;
    const int64_t rem = rows % parts;
    const int64_t begin
            = total->m_begin + (int64_t)idx * step + (idx < rem ? idx : rem);
    rvcl_work_range_t out = *total;
    out.m_begin = begin;
    out.m_end = begin + step + (idx < rem ? 1 : 0);
    return out;
}

/* Other workers already wait at the gate when a spawn fails late: abort
 * the whole run instead of deadlocking the barrier. */
static void spawn_worker(
        pthread_t *thr, slice_worker_arg_t *arg, pthread_barrier_t *gate) {
    arg->start = gate;
    const int rc = pthread_create(thr, NULL, slice_worker, arg);
    if (rc == 0) return;
    fprintf(stderr, "  FAIL: pthread_create: %s\n", strerror(rc));
    exit(1);
}

/* ------------------------------------------- capability concurrent first-call
 *
 * The first rvcl_get_capability() of the process is issued by T threads at
 * once, gated so they race into the once. This must run before any other
 * capability call (rvcl_init does not touch capability). The magic-static
 * initializer (C++11 [stmt.dcl]) converges them to a single detection pass,
 * so every thread must observe the identical cached result — the same
 * pointer and identical content — and ThreadSanitizer must stay clean.
 */
typedef struct {
    pthread_barrier_t *start;
    const rvcl_capability_t *cap;
} cap_worker_arg_t;

static void *cap_worker(void *p) {
    cap_worker_arg_t *arg = (cap_worker_arg_t *)p;
    pthread_barrier_wait(arg->start);
    arg->cap = rvcl_get_capability();
    return NULL;
}

static void capability_concurrent_firstcall(void) {
    const int T = CONCURRENT_THREADS;
    pthread_barrier_t gate;
    if (pthread_barrier_init(&gate, NULL, T) != 0) {
        printf("  FAIL: capability gate init (line %d)\n", __LINE__);
        g_failures++;
        return;
    }
    pthread_t th[T];
    cap_worker_arg_t args[T];
    for (int t = 0; t < T; t++) {
        args[t].start = &gate;
        args[t].cap = NULL;
        const int rc = pthread_create(&th[t], NULL, cap_worker, &args[t]);
        if (rc == 0) continue;
        fprintf(stderr, "  FAIL: pthread_create: %s\n", strerror(rc));
        exit(1);
    }
    for (int t = 0; t < T; t++)
        pthread_join(th[t], NULL);
    pthread_barrier_destroy(&gate);

    int null_caps = 0, ptr_mismatch = 0, content_mismatch = 0;
    const rvcl_capability_t *ref = NULL;
    for (int t = 0; t < T; t++) {
        if (!args[t].cap) {
            null_caps++;
            continue;
        }
        if (!ref) ref = args[t].cap;
        if (args[t].cap != ref) ptr_mismatch++;
        if (memcmp(args[t].cap, ref, sizeof(*ref)) != 0) content_mismatch++;
    }
    CHECK(null_caps == 0, "capability: every thread got a result");
    CHECK(ptr_mismatch == 0, "capability: same cached pointer (once)");
    CHECK(content_mismatch == 0,
            "capability: identical content across threads");
}

/* ------------------------------------------------ dispatch concurrent first-call
 *
 * T threads issue their first rvcl_dispatch_matmul() at once, gated so the
 * cold cache races through the insertion path under the cache mutex. Two
 * groups: every thread dispatches the SAME desc (one cache key, concurrent
 * first-call hit on it) and a DIFFERENT desc (varied shapes that hash into
 * overlapping slots, so later first-calls evict earlier ones). All must
 * succeed and hand out a releasable result — a cache-lock regression.
 */
typedef struct {
    pthread_barrier_t *start;
    rvcl_matmul_desc_t desc;
    rvcl_status_t status;
} dispatch_worker_arg_t;

static void *dispatch_worker(void *p) {
    dispatch_worker_arg_t *arg = (dispatch_worker_arg_t *)p;
    pthread_barrier_wait(arg->start);
    rvcl_dispatch_result_t *r = NULL;
    arg->status = rvcl_dispatch_matmul(&arg->desc, &r);
    if (arg->status == RVCL_SUCCESS) rvcl_dispatch_release(r);
    return NULL;
}

static void dispatch_burst(
        const char *label, const rvcl_matmul_desc_t *base, int vary) {
    const int T = CONCURRENT_THREADS;
    pthread_barrier_t gate;
    if (pthread_barrier_init(&gate, NULL, T) != 0) {
        printf("  FAIL: dispatch %s gate init (line %d)\n", label, __LINE__);
        g_failures++;
        return;
    }
    pthread_t th[T];
    dispatch_worker_arg_t args[T];
    for (int t = 0; t < T; t++) {
        args[t].start = &gate;
        args[t].desc = *base;
        if (vary) { /* varied shape per thread; hashes may share a cache slot */
            args[t].desc.m = base->m + (int64_t)(t + 1) * 8;
            args[t].desc.n = base->n + (int64_t)(t + 1) * 4;
            args[t].desc.k = base->k + (int64_t)(t + 1) * 8;
        }
        args[t].status = RVCL_ERROR_INVALID_ARGUMENT;
        const int rc = pthread_create(&th[t], NULL, dispatch_worker, &args[t]);
        if (rc == 0) continue;
        fprintf(stderr, "  FAIL: pthread_create: %s\n", strerror(rc));
        exit(1);
    }
    for (int t = 0; t < T; t++)
        pthread_join(th[t], NULL);
    pthread_barrier_destroy(&gate);

    int bad = 0;
    for (int t = 0; t < T; t++)
        if (args[t].status != RVCL_SUCCESS) bad++;
    char msg[96];
    snprintf(
            msg, sizeof(msg), "dispatch %s: every first call succeeded", label);
    CHECK(bad == 0, msg);
}

/* ------------------------------------------------------- semantic concurrent
 *
 * T threads each run rvcl_matmul() end-to-end on their own independent
 * tensors at once (own dispatch result, own pack-once workspace, own output
 * buffer). Every result must be bit-exact vs its own golden — the pack-once
 * end-to-end path must stay safe when concurrent calls own disjoint buffers.
 */
typedef struct {
    pthread_barrier_t *start;
    int shape_index;
    rvcl_status_t status;
    int64_t mismatches;
} semantic_worker_arg_t;

static void *semantic_worker(void *p) {
    semantic_worker_arg_t *arg = (semantic_worker_arg_t *)p;
    pthread_barrier_wait(arg->start);

    const int m = 24 + (arg->shape_index % 4) * 8;
    const int n = 20 + (arg->shape_index % 3) * 8;
    const int k = 16 + (arg->shape_index % 4) * 8;
    const int64_t elems = (int64_t)m * n;
    const uint32_t seed
            = golden_case_seed(m, n, k) ^ (uint32_t)arg->shape_index;

    float *a = (float *)malloc((size_t)m * k * 4);
    float *b = (float *)malloc((size_t)k * n * 4);
    float *bias = (float *)malloc((size_t)n * 4);
    float *c = (float *)malloc((size_t)elems * 4);
    float *golden = (float *)malloc((size_t)elems * 4);
    if (!a || !b || !bias || !c || !golden) {
        arg->status = RVCL_ERROR_OUT_OF_MEMORY;
        arg->mismatches = 0; /* no compare ran; run_bad reports the failure */
        free(a);
        free(b);
        free(bias);
        free(c);
        free(golden);
        return NULL;
    }
    lcg_fill_f32(a, (size_t)m * k, seed);
    lcg_fill_f32(b, (size_t)k * n, seed ^ 0x9e3779b9u);
    lcg_fill_f32(bias, (size_t)n, seed ^ 0x85ebca6bu);
    golden_f32(a, b, bias, golden, m, n, k);

    rvcl_matmul_desc_t d = {
            .m = m,
            .n = n,
            .k = k,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    arg->status = rvcl_matmul(&d, a, b, bias, c);
    arg->mismatches = 0;
    if (arg->status == RVCL_SUCCESS)
        arg->mismatches = golden_cmp_bitexact(c, golden, elems).mismatches;

    free(a);
    free(b);
    free(bias);
    free(c);
    free(golden);
    return NULL;
}

static void semantic_concurrent(void) {
    const int T = CONCURRENT_THREADS;
    pthread_barrier_t gate;
    if (pthread_barrier_init(&gate, NULL, T) != 0) {
        printf("  FAIL: semantic gate init (line %d)\n", __LINE__);
        g_failures++;
        return;
    }
    pthread_t th[T];
    semantic_worker_arg_t args[T];
    for (int t = 0; t < T; t++) {
        args[t].start = &gate;
        args[t].shape_index = t;
        args[t].status = RVCL_ERROR_INVALID_ARGUMENT;
        args[t].mismatches = 0;
        const int rc = pthread_create(&th[t], NULL, semantic_worker, &args[t]);
        if (rc == 0) continue;
        fprintf(stderr, "  FAIL: pthread_create: %s\n", strerror(rc));
        exit(1);
    }
    for (int t = 0; t < T; t++)
        pthread_join(th[t], NULL);
    pthread_barrier_destroy(&gate);

    int run_bad = 0, match_bad = 0;
    for (int t = 0; t < T; t++) {
        if (args[t].status != RVCL_SUCCESS) run_bad++;
        /* only a successful run performs the bit-exact comparison; a failed
         * run (e.g. OOM) is reported once by run_bad, not twice here */
        if (args[t].status == RVCL_SUCCESS && args[t].mismatches != 0)
            match_bad++;
    }
    char msg[96];
    snprintf(
            msg, sizeof(msg), "semantic: every thread's rvcl_matmul succeeded");
    CHECK(run_bad == 0, msg);
    snprintf(msg, sizeof(msg), "semantic: every result bit-exact vs golden");
    CHECK(match_bad == 0, msg);
}

/* ------------------------------------------------------------------ main */

int main(void) {
    CHECK(rvcl_init() == RVCL_SUCCESS, "rvcl_init succeeds");

    /* Capability first-call must precede any other capability use in the
     * process (rvcl_init does not touch capability), so the concurrent
     * first-call really races into the magic-static once. */
    capability_concurrent_firstcall();

    /* Dispatch first-call must precede the factory-bypass kernel section
     * below (which never calls rvcl_dispatch_matmul), so this is genuinely
     * the cold-cache concurrent first dispatch. Two groups: all threads
     * dispatch the SAME desc (one cache key) and DIFFERENT descs (one slot
     * each). */
    rvcl_matmul_desc_t dbase = {
            .m = 32,
            .n = 24,
            .k = 16,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    dispatch_burst("same-desc", &dbase, 0);
    dispatch_burst("different-desc", &dbase, 1);

    const char *name = probe_rvv_f32_name();
    rvcl_kernel_create_fn create = rvcl_registry_find_factory(name);
    CHECK(create != NULL, "rvv f32 kernel present (generic or portable)");
    const rvcl_kernel_desc_t *desc = find_desc(name);
    CHECK(desc != NULL, "rvv f32 descriptor found");
    if (!create || !desc) {
        printf("CONCURRENT: %d failure(s)\n", g_failures);
        return 1;
    }

    rvcl_matmul_desc_t d = {
            .m = CONCURRENT_M,
            .n = CONCURRENT_N,
            .k = CONCURRENT_K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    rvcl_op_desc_t od = {.matmul = d};

    rvcl_kernel_t *k = NULL;
    CHECK(create(&k, desc) == RVCL_SUCCESS, "factory creates the kernel");
    if (!k) {
        printf("CONCURRENT: %d failure(s)\n", g_failures);
        return 1;
    }

    CHECK(k->vt->validate(&od, rvcl_get_capability()) == RVCL_SUCCESS,
            "kernel validates the case");

    rvcl_workspace_req_t ws[RVCL_MAX_WORKSPACES];
    int num_ws = 0;
    CHECK(k->vt->configure(k, &od, ws, &num_ws) == RVCL_SUCCESS,
            "kernel configures and reports workspace");
    CHECK(num_ws >= 1 && ws[0].size > 0, "packed-B workspace reported");
    if (num_ws < 1 || ws[0].size <= 0) {
        rvcl_kernel_destroy(k);
        printf("CONCURRENT: %d failure(s)\n", g_failures);
        return 1;
    }

    /* Deterministic inputs + golden, same data contract as differential.
     * Buffers are declared before the first assignment so the error gotos
     * below do not cross their initialization (C++ jump rules). */
    const int64_t elems = (int64_t)CONCURRENT_M * CONCURRENT_N;
    const uint32_t seed
            = golden_case_seed(CONCURRENT_M, CONCURRENT_N, CONCURRENT_K);
    void *ws_buf = NULL;
    float *a = NULL, *b = NULL, *bias = NULL, *c = NULL, *golden = NULL;
    int exit_code = 0;

    a = (float *)malloc((size_t)CONCURRENT_M * CONCURRENT_K * 4);
    b = (float *)malloc((size_t)CONCURRENT_K * CONCURRENT_N * 4);
    bias = (float *)malloc((size_t)CONCURRENT_N * 4);
    c = (float *)malloc((size_t)elems * 4);
    golden = (float *)malloc((size_t)elems * 4);
    if (!a || !b || !bias || !c || !golden) {
        fprintf(stderr, "error: out of memory\n");
        exit_code = 2;
        goto out;
    }
    lcg_fill_f32(a, (size_t)CONCURRENT_M * CONCURRENT_K, seed);
    lcg_fill_f32(b, (size_t)CONCURRENT_K * CONCURRENT_N, seed ^ 0x9e3779b9u);
    lcg_fill_f32(bias, (size_t)CONCURRENT_N, seed ^ 0x85ebca6bu);
    golden_f32(a, b, bias, golden, CONCURRENT_M, CONCURRENT_N, CONCURRENT_K);

    /* Everything from here to the rounds loop lives in a nested scope so
     * the allocation-error gotos below never cross an initialization
     * (C++ jump rules). */
    {
        /* Import tensors zero-copy and build the ONE pack every thread
         * runs with: inputs, output and the shared packed-B workspace. */
        rvcl_tensor_desc_t td = {.ndims = 2,
                .dims = {CONCURRENT_M, CONCURRENT_K},
                .strides = {CONCURRENT_K, 1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_t ta, tb, tbias, tc, tws;
        rvcl_tensor_init(&ta, &td, a);

        td.dims[0] = CONCURRENT_K;
        td.dims[1] = CONCURRENT_N;
        td.strides[0] = CONCURRENT_N;
        rvcl_tensor_init(&tb, &td, b);

        td.dims[0] = CONCURRENT_M;
        td.dims[1] = CONCURRENT_N;
        td.strides[0] = CONCURRENT_N;
        rvcl_tensor_init(&tc, &td, c);

        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {CONCURRENT_N},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(&tbias, &bd, bias);

        const size_t ws_align
                = ws[0].alignment > 1 ? (size_t)ws[0].alignment : 8;
        if (posix_memalign(&ws_buf, ws_align, (size_t)ws[0].size) != 0) {
            fprintf(stderr, "error: workspace allocation failed\n");
            exit_code = 2;
            goto out;
        }
        rvcl_tensor_desc_t wd = {.ndims = 1,
                .dims = {ws[0].size},
                .strides = {1},
                .dtype = RVCL_DTYPE_UINT8};
        rvcl_tensor_init(&tws, &wd, ws_buf);

        rvcl_tensor_pack_t pack;
        rvcl_tensor_pack_init(&pack);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_B, &tb);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_BIAS, &tbias);
        rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_AUX_0, &tws);

        /* Pre-pack B once (pack-once): fill the shared packing workspace
         * before any thread exists and endorse it via src_b_packed —
         * every concurrent run() then reads the panels instead of
         * writing the shared buffer. */
        const rvcl_status_t pack_st = rvcl_kernel_pack_b(k, &tb, &tws);
        CHECK(pack_st == RVCL_SUCCESS, "rvcl_kernel_pack_b pre-packs B");
        if (pack_st != RVCL_SUCCESS) goto out;
        pack.src_b_packed = 1;

        /* Concurrent rounds: line the threads up, then everyone runs its
         * own exclusive M slice on the shared kernel + pack together. */
        const rvcl_work_range_t total = rvcl_work_range_full(
                CONCURRENT_M, CONCURRENT_N, CONCURRENT_K);
        for (int round = 0; round < CONCURRENT_ROUNDS; round++) {
            golden_sentinel(c, (size_t)elems * 4);

            pthread_barrier_t gate;
            if (pthread_barrier_init(&gate, NULL, CONCURRENT_THREADS) != 0) {
                printf("  FAIL: start barrier init (line %d)\n", __LINE__);
                g_failures++;
                break;
            }

            pthread_t th[CONCURRENT_THREADS];
            slice_worker_arg_t args[CONCURRENT_THREADS];
            for (int t = 0; t < CONCURRENT_THREADS; t++) {
                args[t].kernel = k;
                args[t].pack = &pack;
                args[t].slice = balanced_slice(&total, t, CONCURRENT_THREADS);
                args[t].status = RVCL_ERROR_INVALID_ARGUMENT;
                spawn_worker(&th[t], &args[t], &gate);
            }
            for (int t = 0; t < CONCURRENT_THREADS; t++)
                pthread_join(th[t], NULL);
            pthread_barrier_destroy(&gate);

            int run_bad = 0;
            for (int t = 0; t < CONCURRENT_THREADS; t++)
                if (args[t].status != RVCL_SUCCESS) run_bad++;
            char msg[64];
            snprintf(msg, sizeof(msg), "round %d: every slice run() succeeded",
                    round);
            CHECK(run_bad == 0, msg);

            golden_cmp_t r = golden_cmp_bitexact(c, golden, elems);
            snprintf(msg, sizeof(msg), "round %d: bit-exact vs golden", round);
            CHECK(r.mismatches == 0, msg);
        }
    }

    /* End-to-end semantic concurrency: T threads each own disjoint tensors
     * through the full rvcl_matmul path (dispatch + pack-once + run). */
    semantic_concurrent();

out:
    free(ws_buf);
    free(a);
    free(b);
    free(bias);
    free(c);
    free(golden);
    rvcl_kernel_destroy(k);
    if (exit_code) return exit_code;

    printf(g_failures ? "CONCURRENT: %d failure(s)\n"
                      : "CONCURRENT: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
