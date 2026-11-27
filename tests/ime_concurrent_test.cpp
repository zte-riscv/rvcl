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
 ******************************************************************************/

/* Concurrency driver for the IME f16 kernel (NFR2): one kernel instance
 * and one tensor pack — including the pre-packed B workspace — are shared
 * by T threads that each run() one exclusive M slice, gated so they burst
 * into run() together. B is pre-packed once with rvcl_kernel_pack_b() and
 * endorsed via pack.src_b_packed before any thread starts, so every run()
 * consumes the shared buffer read-only.
 *
 * Functional contract: the slices own disjoint C rows and each row depends
 * only on the full K, so the concurrent result must be bit-exact identical
 * to a single full-range run of the same kernel (same path, same data).
 * ThreadSanitizer must report nothing: concurrent run() holds no
 * cross-slice writes into the shared buffer.
 *
 * The case is host-gated like the other IME suites: on a host whose run
 * time gate reports the instruction path unavailable (or a build without
 * the hardware-bound entry) the test prints a skip and exits 0. Pin the
 * process to the IME-capable cluster (e.g. taskset -c 8-15) before the
 * first capability query. */

#include "../src/backends/ime/ime_internal.hpp"
#include "golden.hpp"
#include "ime_fixtures.hpp"
#include "rvcl/rvcl.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Ragged M so balanced slicing carries a remainder; a few n/k tiles so
 * the shared packed B spans more than one panel. */
#define IME_CONC_M 50
#define IME_CONC_N 40
#define IME_CONC_K 24
#define IME_CONC_THREADS 4
#define IME_CONC_ROUNDS 3

static const char *const IME_KERNEL_NAME = "rvcl_matmul_f16_ime_1024";

static const rvcl_kernel_desc_t *find_desc(const char *name) {
    for (int i = 0, n = rvcl_registry_count(); i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (kd && strcmp(kd->name, name) == 0) return kd;
    }
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

typedef struct {
    pthread_barrier_t *start;
    rvcl_kernel_t *kernel; /* shared: state is read-only after setup */
    const rvcl_tensor_pack_t *pack; /* shared, B pre-packed */
    rvcl_work_range_t slice;
    rvcl_status_t status;
} slice_worker_arg_t;

static void *slice_worker(void *p) {
    slice_worker_arg_t *arg = (slice_worker_arg_t *)p;
    pthread_barrier_wait(arg->start);
    arg->status = rvcl_kernel_run(arg->kernel, arg->pack, &arg->slice);
    return NULL;
}

int main(void) {
    if (rvcl_init() != RVCL_SUCCESS) {
        printf("IME CONCURRENT: init failed\n");
        return 1;
    }
    const int available = ime_instruction_path_available();
    if (!available || !rvcl_registry_find_factory(IME_KERNEL_NAME)) {
        printf("IME CONCURRENT: skipped (instruction path unavailable)\n");
        return 0;
    }

    rvcl_kernel_create_fn create
            = rvcl_registry_find_factory(IME_KERNEL_NAME);
    const rvcl_kernel_desc_t *kd = find_desc(IME_KERNEL_NAME);
    CHECK(kd != NULL, "hardware-bound descriptor found");
    if (!kd) return 1;

    rvcl_matmul_desc_t d = {
            .m = IME_CONC_M,
            .n = IME_CONC_N,
            .k = IME_CONC_K,
            .dtype_a = RVCL_DTYPE_FP16,
            .dtype_b = RVCL_DTYPE_FP16,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    rvcl_op_desc_t od = {.matmul = d};

    rvcl_kernel_t *k = NULL;
    CHECK(create(&k, kd) == RVCL_SUCCESS, "factory creates the kernel");
    if (!k) return 1;
    CHECK(k->vt->validate(&od, rvcl_get_capability()) == RVCL_SUCCESS,
            "kernel validates the case");

    rvcl_workspace_req_t ws[RVCL_MAX_WORKSPACES];
    int num_ws = 0;
    CHECK(k->vt->configure(k, &od, ws, &num_ws) == RVCL_SUCCESS,
            "kernel configures and reports workspace");
    CHECK(num_ws >= 1 && ws[0].size > 0, "packed-B workspace reported");
    if (num_ws < 1 || ws[0].size <= 0) {
        rvcl_kernel_destroy(k);
        return 1;
    }

    const int64_t elems = (int64_t)IME_CONC_M * IME_CONC_N;
    const uint32_t seed
            = golden_case_seed(IME_CONC_M, IME_CONC_N, IME_CONC_K);

    uint16_t *a = NULL, *b = NULL;
    float *bias = NULL, *c = NULL, *ref = NULL;
    void *ws_buf = NULL;
    int exit_code = 0;

    a = (uint16_t *)malloc((size_t)IME_CONC_M * IME_CONC_K * 2);
    b = (uint16_t *)malloc((size_t)IME_CONC_K * IME_CONC_N * 2);
    bias = (float *)malloc((size_t)IME_CONC_N * 4);
    c = (float *)malloc((size_t)elems * 4);
    ref = (float *)malloc((size_t)elems * 4);
    if (!a || !b || !bias || !c || !ref) {
        fprintf(stderr, "error: out of memory\n");
        exit_code = 2;
        goto out;
    }
    ime_fill_f16_normal(a, (size_t)IME_CONC_M * IME_CONC_K, seed);
    ime_fill_f16_normal(b, (size_t)IME_CONC_K * IME_CONC_N, seed ^ 0x9e3779b9u);
    ime_fill_bias_normal(bias, (size_t)IME_CONC_N, seed ^ 0x85ebca6bu);

    {
        rvcl_tensor_desc_t td = {.ndims = 2,
                .dims = {IME_CONC_M, IME_CONC_K},
                .strides = {IME_CONC_K, 1},
                .dtype = RVCL_DTYPE_FP16};
        rvcl_tensor_t ta, tb, tbias, tc, tws;
        rvcl_tensor_init(&ta, &td, a);

        td.dims[0] = IME_CONC_K;
        td.dims[1] = IME_CONC_N;
        td.strides[0] = IME_CONC_N;
        rvcl_tensor_init(&tb, &td, b);

        td.dims[0] = IME_CONC_M;
        td.dims[1] = IME_CONC_N;
        td.strides[0] = IME_CONC_N;
        td.dtype = RVCL_DTYPE_FP32;
        rvcl_tensor_init(&tc, &td, c);

        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {IME_CONC_N},
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

        /* Pre-pack B once; every concurrent run() reads it. */
        const rvcl_status_t pack_st = rvcl_kernel_pack_b(k, &tb, &tws);
        CHECK(pack_st == RVCL_SUCCESS, "rvcl_kernel_pack_b pre-packs B");
        if (pack_st != RVCL_SUCCESS) goto out;
        pack.src_b_packed = 1;

        /* Sequential reference: one full-range run of the same kernel. The
         * concurrent slices must reproduce it bit for bit. */
        const rvcl_work_range_t total
                = rvcl_work_range_full(IME_CONC_M, IME_CONC_N, IME_CONC_K);
        CHECK(rvcl_kernel_run(k, &pack, &total) == RVCL_SUCCESS,
                "reference full-range run succeeds");
        memcpy(ref, c, (size_t)elems * 4);

        for (int round = 0; round < IME_CONC_ROUNDS; round++) {
            golden_sentinel(c, (size_t)elems * 4);

            pthread_barrier_t gate;
            if (pthread_barrier_init(&gate, NULL, IME_CONC_THREADS) != 0) {
                printf("  FAIL: start barrier init (line %d)\n", __LINE__);
                g_failures++;
                break;
            }
            pthread_t th[IME_CONC_THREADS];
            slice_worker_arg_t args[IME_CONC_THREADS];
            for (int t = 0; t < IME_CONC_THREADS; t++) {
                args[t].start = &gate;
                args[t].kernel = k;
                args[t].pack = &pack;
                args[t].slice = balanced_slice(&total, t, IME_CONC_THREADS);
                args[t].status = RVCL_ERROR_INVALID_ARGUMENT;
                const int rc = pthread_create(
                        &th[t], NULL, slice_worker, &args[t]);
                if (rc == 0) continue;
                fprintf(stderr, "  FAIL: pthread_create: %s\n", strerror(rc));
                exit(1);
            }
            for (int t = 0; t < IME_CONC_THREADS; t++)
                pthread_join(th[t], NULL);
            pthread_barrier_destroy(&gate);

            int run_bad = 0;
            for (int t = 0; t < IME_CONC_THREADS; t++)
                if (args[t].status != RVCL_SUCCESS) run_bad++;
            char msg[64];
            snprintf(msg, sizeof(msg), "round %d: every slice run() succeeded",
                    round);
            CHECK(run_bad == 0, msg);

            snprintf(msg, sizeof(msg),
                    "round %d: bit-exact vs full-range reference", round);
            CHECK(memcmp(c, ref, (size_t)elems * 4) == 0, msg);
        }
    }

out:
    free(ws_buf);
    free(a);
    free(b);
    free(bias);
    free(c);
    free(ref);
    rvcl_kernel_destroy(k);
    if (exit_code) return exit_code;

    printf(g_failures ? "IME CONCURRENT: %d failure(s)\n"
                      : "IME CONCURRENT: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
