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
 * unit_capability.cpp — capability unit tests
 *
 * Covers: host detection output and has/get query semantics, snapshot
 * caching and stability, concurrent first-call consistency, the
 * human-readable summary contract, and the VLEN-tier conversion table.
 * Host-dependent expectations branch on what the running snapshot
 * actually reports instead of hardcoding a host, so the binary is
 * runnable everywhere; an exact VLEN may additionally be pinned with
 * RVCL_EXPECT_VLEN=<bits>.
 *
 * The --inject mode (own ctest entry, fresh process) exercises the
 * test-only snapshot override facility: synthetic snapshots, summary
 * regeneration, host restore, and the dispatch-cache invalidation that
 * accompanies every override/restore.
 ******************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "golden.hpp"
#include "rvcl/rvcl.h"

#include "../src/runtime/capability_internal.h"
#include "../src/runtime/cost_model.h"

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

static int is_pow2(uint32_t v) {
    return v != 0 && (v & (v - 1)) == 0;
}

/* ------------------------------------------- concurrent first call (first!) */

#define CONC_THREADS 16
#define CONC_CALLS 64

typedef struct {
    pthread_barrier_t *barrier;
    const rvcl_capability_t *snap;
    rvcl_capability_t copy;
    int pointer_mismatch;
} conc_arg_t;

static void *conc_worker(void *p) {
    conc_arg_t *arg = (conc_arg_t *)p;
    pthread_barrier_wait(arg->barrier);
    for (int i = 0; i < CONC_CALLS; i++) {
        const rvcl_capability_t *c = rvcl_get_capability();
        if (!arg->snap)
            arg->snap = c;
        else if (c != arg->snap)
            arg->pointer_mismatch = 1;
    }
    arg->copy = *arg->snap;
    return NULL;
}

/* Runs before anything else touches capability: the point is that the
 * *first* detection happens under concurrency. Meaningful as a race
 * check under the sanitizer build; plain builds check consistency. */
static void case_concurrent_first_call(void) {
    printf("concurrent first-call consistency:\n");
    pthread_barrier_t barrier;
    pthread_barrier_init(&barrier, NULL, CONC_THREADS);
    conc_arg_t args[CONC_THREADS];
    memset(args, 0, sizeof(args));
    for (int i = 0; i < CONC_THREADS; i++)
        args[i].barrier = &barrier;

    pthread_t tid[CONC_THREADS];
    for (int i = 0; i < CONC_THREADS; i++)
        pthread_create(&tid[i], NULL, conc_worker, &args[i]);
    for (int i = 0; i < CONC_THREADS; i++)
        pthread_join(tid[i], NULL);
    pthread_barrier_destroy(&barrier);

    int mismatches = 0;
    for (int i = 0; i < CONC_THREADS; i++) {
        if (args[i].pointer_mismatch) mismatches++;
        if (args[i].snap != args[0].snap) mismatches++;
        if (memcmp(&args[i].copy, &args[0].copy, sizeof(rvcl_capability_t))
                != 0)
            mismatches++;
    }
    CHECK(mismatches == 0, "all threads observe one stable snapshot");
}

/* -------------------------------------------------------- host detection */

static void case_host_features(void) {
    printf("host detection output:\n");
    const rvcl_capability_t *cap = rvcl_get_capability();
    const int has_rv64 = rvcl_capability_has(cap, RVCL_FEATURE_RV64);
    const int has_rvv = rvcl_capability_has(cap, RVCL_FEATURE_RVV);

    if (has_rv64 && has_rvv) {
        const rvcl_feature_t *rvv = rvcl_capability_get(cap, RVCL_FEATURE_RVV);
        CHECK(rvv != NULL, "rvv detail is queryable");
        // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
        CHECK(rvv->value == 1 || (rvv->value >= 128 && is_pow2(rvv->value)),
                "rvv value is unknown(1) or a power-of-two width >= 128");
        const char *pin = getenv("RVCL_EXPECT_VLEN");
        if (pin && *pin) {
            char msg[64];
            const unsigned long want = strtoul(pin, NULL, 0);
            snprintf(msg, sizeof(msg), "rvv vlen equals pinned expectation %lu",
                    want);
            CHECK(rvv->value == want, msg);
        }
    } else if (has_rv64) {
        /* RISC-V build without runtime RVV confirmation: the snapshot
         * legitimately degrades to the compile-time view (no RVV). */
        CHECK(rvcl_capability_get(cap, RVCL_FEATURE_RVV) == NULL,
                "absent rvv has no detail entry");
        CHECK(rvcl_capability_vlen_tier(cap) == 0,
                "snapshot without rvv converts to no tier");
    } else {
        printf("  skip: rv64 not reported (non-riscv host)\n");
    }
}

static void case_vector_build_fp16(void) {
    printf("vector-build fp16 feature:\n");
    /* The f16 rvv entry exists only when the build compiled with the
     * vector extension and Zvfh/Zvfhmin macros visible — the same
     * macros that set the compile-time FP16 bit. */
    if (rvcl_registry_find_factory("rvcl_matmul_f16_rvv") != NULL) {
        CHECK(rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_FP16),
                "vector build reports the fp16 feature");
    } else {
        printf("  skip: no vector-build f16 entry (portable or baseline "
               "build)\n");
    }
}

static void case_zero_feature_host(void) {
    printf("zero-feature host:\n");
    const rvcl_capability_t *cap = rvcl_get_capability();
    if (rvcl_capability_has(cap, RVCL_FEATURE_RV64)) {
        printf("  skip: riscv hosts always report rv64\n");
        return;
    }
    CHECK(cap->num_features == 0, "non-riscv host detects zero features");
    static const rvcl_feature_id_t known[] = {
            RVCL_FEATURE_RV64,
            RVCL_FEATURE_RVV,
            RVCL_FEATURE_IME,
            RVCL_FEATURE_VME,
            RVCL_FEATURE_AME,
            RVCL_FEATURE_FP32,
            RVCL_FEATURE_FP16,
            RVCL_FEATURE_BF16,
            RVCL_FEATURE_INT8,
            RVCL_FEATURE_INT4,
    };
    int all_absent = 1;
    for (size_t i = 0; i < sizeof(known) / sizeof(known[0]); i++)
        if (rvcl_capability_has(cap, known[i])
                || rvcl_capability_get(cap, known[i]) != NULL)
            all_absent = 0;
    CHECK(all_absent, "every known feature id is absent");
    CHECK(rvcl_capability_vlen_tier(cap) == 0,
            "zero-feature snapshot has no tier");

    /* The library still serves f32 through the featureless fallback. */
    enum { M = 8, N = 8, K = 16 };
    float a[M * K], b[K * N], c[M * N], ref[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = (float)((i * 13) % 29) / 29.0f;
    for (int i = 0; i < K * N; i++)
        b[i] = (float)((i * 7) % 31) / 31.0f;
    golden_f32(a, b, NULL, ref, M, N, K);

    rvcl_matmul_desc_t d = {
            .m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
    memset(c, 0, sizeof(c));
    CHECK(rvcl_matmul(&d, a, b, NULL, c) == RVCL_SUCCESS,
            "f32 matmul runs on a zero-feature host");
    golden_cmp_t cmp = golden_cmp_bitexact(c, ref, (int64_t)M * N);
    CHECK(cmp.mismatches == 0, "f32 output is bit-identical to the golden");
}

/* --------------------------------------------------- caching and queries */

static void case_snapshot_stability(void) {
    printf("snapshot caching and stability:\n");
    const rvcl_capability_t *first = rvcl_get_capability();
    rvcl_capability_t baseline_copy = *first;
    int same = 1;
    for (int i = 0; i < 32; i++)
        if (rvcl_get_capability() != first) same = 0;
    CHECK(same, "repeated calls return the same cached pointer");

    const rvcl_capability_t *again = rvcl_get_capability();
    CHECK(again->num_features == baseline_copy.num_features
                    && memcmp(again->features, baseline_copy.features,
                               sizeof(baseline_copy.features))
                            == 0
                    && strcmp(again->description, baseline_copy.description)
                            == 0,
            "snapshot content is stable");
}

static void case_description(void) {
    printf("human-readable summary:\n");
    const rvcl_capability_t *cap = rvcl_get_capability();
    CHECK(cap->description[0] != '\0', "description is non-empty");
    char head[64];
    snprintf(head, sizeof(head), "rvcl-capability: %d feature(s) detected",
            cap->num_features);
    CHECK(strncmp(cap->description, head, strlen(head)) == 0,
            "description carries the feature count");

    const rvcl_feature_t *rvv = rvcl_capability_get(cap, RVCL_FEATURE_RVV);
    if (rvv && rvv->value > 1) {
        char vlen[32];
        snprintf(vlen, sizeof(vlen), "rvv vlen=%u", rvv->value);
        CHECK(strstr(cap->description, vlen) != NULL,
                "description carries the vlen value");
    } else if (rvv) {
        CHECK(strstr(cap->description, "rvv vlen=unknown") != NULL,
                "description marks an unreadable vlen");
    }
}

static void case_query_semantics(void) {
    printf("has/get query semantics:\n");
    const rvcl_capability_t *cap = rvcl_get_capability();
    int present_ok = 1;
    for (int i = 0; i < cap->num_features; i++) {
        const rvcl_feature_id_t id = cap->features[i].id;
        const rvcl_feature_t *f = rvcl_capability_get(cap, id);
        if (!rvcl_capability_has(cap, id) || !f || f->id != id || f->value < 1)
            present_ok = 0;
    }
    CHECK(present_ok, "every present feature passes has() and get()");

    /* Ids no detection layer ever reports must stay absent on any host. */
    static const rvcl_feature_id_t never[] = {
            RVCL_FEATURE_IME,
            RVCL_FEATURE_VME,
            RVCL_FEATURE_AME,
            RVCL_FEATURE_FP32,
            RVCL_FEATURE_INT8,
            RVCL_FEATURE_INT4,
    };
    int absent_ok = 1;
    for (size_t i = 0; i < sizeof(never) / sizeof(never[0]); i++)
        if (rvcl_capability_has(cap, never[i])
                || rvcl_capability_get(cap, never[i]) != NULL)
            absent_ok = 0;
    CHECK(absent_ok, "undetected feature ids report absent");

    CHECK(!rvcl_capability_has(NULL, RVCL_FEATURE_RV64), "has(NULL) is false");
    CHECK(rvcl_capability_get(NULL, RVCL_FEATURE_RV64) == NULL,
            "get(NULL) is NULL");
}

static void case_vlen_tier_table(void) {
    printf("vlen tier conversion table:\n");
    static const struct {
        uint32_t vlen;
        int tier;
    } table[] = {
            {1, 0},
            {64, 0},
            {127, 0},
            {128, 128},
            {129, 128},
            {192, 128},
            {200, 128},
            {255, 128},
            {256, 256},
            {300, 256},
            {384, 256},
            {511, 256},
            {512, 512},
            {1000, 512},
            {1023, 512},
            {1024, 1024},
            {2048, 1024},
            {4096, 1024},
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        rvcl_capability_t cap;
        memset(&cap, 0, sizeof(cap));
        cap.features[0].id = RVCL_FEATURE_RVV;
        cap.features[0].value = table[i].vlen;
        cap.num_features = 1;
        if (rvcl_capability_vlen_tier(&cap) != table[i].tier) {
            printf("  vlen=%u: expected tier %d\n", table[i].vlen,
                    table[i].tier);
            ok = 0;
        }
    }
    CHECK(ok, "tier = largest tier <= vlen; below 128 (or unknown) is 0");

    rvcl_capability_t none;
    memset(&none, 0, sizeof(none));
    CHECK(rvcl_capability_vlen_tier(&none) == 0,
            "snapshot without rvv -> no tier");
    CHECK(rvcl_capability_vlen_tier(NULL) == 0, "null snapshot -> no tier");
}

/* ------------------------------------------------- snapshot override (--inject) */

static rvcl_capability_t make_snapshot(
        int with_rv64, int with_rvv, uint32_t vlen) {
    rvcl_capability_t cap;
    memset(&cap, 0, sizeof(cap));
    if (with_rv64) {
        cap.features[cap.num_features].id = RVCL_FEATURE_RV64;
        cap.features[cap.num_features].value = 1;
        cap.num_features++;
    }
    if (with_rvv) {
        cap.features[cap.num_features].id = RVCL_FEATURE_RVV;
        cap.features[cap.num_features].value = vlen;
        cap.num_features++;
    }
    return cap;
}

static void case_inject_tier_scan(void) {
    printf("injected vlen tier scan:\n");
    static const struct {
        uint32_t vlen;
        int tier;
    } table[] = {
            {64, 0},
            {127, 0},
            {128, 128},
            {129, 128},
            {200, 128},
            {256, 256},
            {300, 256},
            {512, 512},
            {1000, 512},
            {1024, 1024},
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++) {
        rvcl_capability_t snap = make_snapshot(1, 1, table[i].vlen);
        rvcl_capability_test_override(&snap);
        const rvcl_capability_t *cap = rvcl_get_capability();
        const rvcl_feature_t *rvv = rvcl_capability_get(cap, RVCL_FEATURE_RVV);
        if (!rvcl_capability_has(cap, RVCL_FEATURE_RV64) || !rvv
                || rvv->value != table[i].vlen
                || rvcl_capability_vlen_tier(cap) != table[i].tier) {
            printf("  vlen=%u: expected tier %d\n", table[i].vlen,
                    table[i].tier);
            ok = 0;
        }
    }
    CHECK(ok, "override serves the injected snapshot and converts its tier");
    rvcl_capability_test_override(NULL);
}

static void case_inject_description(void) {
    printf("injected snapshot summary regeneration:\n");
    rvcl_capability_t snap = make_snapshot(1, 1, 256);
    snprintf(snap.description, sizeof(snap.description), "%s",
            "ZZ-test-supplied-summary");
    rvcl_capability_test_override(&snap);
    const rvcl_capability_t *cap = rvcl_get_capability();
    CHECK(strstr(cap->description, "ZZ-test-supplied-summary") == NULL,
            "caller-supplied description is discarded");
    CHECK(strstr(cap->description, "2 feature(s) detected") != NULL,
            "summary is regenerated with the feature count");
    CHECK(strstr(cap->description, "vlen=256") != NULL,
            "summary is regenerated with the vlen value");
    rvcl_capability_test_override(NULL);
}

static void case_inject_zero_feature(void) {
    printf("injected empty snapshot:\n");
    rvcl_capability_t snap = make_snapshot(0, 0, 0);
    rvcl_capability_test_override(&snap);
    const rvcl_capability_t *cap = rvcl_get_capability();
    CHECK(cap->num_features == 0, "override serves an empty snapshot");
    CHECK(rvcl_capability_vlen_tier(cap) == 0, "empty snapshot has no tier");
    rvcl_capability_test_override(NULL);
}

static void case_inject_dispatch_cache(void) {
    printf("override invalidates the dispatch cache:\n");
    rvcl_matmul_desc_t d = {
            .m = 16,
            .n = 16,
            .k = 16,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
    rvcl_dispatch_result_t *r = NULL;

    const int misses0 = rvcl_dispatch_cache_misses();
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "first dispatch succeeds");
    CHECK(rvcl_dispatch_cache_misses() > misses0, "first dispatch is a miss");
    rvcl_dispatch_release(r);
    r = NULL;

    const int hits0 = rvcl_dispatch_cache_hits();
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "repeat dispatch succeeds");
    CHECK(rvcl_dispatch_cache_hits() > hits0, "repeat dispatch hits the cache");
    rvcl_dispatch_release(r);
    r = NULL;

    /* The override changes a selection input the dispatch key does not
     * encode, so the library must drop its cached selection. */
    const int misses1 = rvcl_dispatch_cache_misses();
    rvcl_capability_t snap = make_snapshot(1, 1, 128);
    rvcl_capability_test_override(&snap);
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch after override succeeds");
    CHECK(rvcl_dispatch_cache_misses() > misses1,
            "override invalidates the dispatch cache");
    rvcl_dispatch_release(r);
    r = NULL;

    const int misses2 = rvcl_dispatch_cache_misses();
    rvcl_capability_test_override(NULL);
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch after restore succeeds");
    CHECK(rvcl_dispatch_cache_misses() > misses2,
            "restore invalidates the dispatch cache");
    rvcl_dispatch_release(r);
}

static void case_inject_restore(
        const rvcl_capability_t *host_ptr, const rvcl_capability_t *host_copy) {
    printf("host snapshot restore:\n");
    rvcl_capability_test_override(NULL);
    const rvcl_capability_t *cap = rvcl_get_capability();
    CHECK(cap == host_ptr, "restore returns the original host snapshot");
    CHECK(cap->num_features == host_copy->num_features
                    && memcmp(cap->features, host_copy->features,
                               sizeof(host_copy->features))
                            == 0
                    && strcmp(cap->description, host_copy->description) == 0,
            "restored host snapshot is unchanged");

    rvcl_capability_t snap = make_snapshot(1, 1, 512);
    rvcl_capability_test_override(&snap);
    CHECK(rvcl_capability_vlen_tier(rvcl_get_capability()) == 512,
            "override applies again after a restore");
    rvcl_capability_test_override(NULL);
}

/* ------------------------------------------- vendor probe registration
 *
 * Fresh-process modes (own ctest entries): the capability snapshot is built
 * at most once per process and the probe window closes permanently after
 * the first query, so each vendor-scenario runs in its own invocation.
 * A minimal out-of-tree kernel (INT8 tuple, no builtin occupancy) requiring
 * a vendor feature is registered to prove the probe's write channel changes
 * dispatch visibility — the way an actual plugin would.
 */

#define VENDOR_FEATURE ((rvcl_feature_id_t)0x1001)

static rvcl_status_t vendor_probe_1001(rvcl_capability_t *cap) {
    if (cap->num_features < RVCL_MAX_FEATURES) {
        /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately
         * not an enumerator of rvcl_feature_id_t. */
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        cap->features[cap->num_features].id = VENDOR_FEATURE;
        cap->features[cap->num_features].value = 1;
        cap->num_features++;
    }
    return RVCL_SUCCESS;
}

/* A do-nothing vendor kernel: the tests exercise registration, feature
 * visibility and dispatch selection, not arithmetic. */
static rvcl_status_t vk_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return RVCL_SUCCESS;
}

static rvcl_status_t vk_configure(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
        rvcl_workspace_req_t *ws_req, int *num_ws) {
    (void)k;
    (void)desc;
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t vk_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *range) {
    (void)k;
    (void)pack;
    (void)range;
    return RVCL_SUCCESS;
}

static void vk_destroy(rvcl_kernel_t *k) {
    free(k);
}

static const rvcl_kernel_vtable_t vk_vtable = {
        .validate = vk_validate,
        .configure = vk_configure,
        .pack_a = NULL,
        .pack_b = NULL,
        .run = vk_run,
        .destroy = vk_destroy,
};

static rvcl_status_t vk_create(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    rvcl_kernel_t *k = (rvcl_kernel_t *)calloc(1, sizeof(rvcl_kernel_t));
    if (!k) return RVCL_ERROR_OUT_OF_MEMORY;
    k->vt = &vk_vtable;
    k->desc = *desc;
    *out = k;
    return RVCL_SUCCESS;
}

/* Registers the vendor kernel on the INT8 tuple (unoccupied by any builtin),
 * requiring the vendor feature. */
static rvcl_status_t register_vendor_kernel(const char *name) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = RVCL_DTYPE_INT8;
    d.dtype_b = RVCL_DTYPE_INT8;
    d.dtype_acc = RVCL_DTYPE_INT32;
    d.dtype_c = RVCL_DTYPE_INT32;
    d.quant_support = RVCL_QUANT_NONE;
    /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately not an
     * enumerator of rvcl_feature_id_t. */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    d.required_features[0] = VENDOR_FEATURE;
    d.num_required_features = 1;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_kernel_cost_zero;
    return rvcl_register_kernel(&d, vk_create);
}

static rvcl_matmul_desc_t vendor_request(void) {
    return (rvcl_matmul_desc_t) {
            .m = 16,
            .n = 16,
            .k = 16,
            .dtype_a = RVCL_DTYPE_INT8,
            .dtype_b = RVCL_DTYPE_INT8,
            .dtype_acc = RVCL_DTYPE_INT32,
            .dtype_c = RVCL_DTYPE_INT32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
}

/* U-P2-1a: probe registered -> snapshot carries the vendor feature and the
 * required-feature kernel dispatches to the vendor entry. */
static int vendor_probe_case(void) {
    printf("vendor probe: registered -> snapshot + dispatch\n");
    if (rvcl_capability_register_vendor_probe(vendor_probe_1001)
            != RVCL_SUCCESS) {
        printf("  FAIL: probe registration (line %d)\n", __LINE__);
        return 1;
    }
    CHECK(rvcl_init() == RVCL_SUCCESS, "init succeeds");

    const rvcl_capability_t *cap = rvcl_get_capability();
    /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately not an
     * enumerator of rvcl_feature_id_t. */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(rvcl_capability_has(cap, VENDOR_FEATURE),
            "snapshot carries the vendor feature declared by the probe");
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(rvcl_capability_get(cap, VENDOR_FEATURE) != NULL,
            "vendor feature detail is queryable");

    CHECK(register_vendor_kernel("t_vendor_probe") == RVCL_SUCCESS,
            "vendor kernel registered");
    rvcl_dispatch_cache_clear();
    rvcl_dispatch_result_t *r = NULL;
    const rvcl_matmul_desc_t d = vendor_request();
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch on the vendor tuple succeeds");
    if (r) {
        CHECK(strcmp(rvcl_dispatch_kernel_name(r), "t_vendor_probe") == 0,
                "dispatch selects the vendor kernel");
        rvcl_dispatch_release(r);
    } else {
        CHECK(0, "dispatch produced a result object");
    }
    printf(g_failures ? "VENDOR-PROBE: %d failure(s)\n"
                      : "VENDOR-PROBE: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

/* U-P2-1b: no probe registered -> the same required-feature kernel is
 * filtered at stage 3 and the tuple dispatches UNSUPPORTED. */
static int vendor_probe_noreg_case(void) {
    printf("vendor probe: not registered -> UNSUPPORTED\n");
    CHECK(rvcl_init() == RVCL_SUCCESS, "init succeeds");
    /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately not an
     * enumerator of rvcl_feature_id_t. */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(!rvcl_capability_has(rvcl_get_capability(), VENDOR_FEATURE),
            "snapshot has no vendor feature");

    CHECK(register_vendor_kernel("t_vendor_noreg") == RVCL_SUCCESS,
            "vendor kernel registered");
    rvcl_dispatch_cache_clear();
    rvcl_dispatch_result_t *r = NULL;
    const rvcl_matmul_desc_t d = vendor_request();
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_UNSUPPORTED,
            "vendor tuple without a snapshot feature -> UNSUPPORTED");
    CHECK(r == NULL, "no result leaked on an unsupported request");
    printf(g_failures ? "VENDOR-PROBE-NOREG: %d failure(s)\n"
                      : "VENDOR-PROBE-NOREG: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

/* U-P2-2: registering a probe after the first capability query (window
 * closed) returns RVCL_ERROR_ALREADY_INITIALIZED instead of silently
 * ignoring it. */
static int vendor_probe_late_case(void) {
    printf("vendor probe: post-query registration rejected\n");
    CHECK(rvcl_init() == RVCL_SUCCESS, "init succeeds");
    (void)rvcl_get_capability(); /* first query closes the window */
    CHECK(rvcl_capability_register_vendor_probe(vendor_probe_1001)
                    == RVCL_ERROR_ALREADY_INITIALIZED,
            "late probe registration returns ALREADY_INITIALIZED");
    printf(g_failures ? "VENDOR-PROBE-LATE: %d failure(s)\n"
                      : "VENDOR-PROBE-LATE: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

/* ------------------------------------------------------- probe sanitization
 *
 * U-P2-4: whatever a probe appends is filtered before it reaches the
 * snapshot. Four malformed/undisciplined forms are exercised, all registered
 * before the first query so they run during the single snapshot build:
 *   - a probe appending a standard feature id (dropped: only vendor ids
 *     survive);
 *   - a probe appending a duplicate vendor id (deduped: append-only);
 *   - a probe returning non-SUCCESS (its whole increment discarded, the
 *     remaining probes still run);
 *   - a probe flooding distinct vendor ids up to the table capacity (the
 *     snapshot is capped at RVCL_MAX_FEATURES; standard features survive).
 */

static rvcl_status_t probe_std(rvcl_capability_t *cap) {
    /* a standard-range id must be dropped by the sanitizer */
    if (cap->num_features < RVCL_MAX_FEATURES) {
        cap->features[cap->num_features].id = RVCL_FEATURE_INT8;
        cap->features[cap->num_features].value = 1;
        cap->num_features++;
    }
    return RVCL_SUCCESS;
}

static rvcl_status_t probe_dup(rvcl_capability_t *cap) {
    /* the same vendor id twice: append-only dedup keeps one */
    for (int i = 0; i < 2 && cap->num_features < RVCL_MAX_FEATURES; i++) {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        cap->features[cap->num_features].id = (rvcl_feature_id_t)0x1001;
        cap->features[cap->num_features].value = 1;
        cap->num_features++;
    }
    return RVCL_SUCCESS;
}

static rvcl_status_t probe_fail(rvcl_capability_t *cap) {
    if (cap->num_features < RVCL_MAX_FEATURES) {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        cap->features[cap->num_features].id = (rvcl_feature_id_t)0x1003;
        cap->features[cap->num_features].value = 1;
        cap->num_features++;
    }
    return RVCL_UNSUPPORTED; /* non-SUCCESS: drop this probe's increment */
}

static rvcl_status_t probe_flood(rvcl_capability_t *cap) {
    /* distinct vendor ids until the table is full; the sanitizer caps the
     * snapshot at RVCL_MAX_FEATURES without ever clobbering standard ids. */
    unsigned id = 0x1100;
    while (cap->num_features < RVCL_MAX_FEATURES) {
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        cap->features[cap->num_features].id = (rvcl_feature_id_t)id;
        cap->features[cap->num_features].value = 1;
        cap->num_features++;
        id++;
    }
    return RVCL_SUCCESS;
}

static int vendor_probe_sanitize_case(void) {
    printf("vendor probe: malformed increment sanitization\n");
    if (rvcl_capability_register_vendor_probe(probe_std) != RVCL_SUCCESS
            || rvcl_capability_register_vendor_probe(probe_dup) != RVCL_SUCCESS
            || rvcl_capability_register_vendor_probe(probe_fail) != RVCL_SUCCESS
            || rvcl_capability_register_vendor_probe(probe_flood)
                    != RVCL_SUCCESS) {
        printf("  FAIL: probe registration (line %d)\n", __LINE__);
        return 1;
    }
    CHECK(rvcl_init() == RVCL_SUCCESS, "init succeeds");
    const rvcl_capability_t *cap = rvcl_get_capability();

    CHECK(cap->num_features <= RVCL_MAX_FEATURES,
            "snapshot never exceeds the feature table capacity");
    /* vendor-range ids (>= RVCL_FEATURE_VENDOR_FIRST) are deliberately not
     * enumerators of rvcl_feature_id_t. */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(rvcl_capability_has(cap, (rvcl_feature_id_t)0x1001),
            "duplicate vendor id deduped to one present entry");
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(!rvcl_capability_has(cap, (rvcl_feature_id_t)0x1003),
            "non-SUCCESS probe's increment fully discarded");
    CHECK(!rvcl_capability_has(cap, RVCL_FEATURE_INT8),
            "standard-range id appended by a probe is dropped");
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(rvcl_capability_has(cap, (rvcl_feature_id_t)0x1100),
            "flood probe's first vendor id survives (probing continued)");

    /* The flood fills the table to capacity with vendor ids, but must
     * never displace a host-detected built-in standard feature: assert the
     * built-ins the host carries (RV64 on any RV64 target, RVV on a
     * vector build) are still present afterward. */
#if defined(__riscv) && defined(__riscv_xlen) && (__riscv_xlen == 64)
    CHECK(rvcl_capability_has(cap, RVCL_FEATURE_RV64),
            "built-in RV64 survives the flood");
#endif
#if defined(__riscv_vector)
    CHECK(rvcl_capability_has(cap, RVCL_FEATURE_RVV),
            "built-in RVV survives the flood");
#endif
    printf(g_failures ? "VENDOR-PROBE-SANITIZE: %d failure(s)\n"
                      : "VENDOR-PROBE-SANITIZE: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

/* --------------------------------------------------- snapshot assembly dedup
 *
 * The served snapshot carries at most one entry per feature id, no
 * matter which channel appended: the probe channel self-sanitizes (the
 * case above), and the assembly pass extends the same rule to every
 * channel together. Exercised through the test-override install, itself
 * a served-snapshot path: two channels wrote the same vendor id (an
 * informative value first, a bare presence copy second), and the pass
 * must collapse them keeping the first entry while the survivors keep
 * their assembly order.
 */
static int snapshot_dedup_case(void) {
    printf("snapshot assembly: duplicate ids deduplicated, first wins\n");
    rvcl_capability_t cap;
    memset(&cap, 0, sizeof(cap));
    cap.features[0].id = RVCL_FEATURE_RV64;
    cap.features[0].value = 1;
    cap.features[1].id = RVCL_FEATURE_RVV;
    cap.features[1].value = 128;
    /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately not
     * an enumerator of rvcl_feature_id_t. */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    cap.features[2].id = (rvcl_feature_id_t)0x1000;
    cap.features[2].value = 0x20000110u; /* informative: first channel */
    cap.features[3].id = RVCL_FEATURE_FP16;
    cap.features[3].value = 1;
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    cap.features[4].id = (rvcl_feature_id_t)0x1000;
    cap.features[4].value = 1; /* presence: second channel */
    cap.num_features = 5;

    rvcl_capability_test_override(&cap);
    const rvcl_capability_t *served = rvcl_get_capability();
    CHECK(served->num_features == 4, "duplicate id collapsed to one entry");
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    const rvcl_feature_t *f
            = rvcl_capability_get(served, (rvcl_feature_id_t)0x1000);
    CHECK(f && f->value == 0x20000110u,
            "first entry (informative) wins over the later presence copy");
    CHECK(served->features[0].id == RVCL_FEATURE_RV64
                    && served->features[1].id == RVCL_FEATURE_RVV
                    && served->features[3].id == RVCL_FEATURE_FP16,
            "surviving entries keep the assembly order");
    CHECK(strstr(served->description, "4 feature(s)") != NULL,
            "description counts the deduplicated snapshot");
    rvcl_capability_test_override(NULL);

    printf(g_failures ? "SNAPSHOT-DEDUP: %d failure(s)\n"
                      : "SNAPSHOT-DEDUP: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

/* ------------------------------------------------------------- hint pinning
 *
 * U-P2-3: the public cost hint must agree with the internal category wrapper
 * at the tie tier and stay pinned there regardless of the host VLEN tier
 * (a vendor tile is VLEN-independent). Assertions anchor the semantic tiers
 * (256 == cheap-vector tie tier, coefficients below that never exceed it),
 * not the concrete cycle numbers, so a future coefficient recalibration does
 * not turn this into a maintenance burden.
 */
static int vendor_hint_case(void) {
    printf("cost hint: semantic tier pinning\n");
    static const int64_t shapes[][3] = {
            {16, 16, 16},
            {64, 32, 16},
            {128, 128, 128},
            {13, 7, 20},
    };

    /* At the tie tier (128, gain 256) the hint at coeff 256 must equal the
     * internal cheap-vector wrapper exactly, for every shape. */
    rvcl_capability_t t128 = make_snapshot(1, 1, 128);
    rvcl_capability_test_override(&t128);
    int tie_ok = 1;
    int ref_ok = 1;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        rvcl_matmul_desc_t d = {
                .m = shapes[i][0],
                .n = shapes[i][1],
                .k = shapes[i][2],
                .dtype_a = RVCL_DTYPE_FP32,
                .dtype_b = RVCL_DTYPE_FP32,
                .dtype_acc = RVCL_DTYPE_FP32,
                .dtype_c = RVCL_DTYPE_FP32,
                .dtype_bias = RVCL_DTYPE_AUTO,
                .quant = {},
        };
        const rvcl_op_desc_t od = {.matmul = d};
        const uint64_t hint256 = rvcl_cost_matmul_hint(&od, &t128, 256);
        const uint64_t rvv = rvcl_cost_matmul_rvv(&od, &t128);
        if (hint256 != rvv) {
            printf("  shape %ldx%ldx%ld: hint(256)=%llu rvv=%llu\n",
                    (long)shapes[i][0], (long)shapes[i][1], (long)shapes[i][2],
                    (unsigned long long)hint256, (unsigned long long)rvv);
        }
        tie_ok = tie_ok && (hint256 == rvv);
        /* a sub-256 coefficient scores no higher than the rvv category. */
        const uint64_t h128 = rvcl_cost_matmul_hint(&od, &t128, 128);
        const uint64_t h512 = rvcl_cost_matmul_hint(&od, &t128, 512);
        const uint64_t h1024 = rvcl_cost_matmul_hint(&od, &t128, 1024);
        tie_ok = tie_ok && (h128 <= rvv);
        /* coefficient is monotonic: 512 > 256 > 128 in ordering terms. */
        if (h128 > hint256 || hint256 >= h512 || h512 >= h1024)
            printf("  shape %ldx%ldx%ld: coeff ordering broken\n",
                    (long)shapes[i][0], (long)shapes[i][1], (long)shapes[i][2]);
        tie_ok = tie_ok && (h128 <= hint256 && hint256 < h512 && h512 < h1024);

        /* the VLEN-independent reference category scores through the same
         * public path: the hint at the reference coefficient (64 x 256,
         * mirroring the expensive per-element constant) equals the
         * reference wrapper per shape — exact, because the coefficient is
         * an exact multiple of the fixed-point scale. */
        const uint64_t ref = rvcl_cost_matmul_reference(&od, &t128);
        const uint64_t ref_hint
                = rvcl_cost_matmul_hint(&od, &t128, (uint64_t)64 * 256);
        if (ref != ref_hint) {
            printf("  shape %ldx%ldx%ld: reference=%llu hint=%llu\n",
                    (long)shapes[i][0], (long)shapes[i][1], (long)shapes[i][2],
                    (unsigned long long)ref, (unsigned long long)ref_hint);
        }
        ref_ok = ref_ok && (ref == ref_hint);
    }
    CHECK(tie_ok, "hint ties the cheap-vector wrapper at the tie tier");
    CHECK(ref_ok,
            "reference wrapper equals the hint at the reference coefficient");

    /* VLEN independence: the hint is identical across host tiers while the
     * internal wrapper's tier gain changes. */
    rvcl_capability_t t256 = make_snapshot(1, 1, 256);
    rvcl_capability_t t512 = make_snapshot(1, 1, 512);
    rvcl_matmul_desc_t d = {
            .m = 64,
            .n = 64,
            .k = 64,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_FP32,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
    const rvcl_op_desc_t od = {.matmul = d};
    const uint64_t h_t128 = rvcl_cost_matmul_hint(&od, &t128, 256);
    const uint64_t h_t256 = rvcl_cost_matmul_hint(&od, &t256, 256);
    const uint64_t h_t512 = rvcl_cost_matmul_hint(&od, &t512, 256);
    CHECK(h_t128 == h_t256 && h_t256 == h_t512,
            "hint is independent of the host VLEN tier");
    CHECK(vlen_tier_gain(&t128) != vlen_tier_gain(&t512),
            "internal tier gain does change with the host tier");

    rvcl_capability_test_override(NULL);
    printf(g_failures ? "VENDOR-HINT: %d failure(s)\n"
                      : "VENDOR-HINT: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

static int run_inject_mode(void) {
    printf("== capability unit tests: injection subset ==\n");
    const rvcl_capability_t *host_ptr = rvcl_get_capability();
    rvcl_capability_t host_copy = *host_ptr;

    case_inject_tier_scan();
    case_inject_description();
    case_inject_zero_feature();
    case_inject_dispatch_cache();
    case_inject_restore(host_ptr, &host_copy);

    if (g_failures) printf("FAILURES: %d\n", g_failures);
    return g_failures ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--inject") == 0) return run_inject_mode();
    if (argc > 1 && strcmp(argv[1], "--vendor-probe") == 0)
        return vendor_probe_case();
    if (argc > 1 && strcmp(argv[1], "--vendor-probe-noreg") == 0)
        return vendor_probe_noreg_case();
    if (argc > 1 && strcmp(argv[1], "--vendor-probe-late") == 0)
        return vendor_probe_late_case();
    if (argc > 1 && strcmp(argv[1], "--vendor-probe-sanitize") == 0)
        return vendor_probe_sanitize_case();
    if (argc > 1 && strcmp(argv[1], "--snapshot-dedup") == 0)
        return snapshot_dedup_case();
    if (argc > 1 && strcmp(argv[1], "--vendor-hint") == 0)
        return vendor_hint_case();

    printf("== capability unit tests ==\n");
    /* Must stay the first capability user of the process: the first
     * detection has to happen under concurrency for the case above. */
    case_concurrent_first_call();
    CHECK(rvcl_init() == RVCL_SUCCESS, "library init succeeds");

    case_host_features();
    case_vector_build_fp16();
    case_zero_feature_host();
    case_snapshot_stability();
    case_description();
    case_query_semantics();
    case_vlen_tier_table();

    if (g_failures) printf("FAILURES: %d\n", g_failures);
    return g_failures ? 1 : 0;
}
