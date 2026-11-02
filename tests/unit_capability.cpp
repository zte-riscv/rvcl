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
