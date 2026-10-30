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
 * unit_registry.cpp — registry unit tests (RFC-0002 FR1 / AC1)
 *
 * Covers: full-field registration, lazy init via first computation, init
 * idempotence, duplicate/capacity/malformed rejection, iteration and
 * factory contracts, unregister semantics incl. the per-tuple fallback
 * invariant, builtin registration order, and concurrent
 * register/unregister/query safety (meaningful under the ASan build).
 *
 * Custom kernels use datatype tuples no builtin occupies (INT8/INT4) and
 * are unregistered again at the end of each case, followed by a dispatch
 * cache clear, so cases stay independent.
 *
 * Pre-init cases run in fresh processes (argv modes): they register
 * custom kernels before the first rvcl_init() to exercise the whole-table
 * fallback verifier and the failed-init rollback.
 ******************************************************************************/

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

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

/* ------------------------------------------------------- minimal test kernel
 *
 * A do-nothing kernel: none of these cases execute it; it only has to
 * satisfy the factory contract. */

static rvcl_status_t test_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return RVCL_SUCCESS;
}

static rvcl_status_t test_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    (void)k;
    (void)desc;
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t test_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *range) {
    (void)k;
    (void)pack;
    (void)range;
    return RVCL_SUCCESS;
}

static void test_destroy(rvcl_kernel_t *k) {
    free(k);
}

static const rvcl_kernel_vtable_t g_test_vtable = {
        .validate = test_validate,
        .configure = test_configure,
        .pack_a = NULL,
        .pack_b = NULL,
        .run = test_run,
        .destroy = test_destroy,
};

static rvcl_status_t test_create(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    *out = (rvcl_kernel_t *)calloc(1, sizeof(rvcl_kernel_t));
    if (!*out) return RVCL_ERROR_OUT_OF_MEMORY;
    (*out)->vt = &g_test_vtable;
    (*out)->desc = *desc;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------------- helpers */

static rvcl_kernel_desc_t make_desc(const char *name) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));
    /* precision-bounded copy: GCC cannot prove arbitrary callers fit the
     * fixed name buffer, and -Wformat-truncation is fatal under Werror */
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%.*s", RVCL_KERNEL_NAME_MAX - 1,
            name);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = RVCL_DTYPE_INT8;
    d.dtype_b = RVCL_DTYPE_INT8;
    d.dtype_acc = RVCL_DTYPE_INT32;
    d.dtype_c = RVCL_DTYPE_INT32;
    d.quant_support = RVCL_QUANT_NONE;
    d.num_required_features = 0;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_kernel_cost_zero;
    return d;
}

static void unregister_quiet(const char *name) {
    rvcl_unregister_kernel(name);
}

static int find_name_index(const char *name) {
    const int n = rvcl_registry_count();
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (kd && strcmp(kd->name, name) == 0) return i;
    }
    return -1;
}

static int vector_build(void) {
    return rvcl_registry_find_factory("rvcl_matmul_f32_rvv") != NULL;
}

static int has_feature(const rvcl_kernel_desc_t *kd, rvcl_feature_id_t id) {
    for (int i = 0; i < kd->num_required_features; i++)
        if (kd->required_features[i] == id) return 1;
    return 0;
}

/* Field-wise comparison for the concurrency final-state check (names are
 * matched by lookup, not compared here). */
static int same_desc(const rvcl_kernel_desc_t *a, const rvcl_kernel_desc_t *b) {
    if (a->operation != b->operation || a->dtype_a != b->dtype_a
            || a->dtype_b != b->dtype_b || a->dtype_acc != b->dtype_acc
            || a->dtype_c != b->dtype_c || a->quant_support != b->quant_support
            || a->num_required_features != b->num_required_features
            || a->min_m != b->min_m || a->max_m != b->max_m
            || a->min_n != b->min_n || a->max_n != b->max_n
            || a->min_k != b->min_k || a->max_k != b->max_k
            || a->alignment_a != b->alignment_a
            || a->alignment_b != b->alignment_b
            || a->alignment_c != b->alignment_c || a->vlen_tier != b->vlen_tier
            || a->cost_fn != b->cost_fn)
        return 0;
    for (int i = 0; i < a->num_required_features; i++)
        if (a->required_features[i] != b->required_features[i]) return 0;
    return 1;
}

/* --------------------------------------------------- pre-init init paths
 *
 * Run as separate processes (ctest passes an argument) so registrations
 * made before the first rvcl_init() are observable and every case starts
 * from a clean lazy-init state. */

static int preinit_fallback_case(void) {
    printf("init-path: bounded + unbounded tuple registered before init\n");
    rvcl_kernel_desc_t bounded = make_desc("pre_bounded");
    bounded.max_m = 64;
    rvcl_kernel_desc_t unbounded = make_desc("pre_unbounded");
    CHECK(rvcl_register_kernel(&bounded, test_create) == RVCL_SUCCESS,
            "pre-init bounded registration succeeds");
    CHECK(rvcl_register_kernel(&unbounded, test_create) == RVCL_SUCCESS,
            "pre-init unbounded registration succeeds");
    CHECK(rvcl_init() == RVCL_SUCCESS,
            "init accepts a legal pre-registered bounded+unbounded tuple");
    CHECK(rvcl_registry_find_factory("pre_bounded") != NULL
                    && rvcl_registry_find_factory("pre_unbounded") != NULL,
            "pre-init entries survive init");
    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "built-ins registered after a successful init");
    printf(g_failures ? "UNIT-REGISTRY-PREINIT: %d failure(s)\n"
                      : "UNIT-REGISTRY-PREINIT: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

static int preinit_lone_bounded_case(void) {
    printf("init-path: lone bounded tuple registered before init\n");
    rvcl_kernel_desc_t bounded = make_desc("pre_lone_bounded");
    bounded.max_n = 32;
    CHECK(rvcl_register_kernel(&bounded, test_create) == RVCL_SUCCESS,
            "pre-init bounded registration succeeds");
    CHECK(rvcl_init() == RVCL_ERROR_INTERNAL,
            "init rejects a tuple without an unbounded kernel");
    CHECK(rvcl_registry_count() == 1,
            "failed init rolls back the built-ins it registered");
    CHECK(rvcl_registry_find_factory("pre_lone_bounded") != NULL,
            "pre-init entries survive the failed init");
    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") == NULL,
            "no built-in is left behind by the failed init");
    printf(g_failures ? "UNIT-REGISTRY-PREINIT: %d failure(s)\n"
                      : "UNIT-REGISTRY-PREINIT: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}

/* TC-AC1-02: first computation triggers initialization (must run before
 * any explicit rvcl_init in this process). */
static void tc02_first_compute_lazy_init(void) {
    printf("TC-AC1-02 first computation triggers init\n");
    enum { M = 8, N = 8, K = 16 };
    float a[M * K], b[K * N], bias[N], c[M * N], ref[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = (float)((i * 13) % 29) / 29.0f - 0.5f;
    for (int i = 0; i < K * N; i++)
        b[i] = (float)((i * 7) % 31) / 31.0f - 0.5f;
    for (int j = 0; j < N; j++)
        bias[j] = 0.25f;

    rvcl_matmul_desc_t d = {
            .m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    CHECK(rvcl_matmul(&d, a, b, bias, c) == RVCL_SUCCESS,
            "first computation succeeds without explicit init");
    golden_f32(a, b, bias, ref, M, N, K);
    CHECK(memcmp(c, ref, sizeof(c)) == 0, "lazy-init path matches golden");

    CHECK(rvcl_registry_count() >= 1, "registry populated after lazy init");
    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "f32 fallback entry present");
}

/* TC-AC1-01: legal descriptor registers with every field intact. */
static void tc01_register_full_fields(void) {
    printf("TC-AC1-01 full-field registration\n");
    const int c0 = rvcl_registry_count();
    rvcl_kernel_desc_t d = make_desc("t01_full");
    d.required_features[0] = RVCL_FEATURE_INT8;
    d.required_features[1] = RVCL_FEATURE_RV64;
    d.num_required_features = 2;
    d.min_m = 2;
    d.max_m = 64;
    d.min_n = 1;
    d.max_n = 0; /* unbounded */
    d.min_k = 4;
    d.max_k = 512;
    d.preferred_m = 8;
    d.preferred_n = 8;
    d.preferred_k = 8;
    d.alignment_a = 2;
    d.alignment_b = 4;
    d.alignment_c = 2;
    d.vlen_tier = RVCL_VLEN_TIER_256;
    d.cost_fn = rvcl_kernel_cost_zero;

    CHECK(rvcl_register_kernel(&d, test_create) == RVCL_SUCCESS,
            "valid descriptor registers");
    CHECK(rvcl_registry_count() == c0 + 1, "count incremented by one");

    const int idx = find_name_index("t01_full");
    CHECK(idx >= 0, "entry iterable by name");
    if (idx >= 0) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(idx);
        int same = kd && kd->operation == d.operation
                && kd->dtype_a == d.dtype_a && kd->dtype_b == d.dtype_b
                && kd->dtype_acc == d.dtype_acc && kd->dtype_c == d.dtype_c
                && kd->quant_support == d.quant_support
                && kd->num_required_features == d.num_required_features
                && kd->required_features[0] == d.required_features[0]
                && kd->required_features[1] == d.required_features[1]
                && kd->min_m == d.min_m && kd->max_m == d.max_m
                && kd->min_n == d.min_n && kd->max_n == d.max_n
                && kd->min_k == d.min_k && kd->max_k == d.max_k
                && kd->preferred_m == d.preferred_m
                && kd->preferred_n == d.preferred_n
                && kd->preferred_k == d.preferred_k
                && kd->alignment_a == d.alignment_a
                && kd->alignment_b == d.alignment_b
                && kd->alignment_c == d.alignment_c
                && kd->vlen_tier == d.vlen_tier
                && kd->cost_fn == rvcl_kernel_cost_zero;
        CHECK(same, "every registered field reads back identically");
    }
    unregister_quiet("t01_full");
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-AC1-03: init idempotence; entries for variants that did not compile
 * into the build are absent. */
static void tc03_init_idempotent(void) {
    printf("TC-AC1-03 init idempotence and build guards\n");
    char before[64][RVCL_KERNEL_NAME_MAX];
    const int n0 = rvcl_registry_count();
    for (int i = 0; i < n0 && i < RVCL_REGISTRY_MAX_KERNELS; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        snprintf(before[i], RVCL_KERNEL_NAME_MAX, "%s", kd ? kd->name : "");
    }

    CHECK(rvcl_init() == RVCL_SUCCESS, "repeated init succeeds");
    CHECK(rvcl_init() == RVCL_SUCCESS, "third init succeeds");
    CHECK(rvcl_registry_count() == n0, "count unchanged by re-init");

    int same = 1;
    for (int i = 0; i < n0; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || strcmp(kd->name, before[i]) != 0) same = 0;
    }
    CHECK(same, "name sequence unchanged by re-init");

    if (!vector_build()) {
        int leaked = 0;
        for (int i = 0; i < rvcl_registry_count(); i++) {
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
            if (!kd) continue;
            if (kd->vlen_tier != RVCL_VLEN_TIER_ANY) leaked = 1;
            for (int f = 0; f < kd->num_required_features; f++)
                if (kd->required_features[f] == RVCL_FEATURE_RVV) leaked = 1;
        }
        CHECK(leaked == 0, "non-vector build has no RVV/tier entries");
    } else {
        printf("  ok: (vector build, tier entries expected)\n");
    }
}

/* TC-AC1-04: duplicate names rejected atomically. */
static void tc04_duplicate_rejected(void) {
    printf("TC-AC1-04 duplicate registration rejected\n");
    const int c0 = rvcl_registry_count();

    rvcl_kernel_desc_t a = make_desc("t04_dup");
    CHECK(rvcl_register_kernel(&a, test_create) == RVCL_SUCCESS,
            "first registration succeeds");

    rvcl_kernel_desc_t b = make_desc("t04_dup");
    b.max_k = 128; /* different content, same name */
    CHECK(rvcl_register_kernel(&b, test_create)
                    == RVCL_ERROR_ALREADY_REGISTERED,
            "duplicate name rejected");
    CHECK(rvcl_registry_count() == c0 + 1, "count unchanged by rejection");

    /* builtin names are reserved the same way */
    rvcl_kernel_desc_t c = make_desc("rvcl_matmul_f32_reference");
    c.dtype_a = RVCL_DTYPE_FP32;
    c.dtype_b = RVCL_DTYPE_FP32;
    c.dtype_acc = RVCL_DTYPE_FP32;
    c.dtype_c = RVCL_DTYPE_FP32;
    CHECK(rvcl_register_kernel(&c, test_create)
                    == RVCL_ERROR_ALREADY_REGISTERED,
            "builtin name reserved");

    const int idx = find_name_index("t04_dup");
    if (idx >= 0)
        CHECK(rvcl_registry_get(idx)->max_k == 0,
                "original fields intact after rejected duplicate");
    unregister_quiet("t04_dup");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-05: capacity limit rejects atomically. */
static void tc05_capacity(void) {
    printf("TC-AC1-05 capacity overflow rejected\n");
    const int c0 = rvcl_registry_count();
    char names[RVCL_REGISTRY_MAX_KERNELS][RVCL_KERNEL_NAME_MAX];
    int added = 0;

    for (int i = 0; c0 + added < RVCL_REGISTRY_MAX_KERNELS; i++) {
        snprintf(names[added], RVCL_KERNEL_NAME_MAX, "t05_cap_%d", i);
        rvcl_kernel_desc_t d = make_desc(names[added]);
        if (rvcl_register_kernel(&d, test_create) != RVCL_SUCCESS) break;
        added++;
    }
    CHECK(rvcl_registry_count() == RVCL_REGISTRY_MAX_KERNELS,
            "registry fills to capacity");

    rvcl_kernel_desc_t d = make_desc("t05_over");
    CHECK(rvcl_register_kernel(&d, test_create) == RVCL_ERROR_OUT_OF_MEMORY,
            "entry beyond capacity rejected");
    CHECK(rvcl_registry_count() == RVCL_REGISTRY_MAX_KERNELS,
            "count unchanged by overflow");

    int distinct = 1;
    for (int i = 0; i < RVCL_REGISTRY_MAX_KERNELS && distinct; i++)
        for (int j = i + 1; j < RVCL_REGISTRY_MAX_KERNELS; j++) {
            const rvcl_kernel_desc_t *a = rvcl_registry_get(i);
            const rvcl_kernel_desc_t *b = rvcl_registry_get(j);
            if (a && b && strcmp(a->name, b->name) == 0) distinct = 0;
        }
    CHECK(distinct, "all capacity entries have distinct names");

    for (int i = 0; i < added; i++)
        unregister_quiet(names[i]);
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-AC1-06: malformed descriptors rejected with no side effects. */
static void tc06_invalid_descriptors(void) {
    printf("TC-AC1-06 malformed descriptors rejected\n");
    const int c0 = rvcl_registry_count();
    int rejects = 0, total = 0;

    struct {
        const char *label;
        rvcl_kernel_desc_t d;
    } cases[32];
    int nc = 0;

    rvcl_kernel_desc_t base = make_desc("t06_invalid");

#define ADD_CASE(label_str, mut) \
    do { \
        rvcl_kernel_desc_t d = base; \
        mut; \
        cases[nc].label = label_str; \
        cases[nc].d = d; \
        nc++; \
    } while (0)

    ADD_CASE("empty name", d.name[0] = '\0');
    ADD_CASE("overlong name", memset(d.name, 'x', RVCL_KERNEL_NAME_MAX));
    ADD_CASE("operation out of range", d.operation = RVCL_OP_COUNT);
    ADD_CASE("operation negative", d.operation = (rvcl_operation_t)-1);
    ADD_CASE("dtype_a AUTO", d.dtype_a = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype_b AUTO", d.dtype_b = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype_acc AUTO", d.dtype_acc = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype_c AUTO", d.dtype_c = RVCL_DTYPE_AUTO);
    ADD_CASE("dtype out of range", d.dtype_a = (rvcl_dtype_t)99);
    ADD_CASE("quant out of range", d.quant_support = (rvcl_quant_support_t)99);
    ADD_CASE("too many features",
            d.num_required_features = RVCL_MAX_FEATURES + 1);
    ADD_CASE("feature NONE listed",
            (d.required_features[0] = RVCL_FEATURE_NONE,
                    d.num_required_features = 1));
    ADD_CASE("negative min_m", d.min_m = -1);
    ADD_CASE("min_m above max_m", (d.min_m = 8, d.max_m = 4));
    ADD_CASE("min_n above max_n", (d.min_n = 8, d.max_n = 4));
    ADD_CASE("min_k above max_k", (d.min_k = 8, d.max_k = 4));
    ADD_CASE("tier 64", d.vlen_tier = (rvcl_vlen_tier_t)64);
    ADD_CASE("tier 100", d.vlen_tier = (rvcl_vlen_tier_t)100);
    ADD_CASE("tier 3", d.vlen_tier = (rvcl_vlen_tier_t)3);
    ADD_CASE("tier negative", d.vlen_tier = (rvcl_vlen_tier_t)-1);
    ADD_CASE("NULL cost callback", d.cost_fn = NULL);

    for (int i = 0; i < nc; i++) {
        total++;
        if (rvcl_register_kernel(&cases[i].d, test_create)
                == RVCL_ERROR_INVALID_ARGUMENT)
            rejects++;
        else
            printf("  FAIL: case accepted: %s\n", cases[i].label);
    }
    CHECK(total == 21 && rejects == 21, "all malformed descriptors rejected");
    CHECK(rvcl_register_kernel(NULL, test_create)
                    == RVCL_ERROR_INVALID_ARGUMENT,
            "NULL descriptor rejected");
    rvcl_kernel_desc_t d = make_desc("t06_null_factory");
    CHECK(rvcl_register_kernel(&d, NULL) == RVCL_ERROR_INVALID_ARGUMENT,
            "NULL factory rejected");
    CHECK(rvcl_registry_count() == c0, "registry unchanged by all rejections");
}

/* TC-AC1-07: count / iteration / factory lookup contract. */
static void tc07_count_iterate_factory(void) {
    printf("TC-AC1-07 iteration and factory contract\n");
    rvcl_kernel_desc_t d = make_desc("t07_probe");
    rvcl_register_kernel(&d, test_create);

    const int n = rvcl_registry_count();
    CHECK(n >= 1, "count positive");
    CHECK(rvcl_registry_get(n) == NULL, "index == count returns NULL");
    CHECK(rvcl_registry_get(n + 1) == NULL, "index past count returns NULL");
    CHECK(rvcl_registry_get(-1) == NULL, "negative index returns NULL");

    int all_present = 1;
    for (int i = 0; i < n; i++)
        if (!rvcl_registry_get(i)) all_present = 0;
    CHECK(all_present, "every in-range index returns an entry");

    CHECK(rvcl_registry_find_factory("t07_probe") != NULL,
            "factory resolvable for registered name");
    CHECK(rvcl_registry_find_factory("t07_never_registered") == NULL,
            "unknown name yields NULL factory");
    unregister_quiet("t07_probe");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-08 / TC-AC1-09: unregister success and not-found paths. */
static void tc08_09_unregister(void) {
    printf("TC-AC1-08/09 unregister semantics\n");
    const int c0 = rvcl_registry_count();
    rvcl_kernel_desc_t d = make_desc("t08_x");
    CHECK(rvcl_register_kernel(&d, test_create) == RVCL_SUCCESS,
            "custom kernel registered");

    CHECK(rvcl_unregister_kernel("t08_x") == RVCL_SUCCESS,
            "unregister by name succeeds");
    CHECK(rvcl_registry_count() == c0, "count restored");
    CHECK(find_name_index("t08_x") < 0, "unregistered entry not iterable");
    CHECK(rvcl_registry_find_factory("t08_x") == NULL,
            "unregistered entry has no factory");

    CHECK(rvcl_unregister_kernel("no_such_kernel") == RVCL_ERROR_NOT_FOUND,
            "unregistering unknown name -> NOT_FOUND");
    CHECK(rvcl_unregister_kernel("t08_x") == RVCL_ERROR_NOT_FOUND,
            "double unregister -> NOT_FOUND");
    CHECK(rvcl_registry_count() == c0, "failed unregisters change nothing");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-10: unregistering cannot break the per-tuple fallback
 * invariant. Uses an INT4 tuple no builtin occupies. */
static void tc10_fallback_invariant(void) {
    printf("TC-AC1-10 fallback invariant on unregister\n");
    const int c0 = rvcl_registry_count();

    rvcl_kernel_desc_t bounded = make_desc("t10_b1");
    bounded.dtype_a = RVCL_DTYPE_INT4;
    bounded.dtype_b = RVCL_DTYPE_INT4;
    bounded.max_m = 64; /* bounded */
    rvcl_kernel_desc_t unbnd = make_desc("t10_u1");
    unbnd.dtype_a = RVCL_DTYPE_INT4;
    unbnd.dtype_b = RVCL_DTYPE_INT4;

    CHECK(rvcl_register_kernel(&bounded, test_create) == RVCL_SUCCESS
                    && rvcl_register_kernel(&unbnd, test_create)
                            == RVCL_SUCCESS,
            "bounded + unbounded pair registered");

    CHECK(rvcl_unregister_kernel("t10_u1") == RVCL_ERROR_REGISTRY_INVARIANT,
            "removing the only unbounded entry rejected");
    CHECK(rvcl_registry_find_factory("t10_u1") != NULL,
            "rejected unregister leaves the entry in place");

    rvcl_kernel_desc_t unbnd2 = unbnd;
    snprintf(unbnd2.name, RVCL_KERNEL_NAME_MAX, "%s", "t10_u2");
    CHECK(rvcl_register_kernel(&unbnd2, test_create) == RVCL_SUCCESS,
            "second unbounded entry registered");
    CHECK(rvcl_unregister_kernel("t10_u1") == RVCL_SUCCESS,
            "unregister allowed when another unbounded remains");

    CHECK(rvcl_unregister_kernel("t10_b1") == RVCL_SUCCESS,
            "bounded entry removable");
    CHECK(rvcl_unregister_kernel("t10_u2") == RVCL_SUCCESS,
            "last entry of an emptying tuple removable");

    /* builtin f32 tuple: reject removing a unique unbounded fallback,
     * skip when other unbounded entries keep the tuple covered. */
    int f32_unbounded = 0, reference_unbounded = 0;
    for (int i = 0; i < rvcl_registry_count(); i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || kd->dtype_a != RVCL_DTYPE_FP32) continue;
        if (kd->max_m == 0 && kd->max_n == 0 && kd->max_k == 0) {
            f32_unbounded++;
            if (strcmp(kd->name, "rvcl_matmul_f32_reference") == 0)
                reference_unbounded = 1;
        }
    }
    if (reference_unbounded && f32_unbounded == 1) {
        CHECK(rvcl_unregister_kernel("rvcl_matmul_f32_reference")
                        == RVCL_ERROR_REGISTRY_INVARIANT,
                "builtin unique fallback protected");
        CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
                "builtin fallback still present");
    } else {
        printf("  ok: (f32 tuple has other unbounded entries, skip)\n");
    }
    CHECK(rvcl_registry_count() == c0, "table restored");
    rvcl_dispatch_cache_clear();
}

/* TC-AC1-11: every live datatype tuple keeps an unbounded entry. */
static void tc11_table_invariant(void) {
    printf("TC-AC1-11 table-wide fallback invariant\n");
    const int n = rvcl_registry_count();
    int ok = 1;
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd) continue;
        int covered = 0;
        for (int j = 0; j < n; j++) {
            const rvcl_kernel_desc_t *o = rvcl_registry_get(j);
            if (!o || o->operation != kd->operation || o->dtype_a != kd->dtype_a
                    || o->dtype_b != kd->dtype_b
                    || o->dtype_acc != kd->dtype_acc
                    || o->dtype_c != kd->dtype_c)
                continue;
            if (o->max_m == 0 && o->max_n == 0 && o->max_k == 0) covered = 1;
        }
        if (!covered) ok = 0;
    }
    CHECK(ok, "every live tuple has an unbounded fallback");
}

/* TC-AC1-12: builtin order = dedicated (high tier -> low), then generic,
 * then fallback. Dedicated and generic entries must declare the vector
 * features they depend on (FR1.5). */
static void tc12_builtin_order(void) {
    printf("TC-AC1-12 builtin registration order\n");
    const int n = rvcl_registry_count();
    const int vector = vector_build();

    for (int pass = 0; pass < 2; pass++) {
        const rvcl_dtype_t key = pass == 0 ? RVCL_DTYPE_FP32 : RVCL_DTYPE_FP16;
        int prev_cls = -1; /* 0 dedicated, 1 generic, 2 fallback */
        int prev_tier = 1 << 30;
        int seen = 0, generic_count = 0;
        int group_ok = 1;
        for (int i = 0; i < n; i++) {
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
            if (!kd || kd->dtype_a != key || kd->dtype_b != key) continue;
            seen++;

            int cls;
            if (kd->vlen_tier != RVCL_VLEN_TIER_ANY) {
                cls = 0;
                if (!has_feature(kd, RVCL_FEATURE_RVV)) group_ok = 0;
                if (pass == 1 && !has_feature(kd, RVCL_FEATURE_FP16))
                    group_ok = 0;
                /* tiers must strictly descend among consecutive
                 * dedicated entries */
                if (prev_cls == 0 && (int)kd->vlen_tier >= prev_tier)
                    group_ok = 0;
                prev_tier = (int)kd->vlen_tier;
            } else if (has_feature(kd, RVCL_FEATURE_RVV)) {
                cls = 1; /* generic vector variant */
                generic_count++;
                if (pass == 1 && !has_feature(kd, RVCL_FEATURE_FP16))
                    group_ok = 0;
            } else {
                cls = 2; /* fallback: no vector feature requirement */
                if (kd->max_m != 0 || kd->max_n != 0 || kd->max_k != 0)
                    group_ok = 0;
            }
            if (cls < prev_cls) group_ok = 0; /* never go backwards */
            prev_cls = cls;
        }
        /* Vector builds expose exactly one generic entry per tuple;
         * other builds must have no RVV entry at all. */
        if (vector ? generic_count != 1 : generic_count != 0) group_ok = 0;
        if (pass == 0) CHECK(group_ok && seen >= 2, "f32 group ordered");
        if (pass == 1) CHECK(group_ok && seen >= 1, "f16 group ordered");
    }
}

/* TC-AC1-13: concurrent register / unregister / query. The interesting
 * race window is exercised under the ASan build.
 *
 * Capacity note: the test plan budgets 8 entries per A thread (8x8), but
 * the 64-slot registry also holds the built-ins and the D-group target,
 * so each A thread registers 4 names instead. */

#define T13_THREADS_A 8
#define T13_PER_A 4
#define T13_THREADS_B 4
#define T13_THREADS_C 4
#define T13_QUERIES 2000
/* B covers two A threads each, so the group removes the designated half
 * of all A entries. */
#define T13_UNREG_TOTAL (T13_THREADS_B * 2 * (T13_PER_A / 2))

typedef struct {
    int tid;
    int reg_ok;
    int unreg_ok;
} t13_arg_t;

/* Start gate: every worker blocks until the last one is created, so the
 * groups actually overlap instead of finishing in creation order. */
static pthread_barrier_t g_t13_start;
static std::atomic<int> g_t13_a_done(0);

static int t13_wait_start(void) {
    const int rc = pthread_barrier_wait(&g_t13_start);
    return rc == 0 || rc == PTHREAD_BARRIER_SERIAL_THREAD;
}

static void *t13_register(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    if (t13_wait_start()) {
        for (int i = 0; i < T13_PER_A; i++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "t13_a_%d_%d", arg->tid, i);
            rvcl_kernel_desc_t d = make_desc(name);
            if (rvcl_register_kernel(&d, test_create) == RVCL_SUCCESS)
                arg->reg_ok++;
        }
    }
    /* Published even on a gate failure so B-group waiters cannot spin. */
    g_t13_a_done.fetch_add(1, std::memory_order_release);
    return NULL;
}

static void *t13_unregister(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    if (!t13_wait_start()) return (void *)1;
    for (int a = 0; a < 2; a++) {
        const int a_tid = arg->tid * 2 + a;
        for (int i = 0; i < T13_PER_A / 2; i++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "t13_a_%d_%d", a_tid, i);
            /* The A group may not have published the name yet: NOT_FOUND
             * is a legal race outcome, so retry until the designated half
             * is actually removed. Stop once every A thread has finished
             * — a still-unknown name then means its registration failed. */
            for (;;) {
                if (rvcl_unregister_kernel(name) == RVCL_SUCCESS) {
                    arg->unreg_ok++;
                    break;
                }
                if (g_t13_a_done.load(std::memory_order_acquire)
                        == T13_THREADS_A) {
                    /* Close the race where the previous attempt observed
                     * NOT_FOUND just before the final A registration lands. */
                    if (rvcl_unregister_kernel(name) == RVCL_SUCCESS)
                        arg->unreg_ok++;
                    break;
                }
                sched_yield();
            }
        }
    }
    return NULL;
}

static void *t13_query(void *p) {
    (void)p;
    if (!t13_wait_start()) return (void *)1;
    for (int i = 0; i < T13_QUERIES; i++) {
        const int n = rvcl_registry_count();
        for (int j = 0; j < n; j += 7) {
            /* The pointer is valid only while its entry stays registered
             * (registry.h), so the loop exercises the call path without
             * asserting descriptor content on a raced slot. */
            (void)rvcl_registry_get(j);
        }
        (void)rvcl_registry_find_factory("t13_a_0_0");
    }
    return NULL;
}

static void *t13_same_name_unregister(void *p) {
    t13_arg_t *arg = (t13_arg_t *)p;
    if (!t13_wait_start()) return (void *)1;
    if (rvcl_unregister_kernel("t13_dup_race") == RVCL_SUCCESS) arg->unreg_ok++;
    return NULL;
}

static void t13_spawn(pthread_t *thr, void *(*fn)(void *), t13_arg_t *arg) {
    const int rc = pthread_create(thr, NULL, fn, arg);
    if (rc == 0) return;
    fprintf(stderr, "  FAIL: pthread_create: %s\n", strerror(rc));
    exit(1); /* other workers already wait at the gate: abort cleanly */
}

static void tc13_concurrent(void) {
    printf("TC-AC1-13 concurrent register/unregister/query\n");
    const int c0 = rvcl_registry_count();

    /* same-name race target (counted explicitly in the arithmetic below;
     * its removal by the D-group winner is why c0 itself must not include
     * it) */
    rvcl_kernel_desc_t dup = make_desc("t13_dup_race");
    CHECK(rvcl_register_kernel(&dup, test_create) == RVCL_SUCCESS,
            "same-name race target registered");

    pthread_t ta[T13_THREADS_A], tb[T13_THREADS_B], tc[T13_THREADS_C], td[2];
    t13_arg_t aa[T13_THREADS_A], ab[T13_THREADS_B], ac[T13_THREADS_C], ad[2];

    const int total = T13_THREADS_A + T13_THREADS_B + T13_THREADS_C + 2;
    if (pthread_barrier_init(&g_t13_start, NULL, (unsigned)total) != 0) {
        printf("  FAIL: start barrier init (line %d)\n", __LINE__);
        g_failures++;
        return;
    }
    g_t13_a_done.store(0, std::memory_order_relaxed);

    for (int i = 0; i < T13_THREADS_A; i++) {
        aa[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        t13_spawn(&ta[i], t13_register, &aa[i]);
    }
    for (int i = 0; i < T13_THREADS_B; i++) {
        ab[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        t13_spawn(&tb[i], t13_unregister, &ab[i]);
    }
    for (int i = 0; i < T13_THREADS_C; i++) {
        ac[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        t13_spawn(&tc[i], t13_query, &ac[i]);
    }
    /* two racers on the same name */
    for (int i = 0; i < 2; i++) {
        ad[i] = t13_arg_t {.tid = i, .reg_ok = 0, .unreg_ok = 0};
        t13_spawn(&td[i], t13_same_name_unregister, &ad[i]);
    }

    int reg_ok = 0, unreg_ok = 0, thread_err = 0, dup_ok = 0;
    for (int i = 0; i < T13_THREADS_A; i++) {
        void *ret = NULL;
        pthread_join(ta[i], &ret);
        if (ret) thread_err = 1;
        reg_ok += aa[i].reg_ok;
    }
    for (int i = 0; i < T13_THREADS_B; i++) {
        void *ret = NULL;
        pthread_join(tb[i], &ret);
        if (ret) thread_err = 1;
        unreg_ok += ab[i].unreg_ok;
    }
    for (int i = 0; i < T13_THREADS_C; i++) {
        void *ret = NULL;
        pthread_join(tc[i], &ret);
        if (ret) thread_err = 1;
    }
    for (int i = 0; i < 2; i++) {
        void *ret = NULL;
        pthread_join(td[i], &ret);
        if (ret) thread_err = 1;
        dup_ok += ad[i].unreg_ok;
    }
    pthread_barrier_destroy(&g_t13_start);

    CHECK(thread_err == 0, "all worker threads ran");
    CHECK(reg_ok == T13_THREADS_A * T13_PER_A, "all concurrent registers ok");
    CHECK(unreg_ok == T13_UNREG_TOTAL,
            "every designated half was unregistered");
    CHECK(dup_ok == 1, "exactly one winner for the same-name race");
    CHECK(rvcl_registry_count() == c0 + 1 + reg_ok - unreg_ok - dup_ok,
            "final count matches successful operations");

    /* Every A entry is either fully intact or was removed: no partial
     * registration survives the concurrent phase. */
    int intact = 1, removed = 0;
    for (int t = 0; t < T13_THREADS_A; t++) {
        for (int i = 0; i < T13_PER_A; i++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "t13_a_%d_%d", t, i);
            const int idx = find_name_index(name);
            if (idx < 0) {
                removed++;
                continue;
            }
            const rvcl_kernel_desc_t expected = make_desc(name);
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(idx);
            if (!kd || !same_desc(kd, &expected)) intact = 0;
        }
    }
    CHECK(intact, "every surviving entry is complete");
    CHECK(removed == T13_UNREG_TOTAL,
            "removed entries match the designated unregisters");

    const int n = rvcl_registry_count();
    int distinct = 1;
    for (int i = 0; i < n && distinct; i++) {
        const rvcl_kernel_desc_t *a = rvcl_registry_get(i);
        if (!a) {
            distinct = 0;
            break;
        }
        for (int j = i + 1; j < n; j++) {
            const rvcl_kernel_desc_t *b = rvcl_registry_get(j);
            if (b && strcmp(a->name, b->name) == 0) distinct = 0;
        }
    }
    CHECK(distinct, "iteration yields distinct names");

    /* cleanup the survivors */
    for (int t = 0; t < T13_THREADS_A; t++)
        for (int i = 0; i < T13_PER_A; i++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "t13_a_%d_%d", t, i);
            unregister_quiet(name);
        }
    unregister_quiet("t13_dup_race");
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-AC1-14: builtin entries exist per build configuration and every
 * name resolves. The enabled tier set is a build input (AC1), so this
 * target receives the same RVCL_RVV_TIERS_<T> definitions as the library
 * and asserts exactly that set. */
static void tc14_builtin_entries(void) {
    printf("TC-AC1-14 builtin entries present and queryable\n");
    const int n = rvcl_registry_count();
    int all_resolvable = 1;
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || !rvcl_registry_find_factory(kd->name)) all_resolvable = 0;
    }
    CHECK(all_resolvable, "every registry name resolves to a factory");

    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "f32 unbounded reference present");

    if (vector_build()) {
        int enabled[4];
        int num_enabled = 0;
#if defined(RVCL_RVV_TIERS_1024)
        enabled[num_enabled++] = 1024;
#endif
#if defined(RVCL_RVV_TIERS_512)
        enabled[num_enabled++] = 512;
#endif
#if defined(RVCL_RVV_TIERS_256)
        enabled[num_enabled++] = 256;
#endif
#if defined(RVCL_RVV_TIERS_128)
        enabled[num_enabled++] = 128;
#endif
        CHECK(num_enabled > 0, "vector build enables at least one tier");

        int dedicated = 0, matched[4] = {0, 0, 0, 0};
        for (int i = 0; i < n; i++) {
            const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
            if (!kd || kd->dtype_a != RVCL_DTYPE_FP32
                    || kd->dtype_b != RVCL_DTYPE_FP32)
                continue;
            if (kd->vlen_tier == RVCL_VLEN_TIER_ANY) continue;
            dedicated++;
            for (int t = 0; t < num_enabled; t++)
                if ((int)kd->vlen_tier == enabled[t]) matched[t]++;
        }
        CHECK(dedicated == num_enabled,
                "exactly the build-enabled f32 tiers are registered");
        for (int t = 0; t < num_enabled; t++) {
            char name[RVCL_KERNEL_NAME_MAX];
            snprintf(name, sizeof(name), "rvcl_matmul_f32_rvv_%d", enabled[t]);
            CHECK(find_name_index(name) >= 0, "enabled tier entry present");
            CHECK(matched[t] == 1, "enabled tier registered exactly once");
        }
    } else {
        CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_rvv_c") != NULL,
                "portable f32 entry present on non-vector builds");
        CHECK(rvcl_registry_find_factory("rvcl_matmul_f16_rvv_c") != NULL,
                "portable f16 entry present on non-vector builds");
    }
}

int main(int argc, char **argv) {
    if (argc > 1) {
        if (strcmp(argv[1], "--preinit-fallback") == 0)
            return preinit_fallback_case();
        if (strcmp(argv[1], "--preinit-lone-bounded") == 0)
            return preinit_lone_bounded_case();
        fprintf(stderr,
                "usage: %s [--preinit-fallback|--preinit-lone-bounded]\n",
                argv[0]);
        return 2;
    }

    tc02_first_compute_lazy_init(); /* must be first: fresh-process lazy init */
    CHECK(rvcl_init() == RVCL_SUCCESS, "explicit init after lazy path");
    tc01_register_full_fields();
    tc03_init_idempotent();
    tc04_duplicate_rejected();
    tc05_capacity();
    tc06_invalid_descriptors();
    tc07_count_iterate_factory();
    tc08_09_unregister();
    tc10_fallback_invariant();
    tc11_table_invariant();
    tc12_builtin_order();
    tc13_concurrent();
    tc14_builtin_entries();

    printf(g_failures ? "UNIT-REGISTRY: %d failure(s)\n"
                      : "UNIT-REGISTRY: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
