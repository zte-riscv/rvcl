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

/* Per-operation registry structure tests: locks the whole-table
 * properties that must survive the registry's per-operation split -
 * cross-operation name uniqueness, the shared global capacity, the
 * global registration order under mixed-operation registration,
 * pre-main registration safety (the registry must be reachable from a
 * plugin constructor before main), and the per-tuple fallback invariant
 * being independent across operations.
 *
 * Custom kernels use datatype tuples no builtin occupies (INT8/INT4)
 * and are unregistered again at the end of each case, followed by a
 * dispatch cache clear, so cases stay independent. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/runtime/dispatch_internal.h"
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
    /* rvcl_kernel_destroy() frees the kernel allocation after this
     * callback: destroy only releases subclass state (none here). */
    (void)k;
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

static rvcl_kernel_desc_t make_desc(const char *name, rvcl_operation_t op) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));
    /* precision-bounded copy: GCC cannot prove arbitrary callers fit the
     * fixed name buffer, and -Wformat-truncation is fatal under Werror */
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%.*s", RVCL_KERNEL_NAME_MAX - 1,
            name);
    d.operation = op;
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

/* ----------------------------------------------- pre-main registration
 *
 * The constructor of a global object runs before main, before any
 * rvcl_init() and before other translation units' dynamic
 * initializers: a registration from there must find a usable registry
 * (this is how an LD_PRELOAD plugin's constructor registers). */

namespace {

class premain_registrant_t {
public:
    premain_registrant_t() : status(RVCL_ERROR_INTERNAL) {
        const rvcl_kernel_desc_t d = make_desc("re04_pre", RVCL_OP_MATMUL);
        status = rvcl_register_kernel(&d, test_create);
    }

    rvcl_status_t status;
};

static premain_registrant_t g_premain_registrant;

} /* namespace */

/* TC-RE-04: registration from a pre-main constructor survives, lazy init
 * accepts the app-owned tuple, and dispatch selects the kernel. Runs
 * first: its dispatch performs the process's first (lazy) init. */
static void tc_re04_premain_registration(void) {
    printf("TC-RE-04 pre-main registration\n");
    CHECK(g_premain_registrant.status == RVCL_SUCCESS,
            "constructor-time registration succeeded");
    CHECK(rvcl_registry_find_factory("re04_pre") != NULL,
            "pre-main entry resolvable by name");
    CHECK(rvcl_init() == RVCL_SUCCESS,
            "init accepts the pre-main app-owned tuple");
    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "built-ins registered alongside the pre-main entry");

    const rvcl_matmul_desc_t req = {
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
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&req, &r) == RVCL_SUCCESS,
            "dispatch succeeds for the pre-registered kernel");
    if (r) {
        CHECK(rvcl_dispatch_kernel_name(r)
                        && strcmp(rvcl_dispatch_kernel_name(r), "re04_pre")
                                == 0,
                "the pre-registered kernel is selected on its tuple");
        rvcl_dispatch_release(r);
    }
    unregister_quiet("re04_pre");
    rvcl_dispatch_cache_clear();
}

/* TC-RE-01: names are unique across operations - a GEMV kernel may not
 * reuse a MATMUL kernel's name (and vice versa). */
static void tc_re01_cross_op_duplicate(void) {
    printf("TC-RE-01 cross-operation duplicate rejected\n");
    const int c0 = rvcl_registry_count();

    const rvcl_kernel_desc_t m = make_desc("re01_x", RVCL_OP_MATMUL);
    CHECK(rvcl_register_kernel(&m, test_create) == RVCL_SUCCESS,
            "matmul registration succeeds");

    const rvcl_kernel_desc_t g = make_desc("re01_x", RVCL_OP_GEMV);
    CHECK(rvcl_register_kernel(&g, test_create)
                    == RVCL_ERROR_ALREADY_REGISTERED,
            "same name on another operation rejected");
    CHECK(rvcl_registry_count() == c0 + 1, "count unchanged by rejection");
    CHECK(find_name_index("re01_x") >= 0, "original entry still iterable");

    unregister_quiet("re01_x");
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-RE-02: capacity is one shared budget across operations - mixed
 * matmul/gemv registrations fill the table to the limit together and
 * overflow is rejected for every operation. */
static void tc_re02_global_capacity_shared(void) {
    printf("TC-RE-02 global capacity shared across operations\n");
    const int c0 = rvcl_registry_count();
    char names[RVCL_REGISTRY_MAX_KERNELS][RVCL_KERNEL_NAME_MAX];
    int added = 0;

    for (int i = 0; c0 + added < RVCL_REGISTRY_MAX_KERNELS; i++) {
        const rvcl_operation_t op
                = (i % 2 == 0) ? RVCL_OP_MATMUL : RVCL_OP_GEMV;
        snprintf(names[added], RVCL_KERNEL_NAME_MAX, "re02_%s_%d",
                op == RVCL_OP_MATMUL ? "m" : "g", i);
        const rvcl_kernel_desc_t d = make_desc(names[added], op);
        if (rvcl_register_kernel(&d, test_create) != RVCL_SUCCESS) break;
        added++;
    }
    CHECK(rvcl_registry_count() == RVCL_REGISTRY_MAX_KERNELS,
            "mixed registrations fill the shared table");

    const rvcl_kernel_desc_t over_m = make_desc("re02_over_m", RVCL_OP_MATMUL);
    CHECK(rvcl_register_kernel(&over_m, test_create)
                    == RVCL_ERROR_OUT_OF_MEMORY,
            "matmul beyond capacity rejected");
    const rvcl_kernel_desc_t over_g = make_desc("re02_over_g", RVCL_OP_GEMV);
    CHECK(rvcl_register_kernel(&over_g, test_create)
                    == RVCL_ERROR_OUT_OF_MEMORY,
            "gemv beyond capacity rejected");
    CHECK(rvcl_registry_count() == RVCL_REGISTRY_MAX_KERNELS,
            "count unchanged by overflow");

    for (int i = 0; i < added; i++)
        unregister_quiet(names[i]);
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-RE-03: the public iteration order is the global registration
 * order even when registrations alternate between operations. */
static void tc_re03_mixed_global_order(void) {
    printf("TC-RE-03 global order under mixed registration\n");
    const int c0 = rvcl_registry_count();

    const rvcl_kernel_desc_t m1 = make_desc("re03_m1", RVCL_OP_MATMUL);
    const rvcl_kernel_desc_t g1 = make_desc("re03_g1", RVCL_OP_GEMV);
    const rvcl_kernel_desc_t m2 = make_desc("re03_m2", RVCL_OP_MATMUL);
    const rvcl_kernel_desc_t g2 = make_desc("re03_g2", RVCL_OP_GEMV);
    CHECK(rvcl_register_kernel(&m1, test_create) == RVCL_SUCCESS
                    && rvcl_register_kernel(&g1, test_create) == RVCL_SUCCESS
                    && rvcl_register_kernel(&m2, test_create) == RVCL_SUCCESS
                    && rvcl_register_kernel(&g2, test_create) == RVCL_SUCCESS,
            "four mixed registrations succeed");
    CHECK(rvcl_registry_count() == c0 + 4, "count accounts for all four");

    const int i_m1 = find_name_index("re03_m1");
    const int i_g1 = find_name_index("re03_g1");
    const int i_m2 = find_name_index("re03_m2");
    const int i_g2 = find_name_index("re03_g2");
    CHECK(i_m1 >= 0 && i_g1 >= 0 && i_m2 >= 0 && i_g2 >= 0,
            "all mixed entries iterable");
    CHECK(i_m1 < i_g1 && i_g1 < i_m2 && i_m2 < i_g2,
            "global order equals the interleaved registration order");

    unregister_quiet("re03_m1");
    unregister_quiet("re03_g1");
    unregister_quiet("re03_m2");
    unregister_quiet("re03_g2");
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-RE-05: the per-tuple fallback invariant is enforced per operation -
 * draining one operation's tuple never weakens another operation's
 * protection. Uses INT4 tuples no builtin occupies. */
static void tc_re05_fallback_cross_op_independent(void) {
    printf("TC-RE-05 fallback invariant independent across operations\n");
    const int c0 = rvcl_registry_count();

    rvcl_kernel_desc_t m_bounded = make_desc("re05_mb", RVCL_OP_MATMUL);
    m_bounded.dtype_a = RVCL_DTYPE_INT4;
    m_bounded.dtype_b = RVCL_DTYPE_INT4;
    m_bounded.max_m = 64;
    rvcl_kernel_desc_t m_unbnd = make_desc("re05_mu", RVCL_OP_MATMUL);
    m_unbnd.dtype_a = RVCL_DTYPE_INT4;
    m_unbnd.dtype_b = RVCL_DTYPE_INT4;
    rvcl_kernel_desc_t g_bounded = make_desc("re05_gb", RVCL_OP_GEMV);
    g_bounded.dtype_a = RVCL_DTYPE_INT4;
    g_bounded.dtype_b = RVCL_DTYPE_INT4;
    g_bounded.max_m = 64;
    rvcl_kernel_desc_t g_unbnd = make_desc("re05_gu", RVCL_OP_GEMV);
    g_unbnd.dtype_a = RVCL_DTYPE_INT4;
    g_unbnd.dtype_b = RVCL_DTYPE_INT4;

    CHECK(rvcl_register_kernel(&m_bounded, test_create) == RVCL_SUCCESS
                    && rvcl_register_kernel(&m_unbnd, test_create)
                            == RVCL_SUCCESS
                    && rvcl_register_kernel(&g_bounded, test_create)
                            == RVCL_SUCCESS
                    && rvcl_register_kernel(&g_unbnd, test_create)
                            == RVCL_SUCCESS,
            "bounded + unbounded pairs registered on both operations");

    CHECK(rvcl_unregister_kernel("re05_mu") == RVCL_ERROR_REGISTRY_INVARIANT,
            "matmul unbounded protected while its bounded entry remains");
    CHECK(rvcl_unregister_kernel("re05_gu") == RVCL_ERROR_REGISTRY_INVARIANT,
            "gemv unbounded protected while its bounded entry remains");
    CHECK(rvcl_registry_find_factory("re05_mu") != NULL
                    && rvcl_registry_find_factory("re05_gu") != NULL,
            "rejected unregisters leave both entries in place");

    /* Fully drain the matmul tuple; the gemv tuple must stay protected. */
    CHECK(rvcl_unregister_kernel("re05_mb") == RVCL_SUCCESS,
            "matmul bounded entry removable");
    CHECK(rvcl_unregister_kernel("re05_mu") == RVCL_SUCCESS,
            "last matmul entry removable (tuple empties)");
    CHECK(rvcl_unregister_kernel("re05_gu") == RVCL_ERROR_REGISTRY_INVARIANT,
            "gemv tuple unaffected by the matmul drain");

    CHECK(rvcl_unregister_kernel("re05_gb") == RVCL_SUCCESS,
            "gemv bounded entry removable");
    CHECK(rvcl_unregister_kernel("re05_gu") == RVCL_SUCCESS,
            "last gemv entry removable (tuple empties)");
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
    rvcl_dispatch_cache_clear();
}

/* TC-RE-06: dispatch routing - a crowd of GEMV entries on another tuple
 * neither disturbs MATMUL selection (fresh miss and cache hit) nor
 * leaks into it, and a GEMV request selects from its own entries. */
static void tc_re06_dispatch_routing(void) {
    printf("TC-RE-06 dispatch routing under a cross-op crowd\n");
    const int c0 = rvcl_registry_count();

    const rvcl_matmul_desc_t mm = {
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
    char baseline[RVCL_KERNEL_NAME_MAX] = ""; /* read only after a pass */
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&mm, &r) == RVCL_SUCCESS,
            "baseline matmul dispatch succeeds");
    if (r) {
        snprintf(baseline, RVCL_KERNEL_NAME_MAX, "%s",
                rvcl_dispatch_kernel_name(r) ? rvcl_dispatch_kernel_name(r)
                                             : "");
        rvcl_dispatch_release(r);
    }

    /* Crowd the registry with INT4 GEMV entries (a tuple no builtin
     * occupies). Equal costs tie-break to the earliest registration. */
    const char *crowd[3] = {"re06_g1", "re06_g2", "re06_g3"};
    int crowded = 1;
    for (int i = 0; i < 3; i++) {
        rvcl_kernel_desc_t d = make_desc(crowd[i], RVCL_OP_GEMV);
        d.dtype_a = RVCL_DTYPE_INT4;
        d.dtype_b = RVCL_DTYPE_INT4;
        if (rvcl_register_kernel(&d, test_create) != RVCL_SUCCESS) crowded = 0;
    }
    CHECK(crowded, "gemv crowd registered");
    rvcl_dispatch_cache_clear();

    /* Same matmul request, fresh selection under the crowd. */
    r = NULL;
    CHECK(rvcl_dispatch_matmul(&mm, &r) == RVCL_SUCCESS,
            "matmul dispatch succeeds under the gemv crowd");
    if (r) {
        CHECK(rvcl_dispatch_kernel_name(r)
                        && strcmp(rvcl_dispatch_kernel_name(r), baseline) == 0,
                "matmul selection unchanged under the gemv crowd");
        rvcl_dispatch_release(r);
    }

    /* A GEMV request routes to its own entries, not the matmul one. */
    const rvcl_op_desc_t od = {
            .gemv = {.m = 16,
                    .k = 16,
                    .dtype_a = RVCL_DTYPE_INT4,
                    .dtype_b = RVCL_DTYPE_INT4,
                    .dtype_acc = RVCL_DTYPE_INT32,
                    .dtype_c = RVCL_DTYPE_INT32,
                    .dtype_bias = RVCL_DTYPE_AUTO,
                    .quant = {}},
    };
    rvcl_dispatch_result_t *rg = NULL;
    CHECK(rvcl_dispatch_op(RVCL_OP_GEMV, &od, &rg) == RVCL_SUCCESS,
            "gemv dispatch succeeds on the crowded tuple");
    if (rg) {
        CHECK(rvcl_dispatch_kernel_name(rg)
                        && strcmp(rvcl_dispatch_kernel_name(rg), "re06_g1")
                                == 0,
                "gemv dispatch selects its own earliest entry");
        rvcl_dispatch_release(rg);
    }

    for (int i = 0; i < 3; i++)
        unregister_quiet(crowd[i]);
    rvcl_dispatch_cache_clear();
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
}

/* TC-RE-07: a cross-operation registry change invalidates cached
 * selections (whole-table generation) and the re-dispatch reselects the
 * same kernel. */
static void tc_re07_cross_op_unregister_cache(void) {
    printf("TC-RE-07 cache invalidation on cross-op change\n");
    const int c0 = rvcl_registry_count();
    rvcl_dispatch_cache_clear();

    const rvcl_matmul_desc_t mm = {
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
    rvcl_dispatch_result_t *r1 = NULL, *r2 = NULL;
    CHECK(rvcl_dispatch_matmul(&mm, &r1) == RVCL_SUCCESS,
            "first dispatch succeeds");
    CHECK(rvcl_dispatch_matmul(&mm, &r2) == RVCL_SUCCESS,
            "second dispatch succeeds");
    CHECK(r1 && r1 == r2, "cache hit returns the same result object");
    char baseline[RVCL_KERNEL_NAME_MAX];
    snprintf(baseline, RVCL_KERNEL_NAME_MAX, "%s",
            rvcl_dispatch_kernel_name(r1) ? rvcl_dispatch_kernel_name(r1) : "");
    rvcl_dispatch_release(r1);
    rvcl_dispatch_release(r2);

    /* The GEMV mutation bumps the global generation and clears the
     * cache; the next matmul dispatch reselects from scratch. */
    rvcl_kernel_desc_t g = make_desc("re07_g", RVCL_OP_GEMV);
    g.dtype_a = RVCL_DTYPE_INT4;
    g.dtype_b = RVCL_DTYPE_INT4;
    CHECK(rvcl_register_kernel(&g, test_create) == RVCL_SUCCESS,
            "cross-op gemv registration succeeds");
    CHECK(rvcl_unregister_kernel("re07_g") == RVCL_SUCCESS,
            "cross-op gemv unregister succeeds");

    rvcl_dispatch_result_t *r3 = NULL;
    CHECK(rvcl_dispatch_matmul(&mm, &r3) == RVCL_SUCCESS,
            "redispatch succeeds after the cross-op change");
    if (r3) {
        CHECK(rvcl_dispatch_kernel_name(r3)
                        && strcmp(rvcl_dispatch_kernel_name(r3), baseline) == 0,
                "selection stable after the cross-op invalidation");
        rvcl_dispatch_release(r3);
    }
    CHECK(rvcl_registry_count() == c0, "cleanup restores count");
    rvcl_dispatch_cache_clear();
}

int main(void) {
    tc_re04_premain_registration(); /* first: drives the lazy init */
    tc_re01_cross_op_duplicate();
    tc_re02_global_capacity_shared();
    tc_re03_mixed_global_order();
    tc_re05_fallback_cross_op_independent();
    tc_re06_dispatch_routing();
    tc_re07_cross_op_unregister_cache();

    printf(g_failures ? "UNIT-REGISTRY-EXPAND: %d failure(s)\n"
                      : "UNIT-REGISTRY-EXPAND: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
