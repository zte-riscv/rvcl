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
 * unit_dispatch.cpp — dispatch unit tests
 *
 * Covers the six-stage filter pipeline and the selection rules: built-in
 * selection per build and host, datatype AUTO resolution, required-feature
 * and VLEN-tier filtering, shape bounds (inclusive), validate rejection,
 * cost-driven choice with registration order as the tie-break, cache
 * behavior (full-field key, invalidation on registry change, hit reuse),
 * failure semantics, the dispatch/execute boundary, and concurrent
 * dispatch under registry churn.
 *
 * Host-dependent expectations branch on what the running snapshot and
 * build configuration actually provide, so the binary is runnable
 * everywhere. The --inject mode (own ctest entry, fresh process) uses the
 * test-only capability override to exercise non-host VLEN tiers and
 * feature sets.
 *
 * Custom kernels use datatype tuples no built-in occupies and are
 * unregistered at the end of each case; the dispatch cache is cleared
 * between cases so they stay independent.
 ******************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <atomic>

#include "golden.hpp"
#include "rvcl/rvcl.h"

#include "../src/runtime/capability_internal.h"
#include "../src/runtime/dispatch_internal.h"

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
 * Counting kernels: validate/configure/run increment process-wide
 * counters so a case can observe exactly which lifecycle calls happened.
 * Factories and vtables are parameterized by the validate policy. */

static std::atomic<int> g_validate_calls(0);
static std::atomic<int> g_configure_calls(0);
static std::atomic<int> g_run_calls(0);

static rvcl_status_t test_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    g_validate_calls.fetch_add(1);
    return RVCL_SUCCESS;
}

/* Rejects requests deeper than the K limit; used for the stage-6 case. */
static rvcl_status_t test_validate_k32(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap;
    g_validate_calls.fetch_add(1);
    return desc->matmul.k > 32 ? RVCL_UNSUPPORTED : RVCL_SUCCESS;
}

static rvcl_status_t test_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    (void)k;
    (void)desc;
    (void)ws_req;
    g_configure_calls.fetch_add(1);
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t test_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *range) {
    (void)k;
    (void)pack;
    (void)range;
    g_run_calls.fetch_add(1);
    return RVCL_SUCCESS;
}

/* The instance header is released by rvcl_kernel_destroy(); the vtable
 * only drops state the implementation itself allocated. */
static void test_destroy(rvcl_kernel_t *k) {
    (void)k;
}

static const rvcl_kernel_vtable_t g_vtable = {
        .validate = test_validate,
        .configure = test_configure,
        .pack_a = NULL,
        .pack_b = NULL,
        .run = test_run,
        .destroy = test_destroy,
};

static const rvcl_kernel_vtable_t g_vtable_k32 = {
        .validate = test_validate_k32,
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
    (*out)->vt = &g_vtable;
    (*out)->desc = *desc;
    return RVCL_SUCCESS;
}

static rvcl_status_t test_create_k32(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    *out = (rvcl_kernel_t *)calloc(1, sizeof(rvcl_kernel_t));
    if (!*out) return RVCL_ERROR_OUT_OF_MEMORY;
    (*out)->vt = &g_vtable_k32;
    (*out)->desc = *desc;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------------ cost stubs */

static uint64_t cost_zero(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return 0;
}

static uint64_t cost_one(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return 1;
}

static uint64_t cost_five(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return 5;
}

static uint64_t cost_hundred(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return 100;
}

/* --------------------------------------------------------------- helpers */

static rvcl_kernel_desc_t make_desc(const char *name, rvcl_dtype_t a,
        rvcl_dtype_t b, rvcl_dtype_t acc, rvcl_dtype_t c) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));
    /* precision-bounded copy: -Wformat-truncation is fatal under Werror */
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%.*s", RVCL_KERNEL_NAME_MAX - 1,
            name);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = a;
    d.dtype_b = b;
    d.dtype_acc = acc;
    d.dtype_c = c;
    d.quant_support = RVCL_QUANT_NONE;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = cost_zero;
    return d;
}

static rvcl_matmul_desc_t make_request(int64_t m, int64_t n, int64_t k,
        rvcl_dtype_t a, rvcl_dtype_t b, rvcl_dtype_t acc, rvcl_dtype_t c) {
    rvcl_matmul_desc_t d;
    memset(&d, 0, sizeof(d));
    d.m = m;
    d.n = n;
    d.k = k;
    d.dtype_a = a;
    d.dtype_b = b;
    d.dtype_acc = acc;
    d.dtype_c = c;
    d.dtype_bias = RVCL_DTYPE_AUTO;
    return d;
}

static int register_desc(const rvcl_kernel_desc_t *d) {
    return rvcl_register_kernel(d, test_create) == RVCL_SUCCESS;
}

static void unregister_quiet(const char *name) {
    rvcl_unregister_kernel(name);
}

static int find_index(const char *name) {
    const int n = rvcl_registry_count();
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (kd && strcmp(kd->name, name) == 0) return i;
    }
    return -1;
}

static int registry_has(const char *name) {
    return find_index(name) >= 0;
}

/* VLEN tier declared by the named entry; -1 when absent. */
static int tier_of_name(const char *name) {
    const int idx = find_index(name);
    return idx < 0 ? -1 : (int)rvcl_registry_get(idx)->vlen_tier;
}

static int name_ends_with(const char *name, const char *suffix) {
    if (!name) return 0;
    const size_t n = strlen(name);
    const size_t s = strlen(suffix);
    return n >= s && strcmp(name + n - s, suffix) == 0;
}

static int vector_build(void) {
    return rvcl_registry_find_factory("rvcl_matmul_f32_rvv") != NULL;
}

/* Portable generic entry, registered on builds without vector
 * compilation. */
static int portable_f32_registered(void) {
    return rvcl_registry_find_factory("rvcl_matmul_f32_rvv_c") != NULL;
}

static int host_has_rvv(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

/* Descriptor requires no capability: it survives an empty snapshot. */
static int featureless_by_name(const char *name) {
    const int idx = find_index(name);
    return idx >= 0 && rvcl_registry_get(idx)->num_required_features == 0;
}

static int required_by_name(const char *name, rvcl_feature_id_t id) {
    const int idx = find_index(name);
    if (idx < 0) return 0;
    const rvcl_kernel_desc_t *kd = rvcl_registry_get(idx);
    for (int i = 0; i < kd->num_required_features; i++)
        if (kd->required_features[i] == id) return 1;
    return 0;
}

static int unbounded_by_name(const char *name) {
    const int idx = find_index(name);
    if (idx < 0) return 0;
    const rvcl_kernel_desc_t *kd = rvcl_registry_get(idx);
    return kd->max_m == 0 && kd->max_n == 0 && kd->max_k == 0;
}

/* --------------------------------------------------- build tier vocabulary */

static int g_enabled_tiers[4];
static int g_num_enabled_tiers;

static void collect_enabled_tiers(void) {
#if defined(RVCL_RVV_TIERS_1024)
    g_enabled_tiers[g_num_enabled_tiers++] = 1024;
#endif
#if defined(RVCL_RVV_TIERS_512)
    g_enabled_tiers[g_num_enabled_tiers++] = 512;
#endif
#if defined(RVCL_RVV_TIERS_256)
    g_enabled_tiers[g_num_enabled_tiers++] = 256;
#endif
#if defined(RVCL_RVV_TIERS_128)
    g_enabled_tiers[g_num_enabled_tiers++] = 128;
#endif
}

/* Largest tier the host width supports that the build also registered;
 * 0 when no dedicated tier qualifies (tier-0 entries serve). */
static int expected_tier(uint32_t vlen) {
    int host = 0;
    if (vlen >= 1024)
        host = 1024;
    else if (vlen >= 512)
        host = 512;
    else if (vlen >= 256)
        host = 256;
    else if (vlen >= 128)
        host = 128;
    int best = 0;
    for (int i = 0; i < g_num_enabled_tiers; i++)
        if (g_enabled_tiers[i] <= host && g_enabled_tiers[i] > best)
            best = g_enabled_tiers[i];
    return best;
}

/* --------------------------------------------------------- f32 selection */

/* TC-AC3-01: an RVV host with a vector build selects an RVV kernel at
 * the highest host-supported dedicated tier. */
static void tc01_rvv_selection(void) {
    printf("TC-AC3-01 f32 selection on an RVV host\n");
    if (!vector_build() || !host_has_rvv()) {
        printf("  skip: needs a vector build on an RVV host\n");
        return;
    }
    rvcl_dispatch_cache_clear();
    const rvcl_matmul_desc_t d = make_request(64, 64, 64, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "f32 dispatch succeeds");
    const char *name = r ? rvcl_dispatch_kernel_name(r) : NULL;
    CHECK(name != NULL && !name_ends_with(name, "reference"),
            "selected kernel is not the reference fallback");
    if (name) {
        const rvcl_feature_t *rvv
                = rvcl_capability_get(rvcl_get_capability(), RVCL_FEATURE_RVV);
        const int want = expected_tier(rvv ? rvv->value : 1);
        const int got = tier_of_name(name);
        char msg[128];
        snprintf(msg, sizeof(msg),
                "selected tier %d equals the highest supported tier (%d)", got,
                want);
        CHECK(got == want, msg);
        CHECK(required_by_name(name, RVCL_FEATURE_RVV),
                "selected kernel requires RVV");
    }
    rvcl_dispatch_release(r);
}

/* TC-AC3-02: without a usable vector path, f32 selects the portable
 * generic entry when the build provides one, otherwise the featureless
 * reference fallback. */
static void tc02_reference_selection(void) {
    printf("TC-AC3-02 f32 selection without a usable vector path\n");
    if (vector_build() && host_has_rvv()) {
        printf("  skip: vector build on an RVV host selects RVV\n");
        return;
    }
    rvcl_dispatch_cache_clear();
    const rvcl_matmul_desc_t d = make_request(64, 64, 64, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "f32 dispatch succeeds");
    const char *name = r ? rvcl_dispatch_kernel_name(r) : NULL;
    if (portable_f32_registered()) {
        CHECK(name && strcmp(name, "rvcl_matmul_f32_rvv_c") == 0,
                "portable entry is preferred over the reference fallback");
    } else {
        CHECK(name && name_ends_with(name, "reference"),
                "selected kernel is the reference fallback");
    }
    rvcl_dispatch_release(r);
}

/* TC-AC3-02b: a quantized request must not fall through to a kernel that
 * would compute it as if unquantized. No built-in kernel declares
 * quantization support, so a built-in-only registry rejects the request
 * instead of selecting the reference backend. */
static void tc02b_quantized_rejected(void) {
    printf("TC-AC3-02b quantized built-in request is rejected\n");
    rvcl_dispatch_cache_clear();
    rvcl_matmul_desc_t d = make_request(16, 16, 16, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    d.quant.enabled = 1;
    d.quant.scale_a = 0.5f;
    d.quant.scale_b = 0.25f;
    d.quant.scale_c = 1.0f;

    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_UNSUPPORTED,
            "quantized f32 request -> UNSUPPORTED");
    CHECK(r == NULL, "no result (reference included) is selected");

    rvcl_dispatch_result_t *r2 = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r2) == RVCL_UNSUPPORTED,
            "repeated quantized request stays unsupported");
    CHECK(r2 == NULL, "no result leaked on the repeated request");
}

/* TC-AC3-04: repeated dispatch is deterministic and a cache hit reuses
 * the configured instance without re-validating or re-configuring. */
static void tc04_deterministic_cache(void) {
    printf("TC-AC3-04 deterministic dispatch and cache reuse\n");
    rvcl_kernel_desc_t kd = make_desc("t04_k", RVCL_DTYPE_INT8, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    CHECK(register_desc(&kd), "counting kernel registered");
    rvcl_dispatch_cache_clear();
    g_validate_calls.store(0);
    g_configure_calls.store(0);
    g_run_calls.store(0);

    const rvcl_matmul_desc_t d = make_request(16, 16, 16, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_dispatch_result_t *r1 = NULL, *r2 = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r1) == RVCL_SUCCESS,
            "first dispatch succeeds");
    CHECK(rvcl_dispatch_kernel_name(r1)
                    && strcmp(rvcl_dispatch_kernel_name(r1), "t04_k") == 0,
            "first dispatch selects the counting kernel");
    CHECK(g_validate_calls.load() == 1, "first dispatch validates once");
    CHECK(g_configure_calls.load() == 1, "first dispatch configures once");
    CHECK(g_run_calls.load() == 0, "dispatch never runs the kernel");

    CHECK(rvcl_dispatch_matmul(&d, &r2) == RVCL_SUCCESS,
            "second dispatch succeeds");
    CHECK(r1 == r2, "cache hit returns the same result object");
    CHECK(g_validate_calls.load() == 1, "cache hit skips validate");
    CHECK(g_configure_calls.load() == 1, "cache hit skips configure");

    rvcl_dispatch_release(r1);
    rvcl_dispatch_release(r2);
    unregister_quiet("t04_k");
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-05: hit and miss counters are readable and advance. */
static void tc05_cache_counters(void) {
    printf("TC-AC3-05 hit/miss counters\n");
    rvcl_kernel_desc_t kd = make_desc("t05_k", RVCL_DTYPE_INT8, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    CHECK(register_desc(&kd), "counting kernel registered");
    rvcl_dispatch_cache_clear();

    rvcl_matmul_desc_t d1 = make_request(16, 16, 16, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_matmul_desc_t d2 = d1;
    d2.m = 17;
    rvcl_dispatch_result_t *r = NULL;

    int hits = rvcl_dispatch_cache_hits();
    int misses = rvcl_dispatch_cache_misses();
    CHECK(rvcl_dispatch_matmul(&d1, &r) == RVCL_SUCCESS, "first request ok");
    rvcl_dispatch_release(r);
    r = NULL;
    CHECK(rvcl_dispatch_cache_misses() > misses, "first request is a miss");

    hits = rvcl_dispatch_cache_hits();
    CHECK(rvcl_dispatch_matmul(&d1, &r) == RVCL_SUCCESS, "repeat request ok");
    rvcl_dispatch_release(r);
    r = NULL;
    CHECK(rvcl_dispatch_cache_hits() > hits, "repeat request is a hit");

    misses = rvcl_dispatch_cache_misses();
    CHECK(rvcl_dispatch_matmul(&d2, &r) == RVCL_SUCCESS, "variant request ok");
    rvcl_dispatch_release(r);
    CHECK(rvcl_dispatch_cache_misses() > misses, "variant request is a miss");

    unregister_quiet("t05_k");
    rvcl_dispatch_cache_clear();
}

/* Dispatch one request, require a miss, then repeat and require a hit. */
static void expect_miss_then_hit(
        const rvcl_matmul_desc_t *d, const char *label) {
    rvcl_dispatch_result_t *r1 = NULL, *r2 = NULL;
    char msg[160];

    const int misses0 = rvcl_dispatch_cache_misses();
    snprintf(msg, sizeof(msg), "%s: first dispatch succeeds", label);
    CHECK(rvcl_dispatch_matmul(d, &r1) == RVCL_SUCCESS, msg);
    snprintf(msg, sizeof(msg), "%s: first dispatch is a miss", label);
    CHECK(rvcl_dispatch_cache_misses() > misses0, msg);

    const int hits0 = rvcl_dispatch_cache_hits();
    snprintf(msg, sizeof(msg), "%s: repeat dispatch succeeds", label);
    CHECK(rvcl_dispatch_matmul(d, &r2) == RVCL_SUCCESS, msg);
    snprintf(msg, sizeof(msg), "%s: repeat returns the cached object", label);
    CHECK(r1 == r2, msg);
    snprintf(msg, sizeof(msg), "%s: repeat dispatch is a hit", label);
    CHECK(rvcl_dispatch_cache_hits() > hits0, msg);

    rvcl_dispatch_release(r1);
    rvcl_dispatch_release(r2);
}

/* TC-AC3-06: every request field participates in the cache key. */
static void tc06_cache_key(void) {
    printf("TC-AC3-06 full-field cache key\n");
    rvcl_kernel_desc_t kd = make_desc("t06_k", RVCL_DTYPE_INT8, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    /* The quantized key variants below need a kernel that declares
     * support, or the stage-2 quantization filter rejects them. */
    kd.quant_support = RVCL_QUANT_PER_TENSOR;
    /* One tuple per dtype variant so each changed request still finds a
     * candidate; a miss on these keys is therefore observable. */
    rvcl_kernel_desc_t da = make_desc("t06_a", RVCL_DTYPE_UINT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_kernel_desc_t db = make_desc("t06_b", RVCL_DTYPE_INT8,
            RVCL_DTYPE_UINT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_kernel_desc_t dc = make_desc("t06_c", RVCL_DTYPE_INT8, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT32, RVCL_DTYPE_FP32);
    CHECK(register_desc(&kd) && register_desc(&da) && register_desc(&db)
                    && register_desc(&dc),
            "counting kernels registered");
    rvcl_dispatch_cache_clear();

    const rvcl_matmul_desc_t base = make_request(16, 16, 16, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    expect_miss_then_hit(&base, "base request");

    rvcl_matmul_desc_t v = base;
    v.m = 17;
    expect_miss_then_hit(&v, "m variant");
    v = base;
    v.n = 17;
    expect_miss_then_hit(&v, "n variant");
    v = base;
    v.k = 17;
    expect_miss_then_hit(&v, "k variant");
    v = base;
    v.dtype_a = RVCL_DTYPE_UINT8;
    expect_miss_then_hit(&v, "dtype_a variant");
    v = base;
    v.dtype_b = RVCL_DTYPE_UINT8;
    expect_miss_then_hit(&v, "dtype_b variant");
    v = base;
    v.dtype_c = RVCL_DTYPE_FP32;
    expect_miss_then_hit(&v, "dtype_c variant");
    v = base;
    v.dtype_bias = RVCL_DTYPE_INT32;
    expect_miss_then_hit(&v, "bias dtype variant");
    v = base;
    v.dtype_acc = RVCL_DTYPE_AUTO; /* resolves to INT32, raw value differs */
    expect_miss_then_hit(&v, "AUTO accumulator variant");

    /* Quantization scalars are part of the key. The test kernel accepts
     * quantized requests, so the key coverage is observable even though
     * the quantized compute path is out of scope. */
    v = base;
    v.quant.enabled = 1;
    expect_miss_then_hit(&v, "quant enabled variant");
    const rvcl_matmul_desc_t q = v;
    v = q;
    v.quant.scale_a = 0.5f;
    expect_miss_then_hit(&v, "quant scale_a variant");
    v = q;
    v.quant.scale_b = 0.5f;
    expect_miss_then_hit(&v, "quant scale_b variant");
    v = q;
    v.quant.scale_c = 0.5f;
    expect_miss_then_hit(&v, "quant scale_c variant");
    v = q;
    v.quant.zp_a = 3;
    expect_miss_then_hit(&v, "quant zp_a variant");
    v = q;
    v.quant.zp_b = 3;
    expect_miss_then_hit(&v, "quant zp_b variant");
    v = q;
    v.quant.zp_c = 3;
    expect_miss_then_hit(&v, "quant zp_c variant");

    unregister_quiet("t06_k");
    unregister_quiet("t06_a");
    unregister_quiet("t06_b");
    unregister_quiet("t06_c");
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-07: registry changes invalidate the cache and a removed kernel
 * is never selected again. */
static void tc07_registry_change(void) {
    printf("TC-AC3-07 cache invalidation on registry change\n");
    rvcl_kernel_desc_t f1 = make_desc("t07_f1", RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    f1.cost_fn = cost_five;
    rvcl_kernel_desc_t f2 = f1;
    snprintf(f2.name, RVCL_KERNEL_NAME_MAX, "%s", "t07_f2");
    f2.cost_fn = cost_hundred;
    CHECK(register_desc(&f1) && register_desc(&f2),
            "two cost-differentiated kernels registered");
    rvcl_dispatch_cache_clear();

    const rvcl_matmul_desc_t d = make_request(16, 16, 16, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS, "dispatch succeeds");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && strcmp(rvcl_dispatch_kernel_name(r), "t07_f1") == 0,
            "lower-cost kernel selected");
    rvcl_dispatch_release(r);
    r = NULL;

    /* (a) A newly registered cheaper kernel must win after invalidation. */
    rvcl_kernel_desc_t f0 = f1;
    snprintf(f0.name, RVCL_KERNEL_NAME_MAX, "%s", "t07_f0");
    f0.cost_fn = cost_one;
    int misses = rvcl_dispatch_cache_misses();
    CHECK(register_desc(&f0), "cheaper kernel registered");
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch after registration succeeds");
    CHECK(rvcl_dispatch_cache_misses() > misses,
            "registration invalidates the cache");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && strcmp(rvcl_dispatch_kernel_name(r), "t07_f0") == 0,
            "new cheapest kernel selected");
    rvcl_dispatch_release(r);
    r = NULL;

    /* (b) Removing it must invalidate again and exclude it from results. */
    misses = rvcl_dispatch_cache_misses();
    CHECK(rvcl_unregister_kernel("t07_f0") == RVCL_SUCCESS,
            "cheapest kernel unregistered");
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch after unregistration succeeds");
    CHECK(rvcl_dispatch_cache_misses() > misses,
            "unregistration invalidates the cache");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && strcmp(rvcl_dispatch_kernel_name(r), "t07_f0") != 0,
            "unregistered kernel is not selected");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && registry_has(rvcl_dispatch_kernel_name(r)),
            "selected kernel exists in the registry");
    rvcl_dispatch_release(r);

    unregister_quiet("t07_f1");
    unregister_quiet("t07_f2");
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-08: no surviving candidate returns UNSUPPORTED; failures count
 * as misses and never enter the cache. */
static void tc08_no_candidate(void) {
    printf("TC-AC3-08 no-candidate semantics\n");
    rvcl_dispatch_cache_clear();

    const rvcl_matmul_desc_t bf16 = make_request(16, 16, 16, RVCL_DTYPE_BF16,
            RVCL_DTYPE_BF16, RVCL_DTYPE_AUTO, RVCL_DTYPE_BF16);
    rvcl_dispatch_result_t *r = NULL;
    int misses = rvcl_dispatch_cache_misses();
    CHECK(rvcl_dispatch_matmul(&bf16, &r) == RVCL_UNSUPPORTED,
            "tuple without kernels -> UNSUPPORTED");
    CHECK(r == NULL, "no result leaked on rejection");
    CHECK(rvcl_dispatch_cache_misses() > misses,
            "unsupported request counts as a miss");

    misses = rvcl_dispatch_cache_misses();
    CHECK(rvcl_dispatch_matmul(&bf16, &r) == RVCL_UNSUPPORTED,
            "repeated unsupported request -> UNSUPPORTED");
    CHECK(rvcl_dispatch_cache_misses() > misses,
            "failed request is never cached");

    /* (b) An entry of another operation type cannot serve a matmul. */
    rvcl_kernel_desc_t gemv = make_desc("t08_gemv", RVCL_DTYPE_INT4,
            RVCL_DTYPE_INT4, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    gemv.operation = RVCL_OP_GEMV;
    CHECK(register_desc(&gemv), "GEMV entry registered");
    const rvcl_matmul_desc_t int4 = make_request(16, 16, 16, RVCL_DTYPE_INT4,
            RVCL_DTYPE_INT4, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    CHECK(rvcl_dispatch_matmul(&int4, &r) == RVCL_UNSUPPORTED,
            "operation mismatch leaves no candidate");
    rvcl_unregister_kernel("t08_gemv");

    /* (c) An invalid shape is an argument error, not a dispatch miss. */
    rvcl_matmul_desc_t bad = int4;
    bad.m = -1;
    CHECK(rvcl_dispatch_matmul(&bad, &r) == RVCL_ERROR_INVALID_ARGUMENT,
            "non-positive shape -> INVALID_ARGUMENT");
}

/* TC-AC3-09: illegal datatype tuples and AUTO misuse get the dedicated
 * error code; a legal mixed input has no kernel and is UNSUPPORTED. */
static void tc09_illegal_tuples(void) {
    printf("TC-AC3-09 illegal datatype tuples\n");
    rvcl_dispatch_cache_clear();
    const rvcl_matmul_desc_t base = make_request(16, 16, 16, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);

    struct {
        const char *label;
        rvcl_matmul_desc_t d;
    } cases[8];
    int nc = 0;
    rvcl_matmul_desc_t v = base;
    v.dtype_a = RVCL_DTYPE_FP8_E4M3;
    cases[nc].label = "reserved dtype value";
    cases[nc++].d = v;
    v = base;
    v.dtype_a = (rvcl_dtype_t)99;
    cases[nc].label = "out-of-range dtype value";
    cases[nc++].d = v;
    v = base;
    v.dtype_a = RVCL_DTYPE_AUTO;
    cases[nc].label = "AUTO in A";
    cases[nc++].d = v;
    v = base;
    v.dtype_b = RVCL_DTYPE_AUTO;
    cases[nc].label = "AUTO in B";
    cases[nc++].d = v;
    v = base;
    v.dtype_c = RVCL_DTYPE_AUTO;
    cases[nc].label = "AUTO in C";
    cases[nc++].d = v;
    v = base;
    v.dtype_acc = (rvcl_dtype_t)99;
    cases[nc].label = "out-of-range accumulator dtype";
    cases[nc++].d = v;
    v = base;
    v.dtype_bias = (rvcl_dtype_t)99;
    cases[nc].label = "out-of-range bias dtype";
    cases[nc++].d = v;

    for (int i = 0; i < nc; i++) {
        rvcl_dispatch_result_t *r = NULL;
        char msg[128];
        snprintf(msg, sizeof(msg), "%s -> UNSUPPORTED_DTYPE_COMBO",
                cases[i].label);
        CHECK(rvcl_dispatch_matmul(&cases[i].d, &r)
                        == RVCL_UNSUPPORTED_DTYPE_COMBO,
                msg);
        CHECK(r == NULL, "no result leaked on a rejected tuple");
    }

    /* Legal enums, but the mixed tuple has no kernel: not a tuple error. */
    const rvcl_matmul_desc_t mixed = make_request(16, 16, 16, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP16, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&mixed, &r) == RVCL_UNSUPPORTED,
            "mixed A/B input -> UNSUPPORTED");
    CHECK(r == NULL, "no result leaked for a mixed input");

    /* AUTO in the bias slot is the legal "no bias" marker. */
    const rvcl_matmul_desc_t auto_bias = make_request(16, 16, 16,
            RVCL_DTYPE_FP32, RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *rb = NULL;
    CHECK(rvcl_dispatch_matmul(&auto_bias, &rb) == RVCL_SUCCESS,
            "AUTO bias slot is accepted");
    CHECK(rb != NULL, "AUTO-bias request returns a selection");
    rvcl_dispatch_release(rb);
}

/* TC-AC3-10: AUTO is legal only in the accumulator slot and resolves by
 * the input type habit rule. */
static void tc10_auto_resolution(void) {
    printf("TC-AC3-10 AUTO accumulator resolution\n");

    /* FP32 inputs: the only fp32 matmul entries accumulate in FP32, so
     * success proves the AUTO accumulator resolved to FP32. */
    rvcl_dispatch_cache_clear();
    const rvcl_matmul_desc_t f32 = make_request(16, 16, 16, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *r32 = NULL;
    CHECK(rvcl_dispatch_matmul(&f32, &r32) == RVCL_SUCCESS,
            "f32 inputs: AUTO resolves to an existing FP32 tuple");
    rvcl_dispatch_release(r32);

    struct group_t {
        const char *label;
        rvcl_dtype_t a, b, c;
        rvcl_dtype_t expect_acc;
        const char *i32_name;
        const char *f32_name;
    } groups[3] = {
            {"INT8 inputs", RVCL_DTYPE_INT8, RVCL_DTYPE_INT8, RVCL_DTYPE_INT32,
                    RVCL_DTYPE_INT32, "t10_i8_i32", "t10_i8_f32"},
            {"UINT8 inputs", RVCL_DTYPE_UINT8, RVCL_DTYPE_UINT8,
                    RVCL_DTYPE_INT32, RVCL_DTYPE_INT32, "t10_u8_i32",
                    "t10_u8_f32"},
            {"BF16 inputs", RVCL_DTYPE_BF16, RVCL_DTYPE_BF16, RVCL_DTYPE_BF16,
                    RVCL_DTYPE_FP32, "t10_bf_i32", "t10_bf_f32"},
    };

    for (size_t g = 0; g < sizeof(groups) / sizeof(groups[0]); g++) {
        rvcl_kernel_desc_t i32 = make_desc(groups[g].i32_name, groups[g].a,
                groups[g].b, RVCL_DTYPE_INT32, groups[g].c);
        rvcl_kernel_desc_t f32 = make_desc(groups[g].f32_name, groups[g].a,
                groups[g].b, RVCL_DTYPE_FP32, groups[g].c);
        CHECK(register_desc(&i32) && register_desc(&f32),
                "both accumulator variants registered");

        const rvcl_matmul_desc_t d = make_request(16, 16, 16, groups[g].a,
                groups[g].b, RVCL_DTYPE_AUTO, groups[g].c);
        rvcl_dispatch_result_t *r = NULL;
        CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
                "AUTO request dispatches");
        const char *name = r ? rvcl_dispatch_kernel_name(r) : NULL;
        const char *want = groups[g].expect_acc == RVCL_DTYPE_INT32
                ? groups[g].i32_name
                : groups[g].f32_name;
        char msg[128];
        snprintf(msg, sizeof(msg), "%s: AUTO resolves to the %s accumulator",
                groups[g].label,
                groups[g].expect_acc == RVCL_DTYPE_INT32 ? "INT32" : "FP32");
        CHECK(name && strcmp(name, want) == 0, msg);
        rvcl_dispatch_release(r);

        unregister_quiet(groups[g].i32_name);
        unregister_quiet(groups[g].f32_name);
        rvcl_dispatch_cache_clear();
    }
}

/* TC-AC3-13: every dimension is checked independently against its
 * inclusive bounds. */
static void tc13_shape_bounds(void) {
    printf("TC-AC3-13 shape bounds (inclusive, per dimension)\n");
    const rvcl_dtype_t a = RVCL_DTYPE_INT4;
    const rvcl_dtype_t acc = RVCL_DTYPE_INT32;

    rvcl_kernel_desc_t xm = make_desc("t13_m", a, a, acc, acc);
    xm.min_m = 4;
    xm.max_m = 8;
    rvcl_kernel_desc_t xn = make_desc("t13_n", a, a, acc, acc);
    xn.min_n = 4;
    xn.max_n = 8;
    rvcl_kernel_desc_t xk = make_desc("t13_k", a, a, acc, acc);
    xk.min_k = 4;
    xk.max_k = 8;
    rvcl_kernel_desc_t fb = make_desc("t13_f", a, a, acc, acc);
    CHECK(register_desc(&xm) && register_desc(&xn) && register_desc(&xk)
                    && register_desc(&fb),
            "bounded variants and fallback registered");
    rvcl_dispatch_cache_clear();

    struct {
        int64_t value;
        const char *want;
    } scan[4] = {{3, "t13_f"}, {4, "t13_m"}, {8, "t13_m"}, {9, "t13_f"}};
    for (int i = 0; i < 4; i++) {
        const rvcl_matmul_desc_t d
                = make_request(scan[i].value, 16, 16, a, a, acc, acc);
        rvcl_dispatch_result_t *r = NULL;
        CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
                "M-bound request dispatches");
        char msg[96];
        snprintf(msg, sizeof(msg), "m=%ld selects %s", (long)scan[i].value,
                scan[i].want);
        CHECK(rvcl_dispatch_kernel_name(r)
                        && strcmp(rvcl_dispatch_kernel_name(r), scan[i].want)
                                == 0,
                msg);
        rvcl_dispatch_release(r);
    }

    const int64_t nvals[4] = {3, 4, 8, 9};
    const char *want_n[4] = {"t13_f", "t13_n", "t13_n", "t13_f"};
    for (int i = 0; i < 4; i++) {
        const int64_t nv = nvals[i];
        const rvcl_matmul_desc_t d = make_request(16, nv, 16, a, a, acc, acc);
        rvcl_dispatch_result_t *r = NULL;
        CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
                "N-bound request dispatches");
        char msg[96];
        snprintf(msg, sizeof(msg), "n=%ld selects %s", (long)nv, want_n[i]);
        CHECK(rvcl_dispatch_kernel_name(r)
                        && strcmp(rvcl_dispatch_kernel_name(r), want_n[i]) == 0,
                msg);
        rvcl_dispatch_release(r);
    }

    const int64_t kvals[4] = {3, 4, 8, 9};
    const char *want_k[4] = {"t13_f", "t13_k", "t13_k", "t13_f"};
    for (int i = 0; i < 4; i++) {
        const int64_t kv = kvals[i];
        const rvcl_matmul_desc_t d = make_request(16, 16, kv, a, a, acc, acc);
        rvcl_dispatch_result_t *r = NULL;
        CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
                "K-bound request dispatches");
        char msg[96];
        snprintf(msg, sizeof(msg), "k=%ld selects %s", (long)kv, want_k[i]);
        CHECK(rvcl_dispatch_kernel_name(r)
                        && strcmp(rvcl_dispatch_kernel_name(r), want_k[i]) == 0,
                msg);
        rvcl_dispatch_release(r);
    }

    unregister_quiet("t13_m");
    unregister_quiet("t13_n");
    unregister_quiet("t13_k");
    unregister_quiet("t13_f");
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-14: a validate rejection eliminates only that candidate and is
 * side-effect free across repeated dispatches. */
static void tc14_validate_veto(void) {
    printf("TC-AC3-14 validate veto at stage 6\n");
    const rvcl_dtype_t a = RVCL_DTYPE_INT4;
    const rvcl_dtype_t acc = RVCL_DTYPE_INT32;
    rvcl_kernel_desc_t yes = make_desc("t14_y", a, a, acc, acc);
    rvcl_kernel_desc_t fb = make_desc("t14_f", a, a, acc, acc);
    CHECK(rvcl_register_kernel(&yes, test_create_k32) == RVCL_SUCCESS
                    && register_desc(&fb),
            "rejecting kernel and fallback registered");
    rvcl_dispatch_cache_clear();

    const int64_t ks[3] = {32, 33, 32};
    const char *want[3] = {"t14_y", "t14_f", "t14_y"};
    for (int i = 0; i < 3; i++) {
        /* Force a full pipeline run per iteration: otherwise the third
         * request would be a cache hit and would not exercise the
         * repeated-dispatch side-effect freedom the case is about. */
        rvcl_dispatch_cache_clear();
        const rvcl_matmul_desc_t d
                = make_request(16, 16, ks[i], a, a, acc, acc);
        rvcl_dispatch_result_t *r = NULL;
        CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
                "validate-veto request dispatches");
        char msg[96];
        snprintf(msg, sizeof(msg), "k=%ld selects %s", (long)ks[i], want[i]);
        CHECK(rvcl_dispatch_kernel_name(r)
                        && strcmp(rvcl_dispatch_kernel_name(r), want[i]) == 0,
                msg);
        rvcl_dispatch_release(r);
    }

    unregister_quiet("t14_y");
    unregister_quiet("t14_f");
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-15: an oversized request degrades to an unbounded kernel and
 * the surviving built-in path still computes the golden result. */
static void tc15_shape_fallback(void) {
    printf("TC-AC3-15 oversized shape falls back and computes\n");
    /* Use a tuple no built-in occupies, so the bounded entry is the only
     * candidate that could win before the fallback: with built-in f32
     * entries present, a built-in would win the cost tie even if the
     * shape filter failed. */
    const rvcl_dtype_t a = RVCL_DTYPE_INT4;
    const rvcl_dtype_t acc = RVCL_DTYPE_INT32;
    rvcl_kernel_desc_t bounded = make_desc("t15_bounded", a, a, acc, acc);
    bounded.max_n = 32;
    rvcl_kernel_desc_t fb = make_desc("t15_f", a, a, acc, acc);
    CHECK(register_desc(&bounded) && register_desc(&fb),
            "bounded entry and unbounded fallback registered");
    rvcl_dispatch_cache_clear();

    enum { M = 8, N = 33, K = 16 };
    rvcl_dispatch_result_t *r = NULL;

    /* Inside the bound the bounded entry is selected. */
    const rvcl_matmul_desc_t in = make_request(M, 32, K, a, a, acc, acc);
    CHECK(rvcl_dispatch_matmul(&in, &r) == RVCL_SUCCESS,
            "in-bounds request dispatches");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && strcmp(rvcl_dispatch_kernel_name(r), "t15_bounded") == 0,
            "in-bounds request selects the bounded entry");
    rvcl_dispatch_release(r);
    r = NULL;

    /* n exceeds the bound: the fallback is the only survivor. */
    const rvcl_matmul_desc_t d = make_request(M, N, K, a, a, acc, acc);
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "oversized shape is still served");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && strcmp(rvcl_dispatch_kernel_name(r), "t15_f") == 0,
            "bounded entry is excluded and the fallback is selected");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && unbounded_by_name(rvcl_dispatch_kernel_name(r)),
            "the fallback carries no shape limit");
    rvcl_dispatch_release(r);

    /* An oversized f32 request through the built-in path still computes
     * the golden result. */
    float a32[M * K], b32[K * N], c32[M * N], ref[M * N];
    for (int i = 0; i < M * K; i++)
        a32[i] = (float)((i * 13) % 29) / 29.0f - 0.5f;
    for (int i = 0; i < K * N; i++)
        b32[i] = (float)((i * 7) % 31) / 31.0f - 0.5f;
    memset(c32, 0, sizeof(c32));
    golden_f32(a32, b32, NULL, ref, M, N, K);
    const rvcl_matmul_desc_t f32 = make_request(M, N, K, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    CHECK(rvcl_matmul(&f32, a32, b32, NULL, c32) == RVCL_SUCCESS,
            "oversized fallback runs through the semantic API");
    /* The selected micro-kernel may fuse its multiply-add while the
     * scalar golden's contraction depends on the build and the chosen
     * kernel path, so compare with a tolerance: this case is about the
     * shape fallback, not bit-exactness (the differential and concurrent
     * suites cover that). */
    golden_cmp_t cmp = golden_cmp_tol(c32, ref, (int64_t)M * N, 1e-5f);
    CHECK(cmp.mismatches == 0,
            "fallback path matches the golden within tolerance");

    unregister_quiet("t15_bounded");
    unregister_quiet("t15_f");
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-16: dispatch selects and configures, never executes. */
static void tc16_dispatch_only(void) {
    printf("TC-AC3-16 dispatch does not execute\n");
    rvcl_kernel_desc_t kd = make_desc("t16_k", RVCL_DTYPE_INT4, RVCL_DTYPE_INT4,
            RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    CHECK(register_desc(&kd), "counting kernel registered");
    rvcl_dispatch_cache_clear();
    g_validate_calls.store(0);
    g_configure_calls.store(0);
    g_run_calls.store(0);

    const rvcl_matmul_desc_t d = make_request(16, 16, 16, RVCL_DTYPE_INT4,
            RVCL_DTYPE_INT4, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS, "dispatch succeeds");
    CHECK(g_validate_calls.load() == 1, "dispatch validates once");
    CHECK(g_configure_calls.load() == 1, "dispatch configures once");
    CHECK(g_run_calls.load() == 0, "dispatch does not run the kernel");

    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    const rvcl_work_range_t full = rvcl_work_range_full(16, 16, 16);
    CHECK(rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full)
                    == RVCL_SUCCESS,
            "explicit run succeeds");
    CHECK(g_run_calls.load() == 1, "explicit run reaches the kernel");

    rvcl_dispatch_release(r);
    unregister_quiet("t16_k");
    rvcl_dispatch_cache_clear();
}

/* Check one successful dispatch reports a name that is currently
 * registered. Unsupported requests are reported and skipped by the
 * caller-specific requirements. */
static int check_name_registered(
        const rvcl_matmul_desc_t *d, const char *label, int require_success) {
    rvcl_dispatch_result_t *r = NULL;
    const rvcl_status_t st = rvcl_dispatch_matmul(d, &r);
    char msg[160];
    if (st != RVCL_SUCCESS) {
        snprintf(msg, sizeof(msg), "%s: dispatch unsupported (skipped)", label);
        printf("  skip: %s\n", msg);
        CHECK(!require_success, msg);
        return 0;
    }
    const char *name = rvcl_dispatch_kernel_name(r);
    snprintf(msg, sizeof(msg), "%s: selected name is non-empty", label);
    CHECK(name && name[0] != '\0', msg);
    snprintf(msg, sizeof(msg), "%s: selected name is registered", label);
    CHECK(name && registry_has(name), msg);
    rvcl_dispatch_release(r);
    return 1;
}

/* TC-AC3-17: the selected name is queryable and matches a live entry. */
static void tc17_name_consistency(void) {
    printf("TC-AC3-17 selected name consistency\n");
    rvcl_dispatch_cache_clear();
    const rvcl_matmul_desc_t f32 = make_request(16, 16, 16, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    check_name_registered(&f32, "f32 request", 1);
    const rvcl_matmul_desc_t f16 = make_request(16, 16, 16, RVCL_DTYPE_FP16,
            RVCL_DTYPE_FP16, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP16);
    check_name_registered(&f16, "f16 request", 0);

    const rvcl_matmul_desc_t i8 = make_request(16, 16, 16, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    check_name_registered(&i8, "INT8 request without a kernel", 0);
}

/* TC-AC3-18: concurrent dispatch while the registry churns. */

#define T18_T1 8
#define T18_T2 4
#define T18_T3 2
#define T18_CHURN 64

typedef struct {
    int id;
    rvcl_matmul_desc_t desc;
    /* The kernel name points into the result's kernel instance, which the
     * worker releases before the parent inspects it — keep a copy. */
    char name[RVCL_KERNEL_NAME_MAX];
    int ok;
    int rejected;
} t18_arg_t;

static pthread_barrier_t g_t18_start;

static void *t18_dispatch(void *p) {
    t18_arg_t *arg = (t18_arg_t *)p;
    pthread_barrier_wait(&g_t18_start);
    rvcl_dispatch_result_t *r = NULL;
    if (rvcl_dispatch_matmul(&arg->desc, &r) == RVCL_SUCCESS) {
        const char *name = rvcl_dispatch_kernel_name(r);
        snprintf(arg->name, sizeof(arg->name), "%s", name ? name : "");
        arg->ok = 1;
    } else {
        arg->rejected = 1;
    }
    rvcl_dispatch_release(r);
    return NULL;
}

static void *t18_churn(void *p) {
    t18_arg_t *arg = (t18_arg_t *)p;
    pthread_barrier_wait(&g_t18_start);
    for (int i = 0; i < T18_CHURN; i++) {
        char name[RVCL_KERNEL_NAME_MAX];
        snprintf(name, sizeof(name), "t18_churn_%d_%d", arg->id, i);
        rvcl_kernel_desc_t d = make_desc(name, RVCL_DTYPE_INT4, RVCL_DTYPE_INT4,
                RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
        if (register_desc(&d)) rvcl_unregister_kernel(name);
    }
    return NULL;
}

static void tc18_concurrent(void) {
    printf("TC-AC3-18 concurrent dispatch under registry churn\n");
    rvcl_kernel_desc_t kd = make_desc("t18_k", RVCL_DTYPE_INT8, RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    rvcl_kernel_desc_t fb = make_desc("t18_f", RVCL_DTYPE_INT4, RVCL_DTYPE_INT4,
            RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    CHECK(register_desc(&kd) && register_desc(&fb),
            "dispatch target and churn fallback registered");
    rvcl_dispatch_cache_clear();

    const int total = T18_T1 + T18_T2 + T18_T3;
    if (pthread_barrier_init(&g_t18_start, NULL, (unsigned)total) != 0) {
        printf("  FAIL: barrier init (line %d)\n", __LINE__);
        g_failures++;
        return;
    }

    const int hits0 = rvcl_dispatch_cache_hits();
    const int misses0 = rvcl_dispatch_cache_misses();

    pthread_t thr[T18_T1 + T18_T2 + T18_T3];
    t18_arg_t args[T18_T1 + T18_T2 + T18_T3];
    memset(args, 0, sizeof(args));
    int nt = 0;
    for (int i = 0; i < T18_T1; i++, nt++) {
        args[nt].id = i;
        args[nt].desc = make_request(16, 16, 16, RVCL_DTYPE_INT8,
                RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
        pthread_create(&thr[nt], NULL, t18_dispatch, &args[nt]);
    }
    for (int i = 0; i < T18_T2; i++, nt++) {
        args[nt].id = i;
        args[nt].desc = make_request(17 + i, 16, 16, RVCL_DTYPE_INT8,
                RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
        pthread_create(&thr[nt], NULL, t18_dispatch, &args[nt]);
    }
    for (int i = 0; i < T18_T3; i++, nt++) {
        args[nt].id = i;
        pthread_create(&thr[nt], NULL, t18_churn, &args[nt]);
    }

    for (int i = 0; i < nt; i++)
        pthread_join(thr[i], NULL);
    pthread_barrier_destroy(&g_t18_start);

    int ok = 0, rejected = 0, names_ok = 1;
    for (int i = 0; i < T18_T1 + T18_T2; i++) {
        if (args[i].ok) {
            ok++;
            if (strcmp(args[i].name, "t18_k") != 0) names_ok = 0;
        }
        if (args[i].rejected) rejected++;
    }
    CHECK(rejected == 0, "all concurrent dispatches succeed");
    CHECK(ok == T18_T1 + T18_T2, "every dispatch thread observed success");
    CHECK(names_ok, "all concurrent dispatches select the same kernel");
    CHECK(rvcl_dispatch_cache_hits() + rvcl_dispatch_cache_misses() - hits0
                            - misses0
                    >= ok,
            "hit/miss counters cover every successful dispatch");

    unregister_quiet("t18_k");
    unregister_quiet("t18_f");
    rvcl_dispatch_cache_clear();
}

/* -------------------------------------------------- generic op-table dispatch
 *
 * The op-table cases drive the internal rvcl_dispatch_op() entry with
 * synthetic operators whose requests reuse the matmul union member as the
 * payload (the op desc union currently has a single member). Each test op
 * registers a table entry (request validator + cache key generator) and
 * kernels of that operation type, so the six-stage pipeline and cache are
 * exercised per-op. The test key hash returns a constant, forcing every
 * test-op request onto cache slot 0 to prove the op discriminant keeps
 * colliding entries apart. */

static rvcl_status_t test_op_dims_validate(
        const rvcl_op_desc_t *od, rvcl_dispatch_filter_t *out) {
    const rvcl_matmul_desc_t *d = &od->matmul;
    if (d->m <= 0 || d->n <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    out->m = d->m;
    out->n = d->n;
    out->k = d->k;
    out->dtype_a = d->dtype_a;
    out->dtype_b = d->dtype_b;
    out->dtype_acc = d->dtype_acc;
    out->dtype_c = d->dtype_c;
    out->quant_enabled = d->quant.enabled;
    return RVCL_SUCCESS;
}

static int test_op_key_equal(const rvcl_op_desc_t *a, const rvcl_op_desc_t *b) {
    return a->matmul.m == b->matmul.m && a->matmul.n == b->matmul.n
            && a->matmul.k == b->matmul.k
            && a->matmul.dtype_a == b->matmul.dtype_a
            && a->matmul.dtype_b == b->matmul.dtype_b
            && a->matmul.dtype_acc == b->matmul.dtype_acc
            && a->matmul.dtype_c == b->matmul.dtype_c;
}

/* Constant hash: every test-op request maps to cache slot 0. */
static uint32_t test_op_key_hash(const rvcl_op_desc_t *key) {
    (void)key;
    return 0;
}

/* TC-AC3-19: generic op-table dispatch — unregistered op rejected,
 * registered op runs the full pipeline with cache hit, and per-op cache
 * key isolation (same hash, different op, no slot cross-talk). */
static void tc19_op_table(void) {
    printf("TC-AC3-19 generic op-table dispatch\n");
    rvcl_dispatch_cache_clear();

    /* (a) An operator with no table entry is rejected. */
    const rvcl_op_desc_t od
            = {.matmul = make_request(16, 16, 16, RVCL_DTYPE_INT8,
                       RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32)};
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_op(RVCL_OP_GEMV, &od, &r) == RVCL_UNSUPPORTED,
            "op with no table entry is rejected");
    CHECK(r == NULL, "no result leaked on an unregistered op");

    /* (b) Register the op entries once (they persist for this case). */
    const rvcl_dispatch_op_entry_t gemv_e = {RVCL_OP_GEMV,
            test_op_dims_validate, test_op_key_hash, test_op_key_equal};
    const rvcl_dispatch_op_entry_t conv_e = {RVCL_OP_CONV2D,
            test_op_dims_validate, test_op_key_hash, test_op_key_equal};
    CHECK(rvcl_dispatch_register_op(&gemv_e) == RVCL_SUCCESS,
            "GEMV op table entry registered");
    CHECK(rvcl_dispatch_register_op(&conv_e) == RVCL_SUCCESS,
            "CONV2D op table entry registered");
    CHECK(rvcl_dispatch_register_op(&gemv_e) == RVCL_ERROR_ALREADY_REGISTERED,
            "duplicate op table entry rejected");

    /* (c) A registered op runs the full pipeline, then hits the cache. */
    rvcl_kernel_desc_t kd = make_desc("t19_gemv", RVCL_DTYPE_INT8,
            RVCL_DTYPE_INT8, RVCL_DTYPE_INT32, RVCL_DTYPE_INT32);
    kd.operation = RVCL_OP_GEMV;
    CHECK(register_desc(&kd), "GEMV kernel registered");
    rvcl_dispatch_cache_clear();
    g_validate_calls.store(0);
    g_configure_calls.store(0);
    g_run_calls.store(0);

    rvcl_dispatch_result_t *r1 = NULL, *r2 = NULL;
    CHECK(rvcl_dispatch_op(RVCL_OP_GEMV, &od, &r1) == RVCL_SUCCESS,
            "GEMV dispatch succeeds");
    CHECK(rvcl_dispatch_kernel_name(r1)
                    && strcmp(rvcl_dispatch_kernel_name(r1), "t19_gemv") == 0,
            "GEMV selects its own kernel");
    CHECK(g_validate_calls.load() == 1, "GEMV first dispatch validates once");
    CHECK(g_configure_calls.load() == 1, "GEMV first dispatch configures once");
    CHECK(g_run_calls.load() == 0, "dispatch never runs the kernel");

    CHECK(rvcl_dispatch_op(RVCL_OP_GEMV, &od, &r2) == RVCL_SUCCESS,
            "GEMV repeat dispatch succeeds");
    CHECK(r1 == r2, "GEMV cache hit returns the same result object");
    CHECK(g_validate_calls.load() == 1, "GEMV cache hit skips validate");
    CHECK(g_configure_calls.load() == 1, "GEMV cache hit skips configure");
    rvcl_dispatch_release(r1);
    rvcl_dispatch_release(r2);
    unregister_quiet("t19_gemv");
    rvcl_dispatch_cache_clear();

    /* (d) Per-op cache key isolation: both test ops collide on slot 0,
     * but the op discriminant keeps their entries apart. */
    const rvcl_dtype_t a = RVCL_DTYPE_INT4;
    const rvcl_dtype_t acc = RVCL_DTYPE_INT32;
    rvcl_kernel_desc_t gemv = make_desc("t19c_gemv", a, a, acc, acc);
    gemv.operation = RVCL_OP_GEMV;
    rvcl_kernel_desc_t conv = make_desc("t19c_conv", a, a, acc, acc);
    conv.operation = RVCL_OP_CONV2D;
    CHECK(register_desc(&gemv) && register_desc(&conv),
            "GEMV and CONV2D kernels registered");
    rvcl_dispatch_cache_clear();

    const rvcl_op_desc_t gemv_od
            = {.matmul = make_request(8, 8, 8, a, a, acc, acc)};
    rvcl_op_desc_t conv_od = gemv_od;
    conv_od.matmul.m = 9; /* distinct key, still the same slot-0 hash */

    rvcl_dispatch_result_t *rg = NULL;
    CHECK(rvcl_dispatch_op(RVCL_OP_GEMV, &gemv_od, &rg) == RVCL_SUCCESS,
            "GEMV dispatch succeeds");
    CHECK(rvcl_dispatch_kernel_name(rg)
                    && strcmp(rvcl_dispatch_kernel_name(rg), "t19c_gemv") == 0,
            "GEMV selects its own kernel");
    rvcl_dispatch_release(rg);
    rg = NULL;

    int misses = rvcl_dispatch_cache_misses();
    rvcl_dispatch_result_t *rc = NULL;
    CHECK(rvcl_dispatch_op(RVCL_OP_CONV2D, &conv_od, &rc) == RVCL_SUCCESS,
            "colliding-hash CONV2D dispatch succeeds");
    CHECK(rvcl_dispatch_cache_misses() > misses,
            "colliding-hash CONV2D is a cache miss (no cross-talk)");
    CHECK(rvcl_dispatch_kernel_name(rc)
                    && strcmp(rvcl_dispatch_kernel_name(rc), "t19c_conv") == 0,
            "CONV2D selects its own kernel, not the colliding GEMV entry");
    rvcl_dispatch_release(rc);
    rc = NULL;

    unregister_quiet("t19c_gemv");
    unregister_quiet("t19c_conv");
    rvcl_dispatch_cache_clear();
}

/* ------------------------------------------------- capability injection */

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

static void inject_rvv(uint32_t vlen) {
    const rvcl_capability_t cap = make_snapshot(1, 1, vlen);
    rvcl_capability_test_override(&cap);
}

/* TC-AC3-11: a candidate whose required feature is absent is excluded at
 * stage 3; an empty snapshot leaves only featureless kernels. */
static void case_inject_feature_subset(void) {
    printf("TC-AC3-11 required-feature subset filter (injected)\n");
    const rvcl_dtype_t a = RVCL_DTYPE_INT4;
    const rvcl_dtype_t acc = RVCL_DTYPE_INT32;
    rvcl_kernel_desc_t p = make_desc("t11_p", a, a, acc, acc);
    p.required_features[0] = RVCL_FEATURE_RVV;
    p.num_required_features = 1;
    rvcl_kernel_desc_t q = make_desc("t11_q", a, a, acc, acc);
    q.required_features[0] = RVCL_FEATURE_AME;
    q.num_required_features = 1;
    CHECK(register_desc(&p) && register_desc(&q),
            "feature-gated kernels registered");

    inject_rvv(128);
    const rvcl_matmul_desc_t d = make_request(16, 16, 16, a, a, acc, acc);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "kernel whose feature is present is selectable");
    CHECK(rvcl_dispatch_kernel_name(r)
                    && strcmp(rvcl_dispatch_kernel_name(r), "t11_p") == 0,
            "RVV kernel selected with RVV in the snapshot");
    rvcl_dispatch_release(r);
    r = NULL;

    CHECK(rvcl_unregister_kernel("t11_p") == RVCL_SUCCESS,
            "RVV kernel unregistered");
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_UNSUPPORTED,
            "only candidate requiring an absent feature -> UNSUPPORTED");
    CHECK(r == NULL, "no result leaked when the feature is absent");

    /* Empty snapshot: vector kernels are excluded, the featureless
     * fallback still serves f32. */
    const rvcl_capability_t empty = make_snapshot(0, 0, 0);
    rvcl_capability_test_override(&empty);
    const rvcl_matmul_desc_t f32 = make_request(16, 16, 16, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    CHECK(rvcl_dispatch_matmul(&f32, &r) == RVCL_SUCCESS,
            "featureless snapshot still serves f32");
    const char *sel = r ? rvcl_dispatch_kernel_name(r) : NULL;
    CHECK(sel && featureless_by_name(sel),
            "selected kernel requires no feature");
    if (portable_f32_registered()) {
        CHECK(sel && strcmp(sel, "rvcl_matmul_f32_rvv_c") == 0,
                "portable entry is preferred in a featureless snapshot");
    } else {
        CHECK(sel && name_ends_with(sel, "reference"),
                "reference fallback selected in a featureless snapshot");
    }
    rvcl_dispatch_release(r);

    rvcl_unregister_kernel("t11_q");
    rvcl_capability_test_override(NULL);
    rvcl_dispatch_cache_clear();
}

/* TC-AC2-08 (dispatch side): the converted tier of every scanned width
 * selects the matching kernel tier. */
static void case_inject_vlen_scan(void) {
    printf("TC-AC2-08 dispatch-side tier scan (injected)\n");
    static const uint32_t scan[]
            = {64, 127, 128, 129, 200, 256, 300, 512, 1000, 1024};
    for (size_t i = 0; i < sizeof(scan) / sizeof(scan[0]); i++) {
        inject_rvv(scan[i]);
        const rvcl_matmul_desc_t d = make_request(64, 64, 64, RVCL_DTYPE_FP32,
                RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
        rvcl_dispatch_result_t *r = NULL;
        CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
                "dispatch succeeds for the injected width");
        /* Non-vector builds register no dedicated tiers regardless of
         * the configured tier list; only tier-0 entries exist there. */
        const int want = vector_build() ? expected_tier(scan[i]) : 0;
        const int got = r ? tier_of_name(rvcl_dispatch_kernel_name(r)) : -1;
        char msg[128];
        snprintf(msg, sizeof(msg), "vlen=%u selects tier %d (expected %d)",
                scan[i], got, want);
        CHECK(got == want, msg);
        rvcl_dispatch_release(r);
    }
    rvcl_capability_test_override(NULL);
    rvcl_dispatch_cache_clear();
}

/* Save one live entry so a case can remove and restore it. */
static int save_entry(const char *name, rvcl_kernel_desc_t *desc,
        rvcl_kernel_create_fn *create) {
    const int idx = find_index(name);
    if (idx < 0) return 0;
    const rvcl_kernel_desc_t *kd = rvcl_registry_get(idx);
    if (!kd) return 0;
    *desc = *kd;
    *create = rvcl_registry_find_factory(name);
    return *create != NULL;
}

/* TC-AC3-03: an injected VLEN=256 excludes higher tiers and prefers 256
 * over 128; re-injecting 128 (without a manual cache clear) immediately
 * invalidates the cached 256 selection, and removing 256 falls to 128. */
static void case_inject_tier_preference(void) {
    printf("TC-AC3-03 injected VLEN preference (injected)\n");
    if (!vector_build()) {
        printf("  skip: needs a vector build\n");
        return;
    }
    const char *n256 = "rvcl_matmul_f32_rvv_256";
    const char *n128 = "rvcl_matmul_f32_rvv_128";
    if (!registry_has(n256) || !registry_has(n128)) {
        printf("  skip: build does not register both tiers\n");
        return;
    }

    inject_rvv(256);
    const rvcl_matmul_desc_t d = make_request(64, 64, 64, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch at injected VLEN=256 succeeds");
    char msg[128];
    snprintf(msg, sizeof(msg), "VLEN=256 selects the tier-256 kernel (got %d)",
            r ? tier_of_name(rvcl_dispatch_kernel_name(r)) : -1);
    CHECK(r && tier_of_name(rvcl_dispatch_kernel_name(r)) == 256, msg);
    rvcl_dispatch_release(r);
    r = NULL;

    /* Re-injecting 128 must invalidate the cached 256 selection: without
     * the override's cache clear this dispatch would return the stale
     * tier-256 result while the tier-256 kernel is still registered. */
    inject_rvv(128);
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch after re-injecting VLEN=128 succeeds");
    snprintf(msg, sizeof(msg), "re-injection is reflected immediately (got %d)",
            r ? tier_of_name(rvcl_dispatch_kernel_name(r)) : -1);
    CHECK(r && tier_of_name(rvcl_dispatch_kernel_name(r)) == 128, msg);
    rvcl_dispatch_release(r);
    r = NULL;

    rvcl_kernel_desc_t saved;
    rvcl_kernel_create_fn saved_create = NULL;
    CHECK(save_entry(n256, &saved, &saved_create), "tier-256 entry saved");
    CHECK(rvcl_unregister_kernel(n256) == RVCL_SUCCESS,
            "tier-256 entry unregistered");
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch after removing tier 256 succeeds");
    snprintf(msg, sizeof(msg), "tier-128 kernel takes over (got %d)",
            r ? tier_of_name(rvcl_dispatch_kernel_name(r)) : -1);
    CHECK(r && tier_of_name(rvcl_dispatch_kernel_name(r)) == 128, msg);
    rvcl_dispatch_release(r);

    CHECK(rvcl_register_kernel(&saved, saved_create) == RVCL_SUCCESS,
            "tier-256 entry restored");
    rvcl_capability_test_override(NULL);
    rvcl_dispatch_cache_clear();
}

/* TC-AC3-12: a tier above the injected host tier is never selected. */
static void case_inject_tier_exclusion(void) {
    printf("TC-AC3-12 injected tier exclusion (injected)\n");
    if (!vector_build()) {
        printf("  skip: needs a vector build\n");
        return;
    }
    const char *n128 = "rvcl_matmul_f32_rvv_128";
    if (!registry_has(n128)) {
        printf("  skip: build does not register tier 128\n");
        return;
    }

    inject_rvv(128);
    const rvcl_matmul_desc_t d = make_request(64, 64, 64, RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32);
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch at injected VLEN=128 succeeds");
    const int got = r ? tier_of_name(rvcl_dispatch_kernel_name(r)) : -1;
    CHECK(got == 128, "tier-128 kernel selected");
    rvcl_dispatch_release(r);
    r = NULL;

    rvcl_kernel_desc_t saved;
    rvcl_kernel_create_fn saved_create = NULL;
    CHECK(save_entry(n128, &saved, &saved_create), "tier-128 entry saved");
    CHECK(rvcl_unregister_kernel(n128) == RVCL_SUCCESS,
            "tier-128 entry unregistered");
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "dispatch without a dedicated tier succeeds");
    const int fallback_tier
            = r ? tier_of_name(rvcl_dispatch_kernel_name(r)) : -1;
    CHECK(fallback_tier == 0,
            "only tier-0 kernels remain eligible at VLEN=128");
    rvcl_dispatch_release(r);

    CHECK(rvcl_register_kernel(&saved, saved_create) == RVCL_SUCCESS,
            "tier-128 entry restored");
    rvcl_capability_test_override(NULL);
    rvcl_dispatch_cache_clear();
}

static int run_inject_mode(void) {
    printf("== dispatch unit tests: injection subset ==\n");
    CHECK(rvcl_init() == RVCL_SUCCESS, "library init succeeds");
    collect_enabled_tiers();

    case_inject_feature_subset();
    case_inject_vlen_scan();
    case_inject_tier_preference();
    case_inject_tier_exclusion();

    if (g_failures) printf("FAILURES: %d\n", g_failures);
    return g_failures ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "--inject") == 0) return run_inject_mode();

    printf("== dispatch unit tests ==\n");
    CHECK(rvcl_init() == RVCL_SUCCESS, "library init succeeds");
    collect_enabled_tiers();

    tc01_rvv_selection();
    tc02_reference_selection();
    tc02b_quantized_rejected();
    tc04_deterministic_cache();
    tc05_cache_counters();
    tc06_cache_key();
    tc07_registry_change();
    tc08_no_candidate();
    tc09_illegal_tuples();
    tc10_auto_resolution();
    tc13_shape_bounds();
    tc14_validate_veto();
    tc15_shape_fallback();
    tc16_dispatch_only();
    tc17_name_consistency();
    tc18_concurrent();
    tc19_op_table();

    if (g_failures) printf("FAILURES: %d\n", g_failures);
    return g_failures ? 1 : 0;
}
