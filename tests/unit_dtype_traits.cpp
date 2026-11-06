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
 * tests/unit_dtype_traits.cpp — dtype trait table + assertion-level tests
 *
 * Asserts the library dtype fact table (src/common/dtype_traits.hpp):
 * element size facts, the AUTO accumulation habit, the default assertion
 * level and the tolerance class, plus the f16 TOLERANT routing on the
 * registry and the dispatched kernel.
 ******************************************************************************/

#include <stdio.h>
#include <string.h>

#include "../src/common/dtype_traits.hpp"
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

/* size fact + consistency with the public rvcl_dtype_size entry. */
static void check_size(rvcl_dtype_t dt, size_t expected, const char *label) {
    char msg[96];
    snprintf(msg, sizeof(msg), "%s trait size = %zu", label, expected);
    CHECK(rvcl_dtype_traits(dt).size == expected, msg);
    snprintf(msg, sizeof(msg), "%s rvcl_dtype_size agrees", label);
    CHECK(rvcl_dtype_size(dt) == expected, msg);
}

static void tc01_trait_table(void) {
    printf("TC-01 trait table facts\n");

    check_size(RVCL_DTYPE_AUTO, 0, "auto");
    check_size(RVCL_DTYPE_FP32, 4, "f32");
    check_size(RVCL_DTYPE_FP16, 2, "f16");
    check_size(RVCL_DTYPE_BF16, 2, "bf16");
    check_size(RVCL_DTYPE_INT8, 1, "int8");
    check_size(RVCL_DTYPE_UINT8, 1, "uint8");
    check_size(RVCL_DTYPE_INT4, 1, "int4");
    check_size(RVCL_DTYPE_INT32, 4, "int32");
    check_size(RVCL_DTYPE_FP8_E4M3, 1, "fp8e4m3");
    check_size(RVCL_DTYPE_FP8_E5M2, 1, "fp8e5m2");

    /* out-of-range datatypes resolve to a zero row (size 0), matching the
     * pre-table rvcl_dtype_size contract. The out-of-range casts are the
     * point of the assertion (contract test). */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(rvcl_dtype_size((rvcl_dtype_t)99) == 0, "out-of-range size = 0");
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    CHECK(rvcl_dtype_size((rvcl_dtype_t)-1) == 0, "negative size = 0");

    /* AUTO accumulation habit (resolve_acc facts): int8/uint8 -> INT32,
     * floating family -> FP32 */
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_INT8).acc_default == RVCL_DTYPE_INT32,
            "int8 acc habit = INT32");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_UINT8).acc_default == RVCL_DTYPE_INT32,
            "uint8 acc habit = INT32");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_FP32).acc_default == RVCL_DTYPE_FP32,
            "f32 acc habit = FP32");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_FP16).acc_default == RVCL_DTYPE_FP32,
            "f16 acc habit = FP32");
}

static void tc02_assert_level_defaults(void) {
    printf("TC-02 assertion-level defaults and tolerance class\n");

    CHECK(rvcl_dtype_traits(RVCL_DTYPE_FP32).assert_level_default
                    == RVCL_ASSERT_BITEXACT,
            "f32 default assertion level = BITEXACT");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_FP16).assert_level_default
                    == RVCL_ASSERT_TOLERANT,
            "f16 default assertion level = TOLERANT");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_FP16).tolerance == 1e-3f,
            "f16 tolerance class locked at 1e-3");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_FP32).tolerance == 0.0f,
            "f32 tolerance class = 0 (bit-exact)");
    CHECK(rvcl_dtype_traits(RVCL_DTYPE_BF16).assert_level_default
                    == RVCL_ASSERT_BITEXACT,
            "bf16 default assertion level = BITEXACT");
}

/* The backend sets the assertion level from the trait default: every
 * registered f16-tuple kernel is TOLERANT, every f32-tuple kernel
 * BITEXACT. */
static void tc03_registry_assert_level(void) {
    printf("TC-03 registry assert_level from trait default\n");
    const int n = rvcl_registry_count();
    int f16_checked = 0, f32_checked = 0, bad = 0;
    for (int i = 0; i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || kd->operation != RVCL_OP_MATMUL) continue;
        if (kd->dtype_a == RVCL_DTYPE_FP16) {
            f16_checked++;
            if (kd->assert_level != RVCL_ASSERT_TOLERANT) bad = 1;
        } else if (kd->dtype_a == RVCL_DTYPE_FP32) {
            f32_checked++;
            if (kd->assert_level != RVCL_ASSERT_BITEXACT) bad = 1;
        }
    }
    CHECK(f16_checked >= 1, "at least one f16-tuple kernel registered");
    CHECK(f32_checked >= 1, "at least one f32-tuple kernel registered");
    CHECK(bad == 0, "f16 kernels TOLERANT, f32 kernels BITEXACT");
}

/* End-to-end routing: the kernel dispatch selects for an f16 request
 * declares TOLERANT, for f32 BITEXACT. */
static void tc04_dispatch_routing(void) {
    printf("TC-04 dispatch-driven assertion-level routing\n");
    enum { M = 8, N = 8, K = 8 };
    rvcl_matmul_desc_t d = {
            .m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP16,
            .dtype_b = RVCL_DTYPE_FP16,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
    rvcl_dispatch_result_t *r16 = NULL;
    if (rvcl_dispatch_matmul(&d, &r16) == RVCL_SUCCESS && r16) {
        const rvcl_kernel_t *k = rvcl_dispatch_kernel(r16);
        CHECK(k->desc.assert_level == RVCL_ASSERT_TOLERANT,
                "dispatched f16 kernel is TOLERANT");
        rvcl_dispatch_release(r16);
    } else {
        CHECK(0, "f16 dispatch succeeds on this host");
    }

    rvcl_matmul_desc_t d32 = d;
    d32.dtype_a = d32.dtype_b = RVCL_DTYPE_FP32;
    rvcl_dispatch_result_t *r32 = NULL;
    if (rvcl_dispatch_matmul(&d32, &r32) == RVCL_SUCCESS && r32) {
        const rvcl_kernel_t *k = rvcl_dispatch_kernel(r32);
        CHECK(k->desc.assert_level == RVCL_ASSERT_BITEXACT,
                "dispatched f32 kernel is BITEXACT");
        rvcl_dispatch_release(r32);
    } else {
        CHECK(0, "f32 dispatch succeeds on this host");
    }
}

int main(void) {
    CHECK(rvcl_init() == RVCL_SUCCESS, "rvcl_init succeeds");
    tc01_trait_table();
    tc02_assert_level_defaults();
    tc03_registry_assert_level();
    tc04_dispatch_routing();

    printf(g_failures ? "UNIT-DTYPE-TRAITS: %d failure(s)\n"
                      : "UNIT-DTYPE-TRAITS: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
