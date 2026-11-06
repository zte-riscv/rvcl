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
 * tests/dtype_ops.cpp — test-side dtype fact table (see dtype_ops.hpp)
 ******************************************************************************/

#include "dtype_ops.hpp"

#include <string.h>

#include "../src/backends/rvv/rvv_fp16.hpp"
#include "../src/common/dtype_traits.hpp"

/* ----------------------------------------------------- per-dtype wrappers
 * The golden/fill callbacks adapt the golden.hpp helpers to the void*
 * storage the table carries. */

static void golden_f32_op(const void *a, const void *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k) {
    golden_f32((const float *)a, (const float *)b, bias, c, m, n, k);
}

static void golden_f16_op(const void *a, const void *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k) {
    golden_f16((const uint16_t *)a, (const uint16_t *)b, bias, c, m, n, k);
}

static void fill_f32_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_f32((float *)buf, n, seed);
}

static void fill_f16_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_f16((uint16_t *)buf, n, seed);
}

/* fp32 storage is the value itself: the f32_to_storage slot exists for
 * table uniformity and is only meaningful (and only ever invoked) for
 * fp16. */
static uint16_t f32_identity_storage(float f) {
    uint16_t lo;
    __builtin_memcpy(&lo, &f, sizeof(lo));
    return lo;
}

/* Compare routing driven by the kernel desc assertion level. The
 * tolerance class is read from the LIBRARY trait table by dtype (single
 * source of truth); a TOLERANT kernel compares within it, anything else
 * is bit-exact. */
static golden_cmp_t compare_op(const float *got, const float *exp, int64_t n,
        rvcl_assert_level_t level, rvcl_dtype_t dtype) {
    if (level == RVCL_ASSERT_TOLERANT) {
        const float tol = rvcl_dtype_traits(dtype).tolerance;
        return golden_cmp_tol(got, exp, n, tol);
    }
    return golden_cmp_bitexact(got, exp, n);
}

/* ------------------------------------------------------------------ table */

/* The currently supported test dtypes (f32, f16). A new dtype adds one
 * row here plus its golden/fill wrappers. */
static const dtype_ops_t k_dtype_ops[] = {
        {
                .dtstr = "f32",
                .dtype = RVCL_DTYPE_FP32,
                .elt = 4,
                .golden = golden_f32_op,
                .fill = fill_f32_op,
                .compare = compare_op,
                .f32_to_storage = f32_identity_storage,
        },
        {
                .dtstr = "f16",
                .dtype = RVCL_DTYPE_FP16,
                .elt = 2,
                .golden = golden_f16_op,
                .fill = fill_f16_op,
                .compare = compare_op,
                .f32_to_storage = rvv_f32_to_f16,
        },
};

static const int k_dtype_ops_count
        = (int)(sizeof(k_dtype_ops) / sizeof(k_dtype_ops[0]));

const dtype_ops_t *dtype_ops_for(rvcl_dtype_t dt) {
    for (int i = 0; i < k_dtype_ops_count; i++)
        if (k_dtype_ops[i].dtype == dt) return &k_dtype_ops[i];
    return NULL;
}

int dtype_ops_from_dtstr(const char *s, rvcl_dtype_t *out) {
    for (int i = 0; i < k_dtype_ops_count; i++)
        if (!strcmp(s, k_dtype_ops[i].dtstr)) {
            if (out) *out = k_dtype_ops[i].dtype;
            return 1;
        }
    return 0;
}
