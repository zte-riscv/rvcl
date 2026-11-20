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

#include "dtype_ops.hpp"

#include <string.h>

#include "../src/backends/rvv/rvv_fp16.hpp"
#include "../src/backends/rvv/rvv_fp8.hpp"
#include "../src/backends/rvv/rvv_bf16.hpp"
#include "../src/common/dtype_traits.hpp"

/* ----------------------------------------------------- per-dtype wrappers
 * The golden/fill callbacks adapt the golden.hpp helpers to the void*
 * storage the table carries. */

static void golden_f32_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_f32((const float *)a, (const float *)b, bias, c, m, n, k);
}

static void golden_f16_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_f16((const uint16_t *)a, (const uint16_t *)b, bias, c, m, n, k);
}

static void golden_bf16_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_bf16((const uint16_t *)a, (const uint16_t *)b, bias, c, m, n, k);
}

static void golden_fp8_e4m3_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_fp8_e4m3((const uint8_t *)a, (const uint8_t *)b, bias, c, m, n, k);
}

static void golden_fp8_e5m2_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_fp8_e5m2((const uint8_t *)a, (const uint8_t *)b, bias, c, m, n, k);
}

static void golden_mxfp8_e4m3_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    golden_mxfp8_e4m3((const uint8_t *)a, (const uint8_t *)sa,
            (const uint8_t *)b, (const uint8_t *)sb, bias, c, m, n, k, blk);
}

static void golden_mxfp8_e5m2_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    golden_mxfp8_e5m2((const uint8_t *)a, (const uint8_t *)sa,
            (const uint8_t *)b, (const uint8_t *)sb, bias, c, m, n, k, blk);
}

static void golden_nvfp4_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    golden_nvfp4((const uint8_t *)a, (const uint8_t *)sa, (const uint8_t *)b,
            (const uint8_t *)sb, bias, c, m, n, k, blk);
}

static void golden_mxfp4_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    golden_mxfp4((const uint8_t *)a, (const uint8_t *)sa, (const uint8_t *)b,
            (const uint8_t *)sb, bias, c, m, n, k, blk);
}

/* INT8 goldens: the output and bias are int32 (the s8/u8 acc habit), both
 * 4 bytes per element like the float buffers the test rig passes, so the
 * storage is reinterpreted as int32. */
static void golden_s8_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_s8((const int8_t *)a, (const int8_t *)b, (const int32_t *)bias,
            (int32_t *)c, m, n, k);
}

static void golden_u8_op(const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    (void)sa;
    (void)sb;
    (void)blk;
    golden_u8((const uint8_t *)a, (const uint8_t *)b, (const int32_t *)bias,
            (int32_t *)c, m, n, k);
}

static void fill_f32_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_f32((float *)buf, n, seed);
}

static void fill_f16_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_f16((uint16_t *)buf, n, seed);
}

static void fill_bf16_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_bf16((uint16_t *)buf, n, seed);
}

static void fill_fp8_e4m3_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_fp8_e4m3((uint8_t *)buf, n, seed);
}

static void fill_fp8_e5m2_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_fp8_e5m2((uint8_t *)buf, n, seed);
}

static void fill_s8_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_s8((int8_t *)buf, n, seed);
}

static void fill_u8_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_u8((uint8_t *)buf, n, seed);
}

static void fill_e2m1_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_e2m1((uint8_t *)buf, n, seed);
}

static void fill_e8m0_scale_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_e8m0((uint8_t *)buf, n, seed);
}

static void fill_e4m3_scale_op(void *buf, size_t n, uint32_t seed) {
    lcg_fill_e4m3((uint8_t *)buf, n, seed);
}

/* fp32 storage is the value itself (4 bytes) while the f32_to_storage
 * slot returns a narrow (uint16_t) storage word, so it is only meaningful
 * for narrow dtypes (fp16, bf16, fp8). The fp32 row keeps it NULL; no consumer
 * dereferences it for fp32 (both call sites guard on the narrow dtypes). */

static uint16_t f32_to_fp8_e4m3_slot(float f) {
    return (uint16_t)rvv_f32_to_fp8_e4m3(f);
}

static uint16_t f32_to_fp8_e5m2_slot(float f) {
    return (uint16_t)rvv_f32_to_fp8_e5m2(f);
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

static golden_cmp_t compare_op_bits(const float *got, const float *exp,
        int64_t n, rvcl_assert_level_t level, rvcl_dtype_t dtype) {
    if (level == RVCL_ASSERT_TOLERANT) {
        const float tol = rvcl_dtype_traits(dtype).tolerance;
        return golden_cmp_tol(got, exp, n, tol);
    }
    return golden_cmp_bitexact_bits(got, exp, n);
}

/* ------------------------------------------------------------------ table */

/* The currently supported test dtypes (f32, f16, bf16, fp8, mxfp8). A new
 * dtype adds one row here plus its golden/fill wrappers. */
static const dtype_ops_t k_dtype_ops[] = {
        {
                .dtstr = "f32",
                .dtype = RVCL_DTYPE_FP32,
                .elt = 4,
                .golden = golden_f32_op,
                .fill = fill_f32_op,
                .fill_scale = NULL,
                .compare = compare_op,
                .f32_to_storage = NULL, /* narrow-only; see above */
        },
        {
                .dtstr = "f16",
                .dtype = RVCL_DTYPE_FP16,
                .elt = 2,
                .golden = golden_f16_op,
                .fill = fill_f16_op,
                .fill_scale = NULL,
                .compare = compare_op,
                .f32_to_storage = rvv_f32_to_f16,
        },
        {
                .dtstr = "bf16",
                .dtype = RVCL_DTYPE_BF16,
                .elt = 2,
                .golden = golden_bf16_op,
                .fill = fill_bf16_op,
                .fill_scale = NULL,
                .compare = compare_op,
                .f32_to_storage = rvv_f32_to_bf16,
        },
        {
                .dtstr = "f8_e4m3",
                .dtype = RVCL_DTYPE_FP8_E4M3,
                .elt = 1,
                .golden = golden_fp8_e4m3_op,
                .fill = fill_fp8_e4m3_op,
                .fill_scale = NULL,
                .compare = compare_op_bits,
                .f32_to_storage = f32_to_fp8_e4m3_slot,
        },
        {
                .dtstr = "f8_e5m2",
                .dtype = RVCL_DTYPE_FP8_E5M2,
                .elt = 1,
                .golden = golden_fp8_e5m2_op,
                .fill = fill_fp8_e5m2_op,
                .fill_scale = NULL,
                .compare = compare_op_bits,
                .f32_to_storage = f32_to_fp8_e5m2_slot,
        },
        {
                .dtstr = "mxfp8e4m3",
                .dtype = RVCL_DTYPE_MXFP8_E4M3,
                .elt = 1,
                .golden = golden_mxfp8_e4m3_op,
                .fill = fill_fp8_e4m3_op,
                .fill_scale = fill_e8m0_scale_op,
                .compare = compare_op_bits,
                .f32_to_storage = f32_to_fp8_e4m3_slot,
        },
        {
                .dtstr = "mxfp8e5m2",
                .dtype = RVCL_DTYPE_MXFP8_E5M2,
                .elt = 1,
                .golden = golden_mxfp8_e5m2_op,
                .fill = fill_fp8_e5m2_op,
                .fill_scale = fill_e8m0_scale_op,
                .compare = compare_op_bits,
                .f32_to_storage = f32_to_fp8_e5m2_slot,
        },
        {
                .dtstr = "nvfp4",
                .dtype = RVCL_DTYPE_NVFP4,
                .elt = 1,
                .golden = golden_nvfp4_op,
                .fill = fill_e2m1_op,
                .fill_scale = fill_e4m3_scale_op,
                .compare = compare_op_bits,
                .f32_to_storage = NULL, /* sub-byte storage; no f32 convert */
        },
        {
                .dtstr = "mxfp4",
                .dtype = RVCL_DTYPE_MXFP4,
                .elt = 1,
                .golden = golden_mxfp4_op,
                .fill = fill_e2m1_op,
                .fill_scale = fill_e8m0_scale_op,
                .compare = compare_op_bits,
                .f32_to_storage = NULL, /* sub-byte storage; no f32 convert */
        },
        {
                .dtstr = "s8",
                .dtype = RVCL_DTYPE_INT8,
                .elt = 1,
                .golden = golden_s8_op,
                .fill = fill_s8_op,
                .fill_scale = NULL,
                .compare = compare_op_bits,
                .f32_to_storage = NULL, /* integer storage; no f32 convert */
        },
        {
                .dtstr = "u8",
                .dtype = RVCL_DTYPE_UINT8,
                .elt = 1,
                .golden = golden_u8_op,
                .fill = fill_u8_op,
                .fill_scale = NULL,
                .compare = compare_op_bits,
                .f32_to_storage = NULL, /* integer storage; no f32 convert */
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

rvcl_dtype_t dtype_ops_out_dtype(rvcl_dtype_t dt) {
    if (dt == RVCL_DTYPE_INT8 || dt == RVCL_DTYPE_UINT8)
        return RVCL_DTYPE_INT32;
    return RVCL_DTYPE_FP32;
}
