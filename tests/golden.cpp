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
 * tests/golden.cpp — shared golden helpers (see golden.hpp for the contract)
 ******************************************************************************/

#include "golden.hpp"

#include <math.h>
#include <string.h>

#include "../src/backends/rvv/rvv_fp16.hpp"
#include "../src/backends/rvv/rvv_bf16.hpp"
#include "../src/backends/rvv/rvv_fp8.hpp"
#include "../src/backends/rvv/rvv_e8m0.hpp"

/* ------------------------------------------------------------- goldens */

void golden_f32(const float *a, const float *b, const float *bias, float *c,
        int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k; p++)
                acc += a[i * k + p] * b[p * n + j];
            c[i * n + j] = acc;
        }
}

void golden_f16(const uint16_t *a, const uint16_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k; p++)
                acc += rvv_f16_to_f32(a[i * k + p])
                        * rvv_f16_to_f32(b[p * n + j]);
            c[i * n + j] = acc;
        }
}

void golden_bf16(const uint16_t *a, const uint16_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k; p++)
                acc += rvv_bf16_to_f32(a[i * k + p])
                        * rvv_bf16_to_f32(b[p * n + j]);
            c[i * n + j] = acc;
        }
}

/* fp8 goldens share one body parameterized by the dtype upconversion. */
template <float (*UP)(uint8_t)>
static void golden_fp8_impl(const uint8_t *a, const uint8_t *b,
        const float *bias, float *c, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k; p++)
                acc += UP(a[i * k + p]) * UP(b[p * n + j]);
            c[i * n + j] = acc;
        }
}

void golden_fp8_e4m3(const uint8_t *a, const uint8_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k) {
    golden_fp8_impl<rvv_fp8_e4m3_to_f32>(a, b, bias, c, m, n, k);
}

void golden_fp8_e5m2(const uint8_t *a, const uint8_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k) {
    golden_fp8_impl<rvv_fp8_e5m2_to_f32>(a, b, bias, c, m, n, k);
}

/* INT8 goldens share one body parameterized by the signedness. Each byte
 * is widened to int32 and the product is exact in int32, and integer
 * accumulation is associative on the exact values, so the ascending-K
 * order (bias first) is bit-identical to the C and JIT kernels. */
template <typename T> /* int8_t or uint8_t */
static void golden_s8u8_impl(const T *a, const T *b, const int32_t *bias,
        int32_t *c, int64_t m, int64_t n, int64_t k) {
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            int32_t acc = bias ? bias[j] : 0;
            for (int64_t p = 0; p < k; p++)
                acc += (int32_t)a[i * k + p] * (int32_t)b[p * n + j];
            c[i * n + j] = acc;
        }
}

void golden_s8(const int8_t *a, const int8_t *b, const int32_t *bias,
        int32_t *c, int64_t m, int64_t n, int64_t k) {
    golden_s8u8_impl<int8_t>(a, b, bias, c, m, n, k);
}

void golden_u8(const uint8_t *a, const uint8_t *b, const int32_t *bias,
        int32_t *c, int64_t m, int64_t n, int64_t k) {
    golden_s8u8_impl<uint8_t>(a, b, bias, c, m, n, k);
}

/* MXFP8 goldens share one body parameterized by the FP8 dtype upconversion.
 * Each K term folds both E8M0 scales into the B value (the combined scale is
 * an exact power of two, so B*ascale*bscale never rounds a finite term) and
 * accumulates with a single fused multiply-add, acc = fma(A, B*ascale*bscale,
 * acc) — the exact grouping of the C and JIT kernels, so the accumulate
 * order (bias first, ascending k) is bit-identical on finite data. */
template <float (*UP)(uint8_t)>
static void golden_mxfp8_impl(const uint8_t *a, const uint8_t *sa,
        const uint8_t *b, const uint8_t *sb, const float *bias, float *c,
        int64_t m, int64_t n, int64_t k, int64_t blk) {
    const int64_t nblk = k / blk;
    for (int64_t i = 0; i < m; i++)
        for (int64_t j = 0; j < n; j++) {
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k; p++) {
                const int64_t pblk = p / blk;
                const float ascale = rvv_e8m0_to_f32(sa[i * nblk + pblk]);
                const float bscale = rvv_e8m0_to_f32(sb[pblk * n + j]);
                const float bscl = ascale * bscale;
                acc = fmaf(UP(a[i * k + p]), UP(b[p * n + j]) * bscl, acc);
            }
            c[i * n + j] = acc;
        }
}

void golden_mxfp8_e4m3(const uint8_t *a, const uint8_t *sa, const uint8_t *b,
        const uint8_t *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    golden_mxfp8_impl<rvv_fp8_e4m3_to_f32>(a, sa, b, sb, bias, c, m, n, k, blk);
}

void golden_mxfp8_e5m2(const uint8_t *a, const uint8_t *sa, const uint8_t *b,
        const uint8_t *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk) {
    golden_mxfp8_impl<rvv_fp8_e5m2_to_f32>(a, sa, b, sb, bias, c, m, n, k, blk);
}

/* ----------------------------------------------------------- LCG fill */

/* 32-bit LCG (exact in uint32 arithmetic — same stream everywhere). */
static uint32_t lcg_next(uint32_t *state) {
    *state = *state * 1664525u + 1013904223u;
    return *state;
}

void lcg_fill_f32(float *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (float)(lcg_next(&seed) >> 8) / 16777216.0f - 0.5f;
}

void lcg_fill_f16(uint16_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = rvv_f32_to_f16(
                (float)(lcg_next(&seed) >> 8) / 16777216.0f - 0.5f);
}

void lcg_fill_bf16(uint16_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = rvv_f32_to_bf16(
                (float)(lcg_next(&seed) >> 8) / 16777216.0f - 0.5f);
}

void lcg_fill_fp8_e4m3(uint8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = rvv_f32_to_fp8_e4m3(
                (float)(lcg_next(&seed) >> 8) / 16777216.0f - 0.5f);
}

void lcg_fill_fp8_e5m2(uint8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = rvv_f32_to_fp8_e5m2(
                (float)(lcg_next(&seed) >> 8) / 16777216.0f - 0.5f);
}

void lcg_fill_s8(int8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (int8_t)(lcg_next(&seed) >> 24);
}

void lcg_fill_u8(uint8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (uint8_t)(lcg_next(&seed) >> 24);
}

void lcg_fill_i32(int32_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++)
        buf[i] = (int32_t)lcg_next(&seed);
}

void lcg_fill_e8m0(uint8_t *buf, size_t n, uint32_t seed) {
    for (size_t i = 0; i < n; i++) {
        /* Bounded exponent range so a corpus scale byte can never overflow
         * the f32 GEMM accumulation when combined with a full-magnitude fp8
         * element: with E5M2's largest element (57344) and two scale bytes
         * the per-term magnitude is 57344^2 * 2^(2*(e8m0-127)); capping the
         * byte at 0xA0 (2^33) keeps even a K-wide sum well inside f32 range.
         * This preserves the bit-exactness guarantee — E8M0 scales are exact
         * powers of two, so with the shared fma grouping every *finite*
         * product and sum is bit-identical across golden/C/JIT on any
         * compiler. An unbounded random byte (or
         * the 0xFE maximum exponent) can overflow to Inf, and the resulting
         * Inf/NaN pattern is compiler-dependent (0*Inf, Inf+(-Inf)), which
         * is why the extreme boundary exponents are exercised by the
         * conversion smoke in the differential instead. */
        buf[i] = (uint8_t)(0x01 + (lcg_next(&seed) >> 24) % 0xA0);
    }
    /* Subnormal boundary: 0x00 -> 2^-127. Underflow to 0/subnormal is
     * bit-exact, so keeping it in the corpus stays arch-stable and always
     * exercises the special subnormal conversion. */
    if (n >= 1) buf[0] = 0x00;
}

void golden_sentinel(void *buf, size_t bytes) {
    memset(buf, 0x5A, bytes);
}

uint32_t golden_case_seed(int64_t m, int64_t n, int64_t k) {
    uint64_t s = (uint64_t)m * 1000003u ^ (uint64_t)n * 2654435761u
            ^ (uint64_t)k * 40503u;
    return (uint32_t)(s | 1u);
}

/* ------------------------------------------------------------ compare */

golden_cmp_t golden_cmp_bitexact(
        const float *got, const float *exp, int64_t n) {
    golden_cmp_t r = {
            .mismatches = 0, .first = -1, .exp_first = 0.0f, .got_first = 0.0f};
    for (int64_t i = 0; i < n; i++) {
        if (got[i] != exp[i]) {
            if (r.first < 0) {
                r.first = i;
                r.exp_first = exp[i];
                r.got_first = got[i];
            }
            r.mismatches++;
        }
    }
    return r;
}

golden_cmp_t golden_cmp_bitexact_bits(
        const float *got, const float *exp, int64_t n) {
    golden_cmp_t r = {
            .mismatches = 0, .first = -1, .exp_first = 0.0f, .got_first = 0.0f};
    for (int64_t i = 0; i < n; i++) {
        uint32_t gb, eb;
        __builtin_memcpy(&gb, &got[i], 4);
        __builtin_memcpy(&eb, &exp[i], 4);
        if (gb != eb) {
            if (r.first < 0) {
                r.first = i;
                r.exp_first = exp[i];
                r.got_first = got[i];
            }
            r.mismatches++;
        }
    }
    return r;
}

golden_cmp_t golden_cmp_tol(
        const float *got, const float *exp, int64_t n, float rel_tol) {
    golden_cmp_t r = {
            .mismatches = 0, .first = -1, .exp_first = 0.0f, .got_first = 0.0f};
    for (int64_t i = 0; i < n; i++) {
        const float diff = got[i] > exp[i] ? got[i] - exp[i] : exp[i] - got[i];
        if (diff > rel_tol * (1.0f + (exp[i] < 0 ? -exp[i] : exp[i]))) {
            if (r.first < 0) {
                r.first = i;
                r.exp_first = exp[i];
                r.got_first = got[i];
            }
            r.mismatches++;
        }
    }
    return r;
}
