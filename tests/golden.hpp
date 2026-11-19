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
 * tests/golden.hpp — shared golden helpers for the test executables
 *
 * Data semantics have a single point of truth here (linked into both
 * rvcl_differential and rvcl_bench --mode=C); the corpus parsers are
 * deliberately double-implemented so the two consumers cross-validate
 * the format spec against each other.
 *
 * The golden matmuls are hand-written naive loops with zero code shared
 * with the library — an implementation bug cannot hide in a shared ref.
 *
 * Bit-exactness note: vfmacc is a fused (single-rounding) multiply-add,
 * so the scalar golden must contract `acc += a * b` to fmadd the same
 * way — build with optimization (Release / -O2+) on vector builds for
 * the fp32 bit-exact checks to hold.
 ******************************************************************************/

#ifndef RVCL_TEST_GOLDEN_H
#define RVCL_TEST_GOLDEN_H

#include <stddef.h>
#include <stdint.h>

/* C = bias? + A * B, row-major plain, accumulate fp32. */
void golden_f32(const float *a, const float *b, const float *bias, float *c,
        int64_t m, int64_t n, int64_t k);
void golden_f16(const uint16_t *a, const uint16_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k);
void golden_bf16(const uint16_t *a, const uint16_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k);
void golden_fp8_e4m3(const uint8_t *a, const uint8_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k);
void golden_fp8_e5m2(const uint8_t *a, const uint8_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k);

/* MXFP8 block-scale goldens: A/B are FP8 bytes and carry the 1-byte E8M0
 * per-32-block exponent scales sa (m x K/blk, row-major) and sb
 * (K/blk x n, row-major — oneDNN-aligned), exactly the layouts the C/JIT
 * kernels consume. Dequant = element * 2^(e-127). blk is the block width
 * (32). Accumulates bias-first in ascending k, bit-identical to the
 * kernels (E8M0 scales are exact powers of two, so multiply grouping does
 * not round a finite product). */
void golden_mxfp8_e4m3(const uint8_t *a, const uint8_t *sa, const uint8_t *b,
        const uint8_t *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk);
void golden_mxfp8_e5m2(const uint8_t *a, const uint8_t *sa, const uint8_t *b,
        const uint8_t *sb, const float *bias, float *c, int64_t m, int64_t n,
        int64_t k, int64_t blk);

/* Deterministic LCG fill in [-0.5, 0.5): identical sequence on every
 * host and libc (no rand()), so differential and bench C mode see the
 * exact same inputs for a given shape. */
void lcg_fill_f32(float *buf, size_t n, uint32_t seed);
void lcg_fill_f16(uint16_t *buf, size_t n, uint32_t seed);
void lcg_fill_bf16(uint16_t *buf, size_t n, uint32_t seed);
void lcg_fill_fp8_e4m3(uint8_t *buf, size_t n, uint32_t seed);
void lcg_fill_fp8_e5m2(uint8_t *buf, size_t n, uint32_t seed);

/* Deterministic E8M0 scale fill: uniform bytes in a finite-safe exponent
 * range (2^-126 .. 2^33) plus the 0x00 subnormal boundary at buf[0] — a
 * corpus case always exercises the special subnormal conversion, while the
 * bounded exponents keep every product and sum finite so the bit-exact
 * compare holds on any compiler. The extreme boundary bytes (0xFE = 2^127,
 * 0xFF = NaN) are NOT injected here: a 2^127 scale overflows an f32 product
 * with a full-magnitude fp8 element and the resulting Inf/NaN pattern is
 * compiler-dependent, so those conversions are covered by the dedicated
 * differential smoke on rvv_e8m0_to_f32 instead. Uses the same LCG stream
 * as the element fills. */
void lcg_fill_e8m0(uint8_t *buf, size_t n, uint32_t seed);

/* Stable per-shape seed (the mix below is part of the data contract:
 * differential and bench C mode derive identical buffers from it). */
uint32_t golden_case_seed(int64_t m, int64_t n, int64_t k);

/* Prefill output memory with a 0x5A byte pattern so "kernel never wrote
 * this region" is caught by the element compare instead of passing on
 * stale zeros (anti trivially-broken guard). */
void golden_sentinel(void *buf, size_t bytes);

/* Element compare. `first` is the linear index of the first mismatch
 * (-1 when none); exp/got carry its values; mismatches is the count. */
typedef struct {
    int64_t mismatches;
    int64_t first;
    float exp_first, got_first;
} golden_cmp_t;

golden_cmp_t golden_cmp_bitexact(const float *got, const float *exp, int64_t n);
golden_cmp_t golden_cmp_bitexact_bits(
        const float *got, const float *exp, int64_t n);
golden_cmp_t golden_cmp_tol(
        const float *got, const float *exp, int64_t n, float rel_tol);

#endif /* RVCL_TEST_GOLDEN_H */
