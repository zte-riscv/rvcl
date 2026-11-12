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
 * tests/ime_fixtures.hpp — shared fixtures for the IME backend tests
 *
 * Everything here is test-side oracle material, independent of the code
 * under test (the fp16 encoder below is local; the encoded values are
 * asserted by the data-domain guard of the numeric suite).
 *
 *  - upper-triangular 8x8 known answer: A/B are small integers, the
 *    expected C is a hand-computed literal table, so any layout mistake
 *    (transposed B, swapped tiles, lane misorder) fails bit-exactly;
 *  - a normal-domain fp16 filler: the matrix instruction flushes
 *    subnormal inputs to zero, so numeric cases must stay in the finite
 *    normal range for both paths to agree;
 *  - an independent oracle packer implementing the tile-layout formula
 *    from scratch — the pack tests compare against it byte for byte
 *    instead of against the implementation under test;
 *  - boundary shape sets covering the tile grid edges (1, 7, 8, 9, 15,
 *    16, 17, 31 and mixed-axis combinations).
 ******************************************************************************/

#ifndef RVCL_TEST_IME_FIXTURES_H
#define RVCL_TEST_IME_FIXTURES_H

#include "golden.hpp"

#include <stdint.h>
#include <stdlib.h>

/* ------------------------------------------------- local fp16 encoder
 *
 * Test-side round-to-nearest-even f32->f16. The fixtures encode their own
 * values instead of sharing the library-side soft conversion: fixtures
 * must stay independent of the code under test, and the encoded values
 * are asserted afterwards (domain guard below).
 */
inline uint16_t ime_encode_f16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t man = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) /* inf / nan */
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));
    if (exp >= 0x1f) /* overflow -> inf */
        return (uint16_t)(sign | 0x7c00u);
    if (exp <= 0) { /* underflow: subnormal or zero */
        if (exp < -10) return (uint16_t)sign;
        const uint32_t m = man | 0x800000u;
        const int shift = (int)(14 - exp + 13);
        const uint32_t half = 1u << (shift - 1);
        const uint32_t kept = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t out = kept;
        if (rem > half || (rem == half && (kept & 1u))) out++;
        return (uint16_t)(sign | out);
    }

    const uint32_t half = 1u << 12;
    uint32_t out = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > half || (rem == half && (out & 1u))) out++;
    if (out >= 0x7c00u) out = 0x7c00u; /* rounded up to inf */
    return (uint16_t)(sign | out);
}

/* --------------------------------------------- upper-triangular known answer
 *
 * A[i][q] = q+1 when q >= i else 0 (row-major 8x8), plain B[q][j] = j+1
 * when j >= q else 0; C[i][j] = sum_q A[i][q]*B[q][j]. All values are
 * exactly representable in fp16/fp32 and the sums stay below 2^24, so the
 * comparison is bit-exact on both the C and the instruction path.
 */
inline void ime_fixture_upper_a(uint16_t *a) {
    for (int i = 0; i < 8; i++)
        for (int q = 0; q < 8; q++)
            a[i * 8 + q] = ime_encode_f16(q >= i ? (float)(q + 1) : 0.0f);
}

inline void ime_fixture_upper_b(uint16_t *b) {
    for (int q = 0; q < 8; q++)
        for (int j = 0; j < 8; j++)
            b[q * 8 + j] = ime_encode_f16(j >= q ? (float)(j + 1) : 0.0f);
}

static const float ime_fixture_upper_c[64] = {
        1,
        6,
        18,
        40,
        75,
        126,
        196,
        288,
        0,
        4,
        15,
        36,
        70,
        120,
        189,
        280,
        0,
        0,
        9,
        28,
        60,
        108,
        175,
        264,
        0,
        0,
        0,
        16,
        45,
        90,
        154,
        240,
        0,
        0,
        0,
        0,
        25,
        66,
        126,
        208,
        0,
        0,
        0,
        0,
        0,
        36,
        91,
        168,
        0,
        0,
        0,
        0,
        0,
        0,
        49,
        120,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        64,
};

/* ------------------------------------------------ normal-domain fp16 filler
 *
 * The matrix instruction flushes subnormal inputs to zero while the soft
 * conversions do not, so numeric inputs must stay in the finite normal
 * domain: magnitudes are clamped to [2^-13, 0.5). The LCG sequence
 * matches the golden helpers' data contract (same seed, same order).
 */
inline void ime_fill_f16_normal(uint16_t *buf, size_t n, uint32_t seed) {
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) abort();
    lcg_fill_f32(tmp, n, seed);
    for (size_t i = 0; i < n; i++) {
        float v = tmp[i];
        float mag = v < 0.0f ? -v : v;
        if (mag < 0x1p-13f) mag = 0x1p-13f;
        buf[i] = ime_encode_f16(v < 0.0f ? -mag : mag);
    }
    free(tmp);
}

/* Bias values in the same spirit: finite fp32, magnitude within
 * [2^-13, 0.5] or exactly zero. */
inline void ime_fill_bias_normal(float *buf, size_t n, uint32_t seed) {
    float *tmp = (float *)malloc(n * sizeof(float));
    if (!tmp) abort();
    lcg_fill_f32(tmp, n, seed);
    for (size_t i = 0; i < n; i++) {
        float v = tmp[i];
        float mag = v < 0.0f ? -v : v;
        buf[i] = mag < 0x1p-13f ? 0.0f : v;
    }
    free(tmp);
}

/* --------------------------------------------- independent oracle packer
 *
 * Implements the tile layout formula directly (n tile outer, k tile
 * inner, tile element nn*8+kk), sharing no code with ime_pack_b_tiles.
 */
inline void ime_oracle_pack_b(
        uint16_t *dst, const uint16_t *b, int64_t n, int64_t k_dim) {
    const int64_t n_t = n / 8 + (n % 8 != 0);
    const int64_t k_t = k_dim / 8 + (k_dim % 8 != 0);
    for (int64_t nt = 0; nt < n_t; nt++)
        for (int64_t kt = 0; kt < k_t; kt++)
            for (int64_t nn = 0; nn < 8; nn++)
                for (int64_t kk = 0; kk < 8; kk++) {
                    const int64_t gn = nt * 8 + nn;
                    const int64_t gk = kt * 8 + kk;
                    const int64_t idx = (nt * k_t + kt) * 64 + nn * 8 + kk;
                    dst[idx] = (gn < n && gk < k_dim) ? b[gk * n + gn] : 0;
                }
}

/* ------------------------------------------------------ boundary shape set
 *
 * Diagonal shapes walk every tile-grid edge; mixed shapes combine edge
 * sizes across axes; the (N,K) pairs exercise the pack formula around
 * the tile boundaries.
 */
static const int64_t ime_shapes_diag[] = {1, 7, 8, 9, 15, 16, 17, 31};

static const int64_t ime_shapes_mixed[][3] = {
        {1, 8, 8},
        {8, 1, 8},
        {8, 8, 1},
        {7, 9, 15},
        {9, 7, 15},
        {15, 17, 31},
        {31, 17, 15},
        {17, 31, 9},
        {33, 19, 47},
};

static const int64_t ime_shapes_nk[][2] = {
        {1, 31},
        {31, 1},
        {9, 17},
        {17, 9},
        {15, 9},
        {9, 15},
        {31, 8},
        {8, 31},
};

/* -------------------------------------------------------- data-domain guard
 *
 * A value passes when it is +/-0 or a finite normal fp16 (non-zero
 * exponent, not all-ones). Numeric cases assert this on every generated
 * buffer: a subnormal would silently flush to zero on the instruction
 * path and invalidate the comparison.
 */
inline int ime_is_normal_or_zero_fp16(uint16_t h) {
    if ((h & 0x7fffu) == 0) return 1;
    const unsigned exp = (h >> 10) & 0x1fu;
    return exp != 0 && exp != 0x1fu;
}

inline int ime_buffer_all_normal(const uint16_t *buf, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (!ime_is_normal_or_zero_fp16(buf[i])) return 0;
    return 1;
}

#endif /* RVCL_TEST_IME_FIXTURES_H */
