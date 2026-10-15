/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * tests/golden.cpp — shared golden helpers (see golden.hpp for the contract)
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "golden.hpp"

#include <string.h>

#include "../src/backends/rvv/rvv_fp16.hpp"

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
