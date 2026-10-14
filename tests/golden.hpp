/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * tests/golden.hpp — shared golden helpers for the test executables
 *
 * SPDX-License-Identifier: Apache-2.0
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
void golden_f32(const float *a, const float *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k);
void golden_f16(const uint16_t *a, const uint16_t *b, const float *bias,
        float *c, int64_t m, int64_t n, int64_t k);

/* Deterministic LCG fill in [-0.5, 0.5): identical sequence on every
 * host and libc (no rand()), so differential and bench C mode see the
 * exact same inputs for a given shape. */
void lcg_fill_f32(float *buf, size_t n, uint32_t seed);
void lcg_fill_f16(uint16_t *buf, size_t n, uint32_t seed);

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

golden_cmp_t golden_cmp_bitexact(const float *got, const float *exp,
        int64_t n);
golden_cmp_t golden_cmp_tol(const float *got, const float *exp, int64_t n,
        float rel_tol);

#endif /* RVCL_TEST_GOLDEN_H */
