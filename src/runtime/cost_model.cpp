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

/* The cost callback contract (kernel.h) asks each kernel for a non-negative
 * cycle estimate, smaller = better; 0 means "no cost information". This file
 * provides the real, explainable scores that replace the zero default for
 * the matmul categories:
 *   score = per_element_coefficient x shape_heuristic x vlen_tier_gain
 * The per-element coefficient is a structured heuristic default per category
 * (matmul-rvv cheap, matmul-reference expensive), not measured data; the
 * shape_heuristic is a fixed-point penalty for partial tail tiles and is
 * neutral when a kernel declares no preferred hint; the vlen_tier_gain
 * shrinks the matmul-rvv coefficient as the host VLEN tier widens, with no
 * gain for the reference. Ordering guarantee: whenever a matmul-rvv candidate
 * survives filtering it scores below the matmul-reference fallback, so
 * dispatch picks the RVV entry on a capable host; the reference wins only
 * when no matmul-rvv candidate survives. Equal scores keep the earlier
 * registration (dispatch.cpp), which is how the dedicated VLEN tiers and the
 * generic entry tie-break today. */

#include "capability_internal.h"
#include "cost_model.h"

#include <stdint.h>

/* ---------------------------------------------------------- coefficients
 *
 * Structured default constants (estimated cycles per output element),
 * deliberately ordered so the matmul-rvv category is always cheaper than
 * matmul-reference when both are eligible. Real-machine calibration that
 * replaces these with registered per-kernel data is a future step; the
 * shape heuristic and tier gain below already consume the request and
 * capability so the backfill only swaps the coefficient table, not the
 * scoring path.
 */
enum {
    RVCL_COST_ELEM_MATMUL_RVV = 1, /* vector micro-kernel, cheap */
    RVCL_COST_ELEM_MATMUL_IME = 1, /* matrix micro-kernel, same order */
    RVCL_COST_ELEM_MATMUL_REF = 64, /* scalar triple loop, ~64x slower */
};

/* Preferred tile of the matrix micro-kernel: the shape heuristic must
 * see the same 8x8x8 grid the kernel tiles over. */
enum { RVCL_COST_IME_TILE = 8 };

/* Fixed-point scale: 256 represents a 1.0x multiplier. */
enum { RVCL_COST_FP = 256 };

/* -------------------------------------------------------- shape heuristic
 *
 * Preferred-tile utilization: a request whose m/n/k does not fill a kernel
 * preferred tile evenly leaves a partial tail tile that wastes up to one
 * panel of work. Costed as (tiles + 1) / tiles on the affected dimension,
 * folded over m, n, k. A kernel with no preferred hint (pref <= 0) or a
 * dimension that divides evenly is neutral (256 = 1.0x). */
static uint64_t tail_penalty(int64_t dim, int64_t pref) {
    if (pref <= 0 || dim % pref == 0) return RVCL_COST_FP;
    uint64_t tiles = (uint64_t)(dim / pref);
    if (tiles == 0) tiles = 1;
    return (RVCL_COST_FP * (tiles + 1)) / tiles;
}

static uint64_t shape_factor(
        const rvcl_matmul_desc_t *d, int64_t pm, int64_t pn, int64_t pk) {
    uint64_t f = tail_penalty(d->m, pm);
    f = (f * tail_penalty(d->n, pn)) / RVCL_COST_FP;
    f = (f * tail_penalty(d->k, pk)) / RVCL_COST_FP;
    return f;
}

/* ---------------------------------------------------------- tier gain
 *
 * VLEN tier gain for the RVV category (fixed-point, 256 = 1.0x): wider
 * vector registers amortize the micro-kernel overhead, so the score shrinks
 * as the host tier grows. The reference gets no gain — it is a scalar loop
 * regardless of VLEN. Tier-parameterized specialization of individual
 * kernels is a future step and does not change this host-driven gain.
 * Exported for the test executables (see cost_model.h).
 */
RVCL_API uint64_t vlen_tier_gain(const rvcl_capability_t *cap) {
    switch (rvcl_capability_vlen_tier(cap)) {
        case RVCL_VLEN_TIER_1024: return 128; /* 0.5x */
        case RVCL_VLEN_TIER_512: return 160; /* ~0.62x */
        case RVCL_VLEN_TIER_256: return 200; /* ~0.78x */
        case RVCL_VLEN_TIER_128:
        default: return RVCL_COST_FP; /* 1.0x */
    }
}

/* Combine coeff x shape x gain back to an integer cycle estimate, never
 * reporting 0 (which the contract reserves for "no cost information"). */
static uint64_t score(uint64_t coeff, uint64_t shape, uint64_t gain) {
    uint64_t v = (coeff * shape * gain)
            / ((uint64_t)RVCL_COST_FP * (uint64_t)RVCL_COST_FP);
    return v < 1 ? 1 : v;
}

/* ------------------------------------------------------- category scores */

RVCL_API uint64_t rvcl_cost_matmul_rvv(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    /* Preferred-tile hints for the built-in entries are currently 0
     * (neutral); the hints arrive with the real-machine backfill. */
    const rvcl_matmul_desc_t *d = &desc->matmul;
    const uint64_t shape = shape_factor(d, 0, 0, 0);
    const uint64_t gain = vlen_tier_gain(cap);
    return score(RVCL_COST_ELEM_MATMUL_RVV, shape, gain);
}

RVCL_API uint64_t rvcl_cost_matmul_ime(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    const uint64_t rvv = rvcl_cost_matmul_rvv(desc, cap);
    if (!rvcl_capability_has(cap, RVCL_FEATURE_IME)) {
        /* No extension on this host: the hardware-bound entry never
         * survives the feature filter, and the portable entry must sort
         * after the RVV one so non-IME hosts keep their f16 selection. */
        return rvv == UINT64_MAX ? rvv : rvv + 1;
    }
    const rvcl_matmul_desc_t *d = &desc->matmul;
    const uint64_t shape = shape_factor(d, RVCL_COST_IME_TILE,
            RVCL_COST_IME_TILE, RVCL_COST_IME_TILE);
    const uint64_t gain = vlen_tier_gain(cap);
    /* Sub-unit fixed point: 1024 units = one score step, so a partial tail
     * tile stays visible before the hard clamp below. */
    const uint64_t units = (RVCL_COST_ELEM_MATMUL_IME * shape * gain * 1024)
            / ((uint64_t)RVCL_COST_FP * RVCL_COST_FP);
    const uint64_t rounded = (units + 1023) / 1024;
    /* Never above the RVV category: the more specialized entry wins or
     * ties, and registration order resolves the tie. */
    return rounded <= rvv ? rounded : rvv;
}

RVCL_API uint64_t rvcl_cost_matmul_reference(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    /* The reference category is VLEN-independent (a scalar triple loop),
     * so its score is exactly the public hint at the reference
     * coefficient: ceil(64 x 256 / 256) = 64 loses nothing to rounding
     * and the hint pins the neutral gain the direct computation used,
     * making the two identical per shape. Forwarding keeps the category
     * on the one public scoring path instead of a private copy. The
     * cheap-vector wrapper above cannot do the same: its host-dependent
     * tier gain has no hint equivalent (the hint pins the tie tier), so
     * it keeps the direct scoring. */
    return rvcl_cost_matmul_hint(
            desc, cap, (uint64_t)RVCL_COST_ELEM_MATMUL_REF * RVCL_COST_FP);
}

RVCL_API uint64_t rvcl_cost_conv2d_rvv(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    /* The convolution is served by the same cheap vector micro-kernel, so
     * it shares the matmul-rvv coefficient; the shape heuristic is neutral
     * (no preferred tile hints yet), matching the matmul categories. */
    (void)desc;
    const uint64_t gain = vlen_tier_gain(cap);
    return score(RVCL_COST_ELEM_MATMUL_RVV, RVCL_COST_FP, gain);
}

RVCL_API uint64_t rvcl_cost_inner_product_rvv(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    /* The inner product is the same cheap vector micro-kernel over the
     * descriptor's own GEMM projection, so it shares the conv2d shape
     * treatment: the matmul-rvv coefficient with a neutral heuristic. */
    (void)desc;
    const uint64_t gain = vlen_tier_gain(cap);
    return score(RVCL_COST_ELEM_MATMUL_RVV, RVCL_COST_FP, gain);
}

RVCL_API uint64_t rvcl_cost_pooling_rvv(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    /* The pooling kernel streams one output row per vector step, so it
     * shares the matmul-rvv coefficient and the neutral shape heuristic. */
    (void)desc;
    const uint64_t gain = vlen_tier_gain(cap);
    return score(RVCL_COST_ELEM_MATMUL_RVV, RVCL_COST_FP, gain);
}

/* ------------------------------------------------------------- public hint
 *
 * The out-of-tree counterpart of the category wrappers: same shape heuristic
 * and fixed-point scoring, with the per-element coefficient supplied by the
 * vendor as a fixed-point multiplier over the built-in cheap vector category.
 * The gain is pinned to the tie tier (RVCL_COST_FP): a vendor tile machine
 * has no VLEN dependence, so a vendor hint scores like the built-in rvv
 * category at tier 128 (coeff_256 == 256) and stays at or below it for
 * coeff_256 < 256. Ceil division keeps a sub-256 coefficient from rounding
 * its term to zero. */
RVCL_API uint64_t rvcl_cost_matmul_hint(const rvcl_op_desc_t *desc,
        const rvcl_capability_t *cap, uint64_t coeff_256) {
    (void)cap; /* no tier gain: vendor tile cost is independent of VLEN */
    const rvcl_matmul_desc_t *d = &desc->matmul;
    const uint64_t shape = shape_factor(d, 0, 0, 0);
    const uint64_t product = RVCL_COST_ELEM_MATMUL_RVV * coeff_256;
    const uint64_t coeff = (product + RVCL_COST_FP - 1) / RVCL_COST_FP;
    return score(coeff, shape, RVCL_COST_FP);
}
