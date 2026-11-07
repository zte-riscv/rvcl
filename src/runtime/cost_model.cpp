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
 * cost_model.cpp — heuristic kernel cost model
 *
 * The cost callback contract (kernel.h) asks each kernel for a non-negative
 * cycle estimate, smaller = better; 0 means "no cost information". Backends
 * without a real model bind rvcl_kernel_cost_zero(). This file provides the
 * real, explainable scores that replace the zero default for the matmul
 * categories:
 *
 *   score = per_element_coefficient x shape_heuristic x vlen_tier_gain
 *
 *   - per_element_coefficient: a structured default constant per category
 *     (matmul-rvv cheap, matmul-reference expensive). These are heuristic
 *     defaults, not measured data; the real-machine backfill that turns the
 *     constant table into registrable per-kernel data is a future
 *     calibration step (see the table below).
 *   - shape_heuristic: dimensionless fixed-point penalty from the request
 *     against the kernel's preferred-tile hints (partial tail tiles cost a
 *     little extra). Neutral when a kernel declares no preferred hint, so
 *     the current built-in entries (all hints 0) score purely by category
 *     and host tier.
 *   - vlen_tier_gain: matmul-rvv amortizes better at wider VLEN, so its
 *     coefficient shrinks with the host tier; the reference gets no gain.
 *
 * Ordering guarantee: whenever a matmul-rvv candidate survives filtering
 * it scores below the matmul-reference fallback, so dispatch picks the RVV
 * entry on a capable host; the reference wins only when no matmul-rvv
 * candidate survives (feature/shape/datatype filtering). Equal scores keep
 * the earlier registration (dispatch.cpp), which is how the dedicated
 * VLEN tiers and the generic entry tie-break today.
 ******************************************************************************/

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
    RVCL_COST_ELEM_MATMUL_REF = 64, /* scalar triple loop, ~64x slower */
};

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

RVCL_API uint64_t rvcl_cost_matmul_reference(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap; /* the reference is a scalar loop: no VLEN tier gain */
    const rvcl_matmul_desc_t *d = &desc->matmul;
    const uint64_t shape = shape_factor(d, 0, 0, 0);
    return score(RVCL_COST_ELEM_MATMUL_REF, shape, RVCL_COST_FP);
}
