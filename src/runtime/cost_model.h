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
 * cost_model.h — internal kernel cost-model wrappers (not installed)
 *
 * Declarations shared between the library and its test executables only.
 * This header is deliberately outside include/rvcl/: it is not part of
 * the public API, is never installed, and carries no stability promise.
 *
 * The scoring wrappers below are the rvcl_kernel_cost_fn implementations
 * the backends bind instead of rvcl_kernel_cost_zero(). They estimate a
 * non-negative cycle count (smaller = better) from the request shape and
 * the host capability snapshot so dispatch can order surviving candidates
 * by an explainable heuristic. The real-machine calibration that turns
 * the default coefficients into measured data does not change this
 * interface.
 ******************************************************************************/

#ifndef RVCL_COST_MODEL_H
#define RVCL_COST_MODEL_H

#include "rvcl/kernel.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MatMul, RVV backend category: heuristic cost = per-element coefficient
 * (cheap) x shape heuristic x VLEN-tier gain. All matmul-rvv entries share
 * this wrapper, so they tie among themselves and dispatch resolves the tie
 * by registration order (dedicated tiers high -> low, then generic), the
 * same outcome the zero-cost default produced. The wrapper stays below the
 * reference cost whenever it survives filtering, so an RVV entry always
 * wins over the reference on a capable host.
 */
RVCL_API uint64_t rvcl_cost_matmul_rvv(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap);

/*
 * MatMul, reference backend category: heuristic cost = per-element
 * coefficient (expensive, the correctness-first triple loop) x shape
 * heuristic, no tier gain. Registered last as the unbounded fallback, so
 * it only wins when no matmul-rvv candidate survives filtering.
 */
RVCL_API uint64_t rvcl_cost_matmul_reference(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap);

/*
 * VLEN tier gain (fixed-point, 256 = 1.0x) applied to the matmul-rvv
 * category: wider vector registers amortize the micro-kernel overhead, so
 * the gain shrinks as the host tier grows. Exported for the test
 * executables to pin the tier-gain table directly (the combined wrapper
 * score floors to 1 at the current cheap per-element coefficient, so the
 * gain is not observable through rvcl_cost_matmul_rvv alone). Not
 * installed as API.
 */
RVCL_API uint64_t vlen_tier_gain(const rvcl_capability_t *cap);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_COST_MODEL_H */
