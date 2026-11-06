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
 * dispatch.h — kernel selection (Layer: Dispatch)
 *
 * Filters candidates and picks the best legal kernel. Dispatch never
 * contains ISA-specific kernel code and never executes kernels.
 ******************************************************************************/

#ifndef RVCL_DISPATCH_H
#define RVCL_DISPATCH_H

#include "rvcl/kernel.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque dispatch result: selected kernel instance + configure outcome. */
typedef struct rvcl_dispatch_result rvcl_dispatch_result_t;

/*
 * Dispatch pipeline:
 *   1. operation, 2. datatype (exact (A,B,acc,C) match after AUTO
 *   resolution; a quantized request keeps only kernels that declare
 *   quantization support), 3. required-feature subset, 4. VLEN tier,
 *   5. shape bounds, 6. kernel validate().
 * Survivors are ordered by their cost callback; the smallest cost wins
 * and an equal cost keeps the earliest registered candidate. On success
 * the kernel is configured and workspace demands are reported; the
 * result is run repeatedly via rvcl_kernel_run().
 *
 * AUTO is only accepted in the accumulator position (resolved to INT32
 * for INT8/UINT8 inputs, FP32 otherwise) and in the bias slot (no bias).
 * A rejected datatype tuple returns RVCL_UNSUPPORTED_DTYPE_COMBO;
 * returns RVCL_UNSUPPORTED when no candidate survives.
 *
 * The registry may change while candidates are evaluated; dispatch then
 * restarts the selection. If the registry never stays stable for one
 * full attempt (pathological registration churn), the call gives up
 * after a bounded number of retries and returns RVCL_UNSUPPORTED even
 * though no kernel decision was reached.
 */
RVCL_API rvcl_status_t rvcl_dispatch_matmul(
        const rvcl_matmul_desc_t *desc, rvcl_dispatch_result_t **result);

/* Kernel name of the selection (for verbose / tests). */
RVCL_API const char *rvcl_dispatch_kernel_name(const rvcl_dispatch_result_t *r);

/* Workspace demands reported by configure(). */
RVCL_API const rvcl_workspace_req_t *rvcl_dispatch_ws_req(
        const rvcl_dispatch_result_t *r, int *num_ws);

/* The configured kernel instance (borrowed; owned by the result). */
RVCL_API rvcl_kernel_t *rvcl_dispatch_kernel(rvcl_dispatch_result_t *r);

/* Releases the dispatch result (and the kernel instance). */
RVCL_API void rvcl_dispatch_release(rvcl_dispatch_result_t *r);

/* ---------------------------------------------------------- dispatch cache
 * Repeated dispatches with an identical request (all request fields,
 * compared by raw value) reuse the configured kernel instance — no
 * re-filtering, re-validate or re-configure. Any registry change
 * invalidates the whole cache. The capability snapshot is otherwise
 * fixed for the process lifetime; only the internal test-only override
 * replaces it, and that override clears the cache itself.
 * rvcl_dispatch_release() detaches a cache-owned result safely.
 * Clearing the cache does not reset the counters below.
 */
RVCL_API rvcl_status_t rvcl_dispatch_cache_clear(void);

/* Process-lifetime hit/miss counters for verbose/profiling output. */
RVCL_API int rvcl_dispatch_cache_hits(void);
RVCL_API int rvcl_dispatch_cache_misses(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_DISPATCH_H */
