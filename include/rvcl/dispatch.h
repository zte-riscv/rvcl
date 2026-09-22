/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * dispatch.h — kernel selection (Layer: Dispatch)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Filters + score, picks the best legal kernel (PLAN §4.2). Dispatch never
 * contains ISA-specific kernel code and never executes kernels (§4.1).
 * Phase 0 implements the first four filter stages with static priority;
 * the cost model (stage 7) lands in v0.4.
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
 * Dispatch pipeline (PLAN §4.2, Phase 0 scope = stages 1-4):
 *   1. operation filter   2. datatype filter (exact (A,B,acc,C) match)
 *   3. capability filter  4. shape filter
 * On success the kernel is configured and workspace demands are reported;
 * the result is run repeatedly via rvcl_kernel_run().
 * Returns RVCL_UNSUPPORTED when no candidate survives.
 */
RVCL_API rvcl_status_t rvcl_dispatch_matmul(const rvcl_matmul_desc_t *desc,
        rvcl_dispatch_result_t **result);

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
 * PLAN §6.2 item 9: repeated dispatches with an identical matmul request
 * reuse the configured kernel instance (no re-validate/configure).
 * rvcl_dispatch_release() detaches a cache-owned result safely.
 */
RVCL_API rvcl_status_t rvcl_dispatch_cache_clear(void);

/* Counters for verbose/profiling output (PLAN §8.4). */
RVCL_API int rvcl_dispatch_cache_hits(void);
RVCL_API int rvcl_dispatch_cache_misses(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_DISPATCH_H */
