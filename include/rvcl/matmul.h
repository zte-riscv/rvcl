/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * matmul.h — semantic MatMul API (Layer 1: Operator API)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * One-shot convenience path (PLAN §4.7): dispatch + pack + run + epilogue.
 * Applications that need "dispatch once, run many" or custom scheduling
 * should use dispatch.h + kernel.h directly.
 ******************************************************************************/

#ifndef RVCL_MATMUL_H
#define RVCL_MATMUL_H

#include "rvcl/dispatch.h"
#include "rvcl/memory.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * C[m,n] = A[m,k] × B[k,n] (+ bias[n] when bias != NULL), row-major
 * plain tensors. dtypes follow the (A,B,acc,C) tuple in `desc`; pass
 * RVCL_DTYPE_AUTO for acc to apply the habit rules (PLAN §4.5.1).
 *
 * The current scheduler drives work-range parallelism internally; kernels
 * never create threads. Returns RVCL_UNSUPPORTED when no kernel serves
 * the requested tuple on this host.
 */
RVCL_API rvcl_status_t rvcl_matmul(const rvcl_matmul_desc_t *desc, const void *a,
        const void *b, const void *bias, void *c);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_MATMUL_H */
