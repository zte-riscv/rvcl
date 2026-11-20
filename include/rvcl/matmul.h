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

/* One-shot convenience path: dispatch + pack + run + epilogue. Applications
 * that need "dispatch once, run many" or custom scheduling should use
 * dispatch.h + kernel.h directly. */

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
 * RVCL_DTYPE_AUTO for acc to apply the habit rules.
 *
 * The current scheduler drives work-range parallelism internally; kernels
 * never create threads. Returns RVCL_UNSUPPORTED when no kernel serves
 * the requested tuple on this host.
 */
RVCL_API rvcl_status_t rvcl_matmul(const rvcl_matmul_desc_t *desc,
        const void *a, const void *b, const void *bias, void *c);

/*
 * Scaled one-shot: C[m,n] = A[m,k] × B[k,n] (+ bias) with per-block scale
 * tensors for block-scale input dtypes (MX/NV formats). sa/sb are the
 * scale tensors, row-major sa[m][K/blk_a] and sb[K/blk_b][n] (the B scale
 * is indexed by K-block then output column n — the oneDNN MX convention,
 * so the micro-kernels read it as sb[(p/blk)*n + j]); their dtype is the
 * block-scale dtype's scale dtype (E8M0 / E4M3). A block-scale input dtype
 * requires a non-NULL scale, a plain dtype requires NULL; violation
 * returns RVCL_ERROR_INVALID_ARGUMENT.
 */
RVCL_API rvcl_status_t rvcl_matmul_scaled(const rvcl_matmul_desc_t *desc,
        const void *a, const void *sa, const void *b, const void *sb,
        const void *bias, void *c);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_MATMUL_H */
