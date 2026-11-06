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
 * dispatch_internal.h — generic dispatch entry (not installed)
 *
 * Declarations shared between the library and its test executables only.
 * This header is deliberately outside include/rvcl/: it is not part of the
 * public API, is never installed, and carries no stability promise.
 *
 * The six-stage dispatch pipeline is operator-generic (op/datatype/
 * capability/vlen-tier/shape/validate plus cost selection, snapshot retry,
 * generation validation and cache refcounting). Each operator contributes
 * one op-table entry: a request validator that resolves the filter inputs
 * the generic pipeline consumes, plus a cache-key generator (hash and
 * equality) over the operation descriptor's full width. A new operator is
 * served by the whole pipeline and cache once its entry is registered.
 * rvcl_dispatch_op() is the internal generic entry; the public per-operator
 * wrappers (rvcl_dispatch_matmul) look their entry up and call it.
 ******************************************************************************/

#ifndef RVCL_DISPATCH_INTERNAL_H
#define RVCL_DISPATCH_INTERNAL_H

#include "rvcl/dispatch.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Resolved per-operator filter inputs the generic pipeline consumes in
 * stages 2 (datatype) and 5 (shape): the (A,B,acc,C) tuple with the AUTO
 * accumulator already resolved by the operator's habit rule, the shape,
 * and whether the request carries quantization. Produced once per dispatch
 * by the op-table dims_validate callback, so the pipeline body stays free
 * of operator-specific field access.
 */
typedef struct {
    int64_t m, n, k;
    rvcl_dtype_t dtype_a, dtype_b, dtype_acc, dtype_c;
    int quant_enabled;
} rvcl_dispatch_filter_t;

/*
 * One operator's dispatch registration. The op discriminates kernels at
 * stage 1 and disambiguates cache entries; dims_validate both rejects a
 * malformed request (RVCL_ERROR_INVALID_ARGUMENT for illegal shapes,
 * RVCL_UNSUPPORTED_DTYPE_COMBO for an illegal datatype tuple) and fills
 * the resolved filter inputs; key_hash/key_equal drive the dispatch cache
 * over the full-width operation descriptor. hash must agree with equal:
 * equal keys hash equal. Registered by rvcl_dispatch_register_op().
 */
typedef struct {
    rvcl_operation_t op;
    rvcl_status_t (*dims_validate)(
            const rvcl_op_desc_t *od, rvcl_dispatch_filter_t *out);
    uint32_t (*key_hash)(const rvcl_op_desc_t *key);
    int (*key_equal)(const rvcl_op_desc_t *a, const rvcl_op_desc_t *b);
} rvcl_dispatch_op_entry_t;

/*
 * Generic dispatch entry (internal, not installed as API). Runs the full
 * six-stage pipeline over the registry snapshot for the operator's table
 * entry and returns a configured rvcl_dispatch_result_t on success. An
 * operator with no registered table entry is rejected with RVCL_UNSUPPORTED.
 * The public rvcl_dispatch_matmul() wraps this for the matmul entry.
 */
RVCL_API rvcl_status_t rvcl_dispatch_op(rvcl_operation_t op,
        const rvcl_op_desc_t *od, rvcl_dispatch_result_t **result);

/*
 * Registers an operator's dispatch table entry (internal, no stability
 * promise; test executables use it to exercise non-matmul operators).
 * Duplicate op is rejected with RVCL_ERROR_ALREADY_REGISTERED; a full
 * table returns RVCL_ERROR_OUT_OF_MEMORY. Registration is not synchronized
 * with concurrent dispatch — callers register during single-threaded setup.
 */
RVCL_API rvcl_status_t rvcl_dispatch_register_op(
        const rvcl_dispatch_op_entry_t *entry);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_DISPATCH_INTERNAL_H */
