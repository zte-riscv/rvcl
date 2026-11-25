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

#ifndef RVCL_INNER_PRODUCT_H
#define RVCL_INNER_PRODUCT_H

#include "rvcl/dispatch.h"
#include "rvcl/memory.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Inner product described by `desc`: dst, src and w are row-major, src
 * [mb][k], w [oc][k], dst [mb][oc]. bias is a length-oc fp32 vector added
 * per output column, or NULL for no bias (the descriptor's dtype_bias must
 * then be RVCL_DTYPE_AUTO). src and w use dtype_a / dtype_b; dst is fp32
 * and the accumulation is fp32, so the up-conversion of a narrow input is
 * exact and the result is bit-comparable with a scalar reference. Pass
 * RVCL_DTYPE_AUTO for acc to apply the habit rules. The reduction length k
 * is the caller's folded ic*kd*kh*kw; see rvcl_inner_product_desc_t.
 *
 * Returns RVCL_UNSUPPORTED when no kernel serves the requested tuple on
 * this host.
 */
RVCL_API rvcl_status_t rvcl_inner_product(const rvcl_inner_product_desc_t *desc,
        const void *src, const void *w, const void *bias, void *dst);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_INNER_PRODUCT_H */
