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

/* One-shot path: dispatch -> import tensors (zero-copy) -> one run over the
 * whole request through the current scheduler. Pooling allocates no
 * workspace and packs nothing: the kernel reads the NCHW source and writes
 * the fp32 NCHW destination in place, so the single run covers every row and
 * the scheduler slices the m axis inside it. "dispatch once, run many" users
 * call dispatch.h directly. */

#include "rvcl/pooling.h"

#include "rvcl/capability.h"
#include "rvcl/scheduler.h"

#include "../common/pooling_geometry.h"

typedef struct {
    rvcl_kernel_t *kernel;
    const rvcl_tensor_pack_t *pack;
} run_task_ctx_t;

static void run_task(
        rvcl_ischeduler_t *s, const rvcl_work_range_t *slice, void *user_ctx) {
    (void)s;
    const run_task_ctx_t *ctx = (const run_task_ctx_t *)user_ctx;
    rvcl_kernel_run(ctx->kernel, ctx->pack, slice);
}

rvcl_status_t rvcl_pooling(
        const rvcl_pooling_desc_t *desc, const void *src, void *dst) {
    if (!desc || !src || !dst) return RVCL_ERROR_INVALID_ARGUMENT;
    /* Pooling has no bias and no second operand: a request that names one
     * is a caller error, not a silently ignored field. */
    if (desc->dtype_bias != RVCL_DTYPE_AUTO) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!rvcl_pooling_geometry_valid(desc)) return RVCL_ERROR_INVALID_ARGUMENT;

    /* 1. Dispatch (stages 1-4 + validate + configure). */
    rvcl_dispatch_result_t *r = NULL;
    const rvcl_status_t st = rvcl_dispatch_pooling(desc, &r);
    if (st != RVCL_SUCCESS) return st;

    /* 2. Import host memory zero-copy: NCHW src [mb][ic][ih][iw] and fp32
     * dst [mb][ic][oh][ow]. */
    const int64_t oh = rvcl_pooling_out_dim(
            desc->ih, desc->kh, desc->sh, desc->pad_t, desc->pad_b);
    const int64_t ow = rvcl_pooling_out_dim(
            desc->iw, desc->kw, desc->sw, desc->pad_l, desc->pad_r);
    rvcl_tensor_desc_t td = {.ndims = 4,
            .dims = {desc->mb, desc->ic, desc->ih, desc->iw},
            .strides = {desc->ic * desc->ih * desc->iw, desc->ih * desc->iw,
                    desc->iw, 1},
            .dtype = desc->dtype_a};
    rvcl_tensor_t ta, tc;
    rvcl_tensor_init(&ta, &td, (void *)src);

    td.dims[2] = oh;
    td.dims[3] = ow;
    td.strides[0] = desc->ic * oh * ow;
    td.strides[1] = oh * ow;
    td.strides[2] = ow;
    td.strides[3] = 1;
    td.dtype = desc->dtype_c;
    rvcl_tensor_init(&tc, &td, dst);

    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);

    /* 3. No workspace: the kernel needs no staging buffer. */
    run_task_ctx_t ctx = {.kernel = rvcl_dispatch_kernel(r), .pack = &pack};
    const rvcl_work_range_t full = {.m_begin = 0,
            .m_end = desc->mb * oh,
            .n_begin = 0,
            .n_end = ow,
            .k_begin = 0,
            .k_end = 1};

    /* 4. Run the whole request through the current scheduler. */
    rvcl_parallel_for(&full, run_task, &ctx);
    rvcl_dispatch_release(r);
    return RVCL_SUCCESS;
}
