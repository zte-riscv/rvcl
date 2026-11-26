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
 * whole request through the current scheduler. Layer norm allocates no
 * workspace and packs nothing: the kernel reads the [rows][norm_size]
 * source and writes the fp32 [rows][norm_size] destination in place, so the
 * single run covers every row and the scheduler slices the m axis inside
 * it. The optional scale and shift vectors ride the STAT_SCALE / STAT_SHIFT
 * slots. "dispatch once, run many" users call dispatch.h directly. */

#include "rvcl/layer_norm.h"

#include "rvcl/capability.h"
#include "rvcl/scheduler.h"

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

rvcl_status_t rvcl_layer_norm(const rvcl_layer_norm_desc_t *desc,
        const void *src, const void *scale, const void *shift, void *dst) {
    if (!desc || !src || !dst) return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->rows <= 0 || desc->norm_size <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (!(desc->eps >= 0.0f)) return RVCL_ERROR_INVALID_ARGUMENT;
    /* Layer norm has no bias and no second operand: a request that names
     * one is a caller error, not a silently ignored field. */
    if (desc->dtype_bias != RVCL_DTYPE_AUTO) return RVCL_ERROR_INVALID_ARGUMENT;
    /* A live parameter flag needs its vector: substituting the constant
     * here would silently compute a different normalization. */
    if ((desc->use_scale && !scale) || (desc->use_shift && !shift))
        return RVCL_ERROR_INVALID_ARGUMENT;

    /* 1. Dispatch (stages 1-4 + validate + configure). */
    rvcl_dispatch_result_t *r = NULL;
    const rvcl_status_t st = rvcl_dispatch_layer_norm(desc, &r);
    if (st != RVCL_SUCCESS) return st;

    /* 2. Import host memory zero-copy: the source in its own dtype and the
     * fp32 destination, both plain [rows][norm_size]. */
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {desc->rows, desc->norm_size},
            .strides = {desc->norm_size, 1},
            .dtype = desc->dtype_a};
    rvcl_tensor_t ta, tc;
    rvcl_tensor_init(&ta, &td, (void *)src);

    td.dtype = desc->dtype_c;
    rvcl_tensor_init(&tc, &td, dst);

    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);

    /* The optional parameters are length-norm_size fp32 vectors. A flag the
     * request leaves off imports no tensor, which is what the kernel's
     * contract expects for a substituted constant. */
    rvcl_tensor_desc_t pd = {.ndims = 1,
            .dims = {desc->norm_size},
            .strides = {1},
            .dtype = RVCL_DTYPE_FP32};
    rvcl_tensor_t tscale, tshift;
    if (desc->use_scale && scale) {
        rvcl_tensor_init(&tscale, &pd, (void *)scale);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_STAT_SCALE, &tscale);
    }
    if (desc->use_shift && shift) {
        rvcl_tensor_init(&tshift, &pd, (void *)shift);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_STAT_SHIFT, &tshift);
    }

    /* 3. No workspace: the kernel needs no staging buffer. */
    run_task_ctx_t ctx = {.kernel = rvcl_dispatch_kernel(r), .pack = &pack};
    const rvcl_work_range_t full = {.m_begin = 0,
            .m_end = desc->rows,
            .n_begin = 0,
            .n_end = desc->norm_size,
            .k_begin = 0,
            .k_end = 1};

    /* 4. Run the whole request through the current scheduler. */
    rvcl_parallel_for(&full, run_task, &ctx);
    rvcl_dispatch_release(r);
    return RVCL_SUCCESS;
}
