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

/* One-shot path: dispatch -> import tensors (zero-copy) -> allocate the
 * reported workspace -> pack B once -> one run per batch through the
 * current scheduler. The conv kernel stages one batch's row band at a
 * time, so a batch is the unit the scheduler slices inside. The band
 * workspace is single and shared by every slice; what keeps concurrent
 * slices safe is that the band rows a slice writes are exactly the m rows
 * it owns, so the per-slice row intervals stay disjoint and the scheduler
 * never hands the same row to two slices. "dispatch once, run many" users
 * call dispatch.h directly. */

#include "rvcl/conv2d.h"

#include "rvcl/capability.h"
#include "rvcl/scheduler.h"

#include "../common/conv2d_geometry.h"

#include <stdlib.h>
#include <string.h>

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

rvcl_status_t rvcl_conv2d(const rvcl_conv2d_desc_t *desc, const void *src,
        const void *w, const void *bias, void *dst) {
    if (!desc || !src || !w || !dst) return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->groups != 1) return RVCL_ERROR_INVALID_ARGUMENT;
    /* The bias pointer and the descriptor must agree on whether there is a
     * bias, the same rule the matmul entry enforces for its scales. */
    if ((desc->dtype_bias == RVCL_DTYPE_AUTO) != (bias == NULL))
        return RVCL_ERROR_INVALID_ARGUMENT;

    /* 1. Dispatch (stages 1-4 + validate + configure). */
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_conv2d(desc, &r);
    if (st != RVCL_SUCCESS) return st;

    const int64_t oh = rvcl_conv2d_out_dim(desc->ih, desc->kh, desc->sh,
            desc->dil_h, desc->pad_t, desc->pad_b);
    const int64_t ow = rvcl_conv2d_out_dim(desc->iw, desc->kw, desc->sw,
            desc->dil_w, desc->pad_l, desc->pad_r);
    if (oh <= 0 || ow <= 0) {
        rvcl_dispatch_release(r);
        return RVCL_ERROR_INVALID_ARGUMENT;
    }
    const int64_t m_band = oh * ow;
    const int64_t k = desc->ic * desc->kh * desc->kw;

    /* 2. Import host memory zero-copy (NCHW layouts). */
    rvcl_tensor_desc_t td = {.ndims = 4,
            .dims = {desc->mb, desc->ic, desc->ih, desc->iw},
            .strides = {desc->ic * desc->ih * desc->iw, desc->ih * desc->iw,
                    desc->iw, 1},
            .dtype = desc->dtype_a};
    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_init(&ta, &td, (void *)src);

    /* The conv GEMM's B operand is w^T[k, oc]; the NCHW weights already
     * hold exactly those rows, as [oc][ic][kh][kw]. */
    td.ndims = 2;
    td.dims[0] = desc->oc;
    td.dims[1] = k;
    td.strides[0] = k;
    td.strides[1] = 1;
    td.dtype = desc->dtype_b;
    rvcl_tensor_init(&tb, &td, (void *)w);

    td.ndims = 4;
    td.dims[0] = desc->mb;
    td.dims[1] = desc->oc;
    td.dims[2] = oh;
    td.dims[3] = ow;
    td.strides[0] = desc->oc * m_band;
    td.strides[1] = m_band;
    td.strides[2] = ow;
    td.strides[3] = 1;
    td.dtype = desc->dtype_c;
    rvcl_tensor_init(&tc, &td, dst);

    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_B, &tb);
    rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {desc->oc},
                .strides = {1},
                .dtype = desc->dtype_bias};
        rvcl_tensor_init(&tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_BIAS, &tbias);
    }

    /* 3. Workspace: allocate what configure() reported into AUX slots
     * (kernels themselves never malloc). */
    run_task_ctx_t ctx = {.kernel = rvcl_dispatch_kernel(r), .pack = &pack};
    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
    void *ws_buf[RVCL_TENSOR_AUX_COUNT] = {0};
    rvcl_tensor_t ws_t[RVCL_TENSOR_AUX_COUNT];
    for (int i = 0; i < num_ws && i < RVCL_TENSOR_AUX_COUNT; i++) {
        const size_t align = ws[i].alignment > 1 ? (size_t)ws[i].alignment : 8;
        if (posix_memalign(&ws_buf[i], align, (size_t)(ws[i].size + align))
                != 0) {
            st = RVCL_ERROR_OUT_OF_MEMORY;
            goto out;
        }
        rvcl_tensor_desc_t wd = {.ndims = 1,
                .dims = {(int64_t)ws[i].size},
                .strides = {1},
                .dtype = RVCL_DTYPE_UINT8};
        rvcl_tensor_init(&ws_t[i], &wd, ws_buf[i]);
        rvcl_tensor_pack_add(
                &pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + i), &ws_t[i]);
    }

    /* 4. Pack B once: the weights are packed into AUX_0 ahead of the batch
     * runs, and every run() then reads the panels read-only. */
    if (num_ws >= 1 && ctx.kernel->vt->pack_b) {
        st = rvcl_kernel_pack_b(ctx.kernel, &tb, &ws_t[0]);
        if (st != RVCL_SUCCESS) goto out;
        pack.src_b_packed = 1;
    }

    /* 5. Run one batch per work range through the current scheduler. */
    for (int64_t mb = 0; mb < desc->mb; mb++) {
        const rvcl_work_range_t band = {
                .m_begin = mb * m_band,
                .m_end = (mb + 1) * m_band,
                .n_begin = 0,
                .n_end = desc->oc,
                .k_begin = 0,
                .k_end = k,
        };
        rvcl_parallel_for(&band, run_task, &ctx);
    }

    st = RVCL_SUCCESS;

out:
    for (int i = 0; i < RVCL_TENSOR_AUX_COUNT; i++)
        free(ws_buf[i]);
    rvcl_dispatch_release(r);
    return st;
}
