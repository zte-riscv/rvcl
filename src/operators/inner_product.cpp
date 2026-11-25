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
 * reported workspace -> pack B once -> one run over the whole request
 * through the current scheduler. The GEMM writes [mb][oc] row-major
 * straight into the destination, so a single run covers every row and the
 * scheduler slices the m axis inside it. "dispatch once, run many" users
 * call dispatch.h directly. */

#include "rvcl/inner_product.h"

#include "rvcl/capability.h"
#include "rvcl/scheduler.h"

#include <stdlib.h>

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

rvcl_status_t rvcl_inner_product(const rvcl_inner_product_desc_t *desc,
        const void *src, const void *w, const void *bias, void *dst) {
    if (!desc || !src || !w || !dst) return RVCL_ERROR_INVALID_ARGUMENT;
    /* The bias pointer and the descriptor must agree on whether there is a
     * bias, the same rule the matmul entry enforces for its scales. */
    if ((desc->dtype_bias == RVCL_DTYPE_AUTO) != (bias == NULL))
        return RVCL_ERROR_INVALID_ARGUMENT;

    /* 1. Dispatch (stages 1-4 + validate + configure). */
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_inner_product(desc, &r);
    if (st != RVCL_SUCCESS) return st;

    const int64_t m = desc->mb, n = desc->oc, k = desc->k;

    /* 2. Import host memory zero-copy (row-major src [mb][k], w [oc][k],
     * dst [mb][oc]). */
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {m, k},
            .strides = {k, 1},
            .dtype = desc->dtype_a};
    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_init(&ta, &td, (void *)src);

    td.dims[0] = n;
    td.dims[1] = k;
    td.strides[0] = k;
    td.strides[1] = 1;
    td.dtype = desc->dtype_b;
    rvcl_tensor_init(&tb, &td, (void *)w);

    td.dims[0] = m;
    td.dims[1] = n;
    td.strides[0] = n;
    td.strides[1] = 1;
    td.dtype = desc->dtype_c;
    rvcl_tensor_init(&tc, &td, dst);

    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_B, &tb);
    rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {n},
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
    const rvcl_work_range_t full = {.m_begin = 0,
            .m_end = m,
            .n_begin = 0,
            .n_end = n,
            .k_begin = 0,
            .k_end = k};
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

    /* 4. Pack B once: the weights are transposed into AUX_0 ahead of the
     * run, and the run then reads the panels read-only. */
    if (num_ws >= 1 && ctx.kernel->vt->pack_b) {
        st = rvcl_kernel_pack_b(ctx.kernel, &tb, &ws_t[0]);
        if (st != RVCL_SUCCESS) goto out;
        pack.src_b_packed = 1;
    }

    /* 5. Run the whole request through the current scheduler. */
    rvcl_parallel_for(&full, run_task, &ctx);
    st = RVCL_SUCCESS;

out:
    for (int i = 0; i < RVCL_TENSOR_AUX_COUNT; i++)
        free(ws_buf[i]);
    rvcl_dispatch_release(r);
    return st;
}
