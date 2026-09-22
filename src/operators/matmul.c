/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * matmul.c — semantic MatMul API (Layer 1)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * One-shot path: dispatch → import tensors (zero-copy) → allocate the
 * reported workspace on the stack/heap of this call → run via the current
 * scheduler. "dispatch once, run many" users call dispatch.h directly.
 ******************************************************************************/

#include "rvcl/matmul.h"

#include "rvcl/capability.h"
#include "rvcl/scheduler.h"

#include <stdlib.h>
#include <string.h>

/* run() task payload for scheduler-driven execution. */
typedef struct {
    rvcl_kernel_t *kernel;
    rvcl_tensor_pack_t *pack;
} run_task_ctx_t;

static void run_task(rvcl_ischeduler_t *s, const rvcl_work_range_t *slice,
        void *user_ctx) {
    (void)s;
    run_task_ctx_t *ctx = (run_task_ctx_t *)user_ctx;
    rvcl_kernel_run(ctx->kernel, ctx->pack, slice);
}

rvcl_status_t rvcl_matmul(const rvcl_matmul_desc_t *desc, const void *a,
        const void *b, const void *bias, void *c) {
    if (!desc || !a || !b || !c) return RVCL_ERROR_INVALID_ARGUMENT;

    /* 1. Dispatch (stages 1-4 + validate + configure). */
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_matmul(desc, &r);
    if (st != RVCL_SUCCESS) return st;

    /* 2. Import host memory zero-copy (PLAN §4.10). */
    rvcl_tensor_desc_t td2 = {.ndims = 2,
        .dims = {desc->m, desc->k},
        .strides = {desc->k, 1},
        .dtype = desc->dtype_a};
    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_init(&ta, &td2, (void *)a);

    td2.dims[0] = desc->k;
    td2.dims[1] = desc->n;
    td2.strides[0] = desc->n;
    td2.dtype = desc->dtype_b;
    rvcl_tensor_init(&tb, &td2, (void *)b);

    td2.dims[0] = desc->m;
    td2.dims[1] = desc->n;
    td2.strides[0] = desc->n;
    td2.dtype = desc->dtype_c;
    rvcl_tensor_init(&tc, &td2, c);

    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_B, &tb);
    rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
            .dims = {desc->n},
            .strides = {1},
            .dtype = desc->dtype_bias};
        rvcl_tensor_init(&tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_BIAS, &tbias);
    }

    /* 3. Workspace: caller of the semantic API is the caller here, so we
     * allocate what configure() reported and feed AUX slots (kernels
     * themselves never malloc; PLAN §4.3). */
    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
    void *ws_buf[RVCL_MAX_WORKSPACES] = {0};
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    for (int i = 0; i < num_ws && i < RVCL_MAX_WORKSPACES; i++) {
        size_t align = ws[i].alignment > 1 ? (size_t)ws[i].alignment : 8;
        if (posix_memalign(&ws_buf[i], align,
                    (size_t)(ws[i].size + align)) != 0) {
            st = RVCL_ERROR_OUT_OF_MEMORY;
            goto out;
        }
        rvcl_tensor_desc_t wd = {.ndims = 1,
            .dims = {(int64_t)ws[i].size},
            .strides = {1},
            .dtype = RVCL_DTYPE_UINT8};
        rvcl_tensor_init(&ws_t[i], &wd, ws_buf[i]);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_AUX_0 + i, &ws_t[i]);
    }

    /* 4. Run through the current scheduler over the full range. */
    run_task_ctx_t ctx = {.kernel = rvcl_dispatch_kernel(r), .pack = &pack};
    rvcl_work_range_t full = rvcl_work_range_full(desc->m, desc->n, desc->k);
    rvcl_parallel_for(&full, run_task, &ctx);

    st = RVCL_SUCCESS;

out:
    for (int i = 0; i < RVCL_MAX_WORKSPACES; i++)
        free(ws_buf[i]);
    rvcl_dispatch_release(r);
    return st;
}
