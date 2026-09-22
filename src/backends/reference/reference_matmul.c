/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * reference_matmul.c — golden FP32 MatMul (reference backend)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Phase 0 reference backend (PLAN §6.2 step 4): correctness-first triple
 * loop, zero workspace, no packing. Also the differential-testing golden
 * implementation that v0.1 RVV kernels are compared against.
 ******************************************************************************/

#include "rvcl/kernel.h"
#include "rvcl/registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    int64_t m, n, k;
} ref_matmul_t;

/* ------------------------------------------------------------- lifecycle */

static rvcl_status_t ref_validate(const rvcl_op_desc_t *desc,
        const rvcl_capability_t *cap) {
    (void)cap; /* reference kernel runs anywhere, even on x86 CI hosts */
    const rvcl_matmul_desc_t *d = &desc->matmul;
    if (d->m <= 0 || d->n <= 0 || d->k <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    /* Reference covers exactly the P0 fp32 tuple (PLAN §4.5.2). */
    if (d->dtype_a != RVCL_DTYPE_FP32 || d->dtype_b != RVCL_DTYPE_FP32
            || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    return RVCL_SUCCESS;
}

static rvcl_status_t ref_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req,
        int *num_ws) {
    ref_matmul_t *rk = (ref_matmul_t *)k;
    rk->m = desc->matmul.m;
    rk->n = desc->matmul.n;
    rk->k = desc->matmul.k;
    /* Golden path needs no temporary memory. */
    if (ws_req) *ws_req = (rvcl_workspace_req_t){0};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t ref_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *r) {
    (void)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *b = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    if (!a || !b || !c) return RVCL_ERROR_INVALID_ARGUMENT;

    const int64_t k_dim = a->desc.dims[1], n = b->desc.dims[1];
    const float *pa = (const float *)a->data;
    const float *pb = (const float *)b->data;
    const float *pbias = bias ? (const float *)bias->data : NULL;
    float *pc = (float *)c->data;

    /* Honor the requested sub-range; ranges partition M/N (PLAN §4.3). */
    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j = r->n_begin; j < r->n_end; j++) {
            float acc = pbias ? pbias[j] : 0.0f;
            for (int64_t p = 0; p < k_dim; p++)
                acc += pa[i * k_dim + p] * pb[p * n + j];
            pc[i * n + j] = acc;
        }
    }
    return RVCL_SUCCESS;
}

static void ref_destroy(rvcl_kernel_t *k) {
    /* nothing beyond the base free */
    (void)k;
}

static const rvcl_kernel_vtable_t g_ref_vtable = {
    .validate = ref_validate,
    .configure = ref_configure,
    .pack_a = NULL, /* unpacked kernel */
    .pack_b = NULL,
    .run = ref_run,
    .destroy = ref_destroy,
};

static rvcl_status_t ref_create(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc) {
    rvcl_status_t st = rvcl_kernel_alloc(out, desc, sizeof(ref_matmul_t));
    if (st != RVCL_SUCCESS) return st;
    (*out)->vt = &g_ref_vtable;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------- registration */

RVCL_API rvcl_status_t rvcl_reference_register(void) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "rvcl_matmul_f32_reference");
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = RVCL_DTYPE_FP32;
    d.dtype_b = RVCL_DTYPE_FP32;
    d.dtype_acc = RVCL_DTYPE_FP32;
    d.dtype_c = RVCL_DTYPE_FP32;
    d.quant_support = RVCL_QUANT_NONE;
    d.required_features[0] = RVCL_FEATURE_NONE; /* runs anywhere */
    d.num_required_features = 0;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;

    return rvcl_register_kernel(&d, ref_create);
}
