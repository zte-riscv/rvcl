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
 * reference_gemv.cpp — golden FP32 GEMV (reference backend)
 *
 * Reference backend: correctness-first reduction loop, zero workspace,
 * no packing. Also the differential-testing golden implementation that
 * the GEMV kernels are compared against.
 * y[m] = bias? + A[m,k] x[k], row-major A, length-m output vector.
 ******************************************************************************/

#include "rvcl/kernel.h"
#include "rvcl/registry.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    int64_t m, k;
} ref_gemv_t;

/* ------------------------------------------------------------- lifecycle */

static rvcl_status_t ref_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap; /* reference kernel runs anywhere, even on x86 CI hosts */
    const rvcl_gemv_desc_t *d = &desc->gemv;
    if (d->m <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    /* Reference covers exactly the P0 f32 tuple. */
    if (d->dtype_a != RVCL_DTYPE_FP32 || d->dtype_b != RVCL_DTYPE_FP32
            || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* No quantization support: rejecting here (as the other backends do)
     * keeps a quantized request from being computed as if unquantized if
     * this kernel is instantiated outside the dispatch filter. */
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    return RVCL_SUCCESS;
}

static rvcl_status_t ref_configure(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
        rvcl_workspace_req_t *ws_req, int *num_ws) {
    ref_gemv_t *rk = (ref_gemv_t *)k;
    rk->m = desc->gemv.m;
    rk->k = desc->gemv.k;
    /* Golden path needs no temporary memory. */
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t ref_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *r) {
    (void)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *x = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *y = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    if (!a || !x || !y) return RVCL_ERROR_INVALID_ARGUMENT;

    const int64_t k_dim = a->desc.dims[1];
    const float *pa = (const float *)a->data;
    const float *px = (const float *)x->data;
    const float *pbias = bias ? (const float *)bias->data : NULL;
    float *py = (float *)y->data;

    /* Honor the requested sub-range; ranges partition M rows. */
    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        float acc = pbias ? pbias[i] : 0.0f;
        for (int64_t p = 0; p < k_dim; p++)
            acc += pa[i * k_dim + p] * px[p];
        py[i] = acc;
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

static rvcl_status_t ref_create(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    rvcl_status_t st = rvcl_kernel_alloc(out, desc, sizeof(ref_gemv_t));
    if (st != RVCL_SUCCESS) return st;
    (*out)->vt = &g_ref_vtable;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------- registration */

/*
 * The reference GEMV kernel is the f32 tuple's unbounded fallback: no
 * feature requirement, no shape limit, registered after the dedicated,
 * generic and portable variants so it only wins when nothing else survives
 * filtering (RFC-0002 FR1.5).
 */
RVCL_API rvcl_status_t rvcl_gemv_reference_register(void) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "rvcl_gemv_f32_reference");
    d.operation = RVCL_OP_GEMV;
    d.dtype_a = RVCL_DTYPE_FP32;
    d.dtype_b = RVCL_DTYPE_FP32;
    d.dtype_acc = RVCL_DTYPE_FP32;
    d.dtype_c = RVCL_DTYPE_FP32;
    d.quant_support = RVCL_QUANT_NONE;
    d.num_required_features = 0;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;
    d.vlen_tier = RVCL_VLEN_TIER_ANY; /* memset zero, kept explicit */
    d.cost_fn = rvcl_kernel_cost_zero;

    return rvcl_register_kernel(&d, ref_create);
}
