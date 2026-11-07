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
 * rvv_gemv_f32.cpp — FP32 GEMV kernel (RVV backend, C-portable path)
 *
 * y[m] = bias? + A[m,k] x[k], row-major A, length-m output vector.
 * Single-path demonstration operator: a C-portable micro-kernel shared by
 * the generic and portable registry entries, so the operator exercises the
 * whole extension-slot pipeline without a dedicated pick-table row (D-6).
 * The generic entry (rvcl_gemv_f32_rvv) carries the RVV feature and serves
 * vector builds; the portable entry (rvcl_gemv_f32_rvv_c) carries no feature
 * requirement and serves non-vector builds / any host. Both bind the same
 * C loop, which is the correctness oracle for the differential test.
 *
 * There is no B packing: B is a length-k vector, so run() consumes the pack
 * directly and configure() reports no workspace.
 ******************************************************************************/

#include "rvcl/kernel.h"
#include "rvcl/registry.h"

#include "../../common/dtype_traits.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------ C-portable kernel */

static void rvv_gemv_f32_c(const void *a_in, const void *x_in,
        const void *bias_in, void *y_in, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const float *a = (const float *)a_in;
    const float *x = (const float *)x_in;
    const float *bias = (const float *)bias_in;
    float *y = (float *)y_in;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        float acc = bias ? bias[i] : 0.0f;
        for (int64_t p = 0; p < k_dim; p++)
            acc += a[i * k_dim + p] * x[p];
        y[i] = acc;
    }
}

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    int64_t m, k;
} rvv_gemv_t;

/* ------------------------------------------------------------- lifecycle */

static rvcl_status_t rvv_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap;
    const rvcl_gemv_desc_t *d = &desc->gemv;
    if (d->m <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    /* Quantization is not part of the GEMV path: reject a quantized
     * request so it is never computed as if unquantized. */
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    return RVCL_SUCCESS;
}

static rvcl_status_t rvv_configure(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
        rvcl_workspace_req_t *ws_req, int *num_ws) {
    rvv_gemv_t *rk = (rvv_gemv_t *)k;
    rk->m = desc->gemv.m;
    rk->k = desc->gemv.k;
    /* No packing, no workspace. */
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t rvv_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *r) {
    rvv_gemv_t *rk = (rvv_gemv_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *x = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *y = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    if (!a || !x || !y) return RVCL_ERROR_INVALID_ARGUMENT;

    rvv_gemv_f32_c(
            a->data, x->data, bias ? bias->data : NULL, y->data, rk->k, r);
    return RVCL_SUCCESS;
}

static void rvv_destroy(rvcl_kernel_t *k) {
    (void)k;
}

static const rvcl_kernel_vtable_t g_rvv_vtable = {
        .validate = rvv_validate,
        .configure = rvv_configure,
        .pack_a = NULL, /* no packing: B is a vector */
        .pack_b = NULL,
        .run = rvv_run,
        .destroy = rvv_destroy,
};

static rvcl_status_t rvv_create(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    rvcl_status_t st = rvcl_kernel_alloc(out, desc, sizeof(rvv_gemv_t));
    if (st != RVCL_SUCCESS) return st;
    (*out)->vt = &g_rvv_vtable;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------- registration */

/* One registry entry per variant (RFC-0002 FR1.2), register_one-style:
 * the generic entry requires the RVV feature on vector builds; the
 * portable entry carries no feature requirement so it serves any host. */
static rvcl_status_t gemv_register_one(const char *name, rvcl_vlen_tier_t tier,
        rvcl_feature_id_t f0, rvcl_feature_id_t f1) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_GEMV;
    d.dtype_a = RVCL_DTYPE_FP32;
    d.dtype_b = RVCL_DTYPE_FP32;
    d.dtype_acc = RVCL_DTYPE_FP32;
    d.dtype_c = RVCL_DTYPE_FP32;
    d.quant_support = RVCL_QUANT_NONE;
    d.assert_level = rvcl_dtype_traits(RVCL_DTYPE_FP32).assert_level_default;
    d.required_features[0] = f0;
    d.required_features[1] = f1;
    int nfeat = 0;
    if (f0 != RVCL_FEATURE_NONE) {
        nfeat = 1;
        if (f1 != RVCL_FEATURE_NONE) nfeat = 2;
    }
    d.num_required_features = nfeat;
    d.alignment_a = d.alignment_b = d.alignment_c = 4;
    d.vlen_tier = tier;
    d.cost_fn = rvcl_kernel_cost_zero;

    return rvcl_register_kernel(&d, rvv_create);
}

#if defined(__riscv_vector)

RVCL_API rvcl_status_t rvcl_rvv_register_gemv(void) {
    return gemv_register_one("rvcl_gemv_f32_rvv", RVCL_VLEN_TIER_ANY,
            RVCL_FEATURE_RVV, RVCL_FEATURE_NONE);
}

RVCL_API rvcl_status_t rvcl_rvv_register_gemv_portable(void) {
    return RVCL_SUCCESS; /* vector build: portable entry stays out */
}

#else /* !__riscv_vector */

RVCL_API rvcl_status_t rvcl_rvv_register_gemv(void) {
    return RVCL_SUCCESS; /* non-vector build: generic entry stays out */
}

/* Portable scalar entry: no feature requirement (serve any host), the f32
 * tuple's entry on non-vector builds. */
RVCL_API rvcl_status_t rvcl_rvv_register_gemv_portable(void) {
    return gemv_register_one("rvcl_gemv_f32_rvv_c", RVCL_VLEN_TIER_ANY,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE);
}

#endif /* __riscv_vector */
