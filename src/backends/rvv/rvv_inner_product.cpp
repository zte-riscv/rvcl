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

/* Inner product served as a transposing pack plus the dtype's packed-B GEMM
 * micro-kernel:
 *   A[m, k] = src            m = mb, k = folded reduction
 *   B[k, n] = w^T            n = oc
 *   C[m, n] = the GEMM output, row-major, which is exactly dst [mb][oc].
 * The weights are [oc][k] row-major, i.e. B transposed, so pack_b transposes
 * them into the panel layout the GEMM consumes (the shared rvv_pack_b_panel
 * assumes B[k][n] with n innermost and cannot be reused as-is). The GEMM
 * writes [mb][oc] row-major with the bias seeded first, so the destination
 * layout is the output layout and there is no writeback transpose.
 *
 * Workspaces: AUX_0 packed-B panels (KERNEL lifetime, as the matmul path).
 * No per-run band: the GEMM writes straight into the destination, so run()
 * stages nothing and concurrent slices stay disjoint on the m axis.
 *
 * Work ranges: the m axis is mb, n is oc and k is the folded reduction.
 * n and k must be the full ranges (the packed panels and the reduction are
 * not split); the m range may be any slice of [0, mb).
 *
 * The pack loop is plain C compiled on every host; only the GEMM
 * micro-kernel itself is RVV-specialized, and it is selected through the
 * same rvv_pick_gemm_* entry the matmul backend uses, so an inner product
 * and a matmul of one dtype can never run different kernels. run() never
 * mallocs and never threads. */

#include "rvv_internal.hpp"

#include "../../common/dtype_traits.hpp"
#include "../../runtime/cost_model.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    rvcl_inner_product_desc_t d; /* validated request, configured once */
    rvv_gemm_fn gemm; /* dtype-specific micro-kernel entry */
    size_t elt; /* element size of A/B storage */
} rvv_inner_product_t;

/* ------------------------------------------------------------- lifecycle */

/* Every kernel instance serves exactly one (A,B,acc,C) tuple and the output
 * is fp32 (the dtype's accumulation width), so the tuple is part of the
 * instance's contract: a request for another tuple is rejected here rather
 * than computed through a mismatched gemm entry. */
template <rvcl_dtype_t DT>
static rvcl_status_t inner_product_validate_tuple(const rvcl_op_desc_t *desc) {
    const rvcl_inner_product_desc_t *d = &desc->inner_product;
    if (d->mb <= 0 || d->oc <= 0 || d->k <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (d->dtype_a != DT || d->dtype_b != DT || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* AUTO is the request-side "derive it" marker for the accumulator. */
    if (d->dtype_acc != RVCL_DTYPE_AUTO && d->dtype_acc != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Bias is read as f32 by the gemm seed; AUTO means "no bias". */
    if (d->dtype_bias != RVCL_DTYPE_AUTO && d->dtype_bias != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Quantization is not served by the inner_product path yet. */
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    return RVCL_SUCCESS;
}

template <rvcl_dtype_t DT>
static rvcl_status_t inner_product_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap; /* the gemm entry carries its own feature gating */
    return inner_product_validate_tuple<DT>(desc);
}

static rvcl_status_t inner_product_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    rvv_inner_product_t *rk = (rvv_inner_product_t *)k;
    const rvcl_inner_product_desc_t *d = &desc->inner_product;
    if (d->mb <= 0 || d->oc <= 0 || d->k <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    rk->d = *d;
    if (ws_req) {
        /* Slot 0: packed-B panels (the [oc][k] weights transposed). */
        ws_req[RVV_WS_PACKED_B] = rvcl_workspace_req_t {
                .size = (int64_t)rvv_packed_b_bytes(d->k, d->oc, rk->elt),
                .alignment = 64,
                .lifetime = RVCL_WS_LIFETIME_KERNEL,
        };
    }
    if (num_ws) *num_ws = 1;
    return RVCL_SUCCESS;
}

/* pack_b: the B operand of the GEMM is w^T (k rows, oc columns) while the
 * weights are [oc][k] row-major, so the panel packing is a transposing copy
 * into the shared panel layout. */
static rvcl_status_t inner_product_pack_b(
        rvcl_kernel_t *k, const rvcl_tensor_t *src, rvcl_tensor_t *dst) {
    rvv_inner_product_t *rk = (rvv_inner_product_t *)k;
    if (!rk || !src || !src->data || !dst || !dst->data || rk->d.k <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    const int64_t kd = rk->d.k, n = rk->d.oc;
    const uint8_t *w = (const uint8_t *)src->data;
    uint8_t *panels = (uint8_t *)dst->data;
    const size_t elt = rk->elt;

    for (int64_t col = 0; col < n; col++) {
        for (int64_t row = 0; row < kd; row++) {
            const int64_t di = rvv_packed_b_index(kd, col, row);
            memcpy(panels + (size_t)di * elt,
                    w + (size_t)(col * kd + row) * elt, elt);
        }
    }
    return RVCL_SUCCESS;
}

static rvcl_status_t inner_product_run(rvcl_kernel_t *k,
        const rvcl_tensor_pack_t *pack, const rvcl_work_range_t *range) {
    rvv_inner_product_t *rk = (rvv_inner_product_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *b = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    const rvcl_tensor_t *ws_b = rvcl_tensor_pack_get(
            pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + RVV_WS_PACKED_B));
    if (!a || !b || !c || !ws_b) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!range) return RVCL_ERROR_INVALID_ARGUMENT;

    const rvcl_inner_product_desc_t *d = &rk->d;
    /* Full n and k only: the packed panels and the reduction are not split.
     * The m range may be any slice of [0, mb). */
    if (range->m_begin < 0 || range->m_end > d->mb
            || range->m_begin > range->m_end || range->n_begin != 0
            || range->n_end != d->oc || range->k_begin != 0
            || range->k_end != d->k)
        return RVCL_ERROR_INVALID_ARGUMENT;

    if (!pack->src_b_packed) {
        const rvcl_status_t st
                = inner_product_pack_b(k, b, (rvcl_tensor_t *)ws_b);
        if (st != RVCL_SUCCESS) return st;
    }

    /* A is [mb][k] row-major, C is [mb][oc] row-major and the packed panels
     * are read-only across slices; the GEMM seeds the bias first, the same
     * addition order as the golden. */
    rk->gemm(a->data, ws_b->data, bias ? bias->data : NULL, c->data, d->mb,
            d->oc, d->k, range);
    return RVCL_SUCCESS;
}

static void inner_product_destroy(rvcl_kernel_t *k) {
    (void)k;
}

/* One vtable per dtype: validate() carries the instance's tuple. */
template <rvcl_dtype_t DT>
static const rvcl_kernel_vtable_t *inner_product_vtable(void) {
    static const rvcl_kernel_vtable_t vt = {
            .validate = inner_product_validate<DT>,
            .configure = inner_product_configure,
            .pack_a = NULL,
            .pack_b = inner_product_pack_b,
            .run = inner_product_run,
            .destroy = inner_product_destroy,
    };
    return &vt;
}

/* --------------------------------------------------- per-dtype factories */

static rvcl_status_t make_inner_product_kernel(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, const rvcl_kernel_vtable_t *vt,
        rvv_gemm_fn gemm, size_t elt) {
    if (!gemm) return RVCL_ERROR_INVALID_ARGUMENT;
    const rvcl_status_t st
            = rvcl_kernel_alloc(out, desc, sizeof(rvv_inner_product_t));
    if (st != RVCL_SUCCESS) return st;
    rvv_inner_product_t *rk = (rvv_inner_product_t *)*out;
    rk->base.vt = vt;
    rk->gemm = gemm;
    rk->elt = elt;
    return RVCL_SUCCESS;
}

static rvcl_status_t create_inner_product_f32(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return make_inner_product_kernel(out, desc,
            inner_product_vtable<RVCL_DTYPE_FP32>(), rvv_pick_gemm_f32(),
            sizeof(float));
}

static rvcl_status_t create_inner_product_f16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return make_inner_product_kernel(out, desc,
            inner_product_vtable<RVCL_DTYPE_FP16>(), rvv_pick_gemm_f16(), 2);
}

static rvcl_status_t create_inner_product_bf16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return make_inner_product_kernel(out, desc,
            inner_product_vtable<RVCL_DTYPE_BF16>(), rvv_pick_gemm_bf16(), 2);
}

/* ------------------------------------------------------- registration */

/* Registration tuple per dtype: (A,B) are the storage dtype, acc and C are
 * fp32 — the wide-output habit that keeps up-conversion exact, the K
 * accumulation sequential and the result bit-comparable with the golden. */
static rvcl_status_t inner_product_register_one(const char *name,
        rvcl_dtype_t dt, rvcl_feature_id_t f0, rvcl_feature_id_t f1,
        rvcl_kernel_create_fn create) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_INNER_PRODUCT;
    d.dtype_a = dt;
    d.dtype_b = dt;
    d.dtype_acc = RVCL_DTYPE_FP32;
    d.dtype_c = RVCL_DTYPE_FP32;
    d.quant_support = RVCL_QUANT_NONE;
    /* Assertion level from the dtype trait default (fp32 bit-exact, the
     * fp16 family tolerant), same habit as the matmul entries. */
    d.assert_level = rvcl_dtype_traits(dt).assert_level_default;
    d.required_features[0] = f0;
    d.required_features[1] = f1;
    int nfeat = 0;
    if (f0 != RVCL_FEATURE_NONE) {
        nfeat = 1;
        if (f1 != RVCL_FEATURE_NONE) nfeat = 2;
    }
    d.num_required_features = nfeat;
    d.alignment_a = d.alignment_b = d.alignment_c = 4;
    /* ANY on purpose: this backend carries no vector code of its own, its
     * compute is the shared GEMM micro-kernel that rvv_pick_gemm_* picks
     * for the dtype. */
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_cost_inner_product_rvv;

    return rvcl_register_kernel(&d, create);
}

#if defined(__riscv_vector)

RVCL_API rvcl_status_t rvcl_rvv_register_inner_product(void) {
    rvcl_status_t st = inner_product_register_one("rvcl_inner_product_f32_rvv",
            RVCL_DTYPE_FP32, RVCL_FEATURE_RVV, RVCL_FEATURE_NONE,
            create_inner_product_f32);
    if (st != RVCL_SUCCESS) return st;
    st = inner_product_register_one("rvcl_inner_product_f16_rvv",
            RVCL_DTYPE_FP16, RVCL_FEATURE_RVV, RVCL_FEATURE_FP16,
            create_inner_product_f16);
    if (st != RVCL_SUCCESS) return st;
    /* bf16 up-converts by integer widen+shift, no Zvfbfmin needed. */
    return inner_product_register_one("rvcl_inner_product_bf16_rvv",
            RVCL_DTYPE_BF16, RVCL_FEATURE_RVV, RVCL_FEATURE_NONE,
            create_inner_product_bf16);
}

RVCL_API rvcl_status_t rvcl_rvv_register_inner_product_portable(void) {
    return RVCL_SUCCESS; /* vector build: portable entries stay out */
}

#else /* !__riscv_vector */

RVCL_API rvcl_status_t rvcl_rvv_register_inner_product(void) {
    return RVCL_SUCCESS; /* non-vector build: generic entries stay out */
}

/* Portable scalar entries: no feature requirement (serve any host). */
RVCL_API rvcl_status_t rvcl_rvv_register_inner_product_portable(void) {
    rvcl_status_t st = inner_product_register_one(
            "rvcl_inner_product_f32_rvv_c", RVCL_DTYPE_FP32, RVCL_FEATURE_NONE,
            RVCL_FEATURE_NONE, create_inner_product_f32);
    if (st != RVCL_SUCCESS) return st;
    st = inner_product_register_one("rvcl_inner_product_f16_rvv_c",
            RVCL_DTYPE_FP16, RVCL_FEATURE_NONE, RVCL_FEATURE_NONE,
            create_inner_product_f16);
    if (st != RVCL_SUCCESS) return st;
    return inner_product_register_one("rvcl_inner_product_bf16_rvv_c",
            RVCL_DTYPE_BF16, RVCL_FEATURE_NONE, RVCL_FEATURE_NONE,
            create_inner_product_bf16);
}

#endif /* __riscv_vector */
