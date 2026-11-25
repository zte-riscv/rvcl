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

/* NCHW conv2d served as im2col + the dtype's packed-B GEMM micro-kernel:
 *   A[m, k] = src patches   m = mb*OH*OW, k = ic*kh*kw
 *   B[k, n] = w^T           n = oc
 *   C[m, n] = the GEMM output, then transposed to dst [mb][oc][oh][ow].
 * The NCHW weights are [oc][k] row-major, i.e. B transposed, so pack_b
 * transposes them into the panel layout the GEMM consumes (the shared
 * rvv_pack_b_panel assumes B[k][n] with n innermost and cannot be reused
 * as-is). run() then only builds the im2col patch rows of the batches it
 * covers.
 *
 * Workspaces: AUX_0 packed-B panels (KERNEL lifetime, as the matmul path),
 * AUX_1 one GEMM band: A patches, padded up to a 16-byte boundary, followed
 * by the f32 GEMM output. The band covers OH*OW rows of one batch, so a
 * run's staging is bounded by OH*OW*(K+OC) elements regardless of the batch
 * count.
 *
 * Work ranges: the m axis is mb*OH*OW. run() accepts any range and walks
 * the batches it spans one after another, reusing the single band staging
 * serially. Because that staging is shared, concurrent slices must not
 * span (or share) a batch: the semantic layer issues one run per batch and
 * lets the scheduler slice inside it. n and k must be the full ranges (no
 * split-K, no column slicing: the writeback is a whole-batch transpose).
 *
 * The im2col, writeback and transposition loops are plain C compiled on
 * every host; only the GEMM micro-kernel itself is RVV-specialized, and it
 * is selected through the same rvv_pick_gemm_* entry the matmul backend
 * uses, so conv and matmul can never run different kernels of one dtype.
 * run() never mallocs and never threads. */

#include "rvv_internal.hpp"

#include "../../common/conv2d_geometry.h"
#include "../../common/dtype_traits.hpp"
#include "../../runtime/cost_model.h"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    rvcl_conv2d_desc_t d; /* validated request, configured once */
    int64_t oh, ow; /* output height / width */
    int64_t k, m_band; /* GEMM reduction; GEMM rows per batch */
    rvv_gemm_fn gemm; /* dtype-specific micro-kernel entry */
    size_t elt; /* element size of A/B storage */
    /* Band workspace layout: [m_band][k] im2col patches rounded up to a
     * 16-byte multiple, then the f32 GEMM output [m_band][oc]. */
    size_t a_bytes;
    size_t c_bytes;
    void (*pack_patch)(void *dst, const void *src); /* 1 storage element */
} rvv_conv2d_t;

/* ------------------------------------------------------------- helpers */

/* Rejects a geometry the configure() formulas cannot divide: the output
 * size divides by the stride and multiplies by the dilation, and the band
 * sizing multiplies the spatial extents. Checking here as well as in the
 * dispatch request validator keeps a factory-bypass caller (validate then
 * configure, without the dispatch entry) from dividing by zero. */
static rvcl_status_t conv2d_validate_geometry(const rvcl_conv2d_desc_t *d) {
    return rvcl_conv2d_geometry_valid(d) ? RVCL_SUCCESS
                                         : RVCL_ERROR_INVALID_ARGUMENT;
}

/* ------------------------------------------------- per-dtype element ops */

static void patch_f32(void *dst, const void *src) {
    memcpy(dst, src, sizeof(float));
}

static void patch_f16(void *dst, const void *src) {
    memcpy(dst, src, sizeof(uint16_t));
}

static void patch_bf16(void *dst, const void *src) {
    memcpy(dst, src, sizeof(uint16_t));
}

/* ------------------------------------------------------------- lifecycle */

/* Every kernel instance serves exactly one (A,B,acc,C) tuple and the
 * output is fp32 (the dtype's accumulation width), so the tuple is part of
 * the instance's contract: a request for another tuple is rejected here
 * rather than computed through a mismatched gemm entry. */
template <rvcl_dtype_t DT>
static rvcl_status_t conv2d_validate_tuple(const rvcl_op_desc_t *desc) {
    const rvcl_conv2d_desc_t *d = &desc->conv2d;
    /* The geometry is part of the instance's contract too: configure()
     * divides by the stride and multiplies by the dilation, so a zero
     * stride/dilation or a negative pad must not reach it. */
    const rvcl_status_t geo = conv2d_validate_geometry(d);
    if (geo != RVCL_SUCCESS) return geo;
    if (d->dtype_a != DT || d->dtype_b != DT || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* AUTO is the request-side "derive it" marker for the accumulator. */
    if (d->dtype_acc != RVCL_DTYPE_AUTO && d->dtype_acc != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Bias is read as f32 by the gemm seed; AUTO means "no bias". */
    if (d->dtype_bias != RVCL_DTYPE_AUTO && d->dtype_bias != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Quantization is not served by the conv path yet: reject the request
     * here rather than compute it as if it were unquantized. */
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    /* Grouped convolution is reserved and must stay 1. */
    if (d->groups != 1) return RVCL_ERROR_INVALID_ARGUMENT;
    return RVCL_SUCCESS;
}

template <rvcl_dtype_t DT>
static rvcl_status_t conv2d_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap; /* the gemm entry carries its own feature gating */
    return conv2d_validate_tuple<DT>(desc);
}

static rvcl_status_t conv2d_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    rvv_conv2d_t *rk = (rvv_conv2d_t *)k;
    const rvcl_conv2d_desc_t *d = &desc->conv2d;
    rk->d = *d;
    const rvcl_status_t geo = conv2d_validate_geometry(d);
    if (geo != RVCL_SUCCESS) return geo;
    rk->oh = rvcl_conv2d_out_dim(
            d->ih, d->kh, d->sh, d->dil_h, d->pad_t, d->pad_b);
    rk->ow = rvcl_conv2d_out_dim(
            d->iw, d->kw, d->sw, d->dil_w, d->pad_l, d->pad_r);
    if (rk->oh <= 0 || rk->ow <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    rk->m_band = rk->oh * rk->ow;
    rk->k = d->ic * d->kh * d->kw;
    /* The f32 GEMM output starts a_bytes into the band, so the patch area
     * is rounded up to a 16-byte boundary: with a 2-byte storage element
     * an odd m_band*k would otherwise leave the output region at a 2-byte
     * alignment and every float store into it would be misaligned. Only
     * the output region's start depends on a_bytes; the patch row stride
     * stays k*elt (see conv2d_im2col). */
    const size_t patch_bytes = (size_t)(rk->m_band * rk->k) * rk->elt;
    rk->a_bytes = (patch_bytes + 15u) & ~(size_t)15;
    rk->c_bytes = (size_t)(rk->m_band * d->oc) * sizeof(float);

    if (ws_req) {
        /* Slot 0: packed-B panels (the [oc][k] weights transposed). */
        ws_req[RVV_WS_PACKED_B] = rvcl_workspace_req_t {
                .size = (int64_t)rvv_packed_b_bytes(rk->k, d->oc, rk->elt),
                .alignment = 64,
                .lifetime = RVCL_WS_LIFETIME_KERNEL,
        };
        /* Slot 1: one GEMM band — im2col patches, then the f32 GEMM
         * output. RUN lifetime: rewritten by every run(). */
        ws_req[RVV_WS_BAND] = rvcl_workspace_req_t {
                .size = (int64_t)(rk->a_bytes + rk->c_bytes),
                .alignment = 64,
                .lifetime = RVCL_WS_LIFETIME_RUN,
        };
    }
    if (num_ws) *num_ws = 2;
    return RVCL_SUCCESS;
}

/* pack_b: the B operand of the conv GEMM is w^T (k rows, oc columns) while
 * the NCHW weights are [oc][k] row-major, so the panel packing is a
 * transposing copy into the shared panel layout. */
static rvcl_status_t conv2d_pack_b(
        rvcl_kernel_t *k, const rvcl_tensor_t *src, rvcl_tensor_t *dst) {
    rvv_conv2d_t *rk = (rvv_conv2d_t *)k;
    if (!rk || !src || !src->data || !dst || !dst->data || rk->k <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    const int64_t kd = rk->k, n = rk->d.oc;
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

/* Builds the im2col patch rows [row_begin, row_end) of one batch: patch
 * row r = oh*OW + ow gathers src[ic][ih][iw] for the (ic, kh, kw) triple,
 * with out-of-input samples written as zero. */
static void conv2d_im2col(rvv_conv2d_t *rk, const uint8_t *src_mb,
        int64_t row_begin, int64_t row_end, uint8_t *a) {
    const rvcl_conv2d_desc_t *d = &rk->d;
    const int64_t ow = rk->ow;
    const size_t elt = rk->elt;

    for (int64_t r = row_begin; r < row_end; r++) {
        const int64_t oh = r / ow;
        const int64_t ow_i = r - oh * ow;
        uint8_t *dst = a + (size_t)r * (size_t)rk->k * elt;
        int64_t p = 0;
        for (int64_t ic = 0; ic < d->ic; ic++) {
            const uint8_t *src_c = src_mb + (size_t)(ic * d->ih * d->iw) * elt;
            for (int64_t kh = 0; kh < d->kh; kh++) {
                const int64_t ih = oh * d->sh - d->pad_t + kh * d->dil_h;
                const int in_row = ih >= 0 && ih < d->ih;
                for (int64_t kw = 0; kw < d->kw; kw++, p++) {
                    const int64_t iw = ow_i * d->sw - d->pad_l + kw * d->dil_w;
                    uint8_t *out = dst + (size_t)p * elt;
                    if (in_row && iw >= 0 && iw < d->iw) {
                        rk->pack_patch(
                                out, src_c + (size_t)(ih * d->iw + iw) * elt);
                    } else {
                        memset(out, 0, elt);
                    }
                }
            }
        }
    }
}

/* Runs one batch's row slice [row_begin, row_end): im2col, GEMM into the
 * f32 band, then the transpose writeback into the NCHW destination. */
static void conv2d_run_rows(rvv_conv2d_t *rk, const uint8_t *src_mb,
        int64_t row_begin, int64_t row_end, const void *bias, void *ws_b,
        uint8_t *band, void *dst_mb) {
    const rvcl_conv2d_desc_t *d = &rk->d;
    const int64_t m_band = rk->m_band;
    uint8_t *patches = band;
    float *c_f32 = (float *)(band + rk->a_bytes);

    conv2d_im2col(rk, src_mb, row_begin, row_end, patches);

    /* GEMM over the band rows only; A and C share the full band's row
     * stride, so a patch row sits at its own band index. The bias rides
     * the gemm seed (bias-first, the same addition order as the golden). */
    const rvcl_work_range_t band_rng = {
            .m_begin = row_begin,
            .m_end = row_end,
            .n_begin = 0,
            .n_end = d->oc,
            .k_begin = 0,
            .k_end = rk->k,
    };
    rk->gemm(patches, ws_b, bias, c_f32, m_band, d->oc, rk->k, &band_rng);

    /* The GEMM wrote [row][oc]; the destination is [oc][row] inside the
     * batch, so transpose on the way out. The output is fp32 (the dtype's
     * accumulation width), so the writeback is a plain element copy. */
    uint8_t *dst = (uint8_t *)dst_mb;
    for (int64_t oc = 0; oc < d->oc; oc++) {
        uint8_t *dst_row = dst + (size_t)(oc * m_band) * sizeof(float);
        for (int64_t r = row_begin; r < row_end; r++)
            memcpy(dst_row + (size_t)r * sizeof(float), &c_f32[r * d->oc + oc],
                    sizeof(float));
    }
}

static rvcl_status_t conv2d_run(rvcl_kernel_t *k,
        const rvcl_tensor_pack_t *pack, const rvcl_work_range_t *range) {
    rvv_conv2d_t *rk = (rvv_conv2d_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *b = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    const rvcl_tensor_t *ws_b = rvcl_tensor_pack_get(
            pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + RVV_WS_PACKED_B));
    const rvcl_tensor_t *ws_band = rvcl_tensor_pack_get(
            pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + RVV_WS_BAND));
    if (!a || !b || !c || !ws_b || !ws_band) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!range) return RVCL_ERROR_INVALID_ARGUMENT;

    const rvcl_conv2d_desc_t *d = &rk->d;
    const int64_t m_band = rk->m_band;
    const int64_t total = d->mb * m_band;
    /* Full n and k only: the writeback is a whole-batch transpose and the
     * reduction is not split. The m range may span batches; they are
     * walked serially over the single band workspace. */
    if (range->m_begin < 0 || range->m_end > total
            || range->m_begin > range->m_end || range->n_begin != 0
            || range->n_end != d->oc || range->k_begin != 0
            || range->k_end != rk->k)
        return RVCL_ERROR_INVALID_ARGUMENT;

    if (!pack->src_b_packed) {
        const rvcl_status_t st = conv2d_pack_b(k, b, (rvcl_tensor_t *)ws_b);
        if (st != RVCL_SUCCESS) return st;
    }

    const size_t src_mb_bytes = (size_t)(d->ic * d->ih * d->iw) * rk->elt;
    const size_t dst_mb_bytes = (size_t)(d->oc * m_band) * sizeof(float);
    uint8_t *band = (uint8_t *)ws_band->data;

    for (int64_t m = range->m_begin; m < range->m_end;) {
        const int64_t mb = m / m_band;
        const int64_t mb_end = (mb + 1) * m_band;
        const int64_t seg_end = range->m_end < mb_end ? range->m_end : mb_end;
        conv2d_run_rows(rk,
                (const uint8_t *)a->data + (size_t)mb * src_mb_bytes,
                m - mb * m_band, seg_end - mb * m_band,
                bias ? bias->data : NULL, ws_b->data, band,
                (uint8_t *)c->data + (size_t)mb * dst_mb_bytes);
        m = seg_end;
    }
    return RVCL_SUCCESS;
}

static void conv2d_destroy(rvcl_kernel_t *k) {
    (void)k;
}

/* One vtable per dtype: validate() carries the instance's tuple. */
template <rvcl_dtype_t DT>
static const rvcl_kernel_vtable_t *conv2d_vtable(void) {
    static const rvcl_kernel_vtable_t vt = {
            .validate = conv2d_validate<DT>,
            .configure = conv2d_configure,
            .pack_a = NULL,
            .pack_b = conv2d_pack_b,
            .run = conv2d_run,
            .destroy = conv2d_destroy,
    };
    return &vt;
}

/* --------------------------------------------------- per-dtype factories */

static rvcl_status_t make_conv2d_kernel(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, const rvcl_kernel_vtable_t *vt,
        rvv_gemm_fn gemm, size_t elt,
        void (*pack_patch)(void *, const void *)) {
    if (!gemm) return RVCL_ERROR_INVALID_ARGUMENT;
    const rvcl_status_t st = rvcl_kernel_alloc(out, desc, sizeof(rvv_conv2d_t));
    if (st != RVCL_SUCCESS) return st;
    rvv_conv2d_t *rk = (rvv_conv2d_t *)*out;
    rk->base.vt = vt;
    rk->gemm = gemm;
    rk->elt = elt;
    rk->pack_patch = pack_patch;
    return RVCL_SUCCESS;
}

static rvcl_status_t create_conv2d_f32(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return make_conv2d_kernel(out, desc, conv2d_vtable<RVCL_DTYPE_FP32>(),
            rvv_pick_gemm_f32(), sizeof(float), patch_f32);
}

static rvcl_status_t create_conv2d_f16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return make_conv2d_kernel(out, desc, conv2d_vtable<RVCL_DTYPE_FP16>(),
            rvv_pick_gemm_f16(), 2, patch_f16);
}

static rvcl_status_t create_conv2d_bf16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return make_conv2d_kernel(out, desc, conv2d_vtable<RVCL_DTYPE_BF16>(),
            rvv_pick_gemm_bf16(), 2, patch_bf16);
}

/* ------------------------------------------------------- registration */

/* Registration tuple per dtype: (A,B) are the storage dtype, acc and C are
 * fp32 — the wide-output habit that keeps up-conversion exact, the K
 * accumulation sequential and the result bit-comparable with the golden. */
static rvcl_status_t conv2d_register_one(const char *name, rvcl_dtype_t dt,
        rvcl_feature_id_t f0, rvcl_feature_id_t f1,
        rvcl_kernel_create_fn create) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_CONV2D;
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
     * for the dtype. Per-tier rows would register entries with identical
     * behaviour, so the tier split carries no information until a
     * conv-specific vector path exists. */
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_cost_conv2d_rvv;

    return rvcl_register_kernel(&d, create);
}

#if defined(__riscv_vector)

RVCL_API rvcl_status_t rvcl_rvv_register_conv2d(void) {
    rvcl_status_t st
            = conv2d_register_one("rvcl_conv2d_f32_rvv", RVCL_DTYPE_FP32,
                    RVCL_FEATURE_RVV, RVCL_FEATURE_NONE, create_conv2d_f32);
    if (st != RVCL_SUCCESS) return st;
    st = conv2d_register_one("rvcl_conv2d_f16_rvv", RVCL_DTYPE_FP16,
            RVCL_FEATURE_RVV, RVCL_FEATURE_FP16, create_conv2d_f16);
    if (st != RVCL_SUCCESS) return st;
    /* bf16 up-converts by integer widen+shift, no Zvfbfmin needed. */
    return conv2d_register_one("rvcl_conv2d_bf16_rvv", RVCL_DTYPE_BF16,
            RVCL_FEATURE_RVV, RVCL_FEATURE_NONE, create_conv2d_bf16);
}

RVCL_API rvcl_status_t rvcl_rvv_register_conv2d_portable(void) {
    return RVCL_SUCCESS; /* vector build: portable entries stay out */
}

#else /* !__riscv_vector */

RVCL_API rvcl_status_t rvcl_rvv_register_conv2d(void) {
    return RVCL_SUCCESS; /* non-vector build: generic entries stay out */
}

/* Portable scalar entries: no feature requirement (serve any host). */
RVCL_API rvcl_status_t rvcl_rvv_register_conv2d_portable(void) {
    rvcl_status_t st
            = conv2d_register_one("rvcl_conv2d_f32_rvv_c", RVCL_DTYPE_FP32,
                    RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_conv2d_f32);
    if (st != RVCL_SUCCESS) return st;
    st = conv2d_register_one("rvcl_conv2d_f16_rvv_c", RVCL_DTYPE_FP16,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_conv2d_f16);
    if (st != RVCL_SUCCESS) return st;
    return conv2d_register_one("rvcl_conv2d_bf16_rvv_c", RVCL_DTYPE_BF16,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_conv2d_bf16);
}

#endif /* __riscv_vector */
