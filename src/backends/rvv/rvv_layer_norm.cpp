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

/* Row-wise layer normalization: the work row m names one normalized row of
 * norm_size elements, and the three passes over that row run in ascending
 * element order on every path:
 *   - mean: the samples are accumulated left to right, then divided once by
 *     norm_size;
 *   - variance: (x - mean) is squared without fusion and accumulated left to
 *     right, then divided once by norm_size (biased);
 *   - write-back: sm = scale / sqrt(var + eps) is one division per element
 *     (never a reciprocal times the value), and y = sm * (x - mean) + shift
 *     is one fused multiply-add.
 * The RVV path streams the row through vfredosum, whose reduction is
 * ordered, so chaining the chunks of a row through one accumulator scalar
 * reproduces the scalar left-to-right sum bit for bit; vfredusum (the
 * unordered form) is deliberately not used. The write-back uses vfdiv for
 * the scale division and vfmacc for the fused update, and sqrt goes through
 * a scalar sqrtf call, so both paths are bit-identical to each other and to
 * a scalar reference. The destination is fp32 (the dtype's accumulation
 * width), so the up-conversion of a narrow input is exact and the result is
 * bit-comparable with the golden. There is no packing and no workspace:
 * run() reads the source and writes the destination directly. */

#include "rvv_internal.hpp"

#include "../../common/dtype_traits.hpp"
#include "../../runtime/cost_model.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rvv_bf16.hpp"
#include "rvv_fp16.hpp"

#if defined(__riscv_vector)
#include <riscv_vector.h>
#endif

/* ------------------------------------------------------------- helpers */

/* Narrows the per-dtype soft conversions to one conversion pair the row
 * loops call without branching. */
typedef struct {
    size_t elt; /* source element size in bytes */
    float (*to_f32)(const void *src); /* one stored sample -> fp32 */
} ln_conv_t;

static float conv_f32(const void *src) {
    float v;
    memcpy(&v, src, sizeof(v));
    return v;
}

static float conv_f16(const void *src) {
    uint16_t h;
    memcpy(&h, src, sizeof(h));
    return rvv_f16_to_f32(h);
}

static float conv_bf16(const void *src) {
    uint16_t h;
    memcpy(&h, src, sizeof(h));
    return rvv_bf16_to_f32(h);
}

template <rvcl_dtype_t DT>
static const ln_conv_t *ln_conv(void);

template <>
const ln_conv_t *ln_conv<RVCL_DTYPE_FP32>(void) {
    static const ln_conv_t c = {sizeof(float), conv_f32};
    return &c;
}

template <>
const ln_conv_t *ln_conv<RVCL_DTYPE_FP16>(void) {
    static const ln_conv_t c = {sizeof(uint16_t), conv_f16};
    return &c;
}

template <>
const ln_conv_t *ln_conv<RVCL_DTYPE_BF16>(void) {
    static const ln_conv_t c = {sizeof(uint16_t), conv_bf16};
    return &c;
}

/* round(d * d) with no fused-multiply-add contraction: fma(d, d, +0) rounds
 * the exact square once, which is exactly the IEEE multiply, and the
 * explicit call keeps a compiler from contracting `sum += d * d` into a
 * single-rounding fma that would diverge from the vector path's vfmul. */
static inline float ln_square(float d) {
    return fmaf(d, d, 0.0f);
}

/* ------------------------------------------------------- instance state */

typedef struct rvv_layer_norm_s rvv_layer_norm_t;
struct rvv_layer_norm_s {
    rvcl_kernel_t base; /* must be first */
    rvcl_layer_norm_desc_t d; /* validated request, configured once */
    const ln_conv_t *conv; /* source element size + up-conversion */
    /* Row-band executor: the C loop on every host, the row-wise vector path
     * on vector builds. Both are bit-identical (see the file header). */
    void (*rows)(rvv_layer_norm_t *rk, const uint8_t *src, const float *scale,
            const float *shift, float *dst, int64_t row_begin, int64_t row_end);
};

/* ------------------------------------------------------ C-portable path */

/* The three passes of one row, on one stored sample at a time. The
 * destination is fp32 (the dtype's accumulation width), so the result is
 * written as the fp32 value itself and is bit-comparable with the golden. */
static void ln_row_scalar(const rvcl_layer_norm_desc_t *d, const ln_conv_t *cv,
        const uint8_t *src, const float *scale, const float *shift,
        float *dst) {
    const int64_t n = d->norm_size;
    const size_t elt = cv->elt;

    float sum = 0.0f;
    for (int64_t c = 0; c < n; c++)
        sum += cv->to_f32(src + (size_t)c * elt);
    const float mean = sum / (float)n;

    float acc = 0.0f;
    for (int64_t c = 0; c < n; c++) {
        const float dev = cv->to_f32(src + (size_t)c * elt) - mean;
        acc += ln_square(dev);
    }
    const float var = acc / (float)n;
    const float sqrt_var = sqrtf(var + d->eps);

    for (int64_t c = 0; c < n; c++) {
        const float sm = d->use_scale ? scale[c] / sqrt_var : 1.0f / sqrt_var;
        const float sh = d->use_shift ? shift[c] : 0.0f;
        const float dev = cv->to_f32(src + (size_t)c * elt) - mean;
        dst[c] = fmaf(sm, dev, sh);
    }
}

/* The portable row executor: the only path on a non-vector build. */
static void ln_rows_scalar(rvv_layer_norm_t *rk, const uint8_t *src,
        const float *scale, const float *shift, float *dst, int64_t row_begin,
        int64_t row_end) {
    const rvcl_layer_norm_desc_t *d = &rk->d;
    const ln_conv_t *cv = rk->conv;
    const size_t row_in_bytes = (size_t)d->norm_size * cv->elt;
    const size_t row_out_bytes = (size_t)d->norm_size * sizeof(float);

    for (int64_t r = row_begin; r < row_end; r++)
        ln_row_scalar(d, cv, src + (size_t)r * row_in_bytes, scale, shift,
                (float *)((uint8_t *)dst + (size_t)r * row_out_bytes));
}

#if defined(__riscv_vector)

/* ------------------------------------------------------ RVV vector path */

/* Source kinds the widening loader switches on. */
enum { LN_SRC_F32 = 0, LN_SRC_F16 = 1, LN_SRC_BF16 = 2 };

/* One strip of a row widened to fp32 lanes. f32 loads directly, f16 widens
 * through vfwcvt (the exact fp16 -> fp32 conversion) and bf16 widens by the
 * exact integer zero-extend plus shift that rvv_bf16_to_f32 performs.
 * f16 and bf16 load e16m1, whose element count matches e32m2 at the same
 * VLMAX, so one vsetvli serves both the load and the fp32 arithmetic. */
static vfloat32m2_t ln_load_f32v(const uint8_t *p, int kind, long vl) {
    if (kind == LN_SRC_F32)
        return __riscv_vle32_v_f32m2((const float *)p, (size_t)vl);
    if (kind == LN_SRC_F16) {
        const vfloat16m1_t h
                = __riscv_vle16_v_f16m1((const _Float16 *)p, (size_t)vl);
        return __riscv_vfwcvt_f_f_v_f32m2(h, (size_t)vl);
    }
    const vuint16m1_t h
            = __riscv_vle16_v_u16m1((const uint16_t *)p, (size_t)vl);
    const vuint32m2_t w = __riscv_vsll_vx_u32m2(
            __riscv_vzext_vf2_u32m2(h, (size_t)vl), 16, (size_t)vl);
    return __riscv_vreinterpret_f32m2(w);
}

/* Ordered sum of one row: the row is streamed in VL-sized strips and each
 * strip's vfredosum folds into the running scalar, so the accumulation is
 * the plain left-to-right sum of the samples (vfredosum is the ordered
 * form; vfredusum would allow a tree reduction and is not used). When
 * do_var is set the samples are first centered on mean and squared without
 * fusion, mirroring ln_square. */
static float ln_sum_rvv(const uint8_t *src, int64_t n, size_t elt, int kind,
        float mean, int do_var) {
    vfloat32m1_t acc = __riscv_vfmv_v_f_f32m1(0.0f, 1);
    for (int64_t c = 0; c < n;) {
        const long vl = __riscv_vsetvl_e32m2((size_t)(n - c));
        const vfloat32m2_t v = ln_load_f32v(src + (size_t)c * elt, kind, vl);
        vfloat32m2_t term = v;
        if (do_var) {
            const vfloat32m2_t dev
                    = __riscv_vfsub_vf_f32m2(v, mean, (size_t)vl);
            term = __riscv_vfmul_vv_f32m2(dev, dev, (size_t)vl);
        }
        acc = __riscv_vfredosum_vs_f32m2_f32m1(term, acc, (size_t)vl);
        c += (int64_t)vl;
    }
    return __riscv_vfmv_f_s_f32m1_f32(acc);
}

/* One row: the two reduction passes above, then the element-wise write-back
 * with vfdiv for the scale division (one division per element, never a
 * reciprocal times the value) and vfmacc for the fused update. */
static void ln_row_rvv(const rvcl_layer_norm_desc_t *d, int kind, size_t elt,
        const uint8_t *src, const float *scale, const float *shift,
        float *dst) {
    const int64_t n = d->norm_size;
    const float mean = ln_sum_rvv(src, n, elt, kind, 0.0f, 0) / (float)n;
    const float var = ln_sum_rvv(src, n, elt, kind, mean, 1) / (float)n;
    const float sqrt_var = sqrtf(var + d->eps);
    const float inv_sqrt = 1.0f / sqrt_var;

    for (int64_t c = 0; c < n;) {
        const long vl = __riscv_vsetvl_e32m2((size_t)(n - c));
        const size_t avl = (size_t)vl;
        const vfloat32m2_t v = ln_load_f32v(src + (size_t)c * elt, kind, vl);
        const vfloat32m2_t dev = __riscv_vfsub_vf_f32m2(v, mean, avl);
        /* scale absent: the constant 1 / sqrt(var + eps), one division of
         * the scalar, broadcast to the strip. */
        const vfloat32m2_t vsm = d->use_scale
                ? __riscv_vfdiv_vf_f32m2(
                          __riscv_vle32_v_f32m2(scale + c, avl), sqrt_var, avl)
                : __riscv_vfmv_v_f_f32m2(inv_sqrt, avl);
        const vfloat32m2_t vsh = d->use_shift
                ? __riscv_vle32_v_f32m2(shift + c, avl)
                : __riscv_vfmv_v_f_f32m2(0.0f, avl);
        const vfloat32m2_t vy = __riscv_vfmacc_vv_f32m2(vsh, vsm, dev, avl);
        __riscv_vse32_v_f32m2(dst + c, vy, avl);
        c += (int64_t)vl;
    }
}

static void ln_rows_rvv(rvv_layer_norm_t *rk, const uint8_t *src,
        const float *scale, const float *shift, float *dst, int64_t row_begin,
        int64_t row_end) {
    const rvcl_layer_norm_desc_t *d = &rk->d;
    const int kind = d->dtype_a == RVCL_DTYPE_FP32
            ? LN_SRC_F32
            : (d->dtype_a == RVCL_DTYPE_FP16 ? LN_SRC_F16 : LN_SRC_BF16);
    const size_t row_in_bytes = (size_t)d->norm_size * rk->conv->elt;

    for (int64_t r = row_begin; r < row_end; r++)
        ln_row_rvv(d, kind, rk->conv->elt, src + (size_t)r * row_in_bytes,
                scale, shift, dst + (size_t)r * d->norm_size);
}

#endif /* __riscv_vector */

/* ------------------------------------------------------------- lifecycle */

/* Every kernel instance serves exactly one (A,B,acc,C) tuple and the output
 * is fp32 (the dtype's accumulation width), so the tuple is part of the
 * instance's contract: a request for another tuple is rejected here rather
 * than computed through a mismatched conversion. */
template <rvcl_dtype_t DT>
static rvcl_status_t layer_norm_validate_tuple(const rvcl_op_desc_t *desc) {
    const rvcl_layer_norm_desc_t *d = &desc->layer_norm;
    if (d->rows <= 0 || d->norm_size <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    /* eps sits under the square root: a negative value turns the whole row
     * into NaN, which is a request error rather than a numeric outcome. */
    if (!(d->eps >= 0.0f)) return RVCL_ERROR_INVALID_ARGUMENT;
    /* Layer norm has one operand, so the B slot is unused: AUTO ("not
     * specified") and the operand's own dtype are the two spellings that
     * describe it, and both are accepted here exactly as the dispatch
     * validator accepts them. */
    if (d->dtype_a != DT || (d->dtype_b != RVCL_DTYPE_AUTO && d->dtype_b != DT)
            || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* AUTO is the request-side "derive it" marker for the accumulator. */
    if (d->dtype_acc != RVCL_DTYPE_AUTO && d->dtype_acc != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Layer norm has no bias term: AUTO is the only accepted value. */
    if (d->dtype_bias != RVCL_DTYPE_AUTO) return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Quantization is not served by the layer norm path yet: reject the
     * request here rather than compute it as if it were unquantized. */
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    return RVCL_SUCCESS;
}

template <rvcl_dtype_t DT>
static rvcl_status_t layer_norm_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap; /* the row-wise vector path is plain RVV, gated by features */
    return layer_norm_validate_tuple<DT>(desc);
}

static rvcl_status_t layer_norm_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    rvv_layer_norm_t *rk = (rvv_layer_norm_t *)k;
    const rvcl_layer_norm_desc_t *d = &desc->layer_norm;
    if (d->rows <= 0 || d->norm_size <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!(d->eps >= 0.0f)) return RVCL_ERROR_INVALID_ARGUMENT;
    rk->d = *d;
    /* No packing, no workspace: run() reads the source and writes the
     * destination directly. */
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t layer_norm_run(rvcl_kernel_t *k,
        const rvcl_tensor_pack_t *pack, const rvcl_work_range_t *range) {
    rvv_layer_norm_t *rk = (rvv_layer_norm_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    if (!a || !c || !a->data || !c->data) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!range) return RVCL_ERROR_INVALID_ARGUMENT;
    /* The optional per-row parameters ride the stat slots; the request
     * flags say which of them are live, so a missing slot for a live flag
     * is a caller error rather than a silently substituted constant. */
    const rvcl_tensor_t *ts
            = rvcl_tensor_pack_get(pack, RVCL_TENSOR_STAT_SCALE);
    const rvcl_tensor_t *th
            = rvcl_tensor_pack_get(pack, RVCL_TENSOR_STAT_SHIFT);
    const float *scale = ts ? (const float *)ts->data : NULL;
    const float *shift = th ? (const float *)th->data : NULL;
    if ((rk->d.use_scale && !scale) || (rk->d.use_shift && !shift))
        return RVCL_ERROR_INVALID_ARGUMENT;
    /* Full n and k only: a work row is a whole normalized row and there is
     * no splittable reduction axis. */
    if (range->m_begin < 0 || range->m_end > rk->d.rows
            || range->m_begin > range->m_end || range->n_begin != 0
            || range->n_end != rk->d.norm_size || range->k_begin != 0
            || range->k_end != 1)
        return RVCL_ERROR_INVALID_ARGUMENT;

    rk->rows(rk, (const uint8_t *)a->data, scale, shift, (float *)c->data,
            range->m_begin, range->m_end);
    return RVCL_SUCCESS;
}

static void layer_norm_destroy(rvcl_kernel_t *k) {
    (void)k;
}

/* One vtable per dtype: validate() carries the instance's tuple. */
template <rvcl_dtype_t DT>
static const rvcl_kernel_vtable_t *layer_norm_vtable(void) {
    static const rvcl_kernel_vtable_t vt = {
            .validate = layer_norm_validate<DT>,
            .configure = layer_norm_configure,
            .pack_a = NULL, /* no packing: one input tensor */
            .pack_b = NULL,
            .run = layer_norm_run,
            .destroy = layer_norm_destroy,
    };
    return &vt;
}

/* --------------------------------------------------- per-dtype factories */

static rvcl_status_t make_layer_norm_kernel(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, const rvcl_kernel_vtable_t *vt,
        const ln_conv_t *conv,
        void (*rows)(rvv_layer_norm_t *, const uint8_t *, const float *,
                const float *, float *, int64_t, int64_t)) {
    const rvcl_status_t st
            = rvcl_kernel_alloc(out, desc, sizeof(rvv_layer_norm_t));
    if (st != RVCL_SUCCESS) return st;
    rvv_layer_norm_t *rk = (rvv_layer_norm_t *)*out;
    rk->base.vt = vt;
    rk->conv = conv;
    rk->rows = rows;
    return RVCL_SUCCESS;
}

template <rvcl_dtype_t DT>
static rvcl_status_t create_layer_norm(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
#if defined(__riscv_vector)
    /* The vector build keeps the portable executor compiled as the scalar
     * reference of this file; naming it here keeps that use visible. */
    (void)ln_rows_scalar;
    return make_layer_norm_kernel(
            out, desc, layer_norm_vtable<DT>(), ln_conv<DT>(), ln_rows_rvv);
#else
    return make_layer_norm_kernel(
            out, desc, layer_norm_vtable<DT>(), ln_conv<DT>(), ln_rows_scalar);
#endif
}

static rvcl_status_t create_layer_norm_f32(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return create_layer_norm<RVCL_DTYPE_FP32>(out, desc);
}

static rvcl_status_t create_layer_norm_f16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return create_layer_norm<RVCL_DTYPE_FP16>(out, desc);
}

static rvcl_status_t create_layer_norm_bf16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return create_layer_norm<RVCL_DTYPE_BF16>(out, desc);
}

/* ------------------------------------------------------- registration */

/* Registration tuple per dtype: (A,B) are the storage dtype, acc and C are
 * fp32 — the wide-output habit that keeps up-conversion exact and the result
 * bit-comparable with the golden. */
static rvcl_status_t layer_norm_register_one(const char *name, rvcl_dtype_t dt,
        rvcl_feature_id_t f0, rvcl_feature_id_t f1,
        rvcl_kernel_create_fn create) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_LAYER_NORM;
    d.dtype_a = dt;
    d.dtype_b = dt;
    d.dtype_acc = RVCL_DTYPE_FP32;
    d.dtype_c = RVCL_DTYPE_FP32;
    d.quant_support = RVCL_QUANT_NONE;
    /* Assertion level from the dtype trait default (fp32 bit-exact, the
     * fp16 family tolerant), same habit as the pooling entries. */
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
    /* ANY on purpose: the row-wise vector path is plain RVV, so the tier
     * split carries no information (one code serves every VLEN). */
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_cost_layer_norm_rvv;

    return rvcl_register_kernel(&d, create);
}

#if defined(__riscv_vector)

RVCL_API rvcl_status_t rvcl_rvv_register_layer_norm(void) {
    rvcl_status_t st = layer_norm_register_one("rvcl_layer_norm_f32_rvv",
            RVCL_DTYPE_FP32, RVCL_FEATURE_RVV, RVCL_FEATURE_NONE,
            create_layer_norm_f32);
    if (st != RVCL_SUCCESS) return st;
    /* The f16 widening uses vfwcvt, so the entry requires Zvfhmin/Zvfh. */
    st = layer_norm_register_one("rvcl_layer_norm_f16_rvv", RVCL_DTYPE_FP16,
            RVCL_FEATURE_RVV, RVCL_FEATURE_FP16, create_layer_norm_f16);
    if (st != RVCL_SUCCESS) return st;
    /* bf16 up-converts by integer widen+shift, no Zvfbfmin needed. */
    return layer_norm_register_one("rvcl_layer_norm_bf16_rvv", RVCL_DTYPE_BF16,
            RVCL_FEATURE_RVV, RVCL_FEATURE_NONE, create_layer_norm_bf16);
}

RVCL_API rvcl_status_t rvcl_rvv_register_layer_norm_portable(void) {
    return RVCL_SUCCESS; /* vector build: portable entries stay out */
}

#else /* !__riscv_vector */

RVCL_API rvcl_status_t rvcl_rvv_register_layer_norm(void) {
    return RVCL_SUCCESS; /* non-vector build: generic entries stay out */
}

/* Portable scalar entries: no feature requirement (serve any host). */
RVCL_API rvcl_status_t rvcl_rvv_register_layer_norm_portable(void) {
    rvcl_status_t st = layer_norm_register_one("rvcl_layer_norm_f32_rvv_c",
            RVCL_DTYPE_FP32, RVCL_FEATURE_NONE, RVCL_FEATURE_NONE,
            create_layer_norm_f32);
    if (st != RVCL_SUCCESS) return st;
    st = layer_norm_register_one("rvcl_layer_norm_f16_rvv_c", RVCL_DTYPE_FP16,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_layer_norm_f16);
    if (st != RVCL_SUCCESS) return st;
    return layer_norm_register_one("rvcl_layer_norm_bf16_rvv_c",
            RVCL_DTYPE_BF16, RVCL_FEATURE_NONE, RVCL_FEATURE_NONE,
            create_layer_norm_bf16);
}

#endif /* __riscv_vector */
