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

/* Row-wise NCHW pooling: the work row m = mb*OH + oh names one output line
 * of one batch; the row reduction walks the channels, each one independently
 * over the columns. An RVV build vectorizes along the output row (n = OW
 * columns in flight) and a non-vector build walks the same samples scalar by
 * scalar in the same order, so both are bit-identical to each other and to a
 * scalar reference:
 *   - a window sample outside the input is skipped (max, avg_exclude) or
 *     counts as zero (avg_include);
 *   - max reduces over the in-bounds samples in (kh, kw) order and seeds
 *     from the first in-bounds sample, so the result is one raw input value;
 *     a window with no in-bounds sample (reachable when pad >= k) keeps the
 *     -FLT_MAX seed oneDNN uses (lowest_value of the fp32 destination), and
 *     the empty-window average is 0 (avg_include) or NaN (avg_exclude, 0/0);
 *   - the average accumulates the samples in (kh, kw) order starting from
 *     0 and divides once at the end by KH*KW (avg_include) or by the
 *     in-bounds sample count (avg_exclude).
 * The up-conversion of a narrow input is exact, the accumulation is fp32 and
 * the output is fp32 (the dtype's accumulation width), so the result is
 * bit-comparable with the golden. There is no packing and no workspace:
 * run() reads the source and writes the destination directly. */

#include "rvv_internal.hpp"

#include "../../common/dtype_traits.hpp"
#include "../../common/pooling_geometry.h"
#include "../../runtime/cost_model.h"

#include <limits>
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
} pool_conv_t;

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
static const pool_conv_t *pool_conv(void);

template <>
const pool_conv_t *pool_conv<RVCL_DTYPE_FP32>(void) {
    static const pool_conv_t c = {sizeof(float), conv_f32};
    return &c;
}

template <>
const pool_conv_t *pool_conv<RVCL_DTYPE_FP16>(void) {
    static const pool_conv_t c = {sizeof(uint16_t), conv_f16};
    return &c;
}

template <>
const pool_conv_t *pool_conv<RVCL_DTYPE_BF16>(void) {
    static const pool_conv_t c = {sizeof(uint16_t), conv_bf16};
    return &c;
}

/* ------------------------------------------------------- instance state */

typedef struct rvv_pooling_s rvv_pooling_t;
struct rvv_pooling_s {
    rvcl_kernel_t base; /* must be first */
    rvcl_pooling_desc_t d; /* validated request, configured once */
    int64_t oh, ow; /* output height / width */
    const pool_conv_t *conv; /* source element size + up-conversion */
    /* Row-band executor: the C loop on every host, the row-wise vector path
     * on vector builds. Both are bit-identical (see the file header). */
    void (*rows)(rvv_pooling_t *rk, const uint8_t *src, float *dst,
            int64_t row_begin, int64_t row_end);
};

/* ------------------------------------------------------ C-portable path */

/* Max seed for a window with no in-bounds sample: lowest_value of the fp32
 * destination, the seed oneDNN's ref_pooling uses for max pooling. A 0 seed
 * would swallow every negative sample in a partially padded window. */
static const float k_pool_max_seed = -3.4028234663852886e38f; /* -FLT_MAX */

/* One output element at (oh, ow): the window's in-bounds samples in (kh,
 * kw) order. The loop bounds clamp the window into the input, so a window
 * may overhang it on either side. */
static float pool_scalar(const rvcl_pooling_desc_t *d, const pool_conv_t *cv,
        const uint8_t *src_c, int64_t oh, int64_t ow) {
    const int64_t ih0 = oh * d->sh - d->pad_t;
    const int64_t iw0 = ow * d->sw - d->pad_l;
    const int64_t kh_lo = ih0 < 0 ? -ih0 : 0;
    const int64_t kw_lo = iw0 < 0 ? -iw0 : 0;
    const int64_t kh_hi = d->ih - ih0 < d->kh ? d->ih - ih0 : d->kh;
    const int64_t kw_hi = d->iw - iw0 < d->kw ? d->iw - iw0 : d->kw;

    float best = k_pool_max_seed, sum = 0.0f;
    int64_t count = 0;
    for (int64_t kh = kh_lo; kh < kh_hi; kh++) {
        const uint8_t *row
                = src_c + (size_t)((ih0 + kh) * d->iw + iw0) * cv->elt;
        for (int64_t kw = kw_lo; kw < kw_hi; kw++) {
            const float v = cv->to_f32(row + (size_t)kw * cv->elt);
            if (count == 0 || v > best) best = v;
            sum += v;
            count++;
        }
    }

    switch (d->alg) {
        case RVCL_POOL_MAX:
            /* An empty window (pad >= k) keeps the seed. */
            return best;
        case RVCL_POOL_AVG_INCLUDE: return sum / (float)(d->kh * d->kw);
        case RVCL_POOL_AVG_EXCLUDE:
        default: return sum / (float)count;
    }
}

/* The portable row executor: the only path on a non-vector build. */
static void pool_rows_scalar(rvv_pooling_t *rk, const uint8_t *src, float *dst,
        int64_t row_begin, int64_t row_end) {
    const rvcl_pooling_desc_t *d = &rk->d;
    const pool_conv_t *cv = rk->conv;
    const int64_t in_ch = d->ih * d->iw;
    const int64_t out_ch = rk->oh * rk->ow;

    for (int64_t r = row_begin; r < row_end; r++) {
        const int64_t mb = r / rk->oh;
        const int64_t oh = r - mb * rk->oh;
        const uint8_t *src_mb = src + (size_t)(mb * d->ic * in_ch) * cv->elt;
        float *dst_mb = dst + (size_t)mb * d->ic * out_ch;
        for (int64_t c = 0; c < d->ic; c++) {
            const uint8_t *src_c = src_mb + (size_t)c * in_ch * cv->elt;
            float *dst_row = dst_mb + (size_t)c * out_ch + (size_t)oh * rk->ow;
            for (int64_t ow = 0; ow < rk->ow; ow++)
                dst_row[ow] = pool_scalar(d, cv, src_c, oh, ow);
        }
    }
}

#if defined(__riscv_vector)

/* ------------------------------------------------------ RVV vector path */

/* The value of a window with no in-bounds sample, matching pool_scalar:
 * max keeps the seed, avg_include sums zeros and divides by KH*KW, and
 * avg_exclude divides 0 by 0 and yields NaN. */
static float pool_empty_value(rvcl_pooling_alg_t alg) {
    switch (alg) {
        case RVCL_POOL_MAX: return k_pool_max_seed;
        case RVCL_POOL_AVG_INCLUDE: return 0.0f;
        default: return std::numeric_limits<float>::quiet_NaN();
    }
}

/* True when the window at output column ow holds no in-bounds sample: its
 * first in-bounds column lies past the window's last one, which only
 * happens with a left pad >= the window width (kw <= pad_l). A vertically
 * empty window (pad >= kh) has no in-bounds row on any column and is
 * decided once per strip, not per lane. */
static bool pool_window_empty(const rvcl_pooling_desc_t *d, int64_t ow) {
    const int64_t iw0 = ow * d->sw - d->pad_l;
    const int64_t kw_lo = iw0 < 0 ? -iw0 : 0;
    const int64_t kw_hi = d->iw - iw0 < d->kw ? d->iw - iw0 : d->kw;
    return kw_hi <= kw_lo;
}

/* One window tap across the strip: lane i reads the sample at p + i*stride
 * and widens it to fp32. An f32 source loads directly; a narrow source
 * loads unsigned 16-bit and zero-extends it into the 32-bit lanes, which is
 * the low half of the IEEE bit pattern, and then shifts it into the high
 * half — every stored sample widens exactly. */
static vfloat32m4_t pool_load_f32(
        const uint8_t *p, ptrdiff_t stride, bool is_f32, long vl) {
    if (is_f32) return __riscv_vlse32_v_f32m4((const float *)p, stride, vl);
    const vuint16m2_t h
            = __riscv_vlse16_v_u16m2((const uint16_t *)p, stride, vl);
    const vuint32m4_t w
            = __riscv_vsll_vx_u32m4(__riscv_vzext_vf2_u32m4(h, vl), 16, vl);
    return __riscv_vreinterpret_f32m4(w);
}

/* Stores value into n consecutive fp32 outputs, one vfmv + vse per VL. */
static void pool_fill_strip(float *dst, int64_t n, float value) {
    for (int64_t i = 0; i < n;) {
        const long vl = __riscv_vsetvl_e32m4(n - i);
        __riscv_vse32_v_f32m4(dst + i, __riscv_vfmv_v_f_f32m4(value, vl), vl);
        i += (int64_t)vl;
    }
}

/* Computes the [ow0, ow0 + av) strip of one output row of one channel and
 * stores it. Every window tap is a stride-sw / stride-sh load from the
 * tap's first in-input column, so each tap is a contiguous run of lanes
 * with no active lanes past the input edge. The arithmetic mirrors
 * pool_scalar lane for lane: vfmax for max, an ordered vfadd chain for the
 * averages, one division at the end.
 *
 * Only the prefix of lanes whose whole window lies inside the input is
 * vectorized: the tap loads start at the strip's first clamped column and
 * run off the row for a lane whose window closes past the right edge, so
 * that trailing run (the last lane of a padded row, or any lane of a row
 * that does not tile evenly) takes the scalar loop. The same split covers
 * avg_exclude, whose divisor is the per-lane in-bounds count: a
 * partial-count lane never takes a vector rounding path. */
static void pool_strip_rvv(const rvcl_pooling_desc_t *d, const pool_conv_t *cv,
        const uint8_t *src_c, float *dst_row, int64_t oh, int64_t ow0,
        int64_t av) {
    const int64_t ih0 = oh * d->sh - d->pad_t;
    const int64_t kh_lo = ih0 < 0 ? -ih0 : 0;
    const int64_t kh_hi = d->ih - ih0 < d->kh ? d->ih - ih0 : d->kh;
    const int64_t iw0 = ow0 * d->sw - d->pad_l;
    const int64_t kw_lo = iw0 < 0 ? -iw0 : 0;
    const int64_t kw_hi = d->iw - iw0 < d->kw ? d->iw - iw0 : d->kw;
    const int64_t n_kh = kh_hi - kh_lo;
    const int64_t n_kw = kw_hi > kw_lo ? kw_hi - kw_lo : 0;

    /* A vertically empty strip (pad >= kh) has no in-bounds row, so every
     * lane is an empty window. */
    if (n_kh <= 0) {
        pool_fill_strip(dst_row + ow0, av, pool_empty_value(d->alg));
        return;
    }
    /* A horizontally empty lane (kw <= pad_l) has its window start before
     * column 0 with the window closing before it. The start moves left with
     * the lane index, so the empty lanes form a prefix of the strip; lanes
     * after it open inside the input. */
    int64_t empty = 0;
    if (n_kw <= 0)
        while (empty < av && pool_window_empty(d, ow0 + empty))
            empty++;
    if (empty == av) {
        pool_fill_strip(dst_row + ow0, av, pool_empty_value(d->alg));
        return;
    }
    /* With an empty prefix the lanes open at different in-input columns, so
     * the strip is not one strided run: take the scalar loop for all of it.
     * Rare geometry (pad_l >= kw), correctness over throughput. */
    if (empty > 0) {
        for (int64_t j = 0; j < av; j++)
            dst_row[ow0 + j] = pool_scalar(d, cv, src_c, oh, ow0 + j);
        return;
    }

    const int64_t c_first = iw0 + kw_lo;
    const int64_t span = (av - 1) * d->sw;
    /* The full-window prefix: lanes whose window lies inside the input on
     * every side, so every tap load stays on the row. The first lane's
     * window opens at c_first, which a non-zero pad left can put before the
     * input (kw_lo > 0); a window closing past the right edge ends the
     * prefix, so the last lane of a padded row or any lane of a row that
     * does not tile evenly falls out of it. avg_exclude also needs the full
     * KH*KW count vertically, which is why a vertically clamped strip has no
     * prefix at all. */
    int64_t vn = 0;
    if (kw_lo == 0 && n_kh == d->kh && c_first + span + d->kw <= d->iw) {
        const int64_t last = d->iw - d->kw - c_first;
        vn = last < 0 ? 0 : last / d->sw + 1;
        if (vn > av) vn = av;
    }
    if (d->alg == RVCL_POOL_AVG_EXCLUDE && kw_hi < d->kw) vn = 0;
    for (int64_t j = vn; j < av; j++)
        dst_row[ow0 + j] = pool_scalar(d, cv, src_c, oh, ow0 + j);
    av = vn;
    if (av <= 0) return;

    const ptrdiff_t col_stride = (ptrdiff_t)d->sw * (ptrdiff_t)cv->elt;
    const uint8_t *base
            = src_c + (size_t)((ih0 + kh_lo) * d->iw + c_first) * cv->elt;
    const bool is_f32 = cv->elt == sizeof(float);

    for (int64_t i = 0; i < av;) {
        /* One vsetvli per strip: the tap loads and the store below reuse the
         * same AVL, so no reselection is needed inside the window loop. */
        const long vl = __riscv_vsetvl_e32m4(av - i);
        /* Lane i is output column ow0+i, i.e. d->sw elements (sw*elt bytes)
         * past lane 0. */
        const uint8_t *p = base + (size_t)i * (size_t)d->sw * cv->elt;
        vfloat32m4_t acc = pool_load_f32(p, col_stride, is_f32, vl);
        for (int64_t kh = 0; kh < n_kh; kh++)
            for (int64_t kw = 0; kw < n_kw; kw++) {
                if (kh == 0 && kw == 0) continue; /* the seed */
                const uint8_t *q = p + (size_t)(kh * d->iw + kw) * cv->elt;
                const vfloat32m4_t v = pool_load_f32(q, col_stride, is_f32, vl);
                if (d->alg == RVCL_POOL_MAX) {
                    acc = __riscv_vfmax_vv_f32m4(acc, v, vl);
                } else {
                    acc = __riscv_vfadd_vv_f32m4(acc, v, vl);
                }
            }
        if (d->alg != RVCL_POOL_MAX) {
            const float denom = d->alg == RVCL_POOL_AVG_INCLUDE
                    ? (float)(d->kh * d->kw)
                    : (float)(n_kh * n_kw);
            acc = __riscv_vfdiv_vv_f32m4(
                    acc, __riscv_vfmv_v_f_f32m4(denom, vl), vl);
        }
        __riscv_vse32_v_f32m4(dst_row + ow0 + i, acc, vl);
        i += (int64_t)vl;
    }
}

static void pool_rows_rvv(rvv_pooling_t *rk, const uint8_t *src, float *dst,
        int64_t row_begin, int64_t row_end) {
    const rvcl_pooling_desc_t *d = &rk->d;
    const int64_t in_ch = d->ih * d->iw;
    const int64_t out_ch = rk->oh * rk->ow;

    for (int64_t r = row_begin; r < row_end; r++) {
        const int64_t mb = r / rk->oh;
        const int64_t oh = r - mb * rk->oh;
        const uint8_t *src_mb
                = src + (size_t)(mb * d->ic * in_ch) * rk->conv->elt;
        float *dst_mb = dst + (size_t)mb * d->ic * out_ch;
        for (int64_t c = 0; c < d->ic; c++) {
            const uint8_t *src_c = src_mb + (size_t)c * in_ch * rk->conv->elt;
            float *dst_row = dst_mb + (size_t)c * out_ch + (size_t)oh * rk->ow;
            /* The strip takes the whole row; each strip selects its own
             * AVL, so a row of any width is one or more strips. */
            pool_strip_rvv(d, rk->conv, src_c, dst_row, oh, 0, rk->ow);
        }
    }
}

#endif /* __riscv_vector */

/* ------------------------------------------------------------- lifecycle */

/* Every kernel instance serves exactly one (A,B,acc,C) tuple and the output
 * is fp32 (the dtype's accumulation width), so the tuple is part of the
 * instance's contract: a request for another tuple is rejected here rather
 * than computed through a mismatched conversion. */
template <rvcl_dtype_t DT>
static rvcl_status_t pooling_validate_tuple(const rvcl_op_desc_t *desc) {
    const rvcl_pooling_desc_t *d = &desc->pooling;
    /* The geometry is part of the instance's contract too: configure()
     * divides by the stride, so a zero stride must not reach it. */
    if (!rvcl_pooling_geometry_valid(d)) return RVCL_ERROR_INVALID_ARGUMENT;
    /* B is the request-side "unset" marker: pooling has one operand. */
    if (d->dtype_a != DT || d->dtype_b != RVCL_DTYPE_AUTO
            || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* AUTO is the request-side "derive it" marker for the accumulator. */
    if (d->dtype_acc != RVCL_DTYPE_AUTO && d->dtype_acc != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Pooling has no bias term: AUTO is the only accepted value. */
    if (d->dtype_bias != RVCL_DTYPE_AUTO) return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Quantization is not served by the pooling path yet: reject the
     * request here rather than compute it as if it were unquantized. */
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    return RVCL_SUCCESS;
}

template <rvcl_dtype_t DT>
static rvcl_status_t pooling_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap; /* the row-wise vector path is plain RVV, gated by features */
    return pooling_validate_tuple<DT>(desc);
}

static rvcl_status_t pooling_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    rvv_pooling_t *rk = (rvv_pooling_t *)k;
    const rvcl_pooling_desc_t *d = &desc->pooling;
    if (!rvcl_pooling_geometry_valid(d)) return RVCL_ERROR_INVALID_ARGUMENT;
    rk->d = *d;
    rk->oh = rvcl_pooling_out_dim(d->ih, d->kh, d->sh, d->pad_t, d->pad_b);
    rk->ow = rvcl_pooling_out_dim(d->iw, d->kw, d->sw, d->pad_l, d->pad_r);
    if (rk->oh <= 0 || rk->ow <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    /* No packing, no workspace: run() reads the source and writes the
     * destination directly. */
    if (ws_req) *ws_req = rvcl_workspace_req_t {};
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t pooling_run(rvcl_kernel_t *k,
        const rvcl_tensor_pack_t *pack, const rvcl_work_range_t *range) {
    rvv_pooling_t *rk = (rvv_pooling_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    if (!a || !c || !a->data || !c->data) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!range) return RVCL_ERROR_INVALID_ARGUMENT;
    const int64_t total = rk->d.mb * rk->oh;
    /* Full n and k only: a work row is a whole output line and there is no
     * reduction axis to split. */
    if (range->m_begin < 0 || range->m_end > total
            || range->m_begin > range->m_end || range->n_begin != 0
            || range->n_end != rk->ow || range->k_begin != 0
            || range->k_end != 1)
        return RVCL_ERROR_INVALID_ARGUMENT;

    rk->rows(rk, (const uint8_t *)a->data, (float *)c->data, range->m_begin,
            range->m_end);
    return RVCL_SUCCESS;
}

static void pooling_destroy(rvcl_kernel_t *k) {
    (void)k;
}

/* One vtable per dtype: validate() carries the instance's tuple. */
template <rvcl_dtype_t DT>
static const rvcl_kernel_vtable_t *pooling_vtable(void) {
    static const rvcl_kernel_vtable_t vt = {
            .validate = pooling_validate<DT>,
            .configure = pooling_configure,
            .pack_a = NULL, /* no packing: one input tensor */
            .pack_b = NULL,
            .run = pooling_run,
            .destroy = pooling_destroy,
    };
    return &vt;
}

/* --------------------------------------------------- per-dtype factories */

static rvcl_status_t make_pooling_kernel(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, const rvcl_kernel_vtable_t *vt,
        const pool_conv_t *conv,
        void (*rows)(
                rvv_pooling_t *, const uint8_t *, float *, int64_t, int64_t)) {
    const rvcl_status_t st
            = rvcl_kernel_alloc(out, desc, sizeof(rvv_pooling_t));
    if (st != RVCL_SUCCESS) return st;
    rvv_pooling_t *rk = (rvv_pooling_t *)*out;
    rk->base.vt = vt;
    rk->conv = conv;
    rk->rows = rows;
    return RVCL_SUCCESS;
}

template <rvcl_dtype_t DT>
static rvcl_status_t create_pooling(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
#if defined(__riscv_vector)
    return make_pooling_kernel(
            out, desc, pooling_vtable<DT>(), pool_conv<DT>(), pool_rows_rvv);
#else
    return make_pooling_kernel(
            out, desc, pooling_vtable<DT>(), pool_conv<DT>(), pool_rows_scalar);
#endif
    (void)pool_rows_scalar; /* the vector build keeps it as a symbol */
}

static rvcl_status_t create_pooling_f32(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return create_pooling<RVCL_DTYPE_FP32>(out, desc);
}

static rvcl_status_t create_pooling_f16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return create_pooling<RVCL_DTYPE_FP16>(out, desc);
}

static rvcl_status_t create_pooling_bf16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    return create_pooling<RVCL_DTYPE_BF16>(out, desc);
}

/* ------------------------------------------------------- registration */

/* Registration tuple per dtype: (A,B) are the storage dtype, acc and C are
 * fp32 — the wide-output habit that keeps up-conversion exact and the result
 * bit-comparable with the golden. */
static rvcl_status_t pooling_register_one(const char *name, rvcl_dtype_t dt,
        rvcl_feature_id_t f0, rvcl_feature_id_t f1,
        rvcl_kernel_create_fn create) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_POOLING;
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
    /* ANY on purpose: the row-wise vector path is plain RVV, so the tier
     * split carries no information (one code serves every VLEN). */
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = rvcl_cost_pooling_rvv;

    return rvcl_register_kernel(&d, create);
}

#if defined(__riscv_vector)

RVCL_API rvcl_status_t rvcl_rvv_register_pooling(void) {
    rvcl_status_t st
            = pooling_register_one("rvcl_pooling_f32_rvv", RVCL_DTYPE_FP32,
                    RVCL_FEATURE_RVV, RVCL_FEATURE_NONE, create_pooling_f32);
    if (st != RVCL_SUCCESS) return st;
    st = pooling_register_one("rvcl_pooling_f16_rvv", RVCL_DTYPE_FP16,
            RVCL_FEATURE_RVV, RVCL_FEATURE_NONE, create_pooling_f16);
    if (st != RVCL_SUCCESS) return st;
    /* bf16 up-converts by integer widen+shift, no Zvfbfmin needed. */
    return pooling_register_one("rvcl_pooling_bf16_rvv", RVCL_DTYPE_BF16,
            RVCL_FEATURE_RVV, RVCL_FEATURE_NONE, create_pooling_bf16);
}

RVCL_API rvcl_status_t rvcl_rvv_register_pooling_portable(void) {
    return RVCL_SUCCESS; /* vector build: portable entries stay out */
}

#else /* !__riscv_vector */

RVCL_API rvcl_status_t rvcl_rvv_register_pooling(void) {
    return RVCL_SUCCESS; /* non-vector build: generic entries stay out */
}

/* Portable scalar entries: no feature requirement (serve any host). */
RVCL_API rvcl_status_t rvcl_rvv_register_pooling_portable(void) {
    rvcl_status_t st
            = pooling_register_one("rvcl_pooling_f32_rvv_c", RVCL_DTYPE_FP32,
                    RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_pooling_f32);
    if (st != RVCL_SUCCESS) return st;
    st = pooling_register_one("rvcl_pooling_f16_rvv_c", RVCL_DTYPE_FP16,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_pooling_f16);
    if (st != RVCL_SUCCESS) return st;
    return pooling_register_one("rvcl_pooling_bf16_rvv_c", RVCL_DTYPE_BF16,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE, create_pooling_bf16);
}

#endif /* __riscv_vector */
