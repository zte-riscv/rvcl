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

/* Test-side operator slot table (see op_slots.hpp for the contract). The
 * matmul row is the first table entry (migrated from the differential and
 * bench mains); the gemv and conv2d rows follow, each carrying its own
 * descriptor builder, tensor layouts and work range. The shared run rig
 * below never looks at the operation enum: everything operator-specific
 * comes from the row, so adding an operator is adding a row. */

#include "op_slots.hpp"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/common/conv2d_geometry.h"
#include "../src/common/pooling_geometry.h"
#include "../src/runtime/dispatch_internal.h"
#include "../src/backends/rvv/rvv_bf16.hpp"
#include "../src/backends/rvv/rvv_fp16.hpp"
#include "dtype_ops.hpp"

/* ------------------------------------------------------- element counts */

static int64_t matmul_a_elems(const rvcl_op_desc_t *od) {
    return od->matmul.m * od->matmul.k;
}
static int64_t matmul_b_elems(const rvcl_op_desc_t *od) {
    return od->matmul.k * od->matmul.n;
}
static int64_t matmul_bias_elems(const rvcl_op_desc_t *od) {
    return od->matmul.n;
}
static int64_t matmul_out_elems(const rvcl_op_desc_t *od) {
    return od->matmul.m * od->matmul.n;
}
static int64_t gemv_a_elems(const rvcl_op_desc_t *od) {
    return od->gemv.m * od->gemv.k;
}
static int64_t gemv_b_elems(const rvcl_op_desc_t *od) {
    return od->gemv.k;
}
static int64_t gemv_bias_elems(const rvcl_op_desc_t *od) {
    return od->gemv.m;
}
static int64_t gemv_out_elems(const rvcl_op_desc_t *od) {
    return od->gemv.m;
}
static int64_t ip_a_elems(const rvcl_op_desc_t *od) {
    return od->inner_product.mb * od->inner_product.k;
}
static int64_t ip_b_elems(const rvcl_op_desc_t *od) {
    return od->inner_product.oc * od->inner_product.k;
}
static int64_t ip_bias_elems(const rvcl_op_desc_t *od) {
    return od->inner_product.oc;
}
static int64_t ip_out_elems(const rvcl_op_desc_t *od) {
    return od->inner_product.mb * od->inner_product.oc;
}

/* Conv2d output geometry from the descriptor's pad/stride/dilation
 * formula, shared by the row's elems/golden/layout helpers. The formula
 * itself is the library's single copy (src/common/conv2d_geometry.h) so the
 * corpus-vs-params cross-check cannot drift from what the kernel computes. */
static int64_t conv_oh(const rvcl_conv2d_desc_t *d) {
    return rvcl_conv2d_out_dim(
            d->ih, d->kh, d->sh, d->dil_h, d->pad_t, d->pad_b);
}

static int64_t conv_ow(const rvcl_conv2d_desc_t *d) {
    return rvcl_conv2d_out_dim(
            d->iw, d->kw, d->sw, d->dil_w, d->pad_l, d->pad_r);
}

static int64_t conv_a_elems(const rvcl_op_desc_t *od) {
    const rvcl_conv2d_desc_t *d = &od->conv2d;
    return d->mb * d->ic * d->ih * d->iw;
}
static int64_t conv_b_elems(const rvcl_op_desc_t *od) {
    const rvcl_conv2d_desc_t *d = &od->conv2d;
    return d->oc * d->ic * d->kh * d->kw;
}
static int64_t conv_bias_elems(const rvcl_op_desc_t *od) {
    return od->conv2d.oc;
}
static int64_t conv_out_elems(const rvcl_op_desc_t *od) {
    const rvcl_conv2d_desc_t *d = &od->conv2d;
    return d->mb * d->oc * conv_oh(d) * conv_ow(d);
}

/* Pooling output geometry: the library's single formula copy, so the
 * corpus-vs-params cross-check cannot drift from what the kernel computes. */
static int64_t pool_oh(const rvcl_pooling_desc_t *d) {
    return rvcl_pooling_out_dim(d->ih, d->kh, d->sh, d->pad_t, d->pad_b);
}

static int64_t pool_ow(const rvcl_pooling_desc_t *d) {
    return rvcl_pooling_out_dim(d->iw, d->kw, d->sw, d->pad_l, d->pad_r);
}

static int64_t pool_a_elems(const rvcl_op_desc_t *od) {
    const rvcl_pooling_desc_t *d = &od->pooling;
    return d->mb * d->ic * d->ih * d->iw;
}
/* No B operand and no bias: zero-length tensors, which the differential
 * rig allocates as empty buffers and the row leaves out of the pack. */
static int64_t pool_none_elems(const rvcl_op_desc_t *od) {
    (void)od;
    return 0;
}
static int64_t pool_out_elems(const rvcl_op_desc_t *od) {
    const rvcl_pooling_desc_t *d = &od->pooling;
    return d->mb * d->ic * pool_oh(d) * pool_ow(d);
}

static int64_t ln_a_elems(const rvcl_op_desc_t *od) {
    return od->layer_norm.rows * od->layer_norm.norm_size;
}
/* No B operand. The optional scale and shift vectors both ride the bias
 * buffer as its two norm_size halves, so the rig's existing fp32 parameter
 * buffer carries them without a layer_norm-specific allocation. */
static int64_t ln_param_elems(const rvcl_op_desc_t *od) {
    return od->layer_norm.norm_size * 2;
}
static int64_t ln_out_elems(const rvcl_op_desc_t *od) {
    return od->layer_norm.rows * od->layer_norm.norm_size;
}

/* ---------------------------------------------------------------- golden */

static void matmul_golden(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c) {
    const dtype_ops_t *op = dtype_ops_for(od->matmul.dtype_a);
    const int64_t blk = (int64_t)rvcl_dtype_block_size(od->matmul.dtype_a);
    op->golden(a, sa, b, sb, bias, c, od->matmul.m, od->matmul.n, od->matmul.k,
            blk);
}

/* Naive GEMV: y[i] = bias[i] + sum_p a[i][p] * x[p]. sa/sb are NULL (gemv
 * serves fp32 only). */
static void gemv_golden(const rvcl_op_desc_t *od, const void *a, const void *sa,
        const void *b, const void *sb, const float *bias, float *c) {
    (void)sa;
    (void)sb;
    const int64_t m = od->gemv.m, k = od->gemv.k;
    const float *pa = (const float *)a;
    const float *px = (const float *)b;
    for (int64_t i = 0; i < m; i++) {
        float acc = bias ? bias[i] : 0.0f;
        for (int64_t p = 0; p < k; p++)
            acc += pa[i * k + p] * px[p];
        c[i] = acc;
    }
}

/* The convolution golden re-derives the kernel's two inputs in plain test
 * code — the im2col patch matrix and the transposed weight — and then runs
 * the shared matmul golden on them. That keeps the numerics on the already
 * bit-exact matmul contract (bias-first seed, ascending k accumulation,
 * single-rounding fused multiply-add) instead of re-deriving the same sum
 * with a different rounding pattern, and it makes the im2col index math,
 * the weight transpose and the NCHW writeback each an independently written
 * reference rather than a mirror of the kernel. */

/* Patch matrix [mb*OH*OW][ic*kh*kw] of raw storage elements, out-of-window
 * samples zero-filled. */
static void conv2d_im2col_ref(const rvcl_conv2d_desc_t *d, const void *a,
        size_t elt, uint8_t *patches) {
    const int64_t oh = conv_oh(d), ow = conv_ow(d);
    const int64_t k = d->ic * d->kh * d->kw;
    const uint8_t *src = (const uint8_t *)a;
    for (int64_t mb = 0; mb < d->mb; mb++)
        for (int64_t y = 0; y < oh; y++)
            for (int64_t x = 0; x < ow; x++) {
                uint8_t *row = patches
                        + (size_t)((mb * oh + y) * ow + x) * (size_t)k * elt;
                int64_t p = 0;
                for (int64_t ic = 0; ic < d->ic; ic++)
                    for (int64_t kh = 0; kh < d->kh; kh++) {
                        const int64_t ih = y * d->sh - d->pad_t + kh * d->dil_h;
                        for (int64_t kw = 0; kw < d->kw; kw++, p++) {
                            const int64_t iw
                                    = x * d->sw - d->pad_l + kw * d->dil_w;
                            uint8_t *out = row + (size_t)p * elt;
                            if (ih >= 0 && ih < d->ih && iw >= 0
                                    && iw < d->iw) {
                                const int64_t si
                                        = ((mb * d->ic + ic) * d->ih + ih)
                                                * d->iw
                                        + iw;
                                memcpy(out, src + (size_t)si * elt, elt);
                            } else {
                                memset(out, 0, elt);
                            }
                        }
                    }
            }
}

/* w^T [ic*kh*kw][oc] from the [oc][ic*kh*kw] weights. */
static void conv2d_wt_ref(
        const rvcl_conv2d_desc_t *d, const void *b, size_t elt, uint8_t *wt) {
    const int64_t k = d->ic * d->kh * d->kw, n = d->oc;
    const uint8_t *w = (const uint8_t *)b;
    for (int64_t j = 0; j < n; j++)
        for (int64_t p = 0; p < k; p++)
            memcpy(wt + (size_t)(p * n + j) * elt,
                    w + (size_t)(j * k + p) * elt, elt);
}

static void conv2d_golden(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c) {
    (void)sa;
    (void)sb;
    const rvcl_conv2d_desc_t *d = &od->conv2d;
    const int64_t oh = conv_oh(d), ow = conv_ow(d), n = d->oc;
    const int64_t k = d->ic * d->kh * d->kw;
    const int64_t m = d->mb * oh * ow;
    const size_t elt = rvcl_dtype_size(d->dtype_a);
    /* A degenerate geometry (an output window that collapses to zero) has
     * nothing to compare; bailing out keeps the allocations below non-empty
     * and leaves the destination untouched. */
    if (m <= 0 || k <= 0 || n <= 0) return;

    uint8_t *patches = (uint8_t *)malloc((size_t)(m * k) * elt);
    uint8_t *wt = (uint8_t *)malloc((size_t)(k * n) * elt);
    float *cmm = (float *)malloc((size_t)(m * n) * sizeof(float));
    if (!patches || !wt || !cmm) {
        free(patches);
        free(wt);
        free(cmm);
        return;
    }
    conv2d_im2col_ref(d, a, elt, patches);
    conv2d_wt_ref(d, b, elt, wt);

    if (d->dtype_a == RVCL_DTYPE_FP32)
        golden_f32(
                (const float *)patches, (const float *)wt, bias, cmm, m, n, k);
    else if (d->dtype_a == RVCL_DTYPE_FP16)
        golden_f16((const uint16_t *)patches, (const uint16_t *)wt, bias, cmm,
                m, n, k);
    else
        golden_bf16((const uint16_t *)patches, (const uint16_t *)wt, bias, cmm,
                m, n, k);

    /* [m][n] -> NCHW [mb][oc][oh][ow]. */
    for (int64_t mb = 0; mb < d->mb; mb++)
        for (int64_t oc = 0; oc < n; oc++)
            for (int64_t r = 0; r < oh * ow; r++)
                c[(mb * n + oc) * oh * ow + r]
                        = cmm[(mb * oh * ow + r) * n + oc];

    free(patches);
    free(wt);
    free(cmm);
}

/* Inner product golden: the weights are stored [oc][k] (B transposed), so
 * transpose them into the shared matmul golden's [k][oc] B operand and run
 * the already bit-exact matmul contract on the untouched [mb][k] source.
 * The weight transpose is an independently written reference, and the
 * numerics stay on the bias-first, ascending-k, single-rounding path. */
static void inner_product_golden(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c) {
    (void)sa;
    (void)sb;
    const rvcl_inner_product_desc_t *d = &od->inner_product;
    const int64_t m = d->mb, n = d->oc, k = d->k;
    const size_t elt = rvcl_dtype_size(d->dtype_a);
    if (m <= 0 || k <= 0 || n <= 0) return;

    uint8_t *wt = (uint8_t *)malloc((size_t)(k * n) * elt);
    if (!wt) return;
    const uint8_t *w = (const uint8_t *)b;
    for (int64_t j = 0; j < n; j++)
        for (int64_t p = 0; p < k; p++)
            memcpy(wt + (size_t)(p * n + j) * elt,
                    w + (size_t)(j * k + p) * elt, elt);

    if (d->dtype_a == RVCL_DTYPE_FP32)
        golden_f32((const float *)a, (const float *)wt, bias, c, m, n, k);
    else if (d->dtype_a == RVCL_DTYPE_FP16)
        golden_f16((const uint16_t *)a, (const uint16_t *)wt, bias, c, m, n, k);
    else
        golden_bf16(
                (const uint16_t *)a, (const uint16_t *)wt, bias, c, m, n, k);
    free(wt);
}

/* One stored sample as fp32: the narrow formats up-convert exactly, so the
 * golden sees the same values the kernel's conversion produces. */
static float pool_sample_f32(const dtype_ops_t *op, const uint8_t *p) {
    if (op->dtype == RVCL_DTYPE_FP16) {
        uint16_t h;
        memcpy(&h, p, sizeof(h));
        return rvv_f16_to_f32(h);
    }
    if (op->dtype == RVCL_DTYPE_BF16) {
        uint16_t h;
        memcpy(&h, p, sizeof(h));
        return rvv_bf16_to_f32(h);
    }
    float v;
    memcpy(&v, p, sizeof(v));
    return v;
}

/* Pooling golden: an independent walk of the window in (kh, kw) order.
 * max keeps the largest in-bounds sample, or -FLT_MAX when the window holds
 * none (oneDNN seeds max from lowest_value of the dst type, and the dst here
 * is always fp32); avg_include sums every in-bounds sample and divides by
 * KH*KW, so a pad sample counts as zero; avg_exclude divides by the
 * in-bounds count, so an empty window yields 0/0 = NaN. The accumulation is
 * fp32 and the division happens once at the end, so the result is
 * bit-comparable with the kernel. */
static void pooling_golden(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c) {
    (void)sa;
    (void)b;
    (void)sb;
    (void)bias;
    const rvcl_pooling_desc_t *d = &od->pooling;
    /* The input elements are read by the storage dtype's size; the output
     * buffer is always the fp32 view the rig allocates. */
    const dtype_ops_t *op = dtype_ops_for(d->dtype_a);
    const int64_t oh = pool_oh(d), ow = pool_ow(d);
    const int64_t in_ch = d->ih * d->iw, out_ch = oh * ow;
    const uint8_t *src = (const uint8_t *)a;
    if (oh <= 0 || ow <= 0) return;

    for (int64_t mb = 0; mb < d->mb; mb++)
        for (int64_t ic = 0; ic < d->ic; ic++) {
            const uint8_t *src_c
                    = src + (size_t)(mb * d->ic + ic) * (size_t)in_ch * op->elt;
            float *dst_c = c + (size_t)(mb * d->ic + ic) * (size_t)out_ch;
            for (int64_t y = 0; y < oh; y++)
                for (int64_t x = 0; x < ow; x++) {
                    float best = -3.4028234663852886e38f; /* -FLT_MAX */
                    float sum = 0.0f;
                    int64_t count = 0;
                    for (int64_t kh = 0; kh < d->kh; kh++) {
                        const int64_t ih = y * d->sh - d->pad_t + kh;
                        if (ih < 0 || ih >= d->ih) continue;
                        for (int64_t kw = 0; kw < d->kw; kw++) {
                            const int64_t iw = x * d->sw - d->pad_l + kw;
                            if (iw < 0 || iw >= d->iw) continue;
                            const float v = pool_sample_f32(op,
                                    src_c
                                            + (size_t)(ih * d->iw + iw)
                                                    * op->elt);
                            if (count == 0 || v > best) best = v;
                            sum += v;
                            count++;
                        }
                    }
                    float out;
                    if (d->alg == RVCL_POOL_MAX)
                        out = best;
                    else if (d->alg == RVCL_POOL_AVG_INCLUDE)
                        out = sum / (float)(d->kh * d->kw);
                    else /* empty window: NaN, kept out of corpus because the
                          * shared comparators treat NaN != NaN as a mismatch */
                        out = sum / (float)count;
                    dst_c[y * ow + x] = out;
                }
        }
}

/* Layer norm golden: an independent three-pass walk of the row in ascending
 * c order. mean and the biased variance accumulate left to right and divide
 * once; the squared deviation is formed without fusion (fma(d, d, +0) is the
 * single-rounded product); the write-back divides scale by sqrt(var + eps)
 * per element (never a reciprocal times the value) and updates with one
 * fused multiply-add. sqrt goes through libm sqrtf, so the golden is the
 * deterministic scalar reference the kernel is compared against. The rig's
 * fp32 parameter buffer carries the two optional vectors as two norm_size
 * halves: scale first, then shift. */
static void layer_norm_golden(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c) {
    (void)sa;
    (void)sb;
    (void)b;
    const rvcl_layer_norm_desc_t *d = &od->layer_norm;
    const dtype_ops_t *op = dtype_ops_for(d->dtype_a);
    const int64_t n = d->norm_size;
    const uint8_t *src = (const uint8_t *)a;
    if (n <= 0 || d->rows <= 0) return;
    /* The two vectors are read only when the case asks for them and the
     * buffer that carries them exists, which is exactly the pair of slots
     * pack_setup binds: a flag without a buffer imports nothing and the
     * kernel substitutes the constant, so the golden must do the same. A
     * NULL here is therefore the "constant" case, not a missing operand. */
    const float *scale = (d->use_scale && bias) ? bias : NULL;
    const float *shift = (d->use_shift && bias) ? bias + n : NULL;

    for (int64_t r = 0; r < d->rows; r++) {
        const uint8_t *row = src + (size_t)r * (size_t)n * op->elt;
        float *out = c + (size_t)r * (size_t)n;

        float sum = 0.0f;
        for (int64_t i = 0; i < n; i++)
            sum += pool_sample_f32(op, row + (size_t)i * op->elt);
        const float mean = sum / (float)n;

        float acc = 0.0f;
        for (int64_t i = 0; i < n; i++) {
            const float dev
                    = pool_sample_f32(op, row + (size_t)i * op->elt) - mean;
            acc += fmaf(dev, dev, 0.0f);
        }
        const float var = acc / (float)n;
        const float sqrt_var = sqrtf(var + d->eps);

        for (int64_t i = 0; i < n; i++) {
            const float sm = scale ? scale[i] / sqrt_var : 1.0f / sqrt_var;
            const float sh = shift ? shift[i] : 0.0f;
            const float dev
                    = pool_sample_f32(op, row + (size_t)i * op->elt) - mean;
            out[i] = fmaf(sm, dev, sh);
        }
    }
}

/* ------------------------------------------------------------ build desc */

static rvcl_status_t matmul_build_desc(rvcl_op_desc_t *od, int64_t m, int64_t n,
        int64_t k, const char *params, rvcl_dtype_t da, rvcl_dtype_t db,
        rvcl_dtype_t dacc, rvcl_dtype_t dc, rvcl_dtype_t dbias) {
    /* The matmul GEMM view is fully described by the corpus triple; a
     * params column on a matmul case is a corpus error. */
    if (params && *params) return RVCL_ERROR_INVALID_ARGUMENT;
    od->matmul = rvcl_matmul_desc_t {.m = m,
            .n = n,
            .k = k,
            .dtype_a = da,
            .dtype_b = db,
            .dtype_acc = dacc,
            .dtype_c = dc,
            .dtype_bias = dbias,
            .quant = {}};
    return RVCL_SUCCESS;
}

static rvcl_status_t gemv_build_desc(rvcl_op_desc_t *od, int64_t m, int64_t n,
        int64_t k, const char *params, rvcl_dtype_t da, rvcl_dtype_t db,
        rvcl_dtype_t dacc, rvcl_dtype_t dc, rvcl_dtype_t dbias) {
    (void)n; /* a gemv has no n dimension */
    if (params && *params) return RVCL_ERROR_INVALID_ARGUMENT;
    od->gemv = rvcl_gemv_desc_t {.m = m,
            .k = k,
            .dtype_a = da,
            .dtype_b = db,
            .dtype_acc = dacc,
            .dtype_c = dc,
            .dtype_bias = dbias,
            .quant = {}};
    return RVCL_SUCCESS;
}

/* conv2d params lexer: a fixed-order run of <key><decimal> fields, keys
 * "mb","ic","ih","iw","oc","kh","kw" (required) followed by the optional
 * "sh","sw","dh","dw","pt","pb","pl","pr" (defaults 1/1/1/1/0/0/0/0). A
 * key out of order, repeated, unknown or missing is an error. */
typedef struct {
    const char *key;
    int64_t *dst;
    int required;
} conv_param_field_t;

static int64_t conv_param_value(const char **pp, const char *key, int *ok) {
    const char *p = *pp;
    if (strncmp(p, key, 2) != 0) {
        *ok = 0;
        return 0;
    }
    p += 2;
    if (*p < '0' || *p > '9') {
        *ok = 0;
        return 0;
    }
    int64_t v = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        if (v > 1000000000) {
            *ok = 0;
            return 0;
        }
        p++;
    }
    *pp = p;
    return v;
}

static rvcl_status_t conv2d_build_desc(rvcl_op_desc_t *od, int64_t m, int64_t n,
        int64_t k, const char *params, rvcl_dtype_t da, rvcl_dtype_t db,
        rvcl_dtype_t dacc, rvcl_dtype_t dc, rvcl_dtype_t dbias) {
    rvcl_conv2d_desc_t d = {};
    d.sh = d.sw = d.dil_h = d.dil_w = 1;
    d.groups = 1;
    d.dtype_a = da;
    d.dtype_b = db;
    d.dtype_acc = dacc;
    d.dtype_c = dc;
    d.dtype_bias = dbias;

    const conv_param_field_t fields[] = {
            {"mb", &d.mb, 1},
            {"ic", &d.ic, 1},
            {"ih", &d.ih, 1},
            {"iw", &d.iw, 1},
            {"oc", &d.oc, 1},
            {"kh", &d.kh, 1},
            {"kw", &d.kw, 1},
            {"sh", &d.sh, 0},
            {"sw", &d.sw, 0},
            {"dh", &d.dil_h, 0},
            {"dw", &d.dil_w, 0},
            {"pt", &d.pad_t, 0},
            {"pb", &d.pad_b, 0},
            {"pl", &d.pad_l, 0},
            {"pr", &d.pad_r, 0},
    };
    const int nfields = (int)(sizeof(fields) / sizeof(fields[0]));
    const char *p = params ? params : "";
    int next = 0;
    while (*p) {
        int matched = -1;
        for (int i = next; i < nfields; i++)
            if (!strncmp(p, fields[i].key, 2)) {
                matched = i;
                break;
            }
        if (matched < 0) return RVCL_ERROR_INVALID_ARGUMENT;
        int ok = 1;
        const int64_t v = conv_param_value(&p, fields[matched].key, &ok);
        if (!ok) return RVCL_ERROR_INVALID_ARGUMENT;
        *fields[matched].dst = v;
        next = matched + 1;
    }
    for (int i = 0; i < nfields; i++)
        if (fields[i].required && *fields[i].dst == 0)
            return RVCL_ERROR_INVALID_ARGUMENT;

    /* Reject a geometry the output-size formula cannot evaluate: a zero
     * stride or dilation divides by zero, a zero extent has no output, and
     * a negative pad has no meaning. The check is the library's single copy
     * (rvcl_conv2d_geometry_valid), shared with the dispatch and kernel
     * validators. These are corpus errors like any other malformed params
     * column, so they surface as a parse error rather than a crash. */
    if (!rvcl_conv2d_geometry_valid(&d)) return RVCL_ERROR_INVALID_ARGUMENT;

    /* Cross-check the params geometry against the shape column: the corpus
     * triple is the case's GEMM view (m = mb*OH*OW, n = oc, k = ic*kh*kw),
     * so a mismatch between the two columns is a corpus typo. */
    const int64_t oh = conv_oh(&d), ow = conv_ow(&d);
    if (oh <= 0 || ow <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    if (d.mb * oh * ow != m || d.oc != n || d.ic * d.kh * d.kw != k)
        return RVCL_ERROR_INVALID_ARGUMENT;

    od->conv2d = d;
    return RVCL_SUCCESS;
}

/* Pooling params lexer: an optional leading algorithm name ("max",
 * "avg" = avg_include, "avgex" = avg_exclude; default max) followed by a
 * fixed-order run of <key><decimal> fields, keys "mb","ic","ih","iw",
 * "kh","kw" (required) and the optional "sh","sw","pt","pb","pl","pr"
 * (defaults 1/1/0/0/0/0). A key out of order, repeated, unknown or missing
 * is an error. */
typedef struct {
    const char *key;
    int64_t *dst;
    int required;
} pool_param_field_t;

static int64_t pool_param_value(const char **pp, const char *key, int *ok) {
    const char *p = *pp;
    const size_t klen = strlen(key);
    if (strncmp(p, key, klen) != 0) {
        *ok = 0;
        return 0;
    }
    p += klen;
    if (*p < '0' || *p > '9') {
        *ok = 0;
        return 0;
    }
    int64_t v = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        if (v > 1000000000) {
            *ok = 0;
            return 0;
        }
        p++;
    }
    *pp = p;
    return v;
}

static rvcl_status_t pooling_build_desc(rvcl_op_desc_t *od, int64_t m,
        int64_t n, int64_t k, const char *params, rvcl_dtype_t da,
        rvcl_dtype_t db, rvcl_dtype_t dacc, rvcl_dtype_t dc,
        rvcl_dtype_t dbias) {
    rvcl_pooling_desc_t d = {};
    d.alg = RVCL_POOL_MAX;
    d.sh = d.sw = 1;
    d.dtype_a = da;
    /* No B operand and no bias: the request-side "unset" marker, which is
     * also what the kernel's tuple check accepts. */
    d.dtype_b = RVCL_DTYPE_AUTO;
    d.dtype_acc = dacc;
    d.dtype_c = dc;
    d.dtype_bias = RVCL_DTYPE_AUTO;
    (void)db;
    (void)dbias;

    const char *p = params ? params : "";
    if (!strncmp(p, "avgex", 5)) {
        d.alg = RVCL_POOL_AVG_EXCLUDE;
        p += 5;
    } else if (!strncmp(p, "avg", 3)) {
        d.alg = RVCL_POOL_AVG_INCLUDE;
        p += 3;
    } else if (!strncmp(p, "max", 3)) {
        p += 3;
    }

    const pool_param_field_t fields[] = {
            {"mb", &d.mb, 1},
            {"ic", &d.ic, 1},
            {"ih", &d.ih, 1},
            {"iw", &d.iw, 1},
            {"kh", &d.kh, 1},
            {"kw", &d.kw, 1},
            {"sh", &d.sh, 0},
            {"sw", &d.sw, 0},
            {"pt", &d.pad_t, 0},
            {"pb", &d.pad_b, 0},
            {"pl", &d.pad_l, 0},
            {"pr", &d.pad_r, 0},
    };
    const int nfields = (int)(sizeof(fields) / sizeof(fields[0]));
    int next = 0;
    while (*p) {
        int matched = -1;
        for (int i = next; i < nfields; i++)
            if (!strncmp(p, fields[i].key, 2)) {
                matched = i;
                break;
            }
        if (matched < 0) return RVCL_ERROR_INVALID_ARGUMENT;
        int ok = 1;
        const int64_t v = pool_param_value(&p, fields[matched].key, &ok);
        if (!ok) return RVCL_ERROR_INVALID_ARGUMENT;
        *fields[matched].dst = v;
        next = matched + 1;
    }
    for (int i = 0; i < nfields; i++)
        if (fields[i].required && *fields[i].dst == 0)
            return RVCL_ERROR_INVALID_ARGUMENT;

    /* The same geometry check the dispatch and kernel validators run. */
    if (!rvcl_pooling_geometry_valid(&d)) return RVCL_ERROR_INVALID_ARGUMENT;

    /* Cross-check the params geometry against the shape column: the corpus
     * triple is the case's view (m = mb*OH, n = OW, k = 1), so a mismatch
     * between the two columns is a corpus typo. */
    const int64_t oh = pool_oh(&d), ow = pool_ow(&d);
    if (oh <= 0 || ow <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    if (d.mb * oh != m || ow != n || k != 1) return RVCL_ERROR_INVALID_ARGUMENT;

    od->pooling = d;
    return RVCL_SUCCESS;
}

/* Layer norm params lexer: a fixed-order run of <key><decimal> fields, keys
 * "rows","ns" (required) then the optional "e" (eps x 1e9, an integer so the
 * value fits the column's [a-z0-9:] alphabet), "sc" and "sh" (1/0 flags for
 * use_scale / use_shift; defaults 0). A key out of order, repeated, unknown
 * or missing is an error. eps is decoded as e * 1e-9f, so e0 is 0 and e10000
 * is 1e-5. */
static rvcl_status_t layer_norm_build_desc(rvcl_op_desc_t *od, int64_t m,
        int64_t n, int64_t k, const char *params, rvcl_dtype_t da,
        rvcl_dtype_t db, rvcl_dtype_t dacc, rvcl_dtype_t dc,
        rvcl_dtype_t dbias) {
    rvcl_layer_norm_desc_t d = {};
    d.dtype_a = da;
    /* No B operand and no bias: the request-side "unset" marker, which is
     * also what the kernel's tuple check accepts. */
    d.dtype_b = RVCL_DTYPE_AUTO;
    d.dtype_acc = dacc;
    d.dtype_c = dc;
    d.dtype_bias = RVCL_DTYPE_AUTO;
    (void)db;
    (void)dbias;

    /* Every field lands in an int64_t first, so one uniform key table
     * covers the sizes and the flags without a per-field pointer type. */
    int64_t rows = 0, norm_size = 0, eps_e9 = 0, use_scale = 0, use_shift = 0;
    const struct {
        const char *key;
        int64_t *dst;
        int required;
    } fields[] = {
            {"rows", &rows, 1},
            {"ns", &norm_size, 1},
            {"e", &eps_e9, 0},
            {"sc", &use_scale, 0},
            {"sh", &use_shift, 0},
    };
    const int nfields = (int)(sizeof(fields) / sizeof(fields[0]));
    const char *p = params ? params : "";
    int next = 0;
    while (*p) {
        int matched = -1;
        for (int i = next; i < nfields; i++)
            if (!strncmp(p, fields[i].key, strlen(fields[i].key))) {
                matched = i;
                break;
            }
        if (matched < 0) return RVCL_ERROR_INVALID_ARGUMENT;
        int ok = 1;
        const int64_t v = pool_param_value(&p, fields[matched].key, &ok);
        if (!ok) return RVCL_ERROR_INVALID_ARGUMENT;
        *fields[matched].dst = v;
        next = matched + 1;
    }
    for (int i = 0; i < nfields; i++)
        if (fields[i].required && *fields[i].dst == 0)
            return RVCL_ERROR_INVALID_ARGUMENT;
    /* The flags are booleans and eps must stay under the square root's
     * domain, so an out-of-range column is a corpus error. */
    if ((use_scale != 0 && use_scale != 1) || (use_shift != 0 && use_shift != 1)
            || eps_e9 > 1000000000)
        return RVCL_ERROR_INVALID_ARGUMENT;

    d.rows = rows;
    d.norm_size = norm_size;
    /* The scale is applied in double so the usual values land on their
     * nearest fp32: e0 is 0 and e10000 is 1e-5, not 10000 times the
     * rounded 1e-9f. */
    d.eps = (float)((double)eps_e9 * 1e-9);
    d.use_scale = (int)use_scale;
    d.use_shift = (int)use_shift;

    /* The corpus triple is the row view (m = rows, n = norm_size, k = 1);
     * a mismatch between the two columns is a corpus typo. */
    if (d.rows != m || d.norm_size != n || k != 1)
        return RVCL_ERROR_INVALID_ARGUMENT;

    od->layer_norm = d;
    return RVCL_SUCCESS;
}

/* Inner product: the descriptor already is the corpus triple's GEMM
 * projection (mb = m, oc = n, k = k), so a params column on an
 * inner_product case is a corpus error. */
static rvcl_status_t inner_product_build_desc(rvcl_op_desc_t *od, int64_t m,
        int64_t n, int64_t k, const char *params, rvcl_dtype_t da,
        rvcl_dtype_t db, rvcl_dtype_t dacc, rvcl_dtype_t dc,
        rvcl_dtype_t dbias) {
    if (params && *params) return RVCL_ERROR_INVALID_ARGUMENT;
    od->inner_product = rvcl_inner_product_desc_t {.mb = m,
            .k = k,
            .oc = n,
            .dtype_a = da,
            .dtype_b = db,
            .dtype_acc = dacc,
            .dtype_c = dc,
            .dtype_bias = dbias,
            .quant = {}};
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------- params lexing */

int op_slot_lex_params(const char *rest, const char **params, size_t *len) {
    /* A label's leading n" and a params run never collide, so the "is this
     * a label?" test stays a two-character check. */
    const int at_label = rest[0] == 'n' && rest[1] == '"';
    if (*rest == '\0' || *rest == '#' || at_label) return 0;
    const size_t plen = strspn(rest, "abcdefghijklmnopqrstuvwxyz0123456789:");
    if (plen == 0) return -1;
    *params = rest;
    *len = plen;
    return 1;
}

/* ---------------------------------------------------------- pack setups */

/* All tensors are imported zero-copy. The bias length and dtype follow the
 * row's output habit (n / int32 for the matmul integer tuples, fp32
 * otherwise). */
static void matmul_pack_setup(rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, rvcl_tensor_t *ta,
        rvcl_tensor_t *tsa, rvcl_tensor_t *tb, rvcl_tensor_t *tsb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias) {
    const rvcl_matmul_desc_t *d = &od->matmul;
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {d->m, d->k},
            .strides = {d->k, 1},
            .dtype = d->dtype_a};
    rvcl_tensor_init(ta, &td, (void *)a);

    /* A-block scale: m x K/blk row-major, dtype = scale dtype of A. */
    if (sa) {
        const int64_t blk = (int64_t)rvcl_dtype_block_size(d->dtype_a);
        rvcl_tensor_desc_t sd = {.ndims = 2,
                .dims = {d->m, d->k / blk},
                .strides = {d->k / blk, 1},
                .dtype = rvcl_dtype_scale_dtype(d->dtype_a)};
        rvcl_tensor_init(tsa, &sd, (void *)sa);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_SCALE_A, tsa);
    }

    td.dims[0] = d->k;
    td.dims[1] = d->n;
    td.strides[0] = d->n;
    td.dtype = d->dtype_b;
    rvcl_tensor_init(tb, &td, (void *)b);

    /* B-block scale: K/blk x n row-major (oneDNN-aligned). */
    if (sb) {
        const int64_t blk = (int64_t)rvcl_dtype_block_size(d->dtype_b);
        rvcl_tensor_desc_t sd = {.ndims = 2,
                .dims = {d->k / blk, d->n},
                .strides = {d->n, 1},
                .dtype = rvcl_dtype_scale_dtype(d->dtype_b)};
        rvcl_tensor_init(tsb, &sd, (void *)sb);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_SCALE_B, tsb);
    }

    td.dims[0] = d->m;
    td.dims[1] = d->n;
    td.strides[0] = d->n;
    td.dtype = d->dtype_c;
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_B, tb);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {d->n},
                .strides = {1},
                .dtype = dtype_ops_out_dtype(d->dtype_a)};
        rvcl_tensor_init(tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_BIAS, tbias);
    }
}

static void gemv_pack_setup(rvcl_tensor_pack_t *pack, const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c, rvcl_tensor_t *ta, rvcl_tensor_t *tsa,
        rvcl_tensor_t *tb, rvcl_tensor_t *tsb, rvcl_tensor_t *tc,
        rvcl_tensor_t *tbias) {
    (void)sa;
    (void)sb;
    (void)tsa;
    (void)tsb;
    const rvcl_gemv_desc_t *d = &od->gemv;
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {d->m, d->k},
            .strides = {d->k, 1},
            .dtype = d->dtype_a};
    rvcl_tensor_init(ta, &td, (void *)a);

    td.ndims = 1;
    td.dims[0] = d->k;
    td.strides[0] = 1;
    td.dtype = d->dtype_b;
    rvcl_tensor_init(tb, &td, (void *)b);

    td.dims[0] = d->m;
    td.dtype = d->dtype_c;
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_B, tb);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {d->m},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_BIAS, tbias);
    }
}

/* NCHW src, [oc][k] weights, NCHW fp32 dst, length-oc fp32 bias. */
static void conv2d_pack_setup(rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, rvcl_tensor_t *ta,
        rvcl_tensor_t *tsa, rvcl_tensor_t *tb, rvcl_tensor_t *tsb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias) {
    (void)sa;
    (void)sb;
    (void)tsa;
    (void)tsb;
    const rvcl_conv2d_desc_t *d = &od->conv2d;
    const int64_t oh = conv_oh(d), ow = conv_ow(d);
    const int64_t k = d->ic * d->kh * d->kw;

    rvcl_tensor_desc_t td = {.ndims = 4,
            .dims = {d->mb, d->ic, d->ih, d->iw},
            .strides = {d->ic * d->ih * d->iw, d->ih * d->iw, d->iw, 1},
            .dtype = d->dtype_a};
    rvcl_tensor_init(ta, &td, (void *)a);

    td.ndims = 2;
    td.dims[0] = d->oc;
    td.dims[1] = k;
    td.strides[0] = k;
    td.strides[1] = 1;
    td.dtype = d->dtype_b;
    rvcl_tensor_init(tb, &td, (void *)b);

    td.ndims = 4;
    td.dims[0] = d->mb;
    td.dims[1] = d->oc;
    td.dims[2] = oh;
    td.dims[3] = ow;
    td.strides[0] = d->oc * oh * ow;
    td.strides[1] = oh * ow;
    td.strides[2] = ow;
    td.strides[3] = 1;
    td.dtype = d->dtype_c;
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_B, tb);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {d->oc},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_BIAS, tbias);
    }
}

/* src [mb][k], w [oc][k] (row-major, B transposed), fp32 dst [mb][oc] and a
 * length-oc fp32 bias. */
static void inner_product_pack_setup(rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, rvcl_tensor_t *ta,
        rvcl_tensor_t *tsa, rvcl_tensor_t *tb, rvcl_tensor_t *tsb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias) {
    (void)sa;
    (void)sb;
    (void)tsa;
    (void)tsb;
    const rvcl_inner_product_desc_t *d = &od->inner_product;
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {d->mb, d->k},
            .strides = {d->k, 1},
            .dtype = d->dtype_a};
    rvcl_tensor_init(ta, &td, (void *)a);

    td.dims[0] = d->oc;
    td.dims[1] = d->k;
    td.strides[0] = d->k;
    td.strides[1] = 1;
    td.dtype = d->dtype_b;
    rvcl_tensor_init(tb, &td, (void *)b);

    td.dims[0] = d->mb;
    td.dims[1] = d->oc;
    td.strides[0] = d->oc;
    td.strides[1] = 1;
    td.dtype = d->dtype_c;
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_B, tb);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);
    if (bias) {
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {d->oc},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_BIAS, tbias);
    }
}

/* NCHW src and NCHW fp32 dst; no B, no bias. */
static void pooling_pack_setup(rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, rvcl_tensor_t *ta,
        rvcl_tensor_t *tsa, rvcl_tensor_t *tb, rvcl_tensor_t *tsb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias) {
    (void)sa;
    (void)sb;
    (void)tsa;
    (void)tsb;
    (void)b;
    (void)tb;
    (void)bias;
    (void)tbias;
    const rvcl_pooling_desc_t *d = &od->pooling;
    const int64_t oh = pool_oh(d), ow = pool_ow(d);

    rvcl_tensor_desc_t td = {.ndims = 4,
            .dims = {d->mb, d->ic, d->ih, d->iw},
            .strides = {d->ic * d->ih * d->iw, d->ih * d->iw, d->iw, 1},
            .dtype = d->dtype_a};
    rvcl_tensor_init(ta, &td, (void *)a);

    td.dims[2] = oh;
    td.dims[3] = ow;
    td.strides[0] = d->ic * oh * ow;
    td.strides[1] = oh * ow;
    td.strides[2] = ow;
    td.dtype = d->dtype_c;
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);
}

/* src [rows][ns] in the case dtype, fp32 dst [rows][ns]; the optional scale
 * and shift vectors both live in the rig's fp32 bias buffer as two ns-sized
 * halves (scale first, then shift), so the golden and the pack read the same
 * buffer through the same layout. A flag the case leaves off imports no
 * tensor, which is the constant the kernel substitutes. */
static void layer_norm_pack_setup(rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, rvcl_tensor_t *ta,
        rvcl_tensor_t *tsa, rvcl_tensor_t *tb, rvcl_tensor_t *tsb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias) {
    (void)sa;
    (void)sb;
    (void)b;
    (void)tb;
    (void)tsb;
    const rvcl_layer_norm_desc_t *d = &od->layer_norm;

    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {d->rows, d->norm_size},
            .strides = {d->norm_size, 1},
            .dtype = d->dtype_a};
    rvcl_tensor_init(ta, &td, (void *)a);

    td.dtype = d->dtype_c;
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);

    rvcl_tensor_desc_t pd = {.ndims = 1,
            .dims = {d->norm_size},
            .strides = {1},
            .dtype = RVCL_DTYPE_FP32};
    if (d->use_scale && bias) {
        rvcl_tensor_init(tsa, &pd, (void *)bias);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_STAT_SCALE, tsa);
    }
    if (d->use_shift && bias) {
        rvcl_tensor_init(tbias, &pd, (void *)(bias + d->norm_size));
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_STAT_SHIFT, tbias);
    }
}

/* -------------------------------------------------------- full ranges */

static rvcl_work_range_t matmul_full_range(const rvcl_op_desc_t *od) {
    return rvcl_work_range_full(od->matmul.m, od->matmul.n, od->matmul.k);
}

static rvcl_work_range_t gemv_full_range(const rvcl_op_desc_t *od) {
    return rvcl_work_range_full(od->gemv.m, 1, od->gemv.k);
}

/* The conv GEMM view: one row per output pixel over the whole batch. */
static rvcl_work_range_t conv2d_full_range(const rvcl_op_desc_t *od) {
    const rvcl_conv2d_desc_t *d = &od->conv2d;
    return rvcl_work_range_full(
            d->mb * conv_oh(d) * conv_ow(d), d->oc, d->ic * d->kh * d->kw);
}

/* One work row per output line of every batch: the whole width, k = 1. */
static rvcl_work_range_t pooling_full_range(const rvcl_op_desc_t *od) {
    const rvcl_pooling_desc_t *d = &od->pooling;
    return rvcl_work_range_full(d->mb * pool_oh(d), pool_ow(d), 1);
}

/* The inner product GEMM projection: every row over the full width and
 * reduction. */
static rvcl_work_range_t inner_product_full_range(const rvcl_op_desc_t *od) {
    const rvcl_inner_product_desc_t *d = &od->inner_product;
    return rvcl_work_range_full(d->mb, d->oc, d->k);
}

/* One work row per normalized row: the whole normalized axis, k = 1. */
static rvcl_work_range_t layer_norm_full_range(const rvcl_op_desc_t *od) {
    const rvcl_layer_norm_desc_t *d = &od->layer_norm;
    return rvcl_work_range_full(d->rows, d->norm_size, 1);
}

/* --------------------------------------------------------- out widths */

static int64_t matmul_out_width(const rvcl_op_desc_t *od) {
    return od->matmul.n;
}
static int64_t gemv_out_width(const rvcl_op_desc_t *od) {
    (void)od;
    return 1;
}
static int64_t conv2d_out_width(const rvcl_op_desc_t *od) {
    return conv_ow(&od->conv2d);
}
static int64_t pooling_out_width(const rvcl_op_desc_t *od) {
    return pool_ow(&od->pooling);
}
static int64_t inner_product_out_width(const rvcl_op_desc_t *od) {
    return od->inner_product.oc;
}
static int64_t layer_norm_out_width(const rvcl_op_desc_t *od) {
    return od->layer_norm.norm_size;
}

/* ------------------------------------------------------ shared run rig */

/* Allocates the workspaces configure() reported and feeds the AUX slots. */
rvcl_status_t op_slot_pack_ws(const rvcl_workspace_req_t *ws, int num_ws,
        void *ws_buf[], rvcl_tensor_t ws_t[], rvcl_tensor_pack_t *pack) {
    for (int i = 0; i < num_ws && i < RVCL_MAX_WORKSPACES; i++) {
        size_t align = ws[i].alignment > 1 ? (size_t)ws[i].alignment : 8;
        if (posix_memalign(&ws_buf[i], align, (size_t)(ws[i].size + align))
                != 0)
            return RVCL_ERROR_OUT_OF_MEMORY;
        rvcl_tensor_desc_t wd = {.ndims = 1,
                .dims = {ws[i].size + 1},
                .strides = {1},
                .dtype = RVCL_DTYPE_UINT8};
        rvcl_tensor_init(&ws_t[i], &wd, ws_buf[i]);
        rvcl_tensor_pack_add(
                pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + i), &ws_t[i]);
    }
    return RVCL_SUCCESS;
}

void op_slot_pack_free_ws(void *ws_buf[], int num_ws) {
    (void)num_ws; /* the buffers were zero-initialized; free all slots */
    for (int i = 0; i < RVCL_MAX_WORKSPACES; i++)
        free(ws_buf[i]);
}

/* The dispatch-driven run family entry: routes through the internal
 * rvcl_dispatch_op so the test drives exactly what the library pipeline
 * selects (bench C mode and the differential dispatch check). */
static rvcl_status_t run_dispatch_impl(const op_slot_t *slot,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c) {
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_op(slot->op, od, &r);
    if (st != RVCL_SUCCESS || !r) return st;

    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
    void *ws_buf[RVCL_MAX_WORKSPACES] = {0};
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    rvcl_tensor_t ta, tb, tc, tbias, tsa, tsb;
    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    slot->pack_setup(&pack, od, a, sa, b, sb, bias, c, &ta, &tsa, &tb, &tsb,
            &tc, &tbias);
    st = op_slot_pack_ws(ws, num_ws, ws_buf, ws_t, &pack);
    if (st != RVCL_SUCCESS) {
        op_slot_pack_free_ws(ws_buf, num_ws);
        rvcl_dispatch_release(r);
        return st;
    }
    const rvcl_work_range_t full = slot->full_range(od);
    st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
    op_slot_pack_free_ws(ws_buf, num_ws);
    rvcl_dispatch_release(r);
    return st;
}

static rvcl_status_t matmul_run_dispatch(const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c) {
    return run_dispatch_impl(
            op_slot_for(RVCL_OP_MATMUL), od, a, sa, b, sb, bias, c);
}

static rvcl_status_t gemv_run_dispatch(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c) {
    return run_dispatch_impl(
            op_slot_for(RVCL_OP_GEMV), od, a, sa, b, sb, bias, c);
}

static rvcl_status_t conv2d_run_dispatch(const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c) {
    return run_dispatch_impl(
            op_slot_for(RVCL_OP_CONV2D), od, a, sa, b, sb, bias, c);
}

/* Pooling routes through the public dispatch entry (the internal table is
 * keyed on the same row, but the entry is what a user calls). */
static rvcl_status_t pooling_run_dispatch(const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c) {
    (void)sa;
    (void)sb;
    (void)b;
    (void)bias;
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_pooling(&od->pooling, &r);
    if (st != RVCL_SUCCESS || !r) return st;

    const op_slot_t *slot = op_slot_for(RVCL_OP_POOLING);
    rvcl_tensor_t ta, tb, tc, tbias, tsa, tsb;
    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    slot->pack_setup(&pack, od, a, sa, b, sb, bias, c, &ta, &tsa, &tb, &tsb,
            &tc, &tbias);
    const rvcl_work_range_t full = slot->full_range(od);
    st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
    rvcl_dispatch_release(r);
    return st;
}

static rvcl_status_t inner_product_run_dispatch(const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c) {
    return run_dispatch_impl(
            op_slot_for(RVCL_OP_INNER_PRODUCT), od, a, sa, b, sb, bias, c);
}

/* Layer norm routes through the public dispatch entry, like pooling: the
 * entry is what a user calls and it resolves the same row. */
static rvcl_status_t layer_norm_run_dispatch(const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c) {
    (void)sa;
    (void)sb;
    (void)b;
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_layer_norm(&od->layer_norm, &r);
    if (st != RVCL_SUCCESS || !r) return st;

    const op_slot_t *slot = op_slot_for(RVCL_OP_LAYER_NORM);
    rvcl_tensor_t ta, tb, tc, tbias, tsa, tsb;
    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    slot->pack_setup(&pack, od, a, sa, b, sb, bias, c, &ta, &tsa, &tb, &tsb,
            &tc, &tbias);
    const rvcl_work_range_t full = slot->full_range(od);
    st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
    rvcl_dispatch_release(r);
    return st;
}

/* ----------------------------------------------------------- run named */

/* Runs a named kernel by direct factory use (bypasses dispatch selection
 * so every registered kernel can be exercised, not just the selected one).
 * r_m0/r_m1 restrict the executed row range (negative = full). */
static rvcl_status_t run_named_impl(const op_slot_t *slot, const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    rvcl_kernel_create_fn create = rvcl_registry_find_factory(name);
    if (!create) return RVCL_ERROR_NOT_FOUND;

    const rvcl_kernel_desc_t *found = NULL;
    for (int i = 0, n = rvcl_registry_count(); i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (kd && strcmp(kd->name, name) == 0) {
            found = kd;
            break;
        }
    }
    if (!found) return RVCL_ERROR_NOT_FOUND;

    rvcl_kernel_t *k = NULL;
    rvcl_status_t st = create(&k, found);
    if (st != RVCL_SUCCESS) return st;

    st = k->vt->validate(od, rvcl_get_capability());
    if (st != RVCL_SUCCESS) {
        rvcl_kernel_destroy(k);
        return st;
    }

    rvcl_workspace_req_t ws[RVCL_MAX_WORKSPACES];
    int num_ws = 0;
    st = k->vt->configure(k, od, ws, &num_ws);
    if (st != RVCL_SUCCESS) {
        rvcl_kernel_destroy(k);
        return st;
    }

    rvcl_tensor_t ta, tb, tc, tbias, tsa, tsb;
    rvcl_tensor_pack_t pack;
    rvcl_tensor_pack_init(&pack);
    slot->pack_setup(&pack, od, a, sa, b, sb, bias, c, &ta, &tsa, &tb, &tsb,
            &tc, &tbias);
    void *ws_buf[RVCL_MAX_WORKSPACES] = {0};
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    st = op_slot_pack_ws(ws, num_ws, ws_buf, ws_t, &pack);
    if (st != RVCL_SUCCESS) {
        op_slot_pack_free_ws(ws_buf, num_ws);
        rvcl_kernel_destroy(k);
        return st;
    }

    rvcl_work_range_t rng = slot->full_range(od);
    if (r_m0 >= 0) {
        rng.m_begin = r_m0;
        rng.m_end = r_m1;
    }
    st = k->vt->run(k, &pack, &rng);

    op_slot_pack_free_ws(ws_buf, num_ws);
    rvcl_kernel_destroy(k);
    return st;
}

static rvcl_status_t matmul_run_named(const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    return run_named_impl(op_slot_for(RVCL_OP_MATMUL), name, od, a, sa, b, sb,
            bias, c, r_m0, r_m1);
}

static rvcl_status_t gemv_run_named(const char *name, const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c, int64_t r_m0, int64_t r_m1) {
    return run_named_impl(op_slot_for(RVCL_OP_GEMV), name, od, a, sa, b, sb,
            bias, c, r_m0, r_m1);
}

static rvcl_status_t conv2d_run_named(const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    return run_named_impl(op_slot_for(RVCL_OP_CONV2D), name, od, a, sa, b, sb,
            bias, c, r_m0, r_m1);
}

static rvcl_status_t pooling_run_named(const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    return run_named_impl(op_slot_for(RVCL_OP_POOLING), name, od, a, sa, b, sb,
            bias, c, r_m0, r_m1);
}

static rvcl_status_t inner_product_run_named(const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    return run_named_impl(op_slot_for(RVCL_OP_INNER_PRODUCT), name, od, a, sa,
            b, sb, bias, c, r_m0, r_m1);
}

static rvcl_status_t layer_norm_run_named(const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    return run_named_impl(op_slot_for(RVCL_OP_LAYER_NORM), name, od, a, sa, b,
            sb, bias, c, r_m0, r_m1);
}

/* ---------------------------------------------------------------- table */

static const rvcl_dtype_t k_matmul_dtypes[] = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16,
        RVCL_DTYPE_BF16, RVCL_DTYPE_FP8_E4M3, RVCL_DTYPE_FP8_E5M2,
        RVCL_DTYPE_MXFP8_E4M3, RVCL_DTYPE_MXFP8_E5M2, RVCL_DTYPE_NVFP4,
        RVCL_DTYPE_MXFP4, RVCL_DTYPE_INT8, RVCL_DTYPE_UINT8, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_gemv_dtypes[] = {RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_conv2d_dtypes[]
        = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16, RVCL_DTYPE_BF16, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_inner_product_dtypes[]
        = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16, RVCL_DTYPE_BF16, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_pooling_dtypes[]
        = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16, RVCL_DTYPE_BF16, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_layer_norm_dtypes[]
        = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16, RVCL_DTYPE_BF16, RVCL_DTYPE_AUTO};

static const op_slot_t k_op_slots[] = {
        {
                .name = "matmul",
                .op = RVCL_OP_MATMUL,
                .dtypes = k_matmul_dtypes,
                .a_elems = matmul_a_elems,
                .b_elems = matmul_b_elems,
                .bias_elems = matmul_bias_elems,
                .out_elems = matmul_out_elems,
                .golden = matmul_golden,
                .build_desc = matmul_build_desc,
                .pack_setup = matmul_pack_setup,
                .full_range = matmul_full_range,
                .out_width = matmul_out_width,
                .run_dispatch = matmul_run_dispatch,
                .run_named = matmul_run_named,
                .can_timing = 1,
        },
        {
                .name = "gemv",
                .op = RVCL_OP_GEMV,
                .dtypes = k_gemv_dtypes,
                .a_elems = gemv_a_elems,
                .b_elems = gemv_b_elems,
                .bias_elems = gemv_bias_elems,
                .out_elems = gemv_out_elems,
                .golden = gemv_golden,
                .build_desc = gemv_build_desc,
                .pack_setup = gemv_pack_setup,
                .full_range = gemv_full_range,
                .out_width = gemv_out_width,
                .run_dispatch = gemv_run_dispatch,
                .run_named = gemv_run_named,
                .can_timing = 0,
        },
        {
                .name = "conv2d",
                .op = RVCL_OP_CONV2D,
                .dtypes = k_conv2d_dtypes,
                .a_elems = conv_a_elems,
                .b_elems = conv_b_elems,
                .bias_elems = conv_bias_elems,
                .out_elems = conv_out_elems,
                .golden = conv2d_golden,
                .build_desc = conv2d_build_desc,
                .pack_setup = conv2d_pack_setup,
                .full_range = conv2d_full_range,
                .out_width = conv2d_out_width,
                .run_dispatch = conv2d_run_dispatch,
                .run_named = conv2d_run_named,
                /* im2col + GEMM has no micro-kernel-only timing path (the
                 * band staging is part of every run). */
                .can_timing = 0,
        },
        {
                .name = "inner_product",
                .op = RVCL_OP_INNER_PRODUCT,
                .dtypes = k_inner_product_dtypes,
                .a_elems = ip_a_elems,
                .b_elems = ip_b_elems,
                .bias_elems = ip_bias_elems,
                .out_elems = ip_out_elems,
                .golden = inner_product_golden,
                .build_desc = inner_product_build_desc,
                .pack_setup = inner_product_pack_setup,
                .full_range = inner_product_full_range,
                .out_width = inner_product_out_width,
                .run_dispatch = inner_product_run_dispatch,
                .run_named = inner_product_run_named,
                /* transposing pack + GEMM has no micro-kernel-only timing
                 * path (the pack is part of every run). */
                .can_timing = 0,
        },
        {
                .name = "pooling",
                .op = RVCL_OP_POOLING,
                .dtypes = k_pooling_dtypes,
                .a_elems = pool_a_elems,
                .b_elems = pool_none_elems,
                .bias_elems = pool_none_elems,
                .out_elems = pool_out_elems,
                .golden = pooling_golden,
                .build_desc = pooling_build_desc,
                .pack_setup = pooling_pack_setup,
                .full_range = pooling_full_range,
                .out_width = pooling_out_width,
                .run_dispatch = pooling_run_dispatch,
                .run_named = pooling_run_named,
                /* the row walk is the only execution path: no separate
                 * micro-kernel to time. */
                .can_timing = 0,
        },
        {
                .name = "layer_norm",
                .op = RVCL_OP_LAYER_NORM,
                .dtypes = k_layer_norm_dtypes,
                .a_elems = ln_a_elems,
                .b_elems = pool_none_elems,
                /* the two optional vectors share the fp32 parameter buffer */
                .bias_elems = ln_param_elems,
                .out_elems = ln_out_elems,
                .golden = layer_norm_golden,
                .build_desc = layer_norm_build_desc,
                .pack_setup = layer_norm_pack_setup,
                .full_range = layer_norm_full_range,
                .out_width = layer_norm_out_width,
                .run_dispatch = layer_norm_run_dispatch,
                .run_named = layer_norm_run_named,
                /* the row walk is the only execution path: no separate
                 * micro-kernel to time. */
                .can_timing = 0,
        },
};

static const int k_op_slots_count
        = (int)(sizeof(k_op_slots) / sizeof(k_op_slots[0]));

const op_slot_t *op_slot_for(rvcl_operation_t op) {
    for (int i = 0; i < k_op_slots_count; i++)
        if (k_op_slots[i].op == op) return &k_op_slots[i];
    return NULL;
}

const op_slot_t *op_slot_from_name(const char *name) {
    for (int i = 0; i < k_op_slots_count; i++)
        if (strcmp(name, k_op_slots[i].name) == 0) return &k_op_slots[i];
    return NULL;
}

const op_slot_t *op_slot_from_batch_path(const char *path) {
    const char *slash = strrchr(path, '/');
    size_t flen = slash ? (size_t)(slash - path) : 0;
    const char *start = path;
    for (size_t i = flen; i > 0;)
        if (path[--i] == '/') {
            start = path + i + 1;
            break;
        }
    if (flen == 0 || start >= path + flen) return op_slot_default();
    char dir[32];
    size_t n = (size_t)(path + flen - start);
    if (n >= sizeof(dir)) n = sizeof(dir) - 1;
    memcpy(dir, start, n);
    dir[n] = '\0';
    const op_slot_t *slot = op_slot_from_name(dir);
    return slot ? slot : op_slot_default();
}

const op_slot_t *op_slot_default(void) {
    return op_slot_for(RVCL_OP_MATMUL);
}
