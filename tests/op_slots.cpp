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

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/common/conv2d_geometry.h"
#include "../src/runtime/dispatch_internal.h"
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

/* ---------------------------------------------------------------- table */

static const rvcl_dtype_t k_matmul_dtypes[] = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16,
        RVCL_DTYPE_BF16, RVCL_DTYPE_FP8_E4M3, RVCL_DTYPE_FP8_E5M2,
        RVCL_DTYPE_MXFP8_E4M3, RVCL_DTYPE_MXFP8_E5M2, RVCL_DTYPE_NVFP4,
        RVCL_DTYPE_MXFP4, RVCL_DTYPE_INT8, RVCL_DTYPE_UINT8, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_gemv_dtypes[] = {RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_conv2d_dtypes[]
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
