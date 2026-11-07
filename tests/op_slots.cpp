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
 * tests/op_slots.cpp — test-side operator slot table (see op_slots.hpp)
 *
 * The matmul row is the first table entry (migrated from the differential
 * and bench mains); the gemv row is the second, the demonstration of the
 * "add an operator = one row + a corpus dir" contract. Both run paths are
 * shared below and only branch on the operator inside the table
 * implementation, never in the test mains.
 ******************************************************************************/

#include "op_slots.hpp"

#include <stdlib.h>
#include <string.h>

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

/* ---------------------------------------------------------------- golden */

static void matmul_golden(const rvcl_op_desc_t *od, const void *a,
        const void *b, const float *bias, float *c) {
    const dtype_ops_t *op = dtype_ops_for(od->matmul.dtype_a);
    op->golden(a, b, bias, c, od->matmul.m, od->matmul.n, od->matmul.k);
}

/* Naive GEMV: y[i] = bias[i] + sum_p a[i][p] * x[p]. */
static void gemv_golden(const rvcl_op_desc_t *od, const void *a, const void *b,
        const float *bias, float *c) {
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

/* ------------------------------------------------------------ build desc */

static void matmul_build_desc(rvcl_op_desc_t *od, int64_t m, int64_t n,
        int64_t k, rvcl_dtype_t da, rvcl_dtype_t db, rvcl_dtype_t dacc,
        rvcl_dtype_t dc, rvcl_dtype_t dbias) {
    od->matmul = rvcl_matmul_desc_t {.m = m,
            .n = n,
            .k = k,
            .dtype_a = da,
            .dtype_b = db,
            .dtype_acc = dacc,
            .dtype_c = dc,
            .dtype_bias = dbias,
            .quant = {}};
}

static void gemv_build_desc(rvcl_op_desc_t *od, int64_t m, int64_t n, int64_t k,
        rvcl_dtype_t da, rvcl_dtype_t db, rvcl_dtype_t dacc, rvcl_dtype_t dc,
        rvcl_dtype_t dbias) {
    (void)n; /* a gemv has no n dimension */
    od->gemv = rvcl_gemv_desc_t {.m = m,
            .k = k,
            .dtype_a = da,
            .dtype_b = db,
            .dtype_acc = dacc,
            .dtype_c = dc,
            .dtype_bias = dbias,
            .quant = {}};
}

/* ------------------------------------------------------ shared run rig */

/* Builds the tensor pack for the operator's descriptor. The B and C
 * tensors are 2D for matmul (k x n, m x n) and 1D vectors for gemv
 * (k, m); bias is a vector of the output length (n for matmul, m for
 * gemv). All tensors are imported zero-copy. */
void op_slot_pack_setup(const op_slot_t *slot, rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *b,
        const float *bias, float *c, rvcl_tensor_t *ta, rvcl_tensor_t *tb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias) {
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {0, 0},
            .strides = {0, 1},
            .dtype = RVCL_DTYPE_FP32};

    const int64_t m = slot->op == RVCL_OP_MATMUL ? od->matmul.m : od->gemv.m;
    const int64_t k = slot->op == RVCL_OP_MATMUL ? od->matmul.k : od->gemv.k;
    const int64_t n = slot->op == RVCL_OP_MATMUL ? od->matmul.n : 1;

    /* A: m x k row-major */
    td.dims[0] = m;
    td.dims[1] = k;
    td.strides[0] = k;
    td.dtype = slot->op == RVCL_OP_MATMUL ? od->matmul.dtype_a
                                          : od->gemv.dtype_a;
    rvcl_tensor_init(ta, &td, (void *)a);

    /* B: matmul k x n; gemv k-vector */
    if (slot->op == RVCL_OP_MATMUL) {
        td.ndims = 2;
        td.dims[0] = k;
        td.dims[1] = n;
        td.strides[0] = n;
        td.dtype = od->matmul.dtype_b;
    } else {
        td.ndims = 1;
        td.dims[0] = k;
        td.strides[0] = 1;
        td.dtype = od->gemv.dtype_b;
    }
    rvcl_tensor_init(tb, &td, (void *)b);

    /* C: matmul m x n; gemv m-vector */
    if (slot->op == RVCL_OP_MATMUL) {
        td.ndims = 2;
        td.dims[0] = m;
        td.dims[1] = n;
        td.strides[0] = n;
        td.dtype = od->matmul.dtype_c;
    } else {
        td.ndims = 1;
        td.dims[0] = m;
        td.strides[0] = 1;
        td.dtype = od->gemv.dtype_c;
    }
    rvcl_tensor_init(tc, &td, c);

    rvcl_tensor_pack_init(pack);
    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_A, ta);
    rvcl_tensor_pack_add(pack, RVCL_TENSOR_SRC_B, tb);
    rvcl_tensor_pack_add_out(pack, RVCL_TENSOR_DST, tc);
    if (bias) {
        /* bias length is the output length: n for matmul, m for gemv. */
        const int64_t bias_len = slot->op == RVCL_OP_MATMUL ? n : m;
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {bias_len},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(tbias, &bd, (void *)bias);
        rvcl_tensor_pack_add(pack, RVCL_TENSOR_BIAS, tbias);
    }
}

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

/* --------------------------------------------------------- run dispatch */

/* The dispatch-driven run family entry: routes through the internal
 * rvcl_dispatch_op so the test drives exactly what the library pipeline
 * selects (bench C mode and the differential dispatch check). */
static rvcl_status_t run_dispatch_impl(const op_slot_t *slot,
        const rvcl_op_desc_t *od, const void *a, const void *b,
        const float *bias, float *c) {
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_op(slot->op, od, &r);
    if (st != RVCL_SUCCESS || !r) return st;

    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
    void *ws_buf[RVCL_MAX_WORKSPACES] = {0};
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_pack_t pack;
    op_slot_pack_setup(slot, &pack, od, a, b, bias, c, &ta, &tb, &tc, &tbias);
    st = op_slot_pack_ws(ws, num_ws, ws_buf, ws_t, &pack);
    if (st != RVCL_SUCCESS) {
        op_slot_pack_free_ws(ws_buf, num_ws);
        rvcl_dispatch_release(r);
        return st;
    }
    const int64_t m = slot->op == RVCL_OP_MATMUL ? od->matmul.m : od->gemv.m;
    const int64_t n = slot->op == RVCL_OP_MATMUL ? od->matmul.n : 1;
    const int64_t kk = slot->op == RVCL_OP_MATMUL ? od->matmul.k : od->gemv.k;
    rvcl_work_range_t full = rvcl_work_range_full(m, n, kk);
    st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
    op_slot_pack_free_ws(ws_buf, num_ws);
    rvcl_dispatch_release(r);
    return st;
}

static rvcl_status_t matmul_run_dispatch(const rvcl_op_desc_t *od,
        const void *a, const void *b, const float *bias, float *c) {
    return run_dispatch_impl(op_slot_for(RVCL_OP_MATMUL), od, a, b, bias, c);
}

static rvcl_status_t gemv_run_dispatch(const rvcl_op_desc_t *od, const void *a,
        const void *b, const float *bias, float *c) {
    return run_dispatch_impl(op_slot_for(RVCL_OP_GEMV), od, a, b, bias, c);
}

/* ----------------------------------------------------------- run named */

/* Runs a named kernel by direct factory use (bypasses dispatch selection
 * so every registered kernel can be exercised, not just the selected one).
 * r_m0/r_m1 restrict the executed row range (negative = full). */
static rvcl_status_t run_named_impl(const op_slot_t *slot, const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *b,
        const float *bias, float *c, int64_t r_m0, int64_t r_m1) {
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

    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_pack_t pack;
    op_slot_pack_setup(slot, &pack, od, a, b, bias, c, &ta, &tb, &tc, &tbias);
    void *ws_buf[RVCL_MAX_WORKSPACES] = {0};
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    st = op_slot_pack_ws(ws, num_ws, ws_buf, ws_t, &pack);
    if (st != RVCL_SUCCESS) {
        op_slot_pack_free_ws(ws_buf, num_ws);
        rvcl_kernel_destroy(k);
        return st;
    }

    const int64_t m = slot->op == RVCL_OP_MATMUL ? od->matmul.m : od->gemv.m;
    const int64_t n = slot->op == RVCL_OP_MATMUL ? od->matmul.n : 1;
    const int64_t kk = slot->op == RVCL_OP_MATMUL ? od->matmul.k : od->gemv.k;
    rvcl_work_range_t rng = rvcl_work_range_full(m, n, kk);
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
        const rvcl_op_desc_t *od, const void *a, const void *b,
        const float *bias, float *c, int64_t r_m0, int64_t r_m1) {
    return run_named_impl(
            op_slot_for(RVCL_OP_MATMUL), name, od, a, b, bias, c, r_m0, r_m1);
}

static rvcl_status_t gemv_run_named(const char *name, const rvcl_op_desc_t *od,
        const void *a, const void *b, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1) {
    return run_named_impl(
            op_slot_for(RVCL_OP_GEMV), name, od, a, b, bias, c, r_m0, r_m1);
}

/* ---------------------------------------------------------------- table */

static const rvcl_dtype_t k_matmul_dtypes[]
        = {RVCL_DTYPE_FP32, RVCL_DTYPE_FP16, RVCL_DTYPE_AUTO};

static const rvcl_dtype_t k_gemv_dtypes[] = {RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO};

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
                .run_dispatch = gemv_run_dispatch,
                .run_named = gemv_run_named,
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
