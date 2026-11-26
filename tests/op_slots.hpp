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

/* One row per operator under test. A row carries everything the test
 * executables need to drive the operator without an operator branch in the
 * main: the corpus/CLI name, the operator enum, the supported test-dtype
 * set, the input/output element counts for buffer sizing, the golden
 * routing, the descriptor builder, the tensor-pack setup, the full work
 * range and the run family. The run family is the dispatch-driven path
 * through the internal rvcl_dispatch_op entry (run_dispatch) plus the
 * factory-bypass path that runs one named kernel (run_named, the
 * differential per-kernel coverage).
 *
 * Shape, dtype and layout knowledge lives in the row: the shared rig only
 * does "pack init -> row's pack_setup -> workspace -> run", so it never
 * reads the operation descriptor itself and a new operator cannot land in
 * another operator's descriptor branch.
 *
 * The library counterpart is the dispatch op table in src/runtime/
 * dispatch.cpp; the test table is the landing point a new operator registers
 * its test facts in. Adding an operator = adding one row here plus its
 * corpus directory and one rvcl_add_corpus() line. */

#ifndef RVCL_TEST_OP_SLOTS_H
#define RVCL_TEST_OP_SLOTS_H

#include <stddef.h>
#include <stdint.h>

#include "rvcl/rvcl.h"

/* Element count of an input/output tensor for a built descriptor. */
typedef int64_t (*op_slot_elems_fn)(const rvcl_op_desc_t *od);

/* Golden: dtype-stored a/b (+ scale tensors sa/sb for block-scale dtypes,
 * NULL otherwise) -> fp32 output buffer c. */
typedef void (*op_slot_golden_fn)(const rvcl_op_desc_t *od, const void *a,
        const void *sa, const void *b, const void *sb, const float *bias,
        float *c);

/* Build the operator descriptor for a case. m/n/k are the corpus triple;
 * params is the case's operator-specific parameter column (empty string
 * when the operator needs none). dacc is RVCL_DTYPE_AUTO unless the caller
 * pins it. A row parses params into the descriptor fields the corpus
 * triple cannot express; a malformed params column, or one that disagrees
 * with the shape triple, is rejected with RVCL_ERROR_INVALID_ARGUMENT so
 * the consumer can report a corpus error instead of running a shape the
 * case does not describe. */
typedef rvcl_status_t (*op_slot_build_desc_fn)(rvcl_op_desc_t *od, int64_t m,
        int64_t n, int64_t k, const char *params, rvcl_dtype_t da,
        rvcl_dtype_t db, rvcl_dtype_t dacc, rvcl_dtype_t dc,
        rvcl_dtype_t dbias);

/* Fill the tensor pack from the descriptor and the caller's buffers: the
 * row owns the tensor layouts (dims, strides, dtypes), the bias length and
 * which slots exist. The rig calls rvcl_tensor_pack_init() first. */
typedef void (*op_slot_pack_setup_fn)(rvcl_tensor_pack_t *pack,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, rvcl_tensor_t *ta,
        rvcl_tensor_t *tsa, rvcl_tensor_t *tb, rvcl_tensor_t *tsb,
        rvcl_tensor_t *tc, rvcl_tensor_t *tbias);

/* The full work range of a case in the operator's own GEMM view. */
typedef rvcl_work_range_t (*op_slot_range_fn)(const rvcl_op_desc_t *od);

/* Output width used to print a mismatch index as (i, j); 1 for a
 * vector-shaped output. */
typedef int64_t (*op_slot_width_fn)(const rvcl_op_desc_t *od);

/* Run the dispatch-selected kernel via rvcl_dispatch_op into fp32 c. */
typedef rvcl_status_t (*op_slot_run_dispatch_fn)(const rvcl_op_desc_t *od,
        const void *a, const void *sa, const void *b, const void *sb,
        const float *bias, float *c);

/* Run one named kernel by factory bypass into fp32 c; r_m0/r_m1 restrict
 * the executed row range (negative = full range). */
typedef rvcl_status_t (*op_slot_run_named_fn)(const char *name,
        const rvcl_op_desc_t *od, const void *a, const void *sa, const void *b,
        const void *sb, const float *bias, float *c, int64_t r_m0,
        int64_t r_m1);

typedef struct {
    /* Corpus directory / CLI operator name ("matmul", "gemv", "conv2d",
     * "inner_product", "pooling", "layer_norm"). */
    const char *name;
    rvcl_operation_t op;
    /* Supported test dtypes, RVCL_DTYPE_AUTO-terminated. */
    const rvcl_dtype_t *dtypes;
    /* Buffer sizing: element counts of A, B, the bias and the output C. */
    op_slot_elems_fn a_elems;
    op_slot_elems_fn b_elems;
    op_slot_elems_fn bias_elems;
    op_slot_elems_fn out_elems;
    op_slot_golden_fn golden;
    op_slot_build_desc_fn build_desc;
    op_slot_pack_setup_fn pack_setup;
    op_slot_range_fn full_range;
    op_slot_width_fn out_width;
    op_slot_run_dispatch_fn run_dispatch;
    op_slot_run_named_fn run_named;
    /* 1 when the operator has a micro-kernel timing path (bench P/S). */
    int can_timing;
} op_slot_t;

/* Lexes the optional params column that follows a case's shape token.
 * rest is the line text already advanced past the shape token and its
 * separator, so both corpus consumers share one lexical form: a params run
 * is [a-z0-9:]+ and a label always starts with n", so the two never
 * collide. Returns 1 when a column is present (params and len set), 0 when
 * the line carries none, -1 when the column is malformed (empty run). The
 * caller owns copying the run out of the line buffer, because the buffer is
 * reused per line and each consumer has its own buffer policy. */
int op_slot_lex_params(const char *rest, const char **params, size_t *len);

/* Row lookup by operator enum; NULL when unknown. */
const op_slot_t *op_slot_for(rvcl_operation_t op);

/* Row lookup by name (corpus dir / CLI string); NULL when unknown. */
const op_slot_t *op_slot_from_name(const char *name);

/* Row derived from a corpus file path's directory name (the directory
 * name IS the operator); the matmul default when it does not match. */
const op_slot_t *op_slot_from_batch_path(const char *path);

/* The default operator when none is named on the CLI (matmul). */
const op_slot_t *op_slot_default(void);

/* Allocates the workspaces configure() reported and feeds the AUX slots.
 * Operator-agnostic: a kernel reports its workspace demands and the slots
 * are the AUX group. */
rvcl_status_t op_slot_pack_ws(const rvcl_workspace_req_t *ws, int num_ws,
        void *ws_buf[], rvcl_tensor_t ws_t[], rvcl_tensor_pack_t *pack);
void op_slot_pack_free_ws(void *ws_buf[], int num_ws);

#endif /* RVCL_TEST_OP_SLOTS_H */
