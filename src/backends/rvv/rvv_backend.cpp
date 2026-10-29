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
 * rvv_backend.cpp — RVV backend framework adapter: validate/configure/run
 *
 * One kernel descriptor family per dtype (PLAN §6.2 item 7):
 *   rvcl_matmul_f32_rvv   (f32,f32,f32,f32)     requires RVCL_FEATURE_RVV
 *   rvcl_matmul_f16_rvv   (f16,f16,f32,f32)     requires RVV + FP16 (Zvfh)
 *
 * run() contract (PLAN §4.3): consumes only pack memory; workspace is the
 * packed-B panel buffer (AUX_0) reported by configure(); never mallocs,
 * never creates threads. The work range is honored exactly.
 ******************************************************************************/

#include "rvv_internal.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    int64_t m, n, k;
    rvv_gemm_fn gemm; /* dtype-specific micro-kernel entry */
    size_t elt; /* element size of A/B */
} rvv_matmul_t;

/* ------------------------------------------------------------- lifecycle */

static rvcl_status_t rvv_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap;
    const rvcl_matmul_desc_t *d = &desc->matmul;
    if (d->m <= 0 || d->n <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    if (d->quant.enabled) return RVCL_UNSUPPORTED; /* quant lands v0.4 */
    return RVCL_SUCCESS;
}

static rvcl_status_t rvv_configure(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
        rvcl_workspace_req_t *ws_req, int *num_ws) {
    rvv_matmul_t *rk = (rvv_matmul_t *)k;
    const rvcl_matmul_desc_t *d = &desc->matmul;
    rk->m = d->m;
    rk->n = d->n;
    rk->k = d->k;

    /* Workspace slot 0: packed-B panel buffer (lifetime RUN: rebuilt or
     * reused per run; the host decides reuse across runs of identical B).
     * Slot 1 (reserved for split-K accumulation) is inactive in v0.1. */
    size_t elt = rvcl_dtype_size(d->dtype_a);
    size_t bytes = rvv_packed_b_bytes(d->k, d->n, elt);
    if (ws_req) {
        ws_req[RVV_WS_PACKED_B] = rvcl_workspace_req_t {
                .size = (int64_t)bytes,
                .alignment = 64,
                .lifetime = RVCL_WS_LIFETIME_RUN,
        };
        ws_req[RVV_WS_ACC] = rvcl_workspace_req_t {};
    }
    if (num_ws) *num_ws = 1;
    return RVCL_SUCCESS;
}

static rvcl_status_t rvv_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *r) {
    rvv_matmul_t *rk = (rvv_matmul_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *b = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    const rvcl_tensor_t *ws = rvcl_tensor_pack_get(
            pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + RVV_WS_PACKED_B));
    if (!a || !b || !c || !ws) return RVCL_ERROR_INVALID_ARGUMENT;

    /* Pack B into the workspace (dispatch-once-run-many callers may pre-pack
     * via pack_b and keep the buffer across runs). */
    rvcl_status_t st = rvv_pack_b_panel(
            b, (rvcl_tensor_t *)ws, rk->m, rk->n, rk->k, rk->elt);
    if (st != RVCL_SUCCESS) return st;

    rk->gemm(a->data, ws->data, bias ? bias->data : NULL, c->data, rk->m, rk->n,
            rk->k, r);
    return RVCL_SUCCESS;
}

static void rvv_destroy(rvcl_kernel_t *k) {
    (void)k;
}

/* pack_a: A consumed unpacked in v0.1 — metadata pass-through (the vtable
 * slot stays NULL; this function exists for future panel-A kernels). */
static rvcl_status_t rvv_pack_a(
        rvcl_kernel_t *k, const rvcl_tensor_t *src, rvcl_tensor_t *dst) {
    (void)k;
    if (!src || !dst) return RVCL_ERROR_INVALID_ARGUMENT;
    *dst = *src;
    return RVCL_SUCCESS;
}

static const rvcl_kernel_vtable_t g_rvv_vtable = {
        .validate = rvv_validate,
        .configure = rvv_configure,
        .pack_a = rvv_pack_a,
        .pack_b
        = NULL, /* bound per-dtype in create (panel packing differs by elt) */
        .run = rvv_run,
        .destroy = rvv_destroy,
};

/* ---------------------------------------------- micro-kernel selection */

/* RVCL_RVV_KERNEL knob, read once per process (the C++11 static guard
 * makes concurrent first calls safe). Values: auto | jit | c |
 * intrinsics. intrinsics forces the legacy intrinsics kernels for
 * regression runs and only resolves where they are compiled in (the
 * vector -march override); elsewhere the by-name resolver fails loudly. */
static const char *rvv_kernel_knob(void) {
    static const char *knob = getenv("RVCL_RVV_KERNEL");
    return knob;
}

__attribute__((visibility("default"))) rvv_gemm_fn rvv_pick_gemm_f32_by_name(
        const char *name) {
    if (!strcmp(name, "c")) return rvv_gemm_f32_c;
#if defined(__riscv_vector)
    if (!strcmp(name, "intrinsics")) return rvv_gemm_f32_rvv;
#endif
#if defined(RVCL_HAVE_JIT)
    if (!strcmp(name, "auto") || !strcmp(name, "jit")) {
        const int forced = strcmp(name, "jit") == 0;
        if (rvv_gemm_f32_jit_available()) {
            rvv_gemm_fn jit = rvv_gemm_f32_jit_entry();
            if (jit) return jit;
            /* generation failed and already warned; the forced request
             * has no fallback, auto degrades to the C kernel below */
            if (forced) return NULL;
        } else if (forced) {
            fprintf(stderr,
                    "rvcl: forced JIT f32 path requires the vector "
                    "extension\n");
            return NULL;
        }
    }
#endif
    if (!strcmp(name, "auto")) return rvv_gemm_f32_c;
    fprintf(stderr,
            "rvcl: f32 kernel path '%s' is unavailable "
            "(want auto, c, jit or intrinsics)\n",
            name);
    return NULL;
}

__attribute__((visibility("default"))) rvv_gemm_fn rvv_pick_gemm_f16_by_name(
        const char *name) {
    if (!strcmp(name, "c")) return rvv_gemm_f16_c;
#if defined(__riscv_vector)
    if (!strcmp(name, "intrinsics") || !strcmp(name, "auto"))
        return rvv_gemm_f16_rvv;
#else
    if (!strcmp(name, "auto")) return rvv_gemm_f16_c;
#endif
    if (!strcmp(name, "jit")) {
        fprintf(stderr,
                "rvcl: f16 has no JIT path (want auto, c or intrinsics)\n");
        return NULL;
    }
    fprintf(stderr,
            "rvcl: f16 kernel path '%s' is unavailable "
            "(want auto, c or intrinsics)\n",
            name);
    return NULL;
}

/* Single source of truth for the live micro-kernel per dtype (see
 * rvv_internal.hpp): knob-free auto selection is JIT-when-RVV-else-C;
 * the intrinsics kernels stay reachable only through the by-name
 * resolver. Bench kernel-only timing calls the same selectors, so it
 * always measures what dispatch actually bound. */
__attribute__((visibility("default"))) rvv_gemm_fn rvv_pick_gemm_f32(void) {
    const char *knob = rvv_kernel_knob();
    if (knob && *knob) return rvv_pick_gemm_f32_by_name(knob);
    return rvv_pick_gemm_f32_by_name("auto");
}

__attribute__((visibility("default"))) rvv_gemm_fn rvv_pick_gemm_f16(void) {
    const char *knob = rvv_kernel_knob();
    if (knob && *knob) return rvv_pick_gemm_f16_by_name(knob);
    return rvv_pick_gemm_f16_by_name("auto");
}

/* --------------------------------------------------- per-dtype factories */

static rvcl_status_t make_rvv_kernel(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, rvv_gemm_fn gemm, size_t elt) {
    rvcl_status_t st = rvcl_kernel_alloc(out, desc, sizeof(rvv_matmul_t));
    if (st != RVCL_SUCCESS) return st;
    rvv_matmul_t *rk = (rvv_matmul_t *)*out;
    rk->base.vt = &g_rvv_vtable;
    rk->gemm = gemm;
    rk->elt = elt;
    return RVCL_SUCCESS;
}

static rvcl_status_t create_f32(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    /* a NULL pick (invalid knob or a forced-but-unavailable path) must
     * fail creation loudly, never bind a NULL micro-kernel */
    rvv_gemm_fn gemm = rvv_pick_gemm_f32();
    if (!gemm) return RVCL_ERROR_INVALID_ARGUMENT;
    return make_rvv_kernel(out, desc, gemm, sizeof(float));
}

static rvcl_status_t create_f16(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    rvv_gemm_fn gemm = rvv_pick_gemm_f16();
    if (!gemm) return RVCL_ERROR_INVALID_ARGUMENT;
    return make_rvv_kernel(out, desc, gemm, 2);
}

/* pack entry used by run(): dispatch to the shared panel packer */

/* ------------------------------------------------------- registration */

static rvcl_status_t register_one(const char *name, rvcl_dtype_t da,
        rvcl_dtype_t db, rvcl_dtype_t dacc, rvcl_dtype_t dc,
        rvcl_feature_id_t f0, rvcl_feature_id_t f1,
        rvcl_kernel_create_fn create) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));

    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = da;
    d.dtype_b = db;
    d.dtype_acc = dacc;
    d.dtype_c = dc;
    d.quant_support = RVCL_QUANT_NONE;
    d.required_features[0] = f0;
    d.required_features[1] = f1;
    d.num_required_features = f1 ? 2 : 1;
    d.alignment_a = d.alignment_b = d.alignment_c = 4;

    return rvcl_register_kernel(&d, create);
}

RVCL_API rvcl_status_t rvcl_rvv_register(void) {
    /* f32: RVV only */
    rvcl_status_t st = register_one("rvcl_matmul_f32_rvv", RVCL_DTYPE_FP32,
            RVCL_DTYPE_FP32, RVCL_DTYPE_FP32, RVCL_DTYPE_FP32, RVCL_FEATURE_RVV,
            RVCL_FEATURE_NONE, create_f32);
    if (st != RVCL_SUCCESS) return st;

    /* f16: RVV + Zvfh (habit rule: storage f16, accumulate fp32) */
    return register_one("rvcl_matmul_f16_rvv", RVCL_DTYPE_FP16, RVCL_DTYPE_FP16,
            RVCL_DTYPE_FP32, RVCL_DTYPE_FP32, RVCL_FEATURE_RVV,
            RVCL_FEATURE_FP16, create_f16);
}
