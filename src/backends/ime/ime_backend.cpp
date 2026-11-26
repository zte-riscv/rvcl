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

/* IME backend adapter: f16 path selection, host gate and registry
 * integration.
 *
 * The knob RVCL_IME_KERNEL (auto | c | ime, read once per process) resolves
 * through the path table in ime_paths.hpp; an unknown name or a
 * forced-but-unavailable path fails loudly (stderr reason, NULL return),
 * never silently binds a different path.
 *
 * The host gate (ime_instruction_path_available) answers what the host can
 * really execute: the instruction path is compiled in and the capability
 * snapshot reports the floating-point matrix extension. Detection lives in
 * the runtime capability layer and is conservative (any failure reports
 * unavailable), because executing the instruction on a core without it
 * raises an illegal instruction fault. The gate reads the host snapshot,
 * not the served one, so a test-injected feature can never make the
 * instruction path run where the host cannot execute it.
 *
 * The adapter mirrors the RVV backend: one registry entry per build flavor
 * (hardware-bound on RISC-V, portable C otherwise), one shared vtable, and a
 * factory that binds the knob-aware selector, so what dispatch runs and what
 * a kernel-only benchmark measures are the same function. */

#include "ime_internal.hpp"
#include "ime_paths.hpp"
#include "rvcl/capability.h"

#include "../../common/dtype_traits.hpp"
#include "../../runtime/capability_internal.h"
#include "../../runtime/cost_model.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ------------------------------------------------------------- host gate */

__attribute__((visibility("default"))) int ime_instruction_path_available(
        void) {
#if defined(__riscv) && defined(__riscv_vector)
    /* The snapshot is built once per process and cached by the capability
     * layer; the process affinity mask is expected to be pinned before the
     * first query. */
    return rvcl_capability_has(rvcl_capability_host(), RVCL_FEATURE_IME);
#else
    return 0; /* instruction path not compiled in on this target */
#endif
}

/* ------------------------------------------------- micro-kernel selection */

/* RVCL_IME_KERNEL knob, read once per process. Values: auto | c | ime.
 * An empty value counts as unset, matching the RVV knob's behavior. */
static const char *ime_kernel_knob(void) {
    static const char *knob = getenv("RVCL_IME_KERNEL");
    return knob;
}

__attribute__((visibility("default"))) ime_gemm_fn ime_pick_gemm_f16_by_name(
        const char *name) {
    if (!name) {
        fprintf(stderr,
                "rvcl: f16 kernel path is NULL (want auto, c or ime)\n");
        return NULL;
    }
    return ime_path_by_name(&ime_f16_paths, name);
}

__attribute__((visibility("default"))) ime_gemm_fn ime_pick_gemm_f16(void) {
    const char *knob = ime_kernel_knob();
    if (knob && *knob) return ime_pick_gemm_f16_by_name(knob);
    return ime_pick_gemm_f16_by_name("auto");
}

/* ------------------------------------------------------- instance state */

typedef struct {
    rvcl_kernel_t base; /* must be first */
    int64_t m, n, k;
    ime_gemm_fn gemm;
} ime_matmul_t;

/* ------------------------------------------------------------- lifecycle */

/* Shared by both entries: the dtype tuple, the shape and the quantization
 * contract are identical; only the entry's required features differ, and
 * dispatch's capability stage enforces those. */
static rvcl_status_t ime_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)cap;
    const rvcl_matmul_desc_t *d = &desc->matmul;
    if (d->m <= 0 || d->n <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    if (d->quant.enabled) return RVCL_UNSUPPORTED;
    if (d->dtype_a != RVCL_DTYPE_FP16 || d->dtype_b != RVCL_DTYPE_FP16
            || d->dtype_c != RVCL_DTYPE_FP32)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* AUTO is a request-side marker; the semantic layer resolves it before
     * dispatch, but a by-name drive may still see the raw value. */
    if (d->dtype_acc != RVCL_DTYPE_FP32 && d->dtype_acc != RVCL_DTYPE_AUTO)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    return RVCL_SUCCESS;
}

static rvcl_status_t ime_configure(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
        rvcl_workspace_req_t *ws_req, int *num_ws) {
    ime_matmul_t *rk = (ime_matmul_t *)k;
    const rvcl_matmul_desc_t *d = &desc->matmul;
    if (d->m <= 0 || d->n <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    rk->m = d->m;
    rk->n = d->n;
    rk->k = d->k;

    /* Slot 0: packed-B tile buffer. Lifetime KERNEL so the host may pack
     * once via rvcl_kernel_pack_b() and hold the tiles across runs of the
     * same B (the host guarantees B stays unchanged for that span). */
    const size_t bytes = ime_packed_b_bytes(d->k, d->n, sizeof(uint16_t));
    if (bytes == 0) return RVCL_ERROR_OUT_OF_MEMORY; /* size overflow */
    if (ws_req) {
        ws_req[IME_WS_PACKED_B] = rvcl_workspace_req_t {
                .size = (int64_t)bytes,
                .alignment = 64,
                .lifetime = RVCL_WS_LIFETIME_KERNEL,
        };
    }
    if (num_ws) *num_ws = 1;
    return RVCL_SUCCESS;
}

/* pack_b: pre-packs the B tiles for rvcl_kernel_pack_b(); the instance
 * carries the configured shape, so the layout needs only that. */
static rvcl_status_t ime_pack_b(
        rvcl_kernel_t *k, const rvcl_tensor_t *src, rvcl_tensor_t *dst) {
    ime_matmul_t *rk = (ime_matmul_t *)k;
    if (!rk || rk->m <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    return ime_pack_b_tiles(src, dst, rk->m, rk->n, rk->k, sizeof(uint16_t));
}

static rvcl_status_t ime_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *r) {
    const ime_matmul_t *rk = (const ime_matmul_t *)k;
    const rvcl_tensor_t *a = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_A);
    const rvcl_tensor_t *b = rvcl_tensor_pack_get(pack, RVCL_TENSOR_SRC_B);
    const rvcl_tensor_t *c = rvcl_tensor_pack_get(pack, RVCL_TENSOR_DST);
    const rvcl_tensor_t *bias = rvcl_tensor_pack_get(pack, RVCL_TENSOR_BIAS);
    const rvcl_tensor_t *ws = rvcl_tensor_pack_get(
            pack, (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + IME_WS_PACKED_B));
    /* ws is required either way: the in-run packing target, or the
     * endorsed pre-packed buffer (flag set but no workspace is a broken
     * host pack — fail loudly, never read a missing buffer). */
    if (!a || !b || !c || !ws) return RVCL_ERROR_INVALID_ARGUMENT;

    if (!pack->src_b_packed) {
        const rvcl_status_t st = ime_pack_b_tiles(
                b, (rvcl_tensor_t *)ws, rk->m, rk->n, rk->k, sizeof(uint16_t));
        if (st != RVCL_SUCCESS) return st;
    }

#if defined(__riscv_vector)
    /* Defensive re-gate: an injected snapshot may select an entry that
     * bound the instruction path, but the instruction must never run on a
     * host without the extension. */
    if (rk->gemm == ime_gemm_f16_ime && !ime_instruction_path_available())
        return RVCL_UNSUPPORTED;
#endif
    rk->gemm(a->data, ws->data, bias ? bias->data : NULL, c->data, rk->m,
            rk->n, rk->k, r);
    return RVCL_SUCCESS;
}

static void ime_destroy(rvcl_kernel_t *k) {
    (void)k;
}

static const rvcl_kernel_vtable_t g_ime_vtable = {
        .validate = ime_validate,
        .configure = ime_configure,
        .pack_a = NULL, /* A consumed unpacked; the micro-kernel stages tiles */
        .pack_b = ime_pack_b,
        .run = ime_run,
        .destroy = ime_destroy,
};

/* ------------------------------------------------------------- factory */

static rvcl_status_t make_ime_kernel(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc, ime_gemm_fn gemm) {
    rvcl_status_t st = rvcl_kernel_alloc(out, desc, sizeof(ime_matmul_t));
    if (st != RVCL_SUCCESS) return st;
    ime_matmul_t *rk = (ime_matmul_t *)*out;
    rk->base.vt = &g_ime_vtable;
    rk->gemm = gemm;
    return RVCL_SUCCESS;
}

/* The hardware-bound entry must be creatable only when the effective
 * capability snapshot says the extension is usable: dispatch's feature
 * filter already enforces this, but a by-name drive reaches the factory
 * directly and must fail the same way instead of binding a kernel that
 * cannot run. The selector's auto policy binds the instruction path when
 * the host allows it and the portable C path otherwise; a forced-but-
 * unavailable path returns NULL and the creation fails loudly. */
static rvcl_status_t create_ime(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
#if defined(__riscv)
    if (!rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_IME))
        return RVCL_UNSUPPORTED;
#endif
    ime_gemm_fn gemm = ime_pick_gemm_f16();
    if (!gemm) return RVCL_UNSUPPORTED; /* the selector already explained */
    return make_ime_kernel(out, desc, gemm);
}

/* ---------------------------------------------------------- registration */

static rvcl_status_t register_ime(const char *name, rvcl_vlen_tier_t tier,
        rvcl_feature_id_t f0, rvcl_feature_id_t f1) {
    rvcl_kernel_desc_t d;
    memset(&d, 0, sizeof(d));
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", name);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = RVCL_DTYPE_FP16;
    d.dtype_b = RVCL_DTYPE_FP16;
    d.dtype_acc = RVCL_DTYPE_FP32;
    d.dtype_c = RVCL_DTYPE_FP32;
    d.quant_support = RVCL_QUANT_NONE;
    d.assert_level = rvcl_dtype_traits(RVCL_DTYPE_FP16).assert_level_default;
    d.required_features[0] = f0;
    d.required_features[1] = f1;
    int nfeat = 0;
    if (f0 != RVCL_FEATURE_NONE) {
        nfeat = 1;
        if (f1 != RVCL_FEATURE_NONE) nfeat = 2;
    }
    d.num_required_features = nfeat;
    d.preferred_m = IME_M_TILE;
    d.preferred_n = IME_N_TILE;
    d.preferred_k = IME_K_TILE;
    d.alignment_a = 2;
    d.alignment_b = 2;
    d.alignment_c = 4;
    d.vlen_tier = tier;
    d.cost_fn = rvcl_cost_matmul_ime;
    return rvcl_register_kernel(&d, create_ime);
}

__attribute__((visibility("default"))) rvcl_status_t rvcl_ime_register(void) {
#if defined(__riscv)
    return register_ime("rvcl_matmul_f16_ime_1024", RVCL_VLEN_TIER_1024,
            RVCL_FEATURE_IME, RVCL_FEATURE_FP16);
#else
    return RVCL_SUCCESS; /* non-RISC-V builds register the portable entry */
#endif
}

__attribute__((visibility("default"))) rvcl_status_t
rvcl_ime_register_portable(void) {
#if defined(__riscv)
    return RVCL_SUCCESS; /* RISC-V builds register the hardware-bound entry */
#else
    return register_ime("rvcl_matmul_f16_ime_c", RVCL_VLEN_TIER_ANY,
            RVCL_FEATURE_NONE, RVCL_FEATURE_NONE);
#endif
}
