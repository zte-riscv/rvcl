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
 * kernel.h — kernel ABI (Layer 2, project-stable interface)
 *
 * Kernel contract (PLAN §4.3): a kernel is metadata + feature requirements
 * + packing contract + execution functions with the ACL-style three-phase
 * lifecycle:
 *
 *   validate(desc, capability)  — side-effect-free probe (dispatch time)
 *   configure(desc) -> ws_req   — one per dispatch result; reports workspace
 *   pack_a / pack_b             — packing, callable independently of run
 *   run(tensor_pack, work_range)— executes; never mallocs, never threads
 ******************************************************************************/

#ifndef RVCL_KERNEL_H
#define RVCL_KERNEL_H

#include "rvcl/capability.h"
#include "rvcl/memory.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* -------------------------------------------------------- kernel metadata */

#define RVCL_KERNEL_NAME_MAX 64

/* Operation descriptor (completed below); forward-declared so the cost
 * callback type can name it before its full definition. */
typedef union rvcl_op_desc rvcl_op_desc_t;

/* --------------------------------------------------------------- ABI
 *
 * Kernel ABI version for out-of-tree kernel/plugin registration. The
 * descriptor (rvcl_kernel_desc_t) is copied byte-for-byte across the
 * boundary, so a plugin built against a differently-laid-out header
 * (e.g. a shorter or reordered struct) would read memory incorrectly —
 * undefined behavior the C interface cannot verify from a pointer alone.
 * rvcl_kernel_abi_version() returns the version this library was built
 * with; a plugin's constructor should compare it against its own
 * RVCL_KERNEL_ABI_VERSION and refuse to register (reporting to stderr)
 * when they differ. The check is advisory — the library cannot inspect
 * the plugin's struct layout — so every plugin is expected to honor it.
 *
 * Desc layout discipline: the descriptor only ever grows by appending
 * fields at the end (offset-preserving additions, like assert_level), and
 * ANY layout change, including a merely appended field, must bump
 * RVCL_KERNEL_ABI_VERSION so old plugins stop registering rather than
 * passing a short struct.
 */
#define RVCL_KERNEL_ABI_VERSION 1

/* Returns the kernel ABI version of this library build. */
RVCL_API unsigned rvcl_kernel_abi_version(void);

/* ------------------------------------------------------------ cost model */

/*
 * Cost estimation callback (RFC-0002 cost). Contract:
 *   - input: the operation description and the capability snapshot;
 *   - output: non-negative integer (estimated cycles), smaller = better;
 *     0 means "no cost information";
 *   - side-effect-free and thread-safe (dispatch may call concurrently);
 *   - the value only orders candidates — it never affects correctness.
 * The registry stores the pointer but never calls it; dispatch calls it
 * once per surviving candidate and compares with plain unsigned `<`.
 */
typedef uint64_t (*rvcl_kernel_cost_fn)(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap);

/*
 * Canonical zero-cost implementation: "no cost information" (RFC-0002
 * FR4.2). Kernels without a real cost model must bind this so descriptor
 * iteration never sees a NULL callback.
 */
RVCL_API uint64_t rvcl_kernel_cost_zero(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap);

/*
 * Public cost hint for a matmul backend category, the out-of-tree counterpart
 * of the internal category wrappers. A vendor kernel with no independent cost
 * model binds a thin rvcl_kernel_cost_fn that forwards to this helper with a
 * fixed coefficient, letting the framework own the shape heuristic and the
 * fixed-point scoring convention while the coefficient stays vendor-owned.
 *
 * coeff_256 is a fixed-point multiplier (256 = 1.0x): 256 scores the same as
 * the built-in cheap vector (rvv) category, below 256 scores no higher than
 * that category, above 256 scores more expensive. The value only orders
 * candidates — it never affects correctness — and is clamped to >= 1. `cap`
 * is currently advisory (no tier gain for a non-VLEN-scaled vendor tile); it
 * is passed for symmetry so a vendor that needs a VLEN adjustment can read
 * the host's RVV width itself.
 */
RVCL_API uint64_t rvcl_cost_matmul_hint(const rvcl_op_desc_t *desc,
        const rvcl_capability_t *cap, uint64_t coeff_256);

typedef enum {
    RVCL_QUANT_NONE = 0,
    RVCL_QUANT_PER_TENSOR,
    RVCL_QUANT_PER_CHANNEL, /* per output channel */
} rvcl_quant_support_t;

/*
 * Numeric assertion level for a kernel's outputs, the metadata bit behind
 * the v0.2 tolerance infrastructure:
 *   RVCL_ASSERT_BITEXACT (0) — outputs must match the golden bit-exactly.
 *       This is the zero-init default, so every memset-based registration
 *       path (including the reference backend) is bit-exact without code.
 *   RVCL_ASSERT_TOLERANT — outputs are compared within the tolerance class
 *       the dtype trait table gives this kernel's datatype (e.g. fp16's
 *       1e-3 relative). The tolerance VALUE never lives in the descriptor;
 *       TOLERANT references the trait table by dtype.
 * A per-operator override (e.g. a requantization grade) is a reserved
 * extension slot and is not part of v0.1.
 */
typedef enum {
    RVCL_ASSERT_BITEXACT = 0,
    RVCL_ASSERT_TOLERANT,
} rvcl_assert_level_t;

typedef struct {
    char name[RVCL_KERNEL_NAME_MAX]; /* rvcl_<op>_<dtype>_<tile>_<backend> */

    rvcl_operation_t operation;

    /* (A, B, acc, C) tuple — the unit of datatype filtering (PLAN §4.5.2). */
    rvcl_dtype_t dtype_a;
    rvcl_dtype_t dtype_b;
    rvcl_dtype_t dtype_acc;
    rvcl_dtype_t dtype_c;
    rvcl_quant_support_t quant_support;

    /* Feature requirements; count <= RVCL_MAX_FEATURES. */
    rvcl_feature_id_t required_features[RVCL_MAX_FEATURES];
    int num_required_features;

    /* Shape constraints (inclusive bounds; 0 = unbounded). */
    int64_t min_m, max_m, min_n, max_n, min_k, max_k;
    /* Preferred tile for cost-model hints. */
    int64_t preferred_m, preferred_n, preferred_k;

    /* Alignment requirements in bytes (1 = none). */
    int64_t alignment_a, alignment_b, alignment_c;

    /* Required VLEN tier (types.h): RVCL_VLEN_TIER_ANY = no requirement;
     * otherwise the kernel is excluded when the host tier is lower. Lets
     * dispatch filter VLEN-specific variants without ISA-specific code. */
    rvcl_vlen_tier_t vlen_tier;

    /* Cost estimation callback; non-NULL is enforced at registration.
     * Bind rvcl_kernel_cost_zero() when the kernel has no cost model. */
    rvcl_kernel_cost_fn cost_fn;

    /* Numeric assertion level (see rvcl_assert_level_t above); zero-init
     * means RVCL_ASSERT_BITEXACT. Appended after cost_fn so pre-existing
     * fields keep their byte offsets (offset-preserving v0.2 addition). */
    rvcl_assert_level_t assert_level;
} rvcl_kernel_desc_t;

/* ------------------------------------------------------- operation descs */

/* Quantization context (PLAN §4.5.3); zero-init for non-quantized paths. */
typedef struct {
    int enabled; /* 0 for non-quantized */
    float scale_a, scale_b, scale_c;
    int32_t zp_a, zp_b, zp_c;
} rvcl_quant_ctx_t;

typedef struct {
    int64_t m, n, k;
    rvcl_dtype_t dtype_a, dtype_b, dtype_acc, dtype_c;
    rvcl_dtype_t dtype_bias; /* RVCL_DTYPE_AUTO when no bias */
    rvcl_quant_ctx_t quant; /* zero-init for non-quantized */
} rvcl_matmul_desc_t;

/*
 * Matrix-vector product y[m] = A[m,k] x[k] (+ bias[m]). The output is a
 * length-m vector, so there is no n dimension. Quantization is not part of
 * the v0.x GEMV path: validate() rejects a quantized request, and the
 * quantization context is kept for layout symmetry with the other
 * descriptors.
 */
typedef struct {
    int64_t m, k;
    rvcl_dtype_t dtype_a, dtype_b, dtype_acc, dtype_c;
    rvcl_dtype_t dtype_bias; /* RVCL_DTYPE_AUTO when no bias */
    rvcl_quant_ctx_t quant; /* zero-init for non-quantized */
} rvcl_gemv_desc_t;

typedef union rvcl_op_desc {
    rvcl_matmul_desc_t matmul;
    rvcl_gemv_desc_t gemv;
} rvcl_op_desc_t;

/* ------------------------------------------------------------ kernel vtable */

/*
 * Opaque kernel instance state (configure results, packing plan, workspace
 * layout). Created by the runtime from a registry entry.
 */
typedef struct rvcl_kernel rvcl_kernel_t;

typedef struct {
    /* Side-effect-free probe. Called heavily at dispatch time.
     * Returns RVCL_SUCCESS when this kernel can serve the request. */
    rvcl_status_t (*validate)(
            const rvcl_op_desc_t *desc, const rvcl_capability_t *cap);

    /* Prepares execution metadata and reports workspace demands.
     * Called once per dispatch result; run() may be called many times. */
    rvcl_status_t (*configure)(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
            rvcl_workspace_req_t *ws_req /* out, up to RVCL_MAX_WORKSPACES */,
            int *num_ws /* out */);

    /* Packing (independent of run; see PLAN §4.6). Phase 0 kernels are
     * unpacked, so these are optional (NULL = no packing needed).
     * pack_b fills the packing workspace ahead of run(); implementations
     * whose run() packs B itself must honor pack->src_b_packed (memory.h)
     * by skipping that in-run packing. */
    rvcl_status_t (*pack_a)(rvcl_kernel_t *k, const rvcl_tensor_t *src,
            rvcl_tensor_t *dst_packed);
    rvcl_status_t (*pack_b)(rvcl_kernel_t *k, const rvcl_tensor_t *src,
            rvcl_tensor_t *dst_packed);

    /* Executes [range] of the configured work. Consumes only imported
     * memory and workspace via the pack. Never mallocs, never threads.
     * Concurrent run() calls sharing one pack must pre-pack via
     * rvcl_kernel_pack_b() and set pack->src_b_packed when the kernel has
     * a pack_b step, otherwise the per-run packing writes race on the
     * shared packing workspace. */
    rvcl_status_t (*run)(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
            const rvcl_work_range_t *range);

    /* Releases instance state created by configure. */
    void (*destroy)(rvcl_kernel_t *k);
} rvcl_kernel_vtable_t;

/* Every kernel implementation exposes this factory symbol. */
typedef rvcl_status_t (*rvcl_kernel_create_fn)(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc);

/* The generic kernel header every instance starts with. */
struct rvcl_kernel {
    const rvcl_kernel_vtable_t *vt;
    rvcl_kernel_desc_t desc;
};

/* Helper for implementations: allocates + headers a new instance. */
RVCL_API rvcl_status_t rvcl_kernel_alloc(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, size_t instance_size);

/* Runtime-facing lifecycle wrappers. */
RVCL_API rvcl_status_t rvcl_kernel_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws);
RVCL_API rvcl_status_t rvcl_kernel_run(rvcl_kernel_t *k,
        const rvcl_tensor_pack_t *pack, const rvcl_work_range_t *range);
RVCL_API void rvcl_kernel_destroy(rvcl_kernel_t *k);

/*
 * Packs SRC_B into the kernel's packing workspace ahead of run()
 * (pack once, run many). dst_ws is the packing slot by convention: the
 * FIRST workspace configure() reported, fed back through the pack as
 * AUX_0. Returns RVCL_SUCCESS without touching anything when the kernel
 * has no packing step (vtable pack_b NULL, e.g. the reference backend)
 * — the caller must then leave pack->src_b_packed at 0 so run() keeps
 * its unpacked-B behavior. After a successful pack the caller endorses
 * the buffer by setting pack->src_b_packed = 1 (memory.h): while the
 * LIFETIME_KERNEL buffer is held, the host guarantees B stays unchanged
 * and every run() consumes the packed panels read-only instead of
 * packing per run.
 */
RVCL_API rvcl_status_t rvcl_kernel_pack_b(
        rvcl_kernel_t *k, const rvcl_tensor_t *src_b, rvcl_tensor_t *dst_ws);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_KERNEL_H */
