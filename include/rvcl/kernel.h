/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * kernel.h — kernel ABI (Layer 2, project-stable interface)
 *
 * SPDX-License-Identifier: Apache-2.0
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

typedef enum {
    RVCL_QUANT_NONE = 0,
    RVCL_QUANT_PER_TENSOR,
    RVCL_QUANT_PER_CHANNEL, /* per output channel */
} rvcl_quant_support_t;

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

typedef union {
    rvcl_matmul_desc_t matmul;
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
    rvcl_status_t (*validate)(const rvcl_op_desc_t *desc,
            const rvcl_capability_t *cap);

    /* Prepares execution metadata and reports workspace demands.
     * Called once per dispatch result; run() may be called many times. */
    rvcl_status_t (*configure)(rvcl_kernel_t *k, const rvcl_op_desc_t *desc,
            rvcl_workspace_req_t *ws_req /* out, up to RVCL_MAX_WORKSPACES */,
            int *num_ws /* out */);

    /* Packing (independent of run; see PLAN §4.6). Phase 0 kernels are
     * unpacked, so these are optional (NULL = no packing needed). */
    rvcl_status_t (*pack_a)(rvcl_kernel_t *k, const rvcl_tensor_t *src,
            rvcl_tensor_t *dst_packed);
    rvcl_status_t (*pack_b)(rvcl_kernel_t *k, const rvcl_tensor_t *src,
            rvcl_tensor_t *dst_packed);

    /* Executes [range] of the configured work. Consumes only imported
     * memory and workspace via the pack. Never mallocs, never threads. */
    rvcl_status_t (*run)(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
            const rvcl_work_range_t *range);

    /* Releases instance state created by configure. */
    void (*destroy)(rvcl_kernel_t *k);
} rvcl_kernel_vtable_t;

/* Every kernel implementation exposes this factory symbol. */
typedef rvcl_status_t (*rvcl_kernel_create_fn)(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc);

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
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req,
        int *num_ws);
RVCL_API rvcl_status_t rvcl_kernel_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *range);
RVCL_API void rvcl_kernel_destroy(rvcl_kernel_t *k);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_KERNEL_H */
