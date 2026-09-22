/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * memory.h — memory contract (tensor import / tensor pack / workspace)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Mirrors the ACL trio import_memory / ITensorPack / MemoryRequirements
 * (PLAN §4.10):
 *   - rvcl_tensor_init(): zero-copy import of host memory; RVCL never takes
 *     ownership and never allocates on behalf of the host.
 *   - rvcl_tensor_pack_t: slot-aggregated tensors for one run() call;
 *     decouples execution from object state so one kernel handle can run
 *     concurrently with different packs.
 *   - rvcl_workspace_req_t: workspace demands reported by configure() and
 *     allocated by the caller, then fed back via AUX slots.
 ******************************************************************************/

#ifndef RVCL_MEMORY_H
#define RVCL_MEMORY_H

#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------ tensor desc */

typedef struct {
    int ndims;
    int64_t dims[6];    /* dims[0] outermost ... dims[ndims-1] innermost */
    int64_t strides[6]; /* in elements; strides[ndims-1] must be 1 */
    rvcl_dtype_t dtype;
} rvcl_tensor_desc_t;

/* Imported tensor: metadata + borrowed host pointer (zero copy). */
typedef struct {
    rvcl_tensor_desc_t desc;
    void *data; /* host-owned; const-ness is expressed via the pack API */
} rvcl_tensor_t;

/*
 * Initializes a tensor from an external pointer. Only records metadata —
 * no allocation, no copy, no ownership transfer.
 * Rejects non-plain layouts (innermost stride != 1) with
 * RVCL_UNSUPPORTED per the layout negotiation rule (PLAN §4.10).
 */
RVCL_API rvcl_status_t rvcl_tensor_init(
        rvcl_tensor_t *t, const rvcl_tensor_desc_t *desc, void *external_ptr);

/* Total size of the tensor backing buffer in bytes. */
RVCL_API int64_t rvcl_tensor_bytes(const rvcl_tensor_t *t);

/* ------------------------------------------------------------ tensor pack */

typedef enum {
    RVCL_TENSOR_SRC_A = 0,
    RVCL_TENSOR_SRC_B,
    RVCL_TENSOR_BIAS,
    RVCL_TENSOR_DST,
    RVCL_TENSOR_AUX_0, /* workspace slots (up to 4) */
    RVCL_TENSOR_AUX_1,
    RVCL_TENSOR_AUX_2,
    RVCL_TENSOR_AUX_3,
    RVCL_TENSOR_SLOT_COUNT,
} rvcl_tensor_slot_t;

typedef struct {
    const rvcl_tensor_t *slots[RVCL_TENSOR_SLOT_COUNT];
    /* const tensors (inputs / workspace) vs mutable ones (dst). */
    int is_const[RVCL_TENSOR_SLOT_COUNT];
} rvcl_tensor_pack_t;

/* Resets a pack to empty (all slots NULL). */
RVCL_API void rvcl_tensor_pack_init(rvcl_tensor_pack_t *p);

/* Adds an input (const) tensor to a slot. */
RVCL_API rvcl_status_t rvcl_tensor_pack_add(
        rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot, const rvcl_tensor_t *t);

/* Adds an output (mutable) tensor to a slot. */
RVCL_API rvcl_status_t rvcl_tensor_pack_add_out(
        rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot, rvcl_tensor_t *t);

/* Accessors used by kernels. NULL when the slot is unset. */
RVCL_API const rvcl_tensor_t *rvcl_tensor_pack_get(
        const rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot);

/* --------------------------------------------------------------- workspace */

typedef enum {
    RVCL_WS_LIFETIME_RUN = 0, /* scratch for a single run() */
    RVCL_WS_LIFETIME_KERNEL, /* persistent across runs (e.g. packed weights) */
} rvcl_ws_lifetime_t;

typedef struct {
    int64_t size; /* bytes */
    int64_t alignment; /* bytes */
    rvcl_ws_lifetime_t lifetime;
} rvcl_workspace_req_t;

#define RVCL_MAX_WORKSPACES 8

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_MEMORY_H */
