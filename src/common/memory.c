/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * memory.c — memory contract implementation
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#include "rvcl/memory.h"

#include <string.h>

rvcl_status_t rvcl_tensor_init(
        rvcl_tensor_t *t, const rvcl_tensor_desc_t *desc, void *external_ptr) {
    if (!t || !desc) return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->ndims <= 0 || desc->ndims > 6) return RVCL_ERROR_INVALID_ARGUMENT;
    if (rvcl_dtype_size(desc->dtype) == 0) return RVCL_ERROR_INVALID_ARGUMENT;

    /* Layout negotiation (PLAN §4.10): plain tensors only — innermost
     * stride must be 1. Non-blocking/exotic layouts are explicitly
     * rejected with RVCL_UNSUPPORTED, never silently accepted. */
    if (desc->strides[desc->ndims - 1] != 1) return RVCL_UNSUPPORTED;

    t->desc = *desc;
    t->data = external_ptr; /* borrowed; zero-copy by contract */
    return RVCL_SUCCESS;
}

int64_t rvcl_tensor_bytes(const rvcl_tensor_t *t) {
    if (!t || t->desc.ndims <= 0) return 0;
    int64_t els = 1;
    for (int i = 0; i < t->desc.ndims; i++)
        els *= t->desc.dims[i];
    return els * (int64_t)rvcl_dtype_size(t->desc.dtype);
}

void rvcl_tensor_pack_init(rvcl_tensor_pack_t *p) {
    memset(p, 0, sizeof(*p));
}

static rvcl_status_t pack_add(rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot,
        const rvcl_tensor_t *t, int is_const) {
    if (!p || !t) return RVCL_ERROR_INVALID_ARGUMENT;
    if (slot < 0 || slot >= RVCL_TENSOR_SLOT_COUNT)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (p->slots[slot]) return RVCL_ERROR_INVALID_ARGUMENT; /* slot taken */
    p->slots[slot] = t;
    p->is_const[slot] = is_const;
    return RVCL_SUCCESS;
}

rvcl_status_t rvcl_tensor_pack_add(
        rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot, const rvcl_tensor_t *t) {
    return pack_add(p, slot, t, 1);
}

rvcl_status_t rvcl_tensor_pack_add_out(
        rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot, rvcl_tensor_t *t) {
    return pack_add(p, slot, t, 0);
}

const rvcl_tensor_t *rvcl_tensor_pack_get(
        const rvcl_tensor_pack_t *p, rvcl_tensor_slot_t slot) {
    if (!p || slot < 0 || slot >= RVCL_TENSOR_SLOT_COUNT) return NULL;
    return p->slots[slot];
}
