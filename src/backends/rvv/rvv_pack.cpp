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

/* B panel packing: plain row-major -> NC-wide column panels. Pure data
 * movement, dtype-agnostic via element size. On RVV hosts the copy loop
 * vectorizes trivially (contiguous stores, strided loads); the C loop below
 * is portable and correct on every host. Both packing routes converge here:
 * run()'s in-run fallback and the host-facing pre-pack path
 * rvcl_kernel_pack_b() through the rvv pack_b vtable slot. The same routine
 * serves the f32 and f16 kernels: layout depends only on shapes.
 *
 * Sub-byte dtypes (the FP4 block-scale formats, 4 bits/element nibble-
 * packed at 2 elements per byte) are unpacked to one byte per element in
 * the panel: the micro-kernels read the packed panel as a byte per element,
 * so the nibble unpack happens once at pack time instead of in the inner K
 * loop. The element count `elt` stays 1 (byte granularity) for sizing; the
 * packing branches on the dtype's per-element bit width. */

#include "rvv_fp4.hpp"
#include "rvv_internal.hpp"

#include <string.h>

__attribute__((visibility("default"))) rvcl_status_t rvv_pack_b_panel(
        const rvcl_tensor_t *src, rvcl_tensor_t *dst, int64_t m, int64_t n,
        int64_t k_dim, size_t elt) {
    (void)m;
    if (!src || !dst || !dst->data) return RVCL_ERROR_INVALID_ARGUMENT;
    if (src->desc.dtype == RVCL_DTYPE_AUTO) return RVCL_ERROR_INVALID_ARGUMENT;

    const uint8_t *sb = (const uint8_t *)src->data;
    uint8_t *db = (uint8_t *)dst->data;
    const int64_t k = k_dim, nn = n;

    /* Sub-byte (FP4) source: unpack the nibble at element (row,col) into a
     * byte in the panel slot. The source element index is row*nn + col
     * (B[k][n] row-major, nibble-packed). */
    const int subbyte = rvcl_dtype_bits(src->desc.dtype) < 8;
    for (int64_t col = 0; col < nn; col++) {
        const int64_t panel = col / RVV_PACK_NC;
        const int64_t in = col - panel * RVV_PACK_NC;
        for (int64_t row = 0; row < k; row++) {
            const int64_t di = (panel * k + row) * RVV_PACK_NC + in;
            if (subbyte) {
                db[di] = rvv_fp4_nibble_get(sb, row * nn + col);
            } else {
                const size_t si = (size_t)(row * nn + col) * elt;
                memcpy(db + (size_t)di * elt, sb + si, elt);
            }
        }
    }
    return RVCL_SUCCESS;
}
