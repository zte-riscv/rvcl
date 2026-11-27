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

/* Pure data movement. B[k][n] is repacked so that each instruction of the
 * matrix extension consumes one 128-byte tile: within a tile, the 8 K
 * values of one output column are contiguous. Tail tiles are zero-filled
 * explicitly, so out-of-range multiply-accumulate lanes never disturb the
 * valid outputs — the packer never relies on pre-zeroed destination
 * memory. The same routine serves the in-run fallback and the host-facing
 * pre-pack path (kernel pack_b): layout depends only on the shapes.
 *
 * Tile order is n-tile outer, k-tile inner, matching the micro-kernel's
 * K loop, which then walks the buffer linearly.
 */

#include "ime_internal.hpp"

__attribute__((visibility("default"))) size_t ime_packed_b_bytes(
        int64_t k_dim, int64_t n, size_t elt) {
    if (k_dim <= 0 || n <= 0 || elt == 0) return 0;
    /* quotient/remainder form: (dim + 7) can overflow near INT64_MAX */
    const uint64_t n_t = (uint64_t)(n / IME_N_TILE + (n % IME_N_TILE != 0));
    const uint64_t k_t
            = (uint64_t)(k_dim / IME_K_TILE + (k_dim % IME_K_TILE != 0));
    uint64_t tiles = n_t * k_t;
    if (n_t != 0 && tiles / n_t != k_t) return 0; /* tile-count overflow */
    uint64_t bytes = tiles * (uint64_t)IME_TILE_ELT;
    if (tiles != 0 && bytes / tiles != (uint64_t)IME_TILE_ELT) return 0;
    if (bytes > SIZE_MAX / elt) return 0; /* byte-count overflow */
    return (size_t)(bytes * elt);
}

__attribute__((visibility("default"))) int64_t ime_packed_b_index(
        int64_t n_dim, int64_t k_dim, int64_t k, int64_t n) {
    if (n_dim <= 0 || k_dim <= 0) return -1;
    if (k < 0 || n < 0 || k >= k_dim || n >= n_dim) return -1;
    const int64_t k_tiles = k_dim / IME_K_TILE + (k_dim % IME_K_TILE != 0);
    return (n / IME_N_TILE * k_tiles + k / IME_K_TILE) * IME_TILE_ELT
            + (n % IME_N_TILE) * IME_K_TILE + (k % IME_K_TILE);
}

__attribute__((visibility("default"))) rvcl_status_t ime_pack_b_tiles(
        const rvcl_tensor_t *src, rvcl_tensor_t *dst, int64_t m, int64_t n,
        int64_t k_dim, size_t elt) {
    (void)m; /* the M dimension does not participate in B's layout */
    if (!src || !dst || !src->data || !dst->data)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (src->desc.dtype == RVCL_DTYPE_AUTO
            || dst->desc.dtype == RVCL_DTYPE_AUTO)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (src->desc.dtype != RVCL_DTYPE_FP16) return RVCL_UNSUPPORTED_DTYPE_COMBO;
    /* Destination is the pack workspace: imported either as a raw byte view
     * or as an element view of the same buffer. */
    if (dst->desc.dtype != RVCL_DTYPE_UINT8
            && dst->desc.dtype != RVCL_DTYPE_FP16)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    if (n <= 0 || k_dim <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    if (elt != sizeof(uint16_t)) return RVCL_ERROR_INVALID_ARGUMENT;
    const size_t bytes = ime_packed_b_bytes(k_dim, n, elt);
    if (bytes == 0) return RVCL_ERROR_OUT_OF_MEMORY; /* size overflow */

    /* FP16 is validated just above (elt == sizeof(uint16_t)), so both the
     * source and destination are element views of 16-bit values and the
     * tile fill is a plain copy: a per-element memcpy/memset here would
     * dominate the pack of a large B. */
    const uint16_t *sb = (const uint16_t *)src->data;
    uint16_t *db = (uint16_t *)dst->data;
    const int64_t n_t = n / IME_N_TILE + (n % IME_N_TILE != 0);
    const int64_t k_t = k_dim / IME_K_TILE + (k_dim % IME_K_TILE != 0);

    /* nt -> kt -> nn -> kk: the destination tile's 64 elements are written
     * contiguously; tail positions are zeroed so every element of the
     * buffer is written on every call. */
    for (int64_t nt = 0; nt < n_t; nt++) {
        for (int64_t kt = 0; kt < k_t; kt++) {
            for (int64_t nn = 0; nn < IME_N_TILE; nn++) {
                for (int64_t kk = 0; kk < IME_K_TILE; kk++) {
                    const int64_t gn = nt * IME_N_TILE + nn;
                    const int64_t gk = kt * IME_K_TILE + kk;
                    const int64_t di = (nt * k_t + kt) * IME_TILE_ELT
                            + nn * IME_K_TILE + kk;
                    db[di] = (gn < n && gk < k_dim) ? sb[gk * n + gn] : 0;
                }
            }
        }
    }
    return RVCL_SUCCESS;
}
