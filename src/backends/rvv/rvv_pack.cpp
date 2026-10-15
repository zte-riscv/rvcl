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
 * rvv_pack.cpp — B panel packing (plain row-major -> NC-wide column panels)
 *
 * Pure data movement, dtype-agnostic via element size. On RVV hosts the
 * copy loop vectorizes trivially (contiguous stores, strided loads); the
 * C loop below is portable and correct on every host, which is what the
 * x86 CI differential tests exercise (PLAN §6.2 item 11).
 *
 * This is the run()-side packer used through rvcl_kernel_pack_b(). The
 * same routine serves f32 and f16 kernels: layout depends only on shapes.
 ******************************************************************************/

#include "rvv_internal.hpp"

#include <string.h>

__attribute__((visibility("default"))) rvcl_status_t rvv_pack_b_panel(
        const rvcl_tensor_t *src, rvcl_tensor_t *dst, int64_t m, int64_t n,
        int64_t k_dim, size_t elt) {
    (void)m;
    if (!src || !dst || !dst->data) return RVCL_ERROR_INVALID_ARGUMENT;
    if (src->desc.dtype == RVCL_DTYPE_AUTO) return RVCL_ERROR_INVALID_ARGUMENT;

    const char *sb = (const char *)src->data;
    char *db = (char *)dst->data;
    const int64_t k = k_dim, nn = n;

    /* For each column, copy its K elements into the panel slot:
     * dst[(panel*k + row)*NC + in] = src[row*n + col]. Element-wise copy
     * keeps this correct for any elt (1/2/4 bytes). */
    for (int64_t col = 0; col < nn; col++) {
        const int64_t panel = col / RVV_PACK_NC;
        const int64_t in = col - panel * RVV_PACK_NC;
        for (int64_t row = 0; row < k; row++) {
            const size_t si = (size_t)(row * nn + col) * elt;
            const size_t di
                    = (size_t)((panel * k + row) * RVV_PACK_NC + in) * elt;
            memcpy(db + di, sb + si, elt);
        }
    }
    return RVCL_SUCCESS;
}
