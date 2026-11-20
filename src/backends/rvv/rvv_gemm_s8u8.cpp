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
 * rvv_gemm_s8u8.cpp — INT8 (signed/unsigned byte) GEMM C-portable kernels
 *
 * Tuple (s8, s8, s32, s32) and (u8, u8, s32, s32): storage is one byte per
 * element, accumulation and output are INT32. Both are integer-exact: each
 * product of two bytes is widened to int32 and accumulated in ascending K
 * order (bias first), bit-identical to the golden and the JIT kernel. No
 * requantization/quantization is performed here (v0.5 INT8 line). The two
 * dtypes differ only in the signedness of the load, so one templated body
 * serves both. The JIT counterpart lives in jit/rvv_gemm_s8u8_jit.cpp.
 ******************************************************************************/

#include "rvv_internal.hpp"

#include <stdint.h>

namespace {

template <typename T> /* int8_t or uint8_t */
void rvv_gemm_s8u8_c_impl(const void *a_in, const void *b_packed,
        const void *bias_in, void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const T *a = (const T *)a_in;
    const T *bp = (const T *)b_packed;
    const int32_t *bias = (const int32_t *)bias_in;
    int32_t *c = (int32_t *)c_in;
    (void)m;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j = r->n_begin; j < r->n_end; j++) {
            const int64_t panel = j / RVV_PACK_NC;
            const int64_t in = j - panel * RVV_PACK_NC;
            const T *bcol = bp + (panel * k_dim) * RVV_PACK_NC + in;
            int32_t acc = bias ? bias[j] : 0;
            for (int64_t p = 0; p < k_dim; p++)
                acc += (int32_t)a[i * k_dim + p]
                        * (int32_t)bcol[p * RVV_PACK_NC];
            c[i * n + j] = acc;
        }
    }
}

} // namespace

__attribute__((visibility("default"))) void rvv_gemm_s8_c(const void *a,
        const void *b_packed, const void *bias, void *c, int64_t m, int64_t n,
        int64_t k_dim, const rvcl_work_range_t *r) {
    rvv_gemm_s8u8_c_impl<int8_t>(a, b_packed, bias, c, m, n, k_dim, r);
}

__attribute__((visibility("default"))) void rvv_gemm_u8_c(const void *a,
        const void *b_packed, const void *bias, void *c, int64_t m, int64_t n,
        int64_t k_dim, const rvcl_work_range_t *r) {
    rvv_gemm_s8u8_c_impl<uint8_t>(a, b_packed, bias, c, m, n, k_dim, r);
}
