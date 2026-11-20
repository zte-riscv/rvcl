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

/* Tuple (fp8, fp8, f32, f32): storage fp8 (one byte per element),
 * accumulation and output fp32. FP8 has no ratified base hardware
 * extension, so both kernels upconvert every byte through the shared
 * 256-entry f32 lookup tables in rvv_fp8.hpp — the same tables the golden
 * and the JIT kernels use, so all three agree bit-for-bit on FP8 inputs.
 * The JIT counterpart lives in jit/rvv_gemm_fp8_jit.cpp. K accumulates in
 * ascending order (bias first), bit-identical to the golden and the JIT
 * path. E4M3 and E5M2 differ only in the upconversion function, so a
 * single templated body serves both dtypes. */

#include "rvv_fp8.hpp"
#include "rvv_internal.hpp"

#include <string.h>

namespace {

/* One shared loop body, parameterized by the dtype's upconversion. The
 * anonymous namespace gives it internal linkage; `static` would be
 * redundant here. */
template <float (*UP)(uint8_t)>
void rvv_gemm_fp8_c_impl(const void *a_in, const void *b_packed,
        const void *bias_in, void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const uint8_t *a = (const uint8_t *)a_in;
    const uint8_t *bp = (const uint8_t *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    (void)m;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j = r->n_begin; j < r->n_end; j++) {
            const int64_t panel = j / RVV_PACK_NC;
            const int64_t in = j - panel * RVV_PACK_NC;
            const uint8_t *bcol = bp + (panel * k_dim) * RVV_PACK_NC + in;
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k_dim; p++)
                acc += UP(a[i * k_dim + p]) * UP(bcol[p * RVV_PACK_NC]);
            c[i * n + j] = acc;
        }
    }
}

} // namespace

__attribute__((visibility("default"))) void rvv_gemm_fp8_e4m3_c(const void *a,
        const void *b_packed, const void *bias, void *c, int64_t m, int64_t n,
        int64_t k_dim, const rvcl_work_range_t *r) {
    rvv_gemm_fp8_c_impl<rvv_fp8_e4m3_to_f32>(
            a, b_packed, bias, c, m, n, k_dim, r);
}

__attribute__((visibility("default"))) void rvv_gemm_fp8_e5m2_c(const void *a,
        const void *b_packed, const void *bias, void *c, int64_t m, int64_t n,
        int64_t k_dim, const rvcl_work_range_t *r) {
    rvv_gemm_fp8_c_impl<rvv_fp8_e5m2_to_f32>(
            a, b_packed, bias, c, m, n, k_dim, r);
}
