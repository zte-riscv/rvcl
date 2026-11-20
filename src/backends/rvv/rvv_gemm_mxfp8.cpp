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

/* OCP MX block-scale matmul: A/B are FP8 (E4M3 or E5M2) bytes, each 32-wide
 * block of K carries a shared 1-byte E8M0 exponent scale, and the output is
 * fp32. The element dequantization is
 *     a[i][p]  -> UP_fp8(a[i][p]) * UP_e8m0(sa[i][p/32])
 *     b[p][j]  -> UP_fp8(b[p][j]) * UP_e8m0(sb[p/32][j])
 * so C[i][j] = bias[j] + sum_p (a-scale) * (b-scale). Both upconversions are
 * the same 256-entry f32 lookup-table infrastructure the golden and the JIT
 * kernels share (rvv_fp8.hpp, rvv_e8m0.hpp), so all three agree bit-for-bit
 * on the same bytes — including E8M0 0x00 (smallest subnormal 2^-127) and
 * 0xFF (NaN propagation). Scale layouts: sa is [m][K/blk] row-major; sb is
 * [K/blk][n] row-major, accessed as sb[(p/blk)*n + j].
 *
 * Each K term folds both E8M0 scales into the B value and accumulates with a
 * single fused multiply-add, acc = fma(A, B*ascale*bscale, acc). The combined
 * scale is an exact power of two, so B*ascale*bscale does not round, and the
 * lone rounding per K step matches the JIT's vfmacc_vf grouping bit-for-bit.
 * K accumulates in ascending order (bias first), bit-identical to the golden
 * and the JIT path. The JIT counterpart lives in jit/rvv_gemm_mxfp8_jit.cpp. */

#include "rvv_e8m0.hpp"
#include "rvv_fp8.hpp"
#include "rvv_internal.hpp"

#include <math.h>

namespace {

/* One shared loop body, parameterized by the FP8 dtype's upconversion. The
 * anonymous namespace gives it internal linkage; `static` would be
 * redundant here. */
template <float (*UP_FP8)(uint8_t)>
void rvv_gemm_mxfp8_c_impl(const void *a_in, const void *b_packed,
        const void *sa_in, const void *sb_in, const void *bias_in, void *c_in,
        int64_t m, int64_t n, int64_t k_dim, int64_t blk,
        const rvcl_work_range_t *r) {
    const uint8_t *a = (const uint8_t *)a_in;
    const uint8_t *bp = (const uint8_t *)b_packed;
    const uint8_t *sa = (const uint8_t *)sa_in;
    const uint8_t *sb = (const uint8_t *)sb_in;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    const int64_t nblk = k_dim / blk;
    (void)m;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j = r->n_begin; j < r->n_end; j++) {
            const int64_t panel = j / RVV_PACK_NC;
            const int64_t in = j - panel * RVV_PACK_NC;
            const uint8_t *bcol = bp + (panel * k_dim) * RVV_PACK_NC + in;
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k_dim; p++) {
                const int64_t pblk = p / blk;
                const float ascale = rvv_e8m0_to_f32(sa[i * nblk + pblk]);
                const float bscale = rvv_e8m0_to_f32(sb[pblk * n + j]);
                /* Fold both scales into the B value (exact power of two, so
                 * no rounding) and accumulate with one fused multiply-add,
                 * matching the JIT's vfmacc_vf grouping bit-for-bit. */
                const float bscl = ascale * bscale;
                acc = fmaf(UP_FP8(a[i * k_dim + p]),
                        UP_FP8(bcol[p * RVV_PACK_NC]) * bscl, acc);
            }
            c[i * n + j] = acc;
        }
    }
}

} // namespace

__attribute__((visibility("default"))) void rvv_gemm_mxfp8_e4m3_c(const void *a,
        const void *b_packed, const void *sa, const void *sb, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim, int64_t blk,
        const rvcl_work_range_t *r) {
    rvv_gemm_mxfp8_c_impl<rvv_fp8_e4m3_to_f32>(
            a, b_packed, sa, sb, bias, c, m, n, k_dim, blk, r);
}

__attribute__((visibility("default"))) void rvv_gemm_mxfp8_e5m2_c(const void *a,
        const void *b_packed, const void *sa, const void *sb, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim, int64_t blk,
        const rvcl_work_range_t *r) {
    rvv_gemm_mxfp8_c_impl<rvv_fp8_e5m2_to_f32>(
            a, b_packed, sa, sb, bias, c, m, n, k_dim, blk, r);
}
