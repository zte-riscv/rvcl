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

/* FP4 block-scale matmul: A/B are E2M1 4-bit elements, nibble-packed (even
 * index = low nibble, 2 elements per byte). Each K block carries a shared
 * scale byte — NVFP4 uses block 16 with an E4M3 scale, MXFP4 uses block 32
 * with an E8M0 scale. The element dequantization is
 *     a[i][p]  -> UP_e2m1(a_nibble) * UP_scale(sa[i][p/blk])
 *     b[p][j]  -> UP_e2m1(b_byte)   * UP_scale(sb[p/blk][j])
 * so C[i][j] = bias[j] + sum_p (a-scale) * (b-scale). A is unpacked on the
 * fly from its nibble storage; B is pre-unpacked to one byte per element by
 * the panel packer, so the packed panel holds the raw E2M1 encodings that
 * UP_e2m1 reads directly. Both upconversions are the same lookup-table
 * infrastructure the golden shares (rvv_fp4.hpp, rvv_e8m0.hpp, rvv_fp8.hpp),
 * so golden and kernel agree bit-for-bit on the same bytes.
 *
 * Scale layouts: sa is [m][K/blk] row-major; sb is [K/blk][n] row-major,
 * accessed as sb[(p/blk)*n + j]. Each K term folds both scales into the B
 * value and accumulates with a single fused multiply-add,
 * acc = fma(A, B*ascale*bscale, acc) — the exact grouping the golden uses,
 * so the accumulate order (bias first, ascending k) is bit-identical on
 * finite data. K accumulates in ascending order. */

#include "rvv_e8m0.hpp"
#include "rvv_fp4.hpp"
#include "rvv_fp8.hpp"
#include "rvv_internal.hpp"

#include <math.h>

namespace {

/* One shared loop body, parameterized by the E2M1 element upconversion and
 * the scale upconversion (E4M3 for NVFP4, E8M0 for MXFP4). A is nibble-
 * packed (2 elements/byte), so the A element is unpacked before lookup; B
 * was pre-unpacked to a byte per element by the packer. */
template <float (*UP_ELEM)(uint8_t), float (*UP_SCALE)(uint8_t)>
void rvv_gemm_fp4_c_impl(const void *a_in, const void *b_packed,
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
                const float ascale = UP_SCALE(sa[i * nblk + pblk]);
                const float bscale = UP_SCALE(sb[pblk * n + j]);
                /* Fold both scales into the B value and accumulate with one
                 * fused multiply-add, matching the golden's grouping so the
                 * lone rounding per K step is bit-identical. */
                const float bscl = ascale * bscale;
                const uint8_t an = rvv_fp4_nibble_get(a, i * k_dim + p);
                acc = fmaf(UP_ELEM(an), UP_ELEM(bcol[p * RVV_PACK_NC]) * bscl,
                        acc);
            }
            c[i * n + j] = acc;
        }
    }
}

} // namespace

__attribute__((visibility("default"))) void rvv_gemm_nvfp4_c(const void *a,
        const void *b_packed, const void *sa, const void *sb, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim, int64_t blk,
        const rvcl_work_range_t *r) {
    rvv_gemm_fp4_c_impl<rvv_e2m1_to_f32, rvv_fp8_e4m3_to_f32>(
            a, b_packed, sa, sb, bias, c, m, n, k_dim, blk, r);
}

__attribute__((visibility("default"))) void rvv_gemm_mxfp4_c(const void *a,
        const void *b_packed, const void *sa, const void *sb, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim, int64_t blk,
        const rvcl_work_range_t *r) {
    rvv_gemm_fp4_c_impl<rvv_e2m1_to_f32, rvv_e8m0_to_f32>(
            a, b_packed, sa, sb, bias, c, m, n, k_dim, blk, r);
}
