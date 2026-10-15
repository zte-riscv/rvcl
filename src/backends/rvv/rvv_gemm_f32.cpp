/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * rvv_gemm_f32.cpp — FP32 GEMM micro-kernels (RVV intrinsics + C fallback)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Contract: C[M,N] = bias? + A[M,K] × B_packed[K,N] over the work range.
 * B arrives panel-packed (RVV_PACK_NC columns per panel, k-major inside).
 * Accumulation in fp32; C written as fp32 (tuple f32,f32,f32,f32).
 *
 * The C path is both the correctness oracle (differential-tested against
 * the reference backend) and the x86 CI path. The RVV path vectorizes
 * across the NC columns of a B panel — one accumulator vector per
 * (row, panel), K streamed in order — so both paths accumulate K in the
 * same ascending order and produce bit-identical results.
 ******************************************************************************/

#include "rvv_internal.hpp"

#include <string.h>

/* ------------------------------------------------------ C-portable kernel */

__attribute__((visibility("default"))) void rvv_gemm_f32_c(const void *a_in,
        const void *b_packed, const void *bias_in, void *c_in, int64_t m,
        int64_t n, int64_t k_dim, const rvcl_work_range_t *r) {
    const float *a = (const float *)a_in;
    const float *bp = (const float *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    (void)m;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j = r->n_begin; j < r->n_end; j++) {
            const int64_t panel = j / RVV_PACK_NC;
            const int64_t in = j - panel * RVV_PACK_NC;
            const float *bcol = bp + (panel * k_dim) * RVV_PACK_NC + in;
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k_dim; p++)
                acc += a[i * k_dim + p] * bcol[p * RVV_PACK_NC];
            c[i * n + j] = acc;
        }
    }
}

/* ------------------------------------------------- RVV intrinsics kernel */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/*
 * Row × panel kernel: for row i and panel (NC columns), strip-mine the
 * accumulator across the NC lanes (one strip at a time — RVV types cannot
 * be array elements), stream K once per strip, fmacc the B segment with
 * scalar a[i,p]. Strips own disjoint column slices, so every B element is
 * read exactly once and the K order matches the C path exactly, giving
 * bit-identical output.
 *
 * Bias-first (R4): accumulator lanes are seeded with bias[j] before the
 * K loop, exactly like the C path's `acc = bias[j]`, so the addition
 * order — bias, then K products ascending — is identical in both paths.
 */
void rvv_gemm_f32_rvv(const void *a_in, const void *b_packed,
        const void *bias_in, void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const float *a = (const float *)a_in;
    const float *bp = (const float *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    const int64_t NC = RVV_PACK_NC;
    (void)m;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        const float *arow = a + i * k_dim;
        const int64_t p0 = (r->n_begin / NC) * NC;
        const int64_t p1 = r->n_end;

        for (int64_t j0 = p0; j0 < p1; j0 += NC) {
            const int64_t panel = j0 / NC;
            const float *bpanel = bp + (panel * k_dim) * NC;
            const long vl = __riscv_vsetvl_e32m1(NC);
            const long nv = (long)((NC + vl - 1) / vl);

            for (long v = 0; v < nv; v++) {
                const long off = v * vl;
                const long len = (off + vl <= NC) ? vl : (NC - off);

                long blen = len;
                if (bias) {
                    /* Clamp to the n valid bias entries: a ragged last
                     * panel spans columns >= n, whose lanes are filtered
                     * out at the store below (bias has no NC padding). */
                    const long rem = (long)(n - (j0 + off));
                    blen = rem < len ? rem : len;
                }
                vfloat32m1_t acc = (bias && blen > 0)
                        ? __riscv_vle32_v_f32m1(bias + j0 + off, blen)
                        : __riscv_vfmv_v_f_f32m1(0.0f, vl);

                for (int64_t p = 0; p < k_dim; p++) {
                    const vfloat32m1_t bv
                            = __riscv_vle32_v_f32m1(bpanel + p * NC + off, len);
                    acc = __riscv_vfmacc_vf_f32m1(acc, arow[p], bv, len);
                }

                for (long e = 0; e < len; e++) {
                    const int64_t j = j0 + off + e;
                    if (j < r->n_begin || j >= r->n_end) continue;
                    const vfloat32m1_t s
                            = __riscv_vslidedown_vx_f32m1(acc, e, len);
                    c[i * n + j] = __riscv_vfmv_f_s_f32m1_f32(s);
                }
            }
        }
    }
}

#endif /* __riscv_vector */
