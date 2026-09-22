/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * rvv_gemm_f32.c — FP32 GEMM micro-kernels (RVV intrinsics + C fallback)
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

#include "rvv_internal.h"

#include <string.h>

/* ------------------------------------------------------ C-portable kernel */

__attribute__((visibility("default"))) void rvv_gemm_f32_c(const void *a_in, const void *b_packed, const void *bias_in,
        void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
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
 * Row × panel kernel: for row i and panel (NC columns), hold a strip-mined
 * accumulator vector set, stream K once, fmacc the NC-wide B segment with
 * scalar a[i,p]. All loads of B are unit-stride NC-segments; the K order
 * matches the C path exactly, giving bit-identical output.
 */
void rvv_gemm_f32_rvv(const void *a_in, const void *b_packed,
        const void *bias_in, void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const float *a = (const float *)a_in;
    const float *bp = (const float *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    const int64_t NC = RVV_PACK_NC;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        const float *arow = a + i * k_dim;
        const int64_t p0 = (r->n_begin / NC) * NC;
        const int64_t p1 = r->n_end;

        for (int64_t j0 = p0; j0 < p1; j0 += NC) {
            const int64_t panel = j0 / NC;
            const float *bpanel = bp + (panel * k_dim) * NC;
            const long vl = __riscv_vsetvl_e32m1(NC);
            vfloat32m1_t acc[RVV_PACK_NC]; /* strip-mined across NC */
            const long nv = (long)((NC + vl - 1) / vl);

            for (long v = 0; v < nv; v++)
                acc[v] = __riscv_vfmv_v_f_f32m1(0.0f, vl);

            for (int64_t p = 0; p < k_dim; p++) {
                const float av = arow[p];
                for (long v = 0; v < nv; v++) {
                    const long off = v * vl;
                    const long len = (off + vl <= NC) ? vl : (NC - off);
                    const vfloat32m1_t bv = __riscv_vle32_v_f32m1(
                            bpanel + p * NC + off, len);
                    acc[v] = __riscv_vfmacc_vf_f32m1(acc[v], av, bv, len);
                }
            }

            for (long v = 0; v < nv; v++) {
                const long off = v * vl;
                const long len = (off + vl <= NC) ? vl : (NC - off);
                for (long e = 0; e < len; e++) {
                    const int64_t j = j0 + off + e;
                    if (j < r->n_begin || j >= r->n_end) continue;
                    const vfloat32m1_t s = __riscv_vslidedown_vx_f32m1(
                            acc[v], e, len);
                    float val = __riscv_vfmv_f_s_f32m1_f32(s);
                    if (bias) val += bias[j];
                    c[i * n + j] = val;
                }
            }
        }
    }
}

#endif /* __riscv_vector */
