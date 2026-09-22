/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * rvv_gemm_f16.c — FP16 GEMM micro-kernels (RVV intrinsics + C fallback)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Tuple (f16,f16,f32,f32): storage fp16 (Zvfh / Zvfhmin), accumulation and
 * output fp32 (PLAN §4.5.1 habit rule). On RVV hosts with Zvfh the K loop
 * widens fp16 segments to fp32 vectors and accumulates in fp32. On hosts
 * without Zvfh the C path uses the bit-exact soft conversions in
 * rvv_fp16.h, keeping the x86 CI differential tests meaningful.
 *
 * Both paths accumulate K in ascending order → bit-identical results.
 ******************************************************************************/

#include "rvv_fp16.h"
#include "rvv_internal.h"

#include <string.h>

/* ------------------------------------------------------ C-portable kernel */

__attribute__((visibility("default"))) void rvv_gemm_f16_c(const void *a_in, const void *b_packed, const void *bias_in,
        void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const uint16_t *a = (const uint16_t *)a_in;
    const uint16_t *bp = (const uint16_t *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    (void)m;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j = r->n_begin; j < r->n_end; j++) {
            const int64_t panel = j / RVV_PACK_NC;
            const int64_t in = j - panel * RVV_PACK_NC;
            const uint16_t *bcol = bp + (panel * k_dim) * RVV_PACK_NC + in;
            float acc = bias ? bias[j] : 0.0f;
            for (int64_t p = 0; p < k_dim; p++)
                acc += rvv_f16_to_f32(a[i * k_dim + p])
                        * rvv_f16_to_f32(bcol[p * RVV_PACK_NC]);
            c[i * n + j] = acc;
        }
    }
}

/* ------------------------------------------------- RVV intrinsics kernel */

#if defined(__riscv_vector)
#include <riscv_vector.h>

/*
 * Same row × panel structure as the f32 kernel: NC-wide fp16 segment
 * widened to fp32 accumulators (vw load + fwcvt), K streamed in order.
 * Zvfh is required for fp16 vector loads (checked at configure/validate
 * time via RVCL_FEATURE_FP16); on Zvfhmin-only parts the backend falls
 * back to scalar conversion.
 */
void rvv_gemm_f16_rvv(const void *a_in, const void *b_packed,
        const void *bias_in, void *c_in, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r) {
    const uint16_t *a = (const uint16_t *)a_in;
    const uint16_t *bp = (const uint16_t *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;
    const int64_t NC = RVV_PACK_NC;

    for (int64_t i = r->m_begin; i < r->m_end; i++) {
        for (int64_t j0 = (r->n_begin / NC) * NC; j0 < r->n_end; j0 += NC) {
            const int64_t panel = j0 / NC;
            const uint16_t *bpanel = bp + (panel * k_dim) * NC;

            const long vl_h = __riscv_vsetvl_e16m1(NC);
            const long nv = (long)((NC + vl_h - 1) / vl_h);
            vfloat32m2_t acc[RVV_PACK_NC];

            for (long v = 0; v < nv; v++)
                acc[v] = __riscv_vfmv_v_f_f32m2(0.0f, vl_h);

            for (int64_t p = 0; p < k_dim; p++) {
                const float av = rvv_f16_to_f32(a[i * k_dim + p]);
                for (long v = 0; v < nv; v++) {
                    const long off = v * vl_h;
                    const long len = (off + vl_h <= NC) ? vl_h : (NC - off);
                    const vfloat16m1_t bh = __riscv_vle16_v_f16m1(
                            bpanel + p * NC + off, len);
                    const vfloat32m2_t bv = __riscv_vfwcvt_f_f16_f32m2(
                            bh, len);
                    acc[v] = __riscv_vfmacc_vf_f32m2(acc[v], av, bv, len);
                }
            }

            for (long v = 0; v < nv; v++) {
                const long off = v * vl_h;
                const long len = (off + vl_h <= NC) ? vl_h : (NC - off);
                for (long e = 0; e < len; e++) {
                    const int64_t j = j0 + off + e;
                    if (j < r->n_begin || j >= r->n_end) continue;
                    float val = __riscv_vfmv_f_s_f32m2_f32(
                            __riscv_vslidedown_vx_f32m2(acc[v], e, len));
                    if (bias) val += bias[j];
                    c[i * n + j] = val;
                }
            }
        }
    }
}

#endif /* __riscv_vector */
