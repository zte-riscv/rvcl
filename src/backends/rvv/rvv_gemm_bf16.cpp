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

/* Tuple (bf16,bf16,f32,f32): storage bfloat16 (u16), accumulation and
 * output fp32. bf16 is the top 16 bits of an fp32, so the up-conversion is
 * an exact integer widen-and-shift (u16 << 16) that needs no Zvfbfmin; fp32
 * accumulation is sequential over K. The C-portable path uses the soft
 * conversions in rvv_bf16.hpp and is both the correctness oracle and the
 * non-vector host path. The JIT micro-kernel (xbyak_riscv) mirrors the f32
 * JIT skeleton: per output column j the accumulator starts at bias[j] and
 * takes one fused multiply-add per k in ascending order, so it is
 * bit-identical to the C path under the same inputs. A BF16 panel row is
 * widened to fp32 by zero-extend + shift-left-16; the scalar A element is
 * loaded as u16, shifted and moved into an fp32 register by the same bit
 * identity. The JIT kernel compiles at the baseline rv64gc -march, with
 * vector code running only after a run-time capability probe (RVV only —
 * bf16 needs no extension beyond the base vector), keeping the whole
 * backend in a single baseline binary. */

#include "rvv_bf16.hpp"
#include "rvv_internal.hpp"

#include <stdio.h>
#include <string.h>

/* ------------------------------------------------------ C-portable kernel */

__attribute__((visibility("default"))) void rvv_gemm_bf16_c(const void *a_in,
        const void *b_packed, const void *bias_in, void *c_in, int64_t m,
        int64_t n, int64_t k_dim, const rvcl_work_range_t *r) {
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
                acc += rvv_bf16_to_f32(a[i * k_dim + p])
                        * rvv_bf16_to_f32(bcol[p * RVV_PACK_NC]);
            c[i * n + j] = acc;
        }
    }
}

/* ------------------------------------------------------------- JIT kernel */

#if defined(RVCL_HAVE_JIT)

#include <exception>

#include "rvcl/capability.h"

#define XBYAK_RISCV_V 1
#define XBYAK_RISCV_VSETV_DEFAULT_OLD 0

#include "xbyak_riscv/xbyak_riscv.hpp"

static_assert(Xbyak_riscv::VERSION == 0x1310,
        "vendored xbyak_riscv must stay at tag 1.31");

namespace {

using namespace Xbyak_riscv;

static_assert((RVV_PACK_NC & (RVV_PACK_NC - 1)) == 0,
        "RVV_PACK_NC must be a power of two (NC-derived shifts)");
constexpr int nc_log2 = __builtin_ctz(RVV_PACK_NC);

/*
 * Row x panel strip kernel, ABI-mapped to rvv_gemm_fn. The accumulator is
 * an fp32 vector kept at e32/m1; the BF16 panel row is widened into a
 * separate e32/m2 destination (zero-extend + shift-left-16 restores the
 * fp32 bit pattern) so that widening and compute never share a register
 * (the core rejects in-place widening and m2 compute). The K loop
 * alternates two vtypes: e16/m1 to load the BF16 panel row, e32/m1 to
 * accumulate and store. av is clamped to the panel-boundary lanes
 * (min(NC, columns left, lanes left in the panel row)) so no load crosses
 * a panel row boundary.
 *
 *   for i in [r->m_begin, r->m_end)
 *     for j0 stepping over [r->n_begin, r->n_end) in vl-sized strips
 *       vl  = min(NC, n_end - j0, NC - j0 % NC)
 *       acc = bias ? bias[j0 .. j0+vl) : 0           (e32/m1)
 *       for k in 0..K:
 *         vw  = widen(B[panel(j0)][k][j0%NC .. +vl)) (e16/m1 load + e32/m2 widen)
 *         acc += A[i,k] * vw                          (e32/m1 fmacc)
 *       C[i, j0 .. j0+vl) = acc                       (e32/m1 store)
 */
class rvv_gemm_bf16_jit_t : public CodeGenerator {
public:
    rvv_gemm_bf16_jit_t() : CodeGenerator(size_t {64} * 1024) {
        generate();
        ready(PROTECT_RWE);
    }

private:
    void generate() {
        const Reg a = a0, bp = a1, bias = a2, c = a3, vl = a4, n = a5, k = a6,
                  r = a7;
        const Reg i = t0, m_end = t1, n_end = t2, j0 = t3, av = t4, tp = t5,
                  p = t6;
        const Reg acur = s0, bcur = s1;
        const FReg fa = fa0;
        /* vacc: e32/m1 accumulator; vb: e16/m1 panel-row load; vw: e32/m2
         * widening destination (never a compute source at m2). v0 stays
         * free for masks. */
        const VReg vacc = v1, vb = v3, vw = v4;

        Label row, strip, avl_nc, avl_panel, no_bias, khead, kloop, store,
                row_end, done;

        addi(sp, sp, -16);
        sd(s0, sp, 0);
        sd(s1, sp, 8);

        ld(i, r, 0);
        ld(m_end, r, 8);
        ld(n_end, r, 24);

        L(row);
        bge(i, m_end, done);
        ld(j0, r, 16);

        L(strip);
        bge(j0, n_end, row_end);
        sub(av, n_end, j0);
        li(tp, RVV_PACK_NC);
        blt(av, tp, avl_nc);
        mv(av, tp);
        L(avl_nc);
        andi(tp, j0, RVV_PACK_NC - 1);
        li(p, RVV_PACK_NC);
        sub(tp, p, tp);
        blt(av, tp, avl_panel);
        mv(av, tp);
        L(avl_panel);

        /* bias-first: e32/m1 seed, same lane order as the C kernel */
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        beqz(bias, no_bias);
        slli(tp, j0, 2);
        add(tp, bias, tp);
        vle32_v(vacc, tp);
        j_(khead);
        L(no_bias);
        vxor_vv(vacc, vacc, vacc); /* zero the accumulator (vmv.v.i is
                                    * unsupported on some cores) */

        L(khead);
        mul(tp, i, k);
        slli(tp, tp, 1); /* 2-byte A elements */
        add(acur, a, tp); /* &A[i, 0] */
        srai(tp, j0, nc_log2);
        mul(tp, tp, k);
        slli(tp, tp, nc_log2 + 1); /* panel stride: NC lanes x 2 bytes */
        add(bcur, bp, tp);
        andi(tp, j0, RVV_PACK_NC - 1);
        slli(tp, tp, 1);
        add(bcur, bcur, tp); /* &B[panel][0][j0 % NC] */
        li(p, 0);
        L(kloop);
        bge(p, k, store);
        lhu(tp, acur, 0); /* A[i, p] as u16 */
        slli(tp, tp, 16); /* bf16 -> fp32 bit pattern */
        fmv_w_x(fa, tp);
        vsetvli(vl, av, SEW::e16, LMUL::m1, VTA::ta, VMA::ma);
        vle16_v(vb, bcur); /* B[p, j0 .. j0+vl) as bf16 */
        /* zero-extend u16 -> u32 into a fresh e32/m2 destination: widening
         * is rejected when it aliases its source, so vw must not overlap
         * vb. vwaddu_vx(vw, vb, x0) adds zero, i.e. a pure zero-extend. */
        vwaddu_vx(vw, vb, x0);
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        vsll_vi(vw, vw, 16); /* shift into the fp32 bit pattern */
        vfmacc_vf(vacc, fa, vw); /* acc += A[i,p] * B[p, j0..j0+vl) */
        addi(acur, acur, 2);
        addi(bcur, bcur, RVV_PACK_NC * 2); /* next k row: NC x 2 bytes */
        addi(p, p, 1);
        j_(kloop);
        L(store);
        mul(tp, i, n);
        add(tp, tp, j0);
        slli(tp, tp, 2);
        add(tp, c, tp);
        vse32_v(vacc, tp);
        add(j0, j0, vl);
        j_(strip);
        L(row_end);
        addi(i, i, 1);
        j_(row);
        L(done);
        ld(s1, sp, 8);
        ld(s0, sp, 0);
        addi(sp, sp, 16);
        ret();
    }
};

} // namespace

int rvv_gemm_bf16_jit_available(void) {
    /* bf16 needs only the base vector extension: the up-conversion is an
     * integer widen+shift, not a Zvfbfmin instruction. */
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_bf16_jit_entry(void) {
    static rvv_gemm_fn entry = []() -> rvv_gemm_fn {
        try {
            static rvv_gemm_bf16_jit_t kern;
            return kern.getCode<rvv_gemm_fn>();
        } catch (const std::exception &e) {
            fprintf(stderr, "rvcl: bf16 JIT kernel generation failed: %s\n",
                    e.what());
            return nullptr;
        } catch (...) {
            fprintf(stderr, "rvcl: bf16 JIT kernel generation failed\n");
            return nullptr;
        }
    }();
    return entry;
}

#endif /* RVCL_HAVE_JIT */
