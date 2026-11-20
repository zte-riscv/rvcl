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

/* Run-time generated counterpart of the C kernels in rvv_gemm_mxfp8.cpp:
 * this translation unit compiles at the baseline rv64gc -march, and vector
 * code exists only after a run-time capability probe, keeping the whole
 * backend in a single baseline binary.
 *
 * Both operands are FP8 bytes, each 32-wide K-block sharing a 1-byte E8M0
 * exponent scale (dequant = element * 2^(e-127)). The scalar A element and
 * A-block scale are loaded with lbu and indexed into the shared 256-entry
 * f32 tables; the NC-wide B strip and B-block scale are upconverted
 * arithmetically with base-RVV ops, bit-identical to the tables. Because
 * E8M0 scales are exact powers of two (the NaN/0x00 cases apart),
 * multiplying a product by them never rounds, so the accumulate order (bias
 * first, ascending k) stays bit-identical to the C path.
 *
 * Kernel argument layout (LP64D): the eight register args are a0=a,
 * a1=b_packed, a2=sa, a3=sb, a4=bias, a5=c, a6=m, a7=n; k_dim and r are the
 * first two stack arguments (caller sp+0, sp+16). blk is a fixed
 * compile-time constant (32, the OCP MX block width) and is not read from
 * the stack. One code object serves every VLEN. Host side builds the entry
 * once per process (magic static); a generation failure is latched as NULL
 * so auto selection degrades to the C kernel. */

#include "../rvv_e8m0.hpp"
#include "../rvv_fp8.hpp"
#include "../rvv_internal.hpp"

#include <stdint.h>
#include <stdio.h>

#include <exception>

#include "rvcl/capability.h"

#define XBYAK_RISCV_V 1
#define XBYAK_RISCV_VSETV_DEFAULT_OLD 0

#include "xbyak_riscv/xbyak_riscv.hpp"

// Pins the vendored copy to the upstream tag recorded in
// THIRD-PARTY-PROGRAMS: 0x1310 encodes version 1.31.
static_assert(Xbyak_riscv::VERSION == 0x1310,
        "vendored xbyak_riscv must stay at tag 1.31");

namespace {

using namespace Xbyak_riscv;

static_assert((RVV_PACK_NC & (RVV_PACK_NC - 1)) == 0,
        "RVV_PACK_NC must be a power of two (NC-derived shifts)");
constexpr int nc_log2 = __builtin_ctz(RVV_PACK_NC);
constexpr int blk_log2 = 5; /* OCP MX block width 32 = 2^5 */

/* Shared row x panel strip kernel parameterized by the FP8 dtype's 256-entry
 * f32 upconversion table and layout constants (the same arithmetic recipe
 * as rvv_gemm_fp8_jit.cpp reproduces the FP8 table). */
template <const uint32_t (*TABLE)[256], uint32_t EXP_BIAS, uint32_t MAN_SHIFT,
        uint32_t MAN_MASK, uint32_t EXP_MASK, uint32_t DIV_C, uint32_t SUB_C,
        bool HAS_INF>
class rvv_gemm_mxfp8_jit_t : public CodeGenerator {
public:
    rvv_gemm_mxfp8_jit_t() : CodeGenerator(size_t {128} * 1024) {
        generate();
        ready(PROTECT_RWE); /* mprotect RWX + fence.i icache sync */
    }

private:
    void generate() {
        const Reg a = a0, bp = a1, sa = a2, sb = a3, bias = a4, c = a5, vl = a6,
                  n = a7;
        const Reg k = t0, r = t1, i = t2, m_end = t3, n_end = t4, j0 = s3,
                  av = s4, tp = t5, p = t6;
        const Reg acur = s0, bcur = s1,
                  tb = s2, /* FP8 table base (loop-invariant) */
                e8 = s5, /* E8M0 table base (loop-invariant) */
                sa_base = s6; /* &sa[i, 0], loop-invariant per row */
        const FReg fa = fa0; /* scalar A[i, p] */
        const FReg fa_scale = fa1; /* scalar A-block E8M0 scale */
        const FReg fc_div = fa2, fc_one = fa3, fc_sub = fa4, fc_qnan = fa5;
        const FReg fc_pinf = fa6, fc_ninf = fa7;
        /* v1: e32/m1 accumulator. v3: 8-bit B strip / scale at e8/m1.
         * v4: base of the e32/m4 vzext.vf4 destination group. v2/v5/v6:
         * extracted fields; v7: transient scratch; v8: the upconverted B
         * value strip; v9: the upconverted B-block scale. */
        const VReg vacc = v1, vb = v3, vidx = v4;
        const VReg sreg = v2, ereg = v5, mreg = v6, scratch = v7;
        const VReg bval = v8, bscl = v9;

        Label row, strip, avl_nc, avl_panel, no_bias, khead, kloop, store,
                row_end, done;

        addi(sp, sp, -64); /* save s0..s6 (7 regs) in a 16-byte-aligned frame */
        sd(s0, sp, 0);
        sd(s1, sp, 8);
        sd(s2, sp, 16);
        sd(s3, sp, 24);
        sd(s4, sp, 32);
        sd(s5, sp, 40);
        sd(s6, sp, 48);

        /* Stack arguments: k_dim at caller's sp+0, r at caller's sp+16
         * (the callee's own 64-byte frame starts at caller sp - 64). */
        ld(k, sp, 64 + 0);
        ld(r, sp, 64 + 16);

        /* Embed the static FP8 table base address once (loop-invariant).
         * li() sign-extends 32-bit constants, so the low word is
         * zero-extended with a shift-pair before combining to keep
         * arbitrary (48-bit) addresses exact. */
        li(tb, (uint32_t)((uintptr_t)TABLE >> 32));
        slli(tb, tb, 32);
        li(tp, (uint32_t)(uintptr_t)TABLE);
        slli(tp, tp, 32);
        srli(tp, tp, 32);
        add(tb, tb, tp);
        /* Same for the E8M0 table base. */
        li(e8, (uint32_t)((uintptr_t)rvv_e8m0_to_f32_bits >> 32));
        slli(e8, e8, 32);
        li(tp, (uint32_t)(uintptr_t)rvv_e8m0_to_f32_bits);
        slli(tp, tp, 32);
        srli(tp, tp, 32);
        add(e8, e8, tp);

        ld(i, r, 0); /* i = r->m_begin */
        ld(m_end, r, 8);
        ld(n_end, r, 24);

        /* Arithmetic-upconversion constants (loop-invariant). */
        li(tp, 0x3f800000); /* 1.0f */
        fmv_w_x(fc_one, tp);
        li(tp, DIV_C);
        fmv_w_x(fc_div, tp);
        li(tp, SUB_C);
        fmv_w_x(fc_sub, tp);
        li(tp, 0x7fc00000u);
        fmv_w_x(fc_qnan, tp);
        if (HAS_INF) {
            li(tp, 0x7f800000u);
            fmv_w_x(fc_pinf, tp);
            li(tp, 0xff800000u);
            fmv_w_x(fc_ninf, tp);
        }

        L(row);
        bge(i, m_end, done);
        /* sa_base = sa + i * nblk, where nblk = k/32. */
        srli(tp, k, blk_log2); /* nblk */
        mul(tp, i, tp);
        add(sa_base, sa, tp);
        ld(j0, r, 16); /* strips restart from n_begin on every row */

        L(strip);
        bge(j0, n_end, row_end);
        sub(av, n_end, j0); /* av = columns left */
        li(tp, RVV_PACK_NC);
        blt(av, tp, avl_nc);
        mv(av, tp);
        L(avl_nc);
        andi(tp, j0, RVV_PACK_NC - 1);
        li(p, RVV_PACK_NC);
        sub(tp, p, tp); /* NC - (j0 % NC): lanes left in this panel row */
        blt(av, tp, avl_panel);
        mv(av, tp);
        L(avl_panel);
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);

        beqz(bias, no_bias);
        slli(tp, j0, 2);
        add(tp, bias, tp);
        vle32_v(vacc, tp); /* bias-first: same lane order as the C kernel */
        j_(khead);
        L(no_bias);
        vmv_v_i(vacc, 0);

        L(khead);
        /* A row base and the B panel cursor are re-derived per strip. */
        mul(tp, i, k);
        add(acur, a, tp); /* &A[i, 0] (element stride 1 byte) */
        srai(tp, j0, nc_log2); /* panel = j0 / NC */
        mul(tp, tp, k);
        slli(tp, tp, nc_log2); /* panel stride: NC lanes x 1 byte */
        add(bcur, bp, tp);
        andi(tp, j0, RVV_PACK_NC - 1);
        add(bcur, bcur, tp); /* &B[panel][0][j0 % NC] */
        li(p, 0);
        L(kloop);
        bge(p, k, store);

        /* A scalar element: UP_fp8(a[i*k+p]) via the shared table. */
        lbu(tp, acur, 0); /* A[i, p] as a byte index */
        slli(tp, tp, 2);
        add(tp, tp, tb);
        flw(fa, tp, 0);
        /* A-block scale scalar: e8m0(sa[i*nblk + pblk]) via its table. */
        srli(tp, p, blk_log2); /* pblk */
        add(tp, sa_base, tp);
        lbu(tp, tp, 0);
        slli(tp, tp, 2);
        add(tp, tp, e8);
        flw(fa_scale, tp, 0);

        /* ---- Load the NC-wide B strip and upconvert it arithmetically
         * to f32 (same recipe as the FP8 JIT, bit-identical to the shared
         * FP8 table). ---- */
        vsetvli(vl, av, SEW::e8, LMUL::m1, VTA::ta, VMA::ma); /* e8/m1 */
        vle8_v(vb, bcur); /* B[p, j0 .. j0+vl) as byte indices */
        vsetvli(vl, av, SEW::e32, LMUL::m4, VTA::ta, VMA::ma); /* widen */
        vzext_vf4(vidx, vb); /* zero-extend the bytes to e32 indices */
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma); /* compute */
        /* sreg = sign bit, ereg = exponent field, mreg = mantissa field. */
        vsrl_vi(sreg, vidx, 7);
        vand_vi(sreg, sreg, 1);
        vsrl_vi(ereg, vidx, MAN_SHIFT);
        /* vand.vi with 31 sign-extends the immediate to all ones (an
         * identity AND), which would leak the shifted-out sign bit into
         * the exponent field; the mask goes through a scalar register. */
        li(tp, EXP_MASK);
        vand_vx(ereg, ereg, tp);
        vand_vi(mreg, vidx, MAN_MASK);
        /* vidx is reused as scratch after the fields are extracted. */
        vfcvt_f_xu_v(scratch, mreg); /* mf = (float)mantissa */
        /* OPIVI immediates are simm5; out-of-simm5 constants go through a
         * scalar register. */
        li(tp, EXP_BIAS);
        vadd_vx(vb, ereg, tp); /* vb = exponent + bias */
        vsll_vi(vb, vb, 23); /* vb = float bits of 2^(exp) */
        vfdiv_vf(scratch, scratch, fc_div); /* mf / divisor */
        vfadd_vf(scratch, scratch, fc_one); /* 1 + mf / divisor */
        vfmul_vv(scratch, vb, scratch); /* normal value in scratch */
        vfcvt_f_xu_v(vidx, mreg); /* (recompute) mf for the subnormal */
        vfmul_vf(vidx, vidx, fc_sub); /* subnormal = mantissa x 2^-n */
        vmseq_vi(v0, ereg, 0); /* exponent == 0 -> subnormal */
        vmerge_vvm(scratch, scratch, vidx); /* val = e0 ? sub : normal */
        vmsne_vi(v0, sreg, 0); /* sign set */
        vfneg_v(vidx, scratch); /* -val */
        vmerge_vvm(scratch, scratch, vidx); /* val = sign ? -val : val */
        li(tp, EXP_MASK); /* simm5-unsafe constant (31 for E5M2) */
        vmseq_vx(v0, ereg, tp); /* max exponent */
        if (HAS_INF) {
            vmsne_vi(vb, mreg, 0); /* m != 0 */
            vmand_mm(v0, v0, vb); /* e==max && m!=0 -> NaN */
            vfmv_v_f(vb, fc_qnan);
        } else {
            vmseq_vi(vb, mreg, 7); /* m == 7 -> NaN (E4M3) */
            vmand_mm(v0, v0, vb);
            vfmv_v_f(vb, fc_qnan);
        }
        vmerge_vvm(scratch, scratch, vb); /* val = NaN ? qNaN : val */
        if (HAS_INF) {
            vmsne_vi(v0, sreg, 0); /* sign (recomputed) */
            vfmv_v_f(vb, fc_pinf); /* +inf */
            vfmv_v_f(vidx, fc_ninf); /* -inf */
            vmerge_vvm(vb, vb, vidx); /* signed inf */
            vmseq_vx(v0, ereg, tp); /* e == 31; tp still holds EXP_MASK */
            vmsne_vi(vidx, mreg, 0); /* m != 0 */
            vmandn_mm(v0, v0, vidx); /* e==31 && m==0 -> inf */
            vmerge_vvm(scratch, scratch, vb); /* val = inf ? s-inf : val */
        }
        vmv_v_v(bval, scratch); /* keep the upconverted B value strip */

        /* ---- E8M0 B-block scale: sb[pblk*n + (j0..j0+vl)), arithmetic
         * upconversion bit-identical to rvv_e8m0_to_f32_bits. ---- */
        srli(tp, p, blk_log2); /* pblk */
        mul(tp, tp, n);
        add(tp, tp, j0);
        add(tp, sb, tp); /* &sb[pblk][j0] (bytes) */
        vsetvli(vl, av, SEW::e8, LMUL::m1, VTA::ta, VMA::ma); /* e8/m1 */
        vle8_v(vb, tp); /* scale bytes */
        vsetvli(vl, av, SEW::e32, LMUL::m4, VTA::ta, VMA::ma); /* widen */
        vzext_vf4(vidx, vb);
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma);
        vsll_vi(bscl, vidx, 23); /* normal scale bit pattern 2^(e-127) */
        /* fix up 0x00 -> smallest subnormal 0x00400000 (never a literal 0) */
        vmseq_vi(v0, vidx, 0); /* byte == 0 */
        li(tp, 0x00400000u);
        vmv_v_x(vb, tp);
        vmerge_vvm(bscl, bscl, vb); /* scale = e0 ? 2^-127 : scale */
        /* fix up 0xFF -> NaN 0x7fc00000 (propagated, never clipped) */
        li(tp, 0xffu);
        vmseq_vx(v0, vidx, tp); /* byte == 0xff */
        li(tp, 0x7fc00000u);
        vmv_v_x(vb, tp);
        vmerge_vvm(bscl, bscl, vb); /* scale = ff ? qNaN : scale */
        /* Fold the scalar A-block scale in (an exact power of two, or the
         * 0x00 subnormal / 0xFF NaN from the E8M0 table): multiplying does
         * not round the B value. */
        vfmul_vf(bscl, bscl, fa_scale); /* Bscale *= A-block scale */
        vfmul_vv(bval, bval, bscl); /* B_up *= combined block scale */
        vfmacc_vf(vacc, fa, bval); /* acc += A[i,p] * scaled B */

        addi(acur, acur, 1); /* A stride 1 byte */
        addi(bcur, bcur, RVV_PACK_NC); /* next k row: NC bytes */
        addi(p, p, 1);
        j_(kloop);
        L(store);
        mul(tp, i, n); /* &C[i, j0] */
        add(tp, tp, j0);
        slli(tp, tp, 2);
        add(tp, c, tp);
        vse32_v(vacc, tp);
        add(j0, j0, vl); /* consuming the panel remainder re-aligns j0 */
        j_(strip);
        L(row_end);
        addi(i, i, 1);
        j_(row);
        L(done);
        ld(s6, sp, 48);
        ld(s5, sp, 40);
        ld(s4, sp, 32);
        ld(s3, sp, 24);
        ld(s2, sp, 16);
        ld(s1, sp, 8);
        ld(s0, sp, 0);
        addi(sp, sp, 64);
        ret();
    }
};

template <const uint32_t (*TABLE)[256], uint32_t EXP_BIAS, uint32_t MAN_SHIFT,
        uint32_t MAN_MASK, uint32_t EXP_MASK, uint32_t DIV_C, uint32_t SUB_C,
        bool HAS_INF>
rvv_gemm_mxfp8_fn rvv_gemm_mxfp8_jit_entry_impl(void) {
    static rvv_gemm_mxfp8_fn entry = []() -> rvv_gemm_mxfp8_fn {
        try {
            static rvv_gemm_mxfp8_jit_t<TABLE, EXP_BIAS, MAN_SHIFT, MAN_MASK,
                    EXP_MASK, DIV_C, SUB_C, HAS_INF>
                    kern;
            return kern.template getCode<rvv_gemm_mxfp8_fn>();
        } catch (const std::exception &e) {
            fprintf(stderr, "rvcl: mxfp8 JIT kernel generation failed: %s\n",
                    e.what());
            return nullptr;
        } catch (...) {
            fprintf(stderr, "rvcl: mxfp8 JIT kernel generation failed\n");
            return nullptr;
        }
    }();
    return entry;
}

} // namespace

int rvv_gemm_mxfp8_e4m3_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_mxfp8_fn rvv_gemm_mxfp8_e4m3_jit_entry(void) {
    /* E4M3: exponent field e = bits 6:3 (bias 7), mantissa m = bits 2:0,
     * max exponent 15 with m==7 -> qNaN; no infinity. */
    return rvv_gemm_mxfp8_jit_entry_impl<&rvv_fp8_e4m3_to_f32_bits, 120, 3, 7,
            15, 0x41000000u, 0x3B000000u, false>();
}

int rvv_gemm_mxfp8_e5m2_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_mxfp8_fn rvv_gemm_mxfp8_e5m2_jit_entry(void) {
    /* E5M2: exponent field e = bits 6:2 (bias 15), mantissa m = bits 1:0,
     * max exponent 31: m==0 -> +/-inf, m!=0 -> qNaN. */
    return rvv_gemm_mxfp8_jit_entry_impl<&rvv_fp8_e5m2_to_f32_bits, 112, 2, 3,
            31, 0x40800000u, 0x37800000u, true>();
}
