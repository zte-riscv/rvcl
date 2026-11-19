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
 * rvv_gemm_fp8_jit.cpp — FP8 (E4M3 / E5M2) GEMM JIT micro-kernel (xbyak_riscv)
 *
 * Run-time generated counterpart of the C kernels in rvv_gemm_fp8.cpp:
 * this translation unit compiles at the baseline rv64gc -march, and
 * vector code exists only after a run-time capability probe — the
 * property that keeps the whole backend in a single baseline binary.
 *
 * FP8 has no base hardware extension, so the generated loop upconverts
 * every byte to f32. The scalar A element is loaded with lbu and indexed
 * into the shared 256-entry f32 table (a scalar load, not a vector
 * gather), and the NC-wide B strip is upconverted arithmetically: its
 * 8-bit bytes are loaded with vle8.v, zero-extended to 32-bit indices
 * with vzext.vf4, then sign/exponent/mantissa are extracted and rebuilt
 * into the exact f32 bit pattern of the shared tables using base-RVV
 * element-wise integer and float ops. The arithmetic path is bit-identical
 * to the tables (verified for every one of the 256 encodings of both
 * dtypes) and depends on no indexed-load gather, so the kernel runs on
 * emulators and hardware alike. The table address is a compile-time
 * static, so it is embedded as a 64-bit immediate once, before the K
 * loop. Per output column j the accumulator starts at bias[j] and takes
 * one fused multiply-add per k in ascending order — bit-identical to the
 * C path under the same inputs. One code object serves every VLEN (the
 * AVL is clamped like the f32 JIT).
 *
 * Host side mirrors rvv_gemm_f32_jit.cpp: the entry is built once per
 * process (magic static) and a generation failure is latched as NULL so
 * auto selection degrades to the C kernel.
 ******************************************************************************/

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

/* Shared row x panel strip kernel parameterized by the dtype's 256-entry
 * f32 upconversion table and by the FP8 layout constants that reproduce
 * that table arithmetically. All eight kernel arguments arrive in a0..a7
 * (LP64D, no stack arguments) and the kernel is a leaf. */
template <const uint32_t (*TABLE)[256], uint32_t EXP_BIAS, uint32_t MAN_SHIFT,
        uint32_t MAN_MASK, uint32_t DIV_C, uint32_t SUB_C, bool HAS_INF>
class rvv_gemm_fp8_jit_t : public CodeGenerator {
public:
    rvv_gemm_fp8_jit_t() : CodeGenerator(size_t {64} * 1024) {
        generate();
        ready(PROTECT_RWE); /* mprotect RWX + fence.i icache sync */
    }

private:
    void generate() {
        const Reg a = a0, bp = a1, bias = a2, c = a3, vl = a4, n = a5, k = a6,
                  r = a7;
        const Reg i = t0, m_end = t1, n_end = t2, j0 = t3, av = t4, tp = t5,
                  p = t6;
        const Reg acur = s0, bcur = s1,
                  tb = s2; /* table base (loop-invariant) */
        const FReg fa = fa0; /* scalar A[i, p], broadcast */
        const FReg fc_div = fa1, fc_one = fa2, fc_sub = fa3, fc_qnan = fa4,
                   fc_pinf = fa5, fc_ninf = fa6;
        /* v1: e32/m1 accumulator. v3: the 8-bit B strip at e8/m1 and the
         * widening source. v4: the base of the e32/m4 vzext.vf4
         * destination group (aligned to a multiple of 4), whose first
         * register is re-read at e32/m1 as the zero-extended byte indices
         * and then reused as scratch for the arithmetic upconversion.
         * v2/v5/v6 hold the extracted sign/exponent/mantissa through the
         * whole conversion; v7 is transient scratch. v0 stays free for
         * the vmerge mask. The upconversion is purely arithmetic (base
         * RVV integer + float element-wise ops), so the generated kernel
         * depends on no indexed-load emulation. */
        const VReg vacc = v1, vb = v3, vidx = v4;
        const VReg sreg = v2, ereg = v5, mreg = v6, scratch = v7;

        Label row, strip, avl_nc, avl_panel, no_bias, khead, kloop, store,
                row_end, done;

        addi(sp, sp, -32); /* RV64 psABI: sp stays 16-byte aligned */
        sd(s0, sp, 0);
        sd(s1, sp, 8);
        sd(s2, sp, 16);

        /* Embed the static table base address once (loop-invariant). li()
         * sign-extends 32-bit constants, so the low word is zero-extended
         * with a shift-pair before combining to keep arbitrary (48-bit)
         * addresses exact. */
        li(tb, (uint32_t)((uintptr_t)TABLE >> 32));
        slli(tb, tb, 32); /* hi in bits 63..32 */
        li(tp, (uint32_t)(uintptr_t)TABLE);
        slli(tp, tp, 32);
        srli(tp, tp, 32); /* unsigned low 32 */
        add(tb, tb, tp);

        ld(i, r, 0); /* i = r->m_begin */
        ld(m_end, r, 8);
        ld(n_end, r, 24);

        /* Arithmetic-upconversion constants, loop-invariant across every
         * strip and row. The qNaN value (0x7fc00000) is the canonical
         * quiet NaN the shared tables use for every NaN encoding. */
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
        lbu(tp, acur, 0); /* A[i, p] as a byte index */
        slli(tp, tp, 2); /* index x 4 bytes per f32 table entry */
        add(tp, tp, tb);
        flw(fa, tp, 0); /* broadcast scalar A[i, p] upconverted */
        /* Load the NC-wide B strip as 8-bit bytes and widen them to 32-bit
         * with vzext.vf4. The widening vtype must be e32/m4: for vzext.vf4
         * the vtype SEW is the *destination* size (32) and the 8-bit source
         * group EMUL = LMUL x 1/4, so LMUL=m4 makes both the source
         * (EMUL=1) and destination (EMUL=4) groups full-length. The AVL is
         * below VLMAX at e8/m1, hence also at e32/m4, so VL is preserved.
         * The first register of the m4 destination group (v4) holds the VL
         * active zero-extended bytes, re-read at e32/m1. The byte indices
         * are then upconverted to f32 arithmetically (bit-identical to the
         * shared tables) with base-RVV element-wise ops, so the kernel
         * needs no indexed-load gather at all. */
        vsetvli(vl, av, SEW::e8, LMUL::m1, VTA::ta, VMA::ma); /* e8/m1 */
        vle8_v(vb, bcur); /* B[p, j0 .. j0+vl) as byte indices */
        vsetvli(vl, av, SEW::e32, LMUL::m4, VTA::ta, VMA::ma); /* widen */
        /* vzext.vf4 vd, vs2: funct6 010010, funct3 010. The vs1 field is a
         * reserved, don't-care operand (real hardware ignores it), but the
         * qemu emulator decodes a zero vs1 as an illegal instruction, so the
         * field is set to vd's register number (as GCC does) to run correctly
         * on both the emulator and real silicon. */
        append4B(0x4a002057u | (vb.getIdx() << 20) | (vidx.getIdx() << 15)
                | (vidx.getIdx() << 7));
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma); /* compute */
        /* sreg = sign bit, ereg = exponent field, mreg = mantissa field. */
        vsrl_vi(sreg, vidx, 7);
        vand_vi(sreg, sreg, 1);
        vsrl_vi(ereg, vidx, MAN_SHIFT);
        vand_vi(ereg, ereg, MAN_MASK);
        vand_vi(mreg, vidx, MAN_MASK);
        /* vzext.vf4 source (vb) holds the zero-extended bytes; after the
         * field extraction the bytes themselves (vidx) are no longer
         * needed and vidx is reused as a scratch f32 register. */
        vfcvt_f_xu_v(scratch, mreg); /* mf = (float)mantissa */
        vadd_vi(vb, ereg, EXP_BIAS); /* vb = exponent + bias */
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
        vmseq_vi(v0, ereg, MAN_MASK == 7 ? 15 : 31); /* max exponent */
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
            vmsne_vi(v0, sreg, 0); /* sign (recomputed; mreg still intact) */
            vfmv_v_f(vb, fc_pinf); /* +inf */
            vfmv_v_f(vidx, fc_ninf); /* -inf */
            vmerge_vvm(vb, vb, vidx); /* signed inf */
            vmseq_vi(v0, ereg, 31); /* e == 31 */
            vmsne_vi(vidx, mreg, 0); /* m != 0 */
            vmandn_mm(v0, v0, vidx); /* e==31 && m==0 -> inf */
            vmerge_vvm(scratch, scratch, vb); /* val = inf ? s-inf : val */
        }
        vfmacc_vf(vacc, fa, scratch); /* acc += A[i,p] * B[p, j0..j0+vl) */
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
        ld(s2, sp, 16);
        ld(s1, sp, 8);
        ld(s0, sp, 0);
        addi(sp, sp, 32);
        ret();
    }
};

template <const uint32_t (*TABLE)[256], uint32_t EXP_BIAS, uint32_t MAN_SHIFT,
        uint32_t MAN_MASK, uint32_t DIV_C, uint32_t SUB_C, bool HAS_INF>
rvv_gemm_fn rvv_gemm_fp8_jit_entry_impl(void) {
    static rvv_gemm_fn entry = []() -> rvv_gemm_fn {
        try {
            static rvv_gemm_fp8_jit_t<TABLE, EXP_BIAS, MAN_SHIFT, MAN_MASK,
                    DIV_C, SUB_C, HAS_INF>
                    kern;
            return kern.template getCode<rvv_gemm_fn>();
        } catch (const std::exception &e) {
            fprintf(stderr, "rvcl: fp8 JIT kernel generation failed: %s\n",
                    e.what());
            return nullptr;
        } catch (...) {
            fprintf(stderr, "rvcl: fp8 JIT kernel generation failed\n");
            return nullptr;
        }
    }();
    return entry;
}

} // namespace

int rvv_gemm_fp8_e4m3_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_fp8_e4m3_jit_entry(void) {
    /* E4M3: exponent field e = bits 6:3 (bias 7), mantissa m = bits 2:0,
     * max exponent 15 with m==7 -> qNaN; no infinity. */
    return rvv_gemm_fp8_jit_entry_impl<&rvv_fp8_e4m3_to_f32_bits, 120, 3, 7,
            0x41000000u, 0x39000000u, false>();
}

int rvv_gemm_fp8_e5m2_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_fp8_e5m2_jit_entry(void) {
    /* E5M2: exponent field e = bits 6:2 (bias 15), mantissa m = bits 1:0,
     * max exponent 31: m==0 -> +/-inf, m!=0 -> qNaN. */
    return rvv_gemm_fp8_jit_entry_impl<&rvv_fp8_e5m2_to_f32_bits, 112, 2, 3,
            0x40800000u, 0x37800000u, true>();
}
