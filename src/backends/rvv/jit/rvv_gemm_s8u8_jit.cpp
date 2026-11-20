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
 * rvv_gemm_s8u8_jit.cpp — INT8 (signed/unsigned byte) GEMM JIT micro-kernel
 * (xbyak_riscv)
 *
 * Run-time generated counterpart of the C kernels in rvv_gemm_s8u8.cpp:
 * this translation unit compiles at the baseline rv64gc -march, and vector
 * code exists only after a run-time capability probe — the property that
 * keeps the whole backend in a single baseline binary.
 *
 * Tuple (s8,s8,s32,s32) and (u8,u8,s32,s32): the NC-wide B strip is loaded
 * with vle8.v and widened to e32 with two widening multiplies by the
 * scalar 1 (vwmul.vx for signed, vwmulu.vx for unsigned): e8 -> e16 -> e32.
 * Two steps are required because there is no single-instruction 8-to-32
 * sign extension in base RVV (vzext.vf4 only zero-extends, and vsext.vf2
 * traps on some hardware, see D-7 — so the mandated form is the widening
 * multiply). The scalar A element is loaded with lb (signed) or lbu
 * (unsigned), which sign-/zero-extends it in the GPR to a full int32
 * scalar, and the accumulator is advanced with one vmacc.vx per k in
 * ascending order (bias first) — bit-identical to the C path. Both dtypes
 * differ only in the signedness of the load and the widening multiply, so
 * one template serves both. One code object serves every VLEN (the AVL is
 * clamped like the f32 JIT).
 *
 * K limit: like the C path, the int32 accumulator is exact only while the
 * product sum stays in int32 range. The worst per-element product is 127*127
 * (s8) or 255*255 (u8), so a K reduction of length K can reach 16129*K or
 * 65025*K and wraps the int32 accumulator once K exceeds ~133k (s8) or ~33k
 * (u8). Keep K below those integer-exact bounds.
 *
 * Host side mirrors rvv_gemm_fp8_jit.cpp: the entry is built once per
 * process (magic static) and a generation failure is latched as NULL so
 * auto selection degrades to the C kernel.
 ******************************************************************************/

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

/* Shared row x panel strip kernel parameterized by the signedness of the
 * byte dtypes. All eight kernel arguments arrive in a0..a7 (LP64D, no
 * stack arguments) and the kernel is a leaf. */
template <bool SIGNED>
class rvv_gemm_s8u8_jit_t : public CodeGenerator {
public:
    rvv_gemm_s8u8_jit_t() : CodeGenerator(size_t {64} * 1024) {
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
                  one = s2; /* scalar 1 (widening multiply), loop-invariant */
        const Reg aval = s3; /* scalar A[i, p] byte, widened in the GPR */
        /* v1: e32/m1 int32 accumulator. v3: the 8-bit B strip at e8/m1 and
         * the widening source. v4: the m2-aligned e16 widening result.
         * v8: the m4-aligned e32 widening result, whose first register is
         * re-read at e32/m1 as the sign-/zero-extended B values for the
         * vmacc. The m4 group (v8..v11) and the m2 group (v4..v5) overlap
         * no other live register. v0 stays free for the mask registers. */
        const VReg vacc = v1, vb = v3, ve16 = v4, ve32 = v8;

        Label row, strip, avl_nc, avl_panel, no_bias, khead, kloop, store,
                row_end, done;

        addi(sp, sp, -32); /* RV64 psABI: sp stays 16-byte aligned */
        sd(s0, sp, 0);
        sd(s1, sp, 8);
        sd(s2, sp, 16);
        sd(s3, sp, 24);

        li(one, 1); /* the widening multiply by 1 (D-7 sign extension) */

        ld(i, r, 0); /* i = r->m_begin */
        ld(m_end, r, 8);
        ld(n_end, r, 24);

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
        slli(tp, j0, 2); /* int32 bias stride */
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
        if (SIGNED)
            lb(aval, acur, 0); /* sign-extends the byte to a 64-bit int */
        else
            lbu(aval, acur, 0); /* zero-extends the byte */
        /* Load the NC-wide B strip as 8-bit bytes and widen them to e32 in
         * two steps, each a widening multiply by the scalar 1. vwmul.vx
         * sign-extends and vwmulu.vx zero-extends its source, which is the
         * D-7-mandated replacement for vsext.vf2 (illegal-instruction trap
         * on some hardware). The vtype SEW/LMUL before each vwmul is the
         * *source* SEW and the source's LMUL: a widening multiply reads
         * vs2 at EMUL=LMUL and writes vd at EMUL=2*LMUL, so the 8->16 step
         * runs at e8/m1 (source m1 -> destination m2, registers v3 ->
         * v4..v5) and the 16->32 step at e16/m2 (source m2 -> destination
         * m4, registers v4..v5 -> v8..v11). The source and destination
         * register groups never overlap. The AVL stays at or below VLMAX at
         * every intermediate vtype, so VL is preserved through the
         * widening. The first register of the final m4 destination group
         * (v8) holds the first VLMAX(e32/m1) active widened e32 values,
         * re-read at e32/m1 — exactly the fp8 JIT's pattern. */
        vsetvli(vl, av, SEW::e8, LMUL::m1, VTA::ta, VMA::ma); /* e8/m1 */
        vle8_v(vb, bcur); /* B[p, j0 .. j0+vl) as bytes */
        vsetvli(vl, av, SEW::e8, LMUL::m1, VTA::ta, VMA::ma); /* 8->16 */
        if (SIGNED)
            vwmul_vx(ve16, vb, one); /* v4 (m2) = sign-extend bytes to e16 */
        else
            vwmulu_vx(ve16, vb, one); /* v4 (m2) = zero-extend bytes to e16 */
        vsetvli(vl, av, SEW::e16, LMUL::m2, VTA::ta, VMA::ma); /* 16->32 */
        if (SIGNED)
            vwmul_vx(ve32, ve16, one); /* v8 (m4) = sign-extend e16 to e32 */
        else
            vwmulu_vx(ve32, ve16, one); /* v8 (m4) = zero-extend e16 to e32 */
        vsetvli(vl, av, SEW::e32, LMUL::m1, VTA::ta, VMA::ma); /* compute */
        vmacc_vx(vacc, aval, ve32); /* acc += A[i,p] * B[p, j0..j0+vl) */
        addi(acur, acur, 1); /* A stride 1 byte */
        addi(bcur, bcur, RVV_PACK_NC); /* next k row: NC bytes */
        addi(p, p, 1);
        j_(kloop);
        L(store);
        mul(tp, i, n); /* &C[i, j0] */
        add(tp, tp, j0);
        slli(tp, tp, 2); /* int32 C stride */
        add(tp, c, tp);
        vse32_v(vacc, tp);
        add(j0, j0, vl); /* consuming the panel remainder re-aligns j0 */
        j_(strip);
        L(row_end);
        addi(i, i, 1);
        j_(row);
        L(done);
        ld(s3, sp, 24);
        ld(s2, sp, 16);
        ld(s1, sp, 8);
        ld(s0, sp, 0);
        addi(sp, sp, 32);
        ret();
    }
};

template <bool SIGNED>
rvv_gemm_fn rvv_gemm_s8u8_jit_entry_impl(void) {
    static rvv_gemm_fn entry = []() -> rvv_gemm_fn {
        try {
            static rvv_gemm_s8u8_jit_t<SIGNED> kern;
            return kern.template getCode<rvv_gemm_fn>();
        } catch (const std::exception &e) {
            fprintf(stderr, "rvcl: INT8 JIT kernel generation failed: %s\n",
                    e.what());
            return nullptr;
        } catch (...) {
            fprintf(stderr, "rvcl: INT8 JIT kernel generation failed\n");
            return nullptr;
        }
    }();
    return entry;
}

} // namespace

int rvv_gemm_s8_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_s8_jit_entry(void) {
    return rvv_gemm_s8u8_jit_entry_impl<true>();
}

int rvv_gemm_u8_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_u8_jit_entry(void) {
    return rvv_gemm_s8u8_jit_entry_impl<false>();
}
