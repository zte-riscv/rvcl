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
 * rvv_gemm_f32_jit.cpp — FP32 GEMM JIT micro-kernel (xbyak_riscv)
 *
 * Run-time generated counterpart of the f32 micro-kernels in
 * rvv_gemm_f32.cpp: this translation unit compiles at the baseline
 * rv64gc -march, and vector code exists only after a run-time capability
 * probe — the property that keeps the whole backend in a single
 * baseline binary.
 *
 * The generated loop mirrors rvv_gemm_f32_c lane for lane: per output
 * column j the accumulator starts at bias[j] and takes one fused
 * multiply-add per k in ascending order, so the JIT path is bit-identical
 * to the C path under the same inputs. One code object serves every
 * VLEN: the AVL is clamped at run time to min(RVV_PACK_NC, columns
 * left, lanes left in the current panel row), which at VLEN=128 splits
 * each 8-wide panel into two strips and at VLEN>=256 consumes it in one.
 *
 * Host side: rvv_gemm_f32_jit_entry() builds the code object once per
 * process (magic static) and hands out a plain function pointer, so the
 * run path performs no allocation. A generation failure is caught,
 * reported once and latched as NULL, letting the auto selection degrade
 * to the C kernel instead of crashing.
 *
 * The macros below configure the vendored generator before its headers
 * are included. XBYAK_RISCV_V selects the vector instruction API.
 * XBYAK_RISCV_VSETV_DEFAULT_OLD opts out of the generator's backwards
 * compatible vsetvli/vsetivli default arguments (LMUL=m1, VTA=tu,
 * VMA=mu), forcing every call site to pass them explicitly — kernels
 * here always emit VTA::ta, VMA::ma.
 ******************************************************************************/

#include "../rvv_internal.hpp"

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

/*
 * Row x panel strip kernel, ABI-mapped to rvv_gemm_fn:
 *
 *   for i in [r->m_begin, r->m_end)
 *     for j0 stepping over [r->n_begin, r->n_end) in vl-sized strips
 *       vl  = min(NC, n_end - j0, NC - j0 % NC)
 *       acc = bias ? bias[j0 .. j0+vl) : 0
 *       for k in 0..K: acc += A[i,k] * B[panel(j0)][k][j0%NC .. +vl)
 *       C[i, j0 .. j0+vl) = acc
 *
 * The panel clamp inside vl keeps every vector load within one k-row of
 * one panel, so strips may start mid-panel on a non-NC-aligned n_begin
 * without reading across the row boundary. All eight arguments arrive in
 * a0..a7 (LP64D, no stack arguments) and the kernel is a leaf. The
 * 15 caller-saved GPRs minus the four base pointers and two shape
 * scalars are one short of the two k-loop cursors, so s0/s1 are
 * borrowed behind a 16-byte-aligned frame.
 */
class rvv_gemm_f32_jit_t : public CodeGenerator {
public:
    rvv_gemm_f32_jit_t() : CodeGenerator(size_t {64} * 1024) {
        generate();
        ready(PROTECT_RWE); /* mprotect RWX + fence.i icache sync */
    }

private:
    void generate() {
        /* arguments; a4 carries m, which is dead because the work range
         * bounds every loop, so the register is reused for vl */
        const Reg a = a0, bp = a1, bias = a2, c = a3, vl = a4, n = a5, k = a6,
                  r = a7;
        /* live state: loop cursors and trip counts */
        const Reg i = t0, m_end = t1, n_end = t2, j0 = t3, av = t4, tp = t5,
                  p = t6;
        const Reg acur = s0, bcur = s1; /* k-loop address cursors */
        const FReg fa = fa0; /* broadcast scalar A[i,k] */
        const VReg vacc = v1, vb = v2; /* v0 stays free for masks */

        Label row, strip, avl_nc, avl_panel, no_bias, khead, kloop, store,
                row_end, done;

        addi(sp, sp, -16); /* RV64 psABI: sp stays 16-byte aligned */
        sd(s0, sp, 0);
        sd(s1, sp, 8);

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
        andi(tp, j0, 7);
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
        /* A row base and the B panel cursor are re-derived per strip: a
         * strip may start mid-panel, and the k loop advanced the previous
         * strip's cursors. */
        mul(tp, i, k);
        slli(tp, tp, 2);
        add(acur, a, tp); /* &A[i, 0] */
        srai(tp, j0, 3); /* panel = j0 / NC */
        mul(tp, tp, k);
        slli(tp, tp, 5); /* panel stride: K rows x NC lanes x 4 bytes */
        add(bcur, bp, tp);
        andi(tp, j0, 7);
        slli(tp, tp, 2);
        add(bcur, bcur, tp); /* &B[panel][0][j0 % NC] */
        li(p, 0);
        L(kloop);
        bge(p, k, store);
        flw(fa, acur, 0); /* A[i, p] */
        vle32_v(vb, bcur); /* B[p, j0 .. j0+vl) */
        vfmacc_vf(vacc, fa, vb); /* acc += A[i,p] * B[p, j0..j0+vl) */
        addi(acur, acur, 4);
        addi(bcur, bcur, 32); /* next k row of the panel: NC x 4 bytes */
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
        ld(s1, sp, 8);
        ld(s0, sp, 0);
        addi(sp, sp, 16);
        ret();
    }
};

} // namespace

int rvv_gemm_f32_jit_available(void) {
    /* single source of truth for "can vector code run here": the rvcl
     * capability layer (compile-time macros upgraded by hwprobe), not a
     * second probe inside the JIT framework */
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_f32_jit_entry(void) {
    /* generated once per process; a failure latches NULL so the warning
     * fires exactly once and callers can degrade */
    static rvv_gemm_fn entry = []() -> rvv_gemm_fn {
        try {
            static rvv_gemm_f32_jit_t kern;
            return kern.getCode<rvv_gemm_fn>();
        } catch (const std::exception &e) {
            fprintf(stderr, "rvcl: f32 JIT kernel generation failed: %s\n",
                    e.what());
            return nullptr;
        } catch (...) {
            fprintf(stderr, "rvcl: f32 JIT kernel generation failed\n");
            return nullptr;
        }
    }();
    return entry;
}
