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
 * every byte through the same 256-entry f32 lookup tables as the C
 * kernels and the golden (rvv_fp8.hpp): the scalar A element is loaded
 * with lbu and indexed into the table, and the NC-wide B strip is loaded
 * with vle8.v then gathered through the table with vluxei8.v (a
 * byte-indexed gather producing f32 lanes). The table address is a
 * compile-time static, so it is embedded as a 64-bit immediate once,
 * before the K loop. Per output column j the accumulator starts at
 * bias[j] and takes one fused multiply-add per k in ascending order —
 * bit-identical to the C path under the same inputs. One code object
 * serves every VLEN (the AVL is clamped like the f32 JIT).
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
 * f32 upconversion table. All eight kernel arguments arrive in a0..a7
 * (LP64D, no stack arguments) and the kernel is a leaf. */
template <const uint32_t (*TABLE)[256]>
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
        const FReg fa = fa0;
        const VReg vacc = v1, vidx = v2, vb = v3; /* v0 stays free for masks */

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
        vle8_v(vidx, bcur); /* B[p, j0 .. j0+vl) as byte indices */
        vluxei8_v(vb, tb, vidx); /* gather f32 lanes from the table */
        vfmacc_vf(vacc, fa, vb); /* acc += A[i,p] * B[p, j0..j0+vl) */
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

template <const uint32_t (*TABLE)[256]>
rvv_gemm_fn rvv_gemm_fp8_jit_entry_impl(void) {
    static rvv_gemm_fn entry = []() -> rvv_gemm_fn {
        try {
            static rvv_gemm_fp8_jit_t<TABLE> kern;
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
    return rvv_gemm_fp8_jit_entry_impl<&rvv_fp8_e4m3_to_f32_bits>();
}

int rvv_gemm_fp8_e5m2_jit_available(void) {
    return rvcl_capability_has(rvcl_get_capability(), RVCL_FEATURE_RVV);
}

rvv_gemm_fn rvv_gemm_fp8_e5m2_jit_entry(void) {
    return rvv_gemm_fp8_jit_entry_impl<&rvv_fp8_e5m2_to_f32_bits>();
}
