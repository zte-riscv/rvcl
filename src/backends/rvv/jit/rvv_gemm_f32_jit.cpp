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
 * The macros below configure the vendored generator before its headers
 * are included. XBYAK_RISCV_V selects the vector instruction API.
 * XBYAK_RISCV_VSETV_DEFAULT_OLD opts out of the generator's backwards
 * compatible vsetvli/vsetivli default arguments (LMUL=m1, VTA=tu,
 * VMA=mu), forcing every call site to pass them explicitly — kernels
 * here always emit VTA::ta, VMA::ma.
 ******************************************************************************/

#define XBYAK_RISCV_V 1
#define XBYAK_RISCV_VSETV_DEFAULT_OLD 0

#include "xbyak_riscv/xbyak_riscv.hpp"

// Pins the vendored copy to the upstream tag recorded in
// THIRD-PARTY-PROGRAMS: 0x1310 encodes version 1.31.
static_assert(Xbyak_riscv::VERSION == 0x1310,
        "vendored xbyak_riscv must stay at tag 1.31");
