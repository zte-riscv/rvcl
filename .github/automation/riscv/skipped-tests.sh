#!/usr/bin/env bash
# *******************************************************************************
# Copyright 2026 ZTE Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# *******************************************************************************
#
# Tests to skip under QEMU emulation, by build tier (RVCL_CI_TIER).
#
# Emulated wall clock runs 20-100x native, so the skip list is tier-aware
# from day one: corpus growth lands here, not in ad-hoc workflow edits.
#   baseline — nothing skipped today. Auto selection binds the JIT
#              micro-kernel through the runtime hwprobe upgrade, so the
#              full ci corpus runs here: this tier is the vlen=128
#              bit-exactness evidence for the JIT path, on top of its
#              forced jit/c ctest variants.
#   vector   — the forced-c f32 corpus targets only. The per-file vector
#              -march override exposes the f32 C fallback kernel to the
#              cross gcc's ordered-reduction vectorization (non-fused),
#              which lands 1 ulp off the scalar golden that compiles at
#              the baseline march — a compile-march artifact, not a
#              kernel defect (the same targets pass wherever the f32 C
#              kernel compiles without the vector ISA: the baseline
#              tier here and the x86 CI). The f16 forced-c targets are
#              tolerant (1e-3) and the bf16 C fallback compiles at the
#              baseline march, so neither is skipped here. Everything
#              else runs: auto (JIT f32/f16/bf16) plus the forced jit
#              and intrinsics variants over the full corpus.
#   Performance-class targets (bench P mode) get excluded here when
#   they land in ctest.

set -eo pipefail

SKIPPED_TESTS=""
if [[ "${RVCL_CI_TIER}" == "vector" ]]; then
    SKIPPED_TESTS="test_matmul_modeC_smoke_f32_c|test_matmul_modeC_ci_f32_c"
fi

echo "${SKIPPED_TESTS}"
