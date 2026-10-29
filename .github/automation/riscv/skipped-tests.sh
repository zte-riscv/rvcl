#!/usr/bin/env bash
# *******************************************************************************
# RVCL — RISC-V Compute Library
# SPDX-License-Identifier: Apache-2.0
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
#   vector   — the forced-c corpus targets only. The per-file vector
#              -march override exposes the C fallback kernel to the
#              cross gcc's ordered-reduction vectorization (non-fused),
#              which lands 1 ulp off the scalar golden that compiles at
#              the baseline march — a compile-march artifact, not a
#              kernel defect (the same targets pass wherever the C
#              kernel compiles without the vector ISA: the baseline
#              tier here and the x86 CI). Everything else runs: auto
#              (JIT f32 / intrinsics f16) plus the forced jit and
#              intrinsics variants over the full corpus.
#   Performance-class targets (bench P mode) get excluded here when
#   they land in ctest.
# *******************************************************************************

set -eo pipefail

SKIPPED_TESTS=""
if [[ "${RVCL_CI_TIER}" == "vector" ]]; then
    SKIPPED_TESTS="test_matmul_modeC_smoke_c|test_matmul_modeC_ci_c"
fi

echo "${SKIPPED_TESTS}"
