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
#   vector   — nothing skipped today. Auto runs the ci corpus through
#              the JIT f32 kernel (and the intrinsics f16 kernel); the
#              forced c and intrinsics variants keep both legacy
#              regression surfaces covered on this tier.
#   Performance-class targets (bench P mode) get excluded here when
#   they land in ctest.
# *******************************************************************************

set -eo pipefail

SKIPPED_TESTS=""

echo "${SKIPPED_TESTS}"
