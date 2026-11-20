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
# Test rvcl for RISC-V 64: run the cross-built ctest suite under QEMU
# linux-user (registered through binfmt by the workflow).
#
# Environment (set by the workflow):
#   QEMU_CPU              emulated CPU properties incl. the vlen tier
#   RVCL_CI_TIER          baseline | vector — drives skipped-tests.sh
#   RVCL_EXPECT_FEATURES  capability gate expectations, e.g. "rvv,fp16"
#   CTEST_PARALLEL_LEVEL  test parallelism

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# binfmt_misc routes the riscv64 ELFs through qemu-riscv64; the cross
# sysroot provides the guest dynamic loader and glibc.
export QEMU_LD_PREFIX=/usr/riscv64-linux-gnu

# hwprobe emulation needs QEMU >= 8.2; print the version so a capability
# failure can be attributed without rerunning the job.
qemu-riscv64-static --version 2>/dev/null | head -n 1 || true

# Environment gate FIRST: capability detection degrades by design,
# so a misconfigured emulator (e.g. QEMU too old to emulate hwprobe)
# would silently downgrade and the numerical targets would still pass
# green through the C fallback. RVCL_EXPECT_FEATURES turns "the tier we
# built" into a hard assertion before anything numerical runs.
set -x
ctest --no-tests=error --output-on-failure -R '^capability_assert$'
set +x

# Then everything else, minus the tier's skip list.
skipped="$("${SCRIPT_DIR}"/skipped-tests.sh)"
exclude='capability_assert'
if [[ -n "${skipped}" ]]; then
    exclude="${exclude}|${skipped}"
fi
set -x
ctest --no-tests=error --output-on-failure -E "${exclude}"
set +x
