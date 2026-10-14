#!/usr/bin/env bash
# *******************************************************************************
# RVCL — RISC-V Compute Library
# SPDX-License-Identifier: Apache-2.0
#
# Common variables for the RISC-V CI. Exports: CXX, MP, CMAKE_TOOLCHAIN_FILE.
# *******************************************************************************

set -o errexit -o pipefail -o noclobber

export OS="$(uname)"

# Number of parallel build jobs.
if [[ "${OS}" == "Linux" ]]; then
    export MP="-j$(nproc)"
fi

# Cross-compilation mode: gcc version comes from ci.json via the workflow.
export CXX="riscv64-linux-gnu-g++-${GCC_VERSION:?GCC_VERSION must be set (ci.json)}"
export CMAKE_TOOLCHAIN_FILE="$(dirname "$(readlink -f "$0")")/../../../cmake/toolchains/riscv64.cmake"

echo "Cross-compilation mode for RISC-V"
echo "CXX: ${CXX}"
echo "CMAKE_TOOLCHAIN_FILE: ${CMAKE_TOOLCHAIN_FILE}"
