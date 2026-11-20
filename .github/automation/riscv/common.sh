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
# Common variables for the RISC-V CI. Exports: CXX, MP, CMAKE_TOOLCHAIN_FILE.

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
