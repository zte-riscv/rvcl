# *******************************************************************************
# RVCL — RISC-V Compute Library
# SPDX-License-Identifier: Apache-2.0
#
# RISC-V 64 cross-compilation toolchain file.
#
# Mirrors the compiler the riscv64 CI installs (g++-14-riscv64-linux-gnu,
# see .github/automation/riscv/ci.json). The cross gcc default -march is
# rv64gc (no V): the RVV sources compile to the C-portable fallback
# unless RVCL_RVV_MARCH raises the vector baseline for them
# (CMakeLists.txt, RVV backend block) — that is the CI "baseline" tier.
# *******************************************************************************

set(CMAKE_SYSTEM_NAME Linux)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Cross-compilation tools
set(CMAKE_CXX_COMPILER riscv64-linux-gnu-g++-14)

# Search paths
set(CMAKE_FIND_ROOT_PATH /usr/riscv64-linux-gnu)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_PACKAGE ONLY)
