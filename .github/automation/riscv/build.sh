#!/usr/bin/env bash
# *******************************************************************************
# RVCL — RISC-V Compute Library
# SPDX-License-Identifier: Apache-2.0
#
# Cross-compile rvcl for RISC-V 64.
#
# Environment:
#   RVCL_ACTION    configure | build   (required)
#   GCC_VERSION    cross gcc minor version, from ci.json via the workflow
#   RVCL_CI_TIER   baseline | vector   (required; see tier mapping below)
#   CMAKE_BUILD_TYPE  defaults to Release — the differential fp32 checks
#                     are bit-exact and need the optimized scalar golden
#                     (FMA contraction); do not lower it.
# *******************************************************************************

set -o errexit -o pipefail -o noclobber

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

# Defines CXX, MP and CMAKE_TOOLCHAIN_FILE.
source "${SCRIPT_DIR}/common.sh"

CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE:-"Release"}
RVCL_TEST_SET=${RVCL_TEST_SET:-"ci"}

if [[ "$RVCL_ACTION" == "configure" ]]; then
    # Build tiers (test report §3.3 Step 2):
    #   baseline — no -march override: the cross gcc default (rv64gc)
    #              compiles the RVV sources to the C-portable fallback.
    #              Runtime hwprobe still upgrades capability to RVV, so
    #              this tier validates dispatch selection and the
    #              C-fallback numerics.
    #   vector   — the RVV sources compile with the vector ISA: intrinsics
    #              micro-kernels live and capability reports RVV/FP16
    #              from the compile-time macros. Main tier.
    case "${RVCL_CI_TIER:?RVCL_CI_TIER must be set (baseline|vector)}" in
        baseline) RVCL_RVV_MARCH="" ;;
        vector)   RVCL_RVV_MARCH="rv64gcv_zvfh" ;;
        *) echo "Unknown RVCL_CI_TIER: ${RVCL_CI_TIER} (want baseline|vector)"
           exit 1 ;;
    esac

    set -x
    cmake -B build -S . -G Ninja \
        -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE}" \
        -DCMAKE_TOOLCHAIN_FILE="${CMAKE_TOOLCHAIN_FILE}" \
        -DRVCL_TEST_SET="${RVCL_TEST_SET}" \
        -DRVCL_ENABLE_WERROR=ON \
        -DRVCL_RVV_MARCH="${RVCL_RVV_MARCH}"
    set +x

elif [[ "$RVCL_ACTION" == "build" ]]; then
    # Tier-agnostic: builds whatever the configure step set up.
    set -x
    cmake --build build --parallel "${MP#-j}"
    set +x

else
    echo "Unknown action: $RVCL_ACTION"
    exit 1
fi
