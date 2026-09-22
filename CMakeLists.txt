cmake_minimum_required(VERSION 3.16)
project(rvcl
    VERSION 0.1.0
    DESCRIPTION "RISC-V Compute Library — unified kernel abstraction and dispatch runtime"
    LANGUAGES C)

# ------------------------------------------------------------------ options
# Backends are independently buildable (PLAN §4.12).
option(RVCL_ENABLE_RVV   "Enable RVV backend"        ON)  # v0.1
option(RVCL_ENABLE_IME   "Enable IME backend"        OFF) # lands in v0.2
option(RVCL_ENABLE_VME   "Enable VME backend"        OFF) # lands in v0.3
option(RVCL_ENABLE_AME   "Enable AME backend"        OFF) # lands in v0.3
option(RVCL_BUILD_TESTS  "Build tests"               ON)
option(RVCL_BUILD_EXAMPLES "Build examples"          ON)
option(RVCL_BUILD_BENCH  "Build benchmark harness"   ON)

# The reference backend is always built: golden path + host-agnostic CI.
set(RVCL_ENABLE_REFERENCE ON)

# Guard against in-source builds: a stale ./CMakeCache.txt next to the
# sources builds an out-of-date configuration (e.g. RVV disabled) that
# shadows build/ and breaks `make` at the repo root.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/CMakeCache.txt
                ${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles)
    message(FATAL_ERROR
        "In-source builds are disabled. Use:\n"
        "    cmake -B build && cmake --build build\n"
        "or simply `make` at the repo root (wrapper Makefile).")
endif()

set(RVCL_RUNTIME_SOURCES
    src/common/types.c
    src/common/memory.c
    src/runtime/capability.c
    src/runtime/kernel.c
    src/runtime/registry.c
    src/runtime/dispatch.c
    src/runtime/init.c
    src/runtime/scheduler/scheduler.c
    src/operators/matmul.c
    src/backends/reference/reference_matmul.c
)

# ------------------------------------------------------------ RVV backend
# Compiles on any host: the micro-kernels are guarded by __riscv_vector;
# on non-RISC-V hosts the translation units reduce to the C-portable
# fallback so the framework side (packing, registration, cache) is fully
# testable on x86 CI (PLAN §6.2 item 12).
if(RVCL_ENABLE_RVV)
    list(APPEND RVCL_RUNTIME_SOURCES
        src/backends/rvv/rvv_backend.c
        src/backends/rvv/rvv_pack.c
        src/backends/rvv/rvv_gemm_f32.c
        src/backends/rvv/rvv_gemm_f16.c
    )
endif()

add_library(rvcl SHARED ${RVCL_RUNTIME_SOURCES})

set_target_properties(rvcl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    C_STANDARD 11
    C_STANDARD_REQUIRED YES
    PUBLIC_HEADER
        "include/rvcl/rvcl.h;include/rvcl/types.h;include/rvcl/init.h;\
include/rvcl/capability.h;include/rvcl/memory.h;include/rvcl/scheduler.h;\
include/rvcl/kernel.h;include/rvcl/registry.h;include/rvcl/dispatch.h;\
include/rvcl/matmul.h"
)

target_include_directories(rvcl
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)

target_compile_options(rvcl PRIVATE -Wall -Wextra -fvisibility=hidden)
find_package(Threads REQUIRED)
target_link_libraries(rvcl PUBLIC Threads::Threads)

target_compile_definitions(rvcl PRIVATE RVCL_BUILDING)

if(RVCL_ENABLE_RVV)
    target_compile_definitions(rvcl PRIVATE RVCL_ENABLE_RVV_BUILD)
endif()

# On real RVV targets ensure the vector extension is enabled in -march.
if(RVCL_ENABLE_RVV AND CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(RVCL_RVV_MARCH "" CACHE STRING "Extra -march vector flags for RVV files")
    if(RVCL_RVV_MARCH)
        set_source_files_properties(
            src/backends/rvv/rvv_pack.c
            src/backends/rvv/rvv_gemm_f32.c
            src/backends/rvv/rvv_gemm_f16.c
            PROPERTIES COMPILE_OPTIONS "-march=${RVCL_RVV_MARCH}")
    endif()
endif()

# Backends (RVCL_ENABLE_*): each adds sources + a rvcl_<backend>_register()
# call in init.c's register_all(). Gated per PLAN §4.12:
#   cmake -DRVCL_ENABLE_RVV=ON -DRVCL_ENABLE_IME=OFF ...

# ------------------------------------------------------------------- tests
if(RVCL_BUILD_TESTS)
    enable_testing()
    add_executable(rvcl_smoke tests/smoke_test.c)
    target_link_libraries(rvcl_smoke PRIVATE rvcl)
    add_test(NAME smoke COMMAND rvcl_smoke)

    add_executable(rvcl_differential tests/differential_test.c)
    target_link_libraries(rvcl_differential PRIVATE rvcl m)
    add_test(NAME differential COMMAND rvcl_differential)
endif()

# ---------------------------------------------------------------- examples
if(RVCL_BUILD_EXAMPLES)
    add_executable(example_matmul examples/matmul.c)
    target_link_libraries(example_matmul PRIVATE rvcl)

    add_executable(example_dispatch examples/dispatch.c)
    target_link_libraries(example_dispatch PRIVATE rvcl)

    add_executable(example_scheduler examples/scheduler.c)
    target_link_libraries(example_scheduler PRIVATE rvcl)
endif()

# -------------------------------------------------------------------- bench
if(RVCL_BUILD_BENCH)
    add_executable(rvcl_bench tests/bench_matmul.c)
    target_link_libraries(rvcl_bench PRIVATE rvcl m)
endif()

# ------------------------------------------------------------------ install
include(GNUInstallDirs)
install(TARGETS rvcl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rvcl)
