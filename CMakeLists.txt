cmake_minimum_required(VERSION 3.16)
project(rvcl
    VERSION 0.1.0
    DESCRIPTION "RISC-V Compute Library — unified kernel abstraction and dispatch runtime"
    LANGUAGES CXX)

# ------------------------------------------------------------------ options
# Backends are independently buildable (PLAN §4.12).
option(RVCL_ENABLE_RVV   "Enable RVV backend"        ON)  # v0.1
option(RVCL_ENABLE_IME   "Enable IME backend"        OFF) # lands in v0.2
option(RVCL_ENABLE_VME   "Enable VME backend"        OFF) # lands in v0.3
option(RVCL_ENABLE_AME   "Enable AME backend"        OFF) # lands in v0.3
option(RVCL_BUILD_TESTS  "Build tests"               ON)
option(RVCL_BUILD_EXAMPLES "Build examples"          ON)
option(RVCL_BUILD_BENCH  "Build benchmark harness"   ON)
option(RVCL_ENABLE_WERROR "Treat compiler warnings as errors" OFF)

# The rvcl target always carries -Wall -Wextra; this gate extends the same
# bar to tests/examples/bench and makes any warning fatal.
if(RVCL_ENABLE_WERROR)
    add_compile_options(-Wall -Wextra -Werror)
endif()

# The reference backend is always built: golden path + host-agnostic CI.
set(RVCL_ENABLE_REFERENCE ON)

# Guard against in-source builds: a stale ./CMakeCache.txt next to the
# sources builds an out-of-date configuration (e.g. RVV disabled) that
# shadows build/.
if(CMAKE_CURRENT_SOURCE_DIR STREQUAL CMAKE_CURRENT_BINARY_DIR)
    file(REMOVE ${CMAKE_CURRENT_SOURCE_DIR}/CMakeCache.txt
                ${CMAKE_CURRENT_SOURCE_DIR}/CMakeFiles)
    message(FATAL_ERROR
        "In-source builds are disabled. Use:\n"
        "    cmake -B build && cmake --build build")
endif()

set(RVCL_RUNTIME_SOURCES
    src/common/types.cpp
    src/common/memory.cpp
    src/runtime/capability.cpp
    src/runtime/kernel.cpp
    src/runtime/registry.cpp
    src/runtime/dispatch.cpp
    src/runtime/init.cpp
    src/runtime/scheduler/scheduler.cpp
    src/operators/matmul.cpp
    src/backends/reference/reference_matmul.cpp
)

# ------------------------------------------------------------ RVV backend
# Compiles on any host: the micro-kernels are guarded by __riscv_vector;
# on non-RISC-V hosts the translation units reduce to the C-portable
# fallback so the framework side (packing, registration, cache) is fully
# testable on x86 CI (PLAN §6.2 item 12).
if(RVCL_ENABLE_RVV)
    list(APPEND RVCL_RUNTIME_SOURCES
        src/backends/rvv/rvv_backend.cpp
        src/backends/rvv/rvv_pack.cpp
        src/backends/rvv/rvv_gemm_f32.cpp
        src/backends/rvv/rvv_gemm_f16.cpp
    )
endif()

add_library(rvcl SHARED ${RVCL_RUNTIME_SOURCES})

set_target_properties(rvcl PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED YES
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
# rvv_backend.cpp MUST be in the list: create_f32/create_f16 bind the
# intrinsics micro-kernels only when __riscv_vector is defined in that
# translation unit — without the override they silently bind the C
# fallback and the intrinsics path is unreachable. capability.cpp MUST
# be in the list too: the compile-time detection layer only reports RVV
# / FP16 features when the __riscv_* macros are visible there.
if(RVCL_ENABLE_RVV AND CMAKE_SYSTEM_PROCESSOR MATCHES "riscv")
    set(RVCL_RVV_MARCH "" CACHE STRING "Extra -march vector flags for RVV files")
    if(RVCL_RVV_MARCH)
        set_source_files_properties(
            src/runtime/capability.cpp
            src/backends/rvv/rvv_backend.cpp
            src/backends/rvv/rvv_pack.cpp
            src/backends/rvv/rvv_gemm_f32.cpp
            src/backends/rvv/rvv_gemm_f16.cpp
            PROPERTIES COMPILE_OPTIONS "-march=${RVCL_RVV_MARCH}")
    endif()
endif()

# Backends (RVCL_ENABLE_*): each adds sources + a rvcl_<backend>_register()
# call in init.cpp's register_all(). Gated per PLAN §4.12:
#   cmake -DRVCL_ENABLE_RVV=ON -DRVCL_ENABLE_IME=OFF ...

# ------------------------------------------------------------------- tests
# Shared golden helpers: the single data-semantics point for the test
# executables (differential + bench C mode). Corpus parsers stay
# deliberately double-implemented — cross-validation is the feature.
if(RVCL_BUILD_TESTS OR RVCL_BUILD_BENCH)
    add_library(rvcl_test_golden STATIC tests/golden.cpp)
endif()

if(RVCL_BUILD_TESTS)
    enable_testing()

    # Environment gate, registered first: prints the capability summary
    # and asserts RVCL_EXPECT_FEATURES (comma list) when set. The
    # riscv64 CI runs it before any numerical target — a misconfigured
    # emulator (e.g. QEMU too old for hwprobe) must fail loudly instead
    # of degrading to a green-but-meaningless run. Without the variable
    # it is print-only, so local ctest runs stay host-agnostic.
    add_executable(rvcl_capability_assert tests/capability_assert.cpp)
    target_link_libraries(rvcl_capability_assert PRIVATE rvcl)
    add_test(NAME capability_assert COMMAND rvcl_capability_assert)

    add_executable(rvcl_smoke tests/smoke_test.cpp)
    target_link_libraries(rvcl_smoke PRIVATE rvcl)
    add_test(NAME smoke COMMAND rvcl_smoke)

    add_executable(rvcl_differential tests/differential_test.cpp)
    target_link_libraries(rvcl_differential PRIVATE rvcl m rvcl_test_golden)
    add_test(NAME differential COMMAND rvcl_differential)

    # Corpus registration: one line per corpus file -> one modeC ctest
    # target running the differential (the assertion engine) over it.
    # RVCL_TEST_SET picks the tier at configure time (smoke < ci < all,
    # default smoke — fast local builds; CI raises it explicitly);
    # run-time LABELS allow `ctest -L corpus|smoke|ci` selection. The <op>
    # parameter is the multi-operator landing point: a new operator
    # directory registers the same way.
    set(RVCL_TEST_SET "smoke" CACHE STRING
        "Corpus tier registered as ctest targets: smoke | ci | all")
    if(NOT RVCL_TEST_SET MATCHES "^(smoke|ci|all)$")
        message(FATAL_ERROR "RVCL_TEST_SET must be smoke, ci or all")
    endif()
    function(rvcl_add_corpus op set file)
        if(RVCL_TEST_SET STREQUAL "all"
                OR (RVCL_TEST_SET STREQUAL "ci" AND NOT "${set}" STREQUAL "weekly")
                OR (RVCL_TEST_SET STREQUAL "smoke" AND "${set}" STREQUAL "smoke"))
            add_test(NAME test_${op}_modeC_${set}
                COMMAND rvcl_differential
                        --batch=${CMAKE_CURRENT_SOURCE_DIR}/tests/inputs/${op}/${file})
            set_tests_properties(test_${op}_modeC_${set}
                PROPERTIES LABELS "corpus;${set}")
        endif()
    endfunction()

    rvcl_add_corpus(matmul smoke test_matmul_smoke)
    rvcl_add_corpus(matmul ci    test_matmul_ci)
endif()

# ---------------------------------------------------------------- examples
if(RVCL_BUILD_EXAMPLES)
    add_executable(example_matmul examples/matmul.cpp)
    target_link_libraries(example_matmul PRIVATE rvcl)
    add_test(NAME example_matmul COMMAND example_matmul)

    add_executable(example_dispatch examples/dispatch.cpp)
    target_link_libraries(example_dispatch PRIVATE rvcl)
    add_test(NAME example_dispatch COMMAND example_dispatch)

    add_executable(example_scheduler examples/scheduler.cpp)
    target_link_libraries(example_scheduler PRIVATE rvcl)
    add_test(NAME example_scheduler COMMAND example_scheduler)
endif()

# -------------------------------------------------------------------- bench
if(RVCL_BUILD_BENCH)
    add_executable(rvcl_bench tests/bench_matmul.cpp)
    target_link_libraries(rvcl_bench PRIVATE rvcl m rvcl_test_golden)
endif()

# ------------------------------------------------------------------ install
include(GNUInstallDirs)
install(TARGETS rvcl
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rvcl)
