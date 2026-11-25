# RVCL — RISC-V Compute Library

[![CI x86](https://github.com/zte-riscv/rvcl/actions/workflows/ci-x86.yml/badge.svg?branch=main)](https://github.com/zte-riscv/rvcl/actions/workflows/ci-x86.yml)
[![CI riscv64](https://github.com/zte-riscv/rvcl/actions/workflows/ci-riscv.yml/badge.svg?branch=main)](https://github.com/zte-riscv/rvcl/actions/workflows/ci-riscv.yml)

**A unified compute kernel abstraction and dispatch runtime for RISC-V vector
and matrix architectures — covering RVV, IME, VME, AME and vendor extensions,
with capability detection, kernel registry, dispatch, and framework
integration.**

RVCL (RISC-V Compute Library) is an open-source project, licensed under
Apache-2.0 with a DCO contribution process. It provides **one rv64 operator API
that at run time picks the best kernel for the rv64 hardware it finds**,
featuring four stages — **Capability** (what can this machine do) → **Registry**
(which kernel implementations exist) → **Dispatch** (which one should run) →
**Kernel** (do the actual math) — with **Scheduler** and **Memory** as
cross-cutting contracts that keep parallelism and allocation under the host's
control.

---

## Table of Contents

- [Why RVCL](#why-rvcl)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Extending rvcl](#extending-rvcl)
- [Project Layout](#project-layout)
- [Testing and CI](#testing-and-ci)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why RVCL

RISC-V's compute extensions are more scattered than any other architecture's.
The vector extension (RVV) is ratified, but VLEN ranges from 128 to 1024 bits;
the matrix extensions (IME, VME, AME) sit at different points between draft and
stabilization; vendor-private extensions evolve independently. An operator
developer writing directly against each of them pays three costs repeatedly:

- **framework side** — oneDNN, PyTorch and friends must maintain a separate
  backend path per hardware target;
- **kernel side** — scheduling, memory and type conversion are re-solved in
  every implementation, though none of it is ISA-specific;
- **ecosystem side** — every new piece of silicon forces the layers above it to
  be rewritten rather than recompiled.

The tempting alternative is one adapter directory per vendor library inside the
framework. That cost grows linearly with the number of vendors — an independent
review, an adapter directory, a build dependency, a support-list line, plus
threading and memory conventions to reconcile and audit for each. Since the
vendor × extension × VLEN combination space has no enumerable upper bound, the
linear path is not sustainable. That observation is the direct reason RVCL
exists.

To address these issues, we introduce the RVCL project to **collapse the
hardware differences inside one kernel-level layer**. The abstraction sits at
the kernel level with single-threaded, no-allocation-inside-the-kernel operators
that hand thread and memory control back to the host. RVCL includes four layers:

| Layer      | Responsibility                                                            |
| ---------- | ------------------------------------------------------------------------- |
| Capability | Detect ISA features, datatype extensions and vendor interfaces            |
| Registry   | Index kernel metadata by operation × (A,B,acc,C) dtype tuple              |
| Dispatch   | Six-stage filtering plus cost ordering, pick the best legal kernel        |
| Kernel     | `validate / configure / run` three-phase lifecycle                        |

Two cross-cutting contracts stay under host control and invisible to kernels:

- **Scheduler** — parallelism belongs to the host. Kernels express work as
  work-range slices and never create threads; a host may slice work ranges
  itself or inject an orchestrator via `rvcl_scheduler_set()`.
- **Memory** — zero-copy import (`rvcl_tensor_init()` only records metadata and
  borrows the host pointer) plus workspace declaration: kernels report size,
  alignment and lifetime in `configure()`, the host allocates, and the buffers
  come back through AUX slots. No `malloc` happens on a kernel's `run()` path.

## Architecture

```text
┌───────────────────────────────────────────────────┐
│ Operator API  rvcl_matmul / _scaled / rvcl_conv2d │
│               rvcl_inner_product                  │
├───────────────────────────────────────────────────┤
│ Dispatch       six-stage filter + cost ordering   │
├───────────────────────────────────────────────────┤
│ Registry       operation x dtype tuple index      │
├───────────────────────────────────────────────────┤
│ Kernel         validate / configure / run         │
└──────────────────────────┬────────────────────────┘
                           ▼
                      RISC-V ISA
```

Capability feeds dispatch from the side; Scheduler and Memory cross every layer.
The layers are coupled by metadata, not by code: extending one layer does not
touch the implementation body of its neighbours.

**Dispatch** runs six stages — operation, datatype tuple, required-feature
subset, VLEN tier, shape bounds, kernel `validate`. Surviving candidates are
ordered by each kernel's own cost callback; equal costs keep the earliest
registration, and the reference kernel is always the unbounded fallback. The
winning selection is reused through an 8-slot direct-mapped cache whose key
covers every request field; any registry change invalidates the whole cache.

## Quick Start

Requirements: CMake ≥ 3.16, a C++17 compiler, pthread.

```bash
git clone https://github.com/zte-riscv/rvcl.git && cd rvcl
cmake -B build -S .
cmake --build build
ctest --test-dir build
./build/example_matmul
```

Runnable examples (each registered as a ctest target):

| Target              | Shows                                                                                                                    |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| `example_matmul`    | one-shot `rvcl_matmul()` and what dispatch selected                                                                      |
| `example_dispatch`  | dispatch once / run many, caller-allocated workspace, pack-once, cache counters                                          |
| `example_scheduler` | injecting a host thread pool through `rvcl_ischeduler_t`                                                                 |
| `example_multicore` | `pool` (host-injected pool) and `builtin` (zero-injection, needs `RVCL_ENABLE_OMP`) modes, each sweeping 1/2/4/8 threads |
| `example_vendor`    | the out-of-tree plugin contract: ABI handshake, probe, kernel register                                                   |

### Build options

| Option               | Default            | Meaning                                                                                                 |
| -------------------- | ------------------ | ------------------------------------------------------------------------------------------------------- |
| `RVCL_ENABLE_RVV`    | `ON`               | RVV backend; the intrinsics and JIT paths compile on RISC-V hosts, other hosts keep the portable C path |
| `RVCL_ENABLE_IME`    | `OFF`              | IME backend                                                                                             |
| `RVCL_ENABLE_VME`    | `OFF`              | VME backend                                                                                             |
| `RVCL_ENABLE_AME`    | `OFF`              | AME backend                                                                                             |
| `RVCL_ENABLE_OMP`    | `OFF`              | Built-in OpenMP scheduler (standalone product shape)                                                    |
| `RVCL_RVV_TIERS`     | `128,256,512,1024` | Enabled VLEN tiers, one registry descriptor each                                                        |
| `RVCL_TEST_SET`      | `smoke`            | Corpus tier registered as ctest targets: `smoke` / `ci` / `all`                                         |
| `RVCL_SANITIZER`     | empty              | e.g. `address,undefined`, or `thread` alone                                                             |
| `RVCL_ENABLE_WERROR` | `OFF`              | Treat compiler warnings as errors                                                                       |

**Two build tiers for the OpenMP scheduler.** The default tier links no libgomp
— the OpenMP translation unit, its macro and its link are all absent — and the
built-in scheduler resolves to the serial backend. The standalone tier compiles
the `omp` scheduler and makes it the lazy default, so a zero-injection call gets
multi-core behaviour:

```bash
cmake -B build -S . -DRVCL_ENABLE_OMP=ON
OMP_NUM_THREADS=4 ./build/example_matmul
```

## Usage Examples

**One-shot call** (full example in `examples/matmul.cpp`):

```c
#include "rvcl/rvcl.h"

rvcl_matmul_desc_t d = {
    .m = M, .n = N, .k = K,
    .dtype_a = RVCL_DTYPE_FP32, .dtype_b = RVCL_DTYPE_FP32,
    .dtype_acc = RVCL_DTYPE_AUTO,
    .dtype_c = RVCL_DTYPE_FP32,
    .dtype_bias = RVCL_DTYPE_FP32,
    .quant = {},                         /* non-quantized */
};
float c[M * N];
rvcl_status_t st = rvcl_matmul(&d, a, b, bias, c);
```

For block-scale inputs (MX / NV formats) the scaled entry point takes the scale
tensors: `rvcl_matmul_scaled(&d, a, sa, b, sb, bias, c)`, with `sa[m][K/blk_a]`
and `sb[K/blk_b][n]` row-major (the oneDNN MX convention).

**Dispatch once, run many times** (hot path / framework-integration usage):

```c
rvcl_dispatch_result_t *r = NULL;
rvcl_dispatch_matmul(&d, &r);            /* select + configure + report ws */

int num_ws = 0;
const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
/* the host allocates ws[0..num_ws), imports each as a tensor,
 * and feeds it back through RVCL_TENSOR_AUX_0 + i */

for (each batch) {
    rvcl_tensor_pack_t pack;             /* zero-copy import of this batch */
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    /* … SRC_B / BIAS / DST / AUX(workspace) … */
    rvcl_work_range_t full = rvcl_work_range_full(M, N, K);
    rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
}
rvcl_dispatch_release(r);
```

**Pack B once, run many times**: `rvcl_kernel_pack_b()` fills the first
workspace (lifetime `RVCL_WS_LIFETIME_KERNEL`) with packed panels; endorsing it
with `pack.src_b_packed = 1` tells `run()` to consume those panels read-only
instead of packing per run. The flag is a host guarantee that B does not change
while the buffer is held, and the mechanism is voluntary — a host that leaves it
at zero gets per-run packing as before. See `examples/dispatch.cpp`.

**Inject a host scheduler**: process-wide and one-shot — must happen before the
first RVCL compute call:

```c
extern rvcl_ischeduler_t my_pool_scheduler;   /* implements parallel_for */
rvcl_scheduler_set(&my_pool_scheduler);       /* a second call returns ALREADY_INITIALIZED */
```

Under framework integration, `parallel()` on the framework side is the only
parallel source and RVCL supplies single-threaded kernels, so double parallelism
does not structurally exist. When RVCL is used standalone,
`examples/matmul_multicore.cpp` shows both shapes — `pool` injects the example's
own worker pool, `builtin` runs with zero injection on the built-in OpenMP
scheduler.

**Register an out-of-tree kernel** — see `examples/vendor_kernel.cpp` and
[Extending rvcl](#extending-rvcl).

## Extending rvcl

RVCL is built around three in-tree extension slots — **dtype**, **operator** and
**backend** — and one out-of-tree slot, **vendor plugins**. Each has a fixed
landing point, so growing the library means filling a table row or adding a
corpus directory rather than editing the pipeline. The six-stage dispatch
pipeline (`dispatch.cpp`) is operator-generic: it only reads the op-table row
for the request validator and cache key, then filters registered kernels by
dtype tuple, capability, VLEN tier, shape and validation.

### Add a dtype

A new datatype touches four places: a trait-table row, a micro-kernel file, a
registration row and a corpus line. The pipeline body and the two test mains
need no changes.

### Add an operator

An operator is a dispatch-table row plus a kernel contract. Its framework
footprint is three actions: one descriptor member in the `rvcl_op_desc_t` union
in `include/rvcl/kernel.h`, one row in the dispatch op table
(`src/runtime/dispatch.cpp`, a request validator plus a cache-key hash and
equality), and one row in the test-side op table `tests/op_slots.cpp` (corpus
name, dtype set, golden routing, run family) with a new `tests/inputs/<op>/`
corpus directory and one `rvcl_add_corpus()` line. The differential and bench
mains then drive the new operator unchanged.

**GEMV** is the working second operator: `src/backends/rvv/rvv_gemv_f32.cpp`
implements the micro-kernel and `src/backends/reference/reference_gemv.cpp` the
golden reduction, both registered through the same table row.

**CONV2D** is the first operator whose shape does not fit the GEMM triple:
`src/operators/conv2d.cpp` is the semantic layer, `src/backends/rvv/rvv_conv2d.cpp`
the kernel. It reuses the per-dtype GEMM micro-kernels — the patch matrix is
materialised per row band (im2col) and the result is transposed back to NCHW — so
a convolution and a matmul of the same dtype always run the same inner kernel.
Because the convolution geometry (batch, channels, spatial sizes, kernel, stride,
dilation, padding) cannot travel in the corpus shape column, its corpus cases
carry a params column; see `tests/inputs/README.md`. Its dtypes are f32, f16 and
bf16, always accumulating into and writing out f32.

**INNER_PRODUCT** is the GEMM projection of the inner product:
`dst[mb,oc] = bias[oc] + sum_p src[mb,p] * w[oc,p]`, with the reduction
`K = ic*kd*kh*kw` folded by the caller. `src/operators/inner_product.cpp` is the
semantic layer and `src/backends/rvv/rvv_inner_product.cpp` the kernel. It reuses
the per-dtype GEMM micro-kernels: `pack_b` transposes the `[oc][k]` weights into
the packed panel layout once, and the GEMM writes `[mb][oc]` row-major straight
into the destination, so there is no writeback transpose. Its shape triple is the
GEMM view directly (`M = mb`, `N = oc`, `K` = folded reduction), so its corpus
cases carry no params column. Its dtypes are f32, f16 and bf16, always
accumulating into and writing out f32.

### Add a backend

A new backend takes four fixed steps, and no framework code changes:

1. create `src/backends/<name>/` and implement `<name>_register()`;
2. add a `RVCL_ENABLE_<NAME>` CMake option and the sources behind it;
3. add one registration line in `src/runtime/init.cpp` `register_all()`;
4. wire differential coverage against the reference backend where dtypes
   overlap.

### Vendor kernels: in-tree backend or out-of-tree plugin

Two paths exist, and the choice is a packaging question rather than a semantic
one. If the library compiles with RVCL's source and its license permits it, use
the **in-tree backend** path: same shape as a native backend plus one
`RVCL_ENABLE_<NAME>` switch. If it is a closed-source binary or must ship
independently, use the **out-of-tree plugin** path, where the RVCL tree stays
untouched and the plugin programs only against installed headers. The same
operator has been verified along both paths; the only remaining difference is
the registration channel (build-time `register_all()` vs. a runtime
constructor).

The public API that makes out-of-tree kernels equivalent to in-tree ones:

| Function                                  | Purpose                                                             |
| ----------------------------------------- | ------------------------------------------------------------------- |
| `rvcl_capability_register_vendor_probe()` | write channel into the capability snapshot (`rvcl_vendor_probe_fn`) |
| `rvcl_capability_try_csr()`               | read an optional CSR under SIGILL protection; returns 0 when absent |
| `rvcl_cost_matmul_hint()`                 | public cost-scoring channel with a vendor-owned coefficient         |
| `rvcl_kernel_abi_version()`               | advisory descriptor-layout handshake (`RVCL_KERNEL_ABI_VERSION`)    |

The plugin recipe, as demonstrated by `examples/vendor_kernel.cpp`:

1. **ABI handshake** — compare `rvcl_kernel_abi_version()` against the
   `RVCL_KERNEL_ABI_VERSION` in the headers the plugin was built with, and
   refuse to register (reporting to stderr) on mismatch. The check is advisory;
   the library cannot inspect a plugin's struct layout, so every plugin is
   expected to perform it.
2. **Register a capability probe** — it runs once, single-threaded, while the
   snapshot is built, and appends feature ids at or above
   `RVCL_FEATURE_VENDOR_FIRST`. Registration must happen before the first
   capability query or dispatch (a constructor satisfies this); a later
   registration returns `RVCL_ERROR_ALREADY_INITIALIZED`.
3. **Register the kernel** with `rvcl_register_kernel()`, declaring the vendor
   feature in `required_features` so dispatch's stage 3 hides it on incapable
   hosts at zero cost. Do not bind `rvcl_kernel_cost_zero()`, or candidates
   cannot be ordered — forward to `rvcl_cost_matmul_hint()` with your own
   coefficient when you have no independent cost model.
4. **Implement packing and execution** — declare workspace in `configure()`,
   pack into the `RVCL_WS_LIFETIME_KERNEL` slot, slice by work range in `run()`,
   and skip packing when `pack->src_b_packed` is set.
5. **Load and verify** — either `LD_PRELOAD` (the plugin's `DT_NEEDED` pulls in
   librvcl and the constructor registers before `main`) or link the plugin into
   the application. RVCL never `dlopen`s actively. Verify with
   `rvcl_registry_find_factory()` and by dispatch attribution: with the feature
   present the plugin kernel name should be selected; without it, dispatch falls
   back to a built-in kernel and still succeeds.

### Kernel invariants

Five constraints hold for every kernel, enforced by review rather than by the
compiler, because they decide whether a kernel works under arbitrary host
orchestration:

1. the `run()` path performs **no dynamic allocation** — everything transient
   comes from the workspace declared in `configure()`;
2. the `run()` path creates **no threads** — parallelism is a work-range slice;
3. it reads and writes **only memory supplied through the tensor pack**, and
   holds no cross-call external state;
4. it executes **any legal work range** the scheduler may slice, including one
   starting mid-panel;
5. its behaviour matches the `rvcl_kernel_desc_t` metadata it declares, because
   dispatch trusts that metadata unconditionally.

## Project Layout

```text
rvcl/
├── include/rvcl/        # public C ABI: types/init/capability/memory/scheduler/
│                        # kernel/registry/dispatch/matmul/conv2d/inner_product + rvcl.h umbrella
├── src/
│   ├── common/          # type, dtype-trait and memory contract implementations
│   ├── runtime/         # capability / registry / dispatch / cost model / init
│   │   └── scheduler/   # serial + OpenMP scheduler backends
│   ├── operators/       # semantic-layer operators (matmul, conv2d, inner_product)
│   └── backends/
│       ├── reference/   # golden implementations (matmul, gemv)
│       └── rvv/         # C / intrinsics kernels, panel packing, path table
│           └── jit/     # xbyak_riscv JIT kernels (f32/f16/fp8/mxfp8/s8u8)
├── examples/            # matmul, dispatch, scheduler, matmul_multicore, vendor_kernel
├── tests/               # unit + smoke + differential + concurrent + bench
│   └── inputs/          # declarative shape corpora (matmul/, gemv/, conv2d/, inner_product/)
├── third_party/         # vendored xbyak_riscv (tag 1.31, BSD-3)
└── CMakeLists.txt
```

## Testing and CI

Three workflows gate the repository:

- **`ci-lint`** — clang-format-18, the Apache-2.0 license header check,
  Conventional Commits (including the PR title, since `main` squash-merges it),
  and clang-tidy-18 with zero-warning tolerance.
- **`ci-x86`** — the default tier, plus an OpenMP tier and two sanitizer tiers
  (ASan+UBSan, and TSan). TSan drives `tests/concurrent_test.cpp` through
  concurrent `kernel_run` (exclusive slices with pre-packing), concurrent
  first-call dispatch, concurrent semantic calls and concurrent first-call
  capability — the evidence chain behind the concurrency claims.
- **`ci-riscv`** — cross-compile plus QEMU execution across three
  configurations: baseline `rv64gc` at vlen=128, vector `rv64gcv_zvfh` at
  vlen=128 and vlen=256. The capability assertion target runs first in every
  tier.

Test tiers are chosen at configure time by `RVCL_TEST_SET` (`smoke` < `ci` <
`all`, default `smoke`) and selected at run time by label
(`ctest -L corpus|smoke|ci`). Correctness comes from the differential harness
(`rvcl_differential`), which runs every capability-legal kernel for a shape
against the reference implementation, driven by one-case-per-line corpora under
`tests/inputs/`; adding a case is adding a line. `rvcl_bench` reports timing and
attribution in a human table or JSON Lines, and its `kernel` and `scheduler`
keys name the selected kernel and the active scheduler (with thread count), so
multi-core timing differences can be attributed to a concrete implementation and
parallelism level.

## Roadmap

| Version | Milestone                                 | Planned    | Status      |
| ------- | ----------------------------------------- | ---------- | ----------- |
| v0.1    | Framework design + RVV matmul             | 2026-10-31 | Done        |
| v0.2    | Vendor plugin demo + FP8 / FP4 dtypes     | 2026-11-15 | Done        |
| v0.3    | Full operator coverage + IME backend demo | 2026-12-15 | In progress |
| v0.4    | Vendor integration                        | 2027-01-31 | Not started |
| v0.5    | VME / AME backend demo                    | 2027-03-31 | Not started |
| v1.0    | Stable perf and release                   | 2027-04-30 | Not started |

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide. The essentials:

- **DCO, not a CLA.** Every commit carries a `Signed-off-by:` line
  (`git commit -s`).
- **Conventional Commits.** `main` squash-merges, so the PR title becomes the
  mainline subject and CI gates it: type from
  `feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`, description an
  imperative phrase under 72 characters.
- **Three local checks before submitting**: `ctest` green,
  `clang-format-18 --dry-run --Werror` clean, `clang-tidy-18` zero warnings.
  Formatter and analyzer versions must be 18.x. Suppress a finding with `NOLINT`
  on the same or previous line plus a reason; unexplained blanket suppressions
  are rejected in review.
- **Kernel PRs** are reviewed against the five invariants above — the contract
  is enforced by review, not by the compiler.
- **Public API stability** is tiered: `types.h`, `init.h`, `capability.h`,
  `memory.h`, `scheduler.h`, `matmul.h`, `conv2d.h`, `inner_product.h` are the
  stable surface and accept additive-only changes before v1.0; `kernel.h`,
  `registry.h` and `dispatch.h` are the project-stable kernel ABI, where
  breaking changes need an RFC and maintainer consensus. Any descriptor layout
  change — even an appended field — must bump `RVCL_KERNEL_ABI_VERSION`, so old
  plugins stop registering instead of passing a short struct.
- **Review bar**: at least one area maintainer who did not author the change;
  public API changes additionally need the project lead. Areas and current
  reviewers are listed in [MAINTAINERS.md](MAINTAINERS.md); external maintainers
  are a project goal, not an exception.
- **Good first tasks**: a new dtype tuple for an existing backend, a backend
  directory skeleton, or a test that pins down contract behaviour (error codes,
  work ranges).

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions follow
the DCO (`git commit -s`); see [CONTRIBUTING.md](CONTRIBUTING.md). Vendored
third-party code and its licenses are recorded in
[THIRD-PARTY-PROGRAMS](THIRD-PARTY-PROGRAMS) — currently only `xbyak_riscv` (tag
1.31, BSD-3), kept as an unmodified upstream copy. Citation metadata lives in
[CITATION.cff](CITATION.cff).
