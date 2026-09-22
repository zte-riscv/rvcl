# RVCL — RISC-V Compute Library

**A unified compute kernel abstraction and dispatch runtime for RISC-V vector and matrix architectures — covering RVV, IME, VME, AME and vendor extensions, with capability detection, kernel registry, and framework integration (oneDNN).**

RVCL is a unified kernel abstraction and dispatch runtime for RISC-V vector and matrix architectures: **Capability** (what the hardware can do) → **Registry** (which kernel implementations exist) → **Dispatch** (which kernel should run) → **Kernel** (do the actual computation), with an injectable **Scheduler** and a zero-copy **Memory** contract crossing all layers. The goal: one operator API that automatically selects the best kernel for the current hardware across RVV / IME / VME / AME / vendor extensions, and integrates upward with oneDNN (and through it, frameworks such as PyTorch).

> **Status:** Phase 0 complete (v0.1.0-dev). The FP32 reference kernel and the full dispatch framework are in place; the RVV backend is under development.
> Design documents (Chinese): [PLAN-0001](../docs/PLAN-0001_RVCL项目开发计划.md) · [ARCH-0001](../docs/ARCH-0001_RVCL系统架构文档.md) · RFC-0001/0002
> Concise English architecture overview: [ARCH-0001-EN](../docs/ARCH-0001_EN_RVCL_Architecture.md)

---

## Table of Contents

- [Why RVCL](#why-rvcl)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Project Layout](#project-layout)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

## Why RVCL

RISC-V compute extensions are far more heterogeneous than any other architecture: the vector extension (RVV) is ratified, while the matrix extensions — IME (draft), VME, AME — are still evolving, and vendors ship their own accelerators (TPE, etc.). Writing operators directly against every extension means:

- frameworks (oneDNN/PyTorch) maintain a separate backend path per hardware target;
- kernel developers re-solve scheduling, memory, and type-conversion problems that have nothing to do with the ISA;
- every new piece of silicon forces the ecosystem above it to rewrite rather than recompile.

RVCL's answer is the **layered contract that ACL (Arm Compute Library) has already validated in the Arm ecosystem**, ported to RISC-V and extended:

| Layer | Responsibility | In one sentence |
| ----- | -------------- | --------------- |
| Capability | Detect hardware features (ISA / datatype extensions / vendor interfaces) | What can this CPU execute? |
| Registry | Register and index kernel metadata (operation × (A,B,acc,C) dtype tuple) | Which kernel implementations exist? |
| Dispatch | Multi-stage filtering + scoring, pick the best legal kernel | Which kernel should run? |
| Kernel | `validate / configure / run` three-phase lifecycle | Do the actual computation. |

Two cross-cutting contracts stay under host control, invisible to kernels:

- **Scheduler** — parallelism belongs to the host (mirroring ACL's `IScheduler`, injected set-once); kernels express work as work-range slices and never create threads;
- **Memory** — zero-copy import (mirroring `import_memory` / `ITensorPack` / `MemoryRequirements`); kernels report workspace demands, the host allocates, and kernels never malloc large buffers.

## Architecture

```text
┌─────────────────────────────────────────────┐
│  Operator API    rvcl_matmul() …            │  semantic layer: one-shot call
├─────────────────────────────────────────────┤
│  Dispatch        7-stage pipeline (1–4 now) │  pick a kernel
├─────────────────────────────────────────────┤
│  Registry        operation × dtype index    │  find kernels
├─────────────────────────────────────────────┤
│  Kernel          validate/configure/run     │  do the math
└────────────────────────────┬────────────────┘
                             ▼
                        RISC-V ISA
   (Capability feeds dispatch from the side; Scheduler/Memory cross all layers)
```

The datatype model separates **roles** (storage / compute / accumulation); the unit of dispatch matching is the `(A, B, acc, C)` tuple. Habit rules such as `fp32→fp32`, `bf16→fp32`, and `s8→s32` are resolved automatically via `RVCL_DTYPE_AUTO`. INT8 quantization (requant) and INT4 weight-only land in v0.4 / v0.5 respectively. See PLAN-0001 §4.5 for details.

## Quick Start

Requirements: CMake ≥ 3.16, a C11 compiler, pthread. No RISC-V toolchain needed for now — the reference backend builds and runs on x86_64 hosts (it is the golden implementation for differential testing).

```bash
git clone <repo-url> && cd rvcl
cmake -B build -S .
cmake --build build

ctest --test-dir build        # or: cd build && ctest (ctest < 3.21)
./build/example_matmul
```

Actual output (x86_64 host — empty capability set, but the reference kernel has no feature requirements and runs anyway):

```text
RVCL 0.1.0-dev — rvcl-capability: 0 feature(s) detected
dispatch selected: rvcl_matmul_f32_reference (ws=0)
OK — reference matmul verified
```

Backend switches (all OFF by default, enabled per the release roadmap):

```bash
cmake -B build -S . -DRVCL_ENABLE_RVV=ON   # likewise IME / VME / AME
```

## Usage Examples

**One-shot call** (full example in `examples/matmul.c`):

```c
#include "rvcl/rvcl.h"

rvcl_matmul_desc_t d = {
    .m = M, .n = N, .k = K,
    .dtype_a = RVCL_DTYPE_FP32, .dtype_b = RVCL_DTYPE_FP32,
    .dtype_acc = RVCL_DTYPE_AUTO,        /* habit rule: fp32 → fp32 */
    .dtype_c = RVCL_DTYPE_FP32,
};
float c[M * N];
rvcl_status_t st = rvcl_matmul(&d, a, b, /*bias=*/NULL, c);
```

**Dispatch once, run many times** (hot path / framework-integration usage):

```c
rvcl_dispatch_result_t *r = NULL;
rvcl_dispatch_matmul(&d, &r);            /* select + configure + report ws */

for (each batch) {
    rvcl_tensor_pack_t pack;             /* zero-copy import of this batch */
    rvcl_tensor_pack_init(&pack);
    rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
    /* … SRC_B / DST / AUX(workspace) … */
    rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &range);
}
rvcl_dispatch_release(r);
```

**Inject a host scheduler** (mirrors ACL `IScheduler`; process-wide and one-shot — must happen before the first RVCL compute call):

```c
extern rvcl_ischeduler_t my_pool_scheduler;   /* implements parallel_for */
rvcl_scheduler_set(&my_pool_scheduler);       /* a second call returns ALREADY_INITIALIZED */
```

## Project Layout

```text
rvcl/
├── include/rvcl/        # public API: types/init/capability/memory/scheduler/
│                        # kernel/registry/dispatch/matmul + the rvcl.h umbrella
├── src/
│   ├── common/          # type & memory contract implementations
│   ├── runtime/         # capability / registry / dispatch / init / scheduler
│   ├── operators/       # semantic-layer operators (matmul)
│   └── backends/        # reference/ (golden FP32); rvv/ ime/ vme/ ame/ per roadmap
├── examples/matmul.c
├── tests/smoke_test.c   # 24 contract checks
└── CMakeLists.txt
```

Adding a backend = a new `src/backends/<name>/` directory + one registration line in `init.c` + one CMake switch. The dispatch/registry/scheduler/memory framework needs zero changes.

## Roadmap

| Version | Milestone |
| ------- | --------- |
| v0.1 | RVV backend (FP32/FP16), differential testing against the reference |
| v0.2 | IME backend — the first proof of the abstraction's value (same API on a matrix extension) |
| v0.3 | VME / AME + oneDNN vertical slice (`DNNL_RISCV_USE_RVCL`) |
| v0.4 | INT8 quantized paths + dispatch cost model (stages 5–7) |
| v0.5 | Full oneDNN integration + INT4 weight-only + PyTorch support |
| v1.0 | Stable ABI commitment |

Full breakdown in PLAN-0001 §5–§6; the first full-stack use case (PyTorch + oneDNN + RVCL) in PLAN-0001 §6.4.1.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full guide (DCO sign-off,
workflow, review bar). The essentials:

- Kernel implementations must honor the kernel-contract invariants (review checklist in ARCH-0001 §6): no malloc and no thread creation on the run() path, all memory from pack slots, correct handling of arbitrary work ranges;
- Public API changes go through the RFC process (RFC-0001/0002 came first); breaking changes to the project-stable layer (kernel.h/registry.h/dispatch.h) require review;
- Before submitting: `cmake --build build && ctest --test-dir build` must be green.

## License

Apache-2.0 — see [LICENSE](LICENSE) and [NOTICE](NOTICE). Contributions
follow the DCO (`git commit -s`); see [CONTRIBUTING.md](CONTRIBUTING.md).
