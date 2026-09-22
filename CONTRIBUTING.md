# Contributing to RVCL

Thank you for considering contributing to RVCL — RISC-V Compute Library!
This document covers the practicalities: licensing of contributions, the
development workflow, and the review bar for kernels and public APIs.

For the architecture behind the rules below, read
[ARCH-0001](../docs/ARCH-0001_RVCL系统架构文档.md) (Chinese; §6 has the
kernel review checklist). For design rationale, see PLAN-0001 and the
RFC series in the same directory.

## Code of Conduct

By participating in this project you agree to uphold the
[Code of Conduct](CODE_OF_CONDUCT.md).

## License and DCO

RVCL is Apache-2.0. Contributions are accepted under the same license.

We follow the **Developer Certificate of Origin** (DCO, `developercertificate.org`):
every commit must carry a `Signed-off-by:` line (`git commit -s`) attesting
you have the right to submit the work under its license.

## Getting started

```bash
cmake -B build -S .
cmake --build build
ctest --test-dir build        # or: cd build && ctest (ctest < 3.21)
```

No RISC-V toolchain is required: the reference backend builds and runs on
x86_64 hosts and is the golden implementation for differential testing.

Good first tasks:

- a new dtype tuple for an existing backend (e.g. fp16 reference);
- `src/backends/<name>/` backend skeletons;
- tests that pin down contract behavior (error codes, work ranges).

## Workflow

1. Fork / branch, develop against the smallest reviewable unit.
2. Keep the build green: `-Wall -Wextra` is on; fix new warnings you introduce.
3. `ctest` must pass on your host before opening a PR.
4. Open a PR describing **what** and **why**; link any related RFC issue.

## Kernel contributions — hard invariants

Kernels run under contracts the framework enforces by review, not by
compiler. A kernel PR will not be merged unless the `run()` path:

1. performs **no dynamic allocation** (malloc/calloc/realloc/posix_memalign);
   everything transient belongs to workspace reported by `configure()`;
2. creates **no threads** — parallelism is expressed as work-range slices
   for the host scheduler;
3. reads/writes **only memory supplied via the tensor pack**;
4. correctly executes **any legal work range** the scheduler may slice;
5. matches its declared `rvcl_kernel_desc_t` metadata (dtype tuple,
   feature requirements, shape bounds, alignments) — dispatch trusts it.

Backend layout: one directory per backend under `src/backends/<name>/`,
plus one `rvcl_<name>_register()` call in `src/runtime/init.c` and one
CMake `RVCL_ENABLE_<NAME>` switch. Framework code should need zero changes.

## Public API changes

- `types.h / init.h / capability.h / memory.h / scheduler.h / matmul.h`
  are the stable surface: changes are **additive-only** before v1.0.
- `kernel.h / registry.h / dispatch.h` are project-stable (the kernel ABI):
  breaking changes require an RFC and maintainer consensus.
- New operations / datatypes / feature ids follow the extension recipes in
  ARCH-0001 §10.3 — enumerate which files change and which must not.

## Adding a backend (checklist)

- [ ] `src/backends/<name>/` with kernels + `<name>_register()`
- [ ] CMake option `RVCL_ENABLE_<NAME>` (default OFF)
- [ ] one registration line in `src/runtime/init.c` `register_all()`
- [ ] differential test vs the reference backend where dtypes overlap
- [ ] kernels honor the five invariants above (self-review against
      ARCH-0001 §6)

## Reporting bugs / security issues

Bugs: open a GitHub issue with host info (`rvcl_get_capability()`
description string helps), build flags, and a reproducer.

Security: do **not** open public issues for security problems — see
[SECURITY.md](SECURITY.md).
