# Contributing to RVCL

Thank you for considering contributing to RVCL — RISC-V Compute Library!
This document covers the practicalities: licensing of contributions, the
development workflow, and the review bar for kernels and public APIs.

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

## Languages

The build language is C++17 (gnu++17) across the implementation, tests and
examples. The boundary:

- `include/rvcl/*.h` are the public headers and stay plain C headers with a
  C ABI (`extern "C"` guards throughout): both C and C++ consumers link
  against the library unchanged.
- Implementation, tests and examples are C++17 (`.cpp` / internal `.hpp`).
  Existing designated initializers rely on GNU extensions (`-std=gnu++17`),
  so MSVC is not a supported host compiler.
- The upcoming JIT layer (vendored xbyak_riscv, pure C++) links into the
  same single-language build — that is why the tree is not bilingual.

Good first tasks:

- a new dtype tuple for an existing backend (e.g. fp16 reference);
- `src/backends/<name>/` backend skeletons;
- tests that pin down contract behavior (error codes, work ranges).

## Code formatting

The authoritative formatter is **clang-format 18** (`.clang-format`, derived
from oneDNN). CI enforces it on the C/C++ files touched by a PR diff, so a
merge requires `clang-format-18 --dry-run --Werror <files>` to pass.

Format changed files before committing:

```bash
clang-format-18 -i <your .c/.h/.cpp/.hpp files>
```

Version matters: use 18.x (e.g. `pip install clang-format==18.1.*` or the
system `clang-format-18`); an older formatter can produce different output.
The one-time reformat commit that introduced the style is listed in
`.git-blame-ignore-revs` — enable it once so `git blame` skips that commit:

```bash
git config blame.ignoreRevsFile .git-blame-ignore-revs
```

## Static analysis (clang-tidy)

CI runs **clang-tidy 18** (`clang-tidy-18`) on the C/C++ files touched by a
PR diff with zero-warning tolerance — a merge requires the same check to be
clean locally. The check set and exclusions live in `.clang-tidy`
(`bugprone-*`, `cert-*`, `readability-*`, minus the rules that are style
noise for this C-ABI library). Analysis needs the compile database:

```bash
cmake -B build -G Ninja \
    -DCMAKE_CXX_COMPILER=clang++-18 \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    -DRVCL_TEST_SET=ci
clang-tidy-18 -p build <your changed .c/.h/.cpp/.hpp files>
```

Use clang-tidy-18.x to match CI. If a check fires on code that is intentional
(e.g. a `printf`-style varargs helper, or an allocator that is required to
return a `const`-qualified pointer), silence it with a `// NOLINT(<check>)`
on the same line or `// NOLINTNEXTLINE(<check>)` above it, and leave a
comment explaining why the suppression is justified — blanket suppressions
without an explanation are rejected in review.

## Commit message conventions

Commits follow **Conventional Commits** and CI enforces it
(`.github/automation/commit-msg-check.py` on `ci-lint`). A subject is:

```
<type>(<scope>)?(!)?: <description>
```

- `<type>` is one of `feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert`;
- `<scope>` is optional and lower-case `[a-z0-9_-]` (e.g. `ci(riscv):`);
- `!` marks a breaking change;
- `<description>` is a short, imperative summary under **72 characters**,
  starting with a non-space character.

RVCL merges with **squash merge**: the PR title becomes the mainline commit
subject, so the title must satisfy the same rule and the history stays
linear — every mainline entry is one Conventional Commit. Intermediate
commits inside a PR are not separately gated, but keeping them in the same
style makes the squashed subject easy to write.

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
plus one `rvcl_<name>_register()` call in `src/runtime/init.cpp` and one
CMake `RVCL_ENABLE_<NAME>` switch. Framework code should need zero changes.

## Public API changes

- `types.h / init.h / capability.h / memory.h / scheduler.h / matmul.h`
  are the stable surface: changes are **additive-only** before v1.0.
- `kernel.h / registry.h / dispatch.h` are project-stable (the kernel ABI):
  breaking changes require an RFC and maintainer consensus.
- New operations / datatypes / feature ids enumerate which files change and
  which must not.

## Adding a backend (checklist)

- [ ] `src/backends/<name>/` with kernels + `<name>_register()`
- [ ] CMake option `RVCL_ENABLE_<NAME>` (default OFF)
- [ ] one registration line in `src/runtime/init.cpp` `register_all()`
- [ ] differential test vs the reference backend where dtypes overlap
- [ ] kernels honor the five invariants above

## Reporting bugs / security issues

Bugs: open a GitHub issue with host info (`rvcl_get_capability()`
description string helps), build flags, and a reproducer.

Security: do **not** open public issues for security problems — see
[SECURITY.md](SECURITY.md).
