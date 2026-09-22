# RVCL Maintainers

This file lists the maintainers of RVCL and their areas of responsibility.
It is the authoritative answer to "who reviews and merges what".

## Current maintainers

| Name | Affiliation | Areas | GitHub |
| ---- | ----------- | ----- | ------ |
| <TODO: name> | ZTE | project lead; dispatch / registry / capability runtime; API & RFC process | <TODO> |
| <TODO: name> | ZTE | kernel contracts; reference & RVV backends; differential testing | <TODO> |

> Initial maintainer list at project opening. Additional maintainers are
> expected as the community grows — see "Becoming a maintainer" below.
> External (non-founding-organization) maintainers are a project goal, not
> an exception; see PLAN-0001 and COMM-0001 §8.

## Ownership by area

| Area | Path | Default reviewer |
| ---- | ---- | ---------------- |
| Public API surface | `include/rvcl/` | project lead |
| Dispatch / registry / capability | `src/runtime/` | project lead |
| Memory & scheduler contracts | `src/common/`, `src/runtime/scheduler/` | project lead |
| Reference backend (golden) | `src/backends/reference/` | backend lead |
| RVV backend | `src/backends/rvv/` | backend lead |
| IME / VME / AME backends | `src/backends/{ime,vme,ame}/` | area maintainers (TBD as they land) |
| Build system & CI | `CMakeLists.txt`, `.github/workflows/` | any maintainer |
| Tests & examples | `tests/`, `examples/` | any maintainer |

## Merge rules

- PRs require at least **one** approval from an area maintainer who did
  not author the change.
- Changes to the public API (`include/rvcl/*.h`) additionally require the
  project lead's approval; breaking changes to the project-stable layer
  (kernel.h / registry.h / dispatch.h) require an RFC and maintainer
  consensus (see [CONTRIBUTING.md](CONTRIBUTING.md)).
- Kernel PRs are reviewed against the invariant checklist in ARCH-0001 §6.

## Becoming a maintainer

Sustained, high-quality contributions to an area (typically across several
release cycles — code + reviews + design participation) qualify a
contributor for area maintainership. Any current maintainer may nominate;
the maintainers list decides by consensus. Maintainers who are inactive
for two release cycles may move to emeritus status.

Changes to this file go through the standard PR process with maintainer
consensus.
