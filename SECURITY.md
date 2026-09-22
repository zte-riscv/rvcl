# RVCL Security Policy

RVCL is a compute library: it processes in-memory data supplied by the
host application and performs no I/O, networking, or privileged
operations. Its attack surface is correspondingly narrow, but memory
safety in layout/stride handling and input validation still matters.

## Supported versions

Security fixes are applied to the latest released version line only
(pre-1.0: the current `main` branch).

| Version | Supported |
| ------- | --------- |
| main (development) | yes |
| tagged releases, latest line | yes |
| older tagged releases | until superseded |

## Reporting a vulnerability

**Do not open a public GitHub issue for security problems.**

Please report privately via GitHub's **"Report a vulnerability"** action on
the repository's *Security* tab, or by email to
`<TODO: security@example.org>` (PGP key published at the same address).

Include where possible:

- a description of the issue and its impact;
- the RVCL version / commit and build flags;
- host info (the `rvcl_get_capability()` description string helps);
- a minimal reproducer (input descriptors, strides, shapes).

We aim to acknowledge reports within **2 business days** and will keep you
informed of remediation progress. Coordinated disclosure (CVE assignment
where applicable, advisory published together with the fix) follows the
norms of the hosting organization once the project is under RISE/RVI
governance.

## Scope

In scope:

- memory safety bugs in public API entry points reachable with malformed
  but non-NULL arguments (strides, dims, dtype combinations, tensor pack
  misuse);
- integer overflow in size/stride computation (`rvcl_tensor_bytes`,
  workspace sizing);
- violation of the kernel contract invariants leading to out-of-bounds
  access inside `run()`.

Out of scope:

- crashes requiring deliberately hostile host code that already owns the
  process (RVCL makes no sandboxing claims);
- hardware faults, side channels, or emulator (QEMU) bugs;
- volumetric denial of service.
