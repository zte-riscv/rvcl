# Test Corpora

Declarative shape corpora: **one case per line**, consumed by
`rvcl_differential --batch=<file>` (correctness, all capability-legal
kernels) and `rvcl_bench --batch=<file> --mode=C|P|S` (attribution and
timing). Adding a test = adding a line; see "Registering" below.

## Layout

```
tests/inputs/
├── matmul/                  # operator directory — the directory name IS the op
│   ├── test_matmul_smoke    # PR gate set (7 cases)
│   ├── test_matmul_ci       # CI set (53 cases: scan + tails + LCG)
│   └── tools/gen_shapes_lcg.sh
└── <op>/ …                  # new operator = new directory (v0.5+, follows the library)
```

A new operator directory needs: corpus files in the same format, a
`rvcl_add_corpus(<op> …)` line in `CMakeLists.txt`, and a subcommand slot
in the bench/differential entry points. The format itself is operator-
agnostic.

## File format

```ebnf
corpus_file := { blank | comment | header_opt } case_line { blank | comment | case_line } ;
blank       := [ \t ]* EOL ;
comment     := '#' ANY* EOL ;            (* full-line; inline after a case is fine *)
header_opt  := ( '--mode=' ( 'C' | 'P' | 'S' ) | '--dt=' ( 'f32' | 'f16' ) ) EOL ;
              (* only before the first case; any '--' line after it is an error *)
case_line   := shape [ occurrence ] [ sep label ] EOL ;
shape       := triple | triple ':' triple ;  (* MxNxK; or MxK:KxN, both K must match *)
triple      := int 'x' int 'x' int ;
int         := decimal, 1 <= value <= 4096 ;
occurrence  := '*' int ;                 (* benchdnn topology count — tolerated, ignored *)
sep         := [ \t ]* [ '_' ] ;         (* glued '_n"..."' form is accepted *)
label       := 'n' '"' NAME '"' ;        (* NAME in [A-Za-z0-9_.*-]+, unique per file *)
```

Examples — all equivalent shapes, the first two also benchdnn-compatible:

```text
33x19x47 n"ragged_panel_edge"
128x256x1536:128x1536x512_n"deepseek_style_example"   # MxK:KxN, glue form, *n allowed
```

The double-triple form and the `_n"…"` label syntax keep a deliberate
subset compatibility with the oneDNN benchdnn input conventions so
corpora can be exchanged both ways (v0.5 will excerpt `shapes_perf_basic`
lines unmodified).

## Semantics

| Rule | Meaning |
| --- | --- |
| header options | whole-file defaults; bench CLI `--mode` overrides the file header; differential ignores `--mode` (it is inherently correctness) |
| `--dt` default | no header = run f32 **and** f16; explicit = run only that dtype and **assert** it is served (see exit codes) |
| `--dt` CLI | differential `--dt=f32\|f16` restricts the whole run — fixed contract sections and corpus cases alike — to one dtype and overrides the file header; the forced-path ctest targets use it |
| label | unique per file (duplicate = parse error); a case without a label gets the normalized `MxNxK` string as its label |
| double triple | the two K values must be equal, else parse error; normalized to MxNxK |
| bounds | any dimension outside 1..4096 = parse error (foot-guard against runaway shapes on QEMU) |
| empty file | a corpus file with no case line = error (no silently-green empty targets) |

## Exit codes (both consumers)

| Code | When | Shape of the output |
| --- | --- | --- |
| 0 | all cases pass | `CASES: <n> ok, 0 fail, <n> skip` |
| 1 | a case assertion failed | `FAIL <label> (<shape>, <dtype>): first mismatch at (i,j) exp=… got=… (n mismatch)` |
| 2 | corpus / usage error | `error: <file>:<line>: …` (unknown header option, `--` after first case, unparsable line, K mismatch, duplicate label, out-of-bounds dim, empty file) |

A dtype with no capability-legal kernel on the host is a **skip** (exit 0,
counted in the summary) when `--dt` is defaulted — an environment fact,
not a silent degradation; with an explicit `--dt` header it is a failure.

## Naming and tiers

`test_<op>_<set>` for correctness sets, `shapes_<label>` for pure shape
sets (perf excerpts, v0.5). Suffixes: `_smoke` (PR gate), `_ci` (CI
full set); `_weekly`/`_large` arrive with the v0.5 weekly job.

## Registering

One line in `CMakeLists.txt` per corpus file:

```cmake
rvcl_add_corpus(matmul ci test_matmul_ci)   # -> ctest target test_matmul_modeC_ci
```

On RISC-V hosts the same line also registers forced f32 micro-kernel-path
variants of each target (`_jit`, `_c`, plus `_intrinsics` under the vector
`-march` override), driven through the `RVCL_RVV_KERNEL` environment pin
and the differential `--dt=f32` restriction.

`RVCL_TEST_SET` (`smoke` < `ci` < `all`, default `smoke`) selects at
configure time which tiers become ctest targets; CI raises it explicitly.
Run-time selection: `ctest -L corpus` / `ctest -L smoke` / `ctest -L ci`.

## Rebuilding the LCG set

```sh
tests/inputs/matmul/tools/gen_shapes_lcg.sh 42 16 257   # seed, count, max dim
```

Deterministic (Lehmer RNG, exact in awk doubles) — the same command
reproduces the lines in `test_matmul_ci` byte-for-byte.
