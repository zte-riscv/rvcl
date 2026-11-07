/*******************************************************************************
 * Copyright 2026 ZTE Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * tests/differential_test.cpp — reference vs RVV backend differential testing
 *
 * PLAN §6.2 item 11 + §8.1 level 3: the same numeric cases run through
 * both backends by forcing kernel selection, then compare within
 * tolerance. fp32: bit-identical expected (same accumulation order);
 * fp16 storage path: tolerance 1e-3 relative (conversion rounding).
 *
 * Bit-exactness note for the RVV intrinsics path: vfmacc is a fused
 * (single-rounding) multiply-add, so the scalar golden must contract
 * `acc += a * b` to fmadd the same way — build with optimization
 * (Release / -O2+) on vector builds for the fp32 bit-exact checks to
 * hold. At -O0 the scalar golden double-rounds and only the tolerance
 * comparison would pass.
 *
 * Also covers the v0.1 acceptance criteria (PLAN §6.2):
 *   - scheduler injection does not change numeric results;
 *   - work-range partitioning covers without overlap or gap.
 *
 * Corpus mode: `rvcl_differential --batch=<file>` appends the shape
 * corpus (tests/inputs/README.md, format spec) after the fixed contract
 * sections below. Fixed sections test framework contracts, not shape
 * numerics, and stay out of the batch loop by design. With no arguments
 * the behavior is byte-for-byte the Phase 0 run.
 *
 * `--dt=f32|f16` restricts the whole run — fixed sections and corpus
 * cases alike — to one dtype. The forced-path ctest targets use it: a
 * jit-forced f32 run must fail loudly on f16's lack of a JIT path only
 * when f16 is actually in scope, not because the fixed sections ran it.
 *
 * This file holds one of the two independent corpus parsers (bench has
 * the other): both follow the spec's error table, and ctest runs the
 * same corpus through both — cross-validation is the feature.
 ******************************************************************************/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../src/common/dtype_traits.hpp"
#include "dtype_ops.hpp"
#include "golden.hpp"
#include "op_slots.hpp"
#include "rvcl/rvcl.h"

static int g_failures;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  ok: %s\n", msg); \
        } else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

/* ------------------------------------------------------ run-one-backend */

/* The matmul contract sections below drive the matmul kernels through the
 * test op table's named-run path (the same factory-bypass the differential
 * corpus loop uses), so the tensor rig lives in op_slots, not here. */
static rvcl_status_t run_matmul_kernel(const char *name,
        const rvcl_matmul_desc_t *d, const void *a, const void *b,
        const void *bias, void *c, int64_t r_m0, int64_t r_m1) {
    const rvcl_op_desc_t od = {.matmul = *d};
    return op_slot_for(RVCL_OP_MATMUL)
            ->run_named(name, &od, a, b, (const float *)bias, (float *)c, r_m0,
                    r_m1);
}

/* ------------------------------------------------- corpus: parser (own)
 *
 * Independent implementation #1 of the tests/inputs format (bench has
 * #2). Conformance contract: the error table in tests/inputs/README.md
 * — same messages, same exit code 2.
 *
 * Header option key table: keys are operator-independent today; a new
 * operator directory extends its own keys at this table without
 * touching the grammar.
 */

typedef struct {
    int64_t m, n, k;
    char *label; /* effective label: n"…" or the normalized shape */
} corpus_case_t;

typedef struct {
    corpus_case_t *cases;
    int count, cap;
    int dt_explicit; /* 0 = run both dtypes */
    rvcl_dtype_t dt; /* valid when dt_explicit */
} corpus_t;

static const char *const g_corpus_header_keys[] = {"--mode", "--dt", NULL};

static void corpus_fatal(const char *file, int line, const char *fmt, ...)
        __attribute__((noreturn, format(printf, 3, 4)));

/* printf-style fatal helper; the C varargs are the point of this function.
 * NOLINTNEXTLINE(cert-dcl50-cpp) */
static void corpus_fatal(const char *file, int line, const char *fmt, ...) {
    va_list ap;
    fprintf(stderr, "error: %s:%d: ", file, line);
    va_start(ap, fmt);
    /* va_list is started on the line above; clang-analyzer cannot track it
     * across the preceding fprintf and reports a false positive.
     * NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized) */
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(2);
}

/* parses one decimal int at *pp; advances *pp; fails fatally on bad
 * syntax or range */
static int64_t parse_dim(
        const char **pp, const char *file, int line, const char *orig) {
    const char *p = *pp;
    if (*p < '0' || *p > '9')
        corpus_fatal(file, line, "unparsable case '%s'", orig);
    int64_t v = 0;
    while (*p >= '0' && *p <= '9') {
        v = v * 10 + (*p - '0');
        if (v > 1000000000)
            corpus_fatal(file, line, "dimension overflow in '%s'", orig);
        p++;
    }
    if (v < 1 || v > 4096)
        corpus_fatal(file, line, "dimension %lld out of bounds 1..4096 in '%s'",
                (long long)v, orig);
    *pp = p;
    return v;
}

/* parses `int x int x int`, advances *pp past it */
static void parse_triple(const char **pp, int64_t t[3], const char *file,
        int line, const char *orig) {
    t[0] = parse_dim(pp, file, line, orig);
    if (**pp != 'x') corpus_fatal(file, line, "unparsable case '%s'", orig);
    (*pp)++;
    t[1] = parse_dim(pp, file, line, orig);
    if (**pp != 'x') corpus_fatal(file, line, "unparsable case '%s'", orig);
    (*pp)++;
    t[2] = parse_dim(pp, file, line, orig);
}

static void corpus_add(corpus_t *c, int64_t m, int64_t n, int64_t k,
        char *label, const char *file, int line) {
    /* label uniqueness (the point: unambiguous failure localization) */
    for (int i = 0; i < c->count; i++)
        if (strcmp(c->cases[i].label, label) == 0)
            corpus_fatal(file, line, "duplicate label n\"%s\"", label);
    if (c->count == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 64;
        corpus_case_t *narr = (corpus_case_t *)realloc(
                c->cases, (size_t)c->cap * sizeof(*c->cases));
        if (!narr) {
            fprintf(stderr, "error: out of memory\n");
            exit(2);
        }
        c->cases = narr;
    }
    c->cases[c->count++] = corpus_case_t {m, n, k, label};
}

static void parse_header(
        corpus_t *c, const char *text, const char *file, int line) {
    const char *eq = strchr(text, '=');
    char key[32];
    size_t klen = eq ? (size_t)(eq - text) : strlen(text);
    if (klen >= sizeof(key)) klen = sizeof(key) - 1;
    memcpy(key, text, klen);
    key[klen] = '\0';

    int known = 0;
    for (int i = 0; g_corpus_header_keys[i]; i++)
        if (!strcmp(key, g_corpus_header_keys[i])) known = 1;
    if (!known) corpus_fatal(file, line, "unknown header option '%s'", key);
    if (!eq || !eq[1])
        corpus_fatal(file, line, "header option '%s' needs a value", key);
    const char *val = eq + 1;

    if (!strcmp(key, "--mode")) {
        if (strcmp(val, "C") != 0 && strcmp(val, "P") != 0
                && strcmp(val, "S") != 0)
            corpus_fatal(file, line,
                    "bad value for --mode: '%s' (want C, P or S)", val);
        /* differential is inherently mode C: the value is validated and
         * ignored — bench honors it */
    } else { /* --dt */
        rvcl_dtype_t dt2;
        if (dtype_ops_from_dtstr(val, &dt2)) {
            c->dt_explicit = 1;
            c->dt = dt2;
        } else {
            corpus_fatal(file, line,
                    "bad value for --dt: '%s' (want f32 or f16)", val);
        }
    }
}

static void parse_case(corpus_t *c, char *text, const char *file, int line) {
    /* shape token: ends at whitespace, '_' (glued label) or EOL */
    char *tok = text;
    size_t tlen = strcspn(text, " \t_");
    char saved = tok[tlen];
    tok[tlen] = '\0';

    const char *p = tok;
    int64_t t1[3], m, n, k;
    parse_triple(&p, t1, file, line, tok);
    if (*p == ':') { /* double triple MxK:KxN — K values must match */
        p++;
        int64_t t2[3];
        parse_triple(&p, t2, file, line, tok);
        if (t1[2] != t2[1])
            corpus_fatal(file, line,
                    "K mismatch in double triple '%s': %lld vs %lld", tok,
                    (long long)t1[2], (long long)t2[1]);
        m = t1[1];
        n = t2[2];
        k = t1[2];
    } else {
        m = t1[0];
        n = t1[1];
        k = t1[2];
    }
    if (*p == '*') { /* occurrence count: tolerated and ignored */
        p++;
        if (*p < '0' || *p > '9')
            corpus_fatal(file, line, "unparsable case '%s'", tok);
        while (*p >= '0' && *p <= '9')
            p++;
    }
    if (*p != '\0') corpus_fatal(file, line, "unparsable case '%s'", tok);

    tok[tlen] = saved;

    /* separator then optional label */
    char *rest = text + tlen;
    rest += strspn(rest, " \t_");
    if (*rest == '\0' || *rest == '#') { /* no label: shape is the label */
        char shape_s[64];
        snprintf(shape_s, sizeof(shape_s), "%lldx%lldx%lld", (long long)m,
                (long long)n, (long long)k);
        corpus_add(c, m, n, k, strdup(shape_s), file, line);
        return;
    }
    if (rest[0] != 'n' || rest[1] != '"')
        corpus_fatal(
                file, line, "expected n\"label\" after shape in '%s'", text);
    char *name = rest + 2;
    char *close = strchr(name, '"');
    if (!close) corpus_fatal(file, line, "unterminated label in '%s'", text);
    for (const char *q = name; q < close; q++)
        /* Whitelist check reads clearer as "NOT one of the valid chars"
         * than the DeMorgan-expanded conjunction.
         * NOLINTNEXTLINE(readability-simplify-boolean-expr) */
        if (!((*q >= 'A' && *q <= 'Z') || (*q >= 'a' && *q <= 'z')
                    || (*q >= '0' && *q <= '9') || *q == '_' || *q == '.'
                    || *q == '*' || *q == '-'))
            corpus_fatal(
                    file, line, "bad label character '%c' in '%s'", *q, text);
    if (close == name) corpus_fatal(file, line, "empty label in '%s'", text);
    *close = '\0';
    close++;
    close += strspn(close, " \t");
    if (*close != '\0' && *close != '#')
        corpus_fatal(file, line, "trailing junk after label in '%s'", text);

    corpus_add(c, m, n, k, strdup(name), file, line);
}

static void parse_corpus(const char *path, corpus_t *c) {
    memset(c, 0, sizeof(*c));
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, "error: cannot open corpus '%s'\n", path);
        exit(2);
    }

    char *line = NULL;
    size_t cap = 0;
    ssize_t len;
    int lineno = 0, seen_case = 0;
    while ((len = getline(&line, &cap, f)) != -1) {
        lineno++;
        while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r'))
            line[--len] = '\0';
        char *p = line + strspn(line, " \t");
        if (*p == '\0' || *p == '#') continue;
        if (!strncmp(p, "--", 2)) {
            if (seen_case)
                corpus_fatal(path, lineno,
                        "header option after first case: '%s'", p);
            /* trim trailing comment/blank off the option line */
            char *e = strpbrk(p, " \t#");
            if (e) *e = '\0';
            parse_header(c, p, path, lineno);
            continue;
        }
        seen_case = 1;
        parse_case(c, p, path, lineno);
    }
    free(line);
    fclose(f);

    if (c->count == 0)
        corpus_fatal(path, lineno, "empty corpus (no case lines)");
}

/* -------------------------------------------------- corpus: run one case
 *
 * Runs every capability-legal registered kernel for the dtype through
 * the factory bypass and compares against the shared golden. Returns
 * 0 = ok, 1 = failure, -1 = skip (no legal kernel, dtype defaulted).
 */

/* Covers every kernel legal for one dtype with headroom; growth beyond
 * this fails loudly instead of silently dropping corpus coverage. */
#define CORPUS_MAX_KERNELS 8

static int legal_kernels_for(rvcl_operation_t op, rvcl_dtype_t dt,
        const char *names[CORPUS_MAX_KERNELS],
        rvcl_assert_level_t levels[CORPUS_MAX_KERNELS]) {
    const rvcl_capability_t *cap = rvcl_get_capability();
    int nk = 0;
    for (int i = 0; i < rvcl_registry_count(); i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd || kd->operation != op) continue;
        if (kd->dtype_a != dt) continue;
        int legal = 1;
        for (int f = 0; f < kd->num_required_features; f++)
            if (!rvcl_capability_has(cap, kd->required_features[f])) legal = 0;
        if (!legal) continue;
        /* Fail loudly instead of silently dropping kernels: a growing
         * registry must not shrink corpus coverage unnoticed. */
        if (nk == CORPUS_MAX_KERNELS) {
            fprintf(stderr,
                    "corpus: more than %d legal kernels for the dtype, "
                    "raise CORPUS_MAX_KERNELS\n",
                    CORPUS_MAX_KERNELS);
            /* exit(2) = corpus/usage error, as used by the parser paths. */
            exit(2);
        }
        names[nk] = kd->name;
        levels[nk] = kd->assert_level;
        nk++;
    }
    return nk;
}

/* Runs every capability-legal registered kernel for the operator/dtype
 * through the slot's factory-bypass run and compares against the slot's
 * golden. Returns 0 = ok, 1 = failure, -1 = skip (no legal kernel). */
static int run_case(const op_slot_t *slot, const corpus_case_t *cc,
        rvcl_dtype_t dt, int dt_explicit) {
    const dtype_ops_t *op = dtype_ops_for(dt);
    /* An unsupported dtype has no test row: report like a missing kernel
     * instead of dereferencing a NULL operation bundle. */
    if (!op) {
        if (dt_explicit) {
            printf("FAIL %s: no test table row for dtype %d\n", cc->label,
                    (int)dt);
            return 1;
        }
        return -1;
    }
    const char *names[CORPUS_MAX_KERNELS];
    rvcl_assert_level_t levels[CORPUS_MAX_KERNELS];
    const int nk = legal_kernels_for(slot->op, dt, names, levels);
    const char *dtstr = op->dtstr;
    char shape_s[64];
    snprintf(shape_s, sizeof(shape_s), "%lldx%lldx%lld", (long long)cc->m,
            (long long)cc->n, (long long)cc->k);

    if (nk == 0) {
        if (dt_explicit) {
            printf("FAIL %s (%s, %s): no kernel registered for %s\n", cc->label,
                    shape_s, dtstr, dtstr);
            return 1;
        }
        printf("skip: %s (no %s kernel on this host)\n", cc->label, dtstr);
        return -1;
    }

    /* Build the operator descriptor through the slot. */
    rvcl_op_desc_t od;
    slot->build_desc(&od, cc->m, cc->n, cc->k, dt, dt, RVCL_DTYPE_AUTO,
            RVCL_DTYPE_FP32, RVCL_DTYPE_FP32);

    const int64_t a_elems = slot->a_elems(&od);
    const int64_t b_elems = slot->b_elems(&od);
    const int64_t bias_elems = slot->bias_elems(&od);
    const int64_t elems = slot->out_elems(&od);
    const uint32_t seed = golden_case_seed(cc->m, cc->n, cc->k);
    float *golden = (float *)malloc((size_t)elems * 4);
    float *c = (float *)malloc((size_t)elems * 4);
    int fails = 0;

    float *bias = (float *)malloc((size_t)bias_elems * 4);
    lcg_fill_f32(bias, (size_t)bias_elems, seed ^ 0x85ebca6bu);

    /* dtype-storage A/B buffers, sized and filled through the test table */
    void *a = malloc((size_t)a_elems * op->elt);
    void *b = malloc((size_t)b_elems * op->elt);
    op->fill(a, (size_t)a_elems, seed);
    op->fill(b, (size_t)b_elems, seed ^ 0x9e3779b9u);
    slot->golden(&od, a, b, bias, golden);

    for (int i = 0; i < nk; i++) {
        golden_sentinel(c, (size_t)elems * 4);
        rvcl_status_t st
                = slot->run_named(names[i], &od, a, b, bias, c, -1, -1);
        if (st != RVCL_SUCCESS) {
            printf("FAIL %s (%s, %s): kernel %s run status %d\n", cc->label,
                    shape_s, dtstr, names[i], (int)st);
            fails++;
            continue;
        }
        golden_cmp_t r = op->compare(c, golden, elems, levels[i], dt);
        if (r.mismatches == 0) {
            printf("  ok: %s (%s, %s) %s\n", cc->label, shape_s, dtstr,
                    names[i]);
        } else {
            const int64_t width = slot->op == RVCL_OP_MATMUL ? cc->n : 1;
            printf("FAIL %s (%s, %s): %s: first mismatch at "
                   "(i=%lld,j=%lld) exp=%g got=%g (%lld mismatch)\n",
                    cc->label, shape_s, dtstr, names[i],
                    (long long)(r.first / width), (long long)(r.first % width),
                    r.exp_first, r.got_first, (long long)r.mismatches);
            fails++;
        }
    }
    free(a);
    free(b);
    free(bias);
    free(golden);
    free(c);
    return fails ? 1 : 0;
}

/* -------------------------------------------------- scheduler test rig */

/* A scheduler that splits M into 4 slices and runs them serially in this
 * thread, recording the slices it executed. */
#define MAX_SLICES 32
static rvcl_work_range_t g_slices[MAX_SLICES];
static int g_num_slices;

static void no_op_task(
        rvcl_ischeduler_t *s, const rvcl_work_range_t *r, void *ctx) {
    (void)s;
    (void)r;
    (void)ctx;
}

static void recording_parallel_for(rvcl_ischeduler_t *s,
        const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
        void *user_ctx) {
    (void)s;
    g_num_slices = 0;
    const int64_t rows = total->m_end - total->m_begin;
    const int parts = 4;
    const int64_t step = (rows + parts - 1) / parts;
    for (int64_t b = total->m_begin; b < total->m_end; b += step) {
        rvcl_work_range_t slice = *total;
        slice.m_begin = b;
        slice.m_end = b + step < total->m_end ? b + step : total->m_end;
        g_slices[g_num_slices++] = slice;
        task(s, &slice, user_ctx);
    }
}

/* --------------------------------------------------------------- main */

/*
 * The generic rvv names exist only on vector builds; non-vector builds
 * register the portable _rvv_c entries instead. Exactly one of the two
 * exists per dtype in any build, so resolve the target at runtime instead
 * of hard-coding build-dependent names.
 */
static const char *probe_rvv_name(const char *generic, const char *portable) {
    return rvcl_registry_find_factory(generic) ? generic : portable;
}

int main(int argc, char **argv) {
    const char *batch = NULL;
    const char *dt_arg = NULL;
    const char *op_arg = NULL;
    for (int i = 1; i < argc; i++) {
        if (!strncmp(argv[i], "--batch=", 8) && argv[i][8])
            batch = argv[i] + 8;
        else if (!strncmp(argv[i], "--dt=", 5) && argv[i][5])
            dt_arg = argv[i] + 5;
        else if (!strncmp(argv[i], "--op=", 5) && argv[i][5])
            op_arg = argv[i] + 5;
        else {
            fprintf(stderr,
                    "usage: %s [--batch=<corpus-file>] "
                    "[--dt=f32|f16] [--op=matmul|gemv]\n",
                    argv[0]);
            return 2;
        }
    }

    /* The operator under test: an explicit --op wins, otherwise derive it
     * from the corpus directory name (which is the operator), defaulting
     * to matmul with no batch. */
    const op_slot_t *slot;
    if (op_arg) {
        slot = op_slot_from_name(op_arg);
        if (!slot) {
            fprintf(stderr,
                    "error: bad value for --op: '%s' "
                    "(want matmul or gemv)\n",
                    op_arg);
            return 2;
        }
    } else {
        slot = batch ? op_slot_from_batch_path(batch) : op_slot_default();
    }

    /* --dt overrides a --dt header in the corpus file (the bench --mode
     * override rule) so a forced-path run can pin the dtype from ctest
     * regardless of what the corpus file declares. */
    int dt_only_f32 = 0, dt_only_f16 = 0;
    if (dt_arg) {
        rvcl_dtype_t dt2;
        if (dtype_ops_from_dtstr(dt_arg, &dt2)) {
            dt_only_f32 = dt2 == RVCL_DTYPE_FP32;
            dt_only_f16 = dt2 == RVCL_DTYPE_FP16;
        } else {
            fprintf(stderr,
                    "error: bad value for --dt: '%s' (want f32 or f16)\n",
                    dt_arg);
            return 2;
        }
    }

    CHECK(rvcl_init() == RVCL_SUCCESS, "rvcl_init succeeds");

    const char *f32_rvv_name
            = probe_rvv_name("rvcl_matmul_f32_rvv", "rvcl_matmul_f32_rvv_c");
    const char *f16_rvv_name
            = probe_rvv_name("rvcl_matmul_f16_rvv", "rvcl_matmul_f16_rvv_c");

    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "reference f32 kernel present");
    CHECK(rvcl_registry_find_factory(f32_rvv_name) != NULL,
            "rvv f32 kernel present (generic or portable)");
    CHECK(rvcl_registry_find_factory(f16_rvv_name) != NULL,
            "rvv f16 kernel present (generic or portable)");

    /* The fixed contract block below is matmul-specific: it drives the
     * matmul reference/rvv pair, matmul dispatch and matmul work ranges.
     * A non-matmul operator (e.g. --op=gemv) runs only its corpus batch and
     * must not inherit this block, whose forced C path would otherwise pull
     * the matmul vector-tier 1-ulp case into a gemv-targeted run. */
    if (slot->op == RVCL_OP_MATMUL) {
        /* ---- fp32 differential: reference vs rvv (C path on x86) ---- */
        enum { M = 33, N = 19, K = 47 }; /* ragged sizes to hit panel edges */
        float *a = NULL, *b = NULL;
        float *bias = (float *)malloc((size_t)N * 4);
        float *c_ref = (float *)malloc((size_t)M * N * 4),
              *c_rvv = (float *)malloc((size_t)M * N * 4);
        float *golden = (float *)malloc((size_t)M * N * 4);
        if (!dt_only_f16) {
            a = (float *)malloc((size_t)M * K * 4);
            b = (float *)malloc((size_t)K * N * 4);
            for (int i = 0; i < M * K; i++)
                a[i] = (float)((i * 13) % 17) / 17.0f - 0.5f;
            for (int i = 0; i < K * N; i++)
                b[i] = (float)((i * 7) % 19) / 19.0f - 0.5f;
        }
        for (int j = 0; j < N; j++)
            bias[j] = 0.25f;

        rvcl_matmul_desc_t d32 = {
                .m = M,
                .n = N,
                .k = K,
                .dtype_a = RVCL_DTYPE_FP32,
                .dtype_b = RVCL_DTYPE_FP32,
                .dtype_acc = RVCL_DTYPE_AUTO,
                .dtype_c = RVCL_DTYPE_FP32,
                .dtype_bias = RVCL_DTYPE_FP32,
                .quant = {},
        };

        if (!dt_only_f16) {
            golden_f32(a, b, bias, golden, M, N, K);

            CHECK(run_matmul_kernel("rvcl_matmul_f32_reference", &d32, a, b,
                          bias, c_ref, -1, -1)
                            == RVCL_SUCCESS,
                    "reference f32 runs");
            CHECK(run_matmul_kernel(
                          f32_rvv_name, &d32, a, b, bias, c_rvv, -1, -1)
                            == RVCL_SUCCESS,
                    "rvv f32 runs (packed path)");

            int bad_ref = 0, bad_rvv = 0;
            for (int i = 0; i < M * N; i++) {
                if (c_ref[i] != golden[i]) bad_ref++;
                if (c_rvv[i] != golden[i]) bad_rvv++;
            }
            CHECK(bad_ref == 0, "reference matches naive golden (bit-exact)");
            CHECK(bad_rvv == 0,
                    "rvv f32 (packed) matches naive golden (bit-exact)");

            /* fp32 must be identical between backends too */
            int diff_backends = 0;
            for (int i = 0; i < M * N; i++)
                if (c_ref[i] != c_rvv[i]) diff_backends++;
            CHECK(diff_backends == 0,
                    "reference and rvv f32 outputs identical");

            /* ---- fp32 without bias: NULL-bias branch of the same kernels ---- */
            golden_f32(a, b, NULL, golden, M, N, K);
            CHECK(run_matmul_kernel(
                          f32_rvv_name, &d32, a, b, NULL, c_rvv, -1, -1)
                            == RVCL_SUCCESS,
                    "rvv f32 runs without bias");
            int bad_nobias = 0;
            for (int i = 0; i < M * N; i++)
                if (c_rvv[i] != golden[i]) bad_nobias++;
            CHECK(bad_nobias == 0,
                    "rvv f32 no-bias matches naive golden (bit-exact)");
        }

        /* ---- fp16 differential: rvv f16 vs golden soft-convert ---- */
        uint16_t *ah = NULL, *bh = NULL;
        if (!dt_only_f32) {
            const dtype_ops_t *op16 = dtype_ops_for(RVCL_DTYPE_FP16);
            ah = (uint16_t *)malloc((size_t)M * K * 2);
            bh = (uint16_t *)malloc((size_t)K * N * 2);
            for (int i = 0; i < M * K; i++)
                ah[i] = op16->f32_to_storage(
                        (float)((i * 11) % 23) / 23.0f - 0.5f);
            for (int i = 0; i < K * N; i++)
                bh[i] = op16->f32_to_storage(
                        (float)((i * 5) % 29) / 29.0f - 0.5f);

            rvcl_matmul_desc_t d16 = {
                    .m = M,
                    .n = N,
                    .k = K,
                    .dtype_a = RVCL_DTYPE_FP16,
                    .dtype_b = RVCL_DTYPE_FP16,
                    .dtype_acc = RVCL_DTYPE_AUTO,
                    .dtype_c = RVCL_DTYPE_FP32,
                    .dtype_bias = RVCL_DTYPE_FP32,
                    .quant = {},
            };
            golden_f16(ah, bh, bias, golden, M, N, K);
            CHECK(run_matmul_kernel(
                          f16_rvv_name, &d16, ah, bh, bias, c_rvv, -1, -1)
                            == RVCL_SUCCESS,
                    "rvv f16 runs (packed path)");
            const golden_cmp_t cmp16 = op16->compare(c_rvv, golden,
                    (int64_t)M * N, RVCL_ASSERT_TOLERANT, RVCL_DTYPE_FP16);
            CHECK(cmp16.mismatches == 0,
                    "rvv f16 (packed) matches soft-converted golden");
        }

        /* ---- dispatch selection through the full pipeline ---- */
        {
            rvcl_dispatch_result_t *r = NULL;
            /* Both this backend pair and the reference entry serve the shape;
         * the selection itself is covered by the dispatch unit tests, so
         * here only assert that a full dispatch call succeeds. */
            CHECK(rvcl_dispatch_matmul(&d32, &r) == RVCL_SUCCESS,
                    "dispatch selects a kernel for fp32");
            if (r) {
                printf("  (host selects: %s)\n", rvcl_dispatch_kernel_name(r));
                rvcl_dispatch_release(r);
            }
        }

        /* ---- scheduler acceptance: injection does not change results ---- */
        {
            rvcl_ischeduler_t rec = {
                    .parallel_for = recording_parallel_for,
                    .num_threads = 4,
                    .name = "recording-4-slice",
            };
            /* NOTE: set-once means we cannot inject after the serial scheduler
         * was installed by default. The smoke suite owns the serial set;
         * here we verify slice coverage only (same process constraint). */
            g_num_slices = 0;
            rvcl_work_range_t total = rvcl_work_range_full(M, N, K);
            recording_parallel_for(&rec, &total, no_op_task, NULL);
            CHECK(g_num_slices == 4, "4-slice scheduler produced 4 slices");

            /* coverage: no overlap, no gap over M */
            int64_t covered = 0, prev_end = 0;
            int overlap = 0;
            for (int i = 0; i < g_num_slices; i++) {
                if (g_slices[i].m_begin < prev_end) overlap = 1;
                covered += g_slices[i].m_end - g_slices[i].m_begin;
                prev_end = g_slices[i].m_end;
            }
            CHECK(overlap == 0, "slices do not overlap");
            CHECK(covered == M, "slices cover all rows without gaps");
        }

        /* ---- work-range honoring inside a kernel ---- */
        if (!dt_only_f16) {
            /* run_matmul_kernel uses the full range; exercise a sub-range by
         * passing an explicit row span [4,9). */
            memset(c_rvv, 0, (size_t)M * N * 4);
            rvcl_status_t st = run_matmul_kernel(f32_rvv_name, &d32, a, b, bias,
                    c_rvv, 4, 9 /* rows [4,9) only */);
            CHECK(st == RVCL_SUCCESS, "rvv f32 runs on sub-range");
            int bad_full = 0, untouched_ok = 1;
            golden_f32(a, b, bias, golden, M, N, K);
            for (int i = 0; i < M; i++)
                for (int j = 0; j < N; j++) {
                    if (i >= 4 && i < 9) {
                        if (c_rvv[i * N + j] != golden[i * N + j]) bad_full++;
                    } else if (c_rvv[i * N + j] != 0.0f) {
                        untouched_ok = 0;
                    }
                }
            CHECK(bad_full == 0, "sub-range block matches golden");
            CHECK(untouched_ok, "rows outside sub-range untouched");
        }

        /* ---- dispatch cache ---- */
        {
            rvcl_dispatch_cache_clear();
            rvcl_dispatch_result_t *r1 = NULL, *r2 = NULL;
            CHECK(rvcl_dispatch_matmul(&d32, &r1) == RVCL_SUCCESS,
                    "dispatch after clear (miss)");
            CHECK(rvcl_dispatch_matmul(&d32, &r2) == RVCL_SUCCESS,
                    "second identical dispatch (hit)");
            CHECK(r1 == r2, "cache hit returns the same result object");
            CHECK(rvcl_dispatch_cache_misses() >= 1, "miss counter advanced");
            CHECK(rvcl_dispatch_cache_hits() >= 1, "hit counter advanced");
            /* r1 == r2 (cache-owned single object): one release suffices. */
            rvcl_dispatch_release(r1);
        }

        free(a);
        free(b);
        free(bias);
        free(c_ref);
        free(c_rvv);
        free(golden);
        free(ah);
        free(bh);
    }

    /* ---- corpus batch (shape numerics; contract cases stay above) ---- */
    int batch_fail = 0;
    if (batch) {
        corpus_t corpus;
        parse_corpus(batch, &corpus);
        if (dt_arg) {
            corpus.dt_explicit = 1;
            corpus.dt = dt_only_f32 ? RVCL_DTYPE_FP32 : RVCL_DTYPE_FP16;
        }

        int ok = 0, fail = 0, skip = 0, fi = 0;
        const char **failed_labels = (const char **)calloc(
                (size_t)corpus.count * 2, sizeof(*failed_labels));
        for (int i = 0; i < corpus.count; i++) {
            for (int dti = 0; slot->dtypes[dti] != RVCL_DTYPE_AUTO; dti++) {
                const rvcl_dtype_t dt = slot->dtypes[dti];
                if (corpus.dt_explicit && corpus.dt != dt) continue;
                int rc = run_case(
                        slot, &corpus.cases[i], dt, corpus.dt_explicit);
                if (rc < 0)
                    skip++;
                else if (rc > 0) {
                    fail++;
                    failed_labels[fi++] = corpus.cases[i].label;
                } else
                    ok++;
            }
        }
        printf("CASES: %d ok, %d fail, %d skip\n", ok, fail, skip);
        if (fail) {
            printf("failed:");
            for (int i = 0; i < fi; i++)
                printf(" %s", failed_labels[i]);
            printf("\n");
        }
        batch_fail = fail;
        for (int i = 0; i < corpus.count; i++)
            free(corpus.cases[i].label);
        free(corpus.cases);
        /* free() of a calloc'd const char** array; the multi-level -> void*
         * conversion is inherent to free() and cannot be avoided. */
        free(failed_labels); // NOLINT(bugprone-multi-level-implicit-pointer-conversion)
    }

    printf(g_failures || batch_fail ? "DIFFERENTIAL: %d failure(s)\n"
                                    : "DIFFERENTIAL: all passed\n",
            g_failures + batch_fail);
    return (g_failures || batch_fail) ? 1 : 0;
}
