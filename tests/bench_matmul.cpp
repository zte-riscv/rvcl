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
 * tests/bench_matmul.cpp — benchmark harness with mode separation + JSONL
 *
 * PLAN §6.2 item 10 + §8.2: four metric classes per (shape, dtype):
 *   kernel-only  — micro-kernel on pre-packed data (panelized B); goes
 *                  through rvv_pick_gemm_* — the same selector the
 *                  backend create binds, so it always measures what
 *                  dispatch runs (--path=NAME forces a specific
 *                  micro-kernel through the shared by-name resolver,
 *                  the same one behind RVCL_RVV_KERNEL)
 *   pack-only    — the packing transform alone
 *   pack+kernel  — run() as the framework delivers it (pack inside)
 *   end-to-end   — rvcl_matmul() including dispatch + cache
 *
 * Modes (--mode=, or the corpus file header; CLI wins):
 *   C  correctness: dispatch -> run -> golden check per case, fail = 1
 *   P  performance (default): warmup once + best-of-N (--reps, default 5)
 *   S  survival (QEMU/TCG): single pass, no warmup, direct fills, reps 1
 *
 * Usage: rvcl_bench [--json] [--reps N] [--mode=C|P|S]
 *                   [--path={auto,c,jit,intrinsics}] [--batch=FILE]
 *                   [MxNxK ...]
 *   shape syntax: MxNxK, or the benchdnn-compatible MxK:KxN in corpora.
 *   --batch and CLI shapes are mutually exclusive; with neither, the
 *   default sweep below runs.
 *
 * Output: human table (with the dispatched kernel name), or JSON Lines
 * (schema v1): a leading _meta object, then one object per case. The
 * `kernel` key attributes each case to the kernel dispatch actually
 * selected — the observability point for capability-tiered builds. The
 * `scheduler` key is an additive, schema-stable attribution field: it
 * names the scheduler that handles the end-to-end metrics
 * (rvcl_scheduler_get(): lazy default or an injected one), so multi-thread
 * timing is attributable (D-11). Being additive, it does not advance the
 * JSON schema — the schema stays at v1 and existing consumers reading
 * schema v1 remain valid.
 *
 * This file holds corpus parser implementation #2 (differential has
 * #1): both follow the error table in tests/inputs/README.md, and the
 * same corpus runs green through both — cross-validation is the
 * feature.
 ******************************************************************************/

#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "../src/backends/rvv/rvv_internal.hpp"
#include "../src/runtime/dispatch_internal.h"
#include "dtype_ops.hpp"
#include "golden.hpp"
#include "op_slots.hpp"
#include "rvcl/rvcl.h"

/* ------------------------------------------------------------- timing */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* P mode: warmup once, best-of over reps (micro-benchmark standard) */
static double time_best_of(void (*fn)(void *), void *ctx, int reps) {
    fn(ctx); /* warmup */
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        double t0 = now_ms();
        fn(ctx);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* S mode: one cold pass, no warmup — timing here is not a performance
 * conclusion (the JSON row carries mode:"S" so downstream knows) */
static double time_once(void (*fn)(void *), void *ctx) {
    double t0 = now_ms();
    fn(ctx);
    return now_ms() - t0;
}

/* ------------------------------------------------- corpus: parser #2 */

typedef struct {
    int64_t m, n, k;
    char label[96]; /* n"…" or the normalized shape string */
} bcase_t;

typedef struct {
    bcase_t *cases;
    int count, cap;
    char file_mode; /* 'C'|'P'|'S' from the --mode header, 0 if absent */
    int dt_fixed; /* 0 = both dtypes */
    rvcl_dtype_t dt;
} bcorpus_t;

/* bench's own key table (operator-independent keys today; a new
 * operator directory extends its own keys here without touching the
 * grammar) */
static const char *const g_bench_header_keys[] = {"--mode", "--dt", NULL};

static void bcorpus_fatal(const char *file, int line, const char *fmt, ...)
        __attribute__((noreturn, format(printf, 3, 4)));

/* printf-style fatal helper; the C varargs are the point of this function.
 * NOLINTNEXTLINE(cert-dcl50-cpp) */
static void bcorpus_fatal(const char *file, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: %s:%d: ", file, line);
    /* va_list is started above; clang-analyzer cannot track it across the
     * intervening fprintf and reports a false positive.
     * NOLINTNEXTLINE(clang-analyzer-valist.Uninitialized) */
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(2);
}

static void bcheck_dim(
        long long v, const char *file, int line, const char *tok) {
    if (v < 1 || v > 4096)
        bcorpus_fatal(file, line,
                "dimension %lld out of bounds 1..4096 in '%s'", v, tok);
}

static void bcorpus_push(bcorpus_t *c, int64_t m, int64_t n, int64_t k,
        const char *label, const char *file, int line) {
    for (int i = 0; i < c->count; i++)
        if (!strcmp(c->cases[i].label, label))
            bcorpus_fatal(file, line, "duplicate label n\"%s\"", label);
    if (c->count == c->cap) {
        c->cap = c->cap ? c->cap * 2 : 64;
        bcase_t *narr = (bcase_t *)realloc(
                c->cases, (size_t)c->cap * sizeof(*c->cases));
        if (!narr) {
            fprintf(stderr, "error: out of memory\n");
            exit(2);
        }
        c->cases = narr;
    }
    bcase_t *bc = &c->cases[c->count++];
    bc->m = m;
    bc->n = n;
    bc->k = k;
    snprintf(bc->label, sizeof(bc->label), "%s", label);
}

static void bparse_header(
        bcorpus_t *c, char *text, const char *file, int line) {
    char *e = strpbrk(text, " \t#");
    if (e) *e = '\0';
    char *eq = strchr(text, '=');
    if (!eq || !eq[1]) {
        char key[64];
        snprintf(key, sizeof(key), "%s", text);
        bcorpus_fatal(file, line, "header option '%s' needs a value", key);
    }
    *eq = '\0';
    const char *val = eq + 1;

    int known = 0;
    for (int i = 0; g_bench_header_keys[i]; i++)
        if (!strcmp(text, g_bench_header_keys[i])) known = 1;
    if (!known) bcorpus_fatal(file, line, "unknown header option '%s'", text);

    if (!strcmp(text, "--mode")) {
        if (strlen(val) != 1
                || (val[0] != 'C' && val[0] != 'P' && val[0] != 'S'))
            bcorpus_fatal(file, line,
                    "bad value for --mode: '%s' (want C, P or S)", val);
        c->file_mode = val[0];
    } else { /* --dt */
        rvcl_dtype_t dt2;
        if (dtype_ops_from_dtstr(val, &dt2)) {
            c->dt_fixed = 1;
            c->dt = dt2;
        } else {
            bcorpus_fatal(file, line,
                    "bad value for --dt: '%s' (want f32 or f16)", val);
        }
    }
}

/* shape token parse via scanf pattern on a NUL-terminated token */
static void bparse_shape(char *tok, int64_t *pm, int64_t *pn, int64_t *pk,
        const char *file, int line) {
    long long t1[3], t2[3], occ;
    int pos = 0;
    if (sscanf(tok, "%lldx%lldx%lld%n", &t1[0], &t1[1], &t1[2], &pos) != 3
            || pos <= 0)
        bcorpus_fatal(file, line, "unparsable case '%s'", tok);
    bcheck_dim(t1[0], file, line, tok);
    bcheck_dim(t1[1], file, line, tok);
    bcheck_dim(t1[2], file, line, tok);

    char *rest = tok + pos;
    if (*rest == ':') { /* MxK:KxN — the two K values must match */
        int pos2 = 0;
        rest++;
        if (sscanf(rest, "%lldx%lldx%lld%n", &t2[0], &t2[1], &t2[2], &pos2) != 3
                || pos2 <= 0)
            bcorpus_fatal(file, line, "unparsable case '%s'", tok);
        bcheck_dim(t2[0], file, line, tok);
        bcheck_dim(t2[1], file, line, tok);
        bcheck_dim(t2[2], file, line, tok);
        if (t1[2] != t2[1])
            bcorpus_fatal(file, line,
                    "K mismatch in double triple '%s': %lld vs %lld", tok,
                    t1[2], t2[1]);
        *pm = t1[1];
        *pn = t2[2];
        *pk = t1[2];
        rest += pos2;
    } else {
        *pm = t1[0];
        *pn = t1[1];
        *pk = t1[2];
    }

    if (*rest == '*') { /* occurrence count: tolerated and ignored */
        int pos3 = 0;
        rest++;
        if (sscanf(rest, "%lld%n", &occ, &pos3) != 1 || pos3 <= 0)
            bcorpus_fatal(file, line, "unparsable case '%s'", tok);
        rest += pos3;
    }
    if (*rest != '\0') bcorpus_fatal(file, line, "unparsable case '%s'", tok);
}

static void bparse_case(bcorpus_t *c, char *text, const char *file, int line) {
    /* token = shape(+occurrence); ends at whitespace, '_' or EOL */
    size_t tlen = strcspn(text, " \t_");
    char saved = text[tlen];
    text[tlen] = '\0';

    int64_t m, n, k;
    bparse_shape(text, &m, &n, &k, file, line);
    text[tlen] = saved;

    char *rest = text + tlen;
    rest += strspn(rest, " \t_");
    if (*rest == '\0' || *rest == '#') { /* no label: shape is the label */
        char shape_s[64];
        snprintf(shape_s, sizeof(shape_s), "%lldx%lldx%lld", (long long)m,
                (long long)n, (long long)k);
        bcorpus_push(c, m, n, k, shape_s, file, line);
        return;
    }
    if (rest[0] != 'n' || rest[1] != '"')
        bcorpus_fatal(
                file, line, "expected n\"label\" after shape in '%s'", text);
    char *name = rest + 2;
    char *close = strchr(name, '"');
    if (!close || close == name)
        bcorpus_fatal(file, line, "unterminated or empty label in '%s'", text);
    for (const char *q = name; q < close; q++)
        /* Whitelist check reads clearer as "NOT one of the valid chars"
         * than the DeMorgan-expanded conjunction.
         * NOLINTNEXTLINE(readability-simplify-boolean-expr) */
        if (!((*q >= 'A' && *q <= 'Z') || (*q >= 'a' && *q <= 'z')
                    || (*q >= '0' && *q <= '9') || *q == '_' || *q == '.'
                    || *q == '*' || *q == '-'))
            bcorpus_fatal(
                    file, line, "bad label character '%c' in '%s'", *q, text);
    *close = '\0';
    close++;
    close += strspn(close, " \t");
    if (*close != '\0' && *close != '#')
        bcorpus_fatal(file, line, "trailing junk after label in '%s'", text);

    bcorpus_push(c, m, n, k, name, file, line);
}

static void bparse_corpus(const char *path, bcorpus_t *c) {
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
                bcorpus_fatal(path, lineno,
                        "header option after first case: '%s'", p);
            bparse_header(c, p, path, lineno);
            continue;
        }
        seen_case = 1;
        bparse_case(c, p, path, lineno);
    }
    free(line);
    fclose(f);
    if (c->count == 0)
        bcorpus_fatal(path, lineno, "empty corpus (no case lines)");
}

/* the op key is the corpus directory name (multi-operator landing
 * point: a corpus under tests/inputs/<op>/ reports op:"<op>") */
static void op_from_batch_path(const char *path, char *out, size_t sz) {
    const char *slash = strrchr(path, '/');
    size_t flen = slash ? (size_t)(slash - path) : 0;
    const char *start = path;
    for (size_t i = flen; i > 0;)
        if (path[--i] == '/') {
            start = path + i + 1;
            break;
        }
    if (flen == 0 || start >= path + flen) {
        snprintf(out, sz, "matmul");
        return;
    }
    size_t n = (size_t)(path + flen - start);
    if (n >= sz) n = sz - 1;
    memcpy(out, start, n);
    out[n] = '\0';
}

/* ------------------------------------------------------- bench contexts */

typedef struct {
    int64_t m, n, k;
    const char *label;
    float *a, *b, *c;
    float *bias;
    uint16_t *ah, *bh;
    const op_slot_t *slot;
    rvcl_op_desc_t od;
} case_t;

typedef struct {
    const case_t *cs;
    void *packed;
    rvv_gemm_fn gemm;
} konly_ctx_t;

static void bench_kernel_only(void *p) {
    konly_ctx_t *kc = (konly_ctx_t *)p;
    const case_t *cs = kc->cs;
    rvcl_work_range_t full = rvcl_work_range_full(cs->m, cs->n, cs->k);
    kc->gemm(cs->od.matmul.dtype_a == RVCL_DTYPE_FP32 ? (const void *)cs->a
                                                      : (const void *)cs->ah,
            kc->packed, cs->bias, cs->c, cs->m, cs->n, cs->k, &full);
}

typedef struct {
    const case_t *cs;
    const rvcl_tensor_t *b;
    rvcl_tensor_t *dst;
} pack_ctx_t;

static void bench_pack_only(void *p) {
    pack_ctx_t *pc = (pack_ctx_t *)p;
    const case_t *cs = pc->cs;
    size_t elt = rvcl_dtype_size(cs->od.matmul.dtype_a);
    rvv_pack_b_panel(pc->b, pc->dst, cs->m, cs->n, cs->k, elt);
}

typedef struct {
    rvcl_kernel_t *k;
    const rvcl_tensor_pack_t *pack;
    const rvcl_work_range_t *range;
} run_ctx_t;

static void bench_pack_kernel(void *p) {
    run_ctx_t *rc = (run_ctx_t *)p;
    rvcl_kernel_run(rc->k, rc->pack, rc->range);
}

typedef struct {
    const case_t *cs;
} e2e_ctx_t;

static void bench_e2e(void *p) {
    e2e_ctx_t *ec = (e2e_ctx_t *)p;
    const case_t *cs = ec->cs;
    if (cs->od.matmul.dtype_a == RVCL_DTYPE_FP32)
        rvcl_matmul(&cs->od.matmul, cs->a, cs->b, cs->bias, cs->c);
    else
        rvcl_matmul(&cs->od.matmul, cs->ah, cs->bh, cs->bias, cs->c);
}

/* ------------------------------------------------------- JSON helpers */

static const char *jms(char *buf, size_t sz, double v, int avail) {
    if (!avail) {
        snprintf(buf, sz, "null");
    } else {
        snprintf(buf, sz, "%.4f", v);
    }
    return buf;
}

/* --------------------------------------------------------------- main */

typedef enum { MODE_C, MODE_P, MODE_S } bench_mode_t;

/* fills a/b(/ah/bh)/bias deterministically; f16 narrow-from-f32 for P
 * (legacy data path), direct f16 fill for C/S (S skips the fill
 * optimization by design; C shares data with the differential run) */
static void fill_case(case_t *cs, const dtype_ops_t *op, bench_mode_t mode) {
    const int f16 = op->dtype == RVCL_DTYPE_FP16;
    const uint32_t seed = golden_case_seed(cs->m, cs->n, cs->k);
    const int64_t a_elems = cs->slot->a_elems(&cs->od);
    const int64_t b_elems = cs->slot->b_elems(&cs->od);
    const int64_t bias_elems = cs->slot->bias_elems(&cs->od);
    const int64_t c_elems = cs->slot->out_elems(&cs->od);
    cs->a = (float *)calloc((size_t)a_elems, 4);
    cs->b = (float *)calloc((size_t)b_elems, 4);
    cs->c = (float *)calloc((size_t)c_elems, 4);
    cs->bias = (float *)calloc((size_t)bias_elems, 4);
    lcg_fill_f32(cs->a, (size_t)a_elems, seed);
    lcg_fill_f32(cs->b, (size_t)b_elems, seed ^ 0x9e3779b9u);
    lcg_fill_f32(cs->bias, (size_t)bias_elems, seed ^ 0x85ebca6bu);
    if (!f16) return;

    if (mode == MODE_P) {
        cs->ah = (uint16_t *)calloc((size_t)a_elems, 2);
        cs->bh = (uint16_t *)calloc((size_t)b_elems, 2);
        for (int64_t i = 0; i < a_elems; i++)
            cs->ah[i] = op->f32_to_storage(cs->a[i]);
        for (int64_t i = 0; i < b_elems; i++)
            cs->bh[i] = op->f32_to_storage(cs->b[i]);
    } else {
        free(cs->a);
        free(cs->b);
        cs->a = NULL;
        cs->b = NULL;
        cs->ah = (uint16_t *)calloc((size_t)a_elems, 2);
        cs->bh = (uint16_t *)calloc((size_t)b_elems, 2);
        op->fill(cs->ah, (size_t)a_elems, seed);
        op->fill(cs->bh, (size_t)b_elems, seed ^ 0x9e3779b9u);
    }
}

static void free_case(case_t *cs) {
    free(cs->a);
    free(cs->b);
    free(cs->c);
    free(cs->bias);
    free(cs->ah);
    free(cs->bh);
}

/* dispatch-result run rig: pack + workspace, ready for kernel_run */
typedef struct {
    rvcl_dispatch_result_t *r;
    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    void *ws_buf[RVCL_MAX_WORKSPACES];
    int num_ws;
    rvcl_tensor_pack_t pack;
    rvcl_work_range_t full;
} dispatch_rig_t;

static int rig_setup(dispatch_rig_t *g, const case_t *cs) {
    memset(g, 0, sizeof(*g));
    if (rvcl_dispatch_op(cs->slot->op, &cs->od, &g->r) != RVCL_SUCCESS || !g->r)
        return 0;
    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(g->r, &num_ws);
    g->num_ws = num_ws;

    /* fp16 storage lives in the narrow ah/bh buffers; fp32 in a/b. */
    const int f16 = cs->slot->op == RVCL_OP_MATMUL
            && cs->od.matmul.dtype_a == RVCL_DTYPE_FP16;
    const void *a = f16 ? (const void *)cs->ah : (const void *)cs->a;
    const void *b = f16 ? (const void *)cs->bh : (const void *)cs->b;
    op_slot_pack_setup(cs->slot, &g->pack, &cs->od, a, b, cs->bias, cs->c,
            &g->ta, &g->tb, &g->tc, &g->tbias);

    if (op_slot_pack_ws(ws, num_ws, g->ws_buf, g->ws_t, &g->pack)
            != RVCL_SUCCESS) {
        op_slot_pack_free_ws(g->ws_buf, num_ws);
        rvcl_dispatch_release(g->r);
        g->r = NULL;
        return 0;
    }

    const int64_t m
            = cs->slot->op == RVCL_OP_MATMUL ? cs->od.matmul.m : cs->od.gemv.m;
    const int64_t n = cs->slot->op == RVCL_OP_MATMUL ? cs->od.matmul.n : 1;
    const int64_t k
            = cs->slot->op == RVCL_OP_MATMUL ? cs->od.matmul.k : cs->od.gemv.k;
    g->full = rvcl_work_range_full(m, n, k);
    return 1;
}

static void rig_release(dispatch_rig_t *g) {
    op_slot_pack_free_ws(g->ws_buf, g->num_ws);
    if (g->r) rvcl_dispatch_release(g->r);
}

int main(int argc, char **argv) {
    int json = 0, reps = 5, mode_given = 0;
    const char *path = NULL; /* NULL = backend auto selection */
    bench_mode_t mode = MODE_P;
    const char *batch = NULL;
    static const int64_t defaults[][3]
            = {{64, 64, 64}, {128, 128, 128}, {256, 256, 256}, {128, 256, 512}};
    int64_t shapes[16][3];
    int num_shapes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json"))
            json = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strncmp(argv[i], "--mode=", 7) && argv[i][7]) {
            const char *m = argv[i] + 7;
            if (!strcmp(m, "C"))
                mode = MODE_C;
            else if (!strcmp(m, "P"))
                mode = MODE_P;
            else if (!strcmp(m, "S"))
                mode = MODE_S;
            else {
                fprintf(stderr, "error: bad --mode '%s' (want C, P, S)\n", m);
                return 2;
            }
            mode_given = 1;
        } else if (!strncmp(argv[i], "--path=", 7) && argv[i][7])
            path = argv[i] + 7;
        else if (!strncmp(argv[i], "--batch=", 8) && argv[i][8])
            batch = argv[i] + 8;
        else if (num_shapes < 16) {
            long long m, n, k;
            if (sscanf(argv[i], "%lldx%lldx%lld", &m, &n, &k) == 3 && m > 0
                    && n > 0 && k > 0) {
                shapes[num_shapes][0] = m;
                shapes[num_shapes][1] = n;
                shapes[num_shapes][2] = k;
                num_shapes++;
            } else {
                fprintf(stderr,
                        "error: bad argument '%s'\n"
                        "usage: %s [--json] [--reps N] [--mode=C|P|S] "
                        "[--path={auto,c,jit,intrinsics}] [--batch=FILE] "
                        "[MxNxK ...]\n",
                        argv[i], argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "error: too many shape arguments\n");
            return 2;
        }
    }
    if (batch && num_shapes) {
        fprintf(stderr,
                "error: --batch and shape arguments are mutually "
                "exclusive\n");
        return 2;
    }

    bcorpus_t corpus = {};
    char op[32] = "matmul";
    if (batch) {
        bparse_corpus(batch, &corpus);
        op_from_batch_path(batch, op, sizeof(op));
        /* file header --mode is the default; an explicit CLI --mode wins */
        if (!mode_given && corpus.file_mode) {
            if (corpus.file_mode == 'C')
                mode = MODE_C;
            else if (corpus.file_mode == 'S')
                mode = MODE_S;
            else
                mode = MODE_P;
        }
    } else if (num_shapes == 0) {
        num_shapes = (int)(sizeof(defaults) / sizeof(defaults[0]));
        for (int i = 0; i < num_shapes; i++)
            for (int j = 0; j < 3; j++)
                shapes[i][j] = defaults[i][j];
    }
    if (mode == MODE_S) reps = 1;
    if (reps < 1) reps = 1;

    if (rvcl_init() != RVCL_SUCCESS) return 1;

    /* Scheduler attribution (D-11): which scheduler handles the end-to-end
     * metrics. kernel-only / pack+kernel micro-kernel direct calls are
     * naturally single-core; only rvcl_matmul() end-to-end goes through the
     * current scheduler (lazy default or an injected one). */
    rvcl_ischeduler_t *sched = rvcl_scheduler_get();
    const char *sched_name = sched->name;
    const unsigned sched_num_threads = sched->num_threads;

    const int total = batch ? corpus.count : num_shapes;
    char mode_ch;
    if (mode == MODE_C)
        mode_ch = 'C';
    else if (mode == MODE_S)
        mode_ch = 'S';
    else
        mode_ch = 'P';
    if (json) {
        printf("{\"_meta\":{\"schema\":1,\"rvcl\":\"%s\","
               "\"capability\":\"%s\",\"mode\":\"%c\",\"reps\":%d,"
               "\"batch\":%s%s%s}}\n",
                rvcl_get_version(), rvcl_get_capability()->description, mode_ch,
                reps, batch ? "\"" : "null", batch ? batch : "",
                batch ? "\"" : "");
    } else {
        printf("RVCL %s bench (mode %c, reps %d) — %s\n", rvcl_get_version(),
                mode_ch, reps, rvcl_get_capability()->description);
        printf("%-16s %-5s %-24s %10s %10s %10s %10s\n", "shape", "dtype",
                "kernel", "k-ms", "pack", "p+k", "e2e");
    }

    /* The operator is named on the CLI (--op=) or derived from the batch
     * directory; the slot row carries all op-specific facts. */
    const op_slot_t *slot = op_slot_from_name(op);
    if (!slot) slot = op_slot_default();

    int nok = 0, nfail = 0, nskip = 0;
    for (int s = 0; s < total; s++) {
        case_t cs = {};
        if (batch) {
            cs.m = corpus.cases[s].m;
            cs.n = corpus.cases[s].n;
            cs.k = corpus.cases[s].k;
            cs.label = corpus.cases[s].label;
        } else {
            cs.m = shapes[s][0];
            cs.n = shapes[s][1];
            cs.k = shapes[s][2];
        }
        char shape_s[64];
        snprintf(shape_s, sizeof(shape_s), "%lldx%lldx%lld", (long long)cs.m,
                (long long)cs.n, (long long)cs.k);
        if (!cs.label) cs.label = shape_s;
        char label_s[96];
        snprintf(label_s, sizeof(label_s), "%s", cs.label);

        for (int di = 0; slot->dtypes[di] != RVCL_DTYPE_AUTO; di++) {
            const dtype_ops_t *dop = dtype_ops_for(slot->dtypes[di]);
            if (!dop) continue; /* missing test row (never happens today) */
            const int dt = dop->dtype == RVCL_DTYPE_FP16; /* 0 f32, 1 f16 */
            if (corpus.dt_fixed && dop->dtype != corpus.dt) continue;
            const char *dtstr = dop->dtstr;

            cs.slot = slot;
            slot->build_desc(&cs.od, cs.m, cs.n, cs.k, dop->dtype, dop->dtype,
                    RVCL_DTYPE_AUTO, RVCL_DTYPE_FP32, RVCL_DTYPE_AUTO);
            fill_case(&cs, dop, mode);
            const double flops
                    = 2.0 * (double)cs.m * (double)cs.n * (double)cs.k;
            dispatch_rig_t rig;

            if (mode == MODE_C || !slot->can_timing) {
                /* correctness: dispatch -> run -> shared golden; a
                 * dispatch failure is a case failure here. Ops without a
                 * micro-kernel timing path (can_timing == 0) also land
                 * here for P/S modes. */
                int ok = rig_setup(&rig, &cs);
                const char *detail = "";
                char dbuf[128];
                if (!ok) {
                    snprintf(dbuf, sizeof(dbuf), "dispatch failed");
                    detail = dbuf;
                } else {
                    const int64_t out_elems = slot->out_elems(&cs.od);
                    golden_sentinel(cs.c, (size_t)out_elems * 4);
                    run_ctx_t rc = {.k = rvcl_dispatch_kernel(rig.r),
                            .pack = &rig.pack,
                            .range = &rig.full};
                    rvcl_status_t st = rvcl_kernel_run(rc.k, rc.pack, rc.range);
                    if (st != RVCL_SUCCESS) {
                        snprintf(dbuf, sizeof(dbuf), "run status %d", (int)st);
                        detail = dbuf;
                        ok = 0;
                    } else {
                        float *golden = (float *)malloc((size_t)out_elems * 4);
                        slot->golden(&cs.od,
                                dt ? (const void *)cs.ah : (const void *)cs.a,
                                dt ? (const void *)cs.bh : (const void *)cs.b,
                                cs.bias, golden);
                        const rvcl_kernel_t *k = rvcl_dispatch_kernel(rig.r);
                        golden_cmp_t cmp = dop->compare(cs.c, golden, out_elems,
                                k->desc.assert_level, dop->dtype);
                        free(golden);
                        if (cmp.mismatches) {
                            const int64_t width
                                    = slot->op == RVCL_OP_MATMUL ? cs.n : 1;
                            snprintf(dbuf, sizeof(dbuf),
                                    "first mismatch (i=%lld,j=%lld) "
                                    "exp=%g got=%g",
                                    (long long)(cmp.first / width),
                                    (long long)(cmp.first % width),
                                    cmp.exp_first, cmp.got_first);
                            detail = dbuf;
                            ok = 0;
                        }
                    }
                }
                if (json) {
                    printf("{\"op\":\"%s\",\"shape\":\"%s\","
                           "\"label\":\"%s\",\"dtype\":\"%s\","
                           "\"mode\":\"C\",\"kernel\":%s%s%s,"
                           "\"kernel_ms\":null,\"pack_ms\":null,"
                           "\"pack_kernel_ms\":null,\"e2e_ms\":null,"
                           "\"gflops_kernel\":null,\"gflops_e2e\":null,"
                           "\"scheduler\":{\"name\":\"%s\","
                           "\"num_threads\":%u},\"status\":\"%s\"%s%s%s%s}\n",
                            op, shape_s, label_s, dtstr, ok ? "\"" : "null",
                            ok ? rvcl_dispatch_kernel_name(rig.r) : "",
                            ok ? "\"" : "", sched_name, sched_num_threads,
                            ok ? "ok" : "fail", *detail ? ",\"detail\":" : "",
                            *detail ? "\"" : "", detail, *detail ? "\"" : "");
                } else {
                    printf("%-16s %-5s %-24s C: %s%s%s\n", shape_s, dtstr,
                            ok ? rvcl_dispatch_kernel_name(rig.r) : "-",
                            ok ? "ok" : "FAIL", *detail ? " — " : "", detail);
                }
                if (ok) {
                    nok++;
                    rig_release(&rig);
                } else {
                    nfail++;
                    if (rig.r) rig_release(&rig);
                }
                free_case(&cs);
                continue;
            }

            /* ---- P / S: the four metric classes ---- */
            size_t elt = dt ? 2 : 4;
            void *packed = NULL;
            size_t pbytes = rvv_packed_b_bytes(cs.k, cs.n, elt);
            if (posix_memalign(&packed, 64, pbytes) != 0) {
                fprintf(stderr, "error: packed-B buffer allocation failed\n");
                free_case(&cs); /* do not leak this case's buffers */
                return 1;
            }
            rvcl_tensor_t b_src, b_dst;
            {
                rvcl_tensor_desc_t td = {.ndims = 2,
                        .dims = {cs.k, cs.n},
                        .strides = {cs.n, 1},
                        .dtype = cs.od.matmul.dtype_b};
                rvcl_tensor_init(
                        &b_src, &td, dt ? (void *)cs.bh : (void *)cs.b);
                rvcl_tensor_desc_t pd = {.ndims = 1,
                        .dims = {(int64_t)pbytes},
                        .strides = {1},
                        .dtype = RVCL_DTYPE_UINT8};
                rvcl_tensor_init(&b_dst, &pd, packed);
                rvv_pack_b_panel(&b_src, &b_dst, cs.m, cs.n, cs.k, elt);
            }

            konly_ctx_t kc = {.cs = &cs, .packed = packed, .gemm = NULL};
            /* kernel-only goes through the same selector the backend
             * binds (rvv_pick_*); --path=NAME forces a specific
             * micro-kernel through the shared by-name resolver */
            if (path) {
                kc.gemm = dt ? rvv_pick_gemm_f16_by_name(path)
                             : rvv_pick_gemm_f32_by_name(path);
                if (!kc.gemm) {
                    fprintf(stderr, "error: --path=%s is unavailable for %s\n",
                            path, dtstr);
                    free(packed);
                    free_case(&cs);
                    return 2;
                }
            } else {
                kc.gemm = dt ? rvv_pick_gemm_f16() : rvv_pick_gemm_f32();
            }
            double t_kernel = mode == MODE_S
                    ? time_once(bench_kernel_only, &kc)
                    : time_best_of(bench_kernel_only, &kc, reps);

            pack_ctx_t pc = {.cs = &cs, .b = &b_src, .dst = &b_dst};
            double t_pack = mode == MODE_S
                    ? time_once(bench_pack_only, &pc)
                    : time_best_of(bench_pack_only, &pc, reps);

            int has_dispatch = rig_setup(&rig, &cs);
            double t_pk = 0, t_e2e = 0;
            if (has_dispatch) {
                run_ctx_t rc = {.k = rvcl_dispatch_kernel(rig.r),
                        .pack = &rig.pack,
                        .range = &rig.full};
                t_pk = mode == MODE_S
                        ? time_once(bench_pack_kernel, &rc)
                        : time_best_of(bench_pack_kernel, &rc, reps);
                e2e_ctx_t ec = {.cs = &cs};
                t_e2e = mode == MODE_S ? time_once(bench_e2e, &ec)
                                       : time_best_of(bench_e2e, &ec, reps);
            }
            const char *kname
                    = has_dispatch ? rvcl_dispatch_kernel_name(rig.r) : NULL;

            if (json) {
                char b1[32], b2[32], b3[32], b4[32], b5[32];
                snprintf(b5, sizeof(b5), "%.2f",
                        t_kernel > 0 ? flops / (t_kernel * 1e6) : 0.0);
                printf("{\"op\":\"%s\",\"shape\":\"%s\",\"label\":\"%s\","
                       "\"dtype\":\"%s\",\"mode\":\"%c\",\"kernel\":%s%s%s,"
                       "\"kernel_ms\":%.4f,\"pack_ms\":%s,"
                       "\"pack_kernel_ms\":%s,\"e2e_ms\":%s,"
                       "\"gflops_kernel\":%s,\"gflops_e2e\":%s,"
                       "\"scheduler\":{\"name\":\"%s\",\"num_threads\":%u},"
                       "\"status\":\"%s\"}\n",
                        op, shape_s, label_s, dtstr, mode_ch,
                        kname ? "\"" : "null", kname ? kname : "",
                        kname ? "\"" : "", t_kernel,
                        jms(b1, sizeof(b1), t_pack, 1),
                        jms(b2, sizeof(b2), t_pk, has_dispatch),
                        jms(b3, sizeof(b3), t_e2e, has_dispatch), b5,
                        jms(b4, sizeof(b4),
                                t_e2e > 0 ? flops / (t_e2e * 1e6) : 0.0,
                                has_dispatch),
                        sched_name, sched_num_threads,
                        has_dispatch ? "ok" : "nodispatch");
            } else {
                char sh[32];
                snprintf(sh, sizeof(sh), "%lldx%lldx%lld", (long long)cs.m,
                        (long long)cs.n, (long long)cs.k);
                printf("%-16s %-5s %-24.24s %10.3f %10.3f %10.3f %10.3f\n", sh,
                        dtstr, kname ? kname : "-", t_kernel, t_pack,
                        has_dispatch ? t_pk : 0, has_dispatch ? t_e2e : 0);
            }

            if (has_dispatch) {
                nok++;
                rig_release(&rig);
            } else {
                nskip++; /* environment fact: nothing dispatchable */
            }
            free(packed);
            free_case(&cs);
        }
    }

    if (batch) {
        printf("CASES: %d ok, %d fail, %d skip\n", nok, nfail, nskip);
        free(corpus.cases);
    }
    return nfail ? 1 : 0;
}
