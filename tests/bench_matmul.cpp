/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * tests/bench_matmul.cpp — benchmark harness with mode separation + JSONL
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLAN §6.2 item 10 + §8.2: four metric classes per (shape, dtype):
 *   kernel-only  — micro-kernel on pre-packed data (panelized B); goes
 *                  through rvv_pick_gemm_* — the same selector the
 *                  backend create binds, so it always measures what
 *                  dispatch runs (--path=c forces the C micro-kernel
 *                  as the comparison baseline arm)
 *   pack-only    — the packing transform alone
 *   pack+kernel  — run() as the framework delivers it (pack inside)
 *   end-to-end   — rvcl_matmul() including dispatch + cache
 *
 * Modes (--mode=, or the corpus file header; CLI wins):
 *   C  correctness: dispatch -> run -> golden check per case, fail = 1
 *   P  performance (default): warmup once + best-of-N (--reps, default 5)
 *   S  survival (QEMU/TCG): single pass, no warmup, direct fills, reps 1
 *
 * Usage: rvcl_bench [--json] [--reps N] [--mode=C|P|S] [--path=c]
 *                   [--batch=FILE] [MxNxK ...]
 *   shape syntax: MxNxK, or the benchdnn-compatible MxK:KxN in corpora.
 *   --batch and CLI shapes are mutually exclusive; with neither, the
 *   default sweep below runs.
 *
 * Output: human table (with the dispatched kernel name), or JSON Lines
 * (schema v1): a leading _meta object, then one object per case. The
 * `kernel` key attributes each case to the kernel dispatch actually
 * selected — the observability point for capability-tiered builds.
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

#include "rvcl/rvcl.h"
#include "golden.hpp"
#include "../src/backends/rvv/rvv_fp16.hpp"
#include "../src/backends/rvv/rvv_internal.hpp"

/* ------------------------------------------------------------- timing */

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
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
    char file_mode;     /* 'C'|'P'|'S' from the --mode header, 0 if absent */
    int dt_fixed;       /* 0 = both dtypes */
    rvcl_dtype_t dt;
} bcorpus_t;

/* bench's own key table (operator-independent keys today; a new
 * operator directory extends its own keys here without touching the
 * grammar) */
static const char *const g_bench_header_keys[] = {"--mode", "--dt", NULL};

static void bcorpus_fatal(const char *file, int line, const char *fmt, ...)
        __attribute__((noreturn, format(printf, 3, 4)));

static void bcorpus_fatal(const char *file, int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "error: %s:%d: ", file, line);
    vfprintf(stderr, fmt, ap);
    va_end(ap);
    fputc('\n', stderr);
    exit(2);
}

static void bcheck_dim(long long v, const char *file, int line,
        const char *tok) {
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
        c->cases = (bcase_t *)realloc(c->cases,
                (size_t)c->cap * sizeof(*c->cases));
        if (!c->cases) {
            fprintf(stderr, "error: out of memory\n");
            exit(2);
        }
    }
    bcase_t *bc = &c->cases[c->count++];
    bc->m = m;
    bc->n = n;
    bc->k = k;
    snprintf(bc->label, sizeof(bc->label), "%s", label);
}

static void bparse_header(bcorpus_t *c, char *text, const char *file,
        int line) {
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
    if (!known)
        bcorpus_fatal(file, line, "unknown header option '%s'", text);

    if (!strcmp(text, "--mode")) {
        if (strlen(val) != 1 || (val[0] != 'C' && val[0] != 'P'
                    && val[0] != 'S'))
            bcorpus_fatal(file, line,
                    "bad value for --mode: '%s' (want C, P or S)", val);
        c->file_mode = val[0];
    } else { /* --dt */
        if (!strcmp(val, "f32")) {
            c->dt_fixed = 1;
            c->dt = RVCL_DTYPE_FP32;
        } else if (!strcmp(val, "f16")) {
            c->dt_fixed = 1;
            c->dt = RVCL_DTYPE_FP16;
        } else {
            bcorpus_fatal(file, line,
                    "bad value for --dt: '%s' (want f32 or f16)", val);
        }
    }
}

/* shape token parse via scanf pattern on a NUL-terminated token */
static void bparse_shape(char *tok, int64_t *pm, int64_t *pn,
        int64_t *pk, const char *file, int line) {
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
        if (sscanf(rest, "%lldx%lldx%lld%n", &t2[0], &t2[1], &t2[2],
                    &pos2) != 3
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
    if (*rest != '\0')
        bcorpus_fatal(file, line, "unparsable case '%s'", tok);
}

static void bparse_case(bcorpus_t *c, char *text, const char *file,
        int line) {
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
        snprintf(shape_s, sizeof(shape_s), "%lldx%lldx%lld",
                (long long)m, (long long)n, (long long)k);
        bcorpus_push(c, m, n, k, shape_s, file, line);
        return;
    }
    if (rest[0] != 'n' || rest[1] != '"')
        bcorpus_fatal(file, line, "expected n\"label\" after shape in '%s'",
                text);
    char *name = rest + 2;
    char *close = strchr(name, '"');
    if (!close || close == name)
        bcorpus_fatal(file, line, "unterminated or empty label in '%s'",
                text);
    for (const char *q = name; q < close; q++)
        if (!( (*q >= 'A' && *q <= 'Z') || (*q >= 'a' && *q <= 'z')
                    || (*q >= '0' && *q <= '9') || *q == '_' || *q == '.'
                    || *q == '*' || *q == '-'))
            bcorpus_fatal(file, line, "bad label character '%c' in '%s'",
                    *q, text);
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
    rvcl_matmul_desc_t d;
} case_t;

typedef struct {
    const case_t *cs;
    void *packed;
    rvv_gemm_fn gemm;
} konly_ctx_t;

static void bench_kernel_only(void *p) {
    konly_ctx_t *kc = (konly_ctx_t *)p;
    const case_t *cs = kc->cs;
    rvcl_work_range_t full
            = rvcl_work_range_full(cs->m, cs->n, cs->k);
    kc->gemm(cs->d.dtype_a == RVCL_DTYPE_FP32 ? (const void *)cs->a
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
    size_t elt = rvcl_dtype_size(cs->d.dtype_a);
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
    if (cs->d.dtype_a == RVCL_DTYPE_FP32)
        rvcl_matmul(&cs->d, cs->a, cs->b, cs->bias, cs->c);
    else
        rvcl_matmul(&cs->d, cs->ah, cs->bh, cs->bias, cs->c);
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
static void fill_case(case_t *cs, int f16, bench_mode_t mode) {
    const uint32_t seed = golden_case_seed(cs->m, cs->n, cs->k);
    cs->a = (float *)calloc((size_t)(cs->m * cs->k), 4);
    cs->b = (float *)calloc((size_t)(cs->k * cs->n), 4);
    cs->c = (float *)calloc((size_t)(cs->m * cs->n), 4);
    cs->bias = (float *)calloc((size_t)cs->n, 4);
    lcg_fill_f32(cs->a, (size_t)(cs->m * cs->k), seed);
    lcg_fill_f32(cs->b, (size_t)(cs->k * cs->n), seed ^ 0x9e3779b9u);
    lcg_fill_f32(cs->bias, (size_t)cs->n, seed ^ 0x85ebca6bu);
    if (!f16) return;

    if (mode == MODE_P) {
        cs->ah = (uint16_t *)calloc((size_t)(cs->m * cs->k), 2);
        cs->bh = (uint16_t *)calloc((size_t)(cs->k * cs->n), 2);
        for (int64_t i = 0; i < cs->m * cs->k; i++)
            cs->ah[i] = rvv_f32_to_f16(cs->a[i]);
        for (int64_t i = 0; i < cs->k * cs->n; i++)
            cs->bh[i] = rvv_f32_to_f16(cs->b[i]);
    } else {
        free(cs->a);
        free(cs->b);
        cs->a = NULL;
        cs->b = NULL;
        cs->ah = (uint16_t *)calloc((size_t)(cs->m * cs->k), 2);
        cs->bh = (uint16_t *)calloc((size_t)(cs->k * cs->n), 2);
        lcg_fill_f16(cs->ah, (size_t)(cs->m * cs->k), seed);
        lcg_fill_f16(cs->bh, (size_t)(cs->k * cs->n), seed ^ 0x9e3779b9u);
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
    if (rvcl_dispatch_matmul(&cs->d, &g->r) != RVCL_SUCCESS || !g->r)
        return 0;
    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(g->r, &num_ws);
    for (int i = 0; i < num_ws && i < RVCL_MAX_WORKSPACES; i++) {
        size_t align = ws[i].alignment > 1 ? (size_t)ws[i].alignment : 8;
        if (posix_memalign(&g->ws_buf[i], align,
                    (size_t)(ws[i].size + align)) != 0)
            return 0;
        rvcl_tensor_desc_t wd = {.ndims = 1,
            .dims = {ws[i].size + 1},
            .strides = {1},
            .dtype = RVCL_DTYPE_UINT8};
        rvcl_tensor_init(&g->ws_t[i], &wd, g->ws_buf[i]);
    }
    g->num_ws = num_ws;

    rvcl_tensor_desc_t td = {.ndims = 2,
        .dims = {cs->m, cs->k},
        .strides = {cs->k, 1},
        .dtype = cs->d.dtype_a};
    rvcl_tensor_init(&g->ta, &td,
            cs->d.dtype_a == RVCL_DTYPE_FP16 ? (void *)cs->ah
                                             : (void *)cs->a);
    td.dims[0] = cs->k;
    td.dims[1] = cs->n;
    td.strides[0] = cs->n;
    td.dtype = cs->d.dtype_b;
    rvcl_tensor_init(&g->tb, &td,
            cs->d.dtype_b == RVCL_DTYPE_FP16 ? (void *)cs->bh
                                             : (void *)cs->b);
    td.dims[0] = cs->m;
    td.strides[0] = cs->n;
    td.dtype = cs->d.dtype_c;
    rvcl_tensor_init(&g->tc, &td, cs->c);
    rvcl_tensor_desc_t bd = {.ndims = 1,
        .dims = {cs->n},
        .strides = {1},
        .dtype = RVCL_DTYPE_FP32};
    rvcl_tensor_init(&g->tbias, &bd, cs->bias);

    rvcl_tensor_pack_init(&g->pack);
    rvcl_tensor_pack_add(&g->pack, RVCL_TENSOR_SRC_A, &g->ta);
    rvcl_tensor_pack_add(&g->pack, RVCL_TENSOR_SRC_B, &g->tb);
    rvcl_tensor_pack_add_out(&g->pack, RVCL_TENSOR_DST, &g->tc);
    rvcl_tensor_pack_add(&g->pack, RVCL_TENSOR_BIAS, &g->tbias);
    for (int i = 0; i < num_ws && i < RVCL_MAX_WORKSPACES; i++)
        rvcl_tensor_pack_add(&g->pack,
                (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + i), &g->ws_t[i]);
    g->full = rvcl_work_range_full(cs->m, cs->n, cs->k);
    return 1;
}

static void rig_release(dispatch_rig_t *g) {
    for (int i = 0; i < RVCL_MAX_WORKSPACES; i++) free(g->ws_buf[i]);
    rvcl_dispatch_release(g->r);
}

int main(int argc, char **argv) {
    int json = 0, reps = 5, path_c = 0, mode_given = 0;
    bench_mode_t mode = MODE_P;
    const char *batch = NULL;
    static const int64_t defaults[][3] = {{64, 64, 64}, {128, 128, 128},
            {256, 256, 256}, {128, 256, 512}};
    int64_t shapes[16][3];
    int num_shapes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (!strncmp(argv[i], "--mode=", 7) && argv[i][7]) {
            const char *m = argv[i] + 7;
            if (!strcmp(m, "C")) mode = MODE_C;
            else if (!strcmp(m, "P")) mode = MODE_P;
            else if (!strcmp(m, "S")) mode = MODE_S;
            else {
                fprintf(stderr, "error: bad --mode '%s' (want C, P, S)\n",
                        m);
                return 2;
            }
            mode_given = 1;
        } else if (!strcmp(argv[i], "--path=c")) path_c = 1;
        else if (!strncmp(argv[i], "--batch=", 8) && argv[i][8])
            batch = argv[i] + 8;
        else if (num_shapes < 16) {
            long long m, n, k;
            if (sscanf(argv[i], "%lldx%lldx%lld", &m, &n, &k) == 3
                    && m > 0 && n > 0 && k > 0) {
                shapes[num_shapes][0] = m;
                shapes[num_shapes][1] = n;
                shapes[num_shapes][2] = k;
                num_shapes++;
            } else {
                fprintf(stderr, "error: bad argument '%s'\n"
                        "usage: %s [--json] [--reps N] [--mode=C|P|S] "
                        "[--path=c] [--batch=FILE] [MxNxK ...]\n",
                        argv[i], argv[0]);
                return 2;
            }
        } else {
            fprintf(stderr, "error: too many shape arguments\n");
            return 2;
        }
    }
    if (batch && num_shapes) {
        fprintf(stderr, "error: --batch and shape arguments are mutually "
                        "exclusive\n");
        return 2;
    }

    bcorpus_t corpus = {};
    char op[32] = "matmul";
    if (batch) {
        bparse_corpus(batch, &corpus);
        op_from_batch_path(batch, op, sizeof(op));
        /* file header --mode is the default; an explicit CLI --mode wins */
        if (!mode_given && corpus.file_mode)
            mode = corpus.file_mode == 'C' ? MODE_C
                    : corpus.file_mode == 'S' ? MODE_S : MODE_P;
    } else if (num_shapes == 0) {
        num_shapes = (int)(sizeof(defaults) / sizeof(defaults[0]));
        for (int i = 0; i < num_shapes; i++)
            for (int j = 0; j < 3; j++) shapes[i][j] = defaults[i][j];
    }
    if (mode == MODE_S) reps = 1;
    if (reps < 1) reps = 1;

    if (rvcl_init() != RVCL_SUCCESS) return 1;

    const int total = batch ? corpus.count : num_shapes;
    const char mode_ch = mode == MODE_C ? 'C' : mode == MODE_S ? 'S' : 'P';
    if (json) {
        printf("{\"_meta\":{\"schema\":1,\"rvcl\":\"%s\","
               "\"capability\":\"%s\",\"mode\":\"%c\",\"reps\":%d,"
               "\"batch\":%s%s%s}}\n",
                rvcl_get_version(), rvcl_get_capability()->description,
                mode_ch, reps, batch ? "\"" : "null",
                batch ? batch : "", batch ? "\"" : "");
    } else {
        printf("RVCL %s bench (mode %c, reps %d) — %s\n", rvcl_get_version(),
                mode_ch, reps, rvcl_get_capability()->description);
        printf("%-16s %-5s %-24s %10s %10s %10s %10s\n", "shape", "dtype",
                "kernel", "k-ms", "pack", "p+k", "e2e");
    }

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
        snprintf(shape_s, sizeof(shape_s), "%lldx%lldx%lld",
                (long long)cs.m, (long long)cs.n, (long long)cs.k);
        if (!cs.label) cs.label = shape_s;
        char label_s[96];
        snprintf(label_s, sizeof(label_s), "%s", cs.label);

        for (int dt = 0; dt < 2; dt++) { /* 0: f32, 1: f16 */
            if (corpus.dt_fixed
                    && (dt ? RVCL_DTYPE_FP16 : RVCL_DTYPE_FP32) != corpus.dt)
                continue;
            const char *dtstr = dt ? "f16" : "f32";

            cs.d = rvcl_matmul_desc_t{
                .m = cs.m, .n = cs.n, .k = cs.k,
                .dtype_a = dt ? RVCL_DTYPE_FP16 : RVCL_DTYPE_FP32,
                .dtype_b = dt ? RVCL_DTYPE_FP16 : RVCL_DTYPE_FP32,
                .dtype_acc = RVCL_DTYPE_AUTO,
                .dtype_c = RVCL_DTYPE_FP32,
                .dtype_bias = RVCL_DTYPE_AUTO,
                .quant = {},
            };
            fill_case(&cs, dt, mode);
            const double flops = 2.0 * cs.m * cs.n * cs.k;
            dispatch_rig_t rig;

            if (mode == MODE_C) {
                /* correctness: dispatch -> run -> shared golden; a
                 * dispatch failure is a case failure here */
                int ok = rig_setup(&rig, &cs);
                const char *detail = "";
                char dbuf[128];
                if (!ok) {
                    snprintf(dbuf, sizeof(dbuf), "dispatch failed");
                    detail = dbuf;
                } else {
                    golden_sentinel(cs.c, (size_t)(cs.m * cs.n) * 4);
                    run_ctx_t rc = {.k = rvcl_dispatch_kernel(rig.r),
                            .pack = &rig.pack,
                            .range = &rig.full};
                    rvcl_status_t st = rvcl_kernel_run(rc.k, rc.pack,
                            rc.range);
                    if (st != RVCL_SUCCESS) {
                        snprintf(dbuf, sizeof(dbuf), "run status %d",
                                (int)st);
                        detail = dbuf;
                        ok = 0;
                    } else {
                        float *golden = (float *)malloc(
                                (size_t)(cs.m * cs.n) * 4);
                        if (dt)
                            golden_f16(cs.ah, cs.bh, cs.bias, golden,
                                    cs.m, cs.n, cs.k);
                        else
                            golden_f32(cs.a, cs.b, cs.bias, golden,
                                    cs.m, cs.n, cs.k);
                        golden_cmp_t cmp = dt
                                ? golden_cmp_tol(cs.c, golden,
                                        cs.m * cs.n, 1e-3f)
                                : golden_cmp_bitexact(cs.c, golden,
                                        cs.m * cs.n);
                        free(golden);
                        if (cmp.mismatches) {
                            snprintf(dbuf, sizeof(dbuf),
                                    "first mismatch (i=%lld,j=%lld) "
                                    "exp=%g got=%g",
                                    (long long)(cmp.first / cs.n),
                                    (long long)(cmp.first % cs.n),
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
                           "\"status\":\"%s\"%s%s%s%s}\n",
                            op, shape_s, label_s, dtstr,
                            ok ? "\"" : "null",
                            ok ? rvcl_dispatch_kernel_name(rig.r) : "",
                            ok ? "\"" : "", ok ? "ok" : "fail",
                            *detail ? ",\"detail\":" : "",
                            *detail ? "\"" : "", detail,
                            *detail ? "\"" : "");
                } else {
                    printf("%-16s %-5s %-24s C: %s%s%s\n", shape_s, dtstr,
                            ok ? rvcl_dispatch_kernel_name(rig.r) : "-",
                            ok ? "ok" : "FAIL", *detail ? " — " : "",
                            detail);
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
                return 1;
            }
            rvcl_tensor_t b_src, b_dst;
            {
                rvcl_tensor_desc_t td = {.ndims = 2,
                    .dims = {cs.k, cs.n},
                    .strides = {cs.n, 1},
                    .dtype = cs.d.dtype_b};
                rvcl_tensor_init(&b_src, &td,
                        dt ? (void *)cs.bh : (void *)cs.b);
                rvcl_tensor_desc_t pd = {.ndims = 1,
                    .dims = {(int64_t)pbytes},
                    .strides = {1},
                    .dtype = RVCL_DTYPE_UINT8};
                rvcl_tensor_init(&b_dst, &pd, packed);
                rvv_pack_b_panel(&b_src, &b_dst, cs.m, cs.n, cs.k, elt);
            }

            konly_ctx_t kc = {.cs = &cs, .packed = packed, .gemm = NULL};
            /* kernel-only goes through the same selector the backend
             * binds (rvv_pick_*); --path=c forces the C micro-kernel */
            kc.gemm = path_c ? (dt ? rvv_gemm_f16_c : rvv_gemm_f32_c)
                             : (dt ? rvv_pick_gemm_f16()
                                   : rvv_pick_gemm_f32());
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
                t_pk = mode == MODE_S ? time_once(bench_pack_kernel, &rc)
                                      : time_best_of(bench_pack_kernel,
                                              &rc, reps);
                e2e_ctx_t ec = {.cs = &cs};
                t_e2e = mode == MODE_S ? time_once(bench_e2e, &ec)
                                       : time_best_of(bench_e2e, &ec, reps);
            }
            const char *kname = has_dispatch
                    ? rvcl_dispatch_kernel_name(rig.r) : NULL;

            if (json) {
                char b1[32], b2[32], b3[32], b4[32], b5[32];
                snprintf(b5, sizeof(b5), "%.2f",
                        t_kernel > 0 ? flops / (t_kernel * 1e6) : 0.0);
                printf("{\"op\":\"%s\",\"shape\":\"%s\",\"label\":\"%s\","
                       "\"dtype\":\"%s\",\"mode\":\"%c\",\"kernel\":%s%s%s,"
                       "\"kernel_ms\":%.4f,\"pack_ms\":%s,"
                       "\"pack_kernel_ms\":%s,\"e2e_ms\":%s,"
                       "\"gflops_kernel\":%s,\"gflops_e2e\":%s,"
                       "\"status\":\"%s\"}\n",
                        op, shape_s, label_s, dtstr, mode_ch,
                        kname ? "\"" : "null", kname ? kname : "",
                        kname ? "\"" : "",
                        t_kernel, jms(b1, sizeof(b1), t_pack, 1),
                        jms(b2, sizeof(b2), t_pk, has_dispatch),
                        jms(b3, sizeof(b3), t_e2e, has_dispatch), b5,
                        jms(b4, sizeof(b4),
                                t_e2e > 0 ? flops / (t_e2e * 1e6) : 0.0,
                                has_dispatch),
                        has_dispatch ? "ok" : "nodispatch");
            } else {
                char sh[32];
                snprintf(sh, sizeof(sh), "%lldx%lldx%lld",
                        (long long)cs.m, (long long)cs.n, (long long)cs.k);
                printf("%-16s %-5s %-24.24s %10.3f %10.3f %10.3f %10.3f\n",
                        sh, dtstr, kname ? kname : "-", t_kernel, t_pack,
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
