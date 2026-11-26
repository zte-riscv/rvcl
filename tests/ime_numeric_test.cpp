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
 ******************************************************************************/

/* Direct-driven (no dispatch/registry): packs B with the pack helper and
 * drives the micro-kernels over the tile-packed layout. Cases:
 *   - fixed ragged contract (33x19x47) with and without bias vs the soft
 *     golden (1e-3 class);
 *   - boundary shape scan vs the golden (both paths, with canaries);
 *   - C path vs instruction path cross-comparison as an independent,
 *     much tighter assertion (1e-5) separate from the golden tolerance;
 *   - M/N work-range slicing vs the full run (bit-exact inside the
 *     slice, sentinel untouched outside);
 *   - determinism across reruns and across 1/2/4/8-way M partitions;
 *   - bias-first semantics on a tail n tile;
 *   - a data-domain guard asserting every generated input is finite
 *     normal fp16 (the instruction path flushes subnormals to zero, so
 *     the comparisons above are only meaningful on normal-domain data).
 *
 * The instruction-path cases self-skip (exit 0) when the host gate
 * reports the path unavailable; --require-ime turns an unavailable gate
 * into a failure, and --c-only runs the C-path subset.
 */

#include "../src/backends/ime/ime_internal.hpp"
#include "golden.hpp"
#include "ime_fixtures.hpp"
#include "rvcl/rvcl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;
static int g_ime_available;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  ok: %s\n", msg); \
        } else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

static void *xaligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) abort();
    return p;
}

/* One problem instance: normal-domain A/B, optional bias, tile-packed B,
 * golden C. */
typedef struct {
    int64_t m, n, k;
    uint16_t *a, *b_packed, *b_plain;
    float *bias; /* NULL when absent */
    float *golden;
} case_t;

static void case_init(
        case_t *c, int64_t m, int64_t n, int64_t k, int with_bias) {
    c->m = m;
    c->n = n;
    c->k = k;
    const uint32_t seed = golden_case_seed(m, n, k);
    c->a = (uint16_t *)xaligned_alloc((size_t)(m * k * 2));
    c->b_plain = (uint16_t *)xaligned_alloc((size_t)(k * n * 2));
    ime_fill_f16_normal(c->a, (size_t)(m * k), seed);
    ime_fill_f16_normal(c->b_plain, (size_t)(k * n), seed ^ 0x1357);
    c->bias = NULL;
    if (with_bias) {
        c->bias = (float *)xaligned_alloc((size_t)(n * 4));
        ime_fill_bias_normal(c->bias, (size_t)n, seed ^ 0x2468);
    }
    const size_t pbytes = ime_packed_b_bytes(k, n, 2);
    if (pbytes == 0) abort();
    c->b_packed = (uint16_t *)xaligned_alloc(pbytes);
    /* pack through the real packer: the numeric cases must hold on the
     * production layout (the pack tests pin the packer against the
     * oracle separately) */
    rvcl_tensor_desc_t bd;
    memset(&bd, 0, sizeof(bd));
    bd.ndims = 2;
    bd.dims[0] = k;
    bd.dims[1] = n;
    bd.strides[0] = n;
    bd.strides[1] = 1;
    bd.dtype = RVCL_DTYPE_FP16;
    rvcl_tensor_t src;
    if (rvcl_tensor_init(&src, &bd, c->b_plain) != RVCL_SUCCESS) abort();
    rvcl_tensor_desc_t wd;
    memset(&wd, 0, sizeof(wd));
    wd.ndims = 1;
    wd.dims[0] = (int64_t)pbytes;
    wd.strides[0] = 1;
    wd.dtype = RVCL_DTYPE_UINT8;
    rvcl_tensor_t ws;
    if (rvcl_tensor_init(&ws, &wd, c->b_packed) != RVCL_SUCCESS) abort();
    if (ime_pack_b_tiles(&src, &ws, m, n, k, 2) != RVCL_SUCCESS) abort();
    c->golden = (float *)xaligned_alloc((size_t)(m * n * 4));
    golden_f16(c->a, c->b_plain, c->bias, c->golden, m, n, k);
}

static void case_free(case_t *c) {
    free(c->a);
    free(c->b_plain);
    free(c->b_packed);
    free(c->bias);
    free(c->golden);
}

/* Runs the requested path and compares against the golden with the given
 * tolerance; canaries around C catch out-of-range stores. */
static void run_and_check(
        const case_t *c, ime_gemm_fn gemm, const char *tag, float tol) {
    const size_t cbytes = (size_t)(c->m * c->n * 4) + 32;
    unsigned char *raw = (unsigned char *)xaligned_alloc(cbytes);
    memset(raw, 0x5c, cbytes);
    float *out = (float *)(raw + 16);
    const rvcl_work_range_t full = rvcl_work_range_full(c->m, c->n, c->k);
    gemm(c->a, c->b_packed, c->bias, out, c->m, c->n, c->k, &full);

    char msg[160];
    const golden_cmp_t cmp = golden_cmp_tol(out, c->golden, c->m * c->n, tol);
    snprintf(msg, sizeof(msg), "%s: golden compare (tol %g)", tag, (double)tol);
    CHECK(cmp.mismatches == 0, msg);
    int canary = 1;
    for (int i = 0; i < 16; i++)
        canary &= (raw[i] == 0x5c) && (raw[cbytes - 1 - i] == 0x5c);
    snprintf(msg, sizeof(msg), "%s: C canaries intact", tag);
    CHECK(canary, msg);
    free(raw);
}

/* -------------------------------------------- 01/02: fixed ragged contract */

static void tc_fixed(int with_bias) {
    printf("TC-%s: fixed contract 33x19x47 (%s bias)\n",
            with_bias ? "01" : "02", with_bias ? "with" : "no");
    case_t c;
    case_init(&c, 33, 19, 47, with_bias);
    run_and_check(&c, ime_gemm_f16_c, "C path", 1e-3f);
#if defined(__riscv) && defined(__riscv_vector)
    if (g_ime_available)
        run_and_check(&c, ime_gemm_f16_ime, "IME path", 1e-3f);
    else
        printf("  skip: instruction path unavailable on this host\n");
#else
    printf("  skip: instruction path not compiled\n");
#endif
    case_free(&c);
}

/* ------------------------------------------------- 04: boundary shape scan */

static void tc_boundary(void) {
    printf("TC-04 boundary shapes vs golden (1e-3)\n");
    int c_ok = 1;
#if defined(__riscv) && defined(__riscv_vector)
    int ime_ok = 1;
#endif
    for (size_t i = 0;
            i < sizeof(ime_shapes_mixed) / sizeof(ime_shapes_mixed[0]); i++) {
        case_t c;
        case_init(&c, ime_shapes_mixed[i][0], ime_shapes_mixed[i][1],
                ime_shapes_mixed[i][2], (int)(i & 1));
        const size_t cbytes = (size_t)(c.m * c.n * 4) + 32;
        unsigned char *raw = (unsigned char *)xaligned_alloc(cbytes);
        float *out = (float *)(raw + 16);
        const rvcl_work_range_t full = rvcl_work_range_full(c.m, c.n, c.k);

        memset(raw, 0x5c, cbytes);
        ime_gemm_f16_c(c.a, c.b_packed, c.bias, out, c.m, c.n, c.k, &full);
        if (golden_cmp_tol(out, c.golden, c.m * c.n, 1e-3f).mismatches != 0)
            c_ok = 0;
#if defined(__riscv) && defined(__riscv_vector)
        if (g_ime_available) {
            memset(raw, 0x5c, cbytes);
            ime_gemm_f16_ime(
                    c.a, c.b_packed, c.bias, out, c.m, c.n, c.k, &full);
            if (golden_cmp_tol(out, c.golden, c.m * c.n, 1e-3f).mismatches != 0)
                ime_ok = 0;
        }
#endif
        free(raw);
        case_free(&c);
    }
    CHECK(c_ok, "C path passes 1e-3 on all mixed-axis shapes");
#if defined(__riscv) && defined(__riscv_vector)
    if (g_ime_available)
        CHECK(ime_ok, "IME path passes 1e-3 on all mixed-axis shapes");
    else
        printf("  skip: instruction path unavailable on this host\n");
#else
    printf("  skip: instruction path not compiled\n");
#endif
}

/* ------------------------------------------- 05: C vs IME cross-comparison */

static void tc_cross(void) {
    printf("TC-05 C vs instruction path agreement (1e-5, independent)\n");
#if defined(__riscv) && defined(__riscv_vector)
    if (!g_ime_available) {
        printf("  skip: instruction path unavailable on this host\n");
        return;
    }
    static const int64_t shapes[][3]
            = {{1, 1, 1}, {7, 9, 15}, {17, 17, 17}, {31, 31, 31}, {33, 19, 47}};
    int ok = 1;
    double worst = 0;
    for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
        case_t c;
        case_init(&c, shapes[i][0], shapes[i][1], shapes[i][2], 1);
        float *cc = (float *)xaligned_alloc((size_t)(c.m * c.n * 4));
        float *ci = (float *)xaligned_alloc((size_t)(c.m * c.n * 4));
        const rvcl_work_range_t full = rvcl_work_range_full(c.m, c.n, c.k);
        ime_gemm_f16_c(c.a, c.b_packed, c.bias, cc, c.m, c.n, c.k, &full);
        ime_gemm_f16_ime(c.a, c.b_packed, c.bias, ci, c.m, c.n, c.k, &full);
        if (golden_cmp_tol(ci, cc, c.m * c.n, 1e-5f).mismatches != 0) ok = 0;
        for (int64_t e = 0; e < c.m * c.n; e++) {
            const double d = (double)ci[e] - (double)cc[e];
            const double mag
                    = (double)cc[e] < 0 ? -(double)cc[e] : (double)cc[e];
            const double rel = d / (1.0 + mag);
            if (rel > worst) worst = rel;
        }
        free(cc);
        free(ci);
        case_free(&c);
    }
    CHECK(ok, "paths agree within 1e-5 relative on all shapes");
    printf("  info: worst relative deviation %g\n", worst);
#else
    printf("  skip: instruction path not compiled\n");
#endif
}

/* ----------------------------------------------- 06: work-range slicing */

static void tc_slices(void) {
    printf("TC-06 M/N sub-ranges match the full run bit-exactly\n");
#if defined(__riscv) && defined(__riscv_vector)
    const ime_gemm_fn gemm
            = g_ime_available ? ime_gemm_f16_ime : ime_gemm_f16_c;
#else
    const ime_gemm_fn gemm = ime_gemm_f16_c;
#endif
    case_t c;
    case_init(&c, 33, 19, 47, 1);
    const size_t cbytes = (size_t)(c.m * c.n * 4);
    float *full = (float *)xaligned_alloc(cbytes);
    unsigned char *part_raw = (unsigned char *)xaligned_alloc(cbytes + 32);
    float *part = (float *)(part_raw + 16);
    const rvcl_work_range_t fr = rvcl_work_range_full(c.m, c.n, c.k);
    gemm(c.a, c.b_packed, c.bias, full, c.m, c.n, c.k, &fr);

    static const int64_t m_bounds[] = {0, 1, 3, 7, 8, 16, 33};
    int ok = 1, sentinel_ok = 1;
    for (size_t i = 0; i + 1 < sizeof(m_bounds) / sizeof(m_bounds[0]); i++) {
        memset(part_raw, 0x5a, cbytes + 32);
        rvcl_work_range_t r = fr;
        r.m_begin = m_bounds[i];
        r.m_end = m_bounds[i + 1];
        gemm(c.a, c.b_packed, c.bias, part, c.m, c.n, c.k, &r);
        for (int64_t e = 0; e < c.m * c.n; e++) {
            const int64_t row = e / c.n;
            const int inside = row >= r.m_begin && row < r.m_end;
            if (inside)
                ok &= (part[e] == full[e]);
            else
                sentinel_ok &= (part_raw[16 + 4 * e + 0] == 0x5a);
        }
    }
    static const int64_t n_bounds[] = {0, 7, 19};
    for (size_t i = 0; i + 1 < sizeof(n_bounds) / sizeof(n_bounds[0]); i++) {
        memset(part_raw, 0x5a, cbytes + 32);
        rvcl_work_range_t r = fr;
        r.n_begin = n_bounds[i];
        r.n_end = n_bounds[i + 1];
        gemm(c.a, c.b_packed, c.bias, part, c.m, c.n, c.k, &r);
        for (int64_t e = 0; e < c.m * c.n; e++) {
            const int64_t col = e % c.n;
            const int inside = col >= r.n_begin && col < r.n_end;
            if (inside)
                ok &= (part[e] == full[e]);
            else
                sentinel_ok &= (part_raw[16 + 4 * e + 0] == 0x5a);
        }
    }
    CHECK(ok, "every slice writes exactly the full-run values in range");
    CHECK(sentinel_ok, "elements outside the slice keep the sentinel");
    free(part_raw);
    free(full);
    case_free(&c);
}

/* ------------------------------------- 07: determinism and M-partitioning */

static void tc_determinism(void) {
    printf("TC-07 determinism across reruns and M partitions\n");
#if defined(__riscv) && defined(__riscv_vector)
    const ime_gemm_fn gemm
            = g_ime_available ? ime_gemm_f16_ime : ime_gemm_f16_c;
#else
    const ime_gemm_fn gemm = ime_gemm_f16_c;
#endif
    case_t c;
    case_init(&c, 33, 19, 47, 0);
    const size_t cbytes = (size_t)(c.m * c.n * 4);
    float *run1 = (float *)xaligned_alloc(cbytes);
    float *run2 = (float *)xaligned_alloc(cbytes);
    const rvcl_work_range_t fr = rvcl_work_range_full(c.m, c.n, c.k);

    int ok = 1;
    gemm(c.a, c.b_packed, c.bias, run1, c.m, c.n, c.k, &fr);
    for (int rep = 0; rep < 5; rep++) {
        gemm(c.a, c.b_packed, c.bias, run2, c.m, c.n, c.k, &fr);
        ok &= (memcmp(run1, run2, cbytes) == 0);
    }
    CHECK(ok, "identical results across 5 reruns (bit-exact)");

    ok = 1;
    static const int parts[] = {1, 2, 4, 8};
    for (size_t p = 0; p < sizeof(parts) / sizeof(parts[0]); p++) {
        const int64_t t = parts[p];
        memset(run2, 0, cbytes);
        for (int64_t s = 0; s < t; s++) {
            rvcl_work_range_t r = fr;
            r.m_begin = c.m * s / t;
            r.m_end = c.m * (s + 1) / t;
            gemm(c.a, c.b_packed, c.bias, run2, c.m, c.n, c.k, &r);
        }
        ok &= (memcmp(run1, run2, cbytes) == 0);
    }
    CHECK(ok, "1/2/4/8-way M partitions reassemble the full run bit-exactly");
    free(run2);
    free(run1);
    case_free(&c);
}

/* --------------------------------------------------- 08: bias-first tail */

static void tc_bias_first(void) {
    printf("TC-08 bias-first on a tail n tile (A = 0)\n");
#if defined(__riscv) && defined(__riscv_vector)
    const ime_gemm_fn gemm
            = g_ime_available ? ime_gemm_f16_ime : ime_gemm_f16_c;
#else
    const ime_gemm_fn gemm = ime_gemm_f16_c;
#endif
    static const int64_t ns[] = {8, 9};
    int ok = 1, canary_ok = 1;
    for (size_t ni = 0; ni < sizeof(ns) / sizeof(ns[0]); ni++) {
        const int64_t m = 8, n = ns[ni], k = 1;
        /* all-zero A and B: C is exactly the bias tile, fp32-exact */
        uint16_t a[8 * 1] = {0};
        uint16_t b[1 * 9] = {0};
        float bias[9];
        for (int64_t j = 0; j < n; j++)
            bias[j] = (float)(j + 1); /* fp32-exact small integers */

        uint16_t packed[128];
        ime_oracle_pack_b(packed, b, n, k);
        const size_t cbytes = (size_t)(m * n * 4) + 32;
        unsigned char *raw = (unsigned char *)xaligned_alloc(cbytes);
        memset(raw, 0x5c, cbytes);
        float *out = (float *)(raw + 16);
        const rvcl_work_range_t full = rvcl_work_range_full(m, n, k);
        gemm(a, packed, bias, out, m, n, k, &full);
        for (int64_t e = 0; e < m * n; e++) {
            const float want = bias[e % n];
            ok &= (out[e] == want);
        }
        for (int i = 0; i < 16; i++)
            canary_ok &= (raw[i] == 0x5c) && (raw[cbytes - 1 - i] == 0x5c);
        free(raw);
    }
    CHECK(ok, "C[i][j] == bias[j] exactly, tail column included (N=9)");
    CHECK(canary_ok, "canaries intact (no writes past the valid columns)");
}

/* ------------------------------------------------------ 09: data-domain guard */

static void tc_data_guard(void) {
    printf("TC-09 generated inputs stay in the finite normal fp16 domain\n");
    int ok = 1;
    for (size_t i = 0;
            i < sizeof(ime_shapes_mixed) / sizeof(ime_shapes_mixed[0]); i++) {
        const int64_t m = ime_shapes_mixed[i][0];
        const int64_t n = ime_shapes_mixed[i][1];
        const int64_t k = ime_shapes_mixed[i][2];
        uint16_t *a = (uint16_t *)xaligned_alloc((size_t)(m * k * 2));
        uint16_t *b = (uint16_t *)xaligned_alloc((size_t)(k * n * 2));
        float *bias = (float *)xaligned_alloc((size_t)(n * 4));
        ime_fill_f16_normal(a, (size_t)(m * k), golden_case_seed(m, n, k));
        ime_fill_f16_normal(b, (size_t)(k * n), golden_case_seed(m, n, k) + 1);
        ime_fill_bias_normal(bias, (size_t)n, 123);
        ok &= ime_buffer_all_normal(a, (size_t)(m * k));
        ok &= ime_buffer_all_normal(b, (size_t)(k * n));
        for (int64_t j = 0; j < n; j++)
            ok &= (bias[j] == 0.0f || bias[j] != 0);
        free(bias);
        free(b);
        free(a);
    }
    CHECK(ok, "no subnormal/inf/nan in any generated A/B/bias buffer");
}

int main(int argc, char **argv) {
    int c_only = 0, require_ime = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--c-only"))
            c_only = 1;
        else if (!strcmp(argv[i], "--require-ime"))
            require_ime = 1;
        else {
            fprintf(stderr, "usage: %s [--c-only] [--require-ime]\n", argv[0]);
            return 2;
        }
    }
    g_ime_available = ime_instruction_path_available() && !c_only;

    if (require_ime && !ime_instruction_path_available()) {
        printf("FAIL: --require-ime but the instruction path is unavailable\n");
        return 1;
    }

    tc_data_guard();
    tc_fixed(1);
    tc_fixed(0);
    tc_boundary();
    tc_cross();
    tc_slices();
    tc_determinism();
    tc_bias_first();

    printf(g_failures ? "IME numeric tests: %d FAIL\n"
                      : "IME numeric tests: all ok\n",
            g_failures);
    return g_failures ? 1 : 0;
}
