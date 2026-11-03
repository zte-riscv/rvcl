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
 * tests/scheduler_omp_test.cpp — built-in OpenMP scheduler backend tests
 *
 * Registered as its own ctest process (OpenMP builds only) so the
 * lazy-default and nested-guard assertions run before any set-once
 * injection (guarantee 3). Covers:
 *   - lazy default: not injected -> rvcl_scheduler_get() is "omp";
 *     explicit serial injection later wins (last write wins, set-once).
 *   - nested guard: parallel_for inside an active omp region inlines serial.
 *   - slice coverage: diff-1 balanced M split, no overlap / no gap, rows % T
 *     != 0 case.
 *   - multithread bit-exact: rvcl_matmul vs golden on ragged shapes, T=4.
 *   - T scan: omp_set_num_threads(1/2/4/8) all agree bit-exact with golden
 *     (M split never reorders K accumulation, guarantee 5).
 ******************************************************************************/

#include <omp.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <algorithm>
#include <vector>

#include "golden.hpp"
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

/* ------------------------------------------------ nested-guard probe */

static int g_nest_calls;
static int g_nest_full;

static void nest_probe(
        rvcl_ischeduler_t *s, const rvcl_work_range_t *r, void *ctx) {
    (void)s;
    (void)ctx;
    g_nest_calls++;
    if (r->m_begin == 0 && r->m_end == 8) g_nest_full = 1;
}

/* ------------------------------------------------ slice recorder */

#define MAX_SLICES 32
typedef struct {
    rvcl_work_range_t items[MAX_SLICES];
    int count;
    pthread_mutex_t mu;
} slice_recorder_t;

static slice_recorder_t g_rec;

static void record_task(
        rvcl_ischeduler_t *s, const rvcl_work_range_t *r, void *ctx) {
    (void)s;
    (void)ctx;
    pthread_mutex_lock(&g_rec.mu);
    if (g_rec.count < MAX_SLICES) g_rec.items[g_rec.count++] = *r;
    pthread_mutex_unlock(&g_rec.mu);
}

/* Verify diff-1 balanced coverage of [0, rows) with no overlap and no gap. */
static int check_slices(int64_t rows) {
    if (g_rec.count < 1) return 0;
    std::vector<rvcl_work_range_t> sl;
    for (int i = 0; i < g_rec.count; i++)
        sl.push_back(g_rec.items[i]);
    std::sort(sl.begin(), sl.end(),
            [](const rvcl_work_range_t &a, const rvcl_work_range_t &b) {
        return a.m_begin < b.m_begin;
    });
    int64_t covered = 0;
    int64_t prev = 0;
    int64_t min_rows = INT64_MAX, max_rows = 0;
    for (const auto &s : sl) {
        if (s.m_begin < prev) return 0; /* overlap */
        if (s.m_begin != prev) return 0; /* gap */
        const int64_t r = s.m_end - s.m_begin;
        if (r < min_rows) min_rows = r;
        if (r > max_rows) max_rows = r;
        covered += r;
        prev = s.m_end;
    }
    if (covered != rows) return 0;
    if (max_rows - min_rows > 1) return 0; /* not diff-1 balanced */
    return 1;
}

/* ------------------------------------------------- matmul bit-exact rig */

static rvcl_matmul_desc_t make_desc(int64_t M, int64_t N, int64_t K) {
    rvcl_matmul_desc_t d = {.m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {}};
    return d;
}

/* Run rvcl_matmul on a deterministic ragged shape; return bit-exact mismatch
 * count vs the naive golden (-1 on run failure), and store the result in
 * [out] when non-NULL. */
static int64_t run_check(
        int64_t M, int64_t N, int64_t K, std::vector<float> *out) {
    std::vector<float> a((size_t)M * K), b((size_t)K * N), bias((size_t)N),
            golden((size_t)M * N), c((size_t)M * N);
    lcg_fill_f32(a.data(), a.size(), golden_case_seed(M, N, K));
    lcg_fill_f32(b.data(), b.size(), golden_case_seed(N, K, M));
    for (int64_t j = 0; j < N; j++)
        bias[j] = 0.25f;
    golden_f32(a.data(), b.data(), bias.data(), golden.data(), M, N, K);

    rvcl_matmul_desc_t d = make_desc(M, N, K);
    rvcl_status_t st
            = rvcl_matmul(&d, a.data(), b.data(), bias.data(), c.data());
    if (st != RVCL_SUCCESS) return -1;

    int64_t bad = 0;
    for (int64_t i = 0; i < M * N; i++)
        if (c[i] != golden[i]) bad++;
    if (out) *out = std::move(c);
    return bad;
}

/* ------------------------------------------------------------------ main */

int main(void) {
    /* ---- 1. lazy default: not injected -> omp ---- */
    printf("lazy default\n");
    CHECK(rvcl_scheduler_get() != NULL, "scheduler resolves");
    CHECK(strcmp(rvcl_scheduler_get()->name, "omp") == 0,
            "uninjected default is the omp scheduler");

    /* ---- 2. nested guard: inline serial inside an active omp region ---- */
    printf("nested guard\n");
    g_nest_calls = 0;
    g_nest_full = 0;
#pragma omp parallel num_threads(4)
    {
        if (omp_get_thread_num() == 0) {
            rvcl_work_range_t full = rvcl_work_range_full(8, 8, 8);
            rvcl_parallel_for(&full, nest_probe, NULL);
        }
    }
    CHECK(g_nest_calls == 1, "no nested parallel region spawned");
    CHECK(g_nest_full == 1, "inline serial passes the whole range");

    /* ---- 3. slice coverage: diff-1 balanced, rows % T != 0 ---- */
    printf("slice coverage\n");
    {
        const int64_t rows = 10; /* 10 % 4 != 0 exercises diff-1 balancing */
        omp_set_num_threads(4);
        pthread_mutex_init(&g_rec.mu, NULL);
        g_rec.count = 0;
        rvcl_work_range_t total = rvcl_work_range_full(rows, 7, 5);
        rvcl_ischeduler_t *s = rvcl_omp_scheduler();
        s->parallel_for(s, &total, record_task, NULL);
        CHECK(g_rec.count == 4, "rows=10, T=4 -> 4 slices");
        CHECK(check_slices(rows), "diff-1 balanced, no overlap, no gap");
        pthread_mutex_destroy(&g_rec.mu);
    }

    /* ---- 4. multithread bit-exact vs golden, ragged shape, T=4 ---- */
    printf("multithread bit-exact\n");
    {
        omp_set_num_threads(4);
        const int64_t shapes[][3] = {{97, 65, 33}, {7, 7, 7}, {15, 15, 15},
                {23, 23, 23}, {25, 25, 25}};
        int ok = 1;
        for (size_t i = 0; i < sizeof(shapes) / sizeof(shapes[0]); i++) {
            const int64_t bad
                    = run_check(shapes[i][0], shapes[i][1], shapes[i][2], NULL);
            if (bad != 0) {
                fprintf(stderr, "  ragged %lldx%lldx%lld: %lld mismatches\n",
                        (long long)shapes[i][0], (long long)shapes[i][1],
                        (long long)shapes[i][2], (long long)bad);
                ok = 0;
            }
        }
        CHECK(ok, "ragged shapes bit-exact vs golden at T=4");
    }

    /* ---- 5. T scan: 1/2/4/8 mutually equal and bit-exact with golden ----
     * Mutual equality is the machine proof that M slicing never reorders K
     * accumulation (guarantee 5) — it is evaluated independently of the
     * golden comparison so a host-side kernel 1-ulp drift from the scalar
     * golden cannot mask a genuine T dependence. */
    printf("T scan\n");
    {
        const int scan[] = {1, 2, 4, 8};
        const int64_t M = 128, N = 96, K = 64;
        int ok_golden = 1, ok_mutual = 1;
        std::vector<float> ref;
        int have_ref = 0;
        for (size_t i = 0; i < sizeof(scan) / sizeof(scan[0]); i++) {
            omp_set_num_threads(scan[i]);
            std::vector<float> out;
            const int64_t bad = run_check(M, N, K, &out);
            if (bad != 0) {
                fprintf(stderr, "  T=%d: %lld mismatches vs golden\n", scan[i],
                        (long long)bad);
                ok_golden = 0;
            }
            if (have_ref) {
                for (size_t j = 0; j < out.size(); j++)
                    if (out[j] != ref[j]) {
                        fprintf(stderr,
                                "  T=%d drifts from first bit-exact result\n",
                                scan[i]);
                        ok_mutual = 0;
                        break;
                    }
            } else {
                ref = std::move(out);
                have_ref = 1;
            }
        }
        CHECK(ok_mutual, "T=1/2/4/8 outputs mutually equal");
        CHECK(ok_golden, "T=1/2/4/8 bit-exact vs golden");
    }

    /* ---- 6. explicit serial injection wins (last write wins, set-once) ---- */
    printf("explicit injection\n");
    CHECK(rvcl_scheduler_set(rvcl_serial_scheduler()) == RVCL_SUCCESS,
            "serial injection accepted after lazy default");
    CHECK(strcmp(rvcl_scheduler_get()->name, "serial") == 0,
            "explicit serial injection wins over lazy default");
    CHECK(rvcl_scheduler_set(rvcl_serial_scheduler())
                    == RVCL_ERROR_ALREADY_INITIALIZED,
            "second set rejected (set-once)");

    printf(g_failures ? "SCHEDULER_OMP: %d failure(s)\n"
                      : "SCHEDULER_OMP: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
