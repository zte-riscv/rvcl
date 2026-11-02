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
 * examples/matmul_multicore.cpp — multi-core matmul example (injectable pool)
 *
 * Demonstrates host-owned parallelism (PLAN §4.9) end to end: a resident
 * dynamic-fetching thread pool slices the M dimension and drives the same
 * rvcl_matmul() across a 1/2/4/8 thread scan. Correctness is backed by a
 * triple bit-exact comparison (vs element-wise golden, vs the T=1 result,
 * and ragged panel-boundary shapes); scaling is measured with a warmup +
 * best-of-5 timing table on the host only.
 *
 * CLI: example_multicore [mode] [max_threads] [M] [N] [K]
 *   mode:  pool (default) — inject this example's worker pool
 *          builtin — reserved; needs an OpenMP build (PR-4)
 *   default max_threads = nproc; default shape = 512x512x512.
 *   Timing is enabled with RVCL_MULTICORE_BENCH=1 (ctest never times).
 ******************************************************************************/

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <atomic>
#include <vector>

#include "rvcl/rvcl.h"

/* -------------------------------------------------- deterministic inputs */

static void fill_tensor(float *buf, int64_t n, int mul, int mod) {
    for (int64_t i = 0; i < n; i++)
        buf[i] = (float)((i * mul) % mod) / (float)mod - 0.5f;
}

/* bias-first triple-loop golden; matches the kernels' accumulation order. */
static void matmul_golden(const float *a, const float *b, const float *bias,
        float *c, int64_t M, int64_t N, int64_t K) {
    for (int64_t i = 0; i < M; i++)
        for (int64_t j = 0; j < N; j++) {
            float acc = bias[j];
            for (int64_t p = 0; p < K; p++)
                acc += a[i * K + p] * b[p * N + j];
            c[i * N + j] = acc;
        }
}

/* ---------------------------------------------------------- worker pool */

struct pool_t {
    std::atomic<int> active_threads; /* current T incl. main; pool-internal */
    std::atomic<int> next; /* next slice to grab (dynamic fetch) */
    std::atomic<int> done; /* completed slice count */
    std::atomic<int> num_tasks; /* slices in flight for this round */
    rvcl_work_range_t total;
    rvcl_parallel_task_fn fn;
    rvcl_ischeduler_t *sched;
    void *ctx;
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
    int max_threads;
};

static pool_t g_pool;

/* Diff-1 balanced slice boundaries for slice [idx] of [parts] over [total]. */
static void pool_slice(rvcl_work_range_t *out, const rvcl_work_range_t *total,
        int idx, int parts) {
    const int64_t rows = total->m_end - total->m_begin;
    const int64_t step = rows / parts;
    const int64_t rem = rows % parts;
    const int64_t begin
            = total->m_begin + (int64_t)idx * step + (idx < rem ? idx : rem);
    *out = *total; /* n/k intervals pass through unchanged */
    out->m_begin = begin;
    out->m_end = begin + step + (idx < rem ? 1 : 0);
}

/* Signal the merge barrier when the last slice of a round completes. */
static void pool_mark_done(void) {
    if (g_pool.done.fetch_add(1) + 1 == g_pool.num_tasks.load()) {
        pthread_mutex_lock(&g_pool.mu);
        pthread_cond_signal(&g_pool.cv_done);
        pthread_mutex_unlock(&g_pool.mu);
    }
}

/* Worker: sleep until a round needs it, then fetch slices dynamically. */
static void *pool_worker(void *arg) {
    const int wid = (int)(intptr_t)arg;
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        while (g_pool.num_tasks.load() == 0
                || wid >= g_pool.active_threads.load() - 1)
            pthread_cond_wait(&g_pool.cv_work, &g_pool.mu);
        pthread_mutex_unlock(&g_pool.mu);
        int idx;
        while ((idx = g_pool.next.fetch_add(1)) < g_pool.num_tasks.load()) {
            rvcl_work_range_t slice;
            pool_slice(&slice, &g_pool.total, idx, g_pool.num_tasks.load());
            g_pool.fn(g_pool.sched, &slice, g_pool.ctx);
            pool_mark_done();
        }
    }
    return NULL;
}

/* parallel_for: split M into min(T, rows) diff-1 slices, execute, merge. */
static void pool_parallel_for(rvcl_ischeduler_t *s,
        const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
        void *user_ctx) {
    const int64_t rows = total->m_end - total->m_begin;
    const int active = g_pool.active_threads.load();
    int parts = (int)(rows < (int64_t)active ? rows : active); /* small load */
    if (parts < 1) parts = 1;

    pthread_mutex_lock(&g_pool.mu);
    g_pool.num_tasks.store(parts);
    g_pool.total = *total;
    g_pool.fn = task;
    g_pool.sched = s;
    g_pool.ctx = user_ctx;
    g_pool.next.store(0);
    g_pool.done.store(0);
    pthread_cond_broadcast(&g_pool.cv_work);
    pthread_mutex_unlock(&g_pool.mu);

    /* main participates, sharing the same dynamic-fetch cursor */
    int idx;
    while ((idx = g_pool.next.fetch_add(1)) < parts) {
        rvcl_work_range_t slice;
        pool_slice(&slice, total, idx, parts);
        task(s, &slice, user_ctx);
        pool_mark_done();
    }

    /* blocking merge: wait until every slice (workers included) completed */
    pthread_mutex_lock(&g_pool.mu);
    while (g_pool.done.load() < parts)
        pthread_cond_wait(&g_pool.cv_done, &g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
}

static void pool_start(int max_threads) {
    g_pool.max_threads = max_threads < 1 ? 1 : max_threads;
    g_pool.active_threads.store(g_pool.max_threads);
    g_pool.next.store(0);
    g_pool.done.store(0);
    g_pool.num_tasks.store(0);
    pthread_mutex_init(&g_pool.mu, NULL);
    pthread_cond_init(&g_pool.cv_work, NULL);
    pthread_cond_init(&g_pool.cv_done, NULL);
    for (int i = 0; i < g_pool.max_threads - 1; i++) {
        pthread_t th;
        if (pthread_create(&th, NULL, pool_worker, (void *)(intptr_t)i) == 0)
            pthread_detach(th);
    }
}

/* T is pool-internal state: change it without re-injecting the scheduler. */
static void pool_set_threads(int t) {
    int v;
    if (t < 1)
        v = 1;
    else if (t > g_pool.max_threads)
        v = g_pool.max_threads;
    else
        v = t;
    g_pool.active_threads.store(v);
}

/* ----------------------------------------------------- correctness rig */

struct shape_t {
    int64_t m, n, k;
};

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

/* Fill inputs, compute golden, run rvcl_matmul; return #mismatches vs golden
 * (-1 on rvcl_matmul failure), and optionally store the result in [out]. */
static int64_t check_shape(
        int64_t M, int64_t N, int64_t K, std::vector<float> *out) {
    std::vector<float> a((size_t)M * K), b((size_t)K * N), bias((size_t)N),
            golden((size_t)M * N);
    fill_tensor(a.data(), M * K, 13, 17);
    fill_tensor(b.data(), K * N, 7, 19);
    for (int64_t j = 0; j < N; j++)
        bias[j] = 0.25f;
    matmul_golden(a.data(), b.data(), bias.data(), golden.data(), M, N, K);

    rvcl_matmul_desc_t d = make_desc(M, N, K);
    std::vector<float> c((size_t)M * N);
    rvcl_status_t st
            = rvcl_matmul(&d, a.data(), b.data(), bias.data(), c.data());
    if (st != RVCL_SUCCESS) return -1;

    int64_t bad = 0;
    for (int64_t i = 0; i < M * N; i++)
        if (c[i] != golden[i]) bad++;
    if (out) *out = std::move(c);
    return bad;
}

/* Triple bit-exact check across the T scan: vs golden, vs T=1, ragged. */
static int verify_correctness(const shape_t &primary, const shape_t *ragged,
        size_t n_ragged, const int *scan, int n_scan, int max_threads) {
    int ok = 1;
    std::vector<float> t1_primary;
    bool have_t1 = false;
    for (int si = 0; si < n_scan; si++) {
        const int t = scan[si];
        if (t > max_threads) continue;
        pool_set_threads(t);
        printf("T=%d\n", t);

        std::vector<float> pc;
        int64_t bad = check_shape(primary.m, primary.n, primary.k, &pc);
        if (bad < 0) {
            fprintf(stderr, "  primary run failed\n");
            ok = 0;
        } else {
            if (bad) {
                fprintf(stderr, "  primary %lldx%lldx%lld: %lld vs golden\n",
                        (long long)primary.m, (long long)primary.n,
                        (long long)primary.k, (long long)bad);
                ok = 0;
            } else {
                printf("  primary %lldx%lldx%lld bit-exact vs golden\n",
                        (long long)primary.m, (long long)primary.n,
                        (long long)primary.k);
            }
            if (have_t1) {
                int64_t drift = 0;
                for (size_t i = 0; i < pc.size(); i++)
                    if (pc[i] != t1_primary[i]) drift++;
                if (drift) {
                    fprintf(stderr, "  primary drifts from T=1: %lld\n",
                            (long long)drift);
                    ok = 0;
                } else {
                    printf("  primary matches T=1 result\n");
                }
            }
            if (!have_t1) {
                /* capture the first successful run as the T=1 reference */
                t1_primary = std::move(pc);
                have_t1 = true;
            }
        }

        for (size_t ri = 0; ri < n_ragged; ri++) {
            const shape_t &s = ragged[ri];
            int64_t rb = check_shape(s.m, s.n, s.k, NULL);
            if (rb < 0) {
                fprintf(stderr, "  ragged %lldx%lldx%lld run failed\n",
                        (long long)s.m, (long long)s.n, (long long)s.k);
                ok = 0;
            } else if (rb) {
                fprintf(stderr, "  ragged %lldx%lldx%lld: %lld mismatches\n",
                        (long long)s.m, (long long)s.n, (long long)s.k,
                        (long long)rb);
                ok = 0;
            }
        }
    }
    return ok;
}

/* --------------------------------------------------------- scaling table */

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* Warmup + best-of-5 timing per shape per T; speedup/efficiency vs T=1. */
static void bench_scan(const shape_t *shapes, size_t n_shapes, const int *scan,
        int n_scan, int max_threads) {
    printf("threads\tshape\t\ttime_ms\tspeedup\tefficiency\n");
    for (size_t i = 0; i < n_shapes; i++) {
        const shape_t &s = shapes[i];
        const int64_t M = s.m, N = s.n, K = s.k;
        std::vector<float> a((size_t)M * K), b((size_t)K * N), bias((size_t)N),
                c((size_t)M * N);
        fill_tensor(a.data(), M * K, 13, 17);
        fill_tensor(b.data(), K * N, 7, 19);
        for (int64_t j = 0; j < N; j++)
            bias[j] = 0.25f;
        rvcl_matmul_desc_t d = make_desc(M, N, K);

        double t1 = 0.0;
        for (int si = 0; si < n_scan; si++) {
            const int t = scan[si];
            if (t > max_threads) continue;
            pool_set_threads(t);
            rvcl_matmul(
                    &d, a.data(), b.data(), bias.data(), c.data()); /* warmup */
            double best = 1e30;
            for (int r = 0; r < 5; r++) {
                double t0 = now_sec();
                rvcl_matmul(&d, a.data(), b.data(), bias.data(), c.data());
                double dt = now_sec() - t0;
                if (dt < best) best = dt;
            }
            if (t == 1) t1 = best;
            const double speedup = t1 > 0.0 ? t1 / best : 0.0;
            const double eff = speedup / (double)t;
            printf("%d\t%lldx%lldx%lld\t%.3f\t%.2f\t%.2f\n", t, (long long)M,
                    (long long)N, (long long)K, best * 1e3, speedup, eff);
        }
    }
}

/* ------------------------------------------------------------------ main */

int main(int argc, char **argv) {
    const char *mode = argc > 1 ? argv[1] : "pool";
    int max_threads = argc > 2 ? atoi(argv[2]) : 0;
    int64_t M = argc > 3 ? atoll(argv[3]) : 512;
    int64_t N = argc > 4 ? atoll(argv[4]) : 512;
    int64_t K = argc > 5 ? atoll(argv[5]) : 512;

    if (strcmp(mode, "builtin") == 0) {
        fprintf(stderr,
                "builtin mode needs an OpenMP build; use pool mode here\n");
        return 2;
    }
    if (strcmp(mode, "pool") != 0) {
        fprintf(stderr, "usage: %s [pool|builtin] [max_threads] [M] [N] [K]\n",
                argv[0]);
        return 2;
    }
    if (max_threads <= 0) {
        long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
        max_threads = (int)(ncpu > 0 ? ncpu : 4);
    }

    /* start workers and inject BEFORE any other rvcl call (set-once) */
    pool_start(max_threads);
    static rvcl_ischeduler_t pool_sched = {.parallel_for = pool_parallel_for,
            .num_threads = 0,
            .name = "example-multicore-pool"};
    pool_sched.num_threads = (unsigned)max_threads;
    if (rvcl_scheduler_set(&pool_sched) != RVCL_SUCCESS) {
        fprintf(stderr, "scheduler_set failed: a scheduler is already set\n");
        return 1;
    }
    if (rvcl_init() != RVCL_SUCCESS) {
        fprintf(stderr, "rvcl_init failed\n");
        return 1;
    }
    printf("RVCL %s - %s\n", rvcl_get_version(),
            rvcl_get_capability()->description);
    printf("pool: max_threads=%d\n", max_threads);

    static const int scan[] = {1, 2, 4, 8};
    const int n_scan = 4;

    const shape_t primary = {M, N, K};
    static const shape_t ragged[] = {{97, 65, 33}, {7, 7, 7}, {8, 8, 8},
            {9, 9, 9}, {15, 15, 15}, {16, 16, 16}, {17, 17, 17}, {23, 23, 23},
            {24, 24, 24}, {25, 25, 25}};

    if (!verify_correctness(primary, ragged, sizeof(ragged) / sizeof(ragged[0]),
                scan, n_scan, max_threads)) {
        fprintf(stderr, "correctness check FAILED\n");
        return 1;
    }
    printf("correctness: bit-exact across T, vs golden, and ragged shapes\n");

    const char *bench = getenv("RVCL_MULTICORE_BENCH");
    if (bench && bench[0]) {
        static const shape_t shapes[] = {{256, 256, 256}, {512, 512, 512},
                {1024, 1024, 1024}, {512, 512, 2048}};
        bench_scan(shapes, sizeof(shapes) / sizeof(shapes[0]), scan, n_scan,
                max_threads);
    }

    printf("OK\n");
    return 0;
}
