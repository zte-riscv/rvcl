/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * examples/scheduler.c — inject a custom host scheduler
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Demonstrates the Scheduler contract (PLAN §4.9): the host owns
 * parallelism. A custom rvcl_ischeduler_t slices work ranges and
 * executes the slices (here: on worker threads via a trivial pool);
 * the kernel just computes. Injection is process-wide and one-shot.
 *
 * Must be the FIRST rvcl call of the process to win the set-once race
 * with the lazy default: we inject before rvcl_init()/rvcl_matmul().
 ******************************************************************************/

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rvcl/rvcl.h"

/* ------------------------------------------------- custom thread pool */

#define POOL_THREADS 4
#define POOL_MAX_TASKS 64

typedef struct {
    rvcl_parallel_task_fn fn;
    rvcl_ischeduler_t *sched;
    rvcl_work_range_t range;
    void *ctx;
} pool_task_t;

typedef struct {
    pool_task_t tasks[POOL_MAX_TASKS];
    int num_tasks;
    int next;              /* next task to grab (atomic-ish via mutex) */
    int done;              /* completed task count */
    pthread_mutex_t mu;
    pthread_cond_t cv_work, cv_done;
} pool_t;

static pool_t g_pool = {.mu = PTHREAD_MUTEX_INITIALIZER,
    .cv_work = PTHREAD_COND_INITIALIZER,
    .cv_done = PTHREAD_COND_INITIALIZER};

static void *pool_worker(void *arg) {
    (void)arg;
    for (;;) {
        pthread_mutex_lock(&g_pool.mu);
        while (g_pool.next >= g_pool.num_tasks && g_pool.num_tasks >= 0)
            pthread_cond_wait(&g_pool.cv_work, &g_pool.mu);
        int idx = g_pool.next++;
        pthread_mutex_unlock(&g_pool.mu);
        if (idx >= g_pool.num_tasks) continue; /* spurious */
        pool_task_t *t = &g_pool.tasks[idx];
        t->fn(t->sched, &t->range, t->ctx);

        pthread_mutex_lock(&g_pool.mu);
        if (++g_pool.done >= g_pool.num_tasks)
            pthread_cond_signal(&g_pool.cv_done);
        pthread_mutex_unlock(&g_pool.mu);
    }
    return NULL;
}

/* parallel_for: split M into POOL_THREADS slices, enqueue, wait. */
static void pool_parallel_for(rvcl_ischeduler_t *s,
        const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
        void *user_ctx) {
    const int64_t rows = total->m_end - total->m_begin;
    int parts = rows < POOL_THREADS ? (int)rows : POOL_THREADS;
    if (parts < 1) parts = 1;

    pthread_mutex_lock(&g_pool.mu);
    g_pool.num_tasks = 0;
    g_pool.next = 0;
    g_pool.done = 0;
    const int64_t step = (rows + parts - 1) / parts;
    for (int64_t b = total->m_begin; b < total->m_end; b += step) {
        pool_task_t *t = &g_pool.tasks[g_pool.num_tasks++];
        t->fn = task;
        t->sched = s;
        t->range = *total;
        t->range.m_begin = b;
        t->range.m_end = b + step < total->m_end ? b + step : total->m_end;
        t->ctx = user_ctx;
    }
    pthread_cond_broadcast(&g_pool.cv_work);
    while (g_pool.done < g_pool.num_tasks)
        pthread_cond_wait(&g_pool.cv_done, &g_pool.mu);
    pthread_mutex_unlock(&g_pool.mu);
}

/* --------------------------------------------------------------- main */

int main(void) {
    /* 1. Start workers + inject BEFORE any other rvcl call (set-once). */
    pthread_t th[POOL_THREADS];
    for (int i = 0; i < POOL_THREADS; i++)
        pthread_create(&th[i], NULL, pool_worker, NULL);

    static rvcl_ischeduler_t pool_sched = {
        .parallel_for = pool_parallel_for,
        .num_threads = POOL_THREADS,
        .name = "example-pool-4",
    };
    rvcl_status_t st = rvcl_scheduler_set(&pool_sched);
    printf("scheduler injected: %s (num_threads=%u)\n", pool_sched.name,
            pool_sched.num_threads);
    if (st != RVCL_SUCCESS) {
        fprintf(stderr, "scheduler_set failed: %d "
                        "(a scheduler was already installed)\n",
                (int)st);
        return 1;
    }

    if (rvcl_init() != RVCL_SUCCESS) {
        fprintf(stderr, "rvcl_init failed\n");
        return 1;
    }
    printf("RVCL %s — %s\n", rvcl_get_version(),
            rvcl_get_capability()->description);

    /* 2. Matmul through the injected scheduler. */
    enum { M = 97, N = 65, K = 33 }; /* ragged: exercises slicing edges */
    float *a = malloc(M * K * 4), *b = malloc(K * N * 4), *c = malloc(M * N * 4);
    float *bias = malloc(N * 4), *golden = malloc(M * N * 4);
    for (int i = 0; i < M * K; i++) a[i] = (float)((i * 13) % 17) / 17.0f - 0.5f;
    for (int i = 0; i < K * N; i++) b[i] = (float)((i * 7) % 19) / 19.0f - 0.5f;
    for (int j = 0; j < N; j++) bias[j] = 0.25f;

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = bias[j];
            for (int p = 0; p < K; p++)
                acc += a[i * K + p] * b[p * N + j];
            golden[i * N + j] = acc;
        }

    rvcl_matmul_desc_t d = {
        .m = M, .n = N, .k = K,
        .dtype_a = RVCL_DTYPE_FP32,
        .dtype_b = RVCL_DTYPE_FP32,
        .dtype_acc = RVCL_DTYPE_AUTO,
        .dtype_c = RVCL_DTYPE_FP32,
        .dtype_bias = RVCL_DTYPE_FP32,
    };
    st = rvcl_matmul(&d, a, b, bias, c);
    if (st != RVCL_SUCCESS) {
        fprintf(stderr, "rvcl_matmul failed: %d\n", (int)st);
        return 1;
    }

    int bad = 0;
    for (int i = 0; i < M * N; i++)
        if (c[i] != golden[i]) bad++;
    if (bad) {
        fprintf(stderr, "%d mismatched outputs\n", bad);
        return 1;
    }
    printf("matmul via injected scheduler: OK (%d x %d x %d, bit-exact)\n",
            M, N, K);

    /* 3. Set-once demonstration: a second injection is rejected. */
    st = rvcl_scheduler_set(rvcl_serial_scheduler());
    printf("second scheduler_set rejected: %s\n",
            st == RVCL_ERROR_ALREADY_INITIALIZED ? "yes" : "NO (BUG)");

    free(a); free(b); free(c); free(bias); free(golden);
    printf("OK — custom scheduler drove the computation\n");
    return 0;
}
