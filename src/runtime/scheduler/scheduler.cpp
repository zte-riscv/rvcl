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
 * scheduler.cpp — serial scheduler + host injection (set-once)
 ******************************************************************************/

#include "rvcl/scheduler.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

static void serial_parallel_for(rvcl_ischeduler_t *s,
        const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
        void *user_ctx);

static rvcl_ischeduler_t g_serial = {
        .parallel_for = serial_parallel_for,
        .num_threads = 1,
        .name = "serial",
};

static void serial_parallel_for(rvcl_ischeduler_t *s,
        const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
        void *user_ctx) {
    (void)s;
    /* Serial: hand the whole range to one task invocation. */
    task(&g_serial, total, user_ctx);
}

rvcl_ischeduler_t *rvcl_serial_scheduler(void) {
    return &g_serial;
}

/* ----------------------------------------------------------------- set-once */

static rvcl_ischeduler_t *g_scheduler; /* NULL until set */
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_set_done;

rvcl_status_t rvcl_scheduler_set(rvcl_ischeduler_t *scheduler) {
    pthread_mutex_lock(&g_mutex);
    if (g_set_done) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_ALREADY_INITIALIZED; /* set-once (PLAN §4.9) */
    }
    g_scheduler = scheduler ? scheduler : rvcl_serial_scheduler();
    g_set_done = 1;
    pthread_mutex_unlock(&g_mutex);
    return RVCL_SUCCESS;
}

rvcl_ischeduler_t *rvcl_scheduler_get(void) {
    return g_scheduler ? g_scheduler : rvcl_serial_scheduler();
}

void rvcl_parallel_for(const rvcl_work_range_t *total,
        rvcl_parallel_task_fn task, void *user_ctx) {
    rvcl_ischeduler_t *s = rvcl_scheduler_get();
    s->parallel_for(s, total, task, user_ctx);
}
