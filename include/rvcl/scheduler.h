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
 * scheduler.h — injectable scheduler (ACL IScheduler equivalent)
 *
 * Hosts own the parallelism (PLAN §4.9). Semantic APIs and kernels face
 * only rvcl_ischeduler_t; micro-kernel parallelism is expressed as work
 * range slices. rvcl_scheduler_set() is process-wide and one-shot
 * (set-once semantics, mirroring ACL's single Scheduler instance).
 ******************************************************************************/

#ifndef RVCL_SCHEDULER_H
#define RVCL_SCHEDULER_H

#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Task invoked per work slice: fn(scheduler, slice, user_ctx). */
struct rvcl_ischeduler; /* forward decl: file scope, not parameter scope */

typedef void (*rvcl_parallel_task_fn)(struct rvcl_ischeduler *,
        const rvcl_work_range_t *slice, void *user_ctx);

typedef struct rvcl_ischeduler {
    /* Splits [total] and executes the task over the slices, blocking until
     * all slices complete. Implementations may execute inline (serial). */
    void (*parallel_for)(struct rvcl_ischeduler *,
            const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
            void *user_ctx);
    unsigned num_threads;
    const char *name; /* for verbose/profiling output */
} rvcl_ischeduler_t;

/*
 * Injects a host scheduler. Process-wide, one-shot: a second call returns
 * RVCL_ERROR_ALREADY_INITIALIZED (set-once semantics per PLAN §4.9).
 * Passing NULL installs the built-in serial scheduler. Thread-safe.
 */
RVCL_API rvcl_status_t rvcl_scheduler_set(rvcl_ischeduler_t *scheduler);

/* Currently installed scheduler (never NULL after first use). */
RVCL_API rvcl_ischeduler_t *rvcl_scheduler_get(void);

/* Built-in serial scheduler (single-threaded fallback; tests/emulators). */
RVCL_API rvcl_ischeduler_t *rvcl_serial_scheduler(void);

/* Convenience: parallel_for through the current scheduler. */
RVCL_API void rvcl_parallel_for(const rvcl_work_range_t *total,
        rvcl_parallel_task_fn task, void *user_ctx);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_SCHEDULER_H */
