/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * scheduler.h — injectable scheduler (ACL IScheduler equivalent)
 *
 * SPDX-License-Identifier: Apache-2.0
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
