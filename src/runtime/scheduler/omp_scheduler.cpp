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
 * omp_scheduler.cpp — built-in OpenMP scheduler backend
 *
 * Zero-injection default parallelism for the standalone product shape
 * (PR-3 §5.3.1): hosts get a parallel matmul without injecting a scheduler.
 * Compiled only into OpenMP builds (RVCL_ENABLE_OMP=ON); the non-OpenMP stub
 * lives in scheduler.cpp.
 *
 * Execution follows the oneDNN orientation of "one coarse task per thread,
 * no OpenMP dynamic scheduling": a single `omp parallel` region splits the M
 * dimension into diff-1 balanced slices, one per thread, and each thread
 * executes its own slice. Splitting only M (never K) keeps every slice's
 * accumulation order identical to the serial run, so results are bit-exact
 * independent of the thread count.
 ******************************************************************************/

#include "rvcl/scheduler.h"

#include <omp.h>

/* Diff-1 balanced slice [idx] of [parts] over [total] (D-9): every slice
 * carries either `step` or `step+1` rows so sizes differ by at most one.
 * n/k intervals pass through unchanged (plain GEMM keeps full K per slice). */
static void omp_slice(rvcl_work_range_t *out, const rvcl_work_range_t *total,
        int64_t idx, int64_t parts) {
    const int64_t rows = total->m_end - total->m_begin;
    const int64_t step = rows / parts;
    const int64_t rem = rows % parts;
    const int64_t begin = total->m_begin + idx * step + (idx < rem ? idx : rem);
    *out = *total;
    out->m_begin = begin;
    out->m_end = begin + step + (idx < rem ? 1 : 0);
}

static void omp_parallel_for(rvcl_ischeduler_t *s,
        const rvcl_work_range_t *total, rvcl_parallel_task_fn task,
        void *user_ctx) {
    const int64_t rows = total->m_end - total->m_begin;

    /* Guarantee 4: never nest inside an already-active parallel region —
     * inline the whole range so an OMP runtime does not recurse. */
    if (omp_in_parallel()) {
        task(s, total, user_ctx);
        return;
    }

    /* Small-load protection: no more threads than rows to slice. */
    const int max_threads = omp_get_max_threads();
    const int64_t parts64
            = rows < (int64_t)max_threads ? rows : (int64_t)max_threads;
    const int parts = parts64 < 2 ? 1 : (int)parts64;
    if (parts < 2) {
        task(s, total, user_ctx); /* one slice: run inline */
        return;
    }

    /* Single static region: thread t owns slice t, no dynamic scheduling. */
#pragma omp parallel num_threads(parts)
    {
        rvcl_work_range_t slice;
        omp_slice(&slice, total, omp_get_thread_num(), parts);
        task(s, &slice, user_ctx);
    }
}

rvcl_ischeduler_t *rvcl_omp_scheduler(void) {
    /* Magic static: thread-safe one-time construction snapshots num_threads
     * at first access (OMP_NUM_THREADS / omp_set_num_threads apply live in
     * omp_parallel_for through omp_get_max_threads). */
    static rvcl_ischeduler_t sched = {
            .parallel_for = omp_parallel_for,
            .num_threads = (unsigned)omp_get_max_threads(),
            .name = "omp",
    };
    return &sched;
}
