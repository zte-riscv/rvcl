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
 * init.cpp — library initialization / backend registration entry
 *
 * Backends are registered once at library init. RVCL_ENABLE_<BACKEND>
 * build flags decide which backends compile in (PLAN §4.12); dispatch
 * discovers what exists purely through the registry.
 *
 * Registration order is a dispatch contract (RFC-0002 FR1.5): dedicated
 * variants first (high VLEN tier to low), then generic, then the
 * unbounded fallback last. Dispatch breaks cost ties by this order, so
 * the sequence below directly shapes selection.
 ******************************************************************************/

#include "rvcl/init.h"
#include "rvcl/registry.h"
#include "rvcl/types.h"

#include "registry_internal.h"

#include <atomic>
#include <pthread.h>

/* reference backend (always built: golden path + CI on any host) */
rvcl_status_t rvcl_reference_register(void);
/* rvv backend: f32/f16 GEMM, panel packing. On vector builds registers
 * per-tier dedicated variants plus the generic entry; no-op otherwise. */
rvcl_status_t rvcl_rvv_register(void);
/* rvv portable entries (scalar C loops) for builds without vector
 * compilation; no-op on vector builds. */
rvcl_status_t rvcl_rvv_register_portable(void);
/* ime/vme/ame backends: experimental, land in v0.2/v0.3 */

/* Registration failures propagate: a partially registered backend set
 * must not hide behind a successful init. */
static rvcl_status_t register_all(void) {
    rvcl_status_t st = RVCL_SUCCESS;
#if defined(RVCL_ENABLE_RVV_BUILD)
    /* Dedicated tiers (high -> low) + generic, vector builds only. */
    st = rvcl_rvv_register();
    if (st != RVCL_SUCCESS) return st;
#endif
    /* Unbounded f32 fallback. */
    st = rvcl_reference_register();
    if (st != RVCL_SUCCESS) return st;
#if defined(RVCL_ENABLE_RVV_BUILD)
    /* Portable fallback-class entries, non-vector builds only. */
    return rvcl_rvv_register_portable();
#else
    return RVCL_SUCCESS;
#endif
}

static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;
/* Acquire/release pair with the mutex-protected store publishes the
 * fully registered registry to lock-free readers (dispatch fast path). */
static std::atomic<int> g_initialized(0);

RVCL_API const char *rvcl_get_version(void) {
    return RVCL_VERSION_STRING;
}

/*
 * Idempotent library init: registers built-in backends once. Applications
 * may call rvcl_init() explicitly; dispatch also triggers it lazily, so a
 * first computation without an explicit init still sees a full registry.
 */
RVCL_API rvcl_status_t rvcl_init(void) {
    if (std::atomic_load_explicit(&g_initialized, std::memory_order_acquire))
        return RVCL_SUCCESS;

    pthread_mutex_lock(&g_init_mutex);
    if (!std::atomic_load_explicit(&g_initialized, std::memory_order_relaxed)) {
        /* A failed attempt must leave the table exactly as it found it:
         * entries registered before init (application kernels) stay, and
         * a later rvcl_init() retry starts from the same state instead
         * of failing forever on its own half-registered built-ins. The
         * contract assumes application registration is not concurrent
         * with the first init. */
        const int before = rvcl_registry_count();
        rvcl_status_t st = register_all();
        if (st == RVCL_SUCCESS) {
            const rvcl_status_t inv = rvcl_registry_verify_fallback_invariant();
            if (inv != RVCL_SUCCESS) st = RVCL_ERROR_INTERNAL;
        }
        if (st != RVCL_SUCCESS) {
            rvcl_registry_rollback(before);
            pthread_mutex_unlock(&g_init_mutex);
            return st;
        }
        std::atomic_store_explicit(
                &g_initialized, 1, std::memory_order_release);
    }
    pthread_mutex_unlock(&g_init_mutex);
    return RVCL_SUCCESS;
}
