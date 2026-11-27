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

/* Backends are registered once at library init. RVCL_ENABLE_<BACKEND> build
 * flags decide which backends compile in; dispatch discovers what exists
 * purely through the registry. Registration order is a dispatch contract:
 * dedicated variants first (high VLEN tier to low), then generic (including
 * the portable entries on non-vector builds), then the unbounded fallback
 * last. Dispatch breaks cost ties by this order, so the sequence below
 * directly shapes selection. */

#include "rvcl/init.h"
#include "rvcl/registry.h"
#include "rvcl/types.h"

#include "registry_internal.h"

#include <atomic>
#include <pthread.h>

/* reference backend (always built: golden path + CI on any host) */
rvcl_status_t rvcl_reference_register(void);
/* reference GEMV golden (always built, like the matmul reference) */
rvcl_status_t rvcl_gemv_reference_register(void);
/* rvv backend: f32/f16 GEMM, panel packing. On vector builds registers
 * per-tier dedicated variants plus the generic entry; no-op otherwise. */
rvcl_status_t rvcl_rvv_register(void);
/* rvv portable entries (scalar C loops) for builds without vector
 * compilation; no-op on vector builds. */
rvcl_status_t rvcl_rvv_register_portable(void);
/* rvv GEMV: single C-portable path. The generic entry requires RVV on
 * vector builds; the portable entry serves any host otherwise. */
rvcl_status_t rvcl_rvv_register_gemv(void);
rvcl_status_t rvcl_rvv_register_gemv_portable(void);
/* rvv conv2d: im2col + the dtype's GEMM micro-kernel, generic + portable
 * entries, same ordering habit as the matmul and GEMV entries. */
rvcl_status_t rvcl_rvv_register_conv2d(void);
rvcl_status_t rvcl_rvv_register_conv2d_portable(void);
/* rvv inner_product: transposing pack_b + the dtype's GEMM micro-kernel,
 * generic + portable entries, same ordering habit. */
rvcl_status_t rvcl_rvv_register_inner_product(void);
rvcl_status_t rvcl_rvv_register_inner_product_portable(void);
/* rvv pooling: row-wise window reduction, generic + portable entries, same
 * ordering habit. */
rvcl_status_t rvcl_rvv_register_pooling(void);
rvcl_status_t rvcl_rvv_register_pooling_portable(void);
/* ime backend: f16 matrix-extension GEMM. Registers before RVV so that
 * equal-cost ties resolve to the more specialized entry, and contributes
 * its capability probe before the snapshot is first built. */
rvcl_status_t rvcl_ime_register(void);
rvcl_status_t rvcl_ime_register_portable(void);
rvcl_status_t ime_register_capability_probe(void);
/* vme/ame backends: experimental, land in v0.3 */

/* Registration failures propagate: a partially registered backend set
 * must not hide behind a successful init. */
static rvcl_status_t register_all(void) {
    rvcl_status_t st = RVCL_SUCCESS;
#if defined(RVCL_ENABLE_IME_BUILD)
    /* IME before RVV: the tie-break between equal-cost candidates is
     * registration order, and the hardware matrix entry is the more
     * specialized one. Each call is a no-op on the build flavor whose
     * entry it does not own. The probe registration must precede any
     * capability query; a closed window (the snapshot was already built)
     * means it is in place and is not an error. */
    st = ime_register_capability_probe();
    if (st != RVCL_SUCCESS && st != RVCL_ERROR_ALREADY_INITIALIZED) return st;
    st = rvcl_ime_register();
    if (st != RVCL_SUCCESS) return st;
    st = rvcl_ime_register_portable();
    if (st != RVCL_SUCCESS) return st;
#endif
#if defined(RVCL_ENABLE_RVV_BUILD)
    /* Dedicated tiers (high -> low) + generic, vector builds only. */
    st = rvcl_rvv_register();
    if (st != RVCL_SUCCESS) return st;
    /* Portable generic entries (non-vector builds only) are still
     * preferred over the reference fallback, so they register first. */
    st = rvcl_rvv_register_portable();
    if (st != RVCL_SUCCESS) return st;
    /* GEMV generic + portable entries, same ordering habit. */
    st = rvcl_rvv_register_gemv();
    if (st != RVCL_SUCCESS) return st;
    st = rvcl_rvv_register_gemv_portable();
    if (st != RVCL_SUCCESS) return st;
    /* Conv2D generic + portable entries. */
    st = rvcl_rvv_register_conv2d();
    if (st != RVCL_SUCCESS) return st;
    st = rvcl_rvv_register_conv2d_portable();
    if (st != RVCL_SUCCESS) return st;
    /* Inner product generic + portable entries. */
    st = rvcl_rvv_register_inner_product();
    if (st != RVCL_SUCCESS) return st;
    st = rvcl_rvv_register_inner_product_portable();
    if (st != RVCL_SUCCESS) return st;
    /* Pooling generic + portable entries. */
    st = rvcl_rvv_register_pooling();
    if (st != RVCL_SUCCESS) return st;
    st = rvcl_rvv_register_pooling_portable();
    if (st != RVCL_SUCCESS) return st;
#endif
    /* GEMV golden fallback, before the matmul fallback (same family). */
    st = rvcl_gemv_reference_register();
    if (st != RVCL_SUCCESS) return st;
    /* Unbounded f32 fallback, last. */
    return rvcl_reference_register();
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
            /* App-owned tuples (all entries registered before this pass)
             * are exempt from the fallback check; a plugin's shape-bounded
             * kernel on an otherwise-unoccupied tuple must not fail init. */
            const rvcl_status_t inv
                    = rvcl_registry_verify_fallback_invariant(before);
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
