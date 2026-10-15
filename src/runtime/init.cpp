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
 ******************************************************************************/

#include "rvcl/init.h"
#include "rvcl/registry.h"
#include "rvcl/types.h"

#include <pthread.h>

/* reference backend (always built: golden path + CI on any host) */
rvcl_status_t rvcl_reference_register(void);
/* rvv backend: f32/f16 GEMM, panel packing (v0.1, PLAN §6.2 step 7-8) */
rvcl_status_t rvcl_rvv_register(void);
/* ime/vme/ame backends: experimental, land in v0.2/v0.3 */

static void register_all(void) {
    rvcl_reference_register();
#if defined(RVCL_ENABLE_RVV_BUILD)
    rvcl_rvv_register();
#endif
}

static int g_initialized;
static pthread_mutex_t g_init_mutex = PTHREAD_MUTEX_INITIALIZER;

RVCL_API const char *rvcl_get_version(void) {
    return RVCL_VERSION_STRING;
}

/*
 * Idempotent library init: registers built-in backends. Applications may
 * call rvcl_init() explicitly; rvcl_matmul() and friends call it lazily.
 */
RVCL_API rvcl_status_t rvcl_init(void) {
    pthread_mutex_lock(&g_init_mutex);
    if (!g_initialized) {
        register_all();
        g_initialized = 1;
    }
    pthread_mutex_unlock(&g_init_mutex);
    return RVCL_SUCCESS;
}
