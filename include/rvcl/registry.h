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
 * registry.h — kernel registry (Layer: Registry)
 *
 * Registers and indexes kernel metadata by operation × (A,B,acc,C)
 * (PLAN §4.5.2). The registry NEVER makes load-dependent performance
 * decisions (§4.1 invariant) — that is dispatch's job.
 ******************************************************************************/

#ifndef RVCL_REGISTRY_H
#define RVCL_REGISTRY_H

#include "rvcl/kernel.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

#define RVCL_REGISTRY_MAX_KERNELS 64

/*
 * Registers a kernel descriptor + factory. The descriptor is copied.
 * Fails with RVCL_ERROR_ALREADY_REGISTERED on a duplicate name.
 * Thread-safe. Registration is expected at library init / backend
 * registration time, not on hot paths.
 */
RVCL_API rvcl_status_t rvcl_register_kernel(
        const rvcl_kernel_desc_t *desc, rvcl_kernel_create_fn create);

/* Removes a kernel by name (mainly for tests). */
RVCL_API rvcl_status_t rvcl_unregister_kernel(const char *name);

/* Number of registered kernels. */
RVCL_API int rvcl_registry_count(void);

/* Iteration for dispatch: returns NULL when index out of range. */
RVCL_API const rvcl_kernel_desc_t *rvcl_registry_get(int index);

/* Factory lookup by name; NULL when unknown. */
RVCL_API rvcl_kernel_create_fn rvcl_registry_find_factory(const char *name);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_REGISTRY_H */
