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
 * capability.h — CPU capability model (Layer: Capability)
 *
 * Capability describes what the hardware can do. It NEVER selects kernels
 * (PLAN §4.1 invariant). Detection is done once and cached; failures
 * degrade to the RVV baseline instead of erroring out (PLAN §4.4).
 ******************************************************************************/

#ifndef RVCL_CAPABILITY_H
#define RVCL_CAPABILITY_H

#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    /* Detected features; count <= RVCL_MAX_FEATURES. */
    rvcl_feature_t features[RVCL_MAX_FEATURES];
    int num_features;
    char description[128]; /* human-readable CPU description (verbose) */
} rvcl_capability_t;

/* Returns true if the capability set contains `id`. */
RVCL_API int rvcl_capability_has(
        const rvcl_capability_t *cap, rvcl_feature_id_t id);

/* Looks up a feature; returns NULL when absent. */
RVCL_API const rvcl_feature_t *rvcl_capability_get(
        const rvcl_capability_t *cap, rvcl_feature_id_t id);

/*
 * Detects host capabilities (cached; thread-safe after first call).
 * Detection layers: compile-time __riscv_* macros -> runtime auxv/hwprobe
 * -> vendor interfaces. On any failure the result degrades to the RVV
 * baseline rather than returning an error (PLAN §4.4).
 * The returned pointer is owned by RVCL and valid for the process lifetime.
 */
RVCL_API const rvcl_capability_t *rvcl_get_capability(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_CAPABILITY_H */
