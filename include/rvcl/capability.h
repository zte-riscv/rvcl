/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * capability.h — CPU capability model (Layer: Capability)
 *
 * SPDX-License-Identifier: Apache-2.0
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
RVCL_API int rvcl_capability_has(const rvcl_capability_t *cap, rvcl_feature_id_t id);

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
