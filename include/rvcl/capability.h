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

/* ------------------------------------------------- vendor capability probes
 *
 * Vendor probe registration gives an out-of-tree plugin a public write
 * channel into the capability snapshot: the snapshot is built at most once
 * per process (before the first capability query or dispatch), and each
 * registered probe is invoked during that build to append vendor-specific
 * feature ids (id >= RVCL_FEATURE_VENDOR_FIRST) to the snapshot.
 *
 * Registration timing contract: a probe must be registered before the
 * snapshot is built — i.e. before the first rvcl_get_capability() or the
 * first dispatch, whichever comes first. A plugin's constructor (pre-main,
 * via LD_PRELOAD or direct linking) satisfies this naturally. Registering
 * after the snapshot is built returns RVCL_ERROR_ALREADY_INITIALIZED
 * rather than silently ignoring the probe. The accepted probes are invoked
 * during the single-threaded snapshot build, so no probe is ever run
 * concurrently with another. Registered probes persist for the process
 * lifetime and cannot be unregistered.
 *
 * Sanitization: whatever a probe appends is filtered before it becomes
 * part of the snapshot — only feature ids in the vendor range
 * (>= RVCL_FEATURE_VENDOR_FIRST) are kept, ids already present are
 * dropped (append-only, never overwrite), an over-full feature table is
 * rolled back to the pre-probe count, and a probe returning anything
 * other than RVCL_SUCCESS has its entire increment discarded while the
 * remaining probes still run (degrade, never error). The framework does
 * not interpret a vendor feature; it only lets the probe declare it.
 */
#define RVCL_MAX_VENDOR_PROBES 8

typedef rvcl_status_t (*rvcl_vendor_probe_fn)(rvcl_capability_t *cap);

/* Registers a vendor probe; RVCL_ERROR_ALREADY_INITIALIZED when the
 * capability snapshot is already built. Thread-safe; expected on the
 * single-threaded plugin-init path. */
RVCL_API rvcl_status_t rvcl_capability_register_vendor_probe(
        rvcl_vendor_probe_fn probe);

/*
 * Safe CSR read under SIGILL protection: executes csrr CSR, stores the
 * value and returns 1 on success, or returns 0 when the CSR is not
 * implemented (illegal instruction). Lets a probe detect optional
 * architecture state without process-level signal plumbing. The SIGILL
 * disposition is process-global state, so the probe-time (single-threaded
 * snapshot build) usage is the supported pattern. On hosts without a CSR
 * instruction (e.g. x86) this always returns 0 so the header stays
 * compilable everywhere.
 *
 * csrr requires a compile-time CSR offset, so only the CSR constants the
 * helper enumerates are readable (currently 0x7c0 and 0xCE4); an unknown
 * CSR number returns 0 (treated as absent), it does not trap.
 */
RVCL_API int rvcl_capability_try_csr(unsigned int csr, uint32_t *value);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_CAPABILITY_H */
