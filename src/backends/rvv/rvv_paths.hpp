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
 * rvv_paths.hpp — per-dtype micro-kernel path table
 *
 * One table row per micro-kernel path (c / intrinsics / jit). A row
 * carries the path name, a getter for the live function pointer, an
 * availability probe (compile-time macros upgraded by the run-time
 * capability probe for the JIT path), the auto-selection priority and the
 * loud-failure message for a forced-but-unavailable request. The f32 and
 * f16 selection (rvv_pick_gemm_*) and the bench --path resolver all walk
 * these tables, so a new dtype adds a table row and nothing else — the
 * resolver never branches on the dtype. Semantics stay bit-identical to
 * the hand-written selectors they replace:
 *   f32 auto = JIT-when-RVV-else-C
 *   f16 auto = intrinsics-when-RVV-else-C
 *
 * Internal to the backend; included only by rvv_backend.cpp.
 ******************************************************************************/

#ifndef RVCL_RVV_PATHS_H
#define RVCL_RVV_PATHS_H

#include "rvv_internal.hpp"

#include <stddef.h>
#include <stdio.h>
#include <string.h>

/* Getter for a live micro-kernel function pointer; NULL when the path is
 * not compiled in or its generation failed. Probe answers "can this path
 * ever run here" (compile-time macros + the JIT run-time capability
 * probe); NULL means always available. auto_rank orders the auto chain
 * (lowest wins); -1 excludes the path from it. forced_msg is printed when
 * a forced (non-auto) request hits an unavailable path. */
typedef rvv_gemm_fn (*rvv_path_fn_t)(void);
typedef int (*rvv_path_probe_t)(void);

typedef struct {
    const char *name;
    rvv_path_fn_t getter;
    rvv_path_probe_t probe;
    int auto_rank;
    const char *forced_msg;
} rvv_path_entry_t;

typedef struct {
    const char *unknown_fmt;
    const rvv_path_entry_t *paths;
    size_t npaths;
} rvv_path_table_t;

/* ------------------------------------------------------------- getters */

static rvv_gemm_fn rvv_path_f32_c(void) {
    return rvv_gemm_f32_c;
}

static rvv_gemm_fn rvv_path_f16_c(void) {
    return rvv_gemm_f16_c;
}

#if defined(RVCL_HAVE_JIT)
static int rvv_path_f32_jit_ok(void) {
    return rvv_gemm_f32_jit_available();
}

static rvv_gemm_fn rvv_path_f32_jit(void) {
    return rvv_gemm_f32_jit_entry();
}
#endif

#if defined(__riscv_vector)
static rvv_gemm_fn rvv_path_f32_rvv(void) {
    return rvv_gemm_f32_rvv;
}

static rvv_gemm_fn rvv_path_f16_rvv(void) {
    return rvv_gemm_f16_rvv;
}
#endif

/* Probe that never reports available: a path known by name but never
 * served on this build (f16's JIT), so a forced request fails loudly. */
static int rvv_path_never(void) {
    return 0;
}

/* ----------------------------------------------------------- resolver */

static inline rvv_gemm_fn rvv_path_auto(const rvv_path_table_t *t) {
    const rvv_path_entry_t *best = NULL;
    for (size_t i = 0; i < t->npaths; i++) {
        const rvv_path_entry_t *e = &t->paths[i];
        if (e->auto_rank < 0) continue;
        if (e->probe && !e->probe()) continue;
        if (!e->getter || !e->getter()) continue;
        if (!best || e->auto_rank < best->auto_rank) best = e;
    }
    return best ? best->getter() : NULL;
}

static inline rvv_gemm_fn rvv_path_by_name(
        const rvv_path_table_t *t, const char *name) {
    if (!strcmp(name, "auto")) return rvv_path_auto(t);
    for (size_t i = 0; i < t->npaths; i++) {
        const rvv_path_entry_t *e = &t->paths[i];
        if (strcmp(e->name, name) != 0) continue;
        if (e->probe && !e->probe()) {
            if (e->forced_msg) fprintf(stderr, "%s", e->forced_msg);
            return NULL;
        }
        return e->getter ? e->getter() : NULL;
    }
    fprintf(stderr, t->unknown_fmt, name);
    return NULL;
}

/* --------------------------------------------------------------- f32 */

static const rvv_path_entry_t rvv_f32_path_entries[] = {
#if defined(RVCL_HAVE_JIT)
        {"jit", rvv_path_f32_jit, rvv_path_f32_jit_ok, 0,
                "rvcl: forced JIT f32 path requires the vector extension\n"},
#endif
#if defined(__riscv_vector)
        {"intrinsics", rvv_path_f32_rvv, NULL, -1, NULL},
#endif
        {"c", rvv_path_f32_c, NULL, 2, NULL},
};

static const rvv_path_table_t rvv_f32_paths = {
        "rvcl: f32 kernel path '%s' is unavailable "
        "(want auto, c, jit or intrinsics)\n",
        rvv_f32_path_entries,
        sizeof(rvv_f32_path_entries) / sizeof(rvv_f32_path_entries[0]),
};

/* --------------------------------------------------------------- f16 */

static const rvv_path_entry_t rvv_f16_path_entries[] = {
        {"jit", NULL, rvv_path_never, -1,
                "rvcl: f16 has no JIT path (want auto, c or intrinsics)\n"},
#if defined(__riscv_vector)
        {"intrinsics", rvv_path_f16_rvv, NULL, 0, NULL},
#endif
        {"c", rvv_path_f16_c, NULL, 1, NULL},
};

static const rvv_path_table_t rvv_f16_paths = {
        "rvcl: f16 kernel path '%s' is unavailable "
        "(want auto, c or intrinsics)\n",
        rvv_f16_path_entries,
        sizeof(rvv_f16_path_entries) / sizeof(rvv_f16_path_entries[0]),
};

#endif /* RVCL_RVV_PATHS_H */
