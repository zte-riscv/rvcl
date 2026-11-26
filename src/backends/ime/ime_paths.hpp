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

/* One table row per micro-kernel path (ime / c). A row carries the path
 * name, a getter for the live function pointer, an availability probe
 * (compile-time macros upgraded by the run-time host gate), the
 * auto-selection priority and the loud-failure message for a forced-but-
 * unavailable request. The f16 selection (ime_pick_gemm_f16*) resolves
 * through this table, so the auto policy is table data:
 *   f16 auto = instruction-path-when-the-host-allows-it, else C
 *
 * Internal to the backend; included only by ime_backend.cpp.
 */

#ifndef RVCL_IME_PATHS_H
#define RVCL_IME_PATHS_H

#include "ime_internal.hpp"

#include <stdio.h>
#include <string.h>

/* Getter for a live micro-kernel function pointer; NULL when the path is
 * not compiled in. Probe answers "can this path run here"; NULL means
 * always available. auto_rank orders the auto chain (lowest wins); -1
 * excludes the path from it. forced_msg is printed when a forced
 * (non-auto) request hits an unavailable path. */
typedef ime_gemm_fn (*ime_path_fn_t)(void);
typedef int (*ime_path_probe_t)(void);

typedef struct {
    const char *name;
    ime_path_fn_t getter;
    ime_path_probe_t probe;
    int auto_rank;
    const char *forced_msg;
} ime_path_entry_t;

typedef struct {
    const char *unknown_fmt;
    const ime_path_entry_t *paths;
    size_t npaths;
} ime_path_table_t;

/* ------------------------------------------------------------- getters */

static ime_gemm_fn ime_path_c(void) {
    return ime_gemm_f16_c;
}

#if defined(__riscv_vector)
static ime_gemm_fn ime_path_ime(void) {
    return ime_gemm_f16_ime;
}

static int ime_path_ime_ok(void) {
    return ime_instruction_path_available();
}
#else
/* Probe that never reports available: a path known by name but never
 * served on this build, so a forced request fails loudly instead of
 * silently binding a wrong path. */
static int ime_path_never(void) {
    return 0;
}
#endif

/* ----------------------------------------------------------- resolver */

static inline ime_gemm_fn ime_path_auto(const ime_path_table_t *t) {
    const ime_path_entry_t *best = NULL;
    for (size_t i = 0; i < t->npaths; i++) {
        const ime_path_entry_t *e = &t->paths[i];
        if (e->auto_rank < 0) continue;
        if (e->probe && !e->probe()) continue;
        if (!e->getter || !e->getter()) continue;
        if (!best || e->auto_rank < best->auto_rank) best = e;
    }
    return best ? best->getter() : NULL;
}

static inline ime_gemm_fn ime_path_by_name(
        const ime_path_table_t *t, const char *name) {
    if (!strcmp(name, "auto")) return ime_path_auto(t);
    for (size_t i = 0; i < t->npaths; i++) {
        const ime_path_entry_t *e = &t->paths[i];
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

/* --------------------------------------------------------------- f16 */

static const ime_path_entry_t ime_f16_path_entries[] = {
#if defined(__riscv_vector)
        {"ime", ime_path_ime, ime_path_ime_ok, 0,
                "rvcl: forced IME f16 path requires the matrix extension "
                "on every CPU of the process affinity mask\n"},
#else
        {"ime", NULL, ime_path_never, -1,
                "rvcl: IME f16 instruction path is not compiled on this "
                "target\n"},
#endif
        {"c", ime_path_c, NULL, 1, NULL},
};

static const ime_path_table_t ime_f16_paths = {
        "rvcl: f16 kernel path '%s' is unavailable (want auto, c or ime)\n",
        ime_f16_path_entries,
        sizeof(ime_f16_path_entries) / sizeof(ime_f16_path_entries[0]),
};

#endif /* RVCL_IME_PATHS_H */
