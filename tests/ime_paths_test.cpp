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

/* One ctest entry per RVCL_IME_KERNEL value (the knob is read once per
 * process, so each value needs its own process). Expectations are
 * host-adaptive and derive from the same host gate the selector uses,
 * with the availability itself cross-checked: when the gate says yes the
 * instruction path must really resolve (and really execute elsewhere in
 * the numeric suite), and when it says no the auto policy must bind the
 * C path — the property that keeps an unguarded host away from an
 * illegal instruction. Forced-but-unavailable and unknown knob values
 * must fail loudly (NULL + a stderr reason), never silently bind a
 * different path.
 *
 * stderr capture uses a tmpfile() redirect around the resolver call.
 */

#include "../src/backends/ime/ime_internal.hpp"
#include "rvcl/rvcl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int g_failures;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  ok: %s\n", msg); \
        } else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

/* Captures everything the callee writes to stderr into a temp file;
 * returns the file (positioned at 0) or NULL. On any failure it restores
 * fd 2 and closes whatever it opened, so a later capture (or the caller's
 * own stderr) is never left pointing at a stale descriptor. */
static FILE *stderr_capture_begin(int *saved_fd) {
    fflush(stderr);
    *saved_fd = dup(2);
    if (*saved_fd < 0) return NULL;
    FILE *f = tmpfile();
    if (!f) {
        close(*saved_fd);
        *saved_fd = -1;
        return NULL;
    }
    const int fd = fileno(f);
    if (fd < 0 || dup2(fd, 2) < 0) {
        fclose(f);
        close(*saved_fd);
        *saved_fd = -1;
        return NULL;
    }
    return f;
}

static long stderr_capture_end(FILE *f, int saved_fd) {
    fflush(stderr);
    long size = 0;
    if (f) {
        size = ftell(f);
        fclose(f);
    }
    if (saved_fd >= 0) {
        dup2(saved_fd, 2);
        close(saved_fd);
    }
    return size;
}

/* Driving the hardware-bound entry by name on a host where the run-time
 * gate is closed must refuse with RVCL_UNSUPPORTED and create no kernel,
 * so the instruction can never execute there. The capability check
 * precedes the knob lookup in create_ime(), so this holds for every
 * RVCL_IME_KERNEL value. The entry exists only on RISC-V IME builds; on
 * other hosts the portable entry is a different registration and there is
 * nothing to deny. */
static void check_by_name_gate_deny(int available) {
    if (rvcl_init() != RVCL_SUCCESS) {
        CHECK(0, "library init succeeds for the by-name drive");
        return;
    }
    rvcl_kernel_create_fn create
            = rvcl_registry_find_factory("rvcl_matmul_f16_ime_1024");
    if (!create) {
        printf("  skip: hardware-bound entry not registered on this build\n");
        return;
    }
    if (available) {
        printf("  skip: host executes the extension; nothing to deny\n");
        return;
    }
    const rvcl_kernel_desc_t *kd = NULL;
    for (int i = 0, n = rvcl_registry_count(); i < n; i++) {
        const rvcl_kernel_desc_t *cand = rvcl_registry_get(i);
        if (cand && strcmp(cand->name, "rvcl_matmul_f16_ime_1024") == 0) {
            kd = cand;
            break;
        }
    }
    CHECK(kd != NULL, "hardware-bound descriptor found by name");
    if (!kd) return;
    rvcl_kernel_t *k = NULL;
    const rvcl_status_t st = create(&k, kd);
    CHECK(st == RVCL_UNSUPPORTED,
            "by-name drive returns UNSUPPORTED on a non-IME host");
    CHECK(k == NULL, "no kernel is created on a non-IME host");
    rvcl_kernel_destroy(k);
}

int main(void) {
    const char *knob = getenv("RVCL_IME_KERNEL");
    if (!knob || !*knob) knob = "auto";
    printf("RVCL_IME_KERNEL='%s'\n", knob);

    const int available = ime_instruction_path_available();
    printf("host gate: instruction path %s\n",
            available ? "available" : "unavailable");

    check_by_name_gate_deny(available);

    ime_gemm_fn got;
    int saved = -1;

    if (!strcmp(knob, "auto") || !strcmp(knob, "c")) {
        got = ime_pick_gemm_f16();
        if (!strcmp(knob, "c")) {
            CHECK(got == ime_gemm_f16_c, "forced 'c' binds the C path");
        } else {
            CHECK(got != NULL, "auto resolves to a live path");
#if defined(__riscv) && defined(__riscv_vector)
            if (available)
                CHECK(got == ime_gemm_f16_ime,
                        "auto binds the instruction path when the host allows");
            else
                CHECK(got == ime_gemm_f16_c,
                        "auto binds the C path when the host does not allow");
#else
            CHECK(got == ime_gemm_f16_c,
                    "auto binds the C path (instruction path not compiled)");
#endif
        }
    } else if (!strcmp(knob, "ime")) {
#if defined(__riscv) && defined(__riscv_vector)
        FILE *cap = stderr_capture_begin(&saved);
        got = ime_pick_gemm_f16();
        const long err = stderr_capture_end(cap, saved);
        if (available) {
            CHECK(got == ime_gemm_f16_ime,
                    "forced 'ime' binds the instruction path");
        } else {
            CHECK(got == NULL,
                    "forced 'ime' on an unguarded host returns NULL");
            CHECK(err > 0, "forced 'ime' failure prints a stderr reason");
        }
#else
        FILE *cap = stderr_capture_begin(&saved);
        got = ime_pick_gemm_f16();
        const long err = stderr_capture_end(cap, saved);
        CHECK(got == NULL,
                "forced 'ime' without vector compilation returns NULL");
        CHECK(err > 0, "forced 'ime' failure prints a stderr reason");
#endif
    } else {
        FILE *cap = stderr_capture_begin(&saved);
        got = ime_pick_gemm_f16();
        const long err = stderr_capture_end(cap, saved);
        CHECK(got == NULL, "unknown knob value returns NULL");
        CHECK(err > 0, "unknown knob value prints a stderr reason");
    }

    /* The by-name resolver must agree with the knob-free resolver for
     * every value, and a NULL name fails loudly too. */
    FILE *cap = stderr_capture_begin(&saved);
    got = ime_pick_gemm_f16_by_name(NULL);
    const long err = stderr_capture_end(cap, saved);
    CHECK(got == NULL && err > 0, "NULL path name fails loudly");

    if (!strcmp(knob, "auto") || !strcmp(knob, "c") || !strcmp(knob, "ime")) {
        CHECK(ime_pick_gemm_f16_by_name(knob) == ime_pick_gemm_f16(),
                "by-name resolver agrees with the knob resolver");
    }

    printf(g_failures ? "IME paths tests: %d FAIL\n"
                      : "IME paths tests: all ok\n",
            g_failures);
    return g_failures ? 1 : 0;
}
