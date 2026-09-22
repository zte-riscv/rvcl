/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * capability.c — host capability detection (cached)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Detection layers (PLAN §4.4): compile-time __riscv_* macros first, then
 * runtime hwprobe/auxv when available. Failures degrade to the RVV
 * baseline; capability never selects kernels.
 ******************************************************************************/

#include "rvcl/capability.h"

#include <stdio.h>
#include <string.h>

#if defined(__riscv)
#include <sys/auxv.h>
#endif

static rvcl_capability_t g_cap;
static int g_cap_initialized;

/* Compile-time knowledge of the build target. */
static void detect_compiletime(rvcl_capability_t *cap) {
    int n = 0;
#if defined(__riscv) && defined(__riscv_xlen) && (__riscv_xlen == 64)
    cap->features[n].id = RVCL_FEATURE_RV64;
    cap->features[n].value = 1;
    n++;
#endif
#if defined(__riscv_vector)
    cap->features[n].id = RVCL_FEATURE_RVV;
    cap->features[n].value = 0; /* VLEN filled at runtime when detectable */
    n++;
#endif
#if defined(__riscv_zvfh) || defined(__riscv_zvfhmin)
    cap->features[n].id = RVCL_FEATURE_FP16;
    cap->features[n].value = 1;
    n++;
#endif
#if defined(__riscv_zvfbfmin)
    cap->features[n].id = RVCL_FEATURE_BF16;
    cap->features[n].value = 1;
    n++;
#endif
    cap->num_features = n;
}

/* Runtime layer: hwprobe (Linux 6.4+). Absent or failing probes keep the
 * compile-time view; detection failure degrades instead of erroring. */
#if defined(__riscv) && defined(__linux__)
#include <asm/hwprobe.h>
#include <sys/syscall.h>
#include <unistd.h>

static void detect_runtime(rvcl_capability_t *cap) {
    struct riscv_hwprobe pairs[] = {
        {RISCV_HWPROBE_KEY_IMA_EXT_0, 0},
        {RISCV_HWPROBE_KEY_MVENDORID, 0},
    };
    if (syscall(__NR_riscv_hwprobe, pairs, 2, 0, 0, 0) != 0) return;

    const uint64_t ext = pairs[0].value;
    int n = cap->num_features;

    int has_rvv = (ext & RISCV_HWPROBE_IMA_V) != 0;
    int seen_rvv = 0;
    for (int i = 0; i < n; i++) {
        if (cap->features[i].id == RVCL_FEATURE_RVV) {
            seen_rvv = 1;
            /* HWPROBE may not surface new dtype extensions even on silicon
             * (PLAN §4.5.6): only upgrade here, never downgrade. */
            if (has_rvv) cap->features[i].value = 1;
        }
    }
    if (has_rvv && !seen_rvv && n < RVCL_MAX_FEATURES) {
        cap->features[n].id = RVCL_FEATURE_RVV;
        cap->features[n].value = 1;
        n++;
    }
    cap->num_features = n;
}
#else
static void detect_runtime(rvcl_capability_t *cap) {
    (void)cap; /* non-Linux/non-RISC-V hosts: compile-time view only */
}
#endif

int rvcl_capability_has(const rvcl_capability_t *cap, rvcl_feature_id_t id) {
    if (!cap) return 0;
    for (int i = 0; i < cap->num_features; i++)
        if (cap->features[i].id == id) return 1;
    return 0;
}

const rvcl_feature_t *rvcl_capability_get(
        const rvcl_capability_t *cap, rvcl_feature_id_t id) {
    if (!cap) return NULL;
    for (int i = 0; i < cap->num_features; i++)
        if (cap->features[i].id == id)
            return &cap->features[i];
    return NULL;
}

const rvcl_capability_t *rvcl_get_capability(void) {
    /* Simple once-guard. Concurrent first calls may both run detection;
     * the result is identical, so the race is benign for Phase 0. A
     * call_once port lands with the thread-safety hardening task. */
    if (g_cap_initialized) return &g_cap;

    memset(&g_cap, 0, sizeof(g_cap));
    detect_compiletime(&g_cap);
    detect_runtime(&g_cap);

    snprintf(g_cap.description, sizeof(g_cap.description),
            "rvcl-capability: %d feature(s) detected", g_cap.num_features);

    g_cap_initialized = 1;
    return &g_cap;
}
