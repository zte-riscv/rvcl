/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * capability.cpp — host capability detection (cached)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Detection layers (PLAN §4.4): compile-time __riscv_* macros first, then
 * runtime hwprobe/auxv when available. Failures degrade to the RVV
 * baseline; capability never selects kernels.
 *
 * Feature value semantics (unified across both layers): a present
 * feature carries 1 ("present, no further detail") or an informative
 * value. For RVCL_FEATURE_RVV the informative value is the vector
 * length in bits (128/256/...) read from the vlenb CSR; it stays 1 when
 * the read is unavailable (present-but-unknown). Presence is queried
 * via rvcl_capability_has() — never inferred from the value.
 ******************************************************************************/

#include "rvcl/capability.h"

#include <stdio.h>
#include <string.h>

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
    cap->features[n].value = 1; /* VLEN upgraded at runtime when readable */
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

/* ------------------------------------------------------------- VLEN probe
 *
 * Reads VLEN in bits from the vlenb CSR (VLEN/8, U-mode readable
 * whenever V is implemented). The read is gated on V being confirmed
 * present — by the compile-time __riscv_vector macro or by a runtime
 * hwprobe hit — because csrr traps as an illegal instruction on V-less
 * cores and capability must degrade, not crash. A nonsense reading
 * (vlenb is a power of two, at most VLEN 65536 bits) keeps value=1.
 */
#if defined(__riscv)
static void fill_rvv_vlen(rvcl_capability_t *cap) {
    for (int i = 0; i < cap->num_features; i++) {
        if (cap->features[i].id != RVCL_FEATURE_RVV) continue;
        unsigned long vlenb;
        __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
        if (vlenb >= 1 && vlenb <= 8192 && (vlenb & (vlenb - 1)) == 0)
            cap->features[i].value = (uint32_t)vlenb * 8;
        break;
    }
}
#endif

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
    if (syscall(__NR_riscv_hwprobe, pairs, 2, 0, 0, 0) != 0) {
        /* No hwprobe (pre-6.4 kernel, or an emulator without the
         * syscall): keep the compile-time view. A compile-time V build
         * still implies V on the host — the binary would fault on any
         * vector instruction otherwise — so the VLEN read stays safe. */
#if defined(__riscv_vector)
        fill_rvv_vlen(cap);
#endif
        return;
    }

    const uint64_t ext = pairs[0].value;
    int n = cap->num_features;

    int has_rvv = (ext & RISCV_HWPROBE_IMA_V) != 0;
    int seen_rvv = 0;
    for (int i = 0; i < n; i++) {
        if (cap->features[i].id == RVCL_FEATURE_RVV) {
            seen_rvv = 1;
            /* HWPROBE may not surface new dtype extensions even on silicon
             * (PLAN §4.5.6): only upgrade here, never downgrade. */
        }
    }
    if (has_rvv && !seen_rvv && n < RVCL_MAX_FEATURES) {
        cap->features[n].id = RVCL_FEATURE_RVV;
        cap->features[n].value = 1;
        n++;
        cap->num_features = n;
    }
    /* VLEN upgrade: safe whenever V is confirmed. hwprobe reporting V is
     * the runtime confirmation (baseline-tier builds); the compile-time
     * macro covered the no-hwprobe return above. When hwprobe explicitly
     * reports no V on a compile-time-V build the environment is broken
     * — keep the degraded value=1 rather than trusting a CSR read. */
    if (has_rvv) fill_rvv_vlen(cap);
}
#else
static void detect_runtime(rvcl_capability_t *cap) {
#if defined(__riscv_vector)
    /* non-Linux RISC-V hosts: compile-time view only, but a -march=...v
     * build implies V on the host, so the VLEN read is safe */
    fill_rvv_vlen(cap);
#else
    (void)cap; /* non-RISC-V hosts: compile-time view only */
#endif
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

    /* Human-readable summary; the RVV entry carries the VLEN when the
     * vlenb read succeeded (value > 1). */
    const rvcl_feature_t *rvv = rvcl_capability_get(&g_cap, RVCL_FEATURE_RVV);
    if (rvv && rvv->value > 1)
        snprintf(g_cap.description, sizeof(g_cap.description),
                "rvcl-capability: %d feature(s) detected, rvv vlen=%u",
                g_cap.num_features, rvv->value);
    else if (rvv)
        snprintf(g_cap.description, sizeof(g_cap.description),
                "rvcl-capability: %d feature(s) detected, rvv vlen=unknown",
                g_cap.num_features);
    else
        snprintf(g_cap.description, sizeof(g_cap.description),
                "rvcl-capability: %d feature(s) detected",
                g_cap.num_features);

    g_cap_initialized = 1;
    return &g_cap;
}
