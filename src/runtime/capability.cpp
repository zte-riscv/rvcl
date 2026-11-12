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
 * capability.cpp — host capability detection (cached)
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

#include <pthread.h>
#include <stdio.h>
#include <string.h>

#include "capability_internal.h"
#include "rvcl/dispatch.h"

#include <signal.h>
#include <setjmp.h>

/* Test override state: when active, rvcl_get_capability() serves this
 * copy instead of the host snapshot. The buffer and the flag share one
 * mutex with every snapshot read, so an override install never tears. */
static rvcl_capability_t g_cap_override;
static int g_cap_overridden;
static pthread_mutex_t g_cap_mutex = PTHREAD_MUTEX_INITIALIZER;

/* -------------------------------------------------- vendor probe registry
 *
 * Vendor probes registered before the snapshot is built are invoked once
 * during build_capability(). g_probes_locked turns 1 the first time the
 * snapshot build runs, closing the registration window (see capability.h
 * for the contract); until then registration is mutex-protected and the
 * probe array grows append-only. The snapshot build itself runs under the
 * same mutex so a registration racing the first build is serialized either
 * before the probes are captured or after the window closes. */
static rvcl_vendor_probe_fn g_probes[RVCL_MAX_VENDOR_PROBES];
static int g_num_probes;
static int g_probes_locked;

/* ------------------------------------------------------- probe sanitizer
 *
 * Whatever a probe appends is filtered before it enters the snapshot:
 * only vendor-range ids are kept, an id already present is dropped
 * (append-only, never overwrite), and an over-full table rolls back to
 * the pre-probe count so a misbehaving probe cannot overflow or clobber
 * standard features. */
static void sanitize_probe_increment(rvcl_capability_t *cap, int num_before) {
    int n = num_before;
    for (int i = num_before; i < cap->num_features; i++) {
        if (cap->features[i].id < RVCL_FEATURE_VENDOR_FIRST) continue;
        int dup = 0;
        for (int j = 0; j < n; j++)
            if (cap->features[j].id == cap->features[i].id) dup = 1;
        if (dup) continue;
        if (n >= RVCL_MAX_FEATURES) break;
        cap->features[n++] = cap->features[i];
    }
    cap->num_features = n;
}

static void run_vendor_probes(rvcl_capability_t *cap) {
    /* Snapshot the probe array under the mutex so a registration racing
     * the single snapshot build is never read half-applied. After the
     * window closes no registration can add to the array, so the copy is
     * stable for the duration of the build. */
    rvcl_vendor_probe_fn probes[RVCL_MAX_VENDOR_PROBES];
    int num = 0;
    pthread_mutex_lock(&g_cap_mutex);
    for (int i = 0; i < g_num_probes; i++)
        probes[i] = g_probes[i];
    num = g_num_probes;
    pthread_mutex_unlock(&g_cap_mutex);

    for (int i = 0; i < num; i++) {
        const int num_before = cap->num_features;
        if (probes[i](cap) != RVCL_SUCCESS) {
            /* degrade, never error: discard this probe's whole increment */
            cap->num_features = num_before;
            continue;
        }
        sanitize_probe_increment(cap, num_before);
    }
}

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
#include <unistd.h>
#include <asm/hwprobe.h>
#include <sys/syscall.h>

/* The V bit macro is RISCV_HWPROBE_IMA_V in the original header and
 * RISCV_HWPROBE_EXT_V in newer spellings; both name the same frozen bit
 * position, so fall back across names and finally to the raw bit. */
#if !defined(RISCV_HWPROBE_IMA_V)
#if defined(RISCV_HWPROBE_EXT_V)
#define RISCV_HWPROBE_IMA_V RISCV_HWPROBE_EXT_V
#else
#define RISCV_HWPROBE_IMA_V (1 << 2)
#endif
#endif

/* Append-only merge rule of the runtime layer: a feature may be added
 * when absent, but nothing already present is ever removed or
 * downgraded — the compile-time view is a lower bound. */
static void cap_add_feature(rvcl_capability_t *cap, rvcl_feature_id_t id) {
    if (rvcl_capability_has(cap, id)) return;
    if (cap->num_features >= RVCL_MAX_FEATURES) return;
    cap->features[cap->num_features].id = id;
    cap->features[cap->num_features].value = 1;
    cap->num_features++;
}

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

    /* hwprobe may not surface new dtype extensions even on silicon
     * where they exist: each bit below can only add a feature, never
     * contradict the compile-time view. */
    const int has_rvv = (ext & RISCV_HWPROBE_IMA_V) != 0;
    if (has_rvv) cap_add_feature(cap, RVCL_FEATURE_RVV);

/* FP16 is a single feature bit: either Zvfh (native fp16 arithmetic)
 * or Zvfhmin (conversion-only) counts. The bit macros exist only in
 * newer kernel headers; on older ones this layer contributes nothing
 * for fp16 and the compile-time view stands. */
#if defined(RISCV_HWPROBE_EXT_ZVFH) || defined(RISCV_HWPROBE_EXT_ZVFHMIN)
    {
        int has_fp16 = 0;
#if defined(RISCV_HWPROBE_EXT_ZVFH)
        has_fp16 |= (ext & RISCV_HWPROBE_EXT_ZVFH) != 0;
#endif
#if defined(RISCV_HWPROBE_EXT_ZVFHMIN)
        has_fp16 |= (ext & RISCV_HWPROBE_EXT_ZVFHMIN) != 0;
#endif
        if (has_fp16) cap_add_feature(cap, RVCL_FEATURE_FP16);
    }
#endif
#if defined(RISCV_HWPROBE_EXT_ZVFBFMIN)
    if ((ext & RISCV_HWPROBE_EXT_ZVFBFMIN) != 0)
        cap_add_feature(cap, RVCL_FEATURE_BF16);
#endif

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
        if (cap->features[i].id == id) return &cap->features[i];
    return NULL;
}

/* Human-readable summary, shared by the host detection path and the
 * test override path so an overridden snapshot describes itself with
 * exactly the same semantics: the feature count, and the VLEN bits
 * whenever the RVV entry carries a readable width. */
static void format_description(rvcl_capability_t *cap) {
    const rvcl_feature_t *rvv = rvcl_capability_get(cap, RVCL_FEATURE_RVV);
    if (rvv && rvv->value > 1)
        snprintf(cap->description, sizeof(cap->description),
                "rvcl-capability: %d feature(s) detected, rvv vlen=%u",
                cap->num_features, rvv->value);
    else if (rvv)
        snprintf(cap->description, sizeof(cap->description),
                "rvcl-capability: %d feature(s) detected, rvv vlen=unknown",
                cap->num_features);
    else
        snprintf(cap->description, sizeof(cap->description),
                "rvcl-capability: %d feature(s) detected", cap->num_features);
}

/* Detection + description assembly, run exactly once by the magic-static
 * initializer in rvcl_get_capability(). A function-local static with a
 * non-trivial initializer is thread-safe in C++11 [stmt.dcl]: the standard
 * guarantees the initializer runs on exactly one thread and every other
 * thread blocks until it completes, which converges concurrent first calls
 * into a single detection pass. */
static rvcl_capability_t build_capability(void) {
    rvcl_capability_t cap;
    memset(&cap, 0, sizeof(cap));
    detect_compiletime(&cap);
    detect_runtime(&cap);
    run_vendor_probes(&cap);
    format_description(&cap);
    return cap;
}

int rvcl_capability_vlen_tier(const rvcl_capability_t *cap) {
    if (!cap) return RVCL_VLEN_TIER_ANY;
    const rvcl_feature_t *rvv = rvcl_capability_get(cap, RVCL_FEATURE_RVV);
    /* No RVV entry, or a width below 128 (including the
     * present-but-unknown value 1): no tier, only tier-agnostic kernels
     * pass a tier check against this snapshot. */
    if (!rvv || rvv->value < RVCL_VLEN_TIER_128) return RVCL_VLEN_TIER_ANY;
    if (rvv->value >= RVCL_VLEN_TIER_1024) return RVCL_VLEN_TIER_1024;
    if (rvv->value >= RVCL_VLEN_TIER_512) return RVCL_VLEN_TIER_512;
    if (rvv->value >= RVCL_VLEN_TIER_256) return RVCL_VLEN_TIER_256;
    return RVCL_VLEN_TIER_128;
}

void rvcl_capability_test_override(const rvcl_capability_t *cap) {
    pthread_mutex_lock(&g_cap_mutex);
    if (cap) {
        g_cap_override = *cap;
        /* The copy keeps the injected features but gets a
         * library-generated summary: consumers must not be able to tell
         * an overridden snapshot from a detected one by its formatting. */
        format_description(&g_cap_override);
        g_cap_overridden = 1;
        /* The injected snapshot fixes the process capability view: close
         * the probe window so a later registration is rejected instead of
         * silently having no effect on the served snapshot. */
        g_probes_locked = 1;
    } else {
        g_cap_overridden = 0;
    }
    pthread_mutex_unlock(&g_cap_mutex);
    /* The selection input changed while dispatch keys encode only the
     * request: drop cached selections so the next dispatch reflects the
     * now-active snapshot, in both the override and the restore
     * direction. */
    rvcl_dispatch_cache_clear();
}

const rvcl_capability_t *rvcl_get_capability(void) {
    pthread_mutex_lock(&g_cap_mutex);
    if (g_cap_overridden) {
        const rvcl_capability_t *overridden = &g_cap_override;
        pthread_mutex_unlock(&g_cap_mutex);
        return overridden;
    }
    /* First real snapshot access closes the probe registration window
     * (capability.h contract): the snapshot content is fixed from here on,
     * so a later registration cannot change what dispatch observes. */
    g_probes_locked = 1;
    pthread_mutex_unlock(&g_cap_mutex);

    static rvcl_capability_t g_cap = build_capability();
    return &g_cap;
}

rvcl_status_t rvcl_capability_register_vendor_probe(
        rvcl_vendor_probe_fn probe) {
    if (!probe) return RVCL_ERROR_INVALID_ARGUMENT;
    pthread_mutex_lock(&g_cap_mutex);
    if (g_probes_locked) {
        pthread_mutex_unlock(&g_cap_mutex);
        return RVCL_ERROR_ALREADY_INITIALIZED;
    }
    if (g_num_probes >= RVCL_MAX_VENDOR_PROBES) {
        pthread_mutex_unlock(&g_cap_mutex);
        return RVCL_ERROR_OUT_OF_MEMORY;
    }
    g_probes[g_num_probes++] = probe;
    pthread_mutex_unlock(&g_cap_mutex);
    return RVCL_SUCCESS;
}

/* SIGILL-guarded csrr: the CSR may not exist on this core, and capability
 * must degrade rather than crash. The signal handler is process-global, so
 * the probe-time (single-threaded snapshot build) usage is the supported
 * pattern; a concurrent csrr elsewhere could share the handler slot. */
#if defined(__riscv)
static sigjmp_buf g_csr_jmp;
static volatile sig_atomic_t g_csr_trapped;

static void csr_trap_handler(int) {
    g_csr_trapped = 1;
    siglongjmp(g_csr_jmp, 1);
}

int rvcl_capability_try_csr(unsigned int csr, uint32_t *value) {
    if (!value) return 0;
    struct sigaction sa, old;
    memset(&sa, 0, sizeof(sa));
    sa.sa_handler = csr_trap_handler;
    sigemptyset(&sa.sa_mask);
    if (sigaction(SIGILL, &sa, &old) != 0) return 0;
    if (sigsetjmp(g_csr_jmp, 1) == 0) {
        uint32_t v = 0;
        /* csrr takes a compile-time CSR offset, so each supported CSR is read
         * through its own immediate; an unknown CSR degrades to "absent"
         * rather than trapping. The probe/example use 0x7c0 and 0xCE4. */
        switch (csr) {
            case 0x7c0: __asm__ volatile("csrr %0, 0x7c0" : "=r"(v)); break;
            case 0xCE4: __asm__ volatile("csrr %0, 0xCE4" : "=r"(v)); break;
            default: (void)sigaction(SIGILL, &old, NULL); return 0;
        }
        (void)sigaction(SIGILL, &old, NULL);
        *value = v;
        return 1;
    }
    (void)sigaction(SIGILL, &old, NULL);
    return 0;
}
#else
/* Must match the riscv signature that writes *value; this stub just never
 * dereferences it. */
// NOLINTNEXTLINE(readability-non-const-parameter)
int rvcl_capability_try_csr(unsigned int csr, uint32_t *value) {
    (void)csr;
    (void)value;
    return 0; /* non-RISC-V hosts have no CSR instruction */
}
#endif
