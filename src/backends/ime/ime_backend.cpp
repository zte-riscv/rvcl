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
 * ime_backend.cpp — IME backend: micro-kernel selection + host gate
 *
 * This translation unit owns the f16 path selection. The knob
 * RVCL_IME_KERNEL (auto | c | ime, read once per process) resolves through
 * the path table in ime_paths.hpp; an unknown name or a forced-but-
 * unavailable path fails loudly (stderr reason, NULL return), never
 * silently binds a different path.
 *
 * The host gate (ime_instruction_path_available) is intentionally
 * self-contained here so the backend stays usable before the framework
 * wires its capability snapshot: the instruction path may execute only
 * when it is compiled in AND every CPU the process may run on carries the
 * target matrix profile with a wide enough vector unit. The check is
 * conservative — any detection failure reports unavailable — because
 * executing the instruction on a core without it raises an illegal
 * instruction fault. The result is cached once per process (the process
 * affinity mask is expected to be pinned before the first query).
 *
 * The framework adapter (validate/configure/run, descriptor registration)
 * extends this file when the backend is wired into the registry.
 ******************************************************************************/

#include "ime_internal.hpp"
#include "ime_paths.hpp"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__riscv) && defined(__linux__) && defined(__riscv_vector)
#include <fcntl.h>
#include <sched.h>
#include <unistd.h>

/* Device-tree compatible token of the CPU profile that carries the
 * floating-point matrix extension this backend targets; CPUs are matched
 * by exact token comparison against their of_node/compatible property. */
#define IME_DT_COMPATIBLE "spacemit,a100"

/* Minimum vector length (bits) the tile geometry of the instruction path
 * is built for. */
#define IME_MIN_VLEN_BITS 1024

/* Answers whether the given CPU's of_node/compatible property lists the
 * target profile token. The property is a NUL-separated token list; a
 * missing or unreadable node counts as no match. */
static int ime_cpu_matches_profile(int cpu) {
    char path[96];
    char buf[512];
    snprintf(path, sizeof(path),
            "/sys/devices/system/cpu/cpu%d/of_node/compatible", cpu);
    const int fd = open(path, O_RDONLY);
    if (fd < 0) return 0;
    const ssize_t got = read(fd, buf, sizeof(buf) - 1);
    close(fd);
    if (got <= 0) return 0;
    buf[got] = '\0';

    const size_t toklen = strlen(IME_DT_COMPATIBLE);
    size_t off = 0;
    while (off < (size_t)got) {
        const size_t len = strnlen(buf + off, (size_t)got - off);
        if (len == toklen && memcmp(buf + off, IME_DT_COMPATIBLE, len) == 0)
            return 1;
        off += len + 1;
    }
    return 0;
}

/* Vector length in bits; 0 when the CSR value is not a sane power of two
 * in the supported range. The compile-time __riscv_vector guard of this
 * block already proves the binary requires the vector extension. */
static int ime_vlen_bits(void) {
    unsigned long vlenb;
    __asm__ volatile("csrr %0, vlenb" : "=r"(vlenb));
    if (vlenb == 0 || vlenb > 8192 || (vlenb & (vlenb - 1)) != 0) return 0;
    return (int)(vlenb * 8);
}

static int ime_host_probe(void) {
    /* The process may migrate within its affinity mask: every allowed CPU
     * must carry the extension, otherwise the answer is no. */
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) != 0) return 0;
    if (CPU_COUNT(&set) == 0) return 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        if (!CPU_ISSET(cpu, &set)) continue;
        if (!ime_cpu_matches_profile(cpu)) return 0;
    }
    return ime_vlen_bits() >= IME_MIN_VLEN_BITS;
}

#endif /* riscv + linux + vector */

__attribute__((visibility("default"))) int ime_instruction_path_available(
        void) {
#if defined(__riscv) && defined(__linux__) && defined(__riscv_vector)
    /* cached once per process; the C++11 static guard makes concurrent
     * first calls safe */
    static const int available = ime_host_probe();
    return available;
#else
    return 0; /* instruction path not compiled in on this target */
#endif
}

/* ------------------------------------------------- micro-kernel selection */

/* RVCL_IME_KERNEL knob, read once per process. Values: auto | c | ime.
 * An empty value counts as unset, matching the RVV knob's behavior. */
static const char *ime_kernel_knob(void) {
    static const char *knob = getenv("RVCL_IME_KERNEL");
    return knob;
}

__attribute__((visibility("default"))) ime_gemm_fn ime_pick_gemm_f16_by_name(
        const char *name) {
    if (!name) {
        fprintf(stderr,
                "rvcl: f16 kernel path is NULL (want auto, c or ime)\n");
        return NULL;
    }
    return ime_path_by_name(&ime_f16_paths, name);
}

__attribute__((visibility("default"))) ime_gemm_fn ime_pick_gemm_f16(void) {
    const char *knob = ime_kernel_knob();
    if (knob && *knob) return ime_pick_gemm_f16_by_name(knob);
    return ime_pick_gemm_f16_by_name("auto");
}
