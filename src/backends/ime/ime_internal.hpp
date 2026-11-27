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

/* The IME backend targets the matrix-multiply-accumulate instruction of the
 * integrated matrix extension: one instruction consumes an 8x8 fp16 A tile
 * and an 8x8 fp16 B tile and accumulates into an 8x8 fp32 C tile held in a
 * vector register pair. Three layers, mirroring the RVV backend:
 *   - pack: plain row-major B -> tile layout (column-of-K contiguous);
 *   - micro-kernel: fixed 8x8x8 tile loops (C-portable + instruction path);
 *   - framework adapter: validate/configure/run + registration.
 *
 * Numeric contract: inputs are finite normal fp16 values. Subnormal inputs
 * contribute zero on the instruction path (flush-to-zero), and NaN/Inf
 * propagation is not guaranteed; callers needing those semantics must
 * convert or reject upstream.
 *
 * Every translation unit compiles on any host; the instruction path is
 * guarded by __riscv_vector, so framework-side code (packing, selection,
 * tests) stays verifiable on any CI host.
 */

#ifndef RVCL_IME_INTERNAL_H
#define RVCL_IME_INTERNAL_H

#include "rvcl/kernel.h"
#include "rvcl/registry.h"
#include "rvcl/types.h"

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------- tile geometry
 *
 * One instruction processes M x K x N = 8 x 8 x 8:
 *   A tile: 8 rows x 8 K elements, row-major (vs1);
 *   B tile: 8 output columns x 8 K elements, each column's K values
 *           contiguous (vs2);
 *   C tile: 8 x 8 fp32, row-major, occupying an aligned register pair.
 *
 * The instruction path assumes VLEN == 1024 bits: exactly one 8x8 fp16
 * tile fits in one vector register, and one 8x8 fp32 tile fills a register
 * pair. A wider unit would not fit, so the run-time capability gate admits
 * only the exact width.
 */
#define IME_M_TILE 8
#define IME_N_TILE 8
#define IME_K_TILE 8
#define IME_TILE_ELT 64 /* elements per tile */
#define IME_TILE_BYTES 128 /* A/B tile size in bytes (fp16) */

/* Workspace layout index for the packed-B tile buffer (slot AUX_0). */
#define IME_WS_PACKED_B 0

/* ---------------------------------------------------- micro-kernel entry
 * C-portable implementation (always compiled; also the any-host CI path).
 * Exported non-statically so tests and the benchmark harness can drive
 * them directly. Signature is parameter-for-parameter identical to the
 * RVV backend's rvv_gemm_fn so the two backends stay interchangeable at
 * the adapter layer.
 */
void ime_gemm_f16_c(const void *a, const void *b_packed, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);

/* Instruction path (compiled only under __riscv_vector). */
#if defined(__riscv_vector)
void ime_gemm_f16_ime(const void *a, const void *b_packed, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);
#endif

/* Micro-kernel function type (adapter holds one per dtype). */
typedef void (*ime_gemm_fn)(const void *a, const void *b_packed,
        const void *bias, void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);

/* ------------------------------------------------------- pack helpers
 *
 * Tile layout of the packed B buffer (nt = n tile, kt = k tile):
 *   N_T = ceil(N / 8), K_T = ceil(K / 8)
 *   idx(nt, kt, nn, kk) = (nt * K_T + kt) * 64 + nn * 8 + kk
 *   dst[idx] = B[(kt*8 + kk) * N + (nt*8 + nn)]   when both in range
 *            = 0                                   otherwise (tail fill)
 * nt runs outer and kt inner, so the K loop of the micro-kernel reads
 * consecutive tiles linearly. The buffer holds N_T * K_T * 64 elements
 * of elt bytes each, with no extra padding.
 */

/* Bytes needed for the packed B tile buffer; 0 for invalid sizes or when
 * the byte count would overflow (callers treat 0 as unsatisfiable). */
size_t ime_packed_b_bytes(int64_t k_dim, int64_t n, size_t elt);

/* Destination element index of B[k][n] in the packed buffer; -1 when the
 * element is out of range. */
int64_t ime_packed_b_index(int64_t n_dim, int64_t k_dim, int64_t k, int64_t n);

/* Plain row-major B -> tile layout. Pure data movement: no arithmetic, no
 * allocation, any host. */
rvcl_status_t ime_pack_b_tiles(const rvcl_tensor_t *src, rvcl_tensor_t *dst,
        int64_t m, int64_t n, int64_t k_dim, size_t elt);

/* ------------------------------------------------- micro-kernel selection
 * Single source of truth for "which f16 kernel is live": the backend
 * adapter and any kernel-only benchmark call the same selectors, so what
 * runs and what gets measured can never drift apart. The knob
 * RVCL_IME_KERNEL (auto | c | ime, read once per process) resolves here;
 * an invalid or unavailable path fails loudly with a NULL return. */
ime_gemm_fn ime_pick_gemm_f16(void);
ime_gemm_fn ime_pick_gemm_f16_by_name(const char *name);

/* Hard run-time gate: the instruction path is compiled in AND the host
 * capability snapshot reports the floating-point matrix extension.
 * Detection caches once per process and never surfaces a feature the
 * host cannot execute; test-injected snapshots are ignored. */
int ime_instruction_path_available(void);

/* ------------------------------------------------------- registration
 * rvcl_ime_register() registers the hardware-bound entry on RISC-V
 * builds (the entry is registered in every IME build; availability is
 * decided by the run-time feature filter). rvcl_ime_register_portable()
 * registers the portable C entry on non-RISC-V builds. Each is a no-op
 * where its entry does not belong. init.cpp calls both before the RVV
 * registration so equal-cost ties resolve to the more specialized entry. */
rvcl_status_t rvcl_ime_register(void);
rvcl_status_t rvcl_ime_register_portable(void);

#endif /* RVCL_IME_INTERNAL_H */
