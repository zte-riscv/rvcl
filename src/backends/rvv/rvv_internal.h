/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * rvv_internal.h — shared declarations inside the RVV backend
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * The RVV backend keeps three layers (PLAN §4.6):
 *   - pack: plain -> panel layout conversion (RVV-friendly);
 *   - micro-kernel: fixed-shape vectorized inner loops;
 *   - framework adapter: validate/configure/run + registration.
 *
 * Every translation unit compiles on any host. The vectorized paths are
 * guarded by __riscv_vector; hosts without it fall back to the C-portable
 * loops so framework logic (packing, registry, dispatch cache, tests)
 * stays verifiable on x86 CI (PLAN §6.2 item 12). A build that lands on
 * real RVV silicon gets the intrinsics through the same descriptors.
 ******************************************************************************/

#ifndef RVCL_RVV_INTERNAL_H
#define RVCL_RVV_INTERNAL_H

#include "rvcl/kernel.h"
#include "rvcl/registry.h"
#include "rvcl/types.h"

#include <stddef.h>
#include <stdint.h>

/* ------------------------------------------------------------- pack format
 *
 * Panel layout (identical contract for f32 and f16 kernels):
 *   B[k,n] is repacked into column panels of width RVV_PACK_NC:
 *     panel p covers columns [p*NC, (p+1)*NC)
 *     inside a panel, memory is n-major: [k][nc]
 *   Rationale (PLAN §4.6): the micro-kernel streams contiguous B rows
 *   (K direction) while holding an (MR x NC) accumulator tile, so the
 *   innermost K loop touches unit-stride memory in both operands.
 *
 * A is consumed unpacked (row-major) in v0.1; pack_a exists for API
 * completeness and is a metadata copy.
 */

#define RVV_PACK_NC 8 /* panel width: 8 columns (2 f32 vectors at VLEN=128) */

/* Workspace layout index for the packed-B buffer (slot AUX_0). */
#define RVV_WS_PACKED_B 0
#define RVV_WS_ACC      1

/* bytes needed for the packed B panel buffer: k x n elements, panelized */
static inline size_t rvv_packed_b_bytes(int64_t k, int64_t n, size_t elt) {
    int64_t panels = (n + RVV_PACK_NC - 1) / RVV_PACK_NC;
    return (size_t)(k * panels * RVV_PACK_NC) * elt;
}

/* destination element index (in elements) inside the packed buffer */
static inline int64_t rvv_packed_b_index(int64_t k_dim, int64_t col,
        int64_t row) {
    int64_t panel = col / RVV_PACK_NC;
    int64_t in = col - panel * RVV_PACK_NC;
    return (panel * k_dim + row) * RVV_PACK_NC + in;
}

/* ---------------------------------------------------- micro-kernel entry
 * C-portable implementations (always compiled; also the x86 CI path).
 * Exported non-statically so the benchmark harness can drive them
 * directly for kernel-only timing.
 */
void rvv_gemm_f32_c(const void *a, const void *b_packed, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);
void rvv_gemm_f16_c(const void *a, const void *b_packed, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);

/* Panel packer (rvv_pack.c): plain B -> panel layout, dtype-agnostic. */
rvcl_status_t rvv_pack_b_panel(const rvcl_tensor_t *src, rvcl_tensor_t *dst,
        int64_t m, int64_t n, int64_t k_dim, size_t elt);

/* Intrinsics implementations (compiled only under __riscv_vector). */
#if defined(__riscv_vector)
void rvv_gemm_f32_rvv(const void *a, const void *b_packed, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);
void rvv_gemm_f16_rvv(const void *a, const void *b_packed, const void *bias,
        void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);
#endif

/* ------------------------------------------------------- backend registration */

RVCL_API rvcl_status_t rvcl_rvv_register(void);

/* Micro-kernel function type (backend adapter holds one per dtype). */
typedef void (*rvv_gemm_fn)(const void *a, const void *b_packed,
        const void *bias, void *c, int64_t m, int64_t n, int64_t k_dim,
        const rvcl_work_range_t *r);

#endif /* RVCL_RVV_INTERNAL_H */
