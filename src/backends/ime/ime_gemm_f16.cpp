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

/* Tuple (f16,f16,f32,f32): A/B stored fp16, accumulation and output fp32.
 * Both paths implement the same tile semantics on the tile-packed B
 * buffer produced by ime_pack_b_tiles:
 *
 *   C[i][j] = bias[j]? + sum_k A[i][k] * B[k][j]      (K always full)
 *
 * The C path converts fp16 operands with the bit-exact soft conversions
 * (rvv_fp16.hpp) so it runs on any host and serves as the correctness
 * reference. The instruction path issues one matrix multiply-accumulate
 * per (m-tile, k-tile) after staging the A sub-block into the layout the
 * instruction consumes; results may differ from the C path only in the
 * rounding order of the 8-element dot product.
 *
 * Work ranges: tile starts are rounded down to the tile grid (a tile may
 * straddle a slice boundary and be recomputed), but stores are filtered
 * to the exact [m_begin,m_end) x [n_begin,n_end) window and the problem
 * bounds — no element outside the requested range is ever written. K is
 * never sliced.
 */

#include "../rvv/rvv_fp16.hpp"
#include "ime_internal.hpp"

#include <string.h>

/* ------------------------------------------------------- shared tile logic */

/* Builds the 8x8 fp32 bias tile for the n tile at n0: every row repeats
 * the same 8 bias values, columns beyond n read as zero. */
static void ime_bias_tile(
        float *dst, const float *bias, int64_t n0, int64_t n) {
    for (int64_t i = 0; i < IME_M_TILE; i++)
        for (int64_t j = 0; j < IME_N_TILE; j++)
            dst[i * IME_N_TILE + j] = (n0 + j < n) ? bias[n0 + j] : 0.0f;
}

/* Stores one 8x8 fp32 accumulator tile into C, filtered to the work range
 * intersected with the problem bounds; out-of-window elements are never
 * written (tail rows/columns of a recomputed tile hold bias values, so
 * skipping them is also what keeps the write in bounds). */
static void ime_store_tile(float *c, const float *tile, int64_t m, int64_t n,
        int64_t m0, int64_t n0, const rvcl_work_range_t *r) {
    for (int64_t i = 0; i < IME_M_TILE; i++) {
        const int64_t gi = m0 + i;
        if (gi < r->m_begin || gi >= r->m_end || gi >= m) continue;
        for (int64_t j = 0; j < IME_N_TILE; j++) {
            const int64_t gj = n0 + j;
            if (gj < r->n_begin || gj >= r->n_end || gj >= n) continue;
            c[gi * n + gj] = tile[i * IME_N_TILE + j];
        }
    }
}

/* ------------------------------------------------------ C-portable kernel */

__attribute__((visibility("default"))) void ime_gemm_f16_c(const void *a_in,
        const void *b_packed, const void *bias_in, void *c_in, int64_t m,
        int64_t n, int64_t k_dim, const rvcl_work_range_t *r) {
    const uint16_t *a = (const uint16_t *)a_in;
    const uint16_t *bp = (const uint16_t *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;

    const int64_t k_t = k_dim / IME_K_TILE + (k_dim % IME_K_TILE != 0);
    float acc[IME_TILE_ELT];
    uint16_t at[IME_TILE_ELT];

    for (int64_t m0 = (r->m_begin / IME_M_TILE) * IME_M_TILE; m0 < r->m_end;
            m0 += IME_M_TILE) {
        const int64_t rows = (m - m0 < IME_M_TILE) ? m - m0 : IME_M_TILE;
        for (int64_t n0 = (r->n_begin / IME_N_TILE) * IME_N_TILE; n0 < r->n_end;
                n0 += IME_N_TILE) {
            if (bias) {
                ime_bias_tile(acc, bias, n0, n);
            } else {
                memset(acc, 0, sizeof(acc));
            }
            for (int64_t kt = 0; kt < k_t; kt++) {
                /* stage the A sub-block, zero-filled outside the problem */
                const int64_t k0 = kt * IME_K_TILE;
                for (int64_t i = 0; i < IME_M_TILE; i++)
                    for (int64_t q = 0; q < IME_K_TILE; q++) {
                        const int64_t gk = k0 + q;
                        at[i * IME_K_TILE + q] = (i < rows && gk < k_dim)
                                ? a[(m0 + i) * k_dim + gk]
                                : 0;
                    }
                const uint16_t *bt
                        = bp + (n0 / IME_N_TILE * k_t + kt) * IME_TILE_ELT;
                for (int64_t i = 0; i < IME_M_TILE; i++)
                    for (int64_t j = 0; j < IME_N_TILE; j++) {
                        float sum = acc[i * IME_N_TILE + j];
                        for (int64_t q = 0; q < IME_K_TILE; q++)
                            sum += rvv_f16_to_f32(at[i * IME_K_TILE + q])
                                    * rvv_f16_to_f32(bt[j * IME_K_TILE + q]);
                        acc[i * IME_N_TILE + j] = sum;
                    }
            }
            ime_store_tile(c, acc, m, n, m0, n0, r);
        }
    }
}

/* --------------------------------------------------- IME instruction kernel */

#if defined(__riscv_vector)

/* Processes one m tile against a group of up to two adjacent n tiles in a
 * single assembly block: the accumulators live in fixed vector registers
 * across the whole K loop, so no compiler-generated code may run between
 * the initialization and the final store. Per k tile the block stages the
 * A sub-block into a 128-byte stack buffer (scalar copies; the buffer is
 * zero-filled first so tail rows/columns stay zero), loads the B tile(s)
 * and issues the matrix multiply-accumulate; the staged A data is shared
 * by both B tiles of the group, halving the staging traffic.
 *
 * Register assignment (v0/v1 stay untouched): acc0 = v2/v3, acc1 = v4/v5,
 * A tile = v6, B tile = v7. Every vsetvli writes vl to a GPR: discarding
 * the vl result while also taking the AVL from x0 is not a universally
 * accepted encoding, and the safe form costs nothing.
 *
 * Stack use is fixed-size: one 128-byte staging buffer here, plus the
 * caller's bias/out buffers (256 bytes per n tile, time-shared — the
 * bias tile is consumed into the accumulator before the store reuses the
 * same memory).
 */
__attribute__((noinline)) static void ime_tile_group(const uint16_t *a_row0,
        int64_t lda, const uint16_t *b0, const uint16_t *b1, int64_t k_tiles,
        int64_t rows, int64_t tail_k, const float *bias0, const float *bias1,
        float *out0, float *out1, int has_second) {
    uint16_t stage[IME_TILE_ELT] __attribute__((aligned(64)));
    const uint16_t *ap = a_row0, *b0p = b0, *b1p = b1;
    int64_t kt = k_tiles, rowcnt = 0, i = 0, cnt = 0;

    __asm__ volatile(
            ".option push\n"
            ".option arch, +xsmtvdotii\n"
            /* init accumulators: bias tile or zero, both registers of a pair */
            "vsetvli t0, zero, e32, m2, ta, ma\n"
            "beqz %[bias0], 1f\n"
            "vle32.v v2, (%[bias0])\n"
            "beqz %[has_second], 2f\n"
            "vle32.v v4, (%[bias1])\n"
            "j 2f\n"
            "1:\n"
            "vmv.v.i v2, 0\n"
            "beqz %[has_second], 2f\n"
            "vmv.v.i v4, 0\n"
            "2:\n"
            "vsetvli t0, zero, e16, m1, ta, ma\n"
            /* K tile loop: the last tile copies only tail_k columns */
            "3:\n"
            "  li %[rowcnt], 8\n"
            "  beqz %[tailk], 4f\n"
            "  addi t1, %[kt], -1\n"
            "  bnez t1, 4f\n"
            "  mv %[rowcnt], %[tailk]\n"
            "4:\n"
            /* zero-fill the staging buffer (tail padding + dead rows) */
            "  vmv.v.i v6, 0\n"
            "  vse16.v v6, (%[stage])\n"
            "  li %[i], 0\n"
            /* stage rows: row i copies rowcnt halfwords from A[m0+i][k0] */
            "5:\n"
            "  bge %[i], %[rows], 7f\n"
            "  slli t1, %[i], 1\n"
            "  mul t1, t1, %[lda]\n"
            "  add t1, %[ap], t1\n"
            "  slli t2, %[i], 4\n"
            "  add t2, %[stage], t2\n"
            "  mv %[cnt], %[rowcnt]\n"
            "6:\n"
            "  lh t0, 0(t1)\n"
            "  sh t0, 0(t2)\n"
            "  addi t1, t1, 2\n"
            "  addi t2, t2, 2\n"
            "  addi %[cnt], %[cnt], -1\n"
            "  bnez %[cnt], 6b\n"
            "  addi %[i], %[i], 1\n"
            "  j 5b\n"
            "7:\n"
            "  vle16.v v6, (%[stage])\n"
            "  vle16.v v7, (%[b0])\n"
            "  smt.vfwmadot v2, v6, v7\n"
            "  beqz %[has_second], 8f\n"
            "  vle16.v v7, (%[b1])\n"
            "  smt.vfwmadot v4, v6, v7\n"
            "  addi %[b1], %[b1], 128\n"
            "8:\n"
            "  addi %[b0], %[b0], 128\n"
            "  addi %[ap], %[ap], 16\n"
            "  addi %[kt], %[kt], -1\n"
            "  bnez %[kt], 3b\n"
            "vsetvli t0, zero, e32, m2, ta, ma\n"
            "vse32.v v2, (%[out0])\n"
            "beqz %[has_second], 9f\n"
            "vse32.v v4, (%[out1])\n"
            "9:\n"
            ".option pop\n"
            : [ap] "+r"(ap), [b0] "+r"(b0p), [b1] "+r"(b1p), [kt] "+r"(kt),
            [rowcnt] "+r"(rowcnt), [i] "+r"(i), [cnt] "+r"(cnt)
            : [stage] "r"(stage), [bias0] "r"(bias0), [bias1] "r"(bias1),
            [out0] "r"(out0), [out1] "r"(out1), [has_second] "r"(has_second),
            [rows] "r"(rows), [lda] "r"(lda), [tailk] "r"(tail_k)
            : "v2", "v3", "v4", "v5", "v6", "v7", "t0", "t1", "t2", "memory");
}

__attribute__((visibility("default"))) void ime_gemm_f16_ime(const void *a_in,
        const void *b_packed, const void *bias_in, void *c_in, int64_t m,
        int64_t n, int64_t k_dim, const rvcl_work_range_t *r) {
    const uint16_t *a = (const uint16_t *)a_in;
    const uint16_t *bp = (const uint16_t *)b_packed;
    const float *bias = (const float *)bias_in;
    float *c = (float *)c_in;

    const int64_t k_t = k_dim / IME_K_TILE + (k_dim % IME_K_TILE != 0);
    const int64_t tail_k = k_dim % IME_K_TILE;
    /* the bias tile and the accumulator store time-share one buffer per
     * n tile: the assembly consumes the bias before the store writes it */
    float buf0[IME_TILE_ELT] __attribute__((aligned(64)));
    float buf1[IME_TILE_ELT] __attribute__((aligned(64)));

    for (int64_t m0 = (r->m_begin / IME_M_TILE) * IME_M_TILE; m0 < r->m_end;
            m0 += IME_M_TILE) {
        const int64_t rows = (m - m0 < IME_M_TILE) ? m - m0 : IME_M_TILE;
        /* n tiles advance in groups of two to share each staged A tile */
        for (int64_t n0 = (r->n_begin / (IME_N_TILE * 2)) * (IME_N_TILE * 2);
                n0 < r->n_end; n0 += IME_N_TILE * 2) {
            const int has_second
                    = (n0 + IME_N_TILE < n && n0 + IME_N_TILE < r->n_end);
            const int64_t nt0 = n0 / IME_N_TILE;
            const uint16_t *b0 = bp + nt0 * k_t * IME_TILE_ELT;
            const uint16_t *b1
                    = has_second ? bp + (nt0 + 1) * k_t * IME_TILE_ELT : b0;
            const float *bias0 = NULL, *bias1 = NULL;
            if (bias) {
                ime_bias_tile(buf0, bias, n0, n);
                bias0 = buf0;
                if (has_second) {
                    ime_bias_tile(buf1, bias, n0 + IME_N_TILE, n);
                    bias1 = buf1;
                }
            }
            ime_tile_group(a + m0 * k_dim, k_dim, b0, b1, k_t, rows, tail_k,
                    bias0, bias1, buf0, has_second ? buf1 : buf0, has_second);
            ime_store_tile(c, buf0, m, n, m0, n0, r);
            if (has_second)
                ime_store_tile(c, buf1, m, n, m0, n0 + IME_N_TILE, r);
        }
    }
}

#endif /* __riscv_vector */
