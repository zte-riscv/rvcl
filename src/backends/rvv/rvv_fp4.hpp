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

/* FP4 (E2M1) element format shared by the NVFP4 and MXFP4 block-scale
 * dtypes. E2M1 is the 4-bit encoding used by OCP MX: 1 sign, 2 exponent
 * (bias 1), 1 mantissa. The 16 encodings are 0, +-0.5, +-1, +-1.5, +-2,
 * +-3, +-4, +-6 — every one is exactly representable in f32, so the 16-entry
 * lookup table is the exact reference for golden and differential tests on
 * any host. Storage is nibble-packed (2 elements per byte, even index = low
 * nibble), so kernels unpack before the table lookup. The table is shared
 * verbatim by the golden, the C-portable kernels and (later) the JIT
 * kernels, so all agree bit-for-bit on the same bytes. */

#ifndef RVCL_RVV_FP4_H
#define RVCL_RVV_FP4_H

#include <stdint.h>

/* 16-entry f32 (bit-pattern) lookup table for E2M1, indexed by the raw
 * 4-bit encoding (sign << 3 | exponent << 1 | mantissa). All values are
 * exactly representable in f32. */
static const uint32_t rvv_e2m1_to_f32_bits[16] = {
        0x00000000u, /* 0        */
        0x3f000000u, /* 0.5      */
        0x3f800000u, /* 1.0      */
        0x3fc00000u, /* 1.5      */
        0x40000000u, /* 2.0      */
        0x40400000u, /* 3.0      */
        0x40800000u, /* 4.0      */
        0x40c00000u, /* 6.0      */
        0x80000000u, /* -0       */
        0xbf000000u, /* -0.5     */
        0xbf800000u, /* -1.0     */
        0xbfc00000u, /* -1.5     */
        0xc0000000u, /* -2.0     */
        0xc0400000u, /* -3.0     */
        0xc0800000u, /* -4.0     */
        0xc0c00000u, /* -6.0     */
};

/* E2M1 nibble -> f32, via the shared table. */
static inline float rvv_e2m1_to_f32(uint8_t nib) {
    const uint32_t bits = rvv_e2m1_to_f32_bits[nib & 0xFu];
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

/* Nibble unpack: element e of a nibble-packed byte buffer sits at byte e/2
 * (even e = low nibble, odd e = high nibble). */
static inline uint8_t rvv_fp4_nibble_get(const uint8_t *p, int64_t e) {
    return (uint8_t)((p[e >> 1] >> ((e & 1) ? 4 : 0)) & 0xFu);
}

/* Nibble pack: write element e's low 4 bits into byte e/2 (even e = low
 * nibble, odd e = high nibble). */
static inline void rvv_fp4_nibble_set(uint8_t *p, int64_t e, uint8_t nib) {
    const uint8_t shift = (uint8_t)((e & 1) ? 4 : 0);
    const uint8_t mask = (uint8_t)(0x0Fu << shift);
    const uint8_t keep = (uint8_t)(p[e >> 1] & ~mask);
    p[e >> 1] = (uint8_t)(keep | ((nib & 0x0Fu) << shift));
}

#endif /* RVCL_RVV_FP4_H */
