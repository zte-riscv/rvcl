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
 * rvv_bf16.hpp — bfloat16 soft conversions (internal)
 *
 * bf16 is the top 16 bits of an IEEE 754 fp32 value, so the up-conversion
 * is an exact integer widen-and-shift (u16 << 16) that needs no Zvfbfmin.
 * The down-conversion rounds the dropped low 16 bits to nearest-even so
 * the storage round-trip is the deterministic reference for golden and
 * differential tests on hosts without the bf16 vector instructions.
 *
 * Storage type is uint16_t so tests and kernels compile independent of
 * compiler _Float16 / bf16 support.
 ******************************************************************************/

#ifndef RVCL_RVV_BF16_H
#define RVCL_RVV_BF16_H

#include <stdint.h>

static inline float rvv_bf16_to_f32(uint16_t h) {
    const uint32_t bits = (uint32_t)h << 16;
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

static inline uint16_t rvv_f32_to_bf16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);

    const uint32_t lsb = (x >> 16) & 1u;
    const uint32_t rounding_bias = 0x7fff + lsb;
    x += rounding_bias;
    return (uint16_t)(x >> 16);
}

#endif /* RVCL_RVV_BF16_H */
