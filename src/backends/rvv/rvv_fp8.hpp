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

/* FP8 has no ratified base hardware extension, so the up-conversion is a
 * 256-entry f32 lookup table per dtype; every FP8 bit pattern maps to an
 * exactly-representable f32, making the table the exact reference for golden
 * and differential tests on any host. The two dtypes share the same table
 * infrastructure and differ only in the contents:
 *   - E4M3: 1 sign, 4 exponent (bias 7), 3 mantissa; no infinity, field-15
 *     patterns with mantissa 0b111 are NaN.
 *   - E5M2: 1 sign, 5 exponent (bias 15), 2 mantissa; field 31 carries
 *     +/-inf (mantissa 0) and NaN (mantissa nonzero).
 * The down-conversions (f32 -> fp8, round-to-nearest-even) are the
 * deterministic reference for narrowing fp32 data into FP8 storage. The
 * up-conversion tables are shared verbatim by the golden, the C-portable
 * kernels and the JIT kernels, so all three agree bit-for-bit on FP8 inputs.
 * Storage type is uint8_t for the values and uint16_t for the table's f32
 * bit patterns, so kernels compile independent of any compiler FP8 support. */

#ifndef RVCL_RVV_FP8_H
#define RVCL_RVV_FP8_H

#include <stdint.h>

/* 256-entry f32 (bit-pattern) lookup table for FP8 E4M3. Indexed by the
 * raw FP8 byte, each entry is the exact f32 representation of that bit
 * pattern. NaN patterns map to the canonical quiet NaN 0x7fc00000. */
static const uint32_t rvv_fp8_e4m3_to_f32_bits[256] = {
        0x00000000u,
        0x3b000000u,
        0x3b800000u,
        0x3bc00000u,
        0x3c000000u,
        0x3c200000u,
        0x3c400000u,
        0x3c600000u,
        0x3c800000u,
        0x3c900000u,
        0x3ca00000u,
        0x3cb00000u,
        0x3cc00000u,
        0x3cd00000u,
        0x3ce00000u,
        0x3cf00000u,
        0x3d000000u,
        0x3d100000u,
        0x3d200000u,
        0x3d300000u,
        0x3d400000u,
        0x3d500000u,
        0x3d600000u,
        0x3d700000u,
        0x3d800000u,
        0x3d900000u,
        0x3da00000u,
        0x3db00000u,
        0x3dc00000u,
        0x3dd00000u,
        0x3de00000u,
        0x3df00000u,
        0x3e000000u,
        0x3e100000u,
        0x3e200000u,
        0x3e300000u,
        0x3e400000u,
        0x3e500000u,
        0x3e600000u,
        0x3e700000u,
        0x3e800000u,
        0x3e900000u,
        0x3ea00000u,
        0x3eb00000u,
        0x3ec00000u,
        0x3ed00000u,
        0x3ee00000u,
        0x3ef00000u,
        0x3f000000u,
        0x3f100000u,
        0x3f200000u,
        0x3f300000u,
        0x3f400000u,
        0x3f500000u,
        0x3f600000u,
        0x3f700000u,
        0x3f800000u,
        0x3f900000u,
        0x3fa00000u,
        0x3fb00000u,
        0x3fc00000u,
        0x3fd00000u,
        0x3fe00000u,
        0x3ff00000u,
        0x40000000u,
        0x40100000u,
        0x40200000u,
        0x40300000u,
        0x40400000u,
        0x40500000u,
        0x40600000u,
        0x40700000u,
        0x40800000u,
        0x40900000u,
        0x40a00000u,
        0x40b00000u,
        0x40c00000u,
        0x40d00000u,
        0x40e00000u,
        0x40f00000u,
        0x41000000u,
        0x41100000u,
        0x41200000u,
        0x41300000u,
        0x41400000u,
        0x41500000u,
        0x41600000u,
        0x41700000u,
        0x41800000u,
        0x41900000u,
        0x41a00000u,
        0x41b00000u,
        0x41c00000u,
        0x41d00000u,
        0x41e00000u,
        0x41f00000u,
        0x42000000u,
        0x42100000u,
        0x42200000u,
        0x42300000u,
        0x42400000u,
        0x42500000u,
        0x42600000u,
        0x42700000u,
        0x42800000u,
        0x42900000u,
        0x42a00000u,
        0x42b00000u,
        0x42c00000u,
        0x42d00000u,
        0x42e00000u,
        0x42f00000u,
        0x43000000u,
        0x43100000u,
        0x43200000u,
        0x43300000u,
        0x43400000u,
        0x43500000u,
        0x43600000u,
        0x43700000u,
        0x43800000u,
        0x43900000u,
        0x43a00000u,
        0x43b00000u,
        0x43c00000u,
        0x43d00000u,
        0x43e00000u,
        0x7fc00000u,
        0x80000000u,
        0xbb000000u,
        0xbb800000u,
        0xbbc00000u,
        0xbc000000u,
        0xbc200000u,
        0xbc400000u,
        0xbc600000u,
        0xbc800000u,
        0xbc900000u,
        0xbca00000u,
        0xbcb00000u,
        0xbcc00000u,
        0xbcd00000u,
        0xbce00000u,
        0xbcf00000u,
        0xbd000000u,
        0xbd100000u,
        0xbd200000u,
        0xbd300000u,
        0xbd400000u,
        0xbd500000u,
        0xbd600000u,
        0xbd700000u,
        0xbd800000u,
        0xbd900000u,
        0xbda00000u,
        0xbdb00000u,
        0xbdc00000u,
        0xbdd00000u,
        0xbde00000u,
        0xbdf00000u,
        0xbe000000u,
        0xbe100000u,
        0xbe200000u,
        0xbe300000u,
        0xbe400000u,
        0xbe500000u,
        0xbe600000u,
        0xbe700000u,
        0xbe800000u,
        0xbe900000u,
        0xbea00000u,
        0xbeb00000u,
        0xbec00000u,
        0xbed00000u,
        0xbee00000u,
        0xbef00000u,
        0xbf000000u,
        0xbf100000u,
        0xbf200000u,
        0xbf300000u,
        0xbf400000u,
        0xbf500000u,
        0xbf600000u,
        0xbf700000u,
        0xbf800000u,
        0xbf900000u,
        0xbfa00000u,
        0xbfb00000u,
        0xbfc00000u,
        0xbfd00000u,
        0xbfe00000u,
        0xbff00000u,
        0xc0000000u,
        0xc0100000u,
        0xc0200000u,
        0xc0300000u,
        0xc0400000u,
        0xc0500000u,
        0xc0600000u,
        0xc0700000u,
        0xc0800000u,
        0xc0900000u,
        0xc0a00000u,
        0xc0b00000u,
        0xc0c00000u,
        0xc0d00000u,
        0xc0e00000u,
        0xc0f00000u,
        0xc1000000u,
        0xc1100000u,
        0xc1200000u,
        0xc1300000u,
        0xc1400000u,
        0xc1500000u,
        0xc1600000u,
        0xc1700000u,
        0xc1800000u,
        0xc1900000u,
        0xc1a00000u,
        0xc1b00000u,
        0xc1c00000u,
        0xc1d00000u,
        0xc1e00000u,
        0xc1f00000u,
        0xc2000000u,
        0xc2100000u,
        0xc2200000u,
        0xc2300000u,
        0xc2400000u,
        0xc2500000u,
        0xc2600000u,
        0xc2700000u,
        0xc2800000u,
        0xc2900000u,
        0xc2a00000u,
        0xc2b00000u,
        0xc2c00000u,
        0xc2d00000u,
        0xc2e00000u,
        0xc2f00000u,
        0xc3000000u,
        0xc3100000u,
        0xc3200000u,
        0xc3300000u,
        0xc3400000u,
        0xc3500000u,
        0xc3600000u,
        0xc3700000u,
        0xc3800000u,
        0xc3900000u,
        0xc3a00000u,
        0xc3b00000u,
        0xc3c00000u,
        0xc3d00000u,
        0xc3e00000u,
        0x7fc00000u,
};

/* 256-entry f32 (bit-pattern) lookup table for FP8 E5M2. Indexed by the
 * raw FP8 byte. +/-inf and NaN patterns are preserved (0x7fc00000 is the
 * canonical quiet NaN used for every NaN encoding). */
static const uint32_t rvv_fp8_e5m2_to_f32_bits[256] = {
        0x00000000u,
        0x37800000u,
        0x38000000u,
        0x38400000u,
        0x38800000u,
        0x38a00000u,
        0x38c00000u,
        0x38e00000u,
        0x39000000u,
        0x39200000u,
        0x39400000u,
        0x39600000u,
        0x39800000u,
        0x39a00000u,
        0x39c00000u,
        0x39e00000u,
        0x3a000000u,
        0x3a200000u,
        0x3a400000u,
        0x3a600000u,
        0x3a800000u,
        0x3aa00000u,
        0x3ac00000u,
        0x3ae00000u,
        0x3b000000u,
        0x3b200000u,
        0x3b400000u,
        0x3b600000u,
        0x3b800000u,
        0x3ba00000u,
        0x3bc00000u,
        0x3be00000u,
        0x3c000000u,
        0x3c200000u,
        0x3c400000u,
        0x3c600000u,
        0x3c800000u,
        0x3ca00000u,
        0x3cc00000u,
        0x3ce00000u,
        0x3d000000u,
        0x3d200000u,
        0x3d400000u,
        0x3d600000u,
        0x3d800000u,
        0x3da00000u,
        0x3dc00000u,
        0x3de00000u,
        0x3e000000u,
        0x3e200000u,
        0x3e400000u,
        0x3e600000u,
        0x3e800000u,
        0x3ea00000u,
        0x3ec00000u,
        0x3ee00000u,
        0x3f000000u,
        0x3f200000u,
        0x3f400000u,
        0x3f600000u,
        0x3f800000u,
        0x3fa00000u,
        0x3fc00000u,
        0x3fe00000u,
        0x40000000u,
        0x40200000u,
        0x40400000u,
        0x40600000u,
        0x40800000u,
        0x40a00000u,
        0x40c00000u,
        0x40e00000u,
        0x41000000u,
        0x41200000u,
        0x41400000u,
        0x41600000u,
        0x41800000u,
        0x41a00000u,
        0x41c00000u,
        0x41e00000u,
        0x42000000u,
        0x42200000u,
        0x42400000u,
        0x42600000u,
        0x42800000u,
        0x42a00000u,
        0x42c00000u,
        0x42e00000u,
        0x43000000u,
        0x43200000u,
        0x43400000u,
        0x43600000u,
        0x43800000u,
        0x43a00000u,
        0x43c00000u,
        0x43e00000u,
        0x44000000u,
        0x44200000u,
        0x44400000u,
        0x44600000u,
        0x44800000u,
        0x44a00000u,
        0x44c00000u,
        0x44e00000u,
        0x45000000u,
        0x45200000u,
        0x45400000u,
        0x45600000u,
        0x45800000u,
        0x45a00000u,
        0x45c00000u,
        0x45e00000u,
        0x46000000u,
        0x46200000u,
        0x46400000u,
        0x46600000u,
        0x46800000u,
        0x46a00000u,
        0x46c00000u,
        0x46e00000u,
        0x47000000u,
        0x47200000u,
        0x47400000u,
        0x47600000u,
        0x7f800000u,
        0x7fc00000u,
        0x7fc00000u,
        0x7fc00000u,
        0x80000000u,
        0xb7800000u,
        0xb8000000u,
        0xb8400000u,
        0xb8800000u,
        0xb8a00000u,
        0xb8c00000u,
        0xb8e00000u,
        0xb9000000u,
        0xb9200000u,
        0xb9400000u,
        0xb9600000u,
        0xb9800000u,
        0xb9a00000u,
        0xb9c00000u,
        0xb9e00000u,
        0xba000000u,
        0xba200000u,
        0xba400000u,
        0xba600000u,
        0xba800000u,
        0xbaa00000u,
        0xbac00000u,
        0xbae00000u,
        0xbb000000u,
        0xbb200000u,
        0xbb400000u,
        0xbb600000u,
        0xbb800000u,
        0xbba00000u,
        0xbbc00000u,
        0xbbe00000u,
        0xbc000000u,
        0xbc200000u,
        0xbc400000u,
        0xbc600000u,
        0xbc800000u,
        0xbca00000u,
        0xbcc00000u,
        0xbce00000u,
        0xbd000000u,
        0xbd200000u,
        0xbd400000u,
        0xbd600000u,
        0xbd800000u,
        0xbda00000u,
        0xbdc00000u,
        0xbde00000u,
        0xbe000000u,
        0xbe200000u,
        0xbe400000u,
        0xbe600000u,
        0xbe800000u,
        0xbea00000u,
        0xbec00000u,
        0xbee00000u,
        0xbf000000u,
        0xbf200000u,
        0xbf400000u,
        0xbf600000u,
        0xbf800000u,
        0xbfa00000u,
        0xbfc00000u,
        0xbfe00000u,
        0xc0000000u,
        0xc0200000u,
        0xc0400000u,
        0xc0600000u,
        0xc0800000u,
        0xc0a00000u,
        0xc0c00000u,
        0xc0e00000u,
        0xc1000000u,
        0xc1200000u,
        0xc1400000u,
        0xc1600000u,
        0xc1800000u,
        0xc1a00000u,
        0xc1c00000u,
        0xc1e00000u,
        0xc2000000u,
        0xc2200000u,
        0xc2400000u,
        0xc2600000u,
        0xc2800000u,
        0xc2a00000u,
        0xc2c00000u,
        0xc2e00000u,
        0xc3000000u,
        0xc3200000u,
        0xc3400000u,
        0xc3600000u,
        0xc3800000u,
        0xc3a00000u,
        0xc3c00000u,
        0xc3e00000u,
        0xc4000000u,
        0xc4200000u,
        0xc4400000u,
        0xc4600000u,
        0xc4800000u,
        0xc4a00000u,
        0xc4c00000u,
        0xc4e00000u,
        0xc5000000u,
        0xc5200000u,
        0xc5400000u,
        0xc5600000u,
        0xc5800000u,
        0xc5a00000u,
        0xc5c00000u,
        0xc5e00000u,
        0xc6000000u,
        0xc6200000u,
        0xc6400000u,
        0xc6600000u,
        0xc6800000u,
        0xc6a00000u,
        0xc6c00000u,
        0xc6e00000u,
        0xc7000000u,
        0xc7200000u,
        0xc7400000u,
        0xc7600000u,
        0xff800000u,
        0x7fc00000u,
        0x7fc00000u,
        0x7fc00000u,
};

static inline float rvv_fp8_e4m3_to_f32(uint8_t v) {
    const uint32_t bits = rvv_fp8_e4m3_to_f32_bits[v];
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

static inline float rvv_fp8_e5m2_to_f32(uint8_t v) {
    const uint32_t bits = rvv_fp8_e5m2_to_f32_bits[v];
    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

/* f32 -> FP8 E4M3, round-to-nearest-even. Saturates at the max finite
 * value 448 (0x7E) — E4M3 has no infinity — and maps every NaN input to
 * the canonical NaN 0x7F. The subnormal window is [2^-9, 2^-6): the
 * smallest encoding 0x01 is exactly 2^-9. Values below it flush to signed
 * zero by design (deliberate flush-to-zero, no rounding up of the gap just
 * below the smallest subnormal). */
static inline uint8_t rvv_f32_to_fp8_e4m3(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 1u;
    const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 7;
    const uint32_t man = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) /* nan or inf: E4M3 has no inf */
        return (uint8_t)((sign << 7) | 0x7fu);

    if (exp >= 15) { /* finite values with field-15 exponent */
        if (exp > 15) return (uint8_t)((sign << 7) | 0x7eu); /* > 448 -> max */
        /* exp == 15: normal in [256, 448], rounding a mantissa to 0b111
         * produces the NaN encoding 0x7F. 256.0 (man == 0) is the normal
         * 0x78, not a saturation. */
        const uint32_t half = 1u << 19;
        uint32_t out = (man >> 20) & 0x7u;
        const uint32_t rem = man & 0xfffffu;
        if (rem > half || (rem == half && (out & 1u))) out++;
        if (out == 0x7u) return (uint8_t)((sign << 7) | 0x7fu);
        return (uint8_t)((sign << 7) | 0x78u | out);
    }

    if (exp <= 0) { /* subnormal or zero: [2^-9, 2^-6) or below */
        if (exp < -2) return (uint8_t)(sign << 7); /* < 2^-9 -> signed zero */
        const uint32_t m = man | 0x800000u; /* implicit leading 1 */
        const int shift = 21 - exp;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t kept = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t out = kept;
        if (rem > half || (rem == half && (kept & 1u))) out++;
        if (out > 0x7u)
            out = 0x8u; /* rounded up into min normal (0x08 = 2^-6) */
        return (uint8_t)((sign << 7) | out);
    }

    /* normal: round mantissa from 23 to 3 bits, nearest-even */
    const uint32_t half = 1u << 19;
    uint32_t out = ((uint32_t)exp << 3) | (man >> 20);
    const uint32_t rem = man & 0xfffffu;
    if (rem > half || (rem == half && (out & 1u))) out++;
    if ((out & 0x78u) == 0x78u && (out & 0x7u) == 0x7u) /* rounded to NaN */
        return (uint8_t)((sign << 7) | 0x7fu);
    return (uint8_t)((sign << 7) | out);
}

/* f32 -> FP8 E5M2, round-to-nearest-even. Overflow rounds to +/-inf,
 * NaN inputs map to the canonical NaN 0x7F. The subnormal window is
 * [2^-16, 2^-14): the smallest encoding 0x01 is exactly 2^-16. Values below
 * it flush to signed zero by design (deliberate flush-to-zero, no rounding
 * up of the gap just below the smallest subnormal). */
static inline uint8_t rvv_f32_to_fp8_e5m2(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 31) & 1u;
    const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t man = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) { /* inf / nan */
        if (man) return (uint8_t)((sign << 7) | 0x7fu); /* nan */
        return (uint8_t)((sign << 7) | 0x7cu); /* inf */
    }

    if (exp >= 31) /* overflow -> inf */
        return (uint8_t)((sign << 7) | 0x7cu);

    if (exp <= 0) { /* subnormal or zero: [2^-16, 2^-14) or below */
        if (exp < -4) return (uint8_t)(sign << 7); /* < 2^-16 -> signed zero */
        const uint32_t m = man | 0x800000u;
        const int shift = 22 - exp;
        const uint32_t half = 1u << (shift - 1);
        const uint32_t kept = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t out = kept;
        if (rem > half || (rem == half && (kept & 1u))) out++;
        if (out > 0x3u)
            out = 0x4u; /* rounded up into min normal (0x04 = 2^-14) */
        return (uint8_t)((sign << 7) | out);
    }

    /* normal: round mantissa from 23 to 2 bits, nearest-even */
    const uint32_t half = 1u << 20;
    uint32_t out = ((uint32_t)exp << 2) | (man >> 21);
    const uint32_t rem = man & 0x1fffffu;
    if (rem > half || (rem == half && (out & 1u))) out++;
    if ((out & 0x7cu) == 0x7cu) out = 0x7cu; /* rounded up to inf */
    return (uint8_t)((sign << 7) | out);
}

#endif /* RVCL_RVV_FP8_H */
