/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * rvv_fp16.h — IEEE 754 half-precision soft conversions (internal)
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Used by the f16 kernels on hosts without Zvfh (x86 CI) and as the
 * bit-exact reference for differential tests. Round-to-nearest-even on
 * f32->f16, exact on f16->f32.
 *
 * Storage type is uint16_t so tests and kernels compile independent of
 * compiler _Float16 support.
 ******************************************************************************/

#ifndef RVCL_RVV_FP16_H
#define RVCL_RVV_FP16_H

#include <stdint.h>

static inline float rvv_f16_to_f32(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1fu;
    const uint32_t man = h & 0x3ffu;
    uint32_t bits;

    if (exp == 0) {
        if (man == 0) { /* +/- 0 */
            bits = sign;
        } else { /* subnormal: normalize */
            int e = -1;
            uint32_t m = man;
            do {
                m <<= 1;
                e++;
            } while (!(m & 0x400u));
            bits = sign | ((uint32_t)(127 - 15 - e) << 23) | ((m & 0x3ffu) << 13);
        }
    } else if (exp == 0x1f) { /* inf / nan */
        bits = sign | 0x7f800000u | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }

    float out;
    __builtin_memcpy(&out, &bits, 4);
    return out;
}

static inline uint16_t rvv_f32_to_f16(float f) {
    uint32_t x;
    __builtin_memcpy(&x, &f, 4);

    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xffu) - 127 + 15;
    const uint32_t man = x & 0x7fffffu;

    if (((x >> 23) & 0xffu) == 0xffu) /* inf / nan */
        return (uint16_t)(sign | 0x7c00u | (man ? 0x200u : 0u));

    if (exp >= 0x1f) /* overflow -> inf */
        return (uint16_t)(sign | 0x7c00u);

    if (exp <= 0) { /* underflow: subnormal or zero */
        if (exp < -10) return (uint16_t)sign;
        const uint32_t m = man | 0x800000u; /* implicit 1 */
        const int shift = (int)(14 - exp + 13);
        /* round-to-nearest-even on the dropped bits */
        const uint32_t half = 1u << (shift - 1);
        const uint32_t kept = m >> shift;
        uint32_t rem = m & ((1u << shift) - 1u);
        uint32_t out = kept;
        if (rem > half || (rem == half && (kept & 1u))) out++;
        return (uint16_t)(sign | out);
    }

    /* normal: round mantissa from 23 to 10 bits, nearest-even */
    const uint32_t half = 1u << 12;
    uint32_t out = ((uint32_t)exp << 10) | (man >> 13);
    const uint32_t rem = man & 0x1fffu;
    if (rem > half || (rem == half && (out & 1u))) out++;
    if (out & 0x7c00u) out = 0x7c00u; /* rounded up to inf */
    return (uint16_t)(sign | out);
}

#endif /* RVCL_RVV_FP16_H */
