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

/* Covers the pack helpers of the IME backend against test-side oracles:
 * the tile-layout formula (buffer size, element index, tile order), tail
 * zero-fill, boundary-shape packing against an independently implemented
 * oracle packer, rejection of malformed descriptors, idempotence, and the
 * C micro-kernel's numerics on the packed layout (known answer + boundary
 * shapes + canaries). Runs on any host.
 */

#include "../src/backends/ime/ime_internal.hpp"
#include "golden.hpp"
#include "ime_fixtures.hpp"
#include "rvcl/rvcl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

#define CHECK(cond, msg) \
    do { \
        if (cond) { \
            printf("  ok: %s\n", msg); \
        } else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

/* ------------------------------------------------------------ helpers */

static rvcl_tensor_t make_b_tensor(uint16_t *data, int64_t k, int64_t n) {
    rvcl_tensor_desc_t d;
    memset(&d, 0, sizeof(d));
    d.ndims = 2;
    d.dims[0] = k;
    d.dims[1] = n;
    d.strides[0] = n;
    d.strides[1] = 1;
    d.dtype = RVCL_DTYPE_FP16;
    rvcl_tensor_t t;
    if (rvcl_tensor_init(&t, &d, data) != RVCL_SUCCESS) abort();
    return t;
}

static rvcl_tensor_t make_byte_view(void *data, int64_t bytes) {
    rvcl_tensor_desc_t d;
    memset(&d, 0, sizeof(d));
    d.ndims = 1;
    d.dims[0] = bytes;
    d.strides[0] = 1;
    d.dtype = RVCL_DTYPE_UINT8;
    rvcl_tensor_t t;
    if (rvcl_tensor_init(&t, &d, data) != RVCL_SUCCESS) abort();
    return t;
}

static void *xaligned_alloc(size_t bytes) {
    void *p = NULL;
    if (posix_memalign(&p, 64, bytes) != 0) abort();
    return p;
}

/* Packs plain B (plain row-major, fp16, with sentinel prefill) and
 * returns the packed buffer; also returns its byte size. */
static uint16_t *pack_oracle_alloc(
        const uint16_t *b, int64_t n, int64_t k, size_t *bytes_out) {
    const size_t bytes = ime_packed_b_bytes(k, n, 2);
    if (bytes == 0) abort();
    uint16_t *packed = (uint16_t *)xaligned_alloc(bytes);
    memset(packed, 0x5a, bytes);
    ime_oracle_pack_b(packed, b, n, k);
    *bytes_out = bytes;
    return packed;
}

/* --------------------------------------------------- 01: known answer (C) */

static void tc01_known_answer(void) {
    printf("TC-01 8x8 known answer through pack + C kernel\n");
    uint16_t a[64], b[64];
    ime_fixture_upper_a(a);
    ime_fixture_upper_b(b);

    size_t bytes = 0;
    uint16_t *packed = pack_oracle_alloc(b, 8, 8, &bytes);
    CHECK(bytes == 128, "8x8 packed buffer is 128 bytes");

    float c[64];
    golden_sentinel(c, sizeof(c));
    const rvcl_work_range_t full = rvcl_work_range_full(8, 8, 8);
    ime_gemm_f16_c(a, packed, NULL, c, 8, 8, 8, &full);
    CHECK(golden_cmp_bitexact(c, ime_fixture_upper_c, 64).mismatches == 0,
            "C kernel reproduces the hand-computed C bit-exactly");
    free(packed);
}

/* ------------------------------------------- 03: size and index formulas */

static void tc03_formulas(void) {
    printf("TC-03 buffer size and element index formulas\n");

    struct {
        int64_t n, k, bytes;
    } sizes[] = {
            {1, 1, 128},
            {7, 7, 128},
            {8, 8, 128},
            {9, 9, 512},
            {15, 17, 768},
            {16, 16, 512},
            {17, 31, 1536},
            {31, 31, 2048},
            {1, 31, 512},
            {31, 1, 512},
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        ok &= (ime_packed_b_bytes(sizes[i].k, sizes[i].n, 2)
                == (size_t)sizes[i].bytes);
    CHECK(ok, "ime_packed_b_bytes matches N_T*K_T*64*2 on all shapes");

    ok = 1;
    for (size_t i = 0; i < sizeof(sizes) / sizeof(sizes[0]); i++)
        ok &= (ime_packed_b_bytes(sizes[i].k, sizes[i].n, 4)
                == 2 * ime_packed_b_bytes(sizes[i].k, sizes[i].n, 2));
    CHECK(ok, "size formula is linear in elt");

    CHECK(ime_packed_b_bytes(0, 8, 2) == 0, "k_dim <= 0 rejected with 0");
    CHECK(ime_packed_b_bytes(8, 0, 2) == 0, "n <= 0 rejected with 0");
    CHECK(ime_packed_b_bytes(8, 8, 0) == 0, "elt 0 rejected with 0");

    struct {
        int64_t n, k, k_el, n_el, idx;
    } idxs[] = {
            {8, 8, 0, 0, 0},
            {8, 8, 3, 5, 43},
            {8, 8, 7, 7, 63},
            {8, 16, 8, 0, 64},
            {8, 16, 15, 7, 127},
            {16, 8, 7, 8, 71},
            {16, 8, 0, 9, 72},
            {16, 16, 10, 9, 202},
            {17, 31, 30, 16, 710},
            {17, 31, 0, 8, 256},
            {31, 31, 30, 30, 1014},
            {9, 9, 8, 8, 192},
            {7, 7, 6, 6, 54},
    };
    ok = 1;
    for (size_t i = 0; i < sizeof(idxs) / sizeof(idxs[0]); i++)
        ok &= (ime_packed_b_index(
                       idxs[i].n, idxs[i].k, idxs[i].k_el, idxs[i].n_el)
                == idxs[i].idx);
    CHECK(ok, "ime_packed_b_index matches the tile formula table");

    CHECK(ime_packed_b_index(8, 8, 8, 0) == -1, "k out of range -> -1");
    CHECK(ime_packed_b_index(8, 8, 0, 8) == -1, "n out of range -> -1");
    CHECK(ime_packed_b_index(8, 8, -1, 0) == -1, "negative k -> -1");
    CHECK(ime_packed_b_index(8, 8, 0, -1) == -1, "negative n -> -1");
    CHECK(ime_packed_b_index(0, 8, 0, 0) == -1, "invalid dims -> -1");
}

/* ------------------------------------------------- 04: tile memory order */

static void tc04_tile_order(void) {
    printf("TC-04 tile order: n tile outer, k tile inner\n");
    /* Plain B is [K][N] row-major, so the fill respects each case's N
     * stride: B[q][j] = q*16+j+1 (distinct fp16-exact values 1..256). */
    uint16_t b[16 * 16];

    struct {
        int64_t n, k;
        int64_t idx;
        uint16_t val;
        const char *what;
    } probes[] = {
            {8, 16, 0, 1, "(8,16) packed[0] = B[0][0]"},
            {8, 16, 63, 120, "(8,16) packed[63] = B[7][7]"},
            {8, 16, 64, 129, "(8,16) packed[64] = B[8][0] (k tile follows)"},
            {8, 16, 127, 248, "(8,16) packed[127] = B[15][7]"},
            {16, 8, 64, 9, "(16,8) packed[64] = B[0][8] (n tile follows)"},
            {16, 8, 127, 128, "(16,8) packed[127] = B[7][15]"},
    };
    int ok = 1;
    for (size_t i = 0; i < sizeof(probes) / sizeof(probes[0]); i++) {
        for (int64_t q = 0; q < probes[i].k; q++)
            for (int64_t j = 0; j < probes[i].n; j++)
                b[q * probes[i].n + j] = (uint16_t)(q * 16 + j + 1);
        size_t bytes = 0;
        uint16_t *packed
                = pack_oracle_alloc(b, probes[i].n, probes[i].k, &bytes);
        ok &= (packed[probes[i].idx] == probes[i].val);
        free(packed);
    }
    CHECK(ok, "adjacent tile offsets prove nt-outer/kt-inner order");

    /* (9,9): four tiles at elements 0 / 64 / 128 / 192 in order
     * (nt=0,kt=0), (0,1), (1,0), (1,1) */
    uint16_t b9[9 * 9];
    for (int64_t q = 0; q < 9; q++)
        for (int64_t j = 0; j < 9; j++)
            b9[q * 9 + j] = (uint16_t)((q * 9 + j) % 997 + 1);
    size_t bytes = 0;
    uint16_t *packed = pack_oracle_alloc(b9, 9, 9, &bytes);
    CHECK(bytes == 512, "(9,9) packs into 4 tiles (512 bytes)");
    CHECK(packed[64] == b9[8 * 9 + 0],
            "(9,9) tile 1 starts at k=8 (k tile inner)");
    CHECK(packed[128] == b9[0 * 9 + 8],
            "(9,9) tile 2 starts at n=8 (n tile outer)");
    CHECK(packed[192] == b9[8 * 9 + 8], "(9,9) tile 3 is (n=8,k=8)");
    free(packed);
}

/* ------------------------------------------------------- 05: tail zeroing */

static void tc05_zero_fill(void) {
    printf("TC-05 tail tile zero fill\n");
    struct {
        int64_t n, k;
        int valid;
        const char *what;
    } cases[] = {
            {7, 7, 49, "(7,7): 49 valid elements, 15 zeros"},
            {9, 9, 0, "(9,9): valid counts 64/8/8/1 over 4 tiles"},
            {1, 1, 1, "(1,1): 1 valid element, 63 zeros"},
    };
    for (size_t ci = 0; ci < sizeof(cases) / sizeof(cases[0]); ci++) {
        const int64_t n = cases[ci].n, k = cases[ci].k;
        uint16_t *b = (uint16_t *)xaligned_alloc((size_t)(k * n * 2));
        for (int64_t i = 0; i < k * n; i++)
            b[i] = (uint16_t)((i * 37 + 11) % 997 + 1); /* non-zero */

        const size_t bytes = ime_packed_b_bytes(k, n, 2);
        uint16_t *dst = (uint16_t *)xaligned_alloc(bytes);
        memset(dst, 0x5a, bytes); /* sentinel: packer must overwrite all */

        rvcl_tensor_t src = make_b_tensor(b, k, n);
        rvcl_tensor_t out = make_byte_view(dst, bytes);
        CHECK(ime_pack_b_tiles(&src, &out, 8, n, k, 2) == RVCL_SUCCESS,
                cases[ci].what);

        uint16_t *oracle = (uint16_t *)xaligned_alloc(bytes);
        memset(oracle, 0, bytes);
        ime_oracle_pack_b(oracle, b, n, k);
        CHECK(memcmp(dst, oracle, bytes) == 0,
                "packed bytes equal the oracle (explicit zeros, no residue)");

        int zeros = 0, residue = 0;
        for (size_t i = 0; i < bytes / 2; i++) {
            if (dst[i] == 0) zeros++;
            if (dst[i] == 0x5a5a) residue++;
        }
        CHECK(residue == 0, "no sentinel byte pattern survives");
        if (cases[ci].valid > 0)
            CHECK(zeros == (int)(bytes / 2) - cases[ci].valid,
                    "zero count matches tail expectation");
        free(b);
        free(dst);
        free(oracle);
    }
}

/* --------------------------------------- 06: boundary shapes vs oracle */

static void tc06_boundary_pack(void) {
    printf("TC-06 boundary shape scan against the oracle packer\n");
    int ok = 1, m_ok = 1;
    for (size_t i = 0;
            i < sizeof(ime_shapes_mixed) / sizeof(ime_shapes_mixed[0]); i++) {
        const int64_t m = ime_shapes_mixed[i][0];
        const int64_t n = ime_shapes_mixed[i][1];
        const int64_t k = ime_shapes_mixed[i][2];
        const uint32_t seed = golden_case_seed(m, n, k);
        uint16_t *b = (uint16_t *)xaligned_alloc((size_t)(k * n * 2));
        ime_fill_f16_normal(b, (size_t)(k * n), seed);

        const size_t bytes = ime_packed_b_bytes(k, n, 2);
        uint16_t *dst = (uint16_t *)xaligned_alloc(bytes);
        memset(dst, 0xa5, bytes);
        rvcl_tensor_t src = make_b_tensor(b, k, n);
        rvcl_tensor_t out = make_byte_view(dst, bytes);
        if (ime_pack_b_tiles(&src, &out, m, n, k, 2) != RVCL_SUCCESS) ok = 0;
        if (bytes
                != (size_t)((n / 8 + (n % 8 != 0)) * (k / 8 + (k % 8 != 0)) * 64
                        * 2))
            ok = 0;

        uint16_t *oracle = (uint16_t *)xaligned_alloc(bytes);
        ime_oracle_pack_b(oracle, b, n, k);
        if (memcmp(dst, oracle, bytes) != 0) ok = 0;
        free(oracle);

        /* M does not participate in B's layout */
        uint16_t *dst2 = (uint16_t *)xaligned_alloc(bytes);
        rvcl_tensor_t out2 = make_byte_view(dst2, bytes);
        ime_pack_b_tiles(&src, &out2, 12345, n, k, 2);
        if (memcmp(dst2, dst, bytes) != 0) m_ok = 0;
        free(dst2);

        free(b);
        free(dst);
    }
    for (size_t i = 0; i < sizeof(ime_shapes_nk) / sizeof(ime_shapes_nk[0]);
            i++) {
        const int64_t n = ime_shapes_nk[i][0];
        const int64_t k = ime_shapes_nk[i][1];
        const uint32_t seed = golden_case_seed(n, k, 7);
        uint16_t *b = (uint16_t *)xaligned_alloc((size_t)(k * n * 2));
        ime_fill_f16_normal(b, (size_t)(k * n), seed);
        const size_t bytes = ime_packed_b_bytes(k, n, 2);
        uint16_t *dst = (uint16_t *)xaligned_alloc(bytes);
        rvcl_tensor_t src = make_b_tensor(b, k, n);
        rvcl_tensor_t out = make_byte_view(dst, bytes);
        if (ime_pack_b_tiles(&src, &out, 3, n, k, 2) != RVCL_SUCCESS) ok = 0;
        uint16_t *oracle = (uint16_t *)xaligned_alloc(bytes);
        ime_oracle_pack_b(oracle, b, n, k);
        if (memcmp(dst, oracle, bytes) != 0) ok = 0;
        free(oracle);
        free(b);
        free(dst);
    }
    CHECK(ok, "all boundary shapes pack byte-identical to the oracle");
    CHECK(m_ok, "the packed bytes do not depend on M");
}

/* -------------------------------------------- 07: C kernel boundary numerics */

static void tc07_c_boundary(void) {
    printf("TC-07 C kernel numerics on boundary shapes\n");
    int ok = 1, canary_ok = 1, bias_ok = 1;
    for (size_t i = 0; i < sizeof(ime_shapes_diag) / sizeof(ime_shapes_diag[0]);
            i++) {
        const int64_t m = ime_shapes_diag[i], n = m, k = m;
        const uint32_t seed = golden_case_seed(m, n, k);
        uint16_t *a = (uint16_t *)xaligned_alloc((size_t)(m * k * 2));
        uint16_t *b = (uint16_t *)xaligned_alloc((size_t)(k * n * 2));
        float *bias = (float *)xaligned_alloc((size_t)(n * 4));
        ime_fill_f16_normal(a, (size_t)(m * k), seed);
        ime_fill_f16_normal(b, (size_t)(k * n), seed ^ 0x5a5a);
        ime_fill_bias_normal(bias, (size_t)n, seed ^ 0xa5a5);
        const size_t cbytes = (size_t)(m * n * 4) + 32; /* 16B canaries */
        unsigned char *craw = (unsigned char *)xaligned_alloc(cbytes);
        float *c = (float *)(craw + 16);
        float *golden = (float *)xaligned_alloc((size_t)(m * n * 4));

        size_t pbytes = 0;
        uint16_t *packed = pack_oracle_alloc(b, n, k, &pbytes);
        const rvcl_work_range_t full = rvcl_work_range_full(m, n, k);

        for (int with_bias = 0; with_bias < 2; with_bias++) {
            memset(craw, 0x5c, cbytes);
            ime_gemm_f16_c(
                    a, packed, with_bias ? bias : NULL, c, m, n, k, &full);
            golden_f16(a, b, with_bias ? bias : NULL, golden, m, n, k);
            golden_cmp_t cmp = golden_cmp_tol(c, golden, m * n, 1e-3f);
            if (cmp.mismatches != 0) {
                if (with_bias) bias_ok = 0;
                ok = 0;
            }
            if (craw[0] != 0x5c || craw[1] != 0x5c || craw[cbytes - 1] != 0x5c
                    || craw[cbytes - 2] != 0x5c)
                canary_ok = 0;
        }
        free(packed);
        free(golden);
        free(craw);
        free(bias);
        free(b);
        free(a);
    }
    CHECK(ok && bias_ok, "golden 1e-3 holds on every boundary shape");
    CHECK(canary_ok, "C buffer canaries survive (no out-of-range writes)");
}

/* ------------------------------------------------ 08: rejection semantics */

static void tc08_reject(void) {
    printf("TC-08 malformed descriptor rejection\n");
    uint16_t bdata[64];
    unsigned char wsdata[256];
    for (int i = 0; i < 64; i++)
        bdata[i] = 0x3c00;

    rvcl_tensor_t good = make_b_tensor(bdata, 8, 8);
    rvcl_tensor_t ws = make_byte_view(wsdata, sizeof(wsdata));

    rvcl_tensor_t auto_src = good;
    auto_src.desc.dtype = RVCL_DTYPE_AUTO;
    rvcl_tensor_t auto_dst = ws;
    auto_dst.desc.dtype = RVCL_DTYPE_AUTO;

    rvcl_status_t st;
    int rejected = 1;

    st = ime_pack_b_tiles(NULL, &ws, 8, 8, 8, 2);
    rejected &= (st == RVCL_ERROR_INVALID_ARGUMENT);
    st = ime_pack_b_tiles(&good, NULL, 8, 8, 8, 2);
    rejected &= (st == RVCL_ERROR_INVALID_ARGUMENT);
    good.data = NULL;
    st = ime_pack_b_tiles(&good, &ws, 8, 8, 8, 2);
    rejected &= (st == RVCL_ERROR_INVALID_ARGUMENT);
    good.data = bdata;
    ws.data = NULL;
    st = ime_pack_b_tiles(&good, &ws, 8, 8, 8, 2);
    rejected &= (st == RVCL_ERROR_INVALID_ARGUMENT);
    ws.data = wsdata;
    CHECK(rejected, "null pointers rejected with INVALID_ARGUMENT");

    st = ime_pack_b_tiles(&auto_src, &ws, 8, 8, 8, 2);
    CHECK(st == RVCL_ERROR_INVALID_ARGUMENT, "AUTO src dtype rejected");
    st = ime_pack_b_tiles(&good, &auto_dst, 8, 8, 8, 2);
    CHECK(st == RVCL_ERROR_INVALID_ARGUMENT, "AUTO dst dtype rejected");

    rejected = 1;
    static const rvcl_dtype_t bad[]
            = {RVCL_DTYPE_FP32, RVCL_DTYPE_BF16, RVCL_DTYPE_INT8};
    for (size_t i = 0; i < sizeof(bad) / sizeof(bad[0]); i++) {
        rvcl_tensor_t bad_src = good;
        bad_src.desc.dtype = bad[i];
        st = ime_pack_b_tiles(&bad_src, &ws, 8, 8, 8, 2);
        rejected &= (st == RVCL_ERROR_INVALID_ARGUMENT
                || st == RVCL_UNSUPPORTED_DTYPE_COMBO);
        rvcl_tensor_t bad_dst = ws;
        bad_dst.desc.dtype = bad[i];
        st = ime_pack_b_tiles(&good, &bad_dst, 8, 8, 8, 2);
        rejected &= (st == RVCL_ERROR_INVALID_ARGUMENT
                || st == RVCL_UNSUPPORTED_DTYPE_COMBO);
    }
    CHECK(rejected, "non-fp16 src / non-byte dst dtype combos rejected");

    st = ime_pack_b_tiles(&good, &ws, 8, 8, 0, 2);
    CHECK(st == RVCL_ERROR_INVALID_ARGUMENT, "k_dim <= 0 rejected");
    st = ime_pack_b_tiles(&good, &ws, 8, 0, 8, 2);
    CHECK(st == RVCL_ERROR_INVALID_ARGUMENT, "n <= 0 rejected");
    st = ime_pack_b_tiles(&good, &ws, 8, 8, 8, 4);
    CHECK(st == RVCL_ERROR_INVALID_ARGUMENT, "elt mismatch rejected");
}

/* ------------------------------------------- 09: purity and idempotence */

static void tc09_idempotent(void) {
    printf("TC-09 packing is pure and idempotent\n");
    const int64_t n = 9, k = 15;
    uint16_t b[15 * 9];
    for (int64_t i = 0; i < k * n; i++)
        b[i] = (uint16_t)((i * 53 + 7) % 991 + 1);

    const size_t bytes = ime_packed_b_bytes(k, n, 2);
    uint16_t *dst = (uint16_t *)xaligned_alloc(bytes);
    rvcl_tensor_t src = make_b_tensor(b, k, n);
    rvcl_tensor_t out = make_byte_view(dst, bytes);

    int ok = ime_pack_b_tiles(&src, &out, 5, n, k, 2) == RVCL_SUCCESS;
    uint16_t *first = (uint16_t *)xaligned_alloc(bytes);
    memcpy(first, dst, bytes);
    ok &= ime_pack_b_tiles(&src, &out, 5, n, k, 2) == RVCL_SUCCESS;
    ok &= (memcmp(first, dst, bytes) == 0);

    /* mutate the source between calls: pure data movement re-reads it */
    b[0] = 0x4000;
    ok &= ime_pack_b_tiles(&src, &out, 5, n, k, 2) == RVCL_SUCCESS;
    ok &= (memcmp(first, dst, bytes) != 0);
    CHECK(ok, "same input twice -> same bytes; changed input -> changed bytes");
    free(first);
    free(dst);
}

int main(void) {
    tc01_known_answer();
    tc03_formulas();
    tc04_tile_order();
    tc05_zero_fill();
    tc06_boundary_pack();
    tc07_c_boundary();
    tc08_reject();
    tc09_idempotent();
    printf(g_failures ? "IME pack tests: %d FAIL\n"
                      : "IME pack tests: all ok\n",
            g_failures);
    return g_failures ? 1 : 0;
}
