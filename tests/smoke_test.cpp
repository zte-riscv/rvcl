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

/* Smoke test covering the acceptance surface:
 *   - registry: register / duplicate / lookup / count
 *   - capability: detection runs, cached pointer stable
 *   - scheduler: set-once semantics, serial parallel_for executes task
 *   - memory contract: zero-copy import (pointer pass-through), non-plain
 *     layout rejected
 *   - dispatch: (A,B,acc,C) filtering — unsupported tuple rejected,
 *     supported tuple selects a registered capability-matched kernel
 *     (invariant-checked; precise selection is pinned by the dispatch unit
 *     tests), wrong shape rejected
 *   - kernel lifecycle: configure reports the selected kernel's workspace
 *     demand
 *   - end-to-end: rvcl_matmul, rvcl_conv2d and rvcl_inner_product match a
 *     naive golden, the latter two with and without a bias */

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rvcl/rvcl.h"

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

/* ---------------------------------------------------------- scheduler test */

static int g_task_ran;
static void probe_task(
        rvcl_ischeduler_t *s, const rvcl_work_range_t *r, void *ctx) {
    (void)s;
    (void)r;
    *(int *)ctx = 1;
}

/* --------------------------------------------------------------- main */

int main(void) {
    /* --- init & version --- */
    CHECK(rvcl_init() == RVCL_SUCCESS, "rvcl_init succeeds");
    CHECK(rvcl_init() == RVCL_SUCCESS, "rvcl_init idempotent");
    CHECK(strncmp(rvcl_get_version(), "0.1", 3) == 0, "version string sane");

    /* --- capability --- */
    const rvcl_capability_t *cap = rvcl_get_capability();
    CHECK(cap != NULL, "capability detected");
    CHECK(rvcl_get_capability() == cap, "capability cached (same pointer)");

    /* --- registry --- */
    CHECK(rvcl_registry_count() >= 1, "reference kernel registered");
    CHECK(rvcl_registry_find_factory("rvcl_matmul_f32_reference") != NULL,
            "factory lookup by name");
    CHECK(rvcl_registry_find_factory("nope") == NULL, "unknown name -> NULL");

    /* --- scheduler: set-once --- */
    CHECK(rvcl_scheduler_set(rvcl_serial_scheduler()) == RVCL_SUCCESS,
            "scheduler set accepts serial");
    CHECK(rvcl_scheduler_set(rvcl_serial_scheduler())
                    == RVCL_ERROR_ALREADY_INITIALIZED,
            "second scheduler set rejected (set-once)");
    g_task_ran = 0;
    rvcl_work_range_t full = rvcl_work_range_full(8, 8, 8);
    rvcl_parallel_for(&full, probe_task, &g_task_ran);
    CHECK(g_task_ran, "parallel_for executes task via serial scheduler");

    /* --- memory contract --- */
    float buf[16];
    rvcl_tensor_desc_t td = {.ndims = 2,
            .dims = {4, 4},
            .strides = {4, 1},
            .dtype = RVCL_DTYPE_FP32};
    rvcl_tensor_t t;
    CHECK(rvcl_tensor_init(&t, &td, buf) == RVCL_SUCCESS, "tensor import ok");
    CHECK(t.data == (void *)buf, "zero-copy: pointer passed through");
    CHECK(rvcl_tensor_bytes(&t) == (int64_t)16 * 4,
            "tensor byte size computed");

    rvcl_tensor_desc_t bad_td = td;
    bad_td.strides[1] = 2; /* innermost stride != 1: non-plain layout */
    rvcl_tensor_t t2;
    CHECK(rvcl_tensor_init(&t2, &bad_td, buf) == RVCL_UNSUPPORTED,
            "non-plain layout rejected");

    /* --- dispatch: datatype filtering --- */
    rvcl_matmul_desc_t d = {
            .m = 16,
            .n = 16,
            .k = 16,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "fp32 tuple dispatches");
    /* Selection assertions here are deliberately invariant-based: the
     * precise winner depends on build configuration (which VLEN tiers
     * compiled in) and selection phase, and is pinned by the dispatch unit
     * tests instead. Invariants: the selected kernel is a registered f32
     * matmul entry, and on a vector build with RVV capability present it
     * comes from the rvv family. */
    const char *sel = r ? rvcl_dispatch_kernel_name(r) : NULL;
    CHECK(sel != NULL, "dispatch returned a kernel name");
    if (sel) {
        CHECK(strncmp(sel, "rvcl_matmul_f32_r", 17) == 0,
                "selected kernel is a registered f32 matmul entry");
        const int vector_build
                = rvcl_registry_find_factory("rvcl_matmul_f32_rvv") != NULL;
        const int host_has_rvv = rvcl_capability_has(cap, RVCL_FEATURE_RVV);
        if (vector_build && host_has_rvv)
            CHECK(strncmp(sel, "rvcl_matmul_f32_rvv", 19) == 0,
                    "vector build with RVV host selects an rvv-family kernel");
        /* Selection is cost-driven with registration order as the
         * tie-break: dedicated tier variants first, then the generic or
         * portable entry, then the reference fallback. The prefix
         * assertion above holds for every build configuration. */
        CHECK(rvcl_registry_find_factory(sel) != NULL,
                "selected name resolves in the registry");
    }
    int num_ws = -1;
    rvcl_dispatch_ws_req(r, &num_ws);
    CHECK(num_ws >= 0 && num_ws <= RVCL_MAX_WORKSPACES,
            "selected kernel reports a sane workspace count");
    rvcl_dispatch_release(r);

    rvcl_matmul_desc_t dbad = d;
    dbad.dtype_a = RVCL_DTYPE_BF16; /* no bf16 kernel registered yet */
    CHECK(rvcl_dispatch_matmul(&dbad, &r) == RVCL_UNSUPPORTED,
            "unsupported tuple rejected (bf16 has no kernel)");
    CHECK(r == NULL, "no result leaked on rejection");

    rvcl_matmul_desc_t dshape = d;
    dshape.m = -1;
    CHECK(rvcl_dispatch_matmul(&dshape, &r) == RVCL_ERROR_INVALID_ARGUMENT,
            "invalid shape rejected");

    /* --- end-to-end vs naive golden --- */
    enum { M = 8, N = 8, K = 16 };
    float a[M * K], b[K * N], bias[N], c[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = (float)((i * 13) % 29) / 29.0f;
    for (int i = 0; i < K * N; i++)
        b[i] = (float)((i * 7) % 31) / 31.0f;
    for (int j = 0; j < N; j++)
        bias[j] = 0.5f;
    memset(c, 0, sizeof(c));

    rvcl_matmul_desc_t de = {
            .m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };
    CHECK(rvcl_matmul(&de, a, b, bias, c) == RVCL_SUCCESS,
            "rvcl_matmul succeeds");

    int mismatches = 0;
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = bias[j];
            for (int p = 0; p < K; p++)
                acc += a[i * K + p] * b[p * N + j];
            if (fabsf(c[i * N + j] - acc) > 1e-5f) mismatches++;
        }
    CHECK(mismatches == 0, "end-to-end matches naive golden");

    /* --- conv2d end-to-end: bias and no-bias --- */
    /* A small NCHW convolution with stride 1, dilation 1 and one pad on
     * every side, checked against a direct scalar loop. The no-bias pass
     * reuses the same buffers so a bias left behind by the first pass would
     * show up as a difference. */
    enum { CMB = 1, CIC = 2, CIH = 5, CIW = 5, COC = 3, CKH = 3, CKW = 3 };
    enum { COH = CIH, COW = CIW }; /* pad 1, stride 1, kernel 3 */
    float csrc[CMB * CIC * CIH * CIW];
    float cw[COC * CIC * CKH * CKW];
    float cbias[COC];
    float cdst[CMB * COC * COH * COW];
    for (int i = 0; i < CMB * CIC * CIH * CIW; i++)
        csrc[i] = (float)((i * 11) % 23) / 23.0f - 0.5f;
    for (int i = 0; i < COC * CIC * CKH * CKW; i++)
        cw[i] = (float)((i * 5) % 17) / 17.0f - 0.5f;
    for (int i = 0; i < COC; i++)
        cbias[i] = 0.25f * (float)(i + 1);

    rvcl_conv2d_desc_t cd = {
            .mb = CMB,
            .ic = CIC,
            .ih = CIH,
            .iw = CIW,
            .oc = COC,
            .kh = CKH,
            .kw = CKW,
            .sh = 1,
            .sw = 1,
            .dil_h = 1,
            .dil_w = 1,
            .pad_t = 1,
            .pad_b = 1,
            .pad_l = 1,
            .pad_r = 1,
            .groups = 1,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };

    for (int with_bias = 1; with_bias >= 0; with_bias--) {
        const float *bp = with_bias ? cbias : NULL;
        /* dtype_bias and the pointer must agree: AUTO is the descriptor's
         * "there is no bias" marker, an explicit dtype demands a pointer. */
        cd.dtype_bias = with_bias ? RVCL_DTYPE_FP32 : RVCL_DTYPE_AUTO;
        memset(cdst, 0, sizeof(cdst));
        CHECK(rvcl_conv2d(&cd, csrc, cw, bp, cdst) == RVCL_SUCCESS,
                with_bias ? "rvcl_conv2d with bias succeeds"
                          : "rvcl_conv2d without bias succeeds");

        int cmis = 0;
        for (int oc = 0; oc < COC; oc++)
            for (int y = 0; y < COH; y++)
                for (int x = 0; x < COW; x++) {
                    float acc = bp ? bp[oc] : 0.0f;
                    for (int ic = 0; ic < CIC; ic++)
                        for (int kh = 0; kh < CKH; kh++) {
                            const int ih = y - 1 + kh;
                            if (ih < 0 || ih >= CIH) continue;
                            for (int kw = 0; kw < CKW; kw++) {
                                const int iw = x - 1 + kw;
                                if (iw < 0 || iw >= CIW) continue;
                                acc += csrc[(ic * CIH + ih) * CIW + iw]
                                        * cw[((oc * CIC + ic) * CKH + kh) * CKW
                                                + kw];
                            }
                        }
                    const float got = cdst[(oc * COH + y) * COW + x];
                    if (fabsf(got - acc) > 1e-4f) cmis++;
                }
        CHECK(cmis == 0,
                with_bias ? "conv2d with bias matches naive golden"
                          : "conv2d without bias matches naive golden");
    }

    /* --- inner_product end-to-end: bias and no-bias --- */
    /* [mb,k] x [oc,k]^T + bias[oc] -> [mb,oc], checked against a direct
     * scalar loop. The transposing packer is exercised by the [oc][k]
     * weight layout. The no-bias pass reuses the same buffers. */
    enum { IPMB = 3, IPK = 5, IPOC = 4 };
    float ipsrc[IPMB * IPK];
    float ipw[IPOC * IPK];
    float ipbias[IPOC];
    float ipdst[IPMB * IPOC];
    for (int i = 0; i < IPMB * IPK; i++)
        ipsrc[i] = (float)((i * 7) % 19) / 19.0f - 0.5f;
    for (int i = 0; i < IPOC * IPK; i++)
        ipw[i] = (float)((i * 3) % 13) / 13.0f - 0.5f;
    for (int i = 0; i < IPOC; i++)
        ipbias[i] = 0.5f * (float)(i + 1);

    rvcl_inner_product_desc_t ipd = {
            .mb = IPMB,
            .k = IPK,
            .oc = IPOC,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO,
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {},
    };

    for (int with_bias = 1; with_bias >= 0; with_bias--) {
        const float *bp = with_bias ? ipbias : NULL;
        ipd.dtype_bias = with_bias ? RVCL_DTYPE_FP32 : RVCL_DTYPE_AUTO;
        memset(ipdst, 0, sizeof(ipdst));
        CHECK(rvcl_inner_product(&ipd, ipsrc, ipw, bp, ipdst) == RVCL_SUCCESS,
                with_bias ? "rvcl_inner_product with bias succeeds"
                          : "rvcl_inner_product without bias succeeds");

        int ipmis = 0;
        for (int mb = 0; mb < IPMB; mb++)
            for (int oc = 0; oc < IPOC; oc++) {
                float acc = bp ? bp[oc] : 0.0f;
                for (int kk = 0; kk < IPK; kk++)
                    acc += ipsrc[mb * IPK + kk] * ipw[oc * IPK + kk];
                if (fabsf(ipdst[mb * IPOC + oc] - acc) > 1e-5f) ipmis++;
            }
        CHECK(ipmis == 0,
                with_bias ? "inner_product with bias matches naive golden"
                          : "inner_product without bias matches naive golden");
    }

    /* --- duplicate registration rejected --- */
    /* (re-register the reference kernel by name via public API is not
     * possible without the factory; instead verify the guard through an
     * already-registered name indirectly: count unchanged after init) */
    int before = rvcl_registry_count();
    rvcl_init(); /* idempotent */
    CHECK(rvcl_registry_count() == before, "re-init does not re-register");

    printf(g_failures ? "SMOKE: %d failure(s)\n" : "SMOKE: all passed\n",
            g_failures);
    return g_failures ? 1 : 0;
}
