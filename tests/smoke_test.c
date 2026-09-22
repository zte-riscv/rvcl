/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * tests/smoke_test.c — Phase 0 smoke test
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Covers the Phase-0 acceptance surface (PLAN §8.1, minimal form):
 *   - registry: register / duplicate / lookup / count
 *   - capability: detection runs, cached pointer stable
 *   - scheduler: set-once semantics, serial parallel_for executes task
 *   - memory contract: zero-copy import (pointer pass-through), non-plain
 *     layout rejected
 *   - dispatch: (A,B,acc,C) filtering — unsupported tuple rejected,
 *     supported tuple selects reference kernel, wrong shape rejected
 *   - kernel lifecycle: configure reports zero workspace for reference
 *   - end-to-end: rvcl_matmul matches naive golden
 ******************************************************************************/

#include <math.h>
#include <stdio.h>
#include <string.h>

#include "rvcl/rvcl.h"

static int g_failures;

#define CHECK(cond, msg) \
    do { \
        if (cond) { printf("  ok: %s\n", msg); } \
        else { \
            printf("  FAIL: %s (line %d)\n", msg, __LINE__); \
            g_failures++; \
        } \
    } while (0)

/* ---------------------------------------------------------- scheduler test */

static int g_task_ran;
static void probe_task(rvcl_ischeduler_t *s, const rvcl_work_range_t *r,
        void *ctx) {
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
    CHECK(rvcl_tensor_bytes(&t) == 16 * 4, "tensor byte size computed");

    rvcl_tensor_desc_t bad_td = td;
    bad_td.strides[1] = 2; /* innermost stride != 1: non-plain layout */
    rvcl_tensor_t t2;
    CHECK(rvcl_tensor_init(&t2, &bad_td, buf) == RVCL_UNSUPPORTED,
            "non-plain layout rejected");

    /* --- dispatch: datatype filtering --- */
    rvcl_matmul_desc_t d = {
        .m = 16, .n = 16, .k = 16,
        .dtype_a = RVCL_DTYPE_FP32,
        .dtype_b = RVCL_DTYPE_FP32,
        .dtype_acc = RVCL_DTYPE_AUTO,
        .dtype_c = RVCL_DTYPE_FP32,
        .dtype_bias = RVCL_DTYPE_AUTO,
    };
    rvcl_dispatch_result_t *r = NULL;
    CHECK(rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS,
            "fp32 tuple dispatches");
    CHECK(r && strcmp(rvcl_dispatch_kernel_name(r), "rvcl_matmul_f32_reference")
                    == 0,
            "selected kernel is reference (only backend)");
    int num_ws = -1;
    rvcl_dispatch_ws_req(r, &num_ws);
    CHECK(num_ws == 0, "reference kernel reports zero workspace");
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
    for (int i = 0; i < M * K; i++) a[i] = (float)((i * 13) % 29) / 29.0f;
    for (int i = 0; i < K * N; i++) b[i] = (float)((i * 7) % 31) / 31.0f;
    for (int j = 0; j < N; j++) bias[j] = 0.5f;
    memset(c, 0, sizeof(c));

    rvcl_matmul_desc_t de = {
        .m = M, .n = N, .k = K,
        .dtype_a = RVCL_DTYPE_FP32,
        .dtype_b = RVCL_DTYPE_FP32,
        .dtype_acc = RVCL_DTYPE_AUTO,
        .dtype_c = RVCL_DTYPE_FP32,
        .dtype_bias = RVCL_DTYPE_FP32,
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
