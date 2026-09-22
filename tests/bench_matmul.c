/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * tests/bench_matmul.c — benchmark harness with JSON output
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * PLAN §6.2 item 10 + §8.2: four metric classes per (shape, dtype):
 *   kernel-only  — micro-kernel on pre-packed data (panelized B)
 *   pack-only    — the packing transform alone
 *   pack+kernel  — run() as the framework delivers it (pack inside)
 *   end-to-end   — rvcl_matmul() including dispatch + cache
 *
 * Usage: rvcl_bench [--json] [--reps N] [shape ...]
 *   shape syntax: MxNxK (e.g. 128x128x128). Default sweep below.
 * Output: human table, or one JSON object per line (metrics) with --json.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "rvcl/rvcl.h"
#include "../src/backends/rvv/rvv_fp16.h"
#include "../src/backends/rvv/rvv_internal.h"

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

/* steady-state timing: warmup once, best-of over reps */
static double time_best_of(void (*fn)(void *), void *ctx, int reps) {
    fn(ctx); /* warmup */
    double best = 1e30;
    for (int i = 0; i < reps; i++) {
        double t0 = now_ms();
        fn(ctx);
        double dt = now_ms() - t0;
        if (dt < best) best = dt;
    }
    return best;
}

/* ------------------------------------------------------- bench contexts */

typedef struct {
    int64_t m, n, k;
    float *a, *b, *c;
    float *bias;
    uint16_t *ah, *bh;
    rvcl_matmul_desc_t d;
} case_t;

/* kernel-only: pre-packed B + direct micro-kernel (C path; the RVV
 * intrinsics path is selected automatically on RVV hosts) */
typedef struct {
    const case_t *cs;
    void *packed;
    rvv_gemm_fn gemm;
} konly_ctx_t;

static void bench_kernel_only(void *p) {
    konly_ctx_t *kc = (konly_ctx_t *)p;
    const case_t *cs = kc->cs;
    rvcl_work_range_t full
            = rvcl_work_range_full(cs->m, cs->n, cs->k);
    if (cs->d.dtype_a == RVCL_DTYPE_FP32) {
        rvv_gemm_f32_c(cs->a, kc->packed, cs->bias, cs->c,
                cs->m, cs->n, cs->k, &full);
    } else {
        rvv_gemm_f16_c(cs->ah, kc->packed, cs->bias, cs->c,
                cs->m, cs->n, cs->k, &full);
    }
}

typedef struct {
    const case_t *cs;
    void *packed;
    const rvcl_tensor_t *b;
    rvcl_tensor_t *dst;
} pack_ctx_t;

static void bench_pack_only(void *p) {
    pack_ctx_t *pc = (pack_ctx_t *)p;
    const case_t *cs = pc->cs;
    size_t elt = rvcl_dtype_size(cs->d.dtype_a);
    rvv_pack_b_panel(pc->b, pc->dst, cs->m, cs->n, cs->k, elt);
}

typedef struct {
    const case_t *cs;
    rvcl_dispatch_result_t *r;
    rvcl_tensor_t ta, tb, tc, tbias;
    rvcl_tensor_t ws_t[RVCL_MAX_WORKSPACES];
    void *ws_buf[RVCL_MAX_WORKSPACES];
    int num_ws;
    rvcl_tensor_pack_t pack;
    rvcl_work_range_t full;
} pk_ctx_t;

static void bench_pack_kernel(void *p) {
    pk_ctx_t *pc = (pk_ctx_t *)p;
    rvcl_kernel_run(rvcl_dispatch_kernel(pc->r), &pc->pack, &pc->full);
}

typedef struct {
    const case_t *cs;
} e2e_ctx_t;

static void bench_e2e(void *p) {
    e2e_ctx_t *ec = (e2e_ctx_t *)p;
    const case_t *cs = ec->cs;
    if (cs->d.dtype_a == RVCL_DTYPE_FP32)
        rvcl_matmul(&cs->d, cs->a, cs->b, cs->bias, cs->c);
    else
        rvcl_matmul(&cs->d, cs->ah, cs->bh, cs->bias, cs->c);
}

/* --------------------------------------------------------------- main */

int main(int argc, char **argv) {
    int json = 0, reps = 5;
    static const int64_t defaults[][3] = {{64, 64, 64}, {128, 128, 128},
            {256, 256, 256}, {128, 256, 512}};
    int64_t shapes[16][3];
    int num_shapes = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc)
            reps = atoi(argv[++i]);
        else if (num_shapes < 16) {
            /* shape argument: MxNxK (replaces the default sweep) */
            long long m, n, k;
            if (sscanf(argv[i], "%lldx%lldx%lld", &m, &n, &k) == 3
                    && m > 0 && n > 0 && k > 0) {
                shapes[num_shapes][0] = m;
                shapes[num_shapes][1] = n;
                shapes[num_shapes][2] = k;
                num_shapes++;
            }
        }
    }
    if (num_shapes == 0) { /* no shape args: default sweep */
        num_shapes = (int)(sizeof(defaults) / sizeof(defaults[0]));
        for (int i = 0; i < num_shapes; i++)
            for (int j = 0; j < 3; j++) shapes[i][j] = defaults[i][j];
    }

    if (rvcl_init() != RVCL_SUCCESS) return 1;

    if (!json) {
        printf("RVCL %s bench — %s\n", rvcl_get_version(),
                rvcl_get_capability()->description);
        printf("%-16s %-5s %10s %10s %10s %10s\n", "shape", "dtype",
                "kernel", "pack", "pack+kernel", "end-to-end");
    }

    for (int s = 0; s < num_shapes; s++) {
        case_t cs = {0};
        cs.m = shapes[s][0];
        cs.n = shapes[s][1];
        cs.k = shapes[s][2];

        for (int dt = 0; dt < 2; dt++) { /* 0: f32, 1: f16 */
            cs.d = (rvcl_matmul_desc_t){
                .m = cs.m, .n = cs.n, .k = cs.k,
                .dtype_a = dt ? RVCL_DTYPE_FP16 : RVCL_DTYPE_FP32,
                .dtype_b = dt ? RVCL_DTYPE_FP16 : RVCL_DTYPE_FP32,
                .dtype_acc = RVCL_DTYPE_AUTO,
                .dtype_c = RVCL_DTYPE_FP32,
                .dtype_bias = RVCL_DTYPE_AUTO,
            };

            size_t elt = dt ? 2 : 4;
            /* Fill as f32 first (golden values), then narrow to f16
             * storage: a/b always hold f32-wide fill buffers; the f16
             * path reads only ah/bh. */
            cs.a = calloc((size_t)(cs.m * cs.k), 4);
            cs.b = calloc((size_t)(cs.k * cs.n), 4);
            cs.c = calloc((size_t)(cs.m * cs.n), 4);
            cs.bias = calloc((size_t)cs.n, 4);
            for (int64_t i = 0; i < cs.m * cs.k; i++)
                ((float *)cs.a)[i] = (float)(i % 7) / 7.0f - 0.5f;
            for (int64_t i = 0; i < cs.k * cs.n; i++)
                ((float *)cs.b)[i] = (float)(i % 5) / 5.0f - 0.5f;
            for (int64_t j = 0; j < cs.n; j++) cs.bias[j] = 0.125f;
            if (dt) {
                cs.ah = calloc((size_t)(cs.m * cs.k), 2);
                cs.bh = calloc((size_t)(cs.k * cs.n), 2);
                for (int64_t i = 0; i < cs.m * cs.k; i++)
                    cs.ah[i] = rvv_f32_to_f16(((float *)cs.a)[i]);
                for (int64_t i = 0; i < cs.k * cs.n; i++)
                    cs.bh[i] = rvv_f32_to_f16(((float *)cs.b)[i]);
            }

            const double flops = 2.0 * cs.m * cs.n * cs.k;

            /* --- kernel-only (pre-packed) --- */
            void *packed = NULL;
            size_t pbytes = rvv_packed_b_bytes(cs.k, cs.n, elt);
            posix_memalign(&packed, 64, pbytes);
            {
                rvcl_tensor_desc_t td = {.ndims = 2,
                    .dims = {cs.k, cs.n},
                    .strides = {cs.n, 1},
                    .dtype = cs.d.dtype_b};
                rvcl_tensor_t src, dst;
                rvcl_tensor_init(&src, &td, dt ? cs.bh : cs.b);
                rvcl_tensor_desc_t pd = {.ndims = 1,
                    .dims = {(int64_t)pbytes},
                    .strides = {1},
                    .dtype = RVCL_DTYPE_UINT8};
                rvcl_tensor_init(&dst, &pd, packed);
                rvv_pack_b_panel(&src, &dst, cs.m, cs.n, cs.k, elt);
            }
            konly_ctx_t kc = {.cs = &cs, .packed = packed};
            double t_kernel = time_best_of(bench_kernel_only, &kc, reps);

            /* --- pack-only --- */
            pack_ctx_t pc = {0};
            {
                rvcl_tensor_desc_t td = {.ndims = 2,
                    .dims = {cs.k, cs.n},
                    .strides = {cs.n, 1},
                    .dtype = cs.d.dtype_b};
                static rvcl_tensor_t src_t, dst_t;
                rvcl_tensor_init(&src_t, &td, dt ? cs.bh : cs.b);
                rvcl_tensor_desc_t pd = {.ndims = 1,
                    .dims = {(int64_t)pbytes},
                    .strides = {1},
                    .dtype = RVCL_DTYPE_UINT8};
                rvcl_tensor_init(&dst_t, &pd, packed);
                pc.cs = &cs;
                pc.packed = packed;
                pc.b = &src_t;
                pc.dst = &dst_t;
            }
            double t_pack = time_best_of(bench_pack_only, &pc, reps);

            /* --- pack+kernel (framework run()) --- */
            pk_ctx_t pck = {0};
            rvcl_dispatch_result_t *r = NULL;
            if (rvcl_dispatch_matmul(&cs.d, &r) == RVCL_SUCCESS) {
                int num_ws = 0;
                const rvcl_workspace_req_t *ws
                        = rvcl_dispatch_ws_req(r, &num_ws);
                pck.cs = &cs;
                pck.r = r;
                for (int i = 0; i < num_ws && i < RVCL_MAX_WORKSPACES; i++) {
                    size_t align
                            = ws[i].alignment > 1 ? (size_t)ws[i].alignment
                                                  : 8;
                    posix_memalign(&pck.ws_buf[i], align,
                            (size_t)(ws[i].size + align));
                    rvcl_tensor_desc_t wd = {.ndims = 1,
                        .dims = {ws[i].size + 1},
                        .strides = {1},
                        .dtype = RVCL_DTYPE_UINT8};
                    rvcl_tensor_init(&pck.ws_t[i], &wd, pck.ws_buf[i]);
                }
                pck.num_ws = num_ws;

                rvcl_tensor_desc_t td = {.ndims = 2,
                    .dims = {cs.m, cs.k},
                    .strides = {cs.k, 1},
                    .dtype = cs.d.dtype_a};
                rvcl_tensor_init(&pck.ta, &td, dt ? cs.ah : cs.a);
                td.dims[0] = cs.k;
                td.dims[1] = cs.n;
                td.strides[0] = cs.n;
                td.dtype = cs.d.dtype_b;
                rvcl_tensor_init(&pck.tb, &td, dt ? cs.bh : cs.b);
                td.dims[0] = cs.m;
                td.strides[0] = cs.n;
                td.dtype = cs.d.dtype_c;
                rvcl_tensor_init(&pck.tc, &td, cs.c);
                rvcl_tensor_desc_t bd = {.ndims = 1,
                    .dims = {cs.n},
                    .strides = {1},
                    .dtype = RVCL_DTYPE_FP32};
                rvcl_tensor_init(&pck.tbias, &bd, cs.bias);

                rvcl_tensor_pack_init(&pck.pack);
                rvcl_tensor_pack_add(&pck.pack, RVCL_TENSOR_SRC_A, &pck.ta);
                rvcl_tensor_pack_add(&pck.pack, RVCL_TENSOR_SRC_B, &pck.tb);
                rvcl_tensor_pack_add_out(&pck.pack, RVCL_TENSOR_DST, &pck.tc);
                rvcl_tensor_pack_add(&pck.pack, RVCL_TENSOR_BIAS, &pck.tbias);
                for (int i = 0; i < num_ws && i < RVCL_MAX_WORKSPACES; i++)
                    rvcl_tensor_pack_add(&pck.pack,
                            RVCL_TENSOR_AUX_0 + i, &pck.ws_t[i]);
                pck.full = rvcl_work_range_full(cs.m, cs.n, cs.k);
            }
            double t_pk = r ? time_best_of(bench_pack_kernel, &pck, reps) : 0;
            double t_e2e = 0;
            if (r) { /* semantic API needs a dispatchable kernel */
                e2e_ctx_t ec = {.cs = &cs};
                t_e2e = time_best_of(bench_e2e, &ec, reps);
            }

            if (json) {
                char pk_s[32], e2e_s[32], gfe_s[32];
                if (r) {
                    snprintf(pk_s, sizeof(pk_s), "%.4f", t_pk);
                    snprintf(e2e_s, sizeof(e2e_s), "%.4f", t_e2e);
                    snprintf(gfe_s, sizeof(gfe_s), "%.2f",
                            flops / (t_e2e * 1e6));
                } else { /* no dispatchable kernel on this host */
                    strcpy(pk_s, "null");
                    strcpy(e2e_s, "null");
                    strcpy(gfe_s, "null");
                }
                printf("{\"shape\":\"%lldx%lldx%lld\",\"dtype\":\"%s\","
                       "\"kernel_ms\":%.4f,\"pack_ms\":%.4f,"
                       "\"pack_kernel_ms\":%s,\"e2e_ms\":%s,"
                       "\"gflops_kernel\":%.2f,\"gflops_e2e\":%s}\n",
                        (long long)cs.m, (long long)cs.n, (long long)cs.k,
                        dt ? "f16" : "f32",
                        t_kernel, t_pack, pk_s, e2e_s,
                        flops / (t_kernel * 1e6), gfe_s);
            } else {
                char sh[32];
                snprintf(sh, sizeof(sh), "%lldx%lldx%lld",
                        (long long)cs.m, (long long)cs.n, (long long)cs.k);
                if (r)
                    printf("%-16s %-5s %10.3f %10.3f %10.3f %10.3f\n",
                            sh, dt ? "f16" : "f32",
                            t_kernel, t_pack, t_pk, t_e2e);
                else /* e.g. f16 on a host without RVV capability */
                    printf("%-16s %-5s %10.3f %10.3f %10s %10s\n",
                            sh, dt ? "f16" : "f32",
                            t_kernel, t_pack, "n/a", "n/a");
            }

            /* cleanup per case */
            if (r) {
                for (int i = 0; i < RVCL_MAX_WORKSPACES; i++)
                    free(pck.ws_buf[i]);
                rvcl_dispatch_release(r);
            }
            free(packed);
            free(cs.a); free(cs.b); free(cs.c); free(cs.bias);
            free(cs.ah); free(cs.bh);
        }
    }
    return 0;
}
