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
 * examples/dispatch.cpp — dispatch-once-run-many + cache observability
 *
 * The hot-path usage pattern (PLAN §4.7): dispatch once per problem
 * shape, configure reports workspace, run many times with different
 * data. Demonstrates: kernel selection query, workspace allocation by
 * the caller, repeated runs, pack B once and hold it across runs
 * (LIFETIME_KERNEL, bit-exact vs in-run packing), dispatch cache
 * hit/miss counters.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rvcl/rvcl.h"

int main(void) {
    if (rvcl_init() != RVCL_SUCCESS) {
        fprintf(stderr, "rvcl_init failed\n");
        return 1;
    }
    printf("RVCL %s — %s\n", rvcl_get_version(),
            rvcl_get_capability()->description);

    enum { M = 64, N = 64, K = 64, BATCHES = 3, PACK_RUNS = 3 };
    float a[M * K], b[K * N], bias[N], c[M * N];
    rvcl_matmul_desc_t d = {
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

    /* 1. Dispatch once (populates the cache). Selection filters by
     * operation, datatype tuple, required features, VLEN tier and shape,
     * then takes the lowest cost; equal costs keep the earliest
     * registration, so dedicated tier variants precede the generic entry
     * and the featureless fallback comes last. */
    rvcl_dispatch_result_t *r = NULL;
    rvcl_status_t st = rvcl_dispatch_matmul(&d, &r);
    if (st != RVCL_SUCCESS || !r) {
        fprintf(stderr, "dispatch failed: %d\n", (int)st);
        return 1;
    }
    printf("dispatch selected: %s\n", rvcl_dispatch_kernel_name(r));

    /* 2. Workspace: the caller is the allocator (PLAN §4.10). */
    int num_ws = 0;
    const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
    void *ws_buf[RVCL_TENSOR_AUX_COUNT] = {0};
    rvcl_tensor_t ws_t[RVCL_TENSOR_AUX_COUNT];
    printf("workspace: %d request(s)\n", num_ws);
    for (int i = 0; i < num_ws && i < RVCL_TENSOR_AUX_COUNT; i++) {
        size_t align = ws[i].alignment > 1 ? (size_t)ws[i].alignment : 8;
        if (posix_memalign(&ws_buf[i], align, (size_t)(ws[i].size + align))
                != 0) {
            fprintf(stderr, "workspace alloc failed\n");
            return 1;
        }
        rvcl_tensor_desc_t wd = {.ndims = 1,
                .dims = {ws[i].size + 1},
                .strides = {1},
                .dtype = RVCL_DTYPE_UINT8};
        rvcl_tensor_init(&ws_t[i], &wd, ws_buf[i]);
        printf("  slot %d: %ld bytes, align %ld\n", i, (long)ws[i].size,
                (long)ws[i].alignment);
    }

    /* 3. Run many: fresh data per batch, same configured kernel. */
    for (int batch = 0; batch < BATCHES; batch++) {
        for (int i = 0; i < M * K; i++)
            a[i] = (float)((i + batch * 7) % 11) / 11.0f;
        for (int i = 0; i < K * N; i++)
            b[i] = (float)((i + batch * 5) % 13) / 13.0f;
        for (int j = 0; j < N; j++)
            bias[j] = (float)batch * 0.1f;
        memset(c, 0, sizeof(c));

        rvcl_tensor_desc_t td = {.ndims = 2,
                .dims = {M, K},
                .strides = {K, 1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_t ta, tb, tc, tbias;
        rvcl_tensor_init(&ta, &td, a);
        td.dims[0] = K;
        td.dims[1] = N;
        td.strides[0] = N;
        rvcl_tensor_init(&tb, &td, b);
        td.dims[0] = M;
        td.strides[0] = N;
        rvcl_tensor_init(&tc, &td, c);
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {N},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(&tbias, &bd, bias);

        rvcl_tensor_pack_t pack;
        rvcl_tensor_pack_init(&pack);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_B, &tb);
        rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_BIAS, &tbias);
        for (int i = 0; i < num_ws && i < RVCL_TENSOR_AUX_COUNT; i++)
            rvcl_tensor_pack_add(&pack,
                    (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + i), &ws_t[i]);

        rvcl_work_range_t full = rvcl_work_range_full(M, N, K);
        st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
        if (st != RVCL_SUCCESS) {
            fprintf(stderr, "run %d failed: %d\n", batch, (int)st);
            return 1;
        }

        /* spot-check one output element against a naive dot product */
        float acc = bias[3];
        for (int p = 0; p < K; p++)
            acc += a[2 * K + p] * b[p * N + 3];
        if (c[2 * N + 3] != acc) {
            fprintf(stderr, "batch %d mismatch: got %f want %f\n", batch,
                    c[2 * N + 3], acc);
            return 1;
        }
        printf("batch %d: OK (spot check c[2,3]=%.4f)\n", batch, c[2 * N + 3]);
    }

    /* 4. Pack once, run many (LIFETIME_KERNEL): pre-pack B into the
     * reported packing workspace via rvcl_kernel_pack_b(), endorse the
     * buffer with pack.src_b_packed, and hold it across several runs
     * whose A/bias vary while B stays fixed — every run() then consumes
     * the packed panels read-only instead of packing per run. The
     * semantic API does this internally per call; holding the buffer
     * explicitly across runs is the adapter-layer pattern. Run 0 also
     * executes the unflagged path on identical inputs: pre-packed and
     * in-run packing must be bit-identical. */
    if (rvcl_dispatch_kernel(r)->vt->pack_b && num_ws >= 1) {
        float c_ref[M * N];
        rvcl_tensor_desc_t td = {.ndims = 2,
                .dims = {M, K},
                .strides = {K, 1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_t ta, tb, tc, tbias;
        rvcl_tensor_init(&ta, &td, a);
        td.dims[0] = K;
        td.dims[1] = N;
        td.strides[0] = N;
        rvcl_tensor_init(&tb, &td, b); /* B stays fixed from here on */
        td.dims[0] = M;
        td.strides[0] = N;
        rvcl_tensor_init(&tc, &td, c);
        rvcl_tensor_desc_t bd = {.ndims = 1,
                .dims = {N},
                .strides = {1},
                .dtype = RVCL_DTYPE_FP32};
        rvcl_tensor_init(&tbias, &bd, bias);

        st = rvcl_kernel_pack_b(rvcl_dispatch_kernel(r), &tb, &ws_t[0]);
        if (st != RVCL_SUCCESS) {
            fprintf(stderr, "pack_b failed: %d\n", (int)st);
            return 1;
        }

        rvcl_tensor_pack_t pack;
        rvcl_tensor_pack_init(&pack);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_A, &ta);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_SRC_B, &tb);
        rvcl_tensor_pack_add_out(&pack, RVCL_TENSOR_DST, &tc);
        rvcl_tensor_pack_add(&pack, RVCL_TENSOR_BIAS, &tbias);
        for (int i = 0; i < num_ws && i < RVCL_TENSOR_AUX_COUNT; i++)
            rvcl_tensor_pack_add(&pack,
                    (rvcl_tensor_slot_t)(RVCL_TENSOR_AUX_0 + i), &ws_t[i]);
        rvcl_work_range_t full = rvcl_work_range_full(M, N, K);

        for (int run = 0; run < PACK_RUNS; run++) {
            for (int i = 0; i < M * K; i++)
                a[i] = (float)((i + run * 3) % 17) / 17.0f;
            for (int j = 0; j < N; j++)
                bias[j] = 0.05f * (float)(run + 1);
            memset(c, 0, sizeof(c));

            if (run == 0) {
                /* unflagged reference: run() packs B itself */
                pack.src_b_packed = 0;
                st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
                if (st != RVCL_SUCCESS) {
                    fprintf(stderr, "unflagged run failed: %d\n", (int)st);
                    return 1;
                }
                memcpy(c_ref, c, sizeof(c_ref));
                memset(c, 0, sizeof(c));
            }

            pack.src_b_packed = 1;
            st = rvcl_kernel_run(rvcl_dispatch_kernel(r), &pack, &full);
            if (st != RVCL_SUCCESS) {
                fprintf(stderr, "pack-once run %d failed: %d\n", run, (int)st);
                return 1;
            }
            if (run == 0) {
                /* identical inputs, identical computation order: the
                 * pre-packed and in-run-packed results must match */
                int diff = 0;
                for (int i = 0; i < M * N; i++)
                    if (c[i] != c_ref[i]) diff = 1;
                if (diff) {
                    fprintf(stderr,
                            "pack-once run 0 differs from in-run packing\n");
                    return 1;
                }
            }

            float acc = bias[3];
            for (int p = 0; p < K; p++)
                acc += a[2 * K + p] * b[p * N + 3];
            if (c[2 * N + 3] != acc) {
                fprintf(stderr, "pack-once run %d mismatch: got %f want %f\n",
                        run, c[2 * N + 3], acc);
                return 1;
            }
            printf("pack-once run %d: OK (spot check c[2,3]=%.4f)%s\n", run,
                    c[2 * N + 3],
                    run == 0 ? ", bit-exact vs in-run packing" : "");
        }
    } else {
        printf("pack-once: selected kernel has no packing step\n");
    }

    /* 5. Cache observability: second dispatch with the same key hits. */
    rvcl_dispatch_result_t *r2 = NULL;
    st = rvcl_dispatch_matmul(&d, &r2);
    printf("dispatch cache: %d hit(s), %d miss(es)%s\n",
            rvcl_dispatch_cache_hits(), rvcl_dispatch_cache_misses(),
            (st == RVCL_SUCCESS && r2 == r) ? " — second dispatch hit" : "");
    /* r2 == r (cache-owned single object): release once. */
    rvcl_dispatch_release(r);

    for (int i = 0; i < RVCL_TENSOR_AUX_COUNT; i++)
        free(ws_buf[i]);
    printf("OK — dispatch once, ran %d batches\n", BATCHES);
    return 0;
}
