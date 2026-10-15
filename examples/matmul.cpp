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
 * examples/matmul.cpp — first runnable: dispatch → reference kernel → check
 *
 * The Phase-0 vertical slice (PLAN §6.2): one semantic call exercising
 * capability → registry → dispatch (stages 1-4) → validate/configure →
 * zero-copy import → scheduler → run. RVCL_VERBOSE support lands with
 * observability (PLAN §8.4); for now the selected kernel is printed.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

#include "rvcl/rvcl.h"

int main(void) {
    if (rvcl_init() != RVCL_SUCCESS) {
        fprintf(stderr, "rvcl_init failed\n");
        return 1;
    }
    printf("RVCL %s — %s\n", rvcl_get_version(),
            rvcl_get_capability()->description);

    enum { M = 4, N = 4, K = 4 };
    float a[M * K], b[K * N], bias[N], c[M * N];
    for (int i = 0; i < M * K; i++)
        a[i] = (float)(i % 7);
    for (int i = 0; i < K * N; i++)
        b[i] = (float)((i * 3) % 5);
    for (int j = 0; j < N; j++)
        bias[j] = 1.0f;

    rvcl_matmul_desc_t d = {
            .m = M,
            .n = N,
            .k = K,
            .dtype_a = RVCL_DTYPE_FP32,
            .dtype_b = RVCL_DTYPE_FP32,
            .dtype_acc = RVCL_DTYPE_AUTO, /* habit rule: fp32 → acc fp32 */
            .dtype_c = RVCL_DTYPE_FP32,
            .dtype_bias = RVCL_DTYPE_FP32,
            .quant = {}, /* non-quantized */
    };

    /* Show what dispatch selected (the "dispatch once" half of the API),
     * including the workspace configure() actually reported. */
    rvcl_dispatch_result_t *r = NULL;
    if (rvcl_dispatch_matmul(&d, &r) == RVCL_SUCCESS) {
        int num_ws = 0;
        const rvcl_workspace_req_t *ws = rvcl_dispatch_ws_req(r, &num_ws);
        printf("dispatch selected: %s (ws=%lld)\n",
                rvcl_dispatch_kernel_name(r),
                num_ws > 0 ? (long long)ws[0].size : 0LL);
        rvcl_dispatch_release(r);
    }

    /* Semantic one-shot path. */
    rvcl_status_t st = rvcl_matmul(&d, a, b, bias, c);
    if (st != RVCL_SUCCESS) {
        fprintf(stderr, "rvcl_matmul failed: %d\n", (int)st);
        return 1;
    }

    /* Naive golden check. */
    int bad = 0;
    for (int i = 0; i < M && !bad; i++)
        for (int j = 0; j < N; j++) {
            float acc = bias[j];
            for (int p = 0; p < K; p++)
                acc += a[i * K + p] * b[p * N + j];
            if (c[i * N + j] != acc) {
                bad = 1;
                break;
            }
        }

    printf("%s\n", bad ? "FAIL" : "OK — reference matmul verified");
    return bad ? 1 : 0;
}
