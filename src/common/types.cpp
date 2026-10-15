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
 * types.cpp — basic type helpers
 ******************************************************************************/

#include "rvcl/types.h"

size_t rvcl_dtype_size(rvcl_dtype_t dt) {
    switch (dt) {
        case RVCL_DTYPE_FP32:
        case RVCL_DTYPE_INT32: return 4;
        case RVCL_DTYPE_FP16:
        case RVCL_DTYPE_BF16: return 2;
        case RVCL_DTYPE_INT8:
        case RVCL_DTYPE_UINT8: return 1;
        case RVCL_DTYPE_INT4: return 1; /* packed nibble units of 1 byte/2el */
        case RVCL_DTYPE_FP8_E4M3:
        case RVCL_DTYPE_FP8_E5M2: return 1;
        default: return 0; /* AUTO / unknown */
    }
}

rvcl_work_range_t rvcl_work_range_full(int64_t m, int64_t n, int64_t k) {
    rvcl_work_range_t r = {.m_begin = 0,
            .m_end = m,
            .n_begin = 0,
            .n_end = n,
            .k_begin = 0,
            .k_end = k};
    return r;
}
