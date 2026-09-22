/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * types.c — basic type helpers
 *
 * SPDX-License-Identifier: Apache-2.0
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
    rvcl_work_range_t r
            = {.m_begin = 0, .m_end = m, .n_begin = 0, .n_end = n,
                    .k_begin = 0, .k_end = k};
    return r;
}
