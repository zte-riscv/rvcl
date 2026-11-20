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

#include "dtype_traits.hpp"
#include "rvcl/types.h"

size_t rvcl_dtype_bits(rvcl_dtype_t dt) {
    /* Sub-byte dtype: INT4 packs two elements per byte, so its per-element
     * width is 4 bits rather than size*8. Every other dtype stores one
     * element per size bytes, so its width derives from the trait size. */
    if (dt == RVCL_DTYPE_INT4) return 4;
    return rvcl_dtype_traits(dt).size * 8;
}

size_t rvcl_dtype_size(rvcl_dtype_t dt) {
    /* ceil(bits/8): derived from rvcl_dtype_bits so the two accessors can
     * never contradict. For byte-multiple dtypes this equals the element
     * byte size; for sub-byte dtypes 1 byte is the storage granularity. */
    return (rvcl_dtype_bits(dt) + 7) / 8;
}

/* Block-scale attributes are exposed through these accessors rather than
 * new trait-struct fields: rvcl_dtype_traits_t is exported and returned by
 * value, so appending a field would shift the caller stack frame and break
 * existing binary consumers. Block size / scale dtype are dtype properties,
 * so the desc and the dispatch cache key stay field-free. */
size_t rvcl_dtype_block_size(rvcl_dtype_t dt) {
    switch (dt) {
        case RVCL_DTYPE_MXFP8_E4M3:
        case RVCL_DTYPE_MXFP8_E5M2:
        case RVCL_DTYPE_MXFP4: return 32;
        case RVCL_DTYPE_NVFP4: return 16;
        default: return 0; /* not a block-scale dtype */
    }
}

rvcl_dtype_t rvcl_dtype_scale_dtype(rvcl_dtype_t dt) {
    switch (dt) {
        case RVCL_DTYPE_MXFP8_E4M3:
        case RVCL_DTYPE_MXFP8_E5M2:
        case RVCL_DTYPE_MXFP4: return RVCL_DTYPE_E8M0;
        case RVCL_DTYPE_NVFP4: return RVCL_DTYPE_FP8_E4M3;
        default: return RVCL_DTYPE_AUTO; /* no scale dtype */
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
