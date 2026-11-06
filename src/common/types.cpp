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

#include "dtype_traits.hpp"
#include "rvcl/types.h"

size_t rvcl_dtype_size(rvcl_dtype_t dt) {
    /* Table-backed: the element size is a dtype fact owned by the trait
     * table (dtype_traits.hpp), so a new dtype only adds a table row. */
    return rvcl_dtype_traits(dt).size;
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
