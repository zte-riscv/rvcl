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
 * dtype_traits.cpp — constexpr dtype fact table
 *
 * One row per rvcl_dtype_t (AUTO..FP8_E5M2), indexed by the enum value so
 * a lookup is a direct array read. The AUTO accumulation habit here must
 * stay behavior-identical to the historical resolve_acc: int8/uint8 ->
 * INT32, every other concrete dtype -> FP32.
 ******************************************************************************/

#include "dtype_traits.hpp"

static constexpr rvcl_dtype_traits_t k_dtype_traits[] = {
        /* RVCL_DTYPE_AUTO */
        {0, RVCL_DTYPE_FAMILY_NONE, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT,
                0.0f},
        /* RVCL_DTYPE_FP32 */
        {4, RVCL_DTYPE_FAMILY_FP, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT, 0.0f},
        /* RVCL_DTYPE_FP16 */
        {2, RVCL_DTYPE_FAMILY_FP, RVCL_DTYPE_FP32, RVCL_ASSERT_TOLERANT, 1e-3f},
        /* RVCL_DTYPE_BF16 */
        {2, RVCL_DTYPE_FAMILY_FP, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT, 0.0f},
        /* RVCL_DTYPE_INT8 */
        {1, RVCL_DTYPE_FAMILY_INT, RVCL_DTYPE_INT32, RVCL_ASSERT_BITEXACT,
                0.0f},
        /* RVCL_DTYPE_UINT8 */
        {1, RVCL_DTYPE_FAMILY_INT, RVCL_DTYPE_INT32, RVCL_ASSERT_BITEXACT,
                0.0f},
        /* RVCL_DTYPE_INT4 */
        {1, RVCL_DTYPE_FAMILY_INT, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT, 0.0f},
        /* RVCL_DTYPE_INT32 */
        {4, RVCL_DTYPE_FAMILY_INT, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT, 0.0f},
        /* RVCL_DTYPE_FP8_E4M3 */
        {1, RVCL_DTYPE_FAMILY_FP, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT, 0.0f},
        /* RVCL_DTYPE_FP8_E5M2 */
        {1, RVCL_DTYPE_FAMILY_FP, RVCL_DTYPE_FP32, RVCL_ASSERT_BITEXACT, 0.0f},
};

static constexpr int k_dtype_traits_count
        = (int)(sizeof(k_dtype_traits) / sizeof(k_dtype_traits[0]));

rvcl_dtype_traits_t rvcl_dtype_traits(rvcl_dtype_t dt) {
    const int idx = (int)dt;
    if (idx < 0 || idx >= k_dtype_traits_count) return rvcl_dtype_traits_t {};
    return k_dtype_traits[idx];
}
