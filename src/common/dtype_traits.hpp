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
 * dtype_traits.hpp — library-side dtype fact table (single source of truth)
 *
 * Holds only library facts: element size, family, corpus label, the AUTO
 * accumulation habit, the default assertion level and the tolerance class.
 * Backend headers (rvv_fp16.hpp) and test golden data are deliberately
 * out of scope here to keep the common layer free of a common -> backends/
 * or common -> tests/ dependency inversion; the test-side table in
 * tests/dtype_ops.{hpp,cpp} owns those and references this table.
 ******************************************************************************/

#ifndef RVCL_DTYPE_TRAITS_H
#define RVCL_DTYPE_TRAITS_H

#include "rvcl/kernel.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Numeric family of a datatype (used by the AUTO accumulation habit). */
typedef enum {
    RVCL_DTYPE_FAMILY_NONE = 0, /* AUTO / unknown */
    RVCL_DTYPE_FAMILY_FP,
    RVCL_DTYPE_FAMILY_INT,
} rvcl_dtype_family_t;

/* One row of the library dtype fact table. */
typedef struct {
    /* Element size in bytes; 0 for AUTO / unknown. */
    size_t size;
    rvcl_dtype_family_t family;
    /* Corpus / tool label ("f32", "f16", ...). */
    const char *corpus_name;
    /* The accumulator dtype an AUTO request resolves to when this dtype is
     * the A operand (resolve_acc habit: int8/uint8 -> INT32, else FP32). */
    rvcl_dtype_t acc_default;
    /* Default assertion level for kernels of this dtype
     * (RVCL_ASSERT_BITEXACT for exact dtypes, TOLERANT for fp16). */
    rvcl_assert_level_t assert_level_default;
    /* Relative tolerance class referenced by RVCL_ASSERT_TOLERANT;
     * 0 when the dtype is bit-exact by default. */
    float tolerance;
} rvcl_dtype_traits_t;

/* Bounds-checked row lookup: unknown / out-of-range datatypes return a
 * zeroed row (size 0, NONE family), matching the pre-table
 * rvcl_dtype_size contract. Exported so the unit tests can assert the
 * table facts. */
RVCL_API rvcl_dtype_traits_t rvcl_dtype_traits(rvcl_dtype_t dt);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_DTYPE_TRAITS_H */
