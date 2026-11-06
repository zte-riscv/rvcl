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
 * tests/dtype_ops.hpp — test-side dtype fact table
 *
 * Owns the per-dtype TEST facts (dtstr, golden, fill, the soft-convert
 * reference and the compare routing); the library-side dtype fact table
 * (src/common/dtype_traits.hpp) owns the LIBRARY facts (size, family,
 * corpus name, acc habit, assertion-level default, tolerance class). The
 * test table references the library table for the tolerance class, so the
 * two layers stay single-source-of-truth without the common layer ever
 * referencing test or backend headers.
 ******************************************************************************/

#ifndef RVCL_TEST_DTYPE_OPS_H
#define RVCL_TEST_DTYPE_OPS_H

#include <stddef.h>
#include <stdint.h>

#include "golden.hpp"
#include "rvcl/rvcl.h"

/* Per-dtype test operation bundle. */
typedef struct {
    /* Corpus / CLI dtype string ("f32", "f16"). */
    const char *dtstr;
    rvcl_dtype_t dtype;
    /* Storage element size in bytes (buffer sizing). */
    size_t elt;
    /* Golden matmul: dtype-stored A/B buffers, fp32 output. */
    void (*golden)(const void *a, const void *b, const float *bias, float *c,
            int64_t m, int64_t n, int64_t k);
    /* Deterministic LCG fill of a dtype-storage buffer. */
    void (*fill)(void *buf, size_t n, uint32_t seed);
    /* Compare routing, driven by the kernel desc assertion level; the
     * tolerance class comes from the library trait table by dtype. */
    golden_cmp_t (*compare)(const float *got, const float *exp, int64_t n,
            rvcl_assert_level_t level, rvcl_dtype_t dtype);
    /* Soft-convert reference: fp32 value -> dtype storage
     * (identity for fp32, rvv_f32_to_f16 for fp16). */
    uint16_t (*f32_to_storage)(float);
} dtype_ops_t;

/* Row lookup by datatype; NULL when the datatype has no test bundle. */
const dtype_ops_t *dtype_ops_for(rvcl_dtype_t dt);

/* dtstr -> datatype; returns 1 and sets *out on a known string, else 0. */
int dtype_ops_from_dtstr(const char *s, rvcl_dtype_t *out);

#endif /* RVCL_TEST_DTYPE_OPS_H */
