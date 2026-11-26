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

/* Single source of truth for the pooling geometry rules, the counterpart of
 * conv2d_geometry.h: the semantic entry, the dispatch request validator, the
 * kernel adapter and the test corpus cross-check all need the output-size
 * formula and the request check, so both live here once. Internal to the
 * library and its tests: not installed and not part of the public API. */

#ifndef RVCL_POOLING_GEOMETRY_H
#define RVCL_POOLING_GEOMETRY_H

#include <stdint.h>

#include "rvcl/kernel.h"

/* Per-axis output size of the pad/stride formula:
 * (in + pad_lo + pad_hi - k) / s + 1, or 0 when the window does not fit the
 * padded input. Callers validate s > 0 and k > 0 first. */
static inline int64_t rvcl_pooling_out_dim(
        int64_t in, int64_t k, int64_t s, int64_t pad_lo, int64_t pad_hi) {
    const int64_t span = in + pad_lo + pad_hi - k;
    if (span < 0) return 0;
    return span / s + 1;
}

/* Request check the formula and the kernel loop bounds depend on: every
 * extent (batch, channels, spatial, window) must be >= 1, the stride
 * divides, the pads are non-negative and the algorithm is one of the three
 * defined values. Returns 1 when valid. */
static inline int rvcl_pooling_geometry_valid(const rvcl_pooling_desc_t *d) {
    return d->mb > 0 && d->ic > 0 && d->ih > 0 && d->iw > 0 && d->kh > 0
            && d->kw > 0 && d->sh > 0 && d->sw > 0 && d->pad_t >= 0
            && d->pad_b >= 0 && d->pad_l >= 0 && d->pad_r >= 0
            && (int)d->alg >= (int)RVCL_POOL_MAX
            && (int)d->alg <= (int)RVCL_POOL_AVG_EXCLUDE;
}

#endif /* RVCL_POOLING_GEOMETRY_H */
