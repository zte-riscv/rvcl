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

/* Single source of truth for the convolution geometry rules. The semantic
 * entry, the dispatch request validator, the kernel adapter and the test
 * corpus cross-check all need the output-size formula and the positivity
 * check, and the corpus-vs-params check is only meaningful while all of them
 * agree, so both live here once instead of being copied per translation
 * unit. Internal to the library and its tests: not installed and not part of
 * the public API. */

#ifndef RVCL_CONV2D_GEOMETRY_H
#define RVCL_CONV2D_GEOMETRY_H

#include <stdint.h>

#include "rvcl/kernel.h"

/* Per-axis output size of the pad/stride/dilation formula:
 * (in + pad_lo + pad_hi - (k - 1)*dil - 1) / s + 1, or 0 when the kernel
 * window does not fit the padded input. Callers validate s > 0 first. */
static inline int64_t rvcl_conv2d_out_dim(int64_t in, int64_t k, int64_t s,
        int64_t dil, int64_t pad_lo, int64_t pad_hi) {
    const int64_t span = in + pad_lo + pad_hi - (k - 1) * dil - 1;
    if (span < 0) return 0;
    return span / s + 1;
}

/* Positivity check the formula and the band sizing depend on: every extent
 * (batch, channels, spatial, kernel) must be >= 1, the stride and dilation
 * divide / multiply, and the pads are non-negative. Returns 1 when valid. */
static inline int rvcl_conv2d_geometry_valid(const rvcl_conv2d_desc_t *d) {
    return d->mb > 0 && d->ic > 0 && d->ih > 0 && d->iw > 0 && d->oc > 0
            && d->kh > 0 && d->kw > 0 && d->sh > 0 && d->sw > 0 && d->dil_h > 0
            && d->dil_w > 0 && d->pad_t >= 0 && d->pad_b >= 0 && d->pad_l >= 0
            && d->pad_r >= 0;
}

#endif /* RVCL_CONV2D_GEOMETRY_H */
