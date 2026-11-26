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

/* One-shot convenience path: dispatch + run + release. Applications that
 * need "dispatch once, run many" or custom scheduling should use
 * dispatch.h + kernel.h directly. */

#ifndef RVCL_POOLING_H
#define RVCL_POOLING_H

#include "rvcl/dispatch.h"
#include "rvcl/memory.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * NCHW 2D pooling described by `desc`: src and dst are row-major NCHW
 * [mb][ic][ih][iw] and [mb][ic][oh][ow]. src uses dtype_a; dst is fp32 and
 * the accumulation is fp32, so the up-conversion of a narrow input is
 * exact. The algorithm (max, avg including padding, avg excluding padding)
 * is desc->alg; see rvcl_pooling_desc_t for the sample set. Pass
 * RVCL_DTYPE_AUTO for acc to apply the habit rules.
 *
 * Returns RVCL_UNSUPPORTED when no kernel serves the requested tuple on
 * this host.
 */
RVCL_API rvcl_status_t rvcl_pooling(
        const rvcl_pooling_desc_t *desc, const void *src, void *dst);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_POOLING_H */
