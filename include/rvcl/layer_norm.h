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

#ifndef RVCL_LAYER_NORM_H
#define RVCL_LAYER_NORM_H

#include "rvcl/dispatch.h"
#include "rvcl/memory.h"
#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Layer normalization of every row of a plain row-major [rows][norm_size]
 * tensor described by `desc`. src uses dtype_a; dst is fp32 and the
 * accumulation is fp32, so the up-conversion of a narrow input is exact.
 * scale and shift are optional length-norm_size fp32 vectors: pass NULL for
 * one that desc->use_scale / desc->use_shift leaves off, which is the
 * constant 1 or 0.
 *
 * Returns RVCL_UNSUPPORTED when no kernel serves the requested tuple on
 * this host.
 */
RVCL_API rvcl_status_t rvcl_layer_norm(const rvcl_layer_norm_desc_t *desc,
        const void *src, const void *scale, const void *shift, void *dst);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_LAYER_NORM_H */
