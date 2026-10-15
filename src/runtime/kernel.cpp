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
 * kernel.cpp — kernel instance lifecycle helpers
 ******************************************************************************/

#include "rvcl/kernel.h"

#include <stdlib.h>
#include <string.h>

rvcl_status_t rvcl_kernel_alloc(rvcl_kernel_t **out,
        const rvcl_kernel_desc_t *desc, size_t instance_size) {
    if (!out || !desc || instance_size < sizeof(rvcl_kernel_t))
        return RVCL_ERROR_INVALID_ARGUMENT;

    rvcl_kernel_t *k = (rvcl_kernel_t *)calloc(1, instance_size);
    if (!k) return RVCL_ERROR_OUT_OF_MEMORY;
    k->desc = *desc; /* copies metadata into the instance */
    *out = k;
    return RVCL_SUCCESS;
}

rvcl_status_t rvcl_kernel_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    if (!k || !desc || !k->vt || !k->vt->configure)
        return RVCL_ERROR_INVALID_ARGUMENT;
    return k->vt->configure(k, desc, ws_req, num_ws);
}

rvcl_status_t rvcl_kernel_run(rvcl_kernel_t *k, const rvcl_tensor_pack_t *pack,
        const rvcl_work_range_t *range) {
    if (!k || !pack || !range || !k->vt || !k->vt->run)
        return RVCL_ERROR_INVALID_ARGUMENT;
    return k->vt->run(k, pack, range);
}

void rvcl_kernel_destroy(rvcl_kernel_t *k) {
    if (!k) return;
    if (k->vt && k->vt->destroy) k->vt->destroy(k);
    free(k);
}
