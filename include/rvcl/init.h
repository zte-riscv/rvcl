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
 * init.h — library initialization and version
 ******************************************************************************/

#ifndef RVCL_INIT_H
#define RVCL_INIT_H

#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers built-in backends (idempotent, thread-safe). */
RVCL_API rvcl_status_t rvcl_init(void);

/* "MAJOR.MINOR.PATCH[-dev]" — consumed by FindRVCL-style version gating
 * (PLAN §4.11 item 7, the ACL 53.1-gate equivalent). */
RVCL_API const char *rvcl_get_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_INIT_H */
