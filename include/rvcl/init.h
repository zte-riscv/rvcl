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

/*
 * Registers built-in backends (idempotent, thread-safe). Applications may
 * call rvcl_init() explicitly; dispatch also triggers it lazily, so a
 * first computation without an explicit init still sees a full registry.
 *
 * Registration timing contract: kernels registered before the first
 * rvcl_init() (plugin constructor or application init path, see registry.h)
 * are preserved — a failed init rolls back only the built-ins it
 * registered, never application entries, and a later retry starts from the
 * same pre-init state. A datatype tuple whose every entry was registered
 * before init (purely app-owned) is exempt from the built-in fallback
 * requirement: a plugin may register only shape-bounded kernels on a tuple
 * no built-in occupies, and out-of-range shapes on that tuple then dispatch
 * to RVCL_UNSUPPORTED rather than failing the whole init. Registration is
 * assumed not to be concurrent with the first rvcl_init().
 */
RVCL_API rvcl_status_t rvcl_init(void);

/* "MAJOR.MINOR.PATCH[-dev]" — consumed by FindRVCL-style version gating
 * (PLAN §4.11 item 7, the ACL 53.1-gate equivalent). */
RVCL_API const char *rvcl_get_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_INIT_H */
