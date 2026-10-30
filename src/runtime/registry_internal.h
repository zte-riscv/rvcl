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
 * registry_internal.h — registry hooks used by the initialization path
 *
 * Not installed and not part of the public API; only library translation
 * units include this header.
 ******************************************************************************/

#ifndef RVCL_REGISTRY_INTERNAL_H
#define RVCL_REGISTRY_INTERNAL_H

#include "rvcl/types.h"

/* Whole-table fallback check (FR1.7): every datatype tuple that has at
 * least one live entry must contain a kernel without shape limits. Run
 * once after built-in registration; returns
 * RVCL_ERROR_REGISTRY_INVARIANT on violation. */
rvcl_status_t rvcl_registry_verify_fallback_invariant(void);

/* Drops every entry registered after the first `count` entries, in
 * reverse registration order. The init path uses this to roll back a
 * failed built-in registration pass, restoring the pre-init table. Unlike
 * rvcl_unregister_kernel() the fallback guard does not apply: entries
 * appended by a failed pass must disappear even when the remaining user
 * entries would not satisfy the invariant on their own. */
rvcl_status_t rvcl_registry_rollback(int count);

#endif /* RVCL_REGISTRY_INTERNAL_H */
