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

#include "rvcl/kernel.h"
#include "rvcl/types.h"

/* Whole-table fallback check (FR1.7): every datatype tuple that has at
 * least one live entry must contain a kernel without shape limits. Run
 * once after built-in registration; returns
 * RVCL_ERROR_REGISTRY_INVARIANT on violation.
 *
 * `before` is the registry entry count at the start of the built-in
 * registration pass (the init-path snapshot of app-owned entries). A
 * datatype tuple whose every entry was registered before that pass (i.e.
 * purely app-owned, index < before) is exempt from the check: a plugin's
 * intentionally shape-bounded kernel on an otherwise-unoccupied tuple must
 * not fail the library init, and out-of-range shapes on that tuple simply
 * dispatch to RVCL_UNSUPPORTED. Any tuple a built-in participates in keeps
 * the original requirement (built-ins always carry an unbounded fallback). */
rvcl_status_t rvcl_registry_verify_fallback_invariant(int before);

/* Drops every entry registered after the first `count` entries, in
 * reverse registration order. The init path uses this to roll back a
 * failed built-in registration pass, restoring the pre-init table. Unlike
 * rvcl_unregister_kernel() the fallback guard does not apply: entries
 * appended by a failed pass must disappear even when the remaining user
 * entries would not satisfy the invariant on their own. */
rvcl_status_t rvcl_registry_rollback(int count);

/* Copy of one live registry entry as seen by dispatch. Descriptors are
 * copied out, so the selection loop works on a coherent table without
 * holding the registry lock while it calls factories, cost callbacks or
 * validate(). */
typedef struct {
    rvcl_kernel_desc_t desc;
    rvcl_kernel_create_fn create;
} rvcl_registry_view_t;

/*
 * Copies up to max_entries live entries in registration order into out
 * and stores the registry generation observed for that copy in
 * *generation. Returns the number of entries copied. The copy is
 * consistent: concurrent registrations and unregistrations are never
 * observed half-applied.
 */
int rvcl_registry_snapshot(
        rvcl_registry_view_t *out, int max_entries, unsigned *generation);

/*
 * Monotonic change counter: bumped by every successful registration,
 * unregistration and rollback. Dispatch tags cache entries with the
 * generation they were selected under, so a change racing a selection
 * cannot leave a stale selection behind.
 */
unsigned rvcl_registry_generation(void);

#endif /* RVCL_REGISTRY_INTERNAL_H */
