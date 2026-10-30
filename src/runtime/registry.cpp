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
 * registry.cpp — kernel registration and indexing
 *
 * Stable-slot storage with an ordered index (RFC-0002 registry):
 *   - descriptors live in fixed slots that are never moved by unregister,
 *     so a pointer handed out by rvcl_registry_get() stays valid while
 *     the kernel it refers to is registered;
 *   - g_order keeps the registration sequence; dispatch relies on it as
 *     the deterministic tie-break between equal-cost candidates;
 *   - freed slots are reused only after all never-used slots are spent.
 *     A pointer to an unregistered kernel therefore must not be used: its
 *     slot may later hold a different descriptor.
 *
 * Registrations are fully validated before any state changes (reject
 * atomically, never partially register). Unregistering the last unbounded
 * kernel of a live datatype tuple is rejected so every tuple keeps a
 * fallback. No performance decisions here: cost_fn is stored, never
 * called.
 ******************************************************************************/

#include "rvcl/registry.h"

#include "registry_internal.h"
#include "rvcl/dispatch.h"

#include <pthread.h>
#include <string.h>

typedef struct {
    rvcl_kernel_desc_t desc;
    rvcl_kernel_create_fn create;
    int used;
} registry_slot_t;

static registry_slot_t g_slots[RVCL_REGISTRY_MAX_KERNELS];
/* Slot indices in registration order (live entries only). */
static int g_order[RVCL_REGISTRY_MAX_KERNELS];
static int g_count;
static int g_slots_used; /* high-water mark of slots ever allocated */
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

/* -------------------------------------------------------------- zero cost */

RVCL_API uint64_t rvcl_kernel_cost_zero(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return 0; /* "no cost information" (RFC-0002 FR4.2) */
}

/* ------------------------------------------------------------ validation */

static int dtype_is_concrete(rvcl_dtype_t dt) {
    /* Descriptors are match keys: AUTO is a request-side marker only. */
    return dt > RVCL_DTYPE_AUTO && dt <= RVCL_DTYPE_FP8_E5M2;
}

static int feature_id_is_valid(rvcl_feature_id_t id) {
    if (id == RVCL_FEATURE_NONE) return 0;
    if (id >= RVCL_FEATURE_RV64 && id <= RVCL_FEATURE_INT4) return 1;
    return id >= RVCL_FEATURE_VENDOR_FIRST;
}

static int tier_is_valid(rvcl_vlen_tier_t t) {
    switch (t) {
        case RVCL_VLEN_TIER_ANY:
        case RVCL_VLEN_TIER_128:
        case RVCL_VLEN_TIER_256:
        case RVCL_VLEN_TIER_512:
        case RVCL_VLEN_TIER_1024: return 1;
        default: return 0;
    }
}

static int quant_is_valid(rvcl_quant_support_t q) {
    switch (q) {
        case RVCL_QUANT_NONE:
        case RVCL_QUANT_PER_TENSOR:
        case RVCL_QUANT_PER_CHANNEL: return 1;
        default: return 0;
    }
}

/* Full-descriptor gate; runs before taking the lock so a rejected
 * registration cannot leave partial state behind. */
static rvcl_status_t validate_desc(
        const rvcl_kernel_desc_t *desc, rvcl_kernel_create_fn create) {
    if (!desc || !create) return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->name[0] == '\0') return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->name[RVCL_KERNEL_NAME_MAX - 1] != '\0')
        return RVCL_ERROR_INVALID_ARGUMENT; /* not NUL-terminated */
    if (desc->operation < 0 || desc->operation >= RVCL_OP_COUNT)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (!dtype_is_concrete(desc->dtype_a) || !dtype_is_concrete(desc->dtype_b)
            || !dtype_is_concrete(desc->dtype_acc)
            || !dtype_is_concrete(desc->dtype_c))
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (!quant_is_valid(desc->quant_support))
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->num_required_features < 0
            || desc->num_required_features > RVCL_MAX_FEATURES)
        return RVCL_ERROR_INVALID_ARGUMENT;
    for (int i = 0; i < desc->num_required_features; i++)
        if (!feature_id_is_valid(desc->required_features[i]))
            return RVCL_ERROR_INVALID_ARGUMENT;
    if (desc->min_m < 0 || desc->max_m < 0 || desc->min_n < 0 || desc->max_n < 0
            || desc->min_k < 0 || desc->max_k < 0)
        return RVCL_ERROR_INVALID_ARGUMENT;
    /* max == 0 means unbounded on that side. */
    if ((desc->max_m != 0 && desc->min_m > desc->max_m)
            || (desc->max_n != 0 && desc->min_n > desc->max_n)
            || (desc->max_k != 0 && desc->min_k > desc->max_k))
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (!tier_is_valid(desc->vlen_tier)) return RVCL_ERROR_INVALID_ARGUMENT;
    if (!desc->cost_fn) return RVCL_ERROR_INVALID_ARGUMENT;
    return RVCL_SUCCESS;
}

/* ----------------------------------------------------------- slot lookup */

static int find_slot_locked(const char *name) {
    for (int i = 0; i < g_count; i++) {
        const int s = g_order[i];
        if (strcmp(g_slots[s].desc.name, name) == 0) return s;
    }
    return -1;
}

/* Lowest never-used slot first; freed slots only afterwards (see file
 * header). */
static int alloc_slot_locked(void) {
    if (g_slots_used < RVCL_REGISTRY_MAX_KERNELS) return g_slots_used++;
    for (int s = 0; s < RVCL_REGISTRY_MAX_KERNELS; s++)
        if (!g_slots[s].used) return s;
    return -1;
}

static int desc_is_unbounded(const rvcl_kernel_desc_t *d) {
    return d->max_m == 0 && d->max_n == 0 && d->max_k == 0;
}

/* Does the datatype tuple `d` belongs to contain at least one unbounded
 * kernel? Every live tuple must (FR1.7). */
static int tuple_has_unbounded_locked(const rvcl_kernel_desc_t *d) {
    for (int i = 0; i < g_count; i++) {
        const rvcl_kernel_desc_t *o = &g_slots[g_order[i]].desc;
        if (o->operation != d->operation || o->dtype_a != d->dtype_a
                || o->dtype_b != d->dtype_b || o->dtype_acc != d->dtype_acc
                || o->dtype_c != d->dtype_c)
            continue;
        if (desc_is_unbounded(o)) return 1;
    }
    return 0;
}

/* Would removing `skip` leave its datatype tuple without an unbounded
 * fallback? Only meaningful when other entries of the tuple remain. */
static int tuple_keeps_fallback_locked(int skip) {
    const rvcl_kernel_desc_t *rm = &g_slots[skip].desc;
    int others = 0;
    for (int i = 0; i < g_count; i++) {
        const int s = g_order[i];
        if (s == skip) continue;
        const rvcl_kernel_desc_t *d = &g_slots[s].desc;
        if (d->operation != rm->operation || d->dtype_a != rm->dtype_a
                || d->dtype_b != rm->dtype_b || d->dtype_acc != rm->dtype_acc
                || d->dtype_c != rm->dtype_c)
            continue;
        others = 1;
        if (desc_is_unbounded(d)) return 1;
    }
    /* Empty tuples impose no constraint. */
    return !others;
}

/* Registry changes invalidate the dispatch cache via
 * rvcl_dispatch_cache_clear() (declared in dispatch.h). It is called
 * outside g_mutex (lock order registry -> cache, never nested). */

/* ------------------------------------------------------------- public API */

rvcl_status_t rvcl_register_kernel(
        const rvcl_kernel_desc_t *desc, rvcl_kernel_create_fn create) {
    rvcl_status_t st = validate_desc(desc, create);
    if (st != RVCL_SUCCESS) return st;

    pthread_mutex_lock(&g_mutex);
    if (find_slot_locked(desc->name) >= 0) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_ALREADY_REGISTERED;
    }
    if (g_count >= RVCL_REGISTRY_MAX_KERNELS) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_OUT_OF_MEMORY;
    }
    const int s = alloc_slot_locked();
    g_slots[s].desc = *desc;
    g_slots[s].create = create;
    g_slots[s].used = 1;
    g_order[g_count++] = s;
    pthread_mutex_unlock(&g_mutex);

    rvcl_dispatch_cache_clear();
    return RVCL_SUCCESS;
}

rvcl_status_t rvcl_unregister_kernel(const char *name) {
    if (!name) return RVCL_ERROR_INVALID_ARGUMENT;
    pthread_mutex_lock(&g_mutex);
    const int s = find_slot_locked(name);
    if (s < 0) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_NOT_FOUND;
    }
    if (!tuple_keeps_fallback_locked(s)) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_REGISTRY_INVARIANT;
    }
    /* Remove from the order index; descriptor storage stays in place so
     * pointers to the remaining entries stay coherent. Pointers to the
     * removed kernel become invalid (see registry.h). */
    int pos = 0;
    while (pos < g_count && g_order[pos] != s)
        pos++;
    if (pos == g_count) { /* found by name but missing from the order index */
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_INTERNAL;
    }
    memmove(&g_order[pos], &g_order[pos + 1],
            (size_t)(g_count - pos - 1) * sizeof(int));
    g_count--;
    g_slots[s].used = 0;
    pthread_mutex_unlock(&g_mutex);

    rvcl_dispatch_cache_clear();
    return RVCL_SUCCESS;
}

int rvcl_registry_count(void) {
    pthread_mutex_lock(&g_mutex);
    const int n = g_count;
    pthread_mutex_unlock(&g_mutex);
    return n;
}

const rvcl_kernel_desc_t *rvcl_registry_get(int index) {
    const rvcl_kernel_desc_t *out = NULL;
    pthread_mutex_lock(&g_mutex);
    if (index >= 0 && index < g_count) out = &g_slots[g_order[index]].desc;
    pthread_mutex_unlock(&g_mutex);
    return out;
}

rvcl_kernel_create_fn rvcl_registry_find_factory(const char *name) {
    if (!name) return NULL;
    rvcl_kernel_create_fn fn = NULL;
    pthread_mutex_lock(&g_mutex);
    const int s = find_slot_locked(name);
    if (s >= 0) fn = g_slots[s].create;
    pthread_mutex_unlock(&g_mutex);
    return fn;
}

/* ------------------------------------------------------------ invariant */

/* Whole-table fallback check, run once initialization finished (internal
 * symbol, not public API; declared in registry_internal.h). */
rvcl_status_t rvcl_registry_verify_fallback_invariant(void) {
    pthread_mutex_lock(&g_mutex);
    for (int i = 0; i < g_count; i++) {
        const rvcl_kernel_desc_t *d = &g_slots[g_order[i]].desc;
        if (!tuple_has_unbounded_locked(d)) {
            pthread_mutex_unlock(&g_mutex);
            return RVCL_ERROR_REGISTRY_INVARIANT;
        }
    }
    pthread_mutex_unlock(&g_mutex);
    return RVCL_SUCCESS;
}

/* Init-path rollback: drop the tail of the registration order, leaving
 * the first `count` entries untouched. */
rvcl_status_t rvcl_registry_rollback(int count) {
    pthread_mutex_lock(&g_mutex);
    if (count < 0 || count > g_count) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_INTERNAL;
    }
    while (g_count > count) {
        const int s = g_order[g_count - 1];
        g_slots[s].used = 0;
        g_count--;
    }
    pthread_mutex_unlock(&g_mutex);

    rvcl_dispatch_cache_clear();
    return RVCL_SUCCESS;
}
