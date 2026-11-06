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
 * dispatch.cpp — generic kernel selection pipeline
 *
 * Six-stage filtering, then cost-based selection:
 *   1. operation     — descriptor operation equals the request operation
 *   2. datatype      — exact (A,B,acc,C) match after AUTO resolution,
 *                      plus quantization support for quantized requests
 *   3. capability    — every required feature is present in the snapshot
 *   4. vlen tier     — kernel tier is 0 (any) or not above the host tier
 *   5. shape         — m/n/k inside the kernel's inclusive bounds
 *   6. validate      — the instantiated kernel accepts the request
 * Surviving candidates are ordered by their cost callback; the smallest
 * cost wins and equal costs keep the earliest registration. Kernels are
 * instantiated only when they can still win, so rejections at stage 6
 * simply move on to the next candidate.
 *
 * The pipeline is operator-generic: each operator contributes one op-table
 * entry (a request validator that resolves the datatype/shape filter inputs
 * and a cache-key generator), and every registered operator automatically
 * gets the whole pipeline and dispatch cache. rvcl_dispatch_op() is the
 * internal generic entry; the public per-operator wrappers look their entry
 * up and call it.
 *
 * Dispatch only selects and configures; it never runs computations and
 * holds no ISA-specific code. Selection works on a registry snapshot so
 * a concurrent registration or unregistration is never observed
 * half-applied, and cached selections are tagged with the registry
 * generation observed at selection time.
 ******************************************************************************/

#include "rvcl/dispatch.h"

#include "../common/dtype_traits.hpp"
#include "capability_internal.h"
#include "dispatch_internal.h"
#include "registry_internal.h"
#include "rvcl/capability.h"
#include "rvcl/init.h"
#include "rvcl/registry.h"

#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

struct rvcl_dispatch_result {
    rvcl_kernel_t *kernel;
    rvcl_workspace_req_t ws_req[RVCL_MAX_WORKSPACES];
    int num_ws;
    /* Refcount: the cache holds one reference; every dispatch call (hit
     * or fresh) hands out another. rvcl_dispatch_release() drops a
     * reference and frees only at zero — the semantic API may release
     * its handle while the cache keeps the instance alive. */
    int refs;
};

/* ------------------------------------------------------ dispatch cache
 * Small direct-mapped cache of configured kernels keyed by the full
 * request (operator discriminant + full-width operation descriptor). Hit →
 * reuse the configured instance (no filtering, validate or configure);
 * miss → full pipeline. Entries are tagged with the registry generation
 * they were selected under, so a registry change racing a selection can
 * never leave a stale selection cached. The op discriminant participates
 * in every comparison, so two operators whose descriptors collide in a
 * cache slot never share an entry.
 */

#define RVCL_DISPATCH_CACHE_SLOTS 8
/* A selection restarts when the registry changes before its result is
 * cached. The bound only guards against pathological registration churn;
 * ordinary contention is resolved within one or two attempts. */
#define RVCL_DISPATCH_MAX_ATTEMPTS 64

typedef struct {
    int used;
    unsigned generation;
    rvcl_operation_t op;
    rvcl_op_desc_t key;
    rvcl_dispatch_result_t *result;
} cache_slot_t;

static cache_slot_t g_cache[RVCL_DISPATCH_CACHE_SLOTS];
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_cache_hits, g_cache_misses; /* observability */

/* forward: cache eviction / clear drop their reference through this */
static void result_unref(rvcl_dispatch_result_t *r);

/* --------------------------------------------------------------- op table
 * One entry per operator: a request validator resolving the datatype/shape
 * filter inputs and a cache-key generator. The table is populated at
 * startup with the matmul entry; internal callers and tests may append
 * entries via rvcl_dispatch_register_op(). */
#define RVCL_DISPATCH_MAX_OPS 8

/* forward: the matmul entry's validator and key generator (defined in the
 * keys / request sections below) */
static rvcl_status_t matmul_dims_validate(
        const rvcl_op_desc_t *od, rvcl_dispatch_filter_t *out);
static uint32_t matmul_key_hash(const rvcl_op_desc_t *key);
static int matmul_key_equal(const rvcl_op_desc_t *a, const rvcl_op_desc_t *b);

/* forward: the gemv entry's validator and key generator (request / keys
 * sections below) */
static rvcl_status_t gemv_dims_validate(
        const rvcl_op_desc_t *od, rvcl_dispatch_filter_t *out);
static uint32_t gemv_key_hash(const rvcl_op_desc_t *key);
static int gemv_key_equal(const rvcl_op_desc_t *a, const rvcl_op_desc_t *b);

static rvcl_dispatch_op_entry_t g_op_table[RVCL_DISPATCH_MAX_OPS] = {
        {RVCL_OP_MATMUL, matmul_dims_validate, matmul_key_hash,
                matmul_key_equal},
        {RVCL_OP_GEMV, gemv_dims_validate, gemv_key_hash, gemv_key_equal},
};
static int g_op_count = 2;

static const rvcl_dispatch_op_entry_t *op_entry(rvcl_operation_t op) {
    for (int i = 0; i < g_op_count; i++)
        if (g_op_table[i].op == op) return &g_op_table[i];
    return NULL;
}

rvcl_status_t rvcl_dispatch_register_op(const rvcl_dispatch_op_entry_t *entry) {
    if (!entry || !entry->dims_validate || !entry->key_hash
            || !entry->key_equal)
        return RVCL_ERROR_INVALID_ARGUMENT;
    if (op_entry(entry->op)) return RVCL_ERROR_ALREADY_REGISTERED;
    if (g_op_count >= RVCL_DISPATCH_MAX_OPS) return RVCL_ERROR_OUT_OF_MEMORY;
    g_op_table[g_op_count++] = *entry;
    return RVCL_SUCCESS;
}

/* --------------------------------------------------------------- keys */
/* The cache key is the raw request: every field participates, and an
 * AUTO datatype is not normalized to its resolved explicit value. Float
 * fields compare by bit pattern so equality and hashing agree. Each
 * operator's key generator covers its descriptor's full field set. */

static uint32_t float_bits(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    return u;
}

static int matmul_key_equal(const rvcl_op_desc_t *a, const rvcl_op_desc_t *b) {
    return a->matmul.m == b->matmul.m && a->matmul.n == b->matmul.n
            && a->matmul.k == b->matmul.k
            && a->matmul.dtype_a == b->matmul.dtype_a
            && a->matmul.dtype_b == b->matmul.dtype_b
            && a->matmul.dtype_acc == b->matmul.dtype_acc
            && a->matmul.dtype_c == b->matmul.dtype_c
            && a->matmul.dtype_bias == b->matmul.dtype_bias
            && a->matmul.quant.enabled == b->matmul.quant.enabled
            && float_bits(a->matmul.quant.scale_a)
            == float_bits(b->matmul.quant.scale_a)
            && float_bits(a->matmul.quant.scale_b)
            == float_bits(b->matmul.quant.scale_b)
            && float_bits(a->matmul.quant.scale_c)
            == float_bits(b->matmul.quant.scale_c)
            && a->matmul.quant.zp_a == b->matmul.quant.zp_a
            && a->matmul.quant.zp_b == b->matmul.quant.zp_b
            && a->matmul.quant.zp_c == b->matmul.quant.zp_c;
}

static unsigned hash_mix(unsigned h, uint64_t v) {
    h = h * 31u + (unsigned)(v & 0xffffffffu);
    h = h * 31u + (unsigned)(v >> 32);
    return h;
}

static uint32_t matmul_key_hash(const rvcl_op_desc_t *key) {
    unsigned h = 2166136261u;
    h = hash_mix(h, (uint64_t)key->matmul.m);
    h = hash_mix(h, (uint64_t)key->matmul.n);
    h = hash_mix(h, (uint64_t)key->matmul.k);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.dtype_a);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.dtype_b);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.dtype_acc);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.dtype_c);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.dtype_bias);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.quant.enabled);
    h = hash_mix(h, float_bits(key->matmul.quant.scale_a));
    h = hash_mix(h, float_bits(key->matmul.quant.scale_b));
    h = hash_mix(h, float_bits(key->matmul.quant.scale_c));
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.quant.zp_a);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.quant.zp_b);
    h = hash_mix(h, (uint64_t)(unsigned)key->matmul.quant.zp_c);
    return h;
}

static int gemv_key_equal(const rvcl_op_desc_t *a, const rvcl_op_desc_t *b) {
    return a->gemv.m == b->gemv.m && a->gemv.k == b->gemv.k
            && a->gemv.dtype_a == b->gemv.dtype_a
            && a->gemv.dtype_b == b->gemv.dtype_b
            && a->gemv.dtype_acc == b->gemv.dtype_acc
            && a->gemv.dtype_c == b->gemv.dtype_c
            && a->gemv.dtype_bias == b->gemv.dtype_bias
            && a->gemv.quant.enabled == b->gemv.quant.enabled
            && float_bits(a->gemv.quant.scale_a)
            == float_bits(b->gemv.quant.scale_a)
            && float_bits(a->gemv.quant.scale_b)
            == float_bits(b->gemv.quant.scale_b)
            && float_bits(a->gemv.quant.scale_c)
            == float_bits(b->gemv.quant.scale_c)
            && a->gemv.quant.zp_a == b->gemv.quant.zp_a
            && a->gemv.quant.zp_b == b->gemv.quant.zp_b
            && a->gemv.quant.zp_c == b->gemv.quant.zp_c;
}

static uint32_t gemv_key_hash(const rvcl_op_desc_t *key) {
    unsigned h = 2166136261u;
    h = hash_mix(h, (uint64_t)key->gemv.m);
    h = hash_mix(h, (uint64_t)key->gemv.k);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.dtype_a);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.dtype_b);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.dtype_acc);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.dtype_c);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.dtype_bias);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.quant.enabled);
    h = hash_mix(h, float_bits(key->gemv.quant.scale_a));
    h = hash_mix(h, float_bits(key->gemv.quant.scale_b));
    h = hash_mix(h, float_bits(key->gemv.quant.scale_c));
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.quant.zp_a);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.quant.zp_b);
    h = hash_mix(h, (uint64_t)(unsigned)key->gemv.quant.zp_c);
    return h;
}

/* lookup: returns the result (borrowed, cache-owned) or NULL. An entry
 * selected under an older registry generation is dropped on sight. */
static rvcl_dispatch_result_t *cache_lookup(const rvcl_dispatch_op_entry_t *e,
        const rvcl_op_desc_t *key, unsigned generation) {
    unsigned idx = e->key_hash(key) % RVCL_DISPATCH_CACHE_SLOTS;
    cache_slot_t *s = &g_cache[idx];
    if (!s->used || s->op != e->op || !e->key_equal(&s->key, key)) return NULL;
    if (s->generation != generation) {
        result_unref(s->result);
        s->used = 0;
        s->result = NULL;
        return NULL;
    }
    return s->result;
}

/* insert: returns 1 when r now owns the hashed cache slot, 0 when an
 * equal-key entry of the same generation already holds it (a concurrent
 * first call won) — then *winner is that entry. Replaces any previous
 * entry with a different key or an older generation, dropping its cache
 * reference. */
static int cache_insert(const rvcl_dispatch_op_entry_t *e,
        const rvcl_op_desc_t *key, rvcl_dispatch_result_t *r,
        unsigned generation, rvcl_dispatch_result_t **winner) {
    unsigned idx = e->key_hash(key) % RVCL_DISPATCH_CACHE_SLOTS;
    cache_slot_t *s = &g_cache[idx];
    if (s->used && s->generation == generation && s->op == e->op
            && e->key_equal(&s->key, key)) {
        *winner = s->result; /* same key: keep the older instance */
        return 0;
    }
    if (s->used) result_unref(s->result); /* evict the cache's reference */
    s->used = 1;
    s->op = e->op;
    s->generation = generation;
    s->key = *key;
    s->result = r;
    *winner = r;
    return 1;
}

/* Drops the cache's reference; the instance survives while callers hold
 * their own references. */
static void result_unref(rvcl_dispatch_result_t *r) {
    if (!r) return;
    if (--r->refs > 0) return;
    rvcl_kernel_destroy(r->kernel);
    free(r);
}

rvcl_status_t rvcl_dispatch_cache_clear(void) {
    pthread_mutex_lock(&g_cache_mutex);
    for (int i = 0; i < RVCL_DISPATCH_CACHE_SLOTS; i++) {
        if (g_cache[i].used) {
            result_unref(g_cache[i].result);
            g_cache[i].used = 0;
            g_cache[i].result = NULL;
        }
    }
    pthread_mutex_unlock(&g_cache_mutex);
    return RVCL_SUCCESS;
}

int rvcl_dispatch_cache_hits(void) {
    return g_cache_hits;
}
int rvcl_dispatch_cache_misses(void) {
    return g_cache_misses;
}

/* ------------------------------------------------------------- request */
/* Request-side validation and AUTO resolution are operator-specific and
 * live in each op-table entry's dims_validate. The matmul entry below
 * resolves its filter inputs from the matmul descriptor. */

/* Datatypes this iteration can serve in a request. Reserved enum values
 * are rejected as illegal tuple members rather than reported as "no
 * kernel". */
static int dtype_is_request_valid(rvcl_dtype_t dt) {
    switch (dt) {
        case RVCL_DTYPE_FP32:
        case RVCL_DTYPE_FP16:
        case RVCL_DTYPE_BF16:
        case RVCL_DTYPE_INT8:
        case RVCL_DTYPE_UINT8:
        case RVCL_DTYPE_INT4:
        case RVCL_DTYPE_INT32: return 1;
        default: return 0;
    }
}

/* The accumulation type an AUTO request resolves to. AUTO is only legal
 * in the accumulator position. The habit is table-backed (dtype trait
 * table acc_default): int8/uint8 -> INT32, floating family -> FP32, so
 * adding a dtype never forks this resolver again. */
static rvcl_dtype_t resolve_acc(rvcl_dtype_t dtype_a, rvcl_dtype_t dtype_acc) {
    if (dtype_acc != RVCL_DTYPE_AUTO) return dtype_acc;
    return rvcl_dtype_traits(dtype_a).acc_default;
}

/* Validates the (A,B,acc,C,bias) dtype tuple shared by the operators. */
static rvcl_status_t validate_dtype_tuple(rvcl_dtype_t dtype_a,
        rvcl_dtype_t dtype_b, rvcl_dtype_t dtype_acc, rvcl_dtype_t dtype_c,
        rvcl_dtype_t dtype_bias) {
    /* AUTO is a request-side marker valid only for the accumulator and
     * for the "no bias" bias slot. */
    if (dtype_a == RVCL_DTYPE_AUTO || dtype_b == RVCL_DTYPE_AUTO
            || dtype_c == RVCL_DTYPE_AUTO)
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    if (!dtype_is_request_valid(dtype_a) || !dtype_is_request_valid(dtype_b)
            || !dtype_is_request_valid(dtype_c))
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    if (dtype_acc != RVCL_DTYPE_AUTO && !dtype_is_request_valid(dtype_acc))
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    if (dtype_bias != RVCL_DTYPE_AUTO && !dtype_is_request_valid(dtype_bias))
        return RVCL_UNSUPPORTED_DTYPE_COMBO;
    return RVCL_SUCCESS;
}

static rvcl_status_t validate_request(const rvcl_matmul_desc_t *d) {
    return validate_dtype_tuple(
            d->dtype_a, d->dtype_b, d->dtype_acc, d->dtype_c, d->dtype_bias);
}

/* The matmul op-table entry: validates the request and resolves the
 * generic filter inputs. */
static rvcl_status_t matmul_dims_validate(
        const rvcl_op_desc_t *od, rvcl_dispatch_filter_t *out) {
    const rvcl_matmul_desc_t *d = &od->matmul;
    if (d->m <= 0 || d->n <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    const rvcl_status_t st = validate_request(d);
    if (st != RVCL_SUCCESS) return st;
    out->m = d->m;
    out->n = d->n;
    out->k = d->k;
    out->dtype_a = d->dtype_a;
    out->dtype_b = d->dtype_b;
    out->dtype_acc = resolve_acc(d->dtype_a, d->dtype_acc);
    out->dtype_c = d->dtype_c;
    out->quant_enabled = d->quant.enabled;
    return RVCL_SUCCESS;
}

/* The gemv op-table entry. The descriptor has no n dimension; the filter
 * keeps n at 1 and the pipeline shape stage treats the kernel's unbounded
 * n bounds as "any", so a gemv kernel is never rejected on n. */
static rvcl_status_t gemv_dims_validate(
        const rvcl_op_desc_t *od, rvcl_dispatch_filter_t *out) {
    const rvcl_gemv_desc_t *d = &od->gemv;
    if (d->m <= 0 || d->k <= 0) return RVCL_ERROR_INVALID_ARGUMENT;
    const rvcl_status_t st = validate_dtype_tuple(
            d->dtype_a, d->dtype_b, d->dtype_acc, d->dtype_c, d->dtype_bias);
    if (st != RVCL_SUCCESS) return st;
    out->m = d->m;
    out->n = 1; /* no n dimension in a gemv request */
    out->k = d->k;
    out->dtype_a = d->dtype_a;
    out->dtype_b = d->dtype_b;
    out->dtype_acc = resolve_acc(d->dtype_a, d->dtype_acc);
    out->dtype_c = d->dtype_c;
    out->quant_enabled = d->quant.enabled;
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------------- filters */
/* Stages 2 and 5 consume the resolved filter inputs, so the pipeline body
 * stays operator-generic. */

static int filter_datatype(
        const rvcl_kernel_desc_t *kd, const rvcl_dispatch_filter_t *f) {
    if (kd->dtype_a != f->dtype_a || kd->dtype_b != f->dtype_b
            || kd->dtype_acc != f->dtype_acc || kd->dtype_c != f->dtype_c)
        return 0;
    /* A kernel that declares no quantization support would compute a
     * quantized request as if it were not quantized: exclude it before
     * instantiation rather than relying on validate(). */
    if (f->quant_enabled && kd->quant_support == RVCL_QUANT_NONE) return 0;
    return 1;
}

static int filter_capability(
        const rvcl_kernel_desc_t *kd, const rvcl_capability_t *cap) {
    for (int i = 0; i < kd->num_required_features; i++)
        if (!rvcl_capability_has(cap, kd->required_features[i])) return 0;
    return 1;
}

static int in_bounds(int64_t v, int64_t lo, int64_t hi) {
    if (lo && v < lo) return 0;
    if (hi && v > hi) return 0;
    return 1;
}

static int filter_shape(
        const rvcl_kernel_desc_t *kd, const rvcl_dispatch_filter_t *f) {
    return in_bounds(f->m, kd->min_m, kd->max_m)
            && in_bounds(f->n, kd->min_n, kd->max_n)
            && in_bounds(f->k, kd->min_k, kd->max_k);
}

/* ------------------------------------------------------------- dispatch */

rvcl_status_t rvcl_dispatch_op(rvcl_operation_t op, const rvcl_op_desc_t *od,
        rvcl_dispatch_result_t **result) {
    if (!od || !result) return RVCL_ERROR_INVALID_ARGUMENT;
    *result = NULL;

    /* An operator with no dispatch table entry cannot be served: reject
     * it here rather than reporting an empty selection. */
    const rvcl_dispatch_op_entry_t *e = op_entry(op);
    if (!e) return RVCL_UNSUPPORTED;

    /* Lazy init: a first computation without an explicit rvcl_init()
     * still dispatches over the full built-in registry. The init fast
     * path is an atomic load, so this stays cheap afterwards. A failed
     * init must not be papered over: selection over a partial registry
     * would silently miss installed backends. */
    const rvcl_status_t init_st = rvcl_init();
    if (init_st != RVCL_SUCCESS) return init_st;

    /* Validate the request and resolve the filter inputs the pipeline
     * stages consume (shape, datatype tuple with AUTO resolved, quant). */
    rvcl_dispatch_filter_t f;
    rvcl_status_t st = e->dims_validate(od, &f);
    if (st != RVCL_SUCCESS) return st;

    /* Cache hit: reuse the configured instance and hand out a fresh
     * reference on it. The generation is read under the cache lock so a
     * change that already completed its bump cannot be masked by a stale
     * tag. */
    unsigned generation = 0;
    pthread_mutex_lock(&g_cache_mutex);
    generation = rvcl_registry_generation();
    rvcl_dispatch_result_t *cached = cache_lookup(e, od, generation);
    if (cached) {
        g_cache_hits++;
        cached->refs++;
        pthread_mutex_unlock(&g_cache_mutex);
        *result = cached;
        return RVCL_SUCCESS;
    }
    g_cache_misses++;
    pthread_mutex_unlock(&g_cache_mutex);

    const rvcl_capability_t *cap = rvcl_get_capability();
    const int host_tier = rvcl_capability_vlen_tier(cap);

    /* Snapshot storage: one buffer per dispatch call, reused by every
     * retry instead of being reallocated on the stack each attempt. */
    rvcl_registry_view_t views[RVCL_REGISTRY_MAX_KERNELS];
    for (int attempt = 0; attempt < RVCL_DISPATCH_MAX_ATTEMPTS; attempt++) {
        /* A registry snapshot keeps stages 1-5 on one coherent table.
         * The generation observed with the copy detects a change that
         * lands while the candidates are being evaluated. */
        const int n = rvcl_registry_snapshot(
                views, RVCL_REGISTRY_MAX_KERNELS, &generation);

        rvcl_kernel_t *best = NULL;
        uint64_t best_cost = UINT64_MAX;
        for (int i = 0; i < n; i++) {
            const rvcl_kernel_desc_t *kd = &views[i].desc;

            if (kd->operation != op) continue; /* stage 1 */
            if (!filter_datatype(kd, &f)) continue; /* stage 2 */
            if (!filter_capability(kd, cap)) continue; /* stage 3 */
            if (kd->vlen_tier != RVCL_VLEN_TIER_ANY
                    && (int)kd->vlen_tier > host_tier)
                continue; /* stage 4 */
            if (!filter_shape(kd, &f)) continue; /* stage 5 */

            const uint64_t cost = kd->cost_fn(od, cap);
            /* Equal cost keeps the earlier candidate: no later entry can
             * win over the current best, so skip its instantiation. */
            if (best && cost >= best_cost) continue;

            /* Stage 6: an instantiated kernel gets the final word. A
             * factory or validate failure only eliminates this entry. */
            rvcl_kernel_t *k = NULL;
            if (views[i].create(&k, kd) != RVCL_SUCCESS) {
                if (k) rvcl_kernel_destroy(k);
                continue;
            }
            if (!k) continue;
            if (k->vt && k->vt->validate
                    && k->vt->validate(od, cap) != RVCL_SUCCESS) {
                rvcl_kernel_destroy(k);
                continue;
            }
            if (best) rvcl_kernel_destroy(best);
            best = k;
            best_cost = cost;
        }

        if (!best) return RVCL_UNSUPPORTED;

        rvcl_dispatch_result_t *r
                = (rvcl_dispatch_result_t *)calloc(1, sizeof(*r));
        if (!r) {
            rvcl_kernel_destroy(best);
            return RVCL_ERROR_OUT_OF_MEMORY;
        }
        r->kernel = best;
        r->num_ws = 0;
        r->refs = 0; /* references are taken below, under the cache lock */

        st = rvcl_kernel_configure(best, od, r->ws_req, &r->num_ws);
        if (st != RVCL_SUCCESS) {
            rvcl_kernel_destroy(best);
            free(r);
            return st;
        }
        /* A pack carries workspaces only through the AUX slots, so a
         * kernel reporting more than RVCL_TENSOR_AUX_COUNT would have the
         * extra requests silently dropped by the host mapping loops; fail
         * loudly at dispatch time instead. */
        if (r->num_ws > RVCL_TENSOR_AUX_COUNT) {
            rvcl_kernel_destroy(best);
            free(r);
            return RVCL_UNSUPPORTED;
        }

        /* A registry change during selection invalidates what was chosen:
         * restart so the result reflects the current table and no stale
         * entry can enter the cache. */
        pthread_mutex_lock(&g_cache_mutex);
        if (rvcl_registry_generation() != generation) {
            pthread_mutex_unlock(&g_cache_mutex);
            rvcl_kernel_destroy(best);
            free(r);
            continue;
        }

        rvcl_dispatch_result_t *winner = NULL;
        const int inserted = cache_insert(e, od, r, generation, &winner);
        if (inserted) {
            /* Cache reference + caller reference. */
            r->refs = 2;
        } else {
            /* A concurrent first call cached the same key: drop this
             * result's temporary ownership and hand out the winner. */
            winner->refs++;
            result_unref(r);
        }
        pthread_mutex_unlock(&g_cache_mutex);
        *result = winner;
        return RVCL_SUCCESS;
    }

    return RVCL_UNSUPPORTED;
}

/* Public matmul entry: a thin wrapper that looks the op-table entry up
 * via rvcl_dispatch_op(). Public ABI and semantics unchanged. */
rvcl_status_t rvcl_dispatch_matmul(
        const rvcl_matmul_desc_t *desc, rvcl_dispatch_result_t **result) {
    if (!desc || !result) return RVCL_ERROR_INVALID_ARGUMENT;
    const rvcl_op_desc_t od = {.matmul = *desc};
    return rvcl_dispatch_op(RVCL_OP_MATMUL, &od, result);
}

const char *rvcl_dispatch_kernel_name(const rvcl_dispatch_result_t *r) {
    return r && r->kernel ? r->kernel->desc.name : NULL;
}

const rvcl_workspace_req_t *rvcl_dispatch_ws_req(
        const rvcl_dispatch_result_t *r, int *num_ws) {
    if (!r) return NULL;
    if (num_ws) *num_ws = r->num_ws;
    return r->ws_req;
}

rvcl_kernel_t *rvcl_dispatch_kernel(rvcl_dispatch_result_t *r) {
    return r ? r->kernel : NULL;
}

void rvcl_dispatch_release(rvcl_dispatch_result_t *r) {
    if (!r) return;
    pthread_mutex_lock(&g_cache_mutex);
    result_unref(r);
    pthread_mutex_unlock(&g_cache_mutex);
}
