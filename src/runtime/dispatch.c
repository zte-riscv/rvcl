/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * dispatch.c — kernel selection pipeline
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Phase 0 scope (PLAN §6.2 step 6): stages 1-4 of the 7-stage pipeline
 * with exact (A,B,acc,C) matching and static backend priority as the
 * placeholder "cost model". Layout/packing stages and the real cost model
 * land in v0.2-v0.4. Dispatch never executes kernels and holds no
 * ISA-specific code.
 *
 * Stage semantics:
 *   1. operation — desc->operation == kernel->operation
 *   2. datatype  — exact (A,B,acc,C) tuple match; AUTO handling below
 *   3. capability — every required feature present in host capability
 *   4. shape — m/n/k within [min,max] bounds
 * Survivors are ranked by static priority (reference lowest), then the
 * winner is instantiated + configured; validate() runs before configure.
 ******************************************************************************/

#include "rvcl/dispatch.h"

#include "rvcl/capability.h"
#include "rvcl/registry.h"

#include <pthread.h>
#include <stdlib.h>
#include <string.h>

struct rvcl_dispatch_result {
    rvcl_kernel_t *kernel;
    rvcl_workspace_req_t ws_req[RVCL_MAX_WORKSPACES];
    int num_ws;
    /* Refcount: the cache holds one reference; every rvcl_dispatch_matmul()
     * call (hit or fresh) hands out another. rvcl_dispatch_release() drops
     * a reference and frees only at zero — the semantic API may release
     * its handle while the cache keeps the instance alive. */
    int refs;
};

/* ------------------------------------------------------ dispatch cache
 * PLAN §6.2 item 9 (Step 7): a small direct-mapped cache of configured
 * kernels keyed by the matmul request. Hit → reuse the configured
 * instance (validate+configure skipped); miss → full pipeline. Entries
 * hold their kernel instance; cache_reset releases them. Thread-safe
 * via the registry mutex pattern (own mutex here).
 */

#define RVCL_DISPATCH_CACHE_SLOTS 8

typedef struct {
    int used;
    rvcl_matmul_desc_t key;
    rvcl_dispatch_result_t *result;
} cache_slot_t;

static cache_slot_t g_cache[RVCL_DISPATCH_CACHE_SLOTS];
static pthread_mutex_t g_cache_mutex = PTHREAD_MUTEX_INITIALIZER;
static int g_cache_hits, g_cache_misses; /* observability (PLAN §8.4) */

/* forward: cache eviction / clear drop their reference through this */
static void result_unref(rvcl_dispatch_result_t *r);

static int desc_key_equal(const rvcl_matmul_desc_t *a,
        const rvcl_matmul_desc_t *b) {
    return a->m == b->m && a->n == b->n && a->k == b->k
            && a->dtype_a == b->dtype_a && a->dtype_b == b->dtype_b
            && a->dtype_acc == b->dtype_acc && a->dtype_c == b->dtype_c
            && a->dtype_bias == b->dtype_bias
            && a->quant.enabled == b->quant.enabled;
}

static unsigned desc_key_hash(const rvcl_matmul_desc_t *d) {
    unsigned h = (unsigned)(d->m * 31 + d->n * 7 + d->k);
    h = h * 31 + (unsigned)(d->dtype_a << 4 | d->dtype_b);
    h = h * 31 + (unsigned)(d->dtype_acc << 4 | d->dtype_c);
    return h;
}

/* lookup: returns the result (borrowed, cache-owned) or NULL. */
static rvcl_dispatch_result_t *cache_lookup(
        const rvcl_matmul_desc_t *key) {
    unsigned idx = desc_key_hash(key) % RVCL_DISPATCH_CACHE_SLOTS;
    cache_slot_t *s = &g_cache[idx];
    if (!s->used || !desc_key_equal(&s->key, key)) return NULL;
    return s->result;
}

/* insert: replaces any previous entry in the hashed slot. */
static void cache_insert(const rvcl_matmul_desc_t *key,
        rvcl_dispatch_result_t *r) {
    unsigned idx = desc_key_hash(key) % RVCL_DISPATCH_CACHE_SLOTS;
    cache_slot_t *s = &g_cache[idx];
    if (s->used && !desc_key_equal(&s->key, key)) {
        result_unref(s->result); /* evict: drop the cache's reference */
    } else if (s->used) {
        return; /* same key already cached: keep the older instance */
    }
    s->used = 1;
    s->key = *key;
    s->result = r; /* cache reference taken over from the caller */
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

int rvcl_dispatch_cache_hits(void) { return g_cache_hits; }
int rvcl_dispatch_cache_misses(void) { return g_cache_misses; }

/* ------------------------------------------------------------- filters */

static int filter_datatype(const rvcl_kernel_desc_t *kd,
        const rvcl_matmul_desc_t *d) {
    /* acc=AUTO resolves via habit rules (PLAN §4.5.1). */
    rvcl_dtype_t acc = d->dtype_acc;
    if (acc == RVCL_DTYPE_AUTO) {
        if (d->dtype_a == RVCL_DTYPE_FP32) acc = RVCL_DTYPE_FP32;
        else if (d->dtype_a == RVCL_DTYPE_BF16) acc = RVCL_DTYPE_FP32;
        else if (d->dtype_a == RVCL_DTYPE_INT8 || d->dtype_a == RVCL_DTYPE_UINT8)
            acc = RVCL_DTYPE_INT32;
        else
            acc = RVCL_DTYPE_FP32; /* conservative default */
    }
    return kd->dtype_a == d->dtype_a && kd->dtype_b == d->dtype_b
            && kd->dtype_acc == acc && kd->dtype_c == d->dtype_c;
}

static int filter_capability(const rvcl_kernel_desc_t *kd,
        const rvcl_capability_t *cap) {
    for (int i = 0; i < kd->num_required_features; i++)
        if (!rvcl_capability_has(cap, kd->required_features[i])) return 0;
    return 1;
}

static int in_bounds(int64_t v, int64_t lo, int64_t hi) {
    if (lo && v < lo) return 0;
    if (hi && v > hi) return 0;
    return 1;
}

static int filter_shape(const rvcl_kernel_desc_t *kd,
        const rvcl_matmul_desc_t *d) {
    return in_bounds(d->m, kd->min_m, kd->max_m)
            && in_bounds(d->n, kd->min_n, kd->max_n)
            && in_bounds(d->k, kd->min_k, kd->max_k);
}

/* Static placeholder priority (PLAN §4.2): lower = better.
 * Replaced by the cost model in v0.4. */
static int backend_priority(const rvcl_kernel_desc_t *kd) {
    int has_rvv = 0, has_matrix = 0;
    for (int i = 0; i < kd->num_required_features; i++) {
        switch (kd->required_features[i]) {
            case RVCL_FEATURE_RVV: has_rvv = 1; break;
            case RVCL_FEATURE_IME:
            case RVCL_FEATURE_VME:
            case RVCL_FEATURE_AME: has_matrix = 1; break;
            default: break;
        }
    }
    if (strstr(kd->name, "reference")) return 100; /* last resort */
    if (has_matrix) return 0; /* placeholder: matrix > vector */
    if (has_rvv) return 10;
    return 50; /* generic */
}

/* ------------------------------------------------------------- dispatch */

rvcl_status_t rvcl_dispatch_matmul(const rvcl_matmul_desc_t *desc,
        rvcl_dispatch_result_t **result) {
    if (!desc || !result) return RVCL_ERROR_INVALID_ARGUMENT;
    *result = NULL;

    if (desc->m <= 0 || desc->n <= 0 || desc->k <= 0)
        return RVCL_ERROR_INVALID_ARGUMENT;

    /* Cache hit: reuse the configured instance (PLAN §6.2 item 9) and
     * hand out a fresh reference on it. */
    pthread_mutex_lock(&g_cache_mutex);
    rvcl_dispatch_result_t *cached = cache_lookup(desc);
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

    const rvcl_kernel_desc_t *best = NULL;
    int best_prio = 1 << 30;
    const rvcl_op_desc_t od = {.matmul = *desc};

    for (int i = 0, n = rvcl_registry_count(); i < n; i++) {
        const rvcl_kernel_desc_t *kd = rvcl_registry_get(i);
        if (!kd) continue;

        /* Stages 2-4 filters (stage 1 = operation check above). */
        if (kd->operation != RVCL_OP_MATMUL) continue; /* stage 1 */
        if (!filter_datatype(kd, desc)) continue; /* stage 2 */
        if (!filter_capability(kd, cap)) continue; /* stage 3 */
        if (!filter_shape(kd, desc)) continue; /* stage 4 */

        int prio = backend_priority(kd);
        if (prio < best_prio) {
            best_prio = prio;
            best = kd;
        }
    }

    if (!best) return RVCL_UNSUPPORTED;

    /* Instantiate, validate, configure (validate is side-effect-free and
     * rerun here as the final gate; failure falls back to RVCL_UNSUPPORTED
     * so callers can retry with looser constraints). */
    rvcl_kernel_create_fn create = rvcl_registry_find_factory(best->name);
    if (!create) return RVCL_ERROR_INTERNAL;

    rvcl_kernel_t *k = NULL;
    rvcl_status_t st = create(&k, best);
    if (st != RVCL_SUCCESS) return st;

    if (k->vt && k->vt->validate) {
        st = k->vt->validate(&od, cap);
        if (st != RVCL_SUCCESS) {
            rvcl_kernel_destroy(k);
            return RVCL_UNSUPPORTED;
        }
    }

    rvcl_dispatch_result_t *r = calloc(1, sizeof(*r));
    if (!r) {
        rvcl_kernel_destroy(k);
        return RVCL_ERROR_OUT_OF_MEMORY;
    }
    r->kernel = k;
    r->num_ws = 0;
    r->refs = 1; /* the cache's reference (inserted below) */

    st = rvcl_kernel_configure(k, &od, r->ws_req, &r->num_ws);
    if (st != RVCL_SUCCESS) {
        rvcl_kernel_destroy(k);
        free(r);
        return st;
    }

    /* Insert into the cache (evicts a conflicting slot), then hand out
     * the caller's own reference. */
    pthread_mutex_lock(&g_cache_mutex);
    cache_insert(desc, r);
    pthread_mutex_unlock(&g_cache_mutex);
    r->refs++; /* caller's reference */

    *result = r;
    return RVCL_SUCCESS;
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
