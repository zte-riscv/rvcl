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

/* The registry is a set of per-operator views over one shared slot pool:
 *   - kernel_registry_base_t (one subclass per operation) keeps an
 *     operation's registration-ordered index; dispatch snapshots one
 *     view, so a selection loop never scans other operations' kernels;
 *   - registry_set_t owns the pool plus every global-table property:
 *     the pool itself (stable-slot storage), the global registration
 *     order the public iteration contract exposes, cross-operation name
 *     uniqueness, the shared capacity and the generation counter;
 *   - the pool keeps the stable-slot guarantees: descriptors live in
 *     fixed slots that unregister never moves, so a pointer handed out
 *     by rvcl_registry_get() stays valid while its kernel is
 *     registered; freed slots are reused only after all never-used
 *     slots are spent, so a pointer to an unregistered kernel must not
 *     be used: its slot may later hold a different descriptor.
 *
 * All state lives behind registry_set_t's single mutex; a view's methods
 * are called by the set under that lock (the pool is passed in: views
 * own indexes, not storage). The C API below forwards to the singleton,
 * which is constructed on first use so a plugin constructor registering
 * pre-main is safe regardless of static-init order.
 *
 * Registrations are fully validated before any state changes (reject
 * atomically, never partially register). Unregistering the last
 * unbounded kernel of a live datatype tuple is rejected so every tuple
 * keeps a fallback. No performance decisions here: cost_fn is stored,
 * never called. */

#include "rvcl/registry.h"

#include "registry_internal.h"
#include "rvcl/dispatch.h"

#include <atomic>
#include <pthread.h>
#include <string.h>

namespace {

/* ------------------------------------------------------------ validation */

int dtype_is_concrete(rvcl_dtype_t dt) {
    /* Descriptors are match keys: AUTO is a request-side marker only.
     * Block-scale dtypes (E8M0 / MX / NV formats) are concrete dtypes,
     * so a kernel may declare them in its tuple. */
    return dt > RVCL_DTYPE_AUTO && dt <= RVCL_DTYPE_MXFP4;
}

int feature_id_is_valid(rvcl_feature_id_t id) {
    if (id == RVCL_FEATURE_NONE) return 0;
    /* Standard range derives from the enum sentinel (STD_LAST is not a
     * feature); vendor/TPE ids escape through the vendor slot. */
    if (id >= RVCL_FEATURE_RV64 && id < RVCL_FEATURE_STD_LAST) return 1;
    return id >= RVCL_FEATURE_VENDOR_FIRST;
}

int tier_is_valid(rvcl_vlen_tier_t t) {
    switch (t) {
        case RVCL_VLEN_TIER_ANY:
        case RVCL_VLEN_TIER_128:
        case RVCL_VLEN_TIER_256:
        case RVCL_VLEN_TIER_512:
        case RVCL_VLEN_TIER_1024: return 1;
        default: return 0;
    }
}

int quant_is_valid(rvcl_quant_support_t q) {
    switch (q) {
        case RVCL_QUANT_NONE:
        case RVCL_QUANT_PER_TENSOR:
        case RVCL_QUANT_PER_CHANNEL: return 1;
        default: return 0;
    }
}

int assert_level_is_valid(rvcl_assert_level_t l) {
    switch (l) {
        case RVCL_ASSERT_BITEXACT:
        case RVCL_ASSERT_TOLERANT: return 1;
        default: return 0;
    }
}

/* Full-descriptor gate; runs before taking the lock so a rejected
 * registration cannot leave partial state behind. */
rvcl_status_t validate_desc(
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
    if (!assert_level_is_valid(desc->assert_level))
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

int desc_is_unbounded(const rvcl_kernel_desc_t *d) {
    return d->max_m == 0 && d->max_n == 0 && d->max_k == 0;
}

/* Datatype-tuple identity - operation + (A,B,acc,C) - the unit the
 * fallback invariant groups by. */
typedef struct {
    rvcl_operation_t op;
    rvcl_dtype_t dtype_a;
    rvcl_dtype_t dtype_b;
    rvcl_dtype_t dtype_acc;
    rvcl_dtype_t dtype_c;
} registry_tuple_key_t;

registry_tuple_key_t tuple_key_of(const rvcl_kernel_desc_t *d) {
    registry_tuple_key_t k;
    k.op = d->operation;
    k.dtype_a = d->dtype_a;
    k.dtype_b = d->dtype_b;
    k.dtype_acc = d->dtype_acc;
    k.dtype_c = d->dtype_c;
    return k;
}

int same_tuple(const registry_tuple_key_t *k, const rvcl_kernel_desc_t *d) {
    return k->op == d->operation && k->dtype_a == d->dtype_a
            && k->dtype_b == d->dtype_b && k->dtype_acc == d->dtype_acc
            && k->dtype_c == d->dtype_c;
}

/* ------------------------------------------------------------- storage */

typedef struct {
    rvcl_kernel_desc_t desc;
    rvcl_kernel_create_fn create;
    int used;
} registry_slot_t;

/* One entry of the global registration order: locates a pool slot. */
typedef struct {
    rvcl_operation_t op;
    int slot;
} registry_handle_t;

/* ------------------------------------------------- per-operation view
 *
 * One operation's registration-ordered index over the shared pool. The
 * order doubles as the dispatch tie-break between equal-cost candidates
 * of this operation. Views hold no storage and no locks: the set calls
 * them under its mutex with the pool passed in. */
class kernel_registry_base_t {
public:
    explicit kernel_registry_base_t(rvcl_operation_t op) : op_(op), count_(0) {}

    /* Appends an already-filled pool slot to this operation's order. */
    void link(int slot) {
        order_[count_++] = slot;
        on_linked(slot);
    }

    /* Drops a slot from the order (the set clears the pool entry). The
     * set located the slot through the global order, which is appended
     * together with link(), so the slot is necessarily in this view. */
    void unlink(int slot) {
        int pos = 0;
        while (pos < count_ && order_[pos] != slot)
            pos++;
        if (pos == count_) return; /* defensive: not in this view */
        for (int i = pos; i < count_ - 1; i++)
            order_[i] = order_[i + 1];
        count_--;
        on_unlinked(slot);
    }

    /* Copies this operation's entries in registration order; dispatch
     * works on the copy without holding the registry lock. */
    int snapshot(const registry_slot_t *pool, rvcl_registry_view_t *out,
            int max_entries) const {
        int n = 0;
        for (; n < count_ && n < max_entries; n++) {
            const registry_slot_t *s = &pool[order_[n]];
            out[n].desc = s->desc;
            out[n].create = s->create;
        }
        return n;
    }

    /* Would removing `slot` leave its datatype tuple without an
     * unbounded fallback? Only meaningful when other entries of the
     * tuple remain; an emptying tuple imposes no constraint. The tuple
     * includes the operation, and every entry of this view shares it,
     * so the comparison covers the dtype fields only. */
    int tuple_keeps_fallback(const registry_slot_t *pool, int slot) const {
        const rvcl_kernel_desc_t *rm = &pool[slot].desc;
        int others = 0;
        for (int i = 0; i < count_; i++) {
            const int s = order_[i];
            if (s == slot) continue;
            const rvcl_kernel_desc_t *d = &pool[s].desc;
            if (d->dtype_a != rm->dtype_a || d->dtype_b != rm->dtype_b
                    || d->dtype_acc != rm->dtype_acc
                    || d->dtype_c != rm->dtype_c)
                continue;
            others = 1;
            if (desc_is_unbounded(d)) return 1;
        }
        return !others;
    }

    /* Does `slot`'s datatype tuple contain at least one unbounded
     * kernel? Every live tuple must (whole-table fallback rule). */
    int tuple_has_unbounded(const registry_slot_t *pool, int slot) const {
        const rvcl_kernel_desc_t *d = &pool[slot].desc;
        for (int i = 0; i < count_; i++) {
            const rvcl_kernel_desc_t *o = &pool[order_[i]].desc;
            if (o->dtype_a != d->dtype_a || o->dtype_b != d->dtype_b
                    || o->dtype_acc != d->dtype_acc || o->dtype_c != d->dtype_c)
                continue;
            if (desc_is_unbounded(o)) return 1;
        }
        return 0;
    }

protected:
    /* Views are members of the set's single static instance and never
     * deleted through a base pointer. */
    ~kernel_registry_base_t() = default;

    /* Differentiation hooks for subclasses (called under the set's
     * lock): an operation-specific secondary index maintains itself
     * here without the base class or other operations changing. */
    virtual void on_linked(int slot) { (void)slot; }
    virtual void on_unlinked(int slot) { (void)slot; }

private:
    const rvcl_operation_t op_;
    int order_[RVCL_REGISTRY_MAX_KERNELS];
    int count_;
};

/* One view per operation, covering the full operation enum: the
 * registry accepts kernels for any valid operation regardless of
 * whether built-ins or a dispatch table entry exist for it. Bodies stay
 * empty until an operation grows its own secondary index. */
class matmul_registry_t final : public kernel_registry_base_t {
public:
    matmul_registry_t() : kernel_registry_base_t(RVCL_OP_MATMUL) {}
};

class gemv_registry_t final : public kernel_registry_base_t {
public:
    gemv_registry_t() : kernel_registry_base_t(RVCL_OP_GEMV) {}
};

class conv2d_registry_t final : public kernel_registry_base_t {
public:
    conv2d_registry_t() : kernel_registry_base_t(RVCL_OP_CONV2D) {}
};

class reserved_registry_t final : public kernel_registry_base_t {
public:
    reserved_registry_t() : kernel_registry_base_t(RVCL_OP_RESERVED) {}
};

/* ------------------------------------------------------------ the set
 *
 * Sole owner of storage and every whole-table property: the stable-slot
 * pool, the global registration order behind the public iteration
 * contract, cross-operation name uniqueness, the shared capacity, the
 * mutex and the generation counter. */
class registry_set_t {
public:
    /* First-use singleton: constructing on first call keeps a pre-main
     * registration (a plugin constructor) safe regardless of static
     * initialization order across translation units. The views are
     * members wired by the constructor, so a call is one guard check
     * and nothing else - instance() sits on the dispatch hot path via
     * rvcl_registry_generation(). */
    static registry_set_t &instance();

    rvcl_status_t register_kernel(
            const rvcl_kernel_desc_t *desc, rvcl_kernel_create_fn create);
    rvcl_status_t unregister_kernel(const char *name);
    int count() const;
    const rvcl_kernel_desc_t *get(int index) const;
    rvcl_kernel_create_fn find_factory(const char *name) const;
    int snapshot_op(rvcl_operation_t op, rvcl_registry_view_t *out,
            int max_entries, unsigned *generation) const;
    unsigned generation() const {
        return generation_.load(std::memory_order_acquire);
    }
    rvcl_status_t verify_fallback_invariant(int before) const;
    rvcl_status_t rollback(int count);

private:
    registry_set_t() {
        by_op_[RVCL_OP_MATMUL] = &matmul_;
        by_op_[RVCL_OP_GEMV] = &gemv_;
        by_op_[RVCL_OP_CONV2D] = &conv2d_;
        by_op_[RVCL_OP_RESERVED] = &reserved_;
    }
    registry_set_t(const registry_set_t &) = delete;
    registry_set_t &operator=(const registry_set_t &) = delete;

    /* Position in global_order_ of the named kernel, filling *out; -1
     * when unknown. Caller holds the mutex. */
    int find_handle_locked(const char *name, registry_handle_t *out) const {
        for (int i = 0; i < total_; i++) {
            const registry_handle_t h = global_order_[i];
            if (strcmp(pool_[h.slot].desc.name, name) == 0) {
                if (out) *out = h;
                return i;
            }
        }
        return -1;
    }

    /* Lowest never-used slot first; freed slots only afterwards (see
     * file header). Caller holds the mutex and has checked capacity. */
    int alloc_slot_locked() {
        if (slots_used_ < RVCL_REGISTRY_MAX_KERNELS) return slots_used_++;
        for (int s = 0; s < RVCL_REGISTRY_MAX_KERNELS; s++)
            if (!pool_[s].used) return s;
        return -1;
    }

    /* One view per operation, owned here and wired into by_op_ once by
     * the constructor (see the subclasses above). */
    matmul_registry_t matmul_;
    gemv_registry_t gemv_;
    conv2d_registry_t conv2d_;
    reserved_registry_t reserved_;
    kernel_registry_base_t *by_op_[RVCL_OP_COUNT];
    registry_slot_t pool_[RVCL_REGISTRY_MAX_KERNELS];
    registry_handle_t global_order_[RVCL_REGISTRY_MAX_KERNELS];
    int total_ = 0; /* live entries across all operations */
    int slots_used_ = 0; /* high-water mark of slots ever allocated */
    mutable pthread_mutex_t mutex_ = PTHREAD_MUTEX_INITIALIZER;
    /* Bumped inside the mutex on every registry change. Dispatch reads
     * it without the registry lock when validating a cached selection. */
    std::atomic<unsigned> generation_ {0};
};

registry_set_t &registry_set_t::instance() {
    static registry_set_t the_set;
    return the_set;
}

/* Registry changes invalidate the dispatch cache via
 * rvcl_dispatch_cache_clear() (declared in dispatch.h). It is called
 * outside the mutex (lock order registry -> cache, never nested).
 * Generation tags already make stale entries unusable on their next
 * lookup; the sweep additionally drops their cache references now, so a
 * kernel instance removed from the registry is released without waiting
 * for a colliding key. */

rvcl_status_t registry_set_t::register_kernel(
        const rvcl_kernel_desc_t *desc, rvcl_kernel_create_fn create) {
    rvcl_status_t st = validate_desc(desc, create);
    if (st != RVCL_SUCCESS) return st;

    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    /* Names are unique across the whole table, not per operation. */
    if (set.find_handle_locked(desc->name, NULL) >= 0) {
        pthread_mutex_unlock(&set.mutex_);
        return RVCL_ERROR_ALREADY_REGISTERED;
    }
    if (set.total_ >= RVCL_REGISTRY_MAX_KERNELS) {
        pthread_mutex_unlock(&set.mutex_);
        return RVCL_ERROR_OUT_OF_MEMORY;
    }
    const int s = set.alloc_slot_locked();
    set.pool_[s].desc = *desc;
    set.pool_[s].create = create;
    set.pool_[s].used = 1;
    set.by_op_[desc->operation]->link(s);
    set.global_order_[set.total_++] = registry_handle_t {desc->operation, s};
    set.generation_.fetch_add(1, std::memory_order_release);
    pthread_mutex_unlock(&set.mutex_);

    rvcl_dispatch_cache_clear();
    return RVCL_SUCCESS;
}

rvcl_status_t registry_set_t::unregister_kernel(const char *name) {
    if (!name) return RVCL_ERROR_INVALID_ARGUMENT;
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    registry_handle_t h;
    const int pos = set.find_handle_locked(name, &h);
    if (pos < 0) {
        pthread_mutex_unlock(&set.mutex_);
        return RVCL_ERROR_NOT_FOUND;
    }
    if (!set.by_op_[h.op]->tuple_keeps_fallback(set.pool_, h.slot)) {
        pthread_mutex_unlock(&set.mutex_);
        return RVCL_ERROR_REGISTRY_INVARIANT;
    }
    /* Remove from both order indexes; descriptor storage stays in place
     * so pointers to the remaining entries stay coherent. Pointers to
     * the removed kernel become invalid (see registry.h). */
    set.by_op_[h.op]->unlink(h.slot);
    for (int i = pos; i < set.total_ - 1; i++)
        set.global_order_[i] = set.global_order_[i + 1];
    set.total_--;
    set.pool_[h.slot].used = 0;
    set.generation_.fetch_add(1, std::memory_order_release);
    pthread_mutex_unlock(&set.mutex_);

    rvcl_dispatch_cache_clear();
    return RVCL_SUCCESS;
}

int registry_set_t::count() const {
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    const int n = set.total_;
    pthread_mutex_unlock(&set.mutex_);
    return n;
}

const rvcl_kernel_desc_t *registry_set_t::get(int index) const {
    const rvcl_kernel_desc_t *out = NULL;
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    if (index >= 0 && index < set.total_)
        out = &set.pool_[set.global_order_[index].slot].desc;
    pthread_mutex_unlock(&set.mutex_);
    return out;
}

rvcl_kernel_create_fn registry_set_t::find_factory(const char *name) const {
    if (!name) return NULL;
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    registry_handle_t h;
    rvcl_kernel_create_fn fn = NULL;
    if (set.find_handle_locked(name, &h) >= 0) fn = set.pool_[h.slot].create;
    pthread_mutex_unlock(&set.mutex_);
    return fn;
}

/* Consistent copy of one operation's entries (see registry_internal.h):
 * the generation is read in the same critical section as the copy. */
int registry_set_t::snapshot_op(rvcl_operation_t op, rvcl_registry_view_t *out,
        int max_entries, unsigned *generation) const {
    if (op < 0 || op >= RVCL_OP_COUNT || !out || max_entries <= 0) {
        if (generation) *generation = this->generation();
        return 0;
    }
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    const unsigned gen = set.generation_.load(std::memory_order_relaxed);
    const int n = set.by_op_[op]->snapshot(set.pool_, out, max_entries);
    pthread_mutex_unlock(&set.mutex_);
    if (generation) *generation = gen;
    return n;
}

/* The whole-table snapshot was dropped with its last whole-table
 * consumer: dispatch snapshots per operation now. */

/* ------------------------------------------------------------ invariant */

/* Whole-table fallback check, run once initialization finished (internal
 * symbol, not public API; declared in registry_internal.h). Tuples that
 * are purely app-owned - every entry registered before the built-in
 * pass, i.e. at global order positions < before - are exempt; every
 * other live tuple must keep an unbounded kernel. */
rvcl_status_t registry_set_t::verify_fallback_invariant(int before) const {
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    /* Tuple keys the built-in pass joined: entries at positions >=
     * before. Registration is not concurrent with init, so the split is
     * stable while this runs. */
    registry_tuple_key_t joined[RVCL_REGISTRY_MAX_KERNELS];
    int joined_n = 0;
    for (int i = before; i < set.total_; i++) {
        const rvcl_kernel_desc_t *d
                = &set.pool_[set.global_order_[i].slot].desc;
        int seen = 0;
        for (int k = 0; k < joined_n; k++)
            if (same_tuple(&joined[k], d)) {
                seen = 1;
                break;
            }
        if (!seen) joined[joined_n++] = tuple_key_of(d);
    }
    for (int i = 0; i < set.total_; i++) {
        const int slot = set.global_order_[i].slot;
        const rvcl_kernel_desc_t *d = &set.pool_[slot].desc;
        int touched = 0;
        for (int k = 0; k < joined_n; k++)
            if (same_tuple(&joined[k], d)) {
                touched = 1;
                break;
            }
        if (!touched) continue; /* purely app-owned tuple: exempt */
        if (!set.by_op_[d->operation]->tuple_has_unbounded(set.pool_, slot)) {
            pthread_mutex_unlock(&set.mutex_);
            return RVCL_ERROR_REGISTRY_INVARIANT;
        }
    }
    pthread_mutex_unlock(&set.mutex_);
    return RVCL_SUCCESS;
}

/* Init-path rollback: drop the tail of the global registration order
 * across all operations, leaving the first `count` entries untouched. */
rvcl_status_t registry_set_t::rollback(int count) {
    registry_set_t &set = instance();
    pthread_mutex_lock(&set.mutex_);
    if (count < 0 || count > set.total_) {
        pthread_mutex_unlock(&set.mutex_);
        return RVCL_ERROR_INTERNAL;
    }
    /* Nothing to roll back: leave the generation and the dispatch cache
     * alone instead of invalidating every cached selection for nothing. */
    if (set.total_ == count) {
        pthread_mutex_unlock(&set.mutex_);
        return RVCL_SUCCESS;
    }
    while (set.total_ > count) {
        const registry_handle_t h = set.global_order_[set.total_ - 1];
        set.by_op_[h.op]->unlink(h.slot);
        set.pool_[h.slot].used = 0;
        set.total_--;
    }
    set.generation_.fetch_add(1, std::memory_order_release);
    pthread_mutex_unlock(&set.mutex_);

    rvcl_dispatch_cache_clear();
    return RVCL_SUCCESS;
}

} /* namespace */

/* -------------------------------------------------------------- zero cost */

RVCL_API uint64_t rvcl_kernel_cost_zero(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return 0; /* "no cost information" */
}

/* ------------------------------------------------------------- public API */

rvcl_status_t rvcl_register_kernel(
        const rvcl_kernel_desc_t *desc, rvcl_kernel_create_fn create) {
    return registry_set_t::instance().register_kernel(desc, create);
}

rvcl_status_t rvcl_unregister_kernel(const char *name) {
    return registry_set_t::instance().unregister_kernel(name);
}

int rvcl_registry_count(void) {
    return registry_set_t::instance().count();
}

const rvcl_kernel_desc_t *rvcl_registry_get(int index) {
    return registry_set_t::instance().get(index);
}

rvcl_kernel_create_fn rvcl_registry_find_factory(const char *name) {
    return registry_set_t::instance().find_factory(name);
}

unsigned rvcl_registry_generation(void) {
    return registry_set_t::instance().generation();
}

int rvcl_registry_snapshot_op(rvcl_operation_t op, rvcl_registry_view_t *out,
        int max_entries, unsigned *generation) {
    return registry_set_t::instance().snapshot_op(
            op, out, max_entries, generation);
}

rvcl_status_t rvcl_registry_verify_fallback_invariant(int before) {
    return registry_set_t::instance().verify_fallback_invariant(before);
}

rvcl_status_t rvcl_registry_rollback(int count) {
    return registry_set_t::instance().rollback(count);
}
