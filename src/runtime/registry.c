/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * registry.c — kernel registration and indexing
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Flat array with name index; Phase 0 scale is tens of kernels. The lookup
 * key is operation × (A,B,acc,C) tuples, resolved by dispatch iterating
 * registry entries (PLAN §4.5.2). No performance decisions here.
 ******************************************************************************/

#include "rvcl/registry.h"

#include <pthread.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    rvcl_kernel_desc_t desc;
    rvcl_kernel_create_fn create;
} registry_entry_t;

static registry_entry_t g_entries[RVCL_REGISTRY_MAX_KERNELS];
static int g_count;
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

static int find_index_locked(const char *name) {
    for (int i = 0; i < g_count; i++)
        if (strcmp(g_entries[i].desc.name, name) == 0) return i;
    return -1;
}

rvcl_status_t rvcl_register_kernel(const rvcl_kernel_desc_t *desc,
        rvcl_kernel_create_fn create) {
    if (!desc || !create || desc->name[0] == '\0')
        return RVCL_ERROR_INVALID_ARGUMENT;

    pthread_mutex_lock(&g_mutex);
    if (find_index_locked(desc->name) >= 0) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_ALREADY_REGISTERED;
    }
    if (g_count >= RVCL_REGISTRY_MAX_KERNELS) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_OUT_OF_MEMORY;
    }
    g_entries[g_count].desc = *desc;
    g_entries[g_count].create = create;
    g_count++;
    pthread_mutex_unlock(&g_mutex);
    return RVCL_SUCCESS;
}

rvcl_status_t rvcl_unregister_kernel(const char *name) {
    if (!name) return RVCL_ERROR_INVALID_ARGUMENT;
    pthread_mutex_lock(&g_mutex);
    int i = find_index_locked(name);
    if (i < 0) {
        pthread_mutex_unlock(&g_mutex);
        return RVCL_ERROR_NOT_FOUND;
    }
    /* keep order stable: shift down */
    memmove(&g_entries[i], &g_entries[i + 1],
            (size_t)(g_count - i - 1) * sizeof(registry_entry_t));
    g_count--;
    pthread_mutex_unlock(&g_mutex);
    return RVCL_SUCCESS;
}

int rvcl_registry_count(void) {
    pthread_mutex_lock(&g_mutex);
    int n = g_count;
    pthread_mutex_unlock(&g_mutex);
    return n;
}

const rvcl_kernel_desc_t *rvcl_registry_get(int index) {
    const rvcl_kernel_desc_t *out = NULL;
    pthread_mutex_lock(&g_mutex);
    if (index >= 0 && index < g_count) out = &g_entries[index].desc;
    pthread_mutex_unlock(&g_mutex);
    return out;
}

rvcl_kernel_create_fn rvcl_registry_find_factory(const char *name) {
    rvcl_kernel_create_fn fn = NULL;
    pthread_mutex_lock(&g_mutex);
    int i = find_index_locked(name);
    if (i >= 0) fn = g_entries[i].create;
    pthread_mutex_unlock(&g_mutex);
    return fn;
}
