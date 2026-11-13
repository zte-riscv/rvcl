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
 * examples/vendor_kernel.cpp — out-of-tree vendor kernel integration walkthrough
 *
 * Demonstrates the public contract a plugin uses to hook into rvcl as an
 * out-of-tree backend with the same dispatch semantics as an in-tree one:
 *
 *   1. ABI handshake   — rvcl_kernel_abi_version() vs RVCL_KERNEL_ABI_VERSION;
 *   2. probe register  — rvcl_capability_register_vendor_probe() (before the
 *                        first capability query), plus rvcl_capability_try_csr()
 *                        as the safe CSR-read helper;
 *   3. kernel register — rvcl_register_kernel() requiring the vendor feature,
 *                        cost via the public rvcl_cost_matmul_hint();
 *   4. dispatch        — stage-3 feature pre-filter makes the vendor kernel
 *                        visible on capable hosts, invisible otherwise;
 *   5. self-check      — rvcl_registry_find_factory() resolves the name.
 *
 * Protocol notes (the contract text the plugin must honor):
 *  - Two loading forms are recognized: LD_PRELOAD (environment-level load;
 *    the plugin's constructor registers pre-main) and direct application
 *    linking (the application calls a registration entry point explicitly).
 *    rvcl does no active dlopen of plugins.
 *  - Register a probe BEFORE the first capability query or dispatch — the
 *    snapshot is built once per process. A probe registered after that
 *    returns RVCL_ERROR_ALREADY_INITIALIZED; a runtime-loaded plugin that
 *    missed the window must fall back to a validate()-time self-probe
 *    (side-effect-free, cached) instead of relying on the snapshot.
 *  - Cost trap: do not bind rvcl_kernel_cost_zero() and expect ordering —
 *    a real cost model (here via rvcl_cost_matmul_hint) is what orders
 *    candidates. Equal costs tie-break by registration order.
 *  - Only the vendor-range feature ids (>= RVCL_FEATURE_VENDOR_FIRST) a
 *    probe declares survive into the snapshot; the framework sanitizes the
 *    probe's increment.
 *
 * On a host where the vendor feature is not in the snapshot the vendor
 * kernel is filtered at stage 3 and the dispatch on its tuple returns
 * RVCL_UNSUPPORTED — the expected out-of-band outcome this walkthrough
 * treats as success-for-demonstration.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rvcl/rvcl.h"

/* Vendor feature id this plugin declares and requires (vendor range). */
#define VENDOR_FEATURE ((rvcl_feature_id_t)0x1001)
#define VENDOR_KERNEL_NAME "vendor_matmul_f32_plugin"

/* ---------------------------------------------------------- vendor probe */
/*
 * A probe appends the plugin's feature ids to the capability snapshot during
 * its single build. It runs once, single-threaded, before dispatch; here it
 * demonstrates the safe CSR read (returns 0 when the CSR does not exist) and
 * unconditionally declares the vendor feature so the write channel is shown
 * regardless of the host.
 */
static rvcl_status_t vendor_probe(rvcl_capability_t *cap) {
    uint32_t csr_value = 0;
    /* Safe read of an optional, vendor-specific CSR; the helper traps and
     * returns 0 if the CSR is not implemented, so the probe never faults. */
    (void)rvcl_capability_try_csr(0x7c0, &csr_value);
    if (csr_value == 0)
        csr_value = 1; /* host detail; feature below is the point */
    if (cap->num_features < RVCL_MAX_FEATURES) {
        /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately not
         * an enumerator of rvcl_feature_id_t. */
        // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
        cap->features[cap->num_features].id = VENDOR_FEATURE;
        cap->features[cap->num_features].value = 1;
        cap->num_features++;
    }
    return RVCL_SUCCESS;
}

/* ------------------------------------------------------ vendor cost model */
/* The vendor forwards to the public hint with its own coefficient (128 =
 * 0.5x of the built-in cheap vector category): the framework owns the shape
 * heuristic and fixed-point convention, the coefficient stays vendor-owned. */
static uint64_t vendor_cost(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    return rvcl_cost_matmul_hint(desc, cap, 128);
}

/* ------------------------------------------------- minimal vendor kernel */
/* A do-nothing kernel: the walkthrough demonstrates registration, feature
 * visibility and selection, not arithmetic. */

static rvcl_status_t vendor_validate(
        const rvcl_op_desc_t *desc, const rvcl_capability_t *cap) {
    (void)desc;
    (void)cap;
    return RVCL_SUCCESS;
}

static rvcl_status_t vendor_configure(rvcl_kernel_t *k,
        const rvcl_op_desc_t *desc, rvcl_workspace_req_t *ws_req, int *num_ws) {
    (void)k;
    (void)desc;
    /* bounded by sizeof; demonstration code, no unbounded buffer. */
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)
    if (ws_req) memset(ws_req, 0, sizeof(*ws_req));
    if (num_ws) *num_ws = 0;
    return RVCL_SUCCESS;
}

static rvcl_status_t vendor_run(rvcl_kernel_t *k,
        const rvcl_tensor_pack_t *pack, const rvcl_work_range_t *range) {
    (void)k;
    (void)pack;
    (void)range;
    return RVCL_SUCCESS;
}

static void vendor_destroy(rvcl_kernel_t *k) {
    free(k);
}

static const rvcl_kernel_vtable_t vendor_vtable = {
        .validate = vendor_validate,
        .configure = vendor_configure,
        .pack_a = NULL,
        .pack_b = NULL,
        .run = vendor_run,
        .destroy = vendor_destroy,
};

static rvcl_status_t vendor_create(
        rvcl_kernel_t **out, const rvcl_kernel_desc_t *desc) {
    rvcl_kernel_t *k = (rvcl_kernel_t *)calloc(1, sizeof(rvcl_kernel_t));
    if (!k) return RVCL_ERROR_OUT_OF_MEMORY;
    k->vt = &vendor_vtable;
    k->desc = *desc;
    *out = k;
    return RVCL_SUCCESS;
}

/* ---------------------------------------------------------------- main */

int main(void) {
    int failures = 0;

    /* 1. ABI handshake: a plugin built against a differently-laid-out header
     * must refuse to register rather than pass a short descriptor. Advisory:
     * the library cannot inspect the plugin's struct layout. */
    if (rvcl_kernel_abi_version() != RVCL_KERNEL_ABI_VERSION) {
        /* fixed format string; demonstration code. */
        // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)
        fprintf(stderr, "vendor kernel ABI mismatch: library %u, header %u\n",
                rvcl_kernel_abi_version(), RVCL_KERNEL_ABI_VERSION);
        return 1;
    }
    printf("ABI handshake: ok (v%u)\n", rvcl_kernel_abi_version());

    /* 2. Probe register: must precede the first capability query. */
    if (rvcl_capability_register_vendor_probe(vendor_probe) != RVCL_SUCCESS) {
        /* fixed format string; demonstration code. */
        // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)
        fprintf(stderr, "probe registration failed\n");
        return 1;
    }
    printf("probe registered\n");

    /* 3. Kernel register on a tuple no built-in occupies, requiring the
     * vendor feature; the cost model forwards to the public hint. */
    rvcl_kernel_desc_t d;
    /* bounded by sizeof; demonstration code. */
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)
    memset(&d, 0, sizeof(d));
    /* bounded by RVCL_KERNEL_NAME_MAX; demonstration code. */
    // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)
    snprintf(d.name, RVCL_KERNEL_NAME_MAX, "%s", VENDOR_KERNEL_NAME);
    d.operation = RVCL_OP_MATMUL;
    d.dtype_a = RVCL_DTYPE_INT8;
    d.dtype_b = RVCL_DTYPE_INT8;
    d.dtype_acc = RVCL_DTYPE_INT32;
    d.dtype_c = RVCL_DTYPE_INT32;
    d.quant_support = RVCL_QUANT_NONE;
    /* vendor-range id (>= RVCL_FEATURE_VENDOR_FIRST) is deliberately not an
     * enumerator of rvcl_feature_id_t. */
    // NOLINTNEXTLINE(clang-analyzer-optin.core.EnumCastOutOfRange)
    d.required_features[0] = VENDOR_FEATURE;
    d.num_required_features = 1;
    d.alignment_a = d.alignment_b = d.alignment_c = 1;
    d.vlen_tier = RVCL_VLEN_TIER_ANY;
    d.cost_fn = vendor_cost;
    d.assert_level = RVCL_ASSERT_BITEXACT;
    if (rvcl_register_kernel(&d, vendor_create) != RVCL_SUCCESS) {
        /* fixed format string; demonstration code. */
        // NOLINTNEXTLINE(clang-analyzer-security.insecureAPI.DeprecatedOrUnsafeBufferHandling)
        fprintf(stderr, "kernel registration failed\n");
        return 1;
    }
    printf("kernel registered: %s\n", VENDOR_KERNEL_NAME);

    /* 4. Dispatch: stage-3 pre-filter gates on the vendor feature. The
     * probe ran during the snapshot build, so on this process the feature is
     * present and the vendor kernel is the only candidate on its tuple. */
    rvcl_matmul_desc_t req = {
            .m = 16,
            .n = 16,
            .k = 16,
            .dtype_a = RVCL_DTYPE_INT8,
            .dtype_b = RVCL_DTYPE_INT8,
            .dtype_acc = RVCL_DTYPE_INT32,
            .dtype_c = RVCL_DTYPE_INT32,
            .dtype_bias = RVCL_DTYPE_AUTO,
            .quant = {},
    };
    rvcl_dispatch_result_t *r = NULL;
    const rvcl_status_t st = rvcl_dispatch_matmul(&req, &r);
    if (st == RVCL_SUCCESS && r) {
        printf("dispatch selected: %s\n", rvcl_dispatch_kernel_name(r));
        if (strcmp(rvcl_dispatch_kernel_name(r), VENDOR_KERNEL_NAME) != 0) {
            printf("FAIL: unexpected kernel selected\n");
            failures++;
        }
        rvcl_dispatch_release(r);
    } else {
        /* Feature absent (host without the vendor capability, or a probe
         * that missed the registration window): stage-3 filters the kernel
         * and the tuple dispatches UNSUPPORTED. That is the documented
         * out-of-band fallback, not a failure of the chain. */
        printf("dispatch: %s (feature not in snapshot — validate-fallback "
               "expected)\n",
                st == RVCL_UNSUPPORTED ? "UNSUPPORTED" : "error (unexpected)");
        if (st != RVCL_UNSUPPORTED) failures++;
    }

    /* 5. Self-check: the factory resolves by name. */
    if (rvcl_registry_find_factory(VENDOR_KERNEL_NAME) != NULL) {
        printf("find_factory self-check: ok\n");
    } else {
        printf("FAIL: factory for %s not found\n", VENDOR_KERNEL_NAME);
        failures++;
    }

    if (failures) {
        printf("VENDOR-KERNEL: %d failure(s)\n", failures);
        return 1;
    }
    printf("VENDOR-KERNEL: all passed\n");
    return 0;
}
