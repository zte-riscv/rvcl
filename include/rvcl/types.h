/*******************************************************************************
 * RVCL — RISC-V Compute Library
 *
 * Copyright 2026 RVCL contributors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * Basic types shared by all RVCL public headers: status codes, data types,
 * operations, feature ids. See PLAN-0001 §4.5 (datatype model) and RFC-0002
 * (capability/registry/dispatch vocabulary).
 ******************************************************************************/

#ifndef RVCL_TYPES_H
#define RVCL_TYPES_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Public symbol export: RVCL_BUILDING is defined when compiling the
 * library itself; consumers get default-hidden visibility plus these
 * declarations marked visible. */
#if defined(RVCL_BUILDING)
#define RVCL_API __attribute__((visibility("default")))
#else
#define RVCL_API
#endif

/* ---------------------------------------------------------------- version */

#define RVCL_VERSION_MAJOR 0
#define RVCL_VERSION_MINOR 1
#define RVCL_VERSION_PATCH 0
#define RVCL_VERSION_STRING "0.1.0"

/* ----------------------------------------------------------------- status */

typedef enum {
    RVCL_SUCCESS = 0,
    RVCL_ERROR_INVALID_ARGUMENT = 1,
    RVCL_ERROR_OUT_OF_MEMORY = 2,
    RVCL_ERROR_INTERNAL = 3,
    /* Dispatch / kernel selection */
    RVCL_UNSUPPORTED = 10,             /* no kernel can serve the request */
    RVCL_UNSUPPORTED_DTYPE_COMBO = 11, /* (A,B,acc,C) tuple is rejected */
    RVCL_ERROR_ALREADY_REGISTERED = 12,
    RVCL_ERROR_NOT_FOUND = 13,
    RVCL_ERROR_ALREADY_INITIALIZED = 14,
} rvcl_status_t;

/* ------------------------------------------------------------- data types */
/* Storage / compute / accumulation roles are separate (PLAN §4.5.1). */

typedef enum {
    RVCL_DTYPE_AUTO = 0, /* derive from habit rules when unspecified */
    RVCL_DTYPE_FP32,
    RVCL_DTYPE_FP16,
    RVCL_DTYPE_BF16,
    RVCL_DTYPE_INT8,
    RVCL_DTYPE_UINT8,
    RVCL_DTYPE_INT4, /* weights only in the roadmap (§4.5.2) */
    RVCL_DTYPE_INT32, /* accumulators / quantized intermediates */
    RVCL_DTYPE_FP8_E4M3, /* reserved (§4.5.6: no ratified base ext yet) */
    RVCL_DTYPE_FP8_E5M2, /* reserved */
} rvcl_dtype_t;

/* Element size in bytes; 0 for RVCL_DTYPE_AUTO / unknown. */
RVCL_API size_t rvcl_dtype_size(rvcl_dtype_t dt);

/* --------------------------------------------------------------- calendar */

typedef enum {
    RVCL_OP_MATMUL = 0,
    RVCL_OP_GEMV,
    RVCL_OP_CONV2D,
    RVCL_OP_COUNT,
} rvcl_operation_t;

/* ------------------------------------------------------------- capability */
/*
 * Feature ids are stable identifiers used by both the capability model and
 * kernel descriptors ("required_features"). Vendor/TPE extensions add new
 * ids without touching the dispatch model (PLAN §4.4).
 */

typedef enum {
    RVCL_FEATURE_NONE = 0,
    RVCL_FEATURE_RV64,
    RVCL_FEATURE_RVV,
    RVCL_FEATURE_IME,
    RVCL_FEATURE_VME,
    RVCL_FEATURE_AME,
    /* datatype features */
    RVCL_FEATURE_FP32,
    RVCL_FEATURE_FP16, /* implies Zvfh */
    RVCL_FEATURE_BF16, /* implies Zvfbfmin (+ Zvfbfwma for wfma) */
    RVCL_FEATURE_INT8,
    RVCL_FEATURE_INT4,
    /* vendor/TPE slot: values from RVCL_FEATURE_VENDOR_FIRST are assigned
     * by registration; TPE lands here without model changes (PLAN §4.4). */
    RVCL_FEATURE_VENDOR_FIRST = 0x1000,
} rvcl_feature_id_t;

#define RVCL_MAX_FEATURES 32

typedef struct {
    rvcl_feature_id_t id;
    uint32_t value; /* e.g. VLEN for RVV, tile id for a matrix engine */
} rvcl_feature_t;

/* ------------------------------------------------------------ work ranges */
/*
 * Partial output ranges for external parallelism (PLAN §4.3): kernels never
 * create threads; the scheduler slices work ranges and calls run() per slice.
 */

typedef struct {
    int64_t m_begin, m_end;
    int64_t n_begin, n_end;
    int64_t k_begin, k_end; /* full K per slice for plain GEMM; partial-K
                             * reduction requires a split-K plan (v0.4+) */
} rvcl_work_range_t;

/* Full-range constant initializer. */
RVCL_API rvcl_work_range_t rvcl_work_range_full(int64_t m, int64_t n, int64_t k);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_TYPES_H */
