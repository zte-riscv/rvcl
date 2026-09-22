/*******************************************************************************
 * RVCL — RISC-V Compute Library
 * init.h — library initialization and version
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef RVCL_INIT_H
#define RVCL_INIT_H

#include "rvcl/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Registers built-in backends (idempotent, thread-safe). */
RVCL_API rvcl_status_t rvcl_init(void);

/* "MAJOR.MINOR.PATCH[-dev]" — consumed by FindRVCL-style version gating
 * (PLAN §4.11 item 7, the ACL 53.1-gate equivalent). */
RVCL_API const char *rvcl_get_version(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* RVCL_INIT_H */
