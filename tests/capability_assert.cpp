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
 * tests/capability_assert.cpp — environment self-check gate (G7)
 *
 * Prints the detected capability — the description plus every feature
 * with its value — and, when RVCL_EXPECT_FEATURES is set, asserts that
 * each named feature is present, failing on the first miss.
 *
 * Why this exists: capability detection degrades instead of erroring
 * (capability.h), so a misconfigured environment — e.g. a QEMU too old
 * to emulate the hwprobe syscall — silently downgrades to the
 * compile-time view and the numerical targets still pass green through
 * the C fallback. The riscv64 CI runs this as its FIRST target with
 * per-tier expectations:
 *   vector tier   RVCL_EXPECT_FEATURES=rvv,fp16  (compile-time macros)
 *   baseline tier RVCL_EXPECT_FEATURES=rvv       (runtime hwprobe upgrade)
 *
 * The RVV value (VLEN in bits; 1 = present-but-unknown) is printed as
 * an observation only — the vlenb read is best-effort by contract.
 *
 * With the variable unset the run is print-only and always exits 0, so
 * plain local `ctest` never fails on host capability differences.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "rvcl/rvcl.h"

static const char *feature_name(rvcl_feature_id_t id) {
    switch (id) {
        case RVCL_FEATURE_RV64: return "rv64";
        case RVCL_FEATURE_RVV: return "rvv";
        case RVCL_FEATURE_IME: return "ime";
        case RVCL_FEATURE_VME: return "vme";
        case RVCL_FEATURE_AME: return "ame";
        case RVCL_FEATURE_FP32: return "fp32";
        case RVCL_FEATURE_FP16: return "fp16";
        case RVCL_FEATURE_BF16: return "bf16";
        case RVCL_FEATURE_INT8: return "int8";
        case RVCL_FEATURE_INT4: return "int4";
        default: return "unknown";
    }
}

/* Resolves an expectation name to a feature id; 0 when unknown — a typo
 * in the CI configuration must fail loudly, not silently pass. */
static int name_to_id(const char *name, rvcl_feature_id_t *id) {
    static const struct {
        const char *name;
        rvcl_feature_id_t id;
    } table[] = {
            {"rv64", RVCL_FEATURE_RV64},
            {"rvv", RVCL_FEATURE_RVV},
            {"ime", RVCL_FEATURE_IME},
            {"vme", RVCL_FEATURE_VME},
            {"ame", RVCL_FEATURE_AME},
            {"fp32", RVCL_FEATURE_FP32},
            {"fp16", RVCL_FEATURE_FP16},
            {"bf16", RVCL_FEATURE_BF16},
            {"int8", RVCL_FEATURE_INT8},
            {"int4", RVCL_FEATURE_INT4},
    };
    for (size_t i = 0; i < sizeof(table) / sizeof(table[0]); i++)
        if (strcmp(table[i].name, name) == 0) {
            *id = table[i].id;
            return 1;
        }
    return 0;
}

int main(void) {
    const rvcl_capability_t *cap = rvcl_get_capability();

    printf("capability: %s\n", cap->description);
    for (int i = 0; i < cap->num_features; i++)
        printf("  %s = %u\n", feature_name(cap->features[i].id),
                cap->features[i].value);

    const char *expect = getenv("RVCL_EXPECT_FEATURES");
    if (!expect || !*expect) return 0; /* print-only run */

    /* Comma-separated feature names, e.g. RVCL_EXPECT_FEATURES=rvv,fp16 */
    char buf[256];
    snprintf(buf, sizeof(buf), "%s", expect);

    int failures = 0;
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        rvcl_feature_id_t id;
        if (!name_to_id(tok, &id)) {
            printf("FAIL: unknown feature name '%s' in RVCL_EXPECT_FEATURES\n",
                    tok);
            failures++;
            continue;
        }
        if (rvcl_capability_has(cap, id)) {
            const rvcl_feature_t *f = rvcl_capability_get(cap, id);
            printf("ok: expected feature %s present (value=%u)\n", tok,
                    f->value);
        } else {
            printf("FAIL: expected feature %s NOT detected\n", tok);
            failures++;
        }
    }

    printf(failures ? "CAPABILITY: %d failure(s)\n"
                    : "CAPABILITY: all expectations met\n",
            failures);
    return failures ? 1 : 0;
}
