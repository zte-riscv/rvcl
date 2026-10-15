#!/usr/bin/python3

# *******************************************************************************
# Copyright 2026 ZTE Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
#
# Enforce the Conventional Commits format on every commit in base..head.
# See the "Commit message conventions" section in CONTRIBUTING.md.
# *******************************************************************************

import argparse
import re
import subprocess

# Conventional Commits type whitelist, optional (<scope>) and optional "!"
# breaking-change marker, then ": " and a non-empty subject that starts with a
# non-space character. Leading whitespace and merge commits do not match and
# are rejected below with a specific message.
SUBJECT_RE = re.compile(
    r"^(feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert)"
    r"(\([a-z0-9_-]+\))?!?: \S.*$"
)

SUBJECT_MAX_LEN = 72


def check_subject(subject):
    status = "Commit subject: "
    if subject.startswith("Merge "):
        print(f"{status} FAILED: merge commits are not allowed")
        return False
    if not SUBJECT_RE.match(subject):
        if subject != subject.lstrip():
            print(f"{status} FAILED: subject must not start with whitespace")
        else:
            print(
                f"{status} FAILED: expected '<type>(<scope>)?(!)?: <description>' "
                f"with type in "
                "feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
            )
        return False
    if len(subject) > SUBJECT_MAX_LEN:
        print(
            f"{status} FAILED: subject exceeds {SUBJECT_MAX_LEN} characters"
        )
        return False
    print(f"{status} OK")
    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("head", help="Head commit of the PR branch")
    parser.add_argument("base", help="Base commit of the PR branch")
    parser.add_argument(
        "--pr-title",
        default=None,
        help="Optional PR title to validate (squash merge makes it the "
        "mainline subject, so it must match a commit subject)",
    )
    args = parser.parse_args()

    is_ok = True

    commit_range = args.base + ".." + args.head
    messages = subprocess.check_output(
        ["git", "rev-list", "--format=oneline", commit_range],
        universal_newlines=True,
    )
    for line in messages.splitlines():
        print(line)
        subject = line.split(" ", 1)[1]
        is_ok = check_subject(subject) and is_ok

    if args.pr_title:
        print("PR title:", args.pr_title)
        is_ok = check_subject(args.pr_title) and is_ok

    if is_ok:
        print("All commit subjects are formatted correctly.")
    else:
        print(
            "Some commit subjects failed the Conventional Commits check. "
            "Align them with CONTRIBUTING.md and update the PR."
        )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
