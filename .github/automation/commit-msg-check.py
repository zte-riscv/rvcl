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
# *******************************************************************************

# Enforce the Conventional Commits format on every commit in base..head.
# See the "Commit message conventions" section in CONTRIBUTING.md.
#
# With --pr-title the PR title is validated as well, because a squash merge
# makes it the mainline subject. With --pr-number in addition, the exact
# squash subject "<title> (#<number>)" is validated, so the 72-character limit
# applies to what actually lands on main (the suffix included) rather than to
# the bare title.

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


def check_subject(subject, status="Commit subject: "):
    if subject.startswith("Merge "):
        print(f"{status}FAILED: merge commits are not allowed")
        return False
    if not SUBJECT_RE.match(subject):
        if subject != subject.lstrip():
            print(f"{status}FAILED: subject must not start with whitespace")
        else:
            print(
                f"{status}FAILED: '{subject}' does not match "
                f"'<type>(<scope>)?(!)?: <description>' with type in "
                "feat|fix|docs|style|refactor|perf|test|build|ci|chore|revert"
            )
        return False
    if len(subject) > SUBJECT_MAX_LEN:
        print(f"{status}FAILED: subject exceeds {SUBJECT_MAX_LEN} characters")
        return False
    print(f"{status}OK")
    return True


def check_pr_title(title, pr_number):
    # The bare title must already be a valid subject; the squash subject is
    # then checked separately so a failure names the exact constraint that
    # broke (bad format vs. too long once the ` (#N)` suffix is appended).
    status = "PR title: "
    print(status + title)
    is_ok = check_subject(title, status)

    if pr_number is None:
        return is_ok

    squash = f"{title} (#{pr_number})"
    squash_status = "Squash subject: "
    print(squash_status + squash)
    # Appending " (#<number>)" cannot break a subject that already matched
    # SUBJECT_RE, so the length is the only new failure mode here.
    # The `not is_ok` clause is a deliberate invariant guard: it reports a
    # bare title that already failed as FAILED rather than masking it, so a
    # failed bare title never gets a passing squash subject.
    if not is_ok or len(squash) > SUBJECT_MAX_LEN:
        if len(squash) > SUBJECT_MAX_LEN:
            print(
                f"{squash_status}FAILED: exceeds {SUBJECT_MAX_LEN} characters "
                f"({len(squash)}) once ' (#{pr_number})' is appended; "
                "shorten the PR title"
            )
        else:
            print(f"{squash_status}FAILED: the PR title is not a valid subject")
        return False
    print(squash_status + "OK")
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
    parser.add_argument(
        "--pr-number",
        type=int,
        default=None,
        help="Optional PR number; with --pr-title, also validate the squash "
        "subject '<title> (#<number>)' that lands on main",
    )
    args = parser.parse_args()

    if args.pr_number is not None and args.pr_title is None:
        parser.error("--pr-number requires --pr-title")

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
        is_ok = check_pr_title(args.pr_title, args.pr_number) and is_ok

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
