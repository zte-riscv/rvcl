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
# Table-driven tests for commit-msg-check.py. Standard library only, so CI can
# run them with the system python3:
#
#   python3 .github/automation/test_commit_msg_check.py
# *******************************************************************************

import contextlib
import importlib.util
import io
import os
import unittest

SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "commit-msg-check.py")


def load_module():
    spec = importlib.util.spec_from_file_location("commit_msg_check", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


check = load_module()


def run_quietly(func, *args):
    """Run func with stdout captured; return (result, printed text)."""
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        result = func(*args)
    return result, buffer.getvalue()


class CheckSubjectTest(unittest.TestCase):
    def test_conforming_subject_passes(self):
        for subject in (
            "feat: add fp8 dtype support",
            "fix(ci): stop the title gate from splitting words",
            "refactor!: drop the legacy kernel ABI",
            "revert: feat: something",
        ):
            with self.subTest(subject=subject):
                ok, out = run_quietly(check.check_subject, subject)
                self.assertTrue(ok)
                self.assertIn("Commit subject: OK", out)

    def test_non_conforming_subject_fails(self):
        for subject in (
            "add fp8 dtype support",
            "feat:no space after the colon",
            "feat: ",
            "unknown: something",
            "feat(Scope): upper-case scope",
            "Merge branch 'main' into ci/pr-title-gate",
            " feat: leading space",
        ):
            with self.subTest(subject=subject):
                ok, out = run_quietly(check.check_subject, subject)
                self.assertFalse(ok)
                self.assertIn("FAILED", out)

    def test_length_boundary(self):
        limit = check.SUBJECT_MAX_LEN
        prefix = "feat: "
        self.assertEqual(
            len(prefix + "a" * (limit - len(prefix))), limit, "test fixture is stale"
        )
        ok, out = run_quietly(
            check.check_subject, prefix + "a" * (limit - len(prefix))
        )
        self.assertTrue(ok, out)
        ok, out = run_quietly(
            check.check_subject, prefix + "a" * (limit - len(prefix) + 1)
        )
        self.assertFalse(ok)
        self.assertIn(f"exceeds {limit} characters", out)


class CheckPrTitleTest(unittest.TestCase):
    def test_squash_subject_length_boundary(self):
        # 66-character title + " (#35)" is exactly 72 and passes; a
        # 67-character title reaches 73 and is rejected.
        suffix = " (#35)"
        title = "feat: " + "a" * (66 - len("feat: "))
        self.assertEqual(len(title), 66, "test fixture is stale")
        ok, out = run_quietly(check.check_pr_title, title, 35)
        self.assertTrue(ok, out)
        self.assertIn(f"Squash subject: {title}{suffix}", out)
        self.assertIn("Squash subject: OK", out)

        title = "feat: " + "a" * (67 - len("feat: "))
        ok, out = run_quietly(check.check_pr_title, title, 35)
        self.assertFalse(ok)
        self.assertIn("exceeds 72 characters (73)", out)

    def test_bare_title_failure_is_not_masked(self):
        # The bare-title verdict alone must be FAILED, and the squash subject
        # must not be reported OK for these titles either.
        for title in ("add fp8 dtype support", "Merge feat: x", " feat: x"):
            with self.subTest(title=title):
                ok, out = run_quietly(check.check_pr_title, title, 36)
                self.assertFalse(ok)
                self.assertIn("PR title: FAILED", out)
                self.assertNotIn("Squash subject: OK", out)

    def test_bare_title_only_without_number(self):
        ok, _ = run_quietly(check.check_pr_title, "feat: ok", None)
        self.assertTrue(ok)
        ok, out = run_quietly(check.check_pr_title, "feat: ok", None)
        self.assertNotIn("Squash subject", out)


if __name__ == "__main__":
    unittest.main()
