# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# ===----------------------------------------------------------------------=== #
#
# File originates from:
#   Repo:   git@github.com:psf/black.git
#   Commit: d4a85643a465f5fae2113d07d22d021d4af4795a
#   Path:   scripts/check_pre_commit_rev_in_example.py
#
# ===----------------------------------------------------------------------=== #

"""
Check that the rev value in the example pre-commit configuration matches
the latest version of Black. This saves us from forgetting to update that
during the release process.

Why can't we just use `rev: stable` and call it a day? Well pre-commit
won't auto update the hook as you may expect (and for good reasons, some
technical and some pragmatic). Encouraging bad practice is also just
not ideal. xref: https://github.com/psf/black/issues/420
"""

import os
import sys

import commonmark
import yaml
from bs4 import BeautifulSoup


def main(changes: str, source_version_control: str) -> None:
    changes_html = commonmark.commonmark(changes)
    changes_soup = BeautifulSoup(changes_html, "html.parser")
    headers = changes_soup.find_all("h2")
    latest_tag, *_ = [
        header.string for header in headers if header.string != "Unreleased"
    ]

    source_version_control_html = commonmark.commonmark(source_version_control)
    source_version_control_soup = BeautifulSoup(
        source_version_control_html, "html.parser"
    )
    pre_commit_repos = yaml.safe_load(
        source_version_control_soup.find(class_="language-yaml").string
    )["repos"]

    for repo in pre_commit_repos:
        pre_commit_rev = repo["rev"]
        if pre_commit_rev != latest_tag:
            print(
                "Please set the rev in ``source_version_control.md`` to be the"
                f" latest one.\nExpected {latest_tag}, got {pre_commit_rev}.\n"
            )
            sys.exit(1)


if __name__ == "__main__":
    with open("CHANGES.md", encoding="utf-8") as fd:
        changes = fd.read()
    with open(
        os.path.join("docs", "integrations", "source_version_control.md"),
        encoding="utf-8",
    ) as fd:
        source_version_control = fd.read()
    main(changes, source_version_control)
