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
#   Path:   scripts/check_version_in_basics_example.py
#
# ===----------------------------------------------------------------------=== #

"""
Check that the rev value in the example from ``the_basics.md`` matches
the latest version of Black. This saves us from forgetting to update that
during the release process.
"""

import os
import sys

import commonmark
from bs4 import BeautifulSoup


def main(changes: str, the_basics: str) -> None:
    changes_html = commonmark.commonmark(changes)
    changes_soup = BeautifulSoup(changes_html, "html.parser")
    headers = changes_soup.find_all("h2")
    tags = [
        header.string for header in headers if header.string != "Unreleased"
    ]
    latest_tag = tags[0]

    the_basics_html = commonmark.commonmark(the_basics)
    the_basics_soup = BeautifulSoup(the_basics_html, "html.parser")
    (version_example,) = [
        code_block.string
        for code_block in the_basics_soup.find_all(class_="language-console")
        if "$ black --version" in code_block.string
    ]

    for tag in tags:
        if tag in version_example and tag != latest_tag:
            print(
                "Please set the version in the ``black --version`` "
                "example from ``the_basics.md`` to be the latest one.\n"
                f"Expected {latest_tag}, got {tag}.\n"
            )
            sys.exit(1)


if __name__ == "__main__":
    with open("CHANGES.md", encoding="utf-8") as fd:
        changes = fd.read()
    with open(
        os.path.join("docs", "usage_and_configuration", "the_basics.md"),
        encoding="utf-8",
    ) as fd:
        the_basics = fd.read()
    main(changes, the_basics)
