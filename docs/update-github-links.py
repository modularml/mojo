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

"""Update GitHub branch references in docs.

Example:
cd oss/modular/docs
python3 update-github-links.py . max/v26.3
python3 update-github-links.py ../Mojo/docs mojo/v1.0.0b1
"""

import sys
from pathlib import Path

SELF = Path(__file__).resolve()
PATTERNS = {
    "modular/modular/tree/main": "modular/modular/tree/",
    "modular/modular/blob/main": "modular/modular/blob/",
}


def replace_branch_refs(docs_dir: str, branch_name: str) -> None:
    """Replace all ``modular/modular/{tree,blob}/main`` refs with the given branch."""
    replacements = {
        old: prefix + branch_name for old, prefix in PATTERNS.items()
    }

    for filepath in Path(docs_dir).rglob("*"):
        if not filepath.is_file() or filepath.resolve() == SELF:
            continue
        try:
            content = filepath.read_text(encoding="utf-8")
        except (UnicodeDecodeError, PermissionError):
            continue
        if not any(old in content for old in replacements):
            continue
        updated = content
        for old, new in replacements.items():
            updated = updated.replace(old, new)
        filepath.write_text(updated, encoding="utf-8")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            "Usage: python3 update-github-links.py <docs_directory> <branch_name>"
        )
        sys.exit(1)

    docs_dir = sys.argv[1]
    if not Path(docs_dir).is_dir():
        print(f"Error: '{docs_dir}' is not a valid directory")
        sys.exit(1)

    replace_branch_refs(docs_dir, sys.argv[2])
