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

"""Post-processing for mojo-only docs (mojodocs tarball build)."""

import os
import re
import sys


def replace_relative_paths(docs_path: str, file_path: str) -> None:
    path = docs_path + file_path
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"post-process-mojodocs: expected tree at {path!r} (missing or not a directory)"
        )

    md_count = 0
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".md"):
                md_count += 1
                file_path_joined = os.path.join(root, file)
                with open(file_path_joined) as f:
                    content = f.read()
                # Replace relative link paths "](./"
                new_path = root.replace(docs_path, "")
                new_content = re.sub(r"\]\(\./", f"]({new_path}/", content)
                with open(file_path_joined, "w") as f:
                    f.write(new_content)

    if md_count == 0:
        raise RuntimeError(
            f"post-process-mojodocs: no .md files under {path!r} (nothing to do)"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 post-process-mojodocs.py <directory>")
        sys.exit(1)

    replace_relative_paths(sys.argv[1], "/docs/std")
