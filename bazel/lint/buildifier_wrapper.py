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

import os
from collections.abc import Iterable
from pathlib import Path

from lint_helpers import (
    get_all_files,
    get_changed_files,
    is_check,
    is_fast,
)

_BUILDIFIER_IGNORES = (
    ".derived/*",
    ".git/*",
    "CloudInfra/go/pkg/*",
    "third-party/llvm-project/*",
    "oss/max-llm-book/.pixi/*",
    "oss/skills/port-an-llm-to-max/.pixi/*",
)

_CONFIG_FILES = (
    # Buildifier version
    "common.MODULE.bazel",
    # Config files
    "buildifier-tables-add.json",
    ".buildifier.json",
)


def _config_file_changed(changed_files: set[str]) -> bool:
    return any(f.endswith(_CONFIG_FILES) for f in changed_files)


_SUFFIXES = (
    ".bzl",
    ".bazel",
    "BUILD",
    ".sky",
)


def _filter_files(files: Iterable[str]) -> list[str]:
    def _filter_file(f: str) -> bool:
        return f.endswith(_SUFFIXES) and not f.startswith(_BUILDIFIER_IGNORES)

    return [f for f in files if _filter_file(f)]


def main(buildifier: str) -> None:
    files = None
    if is_fast():
        all_files = get_changed_files()

        if _config_file_changed(all_files):
            files = _filter_files(get_all_files())
        else:
            files = _filter_files(all_files)

            if not files:
                # buildifier reads from stdin if no files passed, early exit.
                return
    else:
        files = _filter_files(get_all_files())

    if is_check():
        args = ["-lint=warn", "-mode=diff"]
    else:
        args = ["-lint=fix", "-mode=fix"]

    os.execv(buildifier, [buildifier, *args, *files])


if __name__ == "__main__":
    buildifier = (
        Path(os.environ.get("RUNFILES_DIR") or Path.cwd().parent)
        / "buildifier_prebuilt+/buildifier/buildifier"
    )

    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)

    main(str(buildifier))
