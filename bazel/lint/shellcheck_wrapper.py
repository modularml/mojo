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
import platform
import sys
from pathlib import Path

from lint_helpers import (
    get_all_files,
    get_changed_files,
    is_fast,
)


def shellcheck_path(runfiles_root: Path) -> Path:
    """Return the shellcheck binary path in runfiles for the host platform."""
    if sys.platform == "darwin":
        suffix = "darwin.aarch64"
    elif platform.machine() == "aarch64":
        suffix = "linux.aarch64"
    else:
        suffix = "linux.x86_64"
    return runfiles_root / f"+http_archive+shellcheck-{suffix}" / "shellcheck"


# TODO: Improve heuristic for finding new versions of these
# https://github.com/koalaman/shellcheck/issues/143
NO_EXTENSION_FILES = {
    "bazelw",
    "oss/modular/tools/bazel",
    "tools/bazel",
    "utils/benchmarking/tmp-tuning-utils/bbench",
    "utils/benchmarking/tmp-tuning-utils/nohup-kbench",
    "utils/packaging/conda/entrypoints/lldb-dap",
}


# If any of these files change, run shellcheck over everything in fast mode.
CONFIG_FILES = {
    ".shellcheckrc",  # Internally this is a symlink, so it doesn't show up in ls-files
    "oss/modular/.shellcheckrc",  # But this does
    "oss/modular/bazel/common.MODULE.bazel",
    "bazel/common.MODULE.bazel",
}


def _is_shell(fname: str) -> bool:
    return fname.endswith(".sh") or fname in NO_EXTENSION_FILES


def main(runfiles_root: Path) -> None:
    shellcheck = shellcheck_path(runfiles_root)

    if is_fast():
        changed = get_changed_files()
        candidates = (
            get_all_files() if changed.intersection(CONFIG_FILES) else changed
        )
    else:
        candidates = get_all_files()

    files = [f for f in candidates if _is_shell(f)]
    if not files:
        return

    os.execv(shellcheck, [str(shellcheck), "--format=gcc", *files])


if __name__ == "__main__":
    # Capture the runfiles root before any chdir so the path stays valid.
    runfiles_root = Path.cwd().parent
    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)
    main(runfiles_root)
