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
import subprocess
import sys
from pathlib import Path

from lint_helpers import (
    get_changed_files,
    is_check,
    is_fast,
)


def _ruff_suffix() -> str:
    if sys.platform == "darwin":
        return "aarch64-apple-darwin"
    elif platform.machine() == "aarch64":
        return "aarch64-unknown-linux-gnu"
    else:
        return "x86_64-unknown-linux-gnu"


def main(runfiles_root: Path) -> int:
    ruff = str(runfiles_root / f"+http_archive+ruff-{_ruff_suffix()}" / "ruff")

    # Empty file list lets ruff auto-discover everything from the workspace root.
    files: list[str] = []
    if is_fast():
        changed = get_changed_files()
        if any(
            f.endswith(("common.MODULE.bazel", "pyproject.toml"))
            for f in changed
        ):
            files = []
        else:
            files = [f for f in changed if f.endswith((".py", ".pyi", ".md"))]
            if not files:
                return 0

    if is_check():
        format_args = ["--check", "--diff"]
        check_args = []
    else:
        format_args = []
        check_args = ["--fix"]

    # --quiet can potentially suppress error diagnostics in `ruff format`,
    # so only output that if it fails. We want to suppress success summaries
    # ("All checks passed!", "N files left unchanged") while still emitting
    # real lint errors / diffs.
    fmt_result = subprocess.run(
        [
            ruff,
            "format",
            # Ignore internal-only broken symlink. This seems to be the only way to ignore it.
            "--exclude=oss/modular/oss.AGENTS.md",
            "--color=always",
            *format_args,
            *files,
        ],
        check=False,
        capture_output=True,
    )
    if fmt_result.returncode != 0:
        print(fmt_result.stdout.decode())
        print(fmt_result.stderr.decode())
    check_result = subprocess.run(
        [
            ruff,
            "check",
            "--color=always",
            *check_args,
            *files,
        ],
        check=False,
        capture_output=True,
    )
    if check_result.returncode != 0:
        print(check_result.stdout.decode())
        print(check_result.stderr.decode())

    if fmt_result.returncode != 0:
        return fmt_result.returncode
    return check_result.returncode


if __name__ == "__main__":
    # `Path.cwd().parent` works when bazel launches us directly (the py_binary
    # launcher cd's into runfiles/_main first), but breaks when we're invoked
    # as a tool from a genrule. `RUNFILES_DIR` is set in both cases.
    runfiles_root = Path(os.environ.get("RUNFILES_DIR") or Path.cwd().parent)
    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)
    sys.exit(main(runfiles_root))
