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
    get_changed_files,
    is_check,
    is_fast,
)


def _rumdl_suffix() -> str:
    if sys.platform == "darwin":
        return "aarch64-apple-darwin"
    elif platform.machine() == "aarch64":
        return "aarch64-unknown-linux-musl"
    else:
        return "x86_64-unknown-linux-musl"


def main(runfiles_root: Path) -> int:
    rumdl = str(runfiles_root / f"+http_archive+rumdl-{_rumdl_suffix()}/rumdl")

    config = "oss/modular/bazel/lint/.rumdl.toml"
    if not Path(config).exists():
        config = "bazel/lint/.rumdl.toml"

    files: list[str] = ["."]
    if is_fast():
        changed = get_changed_files()
        if not any(
            f.endswith(("common.MODULE.bazel", ".rumdl.toml")) for f in changed
        ):
            files = [f for f in changed if f.endswith((".md", ".mdx"))]
            if not files:
                return 0

    cmd = "check" if is_check() else "fmt"

    return os.execv(rumdl, [rumdl, cmd, *files, "--config", config])


if __name__ == "__main__":
    # `Path.cwd().parent` works when bazel launches us directly (the py_binary
    # launcher cd's into runfiles/_main first), but breaks when we're invoked
    # as a tool from a genrule. `RUNFILES_DIR` is set in both cases.
    runfiles_root = Path(os.environ.get("RUNFILES_DIR") or Path.cwd().parent)
    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)
    sys.exit(main(runfiles_root))
