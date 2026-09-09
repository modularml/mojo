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
import sys
from pathlib import Path

from lint_helpers import (
    get_changed_files,
    is_check,
    is_fast,
)
from mblack import patched_main


def _filter(fname: str) -> bool:
    return Path(fname).suffix == ".mojo"


def _config_file_changed(changed_files: set[str]) -> bool:
    return any(f.startswith("Mojo/tools/mblack/src") for f in changed_files)


def main() -> None:
    config = Path("pyproject.toml")
    if not config.is_file():
        sys.exit(f"mblack: config not found at {config.resolve()}")

    if is_fast():
        all_files = get_changed_files()

        if _config_file_changed(all_files):
            files = ["."]
        else:
            files = list(filter(_filter, all_files))

            if not files:
                # mblack errors if no paths are specified, so short circuit here
                return
    else:
        files = ["."]

    args = ["--quiet", "--config", str(config)]
    if is_check():
        args.extend(["--check", "--diff"])
    args.extend(files)

    sys.argv = sys.argv + args

    patched_main()


if __name__ == "__main__":
    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)

    main()
