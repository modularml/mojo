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

# Validates that all actions use a pinned sha. Tags like @v4 can change and
# can introduce vulnerabilities.

import os
import re
import subprocess
import sys
from typing import Any

import yaml

_VALID_SHA = re.compile("[0-9a-f]{40}")


def _get_files() -> list[str]:
    all_files = (
        subprocess.check_output(["git", "ls-files"]).decode().splitlines()
    )

    action_files = [
        file
        for file in all_files
        if (".github/workflows/" in file or ".github/actions/" in file)
        and (file.endswith((".yaml", ".yml")))
    ]

    return action_files


def _validate_file(file: str) -> int:
    with open(file) as f:
        content = yaml.safe_load(f)

    steps: list[dict[str, Any]] = []
    if "jobs" in content:
        # Workflow file
        for job in content["jobs"].values():
            if "steps" in job:
                steps += job["steps"]
    elif "runs" in content:
        # Action file
        steps = content["runs"]["steps"]
    else:
        raise KeyError(f"Neither 'runs' nor 'jobs' found in file {file}")

    result = 0
    for step in steps:
        if "uses" in step:
            use: str = step["uses"]

            if use.startswith("./"):
                # Using a local action, no pin
                continue

            pin = use.split("@")[1]

            if not _VALID_SHA.match(pin):
                print(f"In {file}: {use} does not pin to a specific commit.")
                result = 1

    return result


def main() -> int:
    files = _get_files()

    result = 0
    for file in files:
        if _validate_file(file) != 0:
            result = 1

    return result


if __name__ == "__main__":
    if path := os.getenv("BUILD_WORKSPACE_DIRECTORY"):
        os.chdir(path)
    sys.exit(main())
