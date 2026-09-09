#!/usr/bin/env python3
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

import logging
import shutil
import subprocess
import sys
from pathlib import Path


def main() -> None:
    extension_dir = Path(__file__).parent / "extension"
    modular_dir = Path(__file__).parent.parent.parent.parent

    # Copy over the license file so that we can build the extension.
    shutil.copy(modular_dir / "LICENSE.md", extension_dir)

    # Install the python package.
    try:
        subprocess.check_call(
            ["python3", "-m", "pip", "install", "-e", str(extension_dir)],
        )
    except:
        logging.critical("Failed to install the mojo_jupyter package.")
        sys.exit(1)

    # Build the type script extension.
    try:
        subprocess.check_call(
            ["jlpm", "run", "build"],
            cwd=extension_dir,
        )
    except:
        logging.critical(
            "Failed to build the mojo_jupyter typescript extension."
        )
        sys.exit(1)

    # Link to the extension directory.
    try:
        subprocess.check_call(
            [
                "jupyter",
                "labextension",
                "develop",
                str(extension_dir),
                "--overwrite",
            ],
            cwd=extension_dir,
        )
    except:
        logging.critical("Failed to install the mojo_jupyter extension.")
        sys.exit(1)


if __name__ == "__main__":
    main()
