# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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

# ruff: noqa

import os
from pathlib import Path

import lit.formats
import lit.llvm

config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite.
config.name = "Mojo Public Examples"

# suffixes: A list of file extensions to treat as test files.
# TODO: Enable notebooks
config.suffixes = [".mojo", ".🔥"]

config.excludes = [
    # No RUN: directive, just bare examples
    "hello_interop.mojo",
    "matmul.mojo",
] + [path.name for path in os.scandir("examples") if path.is_dir()]

# Have the examples run in the build directory.
# The `run-examples.sh` script creates the build directory.
build_root = Path.cwd().parent / "build"

# Execute the examples inside this part of the build
# directory to avoid polluting the source tree.
config.test_exec_root = build_root / "examples"

# test_source_root: The root path where tests are located.
config.test_source_root = Path(__file__).parent.resolve()

# Substitute %mojo for just `mojo` itself.
config.substitutions.insert(0, ("%mojo", "mojo"))

pre_built_packages_path = os.environ.get(
    "MODULAR_MOJO_NIGHTLY_IMPORT_PATH",
    Path(os.environ["MODULAR_HOME"])
    / "pkg"
    / "packages.modular.com_nightly_mojo"
    / "lib"
    / "mojo",
)

os.environ["MODULAR_MOJO_NIGHTLY_IMPORT_PATH"] = (
    f"{build_root},{pre_built_packages_path}"
)

# Pass through several environment variables
# to the underlying subprocesses that run the tests.
lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.with_system_environment(
    [
        "MODULAR_HOME",
        "MODULAR_MOJO_NIGHTLY_IMPORT_PATH",
        "PATH",
    ]
)
