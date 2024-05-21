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

import os
import platform
import shutil
from pathlib import Path

import lit.formats
import lit.llvm

config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite.
config.name = "Mojo Standard Library"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

# test_utils does not contain tests, just source code
# that we run `mojo package` on to be used by other tests
config.excludes = ["test_utils"]

# test_source_root: The root path where tests are located.
config.test_source_root = Path(__file__).parent.resolve()

# The `run-tests.sh` script creates the build directory for you.
build_root = Path(__file__).parent.parent.parent / "build"

# The tests are executed inside this build directory to avoid
# polluting the source tree.
config.test_exec_root = build_root / "stdlib"

# This makes the OS name available for `REQUIRE` directives, e.g., `# REQUIRE: darwin`.
config.available_features.add(platform.system().lower())

# Substitute %mojo for just `mojo` itself
# since we're not supporting `--sanitize` initially
# to allow running the tests with LLVM sanitizers.
config.substitutions.insert(0, ("%mojo", "mojo -D MOJO_ENABLE_ASSERTIONS"))
config.substitutions.insert(1, ("%bare-mojo", "mojo"))

# The `mojo` nightly compiler ships with its own `stdlib.mojopkg`. For the
# open-source stdlib, we need to specify the paths to the just-built
# `stdlib.mojopkg` and `test_utils.mojopkg`. Otherwise, without this, the
# `mojo` compiler would use its own `stdlib.mojopkg` it ships with which is not
# what we want. We override both the stable and nightly `mojo` import paths
# here to support both versions of the compiler.
os.environ["MODULAR_MOJO_IMPORT_PATH"] = str(build_root)
os.environ["MODULAR_MOJO_NIGHTLY_IMPORT_PATH"] = str(build_root)


# Check if the `not` binary from LLVM is available.
def has_not():
    return shutil.which("not") is not None


if has_not():
    config.available_features.add("has_not")

# Pass through several environment variables
# to the underlying subprocesses that run the tests.
lit.llvm.initialize(lit_config, config)
lit.llvm.llvm_config.with_system_environment(
    [
        "MODULAR_HOME",
        "MODULAR_MOJO_IMPORT_PATH",
        "MODULAR_MOJO_NIGHTLY_IMPORT_PATH",
    ]
)
