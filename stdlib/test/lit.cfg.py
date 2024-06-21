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

# Configuration file for the 'lit' test runner.

config.test_format = lit.formats.ShTest(True)

# name: The name of this test suite.
config.name = "Mojo Standard Library"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]


# Check if the `not` binary from LLVM is available.
def has_not():
    return shutil.which("not") is not None


if has_not():
    config.available_features.add("has_not")

# Insert at 0th position to intentionally override
# %mojo from the global utils/build/llvm-lit/lit.common.cfg.py
# (only matters internally).
# In the future, we can do other fancy things like with sanitizers
# and build type.
if bool(int(os.environ.get("MOJO_ENABLE_ASSERTIONS_IN_TESTS", 1))):
    base_mojo_command = "mojo -D MOJO_ENABLE_ASSERTIONS"
else:
    print("Running tests with assertions disabled.")
    base_mojo_command = "mojo"
config.substitutions.insert(0, ("%mojo", base_mojo_command))

# Mojo without assertions.  Only use this for known tests that do not work
# with assertions enabled.
config.substitutions.insert(1, ("%bare-mojo", "mojo"))

# This makes the OS name available for `REQUIRE` directives, e.g., `# REQUIRE: darwin`.
config.available_features.add(platform.system().lower())

# Internal testing configuration.  This environment variable
# is set by the internal `start-modular.sh` script.
if "_START_MODULAR_INCLUDED" in os.environ:
    # test_source_root: The root path where tests are located.
    config.test_source_root = os.path.dirname(__file__)

    # test_exec_root: The root path where tests should be run.
    config.test_exec_root = os.path.join(
        config.modular_obj_root, "mojo-stdlib", "test"
    )
# External, public Mojo testing configuration
else:
    # test_utils does not contain tests, just source code
    # that we run `mojo package` on to be used by other tests
    config.excludes = ["test_utils"]

    # test_source_root: The root path where tests are located.
    config.test_source_root = Path(__file__).parent.resolve()

    # The `run-tests.sh` script creates the build directory for you.
    build_root = Path(__file__).parent.parent.parent / "build"

    # The tests are executed inside this build directory to avoid
    # polluting the source tree.
    config.test_exec_root = build_root / "stdlib" / "test"

    # The `mojo` nightly compiler ships with its own `stdlib.mojopkg`. For the
    # open-source stdlib, we need to specify the paths to the just-built
    # `stdlib.mojopkg` and `test_utils.mojopkg`. Otherwise, without this, the
    # `mojo` compiler would use its own `stdlib.mojopkg` it ships with which is not
    # what we want. We override both the stable and nightly `mojo` import paths
    # here to support both versions of the compiler.
    os.environ["MODULAR_MOJO_IMPORT_PATH"] = str(build_root)
    os.environ["MODULAR_MOJO_NIGHTLY_IMPORT_PATH"] = str(build_root)

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
