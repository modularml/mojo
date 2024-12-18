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
config.name = "Mojo Standard Library Benchmarks"

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mojo"]

config.substitutions.insert(0, ("%mojo", "mojo"))
config.substitutions.insert(0, ("%mojo-no-debug", "mojo"))

# Internal testing configuration.  This environment variable
# is set by the internal `start-modular.sh` script.
if "_START_MODULAR_INCLUDED" in os.environ:
    # test_source_root: The root path where tests are located.
    config.test_source_root = Path(__file__).parent.resolve()

    # test_exec_root: The root path where tests should be run.
    config.test_exec_root = os.path.join(
        config.modular_obj_root, "open-source", "mojo", "stdlib", "benchmarks"
    )
else:
    # test_source_root: The root path where tests are located.
    config.test_source_root = Path(__file__).parent.resolve()

    repo_root = Path(__file__).parent.parent.parent

    # This is important since `benchmark` is closed source
    # still right now and is always used by the benchmarks.
    pre_built_packages_path = Path(
        os.environ.get(
            "MODULAR_MOJO_NIGHTLY_IMPORT_PATH",
            os.environ.get(
                "MODULAR_MOJO_IMPORT_PATH",
                repo_root / ".magic" / "envs" / "default" / "lib" / "mojo",
            ),
        )
    )

    # The `run-tests.sh` script creates the build directory for you.
    build_root = repo_root / "build"

    # The tests are executed inside this build directory to avoid
    # polluting the source tree.
    config.test_exec_root = (build_root / "stdlib" / "benchmarks").resolve()

    # Add both the open source, locally built `stdlib.mojopkg`
    # along with the closed source, pre-built packages shipped
    # with the Mojo SDK to the appropriate environment variables.
    # These environment variables are interpreted by the mojo parser
    # when resolving imports.
    joint_path = f"{build_root.resolve()},{pre_built_packages_path.resolve()}"
    os.environ["MODULAR_MOJO_NIGHTLY_IMPORT_PATH"] = joint_path
    os.environ["MODULAR_MOJO_MAX_NIGHTLY_IMPORT_PATH"] = joint_path
    os.environ["MODULAR_MOJO_IMPORT_PATH"] = joint_path
    os.environ["MODULAR_MOJO_MAX_IMPORT_PATH"] = joint_path

    # Pass through several environment variables
    # to the underlying subprocesses that run the tests.
    lit.llvm.initialize(lit_config, config)
    lit.llvm.llvm_config.with_system_environment(
        [
            "MODULAR_HOME",
            "MODULAR_MOJO_NIGHTLY_IMPORT_PATH",
            "MODULAR_MOJO_MAX_NIGHTLY_IMPORT_PATH",
            "PATH",
        ]
    )
