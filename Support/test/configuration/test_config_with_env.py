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

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(tempfile.mkdtemp())
HOME = ROOT / "home/user"


class ConfigWithEnvTest(unittest.TestCase):
    """Tests for configuration with environment variables."""

    tool_path: Path = Path("Support/test/configuration/env_test_cpp")

    def setUp(self) -> None:
        """Set up test environment before each test."""
        # The path to the test tool is determined by Bazel's runfiles mechanism
        self.assertTrue(
            self.tool_path.exists(),
            f"Test tool not found at {self.tool_path.resolve()}",
        )

        # Ensure the test directory exists
        os.makedirs(HOME)

    def tearDown(self) -> None:
        """Clean up test environment after each test."""
        if os.path.exists(ROOT):
            shutil.rmtree(ROOT)

    def test_modular_home_precedence(self) -> None:
        """MODULAR_HOME takes precedence over all other config directories."""
        self._run_env_test(
            env_vars={
                "MODULAR_HOME": f"{HOME}/modular",
                "MODULAR_DERIVED_PATH": f"{HOME}/derived",
                "TEST_TMPDIR": f"{HOME}/test-tmpdir",
                "XDG_CONFIG_HOME": f"{HOME}/test-config",
                "XDG_DATA_HOME": f"{HOME}/test-data",
                "XDG_CACHE_HOME": f"{HOME}/test-data",
            },
            expected={
                "ModularConfigFolderPath": f"{HOME}/modular",
                "ModularDataFolderPath": f"{HOME}/modular",
                "ModularCacheFolderPath": f"{HOME}/modular/cache",
            },
        )

    def test_modular_derived_path_precedence(self) -> None:
        """MODULAR_DERIVED_PATH takes precedence over other config directories."""
        self._run_env_test(
            env_vars={
                "MODULAR_DERIVED_PATH": f"{HOME}/modular",
                "TEST_TMPDIR": f"{HOME}/test-tmpdir",
                "XDG_CONFIG_HOME": f"{HOME}/test-config",
                "XDG_DATA_HOME": f"{HOME}/test-data",
                "XDG_CACHE_HOME": f"{HOME}/test-data",
            },
            expected={
                "ModularConfigFolderPath": f"{HOME}/modular",
                "ModularDataFolderPath": f"{HOME}/modular",
                "ModularCacheFolderPath": f"{HOME}/modular/cache",
            },
        )

    def test_test_tmpdir_precedence(self) -> None:
        """TEST_TMPDIR takes precedence over other config directories."""
        self._run_env_test(
            env_vars={
                "TEST_TMPDIR": f"{HOME}",
                "XDG_CONFIG_HOME": f"{HOME}/test-config",
                "XDG_DATA_HOME": f"{HOME}/test-data",
                "XDG_CACHE_HOME": f"{HOME}/test-data",
            },
            expected={
                "ModularConfigFolderPath": f"{HOME}/.modular",
                "ModularDataFolderPath": f"{HOME}/.modular",
                "ModularCacheFolderPath": f"{HOME}/.modular",
            },
        )

    def test_existing_modular_directory_precedence(self) -> None:
        """Existing ~/.modular directory takes precedence over XDG variables."""
        self._run_env_test(
            env_vars={
                "XDG_CONFIG_HOME": f"{HOME}/test-config",
                "XDG_DATA_HOME": f"{HOME}/test-data",
                "XDG_CACHE_HOME": f"{HOME}/test-data",
            },
            expected={
                "ModularConfigFolderPath": f"{HOME}/.modular",
                "ModularDataFolderPath": f"{HOME}/.modular",
                "ModularCacheFolderPath": f"{HOME}/.modular",
            },
            dirs=[f"{HOME}/.modular"],
        )

    def test_default_xdg_base_directories(self) -> None:
        """Default XDG base directories are used when environment is empty."""
        self._run_env_test(
            env_vars={},
            expected={
                "ModularConfigFolderPath": f"{HOME}/.config/modular",
                "ModularDataFolderPath": f"{HOME}/.local/share/modular",
                "ModularCacheFolderPath": f"{HOME}/.cache/modular",
            },
        )

    def test_xdg_environment_variables(self) -> None:
        """XDG base directory environment variables are respected."""
        self._run_env_test(
            env_vars={
                "XDG_CONFIG_HOME": f"{HOME}/test-config",
                "XDG_DATA_HOME": f"{HOME}/test-data",
                "XDG_CACHE_HOME": f"{HOME}/test-cache",
            },
            expected={
                "ModularConfigFolderPath": f"{HOME}/test-config/modular",
                "ModularDataFolderPath": f"{HOME}/test-data/modular",
                "ModularCacheFolderPath": f"{HOME}/test-cache/modular",
            },
        )

    def _run_env_test(
        self,
        env_vars: dict[str, str],
        expected: dict[str, str],
        dirs: list[str] | None = None,
    ) -> None:
        """Helper method to run the env_test_cpp tool with given environment.

        Args:
            env_vars: Environment variables to set for the test.
            expected: Expected output key-value pairs from env_test_cpp.
            dirs: Optional list of directories to create before running test.
        """
        if dirs is None:
            dirs = []

        # Create the command to build, each key in the expected dict is spelled
        # the same as the command-line option to create options from the keys.
        command = [str(self.tool_path)] + [f"--{key}" for key in expected]

        # Create a copy of the current environment and update it.
        env = os.environ.copy()

        # Clear any env vars that can infulence test results
        env["HOME"] = str(HOME)
        for key in list(env.keys()):
            if key.startswith(
                (
                    "BAZEL_",
                    "GTEST_",
                    "MODULAR_",
                    "TEST_",
                    "XDG_",
                )
            ):
                del env[key]

        # Inject the test environment variables
        env.update(env_vars)

        # Create any directories required by the test case.
        for dir_path in dirs:
            if Path(dir_path).is_absolute():
                # Ensure absolute paths are subdirectories of the test environment
                self.assertTrue(
                    dir_path.startswith(str(ROOT)),
                    f"Directory path must be within {ROOT}: {dir_path}",
                )
            else:
                dir_path = str((ROOT / dir_path).absolute())
            os.makedirs(dir_path)

        # Run the test tool
        result = subprocess.run(
            command,
            env=env,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )

        # Parse and verify the output
        output = json.loads(result.stdout)
        self.assertEqual(
            expected,
            output,
            f"Environment variables: {env_vars}\n"
            f"Expected: {expected}\n"
            f"Actual: {output}",
        )
