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

from __future__ import annotations

import hashlib
import logging
import os
import shlex
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from .run import subprocess_run_mojo


class Error(Exception):
    """Base error for mojo paths."""


@dataclass
class MojoCompilationError(Error):
    """Error encountered compiling a Mojo source package."""

    path: Path
    command: list[str]
    stdout: str
    stderr: str

    @staticmethod
    def from_subprocess_error(
        path: Path,
        args: list[str],
        err: subprocess.CalledProcessError,
    ) -> MojoCompilationError:
        return MojoCompilationError(
            path, args, err.stdout.decode(), err.stderr.decode()
        )

    def __str__(self) -> str:
        command = shlex.join(self.command)
        return f"Error compiling Mojo at {self.path}. Command: {command}\n\n{self.stderr}"


@dataclass
class MojoModulePath:
    """Represents a path to the root file of a Mojo module on the file system."""

    path: Path
    """Mojo source file that is the root of the module. Either an `__init__` file
    or a single-file module. """

    is_package: bool


def is_mojo_source_package_path(path: Path) -> bool:
    """Returns True if the given path is a Mojo package source directory.

    A Mojo package source directory is a directory that contains an
    `__init__.mojo` file.

    Args:
        path: The path to check

    Returns:
        bool: True if the path is a Mojo source package directory
    """
    return _mojo_source_package_root_file(path) is not None


def find_mojo_module_in_dir(
    dir_path: Path, module_name: str
) -> MojoModulePath | None:
    """Searches a directory for Mojo package or single file module.

    Returns:
        A `MojoModulePath` if found, otherwise None.
    """
    # Check for package first: <dir_path>/<module_name>/__init__.mojo
    if init_file_path := _mojo_source_package_root_file(dir_path / module_name):
        return MojoModulePath(init_file_path, True)

    # If not a package, check for single file: <dir_path>/<module_name>.mojo
    for ext in ["mojo"]:
        potential_file = dir_path / f"{module_name}.{ext}"
        if potential_file.is_file():
            # Found single file.
            return MojoModulePath(potential_file, False)

    # Not found in this directory
    return None


def _mojo_source_package_root_file(path: Path) -> Path | None:
    """Returns the path to the `__init__.mojo` package root
    file if this is a Mojo source package directory, otherwise None.

    A Mojo package source directory is a directory that contains an
    `__init__.mojo` file.

    Args:
        path: The path to check

    Returns:
        Path: Path to the root module file if the path is a Mojo source package
          directory
    """
    if not path.is_dir():
        return None

    init_mojo = path / "__init__.mojo"

    if init_mojo.is_file():
        return init_mojo
    else:
        return None


def is_mojo_binary_package_path(path: Path) -> bool:
    """Returns True if the given path is a Mojo binary package file, i.e.
    a file ending in ".mojoc".
    """

    if not path.is_file():
        return False

    return path.suffix == ".mojoc"


def _build_mojo_source_package(path: Path) -> Path:
    assert is_mojo_source_package_path(path)

    # Create a deterministic path in the temp directory based on the source path.
    # Include the user ID in the path to avoid permission conflicts
    # when multiple OS users share the same temp directory (Linux).
    path_hash = hashlib.md5(str(path.absolute()).encode()).hexdigest()
    tmp_path = (
        Path(tempfile.gettempdir())
        / f".modular_{os.getuid()}"
        / "mojo_pkg"
        / f"mojo_pkg_{path_hash}.mojoc"
    )

    # Ensure parent directories exist
    tmp_path.parent.mkdir(parents=True, exist_ok=True)

    args = [
        # `mojo` command argument is impliict.
        "precompile",
        str(path),
        "-o",
        str(tmp_path),
    ]

    try:
        subprocess_run_mojo(args, capture_output=True, check=True)
    except subprocess.CalledProcessError as e:
        error = MojoCompilationError.from_subprocess_error(path, args, e)
        logging.error(str(error))
        raise error from e

    return tmp_path
