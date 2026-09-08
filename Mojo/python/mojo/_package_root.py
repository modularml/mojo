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

import logging
import os
from pathlib import Path


def get_package_root() -> Path | None:
    """Returns the package root of this installation, or None if running in a Bazel environment"""
    # lib/python3.12/site-packages/mojo/_package_root.py
    # ->
    # lib/python3.12/site-packages
    site_packages_root = Path(__file__).parent.parent

    # Look for bin/mojo in the environment
    # This is currently supposed to work for two cases:
    # - Set of Conda packages.
    # lib/python3.12/site-packages
    # ->
    # bin/mojo
    conda_root = site_packages_root.parent.parent.parent
    # - Monolithic Pip wheel installation.
    # lib/python3.12/site-packages
    # ->
    # lib/python3.12/site-packages/modular/bin/mojo
    wheel_root = site_packages_root / "modular"

    # Make sure we check the wheel root first!
    # conda_root / bin / mojo should exist in both cases here, but when using
    # wheels this is just a generated wrapper script, which we shouldn't call
    # from here since we'd end up in an infinite loop.
    for root in (wheel_root, conda_root):
        if (root / "bin/mojo").exists():
            logging.debug(f"Located Modular SDK assets at {root}")
            return root

    # If we're in a development venv we don't have the full wheel layout, but
    # this might work depending on what files are being looked up and what the
    # venv contains.
    if venv_root := os.environ.get("VIRTUAL_ENV"):
        return Path(venv_root)
    elif (
        "MODULAR_DERIVED_PATH" in os.environ
        or "BUILD_WORKSPACE_DIRECTORY" in os.environ
        or "BAZEL_TEST" in os.environ
    ):
        # We're running in a Modular internal Bazel test, so let Bazel handle
        # the environment variables.
        return None
    else:
        raise RuntimeError(
            "Unable to locate Modular SDK library assets root directory."
        )
