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
import shutil
import subprocess
from typing import Any

from ._package_root import get_package_root


def _sdk_default_env() -> dict[str, str]:
    root = get_package_root()

    # Running in Bazel
    if not root:
        return {}

    bin = root / "bin"
    lib = root / "lib"

    # Special case for wheel entrypoint - in
    # the venv it is actually put in the root `bin`.
    # lib/python3.13/site-packages/max/
    # ->
    # bin/mblack
    extra_env = {}
    maybe_mblack_path = root.parent.parent.parent.parent / "bin/mblack"
    if maybe_mblack_path.exists():
        extra_env["MODULAR_MOJO_MAX_MBLACK_PATH"] = str(maybe_mblack_path)

    return {
        "MODULAR_MAX_PACKAGE_ROOT": str(root),
        "MODULAR_MOJO_MAX_PACKAGE_ROOT": str(root),
        "MODULAR_MOJO_MAX_DRIVER_PATH": str(bin / "mojo"),
        "MODULAR_MOJO_MAX_IMPORT_PATH": str(lib / "mojo"),
    } | extra_env


def _mojo_env() -> dict[str, str]:
    """Returns an environment variable set that uses the Mojo SDK environment
    paths by default, but with overrides from the ambient OS environment."""

    return _sdk_default_env() | dict(os.environ)


def subprocess_run_mojo(  # noqa: ANN201
    mojo_args: list[str],
    **kwargs: Any,
):
    """Launches the bundled `mojo` in a subprocess, configured to use the
    `mojo` assets in the `max` package.

    Arguments:
        mojo_args: Arguments supplied to the `mojo` command.
        kwargs: Additional arguments to pass to `subprocess.run()`
    """

    env = _mojo_env()
    mojo = env.get("MODULAR_MOJO_MAX_DRIVER_PATH") or shutil.which("mojo")
    if not mojo or not os.path.exists(mojo):
        raise RuntimeError("error: Could not find `mojo` executable")

    return subprocess.run(
        # Combine the `mojo` executable path with the provided argument list.
        [mojo] + mojo_args,
        env=env,
        **kwargs,
    )
