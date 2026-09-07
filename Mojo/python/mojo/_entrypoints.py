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

"""Contains entrypoints for the Mojo wheel"""

import os
import sys

from ._package_root import get_package_root
from .run import _mojo_env


def _entrypoint(file: str) -> None:
    root = get_package_root()
    assert root
    env = _mojo_env()

    os.execve(root / "bin" / file, sys.argv, env)


def exec_mojo() -> None:
    env = _mojo_env()

    os.execve(env["MODULAR_MOJO_MAX_DRIVER_PATH"], sys.argv, env)


def exec_lld() -> None:
    _entrypoint("lld")


def exec_modular_crashpad_handler() -> None:
    _entrypoint("modular-crashpad-handler")
