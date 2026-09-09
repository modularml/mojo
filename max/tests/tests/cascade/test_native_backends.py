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
"""Guards that aiohttp's transport dependencies use their native C backends.

``aiohttp`` (the Cascade HTTP runtime transport) leans on ``multidict`` for
every HTTP header map and ``frozenlist`` for its signal callbacks. Both ship a
compiled C extension plus a slow pure-Python fallback that the wheel selects
only when no native build matches the interpreter. Older pins (``multidict``
6.0.5, ``frozenlist`` 1.4.1) predated CPython 3.13 native wheels, so on the
default 3.13 toolchain the resolver picked the ``py3-none-any`` fallback and
aiohttp silently ran the pure-Python path. This test fails if that regresses.

See SERVSYS-1295.
"""

from __future__ import annotations

import importlib
import importlib.util

import pytest


@pytest.mark.parametrize(
    ("package", "c_extension"),
    [
        ("multidict", "multidict._multidict"),
        ("frozenlist", "frozenlist._frozenlist"),
    ],
)
def test_uses_native_c_extension(package: str, c_extension: str) -> None:
    """The package resolves its compiled extension, not the pure fallback."""
    # Resolve dynamically: these are transitive deps of aiohttp, not direct
    # imports of this test, so the static pydeps scan should not require them.
    importlib.import_module(package)
    spec = importlib.util.find_spec(c_extension)
    origin = spec.origin if spec else None
    assert origin is not None and origin.endswith((".so", ".pyd")), (
        f"{package!r} is using its pure-Python fallback instead of the native "
        f"C extension {c_extension!r} (found origin: {origin!r}). aiohttp "
        f"transport, used by Cascade's HTTP runtime, depends on the native "
        f"backend for performance. This usually means the locked version "
        f"predates native wheels for the current interpreter -- bump it via "
        f"`uv lock -P {package}`. See SERVSYS-1295."
    )
