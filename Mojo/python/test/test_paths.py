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

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from mojo.paths import _build_mojo_source_package


def test_build_mojo_source_package_path_is_user_specific() -> None:
    """The temp path must include the user ID to isolate per-user caches.

    Without user isolation, two OS users sharing /tmp will collide on
    /tmp/.modular/mojo_pkg/ — the second user cannot write into a
    directory owned by the first.
    """
    fake_src = Path("/fake/mojo/package")
    uid = os.getuid()

    with (
        patch("mojo.paths.is_mojo_source_package_path", return_value=True),
        patch("mojo.paths.subprocess_run_mojo", return_value=MagicMock()),
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("mojo.paths.tempfile.gettempdir", return_value=tmp_dir),
    ):
        result = _build_mojo_source_package(fake_src)

    assert f".modular_{uid}" in str(result), (
        f"Expected path to contain '.modular_{uid}', got: {result}"
    )
    assert "/mojo_pkg/" in str(result)
    assert result.name.endswith(".mojoc")


def test_build_mojo_source_package_no_shared_directory_collision() -> None:
    """Verify the path does NOT use a shared .modular/ directory.

    The old path /tmp/.modular/mojo_pkg/ is shared across all users.
    The fixed path must use a user-specific prefix instead.
    """
    fake_src = Path("/fake/mojo/package")

    with (
        patch("mojo.paths.is_mojo_source_package_path", return_value=True),
        patch("mojo.paths.subprocess_run_mojo", return_value=MagicMock()),
        tempfile.TemporaryDirectory() as tmp_dir,
        patch("mojo.paths.tempfile.gettempdir", return_value=tmp_dir),
    ):
        result = _build_mojo_source_package(fake_src)

    # The path must NOT contain the old shared ".modular/mojo_pkg" pattern
    relative = str(result.relative_to(tmp_dir))
    assert not relative.startswith(".modular/mojo_pkg"), (
        f"Path still uses shared .modular/ directory: {result}"
    )
