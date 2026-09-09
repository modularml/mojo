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

"""Test file URI support in MAX serve."""

from io import BytesIO
from unittest.mock import NonCallableMock

import pytest
from max.serve.config import Settings
from max.serve.router._image_resolution import resolve_image_from_url
from PIL import Image
from pydantic import AnyUrl


def create_test_image_bytes() -> bytes:
    """Create a minimal test image and return it as JPEG bytes."""
    img = Image.new("RGB", (10, 10), color="blue")
    buffer = BytesIO()
    img.save(buffer, format="JPEG")
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_file_uri_absolute_path(tmp_path) -> None:  # noqa: ANN001
    """Test resolving absolute file URIs."""
    test_image = tmp_path / "test.jpg"
    test_data = create_test_image_bytes()
    test_image.write_bytes(test_data)

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    result = await resolve_image_from_url(
        AnyUrl(f"file://{test_image}"), settings
    )
    assert result == test_data


@pytest.mark.asyncio
async def test_file_uri_not_found(tmp_path) -> None:  # noqa: ANN001
    """Test error for missing files."""
    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="not found"):
        await resolve_image_from_url(
            AnyUrl(f"file://{tmp_path}/missing.jpg"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_path_traversal_blocked(tmp_path) -> None:  # noqa: ANN001
    """Test path traversal protection."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    # Create a file outside the allowed directory
    secret_file = tmp_path / "secret.jpg"
    secret_file.write_bytes(create_test_image_bytes())

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(allowed)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="forbidden"):
        await resolve_image_from_url(
            AnyUrl(f"file://{allowed}/../secret.jpg"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_directory_blocked(tmp_path) -> None:  # noqa: ANN001
    """Test directories cannot be accessed."""
    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="directory"):
        await resolve_image_from_url(AnyUrl(f"file://{tmp_path}"), settings)


@pytest.mark.asyncio
async def test_file_uri_outside_root_missing_is_forbidden(
    tmp_path,  # noqa: ANN001
) -> None:
    """Containment is checked before existence.

    A nonexistent path outside the allowed roots must be rejected as
    ``forbidden`` rather than ``not found`` — the ordering must not let the
    server stat attacker-chosen paths and leak their existence.
    """
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(allowed)]
    settings.max_local_image_bytes = 20_000_000

    missing_outside = tmp_path / "missing.jpg"
    with pytest.raises(ValueError, match="forbidden"):
        await resolve_image_from_url(
            AnyUrl(f"file://{missing_outside}"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_outside_root_directory_is_forbidden(
    tmp_path,  # noqa: ANN001
) -> None:
    """Containment is checked before the directory type probe.

    A directory outside the allowed roots must be rejected as ``forbidden``
    rather than leaking that the path is a directory.
    """
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside_dir = tmp_path / "outside_dir"
    outside_dir.mkdir()

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(allowed)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="forbidden"):
        await resolve_image_from_url(AnyUrl(f"file://{outside_dir}"), settings)


@pytest.mark.asyncio
async def test_file_uri_size_limit(tmp_path) -> None:  # noqa: ANN001
    """Test file size limit enforcement."""
    large_file = tmp_path / "large.jpg"
    large_file.write_bytes(b"x" * (21 * 1024 * 1024))  # 21MB

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20 * 1024 * 1024  # 20MB

    with pytest.raises(ValueError, match=r"exceeds.*size limit"):
        await resolve_image_from_url(AnyUrl(f"file://{large_file}"), settings)


@pytest.mark.asyncio
async def test_file_uri_url_encoded_path(tmp_path) -> None:  # noqa: ANN001
    """Test URL-encoded file paths."""
    test_image = tmp_path / "test image.jpg"
    test_data = create_test_image_bytes()
    test_image.write_bytes(test_data)

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    encoded_path = str(test_image).replace(" ", "%20")
    result = await resolve_image_from_url(
        AnyUrl(f"file://{encoded_path}"), settings
    )
    assert result == test_data


@pytest.mark.asyncio
async def test_file_uri_localhost_host(tmp_path) -> None:  # noqa: ANN001
    """Test file URI with localhost host."""
    test_image = tmp_path / "test.jpg"
    test_data = create_test_image_bytes()
    test_image.write_bytes(test_data)

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    result = await resolve_image_from_url(
        AnyUrl(f"file://localhost{test_image}"), settings
    )
    assert result == test_data


@pytest.mark.asyncio
async def test_file_uri_remote_host_rejected(tmp_path) -> None:  # noqa: ANN001
    """Test remote hosts are rejected."""
    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(tmp_path)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="remote host"):
        await resolve_image_from_url(
            AnyUrl("file://example.com/image.jpg"), settings
        )


@pytest.mark.asyncio
async def test_file_uri_no_allowed_roots(tmp_path) -> None:  # noqa: ANN001
    """Test access blocked with no allowed roots."""
    test_file = tmp_path / "test.jpg"
    test_file.write_bytes(create_test_image_bytes())

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = []
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="no allowed roots"):
        await resolve_image_from_url(AnyUrl(f"file://{test_file}"), settings)


@pytest.mark.asyncio
async def test_file_uri_symlink_escape(tmp_path) -> None:  # noqa: ANN001
    """Test symlinks cannot escape allowed roots."""
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    outside = tmp_path / "outside.jpg"
    outside.write_bytes(create_test_image_bytes())

    symlink = allowed / "link.jpg"
    try:
        symlink.symlink_to(outside)
    except OSError:
        pytest.skip("Symlinks not supported")

    settings = NonCallableMock(spec=Settings)
    settings.allowed_image_roots = [str(allowed)]
    settings.max_local_image_bytes = 20_000_000

    with pytest.raises(ValueError, match="forbidden"):
        await resolve_image_from_url(AnyUrl(f"file://{symlink}"), settings)
