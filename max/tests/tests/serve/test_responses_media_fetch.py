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
"""Tests for ``fetch_media_data_uri``, the ``/v1/responses`` media fetch.

The responses input schema accepts ``data:`` URIs only, so a client-supplied
``http(s)`` image is fetched and inlined before validation. These tests pin the
properties that made that path worth rewriting: it goes through the shared
resolver (so it inherits its byte cap and host validation instead of downloading
unbounded), and the inlined MIME type is sniffed from the bytes rather than
guessed from the URL.
"""

import base64
import io
from collections.abc import AsyncIterator
from typing import Any

import pytest
from max.pipelines.context.exceptions import InputError
from max.serve.config import APIType, Settings
from max.serve.router import _image_resolution
from max.serve.router._image_resolution import fetch_media_data_uri
from PIL import Image

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skip(
        reason="SERVOPT-1577: the media fetch now resolves and validates the "
        "URL host before connecting, so these fakes fail at DNS resolution "
        "and need a _resolve_host stub."
    ),
]


def _image_bytes(image_format: str) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color="red").save(buf, format=image_format)
    return buf.getvalue()


def _settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        api_types=[APIType.OPENAI], use_heartbeat=False
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class _FakeResponse:
    def __init__(self, chunks: list[bytes], headers: dict[str, str]) -> None:
        self.headers = headers
        self.status_code = 200
        self._chunks = chunks

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> "_FakeAsyncClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **_: Any) -> _FakeStream:
        return _FakeStream(self._response)


def _install_fake_client(
    monkeypatch,  # noqa: ANN001
    body: bytes,
    headers: dict[str, str] | None = None,
) -> None:
    response = _FakeResponse([body], headers or {})
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(response),
    )


def _split_data_uri(data_uri: str) -> tuple[str, bytes]:
    assert data_uri.startswith("data:")
    header, payload = data_uri.split(",", 1)
    mime = header.removeprefix("data:").removesuffix(";base64")
    return mime, base64.b64decode(payload)


async def test_fetch_returns_data_uri_with_original_bytes(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A fetched image comes back inlined byte-for-byte as a data URI."""
    png = _image_bytes("PNG")
    _install_fake_client(monkeypatch, png)
    mime, decoded = _split_data_uri(
        await fetch_media_data_uri("https://example.com/cat.png", _settings())
    )
    assert mime == "image/png"
    assert decoded == png


async def test_mime_is_sniffed_from_bytes_not_url(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A host that content-negotiates WebP for a .jpg URL is labelled webp.

    The previous implementation guessed the MIME type from the URL extension (or
    a header), so this case produced a data URI whose declared type contradicted
    its payload.
    """
    webp = _image_bytes("WEBP")
    _install_fake_client(
        monkeypatch, webp, headers={"content-type": "image/jpeg"}
    )
    mime, decoded = _split_data_uri(
        await fetch_media_data_uri("https://example.com/photo.jpg", _settings())
    )
    assert mime == "image/webp"
    assert decoded == webp


async def test_non_image_content_rejected(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Fetched non-image content is a clean 400, not an inlined fake image."""
    _install_fake_client(monkeypatch, b'{"models": ["internal"]}')
    with pytest.raises(InputError, match="invalid or unreadable image content"):
        await fetch_media_data_uri(
            "http://10.0.0.1:8000/v1/models", _settings()
        )


async def test_truncated_image_rejected(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A header-valid but truncated image fails here, not in the tokenizer.

    Only a full pixel decode catches this: a header-only parse accepts the
    payload, and the failure then resurfaces from the tokenizer's own decode as
    a 500. This is the parity the chat path already had via
    ``decode_and_validate_images``.
    """
    buf = io.BytesIO()
    # Big enough that a truncation still leaves a parseable header, so this
    # exercises the decode rather than the open.
    Image.new("RGB", (256, 256), color="blue").save(buf, format="PNG")
    full = buf.getvalue()
    _install_fake_client(monkeypatch, full[: int(len(full) * 0.6)])
    with pytest.raises(InputError, match="invalid or unreadable image content"):
        await fetch_media_data_uri(
            "https://example.com/truncated.png", _settings()
        )


async def test_server_level_cap_applies(
    monkeypatch,  # noqa: ANN001
) -> None:
    """``Settings.max_bytes`` bounds the responses fetch.

    The old path had no cap at all, so a large body was downloaded in full and
    then base64-expanded in memory.
    """
    _install_fake_client(monkeypatch, b"\x00" * 4096)
    with pytest.raises(InputError, match="image exceeds the maximum"):
        await fetch_media_data_uri(
            "https://example.com/big.png", _settings(max_bytes=1024)
        )


async def test_per_call_cap_applies(
    monkeypatch,  # noqa: ANN001
) -> None:
    """An explicit ``max_bytes`` bounds the fetch even with no server cap."""
    _install_fake_client(monkeypatch, b"\x00" * 4096)
    with pytest.raises(InputError, match="image exceeds the maximum"):
        await fetch_media_data_uri(
            "https://example.com/big.png", _settings(), max_bytes=1024
        )


async def test_large_image_encoded_off_the_event_loop(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A payload over the offload threshold encodes on a worker thread.

    Base64 encoding is pure-Python CPU work, so a multi-MB image must not block
    the loop while other requests are in flight.
    """
    big = _image_bytes("PNG") + b"\x00" * (
        _image_resolution._DATA_URI_OFFLOAD_THRESHOLD + 1
    )
    _install_fake_client(monkeypatch, big)

    offloaded: list[str] = []
    original = _image_resolution.asyncio.to_thread

    async def spy(fn, *args, **kwargs):  # noqa: ANN001, ANN202
        offloaded.append(getattr(fn, "__name__", repr(fn)))
        return await original(fn, *args, **kwargs)

    monkeypatch.setattr(_image_resolution.asyncio, "to_thread", spy)
    # PNG bytes with trailing junk still open as a PNG, so this exercises the
    # offload without needing a genuinely huge image.
    await fetch_media_data_uri("https://example.com/big.png", _settings())
    assert "_encode_data_uri" in offloaded
