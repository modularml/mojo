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
"""Component-test reproducer for CENG-640.

These tests pin down the preprocessor performance fixes for image/video
downloads without an LLM in the loop:

* base64 ``data:`` decoding is offloaded to a worker thread (does not block the
  event loop) for large payloads, and runs inline for small ones;
* oversized media is rejected *before* its bytes are downloaded/decoded -- via
  the advertised ``Content-Length`` and a streamed-total guard for ``http(s)``,
  and via the base64 length for ``data:`` URIs;
* the OpenAI route enforces the per-request video count and per-video byte caps
  up front, mirroring the image caps.

The headline reproducer is ``test_event_loop_not_blocked_during_decode``: it
deadlocks (and fails) if the decode runs on the event loop, and passes only
when the decode is offloaded -- exactly the regression behind the high TTFT.
"""

from __future__ import annotations

import asyncio
import base64
import io
import threading
from collections.abc import AsyncIterator
from typing import Any

import pytest
from httpx import ConnectError, ReadTimeout
from max.pipelines.context.exceptions import InputError
from max.serve.config import Settings
from max.serve.router import _image_resolution
from max.serve.router._image_resolution import resolve_image_from_url
from max.serve.router.openai_routes import openai_parse_chat_completion_request
from max.serve.schemas.openai import CreateChatCompletionRequest
from PIL import Image
from pydantic import AnyUrl

pytestmark = pytest.mark.asyncio

# A 64MB cap, matching the MiniMax-M3 tokenizer's ``max_image_bytes``.
_CAP = 64 * 1024 * 1024


def _data_uri(payload: bytes) -> str:
    return "data:image/png;base64," + base64.b64encode(payload).decode()


def _png_bytes(size: tuple[int, int] = (8, 8)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color="blue").save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Fix 2: base64 decode runs off the event loop for large payloads.
# ---------------------------------------------------------------------------


async def test_large_data_uri_decode_runs_off_event_loop(monkeypatch) -> None:  # noqa: ANN001
    """A large ``data:`` payload is decoded on a worker thread, not the loop."""
    main_thread = threading.get_ident()
    recorded: dict[str, int] = {}
    original = _image_resolution._decode_base64

    def spy(b64: str) -> bytes:
        recorded["thread"] = threading.get_ident()
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", spy)

    # >256KiB of base64 -> exceeds the offload threshold.
    payload = b"\x00" * (400 * 1024)
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(payload)), settings=Settings(), max_bytes=_CAP
    )
    assert out == payload
    assert recorded["thread"] != main_thread


async def test_small_data_uri_decode_runs_inline(monkeypatch) -> None:  # noqa: ANN001
    """A tiny ``data:`` payload decodes inline (no thread-pool hop)."""
    main_thread = threading.get_ident()
    recorded: dict[str, int] = {}
    original = _image_resolution._decode_base64

    def spy(b64: str) -> bytes:
        recorded["thread"] = threading.get_ident()
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", spy)

    payload = b"\x01" * 512
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(payload)), settings=Settings(), max_bytes=_CAP
    )
    assert out == payload
    assert recorded["thread"] == main_thread


async def test_event_loop_not_blocked_during_decode(monkeypatch) -> None:  # noqa: ANN001
    """Headline reproducer: the event loop stays responsive during decode.

    The decode is made to block on an :class:`threading.Event` that can only be
    released by a coroutine running concurrently on the event loop. If the
    decode ran *on* the loop (the CENG-640 regression), the releaser could never
    run and this would dead-lock until the timeout -> test failure. It passes
    only because the decode is offloaded to a worker thread, leaving the loop
    free to make progress.
    """
    started = threading.Event()
    release = threading.Event()
    original = _image_resolution._decode_base64

    def blocking_decode(b64: str) -> bytes:
        started.set()
        if not release.wait(timeout=5.0):
            raise AssertionError(
                "decode was never released: the event loop was blocked"
            )
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", blocking_decode)

    payload = b"\x00" * (400 * 1024)

    async def releaser() -> None:
        # Runs on the same event loop as resolve. It can only make progress if
        # the loop is free while the decode is in flight.
        while not started.is_set():
            await asyncio.sleep(0.001)
        release.set()

    resolve_task = asyncio.create_task(
        resolve_image_from_url(
            AnyUrl(_data_uri(payload)), settings=Settings(), max_bytes=_CAP
        )
    )
    out, _ = await asyncio.wait_for(
        asyncio.gather(resolve_task, releaser()), timeout=15.0
    )
    assert out == payload


# ---------------------------------------------------------------------------
# Fix 1: reject oversized media before download/decode.
# ---------------------------------------------------------------------------


async def test_oversized_data_uri_rejected_before_decode(monkeypatch) -> None:  # noqa: ANN001
    """An over-cap ``data:`` payload is rejected without ever decoding it."""
    decode_called = False
    original = _image_resolution._decode_base64

    def spy(b64: str) -> bytes:
        nonlocal decode_called
        decode_called = True
        return original(b64)

    monkeypatch.setattr(_image_resolution, "_decode_base64", spy)

    payload = b"\x00" * 4096  # decodes to 4096 bytes, cap is 1024
    with pytest.raises(InputError, match="image exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(payload)), settings=Settings(), max_bytes=1024
        )
    assert not decode_called


async def test_data_uri_within_cap_roundtrips() -> None:
    """A within-cap ``data:`` payload resolves to the exact original bytes."""
    payload = bytes(range(256)) * 8
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(payload)), settings=Settings(), max_bytes=_CAP
    )
    assert out == payload


# ---------------------------------------------------------------------------
# Fix 1: http(s) early-abort on size, via a fake streaming client.
# ---------------------------------------------------------------------------


def _no_ssrf() -> Settings:
    # These fetch tests drive the fake ``client.stream(...)`` directly, so they
    # exercise the break-glass (unvalidated) streaming path rather than the
    # SSRF-guarded path that would resolve the host through real DNS.
    return Settings(media_url_ssrf_protection_enabled=False)


class _FakeResponse:
    def __init__(
        self,
        *,
        headers: dict[str, str],
        chunks: list[bytes],
        read_log: list[int],
        status_code: int = 200,
    ) -> None:
        self.headers = headers
        self.status_code = status_code
        self._chunks = chunks
        self._read_log = read_log

    def raise_for_status(self) -> None:
        return None

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            self._read_log.append(len(chunk))
            yield chunk


class _FakeStream:
    def __init__(self, response: _FakeResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakeAsyncClient:
    def __init__(self, response: _FakeResponse, **_: Any) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeAsyncClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **_: Any) -> _FakeStream:
        return _FakeStream(self._response)


def _install_fake_client(
    monkeypatch,  # noqa: ANN001
    *,
    headers: dict[str, str],
    chunks: list[bytes],
) -> list[int]:
    read_log: list[int] = []
    response = _FakeResponse(headers=headers, chunks=chunks, read_log=read_log)
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(response),
    )
    return read_log


async def test_http_oversized_content_length_rejected_without_download(
    monkeypatch,  # noqa: ANN001
) -> None:
    """An over-cap advertised ``Content-Length`` rejects before any body read."""
    read_log = _install_fake_client(
        monkeypatch,
        headers={"content-length": str(100 * 1024 * 1024)},
        chunks=[b"x" * 1024],
    )
    with pytest.raises(InputError, match="video exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/big.mp4"),
            settings=_no_ssrf(),
            max_bytes=50 * 1024 * 1024,
            media_kind="video",
        )
    # Body was never streamed.
    assert read_log == []


async def test_http_stream_aborts_when_total_exceeds_cap(
    monkeypatch,  # noqa: ANN001
) -> None:
    """With no/short Content-Length, the stream aborts once the total is over."""
    # Ten 60-byte chunks (600 bytes total), cap is 100 bytes; only the first
    # two chunks should be read before the abort.
    read_log = _install_fake_client(
        monkeypatch,
        headers={},  # no content-length advertised
        chunks=[b"a" * 60 for _ in range(10)],
    )
    with pytest.raises(InputError, match="image exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/sneaky.png"),
            settings=_no_ssrf(),
            max_bytes=100,
        )
    assert len(read_log) == 2  # aborted early, not all ten chunks


async def test_http_within_cap_downloads_fully(monkeypatch) -> None:  # noqa: ANN001
    """A within-cap http download returns the concatenated body."""
    read_log = _install_fake_client(
        monkeypatch,
        headers={"content-length": "12"},
        chunks=[b"abcd", b"efgh", b"ijkl"],
    )
    out = await resolve_image_from_url(
        AnyUrl("https://example.com/ok.png"),
        settings=_no_ssrf(),
        max_bytes=_CAP,
    )
    assert out == b"abcdefghijkl"
    assert len(read_log) == 3


# ---------------------------------------------------------------------------
# Fix (PERF-2725): explicit fetch timeout + graceful timeout/transport handling.
# A slow/stalled or failed download must raise a clean InputError (-> 400), not
# an opaque ReadTimeout that surfaces as a 500 the client then retries.
# ---------------------------------------------------------------------------


class _TimeoutResponse(_FakeResponse):
    """Streams one chunk, then raises ``ReadTimeout`` like a stalled download."""

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        self._read_log.append(7)
        yield b"partial"
        raise ReadTimeout("simulated mid-stream stall")


class _RaisingStream:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> _FakeResponse:
        raise self._exc

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _RaisingClient:
    def __init__(self, exc: Exception) -> None:
        self._exc = exc

    async def __aenter__(self) -> _RaisingClient:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **_: Any) -> _RaisingStream:
        return _RaisingStream(self._exc)


async def test_http_read_timeout_raises_clean_input_error(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A mid-stream stall (ReadTimeout) -> clean InputError, not an opaque 500."""
    response = _TimeoutResponse(headers={}, chunks=[], read_log=[])
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _FakeAsyncClient(response),
    )
    with pytest.raises(InputError, match="timed out fetching video"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/slow.mp4"),
            settings=_no_ssrf(),
            max_bytes=_CAP,
            media_kind="video",
        )


async def test_http_transport_error_raises_clean_input_error(
    monkeypatch,  # noqa: ANN001
) -> None:
    """A connect/transport failure -> clean InputError, not a 500."""
    monkeypatch.setattr(
        _image_resolution,
        "AsyncClient",
        lambda **kw: _RaisingClient(ConnectError("simulated connect failure")),
    )
    with pytest.raises(InputError, match="failed to fetch video"):
        await resolve_image_from_url(
            AnyUrl("https://example.com/unreachable.mp4"),
            settings=_no_ssrf(),
            max_bytes=_CAP,
            media_kind="video",
        )


async def test_http_client_uses_explicit_non_default_timeout(
    monkeypatch,  # noqa: ANN001
) -> None:
    """The fetch client is built with an explicit timeout > httpx's 5s default.

    The default httpx timeout (5s on every op) is too aggressive for media
    downloads; this pins that an explicit, more generous read timeout is set.
    """
    captured: dict[str, Any] = {}

    def _factory(**kw: Any) -> _FakeAsyncClient:
        captured.update(kw)
        return _FakeAsyncClient(
            _FakeResponse(
                headers={"content-length": "2"},
                chunks=[b"ok"],
                read_log=[],
            )
        )

    monkeypatch.setattr(_image_resolution, "AsyncClient", _factory)
    await resolve_image_from_url(
        AnyUrl("https://example.com/ok.png"),
        settings=_no_ssrf(),
        max_bytes=_CAP,
    )
    assert "timeout" in captured, (
        "fetch client must be given an explicit timeout"
    )
    read_timeout = getattr(captured["timeout"], "read", None)
    assert read_timeout is not None and read_timeout > 5.0


# ---------------------------------------------------------------------------
# Fix 3: the OpenAI route enforces video count + byte caps up front.
# ---------------------------------------------------------------------------


def _video_request(urls: list[str]) -> CreateChatCompletionRequest:
    content: list[dict[str, Any]] = [{"type": "text", "text": "describe"}]
    content += [{"type": "video_url", "video_url": {"url": u}} for u in urls]
    return CreateChatCompletionRequest.model_validate(
        {"model": "test", "messages": [{"role": "user", "content": content}]}
    )


async def test_parse_rejects_too_many_videos_before_download(
    monkeypatch,  # noqa: ANN001
) -> None:
    """Over the per-request video count -> 400 without resolving any video."""
    resolve_calls = 0
    original = _image_resolution.resolve_image_from_url

    async def spy(*args: Any, **kwargs: Any) -> bytes:
        nonlocal resolve_calls
        resolve_calls += 1
        return await original(*args, **kwargs)

    monkeypatch.setattr(
        "max.serve.router.openai_routes.resolve_image_from_url", spy
    )

    request = _video_request([_data_uri(b"x" * 16) for _ in range(5)])
    with pytest.raises(InputError, match="too many videos"):
        await openai_parse_chat_completion_request(
            request,
            wrap_content=True,
            settings=Settings(),
            max_videos_per_request=3,
        )
    assert resolve_calls == 0


async def test_parse_rejects_oversized_video_before_decode() -> None:
    """An over-cap video data URI -> 400 with a video-specific message."""
    request = _video_request([_data_uri(b"\x00" * 8192)])
    with pytest.raises(InputError, match="video exceeds the maximum"):
        await openai_parse_chat_completion_request(
            request,
            wrap_content=True,
            settings=Settings(),
            max_video_bytes=1024,
        )


async def test_parse_accepts_within_cap_image_and_video() -> None:
    """A small image + small video within all caps parse successfully."""
    png = _png_bytes()
    image_uri = _data_uri(png)
    video_uri = _data_uri(b"\x20" * 64)
    request = CreateChatCompletionRequest.model_validate(
        {
            "model": "test",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": image_uri}},
                        {"type": "video_url", "video_url": {"url": video_uri}},
                    ],
                }
            ],
        }
    )
    parsed = await openai_parse_chat_completion_request(
        request,
        wrap_content=True,
        settings=Settings(),
        max_images_per_request=200,
        max_image_bytes=_CAP,
        max_videos_per_request=20,
        max_video_bytes=50 * 1024 * 1024,
    )
    assert len(parsed.images) == 1
    assert len(parsed.videos) == 1
    assert parsed.images[0] == png
    assert parsed.videos[0] == b"\x20" * 64


# ---------------------------------------------------------------------------
# Settings: server-level max_bytes cap and default media_kind.
# ---------------------------------------------------------------------------


def test_settings_media_defaults() -> None:
    """The new media fields default to no server cap / image labelling."""
    settings = Settings()
    assert settings.max_bytes == 0
    assert settings.media_kind == "image"


async def test_settings_max_bytes_applies_when_no_per_call_cap() -> None:
    """``Settings.max_bytes`` caps media even when the caller passes no cap."""
    settings = Settings(max_bytes=1024)
    with pytest.raises(InputError, match="image exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 4096)), settings=settings
        )
    # Within the server cap -> resolves fine.
    out = await resolve_image_from_url(
        AnyUrl(_data_uri(b"\x00" * 512)), settings=settings
    )
    assert out == b"\x00" * 512


async def test_settings_max_bytes_is_a_ceiling_over_per_call_cap() -> None:
    """The effective cap is the smaller of the per-call and server caps."""
    settings = Settings(max_bytes=1024)
    # Per-call cap is larger (8192), so the 1024 server cap should win.
    with pytest.raises(InputError, match="image exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 4096)),
            settings=settings,
            max_bytes=8192,
        )


async def test_settings_media_kind_used_in_error_message() -> None:
    """``Settings.media_kind`` labels the error when the caller omits it."""
    settings = Settings(max_bytes=1024, media_kind="video")
    with pytest.raises(InputError, match="video exceeds the maximum"):
        await resolve_image_from_url(
            AnyUrl(_data_uri(b"\x00" * 4096)), settings=settings
        )
