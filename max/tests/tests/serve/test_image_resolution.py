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
"""Unit tests for the SSRF-hardened media-URL fetch in ``_image_resolution``.

These exercise the resolution helpers directly rather than through the OpenAI
route: the ``_ip_is_blocked`` block predicate, ``_validate_and_pin`` scheme
guard, and the guarded ``resolve_image_from_url`` fetch path (resolution +
IP pinning, manual redirect re-validation, allowlist handling, size cap, and
the break-glass disable switch).
"""

import io
import ipaddress
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from max.pipelines.context.exceptions import InputError
from max.serve.config import APIType, Settings
from max.serve.router._image_resolution import (
    _ip_is_blocked,
    _validate_and_pin,
    resolve_image_from_url,
)
from PIL import Image
from pydantic import AnyUrl

# --- SSRF protection ------------------------------------------------------- #
#
# These drive the guarded http(s) fetch path added on top of Pranav's streaming
# implementation. The fake client mirrors ``client.stream(...)`` (an async
# context manager whose response exposes ``is_redirect``, ``headers``,
# ``raise_for_status`` and an async ``aiter_bytes``), so the SSRF path exercises
# the same streaming + size-cap + error-mapping machinery as the real fetch.


def _png_bytes(color: str = "red") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (4, 4), color=color).save(buf, format="PNG")
    return buf.getvalue()


def _resolver_settings(**overrides: Any) -> Settings:
    kwargs: dict[str, Any] = dict(
        api_types=[APIType.OPENAI],
        use_heartbeat=False,
    )
    kwargs.update(overrides)
    return Settings(**kwargs)


class _FakeStreamResponse:
    """A single streamed response hop for the fake client."""

    def __init__(
        self,
        *,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        chunks: list[bytes] | None = None,
    ) -> None:
        self.status_code = status_code
        self.headers = headers or {}
        self._chunks = chunks or []

    @property
    def is_redirect(self) -> bool:
        return self.status_code in (301, 302, 303, 307, 308)

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"HTTP {self.status_code}",
                request=httpx.Request("GET", "https://x/"),
                response=httpx.Response(self.status_code),
            )

    async def aiter_bytes(
        self, chunk_size: int | None = None
    ) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


class _FakeStreamCtx:
    def __init__(self, response: _FakeStreamResponse) -> None:
        self._response = response

    async def __aenter__(self) -> _FakeStreamResponse:
        return self._response

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _RecordingStreamClient:
    """Fake ``AsyncClient`` whose ``stream`` records each hop and replays a
    per-connect-host response, so SSRF pinning + manual redirects are testable
    against the real streaming code path.
    """

    def __init__(
        self, responses: dict[str, _FakeStreamResponse], **_: Any
    ) -> None:
        # Keyed by the host httpx actually connects to (the pinned IP literal).
        self._responses = responses
        self.hops: list[dict[str, Any]] = []

    async def __aenter__(self) -> "_RecordingStreamClient":
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False

    def stream(self, method: str, url: str, **kwargs: Any) -> _FakeStreamCtx:
        parsed = httpx.URL(url)
        headers = kwargs.get("headers", {})
        extensions = kwargs.get("extensions", {})
        self.hops.append(
            {
                "connect_host": parsed.host,
                "host_header": headers.get("Host"),
                "sni": extensions.get("sni_hostname"),
                "follow_redirects": kwargs.get("follow_redirects"),
            }
        )
        return _FakeStreamCtx(self._responses[parsed.host])


def _patch_stream_client(client: _RecordingStreamClient) -> Any:
    # Patch the per-request constructor to hand back this recording client so
    # validation + pinning run against the real streaming fetch code.
    return patch(
        "max.serve.router._image_resolution.AsyncClient",
        new=lambda **kw: client,
    )


def test_ip_is_blocked_predicate() -> None:
    # The core block predicate must reject every internal/reserved/metadata
    # range (including IPv4-in-IPv6 and 6to4-embedded forms) and accept public
    # addresses. This is the single point that decides "is this an SSRF target".
    blocked = [
        "127.0.0.1",  # loopback
        "10.0.0.1",  # RFC1918
        "192.168.1.1",  # RFC1918
        "172.16.0.1",  # RFC1918
        "169.254.169.254",  # AWS/GCP/Azure IMDS
        "169.254.170.2",  # ECS task metadata
        "169.254.1.1",  # link-local
        "0.0.0.0",  # unspecified
        "224.0.0.1",  # multicast
        "100.64.0.10",  # RFC6598 CGNAT (EKS/GKE pod-node space)
        "100.127.255.1",  # CGNAT upper edge
        "::1",  # v6 loopback
        "fe80::1",  # v6 link-local
        "fd00::1",  # v6 unique-local
        "fd00:ec2::254",  # AWS IMDS v6
        "fec0::1",  # deprecated v6 site-local
        "::ffff:127.0.0.1",  # v4-mapped loopback
        "::ffff:169.254.169.254",  # v4-mapped IMDS
        "2002:0a00:0001::1",  # 6to4 embedding 10.0.0.1
    ]
    for s in blocked:
        assert _ip_is_blocked(ipaddress.ip_address(s)), f"{s} should be blocked"

    allowed = [
        "8.8.8.8",
        "1.1.1.1",
        "93.184.216.34",
        "2606:2800:220:1:248:1893:25c8:1946",
        "2001:4860:4860::8888",
    ]
    for s in allowed:
        assert not _ip_is_blocked(ipaddress.ip_address(s)), (
            f"{s} should be allowed"
        )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "resolved_ip",
    [
        "169.254.169.254",
        "169.254.170.2",
        "10.0.0.1",
        "127.0.0.1",
        "169.254.0.5",
        "100.64.0.10",
        "0.0.0.0",
        "224.0.0.1",
        "::1",
        "fd00::1",
        "fe80::1",
        "fd00:ec2::254",
        "fec0::1",
        "::ffff:169.254.169.254",
    ],
)
async def test_ssrf_blocks_internal_addresses(resolved_ip: str) -> None:
    # A public-looking hostname that resolves to an internal/metadata address is
    # rejected as a clean 400 and is NEVER fetched (the guard runs before any
    # network I/O). This is the core SSRF defense and the DNS-rebinding answer:
    # whatever the host resolves to is what we check, and a blocked answer stops
    # here.
    client = _RecordingStreamClient({})
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=AsyncMock(return_value=[resolved_ip]),
        ),
        _patch_stream_client(client),
    ):
        with pytest.raises(InputError, match="disallowed"):
            await resolve_image_from_url(
                AnyUrl("https://evil.example/x.png"),
                settings=_resolver_settings(),
            )
    # Never connected -- rejected purely on resolution.
    assert client.hops == []


@pytest.mark.asyncio
async def test_ssrf_pins_to_validated_ip_and_preserves_host() -> None:
    # The load-bearing property: the request connects to the validated IP literal
    # (so DNS is never consulted again -- no rebinding window) while the Host
    # header and TLS SNI still carry the real hostname (so routing and cert
    # verification are correct).
    png = _png_bytes()
    client = _RecordingStreamClient(
        {"93.184.216.34": _FakeStreamResponse(chunks=[png])}
    )
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            # A genuinely global v6 -- 2001:db8::/32 is the RFC 3849
            # documentation range, which the guard blocks, so using it here
            # rejects the request before pinning is ever reached.
            new=AsyncMock(
                return_value=[
                    "2606:2800:220:1:248:1893:25c8:1946",
                    "93.184.216.34",
                ]
            ),
        ),
        _patch_stream_client(client),
    ):
        out = await resolve_image_from_url(
            AnyUrl("https://host.example/image.png"),
            settings=_resolver_settings(),
        )

    assert out == png
    assert len(client.hops) == 1
    # Connected to the pinned IP, not the hostname.
    assert client.hops[0]["connect_host"] == "93.184.216.34"
    # ...but presented the real hostname for Host + TLS SNI.
    assert client.hops[0]["host_header"] == "host.example"
    assert client.hops[0]["sni"] == "host.example"
    # The guarded path never delegates redirect-following to httpx.
    assert client.hops[0]["follow_redirects"] is False


@pytest.mark.asyncio
async def test_ssrf_redirect_to_internal_is_blocked() -> None:
    # A 3xx whose Location points at an internal address is the classic SSRF
    # bypass. Redirects are followed manually and each hop is re-validated, so
    # the internal target is rejected and never fetched.
    client = _RecordingStreamClient(
        {
            "93.184.216.34": _FakeStreamResponse(
                status_code=302,
                headers={"location": "http://169.254.169.254/latest/meta-data"},
            ),
        }
    )

    async def _resolve(host: str, port: int) -> list[str]:
        return {
            "host.example": ["93.184.216.34"],
            "169.254.169.254": ["169.254.169.254"],
        }[host]

    with (
        patch("max.serve.router._image_resolution._resolve_host", new=_resolve),
        _patch_stream_client(client),
    ):
        with pytest.raises(InputError, match="disallowed"):
            await resolve_image_from_url(
                AnyUrl("https://host.example/x.png"),
                settings=_resolver_settings(),
            )

    # Only the first (public) hop reached the transport; the redirect target was
    # rejected during re-validation, before any connect.
    assert [h["connect_host"] for h in client.hops] == ["93.184.216.34"]


@pytest.mark.asyncio
async def test_ssrf_redirect_to_public_is_followed() -> None:
    # A legitimate cross-host redirect to another public host is followed (each
    # hop re-validated + re-pinned) and returns the final bytes.
    png = _png_bytes(color="blue")
    client = _RecordingStreamClient(
        {
            "93.184.216.34": _FakeStreamResponse(
                status_code=302,
                headers={"location": "http://other.example/final.png"},
            ),
            "8.8.4.4": _FakeStreamResponse(chunks=[png]),
        }
    )

    async def _resolve(host: str, port: int) -> list[str]:
        return {
            "host.example": ["93.184.216.34"],
            "other.example": ["8.8.4.4"],
        }[host]

    with (
        patch("max.serve.router._image_resolution._resolve_host", new=_resolve),
        _patch_stream_client(client),
    ):
        out = await resolve_image_from_url(
            AnyUrl("https://host.example/x.png"),
            settings=_resolver_settings(),
        )

    assert out == png
    assert [h["connect_host"] for h in client.hops] == [
        "93.184.216.34",
        "8.8.4.4",
    ]


@pytest.mark.asyncio
async def test_ssrf_allowlist_hostname_permits_internal_but_still_pins() -> (
    None
):
    # An allowlisted hostname may resolve to an internal address (e.g. an
    # in-cluster object store) and is permitted -- but the connection is still
    # pinned to the validated IP and presents the real hostname.
    png = _png_bytes(color="green")
    client = _RecordingStreamClient(
        {"10.0.0.5": _FakeStreamResponse(chunks=[png])}
    )
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=AsyncMock(return_value=["10.0.0.5"]),
        ),
        _patch_stream_client(client),
    ):
        out = await resolve_image_from_url(
            AnyUrl("http://minio.internal/image.png"),
            settings=_resolver_settings(
                media_url_allowed_hosts=["minio.internal"]
            ),
        )

    assert out == png
    assert client.hops[0]["connect_host"] == "10.0.0.5"
    assert client.hops[0]["host_header"] == "minio.internal"


@pytest.mark.asyncio
async def test_ssrf_allowlist_cidr_permits_internal() -> None:
    # A CIDR allowlist permits a host whose every resolved address falls inside
    # an allowlisted range.
    png = _png_bytes()
    client = _RecordingStreamClient(
        {"10.1.2.3": _FakeStreamResponse(chunks=[png])}
    )
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=AsyncMock(return_value=["10.1.2.3"]),
        ),
        _patch_stream_client(client),
    ):
        out = await resolve_image_from_url(
            AnyUrl("http://internal.example/image.png"),
            settings=_resolver_settings(media_url_allowed_hosts=["10.0.0.0/8"]),
        )
    assert out == png
    assert client.hops[0]["connect_host"] == "10.1.2.3"


@pytest.mark.asyncio
async def test_ssrf_allowlist_cidr_rejects_mixed_answer() -> None:
    # A host that mixes an allowlisted address with a disallowed one is still
    # rejected -- the CIDR allowlist requires *every* resolved address to be in
    # range, so a poisoned/rebinding answer can't sneak through.
    client = _RecordingStreamClient({})
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=AsyncMock(return_value=["10.1.2.3", "169.254.169.254"]),
        ),
        _patch_stream_client(client),
    ):
        with pytest.raises(InputError, match="disallowed"):
            await resolve_image_from_url(
                AnyUrl("http://internal.example/image.png"),
                settings=_resolver_settings(
                    media_url_allowed_hosts=["10.0.0.0/8"]
                ),
            )
    assert client.hops == []


@pytest.mark.asyncio
async def test_ssrf_disabled_skips_validation() -> None:
    # The break-glass switch: with the guard disabled, the legacy streaming
    # fetch runs (httpx follows redirects itself) and no resolution/validation
    # happens -- even a metadata IP is fetched. Proves the flag is a true bypass.
    png = _png_bytes()
    client = _RecordingStreamClient(
        {"169.254.169.254": _FakeStreamResponse(chunks=[png])}
    )
    resolve_mock = AsyncMock(return_value=["169.254.169.254"])
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=resolve_mock,
        ),
        _patch_stream_client(client),
    ):
        out = await resolve_image_from_url(
            AnyUrl("http://169.254.169.254/x.png"),
            settings=_resolver_settings(
                media_url_ssrf_protection_enabled=False
            ),
        )
    assert out == png
    # Validation path never ran.
    resolve_mock.assert_not_awaited()
    # Legacy path fetched once, with httpx following redirects itself.
    assert len(client.hops) == 1
    assert client.hops[0]["follow_redirects"] is True


@pytest.mark.asyncio
async def test_ssrf_enforces_size_cap_on_streamed_body() -> None:
    # The guarded path streams under the byte cap exactly like the legacy path:
    # a body whose running total crosses the cap is aborted with a clean 400.
    client = _RecordingStreamClient(
        {"93.184.216.34": _FakeStreamResponse(chunks=[b"a" * 60] * 10)}
    )
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=AsyncMock(return_value=["93.184.216.34"]),
        ),
        _patch_stream_client(client),
    ):
        with pytest.raises(InputError, match="image exceeds the maximum"):
            await resolve_image_from_url(
                AnyUrl("https://host.example/big.png"),
                settings=_resolver_settings(),
                max_bytes=100,
            )


@pytest.mark.asyncio
async def test_validate_and_pin_rejects_non_http_scheme() -> None:
    # A non-http(s) scheme is rejected before any resolution -- defends the
    # fetch path even if a caller routes an unexpected scheme to it.
    with pytest.raises(InputError, match="scheme"):
        await _validate_and_pin("ftp://internal.example/secret", [], set())


@pytest.mark.asyncio
async def test_ssrf_redirect_to_malformed_location_is_client_error() -> None:
    # A server-controlled redirect Location that httpx cannot parse (e.g.
    # dotted-octal host) must surface as a clean 400, not an opaque 500 -- it is
    # never fetched.
    client = _RecordingStreamClient(
        {
            "93.184.216.34": _FakeStreamResponse(
                status_code=302,
                headers={"location": "http://0177.0.0.1/x"},
            ),
        }
    )
    with (
        patch(
            "max.serve.router._image_resolution._resolve_host",
            new=AsyncMock(return_value=["93.184.216.34"]),
        ),
        _patch_stream_client(client),
    ):
        with pytest.raises(InputError, match="malformed"):
            await resolve_image_from_url(
                AnyUrl("https://host.example/x.png"),
                settings=_resolver_settings(),
            )
