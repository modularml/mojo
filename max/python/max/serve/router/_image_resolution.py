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
"""Resolve and validate image references from chat-completion requests.

Turns the ``image_url`` / ``video_url`` references in an OpenAI request into raw
image bytes (from ``http(s):``, ``data:``, or ``file:`` URIs) and fully decodes
them once for validation. Kept in its own module so the image-resolution
concern stays focused and out of the much larger ``openai_routes`` handler.
"""

from __future__ import annotations

import asyncio
import base64
import io
import ipaddress
import logging
import socket
from pathlib import Path
from typing import Protocol
from urllib.parse import unquote, urlparse

import aiofiles
import httpx
from httpx import (
    AsyncClient,
    HTTPStatusError,
    Response,
    Timeout,
    TimeoutException,
    TransportError,
)
from max.pipelines.context.exceptions import InputError
from max.serve.config import Settings
from PIL import Image, UnidentifiedImageError
from pydantic import AnyUrl

logger = logging.getLogger("max.serve")

# SSRF protection for client-supplied media URLs: validate the host, reject
# internal/reserved addresses, and pin to the resolved IP. See
# ``_validate_and_pin``.

_ALLOWED_SCHEMES = ("http", "https")

# Explicit cloud-metadata endpoints. Most are already caught by the link-local /
# private predicates below, but list them so the intent is auditable and so a
# predicate gap can never silently expose them.
_METADATA_IPS = frozenset(
    {
        ipaddress.ip_address("169.254.169.254"),  # AWS/GCP/Azure IMDS
        ipaddress.ip_address("169.254.170.2"),  # AWS ECS task metadata
        ipaddress.ip_address("fd00:ec2::254"),  # AWS IMDS over IPv6
    }
)

# Non-routable ranges that ``ipaddress``'s predicates miss/misreport across
# Python versions: 100.64.0.0/10 (RFC6598 CGNAT, the default in-cluster pod/node
# CIDR on EKS/GKE) and fec0::/10 (deprecated IPv6 site-local).
_EXTRA_BLOCKED_NETWORKS = (
    ipaddress.ip_network("100.64.0.0/10"),
    ipaddress.ip_network("fec0::/10"),
)

# Cap on redirect hops we follow, each re-validated. Media fetches need only a
# hop or two; keep it low to bound re-validation work and egress.
_MAX_REDIRECTS = 4


def _ip_is_blocked(
    ip: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    """Whether an address must never be fetched from (an SSRF target).

    Conservative by construction: blocks the union of private, loopback,
    link-local, reserved, multicast and unspecified ranges plus the explicit
    metadata IPs, and unwraps IPv4-in-IPv6 forms (``::ffff:a.b.c.d``, 6to4,
    teredo) so an embedded internal address cannot slip through.
    """
    # Unwrap IPv4-mapped IPv6 (e.g. ``::ffff:169.254.169.254``) and recheck the
    # embedded v4 address against every rule.
    mapped = getattr(ip, "ipv4_mapped", None)
    if mapped is not None and _ip_is_blocked(mapped):
        return True
    # 6to4 (``2002::/16``) and Teredo (``2001::/32``) embed a v4 address too.
    sixtofour = getattr(ip, "sixtofour", None)
    if sixtofour is not None and _ip_is_blocked(sixtofour):
        return True
    teredo = getattr(ip, "teredo", None)
    if teredo is not None:
        # ``teredo`` is a (server, client) pair of v4 addresses.
        if any(_ip_is_blocked(part) for part in teredo):
            return True
    if ip in _METADATA_IPS:
        return True
    if any(ip in net for net in _EXTRA_BLOCKED_NETWORKS):
        return True
    # ``is_global`` is the catch-all; the explicit predicates below are
    # belt-and-suspenders for versions that misreport it.
    if not ip.is_global:
        return True
    return (
        ip.is_private
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_reserved
        or ip.is_multicast
        or ip.is_unspecified
    )


def _parse_allowed_hosts(
    entries: list[str],
) -> tuple[list[ipaddress.IPv4Network | ipaddress.IPv6Network], set[str]]:
    """Split the configured allowlist into IP networks and hostnames.

    An entry that parses as an IP/CIDR becomes a network; everything else is a
    case-insensitive hostname.
    """
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    names: set[str] = set()
    for raw in entries:
        entry = raw.strip()
        if not entry:
            continue
        try:
            nets.append(ipaddress.ip_network(entry, strict=False))
        except ValueError:
            # Normalize the FQDN trailing-dot form so an allowlisted
            # "minio.internal" also matches the "minio.internal." a client may
            # send (and vice versa).
            names.add(entry.rstrip(".").lower())
    return nets, names


def _host_allowlisted(
    host: str,
    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address],
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    names: set[str],
) -> bool:
    """Whether an otherwise-blocked host is explicitly permitted.

    A hostname match permits the host regardless of the address it resolves to
    (the operator is vouching for that name); a CIDR/IP allowlist permits a host
    only when *every* resolved address falls within an allowlisted range (so a
    host that mixes an allowed and a disallowed answer is still rejected).
    """
    if host.rstrip(".").lower() in names:
        return True
    return bool(
        nets and ips and all(any(ip in net for net in nets) for ip in ips)
    )


async def _resolve_host(host: str, port: int) -> list[str]:
    """Resolve a hostname to its IP strings. Isolated for testability.

    Returns the literal IP strings from ``getaddrinfo`` (A and AAAA). A host that
    is already an IP literal resolves to itself.
    """
    loop = asyncio.get_running_loop()
    infos = await loop.getaddrinfo(host, port, type=socket.SOCK_STREAM)
    # sockaddr[0] is typed str | int because typeshed includes the AF_PACKET
    # form; SOCK_STREAM lookups only yield AF_INET/AF_INET6, where it is a str.
    addrs: list[str] = []
    for info in infos:
        addr = info[4][0]
        assert isinstance(addr, str)
        addrs.append(addr)
    return addrs


async def _validate_and_pin(
    url: str,
    nets: list[ipaddress.IPv4Network | ipaddress.IPv6Network],
    names: set[str],
) -> tuple[str, str, str, httpx.URL]:
    """Validate a media URL and resolve it to a pinned connection target.

    Returns ``(pinned_url, host_header, sni_hostname, parsed)`` where
    ``pinned_url`` has its host replaced by a single validated IP literal so
    httpx connects there without re-resolving (no rebinding window), while
    ``host_header`` / ``sni_hostname`` carry the real hostname so the Host header
    is correct and TLS still verifies against the hostname; ``parsed`` is the
    parsed input URL, returned so the caller need not re-parse it.

    Raises ``InputError`` (mapped to a 400) for a disallowed scheme, a host that
    will not resolve, or a host that resolves to an internal/reserved/metadata
    address and is not allowlisted.
    """
    try:
        parsed = httpx.URL(url)
    except httpx.InvalidURL:
        # A malformed URL (e.g. a dotted-octal host or unbalanced IPv6 bracket,
        # possibly arriving via a redirect Location) is a client error, not a
        # server fault -- surface a 400, never an opaque 500.
        raise InputError("malformed media URL") from None
    scheme = parsed.scheme
    if scheme not in _ALLOWED_SCHEMES:
        raise InputError(f"unsupported media URL scheme '{scheme}'")
    host = parsed.host
    if not host:
        raise InputError("media URL has no host")
    default_port = 443 if scheme == "https" else 80
    port = parsed.port or default_port

    try:
        ip_strings = await _resolve_host(host, port)
    except socket.gaierror:
        raise InputError(f"could not resolve media URL host '{host}'") from None
    if not ip_strings:
        raise InputError(f"could not resolve media URL host '{host}'")

    ips: list[ipaddress.IPv4Address | ipaddress.IPv6Address] = []
    for raw_ip in ip_strings:
        # getaddrinfo can hand back a scoped v6 literal ("fe80::1%eth0"); drop
        # the zone id before parsing.
        try:
            ips.append(ipaddress.ip_address(raw_ip.split("%", 1)[0]))
        except ValueError:
            raise InputError(
                f"media URL host '{host}' resolved to an unparseable address"
            ) from None

    if not _host_allowlisted(host, ips, nets, names):
        if any(_ip_is_blocked(ip) for ip in ips):
            raise InputError(
                "media URL host resolves to a disallowed"
                " (internal/reserved) address"
            )

    # Pin to a single validated (or allowlisted) IP, chosen deterministically.
    pinned_ip_obj = min(ips, key=lambda ip: (ip.version != 4, ip.packed))
    pinned_url = str(parsed.copy_with(host=str(pinned_ip_obj), port=port))
    host_header = host if port == default_port else f"{host}:{port}"
    return pinned_url, host_header, host, parsed


# ``data:`` payloads at or below this size (in bytes of the base64 string) are
# decoded inline; larger ones are offloaded to a worker thread. Base64 decoding
# is pure-Python CPU work that blocks the event loop, so a multi-MB payload
# decoded inline stalls every other in-flight request (the TTFT culprit in
# CENG-640). The threshold keeps the common small-thumbnail case off the thread
# pool while pushing the expensive large-payload case off the loop.
_DATA_URI_OFFLOAD_THRESHOLD = 256 * 1024

# Read remote media in bounded chunks so a streamed download can be aborted the
# moment it crosses the size cap, instead of buffering the whole (potentially
# huge) body before checking its length.
_HTTP_CHUNK_SIZE = 256 * 1024

# Explicit fetch timeouts. httpx's default is a 5s timeout on *every* operation,
# which is far too aggressive for media downloads: a large video (or a slow
# CDN) routinely needs more than 5s, and any stall longer than that raises a
# ``ReadTimeout`` that — left unhandled — surfaces as an opaque HTTP 500 (and
# the client then retries the whole generation). The read timeout below is
# per-chunk (each ``_HTTP_CHUNK_SIZE`` read must complete within it), not a cap
# on total download time; total size is already bounded by the byte cap. Pick
# values generous enough for slow-but-steady transfers while still failing a
# truly stalled connection in bounded time.
_FETCH_TIMEOUT = Timeout(connect=10.0, read=30.0, write=10.0, pool=10.0)

# Some media hosts (e.g. Wikimedia, Google Cloud Storage) reject requests that
# carry a default library User-Agent (httpx sends ``python-httpx/...``) with an
# HTTP 403, which turned valid user-supplied image/video URLs into fetch
# failures. Present a common browser User-Agent (and a permissive Accept) so
# fetching from such hosts succeeds.
_FETCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "*/*",
}


def decode_and_validate_images(
    images: list[bytes], max_image_bytes: int | None = None
) -> list[Image.Image]:
    # Fully decode each image so empty, non-image, or truncated/streamed
    # content (e.g. animated or content-negotiated WebP) fails here as a clean
    # 400 instead of reaching the model worker and crashing it with an
    # unhandled PIL error or OSError (HTTP 500). ``Image.open`` is lazy -- it
    # only parses the header -- so a header-valid but undecodable image slips
    # through and later blows up in the tokenizer's ``to_rgb(...)`` ->
    # ``.convert("RGB")`` decode. ``image.load()`` forces that same pixel
    # decode now, while we can still turn the failure into a 400.
    #
    # The decoded images are returned and carried on the request
    # (``TextGenerationRequest.decoded_images``) so the tokenizer reuses them
    # instead of decoding the same bytes a second time. We therefore do not
    # close the images here (no ``with`` block): ``load()`` has already pulled
    # the pixels into memory and the caller owns the decoded image.
    decoded: list[Image.Image] = []
    for image_bytes in images:
        # Optional model-specific cap on resolved bytes (e.g. 10MB).
        if max_image_bytes is not None and len(image_bytes) > max_image_bytes:
            raise InputError(
                "image exceeds the maximum allowed size of "
                f"{max_image_bytes // (1024 * 1024)}MB"
            )
        try:
            image = Image.open(io.BytesIO(image_bytes))
            image.load()
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            SyntaxError,
            Image.DecompressionBombError,
        ) as e:
            raise InputError("invalid or unreadable image content") from e
        decoded.append(image)
    return decoded


def _raise_media_too_large(media_kind: str, max_bytes: int) -> None:
    """Raise an :class:`InputError` for media that exceeds its byte cap.

    Mirrors the message format used by :func:`decode_and_validate_images` so the
    early (pre-decode) rejection is indistinguishable from the late one.

    Args:
        media_kind: ``"image"`` or ``"video"`` (used in the error message).
        max_bytes: The cap that was exceeded, in bytes.

    Raises:
        InputError: Always.
    """
    raise InputError(
        f"{media_kind} exceeds the maximum allowed size of "
        f"{max_bytes // (1024 * 1024)}MB"
    )


def _clean_data_uri_base64(data_uri: str) -> str:
    """Extract and normalize the base64 payload of a ``data:`` URI.

    Tolerates the two ways real clients (and the OpenRouter image relay)
    routinely deviate from canonical base64: stripped ``=`` padding and the
    URL-safe alphabet (``-``/``_``). Splitting this off from the decode lets the
    caller estimate the decoded size (and reject oversized payloads) before
    paying for the decode.
    """
    parts = data_uri.split(",", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("data URI has no base64 payload")
    # Some clients wrap long payloads across lines; strip any whitespace.
    b64 = "".join(parts[1].split())
    # Re-add stripped padding (base64 length must be a multiple of 4).
    b64 += "=" * (-len(b64) % 4)
    return b64


def _base64_decoded_size(b64: str) -> int:
    """Return the exact decoded byte length of a padded base64 string.

    Cheap (``O(1)`` arithmetic on the length) so an oversized ``data:`` payload
    can be rejected before allocating and decoding the (potentially huge) bytes.
    """
    if not b64:
        return 0
    padding = b64[-2:].count("=")
    return (len(b64) // 4) * 3 - padding


def _decode_base64(b64: str) -> bytes:
    decoder = (
        base64.urlsafe_b64decode
        if ("-" in b64 or "_" in b64)
        else base64.b64decode
    )
    return decoder(b64)


def _decode_data_uri_base64(data_uri: str) -> bytes:
    """Decode the base64 payload of a ``data:`` image URI."""
    return _decode_base64(_clean_data_uri_base64(data_uri))


def _resolve_data_uri(
    data_uri: str, max_bytes: int | None, media_kind: str
) -> bytes:
    """Validate-then-decode a ``data:`` payload (run on a worker thread).

    Estimates the decoded size from the base64 length and rejects an oversized
    payload *before* decoding it, so a too-large request never allocates the
    full decoded buffer.
    """
    b64 = _clean_data_uri_base64(data_uri)
    if max_bytes is not None and _base64_decoded_size(b64) > max_bytes:
        _raise_media_too_large(media_kind, max_bytes)
    return _decode_base64(b64)


def _settings_int(value: object) -> int | None:
    """Return a positive int settings value, or ``None``.

    Tolerates ``settings`` being ``None`` or a test mock whose attribute is not
    a real ``int`` (in which case the server-level cap simply does not apply).
    """
    return value if isinstance(value, int) and value > 0 else None


class MediaRef(Protocol):
    """Structural type for a resolvable media reference."""

    @property
    def scheme(self) -> str: ...

    def unicode_string(self) -> str: ...

    def __str__(self) -> str: ...


class DataUrl:
    """Cheap ``data:`` URL ref that skips pydantic's expensive URL parse."""

    __slots__ = ("_raw",)
    scheme = "data"

    def __init__(self, raw: str) -> None:
        self._raw = raw

    def unicode_string(self) -> str:
        return self._raw

    def __str__(self) -> str:
        return self._raw


def make_media_ref(url: str) -> MediaRef:
    """Build a media reference, skipping pydantic validation for ``data:`` URIs.

    ``AnyUrl`` validation of a large base64 ``data:`` payload is pure overhead
    (there is nothing network-shaped to validate) and blocks the event loop, so
    those are wrapped in the lightweight :class:`DataUrl` instead.
    """
    if url[:5].lower() == "data:":
        return DataUrl(url)
    return AnyUrl(url)


async def _read_streamed_response(
    response: Response, media_kind: str, max_bytes: int | None
) -> bytes:
    """Stream a final (non-redirect) response body under the byte cap.

    Enforces the same limits on both the guarded and break-glass fetch paths:
    the ``Content-Length`` fast path rejects an over-cap body before any bytes
    are read, then the body is streamed in bounded chunks and aborted the moment
    the running total crosses ``max_bytes`` (covering a missing or lying
    ``Content-Length``).

    Args:
        response: An open streaming response.
        media_kind: ``"image"`` or ``"video"`` (used in the error message).
        max_bytes: The effective byte cap, or ``None`` for no cap.

    Returns:
        The full response body.

    Raises:
        InputError: If the body exceeds ``max_bytes``.
        HTTPStatusError: If the response status is 4xx or 5xx.
    """
    response.raise_for_status()
    # Fast path: reject up front when the server advertises an over-cap size, so
    # we never start streaming the body.
    if max_bytes is not None:
        advertised = response.headers.get("content-length")
        if (
            advertised is not None
            and advertised.isdigit()
            and int(advertised) > max_bytes
        ):
            _raise_media_too_large(media_kind, max_bytes)
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes(_HTTP_CHUNK_SIZE):
        total += len(chunk)
        if max_bytes is not None and total > max_bytes:
            _raise_media_too_large(media_kind, max_bytes)
        chunks.append(chunk)
    return b"".join(chunks)


async def _fetch_validated(
    image_ref: MediaRef,
    settings: Settings,
    media_kind: str,
    max_bytes: int | None,
    client: AsyncClient,
) -> bytes:
    """SSRF-safe streaming fetch: validate + pin, following redirects manually so
    every hop is re-validated. Streams the final body under the byte cap and
    returns it, or raises ``HTTPStatusError`` (non-2xx -> caller maps) /
    ``InputError``.
    """
    nets, names = _parse_allowed_hosts(settings.media_url_allowed_hosts)
    current = str(image_ref)
    for _hop in range(_MAX_REDIRECTS + 1):
        pinned_url, host_header, sni_hostname, parsed = await _validate_and_pin(
            current, nets, names
        )
        # Only override Host/SNI when the original URL used a hostname. If the
        # client supplied an IP-literal URL, let httpx derive Host/SNI from the
        # (pinned) URL to avoid emitting an invalid IPv6 Host header.
        headers: dict[str, str] = {}
        try:
            ipaddress.ip_address(parsed.host)
        except ValueError:
            headers["Host"] = host_header

        extensions: dict[str, str] = {}
        if parsed.scheme == "https":
            try:
                ipaddress.ip_address(parsed.host)
            except ValueError:
                extensions["sni_hostname"] = sni_hostname
        try:
            async with client.stream(
                "GET",
                pinned_url,
                headers=headers,
                extensions=extensions,
                follow_redirects=False,
            ) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise InputError(
                            f"media url '{image_ref}' returned a redirect with"
                            " no location"
                        )
                    # Resolve a relative redirect against the original hostname
                    # URL (not the pinned-IP one), then loop to re-validate it.
                    try:
                        next_url = str(parsed.join(location))
                    except httpx.InvalidURL:
                        raise InputError(
                            f"media url '{image_ref}' redirected to a malformed"
                            " location"
                        ) from None
                    current = next_url
                    continue
                return await _read_streamed_response(
                    response, media_kind, max_bytes
                )
        except httpx.RemoteProtocolError as e:
            # A malformed redirect Location that httpx rejects while opening the
            # stream is a client-visible 400, not a transient error.
            if "location" in str(e).lower():
                raise InputError(
                    f"media url '{image_ref}' returned a malformed redirect"
                    " location"
                ) from None
            raise
    raise InputError(f"too many redirects fetching media url '{image_ref}'")


async def _fetch_url_bytes(
    image_ref: MediaRef,
    settings: Settings,
    media_kind: str,
    max_bytes: int | None,
) -> bytes:
    """Fetch an http(s) media reference into raw bytes under the byte cap.

    When SSRF protection is enabled (the default) the host is validated and the
    connection pinned to the resolved IP, following redirects manually and
    re-validating each hop; the break-glass path (protection disabled) restores
    the legacy fetch that lets httpx follow redirects itself. Both paths stream
    the body under ``max_bytes`` and share the same error mapping.
    """
    # TODO: Evaluate creating a single AsyncClient for the app.
    async with AsyncClient(
        headers=_FETCH_HEADERS, timeout=_FETCH_TIMEOUT
    ) as client:
        try:
            if settings.media_url_ssrf_protection_enabled:
                return await _fetch_validated(
                    image_ref, settings, media_kind, max_bytes, client
                )
            # Break-glass: guard disabled -> legacy unvalidated fetch that lets
            # httpx follow redirects itself.
            async with client.stream(
                "GET", str(image_ref), follow_redirects=True
            ) as response:
                return await _read_streamed_response(
                    response, media_kind, max_bytes
                )
        except HTTPStatusError as e:
            raise ValueError(
                f"Failed to fetch {media_kind}: HTTP {e.response.status_code}"
            ) from None
        except TimeoutException:
            # A slow/stalled download must not surface as an opaque 500
            # (which the client then retries). Turn it into a clean input
            # error attributable to the unreachable/slow media source.
            raise InputError(
                f"timed out fetching {media_kind} from its URL; the source "
                "may be too slow or the file too large"
            ) from None
        except TransportError as e:
            # Connection reset / DNS / network failure mid-fetch: same
            # treatment as a timeout, a clean input error rather than a 500.
            raise InputError(
                f"failed to fetch {media_kind} from its URL "
                f"({type(e).__name__})"
            ) from None


async def resolve_image_from_url(
    image_ref: MediaRef,
    settings: Settings,
    max_bytes: int | None = None,
    media_kind: str | None = None,
) -> bytes:
    """Resolve a media reference into raw bytes, enforcing a byte cap early.

    The effective cap is the smaller of the per-call ``max_bytes`` (e.g. a
    per-model cap from the tokenizer) and the server-level
    :attr:`Settings.max_bytes`; ``0``/``None`` on either side means "no cap from
    that source". When a cap applies, an oversized payload is rejected before
    the bytes are fully materialized: an ``http(s)`` download is aborted as soon
    as the advertised ``Content-Length`` (or the streamed total) crosses the
    cap, and a ``data:`` payload is rejected from its base64 length before it is
    decoded.

    ``media_kind`` (``"image"``/``"video"``) only selects the error wording; it
    falls back to :attr:`Settings.media_kind` and then ``"image"``.
    """
    # Combine the per-call cap with the server-level cap (smaller wins).
    settings_cap = (
        _settings_int(getattr(settings, "max_bytes", None))
        if settings is not None
        else None
    )
    caps = [c for c in (max_bytes, settings_cap) if c is not None and c > 0]
    max_bytes = min(caps) if caps else None
    if media_kind is None:
        settings_kind = getattr(settings, "media_kind", None)
        media_kind = (
            settings_kind if isinstance(settings_kind, str) else "image"
        )

    if image_ref.scheme == "http" or image_ref.scheme == "https":
        images_bytes = await _fetch_url_bytes(
            image_ref, settings, media_kind, max_bytes
        )
        logger.debug(
            "ResolvedImageUrl: %s -> %d bytes", image_ref, len(images_bytes)
        )
        return images_bytes
    elif image_ref.scheme == "data":
        data_uri = image_ref.unicode_string()
        # Decode off the event loop for large payloads: base64 decoding is
        # CPU-bound pure-Python work that otherwise stalls every concurrent
        # request (CENG-640). Small thumbnails decode inline to skip the
        # thread-pool hop. The size check happens before the decode either way.
        if len(data_uri) > _DATA_URI_OFFLOAD_THRESHOLD:
            images_bytes = await asyncio.to_thread(
                _resolve_data_uri, data_uri, max_bytes, media_kind
            )
        else:
            images_bytes = _resolve_data_uri(data_uri, max_bytes, media_kind)
        logger.debug(
            "ResolvedImageB64: %s -> %d bytes",
            str(image_ref)[:16],
            len(images_bytes),
        )
        return images_bytes
    elif image_ref.scheme == "file":
        if settings is None:
            raise ValueError("Settings required for file URI resolution")

        # Parse the file URI.
        parsed = urlparse(str(image_ref))

        # Check host - only allow empty or localhost.
        if parsed.netloc and parsed.netloc not in ("", "localhost"):
            raise ValueError(
                f"File URI with remote host '{parsed.netloc}' is not supported"
            )

        # Extract and decode the path.
        file_path = Path(unquote(parsed.path))

        # Validate against allowed roots.
        allowed_roots = [Path(root) for root in settings.allowed_image_roots]
        if not allowed_roots:
            raise ValueError(
                "File URI access denied: no allowed roots configured"
            )

        # Canonicalize the path (resolving symlinks and ``..``) so containment
        # is decided against the real target rather than the requested
        # spelling. Resolve non-strictly: a nonexistent path must not raise
        # here, because the allowed-roots check has to run *before* any
        # existence or type probe. Statting an attacker-chosen absolute path
        # that lies outside the roots would turn this into a filesystem
        # existence/type oracle.
        try:
            resolved_path = file_path.resolve()
        except (OSError, RuntimeError) as e:
            raise ValueError(f"Invalid file path: {file_path}") from e

        # Check if path is within allowed roots before touching the filesystem.
        path_allowed = False
        for root in allowed_roots:
            try:
                resolved_path.relative_to(root)
                path_allowed = True
                break
            except ValueError:
                continue

        if not path_allowed:
            raise ValueError(
                f"Path forbidden: {resolved_path} is outside allowed roots"
            )

        # Only now that the path is contained within an allowed root do we
        # probe the filesystem for existence and type.
        if not resolved_path.exists():
            raise ValueError(f"File not found: {file_path}")

        if resolved_path.is_dir():
            raise ValueError(f"Path is a directory: {resolved_path}")

        # Read the file with size limit.
        max_bytes = settings.max_local_image_bytes

        async with aiofiles.open(resolved_path, "rb") as f:
            images_bytes = await f.read(max_bytes + 1)
            if len(images_bytes) > max_bytes:
                raise ValueError(
                    f"File exceeds size limit of {max_bytes} bytes"
                )
        logger.debug(
            "ResolvedFileUri: %s -> %d bytes", resolved_path, len(images_bytes)
        )
        return images_bytes
    raise ValueError(f"Invalid image ref '{image_ref}'")


def _sniff_image_mime(image: Image.Image) -> str:
    """Return the MIME type of a decoded image, from the format PIL reports.

    Sniffing beats guessing from the URL: a ``.jpg`` URL that a host
    content-negotiates to WebP would otherwise be re-encoded under a MIME type
    that contradicts its own payload.
    """
    image_format = image.format
    if not image_format:
        raise InputError("invalid or unreadable image content")
    # MPO is a multi-picture JPEG; every other format PIL names maps directly.
    if image_format == "MPO":
        return "image/jpeg"
    return f"image/{image_format.lower()}"


def _encode_data_uri(image_bytes: bytes) -> str:
    """Fully validate image bytes, then inline them as a base64 ``data:`` URI.

    Runs the same full-pixel decode as the chat path (see
    :func:`decode_and_validate_images`), so truncated or otherwise undecodable
    content fails here as a clean 400 rather than reaching the tokenizer's own
    decode and surfacing as a 500. Unlike the chat path, nothing downstream
    reuses this decode -- the responses path re-decodes from the ``data:`` URI
    it is handed -- so the image is released as soon as its format is read.
    """
    image = decode_and_validate_images([image_bytes])[0]
    try:
        mime = _sniff_image_mime(image)
    finally:
        image.close()
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


async def fetch_media_data_uri(
    url: str, settings: Settings, max_bytes: int | None = None
) -> str:
    """Fetch a web media URL and return it inlined as a base64 ``data:`` URI.

    The ``/v1/responses`` input schema accepts ``data:`` URIs only (see
    ``InputImageContent.validate_data_uri_only``), so a client-supplied
    ``http(s)`` image must be fetched and inlined before the body validates.
    Routing that through :func:`resolve_image_from_url` keeps the responses path
    on the same fetch as chat completions: one byte cap, one error mapping, one
    place where host validation lives, and the same full-decode validation,
    rather than a second downloader that has to be hardened separately.

    Args:
        url: The client-supplied media URL.
        settings: Server settings, supplying the server-level byte cap.
        max_bytes: Optional additional cap; the smaller of the two wins.

    Returns:
        A ``data:<mime>;base64,<payload>`` URI.

    Raises:
        InputError: If the media is oversized, unfetchable, or not a decodable
            image.
    """
    image_bytes = await resolve_image_from_url(
        make_media_ref(url), settings, max_bytes=max_bytes, media_kind="image"
    )
    # Decoding for validation and base64-encoding are both CPU-bound, so a
    # multi-MB image goes to a worker thread for the same reason the ``data:``
    # decode does.
    if len(image_bytes) > _DATA_URI_OFFLOAD_THRESHOLD:
        return await asyncio.to_thread(_encode_data_uri, image_bytes)
    return _encode_data_uri(image_bytes)
