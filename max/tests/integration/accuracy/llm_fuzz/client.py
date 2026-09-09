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
"""
Low-level async HTTP client for fuzz testing.
Uses ONLY Python stdlib (asyncio + http.client in threadpool).
No external dependencies required.

Deliberately does NOT validate payloads—scenarios need to send
malformed, broken, and adversarial requests.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import http.client
import json
import socket
import ssl
import struct
import time
import urllib.parse
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from model_config import ModelConfig

if TYPE_CHECKING:
    from validator_client import ValidatorClient


@dataclass
class RawResponse:
    status: int
    headers: dict[str, str]
    body: str
    elapsed_ms: float
    error: str | None = None
    chunks: list[str] | None = None  # populated for streaming
    cancelled: bool = False


@dataclass
class RunConfig:
    base_url: str  # e.g. http://localhost:8000
    api_key: str = ""
    model: str = "default"
    timeout: float = 30.0
    max_concurrency: int = 200
    endurance_duration_sec: float = 300.0  # 5 minutes default
    endurance_intensity: str = "medium"  # low=5/s, medium=20/s, high=100/s
    model_config: ModelConfig = field(default_factory=ModelConfig)
    verbose: bool = False  # when True, scenarios may attach full request/response to results
    # Populated by fuzz.py before scenarios run; None when --validation-only is off.
    validator: ValidatorClient | None = None
    k2vv_mode: str = "quick"
    # Node count the image stress scenarios scale their concurrency by, so a
    # multi-node deployment is driven at the same per-node pressure a single
    # pod sees.
    image_stress_nodes: int = 1


class FuzzClient:
    """Thin async wrapper around http.client for adversarial requests."""

    def __init__(self, config: RunConfig):
        self.config = config
        self._parsed = urllib.parse.urlparse(config.base_url)
        self._pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=config.max_concurrency
        )
        self._held_sockets: list[socket.socket] = []

    async def __aenter__(self) -> FuzzClient:
        return self

    async def __aexit__(self, *exc: object) -> None:
        self.close_held_sockets()
        self._pool.shutdown(wait=True, cancel_futures=True)

    def close_held_sockets(self) -> None:
        """Close all held raw sockets from connection exhaustion tests."""
        for s in self._held_sockets:
            try:
                s.close()
            except Exception:
                pass
        self._held_sockets.clear()

    @property
    def _is_https(self) -> bool:
        return self._parsed.scheme == "https"

    @property
    def _host(self) -> str:
        return self._parsed.hostname or "localhost"

    @property
    def _port(self) -> int:
        if self._parsed.port:
            return self._parsed.port
        return 443 if self._is_https else 80

    @property
    def _base_path(self) -> str:
        return self._parsed.path.rstrip("/")

    @property
    def _chat_path(self) -> str:
        return f"{self._base_path}/v1/chat/completions"

    def _base_headers(
        self, extra: dict[str, str] | None = None
    ) -> dict[str, str]:
        h = {"Content-Type": "application/json", "Host": self._host}
        if self.config.api_key:
            h["Authorization"] = f"Bearer {self.config.api_key}"
        if extra:
            h.update(extra)
        return h

    def _make_conn(
        self, timeout: float | None = None
    ) -> http.client.HTTPConnection:
        to = timeout or self.config.timeout
        if self._is_https:
            ctx = ssl.create_default_context()
            return http.client.HTTPSConnection(
                self._host, self._port, timeout=to, context=ctx
            )
        return http.client.HTTPConnection(self._host, self._port, timeout=to)

    def _make_socket(self, timeout: float) -> socket.socket:
        sock = socket.create_connection(
            (self._host, self._port), timeout=timeout
        )
        if self._is_https:
            ctx = ssl.create_default_context()
            sock = ctx.wrap_socket(sock, server_hostname=self._host)
        sock.settimeout(timeout)
        return sock

    def _base_header_items(
        self, include_content_type: bool = True, include_auth: bool = True
    ) -> list[tuple[str, str]]:
        items = [("Host", self._host)]
        if include_content_type:
            items.append(("Content-Type", "application/json"))
        if include_auth and self.config.api_key:
            items.append(("Authorization", f"Bearer {self.config.api_key}"))
        return items

    @staticmethod
    def _format_raw_request(
        method: str,
        path: str,
        header_items: list[tuple[str, str]],
        body: bytes | None,
    ) -> bytes:
        lines = [f"{method} {path} HTTP/1.1"]
        lines.extend(f"{key}: {value}" for key, value in header_items)
        return ("\r\n".join(lines) + "\r\n\r\n").encode(
            "utf-8", errors="replace"
        ) + (body or b"")

    def build_raw_request(
        self,
        method: str,
        path: str,
        *,
        body: bytes | None = None,
        header_items: list[tuple[str, str]] | None = None,
        include_content_type: bool = True,
        include_auth: bool = True,
        auto_content_length: bool = True,
        content_length: int | None = None,
    ) -> bytes:
        items = self._base_header_items(
            include_content_type=include_content_type, include_auth=include_auth
        )
        if header_items:
            items.extend(header_items)
        if auto_content_length and body is not None:
            items.append(("Content-Length", str(len(body))))
        elif content_length is not None:
            items.append(("Content-Length", str(content_length)))
        return self._format_raw_request(method, path, items, body)

    def _recv_raw_http_response(
        self, sock: socket.socket, timeout: float, max_bytes: int = 256 * 1024
    ) -> tuple[int, dict[str, str], bytes]:
        deadline = time.perf_counter() + timeout
        data = b""

        while b"\r\n\r\n" not in data and len(data) < max_bytes:
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError
            sock.settimeout(min(remaining, 0.25))
            chunk = sock.recv(4096)
            if not chunk:
                break
            data += chunk

        if not data:
            return 0, {}, b""

        header_blob, sep, rest = data.partition(b"\r\n\r\n")
        header_text = header_blob.decode("iso-8859-1", errors="replace")
        lines = header_text.split("\r\n")
        status = 0
        if lines:
            parts = lines[0].split(" ", 2)
            if len(parts) >= 2 and parts[1].isdigit():
                status = int(parts[1])

        headers: dict[str, str] = {}
        for line in lines[1:]:
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

        body = rest
        content_length = headers.get("Content-Length")
        if content_length and content_length.isdigit():
            target = int(content_length)
            while (
                len(body) < target
                and len(header_blob) + len(sep) + len(body) < max_bytes
            ):
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                sock.settimeout(min(remaining, 0.25))
                try:
                    chunk = sock.recv(min(4096, target - len(body)))
                except TimeoutError:
                    break
                if not chunk:
                    break
                body += chunk
        else:
            while len(header_blob) + len(sep) + len(body) < max_bytes:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                sock.settimeout(min(remaining, 0.1))
                try:
                    chunk = sock.recv(4096)
                except TimeoutError:
                    break
                if not chunk:
                    break
                body += chunk

        return status, headers, body[:max_bytes]

    def _sync_raw_http_request(
        self,
        request_bytes: bytes,
        timeout: float,
        read_response: bool = True,
        reset_after_send: bool = False,
    ) -> RawResponse:
        t0 = time.perf_counter()
        sock: socket.socket | None = None
        try:
            sock = self._make_socket(timeout)
            sock.sendall(request_bytes)

            if reset_after_send:
                sock.setsockopt(
                    socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
                )
                sock.close()
                return RawResponse(
                    status=0,
                    headers={},
                    body="",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    error="CLIENT_RESET",
                )

            if not read_response:
                return RawResponse(
                    status=0,
                    headers={},
                    body="",
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                )

            status, headers, body = self._recv_raw_http_response(sock, timeout)
            return RawResponse(
                status=status,
                headers=headers,
                body=body.decode("utf-8", errors="replace"),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    def _sync_capture_raw_exchange(
        self, request_bytes: bytes, timeout: float, max_bytes: int = 256 * 1024
    ) -> RawResponse:
        t0 = time.perf_counter()
        sock: socket.socket | None = None
        try:
            sock = self._make_socket(timeout)
            sock.sendall(request_bytes)
            sock.settimeout(min(timeout, 0.1))

            data = b""
            deadline = time.perf_counter() + timeout
            while len(data) < max_bytes:
                remaining = deadline - time.perf_counter()
                if remaining <= 0:
                    break
                sock.settimeout(min(remaining, 0.1))
                try:
                    chunk = sock.recv(4096)
                except TimeoutError:
                    break
                if not chunk:
                    break
                data += chunk

            status = 0
            headers: dict[str, str] = {}
            if data:
                header_blob = data.split(b"\r\n\r\n", 1)[0].decode(
                    "iso-8859-1", errors="replace"
                )
                lines = header_blob.split("\r\n")
                if lines:
                    parts = lines[0].split(" ", 2)
                    if len(parts) >= 2 and parts[1].isdigit():
                        status = int(parts[1])
                for line in lines[1:]:
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    headers[key.strip()] = value.strip()
            return RawResponse(
                status=status,
                headers=headers,
                body=data.decode("utf-8", errors="replace"),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
            )
        finally:
            if sock is not None:
                try:
                    sock.close()
                except Exception:
                    pass

    # ---- core sync methods (run in threadpool) ----

    def _sync_request(
        self,
        method: str,
        path: str,
        data: bytes | None,
        headers: dict[str, str],
        timeout: float,
    ) -> RawResponse:
        """Send an arbitrary HTTP request to an arbitrary path."""
        t0 = time.perf_counter()
        try:
            conn = self._make_conn(timeout)
            conn.request(method, path, body=data, headers=headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = (time.perf_counter() - t0) * 1000
            resp_headers = dict(resp.getheaders())
            conn.close()
            return RawResponse(
                status=resp.status,
                headers=resp_headers,
                body=body,
                elapsed_ms=elapsed,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
            )

    def _sync_raw_socket_hold(
        self, timeout: float, send_partial: bool
    ) -> RawResponse:
        """Open a raw TCP connection. Optionally send partial HTTP headers, then hold open."""
        t0 = time.perf_counter()
        try:
            s = self._make_socket(5)
            if send_partial:
                s.sendall(
                    b"POST " + self._chat_path.encode() + b" HTTP/1.1\r\n"
                )
                s.sendall(f"Host: {self._host}\r\n".encode())
                # Deliberately don't send \r\n to end headers or any body
            self._held_sockets.append(s)
            time.sleep(timeout)
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=None,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
            )

    def _sync_post(
        self, data: bytes, headers: dict[str, str], timeout: float
    ) -> RawResponse:
        t0 = time.perf_counter()
        try:
            conn = self._make_conn(timeout)
            conn.request("POST", self._chat_path, body=data, headers=headers)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            elapsed = (time.perf_counter() - t0) * 1000
            resp_headers = dict(resp.getheaders())
            conn.close()
            return RawResponse(
                status=resp.status,
                headers=resp_headers,
                body=body,
                elapsed_ms=elapsed,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
            )

    def _sync_streaming(
        self,
        data: bytes,
        headers: dict[str, str],
        timeout: float,
        cancel_after_chunks: int | None,
        cancel_after_ms: float | None,
    ) -> RawResponse:
        t0 = time.perf_counter()
        chunks = []
        cancelled = False
        try:
            conn = self._make_conn(timeout)
            conn.request("POST", self._chat_path, body=data, headers=headers)
            resp = conn.getresponse()

            if resp.status != 200:
                body = resp.read().decode("utf-8", errors="replace")
                conn.close()
                return RawResponse(
                    status=resp.status,
                    headers=dict(resp.getheaders()),
                    body=body,
                    elapsed_ms=(time.perf_counter() - t0) * 1000,
                    chunks=[],
                )

            cancel_time = (
                time.perf_counter() + (cancel_after_ms / 1000)
                if cancel_after_ms
                else None
            )
            buf = b""
            if cancel_time and getattr(conn, "sock", None) is not None:
                conn.sock.settimeout(min(timeout, 0.1))

            if cancel_after_chunks is not None and cancel_after_chunks == 0:
                cancelled = True

            while not cancelled:
                try:
                    chunk = resp.read(1024)
                except TimeoutError:
                    if cancel_time and time.perf_counter() >= cancel_time:
                        cancelled = True
                        break
                    continue
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    decoded = line.decode("utf-8", errors="replace").strip()
                    if decoded.startswith("data:"):
                        payload_str = decoded[5:].strip()
                        chunks.append(payload_str)
                        if payload_str == "[DONE]":
                            break
                    if (
                        cancel_after_chunks is not None
                        and len(chunks) >= cancel_after_chunks
                    ):
                        cancelled = True
                        break
                    if cancel_time and time.perf_counter() >= cancel_time:
                        cancelled = True
                        break
                if cancelled or (chunks and chunks[-1] == "[DONE]"):
                    break

            conn.close()
            return RawResponse(
                status=resp.status,
                headers=dict(resp.getheaders()),
                body="\n".join(chunks),
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                chunks=chunks,
                cancelled=cancelled,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
                chunks=chunks,
                cancelled=True,
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
                chunks=chunks,
                cancelled=cancelled,
            )

    def _sync_slow_body(
        self,
        data: bytes,
        headers: dict[str, str],
        chunk_delay: float,
        chunk_size: int,
        timeout: float,
    ) -> RawResponse:
        t0 = time.perf_counter()
        try:
            conn = self._make_conn(timeout)
            conn.putrequest("POST", self._chat_path)
            for k, v in headers.items():
                conn.putheader(k, v)
            conn.putheader("Content-Length", str(len(data)))
            conn.endheaders()
            for i in range(0, len(data), chunk_size):
                conn.send(data[i : i + chunk_size])
                time.sleep(chunk_delay)
            resp = conn.getresponse()
            body = resp.read().decode("utf-8", errors="replace")
            conn.close()
            return RawResponse(
                status=resp.status,
                headers=dict(resp.getheaders()),
                body=body,
                elapsed_ms=(time.perf_counter() - t0) * 1000,
            )
        except TimeoutError:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error="TIMEOUT",
            )
        except Exception as e:
            return RawResponse(
                status=0,
                headers={},
                body="",
                elapsed_ms=(time.perf_counter() - t0) * 1000,
                error=str(e)[:500],
            )

    # ---- async wrappers ----

    async def post_json(
        self,
        payload: Any,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> RawResponse:
        if isinstance(payload, dict):
            data = json.dumps(payload).encode()
        elif isinstance(payload, str):
            data = payload.encode("utf-8", errors="replace")
        else:
            data = payload
        hdrs = self._base_headers(headers)
        hdrs["Content-Length"] = str(len(data))
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_post,
            data,
            hdrs,
            timeout or self.config.timeout,
        )

    async def post_raw_bytes(
        self,
        data: bytes,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> RawResponse:
        hdrs = self._base_headers(headers)
        hdrs["Content-Length"] = str(len(data))
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_post,
            data,
            hdrs,
            timeout or self.config.timeout,
        )

    async def raw_http_request(
        self,
        request_bytes: bytes,
        *,
        timeout: float | None = None,
        read_response: bool = True,
        reset_after_send: bool = False,
    ) -> RawResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_raw_http_request,
            request_bytes,
            timeout or self.config.timeout,
            read_response,
            reset_after_send,
        )

    async def post_raw_http(
        self,
        body: bytes,
        *,
        path: str | None = None,
        header_items: list[tuple[str, str]] | None = None,
        timeout: float | None = None,
        include_content_type: bool = True,
        include_auth: bool = True,
        auto_content_length: bool = True,
        content_length: int | None = None,
        read_response: bool = True,
        reset_after_send: bool = False,
    ) -> RawResponse:
        items = self._base_header_items(
            include_content_type=include_content_type, include_auth=include_auth
        )
        if header_items:
            items.extend(header_items)
        if auto_content_length:
            items.append(("Content-Length", str(len(body))))
        elif content_length is not None:
            items.append(("Content-Length", str(content_length)))

        request_bytes = self._format_raw_request(
            "POST", path or self._chat_path, items, body
        )
        return await self.raw_http_request(
            request_bytes,
            timeout=timeout,
            read_response=read_response,
            reset_after_send=reset_after_send,
        )

    async def send_pipelined_requests(
        self,
        request_bytes_list: list[bytes],
        *,
        timeout: float | None = None,
    ) -> RawResponse:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_capture_raw_exchange,
            b"".join(request_bytes_list),
            timeout or self.config.timeout,
        )

    async def post_streaming(
        self,
        payload: dict[str, Any],
        *,
        cancel_after_chunks: int | None = None,
        cancel_after_ms: float | None = None,
        read_timeout: float | None = None,
    ) -> RawResponse:
        payload = dict(payload)
        payload["stream"] = True
        data = json.dumps(payload).encode()
        hdrs = self._base_headers()
        hdrs["Content-Length"] = str(len(data))
        hdrs["Accept"] = "text/event-stream"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_streaming,
            data,
            hdrs,
            read_timeout or self.config.timeout,
            cancel_after_chunks,
            cancel_after_ms,
        )

    async def post_slow_body(
        self,
        payload: dict[str, Any],
        *,
        chunk_delay: float = 1.0,
        chunk_size: int = 10,
    ) -> RawResponse:
        data = json.dumps(payload).encode()
        hdrs = self._base_headers()
        total_time = len(data) / chunk_size * chunk_delay + self.config.timeout
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_slow_body,
            data,
            hdrs,
            chunk_delay,
            chunk_size,
            total_time,
        )

    async def concurrent_requests(
        self,
        payloads: list[dict[str, Any]],
        max_concurrent: int | None = None,
        timeout: float | None = None,
    ) -> list[RawResponse]:
        """Sends ``payloads`` concurrently, capped at ``max_concurrent``.

        ``timeout`` overrides the run's per-request timeout, which the default
        30s is for interactive-sized requests. A batch of multi-hundred-
        megapixel image requests needs longer, and without an override every
        one of them reports TIMEOUT -- indistinguishable from the hang such a
        batch is usually sent to look for.
        """
        sem = asyncio.Semaphore(max_concurrent or self.config.max_concurrency)

        async def _send(p: dict[str, Any]) -> RawResponse:
            async with sem:
                return await self.post_json(p, timeout=timeout)

        return await asyncio.gather(*[_send(p) for p in payloads])

    async def get_path(
        self,
        path: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> RawResponse:
        """GET request to an arbitrary path."""
        hdrs = self._base_headers(headers)
        hdrs.pop("Content-Type", None)
        full_path = f"{self._base_path}{path}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_request,
            "GET",
            full_path,
            None,
            hdrs,
            timeout or self.config.timeout,
        )

    async def method_to_path(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | bytes | None = None,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> RawResponse:
        """Send an arbitrary HTTP method to an arbitrary path."""
        data = None
        hdrs = self._base_headers(headers)
        if payload is not None:
            data = (
                json.dumps(payload).encode()
                if isinstance(payload, dict)
                else payload
            )
            hdrs["Content-Length"] = str(len(data))
        else:
            hdrs.pop("Content-Type", None)
        full_path = f"{self._base_path}{path}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_request,
            method,
            full_path,
            data,
            hdrs,
            timeout or self.config.timeout,
        )

    async def post_to_path(
        self,
        path: str,
        payload: dict[str, Any] | bytes,
        *,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> RawResponse:
        """POST to an arbitrary path (not just /v1/chat/completions)."""
        data = (
            json.dumps(payload).encode()
            if isinstance(payload, dict)
            else payload
        )
        hdrs = self._base_headers(headers)
        hdrs["Content-Length"] = str(len(data))
        full_path = f"{self._base_path}{path}"
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_request,
            "POST",
            full_path,
            data,
            hdrs,
            timeout or self.config.timeout,
        )

    async def raw_socket_hold(
        self, *, timeout: float | None = None, send_partial: bool = True
    ) -> RawResponse:
        """Open a raw TCP connection and hold it open for connection exhaustion testing."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._pool,
            self._sync_raw_socket_hold,
            timeout or self.config.timeout,
            send_partial,
        )

    async def health_check(self) -> RawResponse:
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        return await self.post_json(payload)
