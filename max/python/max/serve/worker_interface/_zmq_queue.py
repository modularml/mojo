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

"""Multi-process queue based on ZeroMQ. Tested for SPSC case."""

from __future__ import annotations

import logging
import queue
import tempfile
import urllib.parse
import uuid
import weakref
from collections.abc import Callable
from typing import Any, Generic, NewType, TypeVar, overload

import psutil
import zmq
import zmq.asyncio
from max.pipelines.modeling.types import (
    msgpack_numpy_decoder,
    msgpack_numpy_encoder,
    msgpack_numpy_oob_decoder,
    msgpack_numpy_oob_encoder,
)
from max.serve.queue import MAXPullQueue, MAXPushQueue

logger = logging.getLogger("max.serve")

T = TypeVar("T")

Request = TypeVar("Request")
Reply = TypeVar("Reply")

# Encoder for the inter-node Router/Dealer sockets, which cannot use the
# intra-node out-of-band path. It never uses shared memory (which does not work
# across nodes); large numpy arrays do not cross these sockets anyway.
NON_SHARED_MSGPACK_NUMPY_ENCODER = msgpack_numpy_encoder()


# Maximum number of characters a caller may append to a base IPC path
# returned by generate_zmq_ipc_path (a leading "-" plus a socket-name
# suffix). The longest in-tree suffix is "-reset_prefix_cache" (19 chars);
# LoRA adds "-lora_response" (14). We reserve a conservative budget so the
# fully-suffixed socket path stays under zmq.IPC_PATH_MAX_LEN even when the
# platform temp dir is long.
_MAX_IPC_SUFFIX_LEN = 24


def generate_zmq_ipc_path() -> str:
    """Generate a unique ZMQ IPC path."""
    # The full UUID is 36 chars (8-4-4-4-12 hex)
    # However, this may cause the full path to be too long for ZMQ if you append
    # additional characters to it. As such, we truncate the UUID to 18 chars.
    # The chances of collision are still very low, because we don't really make
    # that many ZMQs anyways.
    short_uuid = uuid.uuid4().hex[:18]
    base_rpc_path = tempfile.gettempdir()
    # An ipc:// socket binds to a filesystem path bounded by
    # zmq.IPC_PATH_MAX_LEN (107 on Linux, the sun_path limit). On hosts with a
    # long temp dir (e.g. the BuildBuddy macOS remote-build sandbox, whose
    # gettempdir() is ~80 chars) the base path plus a caller suffix overflows
    # that limit. Fall back to a short, writable directory so the
    # fully-suffixed path always fits.
    suffix_budget = len("/") + len(short_uuid) + _MAX_IPC_SUFFIX_LEN
    if len(base_rpc_path) + suffix_budget > zmq.IPC_PATH_MAX_LEN:
        base_rpc_path = "/tmp"
    return f"ipc://{base_rpc_path}/{short_uuid}"


def _validate_zmq_address(address: str) -> None:
    """
    Check if a ZMQ address is valid.
    """
    # Check for supported protocols
    if not address.startswith(("tcp://", "ipc://", "inproc://")):
        raise ValueError(
            f"ZMQ address must start with tcp://, ipc://, or inproc://. Found: {address}"
        )

    # Protocol-specific validation
    if address.startswith("tcp://"):
        # TCP requires host:port format, including bracketed IPv6
        # e.g. tcp://host:port or tcp://[2001:db8::1]:port
        parsed = urllib.parse.urlparse(address)
        if not parsed.hostname:
            raise ValueError(
                f"ZMQ tcp address must be in the format"
                f" tcp://host:port or tcp://[ipv6]:port."
                f" Found: {address}"
            )
        try:
            port = parsed.port
        except ValueError:
            raise ValueError(
                f"ZMQ tcp port must be a number. Found: {address}"
            ) from None
        if port is None:
            raise ValueError(
                f"ZMQ tcp address must be in the format"
                f" tcp://host:port or tcp://[ipv6]:port."
                f" Found: {address}"
            )
        if not (1 <= port <= 65535):
            raise ValueError(
                f"ZMQ tcp port must be between 1 and 65535. Found: {port}"
            )
    elif address.startswith("ipc://"):
        # On linux, IPC_PATH_MAX_LEN is 107.
        # This is the length of `char sun_path[108]` field of `struct sockaddr_un`
        # subtracted by 1 for the null terminator.
        length = len(address) - len("ipc://")
        if length > zmq.IPC_PATH_MAX_LEN:
            raise ValueError(
                f"ZMQ IPC path is too long: {address}.\n"
                f"The maximum length is {zmq.IPC_PATH_MAX_LEN} characters. Found {length} characters."
            )
        if length == 0:
            raise ValueError(
                f"ZMQ IPC requires a path after the protocol. Found: {address}"
            )
    elif address.startswith("inproc://"):
        length = len(address) - len("inproc://")
        if length == 0:
            raise ValueError(
                f"ZMQ inproc requires a name after the protocol. Found: {address}"
            )


# Adapted from:
#  - vllm: https://github.com/vllm-project/vllm/blob/46c759c165a5a985ce62f019bf684e4a6109e41c/vllm/utils.py#L2093
#  - sglang: https://github.com/sgl-project/sglang/blob/efc52f85e2d5c9b31545d4092f2b361b6ff04d67/python/sglang/srt/utils.py#L783
@overload
def _open_zmq_socket(
    path: str,
    mode: int,
    *,
    ctx: zmq.asyncio.Context,
    high_water_mark: int | None = ...,
) -> zmq.asyncio.Socket: ...


@overload
def _open_zmq_socket(
    path: str, mode: int, *, ctx: None = ..., high_water_mark: int | None = ...
) -> zmq.Socket[bytes]: ...


def _open_zmq_socket(
    path: str,
    mode: int,
    *,
    ctx: zmq.asyncio.Context | None = None,
    high_water_mark: int | None = None,
) -> zmq.asyncio.Socket | zmq.Socket[bytes]:
    """Open a ZMQ socket with the proper bind/connect semantics.

    ``high_water_mark`` bounds the socket's high-water mark (messages buffered before
    backpressure). ``None`` keeps the default unbounded behavior (``0``). When
    set on both ends of the request queue it caps the in-transit backlog to the
    worker; ZMQ enforces it approximately (the effective bound is roughly the
    sum of the send and receive HWMs, and ZMQ may round the value).
    """
    mem = psutil.virtual_memory()
    # 0 is ZMQ's sentinel for "unbounded".
    resolved_high_water_mark = 0 if high_water_mark is None else high_water_mark

    resolved_ctx: zmq.asyncio.Context | zmq.Context[zmq.Socket[bytes]] = (
        ctx if ctx is not None else zmq.Context.instance()
    )
    socket = resolved_ctx.socket(mode)

    # Enable IPv6 so that bind/connect works when hostnames resolve to
    # IPv6 addresses (e.g. Kubernetes pods with IPv6-only networking).
    # With IPV6 enabled, the socket still accepts IPv4 connections.
    socket.setsockopt(zmq.IPV6, 1)

    # Calculate buffer size based on system memory
    GIB = 1024**3
    total_mem_gb = mem.total / GIB
    available_mem_gb = mem.available / GIB
    # For systems with substantial memory (>32GB total, >16GB available):
    # - Set a large 0.5GB buffer to improve throughput
    # For systems with less memory:
    if total_mem_gb > 32 and available_mem_gb > 16:
        buf_size = int(0.5 * GIB)
    else:
        buf_size = -1

    # Configure socket options based on type
    if mode == zmq.PULL:
        socket.setsockopt(zmq.RCVHWM, resolved_high_water_mark)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(path)
    elif mode == zmq.PUSH:
        socket.setsockopt(zmq.SNDHWM, resolved_high_water_mark)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.bind(path)
    elif mode == zmq.ROUTER:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.setsockopt(zmq.ROUTER_MANDATORY, 1)
        socket.bind(path)
    elif mode == zmq.DEALER:
        socket.setsockopt(zmq.RCVHWM, 0)
        socket.setsockopt(zmq.SNDHWM, 0)
        socket.setsockopt(zmq.RCVBUF, buf_size)
        socket.setsockopt(zmq.SNDBUF, buf_size)
        socket.setsockopt(zmq.LINGER, 0)
        socket.connect(path)
    else:
        raise ValueError(f"Unknown Socket Mode: {mode}")

    return socket


def _get_helper(func: Callable[[], Any]) -> Any:
    try:
        msg = func()
    except zmq.ZMQError as e:
        if e.errno == zmq.EAGAIN:
            raise queue.Empty() from e
        raise RuntimeError("Failed to get message on ZMQ socket") from e
    return msg


class ZmqConfig(Generic[T]):
    def __init__(
        self, payload_type: Any, high_water_mark: int | None = None
    ) -> None:
        self._payload_type = payload_type
        self._endpoint = generate_zmq_ipc_path()
        # High-water mark applied to both ends of this queue. ``None`` keeps
        # ZMQ's default unbounded buffering; a finite value bounds the
        # in-transit backlog (used for the request queue cap).
        self._high_water_mark = high_water_mark

    def push(self) -> ZmqPushSocket[T]:
        return ZmqPushSocket(
            endpoint=self._endpoint,
            payload_type=self._payload_type,
            high_water_mark=self._high_water_mark,
        )

    def pull(self) -> ZmqPullSocket[T]:
        return ZmqPullSocket(
            endpoint=self._endpoint,
            payload_type=self._payload_type,
            high_water_mark=self._high_water_mark,
        )

    def pair(self) -> tuple[ZmqPushSocket[T], ZmqPullSocket[T]]:
        return self.push(), self.pull()

    def async_push(self) -> ZmqAsyncPushSocket[T]:
        return ZmqAsyncPushSocket(
            endpoint=self._endpoint,
            payload_type=self._payload_type,
            high_water_mark=self._high_water_mark,
        )

    def async_pull(self) -> ZmqAsyncPullSocket[T]:
        return ZmqAsyncPullSocket(
            endpoint=self._endpoint,
            payload_type=self._payload_type,
            high_water_mark=self._high_water_mark,
        )

    def async_pair(self) -> tuple[ZmqAsyncPushSocket[T], ZmqAsyncPullSocket[T]]:
        return self.async_push(), self.async_pull()


class ZmqSocket:
    def __init__(
        self,
        *,
        endpoint: str,
        mode: int,
        high_water_mark: int | None = None,
    ) -> None:
        _validate_zmq_address(endpoint)
        self._endpoint = endpoint
        self._socket = _open_zmq_socket(
            endpoint, mode, high_water_mark=high_water_mark
        )
        self._finalize = weakref.finalize(self, self.close)
        self._is_closed = False

    def close(self) -> None:
        """Clean up resources during garbage collection."""
        if not self._is_closed:
            self._is_closed = True
            self._socket.close()


class ZmqPushSocket(Generic[T], ZmqSocket, MAXPushQueue[T]):
    def __init__(
        self,
        *,
        endpoint: str,
        payload_type: Any,
        high_water_mark: int | None = None,
    ) -> None:
        # Out-of-band, zero-copy serialization: large numpy arrays (e.g.
        # multimodal pixel_values) ride as their own ZMQ frame instead of being
        # copied into the msgpack stream. This supersedes both the shared-memory
        # and tobytes-copy paths for the intra-node request/response queues --
        # it is faster at every size and correct-by-construction (the received
        # frame's lifetime is tied to the decoded array, so there is no
        # /dev/shm segment to size, leak, or clean up).
        self._serialize = msgpack_numpy_oob_encoder()
        super().__init__(
            endpoint=endpoint, mode=zmq.PUSH, high_water_mark=high_water_mark
        )

    def put(self, msg: T) -> None:
        """Send a message, blocking until the peer is ready."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        # `copy=False`: ZMQ holds a reference to each frame's buffer until its
        # I/O thread has sent it, so the source arrays must not be mutated or
        # freed before then. Each request owns fresh arrays, so this holds.
        frames = self._serialize(msg)
        self._socket.send_multipart(frames, copy=False)

    def put_nowait(self, msg: T) -> None:
        """Send a message without blocking; raises zmq.Again if peer isn't connected."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        # See `put` for why `copy=False` is safe. The NOBLOCK EAGAIN lands at
        # message granularity only because these sockets use unbounded HWM
        # (SNDHWM=0): it fails on frame 0 without committing any frame, never
        # mid-multipart. A finite HWM could accept frame 0 and then EAGAIN,
        # which would desync the stream on retry.
        frames = self._serialize(msg)
        self._socket.send_multipart(frames, flags=zmq.NOBLOCK, copy=False)


class ZmqPullSocket(Generic[T], ZmqSocket, MAXPullQueue[T]):
    def __init__(
        self,
        *,
        endpoint: str,
        payload_type: Any,
        high_water_mark: int | None = None,
    ) -> None:
        self._deserialize = msgpack_numpy_oob_decoder(payload_type)
        super().__init__(
            endpoint=endpoint, mode=zmq.PULL, high_water_mark=high_water_mark
        )

    def get_nowait(self) -> T:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        # `copy=False`: received frames are views into ZMQ-owned memory.
        # Out-of-band arrays decode to zero-copy views that keep their frame
        # mapped for the view's lifetime (see _MsgpackNumpyOOBDecoder).
        frames = _get_helper(
            lambda: self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
        )
        msg = self._deserialize(frames)
        return msg


ClientIdentity = NewType("ClientIdentity", bytes)


class ZmqRouterSocket(Generic[Request, Reply], ZmqSocket):
    def __init__(
        self, *, endpoint: str, request_type: Any, reply_type: Any
    ) -> None:
        self._endpoint = endpoint
        # Do not use shm since it does not work for inter-node communication.
        self._serialize = NON_SHARED_MSGPACK_NUMPY_ENCODER
        self._deserialize = msgpack_numpy_decoder(request_type)
        super().__init__(endpoint=endpoint, mode=zmq.ROUTER)

    def send_reply(self, msg: Reply, identity: ClientIdentity) -> None:
        """Send a reply, blocking until the peer is ready."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        self._socket.send_multipart([identity, serialized_msg])

    def send_reply_nowait(self, msg: Reply, identity: ClientIdentity) -> None:
        """Send a reply without blocking; raises zmq.Again if peer isn't connected."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        self._socket.send_multipart(
            [identity, serialized_msg], flags=zmq.NOBLOCK
        )

    def recv_request_nowait(self) -> tuple[Request, ClientIdentity]:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        identity, serialized_msg = _get_helper(
            lambda: self._socket.recv_multipart(flags=zmq.NOBLOCK)
        )
        msg = self._deserialize(serialized_msg)
        return msg, ClientIdentity(identity)


class ZmqDealerSocket(Generic[Request, Reply], ZmqSocket):
    def __init__(
        self, *, endpoint: str, request_type: Any, reply_type: Any
    ) -> None:
        self._endpoint = endpoint
        # Do not use shm since it does not work for inter-node communication.
        self._serialize = NON_SHARED_MSGPACK_NUMPY_ENCODER
        self._deserialize = msgpack_numpy_decoder(reply_type)
        super().__init__(endpoint=endpoint, mode=zmq.DEALER)

    def send_request(self, msg: Request) -> None:
        """Send a request, blocking until the peer is ready."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        self._socket.send(serialized_msg)

    def send_request_nowait(self, msg: Request) -> None:
        """Send a request without blocking; raises zmq.Again if peer isn't connected."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = self._serialize(msg)
        self._socket.send(serialized_msg, flags=zmq.NOBLOCK)

    def recv_reply_nowait(self) -> Reply:
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        serialized_msg = _get_helper(
            lambda: self._socket.recv(flags=zmq.NOBLOCK)
        )
        msg = self._deserialize(serialized_msg)
        return msg


class ZmqAsyncSocket:
    """Base class for async ZMQ sockets using zmq.asyncio."""

    def __init__(
        self,
        *,
        endpoint: str,
        mode: int,
        high_water_mark: int | None = None,
    ) -> None:
        _validate_zmq_address(endpoint)
        self._endpoint = endpoint
        self._socket: zmq.asyncio.Socket = _open_zmq_socket(
            endpoint,
            mode,
            ctx=zmq.asyncio.Context.instance(),
            high_water_mark=high_water_mark,
        )
        self._finalize = weakref.finalize(self, self.close)
        self._is_closed = False

    def close(self) -> None:
        if not self._is_closed:
            self._is_closed = True
            self._socket.close()


class ZmqAsyncPushSocket(Generic[T], ZmqAsyncSocket):
    """Async ZMQ PUSH socket using zmq.asyncio for native event loop integration."""

    def __init__(
        self,
        *,
        endpoint: str,
        payload_type: type[T] | object,
        high_water_mark: int | None = None,
    ) -> None:
        # The async API-server side and the sync model-worker side are the two
        # ends of the same intra-node queues, so this uses the same out-of-band
        # transport as the sync `ZmqPushSocket` (see its `__init__`).
        self._serialize = msgpack_numpy_oob_encoder()
        super().__init__(
            endpoint=endpoint, mode=zmq.PUSH, high_water_mark=high_water_mark
        )

    async def put(self, msg: T) -> None:
        """Send a message, awaiting until the socket is ready."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        # See `ZmqPushSocket.put` for why `copy=False` is safe here.
        frames = self._serialize(msg)
        await self._socket.send_multipart(frames, copy=False)

    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        """Whether a message can be enqueued now without blocking.

        ``True`` when the socket has a connected peer with room below its
        high-water mark; ``False`` when the queue is full (the peer has stopped
        draining) or no peer is connected. ``timeout_s`` bounds how long to
        wait for writability: the default ``0`` is an immediate, point-in-time
        check for the request hot path (shed load the instant the queue backs
        up), a positive value blocks up to that long for the peer to connect
        (used once at startup so the runtime check is never ambiguous about
        connectivity), and ``None`` blocks indefinitely.
        """
        if self._is_closed:
            return False
        # A Poller lets us select POLLOUT with a timeout (the async socket's own
        # poll() does not expose a flags argument in its type stub).
        poller = zmq.asyncio.Poller()
        poller.register(self._socket, zmq.POLLOUT)
        timeout_ms = None if timeout_s is None else int(timeout_s * 1000)
        events = await poller.poll(timeout_ms)
        return any(mask & zmq.POLLOUT for _socket, mask in events)

    def put_nowait(self, msg: T) -> None:
        """Send a message without blocking; raises on EAGAIN."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        # See `ZmqPushSocket.put_nowait`: unbounded SNDHWM keeps a NOBLOCK
        # EAGAIN at message granularity, never mid-multipart.
        frames = self._serialize(msg)
        self._socket.send_multipart(frames, flags=zmq.NOBLOCK, copy=False)


class ZmqAsyncPullSocket(Generic[T], ZmqAsyncSocket):
    """Async ZMQ PULL socket using zmq.asyncio for native event loop integration."""

    def __init__(
        self,
        *,
        endpoint: str,
        payload_type: type[T] | object,
        high_water_mark: int | None = None,
    ) -> None:
        self._deserialize = msgpack_numpy_oob_decoder(payload_type)
        super().__init__(
            endpoint=endpoint, mode=zmq.PULL, high_water_mark=high_water_mark
        )

    async def get(self) -> T:
        """Receive a message, awaiting until one is available."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        # `copy=False`: received frames are views into ZMQ-owned memory that the
        # decoded arrays keep mapped for their lifetime (see `ZmqPullSocket`).
        frames = await self._socket.recv_multipart(copy=False)
        return self._deserialize(frames)

    def get_nowait(self) -> T:
        """Receive a message without blocking; raises queue.Empty if none available."""
        if self._is_closed:
            raise RuntimeError("Socket is closed")
        frames = _get_helper(
            lambda: self._socket.recv_multipart(flags=zmq.NOBLOCK, copy=False)
        )
        return self._deserialize(frames)
