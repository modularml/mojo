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
# ===----------------------------------------------------------------------===
"""ZMQ frontend/backend pair for fetching and resetting EPLB snapshots
across the API-process / worker-process boundary."""

from __future__ import annotations

import asyncio
import queue
import socket as _socket
import time
from dataclasses import dataclass

from max.pipelines.lib.eplb_stats import (
    EplbStatsAccumulator,
    EplbStatsSnapshot,
)
from max.serve.worker_interface._zmq_queue import (
    ZmqDealerSocket,
    ZmqPullSocket,
    ZmqPushSocket,
    ZmqRouterSocket,
)

ZMQ_EPLB_STATS_ENDPOINT = "eplb_stats"
ZMQ_EPLB_STATS_RESET_ENDPOINT = "eplb_stats_reset"
DEFAULT_FETCH_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class EplbStatsRequest:
    """Empty request; the message arriving is the signal."""


class EplbStatsFrontend:
    """API-process side: ask the worker for a snapshot, await reply."""

    def __init__(self, zmq_endpoint_base: str) -> None:
        self._request_socket = ZmqDealerSocket[
            EplbStatsRequest, EplbStatsSnapshot
        ](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_EPLB_STATS_ENDPOINT}",
            request_type=EplbStatsRequest,
            reply_type=EplbStatsSnapshot,
        )
        self._lock = asyncio.Lock()

    async def fetch_snapshot(
        self, timeout_s: float = DEFAULT_FETCH_TIMEOUT_S
    ) -> EplbStatsSnapshot:
        async with self._lock:
            self._request_socket.send_request_nowait(EplbStatsRequest())
            deadline = time.monotonic() + timeout_s
            while deadline > time.monotonic():
                try:
                    return self._request_socket.recv_reply_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.001)
            raise asyncio.TimeoutError(
                f"Timeout waiting for EPLB stats snapshot after {timeout_s}s"
            )


class EplbStatsBackend:
    """Worker-process side: serve any pending request."""

    def __init__(
        self,
        zmq_endpoint_base: str,
        accumulator: EplbStatsAccumulator,
    ) -> None:
        self._reply_socket = ZmqRouterSocket[
            EplbStatsRequest, EplbStatsSnapshot
        ](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_EPLB_STATS_ENDPOINT}",
            request_type=EplbStatsRequest,
            reply_type=EplbStatsSnapshot,
        )
        self._accumulator = accumulator
        self._hostname = _socket.gethostname()

    def serve_pending_requests(self) -> None:
        try:
            _req, identity = self._reply_socket.recv_request_nowait()
        except queue.Empty:
            return

        snapshot = self._accumulator.snapshot(hostname=self._hostname)
        self._reply_socket.send_reply_nowait(snapshot, identity)


class EplbStatsResetFrontend:
    """API-process side: enqueue a fire-and-forget reset request."""

    def __init__(self, zmq_endpoint_base: str) -> None:
        self._socket = ZmqPushSocket[None](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_EPLB_STATS_RESET_ENDPOINT}",
            payload_type=None,
        )

    def enqueue_reset(self) -> None:
        self._socket.put_nowait(None)


class EplbStatsResetBackend:
    """Worker-process side: drain pending resets and zero the accumulator."""

    def __init__(
        self,
        zmq_endpoint_base: str,
        accumulator: EplbStatsAccumulator,
    ) -> None:
        self._socket = ZmqPullSocket[None](
            endpoint=f"{zmq_endpoint_base}-{ZMQ_EPLB_STATS_RESET_ENDPOINT}",
            payload_type=None,
        )
        self._accumulator = accumulator

    def serve_pending_requests(self) -> None:
        # Coalesce any number of queued resets into a single reset.
        reset = False
        while True:
            try:
                self._socket.get_nowait()
                reset = True
            except queue.Empty:
                break
        if reset:
            self._accumulator.reset()


__all__ = [
    "DEFAULT_FETCH_TIMEOUT_S",
    "ZMQ_EPLB_STATS_ENDPOINT",
    "ZMQ_EPLB_STATS_RESET_ENDPOINT",
    "EplbStatsBackend",
    "EplbStatsFrontend",
    "EplbStatsRequest",
    "EplbStatsResetBackend",
    "EplbStatsResetFrontend",
]
