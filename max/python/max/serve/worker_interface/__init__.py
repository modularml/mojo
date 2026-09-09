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

import asyncio
import logging
from collections.abc import AsyncGenerator
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from typing import Generic

from max.pipelines.context import BaseContextType
from max.pipelines.modeling.types import (
    PipelineOutputType,
    RequestID,
)
from max.serve.queue import MAXPullQueue, MAXPushQueue
from max.serve.scheduler_result import SchedulerResult
from max.serve.telemetry.metrics import METRICS

logger = logging.getLogger("max.serve")

from abc import ABC, abstractmethod


class RequestQueueFull(Exception):
    """Raised when the model-worker request queue is at capacity.

    Surfaced by :meth:`ModelWorkerProxy.stream` when the bounded request queue
    (``Settings.max_queue_size``) has no room for a new request. The API layer
    maps this to an HTTP 429 so callers retry rather than letting the queue to
    the worker grow without bound. Because the push is the admission gate
    (awaited before any response status is committed), the rejection surfaces as
    a clean status code even for streaming requests.
    """


async def sleep_with_backoff(count_no_progress: int) -> None:
    """A basic strategy to avoid busy waiting.

    This function sleeps with a linear backoff.
    The first sleep of 0 enables other async threads to run but otherwise does not sleep.
    The step size is 1ms because of limitations around asyncio to sleep with finer granularity.
    The maximum sleep is 10ms because it resolves CPU usage overhead while maintaining minimal waiting.
    """

    ms_to_sleep = min(max(0, count_no_progress), 10)
    await asyncio.sleep(ms_to_sleep * 0.001)


class ModelWorkerProxy(ABC, Generic[BaseContextType, PipelineOutputType]):
    """Held by API worker to communicate with model worker"""

    # Running count of requests accepted by the API server but not yet handed
    # off to this worker (the ingress backlog: tokenization / pre-submit).
    # Maintained by ``note_awaiting_admission``; implementations with a
    # periodic loop (e.g. the zmq proxy's response worker) sample it into the
    # ``maxserve.requests_awaiting_admission`` histogram. Declared as a
    # class-level default; the ``+=`` in ``note_awaiting_admission`` rebinds it
    # as a per-instance attribute on first use (int is immutable, so instances
    # never share state).
    _awaiting_admission_count: int = 0

    def note_awaiting_admission(self, delta: int) -> None:
        """Adjust the ingress backlog (API-side, not yet handed to the worker).

        Call with ``1`` when a request is accepted by the API server (before
        tokenization) and ``-1`` once it is handed off to the worker. Updates
        both the live ``maxserve.num_requests_awaiting_admission`` up/down
        counter and the running count that implementations sample into the
        ``maxserve.requests_awaiting_admission`` histogram.
        """
        self._awaiting_admission_count += delta
        METRICS.reqs_awaiting_admission(delta)

    async def wait_until_connected(self, timeout_s: float | None) -> None:
        """Block until the proxy is ready to accept requests.

        Transports that establish a connection to the worker (e.g. ZMQ)
        override this to wait for that handshake at startup, before the server
        serves traffic, so runtime admission never has to distinguish "not
        connected yet" from a genuinely full queue. ``timeout_s`` of ``None``
        waits indefinitely. The default is a no-op for transports that are
        ready as soon as they are constructed.
        """
        return

    @abstractmethod
    async def stream(
        self,
        req_id: RequestID,
        data: BaseContextType,
    ) -> AsyncGenerator[tuple[list[PipelineOutputType], int | None], None]:
        """Submit ``data`` to the model worker and return a response generator.

        Awaiting this coroutine performs the handoff to the model worker (for
        example, the request-queue put). A submission failure — such as a dead
        worker — raises here, before any response has been streamed, so callers
        can surface it as an error before response headers are sent. The
        returned async generator yields ``(outputs, batch_id)`` pairs: a batch
        of pipeline outputs and the monotonic batch counter from the scheduler
        that produced them (``None`` for cancelled results).
        """
        ...

    @abstractmethod
    def cancel(self, req_id: RequestID) -> None:
        pass


@dataclass
class WorkerQueues(Generic[BaseContextType, PipelineOutputType]):
    request_queue: MAXPullQueue[BaseContextType]
    response_queue: MAXPushQueue[
        dict[RequestID, SchedulerResult[PipelineOutputType]]
    ]
    cancel_queue: MAXPullQueue[list[RequestID]]


class ModelWorkerInterface(ABC, Generic[BaseContextType, PipelineOutputType]):
    """Abstract Base Class for the communication mechanism between API and Model workers

    This needs to be picklable so it can passed to the worker subprocess

    We use AsyncContextManager to "open" the connection on either end
    giving full control to boot up or shutdown resources, or exit prematurely with errors
    """

    @abstractmethod
    def model_worker_proxy(
        self,
    ) -> AbstractAsyncContextManager[
        ModelWorkerProxy[BaseContextType, PipelineOutputType]
    ]:
        """Called by API worker to communicate with model worker"""

    @abstractmethod
    def model_worker_queues(
        self,
    ) -> AbstractAsyncContextManager[
        WorkerQueues[BaseContextType, PipelineOutputType]
    ]:
        """Called by model worker to get work IO streams"""
