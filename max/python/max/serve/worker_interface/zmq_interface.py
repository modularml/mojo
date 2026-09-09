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
import contextlib
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any, Generic

from max.pipelines.context import (
    AudioContext,
    BaseContextType,
    PixelContext,
    TextContext,
)
from max.pipelines.context.exceptions import InputError
from max.pipelines.modeling.types import (
    EmbeddingsContext,
    PipelineOutputType,
    PipelineTask,
    RequestID,
)
from max.serve.queue import MAXAsyncPullQueue, MAXAsyncPushQueue
from max.serve.scheduler_result import SchedulerResult
from max.serve.telemetry.metrics import METRICS
from max.serve.worker_interface import (
    ModelWorkerInterface,
    ModelWorkerProxy,
    RequestQueueFull,
    WorkerQueues,
)
from max.serve.worker_interface._zmq_queue import ZmqConfig

logger = logging.getLogger("max.serve")

_BACKLOG_SAMPLE_INTERVAL_S = 1.0


class ZmqModelWorkerProxy(
    Generic[BaseContextType, PipelineOutputType],
    ModelWorkerProxy[BaseContextType, PipelineOutputType],
):
    def __init__(
        self,
        request_queue: MAXAsyncPushQueue[BaseContextType],
        response_queue: MAXAsyncPullQueue[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ],
        cancel_queue: MAXAsyncPushQueue[list[RequestID]],
    ):
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.cancel_queue = cancel_queue
        # Serializes admission (the writability probe + push) so the probe is
        # authoritative: only one request is admitted at a time, and since the
        # worker only ever frees request-queue slots, a positive probe
        # guarantees the subsequent push does not block.
        self._admission_lock = asyncio.Lock()

        # Each queued item is ``(enqueue_monotonic_s, result)`` so the
        # streaming layer can measure how long the response waited in the
        # output queue (the egress backlog). The timestamp is attached
        # API-side here, not on ``SchedulerResult`` (which is a msgspec wire
        # type serialized from the model worker).
        self.pending_out_queues: dict[
            RequestID,
            asyncio.Queue[tuple[float, SchedulerResult[PipelineOutputType]]],
        ] = {}

    def egress_backlog(self) -> int:
        """Total responses buffered across all pending output queues."""
        return sum(q.qsize() for q in self.pending_out_queues.values())

    async def wait_until_connected(self, timeout_s: float | None) -> None:
        """Block until the request queue's PUSH/PULL handshake completes.

        Called once at startup (before the API server accepts connections) so
        that at request time an unwritable socket unambiguously means the queue
        is *full* -- never merely "not connected yet." This keeps the runtime
        admission check (:meth:`stream`) a pure, immediate backpressure signal.
        ``timeout_s`` of ``None`` waits indefinitely.

        Raises:
            RuntimeError: If the worker does not connect within ``timeout_s``.
        """
        if not await self.request_queue.writable(timeout_s=timeout_s):
            within = "" if timeout_s is None else f" within {timeout_s:g}s"
            raise RuntimeError(
                f"Model worker request queue did not connect{within}."
            )

    async def stream(
        self, req_id: RequestID, data: BaseContextType
    ) -> AsyncGenerator[tuple[list[PipelineOutputType], int | None], None]:
        """Submit a request to the model worker and return a response generator.

        Awaiting this coroutine registers an output queue for ``req_id`` and
        puts ``data`` on the request queue (the handoff to the model worker).
        The push is the admission gate: if the bounded request queue
        (``Settings.max_queue_size``) has no room, :class:`RequestQueueFull` is
        raised *immediately* (before any response is streamed and with no
        registration left behind), so the API can shed load with HTTP 429
        without adding latency to the rejected request. The returned async
        generator drains responses until the request completes.

        The check is a point-in-time writability probe rather than a timed
        blocking send: connectivity is established once at startup (see
        :meth:`wait_until_connected`), so an unwritable socket here means the
        queue is full, not that the worker is still connecting.

        The yielded lists are guaranteed to be non-empty and ordered.

        Raises:
            RuntimeError: If a queue for the given ``req_id`` already exists,
                indicating a duplicate request.
            RequestQueueFull: If the worker request queue is at capacity.
        """
        if req_id in self.pending_out_queues:
            raise RuntimeError(
                f"Detected multiple requests with `req_id` set to {req_id}. "
                "This WILL lead to unexpected behavior! "
                "Please ensure that the `req_id` is unique for each request."
            )

        # Admission gate. Probe writability and push under a lock so the probe
        # is authoritative: a full request queue (the worker has stopped
        # draining under load) is rejected immediately, and because only one
        # request is admitted at a time and the worker only frees slots, the
        # push after a positive probe does not block. (A bare non-blocking send
        # is unsafe here: with a finite high-water mark a multipart send can
        # accept some frames and then EAGAIN, desyncing the stream.)
        async with self._admission_lock:
            if not await self.request_queue.writable():
                raise RequestQueueFull(
                    f"Model worker request queue is full; rejecting {req_id}."
                )

            out_queue: asyncio.Queue[
                tuple[float, SchedulerResult[PipelineOutputType]]
            ] = asyncio.Queue()
            self.pending_out_queues[req_id] = out_queue
            try:
                await self.request_queue.put(data)
            except BaseException:
                # Submission failed before any response streamed; roll back the
                # registration and cancel so the worker drops partial state.
                del self.pending_out_queues[req_id]
                with contextlib.suppress(Exception):
                    self.cancel(req_id)
                raise

        return self._drain_responses(req_id, out_queue)

    async def _drain_responses(
        self,
        req_id: RequestID,
        queue: asyncio.Queue[tuple[float, SchedulerResult[PipelineOutputType]]],
    ) -> AsyncGenerator[tuple[list[PipelineOutputType], int | None], None]:
        """Drain the output queue for a submitted request until it completes.

        Yields ``(outputs, batch_id)`` pairs. ``batch_id`` is the monotonic
        forward-pass counter from the scheduler that produced the outputs, used
        by upstream callers to correlate OTel spans across the API and model
        worker processes. Cleans up the pending output queue on exit and cancels
        the request with the worker if the stream is abandoned before completing
        normally.
        """
        try:
            # queue.get() will wait until an item is available.
            # This will exit when no result is passed in the SchedulerResult.
            # or the SchedulerResult states that we should stop the stream.
            while True:
                enqueue_s, item = await queue.get()
                # Record how long this head-of-line response waited in the
                # output queue. Sampled once per consumer wake (not on the
                # get_nowait drain below) to bound metric volume while still
                # capturing egress congestion: the head item is the oldest
                # waiter, so this is the per-wake worst-case wait.
                METRICS.response_queue_time(
                    (time.monotonic() - enqueue_s) * 1000
                )
                if item.result is None:
                    # The route layer turns InputError into a client-facing 400.
                    if item.error is not None:
                        raise InputError(item.error)
                    break

                outputs = [item.result]
                batch_id = item.batch_id
                should_stop = item.is_done
                while True:
                    try:
                        _, item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break

                    if item.result is None:
                        if item.error is not None:
                            raise InputError(item.error)
                        should_stop = True
                        break

                    outputs.append(item.result)
                    if item.batch_id is not None:
                        batch_id = item.batch_id
                    if item.is_done:
                        should_stop = True
                        break

                yield outputs, batch_id

                if should_stop:
                    break
        except BaseException:
            # The consumer abandoned the stream (e.g. client disconnect) or an
            # error occurred mid-stream; tell the worker to stop generating.
            with contextlib.suppress(Exception):
                self.cancel(req_id)
            raise
        finally:
            del self.pending_out_queues[req_id]

    def cancel(self, req_id: RequestID) -> None:
        """
        Cancel a specific request by its ID (non-blocking).

        This method sends a cancellation message to the worker for the given request ID.

        Args:
            req_id: The unique identifier of the request to cancel.
        """
        self.cancel_queue.put_nowait([req_id])

    async def response_worker(self) -> None:
        """Awaits responses from the model worker and routes them to pending output queues."""
        while True:
            response_dict = await self.response_queue.get()
            for request_id, response in response_dict.items():
                if request_id in self.pending_out_queues:
                    await self.pending_out_queues[request_id].put(
                        (time.monotonic(), response)
                    )

    async def _metrics_worker(self) -> None:
        """Periodically samples backlog gauges and histograms."""
        while True:
            await asyncio.sleep(_BACKLOG_SAMPLE_INTERVAL_S)
            backlog = self.egress_backlog()
            METRICS.responses_buffered(backlog)
            METRICS.responses_buffered_dist(backlog)
            METRICS.requests_awaiting_admission_dist(
                self._awaiting_admission_count
            )


def _response_type_for_task(
    pipeline_task: PipelineTask,
) -> type[Any]:
    """Maps a PipelineTask to the correct msgspec response type for ZMQ deserialization."""
    from max.pipelines.context import TextGenerationOutput
    from max.pipelines.context.outputs import GenerationOutput
    from max.pipelines.modeling.types.pipeline_variants import (
        EmbeddingsGenerationOutput,
    )

    if pipeline_task == PipelineTask.TEXT_GENERATION:
        return dict[RequestID, SchedulerResult[TextGenerationOutput]]
    elif pipeline_task == PipelineTask.EMBEDDINGS_GENERATION:
        return dict[RequestID, SchedulerResult[EmbeddingsGenerationOutput]]
    elif pipeline_task in (
        PipelineTask.PIXEL_GENERATION,
        PipelineTask.AUDIO_GENERATION,
    ):
        return dict[RequestID, SchedulerResult[GenerationOutput]]
    else:
        raise ValueError(
            f"PipelineTask ({pipeline_task}) does not have a response type defined."
        )


class ZmqModelWorkerInterface(
    Generic[BaseContextType, PipelineOutputType],
    ModelWorkerInterface[BaseContextType, PipelineOutputType],
):
    def __init__(
        self,
        pipeline_task: PipelineTask,
        context_type: type[
            TextContext | EmbeddingsContext | PixelContext | AudioContext
        ],
        request_queue_size: int | None = None,
    ) -> None:
        response_type = _response_type_for_task(pipeline_task)

        # Bound the request queue (cap N) so an overloaded worker exerts
        # backpressure: once the queue is full, ``stream`` raises
        # ``RequestQueueFull`` and the API rejects with HTTP 429 instead of
        # letting the backlog to the worker grow without bound.
        self.request_queue_config = ZmqConfig[BaseContextType](
            context_type, high_water_mark=request_queue_size
        )
        self.response_queue_config = ZmqConfig[
            dict[RequestID, SchedulerResult[PipelineOutputType]]
        ](response_type)
        self.cancel_queue_config = ZmqConfig[list[RequestID]](list[RequestID])

    @contextlib.asynccontextmanager
    async def model_worker_queues(
        self,
    ) -> AsyncGenerator[WorkerQueues[BaseContextType, PipelineOutputType]]:
        yield WorkerQueues[BaseContextType, PipelineOutputType](
            request_queue=self.request_queue_config.pull(),
            response_queue=self.response_queue_config.push(),
            cancel_queue=self.cancel_queue_config.pull(),
        )

    @contextlib.asynccontextmanager
    async def model_worker_proxy(
        self,
    ) -> AsyncGenerator[
        ZmqModelWorkerProxy[BaseContextType, PipelineOutputType]
    ]:
        proxy = ZmqModelWorkerProxy(
            self.request_queue_config.async_push(),
            self.response_queue_config.async_pull(),
            self.cancel_queue_config.async_push(),
        )
        worker_task = asyncio.create_task(proxy.response_worker())
        metrics_task = asyncio.create_task(proxy._metrics_worker())
        try:
            yield proxy
        finally:
            worker_task.cancel()
            metrics_task.cancel()
