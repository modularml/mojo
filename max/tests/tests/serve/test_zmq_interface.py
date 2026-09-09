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
import time
from collections.abc import AsyncGenerator, Callable
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

import pytest
from max.pipelines.context import BaseContext
from max.pipelines.modeling.types import (
    PipelineOutput,
    RequestID,
)
from max.serve.scheduler_result import SchedulerResult
from max.serve.worker_interface import RequestQueueFull
from max.serve.worker_interface.zmq_interface import ZmqModelWorkerProxy

_T = TypeVar("_T")


class FakeAsyncPushQueue(Generic[_T]):
    """Test double mimicking ZmqAsyncPushSocket interface."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_T] = asyncio.Queue()

    def put_nowait(self, item: _T) -> None:
        self._queue.put_nowait(item)

    async def put(self, item: _T) -> None:
        await self._queue.put(item)

    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        return True

    def get_nowait(self) -> _T:
        return self._queue.get_nowait()

    async def get(self) -> _T:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()


class FakeAsyncPullQueue(Generic[_T]):
    """Test double mimicking ZmqAsyncPullSocket interface."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[_T] = asyncio.Queue()

    async def get(self) -> _T:
        return await self._queue.get()

    def get_nowait(self) -> _T:
        return self._queue.get_nowait()

    def put_nowait(self, item: _T) -> None:
        self._queue.put_nowait(item)

    async def put(self, item: _T) -> None:
        await self._queue.put(item)

    def qsize(self) -> int:
        return self._queue.qsize()


async def wait_until(
    predicate: Callable[[], bool], timeout: float = 15.0, interval: float = 0.1
) -> None:
    start = asyncio.get_event_loop().time()
    while True:
        if predicate():
            return
        if asyncio.get_event_loop().time() - start > timeout:
            raise TimeoutError("Condition not met in time")
        await asyncio.sleep(interval)


@dataclass
class FakeContext:
    name: str


@dataclass
class FakeOutput(PipelineOutput):
    data: str
    _is_done: bool = False

    @property
    def is_done(self) -> bool:
        return self._is_done


@contextlib.asynccontextmanager
async def create_worker_proxy(
    request_queue: FakeAsyncPushQueue[BaseContext],
    response_queue: FakeAsyncPullQueue[
        dict[RequestID, SchedulerResult[FakeOutput]]
    ],
    cancel_queue: FakeAsyncPushQueue[list[RequestID]],
) -> AsyncGenerator[ZmqModelWorkerProxy[BaseContext, FakeOutput], None]:
    proxy = ZmqModelWorkerProxy(request_queue, response_queue, cancel_queue)
    response_worker_task = asyncio.create_task(proxy.response_worker())
    try:
        yield proxy
    finally:
        response_worker_task.cancel()


@pytest.mark.asyncio
async def test_buffering() -> None:
    request_queue: FakeAsyncPushQueue[BaseContext] = FakeAsyncPushQueue()
    response_queue: FakeAsyncPullQueue[
        dict[RequestID, SchedulerResult[FakeOutput]]
    ] = FakeAsyncPullQueue()
    cancel_queue: FakeAsyncPushQueue[list[RequestID]] = FakeAsyncPushQueue()

    async with create_worker_proxy(
        request_queue, response_queue, cancel_queue
    ) as proxy:
        req_id = RequestID("my_request_id")
        fake_context = cast(BaseContext, FakeContext(name="fake context"))

        batches: list[list[FakeOutput]] = []

        async def collect_stream() -> list[list[FakeOutput]]:
            async for batch, _batch_id in await proxy.stream(
                req_id, fake_context
            ):
                batches.append(batch)
            return batches

        collect_stream_task = asyncio.create_task(collect_stream())
        await wait_until(lambda: request_queue.qsize() > 0)
        assert request_queue.get_nowait() == fake_context

        def put(data: str, is_done: bool = False) -> None:
            output = FakeOutput(data, is_done)
            sch_result = SchedulerResult(is_done=is_done, result=output)
            response_queue.put_nowait({req_id: sch_result})

        put("a")

        await wait_until(lambda: len(batches) > 0)
        put("b")
        put("c")

        await wait_until(lambda: len(batches) > 1)
        put("d", is_done=False)
        put("e", is_done=False)
        put("f", is_done=True)
        put("g", is_done=True)
        put("h", is_done=True)

        buffered_stream_outputs = await collect_stream_task

        assert buffered_stream_outputs == [
            [FakeOutput("a")],
            [FakeOutput("b"), FakeOutput("c")],
            [FakeOutput("d"), FakeOutput("e"), FakeOutput("f", _is_done=True)],
        ]


class _FullPushQueue(FakeAsyncPushQueue[_T]):
    """Push queue that reports itself as full.

    Mimics a bounded ZMQ PUSH socket at its high-water mark whose consumer has
    stopped draining: ``writable`` returns False, so admission rejects
    immediately rather than attempting (and blocking on) a push.
    """

    async def writable(self, timeout_s: float | None = 0.0) -> bool:
        return False

    async def put(self, item: _T) -> None:
        raise AssertionError("put must not be called when the queue is full")


def _make_proxy(
    request_queue: FakeAsyncPushQueue[BaseContext] | None = None,
) -> ZmqModelWorkerProxy[BaseContext, FakeOutput]:
    return ZmqModelWorkerProxy(
        request_queue or FakeAsyncPushQueue(),
        FakeAsyncPullQueue(),
        FakeAsyncPushQueue(),
    )


@pytest.mark.asyncio
async def test_stream_raises_request_queue_full() -> None:
    # A full request queue must surface as RequestQueueFull from awaiting
    # stream, immediately (without attempting the push) and before any response
    # is drained, and must not leave a dangling output-queue registration.
    proxy = _make_proxy(request_queue=_FullPushQueue())
    req_id = RequestID("rejected")
    fake_context = cast(BaseContext, FakeContext(name="ctx"))

    with pytest.raises(RequestQueueFull):
        await proxy.stream(req_id, fake_context)

    assert req_id not in proxy.pending_out_queues
    assert len(proxy.pending_out_queues) == 0


@pytest.mark.asyncio
async def test_stream_registers_and_cleans_up() -> None:
    request_queue: FakeAsyncPushQueue[BaseContext] = FakeAsyncPushQueue()
    proxy = _make_proxy(request_queue=request_queue)
    req_id = RequestID("ok")
    fake_context = cast(BaseContext, FakeContext(name="ctx"))

    generator = await proxy.stream(req_id, fake_context)
    # Awaiting stream pushed the request and registered the output queue for
    # routing.
    assert request_queue.get_nowait() == fake_context
    assert req_id in proxy.pending_out_queues

    # Feed a terminal result and drain the generator to completion; the drain
    # loop deregisters the output queue in its finally block.
    proxy.pending_out_queues[req_id].put_nowait(
        (time.monotonic(), SchedulerResult(is_done=True, result=None))
    )
    async for _ in generator:
        pass

    assert req_id not in proxy.pending_out_queues
