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
"""Cascade worker framework: base classes for workers, proxies, and runtimes."""

from __future__ import annotations

import functools
import inspect
from abc import ABC, abstractmethod
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
    Mapping,
    Sequence,
)
from contextlib import (
    AbstractAsyncContextManager,
    AsyncExitStack,
    asynccontextmanager,
)
from dataclasses import dataclass, field
from typing import (
    Any,
    Concatenate,
    Generic,
    ParamSpec,
    TypeVar,
    cast,
)

from max.experimental.cascade.core.pipeline_method import _get_pipeline_context
from max.experimental.cascade.core.result import Result, ResultIter
from max.experimental.cascade.core.type_hints import (
    return_type,
    stream_elem_type,
)

T = TypeVar("T")
R = TypeVar("R")
P = ParamSpec("P")


@dataclass
class Worker:
    """Base class for cascade workers with deployment metadata."""

    # hints to guide deployment in heterogenous runtime pools
    deploy_hints: list[str] = field(default_factory=list)
    deploy_timeout: float | None = None

    @asynccontextmanager
    async def open(self) -> AsyncIterator[Worker]:
        """Lifecycle context manager. Override to add setup/teardown."""
        yield self


WorkerType = TypeVar("WorkerType", bound=Worker)


# Note having Runtime as a member on Result is actually a feature
# When Runtime is a thin client proxy Runtime, it's serializable as a URL
# We can explicitly hook the rebuild of runtime to memoize by URL
# to enable efficient grpc connection sharing


class Runtime(AsyncExitStack, ABC):
    """Transport-agnostic runtime for worker execution and result delivery.

    Concrete runtimes are async context managers, sharing the
    :py:class:`AsyncExitStack` lifecycle so any custom setup/teardown a
    subclass needs (connection pools, task groups, subprocess handles,
    ...) plugs in by pushing onto the stack in an overridden
    ``__aenter__``. Exit tears registered resources down in reverse
    order. There is no separate ``open()`` method -- runtimes follow the
    same shape as :py:mod:`asyncio` and standard-library context
    managers.
    """

    @abstractmethod
    async def deploy_worker(self, worker: Worker) -> str:
        """Wire primitive: register a worker and return its stable id.

        This is the primitive that differs between local and remote runtimes.
        Most callers want :py:meth:`deploy` instead, which wraps the returned
        id in a client-side :py:class:`Proxy`.
        """
        ...

    async def deploy(self, worker: WorkerType) -> WorkerType:
        """Register a worker and return a client-side :py:class:`Proxy`.

        The runtime returns a :py:class:`Proxy` that is observationally
        equivalent to ``worker`` (same public method surface, same awaited
        result types, same async-iterator streaming surface). The return type
        is annotated as ``WorkerType`` so callers can type pipeline attributes
        with their worker classes (or :py:class:`typing.Protocol` interfaces)
        and get end-to-end type checking; the runtime cast is safe because
        :py:class:`Proxy` exposes a structurally compatible interface.
        """
        worker_id = await self.deploy_worker(worker)
        return cast(WorkerType, Proxy(self, worker_id, worker))

    @abstractmethod
    def call_method(
        self,
        worker_id: str,
        func: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AbstractAsyncContextManager[str]:
        """Launch a worker method asynchronously.

        Returns an async context manager that yields the ``result_id`` bound
        to the call. The id stays valid for the lifetime of the ``async
        with`` block; exiting cancels the in-flight task and releases the
        result buffer. The context is normally entered via
        :py:meth:`AsyncExitStack.enter_async_context` on the pipeline-scope
        stack returned by :py:func:`_get_pipeline_context`, so the call
        lifetime tracks the surrounding request.
        """
        ...

    @abstractmethod
    def get_result(self, resid: str) -> Awaitable[object]:
        """Await a single result.

        Raises whatever exception the call raised.
        """
        ...

    @abstractmethod
    def stream_result(self, resid: str) -> AsyncIterator[object]:
        """Bind a streaming result and iterate inline.

        Single-consumer: the first call (across :py:meth:`stream_result` and
        :py:meth:`stream_next`) binds the stream; subsequent attempts on the
        same ``resid`` raise. Used for lightweight inline streams (e.g. token
        streams) where backpressure rides on the underlying transport.
        """
        ...

    @abstractmethod
    async def get_metrics(self) -> str:
        """Return Prometheus metrics summary."""
        ...


class Proxy(Generic[T]):
    """Client-side handle to a deployed :py:class:`Worker`.

    Inspects ``type(worker)`` at construction time and attaches one bound
    method per non-underscore worker method. Each generated method picks a
    fresh ``result_id``, hands it to ``runtime.call_method`` for binding,
    and returns it wrapped in a :py:class:`Result` (or :py:class:`ResultIter`
    for async-generator methods).
    """

    runtime: Runtime
    worker_id: str

    def __init__(
        self, runtime: Runtime, worker_id: str, worker: Worker
    ) -> None:
        self.runtime = runtime
        self.worker_id = worker_id
        worker_class = type(worker)
        self._cls_name = worker_class.__name__
        for name in dir(worker_class):
            if name.startswith("_"):
                continue
            method = getattr(worker_class, name, None)
            if not callable(method):
                continue
            if inspect.isasyncgenfunction(method):
                setattr(self, name, self._bind_stream(name, method))
            else:
                setattr(self, name, self._bind_call(name, method))

    def _bind_call(
        self,
        name: str,
        method: Callable[Concatenate[WorkerType, P], Coroutine[Any, Any, R]],
    ) -> Callable[P, Coroutine[Any, Any, Result[R]]]:
        """Build a wrapper for a coroutine worker method."""
        runtime = self.runtime
        worker_id = self.worker_id
        ret_type = return_type(method)

        @functools.wraps(method)
        async def call(*args: P.args, **kwargs: P.kwargs) -> Result[R]:
            context = _get_pipeline_context()
            result_id = await context.enter_async_context(
                runtime.call_method(worker_id, name, args, kwargs)
            )
            return Result(result_id, runtime, ret_type)

        return call

    def _bind_stream(
        self,
        name: str,
        method: Callable[Concatenate[WorkerType, P], AsyncIterator[R]],
    ) -> Callable[P, Coroutine[Any, Any, ResultIter[R]]]:
        """Build a wrapper for an async-generator worker method."""
        runtime = self.runtime
        worker_id = self.worker_id
        elem_type = stream_elem_type(method)

        @functools.wraps(method)
        async def call_stream(
            *args: P.args, **kwargs: P.kwargs
        ) -> ResultIter[R]:
            context = _get_pipeline_context()
            result_id = await context.enter_async_context(
                runtime.call_method(worker_id, name, args, kwargs)
            )
            return ResultIter(result_id, runtime, elem_type)

        return call_stream
