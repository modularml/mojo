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
"""In-process worker runtime used as the core for Cascade transports."""

from __future__ import annotations

import asyncio
import inspect
import itertools
import logging
from collections.abc import AsyncIterator, Callable, Mapping, Sequence
from contextlib import AsyncExitStack, asynccontextmanager
from typing import cast

from max.experimental.cascade.core.interfaces import Runtime, Worker
from max.experimental.cascade.core.type_hints import arg_types
from max.support._taskgroups import CancelGroup


# TODO why do we need this? Where should it really live???
def _default_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    return logger


class LocalRuntime(Runtime):
    """In-process worker runtime; serves as the core for Cascade transports."""

    def __init__(self, logger: logging.Logger | None = None) -> None:
        super().__init__()
        self.logger = logger if logger is not None else _default_logger()
        self._workers: dict[str, Worker] = {}
        # Worker-method results live in one of two dicts, picked at
        # ``call_method`` time based on whether the method is an async
        # generator. Splitting the storage keeps the types precise --
        # ``get_result`` and ``stream_result`` each look at a single,
        # well-typed dict and let the natural ``KeyError`` from a miss
        # serve as the "wrong shape for this endpoint" signal.
        self._value_results: dict[str, asyncio.Future[object]] = {}
        self._stream_results: dict[str, AsyncIterator[object]] = {}
        self._actid_counter = itertools.count()
        self._resid_counter = itertools.count()
        self._worker_stack: AsyncExitStack | None = None
        self._task_group: CancelGroup | None = None

    async def __aenter__(self) -> LocalRuntime:
        await super().__aenter__()
        # use a nested stack to ensure workers are destroyed after all pending tasks
        self._worker_stack = await self.enter_async_context(AsyncExitStack())
        self._task_group = await self.enter_async_context(CancelGroup())
        return self

    async def deploy_worker(self, worker: Worker) -> str:
        """Register a deployed worker and return its stable id.

        Enters the worker's ``open()`` context; it will be exited when the
        runtime's ``async with`` block exits. The base class's
        :py:meth:`deploy` wraps this id in a client-side :py:class:`Proxy`.
        """
        if self._worker_stack is None:
            raise RuntimeError("LocalRuntime context not entered")
        cls_name = type(worker).__name__
        worker_id = f"{cls_name}-{next(self._actid_counter)}"
        self.logger.info("Deploying worker %s", worker_id)
        live_worker = await self._worker_stack.enter_async_context(
            worker.open()
        )
        self._workers[worker_id] = live_worker
        return worker_id

    def _resolve_method(
        self, worker_id: str, func_name: str
    ) -> tuple[Worker, Callable[..., object]]:
        """Look up a worker and one of its public methods by name."""
        if worker_id not in self._workers:
            raise KeyError(f"Unknown worker id: {worker_id!r}")
        worker = self._workers[worker_id]
        method = (
            getattr(worker, func_name, None)
            if not func_name.startswith("_")
            else None
        )
        if not callable(method):
            raise ValueError(
                f"Unknown worker method: {type(worker).__name__}.{func_name}"
            )
        return worker, cast(Callable[..., object], method)

    async def get_func_args_type(
        self, worker_id: str, func_name: str
    ) -> dict[str, object | None]:
        """Resolve the parameter type hints of a worker method by name."""
        _, method = self._resolve_method(worker_id, func_name)
        return arg_types(method)

    @asynccontextmanager
    async def call_method(
        self,
        worker_id: str,
        func_name: str,
        args: Sequence[object],
        kwargs: Mapping[str, object],
    ) -> AsyncIterator[str]:
        """Launch a worker method asynchronously and yield its ``result_id``.

        The result id is minted here and is valid for the duration of the
        yielded context. Coroutine methods drive their body in a
        background task whose lifetime is bound to this context (cancelled
        on exit); async-generator methods produce their iterator
        synchronously and do their work lazily as the consumer pulls, so
        no task is needed -- exit just drops the iterator from the
        registry.
        """
        if self._task_group is None:
            raise RuntimeError("LocalRuntime context not entered")
        worker, method = self._resolve_method(worker_id, func_name)
        cls_name = type(worker).__name__
        result_id = f"{cls_name}.{func_name}-{next(self._resid_counter)}"

        cleanup: Callable[[], None]
        if inspect.isasyncgenfunction(method):
            # Stream path: invoking an async-generator function returns
            # the iterator object synchronously without running any user
            # code, so we can register it directly. Iteration -- and any
            # errors it surfaces -- happens later when the consumer pulls
            # from ``stream_result``.
            stream = method(*args, **kwargs)
            assert isinstance(stream, AsyncIterator)
            self._stream_results[result_id] = stream

            def cleanup() -> None:
                self._stream_results.pop(result_id, None)

        else:
            # Value path: drive the body in a task so any exception is
            # captured in the future and so cancellation on exit halts
            # in-flight work.
            future: asyncio.Future[object] = (
                asyncio.get_running_loop().create_future()
            )

            async def run_call() -> None:
                try:
                    ret = method(*args, **kwargs)
                    value = await ret if inspect.isawaitable(ret) else ret
                    future.set_result(value)
                except asyncio.CancelledError:
                    future.cancel()
                    raise
                except Exception as exc:
                    self.logger.exception(
                        "Cascade worker method %s.%s failed",
                        cls_name,
                        func_name,
                    )
                    future.set_exception(exc)

            task = self._task_group.create_task(run_call())
            self._value_results[result_id] = future

            def cleanup() -> None:
                task.cancel()
                self._value_results.pop(result_id, None)

        try:
            yield result_id
        finally:
            cleanup()

    async def get_result(self, result_id: str) -> object:
        """Await a single (non-streaming) result.

        In-process results are the native Python object already.

        Raises :py:class:`KeyError` if ``result_id`` does not name a
        value-typed result (either unknown, already freed, or registered
        as a stream -- consume those via :py:meth:`stream_result`).
        """
        if result_id not in self._value_results:
            raise KeyError(f"Unknown result id: {result_id!r}")
        return await self._value_results[result_id]

    async def stream_result(self, result_id: str) -> AsyncIterator[object]:
        """Bind a streaming result and iterate inline. Single-consumer.

        Raises :py:class:`KeyError` if ``result_id`` does not name a
        streaming result (either unknown, already freed, or registered
        as a value -- consume those via :py:meth:`get_result`).

        TODO: what happens when trying to iterate the same stream twice?
        Should we pop on first ``__aiter__`` to enforce single-consumer?
        """
        if result_id not in self._stream_results:
            raise KeyError(f"Unknown result id: {result_id!r}")
        source = self._stream_results[result_id]
        async for chunk in source:
            yield chunk

    async def get_metrics(self) -> str:
        """Return Prometheus exposition text for this runtime."""
        return ""
