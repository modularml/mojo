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
"""``pipeline_method`` decorator and the scope primitive backing it.

Worker-method calls dispatched via :py:class:`~.interfaces.Proxy` enter
their ``call_method`` context manager on a per-call
:py:class:`AsyncExitStack`. That stack lives in a
:py:class:`ContextVar` for the duration of a single pipeline-method
invocation, so all the p2p calls fanned out from it share the same
tear-down boundary -- when the scope exits, every in-flight call gets
cancelled and every remote result handle is released in one pass.

The scope is established by :py:func:`pipeline_method` (the user-facing
decorator) or, for framework-internal callers, by
:py:func:`_pipeline_method_scope` directly. Calling a proxy method
outside of a scope is a hard error: it would leak result handles with no
owning lifetime.
"""

from __future__ import annotations

import functools
import inspect
from collections.abc import (
    AsyncIterator,
    Awaitable,
    Callable,
    Coroutine,
)
from contextlib import AsyncExitStack, asynccontextmanager
from contextvars import ContextVar
from typing import Any, ParamSpec, TypeVar, overload

P = ParamSpec("P")
T = TypeVar("T")


_PIPELINE_CONTEXT: ContextVar[AsyncExitStack | None] = ContextVar(
    "cascade_pipeline_context", default=None
)


def _get_pipeline_context() -> AsyncExitStack:
    """Return the currently active pipeline-scope :py:class:`AsyncExitStack`.

    Raises if no scope is set. Decorate the entry point with
    :py:func:`pipeline_method` to establish one.
    """
    stack = _PIPELINE_CONTEXT.get()
    if stack is None:
        raise RuntimeError(
            "No cascade pipeline scope is active. Decorate the entry "
            "point with `@pipeline_method`."
        )
    return stack


@asynccontextmanager
async def _pipeline_method_scope() -> AsyncIterator[AsyncExitStack]:
    """Establish a pipeline-method scope for the duration of the block.

    Framework-internal primitive backing :py:func:`pipeline_method`. All
    proxy method calls made inside the scope share a single
    :py:class:`AsyncExitStack`; exiting the scope cancels their
    :py:meth:`~.interfaces.Runtime.call_method` contexts in one pass,
    releasing in-flight tasks and remote result handles deterministically.

    Not exposed on the public ``cascade`` API: user code establishes
    scopes by decorating pipeline entry points with
    :py:func:`pipeline_method`. Tests reach in here directly when they
    need to exercise proxy machinery without a pipeline wrapper.

    Nesting is not supported; the inner scope shadows the outer one and
    only the inner scope's calls are tied to its lifetime.
    """
    async with AsyncExitStack() as stack:
        token = _PIPELINE_CONTEXT.set(stack)
        try:
            yield stack
        finally:
            _PIPELINE_CONTEXT.reset(token)


# Three overloads. ``inspect.isasyncgenfunction`` is the only runtime
# dispatch; the typing here is what tells mypy whether the body's return
# survives unchanged (passthroughs) or gets one extra ``await`` before
# reaching the caller.
#
# Coroutine bodies whose return is itself awaitable -- ``return await
# proxy.method(...)`` is the canonical case, where the body's return
# type is :py:class:`~.result.Result` (an ``Awaitable``) -- get the
# final handle awaited by the wrapper before the scope exits. Symmetric
# with :py:func:`worker_method`'s ``_fetchall`` argument auto-await:
# bodies read as a chain of single-``await`` calls and the caller sees
# the unwrapped value with one ``await``.
@overload
def pipeline_method(
    func: Callable[P, Coroutine[Any, Any, Awaitable[T]]], /
) -> Callable[P, Coroutine[Any, Any, T]]: ...
@overload
def pipeline_method(
    func: Callable[P, AsyncIterator[T]], /
) -> Callable[P, AsyncIterator[T]]: ...
@overload
def pipeline_method(
    func: Callable[P, Coroutine[Any, Any, T]], /
) -> Callable[P, Coroutine[Any, Any, T]]: ...


def pipeline_method(func: Callable[..., Any]) -> Callable[..., Any]:
    """Decorate a pipeline entry point to run inside a fresh pipeline-method scope.

    Works on ``async def`` coroutines and ``async def`` generators. The
    wrapped function runs with a fresh pipeline scope active for its
    full lifetime -- for async generators that means the scope outlives
    every iteration step, which is what keeps p2p result handles alive
    across a streamed response.

    If a coroutine body returns an :py:class:`Awaitable` (e.g.
    :py:class:`~.result.Result`), the wrapper awaits it before exiting
    the scope, so the body can read as a chain of local ``await`` calls
    (including the final one) and the caller still gets the unwrapped
    value with a single ``await``.
    """
    if inspect.isasyncgenfunction(func):

        @functools.wraps(func)
        async def gen_wrapper(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
            async with _pipeline_method_scope():
                async for item in func(*args, **kwargs):
                    yield item

        return gen_wrapper

    @functools.wraps(func)
    async def coro_wrapper(*args: Any, **kwargs: Any) -> Any:
        async with _pipeline_method_scope():
            result = await func(*args, **kwargs)
            if isinstance(result, Awaitable):
                return await result
            return result

    return coro_wrapper
