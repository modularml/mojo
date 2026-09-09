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

"""Tracing Python functions."""

from __future__ import annotations

import functools
import inspect
from collections.abc import Callable
from types import TracebackType
from typing import Any, TypeVar, overload

from max._core.profiler import Trace, is_profiling_enabled

__all__ = ["Tracer", "traced"]

_FuncType = TypeVar("_FuncType", bound=Callable[..., Any])

# For the list of valid colors, take a look at the struct `Color` in:
# `Mojo/stdlib/stdlib/gpu/host/_tracing.mojo`


@overload
def traced(
    func: _FuncType,
    *,
    message: str | None = None,
    color: str = "modular_purple",
) -> _FuncType: ...


@overload
def traced(
    func: None = None,
    *,
    message: str | None = None,
    color: str = "modular_purple",
) -> Callable[[_FuncType], _FuncType]: ...


def traced(
    func: _FuncType | None = None,
    *,
    message: str | None = None,
    color: str = "modular_purple",
) -> _FuncType | Callable[[_FuncType], _FuncType]:
    """Decorator for creating a profiling span for a function.

    Creates a profiling span that measures the execution time of the decorated
    function. This is useful for identifying performance bottlenecks without
    modifying the function's internal code. The decorator supports both
    synchronous and asynchronous functions.

    .. code-block:: python

        from max.profiler import traced

        # Decorator with custom span name
        @traced(message="inference", color="red")
        def run_model() -> None:
            # The profiling span is named "inference"
            model.execute()

        # Decorator with default span name (uses function name)
        @traced
        def preprocess_data() -> None:
            # The profiling span is named "preprocess_data"
            data.normalize()

    Args:
        func: The function to profile.
        message: The name of the profiling span. If None, uses the function name.
        color: The color of the profiling span for visualization tools.

    Returns:
        Callable: The decorated function wrapped in a trace object.
    """
    if func is None:
        return lambda f: traced(f, message=message, color=color)

    # Default to the function name if message wasn't passed.
    message = message if message is not None else func.__name__

    if inspect.iscoroutinefunction(func):

        @functools.wraps(func)
        async def wrapper(*args, **kwargs):  # noqa: ANN202
            if is_profiling_enabled():
                with Trace(message, color):
                    return await func(*args, **kwargs)
            else:
                return await func(*args, **kwargs)

    else:

        @functools.wraps(func)
        def wrapper(*args, **kwargs):  # noqa: ANN202
            if is_profiling_enabled():
                with Trace(message, color):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)

    return wrapper  # type: ignore


class Tracer:
    """A stack-based profiling manager for creating nested profiling spans.

    Manages a stack of profiling spans that allows for nested tracing without
    requiring deeply nested ``with Trace(name):`` statements. This is especially
    useful when you need to dynamically create and manage profiling spans based
    on runtime conditions or when profiling spans don't align with your code's
    block structure.

    The ``Tracer`` can be used both as a context manager and as a manual stack
    manager. As a context manager, it ensures all pushed spans are properly
    closed when the context exits.

    .. code-block:: python

        from max.profiler import Tracer

        # Manual stack management
        tracer = Tracer("parent_operation", color="modular_purple")
        tracer.push("child_operation")
        # ... perform work ...
        tracer.pop()

        # Context manager with manual stack
        with Tracer("parent_operation", color="modular_purple") as tracer:
            # The parent span is named "parent_operation"
            tracer.push("child_operation")
            # ... perform work ...
            tracer.pop()
            # All spans are automatically closed on context exit
    """

    def __init__(
        self, message: str | None = None, color: str = "modular_purple"
    ) -> None:
        """Initializes the tracer stack.

        Creates an empty trace stack and optionally pushes an initial profiling
        span if a message is provided.

        Args:
            message: The name of the initial profiling span. If None, no initial
                span is created.
            color: The color of the profiling span for visualization tools.
        """
        self.trace_stack: list[Trace | None] = []
        self.push(message, color)

    def push(
        self, message: str | None = None, color: str = "modular_purple"
    ) -> None:
        """Pushes a new profiling span onto the stack.

        Creates and activates a new profiling span. If profiling is disabled or
        no message is provided, pushes a None placeholder to maintain stack
        consistency.

        Args:
            message: The name of the profiling span. If None, no span is created.
            color: The color of the profiling span for visualization tools.
        """
        if not is_profiling_enabled() or message is None:
            self.trace_stack.append(None)
        else:
            trace = Trace(message, color)
            self.trace_stack.append(trace)
            trace.__enter__()

    def pop(
        self,
        exc_type: type[BaseException] | None = None,
        exc_value: BaseException | None = None,
        traceback: TracebackType | None = None,
    ) -> None:
        """Pops a profiling span off the stack and closes it.

        Removes the most recently pushed profiling span from the stack and
        closes it, recording its execution time. Exception information can be
        passed through for proper error handling in context managers.

        Args:
            exc_type: The exception type if an exception occurred, or None.
            exc_value: The exception instance if an exception occurred, or None.
            traceback: The traceback object if an exception occurred, or None.
        """
        trace = self.trace_stack.pop()
        if trace is not None:
            trace.__exit__(exc_type, exc_value, traceback)

    def next(self, message: str, color: str = "modular_purple") -> None:
        """Transitions to the next profiling span.

        Pops the current profiling span and immediately pushes a new one with
        the specified message. This is a convenience method for sequential
        operations at the same nesting level.

        Args:
            message: The name of the new profiling span.
            color: The color of the profiling span for visualization tools.
        """
        self.pop()
        self.push(message, color)

    def cleanup(self) -> None:
        """Closes all remaining profiling spans.

        Pops and closes all profiling spans that were pushed onto the stack.
        This method is automatically called when the tracer is used as a
        context manager or when the object is deleted.
        """
        while self.trace_stack:
            self.pop()

    def __del__(self) -> None:
        self.cleanup()

    def __enter__(self) -> Tracer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.cleanup()

    def mark(self) -> None:
        """Marks the current profiling span with a timestamp.

        Records a timestamp event within the current profiling span. This is
        useful for marking significant events or milestones within a longer
        operation.

        Raises:
            AssertionError: If the stack is empty when mark is called.
        """
        assert self.trace_stack, "stack underflow in Tracer.mark()"
        if self.trace_stack[-1] is not None:
            self.trace_stack[-1].mark()
