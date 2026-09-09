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

"""Executor protocol and concrete implementations for eager graph execution.

An :class:`Executor` receives a finalized :class:`~max.graph.Graph` and a
sequence of input :class:`~max.driver.Buffer` objects and returns output
buffers.  Callers must not rely on the graph module being unmodified after
``execute`` returns.

Selection order (highest to lowest priority):

1. Constructor injection into :class:`~max.experimental.realization_context.EagerRealizationContext`.
2. :func:`set_default_executor` / :func:`default_executor`.
3. ``MAX_EAGER_EXECUTOR`` environment variable (``composite`` | ``jit`` | ``interpreter`` | ``compile``; default ``composite``).
"""

from __future__ import annotations

import hashlib
import os
import threading
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import Protocol, runtime_checkable

from max import _core, driver, engine
from max._core.dialects import rmo
from max.graph import Graph

from .compile_pool import _map_future, pool
from .support import SetterContext, _session


class UnsupportedGraphError(RuntimeError):
    """Raised by an executor when it refuses to execute a graph.

    A composite executor (e.g. :class:`JitExecutor`) may catch this to route
    the graph elsewhere **before** execution starts.  Once execution has
    begun, exceptions are never masked as ``UnsupportedGraphError``.
    """


@runtime_checkable
class Executor(Protocol):
    """Contract for running a finalized eager graph against input buffers."""

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Execute *graph* with *inputs* and return output buffers.

        Args:
            graph: A finalized graph ready for execution.  The executor may
                mutate the module internally (e.g. apply lowering passes);
                callers must not use the module after this call returns.
            inputs: Buffers corresponding to ``graph.inputs``, in order.
                Mutable ``BufferType`` inputs are mutated in place; the
                returned sequence covers declared graph outputs only.

        Returns:
            Output buffers in the order declared by ``graph.output``.

        Raises:
            UnsupportedGraphError: If the executor statically refuses the
                graph before any execution begins.
        """
        ...


EagerCacheKey = tuple[str, tuple[tuple[str, str], ...]]

_INTERPRETER_MAX_OPS_ENV_VAR = "MAX_INTERPRETER_MAX_OPS"
_DEFAULT_INTERPRETER_MAX_OPS = 1024


def _interpreter_max_ops() -> int:
    """Returns the dispatchable-op threshold below which the interpreter runs.

    Reads ``MAX_INTERPRETER_MAX_OPS`` from the environment.  Graphs with more
    dispatchable ops than this value are compiled so the graph compiler can
    fuse across ops.

    Returns:
        The op-count threshold (default 1024).
    """
    raw = os.environ.get(_INTERPRETER_MAX_OPS_ENV_VAR, "")
    if raw.strip().isdigit():
        return int(raw.strip())
    return _DEFAULT_INTERPRETER_MAX_OPS


def _eager_model_cache_key(graph: Graph) -> EagerCacheKey:
    """Builds a compact, stable cache key for a finalized eager graph.

    Uses a SHA-256 hash of the MLIR module bytecode combined with the
    resolved kernel library paths and SHA-256 hashes of their contents.
    Hashing file contents (rather than ``st_mtime``) avoids a
    time-of-check/time-of-use race and produces a deterministic key
    regardless of filesystem timestamp granularity.

    Args:
        graph: A finalized graph ready for compilation.

    Returns:
        A tuple of ``(module_hash, ((resolved_path, content_hash), ...))``.
    """
    module_hash = hashlib.sha256(graph._module.bytecode).hexdigest()
    kernel_paths = tuple(
        (
            str(Path(p).resolve()),
            hashlib.sha256(Path(p).read_bytes()).hexdigest(),
        )
        for p in graph.kernel_libraries_paths
    )
    return (module_hash, kernel_paths)


class InterpreterExecutor:
    """Executes a graph via :func:`max._interpreter.execute`.

    Raises :class:`UnsupportedGraphError` when
    :func:`max._interpreter.can_execute` refuses the graph
    (e.g. ``CustomOp`` present, unregistered op, or over the op-count
    threshold), or when this build lacks the interpreter's op handlers.
    All runtime errors propagate unchanged — an explicit interpreter
    request is deliberately loud.
    """

    def __init__(self, max_ops: int | None = None) -> None:
        """Initializes the executor.

        Args:
            max_ops: Maximum number of dispatchable ops accepted.  Graphs
                with more ops raise :class:`UnsupportedGraphError`.  ``None``
                imposes no limit.
        """
        self._max_ops = max_ops

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph* via the MO interpreter.

        Raises:
            UnsupportedGraphError: If the interpreter cannot handle the graph.
        """
        # Defer import to avoid compiling interpreter bindings unless
        # the interpreter is actually needed.
        from max import _interpreter

        # The interpreter only supports MO-level graphs, so lower RMO -> MO.
        _core.lower(graph._module, [rmo.passes.LegalizeRMOOps()])
        if not _interpreter.can_execute(graph, max_ops=self._max_ops):
            raise UnsupportedGraphError(
                "InterpreterExecutor: graph contains ops that require "
                "compilation (CustomOp, unregistered op, or over op-count "
                f"threshold {self._max_ops!r})."
            )
        return _interpreter.execute(graph, inputs)


class CompilingExecutor:
    """Compiles and executes a graph synchronously via ``session.load``.

    No compilation cache of its own — caching is :class:`JitExecutor`'s job.
    Each call to :meth:`execute` recompiles the graph.
    """

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Compiles *graph* and executes it immediately."""
        model = _session().load(graph)
        return model(*inputs)


class JitExecutor:
    """Executes via the interpreter while compiling in the background.

    The first call for a given graph (keyed by structure) submits the
    graph to a background compile *process*
    (:class:`~max.experimental.compile_pool.ProcessCompilePool`) and
    caches the resulting future for the life of this executor. The future
    resolves to an initialized model: ``session.init`` runs as soon as the
    compile lands, off the calling thread, so a call that finds a done
    future executes immediately. While the compile is pending, calls
    execute via the interpreter. When the interpreter refuses a graph,
    the call waits for that graph's compile.

    Failed compilations are assumed to not be transient and are cached.
    """

    cache: dict[EagerCacheKey, Future[engine.Model]]
    lock: threading.Lock

    def __init__(
        self,
        *,
        interpreter: Executor,
        sync_on_interpreter_fallback: bool = True,
    ) -> None:
        """Initializes the executor.

        Args:
            interpreter: Executor serving graphs whose compile is pending.
            sync_on_interpreter_fallback: Whether a graph the interpreter
                refuses waits for its compile. When False such a graph
                raises :class:`UnsupportedGraphError` instead.
        """
        self.cache = {}
        self.lock = threading.Lock()
        self.interpreter = interpreter
        self.sync_on_interpreter_fallback = sync_on_interpreter_fallback

    def compile(self, graph: Graph) -> Future[engine.Model]:
        """Submits the graph to the compiler on first sight.

        Returns:
            A future resolving to the initialized model.
        """
        key = _eager_model_cache_key(graph)
        with self.lock:
            if (future := self.cache.get(key)) is None:
                future = self.cache[key] = _map_future(
                    pool().compile(graph), _session().init
                )
        return future

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph*, compiling in the background on first call.

        Raises:
            Exception: The interpreter's own error, when it fails and
                ``sync_on_interpreter_fallback`` is False.
        """
        future = self.compile(graph)

        if not future.done():
            try:
                return self.interpreter.execute(graph, inputs)
            # TODO(MXF-595): narrow to a typed interpreter refusal.
            except Exception as e:
                if not self.sync_on_interpreter_fallback:
                    raise
                warnings.warn(
                    (
                        f"The eager interpreter failed on this graph ({e!r});"
                        " serving it with the compiled model instead. Please"
                        " file a bug against MAX Framework."
                    ),
                    stacklevel=2,
                )

        return future.result()(*inputs)


CompositeExecutor = JitExecutor


class _FallbackExecutor:
    """Executes via the interpreter, compiling only graphs it cannot serve.

    Unlike :class:`JitExecutor`, nothing is compiled in the background: a
    graph the interpreter serves never touches the compiler, and a graph it
    refuses is compiled synchronously in-process and cached for the life of
    this executor.

    Internal executor for CI use, selected with
    ``MAX_EAGER_EXECUTOR=fallback-internal``.  Not a supported mode; it may
    change or be removed without notice.
    """

    cache: dict[EagerCacheKey, engine.Model]
    lock: threading.Lock

    def __init__(self, *, interpreter: Executor) -> None:
        """Initializes the executor.

        Args:
            interpreter: Executor serving graphs that don't need compilation.
        """
        self.cache = {}
        self.lock = threading.Lock()
        self.interpreter = interpreter

    def execute(
        self, graph: Graph, inputs: Sequence[driver.Buffer]
    ) -> Sequence[driver.Buffer | None]:
        """Executes *graph*, compiling it only if the interpreter fails."""
        try:
            return self.interpreter.execute(graph, inputs)
        except UnsupportedGraphError:
            pass
        # TODO(MXF-595): narrow to a typed interpreter refusal.
        except Exception as e:
            warnings.warn(
                (
                    f"The eager interpreter failed on this graph ({e!r});"
                    " serving it with a synchronously compiled model instead."
                    " Please file a bug against MAX Framework."
                ),
                stacklevel=2,
            )
        key = _eager_model_cache_key(graph)
        with self.lock:
            model = self.cache.get(key)
        if model is None:
            compiled = _session().load(graph)
            with self.lock:
                model = self.cache.setdefault(key, compiled)
        return model(*inputs)


_MAX_EAGER_EXECUTOR_ENV_VAR = "MAX_EAGER_EXECUTOR"


def _default_composite() -> CompositeExecutor:
    """Builds the auto-selected eager executor.

    Interprets graphs within the ``MAX_INTERPRETER_MAX_OPS`` threshold
    while their background compile is pending, and waits for the compiled
    model for graphs the interpreter refuses.  This is the out-of-the-box
    eager execution path.
    """
    return CompositeExecutor(
        interpreter=InterpreterExecutor(max_ops=_interpreter_max_ops()),
    )


def _fallback_internal() -> _FallbackExecutor:
    """Builds the internal interpret-first, compile-on-refusal executor."""
    return _FallbackExecutor(
        interpreter=InterpreterExecutor(max_ops=_interpreter_max_ops()),
    )


_EXECUTORS: dict[str, Callable[[], Executor]] = {
    "composite": _default_composite,
    "jit": _default_composite,
    "compile": CompilingExecutor,
    "interpreter": InterpreterExecutor,
    # TODO(SDLC-4307): CI stopgap until composite stops background compiles.
    "fallback-internal": _fallback_internal,
}


def _executor_from_env() -> Executor:
    name = (
        os.environ.get(_MAX_EAGER_EXECUTOR_ENV_VAR, "composite").lower().strip()
    )
    if name not in _EXECUTORS:
        supported = sorted(n for n in _EXECUTORS if not n.endswith("-internal"))
        raise ValueError(
            f"{_MAX_EAGER_EXECUTOR_ENV_VAR}={name!r}: expected one of "
            f"{supported}"
        )
    return _EXECUTORS[name]()


_DEFAULT_EXECUTOR: Executor = _executor_from_env()


def default_executor() -> Executor:
    """Returns the ambient default executor.

    The initial default is selected by the ``MAX_EAGER_EXECUTOR``
    environment variable (``composite`` | ``jit`` | ``compile`` |
    ``interpreter``; default ``composite``), read at import time.
    """
    return _DEFAULT_EXECUTOR


def set_default_executor(executor: Executor) -> SetterContext[Executor]:
    """Sets the ambient default executor.

    The set takes effect immediately. The returned
    :class:`~max.experimental.support.SetterContext` may be used as a
    context manager to restore the previous executor on exit, or discarded
    to keep the new one.

    Args:
        executor: The new default executor.  All subsequently constructed
            :class:`~max.experimental.realization_context.EagerRealizationContext`
            instances that receive ``executor=None`` will use this executor.

    Returns:
        An undo handle restoring the previously installed executor.
    """

    def setter(executor: Executor) -> None:
        global _DEFAULT_EXECUTOR
        _DEFAULT_EXECUTOR = executor

    previous = default_executor()
    setter(executor)
    return SetterContext(executor, previous, setter)


__all__ = [
    "CompilingExecutor",
    "CompositeExecutor",
    "Executor",
    "InterpreterExecutor",
    "JitExecutor",
    "UnsupportedGraphError",
    "default_executor",
    "set_default_executor",
]
