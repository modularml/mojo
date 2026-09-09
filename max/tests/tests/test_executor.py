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
"""Tests for max.experimental.executor."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import Future
from typing import cast

import pytest
from max import _interpreter, engine
from max.driver import CPU, Buffer
from max.dtype import DType
from max.experimental import executor as executor_module
from max.experimental.compile_pool import ProcessCompilePool, pool
from max.experimental.executor import (
    CompilingExecutor,
    CompositeExecutor,
    Executor,
    InterpreterExecutor,
    JitExecutor,
    UnsupportedGraphError,
    _eager_model_cache_key,
    _executor_from_env,
    _FallbackExecutor,
    default_executor,
    set_default_executor,
)
from max.graph import Graph, TensorType, ops

_KERNEL_FAILURE = "simulated kernel failure"


def _float_buffer(values: Sequence[float]) -> Buffer:
    """Returns a host float32 buffer holding *values*."""
    buf = Buffer(DType.float32, [len(values)])
    for i, value in enumerate(values):
        buf[i] = value
    return buf


def _values(buf: Buffer | None) -> list[float]:
    """Returns the contents of a rank-1 buffer as a list of floats."""
    assert buf is not None
    (n,) = buf.shape
    return [buf[i].item() for i in range(n)]


def _add_graph() -> tuple[Graph, Buffer]:
    """Returns an add-constant graph and a matching input buffer.

    Executors may mutate a graph in place, so each call builds a fresh
    one. Successive graphs are structurally identical and therefore share
    a cache key.
    """
    input_type = TensorType(DType.float32, [2], CPU())
    with Graph("add", input_types=[input_type]) as graph:
        (x,) = graph.inputs
        one = ops.constant([1.0, 1.0], dtype=DType.float32, device=CPU())
        graph.output(ops.add(x, one))
    return graph, _float_buffer([3.0, 4.0])


def _failing_execute(
    graph: Graph, inputs: Sequence[Buffer]
) -> Sequence[Buffer | None]:
    """Stands in for ``max._interpreter.execute`` failing mid-execution."""
    raise RuntimeError(_KERNEL_FAILURE)


class _RecordingExecutor:
    """An executor that records its calls and returns canned buffers."""

    def __init__(self, result: Sequence[Buffer | None] = ()) -> None:
        self.calls: list[tuple[Graph, Sequence[Buffer]]] = []
        self._result = result

    def execute(
        self, graph: Graph, inputs: Sequence[Buffer]
    ) -> Sequence[Buffer | None]:
        self.calls.append((graph, inputs))
        return self._result


class _FakeModel:
    """A compiled model that records its calls and returns canned buffers."""

    def __init__(self, result: Sequence[Buffer | None]) -> None:
        self.calls: list[Sequence[Buffer]] = []
        self._result = result

    def __call__(self, *inputs: Buffer) -> Sequence[Buffer | None]:
        self.calls.append(inputs)
        return self._result


class _StubPool:
    """Records compile submissions and returns futures the test controls.

    Without an *exception* the futures never resolve, which pins
    :class:`JitExecutor` to its interpreter path; a real compile could
    otherwise win the race.
    """

    def __init__(self, exception: BaseException | None = None) -> None:
        self.submissions: list[Graph] = []
        self._exception = exception

    def compile(self, graph: Graph) -> Future[engine.CompiledModel]:
        self.submissions.append(graph)
        future: Future[engine.CompiledModel] = Future()
        if self._exception is not None:
            future.set_exception(self._exception)
        return future


def _install_pool(
    monkeypatch: pytest.MonkeyPatch, stub: _StubPool
) -> _StubPool:
    """Routes compile submissions to *stub*, patching the module singleton.

    The real pool is left running: it is process-wide, so closing it would
    break every other executor.
    """
    monkeypatch.setattr(
        executor_module, "pool", lambda: cast(ProcessCompilePool, stub)
    )
    return stub


@pytest.fixture
def pending_pool(monkeypatch: pytest.MonkeyPatch) -> _StubPool:
    """A stub compile pool whose compiles never land."""
    return _install_pool(monkeypatch, _StubPool())


class TestExecutorProtocol:
    @pytest.mark.parametrize(
        "executor",
        [
            InterpreterExecutor(),
            CompilingExecutor(),
            JitExecutor(interpreter=InterpreterExecutor()),
            _FallbackExecutor(interpreter=InterpreterExecutor()),
            _RecordingExecutor(),
        ],
        ids=["interpreter", "compiling", "jit", "fallback", "duck-typed"],
    )
    def test_conforms_to_protocol(self, executor: object) -> None:
        """Concrete and duck-typed executors satisfy the Executor protocol."""
        assert isinstance(executor, Executor)

    def test_composite_executor_is_an_alias(self) -> None:
        """CompositeExecutor names JitExecutor rather than a separate class."""
        assert CompositeExecutor is JitExecutor


class TestInterpreterExecutor:
    def test_executes_supported_graph(self) -> None:
        """A graph the interpreter accepts runs and returns its outputs."""
        graph, buf = _add_graph()
        (out,) = InterpreterExecutor().execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])

    def test_refuses_graph_needing_compilation(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An op that requires compilation is refused before execution."""
        monkeypatch.setattr(
            _interpreter, "_COMPILATION_REQUIRED_OP_NAMES", ("ConstantOp",)
        )
        with Graph("constant", input_types=[]) as graph:
            graph.output(ops.constant([1.0], dtype=DType.float32, device=CPU()))
        with pytest.raises(UnsupportedGraphError, match="require compilation"):
            InterpreterExecutor().execute(graph, [])

    def test_refuses_graph_over_max_ops(self) -> None:
        """max_ops caps the graph size the interpreter accepts."""
        graph, buf = _add_graph()
        with pytest.raises(UnsupportedGraphError, match="threshold 1"):
            InterpreterExecutor(max_ops=1).execute(graph, [buf])

    def test_propagates_runtime_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failure mid-execution propagates unchanged, never masked."""
        monkeypatch.setattr(_interpreter, "execute", _failing_execute)
        graph, buf = _add_graph()
        with pytest.raises(RuntimeError, match=_KERNEL_FAILURE):
            InterpreterExecutor().execute(graph, [buf])


class TestCompilingExecutor:
    def test_compiles_and_executes(self) -> None:
        """A graph compiles and runs within a single call."""
        graph, buf = _add_graph()
        results = CompilingExecutor().execute(graph, [buf])
        assert _values(results[0]) == pytest.approx([4.0, 5.0])

    def test_recompiles_every_call(self) -> None:
        """With no cache of its own, an identical second graph recompiles."""
        executor = CompilingExecutor()
        graph1, buf1 = _add_graph()
        graph2, buf2 = _add_graph()
        first = executor.execute(graph1, [buf1])
        second = executor.execute(graph2, [buf2])
        assert _values(first[0]) == pytest.approx(_values(second[0]))


class TestJitExecutor:
    """The interpreter serves while a compile is pending; the model takes over."""

    def test_pending_compile_serves_via_interpreter(
        self, pending_pool: _StubPool
    ) -> None:
        """A call whose compile has not landed is served by the interpreter."""
        graph, buf = _add_graph()
        executor = JitExecutor(interpreter=InterpreterExecutor())
        (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])
        assert len(pending_pool.submissions) == 1

    def test_landed_compile_bypasses_interpreter(
        self, pending_pool: _StubPool
    ) -> None:
        """A cached, resolved future serves the call; nothing else is touched."""
        graph, buf = _add_graph()
        model = _FakeModel([_float_buffer([7.0, 8.0])])
        landed: Future[engine.Model] = Future()
        landed.set_result(cast(engine.Model, model))
        interpreter = _RecordingExecutor()
        executor = JitExecutor(interpreter=interpreter)
        executor.cache[_eager_model_cache_key(graph)] = landed

        (out,) = executor.execute(graph, [buf])

        assert _values(out) == pytest.approx([7.0, 8.0])
        assert len(model.calls) == 1
        assert not interpreter.calls
        assert not pending_pool.submissions

    def test_refused_graph_waits_for_compile(self) -> None:
        """A graph the interpreter refuses waits for its compiled model."""
        graph, buf = _add_graph()
        executor = JitExecutor(interpreter=InterpreterExecutor(max_ops=0))
        (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])

    def test_refused_graph_raises_when_sync_disabled(
        self, pending_pool: _StubPool
    ) -> None:
        """With sync fallback off a refusal raises, but the compile still goes out."""
        graph, buf = _add_graph()
        executor = JitExecutor(
            interpreter=InterpreterExecutor(max_ops=0),
            sync_on_interpreter_fallback=False,
        )
        with pytest.raises(UnsupportedGraphError, match="require compilation"):
            executor.execute(graph, [buf])
        assert len(pending_pool.submissions) == 1

    def test_interpreter_runtime_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch, pending_pool: _StubPool
    ) -> None:
        """A live interpreter error reaches the caller."""
        monkeypatch.setattr(_interpreter, "execute", _failing_execute)
        graph, buf = _add_graph()
        executor = JitExecutor(
            interpreter=InterpreterExecutor(),
            sync_on_interpreter_fallback=False,  # TODO(MXF-595)
        )
        with pytest.raises(RuntimeError, match=_KERNEL_FAILURE):
            executor.execute(graph, [buf])

    def test_identical_graphs_share_one_compile(
        self, pending_pool: _StubPool
    ) -> None:
        """Structurally identical graphs share a cache entry and one submission."""
        executor = JitExecutor(interpreter=InterpreterExecutor())
        graph1, buf1 = _add_graph()
        graph2, buf2 = _add_graph()

        (first,) = executor.execute(graph1, [buf1])
        (cached,) = executor.cache.values()
        (second,) = executor.execute(graph2, [buf2])

        assert _values(first) == pytest.approx([4.0, 5.0])
        assert _values(second) == pytest.approx([4.0, 5.0])
        assert len(executor.cache) == 1
        assert next(iter(executor.cache.values())) is cached
        assert len(pending_pool.submissions) == 1

    def test_failed_compile_is_cached(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A failed compile re-raises on every call and is never retried."""
        stub = _install_pool(
            monkeypatch, _StubPool(RuntimeError("compile exploded"))
        )
        # A refusing interpreter makes every call demand the compiled model.
        executor = JitExecutor(interpreter=InterpreterExecutor(max_ops=0))
        graph, buf = _add_graph()

        for _ in range(2):
            with pytest.raises(RuntimeError, match="compile exploded"):
                executor.execute(graph, [buf])
        assert len(stub.submissions) == 1

    def test_caches_are_per_instance(self, pending_pool: _StubPool) -> None:
        """Two JitExecutors never share a compile cache."""
        graph, buf = _add_graph()
        first = JitExecutor(interpreter=InterpreterExecutor())
        second = JitExecutor(interpreter=InterpreterExecutor())
        first.execute(graph, [buf])
        assert len(first.cache) == 1
        assert not second.cache

    def test_default_composite_reads_max_ops_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MAX_INTERPRETER_MAX_OPS caps what the default composite interprets."""
        monkeypatch.delenv("MAX_EAGER_EXECUTOR", raising=False)
        monkeypatch.setenv("MAX_INTERPRETER_MAX_OPS", "0")
        executor = _executor_from_env()
        assert isinstance(executor, JitExecutor)
        assert isinstance(executor.interpreter, InterpreterExecutor)
        assert executor.interpreter._max_ops == 0

        graph, buf = _add_graph()
        (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])


class TestJitExecutorSnapshot:
    """The background compile owns a snapshot, not the caller's module."""

    def test_compiled_model_agrees_with_interpreter(self) -> None:
        """A landed compile agrees with the interpreter that served earlier.

        The interpreter path legalizes the caller's module in place, which
        must leave the already-submitted compile unaffected.
        """
        graph, buf = _add_graph()
        executor = JitExecutor(interpreter=InterpreterExecutor())
        (interpreted,) = executor.execute(graph, [buf])
        (compiling,) = executor.cache.values()
        compiling.result()
        (compiled,) = executor.execute(graph, [buf])
        assert _values(interpreted) == pytest.approx(_values(compiled))

    def test_repeated_execute_of_one_graph_object(self) -> None:
        """Executing one graph object twice stays cached and correct."""
        graph, buf = _add_graph()
        executor = JitExecutor(interpreter=InterpreterExecutor())
        (first,) = executor.execute(graph, [buf])
        (second,) = executor.execute(graph, [buf])
        assert _values(first) == pytest.approx(_values(second))


class TestJitExecutorConcurrency:
    def test_concurrent_demands_share_one_compile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Threads racing on one graph trigger a single compile and init."""
        real_pool = pool()
        real_session = executor_module._session()
        compiles: list[Graph] = []
        inits: list[engine.CompiledModel] = []

        class _CountingPool:
            def compile(self, graph: Graph) -> Future[engine.CompiledModel]:
                compiles.append(graph)
                return real_pool.compile(graph)

        class _CountingSession:
            def init(self, compiled: engine.CompiledModel) -> engine.Model:
                inits.append(compiled)
                return real_session.init(compiled)

        monkeypatch.setattr(
            executor_module,
            "pool",
            lambda: cast(ProcessCompilePool, _CountingPool()),
        )
        monkeypatch.setattr(
            executor_module, "_session", lambda: _CountingSession()
        )

        # A refusing interpreter forces every thread onto the compile.
        executor = JitExecutor(interpreter=InterpreterExecutor(max_ops=0))
        workloads = [_add_graph() for _ in range(4)]
        results: list[Sequence[Buffer | None]] = []

        def _demand(graph: Graph, buf: Buffer) -> None:
            results.append(executor.execute(graph, [buf]))

        threads = [
            threading.Thread(target=_demand, args=workload)
            for workload in workloads
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=300)

        assert len(results) == 4
        for result in results:
            assert _values(result[0]) == pytest.approx([4.0, 5.0])
        assert len(compiles) == 1
        assert len(inits) == 1


class TestFallbackExecutor:
    def test_served_graph_never_compiles(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A graph the interpreter serves never touches the compiler."""

        def _no_session() -> engine.InferenceSession:
            pytest.fail("compiled an interpreter-served graph")

        monkeypatch.setattr(executor_module, "_session", _no_session)
        executor = _FallbackExecutor(interpreter=InterpreterExecutor())
        graph, buf = _add_graph()
        (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])

    def test_refused_graph_compiles_synchronously(self) -> None:
        """A refused graph is served by an in-process compile."""
        graph, buf = _add_graph()
        executor = _FallbackExecutor(interpreter=InterpreterExecutor(max_ops=0))
        (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])
        assert len(executor.cache) == 1

    def test_identical_refused_graphs_share_one_compile(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Structurally identical refused graphs share a cache entry."""
        real_session = executor_module._session
        sessions: list[engine.InferenceSession] = []

        def _counting_session() -> engine.InferenceSession:
            sessions.append(real_session())
            return sessions[-1]

        monkeypatch.setattr(executor_module, "_session", _counting_session)
        executor = _FallbackExecutor(interpreter=InterpreterExecutor(max_ops=0))
        graph1, buf1 = _add_graph()
        graph2, buf2 = _add_graph()

        (first,) = executor.execute(graph1, [buf1])
        (second,) = executor.execute(graph2, [buf2])

        assert _values(first) == pytest.approx([4.0, 5.0])
        assert _values(second) == pytest.approx([4.0, 5.0])
        assert len(executor.cache) == 1
        assert len(sessions) == 1

    def test_interpreter_runtime_error_falls_back_with_warning(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A mid-execution interpreter failure warns and serves compiled."""
        monkeypatch.setattr(_interpreter, "execute", _failing_execute)
        executor = _FallbackExecutor(interpreter=InterpreterExecutor())
        graph, buf = _add_graph()
        with pytest.warns(UserWarning, match="file a bug"):
            (out,) = executor.execute(graph, [buf])
        assert _values(out) == pytest.approx([4.0, 5.0])


class TestDefaultExecutor:
    @pytest.mark.parametrize(
        ("env_value", "expected"),
        [
            (None, CompositeExecutor),
            ("composite", CompositeExecutor),
            ("jit", JitExecutor),
            ("interpreter", InterpreterExecutor),
            ("compile", CompilingExecutor),
            ("fallback-internal", _FallbackExecutor),
        ],
        ids=[
            "unset",
            "composite",
            "jit",
            "interpreter",
            "compile",
            "fallback-internal",
        ],
    )
    def test_env_selects_executor(
        self,
        monkeypatch: pytest.MonkeyPatch,
        env_value: str | None,
        expected: type[object],
    ) -> None:
        """MAX_EAGER_EXECUTOR picks the executor; unset means the composite."""
        if env_value is None:
            monkeypatch.delenv("MAX_EAGER_EXECUTOR", raising=False)
        else:
            monkeypatch.setenv("MAX_EAGER_EXECUTOR", env_value)
        assert isinstance(_executor_from_env(), expected)

    def test_env_unknown_value_raises(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An unrecognized MAX_EAGER_EXECUTOR fails loudly, never silently."""
        monkeypatch.setenv("MAX_EAGER_EXECUTOR", "compiled")
        with pytest.raises(ValueError, match="MAX_EAGER_EXECUTOR"):
            _executor_from_env()

    def test_set_default_executor_restores_on_exit(self) -> None:
        """The handle installs immediately and restores the previous executor."""
        original = default_executor()
        replacement = InterpreterExecutor()
        with set_default_executor(replacement):
            assert default_executor() is replacement
        assert default_executor() is original

    def test_discarded_handle_keeps_new_executor(self) -> None:
        """The set is eager: dropping the handle keeps the new executor."""
        original = default_executor()
        replacement = InterpreterExecutor()
        set_default_executor(replacement)
        try:
            assert default_executor() is replacement
        finally:
            set_default_executor(original)

    def test_default_executor_is_shared_across_threads(self) -> None:
        """Concurrent callers all see the same singleton."""
        results: list[Executor] = []

        threads = [
            threading.Thread(target=lambda: results.append(default_executor()))
            for _ in range(8)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert len(results) == 8
        assert all(result is results[0] for result in results)
