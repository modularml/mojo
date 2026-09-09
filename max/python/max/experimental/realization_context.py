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

"""Implementations of various realization strategies.

**"Eager" execution**: tensors are realized as soon as the realization context
exits. This is the default behavior.

This has a huge concrete advantage over eagerly executing one operation
at a time: by controlling the boundary of where the eager context starts
and ends, we can give advanced users a tool to *enable fine-grained
bounds for automatic fusion*.

In practice the easiest way to do this is to mark a function as
`F.functional`. This function is then assumed to be "atomic" for the
purposes of eager execution. All ops within the function execute as
part of the same graph, meaning the compiler is free to fuse operations
and generate fused kernels within this region.

**"Lazy" execution**: tensors are realized only when code later tries to use
them.

This enables a class of interface design common in the ML world, in
which layers are constructed with randomized weights which are never
used. Lazy execution neatly allows constructing entire models,
only performing the weight initialization and allocating memory for
them if and when those weights are actually used.

**Graph compilation**: tensors must never be realized.

This allows tensor operations to be composed with direct usage of
the Graph API, for instance `Module.compile`, or using `F.*` operations
in another Graph API usage.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import logging
import weakref
from collections.abc import Callable, Generator, Sequence
from types import TracebackType
from typing import TYPE_CHECKING, Any, TypeVar, cast

from max import _core, driver
from max._core.dialects import builtin
from max._mlir_context import in_default_mlir_context
from max.dtype import DType
from max.experimental import _passes
from max.experimental.executor import (
    CompilingExecutor,
    Executor,
    InterpreterExecutor,
    default_executor,
)
from max.experimental.support import (
    SetterContext,
    driver_tensor_type,
)
from max.experimental.tensor import (
    GraphValue,
    RealizationContext,
    RealizationState,
    Tensor,
    current_realization_context,
    realization_context,
)
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    Type,
    Value,
    ops,
)

if TYPE_CHECKING:
    from max.experimental.sharding import DeviceMapping, DeviceMesh

Ex = TypeVar("Ex", bound=BaseException)
#: What a caller files beside a shared subgraph body, handed back unchanged.
Entry = TypeVar("Entry")

_SEED: Tensor | None = None


def seed() -> Tensor:
    """Gets the global random seed tensor used in eager execution mode."""
    global _SEED
    if _SEED is None:
        seed_type = ops.random.SeedType(DeviceRef.CPU())
        shape = [int(d) for d in seed_type.shape]
        seed_data = driver.Buffer(
            seed_type.dtype, shape, seed_type.device.to_device()
        )
        _SEED = Tensor(storage=seed_data)
    return _SEED


def set_seed(value: int) -> None:
    """Sets the global random seed value.

    Updates the global random seed to the specified value. This affects all
    subsequent random number generation in eager execution mode.

    Args:
        value: The integer seed value to set.
    """
    seed().driver_tensor[0] = value


# ─── Shared signal-buffer cache (allocated once per device set) ──────────


# maxsize=4: one entry per unique device-set configuration.  Most users
# have a single configuration; testing may use a few more.  On eviction
# the next call re-allocates (expensive but correct).
@functools.lru_cache(maxsize=4)
def _cached_signal_buffers(
    device_ids: tuple[int, ...],
) -> tuple[list[driver.Buffer], list[BufferType]]:
    """Returns (runtime_buffers, buffer_types) for the given GPU device IDs.

    Signal buffers are 1025 MB each — far too expensive to re-allocate per
    eager graph.  ``lru_cache`` ensures they are allocated once for each
    unique device set and reused for all subsequent graphs.

    Using ``lru_cache`` on an immutable key (tuple of ints) is thread-safe
    and avoids mutable module-level state.  In pytest-xdist each worker is
    a separate process, so there are no cross-worker conflicts.
    """
    # Signal buffers: 1 MB signal + 256 MB communication scratch per GPU.
    # Must stay in sync with ``Signals.NUM_BYTES`` in ``max.nn.comm.allreduce``
    # and the Mojo ``Signal`` struct size. 1 GiB scratch supports
    # hidden_dim * max_batch_input_tokens * dtype_bytes up to ~1 GiB
    # (e.g., Kimi-K2.5 at hidden_dim=20480, max_batch_input_tokens=16384).
    _NUM_BYTES = (1 + 1024) * 1024 * 1024

    try:
        driver.enable_all_peer_access()
    except RuntimeError:
        logging.getLogger(__name__).warning(
            "Failed to enable peer-to-peer GPU access. "
            "Collective operations will fall back to slower paths."
        )

    accelerators = [driver.Accelerator(id=i) for i in device_ids]
    runtime_bufs = [
        driver.Buffer.zeros(
            shape=(_NUM_BYTES,), dtype=DType.uint8, device=accel
        )
        for accel in accelerators
    ]
    for accel in accelerators:
        accel.synchronize()

    buf_types = [
        BufferType(
            dtype=DType.uint8, shape=(_NUM_BYTES,), device=DeviceRef.GPU(id=i)
        )
        for i in device_ids
    ]
    return runtime_bufs, buf_types


def _make_unrealized(
    ctx: RealizationContext,
    values: tuple[GraphValue, ...],
    mapping: DeviceMapping | None,
) -> Tensor:
    """Wraps graph values into a Tensor, dispatching to sharded constructor if needed."""
    state = RealizationState(values, ctx)
    if mapping is not None and mapping.mesh.num_devices > 1:
        placements = mapping.to_placements()
        return Tensor._from_unrealized_shards(state, mapping.mesh, placements)
    return Tensor(state=state)


class EagerRealizationContext(RealizationContext):
    """Computation graph for managing tensor operations.

    This class manages the directed acyclic graph (DAG) of tensor operations
    for lazy evaluation and optimization. It tracks both realized tensors
    (with concrete data in memory) and unrealized tensors (pending computations)
    to enable efficient batch compilation and execution.
    """

    graph: Graph
    #: Keeps a strong reference to tensor data that we need to compute graph values
    sources: dict[_core.Value[Any], Tensor]
    #: Reverse map of sources (TensorValue for read-only, BufferValue for mutable)
    source_values: dict[int, Value[Any]]
    #: Unrealized values
    unrealized: list[weakref.ref[Tensor]]
    #: Signal buffer graph values for multi-device collectives (lazily created).
    signal_buffers: list[BufferValue] | None

    def __init__(
        self,
        executor: Executor | None = None,
        *,
        use_interpreter: bool | None = None,
    ):
        """Initializes the context.

        Args:
            executor: Executor used to run the finalized graph.  ``None``
                resolves to
                :func:`~max.experimental.executor.default_executor` at
                construction time.
            use_interpreter: Deprecated.  Selects an executor for backward
                compatibility when ``executor`` is not given: ``True`` forces
                the interpreter for any graph the interpreter accepts (runtime
                errors propagate), ``False`` forces compilation, and ``None``
                uses the default executor.
        """
        if executor is not None:
            self._executor: Executor = executor
        elif use_interpreter is None:
            self._executor = default_executor()
        elif use_interpreter:
            self._executor = InterpreterExecutor(max_ops=None)
        else:
            self._executor = CompilingExecutor()
        self.sources = {}
        self.source_values = {}
        self.unrealized = []
        self.signal_buffers = None

        # Inherits process-global default custom extensions (see
        # max.graph.default_custom_extensions), so a backend's kernel overlays
        # are reachable by ops staged for eager realization.
        self.graph = Graph("main", input_types=[])

        with realization_context(self), self.graph:
            ops.random.set_seed(seed())

    def finalize_graph(self) -> tuple[list[Tensor], Graph]:
        """Finalizes the computation graph for execution.

        Prepares the graph for execution by setting outputs, lowering RMO
        ops, and removing dead code and unused arguments.

        Returns:
            tuple[list[Tensor], Graph]: A tuple containing the list of output
                tensors (including the seed) and the finalized graph.
        """
        with realization_context(self), self.graph:
            # peek rather than next! If compilation or execute fails
            # the seed should remain the same.
            outputs = [
                Tensor.from_graph_value(ops.random._peek_seed()),
                *(
                    tensor
                    for ref in self.unrealized
                    if (tensor := ref()) is not None
                ),
            ]
            flat_values = [
                s._graph_value for t in outputs for s in t.local_shards
            ]
            self.graph.output(*flat_values)
        _core.lower(self.graph._module, [builtin.passes.RemoveDeadValuesPass()])
        # The graph symbol is public, so RemoveDeadValues won't remove
        # unused arguments. Do that explicitly.
        _passes.remove_unused_arguments(self.graph)
        return outputs, self.graph

    # Lazy realize fires after the surrounding `with` exits — re-enter on bg threads.
    @in_default_mlir_context
    async def realize_all(self) -> list[Tensor]:
        """Compiles and executes the computation graph, realizing all tensors.

        Finalizes the computation graph, passes it to the bound
        :class:`~max.experimental.executor.Executor`, and applies the results
        to produce concrete values for all pending (unrealized) tensors. After
        execution, all tensors tracked by this context will have their data in
        memory.

        Returns:
            list[Tensor]: The list of realized output tensors (excluding the
                internal seed tensor).

        Raises:
            TypeError: If called while still inside this realization context.
        """
        if current_realization_context(None) is self:
            raise TypeError(
                "Can't realize tensor before realization context is completed."
            )

        outputs, graph = self.finalize_graph()

        # All graph inputs (tensor data + signal buffers) go through
        # self.sources — signal buffers are registered there by
        # ensure_signal_buffers().
        input_buffers = [
            self.sources[inp._mlir_value].driver_tensor for inp in graph.inputs
        ]

        results = self._executor.execute(graph, input_buffers)

        # Update tensors to realized.
        # Each tensor consumes num_shards consecutive results (1 for
        # unsharded, N for sharded).
        result_idx = 0
        for tensor in outputs:
            n = tensor.num_shards
            extracted = results[result_idx : result_idx + n]
            if not all(isinstance(buf, driver.Buffer) for buf in extracted):
                raise TypeError(
                    "Expected all results to be driver.Buffer, got: "
                    + str([type(b).__name__ for b in extracted])
                )
            tensor._storages = tuple(cast(list[driver.Buffer], extracted))
            tensor._state = None
            result_idx += n

        # Update mutated buffer inputs to realized
        for source in self.sources.values():
            # This was set by calling `__buffervalue__` on the source.
            # Mark the tensor as realized again.
            if source._state and source._state.ctx is self:
                source._state = None

        new_seed, *outputs = outputs
        set_seed(new_seed.item())

        return outputs

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a realized tensor as an input source to the computation graph.

        Registers a realized tensor as a graph input, allowing it to be used
        in subsequent graph operations. The tensor's data will be passed to
        the compiled graph during execution. This operation is idempotent;
        adding the same tensor multiple times returns the same state.

        Args:
            tensor: A realized tensor to add as a graph input source.

        Returns:
            RealizationState: The state associating the tensor with its graph
                value and this context.

        Raises:
            TypeError: If the tensor is not realized (has no concrete data).
        """
        if not tensor.real:
            raise TypeError("Only realized tensors may be graph sources.")

        return self._add_source(tensor, mutable=False)

    def add_mutable_source(self, tensor: Tensor) -> RealizationState:
        """Adds a realized tensor as a mutable graph input.

        Like :meth:`add_source` but creates a ``BufferType`` (mutable) input
        so the tensor can be mutated in-place by ``buffer_store``.
        """
        return self._add_source(tensor, mutable=True)

    def _add_source(self, tensor: Tensor, *, mutable: bool) -> RealizationState:
        if not tensor.real:
            raise TypeError("Only realized tensors may be graph sources.")

        # Safe to use IDs because self.sources keeps references alive.
        # If already added, return the cached value — but upgrade to
        # mutable if requested and not already mutable.
        if (cached := self.source_values.get(id(tensor))) is not None:
            if mutable and not isinstance(cached, BufferValue):
                # Need to upgrade: remove old read-only input, add mutable.
                pass  # Fall through to create a new mutable input.
            else:
                return RealizationState((cast(GraphValue, cached),), self)

        assert tensor.storage
        src_type = driver_tensor_type(tensor.storage)
        input_type = src_type.as_buffer() if mutable else src_type
        value = _passes.add_input(self.graph, input_type)
        if mutable:
            assert isinstance(value, BufferValue)
        self.sources[value._mlir_value] = tensor
        self.source_values[id(tensor)] = value
        return RealizationState((cast(GraphValue, value),), self)

    def create_unrealized(
        self,
        values: tuple[GraphValue, ...],
        *,
        mapping: DeviceMapping | None = None,
    ) -> Tensor:
        """Creates an unrealized tensor backed by graph value(s)."""
        tensor = _make_unrealized(self, values, mapping)
        self.unrealized.append(weakref.ref(tensor))
        return tensor

    def ensure_signal_buffers(
        self, mesh: DeviceMesh
    ) -> list[BufferValue] | None:
        """Lazily creates signal buffers for multi-device collectives on *mesh*.

        Called by collective ops when they detect a multi-GPU mesh.  On the
        first call, this adds ``BufferType`` graph inputs and caches the
        resulting ``BufferValue`` list so subsequent collectives in the
        same graph reuse the same buffers.

        The runtime ``driver.Buffer`` objects (1025 MB each) are allocated
        once per device set via :func:`_cached_signal_buffers` and shared
        across all eager contexts to avoid repeated allocation.

        Returns ``None`` for single-device or CPU-only meshes.
        """
        if self.signal_buffers is not None:
            return self.signal_buffers

        from max.driver import Accelerator as _Acc

        gpu_ids: list[int] = []
        seen: set[int] = set()
        for dev in mesh.devices:
            if isinstance(dev, _Acc) and dev.id not in seen:
                gpu_ids.append(dev.id)
                seen.add(dev.id)

        if len(gpu_ids) < 2:
            return None

        # Get or allocate shared runtime buffers (expensive — 1+256 MB each).
        runtime_bufs, buf_types = _cached_signal_buffers(tuple(gpu_ids))

        # Add signal buffer types as new graph inputs (per-graph, cheap).
        # Register them in self.sources so the execution loop picks them
        # up naturally alongside tensor data — no special-casing needed.
        buf_values: list[BufferValue] = []
        for i, bt in enumerate(buf_types):
            value = _passes.add_input(self.graph, bt)
            assert isinstance(value, BufferValue)
            buf_values.append(value)
            self.sources[value._mlir_value] = Tensor(storage=runtime_bufs[i])

        self.signal_buffers = buf_values
        return self.signal_buffers

    def __enter__(self):
        self.graph.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)
        if not exception:
            from max.experimental import functional as F

            F._run(self.realize_all())


class LazyRealizationContext(EagerRealizationContext):
    """A realization context that defers execution until explicitly requested.

    Unlike :class:`~max.experimental.realization_context.EagerRealizationContext`, this context does not automatically
    execute the computation graph when the context exits. Tensors remain
    unrealized until explicitly awaited via ``await tensor.realize``.

    This is useful for batching many operations together before execution,
    improving performance by reducing compilation overhead.

    Example::

        with F.lazy():
            a = Tensor.zeros([5, 5])
            b = a + 1
            c = b * 2
        # No execution yet - all tensors are unrealized
        assert not c.real

        await c.realize  # Now compile and execute
        assert c.real
    """

    #: Subgraph dedup table; armed per instance by ``lazy()``.
    subgraph_cache: dict[Any, Any] | None = None
    #: Output tree structure per keyed subgraph (see
    #: :attr:`GraphRealizationContext.subgraph_out_defs`).
    subgraph_out_defs: dict[str, Any] = {}

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)


def _fresh_subgraph_name(graph: Graph, base: str) -> str:
    """Returns ``base`` or a numbered variant not yet taken on ``graph``.

    ``graph``'s own name counts as taken: a subgraph sharing it would make
    ``mo.call`` resolve to the enclosing graph, failing verification with "Only
    subgraphs can be called". That happens whenever a graph is named after the
    same callable it lowers to a subgraph.
    """
    taken = {*graph._subgraphs, graph.name}
    if base not in taken:
        return base
    i = 1
    while f"{base}_{i}" in taken:
        i += 1
    return f"{base}_{i}"


class GraphRealizationContext(RealizationContext):
    """A realization context for ahead-of-time graph compilation.

    This context is used when building computation graphs that will be compiled
    and executed later (e.g., during :meth:`~max.experimental.nn.Module.compile`). Tensors in this
    context remain as symbolic graph values and cannot be realized.

    Unlike eager contexts, this context does not support executing operations
    immediately. Attempting to realize tensors will raise a TypeError.

    Example::

        graph = Graph("my_model", input_types=[TensorType(...)])
        with GraphRealizationContext(graph) as ctx:
            x = Tensor.from_graph_value(graph.inputs[0])
            y = x + 1  # Creates graph operation, not computation
            graph.output(y)
        # Graph can now be compiled and executed separately
    """

    graph: Graph
    """The graph being constructed in this context."""
    signal_buffers: list[BufferValue] | None
    #: Subgraph dedup table; armed by the root trace, ``None`` inlines.
    subgraph_cache: dict[Any, Any] | None
    #: Output tree structure per keyed subgraph, so a caller that reuses a
    #: cached subgraph (skipping its body trace) can still rebuild the call's
    #: results. Keyed by the same dedup key as :attr:`subgraph_cache`.
    subgraph_out_defs: dict[str, Any]
    #: What names here are relative to: the enclosing call's prefix, or ``""``.
    prefix: str

    def __init__(
        self,
        graph: Graph,
        signal_buffers: list[BufferValue] | None = None,
        prefix: str = "",
    ):
        """Initializes the graph realization context.

        Args:
            graph: The graph to construct operations in.
            signal_buffers: GPU signal buffer graph values for
                multi-device collective ops.
            prefix: What names recorded in this graph are relative to.
        """
        self.graph = graph
        self.signal_buffers = signal_buffers
        self.subgraph_cache = None
        self.subgraph_out_defs = {}
        self.prefix = prefix

    async def realize_all(self) -> list[Tensor]:
        """Raises TypeError - graph contexts cannot realize tensors.

        Raises:
            TypeError: Always raised, as graph contexts are for symbolic
                graph construction only.
        """
        raise TypeError("Can't realize from a graph context.")

    def add_source(self, tensor: Tensor) -> RealizationState:
        """Adds a tensor as a constant in the graph.

        In graph context, source tensors become constant values embedded
        in the graph rather than graph inputs.

        Args:
            tensor: The tensor to embed as a constant.

        Returns:
            RealizationState: The state with the constant graph value.
        """
        return RealizationState((ops.constant(tensor),), self)

    def add_mutable_source(self, tensor: Tensor) -> RealizationState:
        """In graph context, same as add_source (constants are immutable)."""
        return self.add_source(tensor)

    def create_unrealized(
        self,
        values: tuple[GraphValue, ...],
        *,
        mapping: DeviceMapping | None = None,
    ) -> Tensor:
        """Creates a tensor backed by graph value(s)."""
        return _make_unrealized(self, values, mapping)

    def __enter__(self):
        self.graph.__enter__()
        return self

    def __exit__(
        self,
        exception_type: type[Ex] | None,
        exception: Ex | None,
        traceback: TracebackType | None,
    ):
        self.graph.__exit__(exception_type, exception, traceback)


def in_graph_context() -> bool:
    """Returns ``True`` when executing inside a :class:`~max.graph.Graph` context."""
    try:
        _ = Graph.current
    except LookupError:
        return False
    return True


_DEFAULT_REALIZATION_CONTEXT: Callable[[], RealizationContext] = (
    EagerRealizationContext
)


def default_realization_context() -> RealizationContext:
    """Constructs a context for ops realized outside any explicit context."""
    return _DEFAULT_REALIZATION_CONTEXT()


def _set_default_realization_context_raw(
    fn: Callable[[], RealizationContext],
) -> None:
    global _DEFAULT_REALIZATION_CONTEXT
    _DEFAULT_REALIZATION_CONTEXT = fn


def set_default_realization_context(
    fn: Callable[[], RealizationContext],
) -> SetterContext[Callable[[], RealizationContext]]:
    """Sets the constructor used by :func:`default_realization_context`.

    The set takes effect immediately. The returned
    :class:`~max.experimental.support.SetterContext` may be used as a
    context manager to restore the previous constructor on exit, or
    discarded to keep the new one.

    Args:
        fn: A zero-argument callable returning a new realization context,
            invoked each time an op realizes outside any explicit context.

    Returns:
        An undo handle restoring the previously installed constructor.
    """
    previous = _DEFAULT_REALIZATION_CONTEXT
    _set_default_realization_context_raw(fn)
    return SetterContext(fn, previous, _set_default_realization_context_raw)


@contextlib.contextmanager
def ensure_context() -> Generator[None]:
    """Ensures a realization context exists for Tensor / TensorValue conversion."""
    if current_realization_context(None) is not None:
        yield
        return
    ctx: RealizationContext = (
        GraphRealizationContext(Graph.current)
        if in_graph_context()
        else default_realization_context()
    )
    with ctx, realization_context(ctx):
        yield


@contextlib.contextmanager
def lazy() -> Generator[None]:
    """Defers tensor realization until explicitly awaited."""
    with LazyRealizationContext() as ctx, realization_context(ctx):
        # Arm subgraph dedup: a lazy block builds one graph, like compile.
        ctx.subgraph_cache = {}
        ctx.subgraph_out_defs = {}
        yield


def subgraph_context() -> (
    GraphRealizationContext | LazyRealizationContext | None
):
    """The context a subgraph would be defined on here, or what inlines instead.

    Returns:
        The root trace context when one is capturing and armed for
        deduplication, and :obj:`None` when a body belongs inline.
    """
    ctx = current_realization_context(None)
    if (
        isinstance(ctx, (GraphRealizationContext, LazyRealizationContext))
        and ctx.subgraph_cache is not None
    ):
        return ctx
    return None


def define_subgraph(
    ctx: GraphRealizationContext | LazyRealizationContext,
    name: str,
    input_types: Sequence[Type[Any]],
    build_body: Callable[[list[Value[Any]]], Sequence[Value[Any]]],
    *,
    key: str | None = None,
) -> Graph:
    """Defines a deduplicated subgraph on ``ctx`` and returns it.

    Works for any graph-building context — ahead-of-time graph or lazy — since
    it reasons only in graph values and so is independent of when ``ctx``
    realizes. ``build_body(inputs) -> outputs`` traces the body.

    Bodies are deduplicated so a repeated call reuses one definition. When
    ``key`` is given, it is the dedup key: two calls with the same ``key`` share
    a definition and the caller vouches that their bodies match. When ``key`` is
    ``None``, the body's IR hash is the key, so only bodies that print to
    identical IR share.

    ``ctx``'s signal buffers are appended as trailing subgraph inputs so
    collectives in the body work; the caller passes the matching signal values
    (``ctx.signal_buffers``) when it emits :func:`~max.graph.ops.call`.
    """
    cache = ctx.subgraph_cache
    if cache is None:
        raise TypeError("define_subgraph requires the root trace context.")

    # A keyed hit answers before tracing; the caller keeps its own out defs.
    if key is not None and (found := cache.get(key)) is not None:
        return found[0]

    with open_subgraph(ctx, key or name, input_types) as subgraph:
        n = len(input_types)
        subgraph.output(*build_body(list(subgraph.inputs[:n])))
    return share_subgraph(ctx, subgraph, None, key=key)[0]


@contextlib.contextmanager
def open_subgraph(
    ctx: GraphRealizationContext | LazyRealizationContext,
    name: str,
    input_types: Sequence[Type[Any]],
    *,
    prefix: str = "",
) -> Generator[Graph]:
    """Opens a fresh subgraph on ``ctx``, entered under its own child context.

    ``ctx``'s signal buffers become trailing inputs so collectives in the body
    work; the caller appends ``ctx.signal_buffers`` to its
    :func:`~max.graph.ops.call` operands to match.

    Args:
        ctx: The root trace context to add the subgraph to.
        name: The base name for the symbol, numbered when already taken.
        input_types: Operand types, excluding the trailing signal buffers.
        prefix: What names declared in the body are relative to.

    Yields:
        The subgraph, whose leading inputs are the operands'.
    """
    signals = ctx.signal_buffers or []
    subgraph = ctx.graph.add_subgraph(
        _fresh_subgraph_name(ctx.graph, name),
        input_types=[*input_types, *(b.type for b in signals)],
        custom_extensions=ctx.graph.kernel_libraries_paths,
        devices=list(ctx.graph.device_chains),
    )
    child = GraphRealizationContext(
        subgraph,
        signal_buffers=[i.buffer for i in subgraph.inputs[len(input_types) :]]
        or None,
        prefix=prefix,
    )
    # child.subgraph_cache stays None, so a nested call in the body inlines.
    with realization_context(child), child:
        yield subgraph


def share_subgraph(
    ctx: GraphRealizationContext | LazyRealizationContext,
    subgraph: Graph,
    entry: Entry,
    *,
    key: str | None = None,
) -> tuple[Graph, Entry]:
    """Files a traced body for reuse, or discards it for an identical one.

    A body already filed under the same ``key`` -- or, when no key is given,
    one whose IR hashes the same -- stands in for this one, which is erased.

    Args:
        ctx: The root trace context owning the deduplication table.
        subgraph: The body just traced, erased here when an identical body is
            already filed.
        entry: Whatever the caller needs at the call site but can only learn
            while tracing, filed beside the body and handed back unchanged.
            ``as_subgraph`` files the output tree structure, which is what
            rebuilds a return value from the call's flat results.
        key: What identifies the body, or :obj:`None` to hash the traced IR.

    Returns:
        The subgraph to call and the entry filed with it -- the already-filed
        body's, when this one duplicated it.

    Raises:
        TypeError: If ``ctx`` owns no deduplication table.
    """
    if ctx.subgraph_cache is None:
        raise TypeError("subgraph dedup requires the root trace context.")
    if key is None:
        # The symbol name is what a reader sees, so it is not the key: naming
        # it after one would print the argument structure into every mo.call.
        asm = subgraph._mlir_op.get_asm(
            assume_verified=True,
            enable_debug_info=False,
            print_generic_op_form=True,
            use_local_scope=True,
        ).replace(f'"{subgraph.name}"', '"_"', 1)
        key = hashlib.sha256(asm.encode()).hexdigest()
    if (filed := ctx.subgraph_cache.get(key)) is not None:
        ctx.graph._subgraphs.pop(subgraph.name, None)
        subgraph._mlir_op.erase()
        return filed
    ctx.subgraph_cache[key] = (subgraph, entry)
    return subgraph, entry


__all__ = [
    "EagerRealizationContext",
    "GraphRealizationContext",
    "LazyRealizationContext",
    "default_realization_context",
    "ensure_context",
    "in_graph_context",
    "lazy",
    "seed",
    "set_default_realization_context",
    "set_seed",
]
