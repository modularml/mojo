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
"""APIs to trace and compile callables.

:func:`compile` turns a function over tensors into a compiled one in two
calls: first pass a spec (dtype, shape, device) for each tensor argument,
then call the result on real tensors. Arguments that are not tensors are
fixed while tracing, so pass them the same way both times. :func:`stage`
stops after tracing, for inspecting the graph.

.. code-block:: python

    from max.driver import CPU
    from max.dtype import DType
    from max.experimental import compilation
    from max.experimental.tensor import Tensor
    from max.graph import DeviceRef, TensorType

    def step(x: Tensor, *, gain: float) -> Tensor:
        return x * gain

    x_spec = TensorType(DType.float32, ["batch", 2], device=DeviceRef.CPU())

    run = compilation.compile(step)(x_spec, gain=3.0)
    out = run(Tensor.ones([4, 2], device=CPU()), gain=3.0)  # "batch" accepts 4

.. invisible-code-block: python

    import numpy as np

    np.testing.assert_allclose(out.to_numpy(), np.full((4, 2), 3.0))
"""

from __future__ import annotations

import dataclasses
import functools
import itertools
import re
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Generic, ParamSpec, TypeAlias, TypeVar, get_args

from max.driver import Accelerator, Buffer, Device, DLPackArray
from max.engine import CompiledModel, Model
from max.experimental import tree_utils as tree
from max.experimental.realization_context import (
    GraphRealizationContext,
    _cached_signal_buffers,
    in_graph_context,
    open_subgraph,
    share_subgraph,
    subgraph_context,
)
from max.experimental.sharding import (
    DeviceMesh,
    DistributedBufferType,
    PlacementMapping,
    Replicated,
    TensorLayout,
)
from max.experimental.sharding.placements import local_shard_shape_from_global
from max.experimental.support import _session
from max.experimental.tensor import Tensor, realization_context
from max.experimental.tree_utils import TreeDef
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Graph,
    StaticDim,
    TensorType,
    TensorValue,
    Value,
    ops,
)

_P = ParamSpec("_P")
_R = TypeVar("_R")

Spec: TypeAlias = (
    TensorLayout | TensorType | Tensor | BufferType | DistributedBufferType
)
"""What one tensor argument may be staged as."""

NormalizedSpec: TypeAlias = TensorLayout | BufferType | DistributedBufferType
"""A :data:`Spec` that :func:`as_layout` has normalized, and so what the graph
reads."""

# The types accepted as one input spec, and so the leaves of a spec tree.
_SPEC_TYPES: tuple[type, ...] = get_args(Spec)
# The types a graph boundary carries, and so the leaves of a staged result.
_GRAPH_VALUE_TYPES = (BufferValue, TensorValue)


def _sanitized_graph_name(fn: Callable[..., object]) -> str:
    """Returns ``fn``'s name with non-identifier characters folded away."""
    raw = getattr(fn, "__name__", None) or type(fn).__name__
    return re.sub(r"\W+", "_", raw).strip("_") or "fn"


def as_layout(spec: Spec) -> NormalizedSpec:
    """Normalizes a tensor spec into a layout.

    Applied to every tensor argument, which is what lets a real tensor stand
    in for a spec.

    .. code-block:: python

        from max.dtype import DType
        from max.experimental.compilation import as_layout
        from max.graph import DeviceRef, TensorType

        layout = as_layout(
            TensorType(DType.float32, [4], device=DeviceRef.CPU())
        )

    .. invisible-code-block: python

        assert layout.dtype == DType.float32
        assert layout.mesh.num_devices == 1

    Args:
        spec: A :class:`~max.experimental.sharding.TensorLayout`, a
            :class:`~max.graph.TensorType`, an example
            :class:`~max.experimental.tensor.Tensor`, or a buffer spec.

    Returns:
        The equivalent layout, or ``spec`` itself when it is a buffer spec.

    Raises:
        TypeError: If ``spec`` is none of the accepted forms.
    """
    if isinstance(spec, (TensorLayout, BufferType, DistributedBufferType)):
        return spec
    if isinstance(spec, Tensor):
        return TensorLayout(spec.dtype, spec.shape, spec.mapping)
    if isinstance(spec, TensorType):
        mesh = DeviceMesh.single(spec.device.to_device())
        return TensorLayout(
            spec.dtype, spec.shape, PlacementMapping(mesh, (Replicated(),))
        )
    raise TypeError(
        f"expected a TensorLayout, TensorType, Tensor, BufferType or "
        f"DistributedBufferType spec, got {type(spec).__name__}: {spec!r}"
    )


def _graph_input_types(
    spec: NormalizedSpec,
) -> list[TensorType | BufferType]:
    """The graph input types one layout contributes, in mesh order."""
    if isinstance(spec, BufferType):
        return [spec]
    if isinstance(spec, TensorLayout):
        shapes = local_shard_shape_from_global(
            spec.shape, spec.mesh, spec.placements
        )
        return [
            TensorType(spec.dtype, shape, DeviceRef.from_device(device))
            for shape, device in zip(shapes, spec.mesh.devices, strict=True)
        ]
    return list(spec.local_types)


def _fills_one_slot(value: object) -> bool:
    """Whether ``value`` fills one argument slot rather than nesting further."""
    # Deferring to ``tree.is_node`` keeps this agreeing with ``_record_graph``'s walk.
    return isinstance(value, _SPEC_TYPES) or not tree.is_node(value)


@dataclasses.dataclass(frozen=True)
class _Signature:
    """How a flat graph boundary maps onto a Python call."""

    # ``Any`` because a call carries whatever the caller passed, uninspected;
    # no union closes it and ``object`` would not survive the round trip.
    in_specs: Any
    """One layout per tensor argument, nested as the staged call passed them."""

    out_structure: TreeDef
    """How to rebuild the return value from the graph's flat results."""

    def flatten(
        self, args: tuple[Any, ...], kwargs: Mapping[str, Any]
    ) -> list[Buffer]:
        """Checks a call against the specs and flattens it into graph order.

        Args:
            args: One :class:`~max.experimental.tensor.Tensor` per tensor
                argument, nested as the staged call passed them.
            kwargs: The staged signature's keyword arguments.

        Returns:
            One buffer per graph input, a distributed argument expanded into
            one per shard.

        Raises:
            TypeError: If an argument is not a
                :class:`~max.experimental.tensor.Tensor` where one belongs.
            ValueError: If the arguments do not match what was staged.
        """
        # Arity first: the path mismatch below reports it only as an absence.
        wanted_args, wanted_kwargs = self.in_specs
        if len(args) != len(wanted_args) or set(kwargs) != set(wanted_kwargs):
            raise ValueError(
                f"expected {len(wanted_args)} positional argument(s), got "
                f"{len(args)}, and keywords {sorted(wanted_kwargs)}, got "
                f"{sorted(kwargs)}."
            )

        # A call boundary is positional, so a walk across it duplicates.
        wanted = tree.paths(self.in_specs, leaf=_fills_one_slot)
        given = tree.paths((args, dict(kwargs)), leaf=_fills_one_slot)
        if list(wanted) != list(given):
            raise ValueError(
                f"tree structure mismatch: expected {list(wanted)}, "
                f"got {list(given)}"
            )

        buffers: list[Buffer] = []
        for path, layout in wanted.items():
            arg, name = given[path], path.partition(".")[2]
            if not isinstance(layout, _SPEC_TYPES):
                # __eq__ may be elementwise, so only a clean True counts.
                try:
                    matches = arg is layout or bool(arg == layout)
                except Exception:
                    matches = False
                if not matches:
                    raise ValueError(
                        f"tree structure mismatch: argument {name} staged as "
                        f"{layout!r}, got {arg!r}"
                    )
                continue
            if not isinstance(arg, Tensor):
                raise TypeError(
                    f"argument {name}: expected a Tensor, got "
                    f"{type(arg).__name__}; use execute_raw()"
                )
            shards = arg.local_shards
            if isinstance(layout, (BufferType, DistributedBufferType)):
                # Dtype and shard count only: extent is whatever was allocated.
                expected = _graph_input_types(layout)
                if len(shards) != len(expected) or arg.dtype != layout.dtype:
                    raise ValueError(
                        f"argument {name}: expected {layout.dtype} across "
                        f"{len(expected)} device(s), got {arg.dtype} across "
                        f"{len(shards)}"
                    )
                buffers.extend(shard.driver_tensor for shard in shards)
                continue
            # Every non-buffer spec went through ``as_layout`` when staged.
            assert isinstance(layout, TensorLayout)
            devices = [shard.device for shard in shards]
            # A symbolic dim was staged to accept any size, so only match statics.
            if (
                (arg.dtype, len(arg.shape)) != (layout.dtype, len(layout.shape))
                or devices != list(layout.mesh.devices)
                or any(
                    isinstance(dim, StaticDim) and dim != size
                    for dim, size in zip(layout.shape, arg.shape, strict=True)
                )
            ):
                raise ValueError(
                    f"argument {name}: expected {layout.dtype} shape "
                    f"{list(layout.shape)} on {list(layout.mesh.devices)}, got "
                    f"{arg.dtype} shape {list(arg.shape)} on {devices}"
                )
            buffers.extend(shard.driver_tensor for shard in shards)
        return buffers

    def unflatten(self, buffers: Sequence[Buffer]) -> Any:
        """Rebuilds the staged callable's return value from flat results.

        The counterpart of :meth:`flatten`, on the way back out.

        Args:
            buffers: The graph's results, flat and in graph order.

        Returns:
            The staged callable's return value, nested as it returned it.
        """
        return tree.unflatten(self.out_structure, list(buffers))


class StagedGraph(Generic[_P, _R]):
    """A traced graph, ready to inspect or compile.

    Returned by :func:`stage`. Printing one renders the MLIR of the whole
    module, subgraph bodies included. Call :meth:`compile` to make it
    runnable.

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, TensorType

        def scale(x: Tensor) -> Tensor:
            return x * 2

        spec = TensorType(DType.float32, [4], device=DeviceRef.CPU())
        staged = compilation.stage(scale)(spec)

    .. invisible-code-block: python

        assert "mo.mul" in str(staged)
    """

    graph: Graph
    """The recorded graph."""

    # Internal state, and so kept out of a constructor: how a call crosses the
    # graph's boundary, and the accelerators whose collectives need signals.
    _signature: _Signature
    _signal_device_ids: tuple[int, ...]

    @classmethod
    def _new(
        cls,
        graph: Graph,
        signature: _Signature,
        signal_device_ids: tuple[int, ...] = (),
    ) -> StagedGraph[_P, _R]:
        self = cls()
        self.graph = graph
        self._signature = signature
        self._signal_device_ids = signal_device_ids
        return self

    def __str__(self) -> str:
        """Returns the whole module's MLIR, shared subgraph bodies included.

        The graph's own op names the bodies it calls but does not contain
        them; rendering only that would quietly weaken any check that an op
        is absent.
        """
        return str(self.graph._module)

    # The default repr would pass a test asserting an op is absent.
    __repr__ = __str__

    def compile(
        self, *, weights: Mapping[str, DLPackArray] | None = None
    ) -> CompiledCallable[_P, _R]:
        """Compiles the graph into a :class:`CompiledCallable`.

        Args:
            weights: Data for the external constants the graph declares, keyed
                as the graph names them, one entry per shard of a distributed
                weight.

        Returns:
            The compiled function, called on real tensors.
        """
        return CompiledCallable._new(
            self._signature,
            _session().compile(self.graph),
            self._signal_device_ids,
            dict(weights or {}),
        )


class CompiledCallable(Generic[_P, _R]):
    """A compiled function over tensors.

    Returned by :func:`compile`. Call it like the original function, with a
    real tensor in each spec's place. The first call also binds
    :attr:`weights` and allocates device memory; :meth:`export_mef` needs
    neither, so it works before any call.

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, TensorType

        def scale(x: Tensor) -> Tensor:
            return x * 2

        spec = TensorType(DType.float32, [3], device=DeviceRef.CPU())

        run = compilation.compile(scale)(spec)  # compiles here
        run.export_mef("scale.mef")             # no weights, no device memory
        out = run(Tensor.ones([3], device=CPU()))  # [2.0, 2.0, 2.0]

    .. invisible-code-block: python

        import numpy as np

        np.testing.assert_allclose(out.to_numpy(), [2.0, 2.0, 2.0])
    """

    weights: Mapping[str, DLPackArray]
    """Data for the external constants the graph declares, keyed as the graph
    names them, one entry per shard of a distributed weight."""

    # Internal state, and so kept out of a constructor: how a call crosses the
    # graph's boundary, the compiled graph, and the accelerators whose
    # collectives need signals.
    _signature: _Signature
    _artifact: CompiledModel
    _signal_device_ids: tuple[int, ...]

    @classmethod
    def _new(
        cls,
        signature: _Signature,
        artifact: CompiledModel,
        signal_device_ids: tuple[int, ...] = (),
        weights: Mapping[str, DLPackArray] | None = None,
    ) -> CompiledCallable[_P, _R]:
        self = cls()
        self._signature = signature
        self._artifact = artifact
        self._signal_device_ids = signal_device_ids
        self.weights = dict(weights or {})
        return self

    @functools.cached_property
    def _engine_model(self) -> Model:
        """The initialized model this calls, with :attr:`weights` bound in."""
        return _session().init(
            self._artifact, weights_registry=dict(self.weights)
        )

    @property
    def _signal_buffers(self) -> list[Buffer]:
        """Buffers for multi-device collectives, allocated once, not per call."""
        ids = self._signal_device_ids
        return _cached_signal_buffers(ids)[0] if ids else []

    def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R:
        """Runs the compiled function on real tensors.

        Args:
            args: One :class:`~max.experimental.tensor.Tensor` per tensor
                argument of the staged signature, nested as it passed them.
            kwargs: The staged signature's keyword arguments.

        Returns:
            The staged callable's return value.

        Raises:
            TypeError: If an argument is not a
                :class:`~max.experimental.tensor.Tensor` where one belongs.
            ValueError: If the arguments do not match what was staged.
        """
        buffers = self._signature.flatten(args, kwargs)
        return self._signature.unflatten(self.execute_raw(*buffers))

    def execute_raw(self, *buffers: Buffer) -> list[Buffer]:
        """Executes the graph on raw buffers, appending the signal buffers.

        Args:
            buffers: One per graph input, a distributed argument expanded into
                one per shard.

        Returns:
            The result buffers, flat and in graph order.
        """
        return list(self._engine_model(*buffers, *self._signal_buffers))

    def export_mef(self, path: str | Path) -> None:
        """Writes the compiled graph to a MEF file at ``path``.

        MEF is the binary format the runtime executes. Writing one serializes
        the compiled graph directly, without binding :attr:`weights` or
        allocating device memory. Read it back with :func:`max.engine.read` to
        skip compiling again.

        Args:
            path: Where to write the file.
        """
        self._artifact.export_mef(path)


def _signal_device_ids(
    layouts: Sequence[NormalizedSpec], signal_devices: Iterable[Device]
) -> tuple[int, ...]:
    """The accelerators whose collectives need signal buffers.

    Args:
        layouts: The input layouts, whose meshes span devices of their own.
        signal_devices: Devices taking part beyond what the inputs span.

    Returns:
        Their ids, or empty when fewer than two accelerators take part.
    """
    spanned = (
        device
        for layout in layouts
        if not isinstance(layout, BufferType) and layout.mesh.num_devices > 1
        for device in layout.mesh.devices
    )
    ids = tuple(
        dict.fromkeys(
            device.id
            for device in itertools.chain(spanned, signal_devices)
            if isinstance(device, Accelerator)
        )
    )
    return ids if len(ids) > 1 else ()


def _argument_tensor(
    ctx: GraphRealizationContext,
    values: Iterator[Value[Any]],
    spec: NormalizedSpec,
) -> Tensor:
    """One staged argument, rebuilt from the graph inputs its spec claims.

    Args:
        ctx: The context the rebuilt tensor records into.
        values: The graph's inputs, in spec order, drained by what ``spec``
            spans.
        spec: What the argument was staged as.

    Returns:
        The tensor to pass ``fn`` in that argument's place.
    """
    if isinstance(spec, BufferType):
        return Tensor.from_graph_value(next(values).buffer)
    shards = itertools.islice(values, spec.mesh.num_devices)
    if isinstance(spec, DistributedBufferType):
        return ctx.create_unrealized(
            tuple(value.buffer for value in shards),
            mapping=PlacementMapping(spec.mesh, spec.placements),
        )
    return ctx.create_unrealized(
        tuple(value.tensor for value in shards), mapping=spec.mapping
    )


def stage(
    fn: Callable[_P, _R],
    *,
    name: str | None = None,
    custom_extensions: Iterable[Path] = (),
    allow_subgraphs: bool = True,
    signal_devices: Iterable[Device] = (),
    is_device_graph: bool = False,
) -> Callable[..., StagedGraph[_P, _R]]:
    """Traces ``fn`` into a graph, without compiling it.

    Call the returned function with one spec per tensor argument of ``fn``
    to get the :class:`StagedGraph`, which prints as MLIR. Use
    :func:`compile` to run ``fn`` instead.

    Tensor arguments are given as specs: a :class:`~max.graph.TensorType`, a
    :class:`~max.experimental.sharding.TensorLayout`, or a buffer type. A real
    :class:`~max.experimental.tensor.Tensor` also works, converted by
    :func:`as_layout`, which fixes every dimension. Pass a type to keep one
    symbolic.

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, TensorType

        def combine(kv: dict[str, Tensor], alpha: float) -> Tensor:
            return (kv["a"] + kv["b"]) * alpha

        spec = TensorType(DType.float32, [2], device=DeviceRef.CPU())

        # Each tensor in the container is an input; alpha is baked in.
        staged = compilation.stage(combine)({"a": spec, "b": spec}, 2.0)
        print(staged)

    .. invisible-code-block: python

        assert len(staged.graph.inputs) == 2

    Args:
        fn: The callable to record, over
            :class:`~max.experimental.tensor.Tensor` values or containers of
            them.
        name: The graph's name. Defaults to ``fn``'s own name.
        custom_extensions: Paths to custom Mojo kernel libraries.
        allow_subgraphs: Whether :func:`as_subgraph` bodies become shared
            subgraphs rather than inlining into the caller.
        signal_devices: Devices taking part in collectives beyond what the
            specs span.
        is_device_graph: Whether to record a device graph.

    Returns:
        A callable taking one spec per argument of ``fn``, as :func:`as_layout`
        accepts them, and returning the :class:`StagedGraph`.
    """

    def record(*args: Any, **kwargs: Any) -> StagedGraph[_P, _R]:
        # Positional, like _Signature.flatten: one graph input per argument slot.
        in_specs = tree.map(as_layout, (args, dict(kwargs)), leaf=_SPEC_TYPES)
        layouts, structure = tree.flatten(in_specs, leaf=_SPEC_TYPES)
        types = [t for spec in layouts for t in _graph_input_types(spec)]
        ids = _signal_device_ids(layouts, signal_devices)
        graph = Graph(
            name or _sanitized_graph_name(fn),
            input_types=[
                *types,
                *(_cached_signal_buffers(ids)[1] if ids else []),
            ],
            custom_extensions=custom_extensions,
            is_device_graph=is_device_graph,
        )
        ctx = GraphRealizationContext(
            graph,
            signal_buffers=[i.buffer for i in graph.inputs[len(types) :]]
            or None,
        )
        if allow_subgraphs:
            ctx.subgraph_cache = {}

        values = iter(graph.inputs[: len(types)])
        with realization_context(ctx), ctx:
            in_args, in_kwargs = tree.unflatten(
                structure,
                [_argument_tensor(ctx, values, spec) for spec in layouts],
            )
            # Also positional on the way out: ``return x, x`` is two results.
            flat, out_structure = tree.flatten(
                fn(*in_args, **in_kwargs), leaf=_GRAPH_VALUE_TYPES
            )
            graph.output(*flat)
        return StagedGraph._new(graph, _Signature(in_specs, out_structure), ids)

    return record


def compile(
    fn: Callable[_P, _R],
    *,
    weights: Mapping[str, DLPackArray] | None = None,
    name: str | None = None,
    custom_extensions: Iterable[Path] = (),
    allow_subgraphs: bool = True,
    signal_devices: Iterable[Device] = (),
    is_device_graph: bool = False,
) -> Callable[..., CompiledCallable[_P, _R]]:
    """Traces and compiles ``fn``.

    Call the returned function with one spec per tensor argument of ``fn``
    to get the :class:`CompiledCallable`; call that on real tensors.

    Tensor arguments are given as specs: a :class:`~max.graph.TensorType`, a
    :class:`~max.experimental.sharding.TensorLayout`, or a buffer type. A real
    :class:`~max.experimental.tensor.Tensor` also works, converted by
    :func:`as_layout`, which fixes every dimension. Pass a type to keep one
    symbolic.

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental import functional as F
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, TensorType

        w_type = TensorType(DType.float32, [2], device=DeviceRef.CPU())

        def layer(x: Tensor) -> Tensor:
            return x * F.constant_external("w", w_type)

        x_spec = TensorType(DType.float32, ["batch", 2], device=DeviceRef.CPU())
        w = Tensor.ones([2], device=CPU()) * 3

        run = compilation.compile(layer, weights={"w": w})(x_spec)
        out = run(Tensor.ones([4, 2], device=CPU()))  # 4 rows of 3.0

    .. invisible-code-block: python

        import numpy as np

        np.testing.assert_allclose(out.to_numpy(), np.full((4, 2), 3.0))

    Args:
        fn: The callable to compile, over
            :class:`~max.experimental.tensor.Tensor` values or containers of
            them.
        weights: Data for the external constants the graph declares, keyed as
            the graph names them, one entry per shard of a distributed weight.
        name: The graph's name. Defaults to ``fn``'s own name.
        custom_extensions: Paths to custom Mojo kernel libraries.
        allow_subgraphs: Whether :func:`as_subgraph` bodies become shared
            subgraphs rather than inlining into the caller.
        signal_devices: Devices taking part in collectives beyond what the
            specs span.
        is_device_graph: Whether to record a device graph.

    Returns:
        A callable taking one spec per argument of ``fn`` and returning the
        :class:`CompiledCallable`.
    """

    def stage_and_compile(
        *args: Any, **kwargs: Any
    ) -> CompiledCallable[_P, _R]:
        return stage(
            fn,
            name=name,
            custom_extensions=custom_extensions,
            allow_subgraphs=allow_subgraphs,
            signal_devices=signal_devices,
            is_device_graph=is_device_graph,
        )(*args, **kwargs).compile(weights=weights)

    return stage_and_compile


class _InferKey:
    """Sentinel for "no key given", which a caller's own ``None`` is not."""

    def __repr__(self) -> str:
        return "<inferred>"


# ``Any``-typed so ``key`` below renders as the ``str | None`` callers pass.
_INFER_KEY: Any = _InferKey()


def _inferred_key(fn: Callable[..., Any]) -> str | None:
    """A dedup key for ``fn``, or ``None`` when it does not fix its body."""
    if getattr(fn, "__closure__", None) is not None:
        return None
    if hasattr(fn, "__self__"):
        return None
    # A default is captured state the argument structure never sees.
    if getattr(fn, "__defaults__", None) or getattr(fn, "__kwdefaults__", None):
        return None
    qualname = getattr(fn, "__qualname__", None)
    if qualname is None:
        return None
    return f"{getattr(fn, '__module__', '?')}.{qualname}"


def as_subgraph(
    fn: Callable[_P, _R],
    *,
    name: str | None = None,
    prefix: str = "",
    key: str | None = _INFER_KEY,
) -> Callable[_P, _R]:
    """Lowers ``fn`` to one shared subgraph body per distinct stage.

    Usable as a decorator or at the call site.

    .. code-block:: python

        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, TensorType

        @compilation.as_subgraph
        def block(x: Tensor) -> Tensor:
            return x * 2

        spec = TensorType(DType.float32, [4], device=DeviceRef.CPU())
        staged = compilation.stage(lambda x: block(block(block(x))))(spec)

    .. invisible-code-block: python

        # One body definition, called three times.
        assert str(staged).count("mo.graph @block") == 1
        assert str(staged).count("mo.call @block") == 3

    A shared body also shares the weights it declares. At the call site,
    ``prefix`` gives each site its own weights out of the one body:

    .. code-block:: python

        from max.driver import CPU
        from max.dtype import DType
        from max.experimental import compilation
        from max.experimental import functional as F
        from max.experimental.tensor import Tensor
        from max.graph import DeviceRef, TensorType

        w_type = TensorType(DType.float32, [1], device=DeviceRef.CPU())

        def block(x: Tensor) -> Tensor:
            return x * F.constant_external("w", w_type, is_placeholder=True)

        def model(x: Tensor) -> Tensor:
            for layer in ("layers.0.", "layers.1."):
                x = compilation.as_subgraph(block, prefix=layer)(x)
            return x

        one = Tensor.ones([1], device=CPU())
        weights = {"layers.0.w": one * 2, "layers.1.w": one * 10}

        run = compilation.compile(model, weights=weights)(w_type)
        out = run(one)  # [20.0]

    .. invisible-code-block: python

        import numpy as np

        np.testing.assert_allclose(out.to_numpy(), [20.0])

    Args:
        fn: The callable to lower.
        name: The subgraph's name. Defaults to ``fn``'s own name.
        prefix: Prepended to the relative weight names the body declares, so
            each call site resolves its own weights from a shared body.
        key: What identifies this body beyond its arguments, completed here
            with the argument structure and operand types. Pass :obj:`None` to
            compare the staged IR instead. Omitted, a key is derived from
            ``fn`` where that is sound.

    Returns:
        A callable with ``fn``'s signature that emits a call to the shared body.

    Raises:
        TypeError: If called outside a capture. Call ``fn`` directly to run
            eagerly.
    """
    name = name or _sanitized_graph_name(fn)
    if key is _INFER_KEY:
        key = _inferred_key(fn)

    @functools.wraps(fn)
    def emit_subgraph_call(*args: _P.args, **kwargs: _P.kwargs) -> Any:
        if not in_graph_context():
            raise TypeError(
                f"as_subgraph({name}) needs a capture (compile() / stage() / "
                "F.lazy()); call it directly to run eagerly"
            )
        ctx = subgraph_context()
        if ctx is None:
            return fn(*args, **kwargs)
        # Positional, like _Signature.flatten: one operand per argument slot.
        operands, structure = tree.flatten(
            (args, kwargs), leaf=_GRAPH_VALUE_TYPES
        )
        # Completed here, where the operands are flat and the prefix is known.
        types = [operand.type for operand in operands]
        full_key = (
            None
            if key is None
            else f"{key}|{name}|{structure}|{types}|{bool(prefix)}"
        )

        cache = ctx.subgraph_cache
        assert cache is not None, "subgraph_context() returns armed contexts"
        # A None key is never stored: share_subgraph hashes the IR instead.
        if (found := cache.get(full_key)) is None:
            with open_subgraph(ctx, name, types, prefix=prefix) as body:
                in_args, in_kwargs = tree.unflatten(
                    structure, body.inputs[: len(operands)]
                )
                # Also positional out: ``return x, x`` is two results.
                outputs, out_structure = tree.flatten(
                    fn(*in_args, **in_kwargs), leaf=_GRAPH_VALUE_TYPES
                )
                body.output(*outputs)
            found = share_subgraph(ctx, body, out_structure, key=full_key)

        subgraph, out_structure = found
        results = ops.call(
            subgraph, *operands, *(ctx.signal_buffers or []), prefix=prefix
        )
        return tree.unflatten(out_structure, list(results))

    return emit_subgraph_call
