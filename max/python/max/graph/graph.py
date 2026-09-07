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
"""Core graph primitives."""

from __future__ import annotations

import contextlib
import functools
import inspect
import io
import itertools
import traceback
from collections import OrderedDict
from collections.abc import Callable, Generator, Iterable, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, TypeGuard, TypeVar, cast

from max import mlir
from max._core import Attribute as _Attribute
from max._core import Block, OpBuilder, Operation
from max._core import Type as _Type
from max._core import Value as _Value
from max._core import graph as _graph
from max._core.dialects import builtin, kgen
from max._core.dialects import kgen as _kgen
from max._core.dialects import mo as _mo
from max._core.dialects.m import DeviceInfoAttr as _DeviceInfoAttr
from max._core.driver import CPU, Accelerator, Device, accelerator_count
from max._core.engine import InferenceSession as _InferenceSession
from max._mlir_context import (
    default_mlir_context,
    ensure_default_mlir_context,
    in_default_mlir_context,
)
from max.mlir.dialects import mo
from mojo.paths import (
    _build_mojo_source_package,
    is_mojo_binary_package_path,
    is_mojo_source_package_path,
)

from .type import (
    BufferType,
    DeviceRef,
    SymbolicDim,
    TensorType,
    Type,
    _ChainType,
)
from .value import BufferValue, TensorValue, TensorValueLike, Value, _ChainValue
from .weight import Weight

CURRENT_GRAPH: ContextVar[Graph] = ContextVar("CURRENT_GRAPH")
# Stack of active Graph.profile_scope() scopes, outermost first. Each entry is
# a (name, color) tuple; color is None when unset. A tuple (not a list) so
# entering/exiting a scope is a ContextVar set()/reset() rather than a
# mutation shared across contexts.
# NOTE: this is a single module-level ContextVar, not scoped per Graph.
# Building a nested Graph while a sibling graph's profile_scope is active will
# inherit that scope's label; this is not currently supported and should be
# avoided.
_CURRENT_PROFILE_SCOPES: ContextVar[tuple[tuple[str, str | None], ...]] = (
    ContextVar("_CURRENT_PROFILE_SCOPES", default=())
)
_KERNEL_LIBRARY_PATHS_ATTR_NAME = "_kernel_library_paths"
_DEVICE_INFO_MAPPING_ATTR_NAME = "mo.device_info_mapping"

T = TypeVar("T")


def _is_chain_value(value: _Value[Any]) -> TypeGuard[_Value[_mo.ChainType]]:
    return isinstance(value.type, _mo.ChainType)


class _DeviceChainMap(OrderedDict[DeviceRef, _ChainValue]):
    """Dictionary that lazily registers per-device chains on access."""

    def __init__(self, graph: Graph) -> None:
        super().__init__()
        self._graph = graph

    @staticmethod
    def _sort_key(device: DeviceRef) -> tuple[str, int]:
        return (device.device_type.value, device.id)

    def __getitem__(self, device: DeviceRef) -> _ChainValue:
        if device not in self:
            self[device] = (
                self._graph._add_chain_block_arg()
                if self._graph._has_chain_input
                # prevent block arguments from being added when we're inside a
                # region of a control flow op (mo.if, mo.while). For simplicity
                # currently control flow ops merge new device chains with the
                # global chain instead of shuttling device chains from the
                # parent scope via block arguments. Any device chains that
                # exist prior to the control flow op will be used as expected.
                and self._graph._graph_body == self._graph._current_block
                # Seed new device chains from the host (``DeviceRef.CPU()``)
                # chain, which is pre-populated at graph construction.
                else super().__getitem__(DeviceRef.CPU())
            )
        return super().__getitem__(device)

    def __setitem__(self, device: DeviceRef, chain: _ChainValue) -> None:
        assert isinstance(chain, _ChainValue)
        super().__setitem__(device, chain)

    def __iter__(self):
        return iter(sorted(super().keys(), key=self._sort_key))

    def _value(self, device: DeviceRef) -> _ChainValue:
        return super().__getitem__(device)

    def ordered_values(self) -> Generator[_ChainValue]:
        for device in self:
            yield self._value(device)

    def pack(self, values: Iterable[Any] = ()) -> list[Any]:
        """Append the current chain state to ``values``.

        Returns ``[*values, *self.ordered_values()]`` — useful for bundling
        user values with the chain state for control-flow op operands /
        block-arg signatures / terminator operands.
        """
        return [*values, *self.ordered_values()]

    def unpack(self, values: Sequence[Any]) -> list[Any]:
        """Take trailing chain values off ``values`` and return user values.

        Inverse of :meth:`pack`: assumes the last ``len(self)`` entries of
        ``values`` are chain values in the same order as iteration over
        ``self``, assigns them back to ``self[device]``, and returns the
        rest. Side-effecting on ``self``.
        """
        chain_count = len(self)
        user_values = list(values[:-chain_count] if chain_count else values)
        chain_values = values[-chain_count:] if chain_count else ()
        for device, chain in zip(self, chain_values, strict=True):
            assert isinstance(chain, _ChainValue)
            self[device] = chain
        return user_values

    def merge_for(self, devices: Iterable[DeviceRef]) -> _ChainValue:
        """Merge the host chain and per-device chains for ``devices`` into one.

        Multi-device collective ops need a single combined input chain that
        depends on the host orchestration chain and on each participating
        device's compute chain. This emits the ``mo.chain.create`` that
        combines them, updates the host chain to the merged result, and
        returns it.

        Duplicates are removed (e.g., if ``DeviceRef.CPU()`` appears in
        ``devices``, the host chain isn't listed twice).
        """
        keys = list(dict.fromkeys((DeviceRef.CPU(), *devices)))
        chains = [self[device] for device in keys]
        unique_chains = list(dict.fromkeys(chains))

        if len(unique_chains) == 1:
            [chain] = unique_chains
            assert chain == self[DeviceRef.CPU()]
            return chain

        merged = self._graph._add_op_generated(
            _mo.ChainCreateOp,
            result=_mo.ChainType(),
            inputs=unique_chains,
            attach_profile_scopes=False,
        )[0]
        assert isinstance(merged, _ChainValue)
        self[DeviceRef.CPU()] = merged
        return merged

    def copy(self) -> _DeviceChainMap:
        result = _DeviceChainMap(self._graph)
        for device in self:
            result[device] = self._value(device)
        return result

    def __repr__(self) -> str:
        items = ", ".join(f"{device}: {self._value(device)}" for device in self)
        return f"_DeviceChainMap({{{items}}})"


class KernelLibrary:
    """Manages custom kernel libraries and operations for a graph.

    A kernel library provides access to custom operations and kernels that can
    be loaded from various sources including Mojo pre-compiled packages (``.mojoc``)
    and Mojo source directories. The library handles verification and registration
    of custom operations within the MLIR context.
    """

    _analysis: _graph.Analysis
    _analysis_cache: dict[frozenset[str], _graph.Analysis] = {}

    def __init__(self, paths: Iterable[Path] = ()) -> None:
        context = default_mlir_context()
        _graph._init_and_register_max_context(context)
        paths_list = list(paths)
        self._analysis = _graph.Analysis(context, paths_list)

    def library_paths(self) -> list[Path]:
        """Returns the list of kernel library paths.

        Returns:
            A list of :class:`pathlib.Path` objects representing the currently loaded
            kernel library paths.
        """
        return self._analysis.library_paths

    def add_path(self, path: Path) -> None:
        """Adds a kernel library path to the analysis.

        Args:
            path: The :class:`pathlib.Path` to the kernel library to be added to the
                current analysis.
        """
        self._analysis.add_path(path)

    def load_paths(self, custom_extensions: Iterable[Path]) -> None:
        """Loads custom operations from provided library paths.

        Performs additional "smart" library loading logic for custom operation
        libraries in additional formats. The loading logic supports the
        following formats:

        - Compiled Mojo binary packages with ``.mojoc`` extension
        - Mojo source directory with custom operations

        The loaded libraries are added to the current kernel library.

        Args:
            custom_extensions: The file paths to the custom operation libraries.
        """
        for ext_path in custom_extensions:
            if is_mojo_binary_package_path(ext_path):
                self.add_path(ext_path)
            elif is_mojo_source_package_path(ext_path):
                # Builds the source directory into a precompiled Mojo file.
                self.add_path(_build_mojo_source_package(ext_path))
            else:
                raise ValueError(
                    "Path provided as custom extension to Graph must be a "
                    + f"Mojo source or binary package: {ext_path}"
                )

    def __getitem__(self, kernel: str):
        if kernel not in self._analysis.symbol_names:
            raise KeyError(kernel)
        return self._analysis.kernel(kernel)

    def __contains__(self, kernel: str) -> bool:
        return kernel in self._analysis.symbol_names

    def __iter__(self):
        yield from sorted(self._analysis.symbol_names)

    def verify_custom_op(self, custom_op: mlir.Operation) -> None:
        """Verifies that a custom operation is valid within the current context.

        Args:
            custom_op: The :obj:`mlir.Operation` to be verified against the
                current kernel library analysis.
        """
        paths = self._analysis.library_paths
        if paths:
            cache_key = frozenset(str(p) for p in paths)
            cached = KernelLibrary._analysis_cache.get(cache_key)
            if cached is not None:
                self._analysis = cached
                self._analysis.verify_custom_op(custom_op)
                return
            self._analysis.verify_custom_op(custom_op)
            KernelLibrary._analysis_cache[cache_key] = self._analysis
        else:
            self._analysis.verify_custom_op(custom_op)

    def has_shape_function(self, kernel: str) -> bool:
        """Returns whether *kernel* registers a shape function.

        A kernel that registers no shape function has no way to compute its
        output shape at run time, so the graph compiler rejects a
        data-dependent output dimension declared for it.

        Args:
            kernel: The registered name of the kernel to check.

        Returns:
            ``True`` if a shape function is registered for *kernel*.

        Raises:
            KeyError: If no kernel named *kernel* is in the library.
        """
        if kernel not in self:
            raise KeyError(f"no kernel named {kernel!r} in the kernel library")
        return self._analysis.has_shape_function(kernel)


_default_custom_extensions: tuple[Path, ...] = ()


def default_custom_extensions() -> tuple[Path, ...]:
    """Returns the custom-extension paths implicitly loaded by every new graph.

    A backend whose ops need a kernel-overlay library  registers it here so ops
    resolve to the overlays even on graph paths that don't thread
    ``custom_extensions`` explicitly — notably the experimental
    eager-realization ``Graph("main")``. Empty by default, so there is no
    effect unless a backend registers a library.
    """
    return _default_custom_extensions


@contextlib.contextmanager
def default_custom_extensions_scope(*paths: Path) -> Generator[None]:
    """Adds *paths* to :func:`default_custom_extensions` for the block's duration.

    Paths already registered are not duplicated. The previous defaults are
    restored on exit.
    """
    global _default_custom_extensions
    previous = _default_custom_extensions
    merged = list(previous)
    merged.extend(path for path in paths if path not in merged)
    _default_custom_extensions = tuple(merged)
    try:
        yield
    finally:
        _default_custom_extensions = previous


class DevicePlacementPolicy(Enum):
    """Controls behavior when an op implicitly transfers a tensor to CPU.

    Ops that only have CPU kernels must transfer non-CPU tensors before
    executing. This policy controls how that situation is reported:

    - ``Ignore``: transfer silently, no message.
    - ``Warn`` (default): emit a ``UserWarning`` naming the op and the
      tracking ticket for GPU support.
    - ``Error``: raise ``ValueError``, making the implicit transfer a hard
      build-time failure.

    Pass via ``Graph(..., strict_device_placement=DevicePlacementPolicy.Error)``.
    """

    Ignore = "ignore"
    Warn = "warn"
    Error = "error"


# From https://stackoverflow.com/a/76301341
class _classproperty:
    def __init__(self, func) -> None:  # noqa: ANN001
        self.fget = func

    def __get__(self, instance, owner):  # noqa: ANN001
        return self.fget(owner)


@dataclass(frozen=True)
class _GraphWeight:
    weight: Weight
    value: TensorValue


def _location(
    ignore_frames: int = 1, attach_profile_scopes: bool = True
) -> mlir.Location:
    """Creates an MLIR Location with the current Python call stack."""
    if not mlir.Context.current:
        raise RuntimeError("Can't create location: No MLIR context active")

    # Read the live config value rather than caching at import time: the
    # flag can be flipped after import via Graph.debug.source_tracebacks,
    # InferenceSession.debug.sensible_mode, or MODULAR_DEBUG, and each
    # graph op must honor the setting at the moment it is created.
    if not _InferenceSession.debug.source_tracebacks:
        location: mlir.Location = mlir.Location.unknown()
    else:
        # Extract the stack into summaries
        # - Avoids reference cycles
        # - Doesn't keep references to closures

        # Always remove at least _location
        tb = traceback.extract_stack()[: -(ignore_frames + 1)]
        location = (
            _graph.frame_loc(mlir.Context.current, tb)
            if tb
            else mlir.Location.unknown()
        )

    if attach_profile_scopes:
        # Wrap outermost scope first, so the innermost scope ends up as the
        # outermost ProfileScopeLocationAttr and the location assembly reads
        # in nesting order. This runs regardless of source-traceback capture:
        # profile_scope labels are a distinct, always-on mechanism riding the
        # same Location, not a feature of the debug traceback.
        for name, color in _CURRENT_PROFILE_SCOPES.get():
            location = _graph.profile_scope_location(
                mlir.Context.current, name, location, color
            )

    return location


def _to_mlir(o: Any) -> Any:
    # Convert args from instances of Python graph-api Value() to mlir.Value
    if hasattr(o, "to_mlir"):
        return o.to_mlir()
    elif isinstance(o, list | tuple):
        return type(o)(_to_mlir(ov) for ov in o)
    elif isinstance(o, dict):
        return {k: _to_mlir(v) for k, v in o.items()}
    return o


# MLIR assembly text must be read and written through the ``max.mlir`` bindings
# rather than ``max._core``. ``_core`` static-links its own LLVM, so its textual
# float parser and printer compare ``llvm::APFloat`` semantics against their own
# copy of the ``APFloatBase::sem*`` statics and never match the ones the
# attribute was built against. That silently turns f64 constants into denormal
# garbage and aborts on any f32 that MLIR spells as a hex literal. ``max.mlir``
# resolves MLIR from ``libmax``, the copy that owns the context, so a round-trip
# through it agrees with itself. See GEX-4052.


def _asm_via_cmlir(op: Operation) -> str:
    """Serializes an operation to MLIR assembly text.

    The cmlir wrapper is a non-owning view onto ``op``, so whatever owns ``op``
    keeps owning it.
    """
    return str(mlir.Operation._CAPICreate(op._CAPIPtr))


def _parse_module_via_cmlir(
    asm: str, context: mlir.Context
) -> builtin.ModuleOp:
    """Parses MLIR assembly text into a typed ``builtin.ModuleOp``.

    Parses in ``max.mlir``, then moves the module into ``max._core`` as
    bytecode, which is a lossless transport between the two. The result is
    detached, matching what ``_core.parse_module`` returns.
    """
    parsed = mlir.Module.parse(asm, context)
    bytecode = io.BytesIO()
    parsed.operation.write_bytecode(bytecode)
    return Operation.from_bytecode(bytecode.getvalue(), context)


def _set_output_param_decls(op: Operation, params: dict[str, None]) -> None:
    # Interfaces don't yet support isinstance checks, so this is a cheap proxy.
    # - nanobind doesn't allow custom metaclasses, but __instancecheck__
    #   must be defined on a metaclass
    # - Interfaces are protocols even though we know when they are explicitly
    #   implemented so that attrs/types/ops may implement them without declaring
    #   it in the stub files
    # - it's not trivial to define our own `isa`-like check. Such a check needs
    #   to name the template parameter for `mlir::isa<T>` explicitly in C++, but
    #   if `isa` is defined as a staticmethod it interferes with protocol type
    #   checking.
    if not hasattr(op, "output_param_decls"):
        return
    op: Operation & _mo.ParamDeclarationInterface  # type: ignore
    # Add symbolic dims of tensor results to the list of graph params and
    # declared output params of the op
    # Use a dict as an ordered set for new param decls. Maps keys to None.
    result_parameters = dict.fromkeys(
        itertools.chain.from_iterable(
            value.type.parameters
            for result in op.results
            if isinstance(
                value := Value.from_mlir(result), TensorValue | BufferValue
            )
        )
    )
    names = [parameter.name for parameter in result_parameters]
    # Track newly declared parameters. Not `names - params.keys()`: set order
    # is per-process hash order and lands in the MEF cache key (MXF-584).
    if new_params := dict.fromkeys(n for n in names if n not in params):
        params.update(new_params)
        si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True))
        # We can't overload the setter yet, so the interface annotation is wrong
        # TODO(MAXPLAT-306): See https://github.com/wjakob/nanobind/discussions/1063
        op.output_param_decls = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(name, si64) for name in new_params]
        )


class Module:
    """A container for one or more :class:`Graph` instances compiled together.

    A ``Module`` holds an MLIR module that may contain multiple ``mo.graph``
    ops — for example a vision encoder and a language model that should be
    handed to :meth:`max.engine.InferenceSession.load_all` as a single
    compilation unit. Constructing several :class:`Graph` instances with the
    same ``module=`` argument adds each graph as a top-level op in this
    module.

    Example:

    .. code-block:: python

        import numpy as np
        from max.driver import Accelerator, CPU, accelerator_count
        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, Module, ops

        device = Accelerator() if accelerator_count() > 0 else CPU()
        device_ref = DeviceRef.from_device(device)

        module = Module()
        with Graph("encoder", input_types=[], module=module) as encoder:
            encoder.output(
                ops.constant(
                    np.array([1.0, 2.0, 3.0], dtype=np.float32),
                    DType.float32,
                    device=device_ref,
                )
            )
        with Graph("decoder", input_types=[], module=module) as decoder:
            decoder.output(
                ops.constant(
                    np.array([4.0, 5.0, 6.0], dtype=np.float32),
                    DType.float32,
                    device=device_ref,
                )
            )

        models = InferenceSession(devices=[device]).load_all(module)
        (enc_out,) = models[encoder.name].execute()
        (dec_out,) = models[decoder.name].execute()

    .. invisible-code-block: python

        np.testing.assert_allclose(enc_out.to_numpy(), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(dec_out.to_numpy(), [4.0, 5.0, 6.0])

    The wrapped MLIR module is exposed as :attr:`mlir_module` for code that
    must reach the underlying representation (graph compiler internals,
    serializers, etc.); routine users should not need it.
    """

    mlir_module: builtin.ModuleOp
    """The wrapped MLIR module, exposed through the typed nanobind dialect
    bindings."""

    def __init__(self, mlir_module: builtin.ModuleOp | None = None) -> None:
        """Create a new, empty module, or wrap an existing :class:`builtin.ModuleOp`.

        Args:
            mlir_module: An existing typed :class:`builtin.ModuleOp` to
                wrap. When ``None`` (the default), a new empty module is
                created. The wrap-existing form is intended for code that
                already holds a typed ``ModuleOp`` (for example
                :class:`Graph`'s subgraph plumbing or the
                :attr:`Graph.module` property) and needs to expose it
                through the public :class:`Module` surface; routine users
                should leave this argument unset.
        """
        if mlir_module is None:
            mlir_module = builtin.ModuleOp(location=_location())
        self.mlir_module = mlir_module

    def top_level_graph_names(self) -> list[str]:
        """Return the name of every top-level (non-subgraph) graph in this module.

        Walks the wrapped module's body and skips any op whose
        ``is_subgraph`` is set (subgraphs are inlined callees, not loaded
        as standalone models). Non-graph ops are skipped rather than
        raising. The resulting order matches MEF model order, which is the
        order :func:`max.engine.InferenceSession.load_all` returns models
        in.

        Returns:
            The names of the top-level graphs, in MEF order. The list is
            empty for a freshly constructed :class:`Module`.
        """
        names: list[str] = []
        for op in self.mlir_module.body:
            if isinstance(op, _mo.GraphOp) and not op.is_subgraph:
                names.append(op.sym_name)
        return names

    def _to_mlir_str(self, *, source_locations: bool = False) -> str:
        """Serializes this module to MLIR assembly text.

        Internal helper used by graph-dump tooling.

        Args:
            source_locations: When ``True``, annotates each op with the Python
                call stack it was built from. This requires source-traceback
                capture to have been enabled during graph construction (see
                :attr:`max.graph.Graph.debug`); without it the ops carry no
                Python frames and the annotations are empty. The wrapped module
                is left unchanged either way.

        Returns:
            The module's MLIR assembly text.
        """
        if not source_locations:
            return _asm_via_cmlir(self.mlir_module)

        # The materialized locations come back as bytecode; reparsing through
        # max.mlir is what puts the printing in the right LLVM image.
        with_locations = mlir.Module.parse(
            _graph.source_locations_bytecode(self.mlir_module),
            self.mlir_module.context,
        )
        return with_locations.operation.get_asm(enable_debug_info=True)


class GraphDebugConfig:
    """Narrow view of :class:`max.engine.DebugConfig` exposed through :attr:`Graph.debug`.

    The attribute :attr:`source_tracebacks` lives on ``Graph.debug`` because it is
    consumed during graph construction, before an ``InferenceSession`` exists.
    All other debug options are available on ``InferenceSession.debug`` and
    share the same global state.
    """

    @property
    def source_tracebacks(self) -> bool:
        """See :attr:`max.engine.DebugConfig.source_tracebacks`."""
        return _InferenceSession.debug.source_tracebacks

    @source_tracebacks.setter
    def source_tracebacks(self, value: bool) -> None:
        _InferenceSession.debug.source_tracebacks = value


class Graph:
    """Represents a single MAX graph.

    A :class:`Graph` defines a model's computation. You build a graph by
    composing operations that describe how input tensors are transformed into
    outputs. Unlike imperative code that executes operations, a :class:`Graph`
    captures the data flow between operations, which allows MAX to optimize and
    parallelize execution at compile time. Operations run on the compiled object.

    The following code examples show two different strategies for constructing
    graphs.

    **Use the context manager:** Use :class:`Graph` as a context manager to
    define the active graph. Inside the ``with`` block, retrieve inputs from
    :attr:`inputs`, call ops to build nodes, and set the graph output with
    :meth:`output()`. Ops called inside the block find the active graph
    automatically. Ops called outside the block fail because there is no active
    graph.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, Weight

        W = Weight("W", DType.float32, [3, 2], DeviceRef.CPU())
        b = Weight("b", DType.float32, [2], DeviceRef.CPU())

        with Graph(
            "linear_relu",
            input_types=[TensorType(DType.float32, ["batch", 3], device=DeviceRef.CPU())],
        ) as graph:
            x = graph.inputs[0].tensor
            y = x @ W + b
            graph.output(y)

    **Use the graph constructor:** Pass a callable as the ``forward`` argument.
    The graph automatically passes the input :class:`TensorValue` to the
    callable and records the return value as the graph output. Under the hood,
    this still opens and closes a graph context.

    .. code-block:: python

        from max.dtype import DType
        from max.graph import DeviceRef, Graph, TensorType, TensorValue, Weight, ops

        class Linear:
            def __init__(self, in_dim: int, out_dim: int):
                self.weight = Weight("W", DType.float32, [in_dim, out_dim], DeviceRef.CPU())
                self.bias = Weight("b", DType.float32, [out_dim], DeviceRef.CPU())

            def __call__(self, x: TensorValue) -> TensorValue:
                return ops.matmul(x, self.weight) + self.bias

        linear_layer = Linear(2, 2)

        graph = Graph(
            "linear",
            linear_layer,
            input_types=[TensorType(DType.float32, (2,), DeviceRef.CPU())],
        )

    These examples only use the :obj:`max.graph` package, but most models also
    use :class:`~max.nn.Module` and other building blocks from :obj:`max.nn`.
    To learn more, see `Build a model graph with Module
    </develop/modules>`_.

    Args:
        name: A name for the graph.
        forward: The sequence of graph ops for the forward pass (inference).
        input_types: A sequence of :class:`~max.graph.type.Type` instances that
            describe each graph input.
            These are typically :class:`TensorType` instances. You can also
            include :class:`BufferType` instances for mutable in-place inputs.
        path: The path to a saved graph (internal use only).
        custom_extensions: The extensions to load for the model. Supports paths
            to ``.mojoc`` or ``.mojo`` sources with custom ops.
        kernel_library: Optional pre-built kernel library to use. Defaults to
            ``None`` (a new library is created from ``custom_extensions`` if
            needed).
        module: Optional existing MLIR module (internal use only). Defaults to
            ``None``.
    """

    debug = GraphDebugConfig()

    # Use a dict rather than a set to keep params ordered.
    # This is to make IR generation deterministic for model IR cache hits.
    # Note that insertion order in built-in dict has been guaranteed since
    # Python 3.7.
    _params: dict[str, None]
    _mlir_op: mlir.Operation | mlir.OpView
    _graph_body: mlir.Block
    _module: builtin.ModuleOp
    _context_state: list[contextlib.AbstractContextManager[Graph]]
    _weights: dict[str, _GraphWeight]
    _current_block: mlir.Block
    _should_verify_ops: bool
    _has_chain_input: bool
    # Per-device chains that ensure the correct sequence of device execution.
    # ``device_chains[DeviceRef.CPU()]`` is the host orchestration chain (the
    # chain advanced by host-side ops like ``debug.print``, ``mo.call``, and
    # the host side of collectives); all other entries are per-device compute
    # chains. New device entries are seeded from the host chain.
    device_chains: _DeviceChainMap

    _kernel_library: KernelLibrary

    _subgraphs: dict[str, Graph] = {}

    # Ensure the default MLIR context for bg-thread Graph construction.
    @in_default_mlir_context
    def __init__(
        self,
        name: str,
        forward: Callable[..., Value[Any] | Iterable[Value[Any]] | None]
        | None = None,
        input_types: Iterable[Type[Any]] = (),
        path: Path | None = None,
        *args,
        custom_extensions: Iterable[Path] = [],
        kernel_library: KernelLibrary | None = None,
        module: Module | None = None,
        strict_device_placement: DevicePlacementPolicy = DevicePlacementPolicy.Warn,
        is_device_graph: bool = False,
        **kwargs,
    ) -> None:
        self.name = name
        self.strict_device_placement = strict_device_placement
        if path is not None:
            self._load_mlir(path)
            return

        self._params = dict.fromkeys(
            dim.name
            for t in input_types
            if isinstance(t, TensorType | BufferType)
            for dim in t.shape
            if isinstance(dim, SymbolicDim)
        )
        self._context_state = []
        self._should_verify_ops = True

        with _location() as loc:
            # The top-level module op is stored as a typed nanobind
            # :class:`builtin.ModuleOp`. Either reuse the one supplied via
            # ``module=`` or create a fresh one.
            self._module = (
                module.mlir_module if module else builtin.ModuleOp(location=loc)
            )
            builder = OpBuilder(self._module.body.end)

            op = _mo.GraphOp(
                builder,
                loc,
                name=name,
                input_types=[t.to_mlir() for t in input_types],
                result_types=[],
            )

            si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True))
            # TODO(MAXPLAT-306): Type annotations are wrong here
            op.input_parameters = kgen.ParamDeclArrayAttr(
                [kgen.ParamDeclAttr(p, si64) for p in self._params]
            )

            self._mlir_op = mlir.Operation._CAPICreate(op._CAPIPtr)
            self._current_block = self._mlir_op.regions[0].blocks[0]
            self._graph_body = self._current_block
            self._populate_device_info_mapping()

        self._weights = {}
        self._has_chain_input = False
        self.device_chains = _DeviceChainMap(self)

        # Create an always-ready chain that is never advanced by the graph.
        # Use it for operations that are safe to schedule without per-device
        # ordering constraints (e.g., host→device transfers for staging).
        self._always_ready_chain = _ChainValue(
            self._add_op_generated(
                _mo.ChainCreateOp,
                result=_mo.ChainType(),
                inputs=[],
                attach_profile_scopes=False,
            )[0]
        )
        self._update_chain(self._always_ready_chain)

        if self._graph_body.arguments:
            mlir_maybe_chain_value = _Value._from_cmlir(
                self._graph_body.arguments[-1]
            )
            if _is_chain_value(mlir_maybe_chain_value):
                self._has_chain_input = True
                self._update_chain(
                    _ChainValue.from_mlir(mlir_maybe_chain_value)
                )

        # Initialize the kernel library and load custom extensions paths.
        # Process-global defaults are appended after any explicit extensions so
        # graphs built without `custom_extensions` still reach a backend's
        # kernel overlays (see `default_custom_extensions`).
        self._kernel_library = kernel_library or KernelLibrary()
        extensions = list(custom_extensions)
        extensions.extend(
            path
            for path in default_custom_extensions()
            if path not in extensions
        )
        self._import_kernels(extensions)

        self._subgraphs = {}

        # If we're building a device graph, annotate the graph appropriately.
        if is_device_graph:
            op.is_device_graph = builtin.UnitAttr()

        if forward is not None:
            # If the forward method was passed stage the graph directly in the
            # constructor.
            with self:
                result = forward(*self.inputs, *args, **kwargs)
                # Account for forward methods that return None, a single
                # output, or multiple outputs.
                outputs = (
                    ()
                    if result is None
                    else (result,)
                    if not isinstance(result, Iterable)
                    else result
                )
                self.output(*outputs)

    @functools.cached_property
    def inputs(self) -> Sequence[Value[Any]]:
        """The input values of the graph.

        Returns:
            A sequence of :class:`~max.graph.Value` objects corresponding to
            the ``input_types`` passed at construction, excluding internal
            chain values.
        """
        body_args = self._graph_body.arguments
        chain_count = len(self.device_chains) if self._has_chain_input else 0
        if body_args and chain_count:
            body_args = body_args[:-chain_count]

        return tuple(
            Value.from_mlir(_Value._from_cmlir(arg)) for arg in body_args
        )

    def add_subgraph(
        self,
        name: str,
        forward: Callable[..., Value[Any] | Iterable[Value[Any]] | None]
        | None = None,
        input_types: Iterable[Type[Any]] = (),
        path: Path | None = None,
        custom_extensions: Iterable[Path] = [],
        devices: Iterable[DeviceRef] = [],
        is_device_graph: bool = False,
    ) -> Graph:
        """Creates a reusable subgraph for the current graph.

        A subgraph is the graph equivalent of a function: you define a block of
        ops once and call it from the parent graph as many times as you need.
        Use a subgraph when a block of computation repeats, for example, a
        transformer layer that appears 62 times in a model. Wrapping it in a
        subgraph lets the compiler process the definition once instead of once
        per repetition, which can cut compile time by 50x or more.

        Trade-offs to keep in mind:

        - **Memory:** Allocations inside a subgraph can't be shared with
          allocations outside it, so peak memory may be slightly higher.
        - **Kernel fusion:** The compiler can't fuse ops across the subgraph
          boundary, which may reduce throughput marginally.

        For models with a :class:`~max.nn.Module`, prefer
        :meth:`~max.nn.Module.build_subgraph`, which handles weight prefixes
        automatically.

        Examples:
            Define a subgraph that adds 1 to every element, then call it on a
            graph input:

            .. code-block:: python

                from max.dtype import DType
                from max.graph import Graph, ops
                from max.graph.type import TensorType, DeviceRef

                input_type = TensorType(DType.float32, [10], DeviceRef.CPU())

                with Graph("main", input_types=[input_type]) as graph:
                    with graph.add_subgraph(
                        "add_one", input_types=[input_type]
                    ) as sub:
                        x = sub.inputs[0].tensor
                        one = ops.constant(1, DType.float32, device=DeviceRef.CPU())
                        sub.output(ops.elementwise.add(x, one))

                    result = ops.call(sub, graph.inputs[0])
                    graph.output(*result)

        Args:
            name: The name identifier for the subgraph. Must be unique within
                the parent graph. Use the same name when calling the subgraph
                with :func:`~max.graph.ops.call`.
            forward: An optional callable that defines the subgraph's forward
                pass. When provided, the subgraph is built immediately.
            input_types: The tensor types for the subgraph's inputs. A chain
                type is added automatically for operation sequencing.
            path: An optional path to a saved subgraph definition to load
                from disk.
            custom_extensions: Paths to custom op libraries (``.mojoc``
                files or Mojo source directories) to load for the subgraph.
            devices: Devices this subgraph targets.
            is_device_graph: Should the subgraph be synthesized to a device graph
                for device graph based execution.

        Returns:
            A :class:`Graph` instance registered as a subgraph of this graph.
        """
        subgraph = Graph(
            name=name,
            forward=forward,
            input_types=[*input_types, _ChainType()],
            path=path,
            # *args,
            custom_extensions=custom_extensions,
            module=self.module,
            is_device_graph=is_device_graph,
            # **kwargs,
        )

        # Mark the new graph op as a subgraph and set its input parameters.
        op = Operation._from_cmlir(subgraph._mlir_op)
        assert isinstance(op, _mo.GraphOp)
        op.is_subgraph = builtin.UnitAttr()

        # If we're building a device graph, annotate the graph appropriately.
        if is_device_graph:
            op.is_device_graph = builtin.UnitAttr()

        # Union callee's existing params  with the caller's params.
        # This may over-declare but is deterministic and comprehensive.
        union_names = list(dict.fromkeys([*subgraph._params, *self._params]))
        si64 = kgen.SIMDType(1, kgen._KGENDType.get_int(64, True))
        op.input_parameters = kgen.ParamDeclArrayAttr(
            [kgen.ParamDeclAttr(name, si64) for name in union_names]
        )
        subgraph._params = dict.fromkeys(union_names)
        # ``input_types`` already ends with an explicit ``_ChainType()`` that
        # seeds ``device_chains[DeviceRef.CPU()]`` (the host chain). Add chain
        # block args only for non-CPU devices to avoid duplicating the host
        # chain.
        for device in devices:
            if device == DeviceRef.CPU():
                continue
            subgraph.device_chains[device] = subgraph._add_chain_block_arg()
        self._subgraphs[name] = subgraph
        return subgraph

    def _update_chain(self, new_chain: _ChainValue) -> None:
        self.device_chains[DeviceRef.CPU()] = new_chain

    def _add_chain_block_arg(self) -> _ChainValue:
        """Add a new chain as a graph block argument."""
        with _location() as loc:
            block = Block._from_cmlir(self._graph_body)
            block.add_argument(_ChainType().to_mlir(), loc)
        mlir_value = _Value._from_cmlir(self._graph_body.arguments[-1])
        assert _is_chain_value(mlir_value)

        return _ChainValue.from_mlir(mlir_value)

    @property
    def always_ready_chain(self) -> _ChainValue:
        """A graph-global, immutable chain that is always ready.

        Created once per graph and never advanced/merged by the graph itself.
        Use it for operations that are safe to schedule without threading
        per-device ordering (for example, host→device transfers for staging).
        """
        return self._always_ready_chain

    @contextlib.contextmanager
    def profile_scope(
        self, name: str, color: str | None = None
    ) -> Generator[None]:
        """Labels every op created within this block for profiling.

        Ops created while the scope is active carry ``name`` in their MLIR
        location, which downstream NVTX tracing surfaces as a ``[name]``
        suffix on the op's own per-kernel trace name. Scopes nest: entering a
        second :meth:`profile_scope` while the first is still active labels
        ops with both names, outermost scope first (for example
        ``kernel_name [draft_forward/target_forward]``).

        ``color`` is only used by the optional in-region range bracketing
        mechanism (``max-debug.profile-scope-tracing``); it never appears on
        the per-kernel trace name, regardless of what was given.

        .. code-block:: python

            from max.graph import Graph
            with Graph("main") as graph:
                with graph.profile_scope("draft_forward", color="orange"):
                    ...  # ops here trace as "kernel_name [draft_forward]"

        Args:
            name: The scope label to attach to every op created inside this
                block.
            color: Optional NVTX color name for the in-region range bracketing
                mechanism. Has no effect on the per-kernel trace name. Invalid
                values are silently ignored by the backend.

        Note:
            Scopes are tracked in a single module-level ContextVar. Do not
            construct a second nested :class:`Graph` while a sibling graph's
            ``profile_scope`` is still active; labels would leak between the
            two graphs.
        """
        current = _CURRENT_PROFILE_SCOPES.get()
        token = _CURRENT_PROFILE_SCOPES.set((*current, (name, color)))
        try:
            yield
        finally:
            _CURRENT_PROFILE_SCOPES.reset(token)

    def __enter__(self) -> Graph:
        self._context_state.append(state := self._enter())
        return state.__enter__()

    def __exit__(self, *exc) -> None:
        self._context_state.pop().__exit__(*exc)

    @contextlib.contextmanager
    def _enter(self) -> Generator[Graph]:
        # Body of ``with graph:`` creates MLIR ops; ensure context on bg threads.
        with ensure_default_mlir_context():
            token = CURRENT_GRAPH.set(self)
            try:
                yield self
            finally:
                CURRENT_GRAPH.reset(token)

    @contextlib.contextmanager
    def _block(self, block: mlir.Block):  # noqa: ANN202
        """Push ``block`` as the current insertion target, isolating per-block state.

        Snapshots ``device_chains`` and the ``_weights`` dedup cache on entry
        and restores them on exit so that block-local side effects don't leak
        into the outer graph's chain timeline. ``_weights`` is snapshotted
        because :meth:`add_weight` keys its cache by Python identity:
        same-name weights with different ``Weight`` instances across blocks
        (e.g., separate ``Weight("w", ...)`` calls in the ``true_fn`` and
        ``false_fn`` of an ``ops.cond``) need each block to start with an
        empty cache so each branch creates its own ``mo.constant.external``
        op (the graph compiler resolves both to the same registry entry).
        """
        previous_block = self._current_block
        previous_weights = self._weights.copy()
        previous_chains = self.device_chains.copy()
        self._current_block = block
        try:
            yield self._current_block
        finally:
            self._current_block = previous_block
            self._weights = previous_weights
            self.device_chains = previous_chains

    @contextlib.contextmanager
    def _pause_verification(self):  # noqa: ANN202
        """Temporarily disable verification."""
        old_value = self._should_verify_ops
        try:
            self._should_verify_ops = False
            yield
        finally:
            self._should_verify_ops = old_value

    def _verify_op(self, op: mlir.Operation | mlir.OpView | Operation) -> None:
        if self._should_verify_ops:
            with self._capturing_mlir_diagnostics():
                op.verify()

    @contextlib.contextmanager
    def _capturing_mlir_diagnostics(self):  # noqa: ANN202
        diagnostics = []

        def handler(d: mlir.Diagnostic) -> bool:
            diagnostics.append(str(d))
            return True

        # Temporarily hookup a handler to record diagnostics from mlir.
        # These are used to generate a better error message on failure.
        handle = default_mlir_context().attach_diagnostic_handler(handler)
        try:
            yield None
        except (mlir.MLIRError, ValueError) as e:
            # MLIRError is raised from the MLIR Python bindings on MLIR
            # errors, however so is ValueError when operation create fails.
            # So catch both exception types and report diagnostics here.
            diags = "\n  ".join(diagnostics)
            raise ValueError(f"Diagnostics:\n    {diags}\n{e}") from None
        finally:
            handle.detach()  # type: ignore

    @_classproperty
    def current(cls) -> Graph:
        """Gets the currently active graph from the execution context.

        Retrieves the :class:`Graph` instance that is currently active within
        the execution context. The current graph is automatically set when
        entering a graph's context using a ``with`` statement or when the
        graph is being built. This provides access to the active graph from
        within operation definitions and other graph construction code.
        """
        try:
            current = CURRENT_GRAPH.get()
        except LookupError as exc:
            raise LookupError("No graph found") from exc
        assert current
        return current

    @property
    def module(self) -> Module:
        """The :class:`Module` that owns this graph.

        Multiple :class:`Graph` instances built with the same ``module=``
        argument share the same underlying ``Module``; that shared object
        is what you pass to :meth:`max.engine.InferenceSession.load_all`
        when compiling several graphs together.
        """
        return Module(mlir_module=self._module)

    @property
    def _body(self) -> mlir.Block:
        return self._current_block

    def _add_op_generated(
        self,
        op_type: type[Operation],
        *args,
        attach_profile_scopes: bool = True,
        **kwargs,
    ) -> list[Value[Any]]:
        """Wrapper for clients that only require the op results."""
        location = _location(attach_profile_scopes=attach_profile_scopes)
        try:
            with self._capturing_mlir_diagnostics():
                builder = OpBuilder(Block._from_cmlir(self._current_block).end)
                op = op_type(
                    builder, location, *_to_mlir(args), **_to_mlir(kwargs)
                )  # type: ignore
                self._verify_op(op)
        except (mlir.MLIRError, ValueError, TypeError) as e:
            # `TypeError` covers verifier diagnostics, which surface from
            # `RaisingScopedEmitFn` via `PyExc_TypeError`; re-raise everything
            # as `ValueError` so callers don't have to know about MLIR-internal
            # exception types.
            positional = {f"args[{i}]": v for i, v in enumerate(args)}
            raise ValueError(
                f"Failed to create op '{op_type.__name__}':\nInputs:\n"
                + "".join(
                    f"    {k} = {v!r}\n"
                    for k, v in {**positional, **kwargs}.items()
                )
                + f"\n{e}"
            ) from None
        _set_output_param_decls(op, self._params)
        return [Value.from_mlir(result) for result in op.results]

    def _add_op(self, op, *args, **kwargs) -> list[Value[Any]]:  # noqa: ANN001
        """Wrapper for clients that only require the op results."""
        results, _ = self._add_op_get_op_with_results(op, *args, **kwargs)
        return results

    def _add_op_get_op_with_results(
        self,
        op,  # noqa: ANN001
        *args,
        _ip: mlir.InsertionPoint | None = None,
        attach_profile_scopes: bool = True,
        **kwargs,
    ) -> tuple[list[Value[Any]], mlir.OpView]:
        # Convert args from instances of Python graph-api Value() to mlir.Value
        def unwrap(arg: Any) -> Any:
            if isinstance(arg, Value):
                return mlir.Value._CAPICreate(arg._mlir_value._CAPIPtr)  # type: ignore[attr-defined]
            elif isinstance(arg, Type):
                return mlir.Type._CAPICreate(arg.to_mlir()._CAPIPtr)  # type: ignore
            elif isinstance(arg, list | tuple):
                return [unwrap(elem) for elem in arg]
            elif isinstance(arg, _Attribute):
                return mlir.Attribute._CAPICreate(arg._CAPIPtr)  # type: ignore
            elif isinstance(arg, _Type):
                return mlir.Type._CAPICreate(arg._CAPIPtr)  # type: ignore
            elif isinstance(arg, _Value):
                return mlir.Value._CAPICreate(arg._CAPIPtr)  # type: ignore[attr-defined]
            else:
                return arg

        unwrapped_args = tuple(unwrap(arg) for arg in args)
        unwrapped_kwargs = {k: unwrap(arg) for k, arg in kwargs.items()}

        ip = _ip or mlir.InsertionPoint(self._body)

        # Construct and insert an op in the body of the graph
        # Insertion point is where the op is to be created in the IR structure
        # location contains info about the source of the op (e.g. file, line)
        with ip, _location(attach_profile_scopes=attach_profile_scopes):
            try:
                with self._capturing_mlir_diagnostics():
                    results = op(*unwrapped_args, **unwrapped_kwargs)

                    if ip.ref_operation is None:
                        staged_op = _graph.last_operation(self._body)
                    else:
                        staged_op = _graph.prev_operation(ip.ref_operation)

                    assert staged_op is not None, "unable to find staged op"
                    staged_op = staged_op.opview
                    self._verify_op(staged_op)

            except (mlir.MLIRError, ValueError) as e:
                # MLIRError is raised from the MLIR Python bindings on MLIR
                # errors, however so is ValueError when operation creation fails.
                # So catch both exception types here.
                mapped_args: dict[str, Any]
                try:
                    mapped_args = (
                        inspect.signature(op).bind(*args, **kwargs).arguments
                    )
                except TypeError:
                    mapped_args = {"args": list(args), **kwargs}
                raise ValueError(
                    f"Failed to create op '{op.__qualname__}':\nInputs:\n"
                    + "".join(
                        f"    {k} = {v!r}\n" for k, v in mapped_args.items()
                    )
                    + f"\n{e}"
                    # Intentionally suppress extra stack traces from max._mlir.
                ) from None

        _set_output_param_decls(Operation._from_cmlir(staged_op), self._params)
        if isinstance(results, mlir.Operation | mlir.OpView):
            return [], staged_op

        # Convert op results from  mlir.Value to instances of Value graph-api
        if isinstance(results, mlir.Value):
            results = [Value.from_mlir(_Value._from_cmlir(results))]
        else:
            results = [
                Value.from_mlir(_Value._from_cmlir(result))
                for result in results
            ]

        return results, staged_op

    def _populate_device_info_mapping(self) -> None:
        """Attaches mo.device_info_mapping to the module if not already present."""
        module = self._mlir_op.block.owner
        if _DEVICE_INFO_MAPPING_ATTR_NAME in module.attributes:
            return
        devices: list[Device] = [CPU()]
        if accelerator_count() > 0:
            devices.append(Accelerator())
        entries = {}
        for dev in devices:
            try:
                arch = dev.architecture_name
            except Exception:
                arch = "unknown"
            try:
                model = dev.model_name
            except Exception:
                model = "unknown"
            info = _DeviceInfoAttr(
                label=dev.label,
                api=dev.api,
                arch=arch,
                model=model,
                tile_based_fusion=False,
            )
            entries[dev.label] = mlir.Attribute._CAPICreate(info._CAPIPtr)  # type: ignore[attr-defined]
        module.attributes[_DEVICE_INFO_MAPPING_ATTR_NAME] = mlir.DictAttr.get(
            entries
        )

    def output(self, *outputs: Value[Any] | TensorValueLike) -> None:
        """Sets the output values of the graph and finalizes construction.

        Call this once after building all ops. The graph can't be executed
        until ``output()`` has been called. Subsequent calls to
        :attr:`output_types` read back the types of the values passed here.

        Examples:
            Build a graph that doubles its input and set the output:

            .. code-block:: python

                from max.dtype import DType
                from max.graph import DeviceRef, Graph, ops
                from max.graph.type import TensorType

                input_type = TensorType(DType.float32, [4], DeviceRef.CPU())

                with Graph("double", input_types=[input_type]) as graph:
                    x = graph.inputs[0].tensor
                    two = ops.constant(2.0, DType.float32, device=DeviceRef.CPU())
                    graph.output(ops.elementwise.mul(x, two))

        Args:
            outputs: The output values of the graph. Each value may be a
                :class:`Value` or any :class:`~max.graph.TensorValueLike`.
        """
        outputs = tuple(
            o if isinstance(o, Value) else TensorValue(o) for o in outputs
        )
        outputs = cast(tuple[Value[Any], ...], outputs)
        # mo.output doesn't support infer_type
        graph_body_args = self._graph_body.arguments
        output_values: list[Value[Any]] = list(outputs)
        if self._has_chain_input:
            output_values = self.device_chains.pack(output_values)
        mlir_values: list[_Value[Any]] = [v._mlir_value for v in output_values]

        # We have a type mismatch now, these are MLIR types
        # Convert from max._core.Type to mlir.Type using CAPI bridge
        output_types = [
            mlir.Type._CAPICreate(value.type._CAPIPtr)  # type: ignore[attr-defined]
            for value in mlir_values
        ]

        # Need to set some more stuff.
        function_type = mlir.FunctionType.get(
            graph_body_args.types,
            output_types,
        )
        signature = mlir.Type.parse(f"!kgen.generator<{function_type}>")
        self._mlir_op.attributes["signature"] = mlir.TypeAttr.get(signature)
        self._mlir_op.attributes["functionType"] = mlir.TypeAttr.get(
            function_type
        )

        self._add_op_generated(
            _mo.OutputOp,
            mlir_values,
            _kgen.ParameterExprArrayAttr([]),
            attach_profile_scopes=False,
        )

        # Set the result_names metadata on the staged op, which is needed by
        # the engine for execution.
        # Note that result_names here needs to match kMgpModelResultNames.
        input_names = [f'"input{i}"' for i in range(len(self.inputs))]
        self._mlir_op.attributes["argument_names"] = mlir.Attribute.parse(
            f"[{', '.join(input_names)}]"
        )
        output_names = [f'"output{i}"' for i in range(len(output_types))]
        self._mlir_op.attributes["result_names"] = mlir.Attribute.parse(
            f"[{', '.join(output_names)}]"
        )

        self._subgraphs = {}
        # Outputting means the graph is complete. Verify the entire graph.
        try:
            with self._capturing_mlir_diagnostics():
                assert self._mlir_op.verify()
        except Exception as e:
            print(self)
            raise ValueError(
                "Graph failed to verify. Please file an issue. This should be"
                " impossible." + f"\n{e}"
            ) from None

    def _erase_output_if_present(self) -> None:
        terminator = self._body.operations[-1]
        if isinstance(terminator, mo.OutputOp):
            terminator.erase()

    @property
    def output_types(self) -> list[Type[Any]]:
        """The types of the graph output values.

        Returns:
            A list of :class:`~max.graph.type.Type` objects corresponding to
            the values passed to :meth:`output()`, in the same order.

        Raises:
            TypeError: If the graph has not yet been terminated by a call to
                :meth:`output()`.
        """
        terminator = self._body.operations[-1]
        if not isinstance(terminator, mo.OutputOp):
            raise TypeError("Graph not yet terminated by a call to output")

        operand_values = [
            _Value._from_cmlir(terminator.operands[i])
            for i in range(len(terminator.operands))
        ]
        while operand_values and _is_chain_value(operand_values[-1]):
            operand_values.pop()

        return [Value.from_mlir(val).type for val in operand_values]

    def _load_mlir(self, path: Path) -> None:
        self._context_state = []
        with open(path) as f:
            context = default_mlir_context()
            with _location():
                self._module = _parse_module_via_cmlir(f.read(), context)
                # Set the mo.graph op, which is the first operation in the
                # module body block.
                self._mlir_op = mlir.Operation._CAPICreate(
                    self._module.body[0]._CAPIPtr
                )

        # Initialize the Kernel Library
        kernels_paths = []
        if _KERNEL_LIBRARY_PATHS_ATTR_NAME in self._mlir_op.attributes:
            paths_attr = self._mlir_op.attributes[
                _KERNEL_LIBRARY_PATHS_ATTR_NAME
            ]
            if isinstance(paths_attr, mlir.ArrayAttr):
                kernels_paths = [Path(str(x)) for x in paths_attr]
        self._kernel_library = KernelLibrary(kernels_paths)

    def copy(self) -> Graph:
        """Creates a deep copy of this graph.

        The copy shares no MLIR state with the original: staging or lowering
        on either graph leaves the other untouched. Use this to hand a graph
        to another thread (for example, background compilation) while
        continuing to build or execute the original. The kernel library is
        shared, not copied.

        Returns:
            A new :class:`Graph` wrapping a deep copy of this graph's module.
        """
        module = self._module.clone()
        assert isinstance(module, builtin.ModuleOp)
        copied = Graph.__new__(Graph)
        copied.name = self.name
        copied.strict_device_placement = self.strict_device_placement
        copied._context_state = []
        copied._module = module
        # Mirrors _load_mlir: the mo.graph op is the first operation in the
        # module body block.
        copied._mlir_op = mlir.Operation._CAPICreate(module.body[0]._CAPIPtr)
        copied._kernel_library = self._kernel_library
        return copied

    def add_weight(
        # TODO(GEX-2121): Remove `force_initial_weight_on_host`
        self,
        weight: Weight,
        force_initial_weight_on_host: bool = True,
    ) -> TensorValue:
        """Adds a weight to the graph.

        If the weight is in the graph already, return the existing value.

        Args:
            weight: The weight to add to the graph.
            force_initial_weight_on_host: If true, then forces weights
                to initially be allocated on host before being moved to
                the indicated device. This is needed as a stop gap
                until we have a more fleshed out ownership model of
                external constants.

        Returns:
            A :class:`~max.graph.TensorValue` that contains this weight.

        Raises:
            ValueError: If a weight with the same name already exists in the graph.
        """
        if graph_weight := self._weights.get(weight.name):
            if graph_weight.weight is weight:
                if force_initial_weight_on_host:
                    transferred_value = graph_weight.value.to(weight.device)
                    if transferred_value is not graph_weight.value:
                        assert isinstance(
                            transferred_value._mlir_value.owner, Operation
                        )
                        assert isinstance(
                            graph_weight.value._mlir_value.owner, Operation
                        )
                        transferred_value._mlir_value.owner.move_after(
                            graph_weight.value._mlir_value.owner
                        )
                    return transferred_value
                else:
                    return graph_weight.value
            else:
                raise ValueError(
                    f"Weight '{weight.name}' already exists in Graph {self}"
                )

        initial_device = (
            DeviceRef.CPU() if force_initial_weight_on_host else weight.device
        )

        tensor_type = TensorType(
            weight.dtype, weight.shape, device=initial_device
        ).to_mlir()
        weight_tensor = Graph.current._add_op(
            mo.constant_external,
            result=tensor_type,
            name=weight.name,
            align=(
                # Default to dtype alignment unless otherwise specified, for
                # example by checkpoint metadata.
                weight.align if weight.align is not None else weight.dtype.align
            ),
            device=initial_device.to_mlir(),
            is_placeholder=weight._placeholder,
            has_alias=weight._has_alias,
            _ip=mlir.InsertionPoint.at_block_begin(self._graph_body),
            # A weight is compile-time data, not a compute op belonging to
            # whatever profile_scope happens to be active at first use; see
            # ops.constant()/constant_external() for the same rule.
            attach_profile_scopes=False,
        )[0]

        const_external_op = weight_tensor._mlir_value.owner
        self._weights[weight.name] = _GraphWeight(weight, weight_tensor)
        if initial_device != weight.device:
            weight_tensor = weight_tensor.to(weight.device)
            assert isinstance(const_external_op, Operation)
            assert isinstance(weight_tensor._mlir_value.owner, Operation)
            weight_tensor._mlir_value.owner.move_after(const_external_op)
        return weight_tensor

    def __repr__(self) -> str:
        return str(self._mlir_op)

    def _import_kernels(self, paths: Iterable[Path]) -> None:
        self._kernel_library.load_paths(paths)
        context = default_mlir_context()
        self._mlir_op.attributes[_KERNEL_LIBRARY_PATHS_ATTR_NAME] = (
            mlir.ArrayAttr.get(
                [
                    mlir.StringAttr.get(str(path), context)
                    for path in self._kernel_library.library_paths()
                ]
            )
        )

    @property
    def kernel_libraries_paths(self) -> list[Path]:
        """Returns the list of extra kernel libraries paths for the custom ops."""
        return self._kernel_library.library_paths()


class GraphBlock:
    """An MLIR block that can be populated with ops before its owning op exists.

    Use this to build a region's body before creating the op that will own
    the region. This avoids the "create op with empty regions → populate →
    verify" pattern that requires verification pausing and manual erase on
    failure — if block construction raises, no user-visible op exists.

    Within a ``with`` context, the active :class:`Graph`'s ``_current_block``
    points at this block, so the usual ``ops.*`` calls insert into this
    block instead of the graph's main body. Chain state is isolated (the
    block runs with its own ``device_chains``) so block-local side effects
    don't leak into the outer graph.

    Pass the block's :attr:`mlir_block` to an op wrapper that accepts a
    pre-built block (e.g., ``mo.if_(..., then_block=...)``); the wrapper
    moves the block into the op's region.

    Implementation note: the MLIR upstream Python binding requires every
    ``mlir.Block`` to have a parent op (``PyBlock`` holds a ``PyOperation``
    reference for lifetime/context, and ``mlir.InsertionPoint`` reads
    ``block.getParentOperation()`` for validation). To satisfy that
    invariant, each :class:`GraphBlock` owns a private scratch
    ``mo.graph`` op (inside a throwaway ``builtin.module``) that hosts its
    block until it's moved into its real owning op. The scratch op is
    deallocated when the :class:`GraphBlock` Python wrapper is.
    """

    def __init__(self, arg_types: Sequence[Type[Any]] = ()) -> None:
        self._graph = Graph.current
        # `mlir.Block.create_at_start` wants upstream `mlir.Type`, but
        # `Type.to_mlir()` returns `_core.Type`. Bridge across the CAPI.
        mlir_arg_types = [
            mlir.Type._CAPICreate(t.to_mlir()._CAPIPtr)  # type: ignore[attr-defined]
            for t in arg_types
        ]
        with _location() as loc:
            arg_locs = [loc for _ in mlir_arg_types]
            self._scratch_module = builtin.ModuleOp(location=loc)
            scratch_builder = OpBuilder(self._scratch_module.body.end)
            self._scratch_graph_op = _mo.GraphOp(
                scratch_builder,
                loc,
                name="__graph_block_scratch",
                input_types=[],
                result_types=[],
            )
            scratch_region = mlir.Operation._CAPICreate(
                self._scratch_graph_op._CAPIPtr
            ).regions[0]
            self._mlir_block = mlir.Block.create_at_start(
                scratch_region, mlir_arg_types, arg_locs
            )
        self._block_ctx: (
            contextlib.AbstractContextManager[mlir.Block] | None
        ) = None

    @property
    def mlir_block(self) -> mlir.Block:
        """The underlying ``mlir.Block``.

        Pass this to op wrappers that accept a pre-built block (e.g.,
        ``mo.IfOp(..., then_block=...)``).
        """
        return self._mlir_block

    @property
    def arguments(self) -> Sequence[mlir.Value[Any]]:
        """The block's MLIR arguments, in the order declared via ``arg_types``."""
        return self._mlir_block.arguments

    @property
    def output_types(self) -> list[mlir.Type]:
        """Types of the values yielded by this block's terminator.

        Reads from the block's last op. Available after :meth:`output` has
        been called.
        """
        operations = self._mlir_block.operations
        if not operations:
            raise RuntimeError(
                "GraphBlock.output_types is only available after output() "
                "has been called."
            )
        terminator = operations[-1]
        return [operand.type for operand in terminator.operands]

    def __enter__(self) -> GraphBlock:
        self._block_ctx = self._graph._block(self._mlir_block)
        self._block_ctx.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool | None:  # noqa: ANN001
        assert self._block_ctx is not None
        ctx = self._block_ctx
        self._block_ctx = None
        return ctx.__exit__(exc_type, exc_val, exc_tb)

    def output(
        self,
        *values: Value[Any] | TensorValueLike,
        terminator: Callable[..., Any] = mo.YieldOp,
    ) -> None:
        """Terminate this block by emitting ``terminator(*values)``.

        Bare emit — no chain plumbing. Callers that need to bundle device
        chain operands into the yield (control-flow ops like ``mo.if`` /
        ``mo.while``) should do so explicitly via
        :meth:`~_DeviceChainMap.pack` before calling :meth:`output`.

        ``terminator`` defaults to ``mo.YieldOp``. Pass a different
        terminator (or an adapter callable) when the block's owning op
        expects a non-yield terminator — e.g., ``mo.while``'s condition
        block uses ``mo.WhileConditionOp(condition, operands)``, which
        splits its first operand off from the rest.

        Verification is paused around the terminator's construction because
        the block is still parked in scratch — ``mo.yield``'s parent-type
        check will run later, against the real owning op, when the
        surrounding control-flow op (e.g., ``mo.if``) is created.
        """
        graph = self._graph
        with graph._pause_verification():
            graph._add_op(terminator, list(values))
