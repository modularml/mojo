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

"""Graph-compiler scatter model cache for the MO interpreter.

Covers the axis-scatter sub-family: ``ScatterOp`` (overwrite) and the reduce
variants ``ScatterAddOp``/``ScatterMaxOp``/``ScatterMinOp``/``ScatterMulOp``.
Each scatters ``updates`` into a copy of ``input`` along one ``axis`` according
to ``indices``; ``ops.scatter*`` builds the copy, so the handler needs no
explicit memcpy. None of them has a GPU kernel yet, so ``ops.scatter*``
implicitly transfers to CPU regardless of the input device.

Two keying schemes, because the two op kinds differ:

- **Overwrite** (``ScatterOp``) is pure data movement, so -- like gather and
  shape-rearrange -- the handler bit-casts every dtype to its same-width
  unsigned int (:func:`uint_view_dtype`), scatters, and views back. One graph
  per width serves every dtype, including float16.
- **Reduce** (add/max/min/mul) does arithmetic on the accumulator, so a
  bit-cast would corrupt it; these key on the real dtype (see
  :func:`_reduce_dtypes`).

``axis`` is canonicalized out: the handler collapses ``input`` to rank 3
``[outer, axis, inner]`` and ``updates``/``indices`` to ``[outer, updates,
inner]`` (zero-copy views), each graph scatters at ``axis=1``, and the result
is viewed back to the input's shape. ``ScatterOp`` carries ``axis`` as a
compile-time attribute; the reduce variants carry it as a runtime scalar
operand (the handler reads whichever). Both are dropped from the cache key.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from math import prod
from typing import TypeAlias, cast

from max import _core, engine
from max._core.dialects import mo
from max._interpreter_ops import gc_compile
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops

_WIDTH_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]
_UINT_FOR_SIZE = {
    1: DType.uint8,
    2: DType.uint16,
    4: DType.uint32,
    8: DType.uint64,
}

# Index widths the deleted scatter dispatchers accepted; the handler raises on
# anything else. Indices are never bit-cast (their values are indices).
_INDEX_DTYPES = [DType.int32, DType.int64]


def uint_view_dtype(dtype: DType) -> DType:
    """The same-bit-width unsigned int a dtype is bit-cast to for copying.

    Raises:
        NotImplementedError: For sub-byte dtypes (e.g. ``float4_e2m1fn``), which
            pack multiple elements per byte and so cannot be reinterpreted
            element-for-element as a whole-byte unsigned int.
    """
    bits = dtype.size_in_bits
    if bits % 8 != 0 or bits // 8 not in _UINT_FOR_SIZE:
        raise NotImplementedError(
            f"scatter GC path does not support sub-byte dtype {dtype}"
        )
    return _UINT_FOR_SIZE[bits // 8]


class Kind(Enum):
    """Whether an op overwrites (pure copy) or reduces (arithmetic)."""

    OVERWRITE = "overwrite"
    REDUCE = "reduce"


ScatterBuilder: TypeAlias = Callable[[Sequence[TensorValue]], TensorValue]


@dataclass(frozen=True)
class ScatterSpec:
    """How one scatter op builds its graph, plus its kind and axis source."""

    build: ScatterBuilder
    kind: Kind
    axis_from_operand: bool


def _b_scatter(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter(x, updates, indices, axis=1)


def _b_scatter_add(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_add(x, updates, indices, axis=1)


def _b_scatter_max(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_max(x, updates, indices, axis=1)


def _b_scatter_min(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_min(x, updates, indices, axis=1)


def _b_scatter_mul(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_mul(x, updates, indices, axis=1)


_SCATTER_OPS: dict[type[_core.Operation], ScatterSpec] = {
    mo.ScatterOp: ScatterSpec(
        _b_scatter, Kind.OVERWRITE, axis_from_operand=False
    ),
    mo.ScatterAddOp: ScatterSpec(
        _b_scatter_add, Kind.REDUCE, axis_from_operand=True
    ),
    mo.ScatterMaxOp: ScatterSpec(
        _b_scatter_max, Kind.REDUCE, axis_from_operand=True
    ),
    mo.ScatterMinOp: ScatterSpec(
        _b_scatter_min, Kind.REDUCE, axis_from_operand=True
    ),
    mo.ScatterMulOp: ScatterSpec(
        _b_scatter_mul, Kind.REDUCE, axis_from_operand=True
    ),
}

SCATTER_GC_OPS = tuple(_SCATTER_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_SCATTER_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _SCATTER_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> ScatterSpec | None:
    return gc_compile.spec_for(op_type, _SCATTER_OPS_BY_NAME)


def _reduce_dtypes(device: Device) -> list[DType]:
    """Real dtypes the reduce variants (add/max/min/mul) sweep on *device*.

    ``float_dtypes`` (f32/f64 on CPU: 16-bit floats don't compile on CPU;
    f16/f32/bf16 on GPU) plus every signed/unsigned int. The int set is full on
    both devices: scatter transfers to CPU, so 8/16-bit int reductions compile
    on GPU too (unlike ``reduce_axis``, whose reduction runs on the GPU).
    Derived empirically.
    """
    return (
        gc_compile.float_dtypes(device)
        + gc_compile.SIGNED_INT_DTYPES
        + gc_compile.UNSIGNED_INT_DTYPES
    )


def _data_dtypes(spec: ScatterSpec, device: Device) -> list[DType]:
    """Swept data dtypes for one op on *device*: widths (overwrite) or reals."""
    if spec.kind is Kind.OVERWRITE:
        return list(_WIDTH_DTYPES)
    return _reduce_dtypes(device)


def _graph_name(
    op_type: type[_core.Operation], device: Device, dtype: DType, idx: DType
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype, index).

    ``dtype`` is the bit-cast uint width for ``ScatterOp`` and the real dtype for
    the reduce variants.
    """
    name = gc_compile.canonical_op_name(op_type, _SCATTER_OPS_BY_NAME)
    return f"scatter_{name}_{device.label}_{device.id}_{dtype.name}_{idx.name}"


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType, idx: DType
) -> bool:
    """Whether (op, device, dtype, index) is in the swept matrix.

    Single source of truth: the sweep and :func:`scatter_model`'s guard both
    route through it.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    return dtype in _data_dtypes(spec, device) and idx in _INDEX_DTYPES


def _scatter_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: ScatterSpec,
    device: Device,
    dtype: DType,
    idx: DType,
) -> None:
    """Adds one fully-symbolic rank-3 scatter graph into *module* in-place.

    ``input`` is ``[outer, a, inner]`` and ``updates``/``indices`` are
    ``[outer, u, inner]`` (they share ``outer``/``inner`` with the input but
    scatter along their own ``u``); the op scatters at ``axis=1``.
    """
    device_ref = DeviceRef.from_device(device)
    graph = Graph(
        _graph_name(op_type, device, dtype, idx),
        input_types=[
            TensorType(dtype, ["outer", "a", "inner"], device=device_ref),
            TensorType(dtype, ["outer", "u", "inner"], device=device_ref),
            TensorType(idx, ["outer", "u", "inner"], device=device_ref),
        ],
        module=module,
    )
    with graph:
        graph.output(spec.build([v.tensor for v in graph.inputs]))


class _ScatterFamily(gc_compile.GCFamilySpec):
    name = "scatter"

    def build_module(self) -> Module:
        """Batched module: every supported (op, device, dtype, index), all
        devices."""
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Per-slot counterpart of :meth:`build_module`: one *device* only."""
        if module is None:
            module = Module()
        for op_type, spec in _SCATTER_OPS.items():
            for dtype in _data_dtypes(spec, device):
                for idx in _INDEX_DTYPES:
                    _scatter_graph(module, op_type, spec, device, dtype, idx)
        return module


_FAMILY = gc_compile.GCOpFamily(_ScatterFamily())
gc_compile.register_family(_FAMILY)


def scatter_model(
    op_type: type[_core.Operation], device: Device, dtype: DType, idx: DType
) -> engine.Model:
    """Returns the scatter :class:`~max.engine.Model` for the given target.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss a warm cache is adopted whole (manifest force-load,
    else a batched stamp sweep) instead of compiling per target.

    Args:
        op_type: The concrete ``mo.*Op`` type being handled.
        device: The realized input's device.
        dtype: The bit-cast uint width for ``ScatterOp`` (see
            :func:`uint_view_dtype`), else the real data dtype.
        idx: The realized index tensor's dtype (``int32`` or ``int64``).

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype, index) is outside the supported set;
            or, with ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target was not
            swept.
    """
    key = _graph_name(op_type, device, dtype, idx)
    # Cache-check before building the closures below: this runs on every eager
    # op dispatch, so a hit must not pay for closures it won't use.
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, dtype, idx):
            return None
        spec = _spec_for(op_type)
        supported = _data_dtypes(spec, device) if spec else []
        return (
            f"Unsupported scatter op/device/dtype/index for key {key!r}."
            f"  Supported data dtypes for this op/device: {supported};"
            f"  index dtypes: {_INDEX_DTYPES}."
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _scatter_graph(module, op_type, spec, device, dtype, idx)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )


# Handler-facing helpers


def model_dtype(op_type: type[_core.Operation], data_dtype: DType) -> DType:
    """The dtype to build/view operands as for *op_type*.

    The same-width uint for overwrite ``ScatterOp`` (bit-cast copy), else the
    real data dtype for the reduce variants.
    """
    spec = _spec_for(op_type)
    if spec is not None and spec.kind is Kind.OVERWRITE:
        return uint_view_dtype(data_dtype)
    return data_dtype


def scatter_axis(op: _core.Operation, inputs: Sequence[Buffer | None]) -> int:
    """The raw (possibly negative) scatter axis for *op*.

    ``ScatterOp`` carries it as a compile-time attribute; the reduce variants
    carry it as a runtime scalar operand (``inputs[3]``).
    """
    spec = _spec_for(type(op))
    if spec is not None and spec.axis_from_operand:
        axis_buf = inputs[3]
        assert isinstance(axis_buf, Buffer)
        return int(axis_buf.to_numpy().item())
    return cast(mo.ScatterOp, op).axis


def scatter_operand_views(
    in_shape: Sequence[int], upd_shape: Sequence[int], axis: int
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    """Zero-copy view shapes for the input and the updates/indices operands.

    Collapses to ``input = [outer, axis, inner]`` and
    ``updates = indices = [outer, num_updates_axis, inner]``; ``axis`` is already
    normalized non-negative by the caller. The graph scatters at ``axis=1`` and
    the handler views the result back to the input's true shape.
    """
    outer = prod(in_shape[:axis])
    axis_size = in_shape[axis]
    inner = prod(in_shape[axis + 1 :])
    num_updates_axis = upd_shape[axis]
    # outer/inner come from the input, so mismatched input/updates geometry
    # (e.g. sharded input, replicated updates) yields a view that under-covers
    # the updates buffer and the kernel writes out of bounds (KERN-3291).
    updates_elems = outer * num_updates_axis * inner
    if prod(upd_shape) != updates_elems:
        raise ValueError(
            f"scatter updates shape {tuple(upd_shape)} is incompatible with"
            f" input shape {tuple(in_shape)} at axis {axis}: the collapsed"
            f" updates view {(outer, num_updates_axis, inner)} covers"
            f" {updates_elems} elements but updates holds {prod(upd_shape)}."
            " The input and updates/indices must be sharded consistently."
        )
    return (outer, axis_size, inner), (outer, num_updates_axis, inner)
