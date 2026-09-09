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

"""Graph-compiler gather model cache for the MO interpreter.

Covers the gather sub-family: ``GatherOp`` (``ops.gather``, index along one
axis) and ``GatherNdOp`` (``ops.gather_nd``, a static-length index vector).

Both are pure data movement, so -- like the shape-rearrange family -- the
handler bit-casts the data tensor to its same-width unsigned int
(:func:`uint_view_dtype`), gathers, and views the result back: one graph per
width (uint8/16/32/64) serves every dtype, including float16 and bool.
Indices are never bit-cast and stay ``int32``/``int64``. Cache key:
``(op, device, data-width, index-dtype)``.

``GatherOp``'s ``axis`` is a compile-time attribute, canonicalized out: the
handler views the input to rank 3 ``[outer, axis, inner]`` and the indices to
rank 1 ``[num_indices]``, and the graph gathers at ``axis=1``.

``GatherNdOp`` gets its own fused graph rather than delegating to
``GatherOp``: ``ops.gather_nd`` requires ``indices.shape[-1]``
(``index_depth``) to be static, so each graph first flattens the index vector
to one scalar (``flat_idx = sum(indices * strides)``, see
:func:`gc_compile.flat_index_strides`) and feeds it straight into the real
``ops.gather_nd`` -- ``index_depth`` stays fully dynamic, never a cache-key
variant. This keeps the real GPU-capable kernel in the loop, and its own
``batch_dims=1`` natively handles per-batch-distinct indices with no
batch-folding needed in the handler.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

Index-bounds behavior matches the deleted kernel: an out-of-range index is
undefined (no clamp/wrap).
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from math import prod
from typing import TypeAlias

from max import _core, engine
from max._core.dialects import mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops

_WIDTH_DTYPES = [DType.uint8, DType.uint16, DType.uint32, DType.uint64]
_UINT_FOR_SIZE = {
    1: DType.uint8,
    2: DType.uint16,
    4: DType.uint32,
    8: DType.uint64,
}

# Widths the deleted gather/gather_nd dispatchers accepted; the handler raises
# on anything else. Indices are never bit-cast (their values are indices).
_INDEX_DTYPES = [DType.int32, DType.int64]

# No op in this family bakes a variant into its cache key anymore; kept for
# the (op, device, data, idx, variant) shape gather_model's callers still use.
Variant: TypeAlias = tuple[int, ...]
_NO_VARIANT: Variant = ()

GatherBuilder: TypeAlias = Callable[
    [Sequence[TensorValue], Variant], TensorValue
]
InputTypesBuilder: TypeAlias = Callable[
    [Device, DType, DType, Variant], list[TensorType]
]


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
            f"gather GC path does not support sub-byte dtype {dtype}"
        )
    return _UINT_FOR_SIZE[bits // 8]


@dataclass(frozen=True)
class GatherSpec:
    """How to build one gather-family op's canonical graph and its variants."""

    input_types: InputTypesBuilder
    build: GatherBuilder
    variants: tuple[Variant, ...] = (_NO_VARIANT,)


def _gather_input_types(
    device: Device, data: DType, idx: DType, variant: Variant
) -> list[TensorType]:
    ref = DeviceRef.from_device(device)
    return [
        TensorType(data, ["outer", "axis", "inner"], device=ref),
        TensorType(idx, ["num_indices"], device=ref),
    ]


def _gather_body(
    inputs: Sequence[TensorValue], variant: Variant
) -> TensorValue:
    x, indices = inputs
    return ops.gather(x, indices, axis=1)


_GATHER_OPS: dict[type[_core.Operation], GatherSpec] = {
    mo.GatherOp: GatherSpec(_gather_input_types, _gather_body),
}

GATHER_GC_OPS = tuple(_GATHER_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_GATHER_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _GATHER_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> GatherSpec | None:
    return gc_compile.spec_for(op_type, _GATHER_OPS_BY_NAME)


def _variant_tag(variant: Variant) -> str:
    """Cache-key suffix for a variant; empty for the no-variant default."""
    return f"_k{variant[0]}" if variant else ""


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    data: DType,
    idx: DType,
    variant: Variant = _NO_VARIANT,
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, width, index)."""
    name = gc_compile.canonical_op_name(op_type, _GATHER_OPS_BY_NAME)
    return (
        f"gather_{name}_{device.label}_{device.id}_{data.name}_{idx.name}"
        f"{_variant_tag(variant)}"
    )


def _is_supported(
    op_type: type[_core.Operation], device: Device, data: DType, idx: DType
) -> bool:
    """Whether (op, device, data-width, index) is in the swept matrix.

    Single source of truth: the sweep and :func:`gather_model`'s guard both
    route through it. ``data`` is the bit-cast uint width, so every dtype maps
    into the supported set. Variant does not affect dtype support, so it is not
    an argument here.
    """
    if _spec_for(op_type) is None:
        return False
    return data in _WIDTH_DTYPES and idx in _INDEX_DTYPES


def _gather_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: GatherSpec,
    device: Device,
    data: DType,
    idx: DType,
    variant: Variant,
) -> None:
    """Adds one fully-symbolic gather graph into *module* in-place."""
    graph = Graph(
        _graph_name(op_type, device, data, idx, variant),
        input_types=spec.input_types(device, data, idx, variant),
        module=module,
    )
    with graph:
        graph.output(spec.build([v.tensor for v in graph.inputs], variant))


def _gather_nd_graph_name(device: Device, data: DType, idx_dtype: DType) -> str:
    """Graph sym_name and cache key for one (device, data-width, index)."""
    return f"gather_nd_{device.label}_{device.id}_{data.name}_{idx_dtype.name}"


def _gather_nd_graph(
    module: Module, device: Device, data_dtype: DType, idx_dtype: DType
) -> None:
    """Adds gather_nd's fused flatten-index-then-gather graph into *module*.

    ``batch_dims`` is always 1 here: :func:`gather_nd_operand_views` always
    produces a leading ``batch`` dim (1 when the real call had
    ``batch_dims=0``), and ``ops.gather_nd``'s own ``batch_dims=1`` natively
    handles per-batch-distinct indices. ``strides`` rides on CPU regardless of
    ``x``'s device (like ``conv_gc``'s shape params), so it is transferred
    onto ``device_ref`` before the elementwise multiply.
    """
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    x_type = TensorType(
        data_dtype, ["batch", "flat_indexed", "suffix"], device=device_ref
    )
    indices_type = TensorType(idx_dtype, ["batch", "m", "k"], device=device_ref)
    strides_type = TensorType(idx_dtype, ["k"], device=cpu)
    graph = Graph(
        _gather_nd_graph_name(device, data_dtype, idx_dtype),
        input_types=[x_type, indices_type, strides_type],
        module=module,
    )
    with graph:
        x_tv, indices_tv, strides_tv = (v.tensor for v in graph.inputs)
        strides_tv = ops.transfer_to(strides_tv, device_ref)
        weighted = indices_tv * strides_tv
        flat_idx = ops.squeeze(ops.sum(weighted, axis=-1), axis=-1)
        flat_idx_depth1 = ops.unsqueeze(flat_idx, axis=-1)
        result = ops.gather_nd(x_tv, flat_idx_depth1, batch_dims=1)
        graph.output(result)


class _GatherFamily(gc_compile.GCFamilySpec):
    name = "gather"

    def build_module(self) -> Module:
        """Batched module: every (op, device, width, index, variant), all
        devices, plus gather_nd's fused graph per (device, width, index)."""
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
        for op_type, spec in _GATHER_OPS.items():
            for data in _WIDTH_DTYPES:
                for idx in _INDEX_DTYPES:
                    for variant in spec.variants:
                        _gather_graph(
                            module, op_type, spec, device, data, idx, variant
                        )
        for data in _WIDTH_DTYPES:
            for idx_dtype in _INDEX_DTYPES:
                _gather_nd_graph(module, device, data, idx_dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_GatherFamily())
gc_compile.register_family(_FAMILY)


def gather_model(
    op_type: type[_core.Operation],
    device: Device,
    data: DType,
    idx: DType,
    variant: Variant = _NO_VARIANT,
) -> engine.Model:
    """Returns the gather :class:`~max.engine.Model` for the given target.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss a warm cache is adopted whole (manifest force-load,
    else a batched stamp sweep) instead of compiling per target.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled --
            always ``GatherOp`` in this family (``GatherNdOp`` has its own
            fused model, see :func:`gather_nd_model`).
        device: The realized input's device.
        data: The data tensor's bit-cast uint width (see
            :func:`uint_view_dtype`), not its original dtype.
        idx: The realized index tensor's dtype (``int32`` or ``int64``).
        variant: Always ``()`` -- no op in this family bakes a variant into
            its cache key.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, data-width, index) is outside the supported
            set; or, with ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target
            was not swept.
    """
    key = _graph_name(op_type, device, data, idx, variant)
    # Cache-check before building the closures below: this runs on every eager
    # op dispatch, so a hit must not pay for closures it won't use.
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, data, idx):
            return None
        return (
            f"Unsupported gather op/device/data/index for key {key!r}."
            f"  Supported data widths: {_WIDTH_DTYPES}; index dtypes:"
            f" {_INDEX_DTYPES}."
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _gather_graph(module, op_type, spec, device, data, idx, variant)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )


# Shape canonicalization (handler-facing)


def gather_operand_views(
    in_shape: Sequence[int], idx_shape: Sequence[int], axis: int
) -> tuple[tuple[int, int, int], tuple[int]]:
    """Zero-copy view shapes for ``GatherOp`` operands.

    Collapses the input to ``[outer, axis, inner]`` and the indices to a flat
    ``[num_indices]``; ``axis`` is already normalized non-negative by the caller.
    The graph gathers at ``axis=1`` and the handler views the result back to the
    op's true output shape.
    """
    outer = prod(in_shape[:axis])
    axis_size = in_shape[axis]
    inner = prod(in_shape[axis + 1 :])
    num_indices = prod(idx_shape)
    return (outer, axis_size, inner), (num_indices,)


def gather_nd_operand_views(
    in_shape: Sequence[int], idx_shape: Sequence[int], batch_dims: int
) -> tuple[tuple[int, int, int], tuple[int, int, int], list[int]]:
    """Zero-copy view shapes for gather_nd's operands via GatherOp's own
    canonical rank-3 form, plus the strides vector for flat-index reduction.

    Collapses ``input``'s ``index_depth`` indexed dims (after ``batch_dims``)
    into one flat axis, matching GatherOp's own ``[outer, axis, inner]``
    convention exactly: ``outer = batch``, ``axis = flat_indexed``, ``inner =
    suffix``. ``indices`` is viewed to ``[batch, m, index_depth]`` (rank 3,
    fed to :func:`gather_nd_model`'s fused graph). ``strides`` are the
    row-major strides of the real indexed dims for this call (see
    :func:`gc_compile.flat_index_strides`) -- never baked into any cache key,
    always computed fresh in Python from the real shapes.
    """
    index_depth = idx_shape[-1]
    batch = prod(in_shape[:batch_dims])
    num_index_vectors = prod(idx_shape[batch_dims:-1])
    indexed_shape = in_shape[batch_dims : batch_dims + index_depth]
    flat_indexed = prod(indexed_shape)
    suffix = prod(in_shape[batch_dims + index_depth :])
    data_view = (batch, flat_indexed, suffix)
    idx_view = (batch, num_index_vectors, index_depth)
    strides = gc_compile.flat_index_strides(indexed_shape)
    return data_view, idx_view, strides


def gather_nd_model(
    device: Device, data_udtype: DType, idx_dtype: DType
) -> engine.Model:
    """Returns gather_nd's fused flatten-index-then-gather
    :class:`~max.engine.Model` for the given target.

    Lazy by default: compiled on first use and cached for the process
    lifetime, exactly like :func:`gather_model`; part of the ``"gather"``
    family's own sweep, not a separate family.

    Args:
        device: The realized input's device.
        data_udtype: The bit-cast uint width (see :func:`uint_view_dtype`).
        idx_dtype: The realized index tensor's dtype (``int32`` or ``int64``).

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If (device, data_udtype, idx_dtype) is outside the
            supported set.
    """
    key = _gather_nd_graph_name(device, data_udtype, idx_dtype)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if data_udtype in _WIDTH_DTYPES and idx_dtype in _INDEX_DTYPES:
            return None
        return (
            f"Unsupported gather_nd op/device/data/index for key {key!r}."
            f"  Supported data widths: {_WIDTH_DTYPES}; index dtypes:"
            f" {_INDEX_DTYPES}."
        )

    def build(module: Module) -> None:
        _gather_nd_graph(module, device, data_udtype, idx_dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
