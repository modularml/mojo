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

"""Graph-compiler scatter-nd model cache for the MO interpreter.

Covers the scatter-nd sub-family: ``ScatterNdOp`` (overwrite) and the reduce
variants ``ScatterNdAddOp``/``ScatterNdMaxOp``/``ScatterNdMinOp``/
``ScatterNdMulOp``. Each scatters ``updates`` into a copy of ``input`` at the
N-dimensional index vectors in ``indices`` (last dim ``index_depth`` <= rank,
no ``batch_dims``).

``ops.scatter_nd`` requires ``indices.shape[-1]`` (``index_depth``) to be
**static**, which used to bake it into the cache key as a variant
(``1..MAX_RANK``), one graph per depth. Instead each graph flattens each
multi-dim index vector to one scalar (``flat_idx = sum(indices * strides)``,
see :func:`gc_compile.flat_index_strides`) and feeds the depth-1 result
straight into the real ``ops.scatter_nd``/``ops.scatter_nd_add``/etc. in the
same compiled graph -- ``index_depth`` stays fully dynamic, never a cache-key
variant. Unlike the axis-based ``ScatterOp`` family (``scatter_gc.py``), this
keeps the real GPU-capable kernel in the loop: duplicate index vectors resolve
deterministically (overwrite is last-write-wins, reduce accumulates
atomically via the GPU kernel's compare-and-swap loop).

Two keying schemes, matching the two op kinds:

- **Overwrite** (``ScatterNdOp``) is pure data movement, so -- like gather and
  the axis-scatter overwrite -- the handler bit-casts every dtype to its
  same-width unsigned int (:func:`uint_view_dtype`), scatters, and views back.
- **Reduce** (add/max/min/mul) accumulates arithmetically, so a bit-cast would
  corrupt it; these key on the real dtype (see :func:`_reduce_dtypes`).

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
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

# Index widths the deleted scatter-nd dispatchers accepted; the handler raises on
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
            f"scatter-nd GC path does not support sub-byte dtype {dtype}"
        )
    return _UINT_FOR_SIZE[bits // 8]


class Kind(Enum):
    """Whether an op overwrites (pure copy) or reduces (arithmetic)."""

    OVERWRITE = "overwrite"
    REDUCE = "reduce"


# Builds the graph body from its [input, updates, flat_idx_depth1] TensorValues.
ScatterNdBuilder: TypeAlias = Callable[[Sequence[TensorValue]], TensorValue]


@dataclass(frozen=True)
class ScatterNdSpec:
    """How one scatter-nd op builds its graph and its kind."""

    build: ScatterNdBuilder
    kind: Kind


def _b_scatter_nd(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_nd(x, updates, indices)


def _b_scatter_nd_add(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_nd_add(x, updates, indices)


def _b_scatter_nd_max(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_nd_max(x, updates, indices)


def _b_scatter_nd_min(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_nd_min(x, updates, indices)


def _b_scatter_nd_mul(ins: Sequence[TensorValue]) -> TensorValue:
    x, updates, indices = ins
    return ops.scatter_nd_mul(x, updates, indices)


_SCATTER_ND_OPS: dict[type[_core.Operation], ScatterNdSpec] = {
    mo.ScatterNdOp: ScatterNdSpec(_b_scatter_nd, Kind.OVERWRITE),
    mo.ScatterNdAddOp: ScatterNdSpec(_b_scatter_nd_add, Kind.REDUCE),
    mo.ScatterNdMaxOp: ScatterNdSpec(_b_scatter_nd_max, Kind.REDUCE),
    mo.ScatterNdMinOp: ScatterNdSpec(_b_scatter_nd_min, Kind.REDUCE),
    mo.ScatterNdMulOp: ScatterNdSpec(_b_scatter_nd_mul, Kind.REDUCE),
}

SCATTER_ND_GC_OPS = tuple(_SCATTER_ND_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_SCATTER_ND_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _SCATTER_ND_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> ScatterNdSpec | None:
    return gc_compile.spec_for(op_type, _SCATTER_ND_OPS_BY_NAME)


def _reduce_dtypes(device: Device) -> list[DType]:
    """Real dtypes the reduce variants sweep on *device*.

    ``float_dtypes`` (f32/f64 on CPU -- 16-bit floats don't compile on CPU;
    f16/f32/bf16 on GPU) plus every signed/unsigned int width. The int set is
    full on CPU and CUDA/ROCm (empirically all widths compile); Metal is
    32-bit only -- its atomic compare-exchange has no other width, and the
    reduce kernel asserts on it at compile time (KERN-3243).
    """
    dtypes = (
        gc_compile.float_dtypes(device)
        + gc_compile.SIGNED_INT_DTYPES
        + gc_compile.UNSIGNED_INT_DTYPES
    )
    if device.api == "metal":
        dtypes = [d for d in dtypes if d.size_in_bits == 32]
    return dtypes


def _data_dtypes(spec: ScatterNdSpec, device: Device) -> list[DType]:
    """Swept data dtypes for one op on *device*: widths (overwrite) or reals."""
    if spec.kind is Kind.OVERWRITE:
        return list(_WIDTH_DTYPES)
    return _reduce_dtypes(device)


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    idx_dtype: DType,
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype, index).

    ``dtype`` is the bit-cast uint width for ``ScatterNdOp`` and the real dtype
    for the reduce variants. No ``index_depth`` component: it is fully dynamic,
    flattened away inside the graph.
    """
    name = gc_compile.canonical_op_name(op_type, _SCATTER_ND_OPS_BY_NAME)
    return (
        f"scatter_nd_{name}_{device.label}_{device.id}_{dtype.name}"
        f"_{idx_dtype.name}"
    )


def _is_supported(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    idx_dtype: DType,
) -> bool:
    """Whether (op, device, dtype, index) is in the swept matrix.

    Single source of truth: the sweep and :func:`scatter_nd_model`'s guard both
    route through it.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    return dtype in _data_dtypes(spec, device) and idx_dtype in _INDEX_DTYPES


def _scatter_nd_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: ScatterNdSpec,
    device: Device,
    dtype: DType,
    idx_dtype: DType,
) -> None:
    """Adds one fused flatten-index-then-scatter graph into *module* in-place.

    ``x``/``updates`` are already collapsed to ``[flat_indexed,
    suffix]``/``[m, suffix]`` (see :func:`scatter_nd_operand_views`) --
    ``ops.scatter_nd``'s ``indices`` are the compact ``[num_updates, k]``
    list-of-vectors shape, not elementwise-matching ``updates`` like the
    axis-based ``ops.scatter``, so no broadcast across ``suffix`` is needed.
    ``strides`` rides on CPU regardless of ``x``'s device (like ``conv_gc``'s
    shape params), so it is transferred onto ``device_ref`` before the
    elementwise multiply.
    """
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    x_type = TensorType(dtype, ["flat_indexed", "suffix"], device=device_ref)
    updates_type = TensorType(dtype, ["m", "suffix"], device=device_ref)
    indices_type = TensorType(idx_dtype, ["m", "k"], device=device_ref)
    strides_type = TensorType(idx_dtype, ["k"], device=cpu)
    graph = Graph(
        _graph_name(op_type, device, dtype, idx_dtype),
        input_types=[x_type, updates_type, indices_type, strides_type],
        module=module,
    )
    with graph:
        x_tv, updates_tv, indices_tv, strides_tv = (
            v.tensor for v in graph.inputs
        )
        strides_tv = ops.transfer_to(strides_tv, device_ref)
        weighted = indices_tv * strides_tv
        flat_idx = ops.squeeze(ops.sum(weighted, axis=-1), axis=-1)
        flat_idx_depth1 = ops.unsqueeze(flat_idx, axis=-1)
        result = spec.build([x_tv, updates_tv, flat_idx_depth1])
        graph.output(result)


# Maps each scatter_nd op to the bazel-facing family name it gets split
# into -- one family per op, not one family for the whole op group, so
# each op's own warm-cache bazel action stays small. The 4 reduce ops'
# dtype-swept matrix (11 dtypes x 2 idx = 22 graphs each) was the dominant
# cost in the single combined "scatter_nd" family (96 graphs/slot, ~937s
# critical path, over the 900s budget); splitting means each bazel action
# only builds its own op's slice (8 for overwrite, 22 each for the reduce
# ops).
_FAMILY_NAME_FOR_OP: dict[type[_core.Operation], str] = {
    mo.ScatterNdOp: "scatter_nd",
    mo.ScatterNdAddOp: "scatter_nd_add",
    mo.ScatterNdMaxOp: "scatter_nd_max",
    mo.ScatterNdMinOp: "scatter_nd_min",
    mo.ScatterNdMulOp: "scatter_nd_mul",
}


class _ScatterNdFamily(gc_compile.GCFamilySpec):
    """Builds one scatter_nd op's own (dtype, idx_dtype) matrix.

    One instance per op (see _FAMILY_NAME_FOR_OP) -- each instance is its
    own bazel-visible family (its own warm-cache action), so splitting by
    op keeps each action's compile matrix small instead of needing all 5
    ops' graphs in one action.
    """

    def __init__(self, op_type: type[_core.Operation]) -> None:
        self.name = _FAMILY_NAME_FOR_OP[op_type]
        self._op_type = op_type

    def build_module(self) -> Module:
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        if module is None:
            module = Module()
        spec = _SCATTER_ND_OPS[self._op_type]
        for dtype in _data_dtypes(spec, device):
            for idx_dtype in _INDEX_DTYPES:
                _scatter_nd_graph(
                    module, self._op_type, spec, device, dtype, idx_dtype
                )
        return module


_FAMILIES: dict[type[_core.Operation], gc_compile.GCOpFamily] = {
    op_type: gc_compile.GCOpFamily(_ScatterNdFamily(op_type))
    for op_type in _SCATTER_ND_OPS
}
for _family in _FAMILIES.values():
    gc_compile.register_family(_family)


def scatter_nd_model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    idx_dtype: DType,
) -> engine.Model:
    """Returns scatter_nd's fused flatten-index-then-scatter
    :class:`~max.engine.Model` for the given target.

    The multi-dim indices are flattened to a scalar per vector and fed
    straight into the real ``ops.scatter_nd*`` inside one compiled graph (see
    :func:`_scatter_nd_graph`).

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss a warm cache is adopted whole (manifest force-load,
    else a batched stamp sweep) instead of compiling per target.

    Args:
        op_type: The concrete ``mo.*Op`` type being handled.
        device: The realized input's device.
        dtype: The bit-cast uint width for ``ScatterNdOp`` (see
            :func:`uint_view_dtype`), else the real data dtype for the reduce
            variants (see :func:`model_dtype`).
        idx_dtype: The realized index tensor's dtype (``int32`` or ``int64``).

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype, index) is outside the supported
            set; or, with ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target
            was not swept.
    """
    key = _graph_name(op_type, device, dtype, idx_dtype)
    family = _FAMILIES[op_type]
    # Cache-check before building the closures below: this runs on every eager
    # op dispatch, so a hit must not pay for closures it won't use.
    model = family.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, dtype, idx_dtype):
            return None
        spec = _spec_for(op_type)
        supported = _data_dtypes(spec, device) if spec else []
        return (
            f"Unsupported scatter-nd op/device/dtype/index for key {key!r}."
            f"  Supported data dtypes for this op/device: {supported};"
            f"  index dtypes: {_INDEX_DTYPES}."
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _scatter_nd_graph(module, op_type, spec, device, dtype, idx_dtype)

    return family.model_for(
        key, device, build, unsupported_reason=check_supported
    )


# Handler-facing helpers


def model_dtype(op_type: type[_core.Operation], data_dtype: DType) -> DType:
    """The dtype to build/view operands as for *op_type*.

    The same-width uint for overwrite ``ScatterNdOp`` (bit-cast copy), else the
    real data dtype for the reduce variants.
    """
    spec = _spec_for(op_type)
    if spec is not None and spec.kind is Kind.OVERWRITE:
        return uint_view_dtype(data_dtype)
    return data_dtype


def scatter_nd_operand_views(
    in_shape: Sequence[int], idx_shape: Sequence[int]
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int], list[int]]:
    """Zero-copy view shapes for scatter_nd's operands, plus the strides vector
    for flat-index reduction.

    Collapses ``input``'s ``index_depth`` indexed dims into one flat axis:
    ``input_view = (flat_indexed, suffix)``, ``updates_view = (m, suffix)``.
    ``indices`` is viewed to ``[m, index_depth]`` (rank 2, fed to
    :func:`scatter_nd_model`'s fused graph directly -- ``ops.scatter_nd``'s own
    indices shape, no ``[outer, axis, inner]`` wrapper dim like the axis-based
    ``ScatterOp`` convention needs). ``strides`` are the row-major strides of
    the real indexed dims for this call (see :func:`gc_compile.
    flat_index_strides`) -- never baked into any cache key, always computed
    fresh in Python from the real shapes.
    """
    index_depth = idx_shape[-1]
    num_index_vectors = prod(idx_shape[:-1])
    indexed_shape = in_shape[:index_depth]
    flat_indexed = prod(indexed_shape)
    suffix = prod(in_shape[index_depth:])
    input_view = (flat_indexed, suffix)
    updates_view = (num_index_vectors, suffix)
    indices_view = (num_index_vectors, index_depth)
    strides = gc_compile.flat_index_strides(indexed_shape)
    return input_view, updates_view, indices_view, strides
