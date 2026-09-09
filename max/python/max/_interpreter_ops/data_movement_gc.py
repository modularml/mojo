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

"""Graph-compiler data-movement model cache for the MO interpreter.

Covers the last three ops that ran on hand-written Mojo bindings:
``Transpose``, ``BroadcastTo`` (also serving ``StaticBroadcastTo``), and
``MutableStoreSlice``. Each is a pure copy (no value interpretation), so
graphs are built per uint width (:func:`gc_compile.uint_view_dtype`) and
handlers bit-cast in and out with zero-copy views.

Structural parameters are runtime inputs, not baked: the permutation and
slice starts/stops/steps arrive as host tensors, and the broadcast target
shape arrives as the *shapes* of tiny host carrier inputs, so one graph
per ``(op, device, width)`` serves every value of them. Rank is
canonicalized away too: every graph is built at ``MAX_RANK`` and the
execute helpers here (:func:`transpose`, :func:`broadcast_to`,
:func:`store_slice`) pad shapes to it with leading 1s (the same scheme
the deleted Mojo kernels used), so the sweep is ops x widths, not ops x
widths x ranks.

``MutableStoreSlice`` is the family's one unusual member: its graph takes
an in-place ``BufferType`` input, emits ``rmo.MoMutableStoreSliceOp`` with
the device-chain idiom from ``max.graph.ops.buffer_store_slice``, and has
no tensor outputs (``graph.output()`` packs the chain).

Compile modes mirror :mod:`shape_rearrange_gc` (lazy-per-target by
default, batched sweep under ``MAX_EAGER_OP_PRECOMPILE=1``, warm-stamp
adoption). Must not import from ``handlers.py``.
"""

from collections.abc import Callable

import numpy as np
from max import _core, engine
from max._core.dialects import kgen, mo, rmo
from max._interpreter_ops import gc_compile
from max.driver import Buffer, Device
from max.dtype import DType
from max.graph import (
    BufferType,
    DeviceRef,
    Graph,
    Module,
    TensorType,
)
from numpy.typing import NDArray

MAX_RANK = gc_compile.MAX_RANK  # Every graph is built at exactly this rank.

_WIDTH_DTYPES = gc_compile.WIDTH_DTYPES

_BuildGraph = Callable[[Module, Device, DType], None]

_MOVE_OPS: dict[type[_core.Operation], _BuildGraph] = {}

_MOVE_OPS_BY_NAME: dict[str, _BuildGraph] = {}


def _register_spec(op_type: type[_core.Operation], build: _BuildGraph) -> None:
    _MOVE_OPS[op_type] = build


def _refresh_by_name() -> None:
    _MOVE_OPS_BY_NAME.clear()
    _MOVE_OPS_BY_NAME.update(
        {op_type.__name__: build for op_type, build in _MOVE_OPS.items()}
    )


def _spec_for(op_type: type[_core.Operation]) -> _BuildGraph | None:
    return gc_compile.spec_for(op_type, _MOVE_OPS_BY_NAME)


_DEVICES = gc_compile.DISCOVERED_DEVICES


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
) -> str:
    """Graph ``sym_name`` and cache key for one target."""
    name = gc_compile.canonical_op_name(op_type, _MOVE_OPS_BY_NAME)
    return f"data_movement_{name}_{device.label}_{device.id}_{dtype.name}"


def _is_supported(device: Device, dtype: DType) -> bool:
    return dtype in _WIDTH_DTYPES


class _DataMovementFamily(gc_compile.GCFamilySpec):
    name = "data_movement"

    def build_module(self) -> Module:
        """Batched module: every (op, device, width)."""
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        if module is None:
            module = Module()
        for build in _MOVE_OPS.values():
            for dtype in _WIDTH_DTYPES:
                build(module, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_DataMovementFamily())
gc_compile.register_family(_FAMILY)


def model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
) -> engine.Model:
    """Return the compiled model for the given target (lazy by default)."""
    key = _graph_name(op_type, device, dtype)
    # Cache-check before building the closures below: this runs on every
    # eager op dispatch, so a hit must not pay for closures it won't use.
    cached = _FAMILY.cache.get(key)
    if cached is not None:
        return cached

    def check_supported() -> str | None:
        if _spec_for(op_type) is None or not _is_supported(device, dtype):
            return f"Unsupported data-movement op/device/dtype for key {key!r}."
        return None

    def build(module: Module) -> None:
        build_graph = _spec_for(op_type)
        assert build_graph is not None, (
            f"unsupported op {op_type!r} reached compile"
        )
        build_graph(module, device, dtype)

    return _FAMILY.model_for(
        key,
        device,
        build,
        unsupported_reason=check_supported,
    )


def _pad_for(rank: int, op_name: str, what: str) -> int:
    """Returns the leading-axis pad count to reach ``MAX_RANK``."""
    if rank > MAX_RANK:
        raise ValueError(
            f"{op_name} supports at most rank-{MAX_RANK} {what},"
            f" got rank {rank}"
        )
    return MAX_RANK - rank


def _scalar_copy(input_buffer: Buffer, device: Device) -> Buffer:
    output = Buffer(shape=(), dtype=input_buffer.dtype, device=device)
    output.inplace_copy_from(input_buffer)
    return output


def transpose(input_buffer: Buffer, perm: list[int], device: Device) -> Buffer:
    """Transpose via the fixed-rank model.

    Pads the input with leading 1s and the perm with leading identity
    axes, then re-views the output to the real permuted shape. Rank 0
    (empty perm) is a plain copy: the graphs need at least one axis.
    """
    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    if rank == 0:
        return _scalar_copy(input_buffer, device)
    pad = _pad_for(rank, "transpose", "inputs")
    out_shape = [in_shape[p] for p in perm]
    padded_perm = list(range(pad)) + [p + pad for p in perm]
    udtype = gc_compile.uint_view_dtype(input_buffer.dtype)
    gc_model = model(mo.TransposeOp, device, udtype)
    (out,) = gc_model(
        input_buffer.view(udtype, (1,) * pad + tuple(in_shape)),
        Buffer.from_numpy(np.array(padded_perm, dtype=np.int64)),
    )
    return out.view(input_buffer.dtype, tuple(out_shape))


# Broadcast carriers are never read — only their shape binds an output dim —
# so one host buffer per distinct dim value is shared across dispatches.
_BROADCAST_CARRIERS: dict[int, Buffer] = {}


def _carrier(dim: int) -> Buffer:
    carrier = _BROADCAST_CARRIERS.get(dim)
    if carrier is None:
        carrier = Buffer.from_numpy(np.zeros(dim, dtype=np.uint8))
        _BROADCAST_CARRIERS[dim] = carrier
    return carrier


def broadcast_to(
    input_buffer: Buffer, target_shape: list[int], device: Device
) -> Buffer:
    """Broadcast via the fixed-rank model.

    Validates legality on the real dim alignment, pads both sides to
    ``MAX_RANK``, and passes the target shape as memoized uint8 carriers
    whose *shapes* bind the output dims (see
    :func:`_build_broadcast_graph`). Rank-0 target: plain copy.
    """
    rank = len(target_shape)
    if rank == 0:
        return _scalar_copy(input_buffer, device)
    pad = _pad_for(rank, "broadcast_to", "targets")
    padded = [1] * (rank - len(input_buffer.shape)) + list(input_buffer.shape)
    aligned_dims = zip(padded, target_shape, strict=True)
    for i, (in_dim, out_dim) in enumerate(aligned_dims):
        if in_dim != 1 and in_dim != out_dim:
            raise ValueError(
                f"Cannot broadcast dim {i} of size {in_dim} to size "
                f"{out_dim} (padded input shape {tuple(padded)}, target "
                f"shape {tuple(target_shape)})"
            )
    udtype = gc_compile.uint_view_dtype(input_buffer.dtype)
    gc_model = model(mo.BroadcastToOp, device, udtype)
    (out,) = gc_model(
        input_buffer.view(udtype, (1,) * pad + tuple(padded)),
        *(_carrier(dim) for dim in [1] * pad + list(target_shape)),
    )
    return out.view(input_buffer.dtype, tuple(target_shape))


def store_slice(
    in_buffer: Buffer,
    src: Buffer,
    starts: NDArray[np.int64],
    stops: NDArray[np.int64],
    steps: NDArray[np.int64],
) -> None:
    """Writes ``src`` into ``in_buffer``'s slice in place, via zero-copy
    uint views padded to ``MAX_RANK`` with leading full-extent unit axes.

    Sub-byte dtypes (fp4) raise NotImplementedError in uint_view_dtype,
    matching the deleted Mojo kernel's unsupported-dtype behavior.
    """
    udtype = gc_compile.uint_view_dtype(in_buffer.dtype)
    pad = _pad_for(len(in_buffer.shape), "store_slice", "buffers")
    lead = np.zeros(pad, dtype=np.int64)
    gc_model = model(mo.MutableStoreSliceOp, in_buffer.device, udtype)
    gc_model(
        in_buffer.view(udtype, (1,) * pad + tuple(in_buffer.shape)),
        src.view(udtype, (1,) * pad + tuple(src.shape)),
        Buffer.from_numpy(np.concatenate((lead, starts))),
        Buffer.from_numpy(np.concatenate((lead + 1, stops))),
        Buffer.from_numpy(np.concatenate((lead + 1, steps))),
    )


def _symbolic_dims(prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(MAX_RANK)]


def _build_transpose_graph(
    module: Module, device: Device, dtype: DType
) -> None:
    """rmo.MoTransposeOp with the permutation as a runtime [MAX_RANK] host
    operand: one graph serves every permutation. Handlers pad input and
    perm to MAX_RANK and re-view the output to the real shape."""
    device_ref = DeviceRef.from_device(device)
    graph = Graph(
        _graph_name(mo.TransposeOp, device, dtype),
        input_types=[
            TensorType(dtype, _symbolic_dims("d"), device=device_ref),
            TensorType(DType.int64, [MAX_RANK], device=DeviceRef.CPU()),
        ],
        module=module,
    )
    with graph:
        data, perm = (v.tensor for v in graph.inputs)
        out_type = TensorType(dtype, _symbolic_dims("o"), device=device_ref)
        out = Graph.current._add_op_generated(
            rmo.MoTransposeOp,
            result=out_type,
            input=data,
            perm=perm,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(out)


_register_spec(mo.TransposeOp, _build_transpose_graph)


def _build_broadcast_graph(
    module: Module, device: Device, dtype: DType
) -> None:
    """mo.static.broadcast_to with output dims bound by carrier inputs.

    A runtime shape *operand* can't work on GPU: ``mo.broadcast_to``
    grants it no host exemption yet reads its values host-side, and
    ``ops.broadcast_to`` can't prove legality for independent symbolic
    dims. So each output dim ``o{i}`` is bound from the shape of a tiny
    host uint8 carrier input ``[o{i}]``; legality stays the handler's
    job (it has the concrete shapes). Always MAX_RANK carriers; handlers
    pad both sides to MAX_RANK."""
    device_ref = DeviceRef.from_device(device)
    out_dims = _symbolic_dims("o")
    graph = Graph(
        _graph_name(mo.BroadcastToOp, device, dtype),
        input_types=[
            TensorType(dtype, _symbolic_dims("d"), device=device_ref),
            *(
                TensorType(DType.uint8, [dim], device=DeviceRef.CPU())
                for dim in out_dims
            ),
        ],
        module=module,
    )
    with graph:
        data = graph.inputs[0].tensor
        out_type = TensorType(dtype, out_dims, device=device_ref)
        out = Graph.current._add_op_generated(
            mo.StaticBroadcastToOp, out_type, data
        )[0].tensor
        graph.output(out)


_register_spec(mo.BroadcastToOp, _build_broadcast_graph)


def _build_store_slice_graph(
    module: Module, device: Device, dtype: DType
) -> None:
    """rmo.MoMutableStoreSliceOp writing in place into a BufferType input.

    Chain idiom copied from ``max.graph.ops.buffer_store_slice``
    (graph/ops/buffer.py), but with runtime starts/stops/steps operands
    instead of static slice indices. Handlers pad dst/src views and the
    slice bounds to MAX_RANK with leading full-extent unit axes. No tensor
    outputs: ``graph.output()`` packs the device chain.
    """
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    graph = Graph(
        _graph_name(mo.MutableStoreSliceOp, device, dtype),
        input_types=[
            BufferType(dtype, _symbolic_dims("d"), device_ref),
            TensorType(dtype, _symbolic_dims("s"), device=device_ref),
            TensorType(DType.int64, [MAX_RANK], device=cpu),
            TensorType(DType.int64, [MAX_RANK], device=cpu),
            TensorType(DType.int64, [MAX_RANK], device=cpu),
        ],
        module=module,
    )
    with graph:
        dst = graph.inputs[0].buffer
        src, starts, stops, steps = (v.tensor for v in graph.inputs[1:])
        in_chain = Graph.current.device_chains[dst.device]
        out_chain = Graph.current._add_op_generated(
            rmo.MoMutableStoreSliceOp,
            mo.ChainType(),
            dst,
            src,
            starts,
            stops,
            steps,
            kgen.ParamDeclArrayAttr([]),
            in_chain,
        )[-1]
        Graph.current.device_chains[dst.device] = out_chain
        graph.output()


_register_spec(mo.MutableStoreSliceOp, _build_store_slice_graph)


# Must stay last: every _register_spec call above feeds this snapshot.
_refresh_by_name()
DATA_MOVEMENT_GC_OPS = tuple(_MOVE_OPS)
