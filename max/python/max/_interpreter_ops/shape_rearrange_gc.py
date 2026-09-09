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

"""Graph-compiler shape-rearrange model cache for the MO interpreter.

Covers the shape-rearrange family: ``Concat``, ``Split``, ``Slice``, the pads
(``PadConstant``/``PadReflect``/``PadRepeat``), and ``Tile``. Each runs one
graph node that only rearranges/copies data (no arithmetic, no weights).

Structural parameters are runtime, not baked: pad widths, repeat counts, and
slice/split bounds arrive as host tensor operands, so one compiled graph serves
every value of them (the output shape is data-dependent). The handler always
knows the concrete output shape, so it re-views the model output to that shape
(zero-copy).

Two keying schemes:

- ``Concat``/``Split`` canonicalize to rank 3 ``[outer, axis, inner]`` (a
  zero-copy view; see :func:`canonical_rank3`) and key on ``(op, device,
  dtype)``. Concat folds its variadic operand count with a pairwise 2-input
  graph; split slices one chunk at a time.
- ``Pad``/``Tile``/``Slice`` keep their natural N-D form and key on
  ``(op, device, dtype, rank)``: their structural operand (paddings ``[2*rank]``,
  repeats ``[rank]``, starts/stops/steps ``[rank]``) needs a static length tied
  to rank, so rank rides in the key (bounded by ``MAX_RANK``).

Compile modes mirror :mod:`reduce_axis_gc` exactly (lazy-per-target by default,
batched sweep under ``MAX_EAGER_OP_PRECOMPILE=1``, warm-stamp adoption). Must
not import from ``handlers.py``.

These are pure data-movement ops (no arithmetic), so a tensor is copied by bit
width: the handler bit-casts every dtype to the same-width unsigned int (a
zero-copy view), runs the copy, and views the result back. One graph per width
(uint8/16/32/64) therefore serves every dtype uniformly — including float16
(no typed CPU kernel) and bool on GPU — and every width compiles on every
device. See :func:`gc_compile.uint_view_dtype`.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from functools import partial

from max import _core, engine
from max._core.dialects import kgen, mo, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, ops

MAX_RANK = gc_compile.MAX_RANK  # Sweep ranks 1..MAX_RANK.

_WIDTH_DTYPES = gc_compile.WIDTH_DTYPES


class DeviceClass(Enum):
    """Which devices an op supports."""

    ALL = "all"  # CPU + accelerators
    CPU_ONLY = "cpu"  # MO_HostOnly ops


@dataclass(frozen=True)
class RearrangeSpec:
    """How to build one op's graph, its devices, and rank-keying."""

    build_graph: Callable[["Module", str, Device, DType, "int | None"], None]
    devices: DeviceClass
    rank_keyed: bool
    max_rank: int | None = None  # Cap for rank sweep; None uses MAX_RANK.


_REARRANGE_OPS: dict[type[_core.Operation], RearrangeSpec] = {}


def _register_spec(op_type: type[_core.Operation], spec: RearrangeSpec) -> None:
    _REARRANGE_OPS[op_type] = spec


_REARRANGE_OPS_BY_NAME: dict[str, RearrangeSpec] = {}


def _refresh_by_name() -> None:
    _REARRANGE_OPS_BY_NAME.clear()
    _REARRANGE_OPS_BY_NAME.update(
        {op_type.__name__: spec for op_type, spec in _REARRANGE_OPS.items()}
    )


def _spec_for(op_type: type[_core.Operation]) -> RearrangeSpec | None:
    return gc_compile.spec_for(op_type, _REARRANGE_OPS_BY_NAME)


# Every family sweeps the same discovered set; see gc_compile.DISCOVERED_DEVICES.
_DEVICES = gc_compile.DISCOVERED_DEVICES


def _devices_for(spec: RearrangeSpec) -> list[Device]:
    if spec.devices is DeviceClass.CPU_ONLY:
        return [d for d in _DEVICES if d.label == "cpu"]
    return list(_DEVICES)


def _supported_dtypes(spec: RearrangeSpec, device: Device) -> list[DType]:
    return list(_WIDTH_DTYPES)


canonical_rank3 = gc_compile.canonical_rank3


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int | None,
) -> str:
    """Graph ``sym_name`` and cache key for one target."""
    name = gc_compile.canonical_op_name(op_type, _REARRANGE_OPS_BY_NAME)
    rank_tag = f"_r{rank}" if rank is not None else ""
    return f"rearrange_{name}_{device.label}_{device.id}_{dtype.name}{rank_tag}"


def _ranks_for(spec: RearrangeSpec) -> list[int | None]:
    if not spec.rank_keyed:
        return [None]
    top = spec.max_rank if spec.max_rank is not None else MAX_RANK
    return list(range(1, top + 1))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    spec = _spec_for(op_type)
    if spec is None:
        return False
    if device not in _devices_for(spec):
        return False
    return dtype in _supported_dtypes(spec, device)


class _ShapeRearrangeFamily(gc_compile.GCFamilySpec):
    name = "shape_rearrange"

    def build_module(self) -> Module:
        """Batched module: every supported (op, device, dtype, rank), all
        devices."""
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Per-slot counterpart of :meth:`build_module`: one *device* only.
        A ``CPU_ONLY`` op (e.g. ``Tile``) is skipped below via
        :func:`_devices_for`."""
        if module is None:
            module = Module()
        for op_type, spec in _REARRANGE_OPS.items():
            if device not in _devices_for(spec):
                continue
            for dtype in _supported_dtypes(spec, device):
                if not _is_supported(op_type, device, dtype):
                    continue
                for rank in _ranks_for(spec):
                    spec.build_graph(
                        module, op_type.__name__, device, dtype, rank
                    )
        return module


_FAMILY = gc_compile.GCOpFamily(_ShapeRearrangeFamily())
gc_compile.register_family(_FAMILY)


def model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int | None = None,
) -> engine.Model:
    """Return the compiled model for the given target (lazy by default)."""
    key = _graph_name(op_type, device, dtype, rank)
    # Cache-check before building the closures below: this runs on every
    # eager op dispatch, so a hit must not pay for closures it won't use.
    cached = _FAMILY.cache.get(key)
    if cached is not None:
        return cached

    def check_supported() -> str | None:
        if not _is_supported(op_type, device, dtype):
            return (
                f"Unsupported shape-rearrange op/device/dtype for key {key!r}."
            )
        spec = _spec_for(op_type)
        # Same unsupported-target signal as the dtype check above: a clean
        # error rather than the GC kernel's comptime rank assert firing deep
        # in compilation (e.g. tile is capped at rank 4).
        if (
            spec is not None
            and spec.rank_keyed
            and rank not in _ranks_for(spec)
        ):
            return (
                f"Unsupported shape-rearrange rank {rank} for"
                f" {op_type.__name__} (max {_ranks_for(spec)[-1]});"
                f" key {key!r}."
            )
        return None

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        spec.build_graph(module, op_type.__name__, device, dtype, rank)

    return _FAMILY.model_for(
        key,
        device,
        build,
        unsupported_reason=check_supported,
    )


def _build_concat_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """Pairwise 2-input concat at axis=1 of rank-3 [outer, a/b, inner]."""
    device_ref = DeviceRef.from_device(device)
    a = TensorType(dtype, ["d0", "a1", "d2"], device=device_ref)
    b = TensorType(dtype, ["d0", "b1", "d2"], device=device_ref)
    graph = Graph(
        _graph_name(mo.ConcatOp, device, dtype, rank),
        input_types=[a, b],
        module=module,
    )
    with graph:
        x, y = (v.tensor for v in graph.inputs)
        graph.output(ops.concat([x, y], axis=1))


_register_spec(
    mo.ConcatOp,
    RearrangeSpec(
        build_graph=_build_concat_graph,
        devices=DeviceClass.ALL,
        rank_keyed=False,
    ),
)


def _build_split_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """Slice one chunk along axis=1 of rank-3 [outer, D, inner]."""
    device_ref = DeviceRef.from_device(device)
    x = TensorType(dtype, ["d0", "d1", "d2"], device=device_ref)
    start = TensorType(DType.int64, [], device=DeviceRef.CPU())
    stop = TensorType(DType.int64, [], device=DeviceRef.CPU())
    graph = Graph(
        _graph_name(mo.SplitOp, device, dtype, rank),
        input_types=[x, start, stop],
        module=module,
    )
    with graph:
        data, lo, hi = (v.tensor for v in graph.inputs)
        chunk = ops.slice_tensor(
            data,
            [slice(None), (slice(lo, hi), "s"), slice(None)],
        )
        graph.output(chunk)


_register_spec(
    mo.SplitOp,
    RearrangeSpec(
        build_graph=_build_split_graph,
        devices=DeviceClass.ALL,
        rank_keyed=False,
    ),
)


def _symbolic_dims(rank: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(rank)]


def _build_tile_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """rmo.MoTileOp with repeats as a runtime [rank] host operand."""
    assert rank is not None
    device_ref = DeviceRef.from_device(device)
    x = TensorType(dtype, _symbolic_dims(rank, "d"), device=device_ref)
    repeats = TensorType(DType.int64, [rank], device=DeviceRef.CPU())
    graph = Graph(
        _graph_name(mo.TileOp, device, dtype, rank),
        input_types=[x, repeats],
        module=module,
    )
    with graph:
        data, reps = (v.tensor for v in graph.inputs)
        out_type = TensorType(
            dtype, _symbolic_dims(rank, "o"), device=device_ref
        )
        tiled = Graph.current._add_op_generated(
            rmo.MoTileOp, out_type, data, reps, kgen.ParamDeclArrayAttr([])
        )[0].tensor
        graph.output(tiled)


_register_spec(
    mo.TileOp,
    RearrangeSpec(
        build_graph=_build_tile_graph,
        devices=DeviceClass.CPU_ONLY,
        rank_keyed=True,
        max_rank=4,  # GC tile kernel supports up to rank 4.
    ),
)


def _build_pad_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
    *,
    mo_cls: type,
    rmo_cls: type,
    has_constant: bool,
) -> None:
    """Emit rmo.MoPad*Op with runtime paddings [2*rank] (+ rank-0 constant)."""
    assert rank is not None
    device_ref = DeviceRef.from_device(device)
    input_types = [
        TensorType(dtype, _symbolic_dims(rank, "d"), device=device_ref),
        TensorType(DType.int64, [2 * rank], device=DeviceRef.CPU()),
    ]
    if has_constant:
        input_types.append(TensorType(dtype, [], device=DeviceRef.CPU()))
    graph = Graph(
        _graph_name(mo_cls, device, dtype, rank),
        input_types=input_types,
        module=module,
    )
    with graph:
        ins = [v.tensor for v in graph.inputs]
        out_type = TensorType(
            dtype, _symbolic_dims(rank, "o"), device=device_ref
        )
        kwargs = dict(
            result=out_type,
            input=ins[0],
            paddings=ins[1],
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )
        if has_constant:
            kwargs["constant"] = ins[2]
        graph.output(
            Graph.current._add_op_generated(rmo_cls, **kwargs)[0].tensor
        )


def _pad_spec(
    mo_cls: type[_core.Operation],
    rmo_cls: type,
    has_constant: bool,
    devices: DeviceClass,
) -> RearrangeSpec:
    return RearrangeSpec(
        build_graph=partial(
            _build_pad_graph,
            mo_cls=mo_cls,
            rmo_cls=rmo_cls,
            has_constant=has_constant,
        ),
        devices=devices,
        rank_keyed=True,
    )


_register_spec(
    mo.PadConstantOp,
    _pad_spec(mo.PadConstantOp, rmo.MoPadConstantOp, True, DeviceClass.ALL),
)
_register_spec(
    mo.PadReflectOp,
    _pad_spec(mo.PadReflectOp, rmo.MoPadReflectOp, False, DeviceClass.CPU_ONLY),
)
_register_spec(
    mo.PadRepeatOp,
    _pad_spec(mo.PadRepeatOp, rmo.MoPadRepeatOp, False, DeviceClass.CPU_ONLY),
)


def _build_slice_graph(
    module: Module,
    op_name: str,
    device: Device,
    dtype: DType,
    rank: int | None,
) -> None:
    """rmo.MoSliceOp with runtime starts/stops/steps [rank]."""
    assert rank is not None
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    graph = Graph(
        _graph_name(mo.SliceOp, device, dtype, rank),
        input_types=[
            TensorType(dtype, _symbolic_dims(rank, "d"), device=device_ref),
            TensorType(DType.int64, [rank], device=cpu),
            TensorType(DType.int64, [rank], device=cpu),
            TensorType(DType.int64, [rank], device=cpu),
        ],
        module=module,
    )
    with graph:
        data, starts, stops, steps = (v.tensor for v in graph.inputs)
        out_type = TensorType(
            dtype, _symbolic_dims(rank, "o"), device=device_ref
        )
        sliced = Graph.current._add_op_generated(
            rmo.MoSliceOp,
            result=out_type,
            input=data,
            start=starts,
            stop=stops,
            step=steps,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(sliced)


_register_spec(
    mo.SliceOp,
    RearrangeSpec(
        build_graph=_build_slice_graph,
        devices=DeviceClass.ALL,
        rank_keyed=True,
    ),
)


# Must stay last: every _register_spec call above feeds this snapshot.
_refresh_by_name()
SHAPE_REARRANGE_GC_OPS = tuple(_REARRANGE_OPS)
