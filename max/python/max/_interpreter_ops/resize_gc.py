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

"""Graph-compiler resize model cache for the MO interpreter.

Covers ``ResizeLinearOp``/``ResizeNearestOp`` only (``ResizeBicubicOp`` is
intentionally excluded -- see below). ``size`` is a runtime rank-1 int64
tensor operand on the MO-analogue ops (``rmo.mo.resize.linear``/``.nearest``),
not a compile-time attribute -- so one compiled graph per ``(op, device,
dtype, rank, variant)`` serves every output size, the same way
``shape_rearrange_gc`` serves every pad width / repeat count / slice bound.

``coordinate_transform_mode`` (4 values) and ``antialias`` (linear only, 2
values) / ``round_mode`` (nearest only, 4 values) are small closed compile-time
attrs, baked into the cache key as a ``variant`` tuple -- same shape as
``reduce_axis_gc``'s cumsum ``(exclusive, reverse)`` variant.

``ResizeBicubicOp`` has no supported kernel -- see GEX-3990 (GraphCompiler
has no shape-fallback registration for ``MO::ResizeBicubicOp``, unlike
``MO::ResizeLinearOp``/``MO::ResizeNearestOp``).

Both remaining ops are CPU-only (``MO_HostOnly`` at the MLIR level).

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import TypeAlias

from max import _core, engine
from max._core.dialects import builtin, kgen, mo, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

MAX_RANK = gc_compile.MAX_RANK  # Sweep ranks 1..MAX_RANK for linear/nearest.

# Resize is CPU-only, so gc_compile.CPU_FLOAT_DTYPES *is* its dtype set --
# same policy as matmul_gc/unary_elementwise_gc/elementwise_binary_gc/
# reduce_axis_gc/pooling_gc: no 16-bit floats on CPU (bfloat16 and float16
# both fail to compile through the GC backend there; float16's failure is
# empirically confirmed -- "The f16 data type is not supported on device
# 'cpu:0'" -- bfloat16 is excluded for consistency with the same policy
# rather than an independent per-op finding).
_RESIZE_DTYPES = gc_compile.CPU_FLOAT_DTYPES

# Variant: (coord_mode, antialias) for linear; (coord_mode, round_mode) for
# nearest.
Variant: TypeAlias = tuple[int, ...]
_NO_VARIANT: Variant = ()

GraphBuilder: TypeAlias = Callable[[Module, Device, DType, int, Variant], None]


def _linear_variants() -> tuple[Variant, ...]:
    return tuple((cm, int(aa)) for cm in range(4) for aa in (False, True))


def _nearest_variants() -> tuple[Variant, ...]:
    return tuple((cm, rm) for cm in range(4) for rm in range(4))


@dataclass(frozen=True)
class ResizeSpec:
    """How to build one resize op's graph, its rank range, and its variants."""

    build: GraphBuilder
    max_rank: int
    rank_keyed: bool
    variants: tuple[Variant, ...] = (_NO_VARIANT,)


def _symbolic_dims(rank: int, prefix: str) -> list[str]:
    return [f"{prefix}{i}" for i in range(rank)]


def _build_linear_graph(
    module: Module, device: Device, dtype: DType, rank: int, variant: Variant
) -> None:
    coord_mode, antialias = variant
    device_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, _symbolic_dims(rank, "d"), device=device_ref)
    size_type = TensorType(DType.int64, [rank], device=DeviceRef.CPU())
    graph = Graph(
        _graph_name(mo.ResizeLinearOp, device, dtype, rank, variant),
        input_types=[in_type, size_type],
        module=module,
    )
    with graph:
        x, size = (v.tensor for v in graph.inputs)
        out_type = TensorType(
            dtype, _symbolic_dims(rank, "o"), device=device_ref
        )
        result = Graph.current._add_op_generated(
            rmo.MoResizeLinearOp,
            result=out_type,
            input=x,
            size=size,
            coordinate_transform_mode=mo.CoordinateTransformModeAttr(
                mo.CoordinateTransformMode(coord_mode)
            ),
            antialias=builtin.BoolAttr(bool(antialias)),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


def _build_nearest_graph(
    module: Module, device: Device, dtype: DType, rank: int, variant: Variant
) -> None:
    coord_mode, round_mode = variant
    device_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, _symbolic_dims(rank, "d"), device=device_ref)
    size_type = TensorType(DType.int64, [rank], device=DeviceRef.CPU())
    graph = Graph(
        _graph_name(mo.ResizeNearestOp, device, dtype, rank, variant),
        input_types=[in_type, size_type],
        module=module,
    )
    with graph:
        x, size = (v.tensor for v in graph.inputs)
        out_type = TensorType(
            dtype, _symbolic_dims(rank, "o"), device=device_ref
        )
        result = Graph.current._add_op_generated(
            rmo.MoResizeNearestOp,
            result=out_type,
            input=x,
            size=size,
            coordinate_transform_mode=mo.CoordinateTransformModeAttr(
                mo.CoordinateTransformMode(coord_mode)
            ),
            round_mode=builtin.IntegerAttr(builtin.IntegerType(64), round_mode),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


_RESIZE_OPS: dict[type[_core.Operation], ResizeSpec] = {
    mo.ResizeLinearOp: ResizeSpec(
        build=_build_linear_graph,
        max_rank=MAX_RANK,
        rank_keyed=True,
        variants=_linear_variants(),
    ),
    mo.ResizeNearestOp: ResizeSpec(
        build=_build_nearest_graph,
        max_rank=MAX_RANK,
        rank_keyed=True,
        variants=_nearest_variants(),
    ),
}

RESIZE_GC_OPS = tuple(_RESIZE_OPS)

_RESIZE_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _RESIZE_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> ResizeSpec | None:
    return gc_compile.spec_for(op_type, _RESIZE_OPS_BY_NAME)


# Resize is CPU-only for both remaining ops (see module docstring).
_CPU_DEVICES = [d for d in gc_compile.DISCOVERED_DEVICES if d.label == "cpu"]


def _ranks_for(spec: ResizeSpec) -> list[int]:
    if not spec.rank_keyed:
        return [spec.max_rank]
    return list(range(1, spec.max_rank + 1))


def _variant_tag(variant: Variant) -> str:
    if not variant:
        return ""
    return "_v" + "_".join(str(v) for v in variant)


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int,
    variant: Variant = _NO_VARIANT,
) -> str:
    name = gc_compile.canonical_op_name(op_type, _RESIZE_OPS_BY_NAME)
    return (
        f"resize_{name}_{device.label}_{device.id}_{dtype.name}_r{rank}"
        f"{_variant_tag(variant)}"
    )


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType, rank: int
) -> bool:
    spec = _spec_for(op_type)
    if spec is None:
        return False
    if device not in _CPU_DEVICES:
        return False
    if dtype not in _RESIZE_DTYPES:
        return False
    return rank in _ranks_for(spec)


class _ResizeFamily(gc_compile.GCFamilySpec):
    name = "resize"

    def sweep_devices(self) -> list[Device]:
        return list(_CPU_DEVICES)

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
        if device not in _CPU_DEVICES:
            return module
        for spec in _RESIZE_OPS.values():
            for dtype in _RESIZE_DTYPES:
                for rank in _ranks_for(spec):
                    for variant in spec.variants:
                        spec.build(module, device, dtype, rank, variant)
        return module


_FAMILY = gc_compile.GCOpFamily(_ResizeFamily())
gc_compile.register_family(_FAMILY)


def resize_model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    rank: int,
    variant: Variant = _NO_VARIANT,
) -> engine.Model:
    """Returns the resize Model for the given target (lazy by default).

    Args:
        op_type: The concrete ``mo.Resize*Op`` type being handled.
        device: The realized input's device.
        dtype: The realized input's dtype.
        rank: The input's rank.
        variant: ``(coord_mode, antialias)`` for linear, ``(coord_mode,
            round_mode)`` for nearest.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the target is outside the supported set (non-CPU device,
            unsupported dtype, or rank > 5).
        EagerLazyCompileDisallowed: If a supported target is not already
            compiled and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
    """
    key = _graph_name(op_type, device, dtype, rank, variant)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, dtype, rank):
            return None
        return (
            f"Unsupported resize op/device/dtype/rank for key {key!r}."
            f" Resize is CPU-only; supported dtypes: {_RESIZE_DTYPES};"
            f" max rank: {MAX_RANK}."
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        spec.build(module, device, dtype, rank, variant)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
