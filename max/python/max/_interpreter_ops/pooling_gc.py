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

"""Graph-compiler pooling model cache for the MO interpreter.

Covers ``MaxPoolOp``/``MaxPoolCeilModeTrueOp``/``AvgPoolOp``/
``AvgPoolCeilModeTrueOp``. ``filter_shape``/``strides``/``dilations``/
``paddings`` are runtime host int64 tensor operands on the MO-analogue ops
(``rmo.mo.max_pool``/``rmo.mo.avg_pool``/...), not compile-time attributes --
so ONE compiled graph per ``(op, device, dtype[, variant])`` serves every
window config, the same way ``matmul_gc`` serves every M/K/N.

``ceil_mode`` is inherent to which op is used (``MaxPoolOp`` vs
``MaxPoolCeilModeTrueOp``; ``AvgPoolOp`` vs ``AvgPoolCeilModeTrueOp``), not a
cache-key axis -- already how the eager handler dispatches. ``count_boundary``
(avg pool only) IS a small closed bool variant, baked into the cache key --
same shape as ``reduce_axis_gc``'s cumsum ``(exclusive, reverse)`` variant.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

The swept dtype set is ``gc_compile.float_dtypes(device)`` (the same CPU/GPU
float policy ``matmul_gc``/``unary_elementwise_gc``/``elementwise_binary_gc``/
``reduce_axis_gc`` already use -- no 16-bit floats on CPU, no ``float64`` on
GPU) plus every int/uint width and bool. This is narrower on CPU than the old
Mojo bindings' ``dispatch_dtype`` (which also compiled ``bfloat16`` there);
avg_pool additionally excludes bool (its kernel requires a numeric dtype). See
``_supported_dtypes``.
"""

from dataclasses import dataclass
from typing import TypeAlias

from max import _core, engine
from max._core.dialects import builtin, kgen, mo, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

# Variant: None (max_pool has no extra axis) or count_boundary (avg_pool).
Variant: TypeAlias = bool | None
_NO_VARIANT: Variant = None
_AVG_POOL_VARIANTS: tuple[Variant, ...] = (False, True)


@dataclass(frozen=True)
class PoolSpec:
    """How to build one pooling op's graph: its MO-analogue RMO op, whether
    it's an avg (vs max) pool, and its variants."""

    rmo_op: type
    is_avg: bool
    variants: tuple[Variant, ...] = (_NO_VARIANT,)


_POOL_OPS: dict[type[_core.Operation], PoolSpec] = {
    mo.MaxPoolOp: PoolSpec(rmo.MoMaxPoolOp, is_avg=False),
    mo.MaxPoolCeilModeTrueOp: PoolSpec(
        rmo.MoMaxPoolCeilModeTrueOp, is_avg=False
    ),
    mo.AvgPoolOp: PoolSpec(
        rmo.MoAvgPoolOp, is_avg=True, variants=_AVG_POOL_VARIANTS
    ),
    mo.AvgPoolCeilModeTrueOp: PoolSpec(
        rmo.MoAvgPoolCeilModeTrueOp, is_avg=True, variants=_AVG_POOL_VARIANTS
    ),
}

POOLING_GC_OPS = tuple(_POOL_OPS)

_POOL_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _POOL_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> PoolSpec | None:
    return gc_compile.spec_for(op_type, _POOL_OPS_BY_NAME)


def _supported_dtypes(device: Device) -> list[DType]:
    """The full dtype set max_pool sweeps on *device*: ``gc_compile``'s shared
    per-device float set (see the module docstring) plus every int/uint
    width and bool. avg_pool further excludes bool -- see ``_is_supported``.
    """
    return (
        gc_compile.float_dtypes(device)
        + gc_compile.SIGNED_INT_DTYPES
        + gc_compile.UNSIGNED_INT_DTYPES
        + [DType.bool]
    )


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    spec = _spec_for(op_type)
    if spec is None:
        return False
    if dtype not in _supported_dtypes(device):
        return False
    # avg_pool requires a numeric dtype.
    return not (spec.is_avg and dtype == DType.bool)


def _variant_tag(variant: Variant) -> str:
    if variant is None:
        return ""
    return f"_cb{int(variant)}"


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant = _NO_VARIANT,
) -> str:
    name = gc_compile.canonical_op_name(op_type, _POOL_OPS_BY_NAME)
    return (
        f"pool_{name}_{device.label}_{device.id}_{dtype.name}"
        f"{_variant_tag(variant)}"
    )


def _pool_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: PoolSpec,
    device: Device,
    dtype: DType,
    variant: Variant,
) -> None:
    """Adds one fully-symbolic NHWC pooling graph into *module* in-place.

    filter_shape/strides/dilations/paddings are runtime host int64 tensor
    operands -- the same buffers the eager handler already unpacks -- so this
    one graph serves every window config for (op_type, device, dtype)."""
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    in_type = TensorType(dtype, ["n", "h", "w", "c"], device=device_ref)
    filter_shape_t = TensorType(DType.int64, [2], device=cpu)
    strides_t = TensorType(DType.int64, [2], device=cpu)
    dilations_t = TensorType(DType.int64, [2], device=cpu)
    paddings_t = TensorType(DType.int64, [4], device=cpu)
    graph = Graph(
        _graph_name(op_type, device, dtype, variant),
        input_types=[
            in_type,
            filter_shape_t,
            strides_t,
            dilations_t,
            paddings_t,
        ],
        module=module,
    )
    with graph:
        x, filter_shape, strides, dilations, paddings = (
            v.tensor for v in graph.inputs
        )
        out_type = TensorType(
            dtype, ["n_out", "h_out", "w_out", "c"], device=device_ref
        )
        kwargs = dict(
            result=out_type,
            input=x,
            filter_shape=filter_shape,
            strides=strides,
            dilations=dilations,
            paddings=paddings,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )
        if spec.is_avg:
            assert variant is not None
            kwargs["count_boundary"] = builtin.BoolAttr(variant)
        result = Graph.current._add_op_generated(spec.rmo_op, **kwargs)[
            0
        ].tensor
        graph.output(result)


class _PoolingFamily(gc_compile.GCFamilySpec):
    name = "pooling"

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
        for op_type, spec in _POOL_OPS.items():
            for dtype in _supported_dtypes(device):
                if not _is_supported(op_type, device, dtype):
                    continue
                for variant in spec.variants:
                    _pool_graph(module, op_type, spec, device, dtype, variant)
        return module


_FAMILY = gc_compile.GCOpFamily(_PoolingFamily())
gc_compile.register_family(_FAMILY)


def pool_model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant = _NO_VARIANT,
) -> engine.Model:
    """Returns the pooling Model for the given target (lazy by default).

    Args:
        op_type: The concrete ``mo.*Pool*Op`` type being handled.
        device: The realized input's device.
        dtype: The realized input's dtype.
        variant: ``count_boundary`` bool for avg_pool ops, ``None`` for
            max_pool.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If (op, device, dtype) is outside the supported set.
        EagerLazyCompileDisallowed: If a supported target is not already
            compiled and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
    """
    key = _graph_name(op_type, device, dtype, variant)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, dtype):
            return None
        return (
            f"Unsupported pooling op/device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {_supported_dtypes(device)}"
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _pool_graph(module, op_type, spec, device, dtype, variant)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
