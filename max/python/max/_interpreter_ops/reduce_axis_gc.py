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

"""Graph-compiler reduce-along-axis model cache for the MO interpreter.

Covers the reduce-along-axis family: the reductions (``ReduceMax``/``Min``/
``Add``/``Mul``/``Mean``), ``Softmax``/``Logsoftmax``, ``ArgMax``/``ArgMin``
(emit ``int64`` indices), and ``Cumsum``.

Every op applies one operation along a single ``axis``, which is a compile-time
attribute, not an operand. To keep the cache key ``(op, device, dtype)`` without
``axis`` in it, the handler canonicalizes any input to rank 3
``[outer, axis, inner]`` (a zero-copy view; see :func:`canonical_rank3`) and each
graph applies the op at ``axis=1``. The handler reads the op's MLIR result type
for the final shape and dtype, so reduced-axis ops (``[d0, 1, d2]``, ``int64``
for argmax) and same-shape ops (``[d0, d1, d2]``) share one handler.

``Cumsum`` additionally carries compile-time ``exclusive``/``reverse`` flags.
Decomposing them host-side would force device<->host copies on GPU buffers (and
baking them into the graph just produces variants anyway), so the four
``(exclusive, reverse)`` combinations are baked into the cache key as separate
graph variants. ``axis`` is still canonicalized out.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-3 graph.
- **Precompile sweep (``=1``).** The batched sweep compiles the full matrix at
  import; a :func:`reduce_model` miss is then a hard error.

Models serve the eager handler via :func:`reduce_model`. Must not import from
``handlers.py``.

The swept dtype set is deliberately conservative (the IR type category is only a
ceiling): softmax/logsoftmax sweep floats only; the reductions, argmax/argmin,
and cumsum sweep floats + ints; ``ReduceMax``/``ReduceMin`` additionally sweep
``bool`` (the logical-OR/AND reductions, issue #6067); cumsum excludes ``bool``.
CPU floats are f32/f64 (no 16-bit); GPU floats are f16/f32/bf16 (no f64).
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias, cast

from max import _core, engine
from max._core.dialects import mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops

# CUDA's reduction kernels (the reduce/argmax family) only support 32- and
# 64-bit integer reduction: 8/16-bit int reduce fails to compile on the B200
# backend ("Failed to compile the model"). So on accelerators the reduction
# family is narrowed to the wide ints. Cumsum is exempt — on GPU it transfers to
# CPU (KERN-1095), so it keeps the full int set; CPU supports every width.
_WIDE_INT_DTYPES = [DType.int32, DType.int64, DType.uint32, DType.uint64]

# Cumsum's (exclusive, reverse) flags ride in the cache key as a variant; every
# other op uses the empty variant.
Variant: TypeAlias = tuple[bool, ...]
_NO_VARIANT: Variant = ()
_CUMSUM_VARIANTS: tuple[Variant, ...] = (
    (False, False),
    (False, True),
    (True, False),
    (True, True),
)

# Builds the rank-3 graph body; the op is applied at axis=1 of [d0, d1, d2].
ReduceBuilder: TypeAlias = Callable[[TensorValue, Variant], TensorValue]


class DTypeClass(Enum):
    """The input-dtype set an op is swept over (see ``_supported_dtypes``)."""

    FLOAT = "float"
    NUMERIC = "numeric"
    NUMERIC_BOOL = "numeric_bool"
    CUMSUM = "cumsum"


@dataclass(frozen=True)
class ReduceSpec:
    """How to build one op's rank-3 graph, its dtype class, and its variants."""

    build: ReduceBuilder
    dtype_class: DTypeClass
    variants: tuple[Variant, ...] = (_NO_VARIANT,)


def _b_max(x: TensorValue, v: Variant) -> TensorValue:
    return ops.max(x, axis=1)


def _b_min(x: TensorValue, v: Variant) -> TensorValue:
    return ops.min(x, axis=1)


def _b_sum(x: TensorValue, v: Variant) -> TensorValue:
    return ops.sum(x, axis=1)


def _b_prod(x: TensorValue, v: Variant) -> TensorValue:
    return ops.prod(x, axis=1)


def _b_mean(x: TensorValue, v: Variant) -> TensorValue:
    return ops.mean(x, axis=1)


# The backend only reduces the *innermost* axis for these four ("axis other than
# innermost/-1 not supported"), but the canonical form reduces axis=1, so they
# transpose d1 to innermost, reduce at axis=-1, and transpose back (folded into
# the graph). argmax/argmin break ties to the lowest index on CPU but an
# arbitrary index on GPU (the graph compiler's documented contract).


def _b_softmax(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.softmax(xt, axis=-1), 1, 2)


def _b_logsoftmax(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.logsoftmax(xt, axis=-1), 1, 2)


def _b_argmax(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.argmax(xt, axis=-1), 1, 2)


def _b_argmin(x: TensorValue, v: Variant) -> TensorValue:
    xt = ops.transpose(x, 1, 2)
    return ops.transpose(ops.argmin(xt, axis=-1), 1, 2)


def _b_cumsum(x: TensorValue, v: Variant) -> TensorValue:
    return ops.cumsum(x, axis=1, exclusive=v[0], reverse=v[1])


_REDUCE_OPS: dict[type[_core.Operation], ReduceSpec] = {
    mo.ReduceMaxOp: ReduceSpec(_b_max, DTypeClass.NUMERIC_BOOL),
    mo.ReduceMinOp: ReduceSpec(_b_min, DTypeClass.NUMERIC_BOOL),
    mo.ReduceAddOp: ReduceSpec(_b_sum, DTypeClass.NUMERIC),
    mo.ReduceMulOp: ReduceSpec(_b_prod, DTypeClass.NUMERIC),
    mo.ReduceMeanOp: ReduceSpec(_b_mean, DTypeClass.NUMERIC),
    mo.ReduceSoftmaxOp: ReduceSpec(_b_softmax, DTypeClass.FLOAT),
    mo.ReduceLogsoftmaxOp: ReduceSpec(_b_logsoftmax, DTypeClass.FLOAT),
    mo.ReduceArgMaxOp: ReduceSpec(_b_argmax, DTypeClass.NUMERIC),
    mo.ReduceArgMinOp: ReduceSpec(_b_argmin, DTypeClass.NUMERIC),
    mo.CumsumOp: ReduceSpec(
        _b_cumsum, DTypeClass.CUMSUM, variants=_CUMSUM_VARIANTS
    ),
}

REDUCE_AXIS_GC_OPS = tuple(_REDUCE_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_REDUCE_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _REDUCE_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> ReduceSpec | None:
    return gc_compile.spec_for(op_type, _REDUCE_OPS_BY_NAME)


def _reduce_int_dtypes(device: Device) -> list[DType]:
    """Integer dtypes the reduction/argmax family supports on *device*.

    CPU handles every width; accelerators are narrowed to 32/64-bit (CUDA's
    reduce kernels don't compile 8/16-bit int reduction — see ``_WIDE_INT_DTYPES``).
    """
    if device.label == "cpu":
        return gc_compile.SIGNED_INT_DTYPES + gc_compile.UNSIGNED_INT_DTYPES
    return _WIDE_INT_DTYPES


def _supported_dtypes(dtype_class: DTypeClass, device: Device) -> list[DType]:
    """Conservative swept dtype set for a (dtype_class, device)."""
    if dtype_class is DTypeClass.FLOAT:
        return gc_compile.float_dtypes(device)
    if dtype_class is DTypeClass.CUMSUM:
        # Cumsum runs on CPU even for a GPU graph (KERN-1095), so it keeps every
        # int width on all devices.
        return (
            gc_compile.float_dtypes(device)
            + gc_compile.SIGNED_INT_DTYPES
            + gc_compile.UNSIGNED_INT_DTYPES
        )
    numeric = gc_compile.float_dtypes(device) + _reduce_int_dtypes(device)
    if dtype_class is DTypeClass.NUMERIC:
        return numeric
    if dtype_class is DTypeClass.NUMERIC_BOOL:
        # bool max/min are the logical any/all reductions (#6067); swept on GPU
        # too, matching the bool-on-GPU binary logical ops (And/Or/Xor).
        return numeric + [DType.bool]
    raise ValueError(f"Unknown dtype_class: {dtype_class!r}")


def _variant_tag(variant: Variant) -> str:
    """Cache-key suffix for a variant; empty for the no-variant default."""
    if not variant:
        return ""
    exclusive, reverse = variant
    return f"_e{int(exclusive)}r{int(reverse)}"


def _graph_name(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant = _NO_VARIANT,
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype, variant)."""
    name = gc_compile.canonical_op_name(op_type, _REDUCE_OPS_BY_NAME)
    return (
        f"reduce_{name}_{device.label}_{device.id}_{dtype.name}"
        f"{_variant_tag(variant)}"
    )


canonical_rank3 = gc_compile.canonical_rank3


def variant_for(op: _core.Operation) -> Variant:
    """The cache-key variant for *op*: (exclusive, reverse) for cumsum, else ().

    Matches by canonical name (not ``isinstance``) so an ``rmo.MoCumsumOp``
    dispatch carries its variant too, not just ``mo.CumsumOp``.
    """
    if (
        gc_compile.canonical_op_name(type(op), _REDUCE_OPS_BY_NAME)
        == mo.CumsumOp.__name__
    ):
        cumsum = cast(mo.CumsumOp, op)
        return (bool(cumsum.exclusive), bool(cumsum.reverse))
    return _NO_VARIANT


def _reduce_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: ReduceSpec,
    device: Device,
    dtype: DType,
    variant: Variant,
) -> None:
    """Adds one fully-symbolic rank-3 reduce graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, ["d0", "d1", "d2"], device=device_ref)
    graph = Graph(
        _graph_name(op_type, device, dtype, variant),
        input_types=[in_type],
        module=module,
    )
    with graph:
        (x,) = graph.inputs
        graph.output(spec.build(x.tensor, variant))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    """Whether (op, device, dtype) is in the conservatively-supported set.

    Single source of truth for the swept matrix:
    :meth:`_ReduceAxisFamily.build_module_for_device` filters candidates
    through this predicate and lazy mode uses it as the support guard in
    :func:`reduce_model`, so the two can't diverge. Variant does not affect
    dtype support, so it is not an argument here.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    return dtype in _supported_dtypes(spec.dtype_class, device)


class _ReduceAxisFamily(gc_compile.GCFamilySpec):
    name = "reduce_axis"

    def build_module(self) -> Module:
        """Batched module: every supported (op, device, dtype, variant), all
        devices."""
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Per-slot counterpart of :meth:`build_module`: one *device*
        only."""
        if module is None:
            module = Module()
        for op_type, spec in _REDUCE_OPS.items():
            for dtype in _supported_dtypes(spec.dtype_class, device):
                if not _is_supported(op_type, device, dtype):
                    continue
                for variant in spec.variants:
                    _reduce_graph(module, op_type, spec, device, dtype, variant)
        return module


_FAMILY = gc_compile.GCOpFamily(_ReduceAxisFamily())
gc_compile.register_family(_FAMILY)


def reduce_model(
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
    variant: Variant = _NO_VARIANT,
) -> engine.Model:
    """Returns the reduce :class:`~max.engine.Model` for the given target.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss, a warm cache is adopted whole (manifest
    force-load, else a batched stamp sweep) instead of compiling per target.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled.
        device: The realized input's device.
        dtype: The realized input's dtype.
        variant: The cumsum ``(exclusive, reverse)`` pair, or ``()`` otherwise.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set.
        EagerLazyCompileDisallowed: If a supported target is not already
            compiled and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
    """
    key = _graph_name(op_type, device, dtype, variant)
    # Cache-check before building the closures below: this runs on every
    # eager op dispatch, so a hit must not pay for closures it won't use.
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, dtype):
            return None
        spec = _spec_for(op_type)
        supported = _supported_dtypes(spec.dtype_class, device) if spec else []
        return (
            f"Unsupported reduce op/device/dtype for key {key!r}."
            f"  Supported dtypes for this op/device: {supported}"
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _reduce_graph(module, op_type, spec, device, dtype, variant)

    return _FAMILY.model_for(
        key,
        device,
        build,
        unsupported_reason=check_supported,
    )
