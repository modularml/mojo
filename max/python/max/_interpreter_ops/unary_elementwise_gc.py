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

"""Graph-compiler unary-elementwise model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-1 graph.
- **Precompile sweep (``=1``).** The batched sweep compiles the full matrix at
  import; a :func:`unary_model` miss is then a hard error.

Lazy mode avoids a trivial program JIT-compiling the whole kernel library on a
cold cache (~3000+ kernels, minutes; MXF-508). Models serve the eager handler
via :func:`unary_model`. Must not import from ``handlers.py``.

The swept dtype set is deliberately conservative (floats-first): the IR type
category is only a ceiling, so transcendental/activation ops are swept on float
dtypes only, ``Abs``/``Negative`` additionally get integer dtypes, and ``Not``
gets ``bool``. CPU floats are f32/f64 (no 16-bit); GPU floats are f16/f32/bf16
(no f64). ``dtype_class`` keys the *input*; ``IsNan``/``IsInf`` take a float
input and emit a constant ``bool``.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TypeAlias

from max import _core, engine
from max._core.dialects import mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue, ops

# Builds an op's graph body from its input tensor (e.g. ``ops.sqrt``).
MoOpBuilder: TypeAlias = Callable[[TensorValue], TensorValue]


class DTypeClass(Enum):
    """The input-dtype set an op is swept over (see ``_supported_dtypes``)."""

    FLOAT = "float"
    ABS = "abs"
    NEGATIVE = "negative"
    BOOL = "bool"


@dataclass(frozen=True)
class UnarySpec:
    """How to build one unary op's graph and which dtype class it sweeps."""

    builder: MoOpBuilder
    dtype_class: DTypeClass


def _gelu_none(x: TensorValue) -> TensorValue:
    return ops.gelu(x, approximate="none")


def _gelu_tanh(x: TensorValue) -> TensorValue:
    return ops.gelu(x, approximate="tanh")


def _gelu_quick(x: TensorValue) -> TensorValue:
    return ops.gelu(x, approximate="quick")


# The builder is a callable (an op or a named helper), so the Gelu variants and
# the plain one-op wrappers share one registry shape.
_UNARY_OPS: dict[type[_core.Operation], UnarySpec] = {
    mo.NegativeOp: UnarySpec(ops.negate, DTypeClass.NEGATIVE),
    mo.AbsOp: UnarySpec(ops.abs, DTypeClass.ABS),
    mo.CeilOp: UnarySpec(ops.ceil, DTypeClass.FLOAT),
    mo.FloorOp: UnarySpec(ops.floor, DTypeClass.FLOAT),
    mo.RoundOp: UnarySpec(ops.round, DTypeClass.FLOAT),
    mo.ExpOp: UnarySpec(ops.exp, DTypeClass.FLOAT),
    mo.LogOp: UnarySpec(ops.log, DTypeClass.FLOAT),
    mo.Log1pOp: UnarySpec(ops.log1p, DTypeClass.FLOAT),
    mo.SqrtOp: UnarySpec(ops.sqrt, DTypeClass.FLOAT),
    mo.RsqrtOp: UnarySpec(ops.rsqrt, DTypeClass.FLOAT),
    mo.TanhOp: UnarySpec(ops.tanh, DTypeClass.FLOAT),
    mo.AtanhOp: UnarySpec(ops.atanh, DTypeClass.FLOAT),
    mo.TruncOp: UnarySpec(ops.trunc, DTypeClass.FLOAT),
    mo.SinOp: UnarySpec(ops.sin, DTypeClass.FLOAT),
    mo.CosOp: UnarySpec(ops.cos, DTypeClass.FLOAT),
    mo.ErfOp: UnarySpec(ops.erf, DTypeClass.FLOAT),
    mo.SigmoidOp: UnarySpec(ops.sigmoid, DTypeClass.FLOAT),
    mo.SiluOp: UnarySpec(ops.silu, DTypeClass.FLOAT),
    mo.GeluOp: UnarySpec(_gelu_none, DTypeClass.FLOAT),
    mo.GeluTanhOp: UnarySpec(_gelu_tanh, DTypeClass.FLOAT),
    mo.GeluQuickOp: UnarySpec(_gelu_quick, DTypeClass.FLOAT),
    mo.NotOp: UnarySpec(ops.logical_not, DTypeClass.BOOL),
    # Predicates: float input, constant bool output; dtype_class keys input.
    mo.IsNanOp: UnarySpec(ops.is_nan, DTypeClass.FLOAT),
    mo.IsInfOp: UnarySpec(ops.is_inf, DTypeClass.FLOAT),
}

UNARY_GC_OPS = tuple(_UNARY_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_UNARY_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _UNARY_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> UnarySpec | None:
    return gc_compile.spec_for(op_type, _UNARY_OPS_BY_NAME)


# These lower to libm calls the GC backend only supports on CPU ("libm
# operations are only available on CPU targets") — verified failing on both
# Metal and CUDA (B200). Swept on CPU only, matching the historical interpreter
# binding's GPU allowlist, which excluded exactly these four.
_CPU_ONLY_OPS = frozenset({mo.Log1pOp, mo.AtanhOp, mo.ErfOp, mo.GeluOp})
# Keyed by name so the CPU-only guard also fires for an rmo dispatch.
_CPU_ONLY_NAMES = frozenset(op_type.__name__ for op_type in _CPU_ONLY_OPS)


def _supported_dtypes(dtype_class: DTypeClass, device: Device) -> list[DType]:
    """Conservative swept dtype set for a (dtype_class, device)."""
    if dtype_class is DTypeClass.FLOAT:
        return gc_compile.float_dtypes(device)
    if dtype_class is DTypeClass.ABS:
        return (
            gc_compile.float_dtypes(device)
            + gc_compile.SIGNED_INT_DTYPES
            + gc_compile.UNSIGNED_INT_DTYPES
        )
    if dtype_class is DTypeClass.NEGATIVE:
        return gc_compile.float_dtypes(device) + gc_compile.SIGNED_INT_DTYPES
    if dtype_class is DTypeClass.BOOL:
        return [DType.bool]
    raise ValueError(f"Unknown dtype_class: {dtype_class!r}")


def _graph_name(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype)."""
    name = gc_compile.canonical_op_name(op_type, _UNARY_OPS_BY_NAME)
    return f"unary_{name}_{device.label}_{device.id}_{dtype.name}"


canonical_shape = gc_compile.canonical_shape_rank1


def _unary_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: UnarySpec,
    device: Device,
    dtype: DType,
) -> None:
    """Adds one fully-symbolic rank-1 unary graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, ["n"], device=device_ref)
    graph = Graph(
        _graph_name(op_type, device, dtype),
        input_types=[in_type],
        module=module,
    )
    with graph:
        (x,) = graph.inputs
        graph.output(spec.builder(x.tensor))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    """Whether (op, device, dtype) is in the conservatively-supported set.

    Single source of truth for the swept matrix (the sweep filter and
    :func:`unary_model`'s guard both route through it). CPU-only ops are
    unsupported on accelerators; each op supports only its ``dtype_class``.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    name = gc_compile.canonical_op_name(op_type, _UNARY_OPS_BY_NAME)
    if device.label != "cpu" and name in _CPU_ONLY_NAMES:
        return False
    return dtype in _supported_dtypes(spec.dtype_class, device)


class _UnaryFamily(gc_compile.GCFamilySpec):
    name = "unary"

    def build_module(self) -> Module:
        """Build the full batched unary module: every supported (op, device,
        dtype) across CPU + all accelerators, in one module.

        Host-ELF and cubins both embed self-contained in the exported MEF, so
        one force-load populates every device class at once. Shared by the
        warm producer (export) and the batched sweep. Unsupported (op,
        device, dtype) targets are filtered out via :func:`_is_supported`
        (MXF-477).
        """
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Build the unary module for a single device slot: every supported
        (op, dtype) on *device*, and nothing else.

        Per-slot counterpart of :meth:`build_module`. The warm producer
        exports one MEF per slot so the warm is device-count-independent: a
        k-GPU consumer force-loads only slots ``0..k-1``.
        """
        if module is None:
            module = Module()
        for op_type, spec in _UNARY_OPS.items():
            for dtype in _supported_dtypes(spec.dtype_class, device):
                if _is_supported(op_type, device, dtype):
                    _unary_graph(module, op_type, spec, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_UnaryFamily())
gc_compile.register_family(_FAMILY)


def unary_model(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Returns the unary :class:`~max.engine.Model` for *op_type* / *device* / *dtype*.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss an available warm cache is adopted whole instead
    of compiling each target singly, force-loaded from a manifest when one is
    present and adoptable, else via a batched sweep of a matching
    ``warm-interpreter-cache`` stamp.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled.
        device: The realized input's device.
        dtype: The realized input's dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set (e.g.
            a transcendental op on an int dtype).
        EagerLazyCompileDisallowed: If a supported target is not already
            compiled and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
    """
    key = _graph_name(op_type, device, dtype)
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
            f"Unsupported unary op/device/dtype for key {key!r}."
            f"  Supported dtypes for this op/device: {supported}"
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _unary_graph(module, op_type, spec, device, dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
