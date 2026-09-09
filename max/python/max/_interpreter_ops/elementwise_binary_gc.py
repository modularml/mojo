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

"""Graph-compiler binary-elementwise model cache for the MO interpreter.

Covers both arithmetic/bitwise binary ops (``Add``, ``Sub``, ``Mul``, ``Div``,
``Mod``, ``Max``, ``Min``, ``And``, ``Or``, ``Xor``, ``Pow``) and the comparison
predicates (``Equal``, ``Greater``, ``GreaterEqual``, ``NotEqual``). A comparison
graph emits ``bool``; the handler reads the realized output dtype, so the two
families share one builder and cache.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a target compiles just that
  one rank-1 graph.
- **Precompile sweep (``=1``).** The batched sweep compiles the full matrix at
  import; a :func:`binary_model` miss is then a hard error.

Lazy mode avoids a trivial program JIT-compiling the whole kernel library on a
cold cache (~3000+ kernels, minutes; MXF-508). Models serve the eager handler
via :func:`binary_model`. Must not import from ``handlers.py``.

Both operands reach the handler already cast to a common dtype and broadcast to
a common shape (the RMO->MO lowering inserts the ``BroadcastToOp`` chain), so a
graph with one dtype shared by both rank-1 inputs is exact.

The swept dtype set is deliberately conservative (the IR type category is only a
ceiling): general arithmetic (``Add``/``Sub``/``Mul``/``Max``/``Min``/``Mod``),
``Pow``, and the comparisons sweep floats + integers; ``Div`` sweeps floats only;
the logical ops (``And``/``Or``/``Xor``) sweep ``bool``. CPU floats are f32/f64
(no 16-bit); GPU floats are f16/f32/bf16 (no f64). ``dtype_class`` keys the
*input* dtype.
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
from max.graph import DeviceRef, Graph, Module, TensorType, TensorValue
from max.graph.ops import elementwise

# Builds an op's graph body from its two input tensors (e.g. ``elementwise.add``).
MoBinaryOpBuilder: TypeAlias = Callable[[TensorValue, TensorValue], TensorValue]


class DTypeClass(Enum):
    """The input-dtype set an op is swept over (see ``_supported_dtypes``)."""

    NUMERIC = "numeric"
    FLOAT = "float"
    BOOL = "bool"


@dataclass(frozen=True)
class BinarySpec:
    """How to build one binary op's graph and which dtype class it sweeps."""

    builder: MoBinaryOpBuilder
    dtype_class: DTypeClass


# The builder is a two-arg elementwise op; comparison builders emit a bool
# tensor, which the handler picks up from the realized output dtype.
_BINARY_OPS: dict[type[_core.Operation], BinarySpec] = {
    mo.AddOp: BinarySpec(elementwise.add, DTypeClass.NUMERIC),
    mo.SubOp: BinarySpec(elementwise.sub, DTypeClass.NUMERIC),
    mo.MulOp: BinarySpec(elementwise.mul, DTypeClass.NUMERIC),
    mo.MaxOp: BinarySpec(elementwise.max, DTypeClass.NUMERIC),
    mo.MinOp: BinarySpec(elementwise.min, DTypeClass.NUMERIC),
    mo.ModOp: BinarySpec(elementwise.mod, DTypeClass.NUMERIC),
    # div promotes int operands to f64 in the lowering, so an int div never
    # reaches the handler; pow has no such promotion, so it must sweep ints.
    mo.DivOp: BinarySpec(elementwise.div, DTypeClass.FLOAT),
    mo.PowOp: BinarySpec(elementwise.pow, DTypeClass.NUMERIC),
    mo.AndOp: BinarySpec(elementwise.logical_and, DTypeClass.BOOL),
    mo.OrOp: BinarySpec(elementwise.logical_or, DTypeClass.BOOL),
    mo.XorOp: BinarySpec(elementwise.logical_xor, DTypeClass.BOOL),
    # Comparison predicates: numeric input, bool output.
    mo.EqualOp: BinarySpec(elementwise.equal, DTypeClass.NUMERIC),
    mo.GreaterOp: BinarySpec(elementwise.greater, DTypeClass.NUMERIC),
    mo.GreaterEqualOp: BinarySpec(
        elementwise.greater_equal, DTypeClass.NUMERIC
    ),
    mo.NotEqualOp: BinarySpec(elementwise.not_equal, DTypeClass.NUMERIC),
}

BINARY_GC_OPS = tuple(_BINARY_OPS)

# Indexed by op name so an rmo dispatch resolves to the mo-keyed spec; see
# gc_compile.canonical_op_name.
_BINARY_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _BINARY_OPS.items()
}


def _spec_for(op_type: type[_core.Operation]) -> BinarySpec | None:
    return gc_compile.spec_for(op_type, _BINARY_OPS_BY_NAME)


def _supported_dtypes(dtype_class: DTypeClass, device: Device) -> list[DType]:
    """Conservative swept dtype set for a (dtype_class, device)."""
    if dtype_class is DTypeClass.FLOAT:
        return gc_compile.float_dtypes(device)
    if dtype_class is DTypeClass.NUMERIC:
        return (
            gc_compile.float_dtypes(device)
            + gc_compile.SIGNED_INT_DTYPES
            + gc_compile.UNSIGNED_INT_DTYPES
        )
    if dtype_class is DTypeClass.BOOL:
        return [DType.bool]
    raise ValueError(f"Unknown dtype_class: {dtype_class!r}")


def _graph_name(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype)."""
    name = gc_compile.canonical_op_name(op_type, _BINARY_OPS_BY_NAME)
    return f"binary_{name}_{device.label}_{device.id}_{dtype.name}"


canonical_shape = gc_compile.canonical_shape_rank1


def _binary_graph(
    module: Module,
    op_type: type[_core.Operation],
    spec: BinarySpec,
    device: Device,
    dtype: DType,
) -> None:
    """Adds one fully-symbolic rank-1 binary graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    in_type = TensorType(dtype, ["n"], device=device_ref)
    graph = Graph(
        _graph_name(op_type, device, dtype),
        input_types=[in_type, in_type],
        module=module,
    )
    with graph:
        lhs, rhs = graph.inputs
        graph.output(spec.builder(lhs.tensor, rhs.tensor))


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    """Whether (op, device, dtype) is in the conservatively-supported set.

    Single source of truth for the swept matrix:
    :meth:`_BinaryFamily.build_module_for_device` filters its candidates
    through this predicate, and lazy mode uses it as the support guard in
    :func:`binary_model`, so the two can't diverge. Each op supports only
    its ``dtype_class``'s dtypes.
    """
    spec = _spec_for(op_type)
    if spec is None:
        return False
    return dtype in _supported_dtypes(spec.dtype_class, device)


class _BinaryFamily(gc_compile.GCFamilySpec):
    name = "binary"

    def build_module(self) -> Module:
        """Batched module: every supported (op, device, dtype), all devices."""
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
        for op_type, spec in _BINARY_OPS.items():
            for dtype in _supported_dtypes(spec.dtype_class, device):
                if _is_supported(op_type, device, dtype):
                    _binary_graph(module, op_type, spec, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_BinaryFamily())
gc_compile.register_family(_FAMILY)


def binary_model(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Returns the binary :class:`~max.engine.Model` for *op_type* / *device* / *dtype*.

    Lazy by default: compiled on first use and cached for the process lifetime.
    With ``MAX_EAGER_OP_PRECOMPILE=1`` it was precompiled at import and this is a
    lookup. On the first miss, a warm cache is adopted whole (manifest
    force-load, else a batched stamp sweep) instead of compiling per target.

    Args:
        op_type: The concrete ``mo.*Op`` type of the op being handled.
        device: The realized inputs' device.
        dtype: The realized inputs' (shared) dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set (e.g.
            ``Div`` on an int dtype).
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
            f"Unsupported binary op/device/dtype for key {key!r}."
            f"  Supported dtypes for this op/device: {supported}"
        )

    def build(module: Module) -> None:
        spec = _spec_for(op_type)
        assert spec is not None, f"unsupported op {op_type!r} reached compile"
        _binary_graph(module, op_type, spec, device, dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
