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

"""Graph-compiler random model cache for the MO interpreter.

Covers ``mo.random.normal`` and ``mo.random.uniform``. Two compile modes,
selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

Graphs are flat rank 1, with the output shape a runtime operand. Emission
goes directly through ``rmo`` because the public ``ops.random`` helpers
derive their shape operand from a ``TensorType`` this graph can't
materialize, and rotate the seed.

Dtype coverage follows :func:`gc_compile.float_dtypes`.

Normal's f16/bf16 accelerator sweep entries are dropped only to save
compile time, not a support narrowing: ``ops.random.gaussian`` always
promotes those to float32 first, so no public caller can request one, and
``check_supported`` still accepts them.

Normal takes float32 ``mean``/``variance`` regardless of output dtype;
uniform's bounds carry the output dtype, and its samples depend on the
kernel's SIMD grouping, so this flat graph can differ from a shaped
compiled graph for the same seed.
"""

from dataclasses import dataclass

from max import _core, engine
from max._core.dialects import kgen, mo, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

_GRAPH_BASE_NAME = "random"


@dataclass(frozen=True)
class _RandomSpec:
    """Per-op differences: which ``rmo`` op, and its two scalar operands."""

    rmo_cls: type
    scalar_names: tuple[str, str]
    scalars_are_float32: bool
    """Normal's mean/variance are always float32; uniform's bounds are not."""


_RANDOM_OPS: dict[type[_core.Operation], _RandomSpec] = {
    mo.RandomNormalOp: _RandomSpec(
        rmo.MoRandomNormalOp, ("mean", "variance"), True
    ),
    mo.RandomUniformOp: _RandomSpec(
        rmo.MoRandomUniformOp, ("lower_bound", "upper_bound"), False
    ),
}

_RANDOM_OPS_BY_NAME = {
    op_type.__name__: spec for op_type, spec in _RANDOM_OPS.items()
}

_PUBLIC_OP_NAME = {
    mo.RandomNormalOp: "random.normal",
    mo.RandomUniformOp: "random.uniform",
}
"""Public ``ops.random`` spelling for an error message, keyed by ``mo`` type."""


def _spec_for(op_type: type[_core.Operation]) -> _RandomSpec | None:
    return gc_compile.spec_for(op_type, _RANDOM_OPS_BY_NAME)


def scalars_are_float32(op_type: type[_core.Operation]) -> bool:
    """Returns whether *op_type*'s two scalar operands are always float32.

    Resolves through :func:`_spec_for` (i.e. ``gc_compile.canonical_op_name``),
    so this answers correctly for an op arriving via either the ``mo`` or
    ``rmo`` spelling -- unlike an ``isinstance(op, mo.RandomNormalOp)`` check,
    which only matches the ``mo`` type object and would silently disagree
    with :func:`random_model` for an op reached through the name-based
    handler fallback (see ``handlers.register_op_handler``).

    Args:
        op_type: ``mo.RandomNormalOp`` or ``mo.RandomUniformOp``.

    Returns:
        ``True`` for ``mo.RandomNormalOp`` (mean/variance are always
        float32); ``False`` for ``mo.RandomUniformOp`` (bounds carry the
        output dtype).

    Raises:
        KeyError: If *op_type* is not a registered random op.
    """
    spec = _spec_for(op_type)
    if spec is None:
        raise KeyError(f"Unsupported random op {op_type.__name__}")
    return spec.scalars_are_float32


@dataclass(frozen=True)
class CompilationTarget:
    op_name: str
    device: Device
    dtype: DType

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.op_name}_{self.device.label}_"
            f"{self.device.id}_{self.dtype.name}"
        )


# Unreachable via the public API (promoted to float32 first); check_supported
# still accepts them so a non-public caller compiles lazily -- deliberate.
_UNREACHABLE_NORMAL_DTYPES = (DType.float16, DType.bfloat16)


def _sweep_dtypes(
    op_type: type[_core.Operation], device: Device
) -> list[DType]:
    """Dtypes precompiled for one (op, device) pair."""
    dtypes = gc_compile.float_dtypes(device)
    if op_type is mo.RandomNormalOp:
        return [d for d in dtypes if d not in _UNREACHABLE_NORMAL_DTYPES]
    return dtypes


_COMPILATION_TARGETS = [
    CompilationTarget(op_type.__name__, device, dtype)
    for op_type in _RANDOM_OPS
    for device in gc_compile.DISCOVERED_DEVICES
    for dtype in _sweep_dtypes(op_type, device)
]


def _random_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one flat rank-1 random graph into *module* in-place."""
    spec = _RANDOM_OPS_BY_NAME[target.op_name]
    device_ref = DeviceRef.from_device(target.device)
    cpu = DeviceRef.CPU()
    scalar_dtype = DType.float32 if spec.scalars_are_float32 else target.dtype
    out_type = TensorType(target.dtype, ["n"], device=device_ref)
    graph = Graph(
        target.graph_name,
        input_types=[
            TensorType(DType.int64, [1], device=cpu),
            TensorType(scalar_dtype, [], device=cpu),
            TensorType(scalar_dtype, [], device=cpu),
            # seed lives on the result device, not the host: the MLIR
            # verifier rejects a host-only seed (matches ops.random's transfer).
            TensorType(DType.uint64, [1], device=device_ref),
        ],
        module=module,
    )
    with graph:
        shape, lo, hi, seed = (v.tensor for v in graph.inputs)
        kwargs = {
            "result": out_type,
            "shape": shape,
            spec.scalar_names[0]: lo,
            spec.scalar_names[1]: hi,
            "seed": seed,
            "output_param_decls": kgen.ParamDeclArrayAttr([]),
        }
        graph.output(
            Graph.current._add_op_generated(spec.rmo_cls, **kwargs)[0].tensor
        )


class _RandomFamily(gc_compile.GCFamilySpec):
    name = "random"

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
        for target in _COMPILATION_TARGETS:
            if (
                target.device.label == device.label
                and target.device.id == device.id
            ):
                _random_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_RandomFamily())
gc_compile.register_family(_FAMILY)


def random_model(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Returns the random :class:`~max.engine.Model` for one target.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        op_type: ``mo.RandomNormalOp`` or ``mo.RandomUniformOp``.
        device: The target device (CPU or GPU accelerator).
        dtype: The output element dtype.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If the op/device/dtype triple is unsupported, or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not precompiled.
    """
    name = gc_compile.canonical_op_name(op_type, _RANDOM_OPS_BY_NAME)
    target = CompilationTarget(name, device, dtype)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _spec_for(op_type) is None:
            return f"Unsupported random op {op_type.__name__}; key {key!r}."
        if dtype not in gc_compile.float_dtypes(device):
            public_name = _PUBLIC_OP_NAME.get(op_type, op_type.__name__)
            supported = ", ".join(
                str(d) for d in gc_compile.float_dtypes(device)
            )
            return (
                f"{public_name} does not support {dtype} on {device.label};"
                f" supported: {supported}. (key {key!r})"
            )
        return None

    return _FAMILY.model_for(
        key,
        device,
        lambda m: _random_graph(m, target),
        unsupported_reason=check_supported,
    )
