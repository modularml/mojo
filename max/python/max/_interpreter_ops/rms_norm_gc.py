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

"""Graph-compiler rms_norm model cache for the MO interpreter.

Covers ``ReduceRmsNormOp`` only. ``epsilon`` and ``weight_offset`` are runtime
host (CPU) tensor operands on the MO op
(``MO_SingleDeviceWithHostOperands<["epsilon", "weightOffset"]>``, confirmed
in ``MOOps.td``), so both are graph inputs rather than baked constants --
neither enters the cache key. ``multiply_before_cast`` is a genuine
``BoolAttr`` (a compile-time MLIR attribute, not an operand -- also confirmed
in ``MOOps.td``), so it rides in the cache key as a two-value variant,
mirroring ``reduce_axis_gc``'s cumsum ``(exclusive, reverse)`` variant.

Built directly against ``mo.ReduceRmsNormOp`` (not the ``ops.rms_norm``
graph-construction wrapper) for the same reason as ``layer_norm_gc``: that
wrapper bakes ``epsilon``/``weight_offset`` in as Python constants at trace
time.

Supported dtypes mirror the shared float set (see
:func:`gc_compile.float_dtypes`), same policy as ``layer_norm_gc``.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

from max import engine
from max._core.dialects import builtin, kgen, mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

canonical_rank2 = gc_compile.canonical_rank2

_VARIANTS = (False, True)  # multiply_before_cast: Llama-style, Gemma-style


def _variant_tag(multiply_before_cast: bool) -> str:
    return f"_mbc{int(multiply_before_cast)}"


def _graph_name(
    device: Device, dtype: DType, multiply_before_cast: bool
) -> str:
    return (
        f"rms_norm_{device.label}_{device.id}_{dtype.name}"
        f"{_variant_tag(multiply_before_cast)}"
    )


def _rms_norm_graph(
    module: Module, device: Device, dtype: DType, multiply_before_cast: bool
) -> None:
    """Adds one fully-symbolic rank-2 rms_norm graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    graph = Graph(
        _graph_name(device, dtype, multiply_before_cast),
        input_types=[
            TensorType(dtype, ["rows", "features"], device=device_ref),
            TensorType(dtype, ["features"], device=device_ref),
            TensorType(DType.float32, [], device=cpu),
            TensorType(dtype, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        x, weight, epsilon, weight_offset = (v.tensor for v in graph.inputs)
        result = Graph.current._add_op_generated(
            mo.ReduceRmsNormOp,
            result=TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            input=x,
            weight=weight,
            epsilon=epsilon,
            weight_offset=weight_offset,
            multiply_before_cast=builtin.BoolAttr(multiply_before_cast),
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


class _RmsNormFamily(gc_compile.GCFamilySpec):
    name = "rms_norm"

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
        for dtype in gc_compile.float_dtypes(device):
            for multiply_before_cast in _VARIANTS:
                _rms_norm_graph(module, device, dtype, multiply_before_cast)
        return module


_FAMILY = gc_compile.GCOpFamily(_RmsNormFamily())
gc_compile.register_family(_FAMILY)


def rms_norm_model(
    device: Device, dtype: DType, multiply_before_cast: bool
) -> engine.Model:
    """Returns the rms_norm Model for the given target.

    Lazy by default (compiled and cached on first use); the first miss adopts
    a whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: The realized input's device.
        dtype: The realized input's dtype.
        multiply_before_cast: The op's compile-time ``multiply_before_cast``
            flag (Gemma-style vs Llama-style); part of the cache key since it
            is a compile-time MLIR attribute, not a runtime operand.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If (device, dtype) is outside the supported set; or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target was not
            swept.
    """
    key = _graph_name(device, dtype, multiply_before_cast)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        supported = gc_compile.float_dtypes(device)
        if dtype in supported:
            return None
        return (
            f"Unsupported rms_norm device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {supported}"
        )

    def build(module: Module) -> None:
        _rms_norm_graph(module, device, dtype, multiply_before_cast)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
