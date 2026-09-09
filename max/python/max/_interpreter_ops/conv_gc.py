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

"""Graph-compiler conv model cache for the MO interpreter.

Covers ``ConvOp`` only (``ConvTransposeOp`` is intentionally excluded -- see
below). Filter shape (kernel height/width AND channel counts), input shape,
stride, dilation, and padding are all runtime tensor operands on the
MO-analogue op (``rmo.mo.conv``).

``ConvTransposeOp`` has no supported kernel -- see KERN-3233.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

Supported DTypes (mirrors ``matmul_gc``, not the old Mojo binding -- conv
lowers to matmul, so matmul's dtype policy applies; see
:func:`_supported_dtypes`):
  - GPU: bfloat16, float32, float16
  - CPU: float32, float64, int32, int64
"""

from max import engine
from max._core.dialects import kgen, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph.type import ConvInputLayout, FilterLayout


def _supported_dtypes(device: Device) -> list[DType]:
    """Mirrors matmul_gc's dtype policy (see module docstring): the shared
    per-device float set, plus int32/int64 on CPU only (conv lowers to
    matmul there). int8/int16 aren't added even though matmul's CPU set
    includes them -- the old conv kernel never supported them and
    conv-specific correctness is unverified.
    """
    floats = gc_compile.float_dtypes(device)
    if device.label != "cpu":
        return floats
    return [*floats, DType.int32, DType.int64]


def _is_supported(device: Device, dtype: DType) -> bool:
    return dtype in _supported_dtypes(device)


def _graph_name(device: Device, dtype: DType) -> str:
    return f"conv_{device.label}_{device.id}_{dtype.name}"


def _conv_graph(module: Module, device: Device, dtype: DType) -> None:
    """Adds one fully-symbolic NHWC/RSCF conv graph into *module* in-place.

    The handler must pre-check groups == 1 before calling the returned model
    -- see ``_handle_conv``'s docstring for why.

    "c_pg" (filter in_channels/groups) is a symbol distinct from "c" (input
    channels) -- they are only equal when groups == 1, which is the only
    value this graph is actually exercised with, but keeping them distinct
    avoids silently constraining the graph to groups == 1 at the type level.
    """
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    graph = Graph(
        _graph_name(device, dtype),
        input_types=[
            TensorType(dtype, ["n", "h", "w", "c"], device=device_ref),
            TensorType(dtype, ["r", "s", "c_pg", "f"], device=device_ref),
            TensorType(DType.int64, [2], device=cpu),
            TensorType(DType.int64, [2], device=cpu),
            TensorType(DType.int64, [4], device=cpu),
            TensorType(DType.int64, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        x, filt, strides, dilations, paddings, groups = (
            v.tensor for v in graph.inputs
        )
        out_type = TensorType(
            dtype, ["n_out", "h_out", "w_out", "f_out"], device=device_ref
        )
        result = Graph.current._add_op_generated(
            rmo.MoConvOp,
            result=out_type,
            input=x,
            filter=filt._with_layout(FilterLayout.RSCF),
            strides=strides,
            dilations=dilations,
            paddings=paddings,
            num_groups=groups,
            input_layout=ConvInputLayout.NHWC,
            filter_layout="",
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


class _ConvFamily(gc_compile.GCFamilySpec):
    name = "conv"

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
        for dtype in _supported_dtypes(device):
            _conv_graph(module, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_ConvFamily())
gc_compile.register_family(_FAMILY)


def conv_model(device: Device, dtype: DType) -> engine.Model:
    """Returns the conv Model for the given (device, dtype) (lazy by default).

    Callers must pre-check ``num_groups == 1`` before calling this and
    passing the returned model buffers -- this function does not validate
    that (it is a runtime tensor operand the graph itself accepts for any
    value, but the underlying kernel only supports groups == 1; see
    ``_handle_conv``'s docstring).

    Args:
        device: The realized input's device.
        dtype: The realized input's dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If (device, dtype) is outside the supported set.
        EagerLazyCompileDisallowed: If a supported target is not already
            compiled and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
    """
    key = _graph_name(device, dtype)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(device, dtype):
            return None
        return (
            f"Unsupported conv device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {_supported_dtypes(device)}"
        )

    def build(module: Module) -> None:
        _conv_graph(module, device, dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
