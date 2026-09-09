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

"""Graph-compiler layer_norm model cache for the MO interpreter.

Covers ``ReduceLayerNormOp`` only. ``epsilon`` is a runtime host (CPU) tensor
operand on the MO op (``MO_SingleDeviceWithHostOperands<["epsilon"]>``,
confirmed in ``MOOps.td``), so it is a graph input rather than a baked
constant -- the family key is ``(device, dtype)`` with no epsilon-specific
variant, mirroring ``conv_gc``.

Built directly against ``mo.ReduceLayerNormOp`` (not the ``ops.layer_norm``
graph-construction wrapper in ``max/python/max/graph/ops/layer_norm.py``)
because that wrapper bakes ``epsilon`` in as a Python-float constant at trace
time; this family instead needs epsilon as a genuine per-call graph input --
the same reason ``conv_gc`` builds ``rmo.MoConvOp`` directly instead of using
``ops.conv2d``.

Supported dtypes mirror the shared float set (see
:func:`gc_compile.float_dtypes`): CPU float32/float64, GPU float16/float32/
bfloat16 -- matches ``FLOAT_DTYPES`` in ``test_interpreter_ops.py``, the only
dtypes the old eager Mojo binding was ever tested against.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

from max import engine
from max._core.dialects import kgen, mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

canonical_rank2 = gc_compile.canonical_rank2


def _graph_name(device: Device, dtype: DType) -> str:
    return f"layer_norm_{device.label}_{device.id}_{dtype.name}"


def _layer_norm_graph(module: Module, device: Device, dtype: DType) -> None:
    """Adds one fully-symbolic rank-2 layer_norm graph into *module* in-place."""
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    graph = Graph(
        _graph_name(device, dtype),
        input_types=[
            TensorType(dtype, ["rows", "features"], device=device_ref),
            TensorType(dtype, ["features"], device=device_ref),
            TensorType(dtype, ["features"], device=device_ref),
            TensorType(DType.float32, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        x, gamma, beta, epsilon = (v.tensor for v in graph.inputs)
        result = Graph.current._add_op_generated(
            mo.ReduceLayerNormOp,
            result=TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            input=x,
            gamma=gamma,
            beta=beta,
            epsilon=epsilon,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


class _LayerNormFamily(gc_compile.GCFamilySpec):
    name = "layer_norm"

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
            _layer_norm_graph(module, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_LayerNormFamily())
gc_compile.register_family(_FAMILY)


def layer_norm_model(device: Device, dtype: DType) -> engine.Model:
    """Returns the layer_norm Model for the given (device, dtype).

    Lazy by default (compiled and cached on first use); the first miss adopts
    a whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: The realized input's device.
        dtype: The realized input's dtype.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If (device, dtype) is outside the supported set; or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if a supported target was not
            swept.
    """
    key = _graph_name(device, dtype)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        supported = gc_compile.float_dtypes(device)
        if dtype in supported:
            return None
        return (
            f"Unsupported layer_norm device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {supported}"
        )

    def build(module: Module) -> None:
        _layer_norm_graph(module, device, dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
