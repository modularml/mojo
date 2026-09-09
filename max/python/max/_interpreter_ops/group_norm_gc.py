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

"""Graph-compiler group_norm model cache for the MO interpreter.

Covers ``ReduceGroupNormOp`` only.
``epsilon`` and ``num_groups`` are runtime host (CPU) tensor operands on the
MO op (``MO_SingleDeviceWithHostOperands<["epsilon", "numGroups"]>``,
confirmed in ``MOOps.td``), so neither enters the cache key -- the family key
is just ``(device, dtype)``, mirroring ``conv_gc``.

The graph is built at a fixed canonical rank 4 ``[n, c, h, 1]`` regardless of
the real input's rank (2, 3, or 4): ``num_groups`` is a runtime operand the
kernel groups by internally, so no reshape happens at the graph level; the
canonical rank just gives the graph a concrete arity. Collapsing all spatial
dims into one ``h`` (with a trailing size-1 ``w``) does not change the
group's mean/var (it's a sum over all group elements, so adding a size-1 axis
is a no-op), letting one compiled graph per ``(device, dtype)`` serve every
input rank -- the same technique ``matmul_gc``/``conv_gc``/``reduce_axis_gc``
use for their own canonical ranks.

Built directly against ``mo.ReduceGroupNormOp`` (not the ``ops.group_norm``
graph-construction wrapper) for the same reason as ``layer_norm_gc``: that
wrapper bakes ``epsilon``/``num_groups`` in as Python constants at trace time.

Supported dtypes mirror the shared float set (see
:func:`gc_compile.float_dtypes`): CPU float32/float64, GPU float16/float32/
bfloat16.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.
"""

from collections.abc import Sequence
from math import prod

from max import engine
from max._core.dialects import kgen, mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType


def canonical_shape(shape: Sequence[int]) -> tuple[int, int, int, int]:
    """Collapses *shape* (``[n, c, ...]``) to rank 4 ``[n, c, h, 1]``.

    All dims after the channel axis are flattened into ``h``; ``prod(())`` is
    1, so a rank-2 ``[n, c]`` input yields ``h == 1``, keeping that case
    branchless.
    """
    n, c, *spatial = shape
    return (n, c, prod(spatial), 1)


def _graph_name(device: Device, dtype: DType) -> str:
    return f"group_norm_{device.label}_{device.id}_{dtype.name}"


def _group_norm_graph(module: Module, device: Device, dtype: DType) -> None:
    """Adds one fully-symbolic rank-4 group_norm graph into *module*
    in-place."""
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    graph = Graph(
        _graph_name(device, dtype),
        input_types=[
            TensorType(dtype, ["n", "c", "h", "w"], device=device_ref),
            TensorType(dtype, ["c"], device=device_ref),
            TensorType(dtype, ["c"], device=device_ref),
            TensorType(DType.float32, [], device=cpu),
            TensorType(DType.int32, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        x, gamma, beta, epsilon, num_groups = (v.tensor for v in graph.inputs)
        result = Graph.current._add_op_generated(
            mo.ReduceGroupNormOp,
            result=TensorType(dtype=x.dtype, shape=x.shape, device=x.device),
            input=x,
            gamma=gamma,
            beta=beta,
            epsilon=epsilon,
            num_groups=num_groups,
            output_param_decls=kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(result)


class _GroupNormFamily(gc_compile.GCFamilySpec):
    name = "group_norm"

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
            _group_norm_graph(module, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_GroupNormFamily())
gc_compile.register_family(_FAMILY)


def group_norm_model(device: Device, dtype: DType) -> engine.Model:
    """Returns the group_norm Model for the given (device, dtype).

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
            f"Unsupported group_norm device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {supported}"
        )

    def build(module: Module) -> None:
        _group_norm_graph(module, device, dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
