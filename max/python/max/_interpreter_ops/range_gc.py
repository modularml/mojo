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

"""Graph-compiler range model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

Unlike the ``MO_HostOnly`` families (``nonzero_gc``, ``nms_gc``), this one
sweeps accelerators normally and does not override ``sweep_devices()``:
``mo.range`` carries ``MO_SingleDeviceWithHostOperands`` on ``start``/``limit``/
``step`` only, so the three scalars are host tensors while the result itself
lives on the target device.

The output length is data-dependent (``stop``/``start``/``step`` are runtime
values), and the op's registered ``range_shape`` function computes it
(``builtin_kernels/kernels.mojo``), so the graph declares a symbolic ``["n"]``
output and sizes itself at runtime. The public ``ops.range`` is used directly
(no ``rmo`` emission needed): it requires an explicit ``out_dim`` for dynamic
scalar inputs, which is exactly this case, and it validates that the three
scalars are on CPU, which they are.

``MO_SameOperandsAndResultElementType`` means ``start``/``limit``/``step``
already carry the result dtype, so the handler passes them straight through
with no re-materialization.

Dtype coverage follows the shared int/float policy (:func:`gc_compile.
float_dtypes`, :data:`gc_compile.SIGNED_INT_DTYPES`, :data:`gc_compile.
UNSIGNED_INT_DTYPES`). The deleted binding also accepted float16/bfloat16 on
CPU; those are excluded here for the same reason ``matmul_gc`` and the other
graph-compiler-backed float ops exclude them. GPU coverage is unchanged from
the deleted binding (float16/float32/bfloat16, no float64).
"""

from dataclasses import dataclass

from max import engine
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "range"


def _supported_dtypes(device: Device) -> list[DType]:
    """Returns the dtypes range sweeps on *device*.

    The deleted binding rejected only bool.
    """
    return (
        gc_compile.float_dtypes(device)
        + gc_compile.SIGNED_INT_DTYPES
        + gc_compile.UNSIGNED_INT_DTYPES
    )


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType
    """The result (and scalar-operand) dtype."""

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype.name}"
        )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype)
    for device in gc_compile.DISCOVERED_DEVICES
    for dtype in _supported_dtypes(device)
]


def _range_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one symbolic range graph into *module* in-place."""
    device_ref = DeviceRef.from_device(target.device)
    cpu = DeviceRef.CPU()
    scalar_type = TensorType(target.dtype, [], device=cpu)
    graph = Graph(
        target.graph_name,
        input_types=[scalar_type, scalar_type, scalar_type],
        module=module,
    )
    with graph:
        start, stop, step = (v.tensor for v in graph.inputs)
        graph.output(
            graph_ops.range(
                start,
                stop,
                step,
                out_dim="n",
                dtype=target.dtype,
                device=device_ref,
            )
        )


class _RangeFamily(gc_compile.GCFamilySpec):
    name = "range"

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
                _range_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_RangeFamily())
gc_compile.register_family(_FAMILY)


def range_model(device: Device, dtype: DType) -> engine.Model:
    """Returns the range :class:`~max.engine.Model` for *device* and *dtype*.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: The target device (CPU or GPU accelerator) the result lives
            on.
        dtype: The result (and scalar-operand) dtype.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If *dtype* is unsupported on *device*, or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not precompiled.
    """
    target = CompilationTarget(device, dtype)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if dtype not in _supported_dtypes(device):
            supported = ", ".join(str(d) for d in _supported_dtypes(device))
            return (
                f"range does not support dtype {dtype} on {device.label};"
                f" supported: {supported}. (key {key!r})"
            )
        return None

    return _FAMILY.model_for(
        key,
        device,
        lambda m: _range_graph(m, target),
        unsupported_reason=check_supported,
    )
