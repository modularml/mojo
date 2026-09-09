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

"""Graph-compiler non-maximum-suppression model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

CPU only: ``mo.non_maximum_suppression`` carries the ``MO_HostOnly`` trait, so
the handler never receives a device buffer.

The selected-box count is data-dependent, and the op's registered shape function
runs the real suppression to get it (``builtin_kernels/kernels.mojo``), so the
model returns an exactly-sized ``[num_selected, 3]`` output. The interpreter's
old upper-bound work buffer and its truncating host copy are therefore gone --
in exchange, sizing the output this way costs a full suppression pass, so NMS
now runs the O(n^2) suppression core twice per call (once in the shape
function, once in the compute kernel) where the deleted binding ran it once,
deliberately, to avoid exactly this redundant recomputation. This trade is
accepted: the old single pass required an upper-bound allocation of
``batch * num_classes * max_output_boxes_per_class * 3`` int64s -- bounded by
the caller-supplied operands (about 1.9 MB for a 1x80x1000 call), not
unbounded. A compiled model must size its own output, so this shape-function
dry run is unavoidable; eager now simply pays the same double-suppression
cost the compiled graph path already paid.

The three thresholds are already tensor operands on the public op, so no direct
``rmo`` emission is needed here; the handler normalises them to the dtypes the
graph declares.
"""

from dataclasses import dataclass

from max import engine
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "nms"

_CPU_DEVICES = [d for d in gc_compile.DISCOVERED_DEVICES if d.label == "cpu"]


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType
    """The shared ``boxes``/``scores`` float dtype."""

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype.name}"
        )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype)
    for device in _CPU_DEVICES
    for dtype in gc_compile.CPU_FLOAT_DTYPES
]


def _nms_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one symbolic NMS graph into *module* in-place."""
    cpu = DeviceRef.CPU()
    graph = Graph(
        target.graph_name,
        input_types=[
            TensorType(target.dtype, ["batch", "num_boxes", 4], device=cpu),
            TensorType(
                target.dtype, ["batch", "num_classes", "num_boxes"], device=cpu
            ),
            # Rank 0 matches the op's scalar operand types.
            TensorType(DType.int64, [], device=cpu),
            TensorType(DType.float32, [], device=cpu),
            TensorType(DType.float32, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        boxes, scores, max_output, iou, score = (v.tensor for v in graph.inputs)
        graph.output(
            graph_ops.non_maximum_suppression(
                boxes, scores, max_output, iou, score
            )
        )


class _NmsFamily(gc_compile.GCFamilySpec):
    name = "nms"

    def sweep_devices(self) -> list[Device]:
        # MO_HostOnly: sweeping accelerators would make the warm producer
        # (xarch_warm/warm_lib.py export_slots) export a useless MEF per GPU.
        return list(_CPU_DEVICES)

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
                _nms_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_NmsFamily())
gc_compile.register_family(_FAMILY)


def nms_model(dtype: DType) -> engine.Model:
    """Returns the NMS :class:`~max.engine.Model` for *dtype*.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        dtype: The shared ``boxes``/``scores`` float dtype.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If *dtype* is not a supported CPU float, or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not precompiled.
    """
    device = _CPU_DEVICES[0]
    target = CompilationTarget(device, dtype)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if dtype not in gc_compile.CPU_FLOAT_DTYPES:
            supported = ", ".join(str(d) for d in gc_compile.CPU_FLOAT_DTYPES)
            return (
                f"non_maximum_suppression does not support dtype {dtype};"
                f" supported: {supported}. (key {key!r})"
            )
        return None

    return _FAMILY.model_for(
        key,
        device,
        lambda m: _nms_graph(m, target),
        unsupported_reason=check_supported,
    )
