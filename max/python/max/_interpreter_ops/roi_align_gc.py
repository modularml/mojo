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

"""Graph-compiler roi_align model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

CPU only: ``mo.roi_align`` carries the ``MO_HostOnly`` trait, so the handler
never receives a device buffer. (The deleted ``roi_align_ops.mojo`` binding had
a GPU branch, but ``MO_HostOnly`` meant the interpreter could never reach it --
migrating CPU-only loses no reachable path.)

Unlike every other family in this package, ``aligned`` and ``mode`` are genuine
MLIR attributes and the GC kernel takes them as comptime parameters
(``builtin_kernels/kernels.mojo``), not runtime operands, so they ride in the
cache key alongside dtype: one graph per (dtype, aligned, mode). The four
numeric scalars (``output_height``, ``output_width``, ``spatial_scale``,
``sampling_ratio``) stay ordinary runtime operands. The public ``ops.roi_align``
folds them into graph constants, which would grow the cache key with every
distinct value, so this module emits ``rmo.MoRoiAlignOp`` directly instead (the
same reason ``band_part_gc`` bypasses ``ops.band_part``).

The output's ROI and spatial dims depend on two host operands, so the graph
declares them symbolic and the op's registered shape function
(``roi_align_shape``) sizes them -- the same mechanism ``resize_gc`` and
``pooling_gc`` already rely on.
"""

from dataclasses import dataclass

from max import engine
from max._core.dialects import kgen, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

_GRAPH_BASE_NAME = "roi_align"

_CPU_DEVICES = [d for d in gc_compile.DISCOVERED_DEVICES if d.label == "cpu"]


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType
    """The shared ``input``/``rois``/output float dtype."""
    aligned: bool
    mode: str
    """``"AVG"`` or ``"MAX"``."""

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype.name}_aligned{int(self.aligned)}_{self.mode}"
        )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype, aligned, mode)
    for device in _CPU_DEVICES
    for dtype in gc_compile.CPU_FLOAT_DTYPES
    for aligned in (False, True)
    for mode in ("AVG", "MAX")
]


def _roi_align_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one symbolic roi_align graph into *module* in-place."""
    cpu = DeviceRef.CPU()
    graph = Graph(
        target.graph_name,
        input_types=[
            TensorType(target.dtype, ["n", "h", "w", "c"], device=cpu),
            TensorType(target.dtype, ["m", 5], device=cpu),
            # Rank 0 matches MO_RankedScalarIndexTensor /
            # MO_RankedScalarFloatTensor; MO_HostOnly puts all four on host.
            TensorType(DType.int64, [], device=cpu),
            TensorType(DType.int64, [], device=cpu),
            TensorType(DType.float32, [], device=cpu),
            TensorType(DType.float32, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        x, rois, oh, ow, scale, ratio = (v.tensor for v in graph.inputs)
        out_type = TensorType(target.dtype, ["m", "oh", "ow", "c"], device=cpu)
        graph.output(
            Graph.current._add_op_generated(
                rmo.MoRoiAlignOp,
                out_type,
                x,
                rois,
                oh,
                ow,
                scale,
                ratio,
                target.aligned,
                target.mode,
                kgen.ParamDeclArrayAttr([]),
            )[0].tensor
        )


class _RoiAlignFamily(gc_compile.GCFamilySpec):
    name = "roi_align"

    def sweep_devices(self) -> list[Device]:
        # MO_HostOnly: never sweep accelerators, or the warm producer compiles
        # an empty module per GPU lane (see resize_gc's CPU-only family).
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
                _roi_align_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_RoiAlignFamily())
gc_compile.register_family(_FAMILY)


def roi_align_model(dtype: DType, aligned: bool, mode: str) -> engine.Model:
    """Returns the roi_align :class:`~max.engine.Model` for the given key.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        dtype: The shared ``input``/``rois``/output float dtype.
        aligned: Whether to apply the half-pixel coordinate offset.
        mode: The pooling mode, ``"AVG"`` or ``"MAX"``.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: If *dtype* is not a supported CPU float, or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not precompiled.
    """
    device = _CPU_DEVICES[0]
    target = CompilationTarget(device, dtype, aligned, mode)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if dtype not in gc_compile.CPU_FLOAT_DTYPES:
            return f"Unsupported roi_align dtype {dtype}; key {key!r}."
        return None

    return _FAMILY.model_for(
        key,
        device,
        lambda m: _roi_align_graph(m, target),
        unsupported_reason=check_supported,
    )
