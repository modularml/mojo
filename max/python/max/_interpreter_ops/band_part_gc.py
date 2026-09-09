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

"""Graph-compiler band_part model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

The mask depends only on the last two axes, so the handler collapses the leading
batch axes into one (a zero-copy view) and every input compiles as rank 3.
``num_lower``/``num_upper``/``exclude`` stay runtime host operands -- the GC
kernel takes them as tensors, not comptime params
(``builtin_kernels/linalg.mojo``) -- so one graph serves every mask, and the
cache key does not grow with their values. The public ``ops.band_part`` folds
them into graph constants instead, hence the direct ``rmo`` emission below (the
same reason ``shape_rearrange_gc`` emits ``rmo.MoTileOp`` by hand).

band_part only ever copies an element or writes a zero, so a tensor is masked by
bit width: the handler bit-casts every dtype to the same-width unsigned int (see
:func:`gc_compile.uint_view_dtype`) and views the result back. Four graphs per
device therefore cover every dtype, including float16, which has no typed CPU
kernel.
"""

from dataclasses import dataclass

from max import engine
from max._core.dialects import kgen, rmo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType

_GRAPH_BASE_NAME = "band_part"


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType
    """The uint width the masked tensor is viewed as, not its real dtype."""

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype.name}"
        )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype)
    for device in gc_compile.DISCOVERED_DEVICES
    for dtype in gc_compile.WIDTH_DTYPES
]


def _band_part_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one symbolic rank-3 band_part graph into *module* in-place."""
    device_ref = DeviceRef.from_device(target.device)
    cpu = DeviceRef.CPU()
    x_type = TensorType(target.dtype, ["b", "m", "n"], device=device_ref)
    graph = Graph(
        target.graph_name,
        input_types=[
            x_type,
            # Rank 0 matches the op's scalar operand types; all three are
            # host operands.
            TensorType(DType.int64, [], device=cpu),
            TensorType(DType.int64, [], device=cpu),
            TensorType(DType.bool, [], device=cpu),
        ],
        module=module,
    )
    with graph:
        data, num_lower, num_upper, exclude = (v.tensor for v in graph.inputs)
        masked = Graph.current._add_op_generated(
            rmo.MoLinalgBandPartOp,
            x_type,
            data,
            num_lower,
            num_upper,
            exclude,
            kgen.ParamDeclArrayAttr([]),
        )[0].tensor
        graph.output(masked)


class _BandPartFamily(gc_compile.GCFamilySpec):
    name = "band_part"

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
                _band_part_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_BandPartFamily())
gc_compile.register_family(_FAMILY)


def band_part_model(device: Device, dtype: DType) -> engine.Model:
    """Returns the band_part :class:`~max.engine.Model` for *device* and *dtype*.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: The target device (CPU or GPU accelerator).
        dtype: The uint width the masked tensor is viewed as (see
            :func:`gc_compile.uint_view_dtype`), not its original dtype.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        KeyError: With ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not
            precompiled.

    Note:
        No support guard (like matmul): *dtype* only ever comes from
        :func:`gc_compile.uint_view_dtype`, which raises ``NotImplementedError``
        itself for anything outside :data:`gc_compile.WIDTH_DTYPES`, so there is
        no conditionally-unsupported target left to guard here.
    """
    target = CompilationTarget(device, dtype)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model
    return _FAMILY.model_for(key, device, lambda m: _band_part_graph(m, target))
