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

"""Graph-compiler arg_nonzero model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

CPU only: ``mo.arg_nonzero`` carries the ``MO_HostOnly`` trait, so the handler
never receives a device buffer.

The compiled graph is canonicalized to rank 1. arg_nonzero returns coordinates
in the input's index space, so the handler flattens the input to a symbolic
``["n"]``, runs one flat graph, and unravels the flat indices back into
``[nnz, rank]`` row-major coordinates on the host with ``np.unravel_index``
against the real input shape. arg_nonzero emits indices in ascending
row-major order, so flattening preserves ordering.

The output row count is data-dependent: the op's registered shape function
returns the exact ``nnz`` (``builtin_kernels/reductions.mojo``), so the model
sizes its own ``[nnz, 1]`` flat output without a count-then-fill pass.

Dtype axis: floats keep their real dtype, never bitcast -- ``-0.0`` is zero as
a float but a nonzero bit pattern as an unsigned int, so a bitcast would
misclassify it. Signed ints, unsigned ints, and bool each have a unique zero
bit pattern, so they ride the four uint widths instead
(:func:`gc_compile.uint_view_dtype` / :data:`gc_compile.WIDTH_DTYPES`), exactly
as ``band_part_gc`` does for its bit-width-keyed masking. That's
:data:`gc_compile.CPU_FLOAT_DTYPES` (2) plus 4 uint widths = 6 graphs total.
"""

from dataclasses import dataclass

from max import engine
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "nonzero"

_CPU_DEVICES = [d for d in gc_compile.DISCOVERED_DEVICES if d.label == "cpu"]


def _supported_dtypes() -> list[DType]:
    """Returns the CPU dtypes arg_nonzero sweeps.

    The shared CPU float policy (:data:`gc_compile.CPU_FLOAT_DTYPES`), plus
    the four uint widths that stand in for every signed int, unsigned int, and
    bool width (see the module docstring). This is narrower than the deleted
    ``argnonzero_ops.mojo`` binding, which dispatched through the
    interpreter's generic dtype-dispatch helper and also handled
    float16/bfloat16 on CPU.
    """
    return gc_compile.CPU_FLOAT_DTYPES + gc_compile.WIDTH_DTYPES


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType
    """The real dtype for floats, or the uint width it's viewed as for
    int/bool (see :func:`gc_compile.uint_view_dtype`)."""

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype.name}"
        )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype)
    for device in _CPU_DEVICES
    for dtype in _supported_dtypes()
]


def _nonzero_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one flat rank-1 arg_nonzero graph into *module* in-place."""
    graph = Graph(
        target.graph_name,
        input_types=[
            TensorType(
                target.dtype, ["n"], device=DeviceRef.from_device(target.device)
            )
        ],
        module=module,
    )
    with graph:
        (x,) = (v.tensor for v in graph.inputs)
        graph.output(graph_ops.nonzero(x, "nnz"))


class _NonzeroFamily(gc_compile.GCFamilySpec):
    name = "nonzero"

    def sweep_devices(self) -> list[Device]:
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
        # sweep_devices() already excludes accelerators (MO_HostOnly).
        for target in _COMPILATION_TARGETS:
            if (
                target.device.label == device.label
                and target.device.id == device.id
            ):
                _nonzero_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_NonzeroFamily())
gc_compile.register_family(_FAMILY)


def nonzero_model(dtype: DType) -> engine.Model:
    """Returns the flat arg_nonzero :class:`~max.engine.Model` for *dtype*.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        dtype: The input element dtype: a real float dtype, or the uint width
            an int/bool input is bit-cast to (see :func:`gc_compile.
            uint_view_dtype`).

    Returns:
        The compiled :class:`~max.engine.Model`. Its graph takes a flat
        ``["n"]`` input and returns ``[nnz, 1]`` flat indices; the caller
        unravels them against the real input shape.

    Raises:
        KeyError: If *dtype* is outside the supported set, or, with
            ``MAX_EAGER_OP_PRECOMPILE=1``, if the target was not precompiled.
    """
    device = _CPU_DEVICES[0]
    target = CompilationTarget(device, dtype)
    key = target.graph_name
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if dtype not in _supported_dtypes():
            supported = ", ".join(str(d) for d in _supported_dtypes())
            return (
                f"arg_nonzero does not support dtype {dtype}; supported:"
                f" {supported}. (key {key!r})"
            )
        return None

    return _FAMILY.model_for(
        key,
        device,
        lambda m: _nonzero_graph(m, target),
        unsupported_reason=check_supported,
    )
