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

"""Graph-compiler select (ternary where) model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

``cond`` is always ``bool`` (``rmo.SelectOp``'s verifier requires it), so only
the ``x``/``y`` value dtype keys the cache; there is no separate cond-dtype
axis. The swept value-dtype set matches every dtype the deleted
``select_ops.mojo`` handled: float + signed int + unsigned int + bool (``x``/
``y`` can themselves be bool tensors, unlike any binary arithmetic op).
``gc_compile.float_dtypes`` already excludes float64 on GPU, matching the
deleted kernel's explicit GPU float64 rejection.
"""

from dataclasses import dataclass

from max import engine
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "select"

canonical_shape = gc_compile.canonical_shape_rank1


@dataclass(frozen=True)
class CompilationTarget:
    device: Device
    dtype: DType

    @property
    def graph_name(self) -> str:
        return (
            f"{_GRAPH_BASE_NAME}_{self.device.label}_{self.device.id}_"
            f"{self.dtype}"
        )


def _supported_dtypes(device: Device) -> list[DType]:
    """Returns every value dtype the deleted ``select_ops.mojo`` handled on
    *device*.
    """
    return (
        gc_compile.float_dtypes(device)
        + gc_compile.SIGNED_INT_DTYPES
        + gc_compile.UNSIGNED_INT_DTYPES
        + [DType.bool]
    )


_COMPILATION_TARGETS = [
    CompilationTarget(device, dtype)
    for device in gc_compile.DISCOVERED_DEVICES
    for dtype in _supported_dtypes(device)
]


def _select_graph(module: Module, target: CompilationTarget) -> None:
    """Adds one fully-symbolic rank-1 select graph into *module* in-place."""
    device_ref = DeviceRef.from_device(target.device)
    cond_type = TensorType(DType.bool, ["n"], device=device_ref)
    value_type = TensorType(target.dtype, ["n"], device=device_ref)
    graph = Graph(
        target.graph_name,
        input_types=[cond_type, value_type, value_type],
        module=module,
    )
    with graph:
        cond, x, y = graph.inputs
        graph.output(graph_ops.where(cond.tensor, x.tensor, y.tensor))


class _SelectFamily(gc_compile.GCFamilySpec):
    name = "select"

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
                _select_graph(module, target)
        return module


_FAMILY = gc_compile.GCOpFamily(_SelectFamily())
gc_compile.register_family(_FAMILY)


def select_model(device: Device, dtype: DType) -> engine.Model:
    """Returns the select :class:`~max.engine.Model` for *device* and *dtype*.

    Lazy by default (compiled and cached on first use); the first miss adopts
    a whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure
    lookup.

    Args:
        device: The target device (CPU or GPU accelerator).
        dtype: The ``x``/``y`` value dtype (``cond`` is always bool).

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        EagerLazyCompileDisallowed: If the target is not already compiled and
            ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.

    Note:
        No support guard (like matmul): every swept dtype is expected to
        compile cleanly, so there is no conditionally-unsupported target.
    """
    target = CompilationTarget(device, dtype)
    key = target.graph_name
    # Checked before the lambda below is built: this runs on every dispatch,
    # so a hit must not pay for a closure it won't use.
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model
    return _FAMILY.model_for(key, device, lambda m: _select_graph(m, target))
