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

"""Graph-compiler matmul model cache for the MO interpreter.

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`):

- **Lazy per-target (default).** First dispatch for a (device, dtype) compiles
  just that target's fully-symbolic rank-3 batched-matmul graph.
- **Precompile sweep (``=1``).** The batched sweep compiles the full matrix at
  import; a :func:`matmul_model` miss is then a hard error.

Lazy mode avoids a trivial matmul JIT-compiling the whole kernel library on a
cold cache (~3000+ kernels, minutes; MXF-508). Models serve the eager
``mo.matmul`` / ``mo.batch_matmul`` handler via :func:`matmul_model`. Must not
import from ``handlers.py``.
"""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

from max import engine
from max._interpreter_ops import gc_compile
from max.driver import Device, DeviceSpec, accelerator_count, load_devices
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType
from max.graph import ops as graph_ops

_GRAPH_BASE_NAME = "matmul"

_ACCELERATOR_DEVICES = load_devices(
    [DeviceSpec.accelerator(i) for i in range(accelerator_count())]
)

_ACCELERATOR_DTYPES = [DType.float32, DType.float16, DType.bfloat16]

_CPU_DEVICES = load_devices([DeviceSpec.cpu()])

# Conservative set proven to compile on every CI architecture. float16/bfloat16
# fail matmul kernel codegen on ARM, so widen only with per-arch CI confirmation.
_CPU_DTYPES = [
    DType.float32,
    DType.float64,
    DType.int8,
    DType.int16,
    DType.int32,
    DType.int64,
]


@dataclass(frozen=True)
class CompilationTarget:
    graph_op_name: str
    device: Device
    # A single dtype shared by both operands. In principle lhs and rhs can
    # have different dtypes. In that case extend the dataclass
    dtype: DType

    @property
    def graph_name(self) -> str:
        """Returns the string used both as the graph ``sym_name`` and cache key."""
        return f"{self.graph_op_name}_{self.device.label}_{self.device.id}_{self.dtype}"


_COMPILATION_TARGETS = [
    CompilationTarget(_GRAPH_BASE_NAME, device, dtype)
    for device, dtype in itertools.chain(
        itertools.product(_CPU_DEVICES, _CPU_DTYPES),
        itertools.product(_ACCELERATOR_DEVICES, _ACCELERATOR_DTYPES),
    )
]


def canonical_shape(shape: Sequence[int]) -> tuple[int, int, int]:
    """Flattens an arbitrary-rank matmul operand to canonical rank 3.

    ``[d0, ..., dn, i, j]`` becomes ``(d0*...*dn, i, j)``; a rank-2 ``[i, j]``
    becomes ``(1, i, j)`` because ``prod(())`` is the empty product ``1``,
    keeping the rank-2 case branchless.
    """
    *batch_dims, i, j = shape
    return (prod(batch_dims), i, j)


def _matmul_graph(
    module: Module, compilation_target: CompilationTarget
) -> None:
    """Adds one fully-symbolic rank-3 matmul graph into *module* in-place."""
    device_ref = DeviceRef.from_device(compilation_target.device)
    lhs_type = TensorType(
        compilation_target.dtype, ["batch", "m", "k"], device=device_ref
    )
    rhs_type = TensorType(
        compilation_target.dtype, ["batch", "k", "n"], device=device_ref
    )
    graph_name = compilation_target.graph_name
    graph = Graph(graph_name, input_types=[lhs_type, rhs_type], module=module)
    with graph:
        lhs, rhs = graph.inputs
        graph.output(graph_ops.matmul(lhs.tensor, rhs.tensor))


class _MatmulFamily(gc_compile.GCFamilySpec):
    name = "matmul"

    def build_module(self) -> Module:
        """Build the full batched matmul module: every ``_COMPILATION_TARGETS``
        slot (CPU + all accelerators, all dtypes) in one module.

        Host-ELF and cubins both embed self-contained in the exported MEF, so
        one force-load populates every device class at once. Shared by the
        warm producer (export) and the batched sweep (compile into cache).
        """
        module = Module()
        for device in self.sweep_devices():
            self.build_module_for_device(device, module)
        return module

    def build_module_for_device(
        self, device: Device, module: Module | None = None
    ) -> Module:
        """Build the matmul module for a single device slot: every dtype
        target on *device* (matched by label + id), and nothing else.

        Per-slot counterpart of :meth:`build_module`. The warm producer
        exports one MEF per slot so the warm is device-count-independent: a
        k-GPU consumer force-loads only slots ``0..k-1``, letting a warm made
        for a higher count still adopt.
        """
        if module is None:
            module = Module()
        for compilation_target in _COMPILATION_TARGETS:
            if (
                compilation_target.device.label == device.label
                and compilation_target.device.id == device.id
            ):
                _matmul_graph(module, compilation_target)
        return module


_FAMILY = gc_compile.GCOpFamily(_MatmulFamily())
gc_compile.register_family(_FAMILY)


def matmul_model(device: Device, dtype: DType) -> engine.Model:
    """Return the matmul :class:`~max.engine.Model` for *device* + *dtype*.

    Lazy by default (compiled and cached on first use); the first miss adopts a
    whole warm cache. ``MAX_EAGER_OP_PRECOMPILE=1`` makes this a pure lookup.

    Args:
        device: The target device (CPU or GPU accelerator).
        dtype: The element dtype for both operands.

    Returns:
        The compiled :class:`~max.engine.Model`.

    Raises:
        EagerLazyCompileDisallowed: If the target is not already compiled and
            ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.

    Note:
        No support guard (unlike unary): RMO->MO casts both operands to a
        common backend-compilable dtype, so no target is unsupported.
    """
    target = CompilationTarget(_GRAPH_BASE_NAME, device, dtype)
    key = target.graph_name
    # Cache-check before building the lambda below: this runs on every eager
    # op dispatch, so a hit must not pay for a closure it won't use.
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model
    return _FAMILY.model_for(key, device, lambda m: _matmul_graph(m, target))
