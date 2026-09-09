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

"""Graph-compiler top-k/bottom-k model cache for the MO interpreter.

Covers ``mo.TopKOp``/``mo.BottomKOp``. ``k`` is a runtime host operand on
these ops, not a compile-time attribute (see ``MO_SelectKLikeOp`` in
``MOOps.td``), so ONE compiled graph per ``(op, device, dtype)`` serves every
``k``, the same way ``matmul_gc`` serves every M/K/N. The public
``ops.top_k``/``ops.bottom_k`` graph-builder API can't express that -- it
bakes ``k`` in as a compile-time constant -- so both ops are built directly
via ``_add_op_generated`` instead, the same pattern ``ops.split`` uses for
``mo.SplitOp`` (there's also no ``rmo`` MO-analogue for top-k to go through,
unlike ``rmo.MoBottomKOp`` for bottom-k).

``axis`` is canonicalized to the shared rank-3 view (see
``gc_compile.canonical_rank3``) and applied directly at axis=1 on CPU. GPU
only supports the innermost axis, so the middle axis is transposed to last
first -- expressed as the *positive* literal 2, never ``-1``: a
graph-compiler shape-function bug crashes whenever a negative axis reaches
it with a genuinely dynamic ``k``; the positive literal sidesteps it
entirely, verified correct on both Apple Metal and NVIDIA B200 (see the
comment at the axis constant in ``_topk_graph``).

``sorted`` is always baked in as ``True``: the deleted Mojo kernel always
sorted regardless of the caller's flag, and so does the static ``rmo.top_k``
legalization (``LowerTopK``).

Two compile modes, selected by ``MAX_EAGER_OP_PRECOMPILE`` (see
:func:`gc_compile.should_precompile`). Must not import from ``handlers.py``.

Dtype sweep: floats + int/uint, no bool (top-k's ordering comparison has no
bool kernel). GPU further narrows ints to 32/64-bit -- see
``_WIDE_INT_DTYPES``.
"""

from max import _core, engine
from max._core.dialects import kgen, mo
from max._interpreter_ops import gc_compile
from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef, Graph, Module, TensorType, ops

# Both ops build identically (see _topk_graph), so there's no per-op data to
# carry, unlike reduce_axis_gc/pooling_gc's per-op specs.
TOPK_GC_OPS: tuple[type[_core.Operation], ...] = (mo.TopKOp, mo.BottomKOp)

_TOPK_OPS_BY_NAME: dict[str, type[_core.Operation]] = {
    op_type.__name__: op_type for op_type in TOPK_GC_OPS
}


def _is_registered(op_type: type[_core.Operation]) -> bool:
    return gc_compile.spec_for(op_type, _TOPK_OPS_BY_NAME) is not None


# The GPU kernel's warp-shuffle primitive only compiles for 32/64-bit widths
# ("unhandled shuffle dtype" for 8/16-bit int, verified empirically by sweeping
# every dtype) -- the same narrowing ``reduce_axis_gc`` applies for CUDA's
# reduction kernels (there: "8/16-bit int reduce fails to compile").
_WIDE_INT_DTYPES = [DType.int32, DType.int64, DType.uint32, DType.uint64]


def _int_dtypes(device: Device) -> list[DType]:
    """Integer dtypes top_k/bottom_k support on *device*.

    CPU handles every width; GPU is narrowed to 32/64-bit -- see
    ``_WIDE_INT_DTYPES``.
    """
    if device.label == "cpu":
        return gc_compile.SIGNED_INT_DTYPES + gc_compile.UNSIGNED_INT_DTYPES
    return _WIDE_INT_DTYPES


def _supported_dtypes(device: Device) -> list[DType]:
    """The dtype set top_k/bottom_k sweep on *device*: floats + supported
    int/uint widths (see ``_int_dtypes``), no bool (top-k's ordering
    comparison has no bool kernel -- the deleted Mojo kernel rejected it the
    same way via ``DType.is_numeric()``).
    """
    return gc_compile.float_dtypes(device) + _int_dtypes(device)


def _is_supported(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> bool:
    return _is_registered(op_type) and dtype in _supported_dtypes(device)


canonical_rank3 = gc_compile.canonical_rank3


def _graph_name(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> str:
    """Graph ``sym_name`` and cache key for one (op, device, dtype)."""
    name = gc_compile.canonical_op_name(op_type, _TOPK_OPS_BY_NAME)
    return f"{name}_{device.label}_{device.id}_{dtype.name}"


def _topk_graph(
    module: Module,
    op_type: type[_core.Operation],
    device: Device,
    dtype: DType,
) -> None:
    """Adds one fully-symbolic rank-3 top-k/bottom-k graph into *module*.

    See the module docstring for why ``k`` stays a runtime operand and why
    GPU transposes to the positive-literal innermost axis.
    """
    device_ref = DeviceRef.from_device(device)
    cpu = DeviceRef.CPU()
    in_type = TensorType(dtype, ["d0", "d1", "d2"], device=device_ref)
    k_type = TensorType(DType.int64, [], device=cpu)
    graph = Graph(
        _graph_name(op_type, device, dtype),
        input_types=[in_type, k_type],
        module=module,
    )
    with graph:
        x, k = (v.tensor for v in graph.inputs)
        is_sorted = ops.constant(True, dtype=DType.bool, device=cpu)
        on_cpu = device.label == "cpu"
        op_input = x if on_cpu else ops.transpose(x, 1, 2)
        # KERN-3266: a negative axis literal crashes the GPU shape function
        # when k is a genuine runtime operand (not a foldable constant); GPU
        # always uses the positive equivalent instead.
        axis = ops.constant(1 if on_cpu else 2, dtype=DType.int64, device=cpu)
        val_dims = ["d0", "k", "d2"] if on_cpu else ["d0", "d2", "k"]
        out_type = TensorType(dtype, val_dims, device=device_ref)
        idx_type = TensorType(DType.int64, val_dims, device=device_ref)
        # MXF-555: tracks giving operand-based RMO ops (this one included) a
        # public graph-API entry point, so eager callers don't need
        # _add_op_generated directly.
        vals, idxs = Graph.current._add_op_generated(
            op_type,
            out_type,
            idx_type,
            op_input,
            k,
            axis,
            is_sorted,
            kgen.ParamDeclArrayAttr([]),
        )
        vals_tensor, idxs_tensor = vals.tensor, idxs.tensor
        if not on_cpu:
            vals_tensor = ops.transpose(vals_tensor, 1, 2)
            idxs_tensor = ops.transpose(idxs_tensor, 1, 2)
        graph.output(vals_tensor, idxs_tensor)


class _TopKFamily(gc_compile.GCFamilySpec):
    name = "topk"

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
        for op_type in TOPK_GC_OPS:
            for dtype in _supported_dtypes(device):
                _topk_graph(module, op_type, device, dtype)
        return module


_FAMILY = gc_compile.GCOpFamily(_TopKFamily())
gc_compile.register_family(_FAMILY)


def topk_model(
    op_type: type[_core.Operation], device: Device, dtype: DType
) -> engine.Model:
    """Returns the top-k/bottom-k :class:`~max.engine.Model` for the target
    (lazy by default; see the module docstring).

    Args:
        op_type: The concrete ``mo.TopKOp``/``mo.BottomKOp`` type being
            handled.
        device: The realized input's device.
        dtype: The realized input's dtype.

    Returns:
        The compiled model ready for execution.

    Raises:
        KeyError: If the (op, device, dtype) is outside the supported set.
        EagerLazyCompileDisallowed: If a supported target is not already
            compiled and ``MAX_EAGER_ALLOW_LAZY_COMPILE=0``.
    """
    key = _graph_name(op_type, device, dtype)
    model = _FAMILY.cache.get(key)
    if model is not None:
        return model

    def check_supported() -> str | None:
        if _is_supported(op_type, device, dtype):
            return None
        return (
            f"Unsupported top-k op/device/dtype for key {key!r}."
            f"  Supported dtypes for this device: {_supported_dtypes(device)}"
        )

    def build(module: Module) -> None:
        assert _is_registered(op_type), (
            f"unsupported op {op_type!r} reached compile"
        )
        _topk_graph(module, op_type, device, dtype)

    return _FAMILY.model_for(
        key, device, build, unsupported_reason=check_supported
    )
