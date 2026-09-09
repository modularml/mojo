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

"""Op handlers for the MO graph interpreter.

This module contains the operation handlers that implement the actual
computation for MO operations. Each handler takes the interpreter instance,
the operation, and input buffers, and returns output buffers.

Handlers are registered using the @register_op_handler decorator.
"""

from collections.abc import Callable, Sequence
from math import prod
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from max import _core, graph
from max._core.dialects import builtin, kgen, mo, mosh
from max._interpreter_ops import (
    band_part_gc,
    cast_gc,
    conv_gc,
    data_movement_gc,
    elementwise_binary_gc,
    gather_gc,
    gc_compile,
    group_norm_gc,
    layer_norm_gc,
    matmul_gc,
    nms_gc,
    nonzero_gc,
    pooling_gc,
    random_gc,
    range_gc,
    reduce_axis_gc,
    resize_gc,
    rms_norm_gc,
    roi_align_gc,
    scatter_gc,
    scatter_nd_gc,
    select_gc,
    shape_rearrange_gc,
    topk_gc,
    unary_elementwise_gc,
)
from max.driver import CPU, Buffer, Device
from max.dtype import DType


class _HasAxis(Protocol):
    """Structural type for ops carrying a compile-time ``axis`` int attribute.

    The reduce/softmax handlers are written against the generic
    ``_core.Operation`` base (a single handler serves several concrete op
    types), but every such op exposes an ``axis`` int property. Casting to
    this protocol lets the handlers read ``op.axis`` in a type-safe way.
    """

    @property
    def axis(self) -> int: ...


# Type alias for op handlers
# Signature: (op, input_buffers) -> output_buffers
OpHandler = Callable[
    [Any, Sequence[Buffer | None]],
    Sequence[Buffer | None],
]

# Op handler registries
# Maps operation types to handler functions (for isinstance checks)
_MO_OP_HANDLERS: dict[type[_core.Operation], OpHandler] = {}
# Maps operation names to handler functions (for name-based lookup fallback)
_MO_OP_NAME_HANDLERS: dict[str, OpHandler] = {}


def register_op_handler(
    op_type: type[_core.Operation],
) -> Callable[[OpHandler], OpHandler]:
    """Decorator to register an MO op handler.

    Args:
        op_type: The MO operation class to handle (e.g., mo.AddOp).

    Returns:
        Decorator function that registers the handler.

    Example:
        @register_op_handler(mo.AddOp)
        def _handle_add(op, inputs):
            # Implementation
            return [output_buffer]
    """

    def decorator(fn: OpHandler) -> OpHandler:
        _MO_OP_HANDLERS[op_type] = fn
        # Also register by name for fallback lookup
        # Register both the direct name (e.g., "ExpOp") and with "Mo" prefix
        # (e.g., "MoExpOp") since nanobind may use either convention
        name = op_type.__name__
        _MO_OP_NAME_HANDLERS[name] = fn
        # Also register with "Mo" prefix for runtime compatibility
        if not name.startswith("Mo"):
            _MO_OP_NAME_HANDLERS[f"Mo{name}"] = fn
        return fn

    return decorator


def lookup_handler(op: _core.Operation) -> OpHandler | None:
    """Look up the handler for an operation.

    First tries type-based lookup, then falls back to name-based lookup
    to handle cases where nanobind creates different class objects.

    Args:
        op: The operation to look up.

    Returns:
        The handler function, or None if no handler exists.
    """
    # Try type-based lookup first
    if type(op) in _MO_OP_HANDLERS:
        return _MO_OP_HANDLERS[type(op)]

    # Fallback: try name-based lookup
    op_class_name = type(op).__name__
    if op_class_name in _MO_OP_NAME_HANDLERS:
        return _MO_OP_NAME_HANDLERS[op_class_name]

    return None


def _check_cpu_only(op: _core.Operation, target_device: Device) -> None:
    """Check that operation is running on CPU (host device).

    Args:
        op: The operation being executed.
        target_device: The target device for execution.

    Raises:
        NotImplementedError: If target device is not CPU.
    """
    if not target_device.is_host:
        raise NotImplementedError(
            f"GPU execution not supported for {type(op).__name__} "
            "in MO interpreter"
        )


def _get_target_device(op: _core.Operation) -> Device:
    """Get the target device from an op's first result type.

    Accesses the device_ref directly from the MLIR type to avoid
    Shape.from_mlir() crashes on parametric shapes (ParamDeclRefAttr).

    Args:
        op: The operation whose result device to extract.

    Returns:
        The target device for the operation's result.
    """
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    return graph.DeviceRef.from_mlir(result_mlir_type.device_ref).to_device()


# Constant operations


@register_op_handler(mo.ConstantOp)
def _handle_constant(
    op: mo.ConstantOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.constant by materializing its value via C++ binding.

    Constants are mo.constant ops with embedded #M.dense_array values in the
    'value' attribute. Supported attribute types:
    - ArrayElementsAttr (#M.dense_array)
    - DenseResourceElementsAttr (external blob)
    - AlignedBytesAttr (#M.aligned_bytes)

    This implementation always copies data from the MLIR attribute into a new
    Buffer on CPU first, then transfers to the target device if needed.
    For splat constants (1 element in source, many in output), the single
    value is replicated on CPU before transfer.

    Args:
        op: The constant operation.
        inputs: Input buffers (empty for constants).

    Returns:
        List containing the materialized constant buffer.
    """
    # Extract the result type to get dtype and shape info
    result_type = graph.Type.from_mlir(op.results[0].type)
    assert isinstance(result_type, graph.TensorType)
    dtype = result_type.dtype
    shape = result_type.shape

    if not graph.Shape.is_static(shape):
        raise ValueError("Dynamic shapes not supported for constants")

    target_device = result_type.device.to_device()

    # Always create buffer on CPU first (C++ binding uses memcpy which
    # requires host memory). Splatting also happens on CPU.
    cpu_buffer = _core.graph._buffer_from_constant_attr(
        op.value, dtype, graph.Shape(shape).static_dims, CPU()
    )

    # Transfer to target device if not CPU
    if not target_device.is_host:
        return [cpu_buffer.to(target_device)]

    return [cpu_buffer]


@register_op_handler(mo.ConstantScalarOp)
def _handle_constant_scalar(
    op: mo.ConstantScalarOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.constant.scalar by extracting the scalar value attribute.

    Scalar constants have a ``value`` attribute that is an ``IntegerAttr``,
    ``FloatAttr``, or ``BoolAttr``.  The result type is ``!mo.scalar<dtype>``
    which we materialise as a rank-0 ``Buffer`` on CPU.

    Args:
        op: The constant scalar operation.
        inputs: Input buffers (empty for constants).

    Returns:
        List containing a rank-0 Buffer with the scalar value.
    """
    result_type: mo.ScalarType = op.results[0].type  # type: ignore[assignment]
    dtype = DType(result_type.dtype)

    attr = op.value
    value: bool | int | float
    if isinstance(
        attr, builtin.BoolAttr | builtin.IntegerAttr | builtin.FloatAttr
    ):
        value = attr.value
    else:
        raise ValueError(
            f"Unsupported scalar attribute type: {type(attr).__name__}"
        )

    np_val = np.array(value, dtype=dtype.to_numpy())
    # Rank-0 bool arrays are not supported by Buffer.from_dlpack;
    # wrap as int8 (same underlying representation).
    if np_val.dtype == np.bool_:
        np_val = np_val.view(np.int8)
    return [Buffer.from_numpy(np_val)]


# Mutable load operations


@register_op_handler(mo.MutableLoadOp)
def _handle_mutable_load(
    op: mo.MutableLoadOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.load by passing through the input buffer.

    mo.mutable.load reads from a buffer input. The handler receives the
    buffer as the first input (already resolved from slots by the dispatcher).
    The second input is the chain (None since chains are skipped).

    Args:
        op: The mutable load operation (unused).
        inputs: Input buffers - first is the buffer to load, second is the chain
            (None).

    Returns:
        List containing the loaded tensor buffer and None for the chain.
    """
    # MutableLoadOp produces (tensor, chain)
    # The interpreter executes sequentially, so chains are not needed.
    # Use None to avoid unnecessary buffer allocation.
    return [inputs[0], None]


# Mutable store operations


@register_op_handler(mo.MutableStoreOp)
def _handle_mutable_store(
    op: mo.MutableStoreOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.store by copying the tensor into the buffer.

    ``mo.mutable.store`` writes a full tensor value into a mutable tensor
    slot. Operand order is ``(in_buffer, in_tensor, in_chain)`` and the sole
    result is ``out_chain``. The interpreter represents chains as ``None``.

    Args:
        op: The mutable store operation (unused).
        inputs: Input buffers - ``(in_buffer, in_tensor, in_chain)``.
            ``in_chain`` is ``None`` since chain values are skipped.

    Returns:
        List containing ``None`` for the out_chain.
    """
    in_buffer = inputs[0]
    in_tensor = inputs[1]
    assert isinstance(in_buffer, Buffer)
    assert isinstance(in_tensor, Buffer)
    in_buffer.inplace_copy_from(in_tensor)
    return [None]


@register_op_handler(mo.MutableStoreSliceOp)
def _handle_mutable_store_slice(
    op: mo.MutableStoreSliceOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.mutable.store.slice via a graph-compiler model.

    Operand order: ``(in_buffer, slice, start, stop, step, in_chain)``;
    result is ``out_chain``. Numpy slice semantics apply to start/stop/step.
    The op has no tensor outputs: ``data_movement_gc.store_slice`` mutates
    ``in_buffer`` in place.

    Args:
        op: The mutable store slice operation.
        inputs: ``(in_buffer, slice, start, stop, step, in_chain)``.

    Returns:
        List containing ``None`` for the out_chain.
    """
    in_buffer = inputs[0]
    slice_tensor = inputs[1]
    start_buf = inputs[2]
    stop_buf = inputs[3]
    step_buf = inputs[4]
    assert isinstance(in_buffer, Buffer)
    assert isinstance(slice_tensor, Buffer)
    assert isinstance(start_buf, Buffer)
    assert isinstance(stop_buf, Buffer)
    assert isinstance(step_buf, Buffer)

    data_movement_gc.store_slice(
        in_buffer,
        slice_tensor,
        start_buf.to_numpy().astype(np.int64).flatten(),
        stop_buf.to_numpy().astype(np.int64).flatten(),
        step_buf.to_numpy().astype(np.int64).flatten(),
    )
    return [None]


# Transfer operations


@register_op_handler(mo.TransferOp)
def _handle_transfer(
    op: mo.TransferOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.transfer by transferring buffer between devices.

    TransferOp transfers tensor contents between devices (e.g. CPU<->GPU).
    When source and destination devices match and alwaysElideSameDeviceCopy is
    True, the result aliases the input. When the flag is False, a driver copy
    is made.

    Args:
        op: The transfer operation.
        inputs: Input buffers - first is the tensor to transfer, second is the
            chain (None).

    Returns:
        List containing the transferred tensor buffer and None for the chain.
    """
    assert isinstance(inputs[0], Buffer)
    input_buffer = inputs[0]
    target_device = _get_target_device(op)

    if input_buffer.device == target_device:
        if op.always_elide_same_device_copy:
            # Alias: return the input buffer directly (no copy).
            return [input_buffer, None]
        output = Buffer(
            shape=input_buffer.shape,
            dtype=input_buffer.dtype,
            device=target_device,
        )
        output.inplace_copy_from(input_buffer)
        return [output, None]

    # Cross-device transfer
    # TransferOp produces (tensor, chain)
    return [input_buffer.to(target_device), None]


# Buffer operations


@register_op_handler(mo.BufferCreateOp)
def _handle_buffer_create(
    op: mo.BufferCreateOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.buffer.create by allocating a zero-filled buffer.

    ``BufferCreateOp`` has no operands and a single ``!mo.buffer<shape, dtype,
    device>`` result.  Shape, dtype, and target device are extracted from the
    result type.  The interpreter allocates a zeroed buffer so that downstream
    ops (e.g. ``buffer.transfer``) have valid storage to write into.
    """
    result_type = graph.BufferType.from_mlir(
        list(op.results)[0].type  # type: ignore[arg-type]
    )
    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(
            "Dynamic shapes not supported for buffer.create in interpreter"
        )
    target_device = result_type.device.to_device()
    buf = Buffer(
        dtype=result_type.dtype,
        shape=graph.Shape(shape).static_dims,
        device=target_device,
    )
    return [buf]


@register_op_handler(mo.BufferTransferOp)
def _handle_buffer_transfer(
    op: mo.BufferTransferOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.buffer.transfer by copying src contents into dst.

    Operand order: ``(src, dst, inChain)``.  The operation copies data from
    ``src`` into ``dst`` (both must have matching shape and dtype).  The sole
    result is an ``outChain`` which the interpreter represents as ``None``.
    """
    src = inputs[0]
    dst = inputs[1]
    assert isinstance(src, Buffer)
    assert isinstance(dst, Buffer)
    dst.inplace_copy_from(src)
    return [None]


# Debug operations


@register_op_handler(mo.DebugPrintOp)
def _handle_debug_print(
    op: mo.DebugPrintOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.debug.print by printing the string value.

    DebugPrintOp has operands (inChain) and attributes (value, label).
    It produces (outChain). The interpreter prints the string to stdout.

    Args:
        op: The debug print operation.
        inputs: Input buffers - first is the chain (None).

    Returns:
        List containing None for the output chain.
    """
    label = op.label
    value = op.value
    if label:
        print(f"[{label}] {value}")
    else:
        print(value)
    return [None]


@register_op_handler(mo.DebugTensorPrintOp)
def _handle_debug_tensor_print(
    op: mo.DebugTensorPrintOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.debug.tensor.print by printing tensor data.

    DebugTensorPrintOp has operands (inChain, input_tensor) and attribute
    (label). It produces (outChain). The interpreter converts the tensor
    buffer to numpy and prints it to stdout.

    Args:
        op: The debug tensor print operation.
        inputs: Input buffers - first is the chain (None), second is the
            tensor Buffer.

    Returns:
        List containing None for the output chain.
    """
    tensor_buf = inputs[1]
    label = op.label
    if tensor_buf is not None:
        np_array = tensor_buf.to_numpy()
        if label:
            print(f"[{label}] {np_array}")
        else:
            print(np_array)
    else:
        tag = f"[{label}] " if label else ""
        print(f"{tag}<no tensor data>")
    return [None]


# Shape operations


@register_op_handler(mo.RebindOp)
def _handle_rebind(
    op: mo.RebindOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer | None]:
    """Handle mo.rebind by passing through the input buffer.

    Rebind is a shape assertion that doesn't change the underlying data.

    Args:
        op: The rebind operation (unused).
        inputs: Input buffers - contains the tensor to rebind.

    Returns:
        List containing the input buffer unchanged.
    """
    return [inputs[0]]


@register_op_handler(mo.StaticBroadcastToOp)
def _handle_static_broadcast_to(
    op: mo.StaticBroadcastToOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.static.broadcast_to via a data_movement_gc model.

    Args:
        op: The static broadcast operation.
        inputs: Input buffers - contains the tensor to broadcast.

    Returns:
        List containing the broadcast tensor buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    assert isinstance(inputs[0], Buffer)

    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )
    target_shape = graph.Shape(shape).static_dims

    return [
        data_movement_gc.broadcast_to(
            inputs[0], list(target_shape), target_device
        )
    ]


@register_op_handler(mo.BroadcastToOp)
def _handle_broadcast_to(
    op: mo.BroadcastToOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.broadcast_to via a data_movement_gc model.

    Supports both CPU and GPU tensors.

    Args:
        op: The broadcast operation.
        inputs: Input buffers - first is the tensor to broadcast,
            second (optional) is the target shape tensor.

    Returns:
        List containing the broadcast tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)

    # Try to get static shape from result type, fall through to dynamic
    # shape from the second input if the shape is parametric.
    target_shape = None
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    shape_attr = result_mlir_type.shape

    shape = graph.Shape.from_mlir(shape_attr)
    if graph.Shape.is_static(shape):
        target_shape = graph.Shape(shape).static_dims

    if target_shape is None and len(inputs) > 1:
        # For dynamic/parametric shapes, get from the shape operand
        assert isinstance(inputs[1], Buffer)
        target_shape = inputs[1].to_numpy().tolist()

    if target_shape is None:
        raise NotImplementedError(
            f"Cannot determine broadcast target shape for {op}"
        )

    return [
        data_movement_gc.broadcast_to(
            inputs[0], list(target_shape), target_device
        )
    ]


# Shared shape/stride helpers


def _row_major_strides(shape: list[int]) -> tuple[int, ...]:
    """Compute row-major (C-order) strides for the given shape.

    Args:
        shape: Tensor dimensions in order from outermost to innermost.

    Returns:
        Tuple of strides with the same length as ``shape``.
    """
    strides = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        strides[i] = strides[i + 1] * shape[i + 1]
    return tuple(strides)


# Helper for device validation


def _check_buffers_on_device(
    buffers: Sequence[Buffer | None], target_device: Device
) -> None:
    """Check that all non-None buffers are on the target device.

    Args:
        buffers: Sequence of buffers to check (None entries are skipped).
        target_device: The expected device for all buffers.

    Raises:
        ValueError: If any buffer is not on the target device.
    """
    for i, buf in enumerate(buffers):
        if buf is not None and buf.device != target_device:
            raise ValueError(
                f"Input buffer {i} is on {buf.device}, "
                f"but expected {target_device}."
            )


# Binary elementwise operations (arithmetic, bitwise, and comparison)


def _handle_binary_elementwise(
    op: _core.Operation, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager binary-elementwise op to its GC model.

    Models are compiled once per (op, device, input dtype) at rank 1 (see
    :func:`elementwise_binary_gc.binary_model`), so both operands are flattened
    around the call and the result reshaped back (zero-copy views). The two
    operands arrive already cast to a common dtype and broadcast to a common
    shape, so a single rank-1 dtype keys the model exactly. The output dtype is
    whatever the model produces — the same dtype for arithmetic/bitwise ops,
    ``bool`` for the comparison predicates.

    Args:
        op: The binary-elementwise operation being handled.
        inputs: The realized input buffers (``lhs``, ``rhs``).

    Returns:
        A single-element list holding the result buffer.
    """
    lhs, rhs = inputs
    assert isinstance(lhs, Buffer)
    assert isinstance(rhs, Buffer)
    # Validate against the op's declared result device (like the sibling
    # handlers), not just the operand's, so an upstream placement mismatch
    # fails loudly here instead of silently dispatching on the wrong device.
    target_device = _get_target_device(op)
    _check_buffers_on_device(inputs, target_device)

    model = elementwise_binary_gc.binary_model(
        type(op), target_device, lhs.dtype
    )
    shape = elementwise_binary_gc.canonical_shape(lhs.shape)
    lhs_view = lhs.view(lhs.dtype, shape)
    rhs_view = rhs.view(rhs.dtype, shape)
    (out,) = model(lhs_view, rhs_view)
    return [out.view(out.dtype, lhs.shape)]


# Wrapped in a function so the BINARY_GC_OPS access is deferred past the import
# cycle between this module and elementwise_binary_gc (via the package).
def _register_binary_elementwise_handlers() -> None:
    for op_type in elementwise_binary_gc.BINARY_GC_OPS:
        register_op_handler(op_type)(_handle_binary_elementwise)


_register_binary_elementwise_handlers()


# Unary elementwise operations


def _handle_unary_elementwise(
    op: _core.Operation, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager unary-elementwise op to its GC model.

    Models are compiled once per (op, device, input dtype) at rank 1 (see
    :func:`unary_elementwise_gc.unary_model`), so the operand is flattened
    around the call and reshaped back (zero-copy views). The output dtype is
    whatever the model produces — the same dtype for most ops, ``bool`` for the
    ``IsNan``/``IsInf`` predicates.

    Args:
        op: The unary-elementwise operation being handled.
        inputs: The realized input buffers; the first is the operand.

    Returns:
        A single-element list holding the result buffer.
    """
    x = inputs[0]
    assert isinstance(x, Buffer)

    model = unary_elementwise_gc.unary_model(type(op), x.device, x.dtype)
    x_view = x.view(x.dtype, unary_elementwise_gc.canonical_shape(x.shape))
    (out,) = model(x_view)
    return [out.view(out.dtype, x.shape)]


# Wrapped in a function so the UNARY_GC_OPS access is deferred past the import
# cycle between this module and unary_elementwise_gc (via the package).
def _register_unary_elementwise_handlers() -> None:
    for op_type in unary_elementwise_gc.UNARY_GC_OPS:
        register_op_handler(op_type)(_handle_unary_elementwise)


_register_unary_elementwise_handlers()


# Cast (any dtype -> any dtype)


@register_op_handler(mo.CastOp)
def _handle_cast(
    op: mo.CastOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager cast to its GC model.

    The model is compiled once per (device, src dtype, dst dtype) at rank 1 (see
    :func:`cast_gc.cast_model`), so the operand is flattened around the call and
    reshaped back (zero-copy views). The source dtype is the input buffer's; the
    target dtype comes from the MLIR result type, not the input.

    Args:
        op: The cast operation being handled.
        inputs: The realized input buffers; the first is the operand.

    Returns:
        A single-element list holding the result buffer.
    """
    x = inputs[0]
    assert isinstance(x, Buffer)

    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    model = cast_gc.cast_model(target_device, x.dtype, result_type.dtype)
    x_view = x.view(x.dtype, cast_gc.canonical_shape(x.shape))
    (out,) = model(x_view)
    return [out.view(out.dtype, x.shape)]


# Matrix operations


@register_op_handler(mo.MatmulOp)
@register_op_handler(mo.BatchMatmulOp)
def _handle_matmul(
    op: mo.MatmulOp | mo.BatchMatmulOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Routes eager matmul and batch_matmul through a GC model.

    Looks up the rank-3 batched-matmul :class:`~max.engine.Model` for the
    realized input's device and dtype (see :func:`matmul_gc.matmul_model`),
    view-shims the two operands to canonical rank 3, executes the model, and
    view-shims the output back to the result rank. All views are zero-copy.

    The same handler serves both ``mo.matmul`` (rank 2) and ``mo.batch_matmul``
    (rank 3+). The RMO->MO lowering already casts both operands to a common
    dtype and broadcasts them to equal batch/rank before these ops are created,
    so reading device, dtype, and the equal-batch shape from ``lhs`` is exact.

    Args:
        op: The ``mo.matmul`` or ``mo.batch_matmul`` operation being handled.
        inputs: The two realized operand buffers (``lhs``, ``rhs``).

    Returns:
        A single-element list holding the matmul result buffer.
    """
    lhs, rhs = inputs
    assert isinstance(lhs, Buffer)
    assert isinstance(rhs, Buffer)

    model = matmul_gc.matmul_model(lhs.device, lhs.dtype)

    # Forward rank shim: flatten leading dims to one batch dim (zero-copy).
    lhs_view = lhs.view(lhs.dtype, matmul_gc.canonical_shape(lhs.shape))
    rhs_view = rhs.view(rhs.dtype, matmul_gc.canonical_shape(rhs.shape))

    (out,) = model(lhs_view, rhs_view)

    # Inverse rank shim: restore the original leading dims.
    m, n = lhs.shape[-2], rhs.shape[-1]
    result_shape = (*lhs.shape[:-2], m, n)
    return [out.view(out.dtype, result_shape)]


# Shape manipulation operations


def _reshape_common(
    op: _core.Operation,
    inputs: Sequence[Buffer | None],
    op_name: str,
) -> Sequence[Buffer]:
    """Common implementation for reshape operations.

    Uses Buffer.view() to create a reshaped view sharing the underlying
    memory, supporting both CPU and GPU tensors without data movement.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_buffers_on_device(inputs, target_device)

    assert isinstance(inputs[0], Buffer)

    shape = result_type.shape
    if not graph.Shape.is_static(shape):
        raise NotImplementedError(f"Dynamic shapes not supported for {op_name}")
    target_shape = graph.Shape(shape).static_dims

    return [inputs[0].view(inputs[0].dtype, tuple(target_shape))]


@register_op_handler(mo.ReshapeOp)
def _handle_reshape(
    op: mo.ReshapeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.reshape."""
    return _reshape_common(op, inputs, "reshape")


@register_op_handler(mo.StaticReshapeOp)
def _handle_static_reshape(
    op: mo.StaticReshapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.static.reshape - reshape without inferred dimensions."""
    return _reshape_common(op, inputs, "static reshape")


@register_op_handler(mo.SqueezeShapeOp)
def _handle_squeeze_shape(
    op: mo.SqueezeShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.squeeze_shape - computes shape with specified dimensions removed.

    This is a CPU-side shape metadata operation. Given an input shape vector
    and a list of indices, returns a new shape vector with the indicated
    dimensions removed. The indicated dimensions must have size 1.

    Args:
        op: The squeeze shape operation.
        inputs: Input buffers - first is the shape vector, second is the
            indices tensor specifying which dimensions to remove.

    Returns:
        List containing the new shape vector as a 1D si64 buffer.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_shape = inputs[0].to_numpy().tolist()
    remove_indices = inputs[1].to_numpy().tolist()

    rank = len(input_shape)
    # Normalize negative indices
    normalized = set()
    for idx in remove_indices:
        idx = int(idx)
        if idx < 0:
            idx += rank
        normalized.add(idx)

    # Build output shape by removing indicated dimensions
    result_shape = [
        dim for i, dim in enumerate(input_shape) if i not in normalized
    ]
    result_np = np.array(result_shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.UnsqueezeShapeOp)
def _handle_unsqueeze_shape(
    op: mo.UnsqueezeShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.unsqueeze_shape - computes shape with size-1 dimensions inserted.

    This is a CPU-side shape metadata operation. Given an input shape vector
    of rank N and a list of M indices, returns a new shape vector of rank N+M
    where the indicated positions are filled with 1 and the original dimensions
    fill the remaining positions.

    Args:
        op: The unsqueeze shape operation.
        inputs: Input buffers - first is the shape vector, second is the
            padding indices tensor specifying where to insert size-1 dims.

    Returns:
        List containing the new shape vector as a 1D si64 buffer.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    input_shape = inputs[0].to_numpy().tolist()
    padding_indices = inputs[1].to_numpy().tolist()

    new_rank = len(input_shape) + len(padding_indices)
    # Normalize negative indices relative to the new rank
    normalized = set()
    for idx in padding_indices:
        idx = int(idx)
        if idx < 0:
            idx += new_rank
        normalized.add(idx)

    # Build output shape: insert 1s at indicated positions, fill rest from input
    result_shape = []
    input_idx = 0
    for i in range(new_rank):
        if i in normalized:
            result_shape.append(1)
        else:
            result_shape.append(int(input_shape[input_idx]))
            input_idx += 1

    result_np = np.array(result_shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.AddSingletonDimOp)
def _handle_add_singleton_dim(
    op: mo.AddSingletonDimOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.add_singleton_dim - adds a dimension of size 1 at the given axis.

    This is a shape-change op that does not copy data. It uses numpy.reshape
    with the target shape from the MLIR result type.

    Args:
        op: The add singleton dim operation.
        inputs: Input buffers - contains the tensor to reshape.

    Returns:
        List containing the reshaped tensor buffer.
    """
    return _reshape_common(op, inputs, "add_singleton_dim")


@register_op_handler(mo.SplitDimOp)
def _handle_split_dim(
    op: mo.SplitDimOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.split_dim - splits one dimension into two dimensions.

    E.g., a tensor of shape [N, K] with axis=0 becomes [S1, S2, K] where
    S1 * S2 = N. The target shape comes from the MLIR result type.

    Args:
        op: The split dim operation.
        inputs: Input buffers - contains the tensor to reshape.

    Returns:
        List containing the reshaped tensor buffer.
    """
    return _reshape_common(op, inputs, "split_dim")


@register_op_handler(mo.MergeDimOp)
def _handle_merge_dim(
    op: mo.MergeDimOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.merge_dim - merges two adjacent dimensions into one.

    E.g., a tensor of shape [A, B, C, D] with axis=1 becomes [A, B*C, D].
    The target shape comes from the MLIR result type.

    Args:
        op: The merge dim operation.
        inputs: Input buffers - contains the tensor to reshape.

    Returns:
        List containing the reshaped tensor buffer.
    """
    return _reshape_common(op, inputs, "merge_dim")


@register_op_handler(mo.TransposeOp)
def _handle_transpose(
    op: mo.TransposeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.transpose via a data_movement_gc graph-compiler model.

    Supports both CPU and GPU tensors.

    Args:
        op: The transpose operation.
        inputs: Input buffers - first is the tensor to transpose,
            second is the permutation tensor (int64 on CPU).

    Returns:
        List containing the transposed tensor buffer.
    """
    target_device = _get_target_device(op)

    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)

    perm = [int(p) for p in inputs[1].to_numpy()]
    return [data_movement_gc.transpose(inputs[0], perm, target_device)]


@register_op_handler(mo.SliceOp)
def _handle_slice(
    op: mo.SliceOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.slice via rmo.MoSliceOp with runtime starts/stops/steps.

    Inputs: (input, starts, stops, steps); the latter three are rank-1 int64
    tensors with one entry per dimension. Negative starts/stops are normalized
    to the input dims; steps must be positive.
    """
    assert len(inputs) >= 4, f"SliceOp expects 4 inputs, got {len(inputs)}"
    input_buffer, starts_b, stops_b, steps_b = inputs[:4]
    assert isinstance(input_buffer, Buffer)
    assert isinstance(starts_b, Buffer)
    assert isinstance(stops_b, Buffer)
    assert isinstance(steps_b, Buffer)

    start_np = starts_b.to_numpy().astype(np.int64)
    stop_np = stops_b.to_numpy().astype(np.int64)
    step_np = steps_b.to_numpy().astype(np.int64)
    assert (step_np > 0).all(), f"SliceOp steps must be positive, got {step_np}"

    input_shape_np = np.array(input_buffer.shape, dtype=np.int64)
    start_np = np.where(start_np < 0, start_np + input_shape_np, start_np)
    stop_np = np.where(stop_np < 0, stop_np + input_shape_np, stop_np)

    rank = len(start_np)
    udtype = gc_compile.uint_view_dtype(input_buffer.dtype)
    model = shape_rearrange_gc.model(
        mo.SliceOp, input_buffer.device, udtype, rank
    )
    (out,) = model(
        input_buffer.view(udtype, input_buffer.shape),
        Buffer.from_numpy(start_np),
        Buffer.from_numpy(stop_np),
        Buffer.from_numpy(step_np),
    )
    # Trust the GC result's shape: rmo.slice clamps open-ended stops at runtime.
    return [out.view(input_buffer.dtype, tuple(out.shape))]


# Shape/parameter operations


@register_op_handler(mo.ShapeOfOp)
def _handle_shape_of(
    op: mo.ShapeOfOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.shape_of - returns the shape of a tensor as a 1D si64 tensor.

    This is a CPU-side metadata operation. The result is always a CPU buffer
    regardless of the input tensor's device, since shape metadata is always
    host-accessible.
    """
    assert isinstance(inputs[0], Buffer)
    shape = inputs[0].shape
    result_np = np.array(shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.BroadcastShapeOp)
def _handle_broadcast_shape(
    op: mo.BroadcastShapeOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.broadcast_shape - compute broadcast shape of two shapes.

    This is a CPU-side metadata operation. The result is always a CPU buffer
    since it computes shape information from small integer tensors.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    shape_x = tuple(inputs[0].to_numpy().tolist())
    shape_y = tuple(inputs[1].to_numpy().tolist())
    result_shape = np.broadcast_shapes(shape_x, shape_y)
    result_np = np.array(result_shape, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mo.ShapeToTensorOp)
def _handle_shape_to_tensor(
    op: mo.ShapeToTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.shape.to_tensor - converts shape value to tensor.

    The input is a !mosh.ape shape value (already a buffer from ParamToValueOp).
    This op just passes through the buffer since ParamToValueOp already
    created a tensor representation.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()
    _check_cpu_only(op, target_device)

    # The input should already be a buffer containing the shape values
    # Just pass it through
    assert isinstance(inputs[0], Buffer)
    return [inputs[0]]


@register_op_handler(mo.ShapeFromTensorOp)
def _handle_shape_from_tensor(
    op: mo.ShapeFromTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.shape.from_tensor - converts a tensor to a shape value.

    The input is a rank-1 integer tensor containing shape dimension values.
    The output is a !mosh.ape shape value.  In the interpreter both are
    represented as 1-D int64 Buffers, so this is a pass-through (symmetric
    with ``_handle_shape_to_tensor``).
    """
    assert isinstance(inputs[0], Buffer)
    return [inputs[0]]


@register_op_handler(mo.IndexToTensorOp)
def _handle_index_to_tensor(
    op: mo.IndexToTensorOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.index.to_tensor - wraps an SI64 scalar into a rank-0 tensor.

    The input is a scalar int64 value (stored by the interpreter as a
    1-element Buffer from ``ParamToValueOp`` or similar).  The result is
    a rank-0 ``!mo.tensor<[], si64>`` scalar tensor.
    """
    assert isinstance(inputs[0], Buffer)
    val = int(inputs[0].to_numpy().item())
    result_np = np.array(val, dtype=np.int64)
    return [Buffer.from_numpy(result_np)]


@register_op_handler(mosh.ParamToValueOp)
def _handle_param_to_value(
    op: mosh.ParamToValueOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mosh.param.to_value - materializes parameter values.

    This op takes a compile-time parameter expression and produces an SSA value.
    For static shapes like <[0, 0]> or <[1, 3]>, we extract the values and
    create a buffer.
    """
    # Get the value attribute which contains the parameter expression
    value_attr = op.value

    # Get the result type to understand what we're producing
    result = list(op.results)[0]
    result_type = result.type

    # Handle !mosh.ape (shape type) - produces a tensor of indices
    if isinstance(result_type, mosh.ShapeType):
        # value_attr should be a ShapeAttr with values
        if isinstance(value_attr, mosh.ShapeAttr):
            shape_values = []
            for dim_attr in value_attr.values:
                # Cast simd literal to integer.
                if isinstance(dim_attr, kgen.SIMDAttr):
                    dim_attr = kgen.CastToBuiltinAttr(dim_attr)

                if hasattr(dim_attr, "value"):
                    val = dim_attr.value
                    if isinstance(val, int):
                        shape_values.append(val)
                    else:
                        raise NotImplementedError(
                            f"Dynamic dimension in param.to_value: {dim_attr}"
                        )
                else:
                    raise NotImplementedError(
                        "Unsupported dimension attr in param.to_value:"
                        f" {dim_attr}"
                    )
            # Create a 1D tensor of si64 values
            result_np = np.array(shape_values, dtype=np.int64)
            output = Buffer.from_numpy(result_np)
            return [output]
        else:
            raise NotImplementedError(
                f"Unsupported value attr type for shape: {type(value_attr)}"
            )

    # Handle index type (single integer value)
    # Check if it's an index/integer type by looking at the attribute
    if hasattr(value_attr, "value"):
        val = value_attr.value
        if isinstance(val, int):
            result_np = np.array([val], dtype=np.int64)
            output = Buffer.from_numpy(result_np)
            return [output]

    raise NotImplementedError(
        f"Unsupported param.to_value result type: {result_type}, attr:"
        f" {value_attr}"
    )


# Reduce-along-axis operations (reduce / softmax / argmax / cumsum)


def _handle_reduce_axis(
    op: _core.Operation, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager reduce-along-axis op to its GC model.

    Reading the op's MLIR result type for the output shape and dtype keeps this
    handler family-agnostic: reduced-axis ops (``[d0, 1, d2]``, ``int64`` for
    argmax) and same-shape ops (``[d0, d1, d2]``) share one path. The operand is
    view-shimmed to canonical rank 3 and the result back, both zero-copy.

    Args:
        op: The reduce/softmax/argmax/cumsum operation being handled.
        inputs: The realized input buffers; the first is the operand.

    Returns:
        A single-element list holding the result buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    x = inputs[0]
    assert isinstance(x, Buffer)
    _check_buffers_on_device(inputs, target_device)

    axis = cast(_HasAxis, op).axis
    variant = reduce_axis_gc.variant_for(op)
    model = reduce_axis_gc.reduce_model(type(op), x.device, x.dtype, variant)

    d0, d1, d2 = reduce_axis_gc.canonical_rank3(x.shape, axis)
    x_view = x.view(x.dtype, (d0, d1, d2))
    (out,) = model(x_view)
    out_shape = tuple(int(dim) for dim in result_type.shape)
    return [out.view(result_type.dtype, out_shape)]


# Wrapped in a function so the REDUCE_AXIS_GC_OPS access is deferred past the
# import cycle between this module and reduce_axis_gc (via the package).
def _register_reduce_axis_handlers() -> None:
    for op_type in reduce_axis_gc.REDUCE_AXIS_GC_OPS:
        register_op_handler(op_type)(_handle_reduce_axis)


_register_reduce_axis_handlers()


# Layer norm operations


@register_op_handler(mo.ReduceLayerNormOp)
def _handle_layer_norm(
    op: mo.ReduceLayerNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Routes eager layer_norm through a GC model.

    Looks up the rank-2 layer_norm :class:`~max.engine.Model` for the
    realized input's device and dtype (see
    :func:`layer_norm_gc.layer_norm_model`), view-shims the input to
    canonical rank 2 ``[rows, features]``, executes the model, and
    view-shims the result back. All views are zero-copy.

    Args:
        op: The layer_norm operation being handled.
        inputs: Input buffers -- input tensor, gamma, beta, epsilon.
            Epsilon is always on CPU (MO_SingleDeviceWithHostOperands).

    Returns:
        A single-element list holding the normalized tensor buffer.
    """
    x, gamma, beta, epsilon = inputs
    assert isinstance(x, Buffer)
    assert isinstance(gamma, Buffer)
    assert isinstance(beta, Buffer)
    assert isinstance(epsilon, Buffer)

    model = layer_norm_gc.layer_norm_model(x.device, x.dtype)

    rows, features = layer_norm_gc.canonical_rank2(x.shape)
    x_view = x.view(x.dtype, (rows, features))
    (out,) = model(x_view, gamma, beta, epsilon)

    return [out.view(out.dtype, x.shape)]


# RMS norm operations


@register_op_handler(mo.ReduceRmsNormOp)
def _handle_rms_norm(
    op: mo.ReduceRmsNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Routes eager rms_norm through a GC model.

    Looks up the rank-2 rms_norm :class:`~max.engine.Model` for the realized
    input's device, dtype, and compile-time ``multiply_before_cast`` variant
    (see :func:`rms_norm_gc.rms_norm_model`), view-shims the input to
    canonical rank 2 ``[rows, features]``, executes the model, and
    view-shims the result back. All views are zero-copy.

    Args:
        op: The rms_norm operation being handled.
        inputs: Input buffers -- input tensor, weight, epsilon,
            weight_offset. Epsilon and weight_offset are always on CPU
            (MO_SingleDeviceWithHostOperands).

    Returns:
        A single-element list holding the normalized tensor buffer.
    """
    x, weight, epsilon, weight_offset = inputs
    assert isinstance(x, Buffer)
    assert isinstance(weight, Buffer)
    assert isinstance(epsilon, Buffer)
    assert isinstance(weight_offset, Buffer)

    multiply_before_cast = bool(op.multiply_before_cast)
    model = rms_norm_gc.rms_norm_model(x.device, x.dtype, multiply_before_cast)

    rows, features = rms_norm_gc.canonical_rank2(x.shape)
    x_view = x.view(x.dtype, (rows, features))
    (out,) = model(x_view, weight, epsilon, weight_offset)

    return [out.view(out.dtype, x.shape)]


# Group norm operations


@register_op_handler(mo.ReduceGroupNormOp)
def _handle_group_norm(
    op: mo.ReduceGroupNormOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Routes eager group_norm through a GC model.

    Looks up the group_norm :class:`~max.engine.Model` for the realized
    input's device and dtype (see :func:`group_norm_gc.group_norm_model`),
    view-shims the input to canonical rank 4 ``[n, c, h, 1]`` (spatial dims
    collapsed into ``h``), executes the model, and view-shims the result
    back. All views are zero-copy.

    Args:
        op: The group_norm operation being handled.
        inputs: Input buffers -- input tensor, gamma, beta, epsilon,
            num_groups. Epsilon and num_groups are always on CPU
            (MO_SingleDeviceWithHostOperands).

    Returns:
        A single-element list holding the normalized tensor buffer.
    """
    x, gamma, beta, epsilon, num_groups = inputs
    assert isinstance(x, Buffer)
    assert isinstance(gamma, Buffer)
    assert isinstance(beta, Buffer)
    assert isinstance(epsilon, Buffer)
    assert isinstance(num_groups, Buffer)

    model = group_norm_gc.group_norm_model(x.device, x.dtype)

    n, c, h, w = group_norm_gc.canonical_shape(x.shape)
    x_view = x.view(x.dtype, (n, c, h, w))
    (out,) = model(x_view, gamma, beta, epsilon, num_groups)

    return [out.view(out.dtype, x.shape)]


# Range operations


@register_op_handler(mo.RangeOp)
def _handle_range(
    op: mo.RangeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager range to its GC model.

    Operands are start/limit/step, all host scalars in the result dtype
    (``MO_SingleDeviceWithHostOperands`` plus ``MO_SameOperandsAndResult-
    ElementType``), so they pass straight through with no re-materialization.
    The compiled graph is symbolic (``["n"]``); its registered shape function
    sizes the output at runtime, so the handler no longer computes the output
    size itself.

    ``ops.range``'s declared static length floors ``(stop - start) / step``,
    while the shared shape function this graph compiles against rounds up
    (``len(range(...))`` for ints, ``ceil`` for floats; see
    ``max/kernels/src/nn/arange.mojo``). The two agree only when the interval
    is evenly divisible by the step, so a non-divisible interval silently
    disagreed with its own declared type until this check: the interpreter
    otherwise validates only the model's output *count*, never its shape.

    Args:
        op: The range operation.
        inputs: start, limit, step (all scalar tensors on CPU).

    Returns:
        List containing the range tensor buffer.

    Raises:
        RuntimeError: If the declared static result length disagrees with the
            model's actual output length, i.e. the start/stop/step interval
            is not evenly divisible by the step -- matching the ``RuntimeError``
            ``ops.range`` and ``Tensor.arange`` document for this same
            declared-vs-actual mismatch.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    start, stop, step = inputs
    assert isinstance(start, Buffer)
    assert isinstance(stop, Buffer)
    assert isinstance(step, Buffer)

    model = range_gc.range_model(target_device, result_type.dtype)
    (out,) = model(start, stop, step)

    declared_shape = result_type.shape
    if graph.Shape.is_static(declared_shape):
        declared_len = graph.Shape(declared_shape).static_dims[0]
        actual_len = out.shape[0]
        if declared_len != actual_len:
            raise RuntimeError(
                f"range declared a static length of {declared_len} but"
                f" produced {actual_len} elements: the start/stop/step"
                " interval is not evenly divisible by the step."
            )
    return [out]


# Random operations


@register_op_handler(mo.RandomNormalOp)
@register_op_handler(mo.RandomUniformOp)
def _handle_random(
    op: mo.RandomNormalOp | mo.RandomUniformOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Dispatches eager random normal/uniform to its GC model.

    Operands are ``(shape, lower, upper, seed)`` for uniform and
    ``(shape, mean, variance, seed)`` for normal. ``MO_SingleDeviceWithHost-
    Operands`` puts shape/mean-lower/variance-upper on the host; seed is not
    listed, so it arrives on the same device as the output, matching how
    ``ops.random``'s public API places it. The compiled graph is flat rank 1 --
    the sampled values depend only on the element count -- so the handler sends
    the element count as the shape operand and views the result back.

    Args:
        op: The random normal or random uniform operation.
        inputs: The four operands described above.

    Returns:
        List containing the sampled tensor buffer.
    """
    shape_buf, lower, upper, seed = inputs
    assert isinstance(shape_buf, Buffer)
    assert isinstance(lower, Buffer)
    assert isinstance(upper, Buffer)
    assert isinstance(seed, Buffer)

    target_device = _get_target_device(op)
    result_mlir_type: mo.TensorType = list(op.results)[0].type  # type: ignore[assignment]
    output_dtype = result_mlir_type.dtype

    output_shape = [int(d) for d in shape_buf.to_numpy().flatten()]
    numel = prod(output_shape)
    flat_shape = Buffer.from_numpy(np.asarray([numel], dtype=np.int64))

    if random_gc.scalars_are_float32(type(op)):
        # Normal's scalars re-materialize at float32 for the one compiled
        # graph; uniform's bounds already carry the output dtype the graph
        # declares.
        lower = Buffer.from_numpy(
            np.asarray(lower.to_numpy(), dtype=np.float32)
        )
        upper = Buffer.from_numpy(
            np.asarray(upper.to_numpy(), dtype=np.float32)
        )

    model = random_gc.random_model(type(op), target_device, output_dtype)
    (out,) = model(flat_shape, lower, upper, seed)
    return [out.view(output_dtype, output_shape)]


# Select operations


@register_op_handler(mo.SelectOp)
def _handle_select(
    op: mo.SelectOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager select (``cond ? x : y``) to its GC model.

    The model is compiled once per (device, value dtype) at rank 1 (see
    :func:`select_gc.select_model`); all three operands are flattened around
    the call and the result reshaped back (zero-copy views). ``cond``/``x``/
    ``y`` arrive already broadcast to one common shape by the RMO->MO
    lowering, so a single rank-1 dtype keys the model exactly.

    Args:
        op: The select operation being handled.
        inputs: The realized input buffers (``cond``, ``x``, ``y``).

    Returns:
        A single-element list holding the result buffer.
    """
    cond, x, y = inputs
    assert isinstance(cond, Buffer)
    assert isinstance(x, Buffer)
    assert isinstance(y, Buffer)

    target_device = _get_target_device(op)
    _check_buffers_on_device(inputs, target_device)

    model = select_gc.select_model(target_device, x.dtype)
    shape = select_gc.canonical_shape(x.shape)
    cond_view = cond.view(cond.dtype, shape)
    x_view = x.view(x.dtype, shape)
    y_view = y.view(y.dtype, shape)
    (out,) = model(cond_view, x_view, y_view)
    return [out.view(out.dtype, x.shape)]


# Concat operations


@register_op_handler(mo.ConcatOp)
def _handle_concat(
    op: mo.ConcatOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.concat via a pairwise rank-3 GC concat at axis=1.

    Each operand is view-shimmed to canonical rank 3 ``[outer, axis_i, inner]``
    (zero-copy); a 2-input concat model folds them left-to-right; the result is
    re-viewed to the concrete output shape. A single operand short-circuits.
    """
    target_device = _get_target_device(op)
    tensors: list[Buffer] = []
    for buf in inputs:
        assert isinstance(buf, Buffer)
        tensors.append(buf)
    assert len(tensors) >= 1, "ConcatOp requires at least one input tensor"
    _check_buffers_on_device(tensors, target_device)

    if len(tensors) == 1:
        # No-op concat: alias the input (buffers are values here), not a copy.
        return [tensors[0]]

    axis = op.axis
    ndim = len(tensors[0].shape)
    if axis < 0:
        axis += ndim

    outer, _, inner = shape_rearrange_gc.canonical_rank3(tensors[0].shape, axis)
    dtype = tensors[0].dtype
    udtype = gc_compile.uint_view_dtype(dtype)
    gc_model = shape_rearrange_gc.model(mo.ConcatOp, tensors[0].device, udtype)

    def view3(buf: Buffer) -> Buffer:
        a = buf.shape[axis]
        return buf.view(udtype, (outer, a, inner))

    acc = view3(tensors[0])
    for buf in tensors[1:]:
        (acc,) = gc_model(acc, view3(buf))

    out_shape = list(tensors[0].shape)
    out_shape[axis] = sum(t.shape[axis] for t in tensors)
    return [acc.view(dtype, tuple(out_shape))]


# Gather operations


@register_op_handler(mo.GatherOp)
def _handle_gather(
    op: mo.GatherOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager gather to its GC model.

    ``axis`` is a compile-time attribute; the operands are view-shimmed to the
    canonical rank-3 data / rank-1 indices form (zero-copy) and the result back
    to the op's true output shape, read from its MLIR result type.

    Args:
        op: The gather operation.
        inputs: The realized input buffers - data tensor, indices tensor.

    Returns:
        A single-element list holding the gathered buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    x, indices = inputs
    assert isinstance(x, Buffer)
    assert isinstance(indices, Buffer)
    _check_buffers_on_device(inputs, target_device)

    axis = op.axis
    if axis < 0:
        axis += len(x.shape)

    # Pure copy: run it on the same-width uint so one graph serves every dtype.
    udtype = gather_gc.uint_view_dtype(x.dtype)
    model = gather_gc.gather_model(type(op), x.device, udtype, indices.dtype)
    data_view, idx_view = gather_gc.gather_operand_views(
        x.shape, indices.shape, axis
    )
    (out,) = model(
        x.view(udtype, data_view), indices.view(indices.dtype, idx_view)
    )
    out_shape = tuple(int(dim) for dim in result_type.shape)
    return [out.view(result_type.dtype, out_shape)]


@register_op_handler(mo.GatherSumOp)
def _handle_gather_sum(
    op: mo.GatherSumOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.gather_sum via NumPy gather-then-sum.

    This is a fused composite op used by DLRM-style multi-hot embeddings:
    gather along axis 0, then reduce-add along axis 1.

    ``output[i, k] = sum_j(input[indices[i, j], k])``

    Operands: input (2-D+ tensor), indices (index tensor).
    """
    target_device = _get_target_device(op)
    _check_cpu_only(op, target_device)

    x, indices = inputs
    assert isinstance(x, Buffer)
    assert isinstance(indices, Buffer)

    input_np = x.to_numpy()
    indices_np = indices.to_numpy().astype(np.intp)

    gathered = np.take(input_np, indices_np, axis=0)
    result = gathered.sum(axis=1, keepdims=True)

    return [Buffer.from_numpy(np.ascontiguousarray(result))]


@register_op_handler(mo.GatherNdOp)
def _handle_gather_nd(
    op: mo.GatherNdOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager gather_nd via gather_gc's single fused model.

    ``batch_dims`` is a compile-time attribute, but the fused graph itself
    always calls the real ``ops.gather_nd`` with ``batch_dims=1`` (see
    :func:`gather_gc.gather_nd_operand_views`/:func:`gather_gc.gather_nd_model`)
    -- it natively handles per-batch-distinct indices, so no batch-folding is
    needed in this handler.

    Args:
        op: The gather_nd operation.
        inputs: The realized input buffers - data tensor, indices tensor.

    Returns:
        A single-element list holding the gathered buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    x, indices = inputs
    assert isinstance(x, Buffer)
    assert isinstance(indices, Buffer)
    _check_buffers_on_device(inputs, target_device)

    data_view, idx_view, strides = gather_gc.gather_nd_operand_views(
        x.shape, indices.shape, op.batch_dims
    )
    udtype = gather_gc.uint_view_dtype(x.dtype)
    model = gather_gc.gather_nd_model(x.device, udtype, indices.dtype)
    strides_buf = Buffer(
        shape=[len(strides)], dtype=indices.dtype, device=CPU()
    )
    for i, s in enumerate(strides):
        strides_buf[i] = s
    (out,) = model(
        x.view(udtype, data_view),
        indices.view(indices.dtype, idx_view),
        strides_buf,
    )
    out_shape = tuple(int(dim) for dim in result_type.shape)
    return [out.view(result_type.dtype, out_shape)]


# ScatterNd operations


def _handle_scatter_nd_family(
    op: _core.Operation, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager scatter-nd op via scatter_nd_gc's single fused
    model.

    Args:
        op: The scatter-nd / scatter-nd-reduce operation being handled.
        inputs: The realized buffers - input, updates, indices.

    Returns:
        A single-element list holding the scattered buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    x, updates, indices = inputs
    assert isinstance(x, Buffer)
    assert isinstance(updates, Buffer)
    assert isinstance(indices, Buffer)
    _check_buffers_on_device([x, updates, indices], target_device)

    input_view, updates_view, indices_view, strides = (
        scatter_nd_gc.scatter_nd_operand_views(x.shape, indices.shape)
    )
    vdtype = scatter_nd_gc.model_dtype(type(op), x.dtype)
    model = scatter_nd_gc.scatter_nd_model(
        type(op), x.device, vdtype, indices.dtype
    )
    strides_buf = Buffer(
        shape=[len(strides)], dtype=indices.dtype, device=CPU()
    )
    for i, s in enumerate(strides):
        strides_buf[i] = s
    (out,) = model(
        x.view(vdtype, input_view),
        updates.view(vdtype, updates_view),
        indices.view(indices.dtype, indices_view),
        strides_buf,
    )
    out_shape = tuple(int(dim) for dim in result_type.shape)
    return [out.view(result_type.dtype, out_shape)]


# Wrapped in a function so the SCATTER_ND_GC_OPS access is deferred past the
# import cycle between this module and scatter_nd_gc (via the package).
def _register_scatter_nd_handlers() -> None:
    for op_type in scatter_nd_gc.SCATTER_ND_GC_OPS:
        register_op_handler(op_type)(_handle_scatter_nd_family)


_register_scatter_nd_handlers()


# Split operations


@register_op_handler(mo.SplitOp)
def _handle_split(
    op: mo.SplitOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.split via a rank-3 dynamic-slice GC model, one call per chunk.

    The input is view-shimmed to ``[outer, D, inner]``; each chunk is sliced at
    axis=1 with runtime ``(offset, offset+size)`` and re-viewed to its concrete
    shape. Operands: input (device), splitSizes (host int64 rank-1).
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    input_buffer = inputs[0]
    split_sizes = [int(s) for s in inputs[1].to_numpy().flatten()]

    in_shape = list(input_buffer.shape)
    ndim = len(in_shape)
    axis = op.axis
    if axis < 0:
        axis += ndim

    outer, _, inner = shape_rearrange_gc.canonical_rank3(in_shape, axis)
    dtype = input_buffer.dtype
    udtype = gc_compile.uint_view_dtype(dtype)
    x_view = input_buffer.view(udtype, (outer, in_shape[axis], inner))
    model = shape_rearrange_gc.model(mo.SplitOp, input_buffer.device, udtype)

    outputs: list[Buffer] = []
    offset = 0
    for size in split_sizes:
        lo = Buffer.from_numpy(np.array(offset, dtype=np.int64))
        hi = Buffer.from_numpy(np.array(offset + size, dtype=np.int64))
        (chunk,) = model(x_view, lo, hi)
        out_shape = list(in_shape)
        out_shape[axis] = size
        outputs.append(chunk.view(dtype, tuple(out_shape)))
        offset += size
    return outputs


# Scatter operations


def _handle_scatter_family(
    op: _core.Operation, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager axis-scatter op to its GC model.

    ``ScatterOp`` overwrites (a pure copy, so its data is bit-cast to the
    same-width uint); the reduce variants (add/max/min/mul) accumulate in the
    real dtype. ``axis`` (an attribute on ``ScatterOp``, a scalar operand on
    the reduce variants) is canonicalized out: operands are view-shimmed to
    rank 3 (zero-copy) and the result back to the input's shape, read from the
    op's MLIR result type. ``ops.scatter*`` builds the input copy, so there is no
    explicit memcpy.

    Args:
        op: The scatter/scatter-reduce operation being handled.
        inputs: The realized buffers - input, updates, indices (plus a scalar
            axis operand for the reduce variants).

    Returns:
        A single-element list holding the scattered buffer.
    """
    result_type = graph.Type.from_mlir(list(op.results)[0].type)
    assert isinstance(result_type, graph.TensorType)
    target_device = result_type.device.to_device()

    # Reduce variants carry axis as a 4th operand (scatter_axis reads inputs[3]).
    x, updates, indices, *extra = inputs
    assert len(extra) <= 1, f"scatter expects 3 or 4 inputs, got {len(inputs)}"
    assert isinstance(x, Buffer)
    assert isinstance(updates, Buffer)
    assert isinstance(indices, Buffer)
    # Only the data operands must sit on the target device.
    _check_buffers_on_device([x, updates, indices], target_device)

    axis = scatter_gc.scatter_axis(op, inputs)
    if axis < 0:
        axis += len(x.shape)

    vdtype = scatter_gc.model_dtype(type(op), x.dtype)
    input_view, updates_view = scatter_gc.scatter_operand_views(
        x.shape, updates.shape, axis
    )
    model = scatter_gc.scatter_model(type(op), x.device, vdtype, indices.dtype)
    (out,) = model(
        x.view(vdtype, input_view),
        updates.view(vdtype, updates_view),
        indices.view(indices.dtype, updates_view),
    )
    out_shape = tuple(int(dim) for dim in result_type.shape)
    return [out.view(result_type.dtype, out_shape)]


# Wrapped in a function so the SCATTER_GC_OPS access is deferred past the import
# cycle between this module and scatter_gc (via the package).
def _register_scatter_handlers() -> None:
    for op_type in scatter_gc.SCATTER_GC_OPS:
        register_op_handler(op_type)(_handle_scatter_family)


_register_scatter_handlers()


@register_op_handler(mo.ConvOp)
def _handle_conv(
    op: mo.ConvOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.conv (forward convolution, NHWC + RSCF).

    Args:
        op: The conv operation.
        inputs: Six buffers -- input, filter, strides, dilations, paddings,
            num_groups. All already-realized host int64/scalar tensors,
            forwarded straight to the GC model, which infers the output
            shape internally via its registered shape function.

    Returns:
        List containing the convolution output buffer.

    Raises:
        NotImplementedError: If groups != 1 -- grouped conv needs a
            pre-packed filter layout, and there's no Python-exposed op to
            produce one (the packing kernel is only invoked by an internal
            compiler mechanism). TODO(KERN-3239): expose a packing op.
    """
    input_buffer, filter_buffer, strides, dilations, paddings, num_groups = (
        inputs
    )
    assert isinstance(input_buffer, Buffer)  # input (NHWC)
    assert isinstance(filter_buffer, Buffer)  # filter (RSCF)
    assert isinstance(strides, Buffer)
    assert isinstance(dilations, Buffer)
    assert isinstance(paddings, Buffer)
    assert isinstance(num_groups, Buffer)

    groups = int(num_groups.to_numpy().item())
    if groups != 1:
        raise NotImplementedError(
            f"conv: groups != 1 is not supported (got groups={groups})"
        )

    model = conv_gc.conv_model(input_buffer.device, input_buffer.dtype)
    return model(
        input_buffer, filter_buffer, strides, dilations, paddings, num_groups
    )


# TODO(KERN-3233): add a conv_transpose GPU kernel (old Mojo binding's GPU
# path crashed on Apple/failed on CUDA; deleted rather than kept as a
# broken fallback).
@register_op_handler(mo.ConvTransposeOp)
def _handle_conv_transpose(
    op: mo.ConvTransposeOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    raise NotImplementedError("conv_transpose is not yet supported")


# Pooling operations


@register_op_handler(mo.MaxPoolOp)
@register_op_handler(mo.MaxPoolCeilModeTrueOp)
def _handle_max_pool(
    op: mo.MaxPoolOp | mo.MaxPoolCeilModeTrueOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.max_pool / mo.max_pool_ceil_mode_true.

    Args:
        op: The max_pool operation.
        inputs: Tuple of (input, filter_shape, strides, dilations, paddings)
            -- already-realized host int64 tensors, forwarded straight to
            the GC model, which infers the output shape internally.
    """
    input_buffer, filter_shape, strides, dilations, paddings = inputs
    assert isinstance(input_buffer, Buffer)
    assert isinstance(filter_shape, Buffer)
    assert isinstance(strides, Buffer)
    assert isinstance(dilations, Buffer)
    assert isinstance(paddings, Buffer)

    model = pooling_gc.pool_model(
        type(op), input_buffer.device, input_buffer.dtype
    )
    return model(input_buffer, filter_shape, strides, dilations, paddings)


@register_op_handler(mo.TileOp)
def _handle_tile(
    op: mo.TileOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.tile via rmo.MoTileOp with runtime repeats. CPU-only.

    Operands: input (device tensor), repeats (host int64 rank-1). Output
    shape[i] = input shape[i] * repeats[i].
    """
    target_device = _get_target_device(op)
    _check_cpu_only(op, target_device)
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    input_buffer = inputs[0]
    repeats = [int(r) for r in inputs[1].to_numpy().flatten()]

    rank = len(input_buffer.shape)
    out_shape = tuple(input_buffer.shape[i] * repeats[i] for i in range(rank))
    # int64 to match the graph's repeats operand regardless of input width.
    reps = Buffer.from_numpy(np.asarray(repeats, dtype=np.int64))
    udtype = gc_compile.uint_view_dtype(input_buffer.dtype)
    model = shape_rearrange_gc.model(
        mo.TileOp, input_buffer.device, udtype, rank
    )
    (out,) = model(input_buffer.view(udtype, input_buffer.shape), reps)
    return [out.view(input_buffer.dtype, out_shape)]


@register_op_handler(mo.LinalgBandPartOp)
def _handle_band_part(
    op: mo.LinalgBandPartOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager band_part to its GC model.

    Operands: input (device), then num_lower / num_upper / exclude, which
    ``MO_SingleDeviceWithHostOperands`` guarantees are host scalars; they pass
    through as runtime operands. The leading axes are batches, so the input
    collapses to rank 3 and the model is keyed by bit width rather than dtype
    (masking only copies or zeroes an element).

    Args:
        op: The band_part operation.
        inputs: Input buffers.

    Returns:
        List containing the masked output buffer.

    Raises:
        ValueError: If the input rank is below 2.
    """
    input_buffer, num_lower, num_upper, exclude = inputs
    assert isinstance(input_buffer, Buffer)
    assert isinstance(num_lower, Buffer)
    assert isinstance(num_upper, Buffer)
    assert isinstance(exclude, Buffer)

    in_shape = list(input_buffer.shape)
    if len(in_shape) < 2:
        raise ValueError(
            f"band_part expects rank >= 2 input, got rank {len(in_shape)}"
        )

    udtype = gc_compile.uint_view_dtype(input_buffer.dtype)
    rank3 = [prod(in_shape[:-2]), in_shape[-2], in_shape[-1]]
    model = band_part_gc.band_part_model(input_buffer.device, udtype)
    # Re-materialized at int64: one compiled graph fixes a single width.
    (out,) = model(
        input_buffer.view(udtype, rank3),
        Buffer.from_numpy(np.asarray(num_lower.to_numpy(), dtype=np.int64)),
        Buffer.from_numpy(np.asarray(num_upper.to_numpy(), dtype=np.int64)),
        exclude,
    )
    return [out.view(input_buffer.dtype, in_shape)]


# Average pooling


@register_op_handler(mo.AvgPoolOp)
@register_op_handler(mo.AvgPoolCeilModeTrueOp)
def _handle_avg_pool(
    op: mo.AvgPoolOp | mo.AvgPoolCeilModeTrueOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer]:
    """Handle mo.avg_pool / mo.avg_pool_ceil_mode_true.

    Args:
        op: The avg_pool operation. ``count_boundary`` selects the
            compiled variant.
        inputs: Tuple of (input, filter_shape, strides, dilations, paddings)
            -- already-realized host int64 tensors, forwarded straight to
            the GC model.
    """
    input_buffer, filter_shape, strides, dilations, paddings = inputs
    assert isinstance(input_buffer, Buffer)
    assert isinstance(filter_shape, Buffer)
    assert isinstance(strides, Buffer)
    assert isinstance(dilations, Buffer)
    assert isinstance(paddings, Buffer)

    model = pooling_gc.pool_model(
        type(op),
        input_buffer.device,
        input_buffer.dtype,
        bool(op.count_boundary),
    )
    return model(input_buffer, filter_shape, strides, dilations, paddings)


# ROI Align operation


@register_op_handler(mo.RoiAlignOp)
def _handle_roi_align(
    op: mo.RoiAlignOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager roi_align to its GC model.

    ``MO_HostOnly`` keeps every operand on CPU. ``aligned`` and ``mode`` are
    MLIR attributes the GC kernel takes as comptime params, so they join
    dtype in the model cache key (see ``roi_align_gc``'s module docstring);
    the four numeric scalars stay ordinary runtime operands, re-materialized
    at the dtypes the graph declares.

    Args:
        op: The roi_align operation.
        inputs: input ``[N, H, W, C]``, rois ``[M, 5]``, then output_height,
            output_width, spatial_scale, and sampling_ratio scalars.

    Returns:
        List containing the ROI-aligned output buffer.

    Raises:
        ValueError: If the input is not rank-4, rois is not ``[M, 5]``, or
            ``mode`` is not ``"AVG"``/``"MAX"``.
    """
    input_buffer, rois_buffer, out_h, out_w, spatial_scale, sampling_ratio = (
        inputs
    )
    assert isinstance(input_buffer, Buffer)
    assert isinstance(rois_buffer, Buffer)
    assert isinstance(out_h, Buffer)
    assert isinstance(out_w, Buffer)
    assert isinstance(spatial_scale, Buffer)
    assert isinstance(sampling_ratio, Buffer)

    in_shape = list(input_buffer.shape)
    if len(in_shape) != 4:
        raise ValueError(
            f"roi_align expects rank-4 NHWC input, got rank {len(in_shape)}"
        )

    rois_shape = list(rois_buffer.shape)
    if len(rois_shape) != 2 or rois_shape[1] != 5:
        raise ValueError(
            f"roi_align expects [M, 5] rois, got shape {rois_shape}"
        )

    mode_str = str(op.mode.value)
    if mode_str not in ("AVG", "MAX"):
        raise ValueError(
            f"roi_align mode must be 'AVG' or 'MAX', got '{mode_str}'"
        )

    model = roi_align_gc.roi_align_model(
        input_buffer.dtype, bool(op.aligned), mode_str
    )
    # Re-materialized at the dtypes the graph declares: one compiled graph
    # fixes int64 + float32.
    (out,) = model(
        input_buffer,
        rois_buffer,
        Buffer.from_numpy(np.asarray(out_h.to_numpy(), dtype=np.int64)),
        Buffer.from_numpy(np.asarray(out_w.to_numpy(), dtype=np.int64)),
        Buffer.from_numpy(
            np.asarray(spatial_scale.to_numpy(), dtype=np.float32)
        ),
        Buffer.from_numpy(
            np.asarray(sampling_ratio.to_numpy(), dtype=np.float32)
        ),
    )
    return [out]


# Top-K / Bottom-K operations


def _handle_topk(
    op: _core.Operation, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches an eager top_k/bottom_k op to its GC model.

    ``k`` passes straight through as a runtime operand (see topk_gc's module
    docstring); ``sorted`` (``inputs[3]``) is unused since the GC graph always
    sorts. The operand is view-shimmed to canonical rank 3 and both results
    view-shimmed back to their real MLIR result shapes, all zero-copy.

    Args:
        op: The top_k/bottom_k operation being handled.
        inputs: The realized input buffers -- operand, ``k``, ``axis``, and
            ``sorted`` (host scalars per ``MO_SelectKLikeOp``).

    Returns:
        A two-element list holding the values and indices buffers.
    """
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    assert isinstance(inputs[2], Buffer)

    x = inputs[0]
    k = inputs[1]
    axis = int(inputs[2].to_numpy().item())

    model = topk_gc.topk_model(type(op), x.device, x.dtype)

    d0, d1, d2 = topk_gc.canonical_rank3(x.shape, axis)
    x_view = x.view(x.dtype, (d0, d1, d2))
    vals, idxs = model(x_view, k)

    vals_type = graph.Type.from_mlir(list(op.results)[0].type)
    idxs_type = graph.Type.from_mlir(list(op.results)[1].type)
    assert isinstance(vals_type, graph.TensorType)
    assert isinstance(idxs_type, graph.TensorType)

    vals_shape = tuple(int(dim) for dim in vals_type.shape)
    idxs_shape = tuple(int(dim) for dim in idxs_type.shape)
    return [
        vals.view(vals_type.dtype, vals_shape),
        idxs.view(idxs_type.dtype, idxs_shape),
    ]


# Wrapped in a function so the TOPK_GC_OPS access is deferred past the import
# cycle between this module and topk_gc (via the package).
def _register_topk_handlers() -> None:
    for op_type in topk_gc.TOPK_GC_OPS:
        register_op_handler(op_type)(_handle_topk)


_register_topk_handlers()


# Arg-NonZero operation


@register_op_handler(mo.ArgNonzeroOp)
def _handle_arg_nonzero(
    op: mo.ArgNonzeroOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager arg_nonzero to its GC model.

    ``MO_HostOnly`` keeps the input on CPU. The compiled graph is canonicalized
    to rank 1: the handler flattens the input, runs the flat model, and unravels
    the resulting flat indices back into ``[nnz, rank]`` row-major coordinates
    against the input's real shape with ``np.unravel_index`` -- the op's shape
    function computes the exact nonzero count, so the model sizes its own flat
    output and the handler neither counts nor allocates.

    Args:
        op: The arg_nonzero operation.
        inputs: Single input buffer.

    Returns:
        List containing the ``[nnz, rank]`` int64 coordinate buffer.

    Raises:
        ValueError: If the input is rank 0 -- ``ops.nonzero`` rejects scalar
            inputs with the same error, so a public caller never reaches this
            handler with one.
        KeyError: If the input dtype is unsupported.
    """
    input_buffer = inputs[0]
    assert isinstance(input_buffer, Buffer)
    shape = input_buffer.shape
    rank = len(shape)
    if rank == 0:
        raise ValueError(
            "arg_nonzero requires rank >= 1 input; scalar inputs not supported."
        )

    dtype = input_buffer.dtype
    graph_dtype = (
        dtype if dtype.is_float() else gc_compile.uint_view_dtype(dtype)
    )
    flat = input_buffer.view(graph_dtype, (prod(shape),))

    model = nonzero_gc.nonzero_model(graph_dtype)
    (flat_idx,) = model(flat)
    flat_np = flat_idx.to_numpy().reshape(-1)
    # np.unravel_index already returns int64, so astype would otherwise copy.
    coords_np = np.stack(np.unravel_index(flat_np, shape), axis=-1).astype(
        np.int64, copy=False
    )
    return [Buffer.from_numpy(coords_np)]


# Padding operations


def _handle_pad_via_gc(
    op: _core.Operation,
    inputs: Sequence[Buffer | None],
    op_type: type[_core.Operation],
) -> Sequence[Buffer]:
    """Dispatch a pad op to its GC model. Constant pad has a 3rd (constant) operand."""
    assert isinstance(inputs[0], Buffer)
    assert isinstance(inputs[1], Buffer)
    input_buffer = inputs[0]
    paddings = [int(p) for p in inputs[1].to_numpy().flatten()]
    in_shape = list(input_buffer.shape)
    rank = len(in_shape)
    out_shape = tuple(
        in_shape[d] + paddings[2 * d] + paddings[2 * d + 1] for d in range(rank)
    )
    # int64 to match the graph's paddings operand dtype regardless of width.
    pad_buf = Buffer.from_numpy(np.asarray(paddings, dtype=np.int64))
    dtype = input_buffer.dtype
    udtype = gc_compile.uint_view_dtype(dtype)
    model = shape_rearrange_gc.model(op_type, input_buffer.device, udtype, rank)
    x_view = input_buffer.view(udtype, input_buffer.shape)
    if op_type is mo.PadConstantOp:
        assert isinstance(inputs[2], Buffer)
        # Fill value: rank-0 scalar operand, in the copy's uint type.
        const_buf = inputs[2].view(udtype, ())
        (out,) = model(x_view, pad_buf, const_buf)
    else:
        (out,) = model(x_view, pad_buf)
    return [out.view(dtype, out_shape)]


@register_op_handler(mo.PadConstantOp)
def _handle_pad_constant(
    op: mo.PadConstantOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.pad.constant via rmo.MoPadConstantOp (CPU + GPU)."""
    return _handle_pad_via_gc(op, inputs, mo.PadConstantOp)


@register_op_handler(mo.PadReflectOp)
def _handle_pad_reflect(
    op: mo.PadReflectOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.pad.reflect via rmo.MoPadReflectOp (CPU-only)."""
    _check_cpu_only(op, _get_target_device(op))
    return _handle_pad_via_gc(op, inputs, mo.PadReflectOp)


@register_op_handler(mo.PadRepeatOp)
def _handle_pad_repeat(
    op: mo.PadRepeatOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.pad.repeat (edge pad) via rmo.MoPadRepeatOp (CPU-only)."""
    _check_cpu_only(op, _get_target_device(op))
    return _handle_pad_via_gc(op, inputs, mo.PadRepeatOp)


@register_op_handler(mo.ResizeLinearOp)
def _handle_resize_linear(
    op: mo.ResizeLinearOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.resize.linear (CPU-only).

    Args:
        op: The resize-linear operation.
        inputs: tuple of input data, output shape.

    Returns:
        List containing a single output buffer with shape given by ``size``.
    """
    input_buffer, size_buffer = inputs
    assert isinstance(input_buffer, Buffer)
    assert isinstance(size_buffer, Buffer)

    rank = len(input_buffer.shape)
    variant = int(op.coordinate_transform_mode.value), int(op.antialias)
    model = resize_gc.resize_model(
        mo.ResizeLinearOp,
        input_buffer.device,
        input_buffer.dtype,
        rank,
        variant,
    )
    return model(input_buffer, size_buffer)


@register_op_handler(mo.ResizeNearestOp)
def _handle_resize_nearest(
    op: mo.ResizeNearestOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Handle mo.resize.nearest (CPU-only).

    Args:
        op: The resize-nearest operation.
        inputs: tuple of input data, output shape.

    Returns:
        List containing a single output buffer with shape given by ``size``.
    """
    input_buffer, size_buffer = inputs
    assert isinstance(input_buffer, Buffer)
    assert isinstance(size_buffer, Buffer)

    rank = len(input_buffer.shape)
    variant = int(op.coordinate_transform_mode.value), int(op.round_mode)
    model = resize_gc.resize_model(
        mo.ResizeNearestOp,
        input_buffer.device,
        input_buffer.dtype,
        rank,
        variant,
    )
    return model(input_buffer, size_buffer)


# TODO(GEX-3990): GraphCompiler has no shape-fallback registration for
# MO::ResizeBicubicOp (unlike Linear/Nearest); resize_ops.mojo has been
# deleted rather than kept as a fallback that can't be reached.
@register_op_handler(mo.ResizeBicubicOp)
def _handle_resize_bicubic(
    op: mo.ResizeBicubicOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    raise NotImplementedError("resize_bicubic is not yet supported")


# Distributed operations


@register_op_handler(mo.DistributedAllreduceSumOp)
def _handle_distributed_allreduce_sum(
    op: mo.DistributedAllreduceSumOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.allreduce.sum by summing tensors across devices.

    Operands (flat): N input tensors, N signal buffers, 1 input chain.
    Results: N output tensors (one per device, all holding the sum), 1 output chain.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.  Each input tensor is transferred to the CPU,
    summed via NumPy, and the result is placed back on each output device.

    Args:
        op: The allreduce sum operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers (one per device) followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"allreduce input {i} is not a Buffer"
        bufs.append(b)

    # Sum inputs on the CPU via NumPy, one independent sum per group.
    # group_size == 0 (the attribute default) means one full-world group.
    group_size = op.group_size or num_inputs
    totals: list[npt.NDArray[Any]] = []
    for group_start in range(0, num_inputs, group_size):
        total = bufs[group_start].to(CPU()).to_numpy().copy()
        for buf in bufs[group_start + 1 : group_start + group_size]:
            total += buf.to(CPU()).to_numpy()
        totals.extend([total] * group_size)

    # Place each group's sum on its output devices.
    results = list(op.results)
    output_buffers: list[Buffer | None] = []
    for i, result in enumerate(results[:-1]):
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(Buffer.from_numpy(totals[i]).to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers


@register_op_handler(mo.DistributedAllgatherOp)
def _handle_distributed_allgather(
    op: mo.DistributedAllgatherOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.allgather by copying each input to every device.

    Operands (flat): N input tensors, N signal buffers, 1 input chain.
    Results: N*G output tensors, 1 output chain, where G is the group size.

    The MO-level allgather produces raw outputs grouped by destination device:
    for each device d and each source i in d's group, the raw result is a copy
    of that source input on device d.  The Graph API wraps these with separate
    ``ConcatOp`` calls to produce the final gathered tensors.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.

    Args:
        op: The allgather operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N*G output buffers followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"allgather input {i} is not a Buffer"
        bufs.append(b)

    results = list(op.results)
    group_size = op.group_size
    output_buffers: list[Buffer | None] = []
    for idx, result in enumerate(results[:-1]):
        device_idx = idx // group_size
        local_input_idx = idx % group_size
        group_start = (device_idx // group_size) * group_size
        input_idx = group_start + local_input_idx
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(bufs[input_idx].to(device))

    output_buffers.append(None)
    return output_buffers


@register_op_handler(mo.DistributedScatterOp)
def _handle_distributed_scatter(
    op: mo.DistributedScatterOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.scatter by distributing root inputs to devices.

    Operands (flat): N input tensors (all on root), N signal buffers, 1 input chain.
    Results: N output tensors (one per device), 1 output chain.

    Each input[i] is copied to the device indicated by result[i]'s type.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.

    Args:
        op: The scatter operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"scatter input {i} is not a Buffer"
        bufs.append(b)

    # All inputs should reside on the root device.
    if bufs:
        root_device = bufs[0].device
        for i, buf in enumerate(bufs[1:], 1):
            assert buf.device == root_device, (
                f"scatter expects all inputs on root device {root_device}, "
                f"but input {i} is on {buf.device}"
            )

    results = list(op.results)
    num_outputs = len(results) - 1  # exclude trailing chain
    assert num_outputs == num_inputs, (
        "scatter expects N inputs and N outputs, "
        f"got {num_inputs} inputs and {num_outputs} outputs"
    )

    output_buffers: list[Buffer | None] = []
    for idx, result in enumerate(results[:-1]):
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(bufs[idx].to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers


@register_op_handler(mo.DistributedBroadcastOp)
def _handle_distributed_broadcast(
    op: mo.DistributedBroadcastOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.broadcast by replicating input to all devices.

    Operands (flat): 1 input tensor, N signal buffers, 1 input chain.
    Results: N output tensors (one per device, all copies of the input),
    1 output chain.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.

    Args:
        op: The broadcast operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers followed by None for the chain.
    """
    # op.root identifies the source device, but in the flat operand layout
    # inputs[0] is always the root tensor; the interpreter simply copies it
    # to every output device regardless of root index.
    input_buf = inputs[0]
    assert isinstance(input_buf, Buffer), "broadcast input is not a Buffer"

    num_signal_bufs = len(op.signal_buffers)
    results = list(op.results)
    num_outputs = len(results) - 1  # exclude trailing chain
    assert num_outputs == num_signal_bufs, (
        "broadcast expects one output per signal buffer, "
        f"got {num_outputs} outputs and {num_signal_bufs} signal buffers"
    )

    output_buffers: list[Buffer | None] = []
    for result in results[:-1]:
        result_type: mo.TensorType = result.type  # type: ignore[assignment]
        device = graph.DeviceRef.from_mlir(result_type.device_ref).to_device()
        output_buffers.append(input_buf.to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers


# Non-maximum suppression


@register_op_handler(mo.NonMaximumSuppressionOp)
def _handle_non_maximum_suppression(
    op: mo.NonMaximumSuppressionOp, inputs: Sequence[Buffer | None]
) -> Sequence[Buffer]:
    """Dispatches eager non-maximum suppression to its GC model.

    ``MO_HostOnly`` keeps every operand on CPU. The op's shape function runs the
    real suppression to size the output, so the handler no longer allocates an
    upper-bound buffer or truncates the result.

    Args:
        op: The non-maximum suppression operation.
        inputs: boxes ``[batch, num_boxes, 4]``, scores
            ``[batch, num_classes, num_boxes]``, then the
            max-output-per-class, IoU-threshold, and score-threshold scalars.

    Returns:
        List containing a single ``[num_selected, 3]`` int64 buffer whose rows
        are ``[batch_index, class_index, box_index]``.

    Raises:
        ValueError: If ``boxes`` and ``scores`` disagree on dtype: the model is
            keyed on ``boxes.dtype`` alone, and neither the op nor its verifier
            enforces that ``scores`` matches.
    """
    boxes, scores, max_output, iou_threshold, score_threshold = inputs
    assert isinstance(boxes, Buffer)
    assert isinstance(scores, Buffer)
    assert isinstance(max_output, Buffer)
    assert isinstance(iou_threshold, Buffer)
    assert isinstance(score_threshold, Buffer)

    if boxes.dtype != scores.dtype:
        raise ValueError(
            f"non_maximum_suppression requires boxes and scores to share a"
            f" dtype; got boxes={boxes.dtype}, scores={scores.dtype}."
        )

    model = nms_gc.nms_model(boxes.dtype)
    # Re-materialized at the dtypes the graph declares: one compiled graph
    # fixes int64 + float32.
    return model(
        boxes,
        scores,
        Buffer.from_numpy(np.asarray(max_output.to_numpy(), dtype=np.int64)),
        Buffer.from_numpy(
            np.asarray(iou_threshold.to_numpy(), dtype=np.float32)
        ),
        Buffer.from_numpy(
            np.asarray(score_threshold.to_numpy(), dtype=np.float32)
        ),
    )


@register_op_handler(mo.DistributedReducescatterSumOp)
def _handle_distributed_reducescatter_sum(
    op: mo.DistributedReducescatterSumOp,
    inputs: Sequence[Buffer | None],
) -> Sequence[Buffer | None]:
    """Handle mo.distributed.reducescatter.sum by summing then splitting.

    Operands (flat): N input tensors, N signal buffers, 1 input chain.
    Results: N output tensors (one per device, each a chunk of the sum),
    1 output chain.

    The interpreter executes sequentially on the host, so signal buffers
    and chains are unused.  Each input tensor is transferred to the CPU,
    summed via NumPy, and the result is split along the scatter axis so
    that each device receives its disjoint chunk.

    Args:
        op: The reduce-scatter sum operation.
        inputs: Flat operand buffers from the interpreter dispatcher.

    Returns:
        N output buffers followed by None for the chain.
    """
    num_inputs = len(op.inputs)
    bufs: list[Buffer] = []
    for i in range(num_inputs):
        b = inputs[i]
        assert isinstance(b, Buffer), f"reducescatter input {i} is not a Buffer"
        bufs.append(b)

    axis = op.axis
    group_size = op.group_size

    results = list(op.results)
    num_outputs = len(results) - 1  # exclude trailing chain
    assert num_outputs == num_inputs, (
        "reducescatter expects N inputs and N outputs, "
        f"got {num_inputs} inputs and {num_outputs} outputs"
    )

    output_buffers: list[Buffer | None] = []
    for group_start in range(0, num_inputs, group_size):
        group_bufs = bufs[group_start : group_start + group_size]

        # Sum inputs in each group independently on the CPU via NumPy.
        total = group_bufs[0].to(CPU()).to_numpy().copy()
        for buf in group_bufs[1:]:
            total += buf.to(CPU()).to_numpy()

        # Split the summed result along the scatter axis using ragged binning
        # (same formula as ops/reducescatter.py).
        dim = total.shape[axis]
        chunk_sizes = [
            (dim + (group_size - i - 1)) // group_size
            for i in range(group_size)
        ]
        chunks = np.split(total, np.cumsum(chunk_sizes[:-1]), axis=axis)

        for local_idx, chunk in enumerate(chunks):
            result = results[group_start + local_idx]
            result_type: mo.TensorType = result.type  # type: ignore[assignment]
            device = graph.DeviceRef.from_mlir(
                result_type.device_ref
            ).to_device()
            output_buffers.append(Buffer.from_numpy(chunk).to(device))

    # Trailing None for the output chain.
    output_buffers.append(None)
    return output_buffers
