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
"""Core graph primitives."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypeAlias, cast

import numpy as np
from max.dtype import DType

from ..._core import graph as _graph
from ..._core.dialects import builtin as _builtin
from ..._core.dialects import mo as _mo
from ...driver import CPU, Buffer, Device, DLPackArray
from ..graph import Graph
from ..type import DeviceRef, TensorType
from ..value import TensorValue
from .validation import _check_device_placement

Number: TypeAlias = float | np.number[Any]
NestedArray: TypeAlias = Sequence["Number | NestedArray"]


def shape(literal: NestedArray | Number) -> tuple[int, ...]:
    """Returns the nested shape of a literal array or number (scalar gives ``()``)."""
    if not isinstance(literal, Sequence):
        return ()
    outer = len(literal)
    inners: set[tuple[int, ...]] = {shape(inner) for inner in literal}
    if len(inners) > 1:
        raise ValueError(f"Array literals must be rectangular, got {literal=}")
    return outer, *next(iter(inners), ())


def index(literal: NestedArray | Number, idx: Sequence[int]) -> Number:
    """Returns the element at the given index into a nested literal."""
    if not idx:
        assert not isinstance(literal, Sequence)
        return cast(Number, literal)
    first, *rest = idx
    assert isinstance(literal, Sequence)
    return index(literal[first], rest)


def constant(
    value: DLPackArray | NestedArray | Number,
    dtype: DType | None = None,
    device: Device | DeviceRef | None = None,
) -> TensorValue:
    """Creates a constant tensor from a Python literal or array-like value.

    .. code-block:: python

        from max.dtype import DType
        from max.engine import InferenceSession
        from max.graph import DeviceRef, Graph, ops

        device = DeviceRef.CPU()
        with Graph("constant_example") as graph:
            x = ops.constant(
                [[1.0, 2.0], [3.0, 4.0]], DType.float32, device=device
            )
            graph.output(x)

        model = InferenceSession().load(graph)
        result = model.execute()[0]
        # result holds [[1.0, 2.0], [3.0, 4.0]].

    .. caution::

        Loading a constant can lose precision. For example, loading
        ``16777217`` as a ``float32`` produces ``16777216.0``.

    Args:
        value: The value to embed. A Python scalar, a (nested) sequence of
            numbers, or an array-like object that supports DLPack, such as a
            NumPy array.
        dtype: The constant tensor's element type. Required when ``value`` is a
            Python scalar or sequence. For an array-like ``value``, defaults to
            the array's dtype.
        device: The device the constant lives on. Required when ``value`` is a
            Python scalar or sequence. For an array-like ``value``, defaults to
            the array's device.

    Returns:
        A ``TensorValue`` representing the constant, with the same shape as
        ``value``. A scalar ``value`` produces a rank-0 tensor.

    Raises:
        TypeError: If ``dtype`` is a sub-byte type, or if ``value`` is a Python
            scalar or sequence and ``dtype`` or ``device`` isn't set.
        ValueError: If ``value`` is a nested sequence that isn't rectangular,
            if an integer in ``value`` is out of range for ``dtype``, or if
            ``dtype`` doesn't match the dtype of an array-like ``value``.
    """
    if dtype is not None and dtype.size_in_bits < 8:
        raise TypeError(
            f"Cannot create a constant of type '{dtype}' since it is a sub-byte type."
        )

    if not isinstance(value, DLPackArray):
        if dtype is None or device is None:
            raise TypeError(
                "Literal constants must explicitly set a dtype and device."
            )

        min, max = _DTYPE_MIN_AND_MAX[dtype]
        tensor = Buffer(dtype, shape(value), device=CPU())
        for idx in tensor._iterate_indices():
            v = index(value, idx)

            if not dtype.is_float() and not min <= int(v) <= max:
                raise ValueError(
                    "Unsafe cast: Refusing to implicitly promote external "
                    f"array with value {v} out of range for DType {dtype}."
                )

            tensor[idx] = v

        value = tensor
    elif isinstance(value, np.ndarray):
        value = np.ascontiguousarray(value)

    value = Buffer.from_dlpack(value)
    device = DeviceRef.from_device(device or value.device)
    if not value.device.is_host:
        _check_device_placement("ops.constant", "TODO(MXF-249).")
        value = value.to(CPU())  # lint: allow-host-sync
    dtype = dtype or value.dtype
    if dtype != value.dtype:
        raise ValueError(
            f"DType must match input dtype: {dtype=} != {value.dtype=}"
        )

    type = TensorType(dtype, value.shape, device=device)
    attr = _graph.array_attr(value, type.to_mlir())
    # See constant_external() below for why constants never carry a
    # profile_scope label.
    return Graph.current._add_op_generated(
        _mo.ConstantOp, type, attr, attach_profile_scopes=False
    )[0].tensor


def constant_external(
    name: str,
    type: TensorType,
    align: int | None = None,
    is_placeholder: bool = False,
) -> TensorValue:
    """Registers an external constant (weight) in the graph of a given type.

    Two external constants with the same name and type refer to the same weight.

    Two external constants with the same name and different types are
    incompatible and will fail compilation.

    Args:
        name: The name of the external constant.
            This should be the fully-qualified weight name and must be unique.
        type: The type of the constant value.
        align: The alignment of the constant. If not provided,
            the default alignment for the type's dtype will be used.
        is_placeholder: When :obj:`True`, marks the constant as a placeholder
            whose name is resolved at :func:`~max.graph.ops.call` time by the
            call's ``prefix``.

    Returns:
        A ``TensorValue`` of the specified type, representing the weight value
        associated with the name at compile time.
    """
    # Constants/weights are compile-time data that later passes are free to
    # batch/dedupe/hoist; a shared allocation op inheriting one constant's
    # scope label would misdirect anything that searches for "the first op
    # tagged with scope S".
    return Graph.current._add_op_generated(
        _mo.ConstantExternalOp,
        result=type,
        name=name,
        align=_builtin.IntegerAttr(
            _builtin.IntegerType(64, _builtin.SignednessSemantics.unsigned),
            align or type.dtype.align,
        ),
        device=type.device,
        has_alias=False,
        is_placeholder=is_placeholder,
        attach_profile_scopes=False,
    )[0]


# For each DType, this is the full range of representable values.
# Since constant and scalar have explicit users dtypes, we trust that the specified dtype is wanted.
# We still error is a value does not fit in these ranges.
_DTYPE_MIN_AND_MAX = {
    DType.bool: (0, 1),
    DType.int8: (-(2**7), 2**7 - 1),
    DType.int16: (-(2**15), 2**15 - 1),
    DType.int32: (-(2**31), 2**31 - 1),
    DType.int64: (-(2**63), 2**63 - 1),
    DType.uint8: (0, 2**8 - 1),
    DType.uint16: (0, 2**16 - 1),
    DType.uint32: (0, 2**32 - 1),
    DType.uint64: (0, 2**64 - 1),
    DType.float4_e2m1fn: (-0b0111, 0b0111),
    DType.float6_e2m3fn: (-7.5, 7.5),
    DType.float6_e3m2fn: (-28, 28),
    DType.float8_e8m0fnu: (2**-127, 2**127),
    DType.float8_e5m2: (float("-inf"), float("inf")),
    DType.float8_e5m2fnuz: (-57344, 57344),
    DType.float8_e4m3fn: (-448, 448),
    DType.float8_e4m3fnuz: (-240, 240),
    DType.bfloat16: (float("-inf"), float("inf")),
    DType.float16: (float("-inf"), float("inf")),
    DType.float32: (float("-inf"), float("inf")),
    DType.float64: (float("-inf"), float("inf")),
}
