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
"""Handles DType promotion for TensorValues.

DType promotion decides the output DType of an operation that has multiple
inputs with differing DTypes. Not all operations promote, but most binary and
mathematical ops do.

Promotion always returns one of the input DTypes; it never widens to a new
DType. This avoids accidentally over-promoting and harming performance.

To choose a common DType for two values, each input DType is ranked along two
axes:

- Category, ordered ``bool < unsigned int < signed int < float``.
- Bit width (for example, 8, 16, 32, or 64 bits).

The common DType is the input DType with the highest category and the largest
bit width. If an input can't be safely represented in the chosen DType, an
error is raised instead of widening to a different DType. For example,
``uint8`` and ``int8`` promote toward ``int8`` (signed outranks unsigned at
the same bit width), but ``int8`` can't represent the largest ``uint8``
values, so promotion fails.

A weak DType is the implicit DType of a non-max object, such as a Python
``int`` or ``float`` or a NumPy array. When a max object and a non-max object
are promoted together, the result always takes the max object's DType, and the
non-max object is scanned to confirm its values are exactly representable in
that DType. For example, ``16777217`` raises an error when promoted to
``float32``, since it would round to ``16777216.0``. To convert a value while
allowing minor precision loss, use ``ops.constant``.

Promotion fails if every input is a non-max object, because there is no max
object DType to promote toward.
"""

import numpy as np
from max.dtype import DType

from ..driver import Buffer, DLPackArray
from . import ops
from .dim import StaticDim
from .graph import DeviceRef
from .value import TensorValue, TensorValueLike, _is_strong_tensor_value_like


def _restrict_to_strong_dtypes(value: TensorValueLike) -> TensorValue:
    """Converts strong dtype values to TensorValue.

    Raise an error if the input dtype is weak.
    """
    if _is_strong_tensor_value_like(value):
        # Valid unary op with proper dtype.
        return TensorValue(value)
    else:
        # TODO: Maybe special case numpy array with non int64/float64 input.
        # Theoretically, that is an explicitly set dtype.

        # Non-max object with unary op.
        # This often is a bug and leads to overpromotion.
        raise TypeError(
            "Unary ops do not support non-max objects as input. Non-max"
            " objects tend to overpromote the dtype, leading to significant"
            " loss of performance. Please explicitly convert the input to a"
            " graph.Value. This can be done with ops.constant."
        )


def _promote_weak_dtypes(
    x: TensorValueLike, y: TensorValueLike
) -> tuple[TensorValue, TensorValue]:
    """Promote weak dtypes and handle device placement.

    Most of dtype promotion is dealt with in RMO.
    This function specifically deals with promotion of non-max objects.
    All non-max objects have a weak dtype and will promote to a max object dtype.
    That said, we will always scan the non-max object to ensure it is representable in the max object dtype.
    Finally, if a mix of weak and strong types are given, place weak types
    on the strong type's device.
    """
    x_was_strong = _is_strong_tensor_value_like(x)
    y_was_strong = _is_strong_tensor_value_like(y)

    if x_was_strong and y_was_strong:
        return (TensorValue(x), TensorValue(y))

    if not x_was_strong and not y_was_strong:
        raise TypeError(
            "Binary ops require at least one max object as input. Non-max"
            " objects tend to overpromote the dtype, leading to significant"
            " loss of performance. Please explicitly convert at least one"
            " input to a graph.Value. This can be done with ops.constant."
        )

    if x_was_strong:
        max_value = TensorValue(x)
        # TODO(GEX-2125): remove `or DeviceRef.CPU()` if once they are a non-optional field
        return (
            max_value,
            _promote_to_strong(
                y, max_value.dtype, max_value.device or DeviceRef.CPU()
            ),
        )
    else:
        max_value = TensorValue(y)
        return (
            _promote_to_strong(
                x, max_value.dtype, max_value.device or DeviceRef.CPU()
            ),
            max_value,
        )


def _promote_to_strong(
    value: TensorValueLike, strong_dtype: DType, device: DeviceRef
) -> TensorValue:
    """Promotes weak dtypes and handle device placement.

    If the input value is already strong, its dtype will not be changed.
    Instead, strong dtype promotion will be handled by the individual ops in RMO.
    """
    if isinstance(value, StaticDim):
        value = int(value)

    if _is_strong_tensor_value_like(value):
        return TensorValue(value)
    elif isinstance(value, int | np.integer):
        min, max = _DTYPE_MIN_AND_MAX_FULL_PRECISION[strong_dtype]
        if min <= value <= max:
            return ops.constant(value, strong_dtype, device)

        raise ValueError(
            f"Unsafe cast: Can't promote python int with value ({value}) to"
            f" dtype({strong_dtype}). It would lose precision."
        )

    elif isinstance(value, float | np.floating):
        if strong_dtype.is_float():
            return ops.constant(value, strong_dtype, device)

        raise ValueError(
            f"Unsafe cast: Can't promote python float to dtype({strong_dtype})."
        )

    elif isinstance(value, DLPackArray):
        tensor = Buffer.from_dlpack(value)

        if tensor.dtype.is_float() and strong_dtype.is_integral():
            raise ValueError(
                f"Unsafe cast: Refusing to implicitly promote float array "
                f"to dtype({strong_dtype})."
            )

        if tensor.dtype.is_integral():
            min, max = _DTYPE_MIN_AND_MAX_FULL_PRECISION[strong_dtype]
            if not all(
                min <= tensor[idx].item() <= max
                for idx in tensor._iterate_indices()
            ):
                raise ValueError(
                    "Unsafe cast: Refusing to implicitly promote external array "
                    f"with precision loss for DType {strong_dtype}. {value=}"
                )

        result = ops.constant(tensor, device=device)
        if result.dtype != strong_dtype:
            result = result.cast(strong_dtype)
        return result

    else:
        raise TypeError(
            "_promote_weak_dtypes() argument must be a TensorValueLike, not"
            f" '{type(value).__name__}'"
        )


# For each DType, this is the range of values where a conversion would not lose precision.
# This is used for conversions from python/numpy int to said DType.
_DTYPE_MIN_AND_MAX_FULL_PRECISION = {
    DType.bool: (0, 1),
    DType.int8: (-(2**7), 2**7 - 1),
    DType.int16: (-(2**15), 2**15 - 1),
    DType.int32: (-(2**31), 2**31 - 1),
    DType.int64: (-(2**63), 2**63 - 1),
    DType.uint8: (0, 2**8 - 1),
    DType.uint16: (0, 2**16 - 1),
    DType.uint32: (0, 2**32 - 1),
    DType.uint64: (0, 2**64 - 1),
    DType.float4_e2m1fn: (-(1.5 * 2**2), (1.5 * 2**2)),
    DType.float6_e2m3fn: (-(1.875 * 2**2), 1.875 * 2**2),
    DType.float6_e3m2fn: (-(1.75 * 2**4), 1.75 * 2**4),
    DType.float8_e8m0fnu: (2**-127, 2**127),
    DType.float8_e5m2: (-(1.10 * 2**16), 1.10 * 2**16),
    DType.float8_e5m2fnuz: (-(1.75 * 2**15), 1.75 * 2**15),
    DType.float8_e4m3fn: (-(1.75 * 2**8), 1.75 * 2**8),
    DType.float8_e4m3fnuz: (-240, 240),
    # This is two to the power of the number of significand bits plus one.
    DType.bfloat16: (-(2**8), 2**8),
    DType.float16: (-(2**11), 2**11),
    DType.float32: (-(2**24), 2**24),
    DType.float64: (-(2**53), 2**53),
}
