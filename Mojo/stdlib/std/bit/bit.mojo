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
"""Provides functions for bit manipulation.

You can import these APIs from the `bit` package. For example:

```mojo
from std.bit import count_leading_zeros
```
"""

from std.sys import llvm_intrinsic, bit_width_of

from std.bit.mask import is_negative

from std.utils._select import _select_register_value as select

# ===-----------------------------------------------------------------------===#
# count_leading_zeros
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def count_leading_zeros[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Counts the per-element number of leading zeros in a SIMD vector.

    Parameters:
        dtype: `DType` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` contains the number of
        leading zeros at position `i` of the input value.
    """
    comptime assert dtype.is_integral(), "must be integral"
    return llvm_intrinsic["llvm.ctlz", type_of(val), has_side_effect=False](
        val, False
    )


# ===-----------------------------------------------------------------------===#
# count_trailing_zeros
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def count_trailing_zeros[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Counts the per-element number of trailing zeros in a SIMD vector.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` contains the number of
        trailing zeros at position `i` of the input value.
    """
    comptime assert dtype.is_integral(), "must be integral"
    return llvm_intrinsic["llvm.cttz", type_of(val), has_side_effect=False](
        val, False.__mlir_i1__()
    )


# ===-----------------------------------------------------------------------===#
# bit_reverse
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def bit_reverse[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Element-wise reverses the bitpattern of a SIMD vector of integer values.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Args:
        val: The input value.

    Constraints:
        The element type of the input vector must be integral.

    Returns:
        A SIMD value where the element at position `i` has a reversed bitpattern
        of an integer value of the element at position `i` of the input value.
    """
    comptime assert dtype.is_integral(), "must be integral"
    return llvm_intrinsic[
        "llvm.bitreverse", type_of(val), has_side_effect=False
    ](val)


# ===-----------------------------------------------------------------------===#
# byte_swap
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def byte_swap[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Byte-swaps a SIMD vector of integer values with an even number of bytes.

    Byte swap an integer value or vector of integer values with an even number
    of bytes (positive multiple of 16 bits). For example, The Int16 returns an
    Int16 value that has the high and low byte of the input Int16 swapped.
    Similarly, Int32 returns an Int32 value that has the four bytes of the input Int32 swapped,
    so that if the input bytes are numbered 0, 1, 2, 3 then the returned Int32 will
    have its bytes in 3, 2, 1, 0 order. Int64 and other integer type extend this
    concept to additional even-byte lengths (6 bytes, 8 bytes and more, respectively).

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be an integral type.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is the value of the
        element at position `i` of the input value with its bytes swapped.
    """
    comptime assert dtype.is_integral(), "must be integral"

    comptime if bit_width_of[dtype]() < 16:
        return val
    return llvm_intrinsic["llvm.bswap", type_of(val), has_side_effect=False](
        val
    )


# ===-----------------------------------------------------------------------===#
# pop_count
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def pop_count[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Counts the number of bits set in a SIMD vector of integer values.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` contains the number of
        bits set in the element at position `i` of the input value.
    """
    comptime assert dtype.is_integral(), "must be integral"
    return llvm_intrinsic["llvm.ctpop", type_of(val), has_side_effect=False](
        val
    )


# ===-----------------------------------------------------------------------===#
# bit_not
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def bit_not[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Performs a bitwise NOT operation on an SIMD vector of integer values.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is computed as a bitwise
        NOT of the integer value at position `i` of the input value.
    """
    comptime assert dtype.is_integral(), "must be integral"
    return ~val


# ===-----------------------------------------------------------------------===#
# bit_width
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def bit_width[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the minimum number of bits required to represent each element of a SIMD vector of integer values.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` equals the number of bits required to represent the integer at position `i` of the input.
    """
    comptime assert dtype.is_integral(), "must be integral"
    comptime bitwidth = bit_width_of[dtype]()

    comptime if dtype.is_unsigned():
        return SIMD[dtype, width](bitwidth) - count_leading_zeros(val)
    else:
        # For signed integers, handle positive and negative separately
        var abs_val = val.lt(0).select(bit_not(val), val)
        return SIMD[dtype, width](bitwidth) - count_leading_zeros(abs_val)


# ===-----------------------------------------------------------------------===#
# log2_floor
# ===-----------------------------------------------------------------------===#


@always_inline
def log2_floor[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Returns the floor of the base-2 logarithm of an integer value.

    Parameters:
        dtype: The `dtype` of the input SIMD vector.
        width: The width of the input and output SIMD vector.

    Args:
        val: The input value.

    Returns:
        The floor of the base-2 logarithm of the input value, which is equal to
        the position of the highest set bit. Returns -1 if val is 0 or negative.
    """
    comptime assert dtype.is_integral(), "dtype must be integral"

    comptime bitwidth = bit_width_of[dtype]()
    var res = SIMD[dtype, width](bitwidth) - count_leading_zeros(val) - 1

    comptime if dtype.is_signed():
        return res | is_negative(val)
    else:
        return res


# ===-----------------------------------------------------------------------===#
# log2_ceil
# ===-----------------------------------------------------------------------===#


@always_inline
def log2_ceil(val: Scalar) -> type_of(val):
    """Returns the ceiling of the base-2 logarithm of an integer value.

    Args:
        val: The input value.

    Returns:
        The smallest integer `n` such that `2^n` is greater than or equal to
        the input value. Returns 0 if `val` is 0.
    """
    comptime assert val.dtype.is_integral(), "the input dtype must be integral"
    return select(val <= 1, type_of(val)(0), log2_floor(val - 1) + 1)


# ===-----------------------------------------------------------------------===#
# next_power_of_two
# ===-----------------------------------------------------------------------===#
# reference: https://en.cppreference.com/w/cpp/numeric/bit_ceil
# reference: https://doc.rust-lang.org/std/primitive.usize.html#method.next_power_of_two


@always_inline
def next_power_of_two[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the smallest power of 2 that is greater than or equal to the
    input value for each element of a SIMD vector. Any integral value less than
    or equal to 1 will be ceiled to 1.

    This operation is called `bit_ceil()` in C++.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is the smallest power of 2
        that is greater than or equal to the integer at position `i` of the input
        value.
    """
    comptime assert dtype.is_integral(), "must be integral"
    return val.gt(1).select(SIMD[dtype, width](1 << bit_width(val - 1)), 1)


# ===-----------------------------------------------------------------------===#
# prev_power_of_two
# ===-----------------------------------------------------------------------===#
# reference: https://en.cppreference.com/w/cpp/numeric/bit_floor


@always_inline
def prev_power_of_two[
    dtype: DType, width: SIMDLength, //
](val: SIMD[dtype, width]) -> SIMD[dtype, width]:
    """Computes the largest power of 2 that is less than or equal to the input
    value for each element of a SIMD vector. Any integral value less than or
    equal to 0 will be floored to 0.

    This operation is called `bit_floor()` in C++.

    Parameters:
        dtype: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is the largest power of 2
        that is less than or equal to the integer at position `i` of the input
        value.
    """
    comptime assert dtype.is_integral(), "must be integral and unsigned"
    return val.gt(0).select(
        SIMD[dtype, width](1) << (bit_width(val) - 1), SIMD[dtype, width](0)
    )


# ===-----------------------------------------------------------------------===#
# rotate_bits_left
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def rotate_bits_left[
    dtype: DType,
    width: SIMDLength,
    //,
    shift: Int,
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_integral():
    """Shifts bits to the left by `shift` positions (with wrap-around) for each element of a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector. Must be integral.
        width: The width of the SIMD vector.
        shift: The number of positions to rotate left.

    Args:
        x: SIMD vector input.

    Returns:
        SIMD vector with each element rotated left by `shift` bits.
    """

    comptime if shift == 0:
        return x
    elif shift < 0:
        return rotate_bits_right[-shift](x)
    else:
        return llvm_intrinsic["llvm.fshl", type_of(x), has_side_effect=False](
            x, x, type_of(x)(shift)
        )


# ===-----------------------------------------------------------------------===#
# rotate_bits_right
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def rotate_bits_right[
    dtype: DType,
    width: SIMDLength,
    //,
    shift: Int,
](x: SIMD[dtype, width]) -> SIMD[dtype, width] where dtype.is_integral():
    """Shifts bits to the right by `shift` positions (with wrap-around) for each element of a SIMD vector.

    Parameters:
        dtype: The `dtype` of the input and output SIMD vector. Must be integral.
        width: The width of the SIMD vector.
        shift: The number of positions to rotate right.

    Args:
        x: SIMD vector input.

    Returns:
        SIMD vector with each element rotated right by `shift` bits.
    """

    comptime if shift == 0:
        return x
    elif shift < 0:
        return rotate_bits_left[-shift](x)
    else:
        return llvm_intrinsic["llvm.fshr", type_of(x), has_side_effect=False](
            x, x, type_of(x)(shift)
        )
