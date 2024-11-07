# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
from bit import count_leading_zeros
```
"""

from sys import llvm_intrinsic, sizeof
from sys.info import bitwidthof

# ===----------------------------------------------------------------------===#
# count_leading_zeros
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn count_leading_zeros(val: Int) -> Int:
    """Counts the number of leading zeros of an integer.

    Args:
        val: The input value.

    Returns:
        The number of leading zeros of the input.
    """
    return llvm_intrinsic["llvm.ctlz", Int, has_side_effect=False](val, False)


@always_inline("nodebug")
fn count_leading_zeros[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Counts the per-element number of leading zeros in a SIMD vector.

    Parameters:
        type: `DType` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` contains the number of
        leading zeros at position `i` of the input value.
    """
    constrained[type.is_integral(), "must be integral"]()
    return llvm_intrinsic["llvm.ctlz", __type_of(val), has_side_effect=False](
        val, False
    )


# ===----------------------------------------------------------------------===#
# count_trailing_zeros
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn count_trailing_zeros(val: Int) -> Int:
    """Counts the number of trailing zeros for an integer.

    Args:
        val: The input value.

    Returns:
        The number of trailing zeros of the input.
    """
    return llvm_intrinsic["llvm.cttz", Int, has_side_effect=False](val, False)


@always_inline("nodebug")
fn count_trailing_zeros[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Counts the per-element number of trailing zeros in a SIMD vector.

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` contains the number of
        trailing zeros at position `i` of the input value.
    """
    constrained[type.is_integral(), "must be integral"]()
    return llvm_intrinsic["llvm.cttz", __type_of(val), has_side_effect=False](
        val, False
    )


# ===----------------------------------------------------------------------===#
# bit_reverse
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bit_reverse(val: Int) -> Int:
    """Reverses the bitpattern of an integer value.

    Args:
        val: The input value.

    Returns:
        The input value with its bitpattern reversed.
    """
    return llvm_intrinsic["llvm.bitreverse", Int, has_side_effect=False](val)


@always_inline("nodebug")
fn bit_reverse[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Element-wise reverses the bitpattern of a SIMD vector of integer values.

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Args:
        val: The input value.

    Constraints:
        The element type of the input vector must be integral.

    Returns:
        A SIMD value where the element at position `i` has a reversed bitpattern
        of an integer value of the element at position `i` of the input value.
    """
    constrained[type.is_integral(), "must be integral"]()
    return llvm_intrinsic[
        "llvm.bitreverse", __type_of(val), has_side_effect=False
    ](val)


# ===----------------------------------------------------------------------===#
# byte_swap
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn byte_swap(val: Int) -> Int:
    """Byte-swaps an integer value with an even number of bytes.

    Byte swap an integer value (8 bytes) with an even number of bytes (positive multiple
    of 16 bits). This returns an integer value (8 bytes) that has its bytes swapped. For
    example, if the input bytes are numbered 0, 1, 2, 3, 4, 5, 6, 7 then the returned
    integer will have its bytes in 7, 6, 5, 4, 3, 2, 1, 0 order.

    Args:
        val: The input value.

    Returns:
        The input value with its bytes swapped.
    """
    return llvm_intrinsic["llvm.bswap", Int, has_side_effect=False](val)


@always_inline("nodebug")
fn byte_swap[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Byte-swaps a SIMD vector of integer values with an even number of bytes.

    Byte swap an integer value or vector of integer values with an even number
    of bytes (positive multiple of 16 bits). For example, The Int16 returns an
    Int16 value that has the high and low byte of the input Int16 swapped.
    Similarly, Int32 returns an Int32 value that has the four bytes of the input Int32 swapped,
    so that if the input bytes are numbered 0, 1, 2, 3 then the returned Int32 will
    have its bytes in 3, 2, 1, 0 order. Int64 and other integer type extend this
    concept to additional even-byte lengths (6 bytes, 8 bytes and more, respectively).

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be an integral type with an
        even number of bytes (Bitwidth % 16 == 0).

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is the value of the
        element at position `i` of the input value with its bytes swapped.
    """
    constrained[type.is_integral(), "must be integral"]()
    return llvm_intrinsic["llvm.bswap", __type_of(val), has_side_effect=False](
        val
    )


# ===----------------------------------------------------------------------===#
# pop_count
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn pop_count(val: Int) -> Int:
    """Counts the number of bits set in an integer value.

    Args:
        val: The input value.

    Returns:
        The number of bits set in the input value.
    """
    return llvm_intrinsic["llvm.ctpop", Int, has_side_effect=False](val)


@always_inline("nodebug")
fn pop_count[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Counts the number of bits set in a SIMD vector of integer values.

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` contains the number of
        bits set in the element at position `i` of the input value.
    """
    constrained[type.is_integral(), "must be integral"]()
    return llvm_intrinsic["llvm.ctpop", __type_of(val), has_side_effect=False](
        val
    )


# ===----------------------------------------------------------------------===#
# bit_not
# ===----------------------------------------------------------------------===#


@always_inline
fn bit_not[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Performs a bitwise NOT operation on an SIMD vector of integer values.

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is computed as a bitwise
        NOT of the integer value at position `i` of the input value.
    """
    constrained[type.is_integral(), "must be integral"]()
    var neg_one = SIMD[type, width](-1)
    return __mlir_op.`pop.simd.xor`(val._value, neg_one._value)


# ===----------------------------------------------------------------------===#
# bit_width
# ===----------------------------------------------------------------------===#


@always_inline
fn bit_width(val: Int) -> Int:
    """Computes the minimum number of bits required to represent the integer.

    Args:
        val: The input value.

    Returns:
        The number of bits required to represent the integer.
    """
    alias bitwidth = bitwidthof[Int]()

    return bitwidth - count_leading_zeros(~val if val < 0 else val)


@always_inline
fn bit_width[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Computes the minimum number of bits required to represent the SIMD vector
    of integer values.

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` equals to the number of
        bits required to represent the integer at position `i` of the input
        value.
    """

    constrained[type.is_integral(), "must be integral"]()

    alias bitwidth = bitwidthof[type]()

    @parameter
    if type.is_unsigned():
        return bitwidth - count_leading_zeros(val)
    else:
        var leading_zero_pos = count_leading_zeros(val)
        var leading_zero_neg = count_leading_zeros(bit_not(val))
        var leading_zero = (val < 0).select(leading_zero_neg, leading_zero_pos)
        return bitwidth - leading_zero


# ===----------------------------------------------------------------------===#
# is_power_of_two
# ===----------------------------------------------------------------------===#
# reference: https://en.cppreference.com/w/cpp/numeric/has_single_bit


@always_inline
fn is_power_of_two(val: Int) -> Bool:
    """Checks if the input value is a power of 2.

    Args:
        val: The input value.

    Returns:
        True if the input value is a power of 2, False otherwise.
    """
    return (val > 0) & (val & (val - 1) == 0)


@always_inline
fn is_power_of_two[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[DType.bool, width]:
    """Checks if the input value is a power of 2 for each element of a SIMD vector.

    Parameters:
        type: `dtype` used for the computation.
        width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is True if the integer at
        position `i` of the input value is a power of 2, False otherwise.
    """
    constrained[type.is_integral(), "must be integral"]()

    @parameter
    if type.is_unsigned():
        return pop_count(val) == 1
    else:
        return (val > 0) & (val & (val - 1) == 0)


# ===----------------------------------------------------------------------===#
# bit_ceil
# ===----------------------------------------------------------------------===#
# reference: https://en.cppreference.com/w/cpp/numeric/bit_ceil


@always_inline
fn bit_ceil(val: Int) -> Int:
    """Computes the smallest power of 2 that is greater than or equal to the
    input value. Any integral value less than or equal to 1 will be ceiled to 1.

    Args:
        val: The input value.

    Returns:
        The smallest power of 2 that is greater than or equal to the input value.
    """
    if val <= 1:
        return 1

    if is_power_of_two(val):
        return val

    return 1 << bit_width(val - 1)


@always_inline
fn bit_ceil[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Computes the smallest power of 2 that is greater than or equal to the
    input value for each element of a SIMD vector. Any integral value less than
    or equal to 1 will be ceiled to 1.

    Parameters:
        type: `dtype` used for the computation.
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
    constrained[type.is_integral(), "must be integral"]()

    alias ones = SIMD[type, width](1)

    return (val > 1).select(1 << bit_width(val - ones), ones)


# ===----------------------------------------------------------------------===#
# bit_floor
# ===----------------------------------------------------------------------===#
# reference: https://en.cppreference.com/w/cpp/numeric/bit_floor


@always_inline
fn bit_floor(val: Int) -> Int:
    """Computes the largest power of 2 that is less than or equal to the input
    value. Any integral value less than or equal to 0 will be floored to 0.

    Args:
        val: The input value.

    Returns:
        The largest power of 2 that is less than or equal to the input value.
    """
    if val <= 0:
        return 0

    return 1 << (bit_width(val) - 1)


@always_inline
fn bit_floor[
    type: DType, width: Int, //
](val: SIMD[type, width]) -> SIMD[type, width]:
    """Computes the largest power of 2 that is less than or equal to the input
    value for each element of a SIMD vector. Any integral value less than or
    equal to 0 will be floored to 0.

    Parameters:
        type: `dtype` used for the computation.
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
    constrained[type.is_integral(), "must be integral and unsigned"]()

    alias zeros = SIMD[type, width](0)

    return (val > 0).select(1 << (bit_width(val) - 1), zeros)


# ===----------------------------------------------------------------------===#
# rotate_bits_left
# ===----------------------------------------------------------------------===#


@always_inline
fn rotate_bits_left[shift: Int](x: Int) -> Int:
    """Shifts the bits of an input to the left by `shift` bits (with
    wrap-around).

    Constraints:
        `-size <= shift < size`

    Parameters:
        shift: The number of bit positions by which to rotate the bits of the
               integer to the left (with wrap-around).

    Args:
        x: The input value.

    Returns:
        The input rotated to the left by `shift` elements (with wrap-around).
    """
    constrained[
        shift >= -bitwidthof[Int]() and shift < bitwidthof[Int](),
        "Constraints: -bitwidthof[Int]() <= shift < bitwidthof[Int]()",
    ]()

    @parameter
    if shift == 0:
        return x
    elif shift < 0:
        return rotate_bits_right[-shift](x)
    else:
        return llvm_intrinsic["llvm.fshl", Int, has_side_effect=False](
            x, x, shift
        )


fn rotate_bits_left[
    type: DType,
    width: Int, //,
    shift: Int,
](x: SIMD[type, width]) -> SIMD[type, width]:
    """Shifts bits to the left by `shift` positions (with wrap-around) for each
    element of a SIMD vector.

    Constraints:
        `0 <= shift < size`

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
              Constraints: must be integral and unsigned.
        width: The width of the input and output SIMD vector.
        shift: The number of positions by which to shift left the bits for each
               element of a SIMD vector to the left (with wrap-around).

    Args:
        x: SIMD vector to perform the operation on.

    Returns:
        The SIMD vector with each element's bits shifted to the left by `shift`
        bits (with wrap-around).
    """

    constrained[type.is_unsigned(), "Only unsigned types can be rotated."]()

    @parameter
    if shift == 0:
        return x
    elif shift < 0:
        return rotate_bits_right[-shift](x)
    else:
        return llvm_intrinsic["llvm.fshl", __type_of(x), has_side_effect=False](
            x, x, SIMD[type, width](shift)
        )


# ===----------------------------------------------------------------------===#
# rotate_bits_right
# ===----------------------------------------------------------------------===#


@always_inline
fn rotate_bits_right[shift: Int](x: Int) -> Int:
    """Shifts the bits of an input to the right by `shift` bits (with
    wrap-around).

    Constraints:
        `-size <= shift < size`

    Parameters:
        shift: The number of bit positions by which to rotate the bits of the
               integer to the right (with wrap-around).

    Args:
        x: The input value.

    Returns:
        The input rotated to the right by `shift` elements (with wrap-around).
    """
    constrained[
        shift >= -bitwidthof[Int]() and shift < bitwidthof[Int](),
        "Constraints: -bitwidthof[Int]() <= shift < bitwidthof[Int]()",
    ]()

    @parameter
    if shift == 0:
        return x
    elif shift < 0:
        return rotate_bits_left[-shift](x)
    else:
        return llvm_intrinsic["llvm.fshr", Int, has_side_effect=False](
            x, x, shift
        )


fn rotate_bits_right[
    type: DType,
    width: Int, //,
    shift: Int,
](x: SIMD[type, width]) -> SIMD[type, width]:
    """Shifts bits to the right by `shift` positions (with wrap-around) for each
    element of a SIMD vector.

    Constraints:
        `0 <= shift < size`

    Parameters:
        type: The `dtype` of the input and output SIMD vector.
              Constraints: must be integral and unsigned.
        width: The width of the input and output SIMD vector.
        shift: The number of positions by which to shift right the bits for each
               element of a SIMD vector to the left (with wrap-around).

    Args:
        x: SIMD vector to perform the operation on.

    Returns:
        The SIMD vector with each element's bits shifted to the right by `shift`
        bits (with wrap-around).
    """

    constrained[type.is_unsigned(), "Only unsigned types can be rotated."]()

    @parameter
    if shift == 0:
        return x
    elif shift < 0:
        return rotate_bits_left[-shift](x)
    else:
        return llvm_intrinsic["llvm.fshr", __type_of(x), has_side_effect=False](
            x, x, SIMD[type, width](shift)
        )
