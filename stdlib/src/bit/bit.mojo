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
from bit import ctlz
```
"""

from sys import llvm_intrinsic
from sys.info import bitwidthof

# ===----------------------------------------------------------------------===#
# ctlz
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ctlz(val: Int) -> Int:
    """Counts the number of leading zeros of an integer.

    Args:
        val: The input value.

    Returns:
        The number of leading zeros of the input.
    """
    return llvm_intrinsic["llvm.ctlz", Int, has_side_effect=False](val, False)


@always_inline("nodebug")
fn ctlz[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Counts the per-element number of leading zeros in a SIMD vector.

    Parameters:
        type: `DType` used for the computation.
        simd_width: SIMD width used for the computation.

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
# cttz
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn cttz(val: Int) -> Int:
    """Counts the number of trailing zeros for an integer.

    Args:
        val: The input value.

    Returns:
        The number of trailing zeros of the input.
    """
    return llvm_intrinsic["llvm.cttz", Int, has_side_effect=False](val, False)


@always_inline("nodebug")
fn cttz[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Counts the per-element number of trailing zeros in a SIMD vector.

    Parameters:
        type: `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

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
# bitreverse
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bitreverse[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Element-wise reverses the bitpattern of an integral value.

    Parameters:
        type: `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

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
# bswap
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn bswap[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Byte-swaps a value.

    Byte swap an integer value or vector of integer values with an even number
    of bytes (positive multiple of 16 bits). This is equivalent to `llvm.bswap`
    intrinsic that has the following semantics:

    The `llvm.bswap.i16` intrinsic returns an i16 value that has the high and
    low byte of the input i16 swapped. Similarly, the `llvm.bswap.i32` intrinsic
    returns an i32 value that has the four bytes of the input i32 swapped, so
    that if the input bytes are numbered 0, 1, 2, 3 then the returned i32 will
    have its bytes in 3, 2, 1, 0 order. The `llvm.bswap.i48`, `llvm.bswap.i64`
    and other intrinsics extend this concept to additional even-byte lengths (6
    bytes, 8 bytes and more, respectively).

    Parameters:
        type: `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

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
# ctpop
# ===----------------------------------------------------------------------===#


@always_inline("nodebug")
fn ctpop[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Counts the number of bits set in a value.

    Parameters:
        type: `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

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


@always_inline("nodebug")
fn bit_not[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Performs a bitwise NOT operation on an integral.

    Parameters:
        type: `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` is computed as a bitwise
        NOT of the integer value at position `i` of the input value.
    """
    constrained[type.is_integral(), "must be integral"]()
    var neg_one = SIMD[type, simd_width].splat(-1)
    return __mlir_op.`pop.xor`(val.value, neg_one.value)


# ===----------------------------------------------------------------------===#
# bit_length
# ===----------------------------------------------------------------------===#


@always_inline
fn bit_length[
    type: DType, simd_width: Int
](val: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    """Computes the number of digits required to represent the integer.

    Parameters:
        type: `dtype` used for the computation.
        simd_width: SIMD width used for the computation.

    Constraints:
        The element type of the input vector must be integral.

    Args:
        val: The input value.

    Returns:
        A SIMD value where the element at position `i` equals to the number of
        digits required to represent the integer at position `i` of the input
        value.
    """

    constrained[type.is_integral(), "must be integral"]()

    alias bitwidth = bitwidthof[type]()

    @parameter
    if type.is_unsigned():
        return bitwidth - ctlz(val)
    else:
        var leading_zero_pos = ctlz(val)
        var leading_zero_neg = ctlz(bit_not(val))
        var leading_zero = (val > 0).select(leading_zero_pos, leading_zero_neg)
        return bitwidth - leading_zero
