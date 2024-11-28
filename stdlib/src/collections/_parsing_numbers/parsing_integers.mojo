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

from utils.string_slice import StringSlice
from .constants import std_size, maximum_uint64_as_str
from memory import memcpy, memcmp


fn standardize_string_slice(
    x: StringSlice,
) -> InlineArray[UInt8, size=std_size]:
    """Put the input string in an inline array, aligned to the right and padded with "0" on the left.
    """
    var standardized_x = InlineArray[UInt8, size=std_size](ord("0"))
    memcpy(
        dest=(standardized_x.unsafe_ptr() + std_size - len(x)).bitcast[Int8](),
        src=x.unsafe_ptr().bitcast[Int8](),
        count=len(x),
    )
    return standardized_x


# The idea is to end up with a InlineArray of size
# 24, which is enough to store the largest integer
# that can be represented in unsigned 64 bits (size 20), and
# is also SIMD friendly because divisible by 8, 4, 2, 1.
# This 24 could be computed at compile time and adapted
# to the simd width and the base, but Mojo's compile-time
# computation is not yet powerful enough yet.
# For now we focus on base 10.
fn to_integer(x: String) raises -> UInt64:
    return to_integer(x.as_string_slice())


fn to_integer(x: StringSlice) raises -> UInt64:
    """The input does not need to be padded with "0" on the left.
    The function returns the integer value represented by the input string.
    """
    if len(x) > len(maximum_uint64_as_str):
        raise Error("The string size too big. '" + str(x) + "'")
    return to_integer(standardize_string_slice(x))


fn to_integer(
    standardized_x: InlineArray[UInt8, size=std_size]
) raises -> UInt64:
    """Takes a inline array containing the ASCII representation of a number.
    It must be padded with "0" on the left. Using an InlineArray makes
    this SIMD friendly.

    We assume there are no leading or trailing whitespaces, no sign, no underscore.

    The function returns the integer value represented by the input string.

    "000000000048642165487456" -> 48642165487456
    """

    # This could be done with simd if we see it's a bottleneck.
    for i in range(std_size):
        if not (UInt8(ord("0")) <= standardized_x[i] <= UInt8(ord("9"))):
            # We make a string out of this number. +1 for the null terminator.
            number_as_string = String._buffer_type(capacity=std_size + 1)
            for j in range(std_size):
                number_as_string.append(standardized_x[j])
            number_as_string.append(0)
            raise Error(
                "Invalid character(s) in the number: '"
                + String(number_as_string^)
                + "'"
            )

    # 24 is not divisible by 16, so we stop at 8. Later on,
    # when we have better compile-time computation, we can
    # change 24 to be adapted to the simd width.
    alias simd_width = min(sys.simdwidthof[DType.uint64](), 8)

    var accumulator = SIMD[DType.uint64, simd_width](0)

    # We use memcmp to check that the number is not too large.

    # Must be a var and not an alias, otherwise we get use a after free
    # error. Even when using _ = max_standardized_x.
    var max_standardized_x = "000018446744073709551615"
    if (
        memcmp(
            standardized_x.unsafe_ptr(),
            max_standardized_x.unsafe_ptr(),
            count=std_size,
        )
        == 1
    ):  # memcmp is pretty fast
        raise Error("The string is too large to be converted to an integer. '")
    _ = max_standardized_x

    # actual conversion
    # Here it only works for base 10. When we can do more things in the
    # parameter domain, we can make it work for other bases.
    alias vector_with_exponents = InlineArray[
        SIMD[DType.uint64, 1], size=std_size
    ](
        0,
        0,
        0,
        0,
        10000000000000000000,
        1000000000000000000,
        100000000000000000,
        10000000000000000,
        1000000000000000,
        100000000000000,
        10000000000000,
        1000000000000,
        100000000000,
        10000000000,
        1000000000,
        100000000,
        10000000,
        1000000,
        100000,
        10000,
        1000,
        100,
        10,
        1,
    )

    @parameter
    for i in range(std_size // simd_width):
        var ascii_vector = (standardized_x.unsafe_ptr() + i * simd_width).load[
            width=simd_width
        ]()
        var as_digits = ascii_vector - SIMD[DType.uint8, simd_width](ord("0"))
        var as_digits_index = as_digits.cast[DType.uint64]()
        alias vector_slice = (
            vector_with_exponents.unsafe_ptr() + i * simd_width
        ).load[width=simd_width]()
        accumulator += as_digits_index * vector_slice
    return int(accumulator.reduce_add())
