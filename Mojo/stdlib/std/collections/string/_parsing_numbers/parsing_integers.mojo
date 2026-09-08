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

from std.sys import simd_width_of
from std.memory import unsafe_memcmp, unsafe_memcpy

from .constants import CONTAINER_SIZE, MAXIMUM_UINT64_AS_STRING


def standardize_string_slice(
    x: StringSlice[mut=False, _],
) -> Array[Byte, CONTAINER_SIZE]:
    """Put the input string in an inline array, aligned to the right and padded
    with "0" on the left.
    """
    var standardized_x = Array[Byte, CONTAINER_SIZE](fill=Byte(ord("0")))
    var std_x_ptr = standardized_x.unsafe_ptr()
    var x_len = x.byte_length()
    unsafe_memcpy(
        dest=std_x_ptr.unsafe_offset(CONTAINER_SIZE - x_len),
        src=x.as_bytes().unsafe_ptr(),
        count=x_len,
    )
    return standardized_x^


# The idea is to end up with an Array of size
# 24, which is enough to store the largest integer
# that can be represented in unsigned 64 bits (size 20), and
# is also SIMD friendly because divisible by 8, 4, 2, 1.
# This 24 could be computed at compile time and adapted
# to the simd width and the base, but Mojo's compile-time
# computation is not yet powerful enough yet.
# For now we focus on base 10.
def to_integer(x: StringSlice[mut=False, _]) raises -> UInt64:
    """The input does not need to be padded with "0" on the left.
    The function returns the integer value represented by the input string.
    """
    if x.byte_length() > MAXIMUM_UINT64_AS_STRING.byte_length():
        raise Error("The string has too many bytes: '", x.byte_length(), "'.")
    return to_integer(standardize_string_slice(x))


def to_integer(standardized_x: Array[Byte, CONTAINER_SIZE]) raises -> UInt64:
    """Takes a inline array containing the ASCII representation of a number.

    Notes:
        It must be padded with "0" on the left. Using an Array makes
        this SIMD friendly.

        We assume there are no leading or trailing whitespaces, no sign, no
        underscore.

        The function returns the integer value represented by the input string
        `"000000000048642165487456"` -> `48642165487456`.
    """

    var std_x_ptr = standardized_x.unsafe_ptr()
    # This could be done with simd if we see it's a bottleneck.
    for i in range(CONTAINER_SIZE):
        if not (Byte(ord("0")) <= std_x_ptr[unsafe_offset=i] <= Byte(ord("9"))):
            var num_str = StringSlice(
                unsafe_from_utf8=Span(
                    unsafe_ptr=std_x_ptr, length=len(standardized_x)
                )
            ).lstrip("0")

            raise Error(
                "Invalid character(s) in the number: '",
                num_str,
                "' at index: ",
                i,
            )

    # 24 is not divisible by 16, so we stop at 8. Later on,
    # when we have better compile-time computation, we can
    # change 24 to be adapted to the simd width.
    comptime simd_width = min(simd_width_of[DType.uint64](), 8)

    var accumulator = SIMD[.uint64, simd_width](0)

    # We use unsafe_memcmp to check that the number is not too large.
    comptime max_standardized_x = String(UInt64.MAX).ascii_rjust(
        CONTAINER_SIZE, "0"
    )
    var too_large = (
        unsafe_memcmp(
            std_x_ptr,
            max_standardized_x.as_bytes().unsafe_ptr(),
            CONTAINER_SIZE,
        )
        == 1
    )
    if too_large:
        var num_str = StringSlice(
            unsafe_from_utf8=Span(
                unsafe_ptr=std_x_ptr, length=len(standardized_x)
            )
        ).lstrip("0")
        raise Error(
            "The string is too large to be converted to an integer: '",
            num_str,
            "'.",
        )

    # actual conversion
    comptime vector_with_exponents = get_vector_with_exponents()

    comptime for i in range(CONTAINER_SIZE // simd_width):
        var ascii_vector = std_x_ptr.unsafe_offset(i * simd_width).unsafe_load[
            width=simd_width
        ]()
        var as_digits = ascii_vector - SIMD[.uint8, simd_width](ord("0"))
        var as_digits_index = as_digits.cast[.uint64]()
        comptime vector_slice = vector_with_exponents.unsafe_ptr().unsafe_offset(
            i * simd_width
        ).unsafe_load[
            width=simd_width
        ]()
        accumulator += as_digits_index * vector_slice
    return UInt64(Int(accumulator.reduce_add()))


def get_vector_with_exponents() -> Array[UInt64, CONTAINER_SIZE]:
    """Returns (0, 0, 0, 0, 10**19, 10**18, 10**17, ..., 10, 1)."""
    return Array[_, CONTAINER_SIZE](
        fill_with=lambda (i: Int) -> UInt64: (
            UInt64(10) ** UInt64(CONTAINER_SIZE - i - 1) if i
            >= 4 else UInt64(0)
        )
    )
