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

"""
Implementation of the following papers:
- Number Parsing at a Gigabyte per Second by Daniel Lemire
  - https://arxiv.org/abs/2101.11408
- Fast Number Parsing Without Fallback by Noble Mushtak & Daniel Lemire
  - https://arxiv.org/abs/2212.06644

The reference implementation used was the one in C# and can be found here:
- https://github.com/CarlVerret/csFastFloat
"""

from std.collections import Array

import std.bit
import std.memory

from std.builtin.globals import global_constant

from .constants import (
    CONTAINER_SIZE,
    MANTISSA_EXPLICIT_BITS,
    POWERS_OF_10,
    SMALLEST_POWER_OF_5,
    get_power_of_5,
)
from .parsing_integers import to_integer


@fieldwise_init
struct UInt128Decomposed(ImplicitlyCopyable, RegisterPassable):
    var high: UInt64
    var low: UInt64

    def __init__(out self, value: UInt128):
        self.high = UInt64(value >> 64)
        self.low = UInt64(value & 0xFFFFFFFFFFFFFFFF)

    def most_significant_bit(self) -> UInt64:
        return self.high >> 63


def _get_w_and_q_from_float_string(
    input_string: StringSlice[mut=False, _],
) raises -> Tuple[UInt64, Int64]:
    """We suppose the number is in the form '123.2481' or '123' or '123e-2' or '12.3e2'.

    Returns a tuple (w, q) where w is the significand and q is the exponent.
    w is an unsigned integer and q is a signed integer. (64 bits each)

    "123.2481e-5" -> (1232481, -9)
    """
    # We read the number from right to left.
    comptime ord_0 = Byte(ord("0"))
    comptime ord_9 = Byte(ord("9"))
    comptime ord_dot = Byte(ord("."))
    comptime ord_minus = Byte(ord("-"))
    comptime ord_plus = Byte(ord("+"))
    comptime ord_e = Byte(ord("e"))
    comptime ord_E = Byte(ord("E"))

    var additional_exponent = 0
    var exponent_multiplier = 1

    # We'll assume that we'll never go over 24 digit for each number.
    var exponent = Array[Byte, CONTAINER_SIZE](fill=Byte(ord("0")))
    var significand = Array[Byte, CONTAINER_SIZE](fill=Byte(ord("0")))

    comptime array_ptr = Pointer[
        type_of(exponent), origin_of(exponent, significand)
    ]
    var prt_to_array = array_ptr(to=exponent)
    var array_index = CONTAINER_SIZE
    var buffer = input_string.as_bytes().unsafe_ptr()

    if (
        not (ord_0 <= buffer[unsafe_offset=0] <= ord_9)
        and buffer[unsafe_offset=0] != ord_dot
    ):
        raise Error(
            "The first character of '",
            input_string,
            "' should be a digit or dot to convert it to a float.",
        )

    if (
        not (
            ord_0
            <= buffer[unsafe_offset=input_string.byte_length() - 1]
            <= ord_9
        )
        and buffer[unsafe_offset=input_string.byte_length() - 1] != ord_dot
    ):
        raise Error(
            "The last character of '",
            input_string,
            "' should be a digit or dot to convert it to a float.",
        )

    var dot_or_e_found = False

    for i in range(input_string.byte_length() - 1, -1, -1):
        array_index -= 1
        if array_index < 0:
            raise Error(
                "The number is too long, it's not supported yet. '",
                input_string,
                "'",
            )
        if buffer[unsafe_offset=i] == ord_dot:
            dot_or_e_found = True
            if prt_to_array == array_ptr(to=exponent):
                # We thought we were writing the exponent, but we were writing the significand.
                significand = exponent.copy()
                exponent = Array[Byte, CONTAINER_SIZE](fill=Byte(ord("0")))
                prt_to_array = array_ptr(to=significand)

            additional_exponent = CONTAINER_SIZE - array_index - 1
            # We don't want to progress in the significand array.
            array_index += 1
        elif buffer[unsafe_offset=i] == ord_minus:
            # Next should be the E letter (or e), so we'll just continue.
            exponent_multiplier = -1
        elif buffer[unsafe_offset=i] == ord_plus:
            # Next should be the E letter (or e), so we'll just continue.
            pass
        elif (
            buffer[unsafe_offset=i] == ord_e or buffer[unsafe_offset=i] == ord_E
        ):
            dot_or_e_found = True
            # We finished writing the exponent.
            prt_to_array = array_ptr(to=significand)
            array_index = CONTAINER_SIZE
        elif (ord_0 <= buffer[unsafe_offset=i]) and (
            buffer[unsafe_offset=i] <= ord_9
        ):
            prt_to_array[][array_index] = buffer[unsafe_offset=i]
        else:
            raise Error(
                "Invalid character(s) in the number: '", input_string, "'"
            )

    if not dot_or_e_found:
        # We were reading the significand
        significand = exponent.copy()
        exponent = Array[Byte, CONTAINER_SIZE](fill=Byte(ord("0")))

    var exponent_as_integer = UInt64(exponent_multiplier) * to_integer(
        exponent
    ) - UInt64(additional_exponent)
    var significand_as_integer = to_integer(significand)
    return (significand_as_integer, Int64(exponent_as_integer))


def strip_unused_characters(x: StringSlice[mut=False, _]) -> type_of(x):
    return x.strip().removeprefix("+").removesuffix("f").removesuffix("F")


def get_sign(x: StringSlice[mut=False, _]) -> Tuple[Float64, type_of(x)]:
    if x.startswith("-"):
        return (-1.0, x[byte=1:])
    return (1.0, x)


# Powers of 10 and integers below 2**53 are exactly representable as Float64.
# Thus any operation done on them must be exact.
def can_use_clinger_fast_path(w: UInt64, q: Int64) -> Bool:
    return w <= UInt64(2**53) and (Int64(-22) <= q <= Int64(22))


def clinger_fast_path(w: UInt64, q: Int64) -> Float64:
    if q >= 0:
        return Float64(w) * global_constant[POWERS_OF_10]()[q]
    else:
        return Float64(w) / global_constant[POWERS_OF_10]()[-q]


def full_multiplication(x: UInt64, y: UInt64) -> UInt128Decomposed:
    # Note that there are assembly instructions to
    # do all that on some architectures.
    # That should speed things up.
    var result = UInt128(x) * UInt128(y)
    return UInt128Decomposed(result)


def get_128_bit_truncated_product(w: UInt64, q: Int64) -> UInt128Decomposed:
    comptime bit_precision = MANTISSA_EXPLICIT_BITS + 3
    var index = 2 * (q - SMALLEST_POWER_OF_5)
    var first_product = full_multiplication(w, get_power_of_5(Int(index)))

    var precision_mask = (UInt64(1) << bit_precision) - 1
    if (first_product.high & precision_mask) == precision_mask:
        var second_product = full_multiplication(
            w, get_power_of_5(Int(index + 1))
        )
        first_product.low = first_product.low + second_product.high
        if second_product.high > first_product.low:
            first_product.high = first_product.high + 1

    return first_product


def create_subnormal_float64(m: UInt64) -> Float64:
    return create_float64(m, -1023)


def create_float64(m: UInt64, p: Int64) -> Float64:
    var m_mask = UInt64(2**MANTISSA_EXPLICIT_BITS - 1)
    var p_shifted = UInt64(p + 1023) << MANTISSA_EXPLICIT_BITS
    var representation_as_int = (m & m_mask) | p_shifted
    return std.memory.bitcast[.float64](representation_as_int)


def lemire_algorithm(var w: UInt64, var q: Int64) -> Float64:
    # This algorithm has 22 steps described
    # in https://arxiv.org/pdf/2101.11408 (algorithm 1)
    # Step 1
    if w == 0 or q < -342:
        return 0.0

    # Step 2
    if q > 308:
        return FloatLiteral.infinity

    # Step 3
    var l = std.bit.count_leading_zeros(w)

    # Step 4
    w <<= l

    # Step 5
    var product = get_128_bit_truncated_product(w, q)

    # Step 6
    # This step is skipped because it has been proven not necessary.
    # The proof can be found in the following paper by
    # Noble Mushtak & Daniel Lemire:
    # Fast Number Parsing Without Fallback
    # https://arxiv.org/abs/2212.06644

    # Step 8
    # Comes before step 7 because we need the upper_bit
    var upper_bit = product.most_significant_bit()

    # Step 7
    var m = product.high >> (upper_bit + 9)

    # Step 9
    var p: Int64 = (
        (((Int64(152170) + Int64(65536)) * q) >> Int64(16))
        + Int64(63)
        - Int64(l)
        + Int64(upper_bit)
    )

    # Step 10
    if p <= (-1022 - 64):
        return 0.0

    # Step 11-15
    # Subnormal case
    if p < -1022:
        var s = -1022 - p
        m = m // (UInt64(2) ** UInt64(s))
        if m % 2 == 1:
            m += 1
        m >>= 1
        if m >= UInt64(2**MANTISSA_EXPLICIT_BITS):
            return create_float64(m, -1022)
        return create_subnormal_float64(m)

    # Step 16-18
    # Round ties to even
    if product.low <= 1 and (m & 3 == 1) and (Int64(-4) <= q <= Int64(23)):
        if std.bit.pop_count(product.high // m) == 1:
            m -= 2

    # step 19
    if m % 2 == 1:
        m += 1
    m //= 2

    # Step 20
    if m == UInt64(2**53):
        m //= 2
        p = p + 1

    # step 21
    if p > 1023:
        return FloatLiteral.infinity

    # Step 22
    return create_float64(m, p)


comptime _ascii_lower: Byte = Byte(ord("A") ^ ord("a"))


@always_inline
def _is_nan(stripped: StringSlice) -> Bool:
    comptime `n` = Byte(ord("n"))
    comptime `a` = Byte(ord("a"))
    var ptr = stripped.as_bytes().unsafe_ptr()
    return stripped.byte_length() == 3 and (
        (ptr[unsafe_offset=0] | _ascii_lower == `n`)
        and (ptr[unsafe_offset=1] | _ascii_lower == `a`)
        and (ptr[unsafe_offset=2] | _ascii_lower == `n`)
    )


@always_inline
def _is_inf(stripped: StringSlice) -> Bool:
    comptime `i` = Byte(ord("i"))
    comptime `n` = Byte(ord("n"))
    comptime `f` = Byte(ord("f"))
    comptime `t` = Byte(ord("t"))
    comptime `y` = Byte(ord("y"))
    var ptr = stripped.as_bytes().unsafe_ptr()
    var in_start = (ptr[unsafe_offset=0] | _ascii_lower == `i`) and (
        ptr[unsafe_offset=1] | _ascii_lower == `n`
    )
    # f was removed previously
    var is_in = stripped.byte_length() == 2 and in_start
    return in_start and (
        is_in
        or (
            stripped.byte_length() == 8
            and (ptr[unsafe_offset=2] | _ascii_lower == `f`)
            and (ptr[unsafe_offset=3] | _ascii_lower == `i`)
            and (ptr[unsafe_offset=4] | _ascii_lower == `n`)
            and (ptr[unsafe_offset=5] | _ascii_lower == `i`)
            and (ptr[unsafe_offset=6] | _ascii_lower == `t`)
            and (ptr[unsafe_offset=7] | _ascii_lower == `y`)
        )
    )


def _atof(x: StringSlice) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    Raises:
        If the given string cannot be parsed as an floating point value, for
        example in `atof("hi")`.

    Args:
        x: A string to be parsed as a floating point.

    Returns:
        An floating point value that represents the string, or otherwise raises.
    """
    if x == "" or x == ".":
        raise Error("String is not convertible to float: ", repr(x))
    var stripped = strip_unused_characters(x)
    var sign_and_stripped = get_sign(stripped)
    var sign = sign_and_stripped[0]
    stripped = sign_and_stripped[1]
    if _is_nan(stripped):
        return FloatLiteral.nan
    elif _is_inf(stripped):
        return FloatLiteral.infinity * sign
    var w_and_q: Tuple[UInt64, Int64]
    try:
        w_and_q = _get_w_and_q_from_float_string(stripped)
    except e:
        raise Error("String is not convertible to float: ", repr(x), ". ", e)
    var w = w_and_q[0]
    var q = w_and_q[1]

    if can_use_clinger_fast_path(w, q):
        return clinger_fast_path(w, q) * sign
    else:
        return lemire_algorithm(w, q) * sign
