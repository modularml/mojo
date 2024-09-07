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
"""Provides functions for base64 encoding strings.

You can import these APIs from the `base64` package. For example:

```mojo
from base64 import b64encode
```
"""

from collections import List
from sys import simdwidthof
import bit

# We make use of the following papers for the implementation, note that there
# are some small differences.
# Wojciech Muła, Daniel Lemire, Base64 encoding and decoding at almost the
# speed of a memory copy, Software: Practice and Experience 50 (2), 2020.
# https://arxiv.org/abs/1910.05109
#
# Wojciech Muła, Daniel Lemire, Faster Base64 Encoding and Decoding using AVX2
# Instructions, ACM Transactions on the Web 12 (3), 2018.
# https://arxiv.org/abs/1704.00605


# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _ascii_to_value(char: String) -> Int:
    """Converts an ASCII character to its integer value for base64 decoding.

    Args:
        char: A single character string.

    Returns:
        The integer value of the character for base64 decoding, or -1 if invalid.
    """
    var char_val = ord(char)

    if char == "=":
        return 0
    elif ord("A") <= char_val <= ord("Z"):
        return char_val - ord("A")
    elif ord("a") <= char_val <= ord("z"):
        return char_val - ord("a") + 26
    elif ord("0") <= char_val <= ord("9"):
        return char_val - ord("0") + 52
    elif char == "+":
        return 62
    elif char == "/":
        return 63
    else:
        return -1


# ===----------------------------------------------------------------------===#
# b64encode
# ===----------------------------------------------------------------------===#


@always_inline
fn _subtract_with_saturation[
    simd_size: Int, //, b: Int
](a: SIMD[DType.uint8, simd_size]) -> SIMD[DType.uint8, simd_size]:
    """The equivalent of https://doc.rust-lang.org/core/arch/x86_64/fn._mm_subs_epu8.html .
    This can be a single instruction on some architectures.
    """
    alias b_as_vector = SIMD[DType.uint8, simd_size](b)
    return max(a, b_as_vector) - b_as_vector


"""
| 6-bit Value | ASCII Range | Target index | Offset (6-bit to ASCII) |
|-------------|-------------|--------------|-------------------------|
| 0 ... 25    | A ... Z     | 13           | 65                      |
| 26 ... 51   | a ... z     | 0            | 71                      |
| 52 ... 61   | 0 ... 9     | 1 ... 10     | -4                      |
| 62          | +           | 11           | -19                     |
| 63          | /           | 12           | -16                     |
"""
alias UNUSED = 0
alias TABLE_BASE64_OFFSETS = SIMD[DType.uint8, 16](
    71, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -19, -16, 65, UNUSED, UNUSED
)


@always_inline
fn _bitcast[
    new_dtype: DType, new_size: Int
](owned input: SIMD) -> SIMD[new_dtype, new_size]:
    var result = UnsafePointer.address_of(input).bitcast[
        SIMD[new_dtype, new_size]
    ]()[]
    return result


fn _base64_simd_mask[
    simd_width: Int
](nb_value_to_load: Int) -> SIMD[DType.bool, simd_width]:
    # Let's make this less verbose when Mojo is more flexible with compile-time programming
    @parameter
    if simd_width == 16:
        return SIMD[DType.uint8, simd_width](
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15
        ) < UInt8(nb_value_to_load)
    elif simd_width == 32:
        return SIMD[DType.uint8, simd_width](
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
        ) < UInt8(nb_value_to_load)
    elif simd_width == 64:
        return SIMD[DType.uint8, simd_width](
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            27,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
        ) < UInt8(nb_value_to_load)
    else:
        constrained[False, msg="simd_width must be 16, 32 or 64"]()
        return SIMD[DType.bool, simd_width]()  # dummy, unreachable


fn _repeat_until[
    dtype: DType, input_size: Int, //, target_size: Int
](vector: SIMD[dtype, input_size]) -> SIMD[dtype, target_size]:
    @parameter
    if target_size == input_size:
        var same_vector = rebind[SIMD[dtype, target_size]](vector)
        return same_vector
    else:
        return _repeat_until[target_size](vector.join(vector))


fn _move_first_group_of_6_bits[
    simd_width: Int
](shuffled_vector: SIMD[DType.uint8, simd_width]) -> SIMD[
    DType.uint8, simd_width
]:
    alias mask_1 = _repeat_until[simd_width](
        SIMD[DType.uint8, 4](0b11111100, 0, 0, 0)
    )
    var masked_1 = shuffled_vector & mask_1
    var result = masked_1 >> 2
    return result


fn _move_second_group_of_6_bits[
    simd_width: Int
](shuffled_vector: SIMD[DType.uint8, simd_width]) -> SIMD[
    DType.uint8, simd_width
]:
    alias mask_2 = _repeat_until[simd_width](
        SIMD[DType.uint8, 4](0b00000011, 0b11110000, 0, 0)
    )
    var masked_2 = shuffled_vector & mask_2
    var masked_2_as_uint16 = _bitcast[DType.uint16, simd_width // 2](masked_2)
    var rotated_2 = bit.rotate_bits_right[4](masked_2_as_uint16)
    var result = _bitcast[DType.uint8, simd_width](rotated_2)
    return result


fn _move_third_group_of_6_bits[
    simd_width: Int
](shuffled_vector: SIMD[DType.uint8, simd_width]) -> SIMD[
    DType.uint8, simd_width
]:
    alias mask_3 = _repeat_until[simd_width](
        SIMD[DType.uint8, 4](
            0,
            0,
            0b00001111,
            0b11000000,
        )
    )
    var masked_3 = shuffled_vector & mask_3
    var masked_3_as_uint16 = _bitcast[DType.uint16, simd_width // 2](masked_3)
    var rotated_3 = bit.rotate_bits_left[2](masked_3_as_uint16)
    var result = _bitcast[DType.uint8, simd_width](rotated_3)
    return result


fn _move_fourth_group_of_6_bits[
    simd_width: Int
](shuffled_vector: SIMD[DType.uint8, simd_width]) -> SIMD[
    DType.uint8, simd_width
]:
    alias mask_4 = _repeat_until[simd_width](
        SIMD[DType.uint8, 4](
            0,
            0,
            0,
            0b00111111,
        )
    )
    result = shuffled_vector & mask_4
    return result


fn _shuffle_input_vector[
    simd_width: Int
](input_vector: SIMD[DType.uint8, simd_width]) -> SIMD[DType.uint8, simd_width]:
    # We reorder the bytes to fall in their correct 4 bytes chunks
    # When Mojo is a bit more flexible with compile-time programming, we should be
    # able to make this less verbose.
    @parameter
    if simd_width < 4:
        constrained[False, msg="simd_width must be at least 4"]()
        return SIMD[DType.uint8, simd_width]()  # dummy, unreachable
    elif simd_width == 4:
        return input_vector.shuffle[0, 1, 1, 2]()
    elif simd_width == 8:
        return input_vector.shuffle[0, 1, 1, 2, 3, 4, 4, 5]()
    elif simd_width == 16:
        return input_vector.shuffle[
            0, 1, 1, 2, 3, 4, 4, 5, 6, 7, 7, 8, 9, 10, 10, 11
        ]()
    elif simd_width == 32:
        return input_vector.shuffle[
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            9,
            10,
            10,
            11,
            12,
            13,
            13,
            14,
            15,
            16,
            16,
            17,
            18,
            19,
            19,
            20,
            21,
            22,
            22,
            23,
        ]()
    elif simd_width == 64:
        return input_vector.shuffle[
            0,
            1,
            1,
            2,
            3,
            4,
            4,
            5,
            6,
            7,
            7,
            8,
            9,
            10,
            10,
            11,
            12,
            13,
            13,
            14,
            15,
            16,
            16,
            17,
            18,
            19,
            19,
            20,
            21,
            22,
            22,
            23,
            24,
            25,
            25,
            26,
            27,
            28,
            28,
            29,
            30,
            31,
            31,
            32,
            33,
            34,
            34,
            35,
            36,
            37,
            37,
            38,
            39,
            40,
            40,
            41,
            42,
            43,
            43,
            44,
            45,
            46,
            46,
            47,
            48,
            49,
            49,
            50,
            51,
            52,
            52,
            53,
            54,
            55,
            55,
            56,
            57,
            58,
            58,
            59,
            60,
            61,
            61,
            62,
            63,
        ]()
    else:
        constrained[False, msg="simd_width must be at most 64"]()
        return SIMD[DType.uint8, simd_width]()  # dummy, unreachable


fn _to_b64_ascii[
    simd_width: Int
](input_vector: SIMD[DType.uint8, simd_width]) -> SIMD[DType.uint8, simd_width]:
    alias constant_13 = SIMD[DType.uint8, simd_width](13)

    # We reorder the bytes to fall in their correct 4 bytes chunks
    var shuffled_vector = _shuffle_input_vector(input_vector)

    # We have 4 different masks to extract each group of 6 bits from the 4 bytes
    var ready_to_encode_per_byte = (
        _move_first_group_of_6_bits(shuffled_vector)
        | _move_second_group_of_6_bits(shuffled_vector)
        | _move_third_group_of_6_bits(shuffled_vector)
        | _move_fourth_group_of_6_bits(shuffled_vector)
    )

    # See the table above for the offsets, we try to go from 6-bits values to target indexes.
    var saturated = _subtract_with_saturation[51](ready_to_encode_per_byte)

    var mask_below_25 = ready_to_encode_per_byte <= 25

    # Now are have the target indexes
    var indices = mask_below_25.select(constant_13, saturated)

    var offsets = TABLE_BASE64_OFFSETS._dynamic_shuffle(indices)

    return ready_to_encode_per_byte + offsets


@always_inline
fn _get_number_of_elements_to_store_from_number_of_elements_to_load[
    simd_width: Int
]() -> SIMD[DType.uint8, simd_width]:
    # fmt: off
    # We must use a temporary alias to make the compiler happy

    @parameter
    if simd_width == 4:
        alias result = SIMD[DType.uint8, simd_width](
            0, 
            4, 4, 4
        )
        return result
    elif simd_width == 8:
        alias result = SIMD[DType.uint8, simd_width](
            0,
            4, 4, 4,
            8, 8, 8, 
            12
        )
        return result
    elif simd_width == 16:
        alias result = SIMD[DType.uint8, simd_width](
            0, 
            4, 4, 4, 
            8, 8, 8, 
            12, 12, 12, 
            16, 16, 16,
            20, 20, 20
        )
        return result
    elif simd_width == 32:
        alias result = SIMD[DType.uint8, simd_width](
            0, 
            4, 4, 4, 
            8, 8, 8, 
            12, 12, 12, 
            16, 16, 16, 
            20, 20, 20, 
            24, 24, 24, 
            28, 28, 28, 
            32, 32, 32,
            36, 36, 36,
            40, 40, 40,
            44,
        )
        return result
    elif simd_width == 64:
        alias result = SIMD[DType.uint8, simd_width](
            0, 
            4, 4, 4, 
            8, 8, 8, 
            12, 12, 12, 
            16, 16, 16, 
            20, 20, 20, 
            24, 24, 24, 
            28, 28, 28, 
            32, 32, 32, 
            36, 36, 36, 
            40, 40, 40, 
            44, 44, 44, 
            48, 48, 48,
            52, 52, 52,
            56, 56, 56,
            60, 60, 60,
            64, 64, 64,
            68, 68, 68,
            72, 72, 72,
            76, 76, 76,
            80, 80, 80,
            84, 84, 84,
        )
        return result
    # fmt: on

    else:
        constrained[False, msg="simd_width must be 4, 8, 16, 32 or 64"]()
        return SIMD[DType.uint8, simd_width]() # dummy, unreachable


fn _get_number_of_non_equal_from_number_of_elements_to_load[
    simd_width: Int
]() -> SIMD[DType.uint8, simd_width]:
    @parameter
    if simd_width == 4:
        alias result = SIMD[DType.uint8, simd_width](
            0,
            2,
            3,
            4,
        )
        return result
    elif simd_width == 8:
        alias result = SIMD[DType.uint8, simd_width](
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            10,
        )
        return result
    elif simd_width == 16:
        alias result = SIMD[DType.uint8, simd_width](
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            10,
            11,
            12,
            14,
            15,
            16,
            18,
            19,
            20,
        )
        return result
    elif simd_width == 32:
        alias result = SIMD[DType.uint8, simd_width](
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            10,
            11,
            12,
            14,
            15,
            16,
            18,
            19,
            20,
            22,
            23,
            24,
            26,
            27,
            28,
            30,
            31,
            32,
            34,
            35,
            36,
            38,
            39,
            40,
            42,
        )
        return result
    elif simd_width == 64:
        alias result = SIMD[DType.uint8, simd_width](
            0,
            2,
            3,
            4,
            6,
            7,
            8,
            10,
            11,
            12,
            14,
            15,
            16,
            18,
            19,
            20,
            22,
            23,
            24,
            26,
            27,
            28,
            30,
            31,
            32,
            34,
            35,
            36,
            38,
            39,
            40,
            42,
            43,
            44,
            46,
            47,
            48,
            50,
            51,
            52,
            54,
            55,
            56,
            58,
            59,
            60,
            62,
            63,
            64,
            66,
            67,
            68,
            70,
            71,
            72,
            74,
            75,
            76,
            78,
            79,
            80,
            82,
            83,
            84,
        )
        return result
    else:
        constrained[False, msg="simd_width must be 4, 8, 16, 32 or 64"]()
        return SIMD[DType.uint8, simd_width]()  # dummy, unreachable


fn _print_vector_in_binary(vector: SIMD):
    for i in range(len(vector)):
        print(bin(vector[i]), end="")
    print()


# TODO: Use Span instead of List as input when Span is easier to use
fn b64encode(input_bytes: List[UInt8, _], inout result: List[UInt8, _]):
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer. Assumed to be null-terminated.
        result: The buffer in which to store the values.
    """
    alias simd_width = sys.simdbytewidth()
    alias input_simd_width = simd_width * 3 // 4
    alias equal_vector = SIMD[DType.uint8, simd_width](ord("="))

    # Could be computed at compile time when Mojo has better compile-time programming.
    # Otherwise it's fixed and not great if we want to change simd sizes
    alias number_of_non_equal_from_number_of_elements_to_load = _get_number_of_non_equal_from_number_of_elements_to_load[
        simd_width
    ]()
    alias number_of_bytes_to_store_from_nb_of_elements_to_load = _get_number_of_elements_to_store_from_number_of_elements_to_load[
        simd_width
    ]()
    var input_bytes_len = len(input_bytes)

    # TODO: add condition on cpu flags
    var input_index = 0
    while input_index + simd_width <= input_bytes_len:
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index

        # We don't want to read past the input buffer
        var input_vector = start_of_input_chunk.load[width=simd_width]()

        result_vector = _to_b64_ascii(input_vector)

        # We write the result to the output buffer
        (result.unsafe_ptr() + len(result)).store(result_vector)

        result.size += int(simd_width)
        input_index += input_simd_width

    while input_index < input_bytes_len:
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index
        var nb_of_elements_to_load = min(
            input_simd_width, input_bytes_len - input_index
        )
        var mask = _base64_simd_mask[simd_width](nb_of_elements_to_load)

        # We don't want to read past the input buffer
        var input_vector = sys.intrinsics.masked_load[simd_width](
            start_of_input_chunk,
            mask,
            passthrough=SIMD[DType.uint8, simd_width](0),
        )

        result_vector = _to_b64_ascii(input_vector)

        # We place the '=' where needed
        var non_equal_chars_number = number_of_non_equal_from_number_of_elements_to_load[
            nb_of_elements_to_load
        ]
        var equal_mask = _base64_simd_mask[simd_width](
            int(non_equal_chars_number)
        )

        var result_vector_with_equals = equal_mask.select(
            result_vector, equal_vector
        )

        var nb_of_elements_to_store = number_of_bytes_to_store_from_nb_of_elements_to_load[
            nb_of_elements_to_load
        ]
        var mask_store = _base64_simd_mask[simd_width](
            int(nb_of_elements_to_store)
        )
        # We write the result to the output buffer
        sys.intrinsics.masked_store(
            result_vector_with_equals,
            result.unsafe_ptr() + len(result),
            mask_store,
        )
        result.size += int(nb_of_elements_to_store)
        input_index += input_simd_width


# For a nicer API, we provide those overloads:
fn b64encode(input_string: String) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_string: The input string buffer. Assumed to be null-terminated.

    Returns:
        The ASCII base64 encoded string.
    """
    # Slicing triggers a copy, but it should work with Span later on.
    return b64encode(input_string._buffer[:-1])


fn b64encode(input_bytes: List[UInt8, _]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer. Assumed to be null-terminated.

    Returns:
        The ASCII base64 encoded string.
    """
    # +1 for the null terminator and +1 to be sure
    var result = List[UInt8, True](capacity=int(len(input_bytes) * (4 / 3)) + 2)
    b64encode(input_bytes, result)
    # null-terminate the result
    result.append(0)
    return String(result^)


# ===----------------------------------------------------------------------===#
# b64decode
# ===----------------------------------------------------------------------===#


@always_inline
fn b64decode(str: String) -> String:
    """Performs base64 decoding on the input string.

    Args:
      str: A base64 encoded string.

    Returns:
      The decoded string.
    """
    var n = str.byte_length()
    debug_assert(n % 4 == 0, "Input length must be divisible by 4")

    var p = String._buffer_type(capacity=n + 1)

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    for i in range(0, n, 4):
        var a = _ascii_to_value(str[i])
        var b = _ascii_to_value(str[i + 1])
        var c = _ascii_to_value(str[i + 2])
        var d = _ascii_to_value(str[i + 3])

        debug_assert(
            a >= 0 and b >= 0 and c >= 0 and d >= 0,
            "Unexpected character encountered",
        )

        p.append((a << 2) | (b >> 4))
        if str[i + 2] == "=":
            break

        p.append(((b & 0x0F) << 4) | (c >> 2))

        if str[i + 3] == "=":
            break

        p.append(((c & 0x03) << 6) | d)

    p.append(0)
    return p


# ===----------------------------------------------------------------------===#
# b16encode
# ===----------------------------------------------------------------------===#


fn b16encode(str: String) -> String:
    """Performs base16 encoding on the input string.

    Args:
      str: The input string.

    Returns:
      Base16 encoding of the input string.
    """
    alias lookup = "0123456789ABCDEF"
    var b16chars = lookup.unsafe_ptr()

    var length = str.byte_length()
    var out = List[UInt8](capacity=length * 2 + 1)

    @parameter
    @always_inline
    fn str_bytes(idx: UInt8) -> UInt8:
        return str._buffer[int(idx)]

    for i in range(length):
        var str_byte = str_bytes(i)
        var hi = str_byte >> 4
        var lo = str_byte & 0b1111
        out.append(b16chars[int(hi)])
        out.append(b16chars[int(lo)])

    out.append(0)

    return String(out^)


# ===----------------------------------------------------------------------===#
# b16decode
# ===----------------------------------------------------------------------===#


@always_inline
fn b16decode(str: String) -> String:
    """Performs base16 decoding on the input string.

    Args:
      str: A base16 encoded string.

    Returns:
      The decoded string.
    """

    # TODO: Replace with dict literal when possible
    @parameter
    @always_inline
    fn decode(c: String) -> Int:
        var char_val = ord(c)

        if ord("A") <= char_val <= ord("Z"):
            return char_val - ord("A") + 10
        elif ord("a") <= char_val <= ord("z"):
            return char_val - ord("a") + 10
        elif ord("0") <= char_val <= ord("9"):
            return char_val - ord("0")

        return -1

    var n = str.byte_length()
    debug_assert(n % 2 == 0, "Input length must be divisible by 2")

    var p = List[UInt8](capacity=n // 2 + 1)

    for i in range(0, n, 2):
        var hi = str[i]
        var lo = str[i + 1]
        p.append(decode(hi) << 4 | decode(lo))

    p.append(0)
    return p
