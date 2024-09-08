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

"""
We make use of the following papers for the implementation, note that there
are some small differences.
Wojciech Muła, Daniel Lemire, Base64 encoding and decoding at almost the
speed of a memory copy, Software: Practice and Experience 50 (2), 2020.
https://arxiv.org/abs/1910.05109

Wojciech Muła, Daniel Lemire, Faster Base64 Encoding and Decoding using AVX2
Instructions, ACM Transactions on the Web 12 (3), 2018.
https://arxiv.org/abs/1704.00605
"""

from collections import InlineArray
from memory import memcpy
from memory.maybe_uninitialized import UnsafeMaybeUninitialized


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


fn _bitcast[
    new_dtype: DType, new_size: Int
](owned input: SIMD) -> SIMD[new_dtype, new_size]:
    var result = UnsafePointer.address_of(input).bitcast[
        SIMD[new_dtype, new_size]
    ]()[]
    return result


fn _get_simd_range_values[simd_width: Int]() -> SIMD[DType.uint8, simd_width]:
    var a = SIMD[DType.uint8, simd_width](0)
    for i in range(simd_width):
        a[i] = i
    return a


fn _base64_simd_mask[
    simd_width: Int
](nb_value_to_load: Int) -> SIMD[DType.bool, simd_width]:
    alias mask = _get_simd_range_values[simd_width]()
    return mask < UInt8(nb_value_to_load)


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


fn _get_table_number_of_bytes_to_store_from_number_of_bytes_to_load[
    simd_width: Int
]() -> SIMD[DType.uint8, simd_width]:
    """This is a lookup table to know how many bytes we need to store in the output buffer
    for a given number of bytes to encode in base64. Including the '=' sign.

    This table lookup is smaller than the simd size, because we only use it for the last chunk.
    This should be called at compile time, otherwise it's quite slow.
    """
    var result = SIMD[DType.uint8, simd_width](0)
    for i in range(1, simd_width):
        # We have "i" bytes to encode in base64, how many bytes do
        # we need to store in the output buffer? Including the '=' sign.

        # math.ceil cannot be called at compile time, this is a workaround
        var group_of_3_bytes = i // 3
        if i % 3 != 0:
            group_of_3_bytes += 1

        result[i] = group_of_3_bytes * 4
    return result


fn _get_number_of_bytes_to_store_from_number_of_bytes_to_load[
    max_size: Int
](nb_of_elements_to_load: Int) -> Int:
    alias table = _get_table_number_of_bytes_to_store_from_number_of_bytes_to_load[
        max_size
    ]()
    return int(table[nb_of_elements_to_load])


fn _get_table_number_of_bytes_to_store_from_number_of_bytes_to_load_without_equal_sign[
    simd_width: Int
]() -> SIMD[DType.uint8, simd_width]:
    """This is a lookup table to know how many bytes we need to store in the output buffer
    for a given number of bytes to encode in base64. This is **not** including the '=' sign.

    This table lookup is smaller than the simd size, because we only use it for the last chunk.
    This should be called at compile time, otherwise it's quite slow.
    """
    var result = SIMD[DType.uint8, simd_width]()
    for i in range(simd_width):
        # We have "i" bytes to encode in base64, how many bytes do
        # we need to store in the output buffer? NOT including the '=' sign.
        # We count the number of groups of 6 bits and we add 1 byte if there is an incomplete group.
        var number_of_bits = i * 8
        var complete_groups_of_6_bits = number_of_bits // 6
        var incomplete_groups_of_6_bits: Int
        if i * 8 % 6 == 0:
            incomplete_groups_of_6_bits = 0
        else:
            incomplete_groups_of_6_bits = 1

        result[i] = complete_groups_of_6_bits + incomplete_groups_of_6_bits
    return result


fn _get_number_of_bytes_to_store_from_number_of_bytes_to_load_without_equal_sign[
    max_size: Int
](nb_of_elements_to_load: Int) -> Int:
    alias table = _get_table_number_of_bytes_to_store_from_number_of_bytes_to_load_without_equal_sign[
        max_size
    ]()
    return int(table[nb_of_elements_to_load])


fn load_incomplete_simd[
    simd_width: Int
](pointer: UnsafePointer[UInt8], nb_of_elements_to_load: Int) -> SIMD[
    DType.uint8, simd_width
]:
    var result = SIMD[DType.uint8, simd_width](0)
    var tmp_buffer_pointer = UnsafePointer.address_of(result).bitcast[UInt8]()
    memcpy(dest=tmp_buffer_pointer, src=pointer, count=nb_of_elements_to_load)
    return result


fn store_incomplete_simd[
    simd_width: Int
](
    pointer: UnsafePointer[UInt8],
    owned simd_vector: SIMD[DType.uint8, simd_width],
    nb_of_elements_to_store: Int,
):
    var tmp_buffer_pointer = UnsafePointer.address_of(simd_vector).bitcast[
        UInt8
    ]()

    memcpy(dest=pointer, src=tmp_buffer_pointer, count=nb_of_elements_to_store)
    _ = simd_vector  # We make it live long enough


# TODO: Use Span instead of List as input when Span is easier to use
@no_inline
fn b64encode_with_buffers(
    input_bytes: List[UInt8, _], inout result: List[UInt8, _]
):
    alias simd_width = sys.simdbytewidth()
    alias input_simd_width = simd_width * 3 // 4
    alias equal_vector = SIMD[DType.uint8, simd_width](ord("="))

    var input_bytes_len = len(input_bytes)

    var input_index = 0

    # Main loop
    while input_index + simd_width <= input_bytes_len:
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index

        var input_vector = start_of_input_chunk.load[width=simd_width]()

        result_vector = _to_b64_ascii(input_vector)

        (result.unsafe_ptr() + len(result)).store(result_vector)

        result.size += simd_width
        input_index += input_simd_width

    # We handle the last 0, 1 or 2 chunks
    while input_index < input_bytes_len:
        var start_of_input_chunk = input_bytes.unsafe_ptr() + input_index
        var nb_of_elements_to_load = min(
            input_simd_width, input_bytes_len - input_index
        )

        # We don't want to read past the input buffer
        var input_vector = load_incomplete_simd[simd_width](
            start_of_input_chunk,
            nb_of_elements_to_load=nb_of_elements_to_load,
        )

        result_vector = _to_b64_ascii(input_vector)

        # We place the '=' where needed
        var non_equal_chars_number = _get_number_of_bytes_to_store_from_number_of_bytes_to_load_without_equal_sign[
            simd_width
        ](
            nb_of_elements_to_load
        )
        var equal_mask = _base64_simd_mask[simd_width](non_equal_chars_number)

        var result_vector_with_equals = equal_mask.select(
            result_vector, equal_vector
        )

        var nb_of_elements_to_store = _get_number_of_bytes_to_store_from_number_of_bytes_to_load[
            simd_width
        ](
            nb_of_elements_to_load
        )
        store_incomplete_simd(
            result.unsafe_ptr() + len(result),
            result_vector_with_equals,
            nb_of_elements_to_store,
        )
        result.size += nb_of_elements_to_store
        input_index += input_simd_width
