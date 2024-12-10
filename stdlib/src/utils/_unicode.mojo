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
from bit import count_leading_zeros
from memory import UnsafePointer, memcpy

from ._unicode_lookups import *


fn _uppercase_mapping_index(rune: Int) -> Int:
    """Return index for upper case mapping or -1 if no mapping is given."""
    return _to_index[has_uppercase_mapping](rune)


fn _uppercase_mapping2_index(rune: Int) -> Int:
    """Return index for upper case mapping converting the rune to 2 runes, or -1 if no mapping is given.
    """
    return _to_index[has_uppercase_mapping2](rune)


fn _uppercase_mapping3_index(rune: Int) -> Int:
    """Return index for upper case mapping converting the rune to 3 runes, or -1 if no mapping is given.
    """
    return _to_index[has_uppercase_mapping3](rune)


fn _lowercase_mapping_index(rune: Int) -> Int:
    """Return index for lower case mapping or -1 if no mapping is given."""
    return _to_index[has_lowercase_mapping](rune)


@always_inline
fn _to_index[lookup: List[UInt32, **_]](rune: Int) -> Int:
    """Find index of rune in lookup with binary search.
    Returns -1 if not found."""
    var cursor = 0
    var x = UInt32(rune)
    var b = lookup.data
    var length = len(lookup)
    while length > 1:
        var half = length >> 1
        length -= half
        cursor += int(b.load(cursor + half - 1) < x) * half

    return cursor if b.load(cursor) == x else -1


fn is_uppercase(s: String) -> Bool:
    """Returns True if all characters in the string are uppercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are uppercaseand
        there is at least one cased character, False otherwise.
    """
    var found = False
    for c in s:
        var rune = ord(c)
        var index = _lowercase_mapping_index(rune)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping_index(rune)
        if index != -1:
            return False
        index = _uppercase_mapping2_index(rune)
        if index != -1:
            return False
        index = _uppercase_mapping3_index(rune)
        if index != -1:
            return False
    return found


fn is_lowercase(s: String) -> Bool:
    """Returns True if all characters in the string are lowercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are lowercase and
        there is at least one cased character, False otherwise.
    """
    var found = False
    for c in s:
        var rune = ord(c)
        var index = _uppercase_mapping_index(rune)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping2_index(rune)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping3_index(rune)
        if index != -1:
            found = True
            continue
        index = _lowercase_mapping_index(rune)
        if index != -1:
            return False
    return found


fn _ord(_p: UnsafePointer[UInt8]) -> (Int, Int):
    """Return the rune and number of bytes to be consumed, for given UTF-8 string pointer
    """
    var p = _p
    var b1 = p[]
    if (b1 >> 7) == 0:  # This is 1 byte ASCII char
        return int(b1), 1
    var num_bytes = count_leading_zeros(~b1)
    var shift = int((6 * (num_bytes - 1)))
    var b1_mask = 0b11111111 >> (num_bytes + 1)
    var result = int(b1 & b1_mask) << shift
    for _ in range(1, num_bytes):
        p += 1
        shift -= 6
        result |= int(p[] & 0b00111111) << shift
    return result, int(num_bytes)


fn _write_rune(rune: UInt32, p: UnsafePointer[UInt8]) -> Int:
    """Write rune as UTF-8 into provided pointer. Return number of added bytes.
    """
    if (rune >> 7) == 0:  # This is 1 byte ASCII char
        p[0] = rune.cast[DType.uint8]()
        return 1

    @always_inline
    fn _utf8_len(val: UInt32) -> Int:
        alias sizes = SIMD[DType.uint32, 4](
            0, 0b1111_111, 0b1111_1111_111, 0b1111_1111_1111_1111
        )
        var values = SIMD[DType.uint32, 4](val)
        var mask = values > sizes
        return int(mask.cast[DType.uint8]().reduce_add())

    var num_bytes = _utf8_len(rune)
    var shift = 6 * (num_bytes - 1)
    var mask = UInt32(0xFF) >> (num_bytes + 1)
    var num_bytes_marker = UInt32(0xFF) << (8 - num_bytes)
    p[0] = (((rune >> shift) & mask) | num_bytes_marker).cast[DType.uint8]()
    for i in range(1, num_bytes):
        shift -= 6
        p[i] = (((rune >> shift) & 0b00111111) | 0b10000000).cast[DType.uint8]()
    return num_bytes


fn to_lowercase(s: String) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to lowercase.
    """
    var input = s.unsafe_ptr()
    var capacity = (s.byte_length() >> 1) * 3 + 1
    var output = UnsafePointer[UInt8].alloc(capacity)
    var input_offset = 0
    var output_offset = 0
    while input_offset < s.byte_length():
        var rune_and_size = _ord(input + input_offset)
        var index = _lowercase_mapping_index(rune_and_size[0])
        if index == -1:
            memcpy(
                output + output_offset, input + input_offset, rune_and_size[1]
            )
            output_offset += rune_and_size[1]
        else:
            output_offset += _write_rune(
                lowercase_mapping[index], output + output_offset
            )

        input_offset += rune_and_size[1]

        if output_offset >= (
            capacity - 5
        ):  # check if we need to resize the ouput
            capacity += ((s.byte_length() - input_offset) >> 1) * 3 + 1
            var new_output = UnsafePointer[UInt8].alloc(capacity)
            memcpy(new_output, output, output_offset)
            output.free()
            output = new_output

    output[output_offset] = 0
    var list = List[UInt8](
        ptr=output, length=(output_offset + 1), capacity=capacity
    )
    return String(list)


fn to_uppercase(s: String) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    var input = s.unsafe_ptr()
    var capacity = (s.byte_length() >> 1) * 3 + 1
    var output = UnsafePointer[UInt8].alloc(capacity)
    var input_offset = 0
    var output_offset = 0
    while input_offset < s.byte_length():
        var rune_and_size = _ord(input + input_offset)
        var index = _uppercase_mapping_index(rune_and_size[0])
        var index2 = _uppercase_mapping2_index(
            rune_and_size[0]
        ) if index == -1 else -1
        var index3 = _uppercase_mapping3_index(
            rune_and_size[0]
        ) if index == -1 and index2 == -1 else -1
        if index != -1:
            output_offset += _write_rune(
                uppercase_mapping[index], output + output_offset
            )
        elif index2 != -1:
            var runes = uppercase_mapping2[index2]
            output_offset += _write_rune(runes[0], output + output_offset)
            output_offset += _write_rune(runes[1], output + output_offset)
        elif index3 != -1:
            var runes = uppercase_mapping3[index3]
            output_offset += _write_rune(runes[0], output + output_offset)
            output_offset += _write_rune(runes[1], output + output_offset)
            output_offset += _write_rune(runes[2], output + output_offset)
        else:
            memcpy(
                output + output_offset, input + input_offset, rune_and_size[1]
            )
            output_offset += rune_and_size[1]

        input_offset += rune_and_size[1]

        if output_offset >= (
            capacity - 5
        ):  # check if we need to resize the ouput
            capacity += ((s.byte_length() - input_offset) >> 1) * 3 + 1
            var new_output = UnsafePointer[UInt8].alloc(capacity)
            memcpy(new_output, output, output_offset)
            output.free()
            output = new_output

    output[output_offset] = 0
    var list = List[UInt8](
        ptr=output, length=(output_offset + 1), capacity=capacity
    )
    return String(list)
