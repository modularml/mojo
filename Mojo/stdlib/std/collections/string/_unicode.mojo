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

from std.collections.string._unicode_lookups import (
    has_uppercase_mapping,
    has_lowercase_mapping,
    uppercase_mapping,
    lowercase_mapping,
    has_uppercase_mapping2,
    uppercase_mapping2,
    has_uppercase_mapping3,
    uppercase_mapping3,
)

from std.builtin.globals import global_constant
from std.collections import Span
from std.sys.intrinsics import unlikely
from std.sys.info import bit_width_of

comptime BIGGEST_UNICODE_CODEPOINT = UInt32(0x10FFFF)


def _uppercase_mapping_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping or -1 if no mapping is given."""
    return _to_index(Span(global_constant[has_uppercase_mapping]()), rune)


def _uppercase_mapping2_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping converting the rune to 2 runes, or -1 if no mapping is given.
    """
    return _to_index(Span(global_constant[has_uppercase_mapping2]()), rune)


def _uppercase_mapping3_index(rune: Codepoint) -> Int:
    """Return index for upper case mapping converting the rune to 3 runes, or -1 if no mapping is given.
    """
    return _to_index(Span(global_constant[has_uppercase_mapping3]()), rune)


def _lowercase_mapping_index(rune: Codepoint) -> Int:
    """Return index for lower case mapping or -1 if no mapping is given."""
    return _to_index(Span(global_constant[has_lowercase_mapping]()), rune)


def _to_index(lookup: Span[UInt32, ImmStaticOrigin], rune: Codepoint) -> Int:
    """Find index of rune in lookup with binary search.
    Returns -1 if not found."""

    var result = lookup._binary_search_index(rune.to_u32())

    if result:
        return Int(result.unsafe_value())
    else:
        return -1


# TODO:
#   Refactor this to return a Span[Codepoint, ImmStaticOrigin], so that the
#   return `Int` count and fixed-size `Array` are not necessary.
def _get_uppercase_mapping(
    char: Codepoint,
) -> Optional[Tuple[Int, Array[Codepoint, 3]]]:
    """Returns the 1, 2, or 3 character sequence that is the uppercase form of
    `char`.

    Returns None if `char` does not have an uppercase equivalent.
    """
    var array = Array[Codepoint, 3](fill=Codepoint(0))

    var index1 = _uppercase_mapping_index(char)
    if index1 != -1:
        var rune = global_constant[uppercase_mapping]()[index1]
        array[0] = Codepoint(unsafe_unchecked_codepoint=rune)
        return Tuple(Int(1), array^)

    var index2 = _uppercase_mapping2_index(char)
    if index2 != -1:
        var runes = global_constant[uppercase_mapping2]()[index2]
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        return Tuple(Int(2), array^)

    var index3 = _uppercase_mapping3_index(char)
    if index3 != -1:
        var runes = global_constant[uppercase_mapping3]()[index3]
        array[0] = Codepoint(unsafe_unchecked_codepoint=runes[0])
        array[1] = Codepoint(unsafe_unchecked_codepoint=runes[1])
        array[2] = Codepoint(unsafe_unchecked_codepoint=runes[2])
        return Tuple(Int(3), array^)

    return None


def _get_lowercase_mapping(char: Codepoint) -> Optional[Codepoint]:
    var index: Optional[Int] = Span(
        global_constant[has_lowercase_mapping]()
    )._binary_search_index(char.to_u32())

    if index:
        # SAFETY: We just checked that `result` is present.
        var codepoint = global_constant[lowercase_mapping]()[
            index.unsafe_value()
        ]

        # SAFETY:
        #   We know this is a valid `Codepoint` because the mapping data tables
        #   contain only valid codepoints.
        return Codepoint(unsafe_unchecked_codepoint=codepoint)
    else:
        return None


def is_uppercase(s: StringSlice[mut=False, _]) -> Bool:
    """Returns True if all characters in the string are uppercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are uppercaseand
        there is at least one cased character, False otherwise.
    """
    var found = False
    for char in s.codepoints():
        var index = _lowercase_mapping_index(char)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping_index(char)
        if index != -1:
            return False
        index = _uppercase_mapping2_index(char)
        if index != -1:
            return False
        index = _uppercase_mapping3_index(char)
        if index != -1:
            return False
    return found


def is_lowercase(s: StringSlice[mut=False, _]) -> Bool:
    """Returns True if all characters in the string are lowercase, and
        there is at least one cased character.

    Args:
        s: The string to examine.

    Returns:
        True if all characters in the string are lowercase and
        there is at least one cased character, False otherwise.
    """
    var found = False
    for char in s.codepoints():
        var index = _uppercase_mapping_index(char)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping2_index(char)
        if index != -1:
            found = True
            continue
        index = _uppercase_mapping3_index(char)
        if index != -1:
            found = True
            continue
        index = _lowercase_mapping_index(char)
        if index != -1:
            return False
    return found


def to_lowercase(s: StringSlice[mut=False, _]) -> String:
    """Returns a new string with all characters converted to lowercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to lowercase.
    """
    var input = s.as_bytes()
    var result = String(capacity_bytes=_estimate_needed_size(len(input)))
    var input_offset = 0
    while input_offset < len(input):
        var char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            input[input_offset:]
        )
        result.append(_get_lowercase_mapping(char).or_else(char))
        input_offset += num_bytes

    return result^


def to_uppercase(s: StringSlice[mut=False, _]) -> String:
    """Returns a new string with all characters converted to uppercase.

    Args:
        s: Input string.

    Returns:
        A new string where cased letters have been converted to uppercase.
    """
    var input = s.as_bytes()
    var result = String(capacity_bytes=_estimate_needed_size(len(input)))
    var input_offset = 0
    while input_offset < len(input):
        var char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            input[input_offset:]
        )
        var uppercase_replacement_opt = _get_uppercase_mapping(char)

        if uppercase_replacement_opt:
            var count: Int
            # A given character can be replaced with a sequence of characters
            # up to 3 characters in length. A fixed size `Codepoint` array is
            # returned, along with a `count` (1, 2, or 3) of how many
            # replacement characters are in the uppercase replacement sequence.
            count, ref uppercase_replacement_chars = (
                uppercase_replacement_opt.unsafe_value()
            )
            for char_idx in range(count):
                result.append(uppercase_replacement_chars[char_idx])
        else:
            result.append(char)

        input_offset += num_bytes

    return result^


@always_inline
def _estimate_needed_size(byte_len: Int) -> Int:
    return 3 * (byte_len >> 1) + 1


@always_inline
def _is_unicode_scalar_value(codepoint: Scalar) -> Bool:
    """Returns True if `codepoint` is a valid Unicode scalar value.

    Args:
        codepoint: The codepoint integer value to check.

    Returns:
        True if `codepoint` is a valid Unicode scalar value; False otherwise.
    """

    comptime if bit_width_of[codepoint.dtype]() <= 8:
        return True
    comptime S = type_of(codepoint)
    var is_not_unpaired_surrogate = not (S(0xD800) <= codepoint <= S(0xDFFF))

    comptime if bit_width_of[codepoint.dtype]() == 16:
        return is_not_unpaired_surrogate

    return is_not_unpaired_surrogate and (
        codepoint <= S(BIGGEST_UNICODE_CODEPOINT)
    )


def _decode_codepoints[
    dtype: DType,
    //,
    strict: Bool = True,
    replace: Codepoint = Codepoint.ord("�"),
](*, from_codepoints: Span[mut=False, Scalar[dtype], ...]) raises -> String:
    comptime assert dtype.is_integral(), "The dtype must be integral"
    comptime replace_str = String(replace)
    comptime replace_len = UInt(replace_str.byte_length())
    # TODO: this could use reduce_sum with a function that calculates the utf8
    # length from the utf32 codepoints. Performance needs to be measured
    var length = 4 * UInt(len(from_codepoints))
    var result = String(capacity_bytes=Int(length))
    var codepoints_ptr = from_codepoints.unsafe_ptr()
    var c_idx = UInt(0)
    var utf8_offset = UInt(0)

    while c_idx < UInt(len(from_codepoints)):
        var utf8_ptr = result.unsafe_ptr_mut().unsafe_offset(Int(utf8_offset))
        var c_og = codepoints_ptr[unsafe_offset=c_idx]

        comptime if bit_width_of[dtype]() >= 16:
            if unlikely(not _is_unicode_scalar_value(c_og)):
                comptime if strict:
                    raise Error("Invalid unicode codepoint at index: ", c_idx)
                else:
                    result += replace_str
                    utf8_offset += replace_len
                    c_idx += 1
                    continue

        var c = c_og.cast[DType.uint32]()
        var codepoint = Codepoint(unsafe_unchecked_codepoint=c)
        var num_bytes = codepoint.unsafe_write_utf8[
            optimize_ascii=True, branchless=False
        ](utf8_ptr)

        utf8_offset += UInt(num_bytes)
        c_idx += 1

    result._set_byte_length(Int(utf8_offset))
    return result^
