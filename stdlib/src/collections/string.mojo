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
"""Implements basic object methods for working with strings.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import KeyElement, List, Optional
from collections._index_normalization import normalize_index
from hashlib._hasher import _HashableWithHasher, _Hasher
from sys import bitwidthof, llvm_intrinsic
from sys.ffi import c_char
from sys.intrinsics import _type_is_eq

from bit import count_leading_zeros
from memory import UnsafePointer, memcmp, memcpy, Span
from python import PythonObject

from utils import (
    IndexList,
    StaticString,
    StringRef,
    StringSlice,
    Variant,
    Writable,
    Writer,
    write_args,
)
from utils._unicode import (
    is_lowercase,
    is_uppercase,
    to_lowercase,
    to_uppercase,
)
from utils.format import _CurlyEntryFormattable, _FormatCurlyEntry
from utils.string_slice import (
    _shift_unicode_to_utf8,
    _StringSliceIter,
    _to_string_list,
    _unicode_codepoint_utf8_byte_length,
    _utf8_byte_type,
)

# ===----------------------------------------------------------------------=== #
# ord
# ===----------------------------------------------------------------------=== #


fn ord(s: String) -> Int:
    """Returns an integer that represents the given one-character string.

    Given a string representing one character, return an integer
    representing the code point of that character. For example, `ord("a")`
    returns the integer `97`. This is the inverse of the `chr()` function.

    Args:
        s: The input string slice, which must contain only a single character.

    Returns:
        An integer representing the code point of the given character.
    """
    return ord(s.as_string_slice())


fn ord(s: StringSlice) -> Int:
    """Returns an integer that represents the given one-character string.

    Given a string representing one character, return an integer
    representing the code point of that character. For example, `ord("a")`
    returns the integer `97`. This is the inverse of the `chr()` function.

    Args:
        s: The input string, which must contain only a single character.

    Returns:
        An integer representing the code point of the given character.
    """
    # UTF-8 to Unicode conversion:              (represented as UInt32 BE)
    # 1: 0aaaaaaa                            -> 00000000 00000000 00000000 0aaaaaaa     a
    # 2: 110aaaaa 10bbbbbb                   -> 00000000 00000000 00000aaa aabbbbbb     a << 6  | b
    # 3: 1110aaaa 10bbbbbb 10cccccc          -> 00000000 00000000 aaaabbbb bbcccccc     a << 12 | b << 6  | c
    # 4: 11110aaa 10bbbbbb 10cccccc 10dddddd -> 00000000 000aaabb bbbbcccc ccdddddd     a << 18 | b << 12 | c << 6 | d
    var p = s.unsafe_ptr()
    var b1 = p[]
    if (b1 >> 7) == 0:  # This is 1 byte ASCII char
        debug_assert(s.byte_length() == 1, "input string length must be 1")
        return int(b1)
    var num_bytes = count_leading_zeros(~b1)
    debug_assert(
        s.byte_length() == int(num_bytes), "input string must be one character"
    )
    debug_assert(
        1 < int(num_bytes) < 5, "invalid UTF-8 byte ", b1, " at index 0"
    )
    var shift = int((6 * (num_bytes - 1)))
    var b1_mask = 0b11111111 >> (num_bytes + 1)
    var result = int(b1 & b1_mask) << shift
    for i in range(1, num_bytes):
        p += 1
        debug_assert(
            p[] >> 6 == 0b00000010, "invalid UTF-8 byte ", b1, " at index ", i
        )
        shift -= 6
        result |= int(p[] & 0b00111111) << shift
    return result


# ===----------------------------------------------------------------------=== #
# chr
# ===----------------------------------------------------------------------=== #


fn chr(c: Int) -> String:
    """Returns a String based on the given Unicode code point. This is the
    inverse of the `ord()` function.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.

    Examples:
    ```mojo
    print(chr(97)) # "a"
    print(chr(8364)) # "€"
    ```
    .
    """

    if c < 0b1000_0000:  # 1 byte ASCII char
        return String(String._buffer_type(c, 0))

    var num_bytes = _unicode_codepoint_utf8_byte_length(c)
    var p = UnsafePointer[UInt8].alloc(num_bytes + 1)
    _shift_unicode_to_utf8(p, c, num_bytes)
    # TODO: decide whether to use replacement char (�) or raise ValueError
    # if not _is_valid_utf8(p, num_bytes):
    #     debug_assert(False, "Invalid Unicode code point")
    #     p.free()
    #     return chr(0xFFFD)
    p[num_bytes] = 0
    return String(ptr=p, length=num_bytes + 1)


# ===----------------------------------------------------------------------=== #
# ascii
# ===----------------------------------------------------------------------=== #


fn _chr_ascii(c: UInt8) -> String:
    """Returns a string based on the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.
    """
    return String(String._buffer_type(c, 0))


fn _repr_ascii(c: UInt8) -> String:
    """Returns a printable representation of the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a representation of the given code point.
    """
    alias ord_tab = ord("\t")
    alias ord_new_line = ord("\n")
    alias ord_carriage_return = ord("\r")
    alias ord_back_slash = ord("\\")

    if c == ord_back_slash:
        return r"\\"
    elif isprintable(c):
        return _chr_ascii(c)
    elif c == ord_tab:
        return r"\t"
    elif c == ord_new_line:
        return r"\n"
    elif c == ord_carriage_return:
        return r"\r"
    else:
        var uc = c.cast[DType.uint8]()
        if uc < 16:
            return hex(uc, prefix=r"\x0")
        else:
            return hex(uc, prefix=r"\x")


@always_inline
fn ascii(value: String) -> String:
    """Get the ASCII representation of the object.

    Args:
        value: The object to get the ASCII representation of.

    Returns:
        A string containing the ASCII representation of the object.
    """
    alias ord_squote = ord("'")
    var result = String()
    var use_dquote = False

    for idx in range(len(value._buffer) - 1):
        var char = value._buffer[idx]
        result += _repr_ascii(char)
        use_dquote = use_dquote or (char == ord_squote)

    if use_dquote:
        return '"' + result + '"'
    else:
        return "'" + result + "'"


# ===----------------------------------------------------------------------=== #
# strtol
# ===----------------------------------------------------------------------=== #


fn _atol(str_slice: StringSlice, base: Int = 10) raises -> Int:
    """Implementation of `atol` for StringSlice inputs.

    Please see its docstring for details.
    """
    if (base != 0) and (base < 2 or base > 36):
        raise Error("Base must be >= 2 and <= 36, or 0.")
    if not str_slice:
        raise Error(_str_to_base_error(base, str_slice))

    var real_base: Int
    var ord_num_max: Int

    var ord_letter_max = (-1, -1)
    var result = 0
    var is_negative: Bool = False
    var has_prefix: Bool = False
    var start: Int = 0
    var str_len = str_slice.byte_length()

    start, is_negative = _trim_and_handle_sign(str_slice, str_len)

    alias ord_0 = ord("0")
    alias ord_letter_min = (ord("a"), ord("A"))
    alias ord_underscore = ord("_")

    if base == 0:
        var real_base_new_start = _identify_base(str_slice, start)
        real_base = real_base_new_start[0]
        start = real_base_new_start[1]
        has_prefix = real_base != 10
        if real_base == -1:
            raise Error(_str_to_base_error(base, str_slice))
    else:
        start, has_prefix = _handle_base_prefix(start, str_slice, str_len, base)
        real_base = base

    if real_base <= 10:
        ord_num_max = ord(str(real_base - 1))
    else:
        ord_num_max = ord("9")
        ord_letter_max = (
            ord("a") + (real_base - 11),
            ord("A") + (real_base - 11),
        )

    var buff = str_slice.unsafe_ptr()
    var found_valid_chars_after_start = False
    var has_space_after_number = False

    # Prefixed integer literals with real_base 2, 8, 16 may begin with leading
    # underscores under the conditions they have a prefix
    var was_last_digit_underscore = not (real_base in (2, 8, 16) and has_prefix)
    for pos in range(start, str_len):
        var ord_current = int(buff[pos])
        if ord_current == ord_underscore:
            if was_last_digit_underscore:
                raise Error(_str_to_base_error(base, str_slice))
            else:
                was_last_digit_underscore = True
                continue
        else:
            was_last_digit_underscore = False
        if ord_0 <= ord_current <= ord_num_max:
            result += ord_current - ord_0
            found_valid_chars_after_start = True
        elif ord_letter_min[0] <= ord_current <= ord_letter_max[0]:
            result += ord_current - ord_letter_min[0] + 10
            found_valid_chars_after_start = True
        elif ord_letter_min[1] <= ord_current <= ord_letter_max[1]:
            result += ord_current - ord_letter_min[1] + 10
            found_valid_chars_after_start = True
        elif _isspace(ord_current):
            has_space_after_number = True
            start = pos + 1
            break
        else:
            raise Error(_str_to_base_error(base, str_slice))
        if pos + 1 < str_len and not _isspace(buff[pos + 1]):
            var nextresult = result * real_base
            if nextresult < result:
                raise Error(
                    _str_to_base_error(base, str_slice)
                    + " String expresses an integer too large to store in Int."
                )
            result = nextresult

    if was_last_digit_underscore or (not found_valid_chars_after_start):
        raise Error(_str_to_base_error(base, str_slice))

    if has_space_after_number:
        for pos in range(start, str_len):
            if not _isspace(buff[pos]):
                raise Error(_str_to_base_error(base, str_slice))
    if is_negative:
        result = -result
    return result


@always_inline
fn _trim_and_handle_sign(str_slice: StringSlice, str_len: Int) -> (Int, Bool):
    """Trims leading whitespace, handles the sign of the number in the string.

    Args:
        str_slice: A StringSlice containing the number to parse.
        str_len: The length of the string.

    Returns:
        A tuple containing:
        - The starting index of the number after whitespace and sign.
        - A boolean indicating whether the number is negative.
    """
    var buff = str_slice.unsafe_ptr()
    var start: Int = 0
    while start < str_len and _isspace(buff[start]):
        start += 1
    var p: Bool = buff[start] == ord("+")
    var n: Bool = buff[start] == ord("-")
    return start + (p or n), n


@always_inline
fn _handle_base_prefix(
    pos: Int, str_slice: StringSlice, str_len: Int, base: Int
) -> (Int, Bool):
    """Adjusts the starting position if a valid base prefix is present.

    Handles "0b"/"0B" for base 2, "0o"/"0O" for base 8, and "0x"/"0X" for base
    16. Only adjusts if the base matches the prefix.

    Args:
        pos: Current position in the string.
        str_slice: The input StringSlice.
        str_len: Length of the input string.
        base: The specified base.

    Returns:
        A tuple containing:
            - Updated position after the prefix, if applicable.
            - A boolean indicating if the prefix was valid for the given base.
    """
    var start = pos
    var buff = str_slice.unsafe_ptr()
    if start + 1 < str_len:
        var prefix_char = chr(int(buff[start + 1]))
        if buff[start] == ord("0") and (
            (base == 2 and (prefix_char == "b" or prefix_char == "B"))
            or (base == 8 and (prefix_char == "o" or prefix_char == "O"))
            or (base == 16 and (prefix_char == "x" or prefix_char == "X"))
        ):
            start += 2
    return start, start != pos


fn _str_to_base_error(base: Int, str_slice: StringSlice) -> String:
    return (
        "String is not convertible to integer with base "
        + str(base)
        + ": '"
        + str(str_slice)
        + "'"
    )


fn _identify_base(str_slice: StringSlice[_], start: Int) -> Tuple[Int, Int]:
    var length = str_slice.byte_length()
    # just 1 digit, assume base 10
    if start == (length - 1):
        return 10, start
    if str_slice[start] == "0":
        var second_digit = str_slice[start + 1]
        if second_digit == "b" or second_digit == "B":
            return 2, start + 2
        if second_digit == "o" or second_digit == "O":
            return 8, start + 2
        if second_digit == "x" or second_digit == "X":
            return 16, start + 2
        # checking for special case of all "0", "_" are also allowed
        var was_last_character_underscore = False
        for i in range(start + 1, length):
            if str_slice[i] == "_":
                if was_last_character_underscore:
                    return -1, -1
                else:
                    was_last_character_underscore = True
                    continue
            else:
                was_last_character_underscore = False
            if str_slice[i] != "0":
                return -1, -1
    elif ord("1") <= ord(str_slice[start]) <= ord("9"):
        return 10, start
    else:
        return -1, -1

    return 10, start


fn atol(str: String, base: Int = 10) raises -> Int:
    """Parses and returns the given string as an integer in the given base.

    If base is set to 0, the string is parsed as an Integer literal, with the
    following considerations:
    - '0b' or '0B' prefix indicates binary (base 2)
    - '0o' or '0O' prefix indicates octal (base 8)
    - '0x' or '0X' prefix indicates hexadecimal (base 16)
    - Without a prefix, it's treated as decimal (base 10)

    Args:
        str: A string to be parsed as an integer in the given base.
        base: Base used for conversion, value must be between 2 and 36, or 0.

    Returns:
        An integer value that represents the string.

    Raises:
        If the given string cannot be parsed as an integer value or if an
        incorrect base is provided.

    Examples:
        >>> atol("32")
        32
        >>> atol("FF", 16)
        255
        >>> atol("0xFF", 0)
        255
        >>> atol("0b1010", 0)
        10

    Notes:
        This follows [Python's integer literals](
        https://docs.python.org/3/reference/lexical_analysis.html#integers).
    """
    return _atol(str.as_string_slice(), base)


fn _atof_error(str_ref: StringSlice[_]) -> Error:
    return Error("String is not convertible to float: '" + str(str_ref) + "'")


fn _atof(str_ref: StringSlice[_]) raises -> Float64:
    """Implementation of `atof` for StringRef inputs.

    Please see its docstring for details.
    """
    if not str_ref:
        raise _atof_error(str_ref)

    var result: Float64 = 0.0
    var exponent: Int = 0
    var sign: Int = 1

    alias ord_0 = UInt8(ord("0"))
    alias ord_9 = UInt8(ord("9"))
    alias ord_dot = UInt8(ord("."))
    alias ord_plus = UInt8(ord("+"))
    alias ord_minus = UInt8(ord("-"))
    alias ord_f = UInt8(ord("f"))
    alias ord_F = UInt8(ord("F"))
    alias ord_e = UInt8(ord("e"))
    alias ord_E = UInt8(ord("E"))

    var start: Int = 0
    var str_ref_strip = str_ref.strip()
    var str_len = len(str_ref_strip)
    var buff = str_ref_strip.unsafe_ptr()

    # check sign, inf, nan
    if buff[start] == ord_plus:
        start += 1
    elif buff[start] == ord_minus:
        start += 1
        sign = -1
    if (str_len - start) >= 3:
        if StringRef(buff + start, 3) == "nan":
            return FloatLiteral.nan
        if StringRef(buff + start, 3) == "inf":
            return FloatLiteral.infinity * sign
    # read before dot
    for pos in range(start, str_len):
        if ord_0 <= buff[pos] <= ord_9:
            result = result * 10.0 + int(buff[pos] - ord_0)
            start += 1
        else:
            break
    # if dot -> read after dot
    if buff[start] == ord_dot:
        start += 1
        for pos in range(start, str_len):
            if ord_0 <= buff[pos] <= ord_9:
                result = result * 10.0 + int(buff[pos] - ord_0)
                exponent -= 1
            else:
                break
            start += 1
    # if e/E -> read scientific notation
    if buff[start] == ord_e or buff[start] == ord_E:
        start += 1
        var sign: Int = 1
        var shift: Int = 0
        var has_number: Bool = False
        for pos in range(start, str_len):
            if buff[start] == ord_plus:
                pass
            elif buff[pos] == ord_minus:
                sign = -1
            elif ord_0 <= buff[start] <= ord_9:
                has_number = True
                shift = shift * 10 + int(buff[pos] - ord_0)
            else:
                break
            start += 1
        exponent += sign * shift
        if not has_number:
            raise _atof_error(str_ref)
    # check for f/F at the end
    if buff[start] == ord_f or buff[start] == ord_F:
        start += 1
    # check if string got fully parsed
    if start != str_len:
        raise _atof_error(str_ref)
    # apply shift
    # NOTE: Instead of `var result *= 10.0 ** exponent`, we calculate a positive
    # integer factor as shift and multiply or divide by it based on the shift
    # direction. This allows for better precision.
    # TODO: investigate if there is a floating point arithmetic problem.
    var shift: Int = 10 ** abs(exponent)
    if exponent > 0:
        result *= shift
    if exponent < 0:
        result /= shift
    # apply sign
    return result * sign


fn atof(str: String) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    Raises:
        If the given string cannot be parsed as an floating point value, for
        example in `atof("hi")`.

    Args:
        str: A string to be parsed as a floating point.

    Returns:
        An floating point value that represents the string, or otherwise raises.
    """
    return _atof(str.as_string_slice())


# ===----------------------------------------------------------------------=== #
# isdigit
# ===----------------------------------------------------------------------=== #


fn isdigit(c: UInt8) -> Bool:
    """Determines whether the given character is a digit [0-9].

    Args:
        c: The character to check.

    Returns:
        True if the character is a digit.
    """
    alias ord_0 = ord("0")
    alias ord_9 = ord("9")
    return ord_0 <= int(c) <= ord_9


# ===----------------------------------------------------------------------=== #
# isupper
# ===----------------------------------------------------------------------=== #


fn isupper(c: UInt8) -> Bool:
    """Determines whether the given character is an uppercase character.

    This currently only respects the default "C" locale, i.e. returns True iff
    the character specified is one of "ABCDEFGHIJKLMNOPQRSTUVWXYZ".

    Args:
        c: The character to check.

    Returns:
        True if the character is uppercase.
    """
    return _is_ascii_uppercase(c)


fn _is_ascii_uppercase(c: UInt8) -> Bool:
    alias ord_a = ord("A")
    alias ord_z = ord("Z")
    return ord_a <= int(c) <= ord_z


# ===----------------------------------------------------------------------=== #
# islower
# ===----------------------------------------------------------------------=== #


fn islower(c: UInt8) -> Bool:
    """Determines whether the given character is an lowercase character.

    This currently only respects the default "C" locale, i.e. returns True iff
    the character specified is one of "abcdefghijklmnopqrstuvwxyz".

    Args:
        c: The character to check.

    Returns:
        True if the character is lowercase.
    """
    return _is_ascii_lowercase(c)


fn _is_ascii_lowercase(c: UInt8) -> Bool:
    alias ord_a = ord("a")
    alias ord_z = ord("z")
    return ord_a <= int(c) <= ord_z


# ===----------------------------------------------------------------------=== #
# _isspace
# ===----------------------------------------------------------------------=== #


fn _isspace(c: String) -> Bool:
    """Determines whether the given character is a whitespace character.

    This only respects the default "C" locale, i.e. returns True only if the
    character specified is one of " \\t\\n\\v\\f\\r". For semantics similar
    to Python, use `String.isspace()`.

    Args:
        c: The character to check.

    Returns:
        True iff the character is one of the whitespace characters listed above.
    """
    return _isspace(ord(c))


fn _isspace(c: UInt8) -> Bool:
    """Determines whether the given character is a whitespace character.

    This only respects the default "C" locale, i.e. returns True only if the
    character specified is one of " \\t\\n\\v\\f\\r". For semantics similar
    to Python, use `String.isspace()`.

    Args:
        c: The character to check.

    Returns:
        True iff the character is one of the whitespace characters listed above.
    """

    # NOTE: a global LUT doesn't work at compile time so we can't use it here.
    alias ` ` = UInt8(ord(" "))
    alias `\t` = UInt8(ord("\t"))
    alias `\n` = UInt8(ord("\n"))
    alias `\r` = UInt8(ord("\r"))
    alias `\f` = UInt8(ord("\f"))
    alias `\v` = UInt8(ord("\v"))
    alias `\x1c` = UInt8(ord("\x1c"))
    alias `\x1d` = UInt8(ord("\x1d"))
    alias `\x1e` = UInt8(ord("\x1e"))

    # This compiles to something very clever that's even faster than a LUT.
    return (
        c == ` `
        or c == `\t`
        or c == `\n`
        or c == `\r`
        or c == `\f`
        or c == `\v`
        or c == `\x1c`
        or c == `\x1d`
        or c == `\x1e`
    )


# ===----------------------------------------------------------------------=== #
# isprintable
# ===----------------------------------------------------------------------=== #


fn isprintable(c: UInt8) -> Bool:
    """Determines whether the given character is a printable character.

    Args:
        c: The character to check.

    Returns:
        True if the character is a printable character, otherwise False.
    """
    alias ord_space = ord(" ")
    alias ord_tilde = ord("~")
    return ord_space <= int(c) <= ord_tilde


# ===----------------------------------------------------------------------=== #
# String
# ===----------------------------------------------------------------------=== #


@value
struct String(
    Sized,
    Stringable,
    AsBytes,
    Representable,
    IntableRaising,
    KeyElement,
    Comparable,
    Boolable,
    Writable,
    Writer,
    CollectionElementNew,
    FloatableRaising,
    _HashableWithHasher,
):
    """Represents a mutable string."""

    # Fields
    alias _buffer_type = List[UInt8, hint_trivial_type=True]
    var _buffer: Self._buffer_type
    """The underlying storage for the string."""

    """ Useful string aliases. """
    alias ASCII_LOWERCASE = String("abcdefghijklmnopqrstuvwxyz")
    alias ASCII_UPPERCASE = String("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    alias ASCII_LETTERS = String.ASCII_LOWERCASE + String.ASCII_UPPERCASE
    alias DIGITS = String("0123456789")
    alias HEX_DIGITS = String.DIGITS + String("abcdef") + String("ABCDEF")
    alias OCT_DIGITS = String("01234567")
    alias PUNCTUATION = String("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~""")
    alias PRINTABLE = (
        String.DIGITS
        + String.ASCII_LETTERS
        + String.PUNCTUATION
        + " \t\n\r\v\f"  # single byte utf8 whitespaces
    )

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @always_inline
    @implicit
    fn __init__(out self, owned impl: List[UInt8, *_]):
        """Construct a string from a buffer of bytes without copying the
        allocated data.

        The buffer must be terminated with a null byte:

        ```mojo
        var buf = List[UInt8]()
        buf.append(ord('H'))
        buf.append(ord('i'))
        buf.append(0)
        var hi = String(buf)
        ```

        Args:
            impl: The buffer.
        """
        debug_assert(
            len(impl) > 0 and impl[-1] == 0,
            "expected last element of String buffer to be null terminator",
        )
        # We make a backup because steal_data() will clear size and capacity.
        var size = impl.size
        debug_assert(
            impl[size - 1] == 0,
            "expected last element of String buffer to be null terminator",
        )
        var capacity = impl.capacity
        self._buffer = Self._buffer_type(
            ptr=impl.steal_data(), length=size, capacity=capacity
        )

    @always_inline
    @implicit
    fn __init__(out self, impl: Self._buffer_type):
        """Construct a string from a buffer of bytes, copying the allocated
        data. Use the transfer operator ^ to avoid the copy.

        The buffer must be terminated with a null byte:

        ```mojo
        var buf = List[UInt8]()
        buf.append(ord('H'))
        buf.append(ord('i'))
        buf.append(0)
        var hi = String(buf)
        ```

        Args:
            impl: The buffer.
        """
        debug_assert(
            len(impl) > 0 and impl[-1] == 0,
            "expected last element of String buffer to be null terminator",
        )
        # We make a backup because steal_data() will clear size and capacity.
        var size = impl.size
        debug_assert(
            impl[size - 1] == 0,
            "expected last element of String buffer to be null terminator",
        )
        self._buffer = impl

    @always_inline
    fn __init__(out self):
        """Construct an uninitialized string."""
        self._buffer = Self._buffer_type()

    @always_inline
    fn __init__(out self, *, capacity: Int):
        """Construct an uninitialized string with the given capacity.

        Args:
            capacity: The capacity of the string.
        """
        self._buffer = Self._buffer_type(capacity=capacity)

    fn __init__(out self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self = other  # Just use the implicit copyinit.

    @implicit
    fn __init__(out self, str: StringRef):
        """Construct a string from a StringRef object.

        Args:
            str: The StringRef from which to construct this string object.
        """
        var length = len(str)
        var buffer = Self._buffer_type()
        # +1 for null terminator, initialized to 0
        buffer.resize(length + 1, 0)
        memcpy(dest=buffer.data, src=str.data, count=length)
        self = Self(buffer^)

    @implicit
    fn __init__(out self, str_slice: StringSlice):
        """Construct a string from a string slice.

        This will allocate a new string that copies the string contents from
        the provided string slice `str_slice`.

        Args:
            str_slice: The string slice from which to construct this string.
        """

        # Calculate length in bytes
        var length: Int = len(str_slice.as_bytes())
        var buffer = Self._buffer_type()
        # +1 for null terminator, initialized to 0
        buffer.resize(length + 1, 0)
        memcpy(
            dest=buffer.data,
            src=str_slice.as_bytes().unsafe_ptr(),
            count=length,
        )
        self = Self(buffer^)

    @always_inline
    @implicit
    fn __init__(out self, literal: StringLiteral):
        """Constructs a String value given a constant string.

        Args:
            literal: The input constant string.
        """
        self = literal.__str__()

    @always_inline
    fn __init__(out self, *, ptr: UnsafePointer[Byte], length: Int):
        """Creates a string from the buffer. Note that the string now owns
        the buffer.

        The buffer must be terminated with a null byte.

        Args:
            ptr: The pointer to the buffer.
            length: The length of the buffer, including the null terminator.
        """
        # we don't know the capacity of ptr, but we'll assume it's the same or
        # larger than len
        self = Self(Self._buffer_type(ptr=ptr, length=length, capacity=length))

    # ===------------------------------------------------------------------=== #
    # Factory dunders
    # ===------------------------------------------------------------------=== #

    fn write_bytes(mut self, bytes: Span[Byte, _]):
        """Write a byte span to this String.

        Args:
            bytes: The byte span to write to this String. Must NOT be
                null terminated.
        """
        self._iadd[False](bytes)

    fn write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """

        @parameter
        fn write_arg[T: Writable](arg: T):
            arg.write_to(self)

        args.each[write_arg]()

    @staticmethod
    @no_inline
    fn write[
        *Ts: Writable
    ](*args: *Ts, sep: StaticString = "", end: StaticString = "") -> Self:
        """
        Construct a string by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Returns:
            A string formed by formatting the argument sequence.

        Examples:

        Construct a String from several `Writable` arguments:

        ```mojo
        var string = String.write(1, ", ", 2.0, ", ", "three")
        print(string) # "1, 2.0, three"
        %# from testing import assert_equal
        %# assert_equal(string, "1, 2.0, three")
        ```
        .
        """
        var output = String()
        write_args(output, args, sep=sep, end=end)
        return output^

    @staticmethod
    @no_inline
    fn write[
        *Ts: Writable
    ](
        args: VariadicPack[_, Writable, *Ts],
        sep: StaticString = "",
        end: StaticString = "",
    ) -> Self:
        """
        Construct a string by passing a variadic pack.

        Args:
            args: A VariadicPack of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
                `Writable`.

        Returns:
            A string formed by formatting the VariadicPack.

        Examples:

        ```mojo
        fn variadic_pack_to_string[
            *Ts: Writable,
        ](*args: *Ts) -> String:
            return String.write(args)

        string = variadic_pack_to_string(1, ", ", 2.0, ", ", "three")
        %# from testing import assert_equal
        %# assert_equal(string, "1, 2.0, three")
        ```
        .
        """
        var output = String()
        write_args(output, args, sep=sep, end=end)
        return output^

    @staticmethod
    @always_inline
    fn _from_bytes(owned buff: UnsafePointer[UInt8]) -> String:
        """Construct a string from a sequence of bytes.

        This does no validation that the given bytes are valid in any specific
        String encoding.

        Args:
            buff: The buffer. This should have an existing terminator.
        """

        return String(ptr=buff, length=len(StringRef(ptr=buff)) + 1)

    @staticmethod
    fn _from_bytes(owned buff: Self._buffer_type) -> String:
        """Construct a string from a sequence of bytes.

        This does no validation that the given bytes are valid in any specific
        String encoding.

        Args:
            buff: The buffer.
        """

        # If a terminator does not already exist, then add it.
        if buff[-1]:
            buff.append(0)

        return String(buff^)

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    fn __getitem__[IndexerType: Indexer](self, idx: IndexerType) -> String:
        """Gets the character at the specified position.

        Parameters:
            IndexerType: The inferred type of an indexer argument.

        Args:
            idx: The index value.

        Returns:
            A new string containing the character at the specified position.
        """
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time
        var normalized_idx = normalize_index["String"](idx, self)
        var buf = Self._buffer_type(capacity=1)
        buf.append(self._buffer[normalized_idx])
        buf.append(0)
        return String(buf^)

    fn __getitem__(self, span: Slice) -> String:
        """Gets the sequence of characters at the specified positions.

        Args:
            span: A slice that specifies positions of the new substring.

        Returns:
            A new string containing the string at the specified positions.
        """
        var start: Int
        var end: Int
        var step: Int
        # TODO(#933): implement this for unicode when we support llvm intrinsic evaluation at compile time

        start, end, step = span.indices(self.byte_length())
        var r = range(start, end, step)
        if step == 1:
            return StringRef(self._buffer.data + start, len(r))

        var buffer = Self._buffer_type()
        var result_len = len(r)
        buffer.resize(result_len + 1, 0)
        var ptr = self.unsafe_ptr()
        for i in range(result_len):
            buffer[i] = ptr[r[i]]
        buffer[result_len] = 0
        return Self(buffer^)

    @always_inline
    fn __eq__(self, other: String) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        if not self and not other:
            return True
        if len(self) != len(other):
            return False
        # same pointer and length, so equal
        if self.unsafe_ptr() == other.unsafe_ptr():
            return True
        for i in range(len(self)):
            if self.unsafe_ptr()[i] != other.unsafe_ptr()[i]:
                return False
        return True

    @always_inline
    fn __ne__(self, other: String) -> Bool:
        """Compares two Strings if they do not have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are not equal and False otherwise.
        """
        return not (self == other)

    @always_inline
    fn __lt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String is strictly less than the RHS String and False
            otherwise.
        """
        return self.as_string_slice() < rhs.as_string_slice()

    @always_inline
    fn __le__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LE comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is less than or equal to the RHS String.
        """
        return not (rhs < self)

    @always_inline
    fn __gt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using GT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is strictly greater than the RHS String.
        """
        return rhs < self

    @always_inline
    fn __ge__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using GE comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True iff this String is greater than or equal to the RHS String.
        """
        return not (self < rhs)

    @staticmethod
    fn _add[rhs_has_null: Bool](lhs: Span[Byte], rhs: Span[Byte]) -> String:
        var lhs_len = len(lhs)
        var rhs_len = len(rhs)
        var lhs_ptr = lhs.unsafe_ptr()
        var rhs_ptr = rhs.unsafe_ptr()
        alias S = StringSlice[ImmutableAnyOrigin]
        if lhs_len == 0:
            return String(S(ptr=rhs_ptr, length=rhs_len))
        elif rhs_len == 0:
            return String(S(ptr=lhs_ptr, length=lhs_len))
        var sum_len = lhs_len + rhs_len
        var buffer = Self._buffer_type(capacity=sum_len + 1)
        var ptr = buffer.unsafe_ptr()
        memcpy(ptr, lhs_ptr, lhs_len)
        memcpy(ptr + lhs_len, rhs_ptr, rhs_len + int(rhs_has_null))
        buffer.size = sum_len + 1

        @parameter
        if not rhs_has_null:
            ptr[sum_len] = 0
        return Self(buffer^)

    @always_inline
    fn __add__(self, other: String) -> String:
        """Creates a string by appending another string at the end.

        Args:
            other: The string to append.

        Returns:
            The new constructed string.
        """
        return Self._add[True](self.as_bytes(), other.as_bytes())

    @always_inline
    fn __add__(self, other: StringLiteral) -> String:
        """Creates a string by appending a string literal at the end.

        Args:
            other: The string literal to append.

        Returns:
            The new constructed string.
        """
        return Self._add[False](self.as_bytes(), other.as_bytes())

    @always_inline
    fn __add__(self, other: StringSlice) -> String:
        """Creates a string by appending a string slice at the end.

        Args:
            other: The string slice to append.

        Returns:
            The new constructed string.
        """
        return Self._add[False](self.as_bytes(), other.as_bytes())

    @always_inline
    fn __radd__(self, other: String) -> String:
        """Creates a string by prepending another string to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return Self._add[True](other.as_bytes(), self.as_bytes())

    @always_inline
    fn __radd__(self, other: StringLiteral) -> String:
        """Creates a string by prepending another string literal to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return Self._add[True](other.as_bytes(), self.as_bytes())

    @always_inline
    fn __radd__(self, other: StringSlice) -> String:
        """Creates a string by prepending another string slice to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return Self._add[True](other.as_bytes(), self.as_bytes())

    fn _iadd[has_null: Bool](mut self, other: Span[Byte]):
        var s_len = self.byte_length()
        var o_len = len(other)
        var o_ptr = other.unsafe_ptr()
        if s_len == 0:
            alias S = StringSlice[ImmutableAnyOrigin]
            self = String(S(ptr=o_ptr, length=o_len))
            return
        elif o_len == 0:
            return
        var sum_len = s_len + o_len
        self._buffer.reserve(sum_len + 1)
        var s_ptr = self.unsafe_ptr()
        memcpy(s_ptr + s_len, o_ptr, o_len + int(has_null))
        self._buffer.size = sum_len + 1

        @parameter
        if not has_null:
            s_ptr[sum_len] = 0

    @always_inline
    fn __iadd__(mut self, other: String):
        """Appends another string to this string.

        Args:
            other: The string to append.
        """
        self._iadd[True](other.as_bytes())

    @always_inline
    fn __iadd__(mut self, other: StringLiteral):
        """Appends another string literal to this string.

        Args:
            other: The string to append.
        """
        self._iadd[False](other.as_bytes())

    @always_inline
    fn __iadd__(mut self, other: StringSlice):
        """Appends another string slice to this string.

        Args:
            other: The string to append.
        """
        self._iadd[False](other.as_bytes())

    fn __iter__(self) -> _StringSliceIter[__origin_of(self)]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return _StringSliceIter[__origin_of(self)](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    fn __reversed__(self) -> _StringSliceIter[__origin_of(self), False]:
        """Iterate backwards over the string, returning immutable references.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return _StringSliceIter[__origin_of(self), forward=False](
            unsafe_pointer=self.unsafe_ptr(), length=self.byte_length()
        )

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks if the string is not empty.

        Returns:
            True if the string length is greater than zero, and False otherwise.
        """
        return self.byte_length() > 0

    fn __len__(self) -> Int:
        """Gets the string length, in bytes (for now) PREFER:
        String.byte_length(), a future version will make this method return
        Unicode codepoints.

        Returns:
            The string length, in bytes (for now).
        """
        var unicode_length = self.byte_length()

        # TODO: everything uses this method assuming it's byte length
        # for i in range(unicode_length):
        #     if _utf8_byte_type(self._buffer[i]) == 1:
        #         unicode_length -= 1

        return unicode_length

    @always_inline
    fn __str__(self) -> String:
        """Gets the string itself.

        This method ensures that you can pass a `String` to a method that
        takes a `Stringable` value.

        Returns:
            The string itself.
        """
        return self

    fn __repr__(self) -> String:
        """Return a Mojo-compatible representation of the `String` instance.

        Returns:
            A new representation of the string.
        """
        var result = String()
        var use_dquote = False
        for s in self:
            use_dquote = use_dquote or (s == "'")

            if s == "\\":
                result += r"\\"
            elif s == "\t":
                result += r"\t"
            elif s == "\n":
                result += r"\n"
            elif s == "\r":
                result += r"\r"
            else:
                var codepoint = ord(s)
                if isprintable(codepoint):
                    result += s
                elif codepoint < 0x10:
                    result += hex(codepoint, prefix=r"\x0")
                elif codepoint < 0x20 or codepoint == 0x7F:
                    result += hex(codepoint, prefix=r"\x")
                else:  # multi-byte character
                    result += s

        if use_dquote:
            return '"' + result + '"'
        else:
            return "'" + result + "'"

    fn __fspath__(self) -> String:
        """Return the file system path representation (just the string itself).

        Returns:
          The file system path representation as a string.
        """
        return self

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn write_to[W: Writer](self, mut writer: W):
        """
        Formats this string to the provided Writer.

        Parameters:
            W: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """

        writer.write_bytes(self.as_bytes())

    fn join(self, *elems: Int) -> String:
        """Joins the elements from the tuple using the current string as a
        delimiter.

        Args:
            elems: The input tuple.

        Returns:
            The joined string.
        """
        if len(elems) == 0:
            return ""
        var curr = str(elems[0])
        for i in range(1, len(elems)):
            curr += self + str(elems[i])
        return curr

    fn join[*Types: Writable](self, *elems: *Types) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            Types: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """

        var result = String()
        var is_first = True

        @parameter
        fn add_elt[T: Writable](a: T):
            if is_first:
                is_first = False
            else:
                result.write(self)
            result.write(a)

        elems.each[add_elt]()
        return result

    fn join[T: StringableCollectionElement](self, elems: List[T, *_]) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            T: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """

        # TODO(#3403): Simplify this when the linked conditional conformance
        # feature is added.  Runs a faster algorithm if the concrete types are
        # able to be converted to a span of bytes.
        @parameter
        if _type_is_eq[T, String]():
            return self.fast_join(rebind[List[String]](elems))
        elif _type_is_eq[T, StringLiteral]():
            return self.fast_join(rebind[List[StringLiteral]](elems))
        # FIXME(#3597): once StringSlice conforms to CollectionElement trait:
        # if _type_is_eq[T, StringSlice]():
        # return self.fast_join(rebind[List[StringSlice]](elems))
        else:
            var result: String = ""
            var is_first = True

            for e in elems:
                if is_first:
                    is_first = False
                else:
                    result += self
                result += str(e[])

            return result

    fn fast_join[
        T: BytesCollectionElement, //,
    ](self, elems: List[T, *_]) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            T: The types of the elements.

        Args:
            elems: The input values.

        Returns:
            The joined string.
        """
        var n_elems = len(elems)
        if n_elems == 0:
            return String("")
        var len_self = self.byte_length()
        var len_elems = 0
        # Calculate the total size of the elements to join beforehand
        # to prevent alloc syscalls as we know the buffer size.
        # This can hugely improve the performance on large lists
        for e_ref in elems:
            len_elems += len(e_ref[].as_bytes())
        var capacity = len_self * (n_elems - 1) + len_elems
        var buf = Self._buffer_type(capacity=capacity)
        var self_ptr = self.unsafe_ptr()
        var ptr = buf.unsafe_ptr()
        var offset = 0
        var i = 0
        var is_first = True
        while i < n_elems:
            if is_first:
                is_first = False
            else:
                memcpy(dest=ptr + offset, src=self_ptr, count=len_self)
                offset += len_self
            var e = elems[i].as_bytes()
            var e_len = len(e)
            memcpy(dest=ptr + offset, src=e.unsafe_ptr(), count=e_len)
            offset += e_len
            i += 1
        buf.size = capacity
        buf.append(0)
        return String(buf^)

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        return self._buffer.data

    fn unsafe_cstr_ptr(self) -> UnsafePointer[c_char]:
        """Retrieves a C-string-compatible pointer to the underlying memory.

        The returned pointer is guaranteed to be null, or NUL terminated.

        Returns:
            The pointer to the underlying memory.
        """
        return self.unsafe_ptr().bitcast[c_char]()

    @always_inline
    fn as_bytes(ref self) -> Span[Byte, __origin_of(self)]:
        """Returns a contiguous slice of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.

        Notes:
            This does not include the trailing null terminator.
        """

        # Does NOT include the NUL terminator.
        return Span[Byte, __origin_of(self)](
            ptr=self._buffer.unsafe_ptr(), length=self.byte_length()
        )

    @always_inline
    fn as_string_slice(ref self) -> StringSlice[__origin_of(self)]:
        """Returns a string slice of the data owned by this string.

        Returns:
            A string slice pointing to the data owned by this string.
        """
        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in String so this is actually
        #   guaranteed to be valid.
        return StringSlice(unsafe_from_utf8=self.as_bytes())

    @always_inline
    fn byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes, excluding null terminator.

        Notes:
            This does not include the trailing null terminator in the count.
        """
        var length = len(self._buffer)
        return length - int(length > 0)

    fn _steal_ptr(mut self) -> UnsafePointer[UInt8]:
        """Transfer ownership of pointer to the underlying memory.
        The caller is responsible for freeing up the memory.

        Returns:
            The pointer to the underlying memory.
        """
        var ptr = self.unsafe_ptr()
        self._buffer.data = UnsafePointer[UInt8]()
        self._buffer.size = 0
        self._buffer.capacity = 0
        return ptr

    fn count(self, substr: String) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        if not substr:
            return len(self) + 1

        var res = 0
        var offset = 0

        while True:
            var pos = self.find(substr, offset)
            if pos == -1:
                break
            res += 1

            offset = pos + substr.byte_length()

        return res

    fn __contains__(self, substr: String) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return substr.as_string_slice() in self.as_string_slice()

    fn find(self, substr: String, start: Int = 0) -> Int:
        """Finds the offset of the first occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        return self.as_string_slice().find(substr.as_string_slice(), start)

    fn rfind(self, substr: String, start: Int = 0) -> Int:
        """Finds the offset of the last occurrence of `substr` starting at
        `start`. If not found, returns -1.

        Args:
          substr: The substring to find.
          start: The offset from which to find.

        Returns:
          The offset of `substr` relative to the beginning of the string.
        """

        return self.as_string_slice().rfind(
            substr.as_string_slice(), start=start
        )

    fn isspace(self) -> Bool:
        """Determines whether every character in the given String is a
        python whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole String is made up of whitespace characters
                listed above, otherwise False.
        """
        return self.as_string_slice().isspace()

    fn split(self, sep: String, maxsplit: Int = -1) raises -> List[String]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.
                Defaults to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Raises:
            If the separator is empty.

        Examples:

        ```mojo
        # Splitting a space
        _ = String("hello world").split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = String("hello,,world").split(",") # ["hello", "", "world"]
        # Splitting with maxsplit
        _ = String("1,2,3").split(",", 1) # ['1', '2,3']
        ```
        .
        """
        var output = List[String]()

        var str_byte_len = self.byte_length() - 1
        var lhs = 0
        var rhs = 0
        var items = 0
        var sep_len = sep.byte_length()
        if sep_len == 0:
            raise Error("Separator cannot be empty.")
        if str_byte_len < 0:
            output.append("")

        while lhs <= str_byte_len:
            rhs = self.find(sep, lhs)
            if rhs == -1:
                output.append(self[lhs:])
                break

            if maxsplit > -1:
                if items == maxsplit:
                    output.append(self[lhs:])
                    break
                items += 1

            output.append(self[lhs:rhs])
            lhs = rhs + sep_len

        if self.endswith(sep) and (len(output) <= maxsplit or maxsplit == -1):
            output.append("")
        return output

    fn split(self, sep: NoneType = None, maxsplit: Int = -1) -> List[String]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from String. Defaults
                to unlimited.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = String("      ").split() # []
        _ = String("").split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = String("      hello    world     ").split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = String(
            "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029world"
        ).split()  # ["hello", "world"]
        ```
        .
        """

        fn num_bytes(b: UInt8) -> Int:
            var flipped = ~b
            return int(count_leading_zeros(flipped) + (flipped >> 7))

        var output = List[String]()
        var str_byte_len = self.byte_length() - 1
        var lhs = 0
        var rhs = 0
        var items = 0
        while lhs <= str_byte_len:
            # Python adds all "whitespace chars" as one separator
            # if no separator was specified
            for s in self[lhs:]:
                if not s.isspace():
                    break
                lhs += s.byte_length()
            # if it went until the end of the String, then
            # it should be sliced up until the original
            # start of the whitespace which was already appended
            if lhs - 1 == str_byte_len:
                break
            elif lhs == str_byte_len:
                # if the last char is not whitespace
                output.append(self[str_byte_len])
                break
            rhs = lhs + num_bytes(self.unsafe_ptr()[lhs])
            for s in self[lhs + num_bytes(self.unsafe_ptr()[lhs]) :]:
                if s.isspace():
                    break
                rhs += s.byte_length()

            if maxsplit > -1:
                if items == maxsplit:
                    output.append(self[lhs:])
                    break
                items += 1

            output.append(self[lhs:rhs])
            lhs = rhs

        return output

    fn splitlines(self, keepends: Bool = False) -> List[String]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        return _to_string_list(self.as_string_slice().splitlines(keepends))

    fn replace(self, old: String, new: String) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        if not old:
            return self._interleave(new)

        var occurrences = self.count(old)
        if occurrences == -1:
            return self

        var self_start = self.unsafe_ptr()
        var self_ptr = self.unsafe_ptr()
        var new_ptr = new.unsafe_ptr()

        var self_len = self.byte_length()
        var old_len = old.byte_length()
        var new_len = new.byte_length()

        var res = Self._buffer_type()
        res.reserve(self_len + (old_len - new_len) * occurrences + 1)

        for _ in range(occurrences):
            var curr_offset = int(self_ptr) - int(self_start)

            var idx = self.find(old, curr_offset)

            debug_assert(idx >= 0, "expected to find occurrence during find")

            # Copy preceding unchanged chars
            for _ in range(curr_offset, idx):
                res.append(self_ptr[])
                self_ptr += 1

            # Insert a copy of the new replacement string
            for i in range(new_len):
                res.append(new_ptr[i])

            self_ptr += old_len

        while True:
            var val = self_ptr[]
            if val == 0:
                break
            res.append(self_ptr[])
            self_ptr += 1

        res.append(0)
        return String(res^)

    fn strip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading or trailing characters.
        """

        return self.lstrip(chars).rstrip(chars)

    fn strip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading and trailing whitespaces
        removed.

        Returns:
            A copy of the string with no leading or trailing whitespaces.
        """
        return self.lstrip().rstrip()

    fn rstrip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no trailing characters.
        """

        return self.as_string_slice().rstrip(chars)

    fn rstrip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with trailing whitespaces removed.

        Returns:
            A copy of the string with no trailing whitespaces.
        """
        return self.as_string_slice().rstrip()

    fn lstrip(self, chars: StringSlice) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.
        """

        return self.as_string_slice().lstrip(chars)

    fn lstrip(self) -> StringSlice[__origin_of(self)]:
        """Return a copy of the string with leading whitespaces removed.

        Returns:
            A copy of the string with no leading whitespaces.
        """
        return self.as_string_slice().lstrip()

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.as_string_slice())

    fn __hash__[H: _Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_bytes(self.unsafe_ptr(), self.byte_length())

    fn _interleave(self, val: String) -> String:
        var res = Self._buffer_type()
        var val_ptr = val.unsafe_ptr()
        var self_ptr = self.unsafe_ptr()
        res.reserve(val.byte_length() * self.byte_length() + 1)
        for i in range(self.byte_length()):
            for j in range(val.byte_length()):
                res.append(val_ptr[j])
            res.append(self_ptr[i])
        res.append(0)
        return String(res^)

    fn lower(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        # TODO: the _unicode module does not support locale sensitive conversions yet.
        return to_lowercase(self)

    fn upper(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        # TODO: the _unicode module does not support locale sensitive conversions yet.
        return to_uppercase(self)

    fn startswith(
        ref self, prefix: String, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          prefix: The prefix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is prefixed by the input prefix.
        """
        if end == -1:
            return StringSlice[__origin_of(self)](
                ptr=self.unsafe_ptr() + start,
                length=self.byte_length() - start,
            ).startswith(prefix.as_string_slice())

        return StringSlice[__origin_of(self)](
            ptr=self.unsafe_ptr() + start, length=end - start
        ).startswith(prefix.as_string_slice())

    fn endswith(self, suffix: String, start: Int = 0, end: Int = -1) -> Bool:
        """Checks if the string end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        Args:
          suffix: The suffix to check.
          start: The start offset from which to check.
          end: The end offset from which to check.

        Returns:
          True if the self[start:end] is suffixed by the input suffix.
        """
        if end == -1:
            return StringSlice[__origin_of(self)](
                ptr=self.unsafe_ptr() + start,
                length=self.byte_length() - start,
            ).endswith(suffix.as_string_slice())

        return StringSlice[__origin_of(self)](
            ptr=self.unsafe_ptr() + start, length=end - start
        ).endswith(suffix.as_string_slice())

    fn removeprefix(self, prefix: String, /) -> String:
        """Returns a new string with the prefix removed if it was present.

        For example:

        ```mojo
        print(String('TestHook').removeprefix('Test'))
        # 'Hook'
        print(String('BaseTestCase').removeprefix('Test'))
        # 'BaseTestCase'
        ```

        Args:
            prefix: The prefix to remove from the string.

        Returns:
            `string[len(prefix):]` if the string starts with the prefix string,
            or a copy of the original string otherwise.
        """
        if self.startswith(prefix):
            return self[prefix.byte_length() :]
        return self

    fn removesuffix(self, suffix: String, /) -> String:
        """Returns a new string with the suffix removed if it was present.

        For example:

        ```mojo
        print(String('TestHook').removesuffix('Hook'))
        # 'Test'
        print(String('BaseTestCase').removesuffix('Test'))
        # 'BaseTestCase'
        ```

        Args:
            suffix: The suffix to remove from the string.

        Returns:
            `string[:-len(suffix)]` if the string ends with the suffix string,
            or a copy of the original string otherwise.
        """
        if suffix and self.endswith(suffix):
            return self[: -suffix.byte_length()]
        return self

    @always_inline
    fn __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.
        """
        return atol(self)

    @always_inline
    fn __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.
        """
        return atof(self)

    fn __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n : The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """
        return self.as_string_slice() * n

    @always_inline
    fn format[*Ts: _CurlyEntryFormattable](self, *args: *Ts) raises -> String:
        """Format a template with `*args`.

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Representable`
                and `Stringable` (to be changed and made more flexible).

        Returns:
            The template with the given values substituted.

        Examples:

        ```mojo
        # Manual indexing:
        print(String("{0} {1} {0}").format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print(String("{} {}").format(True, "hello world")) # True hello world
        ```
        .
        """
        return _FormatCurlyEntry.format(self, args)

    fn isdigit(self) -> Bool:
        """A string is a digit string if all characters in the string are digits
        and there is at least one character in the string.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits and it's not empty else False.
        """
        if not self:
            return False
        for c in self:
            if not isdigit(ord(c)):
                return False
        return True

    fn isupper(self) -> Bool:
        """Returns True if all cased characters in the string are uppercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are uppercase and there
            is at least one cased character, False otherwise.
        """
        return len(self) > 0 and is_uppercase(self)

    fn islower(self) -> Bool:
        """Returns True if all cased characters in the string are lowercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are lowercase and there
            is at least one cased character, False otherwise.
        """
        return len(self) > 0 and is_lowercase(self)

    fn isprintable(self) -> Bool:
        """Returns True if all characters in the string are ASCII printable.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are printable else False.
        """
        for c in self:
            if not isprintable(ord(c)):
                return False
        return True

    fn rjust(self, width: Int, fillchar: StringLiteral = " ") -> String:
        """Returns the string right justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns right justified string, or self if width is not bigger than self length.
        """
        return self._justify(width - len(self), width, fillchar)

    fn ljust(self, width: Int, fillchar: StringLiteral = " ") -> String:
        """Returns the string left justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns left justified string, or self if width is not bigger than self length.
        """
        return self._justify(0, width, fillchar)

    fn center(self, width: Int, fillchar: StringLiteral = " ") -> String:
        """Returns the string center justified in a string of specified width.

        Args:
            width: The width of the field containing the string.
            fillchar: Specifies the padding character.

        Returns:
            Returns center justified string, or self if width is not bigger than self length.
        """
        return self._justify(width - len(self) >> 1, width, fillchar)

    fn _justify(
        self, start: Int, width: Int, fillchar: StringLiteral
    ) -> String:
        if len(self) >= width:
            return self
        debug_assert(
            len(fillchar) == 1, "fill char needs to be a one byte literal"
        )
        var fillbyte = fillchar.as_bytes()[0]
        var buffer = Self._buffer_type(capacity=width + 1)
        buffer.resize(width, fillbyte)
        buffer.append(0)
        memcpy(buffer.unsafe_ptr().offset(start), self.unsafe_ptr(), len(self))
        var result = String(buffer)
        return result^

    fn reserve(mut self, new_capacity: Int):
        """Reserves the requested capacity.

        Args:
            new_capacity: The new capacity.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the data is moved.
        """
        self._buffer.reserve(new_capacity)


# ===----------------------------------------------------------------------=== #
# Utilities
# ===----------------------------------------------------------------------=== #


fn _toggle_ascii_case(char: UInt8) -> UInt8:
    """Assuming char is a cased ASCII character, this function will return the
    opposite-cased letter.
    """

    # ASCII defines A-Z and a-z as differing only in their 6th bit,
    # so converting is as easy as a bit flip.
    return char ^ (1 << 5)


fn _calc_initial_buffer_size_int32(n0: Int) -> Int:
    # See https://commaok.xyz/post/lookup_tables/ and
    # https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
    # for a description.
    alias lookup_table = VariadicList[Int](
        4294967296,
        8589934582,
        8589934582,
        8589934582,
        12884901788,
        12884901788,
        12884901788,
        17179868184,
        17179868184,
        17179868184,
        21474826480,
        21474826480,
        21474826480,
        21474826480,
        25769703776,
        25769703776,
        25769703776,
        30063771072,
        30063771072,
        30063771072,
        34349738368,
        34349738368,
        34349738368,
        34349738368,
        38554705664,
        38554705664,
        38554705664,
        41949672960,
        41949672960,
        41949672960,
        42949672960,
        42949672960,
    )
    var n = UInt32(n0)
    var log2 = int(
        (bitwidthof[DType.uint32]() - 1) ^ count_leading_zeros(n | 1)
    )
    return (n0 + lookup_table[int(log2)]) >> 32


fn _calc_initial_buffer_size_int64(n0: UInt64) -> Int:
    var result: Int = 1
    var n = n0
    while True:
        if n < 10:
            return result
        if n < 100:
            return result + 1
        if n < 1_000:
            return result + 2
        if n < 10_000:
            return result + 3
        n //= 10_000
        result += 4


fn _calc_initial_buffer_size(n0: Int) -> Int:
    var sign = 0 if n0 > 0 else 1

    # Add 1 for the terminator
    return sign + n0._decimal_digit_count() + 1


fn _calc_initial_buffer_size(n: Float64) -> Int:
    return 128 + 1  # Add 1 for the terminator


fn _calc_initial_buffer_size[type: DType](n0: Scalar[type]) -> Int:
    @parameter
    if type.is_integral():
        var n = abs(n0)
        var sign = 0 if n0 > 0 else 1
        alias is_32bit_system = bitwidthof[DType.index]() == 32

        @parameter
        if is_32bit_system or bitwidthof[type]() <= 32:
            return sign + _calc_initial_buffer_size_int32(int(n)) + 1
        else:
            return (
                sign
                + _calc_initial_buffer_size_int64(n.cast[DType.uint64]())
                + 1
            )

    return 128 + 1  # Add 1 for the terminator


fn _calc_format_buffer_size[type: DType]() -> Int:
    """
    Returns a buffer size in bytes that is large enough to store a formatted
    number of the specified type.
    """

    # TODO:
    #   Use a smaller size based on the `dtype`, e.g. we don't need as much
    #   space to store a formatted int8 as a float64.
    @parameter
    if type.is_integral():
        return 64 + 1
    else:
        return 128 + 1  # Add 1 for the terminator
