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
from sys import bitwidthof, llvm_intrinsic
from sys.ffi import c_char
from utils import StaticString, write_args

from bit import count_leading_zeros
from memory import UnsafePointer, memcmp, memcpy
from python import PythonObject

from sys.intrinsics import _type_is_eq
from hashlib._hasher import _HashableWithHasher, _Hasher

from utils import (
    IndexList,
    StringRef,
    Variant,
    Writable,
    Writer,
)
from utils.span import Span, AsBytes
from utils.string_slice import (
    StringSlice,
    _utf8_byte_type,
    _StringSliceIter,
    _unicode_codepoint_utf8_byte_length,
    _shift_unicode_to_utf8,
    _FormatCurlyEntry,
    _CurlyEntryFormattable,
    _to_string_list,
    Stringlike,
    _utf8_first_byte_sequence_length,
    _is_continuation_byte,
    _is_valid_utf8,
)

from utils._unicode import (
    is_lowercase,
    is_uppercase,
    to_lowercase,
    to_uppercase,
)
from builtin.builtin_list import _lit_mut_cast

# ===----------------------------------------------------------------------=== #
# ord
# ===----------------------------------------------------------------------=== #


fn ord[T: Stringlike, //](ref [_]s: T) -> Int:
    """Returns the unicode codepoint for the character.

    Parameters:
        T: The Stringlike type.

    Args:
        s: The input string slice, which must contain only a single character.

    Returns:
        An integer representing the code point of the given character.

    Examples:
    ```mojo
    print(ord("a"), ord("€")) # 97 8364
    ```
    .
    """

    # FIXME(#933): llvm intrinsic can't recognize !pop.scalar<ui8> value when
    # trying to fold ctlz at comp time
    @parameter
    if _type_is_eq[T, StringLiteral]():
        var v = rebind[StringLiteral](s)
        var p = v.unsafe_ptr()
        var b0 = Byte(0)
        if v.byte_length() > 0:
            b0 = p[0]
        debug_assert(not _is_continuation_byte(b0), "invalid byte at index 0")
        alias c_byte_mask = 0b0011_1111

        if b0 < 0b1000_0000:
            return int(b0)
        elif b0 < 0b1110_0000:
            debug_assert(v.byte_length() == 2, "wrong sized string")
            var b0_mask = 0b1111_1111 >> 3
            debug_assert(_is_continuation_byte(p[1]), "invalid byte at index 1")
            return (int(b0 & b0_mask) << 6) | int(p[1] & c_byte_mask)
        elif b0 < 0b1111_0000:
            debug_assert(v.byte_length() == 3, "wrong sized string")
            var b0_mask = 0b1111_1111 >> 4
            debug_assert(_is_continuation_byte(p[1]), "invalid byte at index 1")
            debug_assert(_is_continuation_byte(p[2]), "invalid byte at index 2")
            return (
                (int(b0 & b0_mask) << 12)
                | (int(p[1] & c_byte_mask) << 6)
                | int(p[2] & c_byte_mask)
            )
        else:
            debug_assert(v.byte_length() == 4, "wrong sized string")
            var b0_mask = 0b1111_1111 >> 5
            debug_assert(_is_continuation_byte(p[1]), "invalid byte at index 1")
            debug_assert(_is_continuation_byte(p[2]), "invalid byte at index 2")
            debug_assert(_is_continuation_byte(p[3]), "invalid byte at index 3")
            return (
                (int(b0 & b0_mask) << 18)
                | (int(p[1] & c_byte_mask) << 12)
                | (int(p[2] & c_byte_mask) << 6)
                | int(p[3] & c_byte_mask)
            )
    else:
        return _ord(
            StringSlice(
                unsafe_from_utf8=Span[
                    Byte, _lit_mut_cast[__origin_of(s), False].result
                ](ptr=s.unsafe_ptr(), length=s.byte_length())
            )
        )


fn _ord[O: ImmutableOrigin, //](s: StringSlice[O]) -> Int:
    # UTF-8 to Unicode conversion:              (represented as UInt32 BE)
    # 1: 0aaaaaaa                            -> 00000000 00000000 00000000 0aaaaaaa     a
    # 2: 110aaaaa 10bbbbbb                   -> 00000000 00000000 00000aaa aabbbbbb     a << 6  | b
    # 3: 1110aaaa 10bbbbbb 10cccccc          -> 00000000 00000000 aaaabbbb bbcccccc     a << 12 | b << 6  | c
    # 4: 11110aaa 10bbbbbb 10cccccc 10dddddd -> 00000000 000aaabb bbbbcccc ccdddddd     a << 18 | b << 12 | c << 6 | d

    if s.byte_length() == 0:
        return 0
    var p = s.unsafe_ptr()
    var b0 = p[0]
    var num_bytes = _utf8_first_byte_sequence_length(b0)
    debug_assert(
        s.byte_length() == num_bytes, "input string must be one character"
    )
    debug_assert(1 <= num_bytes <= 4, "invalid UTF-8 byte ", b0, " at index 0")
    alias c_byte_mask = 0b0011_1111
    var b0_mask = 0b1111_1111 >> (num_bytes + int(num_bytes > 1))
    var shift = int((6 * (num_bytes - 1)))
    var result = int(b0 & b0_mask) << shift
    for i in range(1, num_bytes):
        var b = p[i]
        debug_assert(
            _is_continuation_byte(b), "invalid UTF-8 byte ", b, " at index ", i
        )
        shift -= 6
        result |= int(b & c_byte_mask) << shift
    return result


# FIXME: remove num_bytes once variadic list length can be used at comp time
# and maybe make this public
fn _ord[num_bytes: Int](*args: Byte) -> Int:
    """Returns the unicode codepoint for the given utf8 byte sequence.

    Args:
        args: The input bytes, which must be only a single character.

    Returns:
        An integer representing the code point of the given character.

    Examples:
    ```mojo
    %# from collections.string import _ord
    print(_ord[2](0xC3, 0xBD)) # 0xFD
    ```
    .
    """

    debug_assert(num_bytes == len(args), "invalid num_bytes")
    debug_assert(
        num_bytes == _utf8_first_byte_sequence_length(args[0]),
        "invalid first byte",
    )

    @parameter
    for i in range(1, num_bytes):
        var b = args[i]
        debug_assert(
            _is_continuation_byte(b), "invalid UTF-8 byte ", b, " at index ", i
        )

    alias b0_mask = 0b1111_1111 >> (num_bytes + 1)
    alias c_byte_mask = 0b0011_1111

    @parameter
    if num_bytes == 1:
        return int(args[0])
    elif num_bytes == 2:
        return (int(args[0] & b0_mask) << 6) | int(args[1]) & c_byte_mask
    elif num_bytes == 3:
        return (
            (int(args[0] & b0_mask) << 12)
            | (int(args[1] & c_byte_mask) << 6)
            | (int(args[2] & c_byte_mask))
        )
    else:
        return (
            (int(args[0] & b0_mask) << 18)
            | (int(args[1] & c_byte_mask) << 12)
            | (int(args[2] & c_byte_mask) << 6)
            | int(args[3] & c_byte_mask)
        )


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
    print(chr(97), chr(8364)) # a €
    ```
    .
    """

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
# repr
# ===----------------------------------------------------------------------=== #


fn _repr[T: Stringlike, //](value: T) -> String:
    alias `'` = Byte(ord("'"))
    alias `"` = Byte(ord('"'))
    alias `\\` = Byte(ord("\\"))
    alias `\t` = Byte(ord("\t"))
    alias `t` = Byte(ord("t"))
    alias `\n` = Byte(ord("\n"))
    alias `n` = Byte(ord("n"))
    alias `\r` = Byte(ord("\r"))
    alias `r` = Byte(ord("r"))

    var span = Span[Byte, ImmutableAnyOrigin](
        ptr=value.unsafe_ptr(), length=value.byte_length()
    )
    var span_len = len(span)
    debug_assert(_is_valid_utf8(span), "invalid utf8 sequence")
    var nonprintable_python = span.count[func=_nonprintable_python]()
    var hex_prefix = 3 * nonprintable_python  # \xHH
    var length = span_len + hex_prefix + 2  # for the quotes
    var buf = String._buffer_type(capacity=length + 1)  # null terminator

    var use_dquote = False
    v_ptr, b_ptr = value.unsafe_ptr(), buf.unsafe_ptr()
    v_idx, b_idx = 0, 1

    while v_idx < span_len:
        var b0 = v_ptr[v_idx]
        var seq_len = _utf8_first_byte_sequence_length(b0)
        use_dquote = use_dquote or (b0 == `'`)
        # Python escapes backslashes but they are ASCII printable
        if b0 == `\\`:
            (b_ptr + b_idx).init_pointee_copy(`\\`)
            (b_ptr + b_idx + 1).init_pointee_copy(`\\`)
            b_idx += 2
        elif isprintable(b0):
            (b_ptr + b_idx).init_pointee_copy(b0)
            b_idx += 1
        elif b0 == `\t`:
            (b_ptr + b_idx).init_pointee_copy(`\\`)
            (b_ptr + b_idx + 1).init_pointee_copy(`t`)
            b_idx += 2
        elif b0 == `\n`:
            (b_ptr + b_idx).init_pointee_copy(`\\`)
            (b_ptr + b_idx + 1).init_pointee_copy(`n`)
            b_idx += 2
        elif b0 == `\r`:
            (b_ptr + b_idx).init_pointee_copy(`\\`)
            (b_ptr + b_idx + 1).init_pointee_copy(`r`)
            b_idx += 2
        elif seq_len == 1:
            _write_hex[2](b_ptr + b_idx, int(b0))
            b_idx += 4
        else:
            for i in range(seq_len):
                (b_ptr + b_idx + i).init_pointee_copy(v_ptr[v_idx + i])
            b_idx += seq_len
        v_idx += seq_len

    if use_dquote:
        b_ptr.init_pointee_copy(`"`)
        (b_ptr + b_idx).init_pointee_copy(`"`)
    else:
        b_ptr.init_pointee_copy(`'`)
        (b_ptr + b_idx).init_pointee_copy(`'`)
    (b_ptr + b_idx + 1).init_pointee_copy(0)  # null terminator
    buf.size = b_idx + 2
    return String(buf^)


# ===----------------------------------------------------------------------=== #
# ascii
# ===----------------------------------------------------------------------=== #


@always_inline
fn _isdigit_vec[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `0` = SIMD[DType.uint8, w](Byte(ord("0")))
    alias `9` = SIMD[DType.uint8, w](Byte(ord("9")))
    return (`0` <= v) & (v <= `9`)


@always_inline
fn isdigit(v: SIMD[DType.uint8]) -> Bool:
    """Determines whether the given characters are a digit: [0, 9].

    Args:
        v: The characters to check.

    Returns:
        True if the characters are a digit.
    """
    return _isdigit_vec(v).reduce_and()


@always_inline
fn isdigit(c: Byte) -> Bool:
    """Determines whether the given character is a digit: [0, 9].

    Args:
        c: The character to check.

    Returns:
        True if the character is a digit.
    """
    return _isdigit_vec(c)


@always_inline
fn _is_ascii_printable_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias ` ` = SIMD[DType.uint8, w](Byte(ord(" ")))
    alias `~` = SIMD[DType.uint8, w](Byte(ord("~")))
    return (` ` <= v) & (v <= `~`)


@always_inline
fn isprintable(v: SIMD[DType.uint8]) -> Bool:
    """Determines whether the given characters are ASCII printable.

    Args:
        v: The characters to check.

    Returns:
        True if the characters are printable, otherwise False.
    """
    return _is_ascii_printable_vec(v).reduce_and()


@always_inline
fn isprintable(c: Byte) -> Bool:
    """Determines whether the given character is ASCII printable.

    Args:
        c: The character to check.

    Returns:
        True if the character is printable, otherwise False.
    """
    return _is_ascii_printable_vec(c)


@always_inline
fn isprintable(span: Span[Byte]) -> Bool:
    """Determines whether the given characters are ASCII printable.

    Args:
        span: The characters to check.

    Returns:
        True if the characters are printable, otherwise False.
    """
    return span.count[func=_is_ascii_printable_vec]() == len(span)


@always_inline
fn _nonprintable_ascii[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (~_is_ascii_printable_vec(v)) & (v < 0b1000_0000)


@always_inline
fn _is_python_printable_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `\\` = SIMD[DType.uint8, w](Byte(ord(" ")))
    return (v != `\\`) & _is_ascii_printable_vec(v)


@always_inline
fn _is_python_printable(b: Byte) -> Bool:
    return _is_python_printable_vec(b)


@always_inline
fn _nonprintable_python[w: Int](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    return (~_is_python_printable_vec(v)) & (v < 0b1000_0000)


@always_inline
fn _byte_to_hex_string(b: Byte) -> Byte:
    alias `0` = Byte(ord("0"))
    alias `9` = Byte(ord("9"))
    alias `a` = Byte(ord("a"))
    return `0` + int(b > 9) * (`a` - `9` - 1) + b


@always_inline
fn _write_hex[amnt_hex_bytes: Int](p: UnsafePointer[Byte], codepoint: Int):
    """Write a python compliant hexadecimal value into an uninitialized pointer
    location, assumed to be large enough for the value to be written."""
    alias `\\` = Byte(ord("\\"))
    alias `x` = Byte(ord("x"))
    alias `u` = Byte(ord("u"))
    alias `U` = Byte(ord("U"))

    constrained[amnt_hex_bytes in (2, 4, 8), "only 2 or 4 or 8 sequences"]()
    p.init_pointee_copy(`\\`)

    @parameter
    if amnt_hex_bytes == 2:
        (p + 1).init_pointee_copy(`x`)
    elif amnt_hex_bytes == 4:
        (p + 1).init_pointee_copy(`u`)
    else:
        (p + 1).init_pointee_copy(`U`)
    var idx = 2

    @parameter
    for i in reversed(range(amnt_hex_bytes)):
        (p + idx).init_pointee_copy(
            _byte_to_hex_string((codepoint // (16**i)) % 16)
        )
        idx += 1


trait _HasAscii:
    fn __ascii__(self) -> String:
        ...


fn ascii[T: _HasAscii](value: T) -> String:
    """Get the ASCII representation of the object.

    Parameters:
        T: The type.

    Args:
        value: The object to get the ASCII representation of.

    Returns:
        A string containing the ASCII representation of the object.
    """
    return value.__ascii__()


fn _ascii[T: Stringlike, //](value: T) -> String:
    alias `'` = Byte(ord("'"))
    alias `"` = Byte(ord('"'))

    var span = Span[Byte, ImmutableAnyOrigin](
        ptr=value.unsafe_ptr(), length=value.byte_length()
    )
    var span_len = len(span)
    debug_assert(_is_valid_utf8(span), "invalid utf8 sequence")
    var non_printable_ascii = span.count[func=_nonprintable_ascii]()
    var continuation_bytes = span.count[func=_is_continuation_byte]()
    var hex_prefix = 3 * (non_printable_ascii + continuation_bytes)
    var length = span_len + hex_prefix + 2  # for the quotes
    var buf = String._buffer_type(capacity=length + 1)  # null terminator

    var use_dquote = False
    v_ptr, b_ptr = value.unsafe_ptr(), buf.unsafe_ptr()
    v_idx, b_idx = 0, 1

    while v_idx < span_len:
        var b0 = v_ptr[v_idx]
        use_dquote = use_dquote or (b0 == `'`)
        var seq_len = _utf8_first_byte_sequence_length(b0)
        var b1 = v_ptr[v_idx + int(seq_len > 1)]
        var is_2byte_short = seq_len == 2 and b0 <= 0xC3
        if isprintable(b0):
            b_ptr[b_idx] = b0
            b_idx += 1
        elif seq_len == 1 or is_2byte_short:
            codepoint = int(b0)
            if is_2byte_short:
                codepoint = _ord[2](b0, b1)
            _write_hex[2](b_ptr + b_idx, codepoint)
            b_idx += 4
        elif seq_len < 4:
            var codepoint: Int
            if seq_len == 2:
                codepoint = _ord[2](b0, b1)
            else:
                codepoint = _ord[3](b0, b1, v_ptr[v_idx + 2])
            _write_hex[4](b_ptr + b_idx, codepoint)
            b_idx += 6
        else:
            codepoint = _ord[4](b0, b1, v_ptr[v_idx + 2], v_ptr[v_idx + 3])
            _write_hex[8](b_ptr + b_idx, codepoint)
            b_idx += 10
        v_idx += seq_len

    if use_dquote:
        b_ptr.init_pointee_copy(`"`)
        (b_ptr + b_idx).init_pointee_copy(`"`)
    else:
        b_ptr.init_pointee_copy(`'`)
        (b_ptr + b_idx).init_pointee_copy(`'`)
    (b_ptr + b_idx + 1).init_pointee_copy(0)  # null terminator
    buf.size = b_idx + 2
    return String(buf^)


@always_inline
fn _is_ascii_uppercase_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `A` = SIMD[DType.uint8, w](Byte(ord("A")))
    alias `Z` = SIMD[DType.uint8, w](Byte(ord("Z")))
    return (`A` <= v) & (v <= `Z`)


@always_inline
fn _is_ascii_uppercase(c: Byte) -> Bool:
    return _is_ascii_uppercase_vec(c)


@always_inline
fn _is_ascii_uppercase(v: SIMD[DType.uint8]) -> Bool:
    return _is_ascii_uppercase_vec(v).reduce_and()


@always_inline
fn _is_ascii_uppercase(span: Span[Byte]) -> Bool:
    return span.count[func=_is_ascii_uppercase_vec]() == len(span)


@always_inline
fn _is_ascii_lowercase_vec[
    w: Int
](v: SIMD[DType.uint8, w]) -> SIMD[DType.bool, w]:
    alias `a` = SIMD[DType.uint8, w](Byte(ord("a")))
    alias `z` = SIMD[DType.uint8, w](Byte(ord("z")))
    return (`a` <= v) & (v <= `z`)


@always_inline
fn _is_ascii_lowercase(c: Byte) -> Bool:
    return _is_ascii_lowercase_vec(c)


@always_inline
fn _is_ascii_lowercase(v: SIMD[DType.uint8]) -> Bool:
    return _is_ascii_lowercase_vec(v).reduce_and()


@always_inline
fn _is_ascii_lowercase(span: Span[Byte]) -> Bool:
    return span.count[func=_is_ascii_lowercase_vec]() == len(span)


fn _is_ascii_space(c: Byte) -> Bool:
    """Determines whether the given character is an ASCII whitespace character:
    `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

    Args:
        c: The character to check.

    Returns:
        True if the character is one of the ASCII whitespace characters.

    Notes:
        For semantics similar to Python, use `String.isspace()`.
    """

    # NOTE: a global LUT doesn't work at compile time so we can't use it here.
    alias ` ` = Byte(ord(" "))
    alias `\t` = Byte(ord("\t"))
    alias `\n` = Byte(ord("\n"))
    alias `\r` = Byte(ord("\r"))
    alias `\f` = Byte(ord("\f"))
    alias `\v` = Byte(ord("\v"))
    alias `\x1c` = Byte(ord("\x1c"))
    alias `\x1d` = Byte(ord("\x1d"))
    alias `\x1e` = Byte(ord("\x1e"))

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
# strtol
# ===----------------------------------------------------------------------=== #


fn _atol(str_ref: StringSlice[_], base: Int = 10) raises -> Int:
    """Implementation of `atol` for StringRef inputs.

    Please see its docstring for details.
    """
    if (base != 0) and (base < 2 or base > 36):
        raise Error("Base must be >= 2 and <= 36, or 0.")
    if not str_ref:
        raise Error(_atol_error(base, str_ref))

    var real_base: Int
    var ord_num_max: Int

    var ord_letter_max = (-1, -1)
    var result = 0
    var is_negative: Bool = False
    var has_prefix: Bool = False
    var start: Int = 0
    var str_len = str_ref.byte_length()
    var buff = str_ref.unsafe_ptr()

    for pos in range(start, str_len):
        if _is_ascii_space(buff[pos]):
            continue

        if str_ref[pos] == "-":
            is_negative = True
            start = pos + 1
        elif str_ref[pos] == "+":
            start = pos + 1
        else:
            start = pos
        break

    if str_ref[start] == "0" and start + 1 < str_len:
        if base == 2 and (
            str_ref[start + 1] == "b" or str_ref[start + 1] == "B"
        ):
            start += 2
            has_prefix = True
        elif base == 8 and (
            str_ref[start + 1] == "o" or str_ref[start + 1] == "O"
        ):
            start += 2
            has_prefix = True
        elif base == 16 and (
            str_ref[start + 1] == "x" or str_ref[start + 1] == "X"
        ):
            start += 2
            has_prefix = True

    alias ord_0 = ord("0")
    # FIXME:
    #   Change this to `alias` after fixing support for __getitem__ of alias.
    var ord_letter_min = (ord("a"), ord("A"))
    alias ord_underscore = ord("_")

    if base == 0:
        var real_base_new_start = _identify_base(str_ref, start)
        real_base = real_base_new_start[0]
        start = real_base_new_start[1]
        has_prefix = real_base != 10
        if real_base == -1:
            raise Error(_atol_error(base, str_ref))
    else:
        real_base = base

    if real_base <= 10:
        ord_num_max = ord(str(real_base - 1))
    else:
        ord_num_max = ord("9")
        ord_letter_max = (
            ord("a") + (real_base - 11),
            ord("A") + (real_base - 11),
        )

    var found_valid_chars_after_start = False
    var has_space_after_number = False
    # Prefixed integer literals with real_base 2, 8, 16 may begin with leading
    # underscores under the conditions they have a prefix
    var was_last_digit_undescore = not (real_base in (2, 8, 16) and has_prefix)
    for pos in range(start, str_len):
        var ord_current = int(buff[pos])
        if ord_current == ord_underscore:
            if was_last_digit_undescore:
                raise Error(_atol_error(base, str_ref))
            else:
                was_last_digit_undescore = True
                continue
        else:
            was_last_digit_undescore = False
        if ord_0 <= ord_current <= ord_num_max:
            result += ord_current - ord_0
            found_valid_chars_after_start = True
        elif ord_letter_min[0] <= ord_current <= ord_letter_max[0]:
            result += ord_current - ord_letter_min[0] + 10
            found_valid_chars_after_start = True
        elif ord_letter_min[1] <= ord_current <= ord_letter_max[1]:
            result += ord_current - ord_letter_min[1] + 10
            found_valid_chars_after_start = True
        elif _is_ascii_space(ord_current):
            has_space_after_number = True
            start = pos + 1
            break
        else:
            raise Error(_atol_error(base, str_ref))
        if pos + 1 < str_len and not _is_ascii_space(buff[pos + 1]):
            var nextresult = result * real_base
            if nextresult < result:
                raise Error(
                    _atol_error(base, str_ref)
                    + " String expresses an integer too large to store in Int."
                )
            result = nextresult

    if was_last_digit_undescore or (not found_valid_chars_after_start):
        raise Error(_atol_error(base, str_ref))

    if has_space_after_number:
        for pos in range(start, str_len):
            if not _is_ascii_space(buff[pos]):
                raise Error(_atol_error(base, str_ref))
    if is_negative:
        result = -result
    return result


fn _atol_error(base: Int, str_ref: StringSlice[_]) -> String:
    return (
        "String is not convertible to integer with base "
        + str(base)
        + ": '"
        + str(str_ref)
        + "'"
    )


fn _identify_base(str_ref: StringSlice[_], start: Int) -> Tuple[Int, Int]:
    var length = str_ref.byte_length()
    # just 1 digit, assume base 10
    if start == (length - 1):
        return 10, start
    if str_ref[start] == "0":
        var second_digit = str_ref[start + 1]
        if second_digit == "b" or second_digit == "B":
            return 2, start + 2
        if second_digit == "o" or second_digit == "O":
            return 8, start + 2
        if second_digit == "x" or second_digit == "X":
            return 16, start + 2
        # checking for special case of all "0", "_" are also allowed
        var was_last_character_underscore = False
        for i in range(start + 1, length):
            if str_ref[i] == "_":
                if was_last_character_underscore:
                    return -1, -1
                else:
                    was_last_character_underscore = True
                    continue
            else:
                was_last_character_underscore = False
            if str_ref[i] != "0":
                return -1, -1
    elif ord("1") <= ord(str_ref[start]) <= ord("9"):
        return 10, start
    else:
        return -1, -1

    return 10, start


fn atol(str: String, base: Int = 10) raises -> Int:
    """Parses and returns the given string as an integer in the given base.

    For example, `atol("19")` returns `19`. If base is 0 the the string is
    parsed as an Integer literal, see: https://docs.python.org/3/reference/lexical_analysis.html#integers.

    Raises:
        If the given string cannot be parsed as an integer value. For example in
        `atol("hi")`.

    Args:
        str: A string to be parsed as an integer in the given base.
        base: Base used for conversion, value must be between 2 and 36, or 0.

    Returns:
        An integer value that represents the string, or otherwise raises.
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
    var str_len = str_ref_strip.byte_length()
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
# isupper
# ===----------------------------------------------------------------------=== #


@always_inline
fn isupper(c: UInt8) -> Bool:
    """Determines whether the given character is an ASCII uppercase character:
    `"ABCDEFGHIJKLMNOPQRSTUVWXYZ"`.

    Args:
        c: The character to check.

    Returns:
        True if the character is uppercase.
    """
    return _is_ascii_uppercase(c)


# ===----------------------------------------------------------------------=== #
# islower
# ===----------------------------------------------------------------------=== #


@always_inline
fn islower(c: UInt8) -> Bool:
    """Determines whether the given character is an ASCII lowercase character:
    `"abcdefghijklmnopqrstuvwxyz"`.

    Args:
        c: The character to check.

    Returns:
        True if the character is lowercase.
    """
    return _is_ascii_lowercase(c)


# ===----------------------------------------------------------------------=== #
# String
# ===----------------------------------------------------------------------=== #


@value
struct String(
    Sized,
    Stringable,
    Representable,
    IntableRaising,
    KeyElement,
    Comparable,
    Boolable,
    Writable,
    Writer,
    FloatableRaising,
    _HashableWithHasher,
    Stringlike,
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
        """Construct a string from a buffer of bytes.

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
        self.__copyinit__(other)

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

    fn write_bytes(inout self, bytes: Span[Byte, _]):
        """Write a byte span to this String.

        Args:
            bytes: The byte span to write to this String. Must NOT be
                null terminated.
        """
        self._iadd[False](bytes)

    fn write[*Ts: Writable](inout self, *args: *Ts):
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

    fn _iadd[has_null: Bool](inout self, other: Span[Byte]):
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
    fn __iadd__(inout self, other: String):
        """Appends another string to this string.

        Args:
            other: The string to append.
        """
        self._iadd[True](other.as_bytes())

    @always_inline
    fn __iadd__(inout self, other: StringLiteral):
        """Appends another string literal to this string.

        Args:
            other: The string to append.
        """
        self._iadd[False](other.as_bytes())

    @always_inline
    fn __iadd__(inout self, other: StringSlice):
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

        Returns:
            The string itself.

        Notes:
            This method ensures that you can pass a `String` to a method that
            takes a `Stringable` value.
        """
        return self

    @always_inline
    fn __repr__(self) -> String:
        """Return a representation of the string instance. You don't need to
        call this method directly, use `repr("...")` instead.

        Returns:
            A new representation of the string.
        """
        return _repr(self)

    @always_inline
    fn __ascii__(self) -> String:
        """Get the ASCII representation of the object. You don't need to call
        this method directly, use `ascii("...")` instead.

        Returns:
            A string containing the ASCII representation of the object.
        """
        return _ascii(self)

    fn __fspath__(self) -> String:
        """Return the file system path representation (just the string itself).

        Returns:
          The file system path representation as a string.
        """
        return self

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn write_to[W: Writer](self, inout writer: W):
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
        _ = is_first
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
            var e = elems.unsafe_get(i).as_bytes()
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
        """Returns a contiguous slice of bytes.

        Returns:
            A contiguous slice pointing to bytes.

        Notes:
            This does not include the trailing null terminator.
        """
        return Span[Byte, __origin_of(self)](
            ptr=self.unsafe_ptr(), length=self.byte_length()
        )

    @always_inline
    fn as_string_slice(
        ref self,
    ) -> StringSlice[_lit_mut_cast[__origin_of(self), False].result]:
        """Returns a string slice of the data owned by this string.

        Returns:
            A string slice pointing to the data owned by this string.
        """
        return StringSlice(self)

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

    fn _steal_ptr(inout self) -> UnsafePointer[UInt8]:
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
                if not str(s).isspace():  # TODO: with StringSlice.isspace()
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
                if str(s).isspace():  # TODO: with StringSlice.isspace()
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

    fn strip(self, chars: String) -> String:
        """Return a copy of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading or trailing characters.
        """

        return self.lstrip(chars).rstrip(chars)

    fn strip(self) -> String:
        """Return a copy of the string with leading and trailing whitespaces
        removed.

        Returns:
            A copy of the string with no leading or trailing whitespaces.
        """
        return self.lstrip().rstrip()

    fn rstrip(self, chars: String) -> String:
        """Return a copy of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no trailing characters.
        """

        var r_idx = self.byte_length()
        while r_idx > 0 and self[r_idx - 1] in chars:
            r_idx -= 1

        return self[:r_idx]

    fn rstrip(self) -> String:
        """Return a copy of the string with trailing whitespaces removed.

        Returns:
            A copy of the string with no trailing whitespaces.
        """
        var r_idx = self.byte_length()
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self.__reversed__():
        #     if not s.isspace():
        #         break
        #     r_idx -= 1
        while r_idx > 0 and _is_ascii_space(self._buffer.unsafe_get(r_idx - 1)):
            r_idx -= 1
        return self[:r_idx]

    fn lstrip(self, chars: String) -> String:
        """Return a copy of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A copy of the string with no leading characters.
        """

        var l_idx = 0
        while l_idx < self.byte_length() and self[l_idx] in chars:
            l_idx += 1

        return self[l_idx:]

    fn lstrip(self) -> String:
        """Return a copy of the string with leading whitespaces removed.

        Returns:
            A copy of the string with no leading whitespaces.
        """
        var l_idx = 0
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self:
        #     if not s.isspace():
        #         break
        #     l_idx += 1
        while l_idx < self.byte_length() and _is_ascii_space(
            self._buffer.unsafe_get(l_idx)
        ):
            l_idx += 1
        return self[l_idx:]

    fn __hash__(self) -> UInt:
        """Hash the underlying buffer using builtin hash.

        Returns:
            A 64-bit hash value. This value is _not_ suitable for cryptographic
            uses. Its intended usage is for data structures. See the `hash`
            builtin documentation for more details.
        """
        return hash(self.as_string_slice())

    fn __hash__[H: _Hasher](self, inout hasher: H):
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
        return isprintable(self.as_bytes())

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

    fn reserve(inout self, new_capacity: Int):
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
