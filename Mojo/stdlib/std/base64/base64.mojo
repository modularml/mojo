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
"""Provides functions for base64 encoding strings.

You can import these APIs from the `base64` package. For example:

```mojo
from std.base64 import b64encode
```
"""


from std.collections import Span

from ._b64encode import _b64encode

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


@always_inline
def _ascii_to_value(char: Byte) raises -> Byte:
    """Converts an ASCII character to its integer value for base64 decoding.

    Args:
        char: A single ascii byte.

    Returns:
        The integer value of the character for base64 decoding.

    Raises:
        If the character is outside the base64 alphabet.
    """
    comptime `A` = Byte(ord("A"))
    comptime `a` = Byte(ord("a"))
    comptime `Z` = Byte(ord("Z"))
    comptime `z` = Byte(ord("z"))
    comptime `0` = Byte(ord("0"))
    comptime `9` = Byte(ord("9"))
    comptime `=` = Byte(ord("="))
    comptime `+` = Byte(ord("+"))
    comptime `/` = Byte(ord("/"))

    # TODO: Measure perf against lookup table approach
    if char == `=`:
        return Byte(0)
    elif `A` <= char <= `Z`:
        return char - `A`
    elif `a` <= char <= `z`:
        return char - `a` + Byte(26)
    elif `0` <= char <= `9`:
        return char - `0` + Byte(52)
    elif char == `+`:
        return Byte(62)
    elif char == `/`:
        return Byte(63)
    else:
        raise Error(
            "ValueError: Unexpected character '",
            chr(Int(char)),
            "' encountered",
        )


# ===-----------------------------------------------------------------------===#
# b64encode
# ===-----------------------------------------------------------------------===#


@always_inline
def b64encode(input_bytes: ImmSpan[Byte, _], mut result: String):
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer.
        result: The string in which to store the values.

    Notes:
        This method reserves the necessary capacity. `result` can be a 0
        capacity string.
    """
    _b64encode(input_bytes, result)


@always_inline
def b64encode(input_string: StringSlice[mut=False, _]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_string: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    return b64encode(input_string.as_bytes())


@always_inline
def b64encode(input_bytes: ImmSpan[Byte, _]) -> String:
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer.

    Returns:
        The ASCII base64 encoded string.
    """
    var result = String()
    b64encode(input_bytes, result)
    return result^


# ===-----------------------------------------------------------------------===#
# b64decode
# ===-----------------------------------------------------------------------===#


@always_inline
def _is_ascii_whitespace(char: Byte) -> Bool:
    """Returns True if `char` is one of the six ASCII whitespace bytes.

    `Codepoint.is_posix_space()` is not usable here: it also accepts the file,
    group and record separators (0x1C-0x1E), so it would let those bytes
    through the base64 alphabet check.

    Args:
        char: A single byte.

    Returns:
        True if the byte is a space, tab, newline, vertical tab, form feed, or
        carriage return.
    """
    comptime ` ` = Byte(ord(" "))
    comptime `\t` = Byte(ord("\t"))
    comptime `\r` = Byte(ord("\r"))

    # \t, \n, \v, \f and \r occupy the contiguous range 0x09-0x0D.
    return char == ` ` or (char >= `\t` and char <= `\r`)


@always_inline
def _next_significant_byte(
    data: ImmSpan[Byte, _], pos: Int
) -> Tuple[Byte, Int]:
    """Returns the next non-whitespace byte in `data` and the position past it.

    The caller must guarantee that a non-whitespace byte remains at or after
    `pos`; this is not checked here.

    Args:
        data: The bytes to scan.
        pos: The position to start scanning from.

    Returns:
        The next non-whitespace byte, and the position one past that byte.
    """
    var i = pos
    while _is_ascii_whitespace(data[i]):
        i += 1
    return data[i], i + 1


def b64decode(str: StringSlice[mut=False, _]) raises -> List[Byte]:
    """Performs base64 decoding on the input string.

    Whitespace (spaces, tabs, newlines, carriage returns, form feeds, and
    vertical tabs) is ignored, which allows decoding base64 text that has
    been wrapped across multiple lines. Unlike Python's `base64.b64decode`,
    which discards *any* non-alphabet byte, Mojo only ignores whitespace and
    still rejects other invalid characters.

    Args:
        str: A base64 encoded string.

    Returns:
        The decoded bytes.

    Raises:
        If the input length (ignoring whitespace) is not divisible by 4, or
        the input contains a non-whitespace character outside the base64
        alphabet.
    """
    comptime `=` = Byte(ord("="))

    var input = str.as_bytes()

    var n = 0
    for char in input:
        if not _is_ascii_whitespace(char):
            n += 1

    if n % 4 != 0:
        raise Error(
            t"ValueError: Input length '{n}' (ignoring whitespace) must be"
            t" divisible by 4"
        )

    var result = List[Byte](capacity=(n // 4) * 3)

    # This algorithm is based on https://arxiv.org/abs/1704.00605. It reads
    # directly from `input`, skipping over whitespace as it goes, so it never
    # allocates or copies a whitespace-free version of the input.
    var pos = 0
    for _ in range(0, n, 4):
        var a_byte, a_pos = _next_significant_byte(input, pos)
        var b_byte, b_pos = _next_significant_byte(input, a_pos)
        var c_byte, c_pos = _next_significant_byte(input, b_pos)
        var d_byte, d_pos = _next_significant_byte(input, c_pos)
        pos = d_pos

        var a = _ascii_to_value(a_byte)
        var b = _ascii_to_value(b_byte)
        var c = _ascii_to_value(c_byte)
        var d = _ascii_to_value(d_byte)

        result.append((a << 2) | (b >> 4))
        if c_byte == `=`:
            break
        result.append(((b & 0x0F) << 4) | (c >> 2))
        if d_byte == `=`:
            break
        result.append(((c & 0x03) << 6) | d)

    return result^


# ===-----------------------------------------------------------------------===#
# b16encode
# ===-----------------------------------------------------------------------===#


def b16encode(str: StringSlice[mut=False, _]) -> String:
    """Performs base16 encoding on the input string slice.

    Args:
        str: The input string slice.

    Returns:
        Base16 encoding of the input string.
    """
    comptime lookup = "0123456789ABCDEF"
    var b16chars = lookup.ptr()

    var data = str.as_bytes()
    var length = str.byte_length()
    var result = String(capacity_bytes=length * 2)

    for i in range(length):
        var str_byte = data[i]
        var hi = str_byte >> 4
        var lo = str_byte & 0b1111
        result._unsafe_append_byte(b16chars[unsafe_offset=hi])
        result._unsafe_append_byte(b16chars[unsafe_offset=lo])

    return result^


# ===-----------------------------------------------------------------------===#
# b16decode
# ===-----------------------------------------------------------------------===#


def b16decode(str: StringSlice[mut=False, _]) raises -> List[Byte]:
    """Performs base16 decoding on the input string.

    Args:
        str: A base16 encoded string.

    Returns:
        The decoded bytes.

    Raises:
        If the input length is odd or any character is outside the base16
        alphabet `[0-9A-F]` (per RFC 4648 section 8, which is strictly
        uppercase).
    """

    comptime `A` = Byte(ord("A"))
    comptime `F` = Byte(ord("F"))
    comptime `0` = Byte(ord("0"))
    comptime `9` = Byte(ord("9"))

    # TODO: Measure perf against lookup table approach
    @always_inline
    def decode(c: Byte) raises -> Byte:
        if `0` <= c <= `9`:
            return c - `0`
        elif `A` <= c <= `F`:
            return c - `A` + Byte(10)
        else:
            raise Error(
                "ValueError: Unexpected character '",
                chr(Int(c)),
                "' encountered",
            )

    var data = str.as_bytes()
    var n = str.byte_length()
    if n % 2 != 0:
        raise Error("ValueError: Input length '", n, "' must be divisible by 2")

    var result = List[Byte](capacity=n // 2)

    for i in range(0, n, 2):
        var hi = data[i]
        var lo = data[i + 1]
        result.append(decode(hi) << 4 | decode(lo))

    return result^
