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
"""Provides functions for hex encoding/decoding.

You can import these APIs from the `hex` package. For example:

```mojo
from std.hex import hex_encode, hex_decode
```
"""

comptime _HEX_CHARS_LOWER: StaticString = "0123456789abcdef"
comptime _HEX_CHARS_UPPER: StaticString = "0123456789ABCDEF"


@always_inline
def hex_encode[
    *, is_lowercase: Bool = True
](input_bytes: Span[mut=False, Byte, _], mut result: String):
    """Performs hex encoding on the input bytes.

    Parameters:
        is_lowercase: Whether to emit lowercase (`a`-`f`) hex digits, or
            uppercase (`A`-`F`) if `False`.

    Args:
        input_bytes: The input bytes buffer.
        result: The string in which to store the values.

    Notes:
        This method reserves the necessary capacity. `result` can be a 0
        capacity string.
    """

    comptime hex_chars_str: StaticString = (
        _HEX_CHARS_LOWER if is_lowercase else _HEX_CHARS_UPPER
    )

    result.resize(len(input_bytes) * 2)

    var dest = result.unsafe_ptr_mut()
    var hex_chars = hex_chars_str.unsafe_ptr()
    for i in range(len(input_bytes)):
        var b = Int(input_bytes[i])
        dest[unsafe_offset=2 * i] = hex_chars[unsafe_offset=b >> 4]
        dest[unsafe_offset=2 * i + 1] = hex_chars[unsafe_offset=b & 0xF]


@always_inline
def hex_encode[
    *, is_lowercase: Bool = True
](input_bytes: Span[mut=False, Byte, _]) -> String:
    """Performs hex encoding on the input bytes.

    Parameters:
        is_lowercase: Whether to emit lowercase (`a`-`f`) hex digits, or
            uppercase (`A`-`F`) if `False`.

    Args:
        input_bytes: The input bytes buffer.

    Returns:
        The ASCII hex encoded string.
    """
    var result = String()
    hex_encode[is_lowercase=is_lowercase](input_bytes, result)
    return result^


@always_inline
def hex_decode[
    *, is_lowercase: Bool = True
](str: StringSlice[mut=False, _]) raises -> List[Byte]:
    """Performs hex decoding on the input string.

    Parameters:
        is_lowercase: Whether `str` uses lowercase (`a`-`f`) hex digits,
            or uppercase (`A`-`F`) if `False`; raises if it uses the other
            case.

    Args:
        str: A hex encoded string.

    Returns:
        The decoded bytes.

    Raises:
        If the operation fails.
    """

    var result = List[Byte](length=str.byte_length() // 2, fill=0)
    hex_decode[is_lowercase=is_lowercase](str, result)
    return result^


@always_inline
def hex_decode[
    length: Int, *, is_lowercase: Bool = True
](str: StringSlice[mut=False, _]) raises -> Array[Byte, length]:
    """Performs hex decoding on the input string.

    Parameters:
        length: Resulting fixed-sized array length.
        is_lowercase: Whether `str` uses lowercase (`a`-`f`) hex digits,
            or uppercase (`A`-`F`) if `False`; raises if it uses the other
            case.

    Args:
        str: A hex encoded string.

    Returns:
        The decoded bytes.

    Raises:
        If the operation fails.
    """

    var result = Array[Byte, length](uninitialized=True)
    hex_decode[is_lowercase=is_lowercase](str, result)
    return result^


@always_inline
def hex_decode[
    *, is_lowercase: Bool = True
](str: StringSlice[mut=False, _], result: Span[mut=True, Byte, _]) raises:
    """Performs hex decoding on the input string.

    Parameters:
        is_lowercase: Whether `str` uses lowercase (`a`-`f`) hex digits,
            or uppercase (`A`-`F`) if `False`; raises if it uses the other
            case.

    Args:
        str: A hex encoded string.
        result: The bytes in which to store the values.

    Raises:
        If the operation fails.
    """

    if str.byte_length() != len(result) * 2:
        raise Error(
            "ValueError:: Expected hex string of length {}; got {}".format(
                len(result) * 2, str.byte_length()
            )
        )

    var ptr = str.unsafe_ptr()
    for i in range(len(result)):
        var hi = ptr[unsafe_offset=2 * i]
        var lo = ptr[unsafe_offset=2 * i + 1]
        result[i] = _decode_hex_byte[is_lowercase](hi, lo, 2 * i)


@always_inline
def _decode_hex_byte[
    is_lowercase: Bool
](hi: Byte, lo: Byte, pos: Int) raises -> Byte:
    return (_nibble[is_lowercase](hi, pos) << 4) | _nibble[is_lowercase](
        lo, pos + 1
    )


@always_inline
def _nibble[is_lowercase: Bool](c: Byte, pos: Int) raises -> Byte:
    if c >= 48 and c <= 57:
        return c - 48
    comptime if is_lowercase:
        if c >= 97 and c <= 102:
            return c - 87
    else:
        if c >= 65 and c <= 70:
            return c - 55
    raise Error("ValueError: Invalid hex character at position {}".format(pos))
