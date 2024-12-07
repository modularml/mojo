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

from ._b64encode import b64encode_with_buffers as _b64encode_with_buffers

# ===-----------------------------------------------------------------------===#
# Utilities
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# b64encode
# ===-----------------------------------------------------------------------===#


# TODO: Use Span instead of List as input when Span is easier to use
fn b64encode(input_bytes: List[UInt8, _], mut result: List[UInt8, _]):
    """Performs base64 encoding on the input string.

    Args:
        input_bytes: The input string buffer. Assumed to be null-terminated.
        result: The buffer in which to store the values.
    """
    _b64encode_with_buffers(input_bytes, result)


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


# ===-----------------------------------------------------------------------===#
# b64decode
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# b16encode
# ===-----------------------------------------------------------------------===#


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


# ===-----------------------------------------------------------------------===#
# b16decode
# ===-----------------------------------------------------------------------===#


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
