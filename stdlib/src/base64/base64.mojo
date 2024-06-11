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


fn b64encode(str: String) -> String:
    """Performs base64 encoding on the input string.

    Args:
      str: The input string.

    Returns:
      Base64 encoding of the input string.
    """
    alias lookup = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
    var b64chars = lookup.unsafe_ptr()

    var length = len(str)
    var out = String._buffer_type(capacity=length + 1)

    @parameter
    @always_inline
    fn s(idx: Int) -> Int:
        # TODO: Remove cast once transition to UInt8 string types is complete.
        return int(str.unsafe_ptr().bitcast[UInt8]()[idx])

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    var end = length - (length % 3)
    for i in range(0, end, 3):
        var si = s(i)
        var si_1 = s(i + 1)
        var si_2 = s(i + 2)
        out.append(b64chars[si // 4])
        out.append(b64chars[((si * 16) % 64) + si_1 // 16])
        out.append(b64chars[((si_1 * 4) % 64) + si_2 // 64])
        out.append(b64chars[si_2 % 64])

    if end < length:
        var si = s(end)
        out.append(b64chars[si // 4])
        if end == length - 1:
            out.append(b64chars[(si * 16) % 64])
            out.append(ord("="))
        elif end == length - 2:
            var si_1 = s(end + 1)
            out.append(b64chars[((si * 16) % 64) + si_1 // 16])
            out.append(b64chars[(si_1 * 4) % 64])
        out.append(ord("="))
    out.append(0)
    return String(out^)


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
    var n = len(str)
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

    var length = len(str)
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

    var n = len(str)
    debug_assert(n % 2 == 0, "Input length must be divisible by 2")

    var p = List[UInt8](capacity=int(n / 2) + 1)

    for i in range(0, n, 2):
        var hi = str[i]
        var lo = str[i + 1]
        p.append(decode(hi) << 4 | decode(lo))

    p.append(0)
    return p
