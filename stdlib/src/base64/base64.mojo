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

from collections import List, Optional
from sys import simdwidthof

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _ascii_to_value(char: String, specials: String) -> Int:
    """Converts an ASCII character to its integer value for base64 decoding.

    Args:
        char: A single character string.
        specials: A length-2 string representing the non-alphanumeric characters used
                  for encoded and decoding, "+/" for the default base64 alphabet.

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
    elif char == specials[0]:  # default +
        return 62
    elif char == specials[1]:  # default /
        return 63
    else:
        return -1


@always_inline
fn _validate_altchars(altchars: Optional[String] = None) -> String:
    if altchars is not None:
        var ac = altchars.value()
        debug_assert(
            len(ac) == 2, "altchars should have exactly two ASCII characters"
        )
        return ac[0] + ac[1]
    else:
        return "+/"


@always_inline
fn _remove_whitespace(input: String) -> String:
    alias whitespace = " \t\n\r"
    var output = String()
    for i in range(len(input)):
        var c = input[i]
        if c not in whitespace:
            output += c

    return output


# ===----------------------------------------------------------------------===#
# b64encode
# ===----------------------------------------------------------------------===#


fn b64encode(str: String, altchars: Optional[String] = None) -> String:
    """Performs base64 encoding on the input string.

    Args:
      str: The input string.
      altchars: Optional string of length 2 which specifies the alternative alphabet
                used instead of the + and / characters.

    Returns:
      Base64 encoding of the input string.
    """
    var lookup = String(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    lookup += _validate_altchars(altchars)
    var b64chars = lookup._buffer  # TODO: Leaky abstraction!

    var length = str.byte_length()
    var out = String._buffer_type(capacity=length + 1)

    @parameter
    @always_inline
    fn s(idx: Int) -> Int:
        return int(str.unsafe_ptr()[idx])

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
fn b64decode(
    s: String, altchars: Optional[String] = None, validate: Bool = False
) raises -> String:
    """Performs base64 decoding on the input string.

    Args:
      s: A base64 encoded string.
      altchars: Optional string of length 2 which specifies the alternative alphabet
                used instead of the + and / characters.
      validate: If `False` (the default), characters that are neither in the normal
                base-64 alphabet nor the alternative alphabet are discarded prior to
                the padding check. If validate is True, these non-alphabet characters
                in the input will cause an Error to be raised.

    Returns:
      The decoded string.
    """

    # Base64 alphabet according to RFC 4648
    var base64_alphabet = String(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    )
    var specials = _validate_altchars(altchars)
    base64_alphabet += specials

    # Step 1: Remove ASCII whitespace from data
    var encoded = _remove_whitespace(s)

    # Step 3: Check for invalid characters
    var valid_chars = base64_alphabet + "="
    for char in encoded:
        if char not in valid_chars:
            if validate:
                raise Error("Invalid character found: " + str(char))
            else:
                encoded = encoded.replace(char, "")

    # Step 2: Validate padding and length
    var length = encoded.byte_length()
    print("byte length =", length)
    if length % 4 != 0:
        raise Error(
            "After pruning whitespace and invalid chars, length of input "
            + str(length)
            + " is not divisible by 4."
        )

    var decoded = String._buffer_type(capacity=length + 1)

    # This algorithm is based on https://arxiv.org/abs/1704.00605
    for i in range(0, length, 4):
        var a = _ascii_to_value(encoded[i], specials)
        var b = _ascii_to_value(encoded[i + 1], specials)
        var c = _ascii_to_value(encoded[i + 2], specials)
        var d = _ascii_to_value(encoded[i + 3], specials)

        debug_assert(
            a >= 0 and b >= 0 and c >= 0 and d >= 0,
            "Unexpected character encountered",
        )

        decoded.append((a << 2) | (b >> 4))
        if encoded[i + 2] == "=":
            break

        decoded.append(((b & 0x0F) << 4) | (c >> 2))

        if encoded[i + 3] == "=":
            break

        decoded.append(((c & 0x03) << 6) | d)

    decoded.append(0)
    return decoded


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
