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

"""Provides the `hex` and `bin` functions.

These are Mojo built-ins, so you don't need to import them.
"""

from std.os import abort
from std.sys import bit_width_of

comptime _DEFAULT_DIGIT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


# ===-----------------------------------------------------------------------===#
# bin
# ===-----------------------------------------------------------------------===#


def bin(num: Scalar, /, *, prefix: StaticString = "0b") -> String:
    """Return the binary string representation an integral value.

    ```mojo
    print(bin(123))
    print(bin(-123))
    ```
    ```plaintext
    '0b1111011'
    '-0b1111011'
    ```

    Args:
        num: An integral scalar value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """
    return _format_int[radix=2](num, prefix=prefix)


# Need this until we have constraints to stop the compiler from matching this
# directly to bin[dtype: DType](num: Scalar[dtype]).
def bin(b: Scalar[.bool], /, *, prefix: StaticString = "0b") -> String:
    """Returns the binary representation of a scalar bool.

    Args:
        b: A scalar bool value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of b.
    """
    return bin(b.cast[.int8](), prefix=prefix)


def bin[T: Intable, //](num: T, /, *, prefix: StaticString = "0b") -> String:
    """Returns the binary representation of an indexer type.

    Parameters:
        T: The Intable type.

    Args:
        num: An indexer value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """
    return bin(Int(Int(num)), prefix=prefix)


# ===-----------------------------------------------------------------------===#
# hex
# ===-----------------------------------------------------------------------===#


def hex(value: Scalar, /, *, prefix: StaticString = "0x") -> String:
    """Returns the hex string representation of the given integer.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the hex representation of the given integer.
    """
    return _format_int[radix=16](value, prefix=prefix)


def hex[T: Intable, //](value: T, /, *, prefix: StaticString = "0x") -> String:
    """Returns the hex string representation of the given integer.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        T: The indexer type to represent in hexadecimal.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the hex representation of the given integer.
    """
    return hex(Int(Int(value)), prefix=prefix)


def hex(value: Scalar[.bool], /, *, prefix: StaticString = "0x") -> String:
    """Returns the hex string representation of the given scalar bool.

    The hexadecimal representation is a base-16 encoding of the bool.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Args:
        value: The bool value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the hex representation of the given bool.
    """
    return hex(value.cast[.int8](), prefix=prefix)


# ===-----------------------------------------------------------------------===#
# oct
# ===-----------------------------------------------------------------------===#


def oct(value: Scalar, /, *, prefix: StaticString = "0o") -> String:
    """Returns the octal string representation of the given integer.

    The octal representation is a base-8 encoding of the integer value.

    The returned string will be prefixed with "0o" to indicate that the
    subsequent digits are octal.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the octal representation of the given integer.
    """
    return _format_int[radix=8](value, prefix=prefix)


def oct[T: Intable, //](value: T, /, *, prefix: StaticString = "0o") -> String:
    """Returns the octal string representation of the given integer.

    The octal representation is a base-8 encoding of the integer value.

    The returned string will be prefixed with "0o" to indicate that the
    subsequent digits are octal.

    Parameters:
        T: The intable type to represent in octal.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the octal representation of the given integer.
    """
    return oct(Int(Int(value)), prefix=prefix)


def oct(value: Scalar[.bool], /, *, prefix: StaticString = "0o") -> String:
    """Returns the octal string representation of the given scalar bool.

    The octal representation is a base-8 encoding of the bool.

    The returned string will be prefixed with "0o" to indicate that the
    subsequent digits are octal.

    Args:
        value: The bool value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the octal representation of the given bool.
    """
    return oct(value.cast[.int8](), prefix=prefix)


# ===-----------------------------------------------------------------------===#
# Integer formatting utilities
# ===-----------------------------------------------------------------------===#


def _format_int[
    dtype: DType,
    *,
    radix: Int = 10,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
](value: Scalar[dtype], *, prefix: StaticString = "",) -> String:
    var output = String()
    _write_int[radix=radix, digit_chars=digit_chars](
        output, value, prefix=prefix
    )
    return output^


def _write_int[
    dtype: DType,
    W: Writer,
    //,
    *,
    radix: Int = 10,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
](mut writer: W, value: Scalar[dtype], *, prefix: StaticString = "",):
    """Writes a formatted string representation of the given integer using the
    specified radix.

    The maximum supported radix is 36 unless a custom `digit_chars` mapping is
    provided.
    """
    comptime assert dtype.is_integral(), "Expected integral"
    comptime assert (
        radix >= 2
    ), "Unable to format integer to string with radix < 2"
    comptime assert radix <= digit_chars.byte_length(), (
        "Unable to format integer to string when provided radix is larger than"
        " length of available digit value characters"
    )
    comptime assert digit_chars.byte_length() >= 2, (
        "Unable to format integer to string when provided digit_chars mapping"
        " len is not >= 2"
    )

    # Process the integer value into its corresponding digits

    # TODO(#26444, Unicode support): Get an array of Character, not bytes.
    var digit_chars_array = digit_chars.as_bytes()

    # Prefix a '-' if the original int was negative and make positive.
    if value < 0:
        writer.write_string("-")

    # Add the custom number prefix, e.g. "0x" commonly used for hex numbers.
    # This comes *after* the minus sign, if present.
    writer.write(prefix)

    if value == 0:
        # TODO: Replace with safe digit_chars[:1] syntax.
        # SAFETY:
        #   This static origin is valid as long as we're using a
        #   `StaticString` for `digit_chars`.
        var zero_char = digit_chars_array.unsafe_ptr()[]

        # Construct a null-terminated buffer of single-byte char.
        var zero_buf: Array[UInt8, 2] = [zero_char, 0]

        # TODO(MSTDL-720):
        #   Support printing non-null-terminated strings on GPU and switch
        #   back to this code without a workaround.
        # ptr=digit_chars_array,
        writer.write(
            StringSlice(
                unsafe_from_utf8=Span(
                    unsafe_ptr=zero_buf.unsafe_ptr(), length=1
                )
            )
        )

        return

    # Create a buffer to store the formatted value

    # Stack allocate enough bytes to store any formatted integer up to 256 bits
    # TODO: use a dynamic size when #2194 is resolved
    # +1 for storing NUL terminator.
    comptime CAPACITY: Int = max(64, bit_width_of[dtype]()) + 1

    var buf = Array[UInt8, CAPACITY](uninitialized=True)

    # Start the buf pointer at the end. We will write the least-significant
    # digits later in the buffer, and then decrement the pointer to move
    # earlier in the buffer as we write the more-significant digits.
    var offset = CAPACITY - 1

    buf.unsafe_ptr().unsafe_offset(offset).write(
        0
    )  # Write NUL terminator at the end

    # Position the offset to write the least-significant digit just before the
    # NUL terminator.
    offset -= 1

    # Write the digits of the number
    var remaining_int = value

    def process_digits[
        get_digit_value: def(Scalar[dtype]) thin -> Scalar[dtype],
        div_fn: def(Scalar[dtype]) thin -> Scalar[dtype],
    ]() {mut}:
        while remaining_int:
            var digit_value = get_digit_value(remaining_int)

            # Write the char representing the value of the least significant
            # digit.
            buf.unsafe_ptr().unsafe_offset(offset).write(
                digit_chars_array.unsafe_ptr()[unsafe_offset=Int(digit_value)]
            )

            # Position the offset to write the next digit.
            offset -= 1

            # TODO: (MOCO-3028)
            # We should be able to simply do `remaining_int /= radix` here,
            # however, there is a compiler bug when using `/` in parameter
            # vs argument for signed SIMD values.
            #
            # Drop the least significant digit
            remaining_int = div_fn(remaining_int)

    if remaining_int >= 0:

        def pos_digit_value(value: Scalar[dtype]) -> Scalar[dtype]:
            return value % Scalar[dtype](radix)

        def floor_div(value: Scalar[dtype]) -> Scalar[dtype]:
            return value / Scalar[dtype](radix)

        process_digits[pos_digit_value, floor_div]()
    else:

        def neg_digit_value(value: Scalar[dtype]) -> Scalar[dtype]:
            return abs(value % Scalar[dtype](-radix))

        def ceil_div(value: Scalar[dtype]) -> Scalar[dtype]:
            return value.__ceildiv__(Scalar[dtype](radix))

        process_digits[neg_digit_value, ceil_div]()

    # Re-add +1 byte since the loop ended so we didn't write another char.
    offset += 1

    # Create a span to only those bytes in `buf` that have been initialized.
    # -1 because NUL terminator
    writer.write_string(
        StringSlice(
            unsafe_from_utf8=Span(
                unsafe_ptr=buf.unsafe_ptr().unsafe_offset(offset),
                length=len(buf) - offset - 1,
            )
        )
    )
