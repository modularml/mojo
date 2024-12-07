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

"""Provides the `hex` and `bin` functions.

These are Mojo built-ins, so you don't need to import them.
"""

from collections import InlineArray, List, Optional
from os import abort

from utils import StaticString, StringSlice

alias _DEFAULT_DIGIT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


# ===-----------------------------------------------------------------------===#
# bin
# ===-----------------------------------------------------------------------===#


fn bin(num: Scalar, /, *, prefix: StaticString = "0b") -> String:
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
    return _try_format_int(num, 2, prefix=prefix)


# Need this until we have constraints to stop the compiler from matching this
# directly to bin[type: DType](num: Scalar[type]).
fn bin(b: Scalar[DType.bool], /, *, prefix: StaticString = "0b") -> String:
    """Returns the binary representation of a scalar bool.

    Args:
        b: A scalar bool value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of b.
    """
    return bin(b.cast[DType.int8](), prefix=prefix)


fn bin[T: Indexer, //](num: T, /, *, prefix: StaticString = "0b") -> String:
    """Returns the binary representation of an indexer type.

    Parameters:
        T: The Indexer type.

    Args:
        num: An indexer value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """
    return bin(Scalar[DType.index](index(num)), prefix=prefix)


# ===-----------------------------------------------------------------------===#
# hex
# ===-----------------------------------------------------------------------===#


fn hex(value: Scalar, /, *, prefix: StaticString = "0x") -> String:
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
    return _try_format_int(value, 16, prefix=prefix)


fn hex[T: Indexer, //](value: T, /, *, prefix: StaticString = "0x") -> String:
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
    return hex(Scalar[DType.index](index(value)), prefix=prefix)


fn hex(value: Scalar[DType.bool], /, *, prefix: StaticString = "0x") -> String:
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
    return hex(value.cast[DType.int8](), prefix=prefix)


# ===-----------------------------------------------------------------------===#
# oct
# ===-----------------------------------------------------------------------===#


fn oct(value: Scalar, /, *, prefix: StaticString = "0o") -> String:
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
    return _try_format_int(value, 8, prefix=prefix)


fn oct[T: Indexer, //](value: T, /, *, prefix: StaticString = "0o") -> String:
    """Returns the octal string representation of the given integer.

    The octal representation is a base-8 encoding of the integer value.

    The returned string will be prefixed with "0o" to indicate that the
    subsequent digits are octal.

    Parameters:
        T: The indexer type to represent in octal.

    Args:
        value: The integer value to format.
        prefix: The prefix of the formatted int.

    Returns:
        A string containing the octal representation of the given integer.
    """
    return oct(Scalar[DType.index](index(value)), prefix=prefix)


fn oct(value: Scalar[DType.bool], /, *, prefix: StaticString = "0o") -> String:
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
    return oct(value.cast[DType.int8](), prefix=prefix)


# ===-----------------------------------------------------------------------===#
# Integer formatting utilities
# ===-----------------------------------------------------------------------===#


fn _try_format_int(
    value: Scalar[_],
    /,
    radix: Int = 10,
    *,
    prefix: StaticString = "",
) -> String:
    try:
        return _format_int(value, radix, prefix=prefix)
    except e:
        # This should not be reachable as _format_int only throws if we pass
        # incompatible radix and custom digit chars, which we aren't doing
        # above.
        return abort[String](
            "unexpected exception formatting value as hexadecimal: " + str(e)
        )


fn _format_int[
    dtype: DType
](
    value: Scalar[dtype],
    radix: Int = 10,
    *,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
    prefix: StaticString = "",
) raises -> String:
    var output = String()

    _write_int(output, value, radix, digit_chars=digit_chars, prefix=prefix)

    return output^


fn _write_int[
    type: DType,
    W: Writer,
](
    mut writer: W,
    value: Scalar[type],
    /,
    radix: Int = 10,
    *,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
    prefix: StaticString = "",
) raises:
    var err = _try_write_int(
        writer, value, radix, digit_chars=digit_chars, prefix=prefix
    )
    if err:
        raise err.value()


fn _try_write_int[
    type: DType,
    W: Writer,
](
    mut writer: W,
    value: Scalar[type],
    /,
    radix: Int = 10,
    *,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
    prefix: StaticString = "",
) -> Optional[Error]:
    """Writes a formatted string representation of the given integer using the
    specified radix.

    The maximum supported radix is 36 unless a custom `digit_chars` mapping is
    provided.
    """
    constrained[type.is_integral(), "Expected integral"]()

    # Check that the radix and available digit characters are valid
    if radix < 2:
        return Error("Unable to format integer to string with radix < 2")

    if radix > digit_chars.byte_length():
        return Error(
            "Unable to format integer to string when provided radix is larger "
            "than length of available digit value characters"
        )

    if not digit_chars.byte_length() >= 2:
        return Error(
            "Unable to format integer to string when provided digit_chars"
            " mapping len is not >= 2"
        )

    # Process the integer value into its corresponding digits

    # TODO(#26444, Unicode support): Get an array of Character, not bytes.
    var digit_chars_array = digit_chars.unsafe_ptr()

    # Prefix a '-' if the original int was negative and make positive.
    if value < 0:
        writer.write("-")

    # Add the custom number prefix, e.g. "0x" commonly used for hex numbers.
    # This comes *after* the minus sign, if present.
    writer.write(prefix)

    if value == 0:
        # TODO: Replace with safe digit_chars[:1] syntax.
        # SAFETY:
        #   This static origin is valid as long as we're using a
        #   `StringLiteral` for `digit_chars`.
        var zero_char = digit_chars_array[0]

        # Construct a null-terminated buffer of single-byte char.
        var zero_buf = InlineArray[UInt8, 2](zero_char, 0)

        # TODO(MSTDL-720):
        #   Support printing non-null-terminated strings on GPU and switch
        #   back to this code without a workaround.
        # ptr=digit_chars_array,
        var zero = StringSlice[ImmutableAnyOrigin](
            ptr=zero_buf.unsafe_ptr(), length=1
        )
        writer.write(zero)

        return None

    # Create a buffer to store the formatted value

    # Stack allocate enough bytes to store any formatted 64-bit integer
    # TODO: use a dynamic size when #2194 is resolved
    alias CAPACITY: Int = 64 + 1  # +1 for storing NUL terminator.

    var buf = InlineArray[UInt8, CAPACITY](unsafe_uninitialized=True)

    # Start the buf pointer at the end. We will write the least-significant
    # digits later in the buffer, and then decrement the pointer to move
    # earlier in the buffer as we write the more-significant digits.
    var offset = CAPACITY - 1

    buf.unsafe_ptr().offset(offset).init_pointee_copy(
        0
    )  # Write NUL terminator at the end

    # Position the offset to write the least-significant digit just before the
    # NUL terminator.
    offset -= 1

    # Write the digits of the number
    var remaining_int = value

    @parameter
    fn process_digits[get_digit_value: fn () capturing [_] -> Scalar[type]]():
        while remaining_int:
            var digit_value = get_digit_value()

            # Write the char representing the value of the least significant
            # digit.
            buf.unsafe_ptr().offset(offset).init_pointee_copy(
                digit_chars_array[int(digit_value)]
            )

            # Position the offset to write the next digit.
            offset -= 1

            # Drop the least significant digit
            remaining_int /= radix

    if remaining_int >= 0:

        @parameter
        fn pos_digit_value() -> Scalar[type]:
            return remaining_int % radix

        process_digits[pos_digit_value]()
    else:

        @parameter
        fn neg_digit_value() -> Scalar[type]:
            return abs(remaining_int % -radix)

        process_digits[neg_digit_value]()

    _ = remaining_int
    _ = digit_chars_array

    # Re-add +1 byte since the loop ended so we didn't write another char.
    offset += 1

    var buf_ptr = buf.unsafe_ptr() + offset

    # Calculate the length of the buffer we've filled. This is the number of
    # bytes from our final `buf_ptr` to the end of the buffer.
    var len = (CAPACITY - offset) - 1  # -1 because NUL terminator

    # SAFETY:
    #   Create a slice to only those bytes in `buf` that have been initialized.
    var str_slice = StringSlice[__origin_of(buf)](ptr=buf_ptr, length=len)

    writer.write(str_slice)

    return None
