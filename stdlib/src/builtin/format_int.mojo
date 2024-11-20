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

from os import abort
from collections import List, Optional, InlineArray
from collections.string import _calc_format_buffer_size
from utils import StringSlice, StaticString, Variant


alias _DEFAULT_DIGIT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


# ===----------------------------------------------------------------------===#
# bin
# ===----------------------------------------------------------------------===#


fn bin[
    WriterType: Writer, //
](value: Scalar, /, *, inout writer: WriterType, prefix: StaticString = "0b"):
    """Writes the binary string representation an integral value to a formatter.

    ```mojo
    print(bin(123))
    print(bin(-123))
    ```
    ```plaintext
    '0b1111011'
    '-0b1111011'
    ```

    Parameters:
        WriterType: The type of the `writer` argument.

    Args:
        value: An integral scalar value.
        writer: The formatter to write to.
        prefix: The prefix of the formatted int.
    """

    @parameter
    if value.type is DType.bool:
        _write_int(writer, value.cast[DType.int8](), 2, prefix=prefix)
    else:
        _write_int(writer, value, 2, prefix=prefix)


fn bin(value: Scalar, /, *, prefix: StaticString = "0b") -> String:
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
        value: An integral scalar value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """

    @parameter
    if value.type is DType.bool:
        return _format_int(value.cast[DType.int8](), 2, prefix=prefix)
    else:
        return _format_int(value, 2, prefix=prefix)


fn bin[
    T: Indexer, WriterType: Writer, //
](value: T, /, *, inout writer: WriterType, prefix: StaticString = "0b"):
    """Writes the binary representation of an indexer type to a formatter.

    Parameters:
        T: The Indexer type.
        WriterType: The type of the `writer` argument.

    Args:
        value: An indexer value.
        writer: The formatter to write to.
        prefix: The prefix of the formatted int.
    """
    bin(Scalar[DType.index](index(value)), writer=writer, prefix=prefix)


fn bin[T: Indexer, //](value: T, /, *, prefix: StaticString = "0b") -> String:
    """Returns the binary representation of an indexer type.

    Parameters:
        T: The Indexer type.

    Args:
        value: An indexer value.
        prefix: The prefix of the formatted int.

    Returns:
        The binary string representation of num.
    """
    return bin(Scalar[DType.index](index(value)), prefix=prefix)


# ===----------------------------------------------------------------------===#
# hex
# ===----------------------------------------------------------------------===#


fn hex[
    WriterType: Writer, //
](value: Scalar, /, *, inout writer: WriterType, prefix: StaticString = "0x"):
    """Writes the hex string representation of the given integer to a formatter.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The formatted string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        WriterType: The type of the `writer` argument.

    Args:
        value: The integer value to format.
        writer: The formatter to write to.
        prefix: The prefix of the formatted int.
    """

    @parameter
    if value.type is DType.bool:
        _write_int(writer, value.cast[DType.int8](), 16, prefix=prefix)
    else:
        _write_int(writer, value, 16, prefix=prefix)


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

    @parameter
    if value.type is DType.bool:
        return _format_int(value.cast[DType.int8](), 16, prefix=prefix)
    else:
        return _format_int(value, 16, prefix=prefix)


fn hex[
    T: Indexer, WriterType: Writer, //
](value: T, /, *, inout writer: WriterType, prefix: StaticString = "0x"):
    """Writes the hex string representation of the given integer to a formatter.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The formatted string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        T: The indexer type to represent in hexadecimal.
        WriterType: The type of the `writer` argument.

    Args:
        value: The integer value to format.
        writer: The formatter to write to.
        prefix: The prefix of the formatted int.
    """
    hex(Scalar[DType.index](index(value)), writer=writer, prefix=prefix)


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


# ===----------------------------------------------------------------------===#
# oct
# ===----------------------------------------------------------------------===#


fn oct[
    WriterType: Writer, //
](value: Scalar, /, *, inout writer: WriterType, prefix: StaticString = "0o"):
    """Writes the octal string representation of the given integer to a formatter.

    The octal representation is a base-8 encoding of the integer value.

    The formatted string will be prefixed with "0o" to indicate that the
    subsequent digits are octal.

    Parameters:
        WriterType: The type of the `writer` argument.

    Args:
        value: The integer value to format.
        writer: The formatter to write to.
        prefix: The prefix of the formatted int.
    """

    @parameter
    if value.type is DType.bool:
        _write_int(writer, value.cast[DType.int8](), 8, prefix=prefix)
    else:
        _write_int(writer, value, 8, prefix=prefix)


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

    @parameter
    if value.type is DType.bool:
        return _format_int(value.cast[DType.int8](), 8, prefix=prefix)
    else:
        return _format_int(value, 8, prefix=prefix)


fn oct[
    T: Indexer, WriterType: Writer, //
](value: T, /, *, inout writer: WriterType, prefix: StaticString = "0o"):
    """Writes the octal string representation of the given integer to a formatter.

    The octal representation is a base-8 encoding of the integer value.

    The formatted string will be prefixed with "0o" to indicate that the
    subsequent digits are octal.

    Parameters:
        T: The indexer type to represent in octal.
        WriterType: The type of the `writer` argument.

    Args:
        value: The integer value to format.
        writer: The formatter to write to.
        prefix: The prefix of the formatted int.
    """
    oct(Scalar[DType.index](index(value)), writer=writer, prefix=prefix)


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


# ===----------------------------------------------------------------------===#
# Integer formatting utilities
# ===----------------------------------------------------------------------===#


fn _format_int(
    value: Scalar,
    /,
    radix: Int = 10,
    *,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
    prefix: StaticString = "",
) -> String:
    var string = String(
        capacity=len(prefix) + _calc_format_buffer_size[value.type]()
    )
    _write_int(string, value, radix, digit_chars=digit_chars, prefix=prefix)
    return string^


fn _try_format_int(
    value: Scalar,
    radix: Int = 10,
    *,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
    prefix: StaticString = "",
) -> Variant[String, Error]:
    var string = String(
        capacity=len(prefix) + _calc_format_buffer_size[value.type]()
    )
    var err = _try_write_int(
        string, value, radix, digit_chars=digit_chars, prefix=prefix
    )
    if err:
        return err.value()
    return string^


fn _write_int[
    WriterType: Writer, //,
](
    inout writer: WriterType,
    value: Scalar,
    /,
    radix: Int = 10,
    *,
    digit_chars: StaticString = _DEFAULT_DIGIT_CHARS,
    prefix: StaticString = "",
):
    var err = _try_write_int(
        writer, value, radix, digit_chars=digit_chars, prefix=prefix
    )
    if err:
        abort("unexpected write int failure condition: " + str(err.value()))


fn _try_write_int[
    WriterType: Writer,
](
    inout writer: WriterType,
    value: Scalar,
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
    constrained[value.type.is_integral(), "Expected integral"]()

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
    fn process_digits[
        get_digit_value: fn () capturing [_] -> Scalar[value.type]
    ]():
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
        fn pos_digit_value() -> Scalar[value.type]:
            return remaining_int % radix

        process_digits[pos_digit_value]()
    else:

        @parameter
        fn neg_digit_value() -> Scalar[value.type]:
            return abs(remaining_int % -radix)

        process_digits[neg_digit_value]()

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
