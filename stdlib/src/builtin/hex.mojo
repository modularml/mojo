# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

"""Provides the `hex` function.

These are Mojo built-ins, so you don't need to import them.
"""

from collections.vector import DynamicVector
from math import abs as _abs

alias _DEFAULT_DIGIT_CHARS = "0123456789abcdefghijklmnopqrstuvwxyz"


fn hex[T: Intable](value: T) -> String:
    """Returns the hex string represention of the given integer.

    The hexadecimal representation is a base-16 encoding of the integer value.

    The returned string will be prefixed with "0x" to indicate that the
    subsequent digits are hex.

    Parameters:
        T: The intable type to represent in hexadecimal.

    Args:
        value: The integer value to format.

    Returns:
        A string containing the hex representation of the given integer.
    """

    try:
        return _format_int(int(value), 16, prefix="0x")
    except e:
        # This should not be reachable as _format_int only throws if we pass
        # incompatible radix and custom digit chars, which we aren't doing
        # above.
        trap("unexpected exception formatting value as hexadecimal: " + str(e))
        return ""


# ===----------------------------------------------------------------------===#
# Integer formatting utilities
# ===----------------------------------------------------------------------===#


fn _format_int[
    T: Intable
](
    value: T,
    radix: Int = 10,
    digit_chars: String = _DEFAULT_DIGIT_CHARS,
    prefix: String = "",
) raises -> String:
    var buf = DynamicVector[Int8]()

    _write_int(buf, value, radix, digit_chars, prefix)

    return String._unchecked_from_bytes(buf ^)


fn _write_int[
    T: Intable
](
    inout fmt: DynamicVector[Int8],
    value0: T,
    radix: Int = 10,
    digit_chars: String = _DEFAULT_DIGIT_CHARS,
    prefix: String = "",
) raises:
    """Writes a formatted string representation of the given integer using the specified radix.

    The maximum supported radix is 36 unless a custom `digit_chars` mapping is
    provided.
    """

    #
    # Check that the radix and available digit characters are valid
    #

    if radix < 2:
        raise Error("Unable to format integer to string with radix < 2")

    if radix > len(digit_chars):
        raise Error(
            "Unable to format integer to string when provided radix is larger "
            "than length of available digit value characters"
        )

    if not len(digit_chars) >= 2:
        raise Error(
            "Unable to format integer to string when provided digit_chars"
            " mapping len is not >= 2"
        )

    #
    # Process the integer value into its corresponding digits
    #

    # NOTE:
    #   Use Int64 instead of Int here so that the % and / operators use C-like
    #   behavior of returning remainder and doing truncating division.
    var value: Int64 = Int64(int(value0))

    # TODO(#26444, Unicode support): Get an array of Character, not bytes.
    var digit_chars_array: DynamicVector[Int8] = digit_chars.as_bytes()

    # Prefix a '-' if the original int was negative.
    if value < 0:
        alias minus: Int8 = ord("-")
        fmt.push_back(minus)

    # Add the custom number prefix, e.g. "0x" commonly used for hex numbers.
    # This comes *after* the minus sign, if present.
    fmt.extend(prefix.as_bytes())

    if value == 0:
        fmt.push_back(digit_chars_array[0])
        return

    var first_digit_pos = len(fmt)

    var remaining_int: Int64 = value
    if remaining_int >= 0:
        while remaining_int != 0:
            var digit_value = int(remaining_int % radix)

            # Push the char representing the value of the least significant digit
            var digit_char: Int8 = digit_chars_array[digit_value]
            fmt.push_back(digit_char)

            # Drop the least significant digit
            remaining_int = remaining_int / radix
    else:
        while remaining_int != 0:
            # `%` works differently for negative numbers in Mojo than in C
            var digit_value = int(_abs(remaining_int % -radix))

            # Push the char representing the value of the least significant digit
            var digit_char: Int8 = digit_chars_array[digit_value]
            fmt.push_back(digit_char)

            # Drop the least significant digit
            remaining_int = remaining_int / radix

    # We pushed the digits with least significant digits coming first, but
    # the number should have least significant digits at the end, so reverse
    # the order of the digit characters in the string.
    fmt._reverse(start=first_digit_pos)
