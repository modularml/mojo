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
"""Unicode codepoint handling.

This module provides the `Codepoint` type for representing single Unicode scalar values.
A codepoint represents a single Unicode character, restricted to valid Unicode scalar
values in the ranges `0` to `0xD7FF` and `0xE000` to `0x10FFFF` inclusive.
"""


from std.sys.intrinsics import likely

from std.bit import count_leading_zeros
from std.bit.mask import splat
import std.format._utils as fmt
from std.os import abort


@always_inline
def _is_unicode_scalar_value(codepoint: UInt32) -> Bool:
    """Returns True if `codepoint` is a valid Unicode scalar value.

    Args:
        codepoint: The codepoint integer value to check.

    Returns:
        True if `codepoint` is a valid Unicode scalar value; False otherwise.
    """
    return codepoint <= 0xD7FF or (
        codepoint >= 0xE000 and codepoint <= 0x10FFFF
    )


struct Codepoint(Comparable, ImplicitlyCopyable, Intable, Movable, Writable):
    """A Unicode codepoint, typically a single user-recognizable character;
    restricted to valid Unicode scalar values.

    This type is restricted to store a single Unicode `[*scalar value*][1]`,
    typically encoding a single user-recognizable character.

    All valid Unicode scalar values are in the range(s) `0` to `0xD7FF` and
    `0xE000` to `0x10FFFF`, inclusive. This type guarantees that the stored integer
    value falls in these ranges.

    The `Codepoint` type provides functionality for:
    - Converting between codepoints and UTF-8 encoded bytes.
    - Testing character properties like ASCII, digits, whitespace etc.
    - Converting between codepoints and strings.
    - Safe construction from integers with validation.

    Example:

    ```mojo
    from std.collections.string import Codepoint
    from std.testing import assert_true

    # Create a codepoint from a character
    var c = Codepoint.ord('A')

    # Check properties
    assert_true(c.is_ascii())
    assert_true(c.is_ascii_upper())

    # Convert to string
    var s = String(c)  # "A"
    ```

    [1]: https://www.unicode.org/glossary/#unicode_scalar_value

    **Codepoints versus scalar values:**

    Formally, Unicode defines a codespace of values in the range `0` to
    `0x10FFFF` (inclusive), and a [Unicode
    codepoint](https://www.unicode.org/glossary/#code_point) is any integer
    falling within that range. However, due to historical reasons, it became
    necessary to "carve out" a subset of the codespace known as [Unicode scalar
    values][1], that excludes codepoints in the range `0xD800` - `0xDFFF`. The
    codepoints in the excluded range are known as "surrogate" codepoints and
    will never be assigned a semantic meaning—they can only validly appear in
    UTF-16 encoded text.

    The difference between codepoints and scalar values is a technical
    distinction related to the backwards-compatible workaround chosen to enable
    UTF-16 to encode the full range of the Unicode codespace. For simplicities
    sake, and to avoid a confusing clash with the Mojo `Scalar` type, this type
    is pragmatically named `Codepoint`, even though it is restricted to valid
    scalar values.
    """

    var _scalar_value: UInt32
    """The Unicode scalar value represented by this type."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self, *, unsafe_unchecked_codepoint: UInt32):
        """Construct a `Codepoint` from a code point value without checking that it
        falls in the valid range.

        Safety:
            The provided codepoint value MUST be a valid Unicode scalar value.
            Providing a value outside of the valid range could lead to undefined
            behavior in algorithms that depend on the validity guarantees of
            this type.

        Args:
            unsafe_unchecked_codepoint: A valid Unicode scalar value code point.
        """
        assert _is_unicode_scalar_value(
            unsafe_unchecked_codepoint
        ), "codepoint is not a valid Unicode scalar value"

        self._scalar_value = unsafe_unchecked_codepoint

    @always_inline
    def __init__(out self, codepoint: UInt8):
        """Construct a `Codepoint` from a single byte value.

        This constructor cannot fail because non-negative 8-bit integers are
        valid Unicode scalar values.

        Args:
            codepoint: The 8-bit codepoint value to convert to a `Codepoint`.
        """
        self._scalar_value = UInt32(Int(codepoint))

    @always_inline("nodebug")
    def __init__(out self, lit: StringLiteral):
        """Construct a `Codepoint` from a single-codepoint `StringLiteral`.

        This constructor validates at compile-time that the literal contains
        exactly one Unicode codepoint, and computes the codepoint value as a
        compile-time constant. This provides an ergonomic and efficient way to
        create codepoints from string literals without runtime overhead.

        Args:
            lit: A string literal containing exactly one Unicode codepoint.

        Constraints:
            The string literal must contain exactly one Unicode codepoint.
            Multi-byte UTF-8 sequences are supported (e.g., "🔥").

        Examples:

        ```mojo
        from std.collections.string import Codepoint
        from std.testing import assert_equal

        # Create codepoints from ASCII literals
        var a = Codepoint("A")
        assert_equal(a.to_u32(), 65)

        var space = Codepoint(" ")
        assert_equal(space, Codepoint.ord(" "))

        # Multi-byte UTF-8 codepoints also work
        var emoji = Codepoint("🔥")
        assert_equal(emoji, Codepoint.ord("🔥"))
        ```
        """
        # Reconstruct the literal from the type parameter to force compile-time
        # evaluation.
        comptime sl: StringLiteral[lit.value] = {}

        # Compile-time validation for proper UTF-8 codepoint.
        comptime assert (
            sl.byte_length() > 0
        ), "Cannot construct an empty codepoint"

        # SAFETY:
        #   This is safe because `StringLiteral` is guaranteed to point to
        #   valid UTF-8.
        comptime char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            sl.as_bytes()
        )

        comptime assert sl.byte_length() == Int(
            num_bytes
        ), "input string must be one character"

        self = char

    # ===-------------------------------------------------------------------===#
    # Factory methods
    # ===-------------------------------------------------------------------===#

    @staticmethod
    def from_u32(codepoint: UInt32) -> Optional[Self]:
        """Construct a `Codepoint` from a code point value. Returns None if the
        provided `codepoint` is not in the valid range.

        Args:
            codepoint: An integer representing a Unicode scalar value.

        Returns:
            A `Codepoint` if `codepoint` falls in the valid range for Unicode
            scalar values, otherwise None.
        """

        if _is_unicode_scalar_value(codepoint):
            return Codepoint(unsafe_unchecked_codepoint=codepoint)
        else:
            return None

    @staticmethod
    def ord(string: StringSlice[mut=False, _]) -> Codepoint:
        """Returns the `Codepoint` that represents the given single-character
        string.

        Given a string containing one character, return a `Codepoint`
        representing the codepoint of that character. For example,
        `Codepoint.ord("a")` returns the codepoint `97`. This is the inverse of
        the `chr()` function.

        This function is similar to the `ord()` free function, except that it
        returns a `Codepoint` instead of an `Int`.

        Args:
            string: The input string, which must contain only a single character.

        Returns:
            A `Codepoint` representing the codepoint of the given character.
        """
        if string.byte_length() == 0:
            abort("Codepoint.ord: input string must not be empty")

        # SAFETY:
        #   This is safe because `StringSlice` is guaranteed to point to valid
        #   UTF-8, and we verified above that the input is non-empty.
        var char, num_bytes = Codepoint.unsafe_decode_utf8_codepoint(
            string.as_bytes()
        )

        assert (
            string.byte_length() == num_bytes
        ), "input string must be one character"

        return char

    # TODO: add optimize_ascii and branchless optimization options like unsafe_write_utf8
    @staticmethod
    def unsafe_decode_utf8_codepoint(
        s: ImmSpan[UInt8, _],
    ) -> Tuple[Codepoint, Int]:
        """Decodes a single `Codepoint` and number of bytes read from a given
        UTF-8 string pointer.

        Safety:
            `_ptr` MUST point to the first byte in a **known-valid** UTF-8
            character sequence. This function MUST NOT be used on unvalidated
            input.

        Args:
            s: Span to UTF-8 encoded data containing at least one valid
                encoded codepoint.

        Returns:
            The decoded codepoint `Codepoint`, as well as the number of bytes
            read.

        """
        # UTF-8 to Unicode conversion:              (represented as UInt32 BE)
        # 1: 0aaaaaaa                            -> 00000000 00000000 00000000 0aaaaaaa     a
        # 2: 110aaaaa 10bbbbbb                   -> 00000000 00000000 00000aaa aabbbbbb     a << 6  | b
        # 3: 1110aaaa 10bbbbbb 10cccccc          -> 00000000 00000000 aaaabbbb bbcccccc     a << 12 | b << 6  | c
        # 4: 11110aaa 10bbbbbb 10cccccc 10dddddd -> 00000000 000aaabb bbbbcccc ccdddddd     a << 18 | b << 12 | c << 6 | d
        assert len(s) > 0, "input Span must be non-empty"

        var ptr = s.unsafe_ptr()
        var b1 = ptr[]
        if (b1 >> 7) == 0:  # This is 1 byte ASCII char
            return Codepoint(b1), 1

        # NOTE: _utf8_first_byte_sequence_length does the same + an op to check
        # if it is ascii
        var num_bytes = count_leading_zeros(~b1)
        debug_assert(
            1 < Int(num_bytes) < 5, "invalid UTF-8 byte ", b1, " at index 0"
        )

        var shift = Int((6 * (num_bytes - 1)))
        var b1_mask = 0b11111111 >> (num_bytes + 1)
        var result = Int(b1 & b1_mask) << shift
        for i in range(1, Int(num_bytes)):
            ptr = ptr.unsafe_offset(1)
            # Assert that this is a continuation byte
            debug_assert(
                ptr[] >> 6 == 0b00000010,
                "invalid UTF-8 byte ",
                ptr[],
                " at index ",
                i,
            )
            shift -= 6
            result |= Int(ptr[] & 0b00111111) << shift

        # SAFETY: Safe because the input bytes are required to be valid UTF-8,
        #   and valid UTF-8 will never decode to an out of bounds codepoint
        #   using the above algorithm.
        # FIXME:
        #   UTF-8 encoding algorithms that do not properly exclude surrogate
        #   pair code points are actually relatively common (as I understand
        #   it); the algorithm above does not check for that.
        var char = Codepoint(unsafe_unchecked_codepoint=UInt32(result))
        return char, Int(num_bytes)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __lt__(self, other: Self) -> Bool:
        """Return True if this character is less than a different codepoint value from
        `other`.

        Args:
            other: The codepoint value to compare against.

        Returns:
            True if this character's value is less than the other codepoint value;
            False otherwise.
        """
        return self.to_u32() < other.to_u32()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __int__(self) -> Int:
        """Returns the numeric value of this scalar value as an integer.

        Returns:
            The numeric value of this scalar value as an integer.
        """
        return Int(self._scalar_value)

    def write_to(self, mut w: Some[Writer]):
        """
        Write a string representation of this `Codepoint` to the given writer.

        Args:
            w: The object to write to.
        """
        var char_len = self.utf8_byte_length()
        var result = String(unsafe_uninit_length=char_len)
        _ = self.unsafe_write_utf8(result.unsafe_as_bytes_mut().unsafe_ptr())
        w.write_string(result)

    @no_inline
    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the repr of this `Codepoint` to a writer.

        Writes the codepoint in the format `Codepoint(N)` where N is the
        Unicode scalar value.

        Args:
            writer: The object to write to.
        """
        fmt.FormatStruct(writer, "Codepoint").fields(self._scalar_value)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def is_ascii(self) -> Bool:
        """Returns True if this `Codepoint` is an ASCII character.

        All ASCII characters are less than or equal to codepoint value 127, and
        take exactly 1 byte to encode in UTF-8.

        Returns:
            A boolean indicating if this `Codepoint` is an ASCII character.
        """
        return self._scalar_value <= 0b0111_1111

    def is_ascii_digit(self) -> Bool:
        """Determines whether the given character is a digit [0-9].

        Returns:
            True if the character is a digit.
        """
        comptime ord_0 = UInt32(ord("0"))
        comptime ord_9 = UInt32(ord("9"))
        return ord_0 <= self.to_u32() <= ord_9

    def is_ascii_upper(self) -> Bool:
        """Determines whether the given character is an uppercase character.

        This currently only respects the default "C" locale, i.e. returns True
        iff the character specified is one of "ABCDEFGHIJKLMNOPQRSTUVWXYZ".

        Returns:
            True if the character is uppercase.
        """
        comptime ord_a = UInt32(ord("A"))
        comptime ord_z = UInt32(ord("Z"))
        return ord_a <= self.to_u32() <= ord_z

    def is_ascii_lower(self) -> Bool:
        """Determines whether the given character is an lowercase character.

        This currently only respects the default "C" locale, i.e. returns True
        iff the character specified is one of "abcdefghijklmnopqrstuvwxyz".

        Returns:
            True if the character is lowercase.
        """
        comptime ord_a = UInt32(ord("a"))
        comptime ord_z = UInt32(ord("z"))
        return ord_a <= self.to_u32() <= ord_z

    @staticmethod
    @always_inline
    def _is_ascii_printable(codepoint: Scalar) -> Bool:
        """Determines whether the given character is a printable character.

        Args:
            codepoint: The codepoint to check.

        Returns:
            True if the character is a printable character, otherwise False.
        """
        comptime assert (
            codepoint.dtype.is_integral()
        ), "only integral codepoints exist"
        comptime ` ` = type_of(codepoint)(ord(" "))
        comptime `~` = type_of(codepoint)(ord("~"))
        return ` ` <= codepoint <= `~`

    @always_inline
    def is_ascii_printable(self) -> Bool:
        """Determines whether the given character is a printable character.

        Returns:
            True if the character is a printable character, otherwise False.
        """
        return Self._is_ascii_printable(self.to_u32())

    @always_inline
    def is_python_space(self) -> Bool:
        """Determines whether this character is a Python whitespace string.

        This corresponds to Python's [universal separators](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines):
         `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if this character is one of the whitespace characters listed
            above, otherwise False.

        For example, check if a string contains only whitespace:

        ```mojo
        from std.testing import assert_true, assert_false

        # ASCII space characters
        assert_true(Codepoint.ord(" ").is_python_space())
        assert_true(Codepoint.ord("\t").is_python_space())

        # Unicode paragraph separator:
        assert_true(Codepoint.from_u32(0x2029).value().is_python_space())

        # Letters are not space characters
        assert_false(Codepoint.ord("a").is_python_space())
        ```
        """

        comptime next_line = Codepoint.from_u32(0x85).value()
        comptime unicode_line_sep = Codepoint.from_u32(0x2028).value()
        comptime unicode_paragraph_sep = Codepoint.from_u32(0x2029).value()

        return self.is_posix_space() or self in (
            next_line,
            unicode_line_sep,
            unicode_paragraph_sep,
        )

    def is_posix_space(self) -> Bool:
        """Returns True if this `Codepoint` is a **space** character according to the
        [POSIX locale][1].

        The POSIX locale is also known as the C locale.

        [1]: https://pubs.opengroup.org/onlinepubs/9699919799/basedefs/V1_chap07.html#tag_07_03_01

        This only respects the default "C" locale, i.e. returns True only if the
        character specified is one of " \\t\\n\\v\\f\\r". For semantics similar
        to Python, use `String.isspace()`.

        Returns:
            True iff the character is one of the whitespace characters listed
            above.
        """
        if not self.is_ascii():
            return False

        # ASCII char
        var c = UInt8(Int(self))

        # NOTE: a global LUT doesn't work at compile time so we can't use it here.
        comptime ` ` = UInt8(ord(" "))
        comptime `\t` = UInt8(ord("\t"))
        comptime `\n` = UInt8(ord("\n"))
        comptime `\r` = UInt8(ord("\r"))
        comptime `\f` = UInt8(ord("\f"))
        comptime `\v` = UInt8(ord("\v"))
        comptime `\x1c` = UInt8(ord("\x1c"))
        comptime `\x1d` = UInt8(ord("\x1d"))
        comptime `\x1e` = UInt8(ord("\x1e"))

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

    @always_inline
    def to_u32(self) -> UInt32:
        """Returns the numeric value of this scalar value as an unsigned 32-bit
        integer.

        Returns:
            The numeric value of this scalar value as an unsigned 32-bit
            integer.
        """
        return self._scalar_value

    @always_inline
    def unsafe_write_utf8[
        optimize_ascii: Bool = True, branchless: Bool = False
    ](self, ptr: MutPointer[Byte, ...]) -> Int:
        """Shift unicode to utf8 representation.

        Parameters:
            optimize_ascii: Optimize for languages with mostly ASCII characters.
            branchless: Use a branchless algorithm.

        Args:
            ptr: Pointer value to write the encoded UTF-8 bytes. Must validly
                point to a sufficient number of bytes (1-4) to hold the encoded
                data.

        Returns:
            Returns the number of bytes written.

        Safety:
            `ptr` MUST point to at least `self.utf8_byte_length()` allocated
            bytes or else an out-of-bounds write will occur, which is undefined
            behavior.

        ### Unicode (represented as UInt32 BE) to UTF-8 conversion:
        - 1: 00000000 00000000 00000000 0aaaaaaa -> 0aaaaaaa
            - a
        - 2: 00000000 00000000 00000aaa aabbbbbb -> 110aaaaa 10bbbbbb
            - (a >> 6)  | 0b11000000, b         | 0b10000000
        - 3: 00000000 00000000 aaaabbbb bbcccccc -> 1110aaaa 10bbbbbb 10cccccc
            - (a >> 12) | 0b11100000, (b >> 6)  | 0b10000000, c        | 0b10000000
        - 4: 00000000 000aaabb bbbbcccc ccdddddd -> 11110aaa 10bbbbbb 10cccccc
        10dddddd
            - (a >> 18) | 0b11110000, (b >> 12) | 0b10000000, (c >> 6) | 0b10000000,
            d | 0b10000000
        .
        """
        var c = Int(self)

        var num_bytes = self.utf8_byte_length()

        comptime if not branchless:
            var is_ascii: Bool

            comptime if optimize_ascii:
                is_ascii = likely(num_bytes == 1)
            else:
                is_ascii = num_bytes == 1

            comptime cont_mask = 0b11_1111  # 6 set bits
            comptime cont_marker = 0b1000_0000  # marker for continuation bytes

            if is_ascii:
                ptr[unsafe_offset=0] = Byte(c)
            elif num_bytes == 2:
                ptr[unsafe_offset=0] = Byte(
                    (c >> 6) | 0b1100_0000
                )  # marker for 2 byte sequence
                ptr[unsafe_offset=1] = Byte((c & cont_mask) | cont_marker)
            elif num_bytes == 3:
                ptr[unsafe_offset=0] = Byte(
                    (c >> 12) | 0b1110_0000
                )  # marker for 3 byte sequence
                ptr[unsafe_offset=1] = Byte(
                    ((c >> 6) & cont_mask) | cont_marker
                )
                ptr[unsafe_offset=2] = Byte((c & cont_mask) | cont_marker)
            else:
                ptr[unsafe_offset=0] = Byte(
                    (c >> 18) | 0b1111_0000
                )  # marker for 4 byte sequence
                ptr[unsafe_offset=1] = Byte(
                    ((c >> 12) & cont_mask) | cont_marker
                )
                ptr[unsafe_offset=2] = Byte(
                    ((c >> 6) & cont_mask) | cont_marker
                )
                ptr[unsafe_offset=3] = Byte((c & cont_mask) | cont_marker)
        else:
            comptime if optimize_ascii:
                if likely(num_bytes == 1):
                    ptr[unsafe_offset=0] = UInt8(c)
                    return 1
                var shift = 6 * (num_bytes - 1)
                var mask = UInt8(0xFF) >> UInt8(num_bytes + 1)
                var num_bytes_marker = UInt8(0xFF) << UInt8(8 - num_bytes)
                ptr[unsafe_offset=0] = (
                    UInt8(c >> shift) & mask
                ) | num_bytes_marker
                for i in range(1, num_bytes):
                    shift -= 6
                    ptr[unsafe_offset=i] = Byte(
                        ((c >> shift) & 0b0011_1111) | 0b1000_0000
                    )
            else:
                var shift = 6 * (num_bytes - 1)
                var mask = UInt8(0xFF) >> UInt8(num_bytes + Int(num_bytes > 1))
                var num_bytes_marker = UInt8(0xFF) << UInt8(8 - num_bytes)
                ptr[unsafe_offset=0] = (UInt8(c >> shift) & mask) | (
                    num_bytes_marker & UInt8(splat(num_bytes != 1))
                )
                for i in range(1, num_bytes):
                    shift -= 6
                    ptr[unsafe_offset=i] = Byte(
                        ((c >> shift) & 0b0011_1111) | 0b1000_0000
                    )

        return num_bytes

    @always_inline
    def utf8_byte_length(self) -> Int:
        """Returns the number of UTF-8 bytes required to encode this character.

        Returns:
            Byte count of UTF-8 bytes required to encode this character.

        Notes:
            The returned value is always between 1 and 4 bytes.
        """

        # Minimum codepoint values (respectively) that can fit in a 1, 2, 3,
        # and 4 byte encoded UTF-8 sequence.
        comptime sizes = SIMD[.uint32, 4](
            0, UInt32(2**7), UInt32(2**11), UInt32(2**16)
        )

        # Count how many of the minimums this codepoint exceeds, which is equal
        # to the number of bytes needed to encode it.
        return Int(sizes.le(self.to_u32()).cast[.uint8]().reduce_add())
