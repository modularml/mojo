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

"""Implement UTF-8 utils."""

from std.base64._b64encode import _sub_with_saturation
from std.sys import simd_width_of, simd_byte_width
from std.sys.intrinsics import likely

from std.bit import count_leading_zeros
from std.collections import Span

# ===-----------------------------------------------------------------------===#
# Validate UTF-8
# ===-----------------------------------------------------------------------===#

comptime BIGGEST_UTF8_FIRST_BYTE = Byte(0b1111_0100)
"""Since the biggest unicode codepoint is 0x10FFFF then the biggest
first byte that a utf-8 sequence can have is 0b1111_0100 (0xF4).
"""

comptime TOO_SHORT: UInt8 = 1 << 0
comptime TOO_LONG: UInt8 = 1 << 1
comptime OVERLONG_3: UInt8 = 1 << 2
comptime SURROGATE: UInt8 = 1 << 4
comptime OVERLONG_2: UInt8 = 1 << 5
comptime TWO_CONTS: UInt8 = 1 << 7
comptime TOO_LARGE: UInt8 = 1 << 3
comptime TOO_LARGE_1000: UInt8 = 1 << 6
comptime OVERLONG_4: UInt8 = 1 << 6
comptime CARRY: UInt8 = TOO_SHORT | TOO_LONG | TWO_CONTS


# fmt: off
comptime shuf1 = SIMD[.uint8, 16](
    TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
    TOO_LONG, TOO_LONG, TOO_LONG, TOO_LONG,
    TWO_CONTS, TWO_CONTS, TWO_CONTS, TWO_CONTS,
    TOO_SHORT | OVERLONG_2,
    TOO_SHORT,
    TOO_SHORT | OVERLONG_3 | SURROGATE,
    TOO_SHORT | TOO_LARGE | TOO_LARGE_1000 | OVERLONG_4
)

comptime shuf2 = SIMD[.uint8, 16](
    CARRY | OVERLONG_3 | OVERLONG_2 | OVERLONG_4,
    CARRY | OVERLONG_2,
    CARRY,
    CARRY,
    CARRY | TOO_LARGE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000 | SURROGATE,
    CARRY | TOO_LARGE | TOO_LARGE_1000,
    CARRY | TOO_LARGE | TOO_LARGE_1000
)
comptime shuf3 = SIMD[.uint8, 16](
    TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
    TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE_1000 | OVERLONG_4,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | OVERLONG_3 | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_LONG | OVERLONG_2 | TWO_CONTS | SURROGATE | TOO_LARGE,
    TOO_SHORT, TOO_SHORT, TOO_SHORT, TOO_SHORT
)

comptime UTF8_CHAR_WIDTHS: Array[Byte, 256] = [
    #  1  2  3  4  5  6  7  8  9  A  B  C  D  E  F
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 0
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 1
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 2
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 3
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 4
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 5
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 6
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, # 7
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # 8
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # 9
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # A
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # B
    0, 0, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, # C
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, # D
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, # E
    4, 4, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, # F
]
# fmt: on


@always_inline
def _utf8_char_width(b: Byte) -> Int:
    return Int(materialize[UTF8_CHAR_WIDTHS]()[Int(b)])


@always_inline
def _extract_vector[
    width: SIMDLength, //, offset: Int
](a: SIMD[.uint8, width], b: SIMD[.uint8, width]) -> SIMD[.uint8, width]:
    # generates a single `vpalignr` on x86 with AVX
    return a.join(b).slice[width, offset=offset]()


def validate_chunk[
    simd_size: Int
](
    current_block: SIMD[.uint8, simd_size],
    previous_input_block: SIMD[.uint8, simd_size],
) -> SIMD[.uint8, simd_size]:
    comptime v0f = SIMD[.uint8, simd_size](0x0F)
    comptime v80 = SIMD[.uint8, simd_size](0x80)
    comptime third_byte = 0b11100000 - 0x80
    comptime fourth_byte = 0b11110000 - 0x80
    var prev1 = _extract_vector[simd_size - 1](
        previous_input_block, current_block
    )
    var byte_1_high = shuf1._dynamic_shuffle(prev1 >> 4)
    var byte_1_low = shuf2._dynamic_shuffle(prev1 & v0f)
    var byte_2_high = shuf3._dynamic_shuffle(current_block >> 4)
    var sc = byte_1_high & byte_1_low & byte_2_high

    var prev2 = _extract_vector[simd_size - 2](
        previous_input_block, current_block
    )
    var prev3 = _extract_vector[simd_size - 3](
        previous_input_block, current_block
    )
    var is_third_byte = _sub_with_saturation(prev2, third_byte)
    var is_fourth_byte = _sub_with_saturation(prev3, fourth_byte)
    var must23 = is_third_byte | is_fourth_byte
    var must23_as_80 = must23 & v80
    return must23_as_80 ^ sc


def _is_valid_utf8_runtime(span: ImmSpan[Byte, _]) -> Bool:
    """Fast utf-8 validation using SIMD instructions.

    References for this algorithm:
    J. Keiser, D. Lemire, Validating UTF-8 In Less Than One Instruction Per
    Byte, Software: Practice and Experience 51 (5), 2021
    https://arxiv.org/abs/2010.03090

    Blog post:
    https://lemire.me/blog/2018/10/19/validating-utf-8-bytes-using-only-0-45-cycles-per-byte-avx-edition/

    Code adapted from:
    https://github.com/simdutf/SimdUnicode/blob/main/src/UTF8.cs
    """

    var ptr = span.unsafe_ptr()
    var length = len(span)
    comptime simd_size = simd_byte_width()
    var i: Int = 0
    var previous = SIMD[.uint8, simd_size]()

    while i + simd_size <= length:
        var current_bytes = ptr.unsafe_offset(i).unsafe_load[width=simd_size]()
        var has_error = validate_chunk(current_bytes, previous)
        previous = current_bytes
        if any(has_error.ne(0)):
            return False
        i += simd_size

    var has_error: SIMD[.uint8, simd_size]
    # last incomplete chunk
    if i != length:
        var buffer = SIMD[.uint8, simd_size](0)
        for j in range(i, length):
            buffer[j - i] = ptr.unsafe_offset(j)[]
        has_error = validate_chunk(buffer, previous)
    else:
        # Add a chunk of 0s to the end to validate continuations bytes
        has_error = validate_chunk(SIMD[.uint8, simd_size](), previous)

    return all(has_error.eq(0))


def _is_valid_utf8_comptime(span: ImmSpan[Byte, _]) -> Bool:
    var ptr = span.unsafe_ptr()
    var length = Int(len(span))
    var offset = 0

    while offset < length:
        var b0 = ptr[unsafe_offset=offset]
        if b0 > BIGGEST_UTF8_FIRST_BYTE:
            return False
        var byte_type = _utf8_byte_type(b0)
        if byte_type == 0:
            offset += 1
            continue
        elif byte_type == 1:
            return False

        for i in range(1, Int(byte_type)):
            var idx = offset + i
            if idx >= length or not _is_utf8_continuation_byte(
                ptr[unsafe_offset=idx]
            ):
                return False

        # special unicode ranges
        var b1 = ptr[unsafe_offset=offset + 1]
        if byte_type == 2 and b0 < 0b1100_0010:
            return False
        elif b0 == 0xE0 and b1 < 0xA0:
            return False
        elif b0 == 0xED and b1 > 0x9F:
            return False
        elif b0 == 0xF0 and b1 < 0x90:
            return False
        elif b0 == 0xF4 and b1 > 0x8F:
            return False

        offset += Int(byte_type)

    return True


@always_inline("nodebug")
def _is_valid_utf8(span: ImmSpan[Byte, _]) -> Bool:
    """Verify that the bytes are valid UTF-8.

    Args:
        span: The Span of bytes.

    Returns:
        Whether the data is valid UTF-8.

    #### UTF-8 coding format
    [Table 3-7 page 94](http://www.unicode.org/versions/Unicode6.0.0/ch03.pdf).
    Well-Formed UTF-8 Byte Sequences

    Code Points        | First Byte | Second Byte | Third Byte | Fourth Byte |
    :----------        | :--------- | :---------- | :--------- | :---------- |
    U+0000..U+007F     | 00..7F     |             |            |             |
    U+0080..U+07FF     | **C2**..DF | 80..BF      |            |             |
    U+0800..U+0FFF     | E0         | **A0**..BF  | 80..BF     |             |
    U+1000..U+CFFF     | E1..EC     | 80..BF      | 80..BF     |             |
    U+D000..U+D7FF     | ED         | 80..**9F**  | 80..BF     |             |
    U+E000..U+FFFF     | EE..EF     | 80..BF      | 80..BF     |             |
    U+10000..U+3FFFF   | F0         | **90**..BF  | 80..BF     | 80..BF      |
    U+40000..U+FFFFF   | F1..F3     | 80..BF      | 80..BF     | 80..BF      |
    U+100000..U+10FFFF | F4         | 80..**8F**  | 80..BF     | 80..BF      |
    """
    if __is_run_in_comptime_interpreter:
        return _is_valid_utf8_comptime(span)
    else:
        return _is_valid_utf8_runtime(span)


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline
def _is_utf8_continuation_byte[
    w: SIMDLength
](vec: SIMD[.uint8, w]) -> SIMD[.bool, w]:
    return vec.cast[.int8]().lt(-(0b1000_0000 >> 1))


@always_inline
def _is_utf8_start_byte(w: Byte) -> Bool:
    """Determine if `w` is either an ASCII character or the start
    of a UTF-8 multibyte character. This does _not_ validate `w` in
    any other way, for example `_is_utf8_start_byte(0b1111_1000)`
    will return True.
    """
    return w < 128 or w >= 192


@always_inline
def _count_utf8_continuation_bytes(span: Span[Byte, _]) -> Int:
    return Int(span.count(_is_utf8_continuation_byte))


@always_inline
def _utf8_first_byte_sequence_length(b: Byte) -> Int:
    """Get the length of the sequence starting with given byte. Do note that
    this does not work correctly if given a continuation byte."""

    assert b <= BIGGEST_UTF8_FIRST_BYTE, "first byte is out of range for utf-8"
    assert not _is_utf8_continuation_byte(
        b
    ), "Function does not work correctly if given a continuation byte."
    return Int(count_leading_zeros(~b) | b.lt(0b1000_0000).cast[.uint8]())


def _utf8_byte_type(b: SIMD[.uint8, _], /) -> type_of(b):
    """UTF-8 byte type.

    Returns:
        The byte type.

    Notes:

        - 0 -> ASCII byte.
        - 1 -> continuation byte.
        - 2 -> start of 2 byte long sequence.
        - 3 -> start of 3 byte long sequence.
        - 4 -> start of 4 byte long sequence.
    """
    assert b <= BIGGEST_UTF8_FIRST_BYTE, "first byte is out of range for utf-8"
    return count_leading_zeros(~b)


@always_inline
def _is_newline_char_utf8[
    include_r_n: Bool = False
](p: ImmPointer[Byte, ...], eol_start: Int, b0: Byte, char_len: Int,) -> Bool:
    """Returns whether the char is a newline char.

    Safety:
        This assumes valid utf-8 is passed.
    """
    # highly performance sensitive code, benchmark before touching
    comptime `\r` = UInt8(ord("\r"))
    comptime `\n` = UInt8(ord("\n"))
    comptime `\t` = UInt8(ord("\t"))
    comptime `\x1c` = UInt8(ord("\x1c"))
    comptime `\x1e` = UInt8(ord("\x1e"))

    # Since line-breaks are a relatively uncommon occurrence it is best to
    # branch here because the algorithm that calls this needs low latency rather
    # than high throughput, which is what a branchless algorithm with SIMD would
    # provide. So we do branching and add the likely intrinsic to reorder the
    # machine instructions optimally. Also memory reads are expensive and the
    # "happy path" of char_len == 1 is cheaper because it has none.
    if likely(char_len == 1):
        return `\t` <= b0 <= `\x1e` and not (`\r` < b0 < `\x1c`)
    elif char_len == 4:
        return False

    var b1 = p[unsafe_offset=eol_start + 1]
    if char_len == 2:
        var is_next_line = b0 == 0xC2 and b1 == 0x85  # unicode next line \x85

        comptime if include_r_n:
            return is_next_line or (b0 == `\r` and b1 == `\n`)
        else:
            return is_next_line
    else:  # unicode line sep or paragraph sep: \u2028 , \u2029
        assert char_len == 3, "invalid UTF-8 byte length"
        var b2 = p[unsafe_offset=eol_start + 2]
        return b0 == 0xE2 and b1 == 0x80 and (b2 == 0xA8 or b2 == 0xA9)


struct UTF8Chunk[origin: ImmOrigin](ImplicitlyCopyable):
    var valid: StringSlice[Self.origin]
    """The valid UTF-8 bytes."""

    var invalid: Span[Byte, Self.origin]
    """The invalid UTF-8 bytes."""

    @doc_hidden
    def __init__(
        out self,
        *,
        valid: StringSlice[Self.origin],
        invalid: Span[Byte, Self.origin],
    ):
        self.valid = valid
        self.invalid = invalid


# This is an implementation of Rust's `UTF8Chunk` iterator.
# https://doc.rust-lang.org/src/core/str/lossy.rs.html#194
struct UTF8Chunks[origin: ImmOrigin](ImplicitlyCopyable, Iterable, Iterator):
    """An iterator over valid and invalid UTF-8 chunks."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    comptime Element = UTF8Chunk[Self.origin]

    var _bytes: Span[Byte, Self.origin]

    def __init__(out self, bytes: Span[Byte, Self.origin]):
        self._bytes = bytes

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    def __next__(mut self) raises StopIteration -> UTF8Chunk[Self.origin]:
        if len(self._bytes) == 0:
            raise StopIteration()

        @always_inline
        def safe_get(i: Int) {imm self} -> Byte:
            return self._bytes[i] if i < len(self._bytes) else Byte(0)

        @always_inline
        def in_range(byte: Byte, *, start: Byte, end: Byte) -> Bool:
            return start <= byte <= end

        @always_inline
        def is_continuation_byte(byte: Byte) -> Bool:
            # Check if byte has the pattern 10xxxxxx (continuation byte).
            return byte & 192 == TWO_CONTS

        var i = 0
        var valid_up_to = 0

        while i < len(self._bytes):
            var byte = self._bytes.unsafe_get(i)
            i += 1

            # ASCII bytes (0-127) are always valid single-byte characters
            # Multi-byte sequences start with bytes >= 128
            if byte >= TWO_CONTS:
                var width = _utf8_char_width(byte)

                # 2-byte sequence: one continuation byte must follow
                if width == 2:
                    if not is_continuation_byte(safe_get(i)):
                        break
                    i += 1

                # 3-byte sequence: validate first byte + second byte combination,
                # then verify the remaining continuation bytes.
                elif width == 3:
                    var peek = safe_get(i)
                    # These range checks prevent overlong encodings and invalid
                    # Unicode ranges (like UTF-16 surrogates)
                    if byte == 0xE0 and in_range(peek, start=0xA0, end=0xBF):
                        pass  # Valid: prevents overlong 2-byte as 3-byte
                    elif in_range(byte, start=0xE1, end=0xEC) and in_range(
                        peek, start=0x80, end=0xBF
                    ):
                        pass  # Valid: normal 3-byte range
                    elif byte == 0xED and in_range(peek, start=0x80, end=0x9F):
                        pass  # Valid: excludes UTF-16 surrogate pairs
                    elif in_range(byte, start=0xEE, end=0xEF) and in_range(
                        peek, start=0x80, end=0xBF
                    ):
                        pass  # Valid: normal 3-byte range
                    else:
                        break  # Invalid combination

                    i += 1
                    if not is_continuation_byte(safe_get(i)):
                        break
                    i += 1

                # 4-byte sequence: validate first byte + second byte combination,
                # then verify the remaining continuation bytes.
                elif width == 4:
                    var peek = safe_get(i)
                    # These checks ensure we stay within valid Unicode range (U+10FFFF)
                    if byte == 0xF0 and in_range(peek, start=0x90, end=0xBF):
                        pass  # Valid: prevents overlong 3-byte as 4-byte
                    elif in_range(byte, start=0xF1, end=0xF3) and in_range(
                        peek, start=0x80, end=0xBF
                    ):
                        pass  # Valid: normal 4-byte range
                    elif byte == 0xF4 and in_range(peek, start=0x80, end=0x8F):
                        pass  # Valid: up to U+10FFFF (max Unicode)
                    else:
                        break  # Invalid combination or beyond Unicode range

                    i += 1
                    if not is_continuation_byte(safe_get(i)):
                        break
                    i += 1
                    if not is_continuation_byte(safe_get(i)):
                        break
                    i += 1

                # width == 0 or invalid width means invalid UTF-8 byte
                else:
                    break

            # If we reach here, the current character is valid
            valid_up_to = i

        # Split the inspected bytes into valid and invalid portions
        var inspected = self._bytes[:i]
        var remaining = self._bytes[i:]
        self._bytes = remaining

        var valid = inspected[:valid_up_to]
        var invalid = inspected[valid_up_to:]
        return UTF8Chunk(
            valid=StringSlice(unsafe_from_utf8=valid), invalid=invalid
        )
