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
"""Implements the `StringSpan` type and related utilities for efficient string operations."""

from std.builtin.builtin_slice import ContiguousSlice
from std.builtin.format_int import _write_int
from std.reflection import call_location
from std.collections import check_bounds, check_slice_bounds, Span
from std.collections.string._unicode import (
    is_lowercase,
    is_uppercase,
    to_lowercase,
    to_uppercase,
)
from std.collections.string._utf8 import (
    _count_utf8_continuation_bytes,
    _is_newline_char_utf8,
    _is_valid_utf8,
    _utf8_first_byte_sequence_length,
    _is_utf8_continuation_byte,
    _is_utf8_start_byte,
)
from std.collections.string.format import _FormatUtils
from std.collections.string.iterators import (
    BytesIter,
    CodepointSliceIter,
    CodepointsIter,
    GraphemeIndicesIter,
    GraphemeSliceIter,
)
from std.hashlib.hasher import Hasher
from std.format._utils import _TotalWritableBytes, _FlushingWriteBuffer
from std.math import align_down
from std.os import PathLike, abort
from std.sys import simd_width_of
from std.ffi import c_char, CStringSlice
from std.sys.intrinsics import likely, unlikely

from std.bit import count_leading_zeros, count_trailing_zeros
from std.bit.mask import is_negative, splat
from std.memory import (
    unsafe_memcmp,
    unsafe_memcpy,
    pack_bits,
)
from std.python import Python, PythonObject
from std.format._utils import _write_hex


comptime StringSlice = StringSpan
"""Provides a compatibility alias for `StringSpan`."""

comptime MutStringSpan[origin: MutOrigin] = StringSpan[origin]
"""Provides mutable access to the string data it views.

Parameters:
    origin: The origin of the string data.
"""

comptime MutStringSlice = MutStringSpan
"""Provides a compatibility alias for `MutStringSpan`."""

comptime ImmStringSpan[origin: ImmOrigin] = StringSpan[origin]
"""Provides read-only access to the string data it views.

Parameters:
    origin: The origin of the string data.
"""

comptime ImmStringSlice = ImmStringSpan
"""Provides a compatibility alias for `ImmStringSpan`."""

comptime StaticString = StringSpan[ImmStaticOrigin]
"""An immutable static string span.

This is a type of
[`StringSpan`](/docs/std/collections/string/string_span/StringSpan/)
that's immutable and statically allocated. You might use this for situations
that could also be done with a `String` type, but when you want to
optimize memory usage with zero heap allocations.

The key difference from `StringSpan` is that a `StaticString` is guaranteed
to point to data (string literals, constants) that will never be deallocated
(a regular `StringSpan` may point to any string data that might be freed).
This makes `StaticString` safe to store long-term without lifetime concerns.

Although you can reassign a `StaticString`-typed variable with a new value,
you can't modify the underlying data of a `StaticString` after it's created
the way you can with a `String`, such as using `+=` to append to it.

Because this is still a `StringSpan` type, you can do all the same things
with it, such as format a string:

```mojo
var format_string = StaticString("{}: {}")
print(format_string.format("bats", 6))     # => bats: 6
```
"""


struct StringSpan[mut: Bool, //, origin: Origin[mut=mut]](
    Boolable,
    Defaultable,
    Equatable,
    FloatableRaising,
    Hashable,
    ImplicitlyCopyable,
    IntableRaising,
    KeyElement,
    PathLike,
    TrivialRegisterPassable,
    Writable,
):
    """A non-owning view into encoded string data.

    A `StringSpan` is a lightweight view into string data that lets you look
    at part (or all) of an string without copying the data. Unlike a
    [`String`](/docs/std/collections/string/string/String/), a `StringSpan`
    doesn't own the string data, but it knows where to find it and how long it
    is. It's designed for efficient zero-copy string operations without memory
    allocation, while maintaining memory safety and UTF-8 awareness.

    Key features:

    - Zero-copy string operations for high performance.
    - Lightweight view that doesn't own or allocate memory.
    - UTF-8 aware string processing and character iteration.
    - Compatible ABI with `llvm::StringRef` for C++ interoperability.
    - Memory-safe slicing and substring operations.
    - Efficient string parsing and tokenization.

    Examples:

    ```mojo
    # Create a string span
    var text = StringSpan("Hello, 世界")

    # Zero-copy slicing (byte-level)
    var hello = text[byte=0:5] # Hello

    # Unicode-aware byte slicing
    var world = text[byte=7:13]  # "世界"

    # String comparison
    if text.startswith("Hello"):
        print("Found greeting")

    # String formatting
    var format_string = StaticString("{}: {}")
    print(format_string.format("bats", 6)) # bats: 6
    ```

    Related types:

    - [`String`](/docs/std/collections/string/string/String/): An owning,
      mutable string that allocates and manages its own memory.
    - [`StaticString`](/docs/std/collections/string/string_span/#StaticString): An
      alias for an immutable constant `StringSpan`.
    - [`StringLiteral`](/docs/std/builtin/string_literal/StringLiteral/): A
      string literal. String literals are compile-time values.

    Parameters:
        mut: Whether the slice is mutable.
        origin: The origin of the underlying string data.

    Notes:
        TODO: The underlying string data is guaranteed to be encoded using
        UTF-8.
    """

    # Aliases
    comptime Immutable = StringSpan[ImmOrigin(Self.origin)]
    """The immutable version of the `StringSpan`."""
    # Fields
    var _slice: Span[Byte, Self.origin]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __init__(out self):
        """Create an empty / zero-length slice."""
        self._slice = Span[Byte, Self.origin]()

    @doc_hidden
    @implicit
    @always_inline("nodebug")
    def __init__(
        other: StringSpan,
        out self: StringSpan[ImmOrigin(other.origin)],
    ):
        """Implicitly cast the mutable origin of self to an immutable one.

        Args:
            other: The Span to cast.
        """
        self = rebind[type_of(self)](other)

    @doc_hidden
    @always_inline
    def __init__(out self: StaticString, _kgen: __mlir_type.`!kgen.string`):
        # FIXME(MSTDL-160): !kgen.string's are not guaranteed to be UTF-8
        # encoded, they can be arbitrary binary data.
        var length: Int = Int(
            SIMDLength(mlir_value=__mlir_op.`pop.string.size`(_kgen))
        )
        var ptr = ImmPointer[_, ImmStaticOrigin](
            _mlir_value=__mlir_op.`pop.string.address`(_kgen)
        ).unsafe_bitcast[Byte]()
        self._slice = {unsafe_ptr = ptr, length = length}

    @always_inline
    @implicit
    def __init__(out self: StaticString, lit: StringLiteral):
        """Construct a new `StringSpan` from a `StringLiteral`.

        Args:
            lit: The literal to construct this `StringSpan` from.
        """
        # Since a StaticString has static origin, it will outlive
        # whatever arbitrary `origin` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StaticString is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        self = StaticString(lit.value)

    @always_inline("builtin")
    def __init__(out self, *, unsafe_from_utf8: Span[Byte, Self.origin]):
        """Construct a new `StringSpan` from a sequence of UTF-8 encoded bytes.

        Args:
            unsafe_from_utf8: A `Span[Byte]` encoded in UTF-8.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.
        """
        # FIXME(#3706): can't run at compile time
        # TODO(MOCO-1525):
        #   Support skipping UTF-8 during comptime evaluations, or support
        #   the necessary SIMD intrinsics to allow this to evaluate at compile
        #   time.
        # debug_assert(
        #     _is_valid_utf8(value.as_bytes()), "value is not valid utf8"
        # )
        self._slice = Span[Byte, Self.origin](
            unsafe_ptr=unsafe_from_utf8.unsafe_ptr(),
            length=unsafe_from_utf8.__len__(),
        )

    @always_inline
    def __init__[
        cstring_origin: ImmOrigin,
        //,
    ](
        out self: StringSpan[cstring_origin],
        *,
        unsafe_from_utf8: CStringSlice[cstring_origin],
    ):
        """Construct a new `StringSpan` from a UTF-8 encoded `CStringSlice`.

        Parameters:
            cstring_origin: The origin of the source `CStringSlice`.

        Args:
            unsafe_from_utf8: A `CStringSlice` encoded in UTF-8.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.
        """
        self._slice = unsafe_from_utf8.as_bytes()

    def __init__(out self, *, from_utf8: Span[Byte, Self.origin]) raises:
        """Construct a new `StringSpan` from a buffer containing UTF-8 encoded
        data.

        Args:
            from_utf8: A span of bytes containing UTF-8 encoded data.

        Raises:
            An exception is raised if the provided buffer byte values do not
            form valid UTF-8 encoded codepoints.
        """
        if not _is_valid_utf8(from_utf8.as_imm()):
            raise Error("StringSpan: buffer is not valid UTF-8")

        self = Self(unsafe_from_utf8=from_utf8)

    @implicit
    def __init__(out self, ref[Self.origin] value: String):
        """Construct a StringSpan from a String.

        This constructor propagates the mutability of the reference. If you
        have a mutable reference to a String, you get a mutable StringSpan.
        If you have an immutable reference, you get an immutable StringSpan.

        Args:
            value: The string value.
        """
        comptime if Self.origin.mut:
            # FIXME(MOCO-3906): Needs `unsafe_mut_cast()` because type refinement
            #   based on the `if origin.mut` knowledge is not supported.
            ref value_mut = Pointer(to=value).unsafe_mut_cast[True]()[]

            # Note: unsafe_as_bytes_mut() reallocates the `String` data if it
            #   was originally constructed from a read-only static string.
            # SAFETY:
            #   This is safe because the resulting UTF-8 byte slice is
            #   accessible only through the APIs of StringSpan, which
            #   either guarantee UTF-8 validity, or are unsafe themselves.
            self._slice = rebind[type_of(self._slice)](
                value_mut.unsafe_as_bytes_mut()
            )
        else:
            self._slice = rebind[type_of(self._slice)](value.as_bytes())

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    def write_to(self, mut writer: Some[Writer]):
        """Formats this string span to the provided `Writer`.

        Args:
            writer: The object to write to.
        """
        writer.write_string(self)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Formats this string span to the provided `Writer`.

        Args:
            writer: The object to write to.

        Notes:
            Mojo's repr always prints single quotes (`'`) at the start and end
            of the repr. Any single quote inside a string should be escaped
            (`\\'`).
        """
        comptime `\\` = Byte(ord("\\"))
        comptime `'` = Byte(ord("'"))
        comptime `\t` = Byte(ord("\t"))
        comptime `\n` = Byte(ord("\n"))
        comptime `\r` = Byte(ord("\r"))

        # Always start and end with a single quote
        writer.write_string("'")

        for s in self.codepoint_slices():
            var b0 = s.unsafe_ptr()[unsafe_offset=0]  # safe
            # Python escapes backslashes but they are ASCII printable
            if b0 == `\\`:
                writer.write_string(r"\\")
            elif b0 == `'`:  # escape single quotes
                writer.write_string(r"\'")
            elif Codepoint._is_ascii_printable(b0):
                writer.write_string(s)
            elif b0 == `\t`:
                writer.write_string(r"\t")
            elif b0 == `\n`:
                writer.write_string(r"\n")
            elif b0 == `\r`:
                writer.write_string(r"\r")
            elif b0 < 0b1000_0000:  # non-printable ASCII
                _write_hex[amnt_hex_bytes=2](writer, b0)
            else:  # multi-byte character
                writer.write_string(s)

        writer.write_string("'")

    def __bool__(self) -> Bool:
        """Check if a string span is non-empty.

        Returns:
           True if a string span is non-empty, False otherwise.
        """
        return len(self._slice) > 0

    def __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        hasher._update_with_bytes(
            Span(unsafe_ptr=self.unsafe_ptr(), length=self.byte_length())
        )

    def __fspath__(self) -> String:
        """Return the file system path representation of this string.

        Returns:
          The file system path representation as a string.
        """
        return String(self)

    @always_inline
    def __getitem__(self, *, byte: ContiguousSlice) -> Self:
        """Gets a substring at the specified byte positions.

        This performs byte-level slicing, not character (codepoint) slicing.
        The start and end positions are byte indices. For strings containing
        multi-byte UTF-8 characters, slicing at byte positions that do not fall
        on codepoint boundaries will abort.

        Aborts if `byte`'s start or end index is out of bounds (valid range is
        `0` to `self.byte_length()`, inclusive), or if start is greater than
        end. Negative indices are not supported and always abort.

        Args:
            byte: A slice that specifies byte positions of the new substring.

        Returns:
            A new StringSpan containing the bytes in the specified range.
        """
        var start, end = check_slice_bounds(byte, self.byte_length())

        debug_assert[assert_mode="safe"](
            start == len(self._slice)
            or _is_utf8_start_byte(self._slice.unsafe_get(start)),
            "String span starts on",
            start,
            " which is not a codepoint boundary.",
        )
        debug_assert[assert_mode="safe"](
            end == len(self._slice)
            or _is_utf8_start_byte(self._slice.unsafe_get(end)),
            "String span ends on, ",
            end,
            " which is not a codepoint boundary.",
        )
        return Self(unsafe_from_utf8=self._slice[byte])

    def _codepoint_byte_offset(self, count: Int) -> Int:
        # Byte offset of the boundary before the `count`-th codepoint.
        # Requires `0 <= count <= self.count_codepoints()`.
        var ptr = self.unsafe_ptr()
        var offset = 0
        for _ in range(count):
            offset += _utf8_first_byte_sequence_length(
                ptr[unsafe_offset=offset]
            )
        return offset

    def __getitem__(self, *, codepoint: Some[Indexer]) -> Self:
        """Gets the character at the specified position.

        Aborts if `codepoint` is out of bounds.

        Args:
            codepoint: The codepoint index.

        Returns:
            A `StringSpan` view containing the unicode codepoint at the
            specified position.
        """
        var c_idx = index(codepoint)
        if c_idx < 0 or c_idx >= self.count_codepoints():
            abort(t"codepoint index is out of bounds: {c_idx}")
        var ptr = self.unsafe_ptr()
        var start = self._codepoint_byte_offset(c_idx)
        var length = _utf8_first_byte_sequence_length(ptr[unsafe_offset=start])
        return Self(
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=ptr.unsafe_offset(start), length=length
            )
        )

    @always_inline
    def __getitem__(self, *, codepoint: ContiguousSlice) -> Self:
        """Gets a substring at the specified codepoint positions.

        Aborts if `codepoint`'s start or end index is out of bounds (valid
        range is `0` to `self.count_codepoints()`, inclusive), or if start
        is greater than end. Negative indices are not supported and always
        abort.

        Args:
            codepoint: A slice that specifies codepoint positions of the new
                substring.

        Returns:
            A new StringSpan containing the codepoints in the specified range.
        """
        var start, end = check_slice_bounds(codepoint, self.count_codepoints())
        var start_bytes = self._codepoint_byte_offset(start)
        var end_bytes = (
            start_bytes if end == start else self._codepoint_byte_offset(end)
        )
        return Self(
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self.unsafe_ptr().unsafe_offset(start_bytes),
                length=end_bytes - start_bytes,
            )
        )

    @always_inline
    def __getitem__(self, *, grapheme: Some[Indexer]) -> Self:
        """Gets the character at the specified position.

        Args:
            grapheme: The grapheme index.

        Returns:
            A `StringSpan` view containing the unicode grapheme at the
            specified position.
        """

        var char = self.graphemes().nth(index(grapheme))
        debug_assert[assert_mode="safe"](Bool(char), "Invalid grapheme index")
        return char.take()

    @doc_hidden
    def __init__(
        out self: StringSpan[ImmutAnyOrigin],
        *,
        unsafe_borrowed_obj: PythonObject,
    ) raises:
        """Construct a `StringSpan` from a Python `str` object.

        The caller is responsible for keeping the Python `str` object alive
        until the `StringSpan` is no longer needed.

        Args:
            unsafe_borrowed_obj: The Python `str` object to convert from.

        Raises:
            An error if the conversion failed.
        """
        ref cpython = Python().cpython()
        try:
            self = cpython.PyUnicode_AsUTF8AndSize(
                unsafe_borrowed_obj._obj_ptr
            )[]
        except:
            raise cpython.get_error()

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @doc_hidden
    @unavailable(
        "StringSpan does not support `__len__` because Mojo strings are"
        " UTF-8 encoded, so a single length is ambiguous: it could mean the"
        " number of UTF-8 bytes, the number of Unicode code points, or the"
        " number of user-visible characters (grapheme clusters). Use"
        " `s.byte_length()` or `len(s.bytes())` for the number of UTF-8 bytes,"
        " `len(s.codepoints())` for Unicode code points, or"
        " `len(s.graphemes())` for grapheme clusters."
    )
    def __len__(self) -> Int:
        ...

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_nested_origins_read_only
    def __eq__(self, rhs_same: Self) -> Bool:
        """Verify if a `StringSpan` is equal to another `StringSpan` with the
        same origin.

        Args:
            rhs_same: The `StringSpan` to compare against.

        Returns:
            If the `StringSpan` is equal to the input in length and contents.
        """
        return Self.__eq__(self, rhs=rhs_same)

    def __eq__(self, rhs: String) -> Bool:
        """Verify if a `StringSpan` is equal to another `String`.

        Args:
            rhs: The `String` to compare against.

        Returns:
            If the `StringSpan` is equal to the input in length and contents.
        """
        return self == StringSpan(rhs)

    # This decorator informs the compiler that indirect address spaces are not
    # dereferenced by the method.
    # TODO: replace with a safe model that checks the body of the method for
    # accesses to the origin.
    @__unsafe_nested_origins_read_only
    def __eq__(self, rhs: StringSpan) -> Bool:
        """Verify if a `StringSpan` is equal to another `StringSpan`.

        Args:
            rhs: The `StringSpan` to compare against.

        Returns:
            If the `StringSpan` is equal to the input in length and contents.
        """

        var s_len = self.byte_length()
        var s_ptr = self.unsafe_ptr()
        var rhs_ptr = rhs.unsafe_ptr()
        if s_len != rhs.byte_length():
            return False
        # same pointer and length, so equal
        elif s_len == 0 or s_ptr == rhs_ptr:
            return True
        return unsafe_memcmp(s_ptr, rhs_ptr, s_len) == 0

    @__unsafe_nested_origins_read_only
    def __ne__(self, rhs_same: Self) -> Bool:
        """Verify if a `StringSpan` is not equal to another `StringSpan` with
        the same origin.

        Args:
            rhs_same: The `StringSpan` to compare against.

        Returns:
            If the `StringSpan` is not equal to the input in length and
            contents.
        """
        return Self.__ne__(self, rhs=rhs_same)

    @__unsafe_nested_origins_read_only
    @always_inline
    def __ne__(self, rhs: StringSpan) -> Bool:
        """Verify if span is not equal to another `StringSpan`.

        Args:
            rhs: The `StringSpan` to compare against.

        Returns:
            If the `StringSpan` is not equal to the input in length and
            contents.
        """
        return not self == rhs

    @always_inline
    def __lt__(self, rhs: StringSpan) -> Bool:
        """Verify if the `StringSpan` bytes are strictly less than the input in
        overlapping content.

        Args:
            rhs: The other `StringSpan` to compare against.

        Returns:
            If the `StringSpan` bytes are strictly less than the input in
            overlapping content.
        """
        var len1 = self.byte_length()
        var len2 = rhs.byte_length()
        return Int(len1 < len2) > unsafe_memcmp(
            self.unsafe_ptr(), rhs.unsafe_ptr(), min(len1, len2)
        )

    @always_inline
    def __gt__(self, rhs: StringSpan) -> Bool:
        """Define whether this String span is strictly greater than the RHS.

        Args:
            rhs: The other `StringSpan` to compare against.

        Returns:
            True if this String span is strictly greater than the RHS
            StringSpan.
        """
        return not (self <= rhs)

    @always_inline
    def __le__(self, rhs: StringSpan) -> Bool:
        """Define whether this String span is less than or equal to the RHS.

        Args:
            rhs: The other `StringSpan` to compare against.

        Returns:
            True if this String span is less than or equal to the RHS
            StringSpan.
        """
        return not (rhs < self)

    @always_inline
    def __lt__(self, rhs: String) -> Bool:
        """Define whether this String span is strictly less than the RHS.

        Args:
            rhs: The other `String` to compare against.

        Returns:
            If the `StringSpan` bytes are strictly less than the input in
            overlapping content.
        """
        return self < StringSpan(rhs)

    @always_inline
    def __le__(self, rhs: String) -> Bool:
        """Define whether this String span is less than or equal to the RHS.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String span is less than or equal to the RHS String.
        """
        return self <= StringSpan(rhs)

    @always_inline
    def __gt__(self, rhs: String) -> Bool:
        """Define whether this String span is strictly greater than the RHS.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String span is strictly greater than the RHS String.
        """
        return self > StringSpan(rhs)

    @always_inline
    def __ge__(self, rhs: String) -> Bool:
        """Define whether this String span is greater than or equal to the RHS.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String span is greater than or equal to the RHS String.
        """
        return StringSpan(rhs) <= self

    def __iter__(self) -> GraphemeSliceIter[Self.origin]:
        """Iterate over the grapheme clusters in the string span.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen. See `graphemes()` for the precise
        definition.

        To iterate by Unicode codepoint or by byte instead, use
        `codepoints()`/`codepoint_slices()` or `bytes()`.

        Returns:
            An iterator yielding each grapheme cluster as a `StringSpan`.
        """
        return self.graphemes()

    def __reversed__(self) -> GraphemeSliceIter[Self.origin, False]:
        """Iterate backwards over the grapheme clusters in the string span.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen. See `graphemes()` for the precise
        definition.

        Returns:
            A reverse iterator yielding each grapheme cluster as a
            `StringSpan`.
        """
        return self.graphemes_reversed()

    @always_inline
    def __getitem__[I: Indexer, //](self, *, byte: I) -> Self:
        """Gets a single byte at the specified byte index.

        This performs byte-level indexing, not character (codepoint) indexing.
        For strings containing multi-byte UTF-8 characters `byte` must fall on
        a codepoint boundary and an entire codepoint will be returned.
        Aborts if `byte` does not fall on a codepoint boundary.

        Parameters:
            I: A type that can be used as an index.

        Args:
            byte: The byte index (0-based).

        Returns:
            A StringSpan containing the codepoint starting at the specified
            byte position.
        """
        var idx = index(byte)
        self._check_valid_index(idx)
        return self._unchecked_get_byte(idx)

    @always_inline
    def __getitem__(self, *, byte: IntLiteral) -> Self:
        """Gets a single byte at the specified byte index.

        This performs byte-level indexing, not character (codepoint) indexing.
        For strings containing multi-byte UTF-8 characters `byte` must fall on
        a codepoint boundary and an entire codepoint will be returned.
        Aborts if `byte` does not fall on a codepoint boundary.

        Args:
            byte: The byte index (0-based).

        Returns:
            A StringSpan containing the codepoint starting at the specified
            byte position.
        """
        comptime assert IntLiteral[byte.value]() >= 0, (
            "negative indexing is not supported, use e.g."
            " `slice[byte=slice.byte_length() - 1]`"
        )
        var idx = index(byte)
        self._check_valid_index(idx)
        return self._unchecked_get_byte(idx)

    @always_inline
    def __getitem__(self, *, grapheme: ContiguousSlice) -> Self:
        """Gets a substring at the specified grapheme-cluster positions.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen (see `graphemes()`). Slicing by
        grapheme requires a forward scan of the string and is O(n) in the
        byte length; use `byte=` slicing when you already have byte
        offsets.

        Aborts if `grapheme`'s start or end index is out of bounds (valid
        range is `0` to `self.count_graphemes()`, inclusive), or if start is
        greater than end. Negative indices are not supported and always
        abort. Counting the total number of graphemes to report a precise
        out-of-bounds message is itself an O(n) scan, so it is only done
        once a bad index is detected, not on every call.

        Args:
            grapheme: A slice specifying the grapheme-cluster range of the
                new substring.

        Returns:
            A new `StringSpan` covering the requested grapheme range.

        Examples:

        ```mojo
        from std.testing import assert_equal

        # "café" decomposed: 'c', 'a', 'f', 'e' + combining acute.
        # 5 codepoints, 4 graphemes.
        var s = StringSpan("cafe\\u{0301}")
        assert_equal(s[grapheme=0:3], "caf")
        assert_equal(s[grapheme=3:4], "e\\u{0301}")
        assert_equal(s[grapheme=3:], "e\\u{0301}")
        assert_equal(s[grapheme=:], s)
        ```
        """
        var start_idx = grapheme.start.or_else(0)
        var end_opt = grapheme.end

        # A negative index, or an explicit end before start, is invalid
        # regardless of the actual grapheme count. Recompute the count --
        # an O(n) scan -- only on this error path, and let
        # `check_slice_bounds` raise with a properly scoped message.
        if start_idx < 0 or (
            end_opt and end_opt.unsafe_value() < max(start_idx, 0)
        ):
            _ = check_slice_bounds(grapheme, self.count_graphemes())

        var total_bytes = len(self._slice)
        var iter = self.graphemes()
        var i = 0

        # Skip `start_idx` graphemes. Compute the byte offset once at the end
        # by subtracting the iterator's remaining byte length, instead of
        # summing each grapheme's `byte_length()` per iteration.
        while i < start_idx:
            if not iter.next():
                # Exhausted before reaching `start_idx`; `i` is the real
                # grapheme count, only known now that we've hit it.
                _ = check_slice_bounds(grapheme, i)
                break
            i += 1
        var start_bytes = total_bytes - iter.remaining_byte_length()

        if not end_opt:
            return Self(
                unsafe_from_utf8=Span[Byte, Self.origin](
                    unsafe_ptr=self._slice.unsafe_ptr().unsafe_offset(
                        start_bytes
                    ),
                    length=total_bytes - start_bytes,
                )
            )

        var end_idx = end_opt.unsafe_value()
        while i < end_idx:
            if not iter.next():
                _ = check_slice_bounds(grapheme, i)
                break
            i += 1
        var end_bytes = total_bytes - iter.remaining_byte_length()

        return Self(
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self._slice.unsafe_ptr().unsafe_offset(start_bytes),
                length=end_bytes - start_bytes,
            )
        )

    @always_inline
    def _check_valid_index(self, idx: Int):
        # Show source location where user provided incorrect index by skipping
        # two levels of inlining above this function call.
        var location = call_location[inline_count=2]()
        check_bounds(idx, self.byte_length(), location)
        # Subscripting checks codepoint boundaries unconditionally, to avoid
        # breaking methods that assume valid utf8.
        debug_assert[assert_mode="safe"](
            _is_utf8_start_byte(self._slice.unsafe_get(idx)),
            "String span index, ",
            idx,
            " does not lie on a codepoint boundary.",
            location=location,
        )

    @always_inline
    def _unchecked_get_byte(self, idx: Int) -> Self:
        return StringSpan(
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self.unsafe_ptr().unsafe_offset(idx),
                length=_utf8_first_byte_sequence_length(
                    self._slice.unsafe_get(idx)
                ),
            )
        )

    def __contains__(self, substr: StringSpan) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return self.find(substr) != -1

    @always_inline
    def __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.

        Raises:
            If the operation fails.
        """
        return atol(self)

    @always_inline
    def __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.

        Raises:
            If the operation fails.
        """
        return atof(self)

    def __add__(self, rhs: StringSpan) -> String:
        """Returns a string with this value prefixed on another string.

        Args:
            rhs: The right side of the result.

        Returns:
            The result string.
        """
        return String._add(self._slice, rhs._slice)

    def __radd__(self, lhs: StringSpan) -> String:
        """Returns a string with this value appended to another string.

        Args:
            lhs: The left side of the result.

        Returns:
            The result string.
        """
        return lhs + self

    def __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n: The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """
        var string = String()
        var buffer = _FlushingWriteBuffer(string)
        for _ in range(n):
            buffer.write_string(self)
        buffer.flush()
        return string^

    @always_inline("nodebug")
    def __merge_with__[
        other_type: type_of(StringSpan[_]),
    ](self, out result: StringSpan[origin_of(Self.origin, other_type.origin)],):
        """Returns a string span with merged origins.

        Parameters:
            other_type: The type of the origin to merge with.

        Returns:
            A StringSpan merged with the other origin.
        """
        return {
            unsafe_from_utf8 = Span[Byte, result.origin](
                unsafe_ptr=self.unsafe_ptr()
                .unsafe_mut_cast[result.mut]()
                .unsafe_origin_cast[result.origin](),
                length=self.byte_length(),
            )
        }

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @__allow_legacy_custom_self_type
    @always_inline
    def as_c_string_slice(
        self: StaticString,
    ) -> CStringSlice[ImmStaticOrigin]:
        """Return a CStringSlice for this StaticString.

        Returns:
            A c-compatible CStringSlice.
        """
        return {unsafe_from_ptr = self.unsafe_ptr().unsafe_bitcast[Int8]()}

    @always_inline
    def as_imm(self) -> Self.Immutable:
        """Return an immutable version of this Span.

        Returns:
            An immutable version of the same Span.
        """
        return rebind[Self.Immutable](self)

    def replace(self, old: StringSpan, new: StringSpan) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        if not old:
            return self._interleave(new)

        var occurrences = self.count(old)
        if occurrences == -1:
            return String(self)

        var self_len = self.byte_length()
        var old_len = old.byte_length()
        var new_len = new.byte_length()

        var res = String(
            capacity_bytes=self_len + (new_len - old_len) * occurrences
        )

        var current_pos = 0

        for _ in range(occurrences):
            var idx = self.find(old, current_pos)

            assert idx >= 0, "expected to find occurrence during find"

            # Copy preceding unchanged chars
            res += StringSpan(unsafe_from_utf8=self.as_bytes()[current_pos:idx])

            # Insert a copy of the new replacement string
            res += new

            current_pos = idx + old_len

        # Copy remaining chars
        if current_pos < self_len:
            res += StringSpan(unsafe_from_utf8=self.as_bytes()[current_pos:])

        return res^

    def _interleave(self, val: StringSpan) -> String:
        # TODO: this may be better as:
        # (val.byte_length() * self.count_codepoints()) + self.count_codepoints()
        var estimated_capacity = val.byte_length() * self.byte_length()
        var res = String(capacity_bytes=estimated_capacity)
        for codepoint in self.codepoint_slices():
            res += val
            res += codepoint
        return res^

    def _strip[forward: Bool](self, chars: ImmStringSpan) -> Self:
        var iter = CodepointSliceIter[forward=forward](self)
        while True:
            try:
                var next_it = iter.copy()
                var c = next_it.__next__()
                if c not in chars:
                    break
                iter = next_it^
            except:
                break
        return iter._slice

    @always_inline
    def strip(self, chars: ImmStringSpan) -> Self:
        """Returns a view of the string with leading and trailing characters
        removed. Note character is defined as a single unicode code-point,
        not any kind of displayed character, and strip can break apart
        graphemes.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A view of the string with no leading or trailing characters.

        Example:

        ```mojo
        print("himojohi".strip("hi")) # "mojo"
        ```
        """

        return self.lstrip(chars).rstrip(chars)

    @always_inline
    def strip(self) -> Self:
        """Returns a view of the string with leading and trailing whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A view of the string with no leading or trailing whitespaces.

        Example:

        ```mojo
        print("  mojo  ".strip()) # "mojo"
        ```
        """
        return self.lstrip().rstrip()

    @always_inline
    def rstrip(self, chars: ImmStringSpan) -> Self:
        """Returns a view of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A view of the string with no trailing characters.

        Example:

        ```mojo
        print("mojohi".strip("hi")) # "mojo"
        ```
        """

        return self._strip[forward=False](chars)

    @always_inline
    def rstrip(self) -> Self:
        """Returns a view of the string with trailing whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A view of the string with no trailing whitespaces.

        Example:

        ```mojo
        print("mojo  ".strip()) # "mojo"
        ```
        """
        var r_idx = self.byte_length()
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self.__reversed__():
        #     if not s.isspace():
        #         break
        #     r_idx -= 1
        while (
            r_idx > 0 and Codepoint(self.as_bytes()[r_idx - 1]).is_posix_space()
        ):
            r_idx -= 1
        return Self(unsafe_from_utf8=self.as_bytes()[:r_idx])

    @always_inline
    def lstrip(self, chars: ImmStringSpan) -> Self:
        """Returns a view of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A view of the string with no leading characters.

        Example:

        ```mojo
        print("himojo".strip("hi")) # "mojo"
        ```
        """

        return self._strip[forward=True](chars)

    @always_inline
    def lstrip(self) -> Self:
        """Returns a view of the string with leading whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A view of the string with no leading whitespaces.

        Example:

        ```mojo
        print("  mojo".strip()) # "mojo"
        ```
        """
        var l_idx = 0
        # TODO (#933): should use this once llvm intrinsics can be used at comp time
        # for s in self:
        #     if not s.isspace():
        #         break
        #     l_idx += 1
        while (
            l_idx < self.byte_length()
            and Codepoint(self.as_bytes()[l_idx]).is_posix_space()
        ):
            l_idx += 1
        return Self(unsafe_from_utf8=self.as_bytes()[l_idx:])

    def bytes(self) -> BytesIter[Self.origin]:
        """Returns an iterator over the raw bytes of this string span.

        Unlike `codepoints()` and `graphemes()`, this iterator operates at the
        byte level and yields individual `Byte` values without interpreting
        multi-byte UTF-8 sequences.

        Returns:
            An iterator type that returns successive `Byte` values stored in
            this string span.

        Examples:

        Iterate over the bytes of an ASCII string:

        ```mojo
        from std.testing import assert_equal, assert_raises

        var s = StringSpan("abc")
        var iter = s.bytes()
        assert_equal(next(iter), Byte(ord("a")))
        assert_equal(next(iter), Byte(ord("b")))
        assert_equal(next(iter), Byte(ord("c")))
        with assert_raises():
            _ = next(iter) # raises StopIteration
        ```

        Multi-byte UTF-8 sequences are yielded as individual bytes:

        ```mojo
        from std.testing import assert_equal

        # "é" is encoded in UTF-8 as two bytes: 0xC3 0xA9.
        var s = StringSpan("é")
        var iter = s.bytes()
        assert_equal(next(iter), Byte(0xC3))
        assert_equal(next(iter), Byte(0xA9))
        ```
        """
        return BytesIter(_slice=self)

    @always_inline
    def codepoints(self) -> CodepointsIter[Self.origin]:
        """Returns an iterator over the `Codepoint`s encoded in this string span.

        Returns:
            An iterator type that returns successive `Codepoint` values stored in
            this string span.

        **Examples:**

        Print the characters in a string:

        ```mojo
        from std.testing import assert_equal, assert_raises

        var s = StringSpan("abc")
        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
        assert_equal(iter.__next__(), Codepoint.ord("b"))
        assert_equal(iter.__next__(), Codepoint.ord("c"))
        with assert_raises():
            _ = iter.__next__() # raises StopIteration
        ```

        `codepoints()` iterates over Unicode codepoints, and supports multibyte
        codepoints:

        ```mojo
        from std.testing import assert_equal, assert_raises

        # A visual character composed of a combining sequence of 2 codepoints.
        var s = StringSpan("á")
        assert_equal(s.byte_length(), 3)

        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
         # U+0301 Combining Acute Accent
        assert_equal(iter.__next__().to_u32(), 0x0301)
        with assert_raises():
            _ = iter.__next__() # raises StopIteration
        ```
        """
        return CodepointsIter(self)

    def codepoint_slices(self) -> CodepointSliceIter[Self.origin]:
        """Iterate over the string, returning immutable references.

        Returns:
            An iterator of references to the string elements.
        """
        return CodepointSliceIter[Self.origin](self)

    def codepoint_slices_reversed(
        self,
    ) -> CodepointSliceIter[Self.origin, False]:
        """Iterates backwards over the string span, returning single-character slices.

        Each returned slice points to a single Unicode codepoint encoded in the
        underlying UTF-8 representation of this string span, starting from the end
        and moving towards the beginning.

        Returns:
            A reversed iterator of references to the string span elements.
        """
        return CodepointSliceIter[Self.origin, forward=False](self)

    def graphemes(self) -> GraphemeSliceIter[Self.origin]:
        """Return an iterator over the grapheme clusters in this string.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen. This handles combining marks, emoji
        ZWJ sequences, flag emoji, Hangul syllables, and other
        multi-codepoint clusters as defined by UAX #29.

        Returns:
            An iterator yielding each grapheme cluster as a `StringSpan`.

        Example:

        ```mojo
        from std.testing import assert_equal
        # "café" with combining accent: c, a, f, e + combining acute
        var s = StringSpan("cafe\\u{0301}")
        var count = 0
        for g in s.graphemes():
            count += 1
        assert_equal(count, 4)
        ```
        """
        return GraphemeSliceIter(self)

    def graphemes_reversed(self) -> GraphemeSliceIter[Self.origin, False]:
        """Return an iterator over the grapheme clusters in this string,
        yielding them in reverse order.

        See `graphemes()` for the definition of a grapheme cluster. Reverse
        iteration is more expensive per element than forward iteration: the
        UAX #29 state machine is forward-scanning, so each step backs up
        to a guaranteed grapheme boundary (typically a line break or the
        start of the string) and forward-scans from there.

        Returns:
            A reverse iterator yielding each grapheme cluster as a
            `StringSpan`.

        Example:

        ```mojo
        from std.testing import assert_equal

        var s = StringSpan("abc")
        var result = List[String]()
        for g in s.graphemes_reversed():
            result.append(String(g))
        assert_equal(len(result), 3)
        assert_equal(result[0], "c")
        assert_equal(result[1], "b")
        assert_equal(result[2], "a")
        ```
        """
        return GraphemeSliceIter[Self.origin, False](self)

    def grapheme_indices(self) -> GraphemeIndicesIter[Self.origin]:
        """Return an iterator over grapheme clusters paired with their byte
        offsets.

        Each yielded element is a `Tuple[Int, StringSpan]` where the first
        element is the byte offset (relative to the start of this string)
        at which the grapheme begins, and the second is the grapheme slice.

        Mirrors the shape of Rust's `str::grapheme_indices`.

        Returns:
            An iterator yielding `(byte_offset, grapheme)` pairs.

        Example:

        ```mojo
        from std.testing import assert_equal

        # "café" decomposed: 'c','a','f','e' + combining acute (U+0301)
        var s = StringSpan("cafe\\u{0301}")
        var offsets = List[Int]()
        for off, _ in s.grapheme_indices():
            offsets.append(off)
        # Offsets land at 0, 1, 2, 3; the 4th grapheme spans 3 bytes.
        assert_equal(len(offsets), 4)
        assert_equal(offsets[3], 3)
        ```
        """
        return GraphemeIndicesIter[Self.origin](self)

    def split_at_grapheme(
        self, n: Int
    ) -> Tuple[Self.Immutable, Self.Immutable]:
        """Split this string at the `n`-th grapheme-cluster boundary.

        Returns two slices: the first covers grapheme clusters `[0, n)` and
        the second covers `[n, count)` in a single forward pass.

        `n == 0` yields `("", self)`; `n >= count_graphemes()` yields
        `(self, "")`. Negative `n` is rejected in safe builds.

        Args:
            n: The grapheme-cluster boundary at which to split. Must be
                non-negative.

        Returns:
            A tuple `(prefix, suffix)` of `StringSpan`s.

        Example:

        ```mojo
        from std.testing import assert_equal

        var s = StringSpan("Hello, World!")
        var prefix, suffix = s.split_at_grapheme(5)
        assert_equal(prefix, "Hello")
        assert_equal(suffix, ", World!")
        ```
        """
        debug_assert[assert_mode="safe"](
            n >= 0, "grapheme split index must be non-negative"
        )
        var iter = self.graphemes()
        var split_bytes = 0
        for _ in range(n):
            var g = iter.next()
            if not g:
                break
            split_bytes += g.unsafe_value().byte_length()

        var total = len(self._slice)
        var prefix = Self.Immutable(
            unsafe_from_utf8=Span[Byte, ImmOrigin(Self.origin)](
                unsafe_ptr=self._slice.unsafe_ptr(), length=split_bytes
            )
        )
        var suffix = Self.Immutable(
            unsafe_from_utf8=Span[Byte, ImmOrigin(Self.origin)](
                unsafe_ptr=self._slice.unsafe_ptr().unsafe_offset(split_bytes),
                length=total - split_bytes,
            )
        )
        return (prefix, suffix)

    def count_graphemes(self) -> Int:
        """Count the number of grapheme clusters in this string.

        This is an O(n) operation that scans the full string to identify
        grapheme cluster boundaries using UAX #29 rules.

        Returns:
            The number of grapheme clusters.

        Example:

        ```mojo
        from std.testing import assert_equal
        var s = StringSpan("Hello")
        assert_equal(s.count_graphemes(), 5)
        ```
        """
        return len(self.graphemes())

    @always_inline
    def as_bytes(self) -> Span[Byte, Self.origin]:
        """Get the sequence of encoded bytes of the underlying string.

        Returns:
            A slice containing the underlying sequence of encoded bytes.
        """
        return self._slice

    @always_inline
    def unsafe_ptr(self) -> Pointer[Byte, Self.origin]:
        """Gets a pointer to the first element of this string span.

        Returns:
            A pointer pointing at the first element of this string span.
        """
        return self._slice.unsafe_ptr()

    @always_inline
    def byte_length(self) -> Int:
        """Get the length of this string span in bytes.

        Returns:
            The length of this string span in bytes.
        """

        return len(self._slice)

    def count_codepoints(self) -> Int:
        """Calculates the length in Unicode codepoints encoded in the
        UTF-8 representation of this string.

        This is an O(n) operation, where n is the length of the string, as it
        requires scanning the full string contents.

        Returns:
            The length in Unicode codepoints.

        Examples:

            Query the length of a string, in bytes and Unicode codepoints:

            ```mojo
            from std.testing import assert_equal

            var s = StringSpan("ನಮಸ್ಕಾರ")
            assert_equal(s.count_codepoints(), 7)
            assert_equal(s.byte_length(), 21)
            ```

            Strings containing only ASCII characters have the same byte and
            Unicode codepoint length:

            ```mojo
            from std.testing import assert_equal

            var s = StringSpan("abc")
            assert_equal(s.count_codepoints(), 3)
            assert_equal(s.byte_length(), 3)
            ```

            The character length of a string with visual combining characters is
            the length in Unicode codepoints, not grapheme clusters:

            ```mojo
            from std.testing import assert_equal

            var s = StringSpan("á")
            assert_equal(s.count_codepoints(), 2)
            assert_equal(s.byte_length(), 3)
            ```

        Notes:
            This method needs to traverse the whole string to count, so it has
            a performance hit compared to using the byte length.
        """
        # Every codepoint is encoded as one leading byte + 0 to 3 continuation
        # bytes.
        # The total number of codepoints is equal the number of leading bytes.
        # So we can compute the number of leading bytes (and thereby codepoints)
        # by subtracting the number of continuation bytes length from the
        # overall length in bytes.
        # For a visual explanation of how this UTF-8 codepoint counting works:
        #   https://connorgray.com/ephemera/project-log#2025-01-13
        var continuation_count = _count_utf8_continuation_bytes(self.as_bytes())
        return self.byte_length() - continuation_count

    def is_codepoint_boundary(self, index: Int) -> Bool:
        """Returns True if `index` is the position of the first byte in a UTF-8
        codepoint sequence, or is at the end of the string.

        A byte position is considered a codepoint boundary if a valid subslice
        of the string would end (noninclusive) at `index`.

        Positions `0` and `self.byte_length()` are considered to be codepoint boundaries.

        Positions beyond the length of the string span will return False.

        Args:
            index: An index into the underlying byte representation of the
                string.

        Returns:
            A boolean indicating if `index` gives the position of the first
            byte in a UTF-8 codepoint sequence, or is at the end of the string.

        Examples:

        Check if particular byte positions are codepoint boundaries:

        ```mojo
        from std.testing import assert_equal, assert_true, assert_false
        var abc = StringSpan("abc")
        assert_equal(len(abc), 3)
        assert_true(abc.is_codepoint_boundary(0))
        assert_true(abc.is_codepoint_boundary(1))
        assert_true(abc.is_codepoint_boundary(2))
        assert_true(abc.is_codepoint_boundary(3))
        ```

        Only the index of the first byte in a multi-byte codepoint sequence is
        considered a codepoint boundary:

        ```mojo
        var thumb = StringSpan("👍")
        assert_equal(len(thumb), 4)
        assert_true(thumb.is_codepoint_boundary(0))
        assert_false(thumb.is_codepoint_boundary(1))
        assert_false(thumb.is_codepoint_boundary(2))
        assert_false(thumb.is_codepoint_boundary(3))
        ```

        Visualization showing which bytes are considered codepoint boundaries,
        within a piece of text that includes codepoints whose UTF-8
        representation requires, respectively, 1, 2, 3, and 4-bytes. The
        codepoint boundary byte indices are indicated by a vertical arrow (↑).

        For example, this diagram shows that a slice of bytes formed by the
        half-open range starting at byte 3 and extending up to but not including
        byte 6 (`[3, 6)`) is a valid UTF-8 sequence.

        ```text
        ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
        ┃                a©➇𝄞                  ┃ String
        ┣━━┳━━━━━━━┳━━━━━━━━━━━┳━━━━━━━━━━━━━━━┫
        ┃97┃  169  ┃   10119   ┃    119070     ┃ Unicode Codepoints
        ┣━━╋━━━┳━━━╋━━━┳━━━┳━━━╋━━━┳━━━┳━━━┳━━━┫
        ┃97┃194┃169┃226┃158┃135┃240┃157┃132┃158┃ UTF-8 Bytes
        ┗━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┻━━━┛
        0  1   2   3   4   5   6   7   8   9  10
        ↑  ↑       ↑           ↑               ↑
        ```

        The following program verifies the above diagram:

        ```mojo
        from std.testing import assert_true, assert_false

        var text = StringSpan("a©➇𝄞")
        assert_true(text.is_codepoint_boundary(0))
        assert_true(text.is_codepoint_boundary(1))
        assert_false(text.is_codepoint_boundary(2))
        assert_true(text.is_codepoint_boundary(3))
        assert_false(text.is_codepoint_boundary(4))
        assert_false(text.is_codepoint_boundary(5))
        assert_true(text.is_codepoint_boundary(6))
        assert_false(text.is_codepoint_boundary(7))
        assert_false(text.is_codepoint_boundary(8))
        assert_false(text.is_codepoint_boundary(9))
        assert_true(text.is_codepoint_boundary(10))
        ```
        """
        # TODO: Example: Print the byte indices that are codepoints boundaries:

        if index >= self.byte_length():
            return index == self.byte_length()

        var byte = self.as_bytes()[index]
        # If this is not a continuation byte, then it must be a start byte.
        return not _is_utf8_continuation_byte(byte)

    def startswith(
        self, prefix: StringSpan, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Verify if the `StringSpan` starts with the specified prefix between
        start and end positions.

        The `start` and `end` positions must be offsets given in bytes, and
        must be codepoint boundaries.

        Args:
            prefix: The prefix to check.
            start: The start offset in bytes from which to check.
            end: The end offset in bytes from which to check.

        Returns:
            True if the `self[byte=start:end]` is prefixed by the input prefix.
        """
        if end == -1:
            return self.find(prefix, start) == start
        return StringSpan[Self.origin](
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self.unsafe_ptr().unsafe_offset(start),
                length=end - start,
            )
        ).startswith(prefix)

    def endswith(
        self, suffix: StringSpan, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Verify if the `StringSpan` end with the specified suffix between
        start and end positions.

        The `start` and `end` positions must be offsets given in bytes, and
        must be codepoint boundaries.

        Args:
            suffix: The suffix to check.
            start: The start offset in bytes from which to check.
            end: The end offset in bytes from which to check.

        Returns:
            True if the `self[byte=start:end]` is suffixed by the input suffix.
        """
        if suffix.byte_length() > self.byte_length():
            return False
        if end == -1:
            return (
                self.rfind(suffix, start) + suffix.byte_length()
                == self.byte_length()
            )
        # FIXME: use normalize_index
        return StringSpan[Self.origin](
            unsafe_from_utf8=Span[Byte, Self.origin](
                unsafe_ptr=self.unsafe_ptr().unsafe_offset(start),
                length=end - start,
            )
        ).endswith(suffix)

    def removeprefix(self, prefix: StringSpan, /) -> Self:
        """Returns a view of the string with the prefix removed if it was
        present.

        Args:
            prefix: The prefix to remove from the string.

        Returns:
            `string[byte=prefix.byte_length():]` if the string starts with the
            prefix string, or a view of the whole string otherwise.

        Examples:

        ```mojo
        print(StringSpan('TestHook').removeprefix('Test')) # 'Hook'
        print(StringSpan('BaseTestCase').removeprefix('Test')) # 'BaseTestCase'
        ```
        """
        if self.startswith(prefix):
            return self[byte = prefix.byte_length() :]
        return self

    def removesuffix(self, suffix: StringSpan, /) -> Self:
        """Returns a view of the string with the suffix removed if it was
        present.

        Args:
            suffix: The suffix to remove from the string.

        Returns:
            `string[byte=:(self.byte_length()-suffix.byte_length())]` if the string ends with the
            suffix string, or a view of the whole string otherwise.

        Examples:

        ```mojo
        print(StringSpan('TestHook').removesuffix('Hook')) # 'Test'
        print(StringSpan('BaseTestCase').removesuffix('Test')) # 'BaseTestCase'
        ```
        """
        if suffix and self.endswith(suffix):
            return self[byte = : self.byte_length() - suffix.byte_length()]
        return self

    @always_inline
    def format[*Ts: Writable](self, *args: *Ts) raises -> String:
        """Produce a formatted string using the current string as a template.

        The template, or "format string" can contain literal text and/or
        replacement fields delimited with curly braces (`{}`). Returns a copy of
        the format string with the replacement fields replaced with string
        representations of the `args` arguments.

        For more information, see the discussion in the
        [`format` module](/docs/std/collections/string/format/).

        Args:
            args: The substitution values.

        Parameters:
            Ts: The types of substitution values that implement `Writable`.

        Returns:
            The template with the given values substituted.

        Examples:

        ```mojo
        # Manual indexing:
        print(StringSpan("{0} {1} {0}").format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print(StringSpan("{} {}").format(True, "hello world")) # True hello world
        ```

        Raises:
            If the operation fails.
        """
        return _FormatUtils.format(self, *args)

    def find(self, substr: StringSpan, start: Int = 0) -> Int:
        """Finds the offset in bytes of the first occurrence of `substr`
        starting at `start`. If not found, returns `-1`.

        Args:
            substr: The substring to find.
            start: The offset in bytes from which to find.

        Returns:
            The offset in bytes of `substr` relative to the beginning of the
            string.
        """
        if not substr:
            return 0

        if self.byte_length() < substr.byte_length() + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var start_byte = start if start >= 0 else max(
            start + self.byte_length(), 0
        )
        var haystack = self.as_bytes()[start_byte:]

        var loc = _memmem(
            haystack.as_imm(),
            substr.as_bytes().as_imm(),
        )

        if not loc:
            return -1

        return Int(loc.unsafe_value()) - Int(self.unsafe_ptr())

    def rfind(self, substr: StringSpan, start: Int = 0) -> Int:
        """Finds the offset in bytes of the last occurrence of `substr` starting at
        `start`. If not found, returns `-1`.

        Args:
            substr: The substring to find.
            start: The offset in bytes from which to find.

        Returns:
            The offset in bytes of `substr` relative to the beginning of the
            string.
        """
        if not substr:
            return self.byte_length()

        if self.byte_length() < substr.byte_length() + start:
            return -1

        # The substring to search within, offset from the beginning if `start`
        # is positive, and offset from the end if `start` is negative.
        var start_byte = start if start >= 0 else max(
            start + self.byte_length(), 0
        )
        var haystack = self.as_bytes()[start_byte:]

        var loc = _memrmem(
            haystack,
            substr.as_bytes(),
        )

        if not loc:
            return -1

        return Int(loc.unsafe_value()) - Int(self.unsafe_ptr())

    def isspace[single_character: Bool = False](self) -> Bool:
        """Determines whether every character in the given StringSpan is a
        python whitespace String. This corresponds to Python's
        [universal separators](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines):
         `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Parameters:
            single_character: Whether to evaluate the `StringSpan` as a single
                unicode character (avoids overhead when already iterating).

        Returns:
            True if the whole StringSpan is made up of whitespace characters
            listed above, otherwise False.

        Example:

        Check if a string contains only whitespace:

        ```mojo
        from std.testing import assert_true, assert_false

        # An empty string is not considered to contain only whitespace chars:
        assert_false(StringSpan("").isspace())

        # ASCII space characters
        assert_true(StringSpan(" ").isspace())
        assert_true(StringSpan("\t").isspace())

        # Contains non-space characters
        assert_false(StringSpan(" abc  ").isspace())
        ```
        """

        def _is_space_char(s: StringSpan) -> Bool:
            # sorry for readability, but this has less overhead than memcmp
            # highly performance sensitive code, benchmark before touching
            comptime ` ` = UInt8(ord(" "))
            comptime `\t` = UInt8(ord("\t"))
            comptime `\n` = UInt8(ord("\n"))
            comptime `\r` = UInt8(ord("\r"))
            comptime `\f` = UInt8(ord("\f"))
            comptime `\v` = UInt8(ord("\v"))
            comptime `\x1c` = UInt8(ord("\x1c"))
            comptime `\x1d` = UInt8(ord("\x1d"))
            comptime `\x1e` = UInt8(ord("\x1e"))

            var no_null_len = s.byte_length()
            var ptr = s.unsafe_ptr()
            if likely(no_null_len == 1):
                var c = ptr[unsafe_offset=0]
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
            elif no_null_len == 2:
                return (
                    ptr[unsafe_offset=0] == 0xC2
                    and ptr[unsafe_offset=1] == 0x85
                )  # next_line: \x85
            elif no_null_len == 3:
                # unicode line sep or paragraph sep: \u2028 , \u2029
                var last_byte = (
                    ptr[unsafe_offset=2] == 0xA8 or ptr[unsafe_offset=2] == 0xA9
                )
                return (
                    ptr[unsafe_offset=0] == 0xE2
                    and ptr[unsafe_offset=1] == 0x80
                    and last_byte
                )
            return False

        comptime if single_character:
            return _is_space_char(self)
        else:
            for s in self.codepoint_slices():
                if not _is_space_char(s):
                    return False
            return self.byte_length() != 0

    @always_inline
    def split(self, sep: StringSpan) -> List[Self.Immutable]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting a space
        _ = StringSpan("hello world").split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = StringSpan("hello,,world").split(",") # ["hello", "", "world"]
        # Splitting with starting or ending separators
        _ = StringSpan(",1,2,3,").split(",") # ['', '1', '2', '3', '']
        # Splitting with an empty separator
        _ = StringSpan("123").split("") # ['', '1', '2', '3', '']
        ```
        """
        return _split[has_maxsplit=False](self.as_imm(), sep, -1)

    @always_inline
    def split(self, sep: StringSpan, maxsplit: Int) -> List[Self.Immutable]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:
        ```mojo
        # Splitting with maxsplit
        _ = StringSpan("1,2,3").split(",", maxsplit=1) # ['1', '2,3']
        # Splitting with starting or ending separators
        _ = StringSpan(",1,2,3,").split(",", maxsplit=1) # ['', '1,2,3,']
        # Splitting with an empty separator
        _ = StringSpan("123").split("", maxsplit=1) # ['', '123']
        ```
        """
        return _split[has_maxsplit=True](self.as_imm(), sep, maxsplit)

    @always_inline
    def split(self, sep: NoneType = None) -> List[Self.Immutable]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = StringSpan("      ").split() # []
        _ = StringSpan("").split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = StringSpan("      hello    world     ").split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = StringSpan(
            "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85world"
        ).split()  # ["hello", "world"]
        ```
        """
        return _split[has_maxsplit=False](self.as_imm(), sep, -1)

    @always_inline
    def split(
        self, sep: NoneType = None, *, maxsplit: Int
    ) -> List[Self.Immutable]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:
        ```mojo
        # Splitting with maxsplit
        _ = StringSpan("1     2  3").split(maxsplit=1) # ['1', '2  3']
        ```
        """
        return _split[has_maxsplit=True](self.as_imm(), sep, maxsplit)

    def isnewline[single_character: Bool = False](self) -> Bool:
        """Determines whether every character in the given StringSpan is a
        python newline character. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Parameters:
            single_character: Whether to evaluate the stringslice as a single
                unicode character (avoids overhead when already iterating).

        Returns:
            True if the whole StringSpan is made up of whitespace characters
                listed above, otherwise False.
        """

        var ptr = self.unsafe_ptr()
        var length = self.byte_length()

        comptime if single_character:
            return length != 0 and _is_newline_char_utf8[include_r_n=True](
                ptr, 0, ptr[unsafe_offset=0], length
            )
        else:
            var offset = 0
            for s in self.codepoint_slices():
                var b_len = s.byte_length()
                if not _is_newline_char_utf8(
                    ptr, offset, ptr[unsafe_offset=offset], b_len
                ):
                    return False
                offset += b_len
            return length != 0

    def splitlines(self, keepends: Bool = False) -> List[Self.Immutable]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """

        # highly performance sensitive code, benchmark before touching
        comptime `\r` = UInt8(ord("\r"))
        comptime `\n` = UInt8(ord("\n"))

        var output = List[Self.Immutable](capacity=128)  # guessing
        var ptr = self.as_imm().unsafe_ptr()
        var length = self.byte_length()
        var line_start = 0
        var prev_b0 = Byte(0)

        @always_inline
        def _splitlines[keep: Bool]() {mut}:
            while line_start < length:
                var line_end = line_start
                var is_new_line = False
                var b0 = Byte(0)
                var char_len = 0

                while not is_new_line and line_end < length:
                    b0 = ptr[unsafe_offset=line_end]
                    char_len = _utf8_first_byte_sequence_length(b0)
                    assert (
                        line_end + char_len <= length
                    ), "corrupted sequence causing unsafe memory access"
                    # percentage-wise a newline is uncommon compared to a normal byte
                    is_new_line = unlikely(
                        _is_newline_char_utf8(ptr, line_end, b0, char_len)
                    )
                    line_end += char_len

                var str_len = line_end - line_start

                # NOTE: when keep=True the algorithm needs to check the next
                # character to see whether it is \r\n due to having to store the
                # full line + the line ending.
                # When keep=False it is much faster because the previous
                # character is stored in a variable instead of having to deref a
                # pointer
                comptime if keep:
                    var is_r = unlikely(b0 == `\r`)
                    var may_be_r_n = is_r and likely(line_end < length)
                    var is_r_n = Int(
                        unlikely(
                            may_be_r_n and ptr[unsafe_offset=line_end] == `\n`
                        )
                    )
                    line_end += is_r_n
                    str_len += is_r_n
                else:
                    str_len -= splat(likely(is_new_line)) & char_len
                    var is_r_n = unlikely(prev_b0 == `\r` and b0 == `\n`)
                    prev_b0 = b0
                    if is_r_n:  # the line was already appended
                        line_start = line_end
                        continue
                var s = Self.Immutable(
                    unsafe_from_utf8=Span(
                        unsafe_ptr=ptr.unsafe_offset(line_start),
                        length=Int(str_len),
                    )
                )
                output.append(s)
                line_start = line_end

        if keepends:
            _splitlines[keep=True]()
        else:
            _splitlines[keep=False]()

        return output^

    def count(self, substr: StringSpan) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
            substr: The substring to count.

        Returns:
            The number of occurrences of `substr`.
        """
        if not substr:
            return self.byte_length() + 1

        var res = 0
        var offset = 0

        while True:
            var pos = self.find(substr, offset)
            if pos == -1:
                break
            res += 1

            offset = pos + substr.byte_length()

        return res

    def is_ascii_digit(self) -> Bool:
        """A string is a digit string if all characters in the string are digits
        and there is at least one character in the string.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits and it's not empty else False.
        """
        if not self:
            return False
        for char in self.codepoints():
            if not char.is_ascii_digit():
                return False
        return True

    def isupper(self) -> Bool:
        """Returns True if all cased characters in the string are uppercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are uppercase and there
            is at least one cased character, False otherwise.
        """
        return self.byte_length() > 0 and is_uppercase(self)

    def islower(self) -> Bool:
        """Returns True if all cased characters in the string are lowercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are lowercase and there
            is at least one cased character, False otherwise.
        """
        return self.byte_length() > 0 and is_lowercase(self)

    def lower(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        # TODO: the _unicode module does not support locale sensitive conversions yet.
        return to_lowercase(self)

    def upper(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        # TODO: the _unicode module does not support locale sensitive conversions yet.
        return to_uppercase(self)

    def is_ascii_printable(self) -> Bool:
        """Returns True if all characters in the string are ASCII printable.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are printable else False.
        """
        for char in self.codepoints():
            if not char.is_ascii_printable():
                return False
        return True

    def ascii_rjust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string span right justified in a string of specified width.

        Pads the string span on the left with the specified fill character so
        that the total (byte) length of the resulting string equals `width`. If the
        original string span is already longer than or equal to `width`,
        returns the string span unchanged (as a `String`).

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A right-justified string of (byte) length `width`, or the original string
            slice (as a `String`) if its length is already greater than or
            equal to `width`.

        Examples:

        ```mojo
        var s = StringSpan("hello")
        print(s.ascii_rjust(10))        # "     hello"
        print(s.ascii_rjust(10, "*"))   # "*****hello"
        print(s.ascii_rjust(3))         # "hello" (no padding)
        ```
        """
        return self._justify(width - self.byte_length(), width, fillchar)

    def ascii_ljust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string span left justified in a string of specified width.

        Pads the string span on the right with the specified fill character so
        that the total byte length of the resulting string equals `width`. If the
        original string span is already longer than or equal to `width`,
        returns the string span unchanged (as a `String`).

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A left-justified string of (byte) length `width`, or the original string
            slice (as a `String`) if its length is already greater than or
            equal to `width`.

        Examples:

        ```mojo
        var s = StringSpan("hello")
        print(s.ascii_ljust(10))        # "hello     "
        print(s.ascii_ljust(10, "*"))   # "hello*****"
        print(s.ascii_ljust(3))         # "hello" (no padding)
        ```
        """
        return self._justify(0, width, fillchar)

    def ascii_center(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string span center justified in a string of specified width.

        Pads the string span on both sides with the specified fill character so
        that the total length of the resulting string equals `width`. If the
        padding needed is odd, the extra character goes on the right side. If the
        original string span is already longer than or equal to `width`,
        returns the string span unchanged (as a `String`).

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A center-justified string of length `width`, or the original string
            slice (as a `String`) if its length is already greater than or
            equal to `width`.

        Examples:

        ```mojo
        var s = StringSpan("hello")
        print(s.ascii_center(10))        # "  hello   "
        print(s.ascii_center(11, "*"))   # "***hello***"
        print(s.ascii_center(3))         # "hello" (no padding)
        ```
        """
        return self._justify(width - self.byte_length() >> 1, width, fillchar)

    def _justify(
        self, start: Int, width: Int, fillchar: StaticString
    ) -> String:
        """Internal helper function to justify a string with padding.

        This function pads the string to a specified total width by adding fill
        characters. The `start` parameter controls how many fill characters are
        added before the string (left padding), with remaining fill characters
        added after the string (right padding) to reach the total width.

        Args:
            start: The number of fill characters to add at the beginning (left
                padding). For left justification this is 0, for right
                justification this is `width - self.byte_length()`, and for center
                justification this is `(width - self.byte_length()) >> 1`.
            width: The total width of the resulting string in bytes. If the
                original string is already greater than or equal to this width,
                no padding is added and the original string is returned.
            fillchar: The padding character to use. Must be a single-byte
                character.

        Returns:
            A new string of the specified total width with fill characters
            padding the original string, or the original string if it's already
            at least as wide as the requested width.
        """
        if self.byte_length() >= width:
            return String(self)
        assert (
            fillchar.byte_length() == 1
        ), "fill char needs to be a one byte literal"

        var result = String(capacity_bytes=width)
        for _ in range(start):
            result += fillchar
        result += self

        while result.byte_length() < width:
            result += fillchar
        return result

    def join[
        T: Copyable & Writable,
        //,
    ](self, elems: Span[T, _]) -> String:
        """Joins string elements using the current string as a delimiter.

        Parameters:
            T: The type of the elements, must implement the `Copyable`,
                and `Writable` traits.

        Args:
            elems: The input values.

        Returns:
            The joined string.

        Notes:
            - Defaults to writing directly to the string if the bytes
            fit in an inline `String`, otherwise will process it by chunks.
        """
        if len(elems) == 0:
            return String()

        var sep = StringSpan(
            unsafe_from_utf8=Span(
                unsafe_ptr=self.unsafe_ptr(), length=self.byte_length()
            )
        )
        var total_bytes = _TotalWritableBytes(elems, sep=sep).size
        var result = String(capacity_bytes=total_bytes)

        if result._is_inline():
            # Write directly to the stack address
            result.write(elems[0])
            for i in range(1, len(elems)):
                result.write(self, elems[i])
            return result^

        var buffer = _FlushingWriteBuffer(result)

        buffer.write(elems[0])
        for i in range(1, len(elems)):
            buffer.write(self, elems[i])
        buffer.flush()
        return result^


# ===-----------------------------------------------------------------------===#
# Utils
# ===-----------------------------------------------------------------------===#


@always_inline("nodebug")
def _get_kgen_string[
    string: StaticString, *extra: StaticString
]() -> __mlir_type.`!kgen.string`:
    """Form a `!kgen.string` from compile-time StringSpan values concatenated.

    Parameters:
        string: The first string span to use.
        extra: Additional string spans to concatenate.

    Returns:
        The string value as a `!kgen.string`.
    """
    return __mlir_attr[
        `#kgen.param.expr<data_to_str,`,
        string,
        `,`,
        extra.values,
        `> : !kgen.string`,
    ]


@always_inline("nodebug")
def get_static_string[
    string: StaticString, *extra: StaticString
]() -> StaticString:
    """Form a StaticString from compile-time StringSpan values. This
    guarantees that the returned string is compile-time constant in static
    memory.  It also guarantees that there is a 'nul' zero byte at the end,
    which is not included in the returned range.

    Parameters:
        string: The first StringSpan value.
        extra: Additional StringSpan values to concatenate.

    Returns:
        The string value as a StaticString.
    """
    return StaticString(_get_kgen_string[string, *extra]())


def _to_string_list[
    O: ImmOrigin,
    T: Copyable,
    //,
    len_fn: def(T) thin -> Int,
    unsafe_ptr_fn: def(T) thin -> Pointer[Byte, O],
](items: List[T]) -> List[String]:
    var i_len = len(items)

    var out_list = List[String](capacity=i_len)

    for i in range(i_len):
        var elt_ptr = Pointer(to=items[i])
        var og_len = len_fn(elt_ptr[])
        var og_ptr = unsafe_ptr_fn(elt_ptr[])
        out_list.append(
            String(
                StringSpan(
                    unsafe_from_utf8=Span[Byte, O](
                        unsafe_ptr=og_ptr, length=og_len
                    )
                )
            )
        )
    return out_list^


@always_inline
def _to_string_list[
    O: ImmOrigin, //
](items: List[StringSpan[O]]) -> List[String]:
    """Create a list of Strings **copying** the existing data.

    Parameters:
        O: The origin of the data.

    Args:
        items: The List of string spans.

    Returns:
        The list of created strings.
    """

    return _to_string_list[
        lambda (v: StringSpan[O]) -> Int: v.byte_length(),
        lambda (v: StringSpan[O]) -> Pointer[Byte, O]: v.unsafe_ptr(),
    ](items)


@always_inline
def _to_string_list[
    O: ImmOrigin, //
](items: List[Span[Byte, O]]) -> List[String]:
    """Create a list of Strings **copying** the existing data.

    Parameters:
        O: The origin of the data.

    Args:
        items: The List of Bytes.

    Returns:
        The list of created strings.
    """

    return _to_string_list[
        lambda (v: Span[Byte, O]) -> Int: len(v),
        lambda (v: Span[Byte, O]) -> Pointer[Byte, O]: v.unsafe_ptr(),
    ](items)


@always_inline
def _unsafe_strlen(ptr: ImmPointer[Byte, _], max: Int = Int.MAX) -> Int:
    """Get the length of a null-terminated string from a pointer.

    Args:
        ptr: The null-terminated pointer to the string.
        max: The maximum size of the string.

    Returns:
        The length of the null terminated string without the null terminator.

    Notes:
        The length does NOT include the null terminator.
    """
    var offset = 0
    while offset < max and ptr[unsafe_offset=offset]:
        offset += 1
    return offset


@always_inline
def _memchr[
    dtype: DType, //
](source: ImmSpan[Scalar[dtype], _], char: Scalar[dtype]) -> OptionalPointer[
    Scalar[dtype], source.origin
]:
    if (
        __is_run_in_comptime_interpreter
        or len(source) < simd_width_of[Scalar[dtype]]()
    ):
        var ptr = source.unsafe_ptr()

        for i in range(len(source)):
            if ptr[unsafe_offset=i] == char:
                return ptr.unsafe_offset(i)
        return {}
    else:
        return _memchr_impl(source, char)


@always_inline
def _memchr_impl[
    dtype: DType, //
](
    source: ImmSpan[Scalar[dtype], _],
    char: Scalar[dtype],
) -> OptionalPointer[
    Scalar[dtype], source.origin
]:
    var haystack = source.unsafe_ptr()
    var length = len(source)
    comptime bool_mask_width = simd_width_of[DType.bool]()
    var first_needle = SIMD[dtype, bool_mask_width](char)
    var vectorized_end = align_down(length, bool_mask_width)

    for i in range(0, vectorized_end, bool_mask_width):
        var bool_mask = haystack.unsafe_load[width=bool_mask_width](i).eq(
            first_needle
        )
        var mask = pack_bits(bool_mask)
        if mask:
            return haystack.unsafe_offset(
                Int(type_of(mask)(i) + count_trailing_zeros(mask))
            )

    for i in range(vectorized_end, length):
        if haystack[unsafe_offset=i] == char:
            return haystack.unsafe_offset(i)

    return {}


@always_inline
def _memmem[
    dtype: DType, //
](
    haystack_span: ImmSpan[Scalar[dtype], _],
    needle_span: ImmSpan[Scalar[dtype], _],
) -> OptionalPointer[Scalar[dtype], haystack_span.origin]:
    if (
        __is_run_in_comptime_interpreter
        or len(haystack_span) < simd_width_of[Scalar[dtype]]()
    ):
        var haystack: Pointer[
            Scalar[dtype], haystack_span.origin
        ] = haystack_span.unsafe_ptr()
        var haystack_len = len(haystack_span)
        var needle = needle_span.unsafe_ptr()
        var needle_len = len(needle_span)

        for i in range(haystack_len - needle_len + 1):
            if haystack[unsafe_offset=i] != needle[unsafe_offset=0]:
                continue

            if (
                unsafe_memcmp(
                    haystack.unsafe_offset(i + 1),
                    needle.unsafe_offset(1),
                    needle_len - 1,
                )
                == 0
            ):
                return haystack.unsafe_offset(i)

        return {}
    else:
        return _memmem_impl(haystack_span, needle_span)


@always_inline
def _memmem_impl[
    dtype: DType, //
](
    haystack_span: ImmSpan[Scalar[dtype], _],
    needle_span: ImmSpan[Scalar[dtype], _],
) -> OptionalPointer[Scalar[dtype], haystack_span.origin]:
    var haystack: Pointer[
        Scalar[dtype], haystack_span.origin
    ] = haystack_span.unsafe_ptr()
    var haystack_len = len(haystack_span)
    var needle = needle_span.unsafe_ptr()
    var needle_len = len(needle_span)
    assert needle_len > 0, "needle_len must be > 0"
    if needle_len == 1:
        return _memchr(haystack_span, needle[unsafe_offset=0])
    elif needle_len > haystack_len:
        return {}

    comptime bool_mask_width = simd_width_of[DType.bool]()
    var vectorized_end = align_down(
        haystack_len - needle_len + 1, bool_mask_width
    )

    var first_needle = SIMD[dtype, bool_mask_width](needle[unsafe_offset=0])
    var last_needle = SIMD[dtype, bool_mask_width](
        needle[unsafe_offset=needle_len - 1]
    )

    for i in range(0, vectorized_end, bool_mask_width):
        var first_block = haystack.unsafe_load[width=bool_mask_width](i)
        var last_block = haystack.unsafe_load[width=bool_mask_width](
            i + needle_len - 1
        )

        var bool_mask = first_needle.eq(first_block) & last_needle.eq(
            last_block
        )
        var mask = pack_bits(bool_mask)

        while mask:
            var offset = i + Int(count_trailing_zeros(mask))
            if (
                unsafe_memcmp(
                    haystack.unsafe_offset(offset + 1),
                    needle.unsafe_offset(1),
                    needle_len - 1,
                )
                == 0
            ):
                return haystack.unsafe_offset(offset)
            mask = mask & (mask - 1)

    for i in range(vectorized_end, haystack_len - needle_len + 1):
        if haystack[unsafe_offset=i] != needle[unsafe_offset=0]:
            continue

        if (
            unsafe_memcmp(
                haystack.unsafe_offset(i + 1),
                needle.unsafe_offset(1),
                needle_len - 1,
            )
            == 0
        ):
            return haystack.unsafe_offset(i)
    return {}


@always_inline
def _memrchr[
    dtype: DType
](
    source: ImmSpan[Scalar[dtype], _],
    char: Scalar[dtype],
) -> OptionalPointer[
    Scalar[dtype], source.origin
]:
    var base: Pointer[Scalar[dtype], source.origin] = source.unsafe_ptr()
    for i in reversed(range(len(source))):
        if source.unsafe_get(i) == char:
            return base.unsafe_offset(i)
    return {}


@always_inline
def _memrmem[
    dtype: DType
](
    haystack: ImmSpan[Scalar[dtype], _],
    needle: ImmSpan[Scalar[dtype], _],
) -> OptionalPointer[Scalar[dtype], haystack.origin]:
    var hp: Pointer[Scalar[dtype], haystack.origin] = haystack.unsafe_ptr()
    if not needle:
        return hp
    if len(needle) > len(haystack):
        return {}
    if len(needle) == 1:
        return _memrchr[dtype](haystack, needle.unsafe_get(0))
    if (
        __is_run_in_comptime_interpreter
        or len(haystack) < simd_width_of[DType.bool]()
    ):
        var haystack_ptr = haystack.unsafe_ptr()
        var needle_ptr = needle.unsafe_ptr()
        var needle_len = len(needle)
        for i in reversed(range(len(haystack) - needle_len + 1)):
            if haystack.unsafe_get(i) != needle.unsafe_get(0):
                continue
            if (
                unsafe_memcmp(
                    haystack_ptr.unsafe_offset(i + 1),
                    needle_ptr.unsafe_offset(1),
                    needle_len - 1,
                )
                == 0
            ):
                return haystack_ptr.unsafe_offset(i)
        return {}
    return _memrmem_impl(haystack, needle)


@always_inline
def _memrmem_impl[
    dtype: DType, //
](
    haystack_span: ImmSpan[Scalar[dtype], _],
    needle_span: ImmSpan[Scalar[dtype], _],
) -> OptionalPointer[Scalar[dtype], haystack_span.origin]:
    var haystack = haystack_span.unsafe_ptr()
    var haystack_len = len(haystack_span)
    var needle = needle_span.unsafe_ptr()
    var needle_len = len(needle_span)

    comptime bool_mask_width = simd_width_of[DType.bool]()
    var search_len = haystack_len - needle_len + 1
    var vectorized_end = align_down(search_len, bool_mask_width)

    var first_needle = SIMD[dtype, bool_mask_width](needle[unsafe_offset=0])
    var last_needle = SIMD[dtype, bool_mask_width](
        needle[unsafe_offset=needle_len - 1]
    )

    # Scalar tail: positions [vectorized_end, search_len) from right to left.
    for i in reversed(range(vectorized_end, search_len)):
        if haystack[unsafe_offset=i] != needle[unsafe_offset=0]:
            continue
        if (
            unsafe_memcmp(
                haystack.unsafe_offset(i + 1),
                needle.unsafe_offset(1),
                needle_len - 1,
            )
            == 0
        ):
            return haystack.unsafe_offset(i)

    # SIMD blocks from right to left: each block covers bool_mask_width
    # candidate positions. Within each block, process candidates from right
    # to left (highest index first) using count_leading_zeros.
    var i = vectorized_end - bool_mask_width
    while i >= 0:
        var first_block = haystack.unsafe_load[width=bool_mask_width](i)
        var last_block = haystack.unsafe_load[width=bool_mask_width](
            i + needle_len - 1
        )
        var bool_mask = first_needle.eq(first_block) & last_needle.eq(
            last_block
        )
        var mask = pack_bits(bool_mask)

        while mask:
            var pos_in_block = (
                type_of(mask)(bool_mask_width - 1) - count_leading_zeros(mask)
            )
            var offset = i + Int(pos_in_block)
            if (
                unsafe_memcmp(
                    haystack.unsafe_offset(offset + 1),
                    needle.unsafe_offset(1),
                    needle_len - 1,
                )
                == 0
            ):
                return haystack.unsafe_offset(offset)
            mask = mask ^ (type_of(mask)(1) << pos_in_block)

        i -= bool_mask_width

    return {}


def _split[
    has_maxsplit: Bool
](
    src_str: StringSpan,
    sep: StringSpan,
    maxsplit: Int,
    out output: List[type_of(src_str).Immutable],
):
    comptime S = type_of(src_str).Immutable
    var ptr = src_str.unsafe_ptr().as_imm()
    var sep_len = sep.byte_length()
    if sep_len == 0:
        var iterator = src_str.codepoint_slices()
        var i_len = len(iterator) + 2
        output = {capacity = i_len}
        output.append(S(unsafe_from_utf8=Span(unsafe_ptr=ptr, length=0)))
        for s in iterator:
            output.append(s)
        output.append(
            S(
                unsafe_from_utf8=Span(
                    unsafe_ptr=ptr.unsafe_offset(i_len - 1), length=0
                )
            )
        )
        return

    comptime prealloc = 32  # guessing, Python's implementation uses 12
    var amnt = prealloc

    comptime if has_maxsplit:
        amnt = maxsplit + 1 if maxsplit < prealloc else prealloc
    output = {capacity = amnt}
    var str_byte_len = src_str.byte_length()
    var lhs = 0
    var rhs: Int
    var items = 0
    # var str_span = src_str.as_bytes() # FIXME: solve #3526 with #3548
    # var sep_span = sep.as_bytes() # FIXME: solve #3526 with #3548

    while lhs <= str_byte_len:
        # FIXME(#3526): use str_span and sep_span
        rhs = src_str.find(sep, lhs)
        # if not found go to the end
        rhs += is_negative(rhs) & (str_byte_len + 1)

        comptime if has_maxsplit:
            rhs += splat(items == maxsplit) & (str_byte_len - rhs)
            items += 1

        output.append(
            S(
                unsafe_from_utf8=Span(
                    unsafe_ptr=ptr.unsafe_offset(lhs), length=rhs - lhs
                )
            )
        )
        lhs = rhs + sep_len


def _split[
    has_maxsplit: Bool
](
    src_str: StringSpan,
    sep: NoneType,
    maxsplit: Int,
    out output: List[type_of(src_str).Immutable],
):
    comptime S = type_of(src_str).Immutable
    comptime prealloc = 32  # guessing, Python's implementation uses 12
    var amnt = prealloc

    comptime if has_maxsplit:
        amnt = maxsplit + 1 if maxsplit < prealloc else prealloc
    output = {capacity = amnt}
    var str_byte_len = src_str.byte_length()
    var lhs = 0
    var rhs: Int
    var items = 0
    var ptr = src_str.unsafe_ptr().as_imm()

    comptime PointerType = type_of(ptr)

    @always_inline("nodebug")
    def _build_slice(p: PointerType, start: Int, end: Int) -> S:
        return S(
            unsafe_from_utf8=Span(
                unsafe_ptr=p.unsafe_offset(start), length=end - start
            )
        )

    while lhs <= str_byte_len:
        # Python adds all "whitespace chars" as one separator
        # if no separator was specified
        for s in _build_slice(ptr, lhs, str_byte_len).codepoint_slices():
            if not s.isspace[single_character=True]():
                break
            lhs += s.byte_length()
        # if it went until the end of the String, then it should be sliced
        # until the start of the whitespace which was already appended
        if lhs == str_byte_len:
            break
        rhs = lhs + _utf8_first_byte_sequence_length(ptr[unsafe_offset=lhs])
        for s in _build_slice(ptr, rhs, str_byte_len).codepoint_slices():
            if s.isspace[single_character=True]():
                break
            rhs += s.byte_length()

        comptime if has_maxsplit:
            rhs += splat(items == maxsplit) & (str_byte_len - rhs)
            items += 1

        output.append(
            S(
                unsafe_from_utf8=Span(
                    unsafe_ptr=ptr.unsafe_offset(lhs), length=rhs - lhs
                )
            )
        )
        lhs = rhs
