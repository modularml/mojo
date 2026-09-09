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
"""Implements the core `String` type and related utilities."""

from std.builtin.globals import global_constant
from std.collections import KeyElement, Span, check_slice_bounds
from std.collections.string import CodepointsIter
from std.collections.string._parsing_numbers.parsing_floats import _atof
from std.collections.string._utf8 import UTF8Chunks, _is_valid_utf8
from std.collections.string.format import _FormatUtils
from std.collections.string.string_span import (
    BytesIter,
    CodepointSliceIter,
    GraphemeIndicesIter,
    GraphemeSliceIter,
    _to_string_list,
    _unsafe_strlen,
)
from std.builtin.builtin_slice import ContiguousSlice
from std.hashlib.hasher import Hasher
from std.reflection import call_location
from std.format.tstring import TString
from std.format._utils import (
    FLUSHING_WRITE_BUFFER_BYTES,
    _TotalWritableBytes,
    _FlushingWriteBuffer,
)
from std.os import PathLike, abort
from std.atomic import Atomic, Ordering, fence
from std.sys import size_of, bit_width_of
from std.ffi import c_char, CStringSlice
from std.sys.info import is_32bit, is_apple_gpu

from std.bit import count_leading_zeros
from std.memory import (
    Layout,
    ThinAllocation,
    dealloc,
    unsafe_memcmp,
    unsafe_memcpy,
    unsafe_memset,
)
from std.python import ConvertibleFromPython, PythonObject

# ===----------------------------------------------------------------------=== #
# String
# ===----------------------------------------------------------------------=== #


@stable(since="1.0")
struct String(
    Boolable,
    Comparable,
    ConvertibleFromPython,
    Defaultable,
    FloatableRaising,
    ImplicitlyCopyable,
    IntableRaising,
    KeyElement,
    PathLike,
    Writable,
    Writer,
):
    """Represents a mutable string.

    This is Mojo's primary text representation, designed to efficiently handle
    UTF-8 encoded text while providing a safe and ergonomic interface for
    string manipulation.

    You can create a `String` by assigning a string literal to a variable or
    with the `String` constructor:

    ```mojo
    # From string literals (String type is inferred)
    var hello = "Hello"

    # From String constructor
    var world = String("World")
    print(hello, world)    # "Hello World"
    ```

    You can convert many Mojo types to a `String` because it's common to
    implement the [`Writable`](/docs/std/format/Writable/) trait:

    ```mojo
    var int : Int = 42
    print(String(int))    # "42"
    ```

    If you have a custom type you want to convert to a string, you can implement
    the [`Writable`](/docs/std/format/Writable/) trait like this:

    ```mojo
    @fieldwise_init
    struct Person(Writable):
        var name: String
        var age: Int

        def write_to(self, mut writer: Some[Writer]):
            t"{self.name} ({self.age})".write_to(writer)

    var person = Person("Alice", 30)
    print(String(person))      # => Alice (30)
    ```

    However, `print()` doesn't actually specify `String` as its argument type.
    Instead, it accepts any type that conforms to the
    [`Writable`](/docs/std/format/Writable/) trait (`String` conforms to
    this trait, which is why you can pass it to `print()`). That means it's
    actually more efficient to pass any type that implements `Writable`
    directly to `print()` (instead of first converting it to `String`).
    For example, float types are also writable:

    ```mojo
    var float : Float32 = 3.14
    print(float)
    ```

    Be aware of the following characteristics when working with `String`:

    - **UTF-8 encoding**: Strings store UTF-8 encoded text, so byte length may
      differ from character count. Use `text.count_codepoints()` to get
      the codepoint count:

      ```mojo
      var text = "café"                # 4 Unicode characters
      print(text.byte_length())        # Prints 5 (é is 2 bytes in UTF-8)
      print(text.count_codepoints())   # Prints 4 (correct Unicode count)
      ```

    - **Always mutable**: You can modify strings in-place:

      ```mojo
      var message = "Hello"
      message += " World"        # In-place concatenation
      print(message)             # "Hello World"
      ```

      If you want a compile-time immutable string, use `comptime`:

      ```mojo
      comptime GREETING = "Immutable string"  # Fixed at compile time
      GREETING = "Not gonna happen"        # error: expression must be mutable in assignment
      ```

    - **Value semantics**: String assignment creates a copy, but it's optimized
    with copy-on-write so that the actual copying happens only if/when one of
    the strings is modified.

      ```mojo
      var str1 = "Hello"
      var str2 = str1            # Currently references the same data
      str2 += " World"           # Now str2 becomes a copy of str1
      print(str1)                # "Hello"
      print(str2)                # "Hello World"
      ```

    More examples:

    ```mojo
    var text = "Hello"

    # String properties and indexing
    print(text.byte_length())                 # 5
    print(text[byte=1])                       # e (byte slice)
    print(text[byte=text.byte_length() - 1])  # o (last character)

    # In-place concatenation
    text += " World"
    print(text)

    # Searching and checking
    if "World" in text:
        print("Found 'World' in text")

    var pos = text.find("World")
    if pos != -1:
        print("'World' found at position:", pos)

    # String replacement
    var replaced = text.replace("Hello", "Hi")   # "Hi World"
    print(replaced)

    # String formatting
    var name = "Alice"
    var age = 30
    var formatted = "{} is {} years old".format(name, age)
    print(formatted)    # "Alice is 30 years old"
    ```

    Related functions:

    - String-to-number conversions:
      [`atof()`](/docs/std/collections/string/string/atof/),
      [`atol()`](/docs/std/collections/string/string/atol/)).
    - Character code conversions:
      [`chr()`](/docs/std/collections/string/string/chr/),
      [`ord()`](/docs/std/collections/string/string/ord/)).
    - String formatting:
      [`format()`](/docs/std/collections/string/string/String/#format).

    Related types:

    - [`StringSpan`](/docs/std/collections/string/string_span/StringSpan/): A non-owning
      view of string data, which can be either mutable or immutable.
    - [`StaticString`](/docs/std/collections/string/string_span/#StaticString): An
      alias for an immutable constant `StringSpan`.
    - [`StringLiteral`](/docs/std/builtin/string_literal/StringLiteral/): A
      string literal. String literals are compile-time values.
    """

    # Fields: String has two forms - the declared form here, and the "inline"
    # form when '_capacity_or_data.is_inline()' is true. The inline form
    # clobbers these fields (except the top byte of the capacity field) with
    # the string data.
    var _ptr_or_data: Pointer[UInt8, MutUntrackedOrigin]
    """The underlying storage for the string data."""
    var _len_or_data: Int
    """The number of bytes in the string data."""
    var _capacity_or_data: Int
    """The capacity and bit flags for this String."""

    # Useful string aliases.
    comptime ASCII_LOWERCASE = "abcdefghijklmnopqrstuvwxyz"
    """All lowercase ASCII letters."""

    comptime ASCII_UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    """All uppercase ASCII letters."""

    comptime ASCII_LETTERS = Self.ASCII_LOWERCASE + Self.ASCII_UPPERCASE
    """All ASCII letters (lowercase and uppercase)."""

    comptime DIGITS = "0123456789"
    """All decimal digit characters."""

    comptime HEX_DIGITS = Self.DIGITS + "abcdef" + "ABCDEF"
    """All hexadecimal digit characters."""

    comptime OCT_DIGITS = "01234567"
    """All octal digit characters."""

    comptime PUNCTUATION = """!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""
    """All ASCII punctuation characters."""

    comptime PRINTABLE = Self.DIGITS + Self.ASCII_LETTERS + Self.PUNCTUATION + " \t\n\r\v\f"
    """All printable ASCII characters."""

    # ===------------------------------------------------------------------=== #
    # String Implementation Details
    # ===------------------------------------------------------------------=== #
    # This is the number of bytes that can be stored inline in the string value.
    # 'String' is 3 words in size and we use the top byte of the capacity field
    # to store flags.
    comptime INLINE_CAPACITY = bit_width_of[DType.int]() // 8 * 3 - 1
    """Maximum bytes for inline (SSO) string storage."""

    # When FLAG_HAS_NUL_TERMINATOR is set, the byte past the end of the string
    # is known to be an accessible 'nul' terminator.
    comptime FLAG_HAS_NUL_TERMINATOR = 1 << (bit_width_of[DType.int]() - 3)
    """Flag indicating string has accessible nul terminator."""

    # When FLAG_IS_REF_COUNTED is set, the string is pointing to a mutable buffer
    # that may have other references to it.
    comptime FLAG_IS_REF_COUNTED = 1 << (bit_width_of[DType.int]() - 2)
    """Flag indicating string uses reference-counted storage."""

    # When FLAG_IS_INLINE is set, the string is inline or "Short String
    # Optimized" (SSO). The first 23 bytes of the fields are treated as UTF-8
    # data
    comptime FLAG_IS_INLINE = 1 << (bit_width_of[DType.int]() - 1)
    """Flag indicating string uses inline (SSO) storage."""

    # gives us 5 bits for the length.
    comptime INLINE_LENGTH_START = bit_width_of[DType.int]() - 8
    """Bit position where inline length field starts."""

    comptime INLINE_LENGTH_MASK = 0b1_1111 << Self.INLINE_LENGTH_START
    """Bit mask for extracting inline string length."""

    # This is the size to offset the pointer by, to get access to the
    # atomic reference count prepended to the UTF-8 data.
    comptime REF_COUNT_SIZE = size_of[Atomic[Int]]()
    """Size of the reference count prefix for heap strings."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    @stable(since="1.0")
    @always_inline("nodebug")
    def __deinit__(deinit self):
        """Destroy the string data."""
        self._drop_ref()

    @stable(since="1.1")
    @always_inline("nodebug")
    def __init__(out self):
        """Construct an empty string."""
        # this is UB if we ever touch the pointer, but so is
        # an uninitialized pointer
        self._ptr_or_data = {unsafe_from_address = Int(0)}
        self._len_or_data = 0
        self._capacity_or_data = Self.FLAG_IS_INLINE

    @stable(since="1.1")
    @always_inline("nodebug")
    def __init__(out self, *, capacity_bytes: Int):
        """Construct an empty string with at least a given capacity.

        Args:
            capacity_bytes: The minimum capacity (in bytes) to allocate.
        """
        if capacity_bytes <= Self.INLINE_CAPACITY:
            self = Self()
        else:
            self._capacity_or_data = (capacity_bytes + 7) >> 3
            self._ptr_or_data = Self._alloc(self._capacity_or_data << 3)
            self._len_or_data = 0
            self._set_ref_counted()

    @always_inline("nodebug")
    @stable(since="1.0")
    @implicit  # does not allocate.
    def __init__(out self, data: StaticString, /):
        """Construct a `String` from a `StaticString` without allocating.

        Args:
            data: The static constant string to refer to.
        """
        self._len_or_data = data._slice._len
        # TODO: Validate the safety of this.
        # Safety: This should be safe since we set `capacity_or_data` to 0.
        # Meaning any mutation will cause us to either reallocate or inline
        # the string.
        self._ptr_or_data = data._slice._data.unsafe_mut_cast[
            True
        ]().unsafe_origin_cast[MutUntrackedOrigin]()
        # Always use static constant representation initially, defer inlining
        # decision until mutation to avoid unnecessary unsafe_memcpy.
        self._capacity_or_data = 0

    @always_inline("nodebug")
    @stable(since="1.0")
    @implicit  # does not allocate.
    def __init__(out self, data: StringLiteral, /):
        """Construct a `String` from a `StringLiteral` without allocating.

        Args:
            data: The static constant string to refer to.
        """
        self._len_or_data = Int(
            SIMDLength(mlir_value=__mlir_op.`pop.string.size`(data.value))
        )
        self._ptr_or_data = Pointer[_, MutUntrackedOrigin](
            _mlir_value=__mlir_op.`pop.string.address`(data.value)
        ).unsafe_bitcast[Byte]()
        # Always use static constant representation initially, defer inlining
        # decision until mutation to avoid unnecessary unsafe_memcpy.
        self._capacity_or_data = Self.FLAG_HAS_NUL_TERMINATOR

    def __init__(out self, tstring: TString):
        """Construct a string from a TString (template string).

        This constructor enables explicit conversion from TString to String.

        Args:
            tstring: The TString to convert to a String.
        """
        self = Self()
        tstring.write_to(self)

    def __init__(out self, *, unsafe_from_utf8: Span[Byte, _]):
        """Construct a string by copying the data. This constructor is explicit
        because it can involve memory allocation.

        Consider using the `String(from_utf8=...)` or
        `String(from_utf8_lossy=...)` constructors instead, as they are safer
        alternatives to the `unsafe_from_utf8` constructor.

        Args:
            unsafe_from_utf8: The utf8 bytes to copy.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.
        """
        assert _is_valid_utf8(
            unsafe_from_utf8
        ), "String: span is not valid UTF-8"
        var length = len(unsafe_from_utf8)
        self = Self(unsafe_uninit_length=length)
        unsafe_memcpy(
            dest=self.unsafe_ptr_mut(),
            src=unsafe_from_utf8.unsafe_ptr(),
            count=length,
        )

    @stable(since="1.0")
    def __init__(out self, *, from_utf8_lossy: Span[Byte, _]):
        """Construct a string from a span of bytes, including invalid UTF-8.

        Since `String` is guaranteed to be valid UTF-8, invalid UTF-8 sequences
        are replaced with the `U+FFFD` replacement character: `�`.

        Args:
            from_utf8_lossy: The bytes to convert to a string.

        Examples:

        ```mojo
        from std.testing import assert_equal

        # Valid UTF-8 sequence
        var fire_emoji_bytes = [Byte(0xF0), 0x9F, 0x94, 0xA5]
        var fire_emoji = String(from_utf8_lossy=fire_emoji_bytes)
        assert_equal(fire_emoji, "🔥")

        # Invalid UTF-8 sequence
        # "mojo<invalid sequence>"
        var mojo_bytes = [Byte(0x6D), 0x6F, 0x6A, 0x6F, 0xF0, 0x90, 0x80]
        var mojo = String(from_utf8_lossy=mojo_bytes)
        assert_equal(mojo, "mojo�")
        ```
        """

        comptime REPLACEMENT = StaticString("�")

        self = String(capacity_bytes=len(from_utf8_lossy))
        for chunk in UTF8Chunks(from_utf8_lossy):
            self += chunk.valid
            if len(chunk.invalid) > 0:
                self += REPLACEMENT

    def __init__(out self, *, from_utf8: Span[Byte, _]) raises:
        """Construct a string from a span of bytes, raising an error if the data
        is not valid UTF-8.

        Args:
            from_utf8: The bytes to convert to a string.

        Raises:
            An error if the data is not valid UTF-8.
        """
        if not _is_valid_utf8(from_utf8):
            raise Error("Cannot construct a String from invalid UTF-8 data")
        self = String(unsafe_from_utf8=from_utf8)

    # ===------------------------------------------------------------------=== #
    # Writables
    # ===------------------------------------------------------------------=== #
    # There is duplication here to avoid passing around variadic packs in such
    # a common callsite, as that isn't free for both compilation speed and
    # register pressure.

    def __init__[
        *Ts: Writable,
    ](out self, *args: *Ts, sep: StaticString = "", end: StaticString = ""):
        """
        Construct a string by concatenating a sequence of Writable arguments.

        Args:
            args: A sequence of Writable arguments.
            sep: The separator used between elements.
            end: The String to write after printing the elements.

        Parameters:
            Ts: Types of the provided argument sequence.

        Examples:

        Construct a String from several `Writable` arguments:

        ```mojo
        var string = String(1, 2.0, "three", sep=", ")
        print(string) # "1, 2.0, three"
        ```

        Forwarding from another variadic function:

        ```mojo
        def variadic_pack_to_string[*Ts: Writable](*args: *Ts) -> String:
            return String(*args)

        _ = variadic_pack_to_string(1, ", ", 2.0, ", ", "three")
        ```
        """
        comptime assert Ts.all_conforms_to[Writable]()  # satisfy where clause.

        var total_bytes = _TotalWritableBytes()
        args._write_to(total_bytes, end=end, sep=sep)

        if total_bytes.size <= Self.INLINE_CAPACITY:
            self = String()
            args._write_to(self, end=end, sep=sep)
        else:
            self = String(capacity_bytes=total_bytes.size)
            var buffer = _FlushingWriteBuffer[FLUSHING_WRITE_BUFFER_BYTES](self)
            args._write_to(buffer, end=end, sep=sep)
            buffer.flush()

    def write[*Ts: Writable](mut self, *args: *Ts):
        """Write a sequence of Writable arguments to the provided Writer.

        Parameters:
            Ts: Types of the provided argument sequence.

        Args:
            args: Sequence of arguments to write to this Writer.
        """
        comptime assert Ts.all_conforms_to[Writable]()  # satisfy where clause.

        var total_bytes = _TotalWritableBytes()
        total_bytes.size += self.byte_length()
        args._write_to(total_bytes, sep="")

        if total_bytes.size <= Self.INLINE_CAPACITY:
            args._write_to(self, sep="")
        else:
            self.reserve_bytes(total_bytes.size)
            var buffer = _FlushingWriteBuffer[FLUSHING_WRITE_BUFFER_BYTES](self)
            args._write_to(buffer, sep="")
            buffer.flush()

    def write[T: Writable](mut self, value: T):
        """Write a single Writable argument to the provided Writer.

        Parameters:
            T: The type of the value to write, which must implement `Writable`.

        Args:
            value: The `Writable` argument to write.
        """
        value.write_to(self)

    @always_inline("nodebug")
    def __init__(out self, *, unsafe_uninit_length: Int):
        """Construct a String with the specified length, with uninitialized
        memory. This is unsafe, as it relies on the caller initializing the
        elements with unsafe operations, not assigning over the uninitialized
        data.

        Args:
            unsafe_uninit_length: The number of bytes to allocate.
        """
        self = Self(capacity_bytes=unsafe_uninit_length)
        self._set_byte_length(unsafe_uninit_length)

    def __init__(
        out self,
        *,
        unsafe_from_utf8_ptr: ImmPointer[c_char, _],
    ):
        """Creates a string from a UTF-8 encoded nul-terminated pointer.

        Args:
            unsafe_from_utf8_ptr: A `Pointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        # Copy the data.
        self = String(
            StringSlice(
                unsafe_from_utf8=CStringSlice(
                    unsafe_from_ptr=unsafe_from_utf8_ptr.unsafe_bitcast[Int8]()
                )
            )
        )

    def __init__(out self, *, unsafe_from_utf8_ptr: ImmPointer[UInt8, _]):
        """Creates a string from a UTF-8 encoded nul-terminated pointer.

        Args:
            unsafe_from_utf8_ptr: A `Pointer[Byte]` of null-terminated bytes encoded in UTF-8.

        Safety:
            - `unsafe_from_utf8_ptr` MUST be valid UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` MUST be null terminated.
        """
        # Copy the data.
        self = String(
            StringSlice(
                unsafe_from_utf8=CStringSlice(
                    unsafe_from_ptr=unsafe_from_utf8_ptr.unsafe_bitcast[Int8]()
                )
            )
        )

    @stable(since="1.0")
    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Copy initialize the string from another string.

        Args:
            copy: The string to copy.
        """
        # Keep inline strings inline, and static strings static.
        self._ptr_or_data = copy._ptr_or_data
        self._len_or_data = copy._len_or_data
        self._capacity_or_data = copy._capacity_or_data

        # Increment the refcount if it has a mutable buffer.
        self._add_ref()

    # ===------------------------------------------------------------------=== #
    # Capacity Field Helpers
    # ===------------------------------------------------------------------=== #

    # This includes getting and setting flags from the capacity field such as
    # null terminator, inline, and indirect. If indirect the length is also
    # stored in the capacity field.

    @deprecated(use=capacity_bytes)
    @doc_hidden
    def capacity(self) -> Int:
        return self.capacity_bytes()

    @always_inline("nodebug")
    def capacity_bytes(self) -> Int:
        """Get the current capacity of the `String`'s internal buffer.

        Returns:
            The number of bytes that can be stored before reallocation is needed.
        """
        # Max inline capacity before reallocation.
        if self._is_inline():
            return Self.INLINE_CAPACITY
        if not self._is_ref_counted():
            return self._len_or_data
        return self._capacity_or_data << 3

    @always_inline("nodebug")
    def _set_nul_terminated(mut self):
        self._capacity_or_data |= Self.FLAG_HAS_NUL_TERMINATOR

    @always_inline("nodebug")
    def _has_nul_terminator(self) -> Bool:
        return Bool(
            UInt64(self._capacity_or_data)
            & UInt64(Self.FLAG_HAS_NUL_TERMINATOR)
        )

    @always_inline("nodebug")
    def _clear_nul_terminator(mut self):
        self._capacity_or_data &= ~Self.FLAG_HAS_NUL_TERMINATOR

    @always_inline("nodebug")
    def _is_inline(self) -> Bool:
        return Bool(
            UInt64(self._capacity_or_data) & UInt64(Self.FLAG_IS_INLINE)
        )

    @always_inline("nodebug")
    def _set_ref_counted(mut self):
        self._capacity_or_data |= Self.FLAG_IS_REF_COUNTED

    @always_inline("nodebug")
    def _is_ref_counted(self) -> Bool:
        return Bool(
            UInt64(self._capacity_or_data) & UInt64(Self.FLAG_IS_REF_COUNTED)
        )

    # ===------------------------------------------------------------------=== #
    # Pointer Field Helpers
    # ===------------------------------------------------------------------=== #

    # This includes helpers for the allocated atomic ref count used for
    # out-of-line strings, which is stored before the UTF-8 data.

    @always_inline("nodebug")
    def _refcount(self) -> ref[self._ptr_or_data.origin] Atomic[Int]:
        # The header is stored before the string data.
        return self._ptr_or_data.unsafe_offset(
            -Self.REF_COUNT_SIZE
        ).unsafe_bitcast[Atomic[Int]]()[]

    @always_inline("nodebug")
    def _is_unique(mut self) -> Bool:
        """Return true if the refcount is 1."""
        if UInt64(self._capacity_or_data) & UInt64(Self.FLAG_IS_REF_COUNTED):
            return self._refcount().load[ordering=Ordering.RELAXED]() == 1
        else:
            return False

    @always_inline("nodebug")
    def _add_ref(mut self):
        """Atomically increment the refcount."""
        comptime if is_apple_gpu():
            # No-op: heap-string refcounting is unsupported on Apple GPU.
            # See MSTDL-2907.
            return
        if self._capacity_or_data & Self.FLAG_IS_REF_COUNTED:
            # See `ArcPointer`'s refcount implementation for more details on the
            # use of memory orderings.
            _ = self._refcount().fetch_add[ordering=Ordering.RELAXED](1)

    @always_inline("nodebug")
    def _drop_ref(mut self):
        """Atomically decrement the refcount and deallocate self if the result
        hits zero."""
        comptime if is_apple_gpu():
            # No-op: heap-string refcounting is unsupported on Apple GPU.
            # See MSTDL-2907.
            return
        # If indirect or inline we don't need to do anything.
        if self._capacity_or_data & Self.FLAG_IS_REF_COUNTED:
            var ptr = self._ptr_or_data.unsafe_offset(-Self.REF_COUNT_SIZE)
            var refcount = ptr.unsafe_bitcast[Atomic[Int]]()
            if refcount[].fetch_sub(1) == 1:
                fence[Ordering.ACQUIRE]()
                dealloc(
                    ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(
                        Layout[Byte](
                            count=self.capacity_bytes() + Self.REF_COUNT_SIZE
                        )
                    )
                )

    @staticmethod
    def _alloc(capacity: Int) -> Pointer[Byte, MutUntrackedOrigin]:
        """Allocate space for a new out-of-line string buffer."""
        var ptr = alloc(
            Layout[Byte](count=capacity + Self.REF_COUNT_SIZE)
        ).unsafe_leak()

        # Initialize the Atomic refcount into the header.
        ptr.unsafe_bitcast[Atomic[Int]]().unsafe_write(
            init_with=lambda () -> Atomic[Int]: Atomic[Int](1)
        )

        # Return a pointer to right after the header, which is where the string
        # data will be stored.
        return ptr.unsafe_offset(Self.REF_COUNT_SIZE)

    # ===------------------------------------------------------------------=== #
    # Factory dunders
    # ===------------------------------------------------------------------=== #

    def write_string(mut self, string: StringSlice):
        """
        Write a `StringSlice` to this `String`.

        Args:
            string: The `StringSlice` to write to this String.
        """
        self._iadd(string.as_bytes())

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    @doc_hidden
    @unavailable(
        "String does not support direct positional indexing like `s[i]`"
        " because Mojo strings are UTF-8 encoded, and the same position can"
        " mean three different things. Use one of: `s[byte=i]` for a raw"
        " UTF-8 byte, `s[codepoint=i]` for a Unicode code point, or"
        " `s[grapheme=i]` for a user-visible character (grapheme cluster)."
    )
    def __getitem__(
        self, _index: Some[Indexer], /
    ) -> StringSlice[origin_of(self)]:
        ...

    @doc_hidden
    @unavailable(
        "String does not support direct positional slicing like `s[a:b]`"
        " because Mojo strings are UTF-8 encoded, and the same range can"
        " mean different things. Use `s[byte=a:b]` to slice by raw UTF-8"
        " byte positions, or `s[codepoint=a:b]` to slice by Unicode code"
        " points."
    )
    def __getitem__(
        self, _slice: ContiguousSlice, /
    ) -> StringSlice[origin_of(self)]:
        ...

    @doc_hidden
    @unavailable(
        "String does not support `__len__` because Mojo strings are UTF-8"
        " encoded, so a single length is ambiguous: it could mean the number"
        " of UTF-8 bytes, the number of Unicode code points, or the number of"
        " user-visible characters (grapheme clusters). Use `s.byte_length()`"
        " or `len(s.bytes())` for the number of UTF-8 bytes,"
        " `len(s.codepoints())` for Unicode code points, or"
        " `len(s.graphemes())` for grapheme clusters."
    )
    def __len__(self) -> Int:
        ...

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        self, *, byte: Int
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Gets a single byte at the specified byte index.

        This performs byte-level indexing, not character (codepoint) indexing.
        For strings containing multi-byte UTF-8 characters `byte` must fall on
        a codepoint boundary and an entire codepoint will be returned.
        Aborts if `byte` does not fall on a codepoint boundary.

        Args:
            byte: The byte index (0-based).

        Returns:
            A StringSlice containing a single byte at the specified position.
        """
        var string_slice = self._interior_slice()
        var idx = index(byte)
        string_slice._check_valid_index(idx)
        return string_slice._unchecked_get_byte(idx)

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__[
        I: Indexer, //
    ](self, *, byte: I) -> StringSlice[
        origin_of(self)._get_owned_interior["bytes"]
    ]:
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
            A StringSlice containing a single byte at the specified position.
        """
        var string_slice = self._interior_slice()
        var idx = index(byte)
        string_slice._check_valid_index(idx)
        return string_slice._unchecked_get_byte(idx)

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        self, *, byte: IntLiteral
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Gets a single byte at the specified byte index.

        This performs byte-level indexing, not character (codepoint) indexing.
        For strings containing multi-byte UTF-8 characters `byte` must fall on
        a codepoint boundary and an entire codepoint will be returned.
        Aborts if `byte` does not fall on a codepoint boundary.

        Args:
            byte: The byte index (0-based).

        Returns:
            A StringSlice containing a single byte at the specified position.
        """
        comptime assert IntLiteral[byte.value]() >= 0, (
            "negative indexing is not supported, use e.g."
            " `string[byte=string.byte_length() - 1]`"
        )
        var string_slice = self._interior_slice()
        string_slice._check_valid_index(byte)
        return string_slice._unchecked_get_byte(byte)

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        self, *, byte: ContiguousSlice
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Gets a substring at the specified byte positions.

        This performs byte-level slicing, not character (codepoint) slicing.
        The start and end positions are byte indices. For strings containing
        multi-byte UTF-8 characters, slicing at byte positions that do not fall
        on codepoint boundaries will abort.

        Aborts if `byte`'s start or end index is out of bounds (valid range
        is `0` to `self.byte_length()`, inclusive), or if start is greater
        than end. Negative indices are not supported and always abort.

        Args:
            byte: A slice that specifies byte positions of the new substring.

        Returns:
            A StringSlice containing the bytes in the specified range.
        """
        _ = check_slice_bounds(byte, self.byte_length(), call_location())
        return self._interior_slice()[byte=byte]

    @__unsafe_nested_origins_read_only
    def __getitem__[
        I: Indexer, //
    ](self, *, codepoint: I) -> StringSlice[
        origin_of(self)._get_owned_interior["bytes"]
    ]:
        """Gets the character at the specified position.

        Parameters:
            I: A type that can be used as an index.

        Args:
            codepoint: The codepoint index.

        Returns:
            A `StringSlice` view containing the unicode codepoint at the
            specified position.
        """
        return self._interior_slice()[codepoint=codepoint]

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        self, *, codepoint: ContiguousSlice
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Gets a substring at the specified codepoint positions.

        Args:
            codepoint: A slice that specifies codepoint positions of the new
                substring.

        Returns:
            A StringSlice containing the codepoints in the specified range.
        """
        return self._interior_slice()[codepoint=codepoint]

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        self, *, grapheme: Some[Indexer]
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Gets the character at the specified position.

        Args:
            grapheme: The grapheme index.

        Returns:
            A `StringSlice` view containing the unicode grapheme at the
            specified position.
        """
        return self._interior_slice()[grapheme=grapheme]

    @stable(since="1.0")
    def __eq__(self, rhs: String) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            rhs: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        # Early exit if lengths differ
        var self_len = self.byte_length()
        var other_len = rhs.byte_length()
        if self_len != other_len:
            return False
        var self_ptr = self.unsafe_ptr()
        var rhs_ptr = rhs.unsafe_ptr()
        # same pointer and length, so equal
        if self_len == 0 or self_ptr == rhs_ptr:
            return True
        # Compare memory directly
        return unsafe_memcmp(self_ptr, rhs_ptr, self_len) == 0

    @always_inline("nodebug")
    @stable(since="1.0")
    def __eq__(self, other: StringSlice) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return StringSlice(self) == other

    @always_inline("nodebug")
    @stable(since="1.0")
    def __ne__(self, other: StringSlice) -> Bool:
        """Compares two Strings if they have the same values.

        Args:
            other: The rhs of the operation.

        Returns:
            True if the Strings are equal and False otherwise.
        """
        return StringSlice(self) != other

    @always_inline("nodebug")
    def __lt__(self, rhs: String) -> Bool:
        """Compare this String to the RHS using LT comparison.

        Args:
            rhs: The other String to compare against.

        Returns:
            True if this String is strictly less than the RHS String and False
            otherwise.
        """
        return StringSlice(self) < StringSlice(rhs)

    @staticmethod
    def _add(lhs: Span[Byte, _], rhs: Span[Byte, _]) -> String:
        var lhs_len = len(lhs)
        var rhs_len = len(rhs)

        var result = String(unsafe_uninit_length=lhs_len + rhs_len)
        var result_ptr = result.unsafe_ptr_mut()
        unsafe_memcpy(dest=result_ptr, src=lhs.unsafe_ptr(), count=lhs_len)
        unsafe_memcpy(
            dest=result_ptr.unsafe_offset(lhs_len),
            src=rhs.unsafe_ptr(),
            count=rhs_len,
        )
        return result^

    def __add__(self, other: StringSlice) -> String:
        """Creates a string by appending a string slice at the end.

        Args:
            other: The string slice to append.

        Returns:
            The new constructed string.
        """
        return Self._add(self.as_bytes(), other.as_bytes())

    def _unsafe_append_byte(mut self, byte: Byte):
        """Appends a byte to the string assuming the capacity is sufficient.

        This helper is inherently unsafe as it does not check if the capacity is
        sufficient and does not check UTF-8 validity.
        """
        assert (
            self.capacity_bytes() > self.byte_length()
        ), "String: capacity is not sufficient"
        var length = self.byte_length()
        self.unsafe_ptr_mut().unsafe_offset(length).write(byte)
        self._set_byte_length(length + 1)

    def append(mut self, codepoint: Codepoint):
        """Append a codepoint to the string.

        Args:
            codepoint: The codepoint to append.
        """
        self._clear_nul_terminator()
        var length = self.byte_length()
        var new_length = length + codepoint.utf8_byte_length()
        self.reserve_bytes(new_length)
        _ = codepoint.unsafe_write_utf8(
            self.unsafe_ptr_mut().unsafe_offset(length)
        )
        self._set_byte_length(new_length)

    def __radd__(self, other: StringSlice[mut=False, _]) -> String:
        """Creates a string by prepending another string slice to the start.

        Args:
            other: The string to prepend.

        Returns:
            The new constructed string.
        """
        return Self._add(other.as_bytes(), self.as_bytes())

    def _iadd(mut self, other: ImmSpan[Byte, _]):
        var other_len = len(other)
        if other_len == 0:
            return
        var old_len = self.byte_length()
        var new_len = old_len + other_len
        unsafe_memcpy(
            dest=self.unsafe_ptr_mut(new_len).unsafe_offset(old_len),
            src=other.unsafe_ptr(),
            count=other_len,
        )
        self._set_byte_length(new_len)
        self._clear_nul_terminator()

    def __iadd__(mut self, other: StringSlice[mut=False, _]):
        """Appends another string slice to this string.

        Args:
            other: The string to append.
        """
        self._iadd(other.as_bytes())

    def __iter__(self) -> GraphemeSliceIter[origin_of(self)]:
        """Iterate over the grapheme clusters in the string.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen. See `graphemes()` for the precise
        definition.

        To iterate by Unicode codepoint or by byte instead, use
        `codepoints()`/`codepoint_slices()` or `bytes()`.

        Returns:
            An iterator yielding each grapheme cluster as a `StringSlice`.
        """
        return self.graphemes()

    def __reversed__(self) -> GraphemeSliceIter[origin_of(self), False]:
        """Iterate backwards over the grapheme clusters in the string.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen. See `graphemes()` for the precise
        definition.

        Returns:
            A reverse iterator yielding each grapheme cluster as a
            `StringSlice`.
        """
        return self.graphemes_reversed()

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    def __bool__(self) -> Bool:
        """Checks if the string is not empty.

        Returns:
            True if the string length is greater than zero, and False otherwise.
        """
        return self.byte_length() > 0

    @always_inline("nodebug")
    def __fspath__(self) -> String:
        """Return the file system path representation (just the string itself).

        Returns:
          The file system path representation as a string.
        """
        return self

    def __init__(out self, *, py: PythonObject) raises:
        """Construct a `String` from a PythonObject.

        Args:
            py: The PythonObject to convert from.

        Raises:
            An error if the conversion failed.
        """
        var str_obj = py.__str__()
        self = String(StringSlice(unsafe_borrowed_obj=str_obj))
        # keep python object alive so the copy can occur
        _ = str_obj

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    def write_to(self, mut writer: Some[Writer]):
        """
        Formats this string to the provided Writer.

        Args:
            writer: The object to write to.
        """
        writer.write_string(self)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Formats this string slice to the provided `Writer`.

        Args:
            writer: The object to write to.

        Notes:
            Mojo's repr always prints single quotes (`'`) at the start and end
            of the repr. Any single quote inside a string should be escaped
            (`\\'`).
        """
        StringSlice(self).write_repr_to(writer)

    def join[T: Copyable & Writable](self, elems: Span[T, _]) -> String:
        """Joins string elements using the current string as a delimiter.
        Defaults to writing to the stack if total bytes of `elems` is less than
        `buffer_size`, otherwise will allocate once to the heap and write
        directly into that. The `buffer_size` defaults to 4096 bytes to match
        the default page size on arm64 and x86-64.

        Parameters:
            T: The type of the elements. Must implement the `Copyable`,
                and `Writable` traits.

        Args:
            elems: The input values.

        Returns:
            The joined string.

        Notes:
            - Defaults to writing directly to the string if the bytes
            fit in an inline `String`, otherwise will process it by chunks.
            - The `buffer_size` defaults to 4096 bytes to match the default
            page size on arm64 and x86-64, but you can increase this if you're
            joining a very large `List` of elements to write into the stack
            instead of the heap.
        """
        return StringSlice(self).join(elems)

    def bytes(self) -> BytesIter[origin_of(self)]:
        """Returns an iterator over the raw UTF-8 bytes of this string.

        Unlike `codepoints()` and `graphemes()`, this iterator operates at the
        byte level and yields individual `Byte` values without interpreting
        multi-byte UTF-8 sequences.

        Returns:
            An iterator type that returns successive `Byte` values stored in
            this string.

        Examples:

        Iterate over the bytes of an ASCII string:

        ```mojo
        from std.testing import assert_equal, assert_raises

        var s = String("abc")
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
        var s = String("é")
        var iter = s.bytes()
        assert_equal(next(iter), Byte(0xC3))
        assert_equal(next(iter), Byte(0xA9))
        ```
        """
        return StringSlice(self).bytes()

    def codepoints(self) -> CodepointsIter[origin_of(self)]:
        """Returns an iterator over the `Codepoint`s encoded in this string slice.

        Returns:
            An iterator type that returns successive `Codepoint` values stored in
            this string slice.

        Examples:

        Print the characters in a string:

        ```mojo
        from std.testing import assert_equal, assert_raises

        var s = "abc"
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
        var s = "á"
        assert_equal(s.byte_length(), 3)

        var iter = s.codepoints()
        assert_equal(iter.__next__(), Codepoint.ord("a"))
         # U+0301 Combining Acute Accent
        assert_equal(iter.__next__().to_u32(), 0x0301)
        with assert_raises():
            _ = iter.__next__() # raises StopIteration
        ```
        """
        return StringSlice(self).codepoints()

    def codepoint_slices(self) -> CodepointSliceIter[origin_of(self)]:
        """Returns an iterator over single-character slices of this string.

        Each returned slice points to a single Unicode codepoint encoded in the
        underlying UTF-8 representation of this string.

        Returns:
            An iterator of references to the string elements.

        Examples:

        Iterate over the character slices in a string:

        ```mojo
        from std.testing import assert_equal, assert_raises, assert_true

        var s = "abc"
        var iter = s.codepoint_slices()
        assert_true(iter.__next__() == "a")
        assert_true(iter.__next__() == "b")
        assert_true(iter.__next__() == "c")
        with assert_raises():
            _ = iter.__next__() # raises StopIteration
        ```
        """
        return StringSlice(self).codepoint_slices()

    def codepoint_slices_reversed(
        self,
    ) -> CodepointSliceIter[origin_of(self), False]:
        """Iterates backwards over the string, returning single-character slices.

        Each returned slice points to a single Unicode codepoint encoded in the
        underlying UTF-8 representation of this string, starting from the end
        and moving towards the beginning.

        Returns:
            A reversed iterator of references to the string elements.
        """
        return CodepointSliceIter[origin_of(self), forward=False](self)

    def graphemes(self) -> GraphemeSliceIter[origin_of(self)]:
        """Return an iterator over the grapheme clusters in this string.

        A grapheme cluster is what a user would typically think of as a
        single "character" on screen.

        Returns:
            An iterator yielding each grapheme cluster as a `StringSlice`.
        """
        return StringSlice(self).graphemes()

    def graphemes_reversed(
        self,
    ) -> GraphemeSliceIter[origin_of(self), False]:
        """Return an iterator over the grapheme clusters in this string,
        yielding them in reverse order.

        See `graphemes()` for the definition of a grapheme cluster.

        Returns:
            A reverse iterator yielding each grapheme cluster as a
            `StringSlice`.
        """
        return StringSlice(self).graphemes_reversed()

    def grapheme_indices(self) -> GraphemeIndicesIter[origin_of(self)]:
        """Return an iterator over grapheme clusters paired with their byte
        offsets.

        Each yielded element is a `Tuple[Int, StringSlice]` where the first
        element is the byte offset at which the grapheme begins.

        Returns:
            An iterator yielding `(byte_offset, grapheme)` pairs.
        """
        return StringSlice(self).grapheme_indices()

    def split_at_grapheme(
        self, n: Int
    ) -> Tuple[
        StringSlice[ImmOrigin(origin_of(self))],
        StringSlice[ImmOrigin(origin_of(self))],
    ]:
        """Split this string at the `n`-th grapheme-cluster boundary.

        Args:
            n: The grapheme-cluster boundary at which to split. Must be
                non-negative.

        Returns:
            A tuple `(prefix, suffix)` of `StringSlice`s. The prefix covers
            grapheme clusters `[0, n)` and the suffix covers the rest.
        """
        return StringSlice(self).split_at_grapheme(n)

    def count_graphemes(self) -> Int:
        """Count the number of grapheme clusters in this string.

        This is an O(n) operation.

        Returns:
            The number of grapheme clusters.
        """
        return StringSlice(self).count_graphemes()

    @always_inline("nodebug")
    def unsafe_ptr(
        self,
    ) -> Pointer[Byte, origin_of(self)]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """

        if self._is_inline():
            # The string itself holds the data.
            return (
                Pointer(to=self)
                .unsafe_bitcast[Byte]()
                .as_imm()
                .unsafe_origin_cast[origin_of(self)]()
            )
        else:
            return self._ptr_or_data.as_imm().unsafe_origin_cast[
                origin_of(self)
            ]()

    def unsafe_ptr_mut(
        mut self, var capacity: Int = 0
    ) -> Pointer[Byte, origin_of(self)]:
        """Retrieves a mutable pointer to the unique underlying memory. Passing
        a larger capacity will reallocate the string to the new capacity if
        larger than the existing capacity, allowing you to write more data.

        Args:
            capacity: The new capacity of the string.

        Returns:
            The pointer to the underlying memory.
        """
        var new_cap = max(self.capacity_bytes(), capacity)
        # Decide on strategy for making the string mutable
        if new_cap <= Self.INLINE_CAPACITY:
            if not self._is_inline():
                self._inline_string()
        elif not self._is_unique() or new_cap > self.capacity_bytes():
            self._realloc_mutable(new_cap)

        return self.unsafe_ptr().unsafe_mut_cast[True]()

    @always_inline
    def as_c_string_slice(
        mut self,
    ) -> CStringSlice[ImmOrigin(origin_of(self))]:
        """Return a `CStringSlice` to the underlying memory of the string.

        Returns:
            The `CStringSlice` of the string.
        """
        # Add a nul terminator, making the string mutable if not already
        if not self._has_nul_terminator():
            var ptr = self.unsafe_ptr_mut(capacity=self.byte_length() + 1)
            var len = self.byte_length()
            ptr[unsafe_offset=len] = 0
            self._capacity_or_data |= Self.FLAG_HAS_NUL_TERMINATOR

        # Safety: we ensure the string is null-terminated above.
        return CStringSlice(
            unsafe_from_ptr=self.unsafe_ptr().unsafe_bitcast[c_char]()
        )

    @__unsafe_nested_origins_read_only
    def as_bytes(
        ref self,
    ) -> Span[Byte, origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a contiguous slice of the bytes owned by this string.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.

        Notes:
            The returned span carries an interior origin derived from this
            string, so mutating the string (which may reallocate its buffer)
            invalidates the span at compile time.
        """

        return Span(
            unsafe_ptr=Pointer(
                to=self.unsafe_ptr()._get_ref_with_unsafe_interior_origin[
                    "bytes", origin_of(self)
                ]()
            ),
            length=self.byte_length(),
        )

    @__unsafe_nested_origins_read_only
    def _interior_slice(
        ref self,
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        # Canonical interior-origin string-slice view of the whole string. All
        # view-returning accessors derive from this so element/substring views
        # are invalidated when the string is mutated.
        return StringSlice(unsafe_from_utf8=self.as_bytes())

    @__unsafe_nested_origins_read_only
    def unsafe_as_bytes_mut(
        mut self,
    ) -> Span[Byte, origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a mutable contiguous slice of the bytes owned by this string.
        This name has a _mut suffix so the as_bytes() method doesn't have to
        guarantee mutability.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.

        Safety:
            - Any mutation of the byte slice must uphold UTF-8 validity of the
              overall string.
        """
        return Span(
            unsafe_ptr=Pointer(
                to=self.unsafe_ptr_mut()._get_ref_with_unsafe_interior_origin[
                    "bytes", origin_of(self)
                ]()
            ),
            length=self.byte_length(),
        )

    def byte_length(self) -> Int:
        """Get the string length in bytes.

        Returns:
            The length of this string in bytes.
        """
        if self._is_inline():
            return Int(
                UInt64(self._capacity_or_data & Self.INLINE_LENGTH_MASK)
                >> UInt64(Self.INLINE_LENGTH_START)
            )
        else:
            return self._len_or_data

    @always_inline
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

            var s = StringSlice("ನಮಸ್ಕಾರ")
            assert_equal(s.count_codepoints(), 7)
            assert_equal(s.byte_length(), 21)
            ```

            Strings containing only ASCII characters have the same byte and
            Unicode codepoint length:

            ```mojo
            from std.testing import assert_equal

            var s = StringSlice("abc")
            assert_equal(s.count_codepoints(), 3)
            assert_equal(s.byte_length(), 3)
            ```

            The character length of a string with visual combining characters is
            the length in Unicode codepoints, not grapheme clusters:

            ```mojo
            from std.testing import assert_equal

            var s = StringSlice("á")
            assert_equal(s.count_codepoints(), 2)
            assert_equal(s.byte_length(), 3)
            ```

        Notes:
            This method needs to traverse the whole string to count, so it has
            a performance hit compared to using the byte length.
        """
        return StringSlice(self).count_codepoints()

    def _set_byte_length(mut self, new_len: Int):
        """Set the byte length of the `String`.

        This only updates the length field. The caller is responsible for
        ensuring the buffer holds at least `new_len` initialized bytes.

        Args:
            new_len: The new byte length to set.
        """
        if self._is_inline():
            self._capacity_or_data = (
                self._capacity_or_data & ~Self.INLINE_LENGTH_MASK
            ) | (new_len << Self.INLINE_LENGTH_START)
        else:
            self._len_or_data = new_len

    def count(self, substr: StringSlice) -> Int:
        """Return the number of non-overlapping occurrences of substring
        `substr` in the string.

        If sub is empty, returns the number of empty strings between characters
        which is the length of the string plus one.

        Args:
          substr: The substring to count.

        Returns:
          The number of occurrences of `substr`.
        """
        return StringSlice(self).count(substr)

    def __contains__(self, substr: StringSlice) -> Bool:
        """Returns True if the substring is contained within the current string.

        Args:
          substr: The substring to check.

        Returns:
          True if the string contains the substring.
        """
        return substr in StringSlice(self)

    def find(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset in bytes of the first occurrence of `substr`
        starting at `start`. If not found, returns `-1`.

        Args:
          substr: The substring to find.
          start: The offset in bytes from which to find.

        Returns:
          The offset in bytes of `substr` relative to the beginning of the
          string.
        """

        return StringSlice(self).find(substr, start)

    def rfind(self, substr: StringSlice, start: Int = 0) -> Int:
        """Finds the offset in bytes of the last occurrence of `substr` starting
        at `start`. If not found, returns `-1`.

        Args:
          substr: The substring to find.
          start: The offset in bytes from which to find.

        Returns:
          The offset in bytes of `substr` relative to the beginning of the
          string.
        """

        return StringSlice(self).rfind(substr, start=start)

    def partition(self, sep: StringSlice) -> Tuple[String, String, String]:
        """Splits the string at the first occurrence of `sep`.

        Splits `self` at the first occurrence of `sep`, and returns a
        3-tuple of the form `(before, sep, after)`. If `sep` is not in `self`,
        returns `(self, "", "")`.

        Args:
            sep: The separator to partition on.

        Returns:
            A `Tuple[String, String, String]` containing the part before `sep`,
            `sep` itself, and the part after `sep`. If `sep` is not found,
            returns `(self, "", "")`.

        """
        return StringSlice(self).partition(sep)

    def rpartition(self, sep: StringSlice) -> Tuple[String, String, String]:
        """Splits the string at the last occurrence of `sep`.

        Splits `self` at the last occurrence of `sep`, and returns a
        3-tuple of the form `(before, sep, after)`. If `sep` is not in `self`,
        returns `("", "", self)`.

        Args:
            sep: The separator to partition on.

        Returns:
            A `Tuple[String, String, String]` containing the part before `sep`,
            `sep` itself, and the part after `sep`. If `sep` is not found,
            returns `("", "", self)`.

        """
        return StringSlice(self).rpartition(sep)

    def isspace(self) -> Bool:
        """Determines whether every character in the given String is a
        python whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the whole String is made up of whitespace characters
                listed above, otherwise False.
        """
        return StringSlice(self).isspace()

    @always_inline
    def split(
        self, sep: StringSlice
    ) -> List[StringSlice[origin_of(self)._get_owned_interior["bytes"]]]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting a space
        _ = StringSlice("hello world").split(" ") # ["hello", "world"]
        # Splitting adjacent separators
        _ = StringSlice("hello,,world").split(",") # ["hello", "", "world"]
        # Splitting with starting or ending separators
        _ = StringSlice(",1,2,3,").split(",") # ['', '1', '2', '3', '']
        # Splitting with an empty separator
        _ = StringSlice("123").split("") # ['', '1', '2', '3', '']
        ```
        """
        return self._interior_slice().split(sep)

    @always_inline
    def split(
        self, sep: StringSlice, maxsplit: Int
    ) -> List[StringSlice[origin_of(self)._get_owned_interior["bytes"]]]:
        """Split the string by a separator.

        Args:
            sep: The string to split on.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting with maxsplit
        _ = StringSlice("1,2,3").split(",", maxsplit=1) # ['1', '2,3']
        # Splitting with starting or ending separators
        _ = StringSlice(",1,2,3,").split(",", maxsplit=1) # ['', '1,2,3,']
        # Splitting with an empty separator
        _ = StringSlice("123").split("", maxsplit=1) # ['', '123']
        ```
        """
        return self._interior_slice().split(sep, maxsplit=maxsplit)

    @always_inline
    def split(
        self, sep: NoneType = None
    ) -> List[StringSlice[origin_of(self)._get_owned_interior["bytes"]]]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:

        ```mojo
        # Splitting an empty string or filled with whitespaces
        _ = StringSlice("      ").split() # []
        _ = StringSlice("").split() # []

        # Splitting a string with leading, trailing, and middle whitespaces
        _ = StringSlice("      hello    world     ").split() # ["hello", "world"]
        # Splitting adjacent universal newlines:
        _ = StringSlice(
            "hello \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85world"
        ).split()  # ["hello", "world"]
        ```
        """
        return self._interior_slice().split(sep)

    @always_inline
    def split(
        self, sep: NoneType = None, *, maxsplit: Int
    ) -> List[StringSlice[origin_of(self)._get_owned_interior["bytes"]]]:
        """Split the string by every Whitespace separator.

        Args:
            sep: None.
            maxsplit: The maximum amount of items to split from String.

        Returns:
            A List of Strings containing the input split by the separator.

        Examples:
        ```mojo
        # Splitting with maxsplit
        _ = StringSlice("1     2  3").split(maxsplit=1) # ['1', '2  3']
        ```
        """
        return self._interior_slice().split(sep, maxsplit=maxsplit)

    def splitlines(
        self, keepends: Bool = False
    ) -> List[StringSlice[origin_of(self)._get_owned_interior["bytes"]]]:
        """Split the string at line boundaries. This corresponds to Python's
        [universal newlines:](
        https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `"\\r\\n"` and `"\\t\\n\\v\\f\\r\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Args:
            keepends: If True, line breaks are kept in the resulting strings.

        Returns:
            A List of Strings containing the input split by line boundaries.
        """
        return self._interior_slice().splitlines(keepends)

    def replace(self, old: StringSlice, new: StringSlice) -> String:
        """Return a copy of the string with all occurrences of substring `old`
        if replaced by `new`.

        Args:
            old: The substring to replace.
            new: The substring to replace with.

        Returns:
            The string where all occurrences of `old` are replaced with `new`.
        """
        return StringSlice(self).replace(old, new)

    def strip(
        self, chars: ImmStringSlice
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with leading and trailing characters
        removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A `StringSlice` into this string with no leading or trailing
            characters.
        """

        return self.lstrip(chars).rstrip(chars)

    def strip(
        self,
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with leading and trailing whitespaces
        removed. This only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A `StringSlice` into this string with no leading or trailing
            whitespaces.
        """
        return self.lstrip().rstrip()

    def rstrip(
        self, chars: ImmStringSlice
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with trailing characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A `StringSlice` into this string with no trailing characters.
        """

        return self._interior_slice().rstrip(chars)

    def rstrip(
        self,
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with trailing whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A `StringSlice` into this string with no trailing whitespaces.
        """
        return self._interior_slice().rstrip()

    def lstrip(
        self, chars: ImmStringSlice
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with leading characters removed.

        Args:
            chars: A set of characters to be removed. Defaults to whitespace.

        Returns:
            A `StringSlice` into this string with no leading characters.
        """

        return self._interior_slice().lstrip(chars)

    def lstrip(
        self,
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with leading whitespaces removed. This
        only takes ASCII whitespace into account:
        `" \\t\\n\\v\\f\\r\\x1c\\x1d\\x1e"`.

        Returns:
            A `StringSlice` into this string with no leading whitespaces.
        """
        return self._interior_slice().lstrip()

    def __hash__[H: Hasher](self, mut hasher: H):
        """Updates hasher with the underlying bytes.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        StringSlice(self).__hash__(hasher)

    def lower(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to lowercase.

        Returns:
            A new string where cased letters have been converted to lowercase.
        """

        return StringSlice(self).lower()

    def upper(self) -> String:
        """Returns a copy of the string with all cased characters
        converted to uppercase.

        Returns:
            A new string where cased letters have been converted to uppercase.
        """

        return StringSlice(self).upper()

    def startswith(
        self, prefix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string starts with the specified prefix between start
        and end positions. Returns True if found and False otherwise.

        The `start` and `end` positions must be offsets given in bytes, and
        must be codepoint boundaries.

        Args:
            prefix: The prefix to check.
            start: The start offset in bytes from which to check.
            end: The end offset in bytes from which to check.

        Returns:
            True if the `self[byte=start:end]` is prefixed by the input prefix.
        """
        return StringSlice(self).startswith(prefix, start, end)

    def endswith(
        self, suffix: StringSlice, start: Int = 0, end: Int = -1
    ) -> Bool:
        """Checks if the string end with the specified suffix between start
        and end positions. Returns True if found and False otherwise.

        The `start` and `end` positions must be offsets given in bytes, and
        must be codepoint boundaries.

        Args:
            suffix: The suffix to check.
            start: The start offset in bytes from which to check.
            end: The end offset in bytes from which to check.

        Returns:
            True if the `self[byte=start:end]` is suffixed by the input suffix.
        """
        return StringSlice(self).endswith(suffix, start, end)

    def removeprefix(
        self, prefix: StringSlice, /
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with the prefix removed if it was
        present.

        Args:
            prefix: The prefix to remove from the string.

        Returns:
            A `StringSlice` into this string:
            `string[byte=prefix.byte_length():]` if the string starts with
            the prefix string, or the whole string otherwise.

        Examples:

        ```mojo
        print(String('TestHook').removeprefix('Test')) # 'Hook'
        print(String('BaseTestCase').removeprefix('Test')) # 'BaseTestCase'
        ```
        """
        return self._interior_slice().removeprefix(prefix)

    def removesuffix(
        self, suffix: StringSlice, /
    ) -> StringSlice[origin_of(self)._get_owned_interior["bytes"]]:
        """Returns a view of the string with the suffix removed if it was
        present.

        Args:
            suffix: The suffix to remove from the string.

        Returns:
            A `StringSlice` into this string:
            `string[byte=:(self.byte_length()-suffix.byte_length())]` if the string ends with the suffix string,
            or the whole string otherwise.

        Examples:

        ```mojo
        print(String('TestHook').removesuffix('Hook')) # 'Test'
        print(String('BaseTestCase').removesuffix('Test')) # 'BaseTestCase'
        ```
        """
        return self._interior_slice().removesuffix(suffix)

    def __int__(self) raises -> Int:
        """Parses the given string as a base-10 integer and returns that value.
        If the string cannot be parsed as an int, an error is raised.

        Returns:
            An integer value that represents the string, or otherwise raises.

        Raises:
            If the operation fails.
        """
        return atol(self)

    def __float__(self) raises -> Float64:
        """Parses the string as a float point number and returns that value. If
        the string cannot be parsed as a float, an error is raised.

        Returns:
            A float value that represents the string, or otherwise raises.

        Raises:
            If the operation fails.
        """
        return atof(self)

    def __mul__(self, n: Int) -> String:
        """Concatenates the string `n` times.

        Args:
            n : The number of times to concatenate the string.

        Returns:
            The string concatenated `n` times.
        """
        return StringSlice(self) * n

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

        Example:

        ```mojo
        # Manual indexing:
        print("{0} {1} {0}".format("Mojo", 1.125)) # Mojo 1.125 Mojo
        # Automatic indexing:
        print("{} {}".format(True, "hello world")) # True hello world
        ```

        Raises:
            If the operation fails.
        """
        return _FormatUtils.format(self, *args)

    def is_ascii_digit(self) -> Bool:
        """A string is a digit string if all characters in the string are ASCII digits
        and there is at least one character in the string.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are digits and it's not empty else False.
        """
        return StringSlice(self).is_ascii_digit()

    def isupper(self) -> Bool:
        """Returns True if all cased characters in the string are uppercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are uppercase and there
            is at least one cased character, False otherwise.
        """
        return StringSlice(self).isupper()

    def islower(self) -> Bool:
        """Returns True if all cased characters in the string are lowercase and
        there is at least one cased character.

        Returns:
            True if all cased characters in the string are lowercase and there
            is at least one cased character, False otherwise.
        """
        return StringSlice(self).islower()

    def is_ascii_printable(self) -> Bool:
        """Returns True if all characters in the string are ASCII printable.

        Note that this currently only works with ASCII strings.

        Returns:
            True if all characters are printable else False.
        """
        return StringSlice(self).is_ascii_printable()

    def ascii_rjust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string right justified in a string of specified width.

        Pads the string on the left with the specified fill character so that
        the total (byte) length of the resulting string equals `width`. If the original
        string is already longer than or equal to `width`, returns the original
        string unchanged.

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A right-justified string of (byte) length `width`, or the original string
            if its length is already greater than or equal to `width`.

        Examples:

        ```mojo
        var s = String("hello")
        print(s.ascii_rjust(10))        # "     hello"
        print(s.ascii_rjust(10, "*"))   # "*****hello"
        print(s.ascii_rjust(3))         # "hello" (no padding)
        ```
        """
        return StringSlice(self).ascii_rjust(width, fillchar)

    def ascii_ljust(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string left justified in a string of specified width.

        Pads the string on the right with the specified fill character so that
        the total (byte) length of the resulting string equals `width`. If the original
        string is already longer than or equal to `width`, returns the original
        string unchanged.

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A left-justified string of (byte) length `width`, or the original string
            if its length is already greater than or equal to `width`.

        Examples:

        ```mojo
        var s = String("hello")
        print(s.ascii_ljust(10))        # "hello     "
        print(s.ascii_ljust(10, "*"))   # "hello*****"
        print(s.ascii_ljust(3))         # "hello" (no padding)
        ```
        """
        return StringSlice(self).ascii_ljust(width, fillchar)

    def ascii_center(self, width: Int, fillchar: StaticString = " ") -> String:
        """Returns the string center justified in a string of specified width.

        Pads the string on both sides with the specified fill character so that
        the total length of the resulting string equals `width`. If the padding
        needed is odd, the extra character goes on the right side. If the
        original string is already longer than or equal to `width`, returns the
        original string unchanged.

        Args:
            width: The total width (in bytes) of the resulting string. This is
                not the amount of padding, but the final length of the returned
                string.
            fillchar: The padding character to use (defaults to space). Must be
                a single-byte character.

        Returns:
            A center-justified string of length `width`, or the original string
            if its length is already greater than or equal to `width`.

        Examples:

        ```mojo
        var s = String("hello")
        print(s.ascii_center(10))        # "  hello   "
        print(s.ascii_center(11, "*"))   # "***hello***"
        print(s.ascii_center(3))         # "hello" (no padding)
        ```
        """
        return StringSlice(self).ascii_center(width, fillchar)

    def resize(mut self, length: Int, fill_byte: UInt8 = 0):
        """Resize the string to a new length. Panics if new_len does not
        lie on a codepoint boundary or if fill_byte is non-ascii (>=128).

        Args:
            length: The new length of the string.
            fill_byte: The byte to fill any new space with. Must be a valid single-byte
                       utf-8 character.

        Notes:
            If the new length is greater than the current length, the string is
            extended by the difference, and the new bytes are initialized to
            `fill_byte`.
        """
        debug_assert[assert_mode="safe"](
            fill_byte < 128, "Fill byte is the start of a multi-byte character."
        )
        self._clear_nul_terminator()
        var old_len = self.byte_length()
        if length > old_len:
            unsafe_memset(
                self.unsafe_ptr_mut(length).unsafe_offset(old_len),
                fill_byte,
                length - old_len,
            )
        else:
            debug_assert[assert_mode="safe"](
                StringSlice(self).is_codepoint_boundary(length),
                "String shrunk to length ",
                length,
                " which does not lie on a codepoint boundary.",
            )
        self._set_byte_length(length)

    def resize(mut self, *, unsafe_uninit_length: Int):
        """Resizes the string to the given new size leaving any new data
        uninitialized. Panics if the new length does not lie on a codepoint
        boundary.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        string is extended and the new data is left uninitialized.

        Args:
            unsafe_uninit_length: The new size.
        """
        self._clear_nul_terminator()
        debug_assert(
            unsafe_uninit_length >= self.byte_length()
            or StringSlice(self).is_codepoint_boundary(unsafe_uninit_length),
            "String shrunk to length ",
            unsafe_uninit_length,
            " which does not lie on a codepoint boundary.",
        )
        if unsafe_uninit_length > self.capacity_bytes():
            self.reserve_bytes(unsafe_uninit_length)
        self._set_byte_length(unsafe_uninit_length)

    @deprecated(use=reserve_bytes)
    @doc_hidden
    def reserve(mut self, new_capacity_bytes: Int, /):
        self.reserve_bytes(new_capacity_bytes)

    @stable(since="1.1")
    def reserve_bytes(mut self, new_capacity_bytes: Int, /):
        """Reserves at least the requested capacity.

        Args:
            new_capacity_bytes: The minimum new capacity in bytes.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the data is moved.
        """
        if new_capacity_bytes <= self.capacity_bytes():
            return
        self._realloc_mutable(new_capacity_bytes)

    # Make a string mutable on the stack.
    def _inline_string(mut self):
        var length = self.byte_length()
        var new_string = Self()
        new_string._set_byte_length(length)
        var dst = Pointer(to=new_string).unsafe_bitcast[Byte]()
        var src = self.unsafe_ptr()
        for i in range(length):
            dst[unsafe_offset=i] = src[unsafe_offset=i]
        self = new_string^

    # This is the out-of-line implementation of reserve called when we need
    # to grow the capacity of the string. Make sure our capacity at least
    # doubles to avoid O(n^2) behavior, and make use of extra space if it exists.
    def _realloc_mutable(mut self, capacity: Int):
        # Get these fields before we change _capacity_or_data
        var byte_len = self.byte_length()
        var old_ptr = self.unsafe_ptr()
        var new_capacity = (max(capacity, self.capacity_bytes() * 2) + 7) >> 3
        var new_ptr = self._alloc(new_capacity << 3)
        unsafe_memcpy(dest=new_ptr, src=old_ptr, count=byte_len)
        # If mutable buffer drop the ref count
        self._drop_ref()
        self._len_or_data = byte_len
        self._ptr_or_data = new_ptr
        # Assign directly to clear existing flags
        self._capacity_or_data = new_capacity
        self._set_ref_counted()


# ===----------------------------------------------------------------------=== #
# ord
# ===----------------------------------------------------------------------=== #


def ord(s: StringSlice) -> Int:
    """Returns an integer that represents the codepoint of a single-character
    string.

    Given a string containing a single character `Codepoint`, return an integer
    representing the codepoint of that character. For example, `ord("a")`
    returns the integer `97`. This is the inverse of the `chr()` function.

    This function is in the prelude, so you don't need to import it.

    Args:
        s: The input string, which must contain only a single- character.

    Returns:
        An integer representing the code point of the given character.
    """
    return Int(Codepoint.ord(s))


# ===----------------------------------------------------------------------=== #
# chr
# ===----------------------------------------------------------------------=== #

comptime _LARGEST_UNICODE_ASCII_BYTE = 127


def chr(c: Int) -> String:
    """Returns a String based on the given Unicode code point. This is the
    inverse of the `ord()` function.

    This function is in the prelude, so you don't need to import it.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.

    Example:
    ```mojo
    print(chr(97), chr(8364)) # "a €"
    ```
    """
    if c <= _LARGEST_UNICODE_ASCII_BYTE:
        return _unsafe_chr_ascii(UInt8(c))

    var char_opt = Codepoint.from_u32(UInt32(c))
    if not char_opt:
        # TODO: Raise ValueError instead.
        abort(t"chr({c}) is not a valid Unicode codepoint")

    # SAFETY: We just checked that `char` is present.
    return String(char_opt.unsafe_value())


# ===----------------------------------------------------------------------=== #
# ascii
# ===----------------------------------------------------------------------=== #


def _unsafe_chr_ascii(c: UInt8) -> String:
    """Returns a string based on the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a single character based on the given code point.

    Safety:
        The byte must be a valid single byte ASCII character (0-127).
    """
    assert (
        c <= _LARGEST_UNICODE_ASCII_BYTE
    ), "Character is not single byte unicode"

    return String(unsafe_from_utf8=Span(unsafe_ptr=Pointer(to=c), length=1))


def _repr_ascii(c: UInt8) -> String:
    """Returns a printable representation of the given ASCII code point.

    Args:
        c: An integer that represents a code point.

    Returns:
        A string containing a representation of the given code point.
    """
    comptime ord_tab = UInt8(ord("\t"))
    comptime ord_new_line = UInt8(ord("\n"))
    comptime ord_carriage_return = UInt8(ord("\r"))
    comptime ord_back_slash = UInt8(ord("\\"))

    if c == ord_back_slash:
        return r"\\"
    elif Codepoint(c).is_ascii_printable():
        return _unsafe_chr_ascii(c)
    elif c == ord_tab:
        return r"\t"
    elif c == ord_new_line:
        return r"\n"
    elif c == ord_carriage_return:
        return r"\r"
    else:
        var uc = c.cast[.uint8]()
        if uc < 16:
            return hex(uc, prefix=r"\x0")
        else:
            return hex(uc, prefix=r"\x")


def ascii(value: StringSlice) -> String:
    """Get the ASCII representation of the object.

    Args:
        value: The object to get the ASCII representation of.

    Returns:
        A string containing the ASCII representation of the object.
    """
    comptime ord_squote = UInt8(ord("'"))
    var result = String()
    var use_dquote = False
    var data = value.as_bytes()

    for idx in range(len(data)):
        var char = data[idx]
        result += _repr_ascii(char)
        use_dquote = use_dquote or (char == ord_squote)

    if use_dquote:
        return String(t'"{result}"')
    else:
        return String(t"'{result}'")


# ===----------------------------------------------------------------------=== #
# atol
# ===----------------------------------------------------------------------=== #


def atol(str_slice: StringSlice, base: Int = 10) raises -> Int:
    """Parses and returns the given string as an integer in the given base.

    If base is set to 0, the string is parsed as an integer literal, with the
    following considerations:
    - '0b' or '0B' prefix indicates binary (base 2)
    - '0o' or '0O' prefix indicates octal (base 8)
    - '0x' or '0X' prefix indicates hexadecimal (base 16)
    - Without a prefix, it's treated as decimal (base 10)

    This follows [Python's integer literals format](
    https://docs.python.org/3/reference/lexical_analysis.html#integers).

    This function is in the prelude, so you don't need to import it.

    Notes:
        This function only accepts ASCII digits (0-9 and a-z/A-Z for bases
        greater than 10). Unicode digit characters are not supported. Leading
        and trailing whitespace is trimmed, but only ASCII/POSIX whitespace
        characters are recognized (space, tab, newline, etc.). Unicode
        whitespace characters will cause a parsing error.

    Args:
        str_slice: A string to be parsed as an integer in the given base.
        base: Base used for conversion, value must be between 2 and 36, or 0.

    Returns:
        An integer value that represents the string.

    Raises:
        If the given string cannot be parsed as an integer value or if an
        incorrect base is provided.

    Examples:

    ```text
    >>> atol("32")
    32
    >>> atol("FF", 16)
    255
    >>> atol("0xFF", 0)
    255
    >>> atol("0b1010", 0)
    10
    ```
    """

    if (base != 0) and (base < 2 or base > 36):
        raise Error("Base must be >= 2 and <= 36, or 0.")
    if not str_slice:
        raise Error(_str_to_base_error(base, str_slice))

    var real_base: Int
    var ord_num_max: Int

    var ord_letter_max = (-1, -1)
    var result = UInt(0)
    var is_negative: Bool
    var has_prefix: Bool
    var start: Int
    var str_len = str_slice.byte_length()

    start, is_negative = _trim_and_handle_sign(str_slice, str_len)
    if start >= str_len:
        raise Error(_str_to_base_error(base, str_slice))

    comptime ord_0 = ord("0")
    comptime ord_letter_min = (ord("a"), ord("A"))
    comptime ord_underscore = ord("_")

    if base == 0:
        var real_base_new_start = _identify_base(str_slice, start)
        real_base = real_base_new_start[0]
        start = real_base_new_start[1]
        has_prefix = real_base != 10
        if real_base == -1:
            raise Error(_str_to_base_error(base, str_slice))
    else:
        start, has_prefix = _handle_base_prefix(start, str_slice, str_len, base)
        real_base = base

    if real_base <= 10:
        ord_num_max = ord(String(real_base - 1))
    else:
        ord_num_max = ord("9")
        ord_letter_max = (
            ord("a") + (real_base - 11),
            ord("A") + (real_base - 11),
        )

    var buff = str_slice.unsafe_ptr()
    var found_valid_chars_after_start = False
    var has_space_after_number = False

    # The largest magnitude that fits in `Int`: one more for negative results,
    # since |Int.MIN| == Int.MAX + 1.
    var limit = UInt(Int.MAX) + UInt(1 if is_negative else 0)
    var mul_cutoff = limit // UInt(real_base)
    comptime too_large = (
        " String expresses an integer too large to store in Int."
    )

    # Prefixed integer literals with real_base 2, 8, 16 may begin with leading
    # underscores under the conditions they have a prefix
    var was_last_digit_underscore = not (real_base in (2, 8, 16) and has_prefix)
    for pos in range(start, str_len):
        var ord_current = Int(buff[unsafe_offset=pos])
        if ord_current == ord_underscore:
            if was_last_digit_underscore:
                raise Error(_str_to_base_error(base, str_slice))
            else:
                was_last_digit_underscore = True
                continue
        else:
            was_last_digit_underscore = False

        var digit: UInt
        if ord_0 <= ord_current <= ord_num_max:
            digit = UInt(ord_current - ord_0)
        elif ord_letter_min[0] <= ord_current <= ord_letter_max[0]:
            digit = UInt(ord_current - ord_letter_min[0] + 10)
        elif ord_letter_min[1] <= ord_current <= ord_letter_max[1]:
            digit = UInt(ord_current - ord_letter_min[1] + 10)
        elif Codepoint(UInt8(ord_current)).is_posix_space():
            has_space_after_number = True
            start = pos + 1
            break
        else:
            raise Error(_str_to_base_error(base, str_slice))
        found_valid_chars_after_start = True

        if result > limit - digit:
            raise Error(_str_to_base_error(base, str_slice), too_large)
        result += digit

        if (
            pos + 1 < str_len
            and not Codepoint(buff[unsafe_offset=pos + 1]).is_posix_space()
        ):
            if result > mul_cutoff:
                raise Error(_str_to_base_error(base, str_slice), too_large)
            result *= UInt(real_base)

    if was_last_digit_underscore or (not found_valid_chars_after_start):
        raise Error(_str_to_base_error(base, str_slice))

    if has_space_after_number:
        for pos in range(start, str_len):
            if not Codepoint(buff[unsafe_offset=pos]).is_posix_space():
                raise Error(_str_to_base_error(base, str_slice))
    if is_negative:
        if result == UInt(Int.MAX) + 1:
            return Int.MIN
        return -Int(result)
    return Int(result)


def _trim_and_handle_sign(
    str_slice: StringSlice, str_len: Int
) -> Tuple[Int, Bool]:
    """Trims leading whitespace, handles the sign of the number in the string.

    Args:
        str_slice: A StringSlice containing the number to parse.
        str_len: The length of the string.

    Returns:
        A tuple containing:
        - The starting index of the number after whitespace and sign.
        - A boolean indicating whether the number is negative.
    """
    var buff = str_slice.unsafe_ptr()
    var start: Int = 0
    while (
        start < str_len
        and Codepoint(buff[unsafe_offset=start]).is_posix_space()
    ):
        start += 1
    if start == str_len:
        return start, False
    var p: Bool = buff[unsafe_offset=start] == UInt8(ord("+"))
    var n: Bool = buff[unsafe_offset=start] == UInt8(ord("-"))
    return start + (Int(p) or Int(n)), n


def _handle_base_prefix(
    pos: Int, str_slice: StringSlice, str_len: Int, base: Int
) -> Tuple[Int, Bool]:
    """Adjusts the starting position if a valid base prefix is present.

    Handles "0b"/"0B" for base 2, "0o"/"0O" for base 8, and "0x"/"0X" for base
    16. Only adjusts if the base matches the prefix.

    Args:
        pos: Current position in the string.
        str_slice: The input StringSlice.
        str_len: Length of the input string.
        base: The specified base.

    Returns:
        A tuple containing:
            - Updated position after the prefix, if applicable.
            - A boolean indicating if the prefix was valid for the given base.
    """
    var start = pos
    var buff = str_slice.unsafe_ptr()
    if start + 1 < str_len:
        var prefix_char = chr(Int(buff[unsafe_offset=start + 1]))
        if buff[unsafe_offset=start] == UInt8(ord("0")) and (
            (base == 2 and (prefix_char == "b" or prefix_char == "B"))
            or (base == 8 and (prefix_char == "o" or prefix_char == "O"))
            or (base == 16 and (prefix_char == "x" or prefix_char == "X"))
        ):
            start += 2
    return start, start != pos


def _str_to_base_error(base: Int, str_slice: StringSlice) -> String:
    return String(
        "String is not convertible to integer with base ",
        base,
        ": '",
        str_slice,
        "'",
    )


def _identify_base(str_slice: StringSlice, start: Int) -> Tuple[Int, Int]:
    var length = str_slice.byte_length()
    # just 1 digit, assume base 10
    if start == (length - 1):
        return 10, start
    if str_slice[byte=start] == "0":
        var second_digit = str_slice[byte=start + 1]
        if second_digit == "b" or second_digit == "B":
            return 2, start + 2
        if second_digit == "o" or second_digit == "O":
            return 8, start + 2
        if second_digit == "x" or second_digit == "X":
            return 16, start + 2
        # checking for special case of all "0", "_" are also allowed
        var was_last_character_underscore = False
        for i in range(start + 1, length):
            if str_slice[byte=i] == "_":
                if was_last_character_underscore:
                    return -1, -1
                else:
                    was_last_character_underscore = True
                    continue
            else:
                was_last_character_underscore = False
            if str_slice[byte=i] != StringSlice("0"):
                return -1, -1
    elif ord("1") <= ord(str_slice[byte=start]) <= ord("9"):
        return 10, start
    else:
        return -1, -1

    return 10, start


def _atof_error[reason: StaticString = ""](str_ref: StringSlice) -> Error:
    comptime if reason:
        return Error(
            "String is not convertible to float: '",
            str_ref,
            "' because ",
            reason,
        )
    return Error("String is not convertible to float: '", str_ref, "'")


def atof(str_slice: StringSlice) raises -> Float64:
    """Parses the given string as a floating point and returns that value.

    For example, `atof("2.25")` returns `2.25`.

    This function is in the prelude, so you don't need to import it.

    Args:
        str_slice: A string to be parsed as a floating point.

    Returns:
        A floating-point value that represents the string.

    Raises:
        If the given string cannot be parsed as an floating-point value, for
        example in `atof("hi")`.
    """
    return _atof(str_slice)


# ===----------------------------------------------------------------------=== #
# Other utilities
# ===----------------------------------------------------------------------=== #


def _toggle_ascii_case(char: UInt8) -> UInt8:
    """Assuming char is a cased ASCII character, this function will return the
    opposite-cased letter.
    """

    # ASCII defines A-Z and a-z as differing only in their 6th bit,
    # so converting is as easy as a bit flip.
    return char ^ (1 << 5)


def _calc_initial_buffer_size_int32(n0: Int) -> Int:
    # See https://commaok.xyz/post/lookup_tables/ and
    # https://lemire.me/blog/2021/06/03/computing-the-number-of-digits-of-an-integer-even-faster/
    # for a description.
    comptime lookup_table: Array[Int, 32] = [
        4294967296,
        8589934582,
        8589934582,
        8589934582,
        12884901788,
        12884901788,
        12884901788,
        17179868184,
        17179868184,
        17179868184,
        21474826480,
        21474826480,
        21474826480,
        21474826480,
        25769703776,
        25769703776,
        25769703776,
        30063771072,
        30063771072,
        30063771072,
        34349738368,
        34349738368,
        34349738368,
        34349738368,
        38554705664,
        38554705664,
        38554705664,
        41949672960,
        41949672960,
        41949672960,
        42949672960,
        42949672960,
    ]

    ref lookup_table_ref = global_constant[lookup_table]()

    var n = UInt32(n0)
    var log2 = Int(
        UInt32(bit_width_of[DType.uint32]() - 1) ^ count_leading_zeros(n | 1)
    )
    return (n0 + lookup_table_ref[log2]) >> 32


def _calc_initial_buffer_size_int64(n0: UInt64) -> Int:
    var result: Int = 1
    var n = n0
    while True:
        if n < 10:
            return result
        if n < 100:
            return result + 1
        if n < 1_000:
            return result + 2
        if n < 10_000:
            return result + 3
        n //= 10_000
        result += 4


def _calc_initial_buffer_size(n: Float64) -> Int:
    return 128 + 1  # Add 1 for the terminator


def _calc_initial_buffer_size[dtype: DType](n0: Scalar[dtype]) -> Int:
    comptime if dtype.is_integral():
        var n = abs(n0)
        var sign = 0 if n0 >= 0 else 1

        comptime if is_32bit() or bit_width_of[dtype]() <= 32:
            return sign + _calc_initial_buffer_size_int32(Int(n)) + 1
        else:
            return sign + _calc_initial_buffer_size_int64(n.cast[.uint64]()) + 1

    return 128 + 1  # Add 1 for the terminator


def _calc_format_buffer_size[dtype: DType]() -> Int:
    """Returns a buffer size in bytes that is large enough to store a formatted
    number of the specified dtype.
    """

    # TODO:
    #   Use a smaller size based on the `dtype`, e.g. we don't need as much
    #   space to store a formatted int8 as a float64.
    comptime if dtype.is_integral():
        return 64 + 1
    else:
        return 128 + 1  # Add 1 for the terminator
