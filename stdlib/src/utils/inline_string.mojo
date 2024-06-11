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

"""Implements a string that has a small-string optimization which
   avoids heap allocations for short strings.
"""

from sys import sizeof

from memory import memcpy, LegacyPointer, UnsafePointer

from collections import Optional

from utils import InlineArray, Variant, StringSlice
from utils._format import ToFormatter


# ===----------------------------------------------------------------------===#
# InlineString
# ===----------------------------------------------------------------------===#


@value
struct InlineString(Sized, Stringable, CollectionElement):
    """A string that performs small-string optimization to avoid heap allocations for short strings.
    """

    alias SMALL_CAP: Int = 24

    """The number of bytes of string data that can be stored inline in this
    string before a heap allocation is required.

    If constructed from a heap allocated String that string will be used as the
    layout of this string, even if the given string would fit within the
    small-string capacity of this type."""

    # Fields
    alias Layout = Variant[String, _FixedString[Self.SMALL_CAP]]

    var _storage: Self.Layout

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    fn __init__(inout self):
        """Constructs a new empty string."""
        var fixed = _FixedString[Self.SMALL_CAP]()
        self._storage = Self.Layout(fixed^)

    fn __init__(inout self, literal: StringLiteral):
        """Constructs a InlineString value given a string literal.

        Args:
            literal: The input constant string.
        """

        if len(literal) <= Self.SMALL_CAP:
            try:
                var fixed = _FixedString[Self.SMALL_CAP](literal)
                self._storage = Self.Layout(fixed^)
            except e:
                abort(
                    "unreachable: Construction of FixedString of validated"
                    " string failed"
                )
                # TODO(#11245):
                #   When support for "noreturn" functions is added,
                #   this false initialization of this type should be unnecessary.
                self._storage = Self.Layout(String(""))
        else:
            var heap = String(literal)
            self._storage = Self.Layout(heap^)

    fn __init__(inout self, owned heap_string: String):
        """Construct a new small string by taking ownership of an existing
        heap-allocated String.

        Args:
            heap_string: The heap string to take ownership of.
        """
        self._storage = Self.Layout(heap_string^)

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    fn __iadd__(inout self, literal: StringLiteral):
        """Appends another string to this string.

        Args:
            literal: The string to append.
        """
        self.__iadd__(StringRef(literal))

    fn __iadd__(inout self, string: String):
        """Appends another string to this string.

        Args:
            string: The string to append.
        """
        self.__iadd__(string.as_string_slice())

    fn __iadd__(inout self, str_slice: StringSlice[_]):
        """Appends another string to this string.

        Args:
            str_slice: The string to append.
        """
        var total_len = len(self) + str_slice._byte_length()

        # NOTE: Not guaranteed that we're in the small layout even if our
        #       length is shorter than the small capacity.

        if not self._is_small():
            self._storage[String] += str_slice
        elif total_len < Self.SMALL_CAP:
            try:
                self._storage[_FixedString[Self.SMALL_CAP]] += str_slice
            except e:
                abort(
                    "unreachable: InlineString append to FixedString failed: "
                    + str(e),
                )
        else:
            # We're currently in the small layout but must change to the
            # big layout.

            # Begin by heap allocating enough space to store the combined
            # string.
            var buffer = List[UInt8](capacity=total_len)

            # Copy the bytes from the current small string layout
            memcpy(
                dest=buffer.unsafe_ptr(),
                src=self._storage[_FixedString[Self.SMALL_CAP]].unsafe_ptr(),
                count=len(self),
            )

            # Copy the bytes from the additional string.
            memcpy(
                dest=buffer.unsafe_ptr() + len(self),
                src=str_slice.unsafe_ptr(),
                count=str_slice._byte_length(),
            )

            # Record that we've initialized `total_len` count of elements
            # in `buffer`
            buffer.size = total_len

            # Add the NUL byte
            buffer.append(0)

            self._storage = Self.Layout(String(buffer^))

    fn __add__(self, other: StringLiteral) -> Self:
        """Construct a string by appending another string at the end of this string.

        Args:
            other: The string to append.

        Returns:
            A new string containing the concatenation of `self` and `other`.
        """

        var string = self
        string += StringRef(other)
        return string

    fn __add__(self, other: String) -> Self:
        """Construct a string by appending another string at the end of this string.

        Args:
            other: The string to append.

        Returns:
            A new string containing the concatenation of `self` and `other`.
        """

        var string = self
        string += other.as_string_slice()
        return string

    fn __add__(self, other: InlineString) -> Self:
        """Construct a string by appending another string at the end of this string.

        Args:
            other: The string to append.

        Returns:
            A new string containing the concatenation of `self` and `other`.
        """

        var string = self
        string += other.as_string_slice()
        return string

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    fn __len__(self) -> Int:
        if self._is_small():
            return len(self._storage[_FixedString[Self.SMALL_CAP]])
        else:
            debug_assert(
                self._storage.isa[String](),
                "expected non-small string variant to be String",
            )
            return len(self._storage[String])

    fn __str__(self) -> String:
        if self._is_small():
            return str(self._storage[_FixedString[Self.SMALL_CAP]])
        else:
            return self._storage[String]

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn _is_small(self) -> Bool:
        """Returns True if this string is currently in the small-string
        optimization layout."""
        var res: Bool = self._storage.isa[_FixedString[Self.SMALL_CAP]]()

        return res

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Returns a pointer to the bytes of string data.

        Returns:
            The pointer to the underlying memory.
        """

        if self._is_small():
            return self._storage[_FixedString[Self.SMALL_CAP]].unsafe_ptr()
        else:
            return self._storage[String].unsafe_ptr()

    @always_inline
    fn as_string_slice(ref [_]self: Self) -> StringSlice[__lifetime_of(self)]:
        """Returns a string slice of the data owned by this inline string.

        Returns:
            A string slice pointing to the data owned by this inline string.
        """

        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in _FixedString so this is actually
        #   guaranteed to be valid.
        return StringSlice(unsafe_from_utf8=self.as_bytes_slice())

    @always_inline
    fn as_bytes_slice(ref [_]self: Self) -> Span[UInt8, __lifetime_of(self)]:
        """
        Returns a contiguous slice of the bytes owned by this string.

        This does not include the trailing null terminator.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        return Span[UInt8, __lifetime_of(self)](
            unsafe_ptr=self.unsafe_ptr(),
            # Does NOT include the NUL terminator.
            len=len(self),
        )


# ===----------------------------------------------------------------------===#
# __FixedString
# ===----------------------------------------------------------------------===#


@value
struct _FixedString[CAP: Int](
    Sized, Stringable, Formattable, ToFormatter, CollectionElement
):
    """A string with a fixed available capacity.

    The string data is stored inline in this structs memory layout.

    Parameters:
        CAP: The fixed-size count of bytes of string storage capacity available.
    """

    # Fields
    var buffer: InlineArray[UInt8, CAP]
    """The underlying storage for the fixed string."""
    var size: Int
    """The number of elements in the vector."""

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    fn __init__(inout self):
        """Constructs a new empty string."""
        self.buffer = InlineArray[UInt8, CAP](unsafe_uninitialized=True)
        self.size = 0

    @always_inline
    fn __init__(inout self, literal: StringLiteral) raises:
        """Constructs a FixedString value given a string literal.

        Args:
            literal: The input constant string.
        """
        if len(literal) > CAP:
            raise Error(
                "String literal (len="
                + str(len(literal))
                + ") is longer than FixedString capacity ("
                + str(CAP)
                + ")"
            )

        self.buffer = InlineArray[UInt8, CAP]()
        self.size = len(literal)

        memcpy(self.buffer.unsafe_ptr(), literal.unsafe_ptr(), len(literal))

    # ===------------------------------------------------------------------=== #
    # Factory methods
    # ===------------------------------------------------------------------=== #

    @staticmethod
    fn format_sequence[*Ts: Formattable](*args: *Ts) -> Self:
        """
        Construct a string by concatenating a sequence of formattable arguments.

        Args:
            args: A sequence of formattable arguments.

        Parameters:
            Ts: The types of the arguments to format. Each type must be satisfy
              `Formattable`.

        Returns:
            A string formed by formatting the argument sequence.
        """

        var output = Self()
        var writer = output._unsafe_to_formatter()

        @parameter
        fn write_arg[T: Formattable](arg: T):
            arg.format_to(writer)

        args.each[write_arg]()

        return output^

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    fn __iadd__(inout self, literal: StringLiteral) raises:
        """Appends another string to this string.

        Args:
            literal: The string to append.
        """
        self.__iadd__(literal.as_string_slice())

    fn __iadd__(inout self, string: String) raises:
        """Appends another string to this string.

        Args:
            string: The string to append.
        """
        self.__iadd__(string.as_string_slice())

    @always_inline
    fn __iadd__(inout self, str_slice: StringSlice[_]) raises:
        """Appends another string to this string.

        Args:
            str_slice: The string to append.
        """
        var err = self._iadd_non_raising(str_slice)
        if err:
            raise err.value()

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    fn __str__(self) -> String:
        return String(self.as_string_slice())

    fn __len__(self) -> Int:
        return self.size

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    fn _iadd_non_raising(
        inout self,
        str_slice: StringSlice[_],
    ) -> Optional[Error]:
        var total_len = len(self) + str_slice._byte_length()

        # Ensure there is sufficient capacity to append `str_slice`
        if total_len > CAP:
            return Optional(
                Error(
                    "Insufficient capacity to append len="
                    + str(str_slice._byte_length())
                    + " string to len="
                    + str(len(self))
                    + " FixedString with capacity="
                    + str(CAP),
                )
            )

        # Append the bytes from `str_slice` at the end of the current string
        memcpy(
            dest=self.buffer.unsafe_ptr() + len(self),
            src=str_slice.unsafe_ptr(),
            count=str_slice._byte_length(),
        )

        self.size = total_len

        return None

    fn format_to(self, inout writer: Formatter):
        writer.write_str(self.as_string_slice())

    fn _unsafe_to_formatter(inout self) -> Formatter:
        fn write_to_string(ptr0: UnsafePointer[NoneType], strref: StringRef):
            var ptr: UnsafePointer[Self] = ptr0.bitcast[Self]()

            var str_slice = StringSlice[ImmutableStaticLifetime](
                unsafe_from_utf8_strref=strref
            )

            # FIXME(#37990):
            #   Use `ptr[] += str_slice` and remove _iadd_non_raising after
            #   "failed to fold operation lit.try" is fixed.
            # try:
            #     ptr[] += str_slice
            # except e:
            #     abort("error formatting to FixedString: " + str(e))
            var err = ptr[]._iadd_non_raising(str_slice)
            if err:
                abort("error formatting to FixedString: " + str(err.value()))

        return Formatter(
            write_to_string,
            # Arg data
            UnsafePointer.address_of(self).bitcast[NoneType](),
        )

    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The pointer to the underlying memory.
        """
        return self.buffer.unsafe_ptr()

    @always_inline
    fn as_string_slice(ref [_]self: Self) -> StringSlice[__lifetime_of(self)]:
        """Returns a string slice of the data owned by this fixed string.

        Returns:
            A string slice pointing to the data owned by this fixed string.
        """

        # FIXME(MSTDL-160):
        #   Enforce UTF-8 encoding in _FixedString so this is actually
        #   guaranteed to be valid.
        return StringSlice(unsafe_from_utf8=self.as_bytes_slice())

    @always_inline
    fn as_bytes_slice(ref [_]self: Self) -> Span[UInt8, __lifetime_of(self)]:
        """
        Returns a contiguous slice of the bytes owned by this string.

        This does not include the trailing null terminator.

        Returns:
            A contiguous slice pointing to the bytes owned by this string.
        """

        return Span[UInt8, __lifetime_of(self)](
            unsafe_ptr=self.unsafe_ptr(),
            # Does NOT include the NUL terminator.
            len=self.size,
        )
