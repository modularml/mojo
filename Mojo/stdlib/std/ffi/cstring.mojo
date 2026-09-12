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
"""Implements C string interoperability utilities."""

from std.collections.string.string_span import _unsafe_strlen
from std.memory import MaybeUninit
from std.sys import size_of
from std.utils._nicheable import UnsafeNicheable, NicheIndex


@inline(.always)
def _validate_bytes(slice: Span[Byte, _]) raises:
    var length = Int(_unsafe_strlen(slice.unsafe_ptr(), Int(len(slice))))
    if length == len(slice) - 1:
        return
    elif length == 0 or length == len(slice):
        raise Error("CStringSpan is not nul-terminated")
    else:
        raise Error("CStringSpan has interior nul byte")


struct CStringSpan[origin: ImmOrigin](
    Equatable,
    ImplicitlyCopyable,
    Sized,
    TrivialRegisterPassable,
    UnsafeNicheable,
    Writable,
):
    """A non-owning immutable view to a nul-terminated C string (`const char*`).

    This type can be safely constructed from any sort of `StringSpan` or
    `Span[Byte]` that is nul-terminated, or unsafely from a raw pointer.

    Parameters:
        origin: The origin of the `CStringSpan`.
    """

    comptime _PointerType = Pointer[Int8, Self.origin]

    var _data: Self._PointerType

    @inline(.always)
    def __init__(
        out self,
        *,
        unsafe_from_ptr: Pointer[Int8, Self.origin],
    ):
        """Construct a `CStringSpan` from a `Pointer`.

        Args:
            unsafe_from_ptr: The `Pointer` to construct the `CStringSpan` from.

        Safety:
            The `Pointer` must be a valid nul-terminated C string.
            The pointer cannot be null. To represent nullability, use
            `Optional[CStringSpan]`.

        Example:

        ```mojo
        from std.ffi import c_char, CStringSpan, external_call

        def getenv_wrapper(
            name: CStringSpan,
        ) raises -> CStringSpan[ImmStaticOrigin]:
            # External call to 'getenv'.
            # C signature: const char *getenv(const char *name);
            var result = external_call[
                "getenv",
                Optional[CStringSpan[ImmStaticOrigin]],
            ](name)

            try:
                # Optional.__getitem__ raises an error if empty.
                return result[]
            except:
                raise Error("getenv returned an error!")
        ```
        """
        self._data = unsafe_from_ptr

    @inline(.always)
    def __init__(out self, span: StringSpan[Self.origin]) raises:
        """Construct a `CStringSpan` from a `StringSpan`.

        Args:
            span: The `StringSpan` to construct the `CStringSpan` from.

        Raises:
            An error if the span is not nul-terminated or has interior nul
            bytes.

        Example:

        ```mojo
        from std.ffi import CStringSpan
        from std.testing import assert_raises

        var string = String("Hello, World!")

        with assert_raises():
            # This will raise an error since the string is not nul-terminated.
            _ = CStringSpan(string)
        ```
        """
        _validate_bytes(span.as_bytes())
        # Safety: _validate_bytes ensures span is a non-null terminated cstring.
        self._data = span.as_bytes().unsafe_ptr().unsafe_bitcast[Int8]()

    @inline(.always)
    def __init__(out self, span: Span[Byte, Self.origin]) raises:
        """Construct a `CStringSpan` from a `Span[Byte]`.

        Args:
            span: The `Span[Byte]` to construct the `CStringSpan` from.

        Raises:
            An error if the span is not nul-terminated or has interior nul
            bytes.
        """
        _validate_bytes(span)
        # Safety: _validate_bytes ensures span is a non-null terminated cstring.
        self._data = span.unsafe_ptr().unsafe_bitcast[Int8]()

    @inline(.always)
    def __eq__(self, rhs_same: Self) -> Bool:
        """Compare two `CStringSpan`s for equality.

        Args:
            rhs_same: The `CStringSpan` to compare against.

        Returns:
            True if the `CStringSpan`s are equal, False otherwise.
        """
        return Self.__eq__(self, rhs=rhs_same)

    @inline(.always)
    def __eq__(self, rhs: CStringSpan) -> Bool:
        """Compare two `CStringSpan`s for equality.

        Args:
            rhs: The `CStringSpan` to compare against.

        Returns:
            True if the `CStringSpan`s are equal, False otherwise.
        """
        var a = self.ptr()
        var b = rhs.ptr()
        if a == b:
            return True

        while a[] == b[]:
            if a[] == Int8(0):
                return True
            a = a.unsafe_offset(1)
            b = b.unsafe_offset(1)
        return False

    @inline(.always)
    def __ne__(self, rhs: CStringSpan) -> Bool:
        """Compare two `CStringSpan`s for inequality.

        Args:
            rhs: The `CStringSpan` to compare against.

        Returns:
            True if the `CStringSpan`s are not equal, False otherwise.
        """
        return not (self == rhs)

    @inline(.always)
    def __len__(self) -> Int:
        """Get the length of the C string. Like C's strlen this does not include
        the nul terminator.

        Returns:
            The length of the C string.
        """
        return Int(_unsafe_strlen(self._data.unsafe_bitcast[Byte]()))

    def write_to(self, mut writer: Some[Writer]):
        """Write the `CStringSpan` to a `Writer`, the nul terminator is
        omitted.

        Args:
            writer: The `Writer` to write the `CStringSpan` to.
        """
        # TODO: This should error if the bytes are not valid UTF-8.
        writer.write_string(StringSpan(unsafe_from_utf8=self.as_bytes()))

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the string representation of this `CStringSpan` to a `Writer`.

        Args:
            writer: The `Writer` to write the `CStringSpan` to.
        """
        t"CStringSpan({self.as_bytes_with_nul()})".write_to(writer)

    @inline(.always)
    def ptr(self) -> Pointer[Int8, Self.origin]:
        """Get a pointer to the underlying `CStringSpan`.

        Returns:
            A pointer to the underlying `CStringSpan`.
        """
        return self._data

    @doc_hidden
    @inline(.always)
    @deprecated(use=ptr)
    def unsafe_ptr(self) -> Pointer[Int8, Self.origin]:
        """Get a pointer to the underlying `CStringSpan`.

        Returns:
            A pointer to the underlying `CStringSpan`.
        """
        return self.ptr()

    @inline(.always)
    def as_bytes(self) -> Span[Byte, Self.origin]:
        """Get a span of the underlying `CStringSpan` as bytes.

        The returned span does not include the nul terminator.
        If you want a byte span including the nul terminator, use
        `as_bytes_with_nul()`.

        Returns:
            A span of the underlying `CStringSpan` as bytes.
        """
        return Span[Byte, Self.origin](
            unsafe_ptr=self._data.unsafe_bitcast[Byte](),
            length=len(self),
        )

    @inline(.always)
    def as_bytes_with_nul(self) -> Span[Byte, Self.origin]:
        """Get a span of the underlying `CStringSpan` as bytes including the
        nul terminator.

        If you want a byte span not including the nul terminator, use
        `as_bytes()`.

        Returns:
            A span of the underlying `CStringSpan` as bytes.
        """
        return Span[Byte, Self.origin](
            unsafe_ptr=self._data.unsafe_bitcast[Byte](),
            length=len(self) + 1,
        )

    @doc_hidden
    def as_unsafe_any_origin(self) -> CStringSpan[ImmutAnyOrigin]:
        return {unsafe_from_ptr = self._data.as_unsafe_any_origin()}

    @staticmethod
    @doc_hidden
    @inline(.always)
    def niche_count() -> Int:
        return Self._PointerType.niche_count()

    @staticmethod
    @doc_hidden
    @inline(.always)
    def write_niche[index: Int](memory: MutPointer[MaybeUninit[Self], _]):
        comptime assert size_of[Self]() == size_of[Self._PointerType]()
        Self._PointerType.write_niche[index](
            memory.unsafe_bitcast[MaybeUninit[Self._PointerType]]()
        )

    @staticmethod
    @doc_hidden
    @inline(.always)
    def classify_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> NicheIndex:
        comptime assert size_of[Self]() == size_of[Self._PointerType]()
        return Self._PointerType.classify_niche(
            memory.unsafe_bitcast[MaybeUninit[Self._PointerType]]()
        )


@deprecated(use=CStringSpan)
comptime CStringSlice = CStringSpan
"""Provides a compatibility alias for `CStringSpan`."""
