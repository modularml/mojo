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

"""Implements the StringSlice type.

You can import these APIs from the `utils.string_slice` module. For example:

```mojo
from utils import StringSlice
```
"""

from utils import Span
from builtin.string import _isspace

alias StaticString = StringSlice[ImmutableStaticLifetime]
"""An immutable static string slice."""


struct StringSlice[
    is_mutable: Bool, //,
    lifetime: AnyLifetime[is_mutable].type,
](Stringable, Sized, Formattable):
    """
    A non-owning view to encoded string data.

    TODO:
    The underlying string data is guaranteed to be encoded using UTF-8.

    Parameters:
        is_mutable: Whether the slice is mutable.
        lifetime: The lifetime of the underlying string data.
    """

    var _slice: Span[UInt8, lifetime]

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    fn __init__(inout self, literal: StringLiteral):
        """Construct a new string slice from a string literal.

        Args:
            literal: The literal to construct this string slice from.
        """

        # Its not legal to try to mutate a StringLiteral. String literals are
        # static data.
        constrained[
            not is_mutable, "cannot create mutable StringSlice of StringLiteral"
        ]()

        # Since a StringLiteral has static lifetime, it will outlive
        # whatever arbitrary `lifetime` the user has specified they need this
        # slice to live for.
        # SAFETY:
        #   StringLiteral is guaranteed to use UTF-8 encoding.
        # FIXME(MSTDL-160):
        #   Ensure StringLiteral _actually_ always uses UTF-8 encoding.
        self = StringSlice[lifetime](
            unsafe_from_utf8_ptr=literal.unsafe_ptr(),
            len=literal._byte_length(),
        )

    @always_inline
    fn __init__(inout self, *, owned unsafe_from_utf8: Span[UInt8, lifetime]):
        """
        Construct a new StringSlice from a sequence of UTF-8 encoded bytes.

        Safety:
            `unsafe_from_utf8` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8: A slice of bytes encoded in UTF-8.
        """

        self._slice = unsafe_from_utf8^

    fn __init__(inout self, *, unsafe_from_utf8_strref: StringRef):
        """
        Construct a new StringSlice from a StringRef pointing to UTF-8 encoded
        bytes.

        Safety:
            - `unsafe_from_utf8_strref` MUST point to data that is valid for
              `lifetime`.
            - `unsafe_from_utf8_strref` MUST be valid UTF-8 encoded data.

        Args:
            unsafe_from_utf8_strref: A StringRef of bytes encoded in UTF-8.
        """
        var strref = unsafe_from_utf8_strref

        var byte_slice = Span[UInt8, lifetime](
            unsafe_ptr=strref.unsafe_ptr(),
            len=len(strref),
        )

        self = Self(unsafe_from_utf8=byte_slice)

    @always_inline
    fn __init__(
        inout self,
        *,
        unsafe_from_utf8_ptr: UnsafePointer[UInt8],
        len: Int,
    ):
        """
        Construct a StringSlice from a pointer to a sequence of UTF-8 encoded
        bytes and a length.

        Safety:
            - `unsafe_from_utf8_ptr` MUST point to at least `len` bytes of valid
              UTF-8 encoded data.
            - `unsafe_from_utf8_ptr` must point to data that is live for the
              duration of `lifetime`.

        Args:
            unsafe_from_utf8_ptr: A pointer to a sequence of bytes encoded in
              UTF-8.
            len: The number of bytes of encoded data.
        """
        var byte_slice = Span[UInt8, lifetime](
            unsafe_ptr=unsafe_from_utf8_ptr,
            len=len,
        )

        self._slice = byte_slice

    # ===------------------------------------------------------------------===#
    # Trait implementations
    # ===------------------------------------------------------------------===#

    fn __str__(self) -> String:
        return String(str_slice=self)

    fn __len__(self) -> Int:
        """Nominally returns the _length in Unicode codepoints_ (not bytes!).

        Returns:
            The length in Unicode codepoints.
        """
        # FIXME(MSTDL-160):
        #   Actually perform UTF-8 decoding here to count the codepoints.
        return len(self._slice)

    fn format_to(self, inout writer: Formatter):
        """
        Formats this string slice to the provided formatter.

        Args:
            writer: The formatter to write to.
        """
        writer.write_str(str_slice=self)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    fn as_bytes_slice(self) -> Span[UInt8, lifetime]:
        """
        Get the sequence of encoded bytes as a slice of the underlying string.

        Returns:
            A slice containing the underlying sequence of encoded bytes.
        """
        return self._slice

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[UInt8]:
        """
        Gets a pointer to the first element of this string slice.

        Returns:
            A pointer pointing at the first element of this string slice.
        """

        return self._slice.unsafe_ptr()

    @always_inline
    fn _byte_length(self) -> Int:
        """
        Get the length of this string slice in bytes.

        Returns:
            The length of this string slice in bytes.
        """

        return len(self.as_bytes_slice())

    fn _strref_dangerous(self) -> StringRef:
        """
        Returns an inner pointer to the string as a StringRef.

        Safety:
            This functionality is extremely dangerous because Mojo eagerly
            releases strings.  Using this requires the use of the
            _strref_keepalive() method to keep the underlying string alive long
            enough.
        """
        return StringRef(self.unsafe_ptr(), self._byte_length())

    fn _strref_keepalive(self):
        """
        A no-op that keeps `self` alive through the call.  This
        can be carefully used with `_strref_dangerous()` to wield inner pointers
        without the string getting deallocated early.
        """
        pass

    fn isspace(self) -> Bool:
        """Determines whether the given StringSlice is a python
        whitespace String. This corresponds to Python's
        [universal separators](
            https://docs.python.org/3/library/stdtypes.html#str.splitlines)
        `" \\t\\n\\r\\f\\v\\x1c\\x1d\\x1e\\x85\\u2028\\u2029"`.

        Returns:
            True if the String is one of the whitespace characters
                listed above, otherwise False.
        """
        # TODO add line and paragraph separator as stringliteral
        # once unicode escape secuences are accepted
        var next_line = List[UInt8](0xC2, 0x85)
        """TODO: \\x85"""
        var unicode_line_sep = List[UInt8](0xE2, 0x80, 0xA8)
        """TODO: \\u2028"""
        var unicode_paragraph_sep = List[UInt8](0xE2, 0x80, 0xA9)
        """TODO: \\u2029"""

        @always_inline
        fn _compare(
            item1: UnsafePointer[UInt8], item2: UnsafePointer[UInt8], amnt: Int
        ) -> Bool:
            var ptr1 = DTypePointer(item1)
            var ptr2 = DTypePointer(item2)
            return memcmp(ptr1, ptr2, amnt) == 0

        var no_null_len = len(self)
        var ptr = self.unsafe_ptr()
        if no_null_len == 1 and _isspace(ptr[0]):
            return True
        elif no_null_len == 2 and _compare(ptr, next_line.unsafe_ptr(), 2):
            return True
        elif no_null_len == 3 and (
            _compare(ptr, unicode_line_sep.unsafe_ptr(), 3)
            or _compare(ptr, unicode_paragraph_sep.unsafe_ptr(), 3)
        ):
            return True
        _ = next_line, unicode_line_sep, unicode_paragraph_sep
        return False
