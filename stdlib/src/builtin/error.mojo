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
"""Implements the Error class.

These are Mojo built-ins, so you don't need to import them.
"""

from sys import alignof, sizeof

from memory.memory import _free
from memory import memcmp, memcpy, UnsafePointer

# ===----------------------------------------------------------------------===#
# Error
# ===----------------------------------------------------------------------===#


@register_passable
struct Error(Stringable, Boolable, Representable):
    """This type represents an Error."""

    var data: UnsafePointer[UInt8]
    """A pointer to the beginning of the string data being referenced."""

    var loaded_length: Int
    """The length of the string being referenced.
    Error instances conditionally own their error message. To reduce
    the size of the error instance we use the sign bit of the length field
    to store the ownership value. When loaded_length is negative it indicates
    ownership and a free is executed in the destructor.
    """

    @always_inline("nodebug")
    fn __init__() -> Error:
        """Default constructor.

        Returns:
            The constructed Error object.
        """
        return Error {data: UnsafePointer[UInt8](), loaded_length: 0}

    @always_inline("nodebug")
    fn __init__(value: StringLiteral) -> Error:
        """Construct an Error object with a given string literal.

        Args:
            value: The error message.

        Returns:
            The constructed Error object.
        """
        return Error {
            data: value.unsafe_ptr(),
            loaded_length: len(value),
        }

    @always_inline("nodebug")
    fn __init__(src: String) -> Error:
        """Construct an Error object with a given string.

        Args:
            src: The error message.

        Returns:
            The constructed Error object.
        """
        var length = len(src)
        var dest = UnsafePointer[UInt8].alloc(length + 1)
        memcpy(
            dest=dest,
            # TODO: Remove cast once string UInt8 transition is complete.
            src=src.unsafe_ptr().bitcast[UInt8](),
            count=length,
        )
        dest[length] = 0
        return Error {data: dest, loaded_length: -length}

    @always_inline("nodebug")
    fn __init__(src: StringRef) -> Error:
        """Construct an Error object with a given string ref.

        Args:
            src: The error message.

        Returns:
            The constructed Error object.
        """
        var length = len(src)
        var dest = UnsafePointer[UInt8].alloc(length + 1)
        memcpy(
            dest=dest,
            src=src.unsafe_ptr(),
            count=length,
        )
        dest[length] = 0
        return Error {data: dest, loaded_length: -length}

    fn __del__(owned self):
        """Releases memory if allocated."""
        if self.loaded_length < 0:
            self.data.free()

    @always_inline("nodebug")
    fn __copyinit__(existing: Self) -> Self:
        """Creates a deep copy of an existing error.

        Returns:
            The copy of the original error.
        """
        if existing.loaded_length < 0:
            var length = -existing.loaded_length
            var dest = UnsafePointer[UInt8].alloc(length + 1)
            memcpy(dest, existing.data, length)
            dest[length] = 0
            return Error {data: dest, loaded_length: existing.loaded_length}
        else:
            return Error {
                data: existing.data, loaded_length: existing.loaded_length
            }

    fn __bool__(self) -> Bool:
        """Returns True if the error is set and false otherwise.

        Returns:
          True if the error object contains a value and False otherwise.
        """
        return self.data.__bool__()

    fn __str__(self) -> String:
        """Converts the Error to string representation.

        Returns:
            A String of the error message.
        """
        return self._message()

    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        return "Error(" + repr(self._message()) + ")"

    @always_inline
    fn _message(self) -> String:
        """Converts the Error to string representation.

        Returns:
            A String of the error message.
        """
        if not self:
            return ""

        var length = self.loaded_length
        if length < 0:
            length = -length
        return String(StringRef(self.data, length))
