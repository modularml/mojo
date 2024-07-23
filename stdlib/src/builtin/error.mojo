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

from memory import UnsafePointer, memcpy
from memory.memory import _free

# ===----------------------------------------------------------------------===#
# Error
# ===----------------------------------------------------------------------===#


@register_passable
struct Error(
    Stringable,
    Boolable,
    Representable,
    Formattable,
    CollectionElement,
):
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

    @always_inline
    fn __init__() -> Self:
        """Default constructor.

        Returns:
            The constructed Error object.
        """
        return Error {data: UnsafePointer[UInt8](), loaded_length: 0}

    @always_inline
    fn __init__(value: StringLiteral) -> Self:
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

    fn __init__(src: String) -> Self:
        """Construct an Error object with a given string.

        Args:
            src: The error message.

        Returns:
            The constructed Error object.
        """
        var length = src.byte_length()
        var dest = UnsafePointer[UInt8].alloc(length + 1)
        memcpy(
            dest=dest,
            src=src.unsafe_ptr(),
            count=length,
        )
        dest[length] = 0
        return Error {data: dest, loaded_length: -length}

    fn __init__(src: StringRef) -> Self:
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

    fn __init__(*, other: Self) -> Self:
        """Copy the object.

        Args:
            other: The value to copy.

        Returns:
            The copied `Error`.
        """
        return other

    fn __del__(owned self):
        """Releases memory if allocated."""
        if self.loaded_length < 0:
            self.data.free()

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

    @no_inline
    fn __str__(self) -> String:
        """Converts the Error to string representation.

        Returns:
            A String of the error message.
        """
        return String.format_sequence(self)

    @no_inline
    fn format_to(self, inout writer: Formatter):
        """
        Formats this error to the provided formatter.

        Args:
            writer: The formatter to write to.
        """

        # TODO: Avoid this unnecessary intermediate String allocation.
        writer.write(self._message())

    @no_inline
    fn __repr__(self) -> String:
        """Converts the Error to printable representation.

        Returns:
            A printable representation of the error message.
        """
        return "Error(" + repr(self._message()) + ")"

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


@export("__mojo_debugger_raise_hook")
fn __mojo_debugger_raise_hook():
    """This function is used internally by the Mojo Debugger."""
    pass
