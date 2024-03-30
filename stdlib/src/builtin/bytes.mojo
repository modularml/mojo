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
"""Defines the Bytes type.

The Bytes type is the equalivalent of the bytes type in Python. It is an
array of values between 0 and 255. It is used to represent binary data.

This type cannot be used without imports yet because it's not ready for a wider audience.
"""
from collections import Optional


struct Bytes(Sized, CollectionElement):
    """A mutable sequence of bytes. Behaves like the python version.

    Note that `some_bytes[i]` returns an UInt8.
    some_bytes *= 2 modifies the sequence in-place. Same with +=.

    Also `__setitem__` is available, meaning you can do `some_bytes[7] = 105` or
    even `some_bytes[7] = some_other_bytes` (the latter must be only one byte long).

    You can create bytes from a list of UInt8 values:
    ```mojo
    var some_bytes = Bytes(List[UInt8](1, 2, 3, 4))
    print(some_bytes)
    # b'\x01\x02\x03\x04'
    ```

    or you can create bytes, set to 0, by specifying the size:
    ```mojo
    var some_bytes = Bytes(4)
    print(some_bytes)
    # b'\x00\x00\x00\x00'
    ```

    An empty constructor means a sequence of 0 bytes.
    ```mojo
    var some_bytes = Bytes()
    print(some_bytes)
    # b''
    ```
    """

    alias _storage_type = List[UInt8]

    var _data: Self._storage_type

    fn __init__(inout self, owned data: Self._storage_type, /):
        """Creates a Bytes object from a list of UInt8 values.

        Args:
            data: The list of UInt8 values to create the Bytes object from. Positional-only.
        """
        self._data = data^

    fn __init__(inout self, size: Int = 0, /):
        """Creates a Bytes object of a given size, filled with 0s.

        Args:
            size: The size of the Bytes object. Defaults to 0.
        """
        self.__init__(size, capacity=size)

    fn __init__(inout self, size: Int, /, *, capacity: Int):
        """Creates a Bytes object of a given size, filled with 0s.

        Args:
            size: The size of the Bytes object. Defaults to 0.
            capacity: The capacity of the underlying `List[UInt8]`.
        """
        self._data = Self._storage_type(capacity=capacity)
        for i in range(size):
            self._data.append(0)

    fn __len__(self) -> Int:
        """Returns the number of bytes present in the `Bytes` object."""
        return len(self._data)

    fn __getitem__(self, index: Int) -> UInt8:
        """Returns the byte at the given index.

        The returned value is an UInt8 (so in the range 0-255).
        """
        return self._data[index]

    fn __setitem__(inout self, index: Int, value: UInt8):
        """Sets the byte at the given index to the given value.

        Args:
            index: The index of the byte to set.
            value: The value to set the byte to. Must be convertible to an
                UInt8, for example an integer between 0 and 255 works.
        """
        self._data[index] = value

    fn __copyinit__(inout self, existing: Self):
        self._data = Self._storage_type(existing._data)

    fn __moveinit__(inout self, owned existing: Self):
        self._data = existing._data^
