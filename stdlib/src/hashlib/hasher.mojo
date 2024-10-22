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

"""Defines the `Hashable` and `Hasher` traits and provides the default hasher type."""

from .fnv1a import Fnv1a
from memory import UnsafePointer


trait Hashable:
    """A trait for types which specify a function to hash their data.
    The type receives a `hasher`, and contributes its properties
    by calling the `update` function on the `hasher`.

        ```mojo
        struct Point(Hashable):
            var x: Float64
            var y: Float64

            fn __hash__[H: Hasher](self, inout hasher: H):
                hasher.update(self.x)
                hasher.update(self.y)
        ```
    """

    fn __hash__[H: Hasher](self, mut hasher: H):
        """Function to contribute the properties.

        Parameters:
            H: The hasher type.

        Args:
            hasher: Hasher instance which produces the hash value.
        """
        ...


trait Hasher:
    """A trait for types which implement a hash function.
    The type implements functions to update its internal state.
    The hash value is produced when `finish` function is called.

        ```mojo
        struct DummyHasher(Hasher):
            var _dummy_value: UInt64

            fn __init__(inout self):
                self._dummy_value = 0

            fn _update_with_bytes(inout self, data: UnsafePointer[UInt8], length: Int):
                for i in range(length):
                    self._dummy_value += data[i].cast[DType.uint64]()

            fn _update_with_simd(inout self, value: SIMD[_, _]):
                self._dummy_value += value.cast[DType.uint64]().reduce_add()

            fn update[T: Hashable](inout self, value: T):
                value.__hash__(self)

            fn finish(owned self) -> UInt64:
                return self._dummy_value

        ```
    """

    fn __init__(out self):
        """Initialise the hasher."""
        ...

    fn _update_with_bytes(
        mut self, new_data: UnsafePointer[UInt8], length: Int
    ):
        """Consume provided data to update the internal buffer.

        Args:
            new_data: Pointer to the byte array.
            length: The length of the byte array.
        """
        ...

    fn _update_with_simd(mut self, new_data: SIMD[_, _]):
        """Update the buffer value with new data.

        Args:
            new_data: Value used for update.
        """
        ...

    fn update[T: Hashable](mut self, value: T):
        """Update the buffer value with new hashable value.

        Parameters:
            T: Hashable type.

        Args:
            value: Value used for update.
        """
        ...

    fn finish(owned self) -> UInt64:
        """Computes the hash value based on all the previously provided data.

        Returns:
            Final hash value.
        """
        ...


alias default_hasher = Fnv1a
