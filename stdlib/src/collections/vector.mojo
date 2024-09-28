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
"""Defines InlinedFixedVector.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import InlinedFixedVector
```
"""

from memory.maybe_uninitialized import UnsafeMaybeUninitialized
from memory import Reference, UnsafePointer, memcpy
from collections import inline_array
from sys import sizeof

# ===----------------------------------------------------------------------===#
# _VecIter
# ===----------------------------------------------------------------------===#


@value
struct _VecIter[
    mutability: Bool, //,
    type: CollectionElement,
    static_size: Int,
    lifetime: Lifetime[mutability].type,
](Sized):
    """Iterator for any random-access container"""

    var i: Int
    var size: Int
    var vec: Reference[InlinedFixedVector[type, static_size], lifetime]

    fn __next__(inout self) -> Reference[type, lifetime]:
        # TODO: Return an autoderef
        self.i += 1
        return self.vec[][self.i - 1]

    fn __len__(self) -> Int:
        return self.size - self.i


# ===----------------------------------------------------------------------===#
# InlinedFixedVector
# ===----------------------------------------------------------------------===#


@always_inline
fn _calculate_fixed_vector_default_size[type: CollectionElement]() -> Int:
    alias prefered_bytecount = 64
    alias sizeof_type = sizeof[type]()

    @parameter
    if sizeof_type >= 256:
        return 0

    alias prefered_inline_bytes = prefered_bytecount - sizeof[
        InlinedFixedVector[type, 0]
    ]()
    alias num_elements = prefered_inline_bytes // sizeof_type
    return num_elements or 1


struct InlinedFixedVector[
    type: CollectionElement,
    size: Int = _calculate_fixed_vector_default_size[type](),
](Sized, CollectionElement):
    """A dynamically-allocated vector with small-vector optimization and a fixed
    maximum capacity.

    The `InlinedFixedVector` does not resize or implement bounds checks. It is
    initialized with both a small-vector size (specified at compile time) and a
    maximum capacity (specified at runtime).

    The first `size` elements are stored in the statically-allocated small
    vector storage. Any remaining elements are stored in dynamically-allocated
    storage.

    The destructor of it's elements is called when the vector itself is destructed,
    it then deallocate its memory to complete the cleanup.

    This data structure is useful for applications where the number of required
    elements is not known at compile time, but once known at runtime, is
    guaranteed to be equal to or less than a certain capacity.

    Parameters:
        type: The type of the elements.
        size: The statically-known small-vector size.
    """

    alias static_size: Int = size
    alias static_data_type = InlineArray[
        UnsafeMaybeUninitialized[type], size, run_destructors=False
    ]
    var static_data: Self.static_data_type
    """The underlying static storage, used for small vectors."""
    var dynamic_data: UnsafePointer[type]
    """The underlying dynamic storage, used to grow large vectors."""
    var current_size: Int
    """The number of elements in the vector."""
    var capacity: Int
    """The maximum number of elements that can fit in the vector."""

    @always_inline
    fn __init__(inout self, capacity: Int):
        """Constructs `InlinedFixedVector` with the given capacity.

        The dynamically allocated portion is `capacity - size`.

        Args:
            capacity: The requested maximum capacity of the vector.
        """
        debug_assert(capacity >= 0)
        self.static_data = Self.static_data_type(unsafe_uninitialized=True)
        self.dynamic_data = UnsafePointer[type]()
        if capacity > Self.static_size:
            self.dynamic_data = UnsafePointer[type].alloc(
                capacity - Self.static_size
            )
        self.current_size = 0
        self.capacity = capacity

    @always_inline
    fn __copyinit__(inout self, existing: Self):
        """
        Copy constructor.

        Args:
            existing: The `InlinedFixedVector` to copy.
        """
        self.current_size = existing.current_size
        self.capacity = existing.capacity
        debug_assert(self.capacity >= 0)

        self.static_data = Self.static_data_type(unsafe_uninitialized=True)
        self.dynamic_data = UnsafePointer[type]()

        if self.capacity > self.static_size:
            debug_assert(existing.dynamic_data)
            allocated = self.capacity - self.static_size
            self.dynamic_data = UnsafePointer[type].alloc(allocated)

        remaining_elements = existing.current_size
        if remaining_elements > existing.static_size:
            dyn_size = remaining_elements - self.static_size
            for i in range(dyn_size):
                self.dynamic_data.offset(i).init_pointee_copy(
                    existing.dynamic_data.offset(i)[]
                )
            remaining_elements -= dyn_size

        debug_assert(remaining_elements <= self.static_size)
        for i in range(remaining_elements):
            self.static_data.unsafe_ptr().offset(i)[].write(existing[i])

    @always_inline
    fn __moveinit__(inout self, owned existing: Self):
        """
        Move constructor.

        Args:
            existing: The `InlinedFixedVector` to consume.
        """
        self.static_data = Self.static_data_type(unsafe_uninitialized=True)
        self.dynamic_data = existing.dynamic_data
        memcpy(
            self.static_data.unsafe_ptr(),
            existing.static_data.unsafe_ptr(),
            existing.static_size,
        )
        self.current_size = existing.current_size
        self.capacity = existing.capacity

        existing.dynamic_data = UnsafePointer[type]()
        existing.current_size = 0

    @always_inline
    fn __del__(owned self):
        """
        Destructor.
        """
        debug_assert(self.current_size <= self.capacity)
        debug_assert(self.current_size >= 0)

        if self.current_size > self.static_size:
            dyn_elem = self.current_size - self.static_size
            for i in range(dyn_elem):
                self.dynamic_data.offset(i).destroy_pointee()
            self.current_size -= dyn_elem

        debug_assert(self.current_size <= self.static_size)
        for idx in range(self.current_size):
            (self.static_data[idx]).unsafe_ptr().destroy_pointee()

        self.dynamic_data.free()
        self.dynamic_data = UnsafePointer[type]()

    @always_inline
    fn append(inout self, owned value: type):
        """Appends a value to this vector.

        Args:
            value: The value to append.
        """
        debug_assert(self.current_size < self.capacity)
        debug_assert(self.current_size >= 0)

        if self.current_size < Self.static_size:
            self.static_data[self.current_size].write(value^)
        else:
            debug_assert((self.current_size - self.static_size) >= 0)
            (
                self.dynamic_data + (self.current_size - Self.static_size)
            ).init_pointee_move(value^)

        self.current_size += 1

    @always_inline
    fn __len__(self) -> Int:
        """Gets the number of elements in the vector.

        Returns:
            The number of elements in the vector.
        """
        return self.current_size

    @always_inline
    fn __getitem__(ref [_]self, idx: Int) -> ref [__lifetime_of(self)] type:
        """Gets a vector element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            The element at the given index.
        """

        var normalized_idx = idx
        debug_assert(self.current_size <= self.capacity)
        debug_assert(self.current_size >= 0)
        debug_assert(
            -self.current_size <= normalized_idx < self.current_size,
            "index must be within bounds",
        )

        if normalized_idx < 0:
            normalized_idx += len(self)

        debug_assert(len(self) > normalized_idx >= 0)

        if normalized_idx < Self.static_size:
            return UnsafePointer.address_of(
                self.static_data[normalized_idx].assume_initialized()
            )[]

        return self.dynamic_data[normalized_idx - Self.static_size]

    fn clear(inout self):
        """Clears the elements in the vector."""

        debug_assert(self.current_size <= self.capacity)
        debug_assert(self.current_size >= 0)

        if self.current_size > self.static_size:
            dyn_elem = self.current_size - self.static_size
            for i in range(dyn_elem):
                self.dynamic_data.offset(i).destroy_pointee()
            self.current_size -= dyn_elem

        debug_assert(self.current_size <= self.static_size)
        for idx in range(self.current_size):
            (self.static_data[idx]).unsafe_ptr().destroy_pointee()

        self.current_size = 0

    fn __iter__[
        mutability: Bool, //, L: Lifetime[mutability].type
    ](ref [L]self) -> _VecIter[type, size, L]:
        """Iterate over the vector.

        Returns:
            An iterator to the start of the vector.
        """
        debug_assert(self.current_size <= self.capacity)
        return _VecIter[type, size, L](0, self.current_size, self)
