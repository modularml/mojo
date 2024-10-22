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
"""Defines the Deque type.

You can import these APIs from the `collections` package.

Examples:

```mojo
from collections import Deque
```
"""

from bit import bit_ceil
from collections import Optional
from memory import UnsafePointer

from builtin._documentation import doc_private

# ===----------------------------------------------------------------------===#
# Deque
# ===----------------------------------------------------------------------===#


struct Deque[ElementType: CollectionElement](
    Movable, ExplicitlyCopyable, Sized, Boolable
):
    """Implements a double-ended queue.

    It supports pushing and popping from both ends in O(1) time resizing the
    underlying storage as needed.

    Parameters:
        ElementType: The type of the elements in the deque.
            Must implement the trait `CollectionElement`.
    """

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    alias default_capacity: Int = 64
    """The default capacity of the deque: must be the power of 2."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var data: UnsafePointer[ElementType]
    """The underlying storage for the deque."""

    var head: Int
    """The index of the head: points the first element of the deque."""

    var tail: Int
    """The index of the tail: points behind the last element of the deque."""

    var capacity: Int
    """The amount of elements that can fit in the deque without resizing it."""

    var minlen: Int
    """The minimum required capacity in the number of elements of the deque."""

    var maxlen: Int
    """The maximum allowed capacity in the number of elements of the deque."""

    var shrink: Bool
    """The flag defining if the deque storage is re-allocated to make it smaller when possible."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(
        inout self,
        *,
        owned elements: Optional[List[ElementType]] = None,
        capacity: Int = self.default_capacity,
        minlen: Int = self.default_capacity,
        maxlen: Int = -1,
        shrink: Bool = True,
    ):
        """Constructs a deque.

        Args:
            elements: The optional list of initial deque elements.
            capacity: The initial capacity of the deque.
            minlen: The minimum allowed capacity of the deque when shrinking.
            maxlen: The maximum allowed capacity of the deque when growing.
            shrink: Should storage be de-allocated when not needed.
        """
        if capacity <= 0:
            deque_capacity = self.default_capacity
        else:
            deque_capacity = bit_ceil(capacity)

        if minlen <= 0:
            min_capacity = self.default_capacity
        else:
            min_capacity = bit_ceil(minlen)

        if maxlen <= 0:
            max_capacity = -1
        else:
            max_capacity = maxlen
            deque_capacity = min(deque_capacity, bit_ceil(maxlen))

        self.capacity = deque_capacity
        self.data = UnsafePointer[ElementType].alloc(deque_capacity)
        self.head = 0
        self.tail = 0
        self.minlen = min_capacity
        self.maxlen = max_capacity
        self.shrink = shrink

        if elements is not None:
            self.extend(elements.value())

    fn __init__(inout self, owned *values: ElementType):
        """Constructs a deque from the given values.

        Args:
            values: The values to populate the deque with.
        """
        self = Self(variadic_list=values^)

    fn __init__(
        inout self, *, owned variadic_list: VariadicListMem[ElementType, _]
    ):
        """Constructs a deque from the given values.

        Args:
            variadic_list: The values to populate the deque with.
        """
        args_length = len(variadic_list)

        if args_length < self.default_capacity:
            capacity = self.default_capacity
        else:
            capacity = args_length

        self = Self(capacity=capacity)

        for i in range(args_length):
            src = UnsafePointer.address_of(variadic_list[i])
            dst = self.data + i
            src.move_pointee_into(dst)

        # Mark the elements as unowned to avoid del'ing uninitialized objects.
        variadic_list._is_owned = False

        self.tail = args_length

    fn __init__(inout self, other: Self):
        """Creates a deepcopy of the given deque.

        Args:
            other: The deque to copy.
        """
        self = Self(
            capacity=other.capacity,
            minlen=other.minlen,
            maxlen=other.maxlen,
            shrink=other.shrink,
        )
        for i in range(len(other)):
            offset = (other.head + i) & (other.capacity - 1)
            (self.data + i).init_pointee_copy((other.data + offset)[])

        self.tail = len(other)

    fn __moveinit__(inout self, owned existing: Self):
        """Moves data of an existing deque into a new one.

        Args:
            existing: The existing deque.
        """
        self.data = existing.data
        self.capacity = existing.capacity
        self.head = existing.head
        self.tail = existing.tail
        self.minlen = existing.minlen
        self.maxlen = existing.maxlen
        self.shrink = existing.shrink

    fn __del__(owned self):
        """Destroys all elements in the deque and free its memory."""
        for i in range(len(self)):
            offset = (self.head + i) & (self.capacity - 1)
            (self.data + offset).destroy_pointee()
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks whether the deque has any elements or not.

        Returns:
            `False` if the deque is empty, `True` if there is at least one element.
        """
        return self.head != self.tail

    @always_inline
    fn __len__(self) -> Int:
        """Gets the number of elements in the deque.

        Returns:
            The number of elements in the deque.
        """
        return (self.tail - self.head) & (self.capacity - 1)

    fn __getitem__(ref [_]self, idx: Int) -> ref [self] ElementType:
        """Gets the deque element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        normalized_idx = idx

        debug_assert(
            -len(self) <= normalized_idx < len(self),
            "index: ",
            normalized_idx,
            " is out of bounds for `Deque` of size: ",
            len(self),
        )

        if normalized_idx < 0:
            normalized_idx += len(self)

        offset = (self.head + normalized_idx) & (self.capacity - 1)
        return (self.data + offset)[]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn append(inout self, owned value: ElementType):
        """Appends a value to the right side of the deque.

        Args:
            value: The value to append.
        """
        if len(self) == self.maxlen:
            (self.data + self.head).destroy_pointee()
            self.head = (self.head + 1) & (self.capacity - 1)

        (self.data + self.tail).init_pointee_move(value^)
        self.tail = (self.tail + 1) & (self.capacity - 1)

        if self.head == self.tail:
            self._realloc(self.capacity << 1)

    fn extend(inout self, owned values: List[ElementType]):
        """Extends the right side of the deque by consuming elements of the list argument.

        Args:
            values: List whose elements will be added at the right side of the deque.
        """
        for value in values:
            self.append(value[])

    @doc_private
    fn _realloc(inout self, new_capacity: Int):
        """Relocates data to a new storage buffer.

        Args:
            new_capacity: The new capacity of the buffer.
        """
        deque_len = len(self) if self else self.capacity

        tail_len = self.tail
        head_len = self.capacity - self.head

        if head_len > deque_len:
            head_len = deque_len
            tail_len = 0

        new_data = UnsafePointer[ElementType].alloc(new_capacity)

        src = self.data + self.head
        dsc = new_data
        for i in range(head_len):
            (src + i).move_pointee_into(dsc + i)

        src = self.data
        dsc = new_data + head_len
        for i in range(tail_len):
            (src + i).move_pointee_into(dsc + i)

        self.head = 0
        self.tail = deque_len

        if self.data:
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity
