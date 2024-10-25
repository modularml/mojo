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
from bit import bit_ceil, is_power_of_two
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

    var _data: UnsafePointer[ElementType]
    """The underlying storage for the deque."""

    var _head: Int
    """The index of the head: points the first element of the deque."""

    var _tail: Int
    """The index of the tail: points behind the last element of the deque."""

    var _capacity: Int
    """The amount of elements that can fit in the deque without resizing it."""

    var _min_capacity: Int
    """The minimum required capacity in the number of elements of the deque."""

    var _maxlen: Int
    """The maximum number of elements allowed in the deque.

    If more elements are pushed, causing the total to exceed this limit,
    items will be popped from the opposite end to maintain the maximum length.
    """

    var _shrink: Bool
    """ Indicates whether the deque's storage is re-allocated to a smaller size when possible."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(
        inout self,
        *,
        owned elements: Optional[List[ElementType]] = None,
        capacity: Int = self.default_capacity,
        min_capacity: Int = self.default_capacity,
        maxlen: Int = -1,
        shrink: Bool = True,
    ):
        """Constructs a deque.

        Args:
            elements: The optional list of initial deque elements.
            capacity: The initial capacity of the deque.
            min_capacity: The minimum allowed capacity of the deque when shrinking.
            maxlen: The maximum allowed capacity of the deque when growing.
            shrink: Should storage be de-allocated when not needed.
        """
        if capacity <= 0:
            deque_capacity = self.default_capacity
        else:
            deque_capacity = bit_ceil(capacity)

        if min_capacity <= 0:
            min_deque_capacity = self.default_capacity
        else:
            min_deque_capacity = bit_ceil(min_capacity)

        if maxlen <= 0:
            max_deque_len = -1
        else:
            max_deque_len = maxlen
            max_deque_capacity = bit_ceil(maxlen)
            if is_power_of_two(maxlen):
                max_deque_capacity <<= 1
            deque_capacity = min(deque_capacity, max_deque_capacity)

        self._capacity = deque_capacity
        self._data = UnsafePointer[ElementType].alloc(deque_capacity)
        self._head = 0
        self._tail = 0
        self._min_capacity = min_deque_capacity
        self._maxlen = max_deque_len
        self._shrink = shrink

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
            dst = self._data + i
            src.move_pointee_into(dst)

        # Mark the elements as unowned to avoid del'ing uninitialized objects.
        variadic_list._is_owned = False

        self._tail = args_length

    fn __init__(inout self, other: Self):
        """Creates a deepcopy of the given deque.

        Args:
            other: The deque to copy.
        """
        self = Self(
            capacity=other._capacity,
            min_capacity=other._min_capacity,
            maxlen=other._maxlen,
            shrink=other._shrink,
        )
        for i in range(len(other)):
            offset = other._physical_index(other._head + i)
            (self._data + i).init_pointee_copy((other._data + offset)[])

        self._tail = len(other)

    fn __moveinit__(inout self, owned existing: Self):
        """Moves data of an existing deque into a new one.

        Args:
            existing: The existing deque.
        """
        self._data = existing._data
        self._capacity = existing._capacity
        self._head = existing._head
        self._tail = existing._tail
        self._min_capacity = existing._min_capacity
        self._maxlen = existing._maxlen
        self._shrink = existing._shrink

    fn __del__(owned self):
        """Destroys all elements in the deque and free its memory."""
        for i in range(len(self)):
            offset = self._physical_index(self._head + i)
            (self._data + offset).destroy_pointee()
        self._data.free()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks whether the deque has any elements or not.

        Returns:
            `False` if the deque is empty, `True` if there is at least one element.
        """
        return self._head != self._tail

    @always_inline
    fn __len__(self) -> Int:
        """Gets the number of elements in the deque.

        Returns:
            The number of elements in the deque.
        """
        return (self._tail - self._head) & (self._capacity - 1)

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

        offset = self._physical_index(self._head + normalized_idx)
        return (self._data + offset)[]

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn append(inout self, owned value: ElementType):
        """Appends a value to the right side of the deque.

        Args:
            value: The value to append.
        """
        if len(self) == self._maxlen:
            (self._data + self._head).destroy_pointee()
            self._head = self._physical_index(self._head + 1)

        (self._data + self._tail).init_pointee_move(value^)
        self._tail = self._physical_index(self._tail + 1)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    fn extend(inout self, owned values: List[ElementType]):
        """Extends the right side of the deque by consuming elements of the list argument.

        Args:
            values: List whose elements will be added at the right side of the deque.
        """
        len_self = len(self)
        len_values = len(values)
        len_total = len_self + len_values

        new_capacity = self._capacity
        if self._capacity <= len_total:
            new_capacity = bit_ceil(len_total)
            if is_power_of_two(len_total):
                new_capacity <<= 1

        max_capacity = new_capacity
        if self._maxlen > 0:
            max_capacity = bit_ceil(self._maxlen)
            if is_power_of_two(self._maxlen):
                max_capacity <<= 1

        new_capacity = min(new_capacity, max_capacity)

        # number of elements to move into the final deque
        # first from `values` and then from `self`
        n_move_total = len_total
        if self._maxlen > 0:
            n_move_total = min(len_total, self._maxlen)
        n_move_values = min(len_values, n_move_total)
        n_move_self = n_move_total - n_move_values

        # number of elements that do not fit into `maxlen`
        # and therefore have to be popped and destroyed
        n_pop_self = len_self - n_move_self
        n_pop_values = len_values - n_move_values

        # pop excess `self` elements
        for _ in range(n_pop_self):
            (self._data + self._head).destroy_pointee()
            self._head = self._physical_index(self._head + 1)

        # move from `self` to new location if we have to re-allocate
        if new_capacity > self._capacity:
            new_data = UnsafePointer[ElementType].alloc(new_capacity)
            for i in range(n_move_self):
                offset = self._physical_index(self._head + i)
                (self._data + offset).move_pointee_into(new_data + i)
            if self._data:
                self._data.free()
            self._data = new_data
            self._capacity = new_capacity
            self._head = 0
            self._tail = n_move_self

        # we will consume all elements of `values`
        values.size = 0
        # pop excess elements from `values`
        for i in range(n_pop_values):
            (values.data + i).destroy_pointee()

        # move remaining elements from `values`
        src = values.data.bitcast[origin = __origin_of(values)]() + n_pop_values
        for i in range(n_move_values):
            (src + i).move_pointee_into(self._data + self._tail)
            self._tail = self._physical_index(self._tail + 1)

    @doc_private
    @always_inline
    fn _physical_index(self, logical_index: Int) -> Int:
        """Calculates the physical index in the circular buffer.

        Args:
            logical_index: The logical index, which may fall outside the physical bounds
                of the buffer and needs to be wrapped around.

        The size of the underlying buffer is always a power of two, allowing the use of
        the more efficient bitwise `&` operation instead of the modulo `%` operator.
        """
        return logical_index & (self._capacity - 1)

    @doc_private
    fn _realloc(inout self, new_capacity: Int):
        """Relocates data to a new storage buffer.

        Args:
            new_capacity: The new capacity of the buffer.
        """
        deque_len = len(self) if self else self._capacity

        tail_len = self._tail
        head_len = self._capacity - self._head

        if head_len > deque_len:
            head_len = deque_len
            tail_len = 0

        new_data = UnsafePointer[ElementType].alloc(new_capacity)

        src = self._data + self._head
        dsc = new_data
        for i in range(head_len):
            (src + i).move_pointee_into(dsc + i)

        src = self._data
        dsc = new_data + head_len
        for i in range(tail_len):
            (src + i).move_pointee_into(dsc + i)

        self._head = 0
        self._tail = deque_len

        if self._data:
            self._data.free()
        self._data = new_data
        self._capacity = new_capacity
