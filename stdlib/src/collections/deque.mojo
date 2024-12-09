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

from collections import Optional

from bit import bit_ceil
from memory import UnsafePointer

# ===-----------------------------------------------------------------------===#
# Deque
# ===-----------------------------------------------------------------------===#


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
        out self,
        *,
        owned elements: Optional[List[ElementType]] = None,
        capacity: Int = Self.default_capacity,
        min_capacity: Int = Self.default_capacity,
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
            if max_deque_capacity == maxlen:
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

    @implicit
    fn __init__(out self, owned *values: ElementType):
        """Constructs a deque from the given values.

        Args:
            values: The values to populate the deque with.
        """
        self = Self(elements=values^)

    fn __init__(mut self, *, owned elements: VariadicListMem[ElementType, _]):
        """Constructs a deque from the given values.

        Args:
             elements: The values to populate the deque with.
        """
        args_length = len(elements)

        if args_length < self.default_capacity:
            capacity = self.default_capacity
        else:
            capacity = args_length

        self = Self(capacity=capacity)

        for i in range(args_length):
            src = UnsafePointer.address_of(elements[i])
            dst = self._data + i
            src.move_pointee_into(dst)

        # Do not destroy the elements when their backing storage goes away.
        __mlir_op.`lit.ownership.mark_destroyed`(
            __get_mvalue_as_litref(elements)
        )

        self._tail = args_length

    @implicit
    fn __init__(out self, other: Self):
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

    fn __moveinit__(out self, owned existing: Self):
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
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __add__(self, other: Self) -> Self:
        """Concatenates self with other and returns the result as a new deque.

        Args:
            other: Deque whose elements will be appended to the elements of self.

        Returns:
            The newly created deque with the properties of `self`.
        """
        new = Self(other=self)
        for element in other:
            new.append(element[])
        return new^

    fn __iadd__(mut self, other: Self):
        """Appends the elements of other deque into self.

        Args:
            other: Deque whose elements will be appended to self.
        """
        for element in other:
            self.append(element[])

    fn __mul__(self, n: Int) -> Self:
        """Concatenates `n` deques of `self` and returns a new deque.

        Args:
            n: The multiplier number.

        Returns:
            The new deque.
        """
        if n <= 0:
            return Self(
                capacity=self._min_capacity,
                min_capacity=self._min_capacity,
                maxlen=self._maxlen,
                shrink=self._shrink,
            )
        new = Self(other=self)
        for _ in range(n - 1):
            for element in self:
                new.append(element[])
        return new^

    fn __imul__(mut self, n: Int):
        """Concatenates self `n` times in place.

        Args:
            n: The multiplier number.
        """
        if n <= 0:
            self.clear()
            return

        orig = Self(other=self)
        for _ in range(n - 1):
            for element in orig:
                self.append(element[])

    fn __eq__[
        EqualityElementType: EqualityComparableCollectionElement, //
    ](
        self: Deque[EqualityElementType], other: Deque[EqualityElementType]
    ) -> Bool:
        """Checks if two deques are equal.

        Parameters:
            EqualityElementType: The type of the elements in the deque.
                Must implement the trait `EqualityComparableCollectionElement`.

        Args:
            other: The deque to compare with.

        Returns:
            `True` if the deques are equal, `False` otherwise.
        """
        if len(self) != len(other):
            return False

        for i in range(len(self)):
            offset_self = self._physical_index(self._head + i)
            offset_other = other._physical_index(other._head + i)
            if (self._data + offset_self)[] != (other._data + offset_other)[]:
                return False
        return True

    fn __ne__[
        EqualityElementType: EqualityComparableCollectionElement, //
    ](
        self: Deque[EqualityElementType], other: Deque[EqualityElementType]
    ) -> Bool:
        """Checks if two deques are not equal.

        Parameters:
            EqualityElementType: The type of the elements in the deque.
                Must implement the trait `EqualityComparableCollectionElement`.

        Args:
            other: The deque to compare with.

        Returns:
            `True` if the deques are not equal, `False` otherwise.
        """
        return not (self == other)

    fn __contains__[
        EqualityElementType: EqualityComparableCollectionElement, //
    ](self: Deque[EqualityElementType], value: EqualityElementType) -> Bool:
        """Verify if a given value is present in the deque.

        Parameters:
            EqualityElementType: The type of the elements in the deque.
                Must implement the trait `EqualityComparableCollectionElement`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the deque, False otherwise.
        """
        for i in range(len(self)):
            offset = self._physical_index(self._head + i)
            if (self._data + offset)[] == value:
                return True
        return False

    fn __iter__(
        ref self,
    ) -> _DequeIter[ElementType, __origin_of(self)]:
        """Iterates over elements of the deque, returning the references.

        Returns:
            An iterator of the references to the deque elements.
        """
        return _DequeIter(0, Pointer.address_of(self))

    fn __reversed__(
        ref self,
    ) -> _DequeIter[ElementType, __origin_of(self), False]:
        """Iterate backwards over the deque, returning the references.

        Returns:
            A reversed iterator of the references to the deque elements.
        """
        return _DequeIter[forward=False](len(self), Pointer.address_of(self))

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

    fn __getitem__(ref self, idx: Int) -> ref [self] ElementType:
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

    @no_inline
    fn write_to[
        RepresentableElementType: RepresentableCollectionElement,
        WriterType: Writer, //,
    ](self: Deque[RepresentableElementType], mut writer: WriterType):
        """Writes `my_deque.__str__()` to a `Writer`.

        Parameters:
            RepresentableElementType: The type of the Deque elements.
                Must implement the trait `RepresentableCollectionElement`.
            WriterType: A type conforming to the Writable trait.

        Args:
            writer: The object to write to.
        """
        writer.write("Deque(")
        for i in range(len(self)):
            offset = self._physical_index(self._head + i)
            writer.write(repr((self._data + offset)[]))
            if i < len(self) - 1:
                writer.write(", ")
        writer.write(")")

    @no_inline
    fn __str__[
        RepresentableElementType: RepresentableCollectionElement, //
    ](self: Deque[RepresentableElementType]) -> String:
        """Returns a string representation of a `Deque`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        my_deque = Deque[Int](1, 2, 3)
        print(my_deque.__str__())
        ```

        When the compiler supports conditional methods, then a simple `str(my_deque)` will
        be enough.

        The elements' type must implement the `__repr__()` method for this to work.

        Parameters:
            RepresentableElementType: The type of the elements in the deque.
                Must implement the trait `RepresentableCollectionElement`.

        Returns:
            A string representation of the deque.
        """
        output = String()
        self.write_to(output)
        return output^

    @no_inline
    fn __repr__[
        RepresentableElementType: RepresentableCollectionElement, //
    ](self: Deque[RepresentableElementType]) -> String:
        """Returns a string representation of a `Deque`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        my_deque = Deque[Int](1, 2, 3)
        print(my_deque.__repr__())
        ```

        When the compiler supports conditional methods, then a simple `repr(my_deque)` will
        be enough.

        The elements' type must implement the `__repr__()` for this to work.

        Parameters:
            RepresentableElementType: The type of the elements in the deque.
                Must implement the trait `RepresentableCollectionElement`.

        Returns:
            A string representation of the deque.
        """
        return self.__str__()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn append(mut self, owned value: ElementType):
        """Appends a value to the right side of the deque.

        Args:
            value: The value to append.
        """
        # checking for positive _maxlen first is important for speed
        if self._maxlen > 0 and len(self) == self._maxlen:
            (self._data + self._head).destroy_pointee()
            self._head = self._physical_index(self._head + 1)

        (self._data + self._tail).init_pointee_move(value^)
        self._tail = self._physical_index(self._tail + 1)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    fn appendleft(mut self, owned value: ElementType):
        """Appends a value to the left side of the deque.

        Args:
            value: The value to append.
        """
        # checking for positive _maxlen first is important for speed
        if self._maxlen > 0 and len(self) == self._maxlen:
            self._tail = self._physical_index(self._tail - 1)
            (self._data + self._tail).destroy_pointee()

        self._head = self._physical_index(self._head - 1)
        (self._data + self._head).init_pointee_move(value^)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    fn clear(mut self):
        """Removes all elements from the deque leaving it with length 0.

        Resets the underlying storage capacity to `_min_capacity`.
        """
        for i in range(len(self)):
            offset = self._physical_index(self._head + i)
            (self._data + offset).destroy_pointee()
        self._data.free()
        self._capacity = self._min_capacity
        self._data = UnsafePointer[ElementType].alloc(self._capacity)
        self._head = 0
        self._tail = 0

    fn count[
        EqualityElementType: EqualityComparableCollectionElement, //
    ](self: Deque[EqualityElementType], value: EqualityElementType) -> Int:
        """Counts the number of occurrences of a `value` in the deque.

        Parameters:
            EqualityElementType: The type of the elements in the deque.
                Must implement the trait `EqualityComparableCollectionElement`.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the deque.
        """
        count = 0
        for i in range(len(self)):
            offset = self._physical_index(self._head + i)
            if (self._data + offset)[] == value:
                count += 1
        return count

    fn extend(mut self, owned values: List[ElementType]):
        """Extends the right side of the deque by consuming elements of the list argument.

        Args:
            values: List whose elements will be added at the right side of the deque.
        """
        n_move_total, n_move_self, n_move_values, n_pop_self, n_pop_values = (
            self._compute_pop_and_move_counts(len(self), len(values))
        )

        # pop excess `self` elements
        for _ in range(n_pop_self):
            (self._data + self._head).destroy_pointee()
            self._head = self._physical_index(self._head + 1)

        # move from `self` to new location if we have to re-allocate
        if n_move_total >= self._capacity:
            self._prepare_for_new_elements(n_move_total, n_move_self)

        # we will consume all elements of `values`
        values_data = values.steal_data()

        # pop excess elements from `values`
        for i in range(n_pop_values):
            (values_data + i).destroy_pointee()

        # move remaining elements from `values`
        src = values_data + n_pop_values
        for i in range(n_move_values):
            (src + i).move_pointee_into(self._data + self._tail)
            self._tail = self._physical_index(self._tail + 1)

    fn extendleft(mut self, owned values: List[ElementType]):
        """Extends the left side of the deque by consuming elements from the list argument.

        Acts as series of left appends resulting in reversed order of elements in the list argument.

        Args:
            values: List whose elements will be added at the left side of the deque.
        """
        n_move_total, n_move_self, n_move_values, n_pop_self, n_pop_values = (
            self._compute_pop_and_move_counts(len(self), len(values))
        )

        # pop excess `self` elements
        for _ in range(n_pop_self):
            self._tail = self._physical_index(self._tail - 1)
            (self._data + self._tail).destroy_pointee()

        # move from `self` to new location if we have to re-allocate
        if n_move_total >= self._capacity:
            self._prepare_for_new_elements(n_move_total, n_move_self)

        # we will consume all elements of `values`
        values_data = values.steal_data()

        # pop excess elements from `values`
        for i in range(n_pop_values):
            (values_data + i).destroy_pointee()

        # move remaining elements from `values`
        src = values_data + n_pop_values
        for i in range(n_move_values):
            self._head = self._physical_index(self._head - 1)
            (src + i).move_pointee_into(self._data + self._head)

    fn index[
        EqualityElementType: EqualityComparableCollectionElement, //
    ](
        self: Deque[EqualityElementType],
        value: EqualityElementType,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) raises -> Int:
        """Returns the index of the first occurrence of a `value` in a deque
        restricted by the range given the `start` and `stop` bounds.

        Parameters:
            EqualityElementType: The type of the elements in the deque.
                Must implement the `EqualityComparableCollectionElement` trait.

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to None, which means the end of the deque).

        Returns:
            The index of the first occurrence of the value in the deque.

        Raises:
            ValueError: If the value is not found in the deque.
        """
        start_normalized = start

        if stop is None:
            stop_normalized = len(self)
        else:
            stop_normalized = stop.value()

        if start_normalized < 0:
            start_normalized += len(self)
        if stop_normalized < 0:
            stop_normalized += len(self)

        start_normalized = max(0, min(start_normalized, len(self)))
        stop_normalized = max(0, min(stop_normalized, len(self)))

        for idx in range(start_normalized, stop_normalized):
            offset = self._physical_index(self._head + idx)
            if (self._data + offset)[] == value:
                return idx
        raise "ValueError: Given element is not in deque"

    fn insert(mut self, idx: Int, owned value: ElementType) raises:
        """Inserts the `value` into the deque at position `idx`.

        Args:
            idx: The position to insert the value into.
            value: The value to insert.

        Raises:
            IndexError: If deque is already at its maximum size.
        """
        deque_len = len(self)

        if deque_len == self._maxlen:
            raise "IndexError: Deque is already at its maximum size"

        normalized_idx = idx

        if normalized_idx < -deque_len:
            normalized_idx = 0

        if normalized_idx > deque_len:
            normalized_idx = deque_len

        if normalized_idx < 0:
            normalized_idx += deque_len

        if normalized_idx <= deque_len // 2:
            for i in range(normalized_idx):
                src = self._physical_index(self._head + i)
                dst = self._physical_index(src - 1)
                (self._data + src).move_pointee_into(self._data + dst)
            self._head = self._physical_index(self._head - 1)
        else:
            for i in range(deque_len - normalized_idx):
                dst = self._physical_index(self._tail - i)
                src = self._physical_index(dst - 1)
                (self._data + src).move_pointee_into(self._data + dst)
            self._tail = self._physical_index(self._tail + 1)

        offset = self._physical_index(self._head + normalized_idx)
        (self._data + offset).init_pointee_move(value^)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    fn remove[
        EqualityElementType: EqualityComparableCollectionElement, //
    ](mut self: Deque[EqualityElementType], value: EqualityElementType,) raises:
        """Removes the first occurrence of the `value`.

        Parameters:
            EqualityElementType: The type of the elements in the deque.
                Must implement the `EqualityComparableCollectionElement` trait.

        Args:
            value: The value to remove.

        Raises:
            ValueError: If the value is not found in the deque.
        """
        deque_len = len(self)
        for idx in range(deque_len):
            offset = self._physical_index(self._head + idx)
            if (self._data + offset)[] == value:
                (self._data + offset).destroy_pointee()

                if idx < deque_len // 2:
                    for i in reversed(range(idx)):
                        src = self._physical_index(self._head + i)
                        dst = self._physical_index(src + 1)
                        (self._data + src).move_pointee_into(self._data + dst)
                    self._head = self._physical_index(self._head + 1)
                else:
                    for i in range(idx + 1, deque_len):
                        src = self._physical_index(self._head + i)
                        dst = self._physical_index(src - 1)
                        (self._data + src).move_pointee_into(self._data + dst)
                    self._tail = self._physical_index(self._tail - 1)

                if (
                    self._shrink
                    and self._capacity > self._min_capacity
                    and self._capacity // 4 >= len(self)
                ):
                    self._realloc(self._capacity >> 1)

                return

        raise "ValueError: Given element is not in deque"

    fn peek(self) raises -> ElementType:
        """Inspect the last (rightmost) element of the deque without removing it.

        Returns:
            The the last (rightmost) element of the deque.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        return (self._data + self._physical_index(self._tail - 1))[]

    fn peekleft(self) raises -> ElementType:
        """Inspect the first (leftmost) element of the deque without removing it.

        Returns:
            The the first (leftmost) element of the deque.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        return (self._data + self._head)[]

    fn pop(mut self, out element: ElementType) raises:
        """Removes and returns the element from the right side of the deque.

        Returns:
            The popped value.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        self._tail = self._physical_index(self._tail - 1)
        element = (self._data + self._tail).take_pointee()

        if (
            self._shrink
            and self._capacity > self._min_capacity
            and self._capacity // 4 >= len(self)
        ):
            self._realloc(self._capacity >> 1)

        return

    fn popleft(mut self, out element: ElementType) raises:
        """Removes and returns the element from the left side of the deque.

        Returns:
            The popped value.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        element = (self._data + self._head).take_pointee()
        self._head = self._physical_index(self._head + 1)

        if (
            self._shrink
            and self._capacity > self._min_capacity
            and self._capacity // 4 >= len(self)
        ):
            self._realloc(self._capacity >> 1)

        return

    fn reverse(mut self):
        """Reverses the elements of the deque in-place."""
        last = self._head + len(self) - 1
        for i in range(len(self) // 2):
            src = self._physical_index(self._head + i)
            dst = self._physical_index(last - i)
            tmp = (self._data + dst).take_pointee()
            (self._data + src).move_pointee_into(self._data + dst)
            (self._data + src).init_pointee_move(tmp^)

    fn rotate(mut self, n: Int = 1):
        """Rotates the deque by `n` steps.

        If `n` is positive, rotates to the right.
        If `n` is negative, rotates to the left.

        Args:
            n: Number of steps to rotate the deque
                (defaults to 1).
        """
        if n < 0:
            for _ in range(-n):
                (self._data + self._head).move_pointee_into(
                    self._data + self._tail
                )
                self._tail = self._physical_index(self._tail + 1)
                self._head = self._physical_index(self._head + 1)
        else:
            for _ in range(n):
                self._tail = self._physical_index(self._tail - 1)
                self._head = self._physical_index(self._head - 1)
                (self._data + self._tail).move_pointee_into(
                    self._data + self._head
                )

    fn _compute_pop_and_move_counts(
        self, len_self: Int, len_values: Int
    ) -> (Int, Int, Int, Int, Int):
        """
        Calculates the number of elements to retain, move or discard in the deque and
        in the list of the new values based on the current length of the deque,
        the length of new values to add, and the maximum length constraint `_maxlen`.

        Args:
            len_self: The current number of elements in the deque.
            len_values: The number of new elements to add to the deque.

        Returns:
            A tuple: (n_move_total, n_move_self, n_move_values, n_pop_self, n_pop_values)
                n_move_total: Total final number of elements in the deque.
                n_move_self: Number of existing elements to retain in the deque.
                n_move_values: Number of new elements to add from `values`.
                n_pop_self: Number of existing elements to remove from the deque.
                n_pop_values: Number of new elements that don't fit and will be discarded.
        """
        len_total = len_self + len_values

        n_move_total = (
            min(len_total, self._maxlen) if self._maxlen > 0 else len_total
        )
        n_move_values = min(len_values, n_move_total)
        n_move_self = n_move_total - n_move_values

        n_pop_self = len_self - n_move_self
        n_pop_values = len_values - n_move_values

        return (
            n_move_total,
            n_move_self,
            n_move_values,
            n_pop_self,
            n_pop_values,
        )

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

    fn _prepare_for_new_elements(mut self, n_total: Int, n_retain: Int):
        """Prepares the deque’s internal buffer for adding new elements by
        reallocating memory and retaining the specified number of existing elements.

        Args:
            n_total: The total number of elements the new buffer should support.
            n_retain: The number of existing elements to keep in the deque.
        """
        new_capacity = bit_ceil(n_total)
        if new_capacity == n_total:
            new_capacity <<= 1

        new_data = UnsafePointer[ElementType].alloc(new_capacity)

        for i in range(n_retain):
            offset = self._physical_index(self._head + i)
            (self._data + offset).move_pointee_into(new_data + i)

        if self._data:
            self._data.free()

        self._data = new_data
        self._capacity = new_capacity
        self._head = 0
        self._tail = n_retain

    fn _realloc(mut self, new_capacity: Int):
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


@value
struct _DequeIter[
    deque_mutability: Bool, //,
    ElementType: CollectionElement,
    deque_lifetime: Origin[deque_mutability],
    forward: Bool = True,
]:
    """Iterator for Deque.

    Parameters:
        deque_mutability: Whether the reference to the deque is mutable.
        ElementType: The type of the elements in the deque.
        deque_lifetime: The lifetime of the Deque.
        forward: The iteration direction. `False` is backwards.
    """

    alias deque_type = Deque[ElementType]

    var index: Int
    var src: Pointer[Self.deque_type, deque_lifetime]

    fn __iter__(self) -> Self:
        return self

    fn __next__(mut self) -> Pointer[ElementType, deque_lifetime]:
        @parameter
        if forward:
            self.index += 1
            return Pointer.address_of(self.src[][self.index - 1])
        else:
            self.index -= 1
            return Pointer.address_of(self.src[][self.index])

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0
