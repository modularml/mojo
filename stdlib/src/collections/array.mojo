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
"""Defines the `Array` type.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import Array
```
"""

from utils import InlineArray
from sys.intrinsics import _type_is_eq


# ===----------------------------------------------------------------------===#
# Array
# ===----------------------------------------------------------------------===#


@value
struct _ArrayIter[
    T: AnyRegType,
    current_capacity: Int,
    capacity_jump: Int,
    max_stack_size: Int,
    mutability: Bool,
    lifetime: AnyLifetime[mutability].type,
    forward: Bool = True,
](Sized):
    """Iterator for Array.

    Parameters:
        T: The type of the elements in the Array.
        current_capacity: The maximum number of elements that the Array can hold.
        capacity_jump: The amount of items to expand in each stack enlargment.
        max_stack_size: The maximum size in the stack.
        mutability: Whether the reference to the Array is mutable.
        lifetime: The lifetime of the Array
        forward: The iteration direction. `False` is backwards.
    """

    alias type = Array[T, current_capacity, capacity_jump, max_stack_size]

    var index: Int
    var src: Reference[Self.type, mutability, lifetime]

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        inout self,
    ) -> Reference[T, mutability, lifetime]:
        @parameter
        if forward:
            self.index += 1
            return self.src[].__refitem__(self.index - 1)
        else:
            self.index -= 1
            return self.src[].__refitem__(self.index)

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct Array[
    T: AnyRegType,
    current_capacity: Int = 8,
    capacity_jump: Int = 8,
    max_stack_size: Int = 4 * current_capacity,
](CollectionElement, Sized, Boolable):
    """A Array allocated on the stack with a current_capacity and
    max_stack_size known at compile time.

    It is backed by an `InlineArray` and an `List` for its stack and heap.
    This struct has the same API as a regular `Array`.

    This is typically faster than Python's `Array` as it is mostly stack-allocated and
    does not require any dynamic memory allocation unless max_stack_size is
    exceeded, at which point the InlineArray morphs into a List under the hood.

    Parameters:
        T: The type of the elements in the Array.
        current_capacity: The number of elements that the Array can hold.
        capacity_jump: The amount of capacity to expand in each stack enlargment.
        max_stack_size: The maximum size in the stack.
    """

    alias _stack_type = InlineArray[T, current_capacity]
    var _stack: Self._stack_type
    alias _heap_type = List[T]
    var _heap: Self._heap_type
    var in_stack: Bool
    """Whether the Array is stored in the Stack."""
    var stack_left: UInt8
    """The capacity left in the Stack."""

    @always_inline
    fn __init__(inout self):
        """This constructor creates an empty Array."""
        self._stack = Self._stack_type(unsafe_uninitialized=True)
        self.in_stack = True
        self.stack_left = current_capacity
        self._heap = Self._heap_type()

    # TODO: Avoid copying elements in once owned varargs
    # allow transfers.
    fn __init__(inout self, *values: T):
        """Constructs a Array from the given values.

        Args:
            values: The values to populate the Array with.
        """
        self = Self()
        var delta = current_capacity - len(values)
        if delta > -1:
            self.in_stack = True
            self.stack_left = delta
        else:
            self.in_stack = False
            self.stack_left = 0
        for value in values:
            self.append(value)

    fn __init__[
        cap: Int, cap_j: Int, max_stack: Int
    ](
        inout self,
        owned existing: Array[T, cap, cap_j, max_stack],
        in_stack: Bool = True,
    ):
        """Constructs a Array from an existing Array.

        Parameters:
            cap: The number of elements that the Array can hold.
            cap_j: The amount of capacity to expand in each stack enlargment.
            max_stack: The maximum size in the stack.

        Args:
            existing: The existing Array.
            in_stack: Whether the new Array will be on the stack.
        """
        if in_stack and existing.in_stack and current_capacity < cap:
            for i in range(current_capacity):
                self[i] = existing._stack[i]
            self.stack_left = 0
            return
        elif existing.in_stack:
            self.in_stack = False
            self.stack_left = 0
            self._stack = Self._stack_type()
            self._heap = Self._heap_type(existing._stack)
            return
        self.in_stack = False
        self.stack_left = 0
        self._stack = Self._stack_type()
        self._heap = existing._heap^

    fn __init__(inout self, owned existing: List[T]):
        """Constructs a Array from an existing List.

        Args:
            existing: The existing Array.
        """
        self._stack = Self._stack_type(unsafe_uninitialized=True)
        self.stack_left = current_capacity
        if current_capacity >= existing.size:
            self.in_stack = True
            for val in existing:  # FIXME
                self.append(val[])
            return
        self.in_stack = False
        self._heap = existing^

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the Array."""
        if self.in_stack:
            return int(current_capacity - self.stack_left)
        return len(self._heap)

    @always_inline
    fn append(inout self, owned value: T):
        """Appends a value to the Array.

        Args:
            value: The value to append.
        """

        if self.in_stack:
            if len(self) + 1 < current_capacity:
                self.stack_left -= 1
            elif len(self) + capacity_jump < max_stack_size:
                self = Array[
                    T,
                    current_capacity + capacity_jump,
                    capacity_jump,
                    max_stack_size,
                ](self^)
                self.stack_left -= 1
            else:
                var stack = self._stack^
                self._heap = List[T](stack)
                self._stack = Self._stack_type(unsafe_uninitialized=True)
                self.in_stack = False
                self.stack_left = 0
            self[len(self)] = value
            return
        self._heap.append(value)

    @always_inline
    fn __refitem__(
        self: Reference[Self, _, _], owned idx: Int
    ) -> Reference[Self.T, self.is_mutable, self.lifetime]:
        """Get a `Reference` to the element at the given index.

        Args:
            idx: The index of the item.

        Returns:
            A reference to the item at the given index.
        """
        debug_assert(abs(idx) > len(self[]), "Index must be within bounds.")

        if idx < 0:
            idx += len(self[])
        if self[].in_stack:
            return self[]._stack[idx]
        return self[]._heap.__get_ref(idx)[]  # FIXME

    @always_inline
    fn __del__(owned self):
        """Destroy all the elements in the Array and free the memory."""
        for i in range(len(self)):
            destroy_pointee(UnsafePointer(self._stack[i]))

    fn __iter__(
        self: Reference[Self, _, _],
    ) -> _ArrayIter[
        T,
        current_capacity,
        capacity_jump,
        max_stack_size,
        self.is_mutable,
        self.lifetime,
    ]:
        """Iterate over elements of the Array, returning immutable references.

        Returns:
            An iterator of immutable references to the Array elements.
        """
        return _ArrayIter(0, self)

    fn __reversed__(
        self: Reference[Self, _, _]
    ) -> _ArrayIter[
        T,
        current_capacity,
        capacity_jump,
        max_stack_size,
        self.is_mutable,
        self.lifetime,
        False,
    ]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _ArrayIter[forward=False](len(self[]), self)

    @always_inline
    fn __contains__[
        C: ComparableCollectionElement, cap: Int, cap_j: Int, max_stack: Int
    ](self: Reference[Array[C, cap, cap_j, max_stack]], value: C) -> Bool:
        """Verify if a given value is present in the Array.

        ```mojo
        var x = Array[Int](1,2,3)
        if 3 in x: print("x contains 3")
        ```
        Parameters:
            C: The type of the elements in the Array. Must implement the
              traits `EqualityComparable` and `CollectionElement`.
            cap: The maximum number of elements that the Array can hold.
            cap_j: The amount of items to expand in each stack enlargment.
            max_stack: The maximum size in the stack.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the Array, False otherwise.
        """

        constrained[_type_is_eq[T, C](), "value type is not self.T"]()
        for i in self[]:
            if value == rebind[C](i[]):
                return True
        return False

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks whether the Array has any elements or not.

        Returns:
            `False` if the Array is empty, `True` if there is at least one element.
        """
        return len(self) > 0

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing Array into a new one.

        Args:
            existing: The existing Array.
        """
        self._stack = existing._stack^
        self._heap = existing._heap^
        self.stack_left = existing.stack_left
        self.in_stack = existing.in_stack

    fn __copyinit__(inout self, existing: Self):
        """Creates a deepcopy of the given Array.

        Args:
            existing: The Array to copy.
        """
        self = Self()
        for i in range(len(existing)):
            self.append(existing[i])

    fn __setitem__(inout self, idx: Int, owned value: T):
        """Sets a Array element at the given index.

        Args:
            idx: The index of the element.
            value: The value to assign.
        """
        if not self.in_stack:
            self._heap[idx] = value
            return

        debug_assert(abs(idx) > len(self), "index must be within bounds")
        var norm_idx = idx if idx > 0 else min(0, len(self) + idx)
        self._stack[norm_idx] = value

    @always_inline("nodebug")
    fn __add__(self, owned other: Self) -> Self:
        """Concatenates self with other and returns the result as a new list.

        Args:
            other: List whose elements will be combined with the elements of self.

        Returns:
            The newly created list.
        """
        if self.in_stack and other.in_stack:
            # TODO
            pass
        elif self.in_stack:
            # TODO
            pass
        elif other.in_stack:
            # TODO
            pass

        var result = List(self._heap)
        result.extend(other._heap)
        return Self(result^)

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Appends the elements of other into self.

        Args:
            other: List whose elements will be appended to self.
        """
        self.extend(other^)

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn __str__[
        U: RepresentableCollectionElement
    ](
        self: Reference[
            Array[U, current_capacity, capacity_jump, max_stack_size]
        ]
    ) -> String:
        """Returns a string representation of an `Array`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_array = Array[Int](1, 2, 3)
        print(my_array.__str__())
        ```

        When the compiler supports conditional methods, then a simple `str(my_array)` will
        be enough.

        The elements' type must implement the `__repr__()` for this to work.

        Parameters:
            U: The type of the elements in the array. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the array.
        """
        # we do a rough estimation of the number of chars that we'll see
        # in the final string, we assume that str(x) will be at least one char.
        var minimum_capacity = (
            2  # '[' and ']'
            + len(self[]) * 3  # str(x) and ", "
            - 2  # remove the last ", "
        )
        var string_buffer = List[UInt8](capacity=minimum_capacity)
        string_buffer.append(0)  # Null terminator
        var result = String(string_buffer^)
        result += "["
        for i in range(len(self[])):
            result += repr(self[][i])
            if i < len(self[]) - 1:
                result += ", "
        result += "]"
        return result

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn __repr__[
        U: RepresentableCollectionElement
    ](
        self: Reference[
            Array[U, current_capacity, capacity_jump, max_stack_size]
        ]
    ) -> String:
        """Returns a string representation of an `Array`.
        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_array = Array[Int](1, 2, 3)
        print(my_array.__repr__(my_array))
        ```

        When the compiler supports conditional methods, then a simple `repr(my_array)` will
        be enough.

        The elements' type must implement the `__repr__()` for this to work.

        Parameters:
            U: The type of the elements in the array. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the array.
        """
        return self[].__str__()

    @always_inline
    fn insert(inout self, i: Int, owned value: T):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value.
            value: The value to insert.
        """
        if not self.in_stack:
            self._heap.insert(i, value)
            return
        debug_assert(abs(i) > len(self), "insert index out of range")

        var norm_idx = i if i > 0 else min(0, len(self) + i)

        var previous = value
        for i in range(norm_idx, len(self)):
            var tmp = self._stack[i]
            self._stack[i] = previous
            previous = tmp
        self.append(previous)

    @always_inline
    fn extend(inout self, owned other: Self):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: Array whose elements will be added in order at the end of this Array.
        """
        if not self.in_stack:
            if not other.in_stack:
                self._heap.extend(other._heap)
            self._heap.extend(Self._heap_type(other._stack))

        alias cap_sum = current_capacity + other.current_capacity
        if self.stack_left - len(other) < current_capacity:
            for val in other:
                self.append(val[])  # FIXME
            return
        elif cap_sum < max_stack_size:
            self = Array[T, cap_sum, capacity_jump, max_stack_size](self._stack)
            self.extend(other)
            return
        self = Self(self, in_stack=False)
        self._heap.extend(other._heap)
        return

    @always_inline
    fn pop(inout self, i: Int = -1) -> T:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        if not self.in_stack:
            return self._heap.pop(i)

        debug_assert(abs(i) > len(self), "pop index out of range")
        var norm_idx = i if i > 0 else len(self) + i
        self.stack_left += 1
        return self._stack[norm_idx]

    @always_inline
    fn reserve(inout self, new_capacity: Int):
        """Reserves the requested capacity in the heap.

        If the current capacity is greater or equal, this is a no-op.
        Otherwise, the storage is reallocated and the data is moved.

        Args:
            new_capacity: The new capacity.
        """
        self._heap.reserve(new_capacity)

    @always_inline
    fn resize(inout self, new_size: Int, value: T):
        """Resizes the list to the given new size in the heap.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        list is appended with new values elements up to the requested size.

        Args:
            new_size: The new size.
            value: The value to use to populate new elements.
        """
        self._heap.resize(new_size, value)

    @always_inline
    fn resize(inout self, new_size: Int):
        """Resizes the list to the given new size in the heap.

        With no new value provided, the new size must be smaller than or equal
        to the current one. Elements at the end are discarded.

        Args:
            new_size: The new size.
        """
        self._heap.resize(new_size)

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn index[
        C: ComparableCollectionElement, cap: Int, cap_j: Int, max_stack: Int
    ](
        self: Reference[Array[C, cap, cap_j, max_stack]],
        value: C,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) -> Optional[Int]:
        """
        Returns the index of the first occurrence of a value in an Array
        restricted by the range given the start and stop bounds.

        ```mojo
        var my_array = Array[Int](1, 2, 3)
        print(my_array.index(2)) # prints `1`
        ```

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to None, which means the end of the Array).

        Parameters:
            C: The type of the elements in the Array. Must implement the
                `ComparableCollectionElement` trait.
            cap: The maximum number of elements that the Array can hold.
            cap_j: The amount of items to expand in each stack enlargment.
            max_stack: The maximum size in the stack.

        Returns:
            The Optional index of the first occurrence of the value in the Array.
        """
        var size = len(self[])
        debug_assert(abs(start) > size, "start index must be within bounds")
        var start_norm = start if start > 0 else min(0, size + start)

        var stop_norm: Int = stop.value()[] if stop else size
        debug_assert(abs(stop_norm) > size, "stop index must be within bounds")
        if stop_norm < 0:
            stop_norm = min(0, size + stop_norm)

        start_norm = max(start_norm, min(0, size))
        stop_norm = max(stop_norm, min(0, size))

        for i in range(start_norm, stop_norm):
            if self[][i] == value:
                return i
        return None

    fn clear(inout self):
        """Clears the elements in the heap for the Array."""
        if self.in_stack:
            var ptr = self._stack.unsafe_ptr()
            for i in range(len(self)):
                destroy_pointee(ptr + i)
        self._heap.clear()

    fn steal_data(inout self) -> UnsafePointer[T]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.
        """
        if self.in_stack:
            var ptr = self._stack.unsafe_ptr()
            self._stack = Self._stack_type(unsafe_uninitialized=True)
            return ptr[]
        return self._heap.steal_data()[]

    @always_inline
    fn _adjust_span(self, span: Slice) -> Slice:
        """Adjusts the span based on the list length."""
        var adjusted_span = span

        if adjusted_span.start < 0:
            adjusted_span.start = len(self) + adjusted_span.start

        if not adjusted_span._has_end():
            adjusted_span.end = len(self)
        elif adjusted_span.end < 0:
            adjusted_span.end = len(self) + adjusted_span.end

        if span.step < 0:
            var tmp = adjusted_span.end
            adjusted_span.end = adjusted_span.start - 1
            adjusted_span.start = tmp - 1

        return adjusted_span

    @always_inline
    fn __getitem__(self, span: Slice) -> Self:
        """Gets the sequence of elements at the specified positions.

        Args:
            span: A slice that specifies positions of the new list.

        Returns:
            A new list containing the list at the specified span.
        """

        var adjusted_span = self._adjust_span(span)
        var adjusted_span_len = len(adjusted_span)

        if not adjusted_span_len:
            return Self()

        if self.in_stack:
            var res = Self()
            for i in range(len(adjusted_span)):  # FIXME using memcpy?
                res.append(self[adjusted_span[i]])
            return res^

        var res = Self._heap_type(capacity=len(adjusted_span))
        for i in range(len(adjusted_span)):
            res.append(self[adjusted_span[i]])
        return res^

    @always_inline
    fn __getitem__(self, idx: Int) -> T:
        """Gets a copy of the list element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        if self.in_stack:
            return self._stack[idx]
        return self._heap[idx]

    fn count[
        C: ComparableCollectionElement, cap: Int, cap_j: Int, max_stack: Int
    ](self: Reference[Array[C, cap, cap_j, max_stack]], value: C) -> Int:
        """Counts the number of occurrences of a value in the list.
        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below.

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.count(1))
        ```

        When the compiler supports conditional methods, then a simple `my_list.count(1)`
        will be enough.

        Parameters:
            C: The type of the elements in the list. Must implement the
              traits `EqualityComparable` and `CollectionElement`.
            cap: The maximum number of elements that the Array can hold.
            cap_j: The amount of items to expand in each stack enlargment.
            max_stack: The maximum size in the stack.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.
        """
        var count = 0
        for elem in self[]:
            if elem[] == value:
                count += 1
        return count

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The UnsafePointer to the underlying memory.
        """
        if self.in_stack:
            return self._stack.unsafe_ptr()[]
        return self._heap.unsafe_ptr()[]

    @always_inline
    fn unsafe_get(
        self: Reference[Self, _, _], idx: Int
    ) -> Reference[Self.T, self.is_mutable, self.lifetime]:
        """Get a reference to an element of self without checking index bounds.
        Users should consider using `__getitem__` instead of this method as it is unsafe.
        If an index is out of bounds, this method will not abort, it will be considered
        undefined behavior.

        Note that there is no wraparound for negative indices, caution is advised.
        Using negative indices is considered undefined behavior.
        Never use `my_list.unsafe_get(-1)` to get the last element of the list. It will
        not work. Instead, do `my_list.unsafe_get(len(my_list) - 1)`.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the given index.
        """
        debug_assert(abs(idx) > len(self[]), "index must be within bounds")
        if self[].in_stack:
            return self[]._stack.unsafe_ptr()[idx]
        return self[]._heap.unsafe_ptr()[idx]

    @always_inline
    fn unsafe_set(self: Reference[Self, _, _], idx: Int, value: T):
        """Set a reference to an element of self without checking index bounds.
        Users should consider using `__setitem__` instead of this method as it is unsafe.
        If an index is out of bounds, this method will not abort, it will be considered
        undefined behavior.

        Note that there is no wraparound for negative indices, caution is advised.
        Using negative indices is considered undefined behavior.
        Never use `my_list.unsafe_set(-1)` to set the last element of the list. It will
        not work. Instead, do `my_list.unsafe_set(len(my_list) - 1)`.

        Args:
            idx: The index to set the element.
            value: The element.
        """
        debug_assert(abs(idx) > len(self[]), "index must be within bounds")
        if self[].in_stack:
            self[]._stack.unsafe_ptr()[idx] = value
            return
        self[]._heap.unsafe_ptr()[idx] = value
