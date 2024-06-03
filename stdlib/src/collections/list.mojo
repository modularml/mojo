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
"""Defines the List type.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import List
```
"""


from memory import UnsafePointer, Reference
from sys.intrinsics import _type_is_eq
from .optional import Optional
from utils import Span

# ===----------------------------------------------------------------------===#
# List
# ===----------------------------------------------------------------------===#


@value
struct _ListIter[
    list_mutability: Bool, //,
    T: CollectionElement,
    list_lifetime: AnyLifetime[list_mutability].type,
    forward: Bool = True,
]:
    """Iterator for List.

    Parameters:
        list_mutability: Whether the reference to the list is mutable.
        T: The type of the elements in the list.
        list_lifetime: The lifetime of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias list_type = List[T]

    var index: Int
    var src: Reference[Self.list_type, list_lifetime]

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        inout self,
    ) -> Reference[T, list_lifetime]:
        @parameter
        if forward:
            self.index += 1
            return self.src[].__get_ref(self.index - 1)[]
        else:
            self.index -= 1
            return self.src[].__get_ref(self.index)[]

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct List[T: CollectionElement](CollectionElement, Sized, Boolable):
    """The `List` type is a dynamically-allocated list.

    It supports pushing and popping from the back resizing the underlying
    storage as needed.  When it is deallocated, it frees its memory.

    Parameters:
        T: The type of the elements.
    """

    # Fields
    var data: UnsafePointer[T]
    """The underlying storage for the list."""
    var size: Int
    """The number of elements in the list."""
    var capacity: Int
    """The amount of elements that can fit in the list without resizing it."""

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self):
        """Constructs an empty list."""
        self.data = UnsafePointer[T]()
        self.size = 0
        self.capacity = 0

    fn __init__(inout self, existing: Self):
        """Creates a deep copy of the given list.

        Args:
            existing: The list to copy.
        """
        self.__init__(capacity=existing.capacity)
        for e in existing:
            self.append(e[])

    fn __init__(inout self, *, capacity: Int):
        """Constructs a list with the given capacity.

        Args:
            capacity: The requested capacity of the list.
        """
        self.data = UnsafePointer[T].alloc(capacity)
        self.size = 0
        self.capacity = capacity

    # TODO: Avoid copying elements in once owned varargs
    # allow transfers.
    fn __init__(inout self, *values: T):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
        """
        self = Self(capacity=len(values))
        for value in values:
            self.append(value[])

    fn __init__(inout self, span: Span[T]):
        """Constructs a list from the a Span of values.

        Args:
            span: The span of values to populate the list with.
        """
        self = Self(capacity=len(span))
        for value in span:
            self.append(value[])

    fn __init__(
        inout self: Self,
        *,
        unsafe_pointer: UnsafePointer[T],
        size: Int,
        capacity: Int,
    ):
        """Constructs a list from a pointer, its size, and its capacity.

        Args:
            unsafe_pointer: The pointer to the data.
            size: The number of elements in the list.
            capacity: The capacity of the list.
        """
        self.data = unsafe_pointer
        self.size = size
        self.capacity = capacity

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing list into a new one.

        Args:
            existing: The existing list.
        """
        self.data = existing.data
        self.size = existing.size
        self.capacity = existing.capacity

    fn __copyinit__(inout self, existing: Self):
        """Creates a deepcopy of the given list.

        Args:
            existing: The list to copy.
        """
        self = Self(capacity=existing.capacity)
        for i in range(len(existing)):
            self.append(existing[i])

    @always_inline
    fn __del__(owned self):
        """Destroy all elements in the list and free its memory."""
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        if self.data:
            self.data.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __setitem__(inout self, idx: Int, owned value: T):
        """Sets a list element at the given index.

        Args:
            idx: The index of the element.
            value: The value to assign.
        """
        var normalized_idx = idx
        debug_assert(
            -self.size <= normalized_idx < self.size,
            "index must be within bounds",
        )

        if normalized_idx < 0:
            normalized_idx += len(self)

        (self.data + normalized_idx).destroy_pointee()
        (self.data + normalized_idx).init_pointee_move(value^)

    @always_inline
    fn __contains__[
        T2: ComparableCollectionElement
    ](self: List[T], value: T2) -> Bool:
        """Verify if a given value is present in the list.

        ```mojo
        var x = List[Int](1,2,3)
        if 3 in x: print("x contains 3")
        ```
        Parameters:
            T2: The type of the elements in the list. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.
        """

        constrained[_type_is_eq[T, T2](), "value type is not self.T"]()
        for i in self:
            if rebind[Reference[T2, __lifetime_of(self)]](i)[] == value:
                return True
        return False

    @always_inline("nodebug")
    fn __mul__(self, x: Int) -> Self:
        """Multiplies the list by x and returns a new list.

        Args:
            x: The multiplier number.

        Returns:
            The new list.
        """
        # avoid the copy since it would be cleared immediately anyways
        if x == 0:
            return Self()
        var result = List(self)
        result.__mul(x)
        return result^

    @always_inline("nodebug")
    fn __imul__(inout self, x: Int):
        """Multiplies the list by x in place.

        Args:
            x: The multiplier number.
        """
        self.__mul(x)

    @always_inline("nodebug")
    fn __add__(self, owned other: Self) -> Self:
        """Concatenates self with other and returns the result as a new list.

        Args:
            other: List whose elements will be combined with the elements of self.

        Returns:
            The newly created list.
        """
        var result = List(self)
        result.extend(other^)
        return result^

    @always_inline("nodebug")
    fn __iadd__(inout self, owned other: Self):
        """Appends the elements of other into self.

        Args:
            other: List whose elements will be appended to self.
        """
        self.extend(other^)

    fn __iter__(ref [_]self: Self) -> _ListIter[T, __lifetime_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _ListIter(0, self)

    fn __reversed__(
        ref [_]self: Self,
    ) -> _ListIter[T, __lifetime_of(self), False]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _ListIter[forward=False](len(self), self)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) -> Int:
        """Gets the number of elements in the list.

        Returns:
            The number of elements in the list.
        """
        return self.size

    fn __bool__(self) -> Bool:
        """Checks whether the list has any elements or not.

        Returns:
            `False` if the list is empty, `True` if there is at least one element.
        """
        return len(self) > 0

    fn __str__[U: RepresentableCollectionElement](self: List[U]) -> String:
        """Returns a string representation of a `List`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.__str__())
        ```

        When the compiler supports conditional methods, then a simple `str(my_list)` will
        be enough.

        The elements' type must implement the `__repr__()` for this to work.

        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the list.
        """
        # we do a rough estimation of the number of chars that we'll see
        # in the final string, we assume that str(x) will be at least one char.
        var minimum_capacity = (
            2  # '[' and ']'
            + len(self) * 3  # str(x) and ", "
            - 2  # remove the last ", "
        )
        var string_buffer = List[UInt8](capacity=minimum_capacity)
        string_buffer.append(0)  # Null terminator
        var result = String(string_buffer^)
        result += "["
        for i in range(len(self)):
            result += repr(self[i])
            if i < len(self) - 1:
                result += ", "
        result += "]"
        return result

    fn __repr__[U: RepresentableCollectionElement](self: List[U]) -> String:
        """Returns a string representation of a `List`.
        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.__repr__(my_list))
        ```

        When the compiler supports conditional methods, then a simple `repr(my_list)` will
        be enough.

        The elements' type must implement the `__repr__()` for this to work.

        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the list.
        """
        return self.__str__()

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn _realloc(inout self, new_capacity: Int):
        var new_data = UnsafePointer[T].alloc(new_capacity)

        for i in range(self.size):
            (self.data + i).move_pointee_into(new_data + i)

        if self.data:
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    @always_inline
    fn append(inout self, owned value: T):
        """Appends a value to this list.

        Args:
            value: The value to append.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        (self.data + self.size).init_pointee_move(value^)
        self.size += 1

    @always_inline
    fn insert(inout self, i: Int, owned value: T):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value.
            value: The value to insert.
        """
        debug_assert(i <= self.size, "insert index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx = max(0, len(self) + i)

        var earlier_idx = len(self)
        var later_idx = len(self) - 1
        self.append(value^)

        for _ in range(normalized_idx, len(self) - 1):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = earlier_ptr.take_pointee()
            later_ptr.move_pointee_into(earlier_ptr)
            later_ptr.init_pointee_move(tmp^)

            earlier_idx -= 1
            later_idx -= 1

    @always_inline
    fn __mul(inout self, x: Int):
        """Appends the original elements of this list x-1 times.

        ```mojo
        var a = List[Int](1, 2)
        a.__mul(2) # a = [1, 2, 1, 2]
        ```

        Args:
            x: The multiplier number.
        """
        if x == 0:
            self.clear()
            return
        var orig = List(self)
        self.reserve(len(self) * x)
        for i in range(x - 1):
            self.extend(orig)

    @always_inline
    fn extend(inout self, owned other: List[T]):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: List whose elements will be added in order at the end of this list.
        """

        var final_size = len(self) + len(other)
        var other_original_size = len(other)

        self.reserve(final_size)

        # Defensively mark `other` as logically being empty, as we will be doing
        # consuming moves out of `other`, and so we want to avoid leaving `other`
        # in a partially valid state where some elements have been consumed
        # but are still part of the valid `size` of the list.
        #
        # That invalid intermediate state of `other` could potentially be
        # visible outside this function if a `__moveinit__()` constructor were
        # to throw (not currently possible AFAIK though) part way through the
        # logic below.
        other.size = 0

        var dest_ptr = self.data + len(self)

        for i in range(other_original_size):
            var src_ptr = other.data + i

            # This (TODO: optimistically) moves an element directly from the
            # `other` list into this list using a single `T.__moveinit()__`
            # call, without moving into an intermediate temporary value
            # (avoiding an extra redundant move constructor call).
            src_ptr.move_pointee_into(dest_ptr)

            dest_ptr = dest_ptr + 1

        # Update the size now that all new elements have been moved into this
        # list.
        self.size = final_size

    @always_inline
    fn pop(inout self, i: Int = -1) -> T:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        debug_assert(-len(self) <= i < len(self), "pop index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        var ret_val = (self.data + normalized_idx).take_pointee()
        for j in range(normalized_idx + 1, self.size):
            (self.data + j).move_pointee_into(self.data + j - 1)
        self.size -= 1
        if self.size * 4 < self.capacity:
            if self.capacity > 1:
                self._realloc(self.capacity // 2)
        return ret_val^

    @always_inline
    fn reserve(inout self, new_capacity: Int):
        """Reserves the requested capacity.

        If the current capacity is greater or equal, this is a no-op.
        Otherwise, the storage is reallocated and the date is moved.

        Args:
            new_capacity: The new capacity.
        """
        if self.capacity >= new_capacity:
            return
        self._realloc(new_capacity)

    @always_inline
    fn resize(inout self, new_size: Int, value: T):
        """Resizes the list to the given new size.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        list is appended with new values elements up to the requested size.

        Args:
            new_size: The new size.
            value: The value to use to populate new elements.
        """
        if new_size <= self.size:
            self.resize(new_size)
        else:
            self.reserve(new_size)
            for i in range(new_size, self.size):
                (self.data + i).destroy_pointee()
            for i in range(self.size, new_size):
                (self.data + i).init_pointee_copy(value)
            self.size = new_size

    @always_inline
    fn resize(inout self, new_size: Int):
        """Resizes the list to the given new size.

        With no new value provided, the new size must be smaller than or equal
        to the current one. Elements at the end are discarded.

        Args:
            new_size: The new size.
        """
        debug_assert(
            new_size <= self.size,
            (
                "New size must be smaller than or equal to current size when no"
                " new value is provided."
            ),
        )
        for i in range(new_size, self.size):
            (self.data + i).destroy_pointee()
        self.size = new_size
        self.reserve(new_size)

    fn reverse(inout self):
        """Reverses the elements of the list."""
        try:
            self._reverse()
        except:
            abort("unreachable: default _reverse start unexpectedly fails")

    # This method is private to avoid exposing the non-Pythonic `start` argument.
    @always_inline
    fn _reverse(inout self, start: Int = 0) raises:
        """Reverses the elements of the list at positions after `start`.

        Args:
            start: An integer indicating the position after which to reverse elements.
        """
        var start_idx = start if start >= 0 else len(self) + start
        if start_idx < 0 or start_idx > len(self):
            raise "IndexError: start index out of range."

        var earlier_idx = start_idx
        var later_idx = len(self) - 1

        var effective_len = len(self) - start_idx
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = earlier_ptr.take_pointee()
            later_ptr.move_pointee_into(earlier_ptr)
            later_ptr.init_pointee_move(tmp^)

            earlier_idx += 1
            later_idx -= 1

    # TODO: Remove explicit self type when issue 1876 is resolved.
    fn index[
        C: ComparableCollectionElement
    ](
        ref [_]self: List[C],
        value: C,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) raises -> Int:
        """
        Returns the index of the first occurrence of a value in a list
        restricted by the range given the start and stop bounds.

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.index(2)) # prints `1`
        ```

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to None, which means the end of the list).

        Parameters:
            C: The type of the elements in the list. Must implement the
                `ComparableCollectionElement` trait.

        Returns:
            The index of the first occurrence of the value in the list.

        Raises:
            ValueError: If the value is not found in the list.
        """
        var start_normalized = start

        var stop_normalized: Int
        if stop is None:
            # Default end
            stop_normalized = len(self)
        else:
            stop_normalized = stop.value()

        if start_normalized < 0:
            start_normalized += len(self)
        if stop_normalized < 0:
            stop_normalized += len(self)

        start_normalized = _clip(start_normalized, 0, len(self))
        stop_normalized = _clip(stop_normalized, 0, len(self))

        for i in range(start_normalized, stop_normalized):
            if self[i] == value:
                return i
        raise "ValueError: Given element is not in list"

    fn clear(inout self):
        """Clears the elements in the list."""
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.size = 0

    fn steal_data(inout self) -> UnsafePointer[T]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.
        """
        var ptr = self.data
        self.data = UnsafePointer[T]()
        self.size = 0
        self.capacity = 0
        return ptr

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
        var adjusted_span_len = adjusted_span.unsafe_indices()

        if not adjusted_span_len:
            return Self()

        var res = Self(capacity=adjusted_span_len)
        for i in range(adjusted_span_len):
            res.append(self[adjusted_span[i]])

        return res^

    @always_inline
    fn __getitem__(self, idx: Int) -> T:
        """Gets a copy of the list element at the given index.

        FIXME(lifetimes): This should return a reference, not a copy!

        Args:
            idx: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        var normalized_idx = idx
        debug_assert(
            -self.size <= normalized_idx < self.size,
            "index must be within bounds",
        )
        if normalized_idx < 0:
            normalized_idx += len(self)

        return (self.data + normalized_idx)[]

    # TODO(30737): Replace __getitem__ with this, but lots of places use it
    fn __get_ref(
        ref [_]self: Self, i: Int
    ) -> Reference[T, __lifetime_of(self)]:
        """Gets a reference to the list element at the given index.

        Args:
            i: The index of the element.

        Returns:
            An immutable reference to the element at the given index.
        """
        var normalized_idx = i
        if i < 0:
            normalized_idx += self.size

        return self.unsafe_get(normalized_idx)

    @always_inline
    fn unsafe_get(
        ref [_]self: Self, idx: Int
    ) -> Reference[Self.T, __lifetime_of(self)]:
        """Get a reference to an element of self without checking index bounds.

        Users should consider using `__getitem__` instead of this method as it
        is unsafe. If an index is out of bounds, this method will not abort, it
        will be considered undefined behavior.

        Note that there is no wraparound for negative indices, caution is
        advised. Using negative indices is considered undefined behavior. Never
        use `my_list.unsafe_get(-1)` to get the last element of the list.
        Instead, do `my_list.unsafe_get(len(my_list) - 1)`.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the given index.
        """
        debug_assert(
            0 <= idx < len(self),
            (
                "The index provided must be within the range [0, len(List) -1]"
                " when using List.unsafe_get()"
            ),
        )
        return (self.data + idx)[]

    fn count[T: ComparableCollectionElement](self: List[T], value: T) -> Int:
        """Counts the number of occurrences of a value in the list.
        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below.

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.count(1))
        ```

        When the compiler supports conditional methods, then a simple `my_list.count(1)` will
        be enough.

        Parameters:
            T: The type of the elements in the list. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.
        """
        var count = 0
        for elem in self:
            if elem[] == value:
                count += 1
        return count

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The UnsafePointer to the underlying memory.
        """
        return self.data


fn _clip(value: Int, start: Int, end: Int) -> Int:
    return max(start, min(value, end))
