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


from os import abort
from sys import sizeof
from sys.intrinsics import _type_is_eq

from memory import Pointer, UnsafePointer, memcpy, Span

from .optional import Optional

# ===-----------------------------------------------------------------------===#
# List
# ===-----------------------------------------------------------------------===#


@value
struct _ListIter[
    list_mutability: Bool, //,
    T: CollectionElement,
    hint_trivial_type: Bool,
    list_origin: Origin[list_mutability],
    forward: Bool = True,
]:
    """Iterator for List.

    Parameters:
        list_mutability: Whether the reference to the list is mutable.
        T: The type of the elements in the list.
        hint_trivial_type: Set to `True` if the type `T` is trivial, this is not mandatory,
            but it helps performance. Will go away in the future.
        list_origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias list_type = List[T, hint_trivial_type]

    var index: Int
    var src: Pointer[Self.list_type, list_origin]

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        mut self,
    ) -> Pointer[T, list_origin]:
        @parameter
        if forward:
            self.index += 1
            return Pointer.address_of(self.src[][self.index - 1])
        else:
            self.index -= 1
            return Pointer.address_of(self.src[][self.index])

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        @parameter
        if forward:
            return len(self.src[]) - self.index
        else:
            return self.index


struct List[T: CollectionElement, hint_trivial_type: Bool = False](
    CollectionElement, CollectionElementNew, Sized, Boolable
):
    """The `List` type is a dynamically-allocated list.

    It supports pushing and popping from the back resizing the underlying
    storage as needed.  When it is deallocated, it frees its memory.

    Parameters:
        T: The type of the elements.
        hint_trivial_type: A hint to the compiler that the type T is trivial.
            It's not mandatory, but if set, it allows some optimizations.
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

    fn __init__(out self):
        """Constructs an empty list."""
        self.data = UnsafePointer[T]()
        self.size = 0
        self.capacity = 0

    fn __init__(out self, *, other: Self):
        """Creates a deep copy of the given list.

        Args:
            other: The list to copy.
        """
        self = Self(capacity=other.capacity)
        for e in other:
            self.append(e[])

    fn __init__(out self, *, capacity: Int):
        """Constructs a list with the given capacity.

        Args:
            capacity: The requested capacity of the list.
        """
        self.data = UnsafePointer[T].alloc(capacity)
        self.size = 0
        self.capacity = capacity

    @implicit
    fn __init__(out self, owned *values: T):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
        """
        self = Self(elements=values^)

    fn __init__(out self, *, owned elements: VariadicListMem[T, _]):
        """Constructs a list from the given values.

        Args:
            elements: The values to populate the list with.
        """
        var length = len(elements)

        self = Self(capacity=length)

        for i in range(length):
            var src = UnsafePointer.address_of(elements[i])
            var dest = self.data + i

            src.move_pointee_into(dest)

        # Do not destroy the elements when their backing storage goes away.
        __mlir_op.`lit.ownership.mark_destroyed`(
            __get_mvalue_as_litref(elements)
        )

        self.size = length

    @implicit
    fn __init__(out self, span: Span[T]):
        """Constructs a list from the a Span of values.

        Args:
            span: The span of values to populate the list with.
        """
        self = Self(capacity=len(span))
        for value in span:
            self.append(value[])

    fn __init__(mut self, *, ptr: UnsafePointer[T], length: Int, capacity: Int):
        """Constructs a list from a pointer, its length, and its capacity.

        Args:
            ptr: The pointer to the data.
            length: The number of elements in the list.
            capacity: The capacity of the list.
        """
        self.data = ptr
        self.size = length
        self.capacity = capacity

    fn __moveinit__(out self, owned existing: Self):
        """Move data of an existing list into a new one.

        Args:
            existing: The existing list.
        """
        self.data = existing.data
        self.size = existing.size
        self.capacity = existing.capacity

    fn __copyinit__(out self, existing: Self):
        """Creates a deepcopy of the given list.

        Args:
            existing: The list to copy.
        """
        self = Self(capacity=existing.capacity)
        for i in range(len(existing)):
            self.append(existing[i])

    fn __del__(owned self):
        """Destroy all elements in the list and free its memory."""

        @parameter
        if not hint_trivial_type:
            for i in range(self.size):
                (self.data + i).destroy_pointee()
        self.data.free()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __eq__[
        U: EqualityComparableCollectionElement, //
    ](self: List[U, *_], other: List[U, *_]) -> Bool:
        """Checks if two lists are equal.

        Examples:
        ```mojo
        var x = List[Int](1, 2, 3)
        var y = List[Int](1, 2, 3)
        if x == y: print("x and y are equal")
        ```

        Parameters:
            U: The type of the elements in the list. Must implement the
               traits `EqualityComparable` and `CollectionElement`.

        Args:
            other: The list to compare with.

        Returns:
            True if the lists are equal, False otherwise.
        """
        if len(self) != len(other):
            return False
        var index = 0
        for element in self:
            if element[] != other[index]:
                return False
            index += 1
        return True

    @always_inline
    fn __ne__[
        U: EqualityComparableCollectionElement, //
    ](self: List[U, *_], other: List[U, *_]) -> Bool:
        """Checks if two lists are not equal.

        Examples:

        ```mojo
        var x = List[Int](1, 2, 3)
        var y = List[Int](1, 2, 4)
        if x != y: print("x and y are not equal")
        ```

        Parameters:
            U: The type of the elements in the list. Must implement the
               traits `EqualityComparable` and `CollectionElement`.

        Args:
            other: The list to compare with.

        Returns:
            True if the lists are not equal, False otherwise.
        """
        return not (self == other)

    fn __contains__[
        U: EqualityComparableCollectionElement, //
    ](self: List[U, *_], value: U) -> Bool:
        """Verify if a given value is present in the list.

        ```mojo
        var x = List[Int](1,2,3)
        if 3 in x: print("x contains 3")
        ```
        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.
        """
        for i in self:
            if i[] == value:
                return True
        return False

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
        var result = List(other=self)
        result.__mul(x)
        return result^

    fn __imul__(mut self, x: Int):
        """Multiplies the list by x in place.

        Args:
            x: The multiplier number.
        """
        self.__mul(x)

    fn __add__(self, owned other: Self) -> Self:
        """Concatenates self with other and returns the result as a new list.

        Args:
            other: List whose elements will be combined with the elements of self.

        Returns:
            The newly created list.
        """
        var result = List(other=self)
        result.extend(other^)
        return result^

    fn __iadd__(mut self, owned other: Self):
        """Appends the elements of other into self.

        Args:
            other: List whose elements will be appended to self.
        """
        self.extend(other^)

    fn __iter__(ref self) -> _ListIter[T, hint_trivial_type, __origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _ListIter(0, Pointer.address_of(self))

    fn __reversed__(
        ref self,
    ) -> _ListIter[T, hint_trivial_type, __origin_of(self), False]:
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _ListIter[forward=False](len(self), Pointer.address_of(self))

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

    @no_inline
    fn __str__[
        U: RepresentableCollectionElement, //
    ](self: List[U, *_]) -> String:
        """Returns a string representation of a `List`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.__str__())
        ```

        When the compiler supports conditional methods, then a simple `str(my_list)` will
        be enough.

        The elements' type must implement the `__repr__()` method for this to work.

        Parameters:
            U: The type of the elements in the list. Must implement the
              traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the list.
        """
        var output = String()
        self.write_to(output)
        return output^

    @no_inline
    fn write_to[
        W: Writer, U: RepresentableCollectionElement, //
    ](self: List[U, *_], mut writer: W):
        """Write `my_list.__str__()` to a `Writer`.

        Parameters:
            W: A type conforming to the Writable trait.
            U: The type of the List elements. Must have the trait `RepresentableCollectionElement`.

        Args:
            writer: The object to write to.
        """
        writer.write("[")
        for i in range(len(self)):
            writer.write(repr(self[i]))
            if i < len(self) - 1:
                writer.write(", ")
        writer.write("]")

    @no_inline
    fn __repr__[
        U: RepresentableCollectionElement, //
    ](self: List[U, *_]) -> String:
        """Returns a string representation of a `List`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        var my_list = List[Int](1, 2, 3)
        print(my_list.__repr__())
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

    fn bytecount(self) -> Int:
        """Gets the bytecount of the List.

        Returns:
            The bytecount of the List.
        """
        return len(self) * sizeof[T]()

    fn _realloc(mut self, new_capacity: Int):
        var new_data = UnsafePointer[T].alloc(new_capacity)

        _move_pointee_into_many_elements[hint_trivial_type](
            dest=new_data,
            src=self.data,
            size=self.size,
        )

        if self.data:
            self.data.free()
        self.data = new_data
        self.capacity = new_capacity

    fn append(mut self, owned value: T):
        """Appends a value to this list.

        Args:
            value: The value to append.
        """
        if self.size >= self.capacity:
            self._realloc(max(1, self.capacity * 2))
        (self.data + self.size).init_pointee_move(value^)
        self.size += 1

    fn insert(mut self, i: Int, owned value: T):
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

    fn __mul(mut self, x: Int):
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
        var orig = List(other=self)
        self.reserve(len(self) * x)
        for i in range(x - 1):
            self.extend(orig)

    fn extend(mut self, owned other: List[T, *_]):
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

    fn pop(mut self, i: Int = -1) -> T:
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

    fn reserve(mut self, new_capacity: Int):
        """Reserves the requested capacity.

        If the current capacity is greater or equal, this is a no-op.
        Otherwise, the storage is reallocated and the date is moved.

        Args:
            new_capacity: The new capacity.
        """
        if self.capacity >= new_capacity:
            return
        self._realloc(new_capacity)

    fn resize(mut self, new_size: Int, value: T):
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
            for i in range(self.size, new_size):
                (self.data + i).init_pointee_copy(value)
            self.size = new_size

    fn resize(mut self, new_size: Int):
        """Resizes the list to the given new size.

        With no new value provided, the new size must be smaller than or equal
        to the current one. Elements at the end are discarded.

        Args:
            new_size: The new size.
        """
        if self.size < new_size:
            abort(
                "You are calling List.resize with a new_size bigger than the"
                " current size. If you want to make the List bigger, provide a"
                " value to fill the new slots with. If not, make sure the new"
                " size is smaller than the current size."
            )
        for i in range(new_size, self.size):
            (self.data + i).destroy_pointee()
        self.size = new_size
        self.reserve(new_size)

    fn reverse(mut self):
        """Reverses the elements of the list."""

        var earlier_idx = 0
        var later_idx = len(self) - 1

        var effective_len = len(self)
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
        C: EqualityComparableCollectionElement, //
    ](
        ref self: List[C, *_],
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
                `EqualityComparableCollectionElement` trait.

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

    fn clear(mut self):
        """Clears the elements in the list."""
        for i in range(self.size):
            (self.data + i).destroy_pointee()
        self.size = 0

    fn steal_data(mut self) -> UnsafePointer[T]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.
        """
        var ptr = self.data
        self.data = UnsafePointer[T]()
        self.size = 0
        self.capacity = 0
        return ptr

    fn __getitem__(self, span: Slice) -> Self:
        """Gets the sequence of elements at the specified positions.

        Args:
            span: A slice that specifies positions of the new list.

        Returns:
            A new list containing the list at the specified span.
        """

        var start: Int
        var end: Int
        var step: Int
        start, end, step = span.indices(len(self))
        var r = range(start, end, step)

        if not len(r):
            return Self()

        var res = Self(capacity=len(r))
        for i in r:
            res.append(self[i])

        return res^

    fn __getitem__(ref self, idx: Int) -> ref [self] T:
        """Gets the list element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """

        var normalized_idx = idx

        debug_assert(
            -self.size <= normalized_idx < self.size,
            "index: ",
            normalized_idx,
            " is out of bounds for `List` of size: ",
            self.size,
        )
        if normalized_idx < 0:
            normalized_idx += len(self)

        return (self.data + normalized_idx)[]

    @always_inline
    fn unsafe_get(ref self, idx: Int) -> ref [self] Self.T:
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

    @always_inline
    fn unsafe_set(mut self, idx: Int, owned value: T):
        """Write a value to a given location without checking index bounds.

        Users should consider using `my_list[idx] = value` instead of this method as it
        is unsafe. If an index is out of bounds, this method will not abort, it
        will be considered undefined behavior.

        Note that there is no wraparound for negative indices, caution is
        advised. Using negative indices is considered undefined behavior. Never
        use `my_list.unsafe_set(-1, value)` to set the last element of the list.
        Instead, do `my_list.unsafe_set(len(my_list) - 1, value)`.

        Args:
            idx: The index of the element to set.
            value: The value to set.
        """
        debug_assert(
            0 <= idx < len(self),
            (
                "The index provided must be within the range [0, len(List) -1]"
                " when using List.unsafe_set()"
            ),
        )
        (self.data + idx).destroy_pointee()
        (self.data + idx).init_pointee_move(value^)

    fn count[
        T: EqualityComparableCollectionElement, //
    ](self: List[T, *_], value: T) -> Int:
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

    fn swap_elements(mut self, elt_idx_1: Int, elt_idx_2: Int):
        """Swaps elements at the specified indexes if they are different.

        ```mojo
        var my_list = List[Int](1, 2, 3)
        my_list.swap_elements(0, 2)
        print(my_list) # 3, 2, 1
        ```

        This is useful because `swap(my_list[i], my_list[j])` cannot be
        supported by Mojo, because a mutable alias may be formed.

        Args:
            elt_idx_1: The index of one element.
            elt_idx_2: The index of the other element.
        """
        debug_assert(
            0 <= elt_idx_1 < len(self) and 0 <= elt_idx_2 < len(self),
            (
                "The indices provided to swap_elements must be within the range"
                " [0, len(List)-1]"
            ),
        )
        if elt_idx_1 != elt_idx_2:
            swap((self.data + elt_idx_1)[], (self.data + elt_idx_2)[])

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[T]:
        """Retrieves a pointer to the underlying memory.

        Returns:
            The UnsafePointer to the underlying memory.
        """
        return self.data


fn _clip(value: Int, start: Int, end: Int) -> Int:
    return max(start, min(value, end))


fn _move_pointee_into_many_elements[
    T: CollectionElement, //, hint_trivial_type: Bool
](dest: UnsafePointer[T], src: UnsafePointer[T], size: Int):
    @parameter
    if hint_trivial_type:
        memcpy(
            dest=dest.bitcast[Int8](),
            src=src.bitcast[Int8](),
            count=size * sizeof[T](),
        )
    else:
        for i in range(size):
            (src + i).move_pointee_into(dest + i)
