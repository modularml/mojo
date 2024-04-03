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


from memory.anypointer import AnyPointer
from memory.unsafe import Reference

# ===----------------------------------------------------------------------===#
# Utilties
# ===----------------------------------------------------------------------===#


@always_inline
fn _max(a: Int, b: Int) -> Int:
    return a if a > b else b


# ===----------------------------------------------------------------------===#
# List
# ===----------------------------------------------------------------------===#


@value
struct _ListIter[
    T: CollectionElement,
    list_mutability: __mlir_type.`i1`,
    list_lifetime: AnyLifetime[list_mutability].type,
]:
    """Iterator for List.

    Parameters:
        T: The type of the elements in the list.
        list_mutability: Whether the reference to the list is mutable.
        list_lifetime: The lifetime of the List
    """

    alias list_type = List[T]

    var index: Int
    var src: Reference[Self.list_type, list_mutability, list_lifetime]

    fn __next__(
        inout self,
    ) -> Reference[T, list_mutability, list_lifetime]:
        self.index += 1
        return self.src[].__get_ref[list_mutability, list_lifetime](
            self.index - 1
        )

    fn __len__(self) -> Int:
        return len(self.src[]) - self.index


struct List[T: CollectionElement](CollectionElement, Sized):
    """The `List` type is a dynamically-allocated list.

    It supports pushing and popping from the back resizing the underlying
    storage as needed.  When it is deallocated, it frees its memory.

    Parameters:
        T: The type of the elements.
    """

    var data: AnyPointer[T]
    """The underlying storage for the list."""
    var size: Int
    """The number of elements in the list."""
    var capacity: Int
    """The amount of elements that can fit in the list without resizing it."""

    fn __init__(inout self):
        """Constructs an empty list."""
        self.data = AnyPointer[T]()
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
        self.data = AnyPointer[T].alloc(capacity)
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
            _ = (self.data + i).take_value()
        if self.data:
            self.data.free()

    fn __len__(self) -> Int:
        """Gets the number of elements in the list.

        Returns:
            The number of elements in the list.
        """
        return self.size

    @always_inline
    fn _realloc(inout self, new_capacity: Int):
        var new_data = AnyPointer[T].alloc(new_capacity)

        for i in range(self.size):
            (new_data + i).emplace_value((self.data + i).take_value())

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
            self._realloc(_max(1, self.capacity * 2))
        (self.data + self.size).emplace_value(value^)
        self.size += 1

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
            src_ptr.move_into(dest_ptr)

            dest_ptr = dest_ptr + 1

        # Update the size now that all new elements have been moved into this
        # list.
        self.size = final_size

    @always_inline
    fn pop_back(inout self) -> T:
        """Pops a value from the back of this list.

        Returns:
            The popped value.
        """
        var ret_val = (self.data + (self.size - 1)).take_value()
        self.size -= 1
        if self.size * 4 < self.capacity:
            if self.capacity > 1:
                self._realloc(self.capacity // 2)
        return ret_val^

    @always_inline
    fn pop(inout self, i: Int = -1) -> T:
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.
        """
        debug_assert(-self.size <= i < self.size, "pop index out of range")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        var ret_val = (self.data + normalized_idx).take_value()
        for j in range(normalized_idx + 1, self.size):
            (self.data + j).move_into(self.data + j - 1)
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
        self.reserve(new_size)
        for i in range(new_size, self.size):
            _ = (self.data + i).take_value()
        for i in range(self.size, new_size):
            (self.data + i).emplace_value(value)
        self.size = new_size

    fn reverse(inout self):
        """Reverses the elements of the list."""

        self._reverse()

    # This method is private to avoid exposing the non-Pythonic `start` argument.
    @always_inline
    fn _reverse(inout self, start: Int = 0):
        """Reverses the elements of the list at positions after `start`.

        Args:
            start: A non-negative integer indicating the position after which to reverse elements.
        """

        # TODO(polish): Support a negative slice-like start position here that
        #               counts from the end.
        debug_assert(
            start >= 0,
            "List reverse start position must be non-negative",
        )

        var earlier_idx = start
        var later_idx = len(self) - 1

        var effective_len = len(self) - start
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self.data + earlier_idx
            var later_ptr = self.data + later_idx

            var tmp = earlier_ptr.take_value()
            later_ptr.move_into(earlier_ptr)
            later_ptr.emplace_value(tmp^)

            earlier_idx += 1
            later_idx -= 1

    fn clear(inout self):
        """Clears the elements in the list."""
        for i in range(self.size):
            _ = (self.data + i).take_value()
        self.size = 0

    fn steal_data(inout self) -> AnyPointer[T]:
        """Take ownership of the underlying pointer from the list.

        Returns:
            The underlying data.
        """
        var ptr = self.data
        self.data = AnyPointer[T]()
        self.size = 0
        self.capacity = 0
        return ptr

    fn __setitem__(inout self, i: Int, owned value: T):
        """Sets a list element at the given index.

        Args:
            i: The index of the element.
            value: The value to assign.
        """
        debug_assert(-self.size <= i < self.size, "index must be within bounds")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        _ = (self.data + normalized_idx).take_value()
        (self.data + normalized_idx).emplace_value(value^)

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

        var res = Self(capacity=len(adjusted_span))
        for i in range(len(adjusted_span)):
            res.append(self[adjusted_span[i]])

        return res^

    @always_inline
    fn __getitem__(self, i: Int) -> T:
        """Gets a copy of the list element at the given index.

        FIXME(lifetimes): This should return a reference, not a copy!

        Args:
            i: The index of the element.

        Returns:
            A copy of the element at the given index.
        """
        debug_assert(-self.size <= i < self.size, "index must be within bounds")

        var normalized_idx = i
        if i < 0:
            normalized_idx += len(self)

        return (self.data + normalized_idx)[]

    # TODO(30737): Replace __getitem__ with this as __refitem__, but lots of places use it
    fn __get_ref[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
        i: Int,
    ) -> Reference[T, mutability, self_life]:
        """Gets a reference to the list element at the given index.

        Args:
            i: The index of the element.

        Returns:
            An immutable reference to the element at the given index.
        """
        var normalized_idx = i
        if i < 0:
            normalized_idx += Reference(self)[].size

        # Mutability gets set to the local mutability of this
        # pointer value, ie. because we defined it with `let` it's now an
        # "immutable" reference regardless of the mutability of `self`.
        # This means we can't just use `AnyPointer.__refitem__` here
        # because the mutability won't match.
        var base_ptr = Reference(self)[].data
        return __mlir_op.`lit.ref.from_pointer`[
            _type = Reference[T, mutability, self_life].mlir_ref_type
        ]((base_ptr + normalized_idx).value)

    fn __iter__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _ListIter[
        T, mutability, self_life
    ]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _ListIter[T, mutability, self_life](0, Reference(self))
