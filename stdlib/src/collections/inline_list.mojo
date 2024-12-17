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
"""Defines the `InlineList` type.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import InlineList
```
"""

from sys.intrinsics import _type_is_eq

from memory.maybe_uninitialized import UnsafeMaybeUninitialized


# ===-----------------------------------------------------------------------===#
# InlineList
# ===-----------------------------------------------------------------------===#
@value
struct _InlineListIter[
    list_mutability: Bool, //,
    T: CollectionElementNew,
    capacity: Int,
    list_origin: Origin[list_mutability],
    forward: Bool = True,
]:
    """Iterator for InlineList.

    Parameters:
        list_mutability: Whether the reference to the list is mutable.
        T: The type of the elements in the list.
        capacity: The maximum number of elements that the list can hold.
        list_origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias list_type = InlineList[T, capacity]

    var index: Int
    var src: Pointer[Self.list_type, list_origin]

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        mut self,
    ) -> Pointer[T, __origin_of(self.src[][0])]:
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


# TODO: Provide a smarter default for the capacity.
struct InlineList[ElementType: CollectionElementNew, capacity: Int = 16](Sized):
    """A list allocated on the stack with a maximum size known at compile time.

    It is backed by an `InlineArray` and an `Int` to represent the size.
    This struct has the same API as a regular `List`, but it is not possible to change the
    capacity. In other words, it has a fixed maximum size.

    This is typically faster than a `List` as it is only stack-allocated and does not require
    any dynamic memory allocation.

    Parameters:
        ElementType: The type of the elements in the list.
        capacity: The maximum number of elements that the list can hold.
    """

    # Fields
    var _array: InlineArray[UnsafeMaybeUninitialized[ElementType], capacity]
    var _size: Int

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """This constructor creates an empty InlineList."""
        self._array = InlineArray[
            UnsafeMaybeUninitialized[ElementType], capacity
        ](unsafe_uninitialized=True)
        self._size = 0

    # TODO: Avoid copying elements in once owned varargs
    # allow transfers.
    @implicit
    fn __init__(out self, *values: ElementType):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
        """
        debug_assert(len(values) <= capacity, "List is full.")
        self = Self()
        for value in values:
            self.append(ElementType(other=value[]))

    @always_inline
    fn __del__(owned self):
        """Destroy all the elements in the list and free the memory."""
        for i in range(self._size):
            self._array[i].assume_initialized_destroy()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(
        ref self, owned idx: Int
    ) -> ref [self._array] Self.ElementType:
        """Get a `Pointer` to the element at the given index.

        Args:
            idx: The index of the item.

        Returns:
            A reference to the item at the given index.
        """
        debug_assert(
            -self._size <= idx < self._size, "Index must be within bounds."
        )

        if idx < 0:
            idx += len(self)

        return self._array[idx].assume_initialized()

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __len__(self) -> Int:
        """Returns the length of the list.

        Returns:
            The number of elements in the list.
        """
        return self._size

    @always_inline
    fn __bool__(self) -> Bool:
        """Checks whether the list has any elements or not.

        Returns:
            `False` if the list is empty, `True` if there is at least one element.
        """
        return len(self) > 0

    fn __iter__(
        ref self,
    ) -> _InlineListIter[ElementType, capacity, __origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        return _InlineListIter(0, Pointer.address_of(self))

    fn __contains__[
        C: EqualityComparableCollectionElement, //
    ](self, value: C) -> Bool:
        """Verify if a given value is present in the list.

        ```mojo
        var x = InlineList[Int](1,2,3)
        if 3 in x: print("x contains 3")
        ```
        Parameters:
            C: The type of the elements in the list. Must implement the
              traits `EqualityComparable` and `CollectionElementNew`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.
        """

        constrained[
            _type_is_eq[ElementType, C](), "value type is not self.ElementType"
        ]()
        for e in self:
            if rebind[C](e[]) == value:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn count[C: EqualityComparableCollectionElement, //](self, value: C) -> Int:
        """Counts the number of occurrences of a value in the list.

        ```mojo
        var my_list = InlineList[Int](1, 2, 3)
        print(my_list.count(1))
        ```
        Parameters:
            C: The type of the elements in the list. Must implement the
              traits `EqualityComparable` and `CollectionElementNew`.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.
        """
        constrained[
            _type_is_eq[ElementType, C](), "value type is not self.ElementType"
        ]()

        var count = 0
        for e in self:
            count += int(rebind[C](e[]) == value)
        return count

    fn append(mut self, owned value: ElementType):
        """Appends a value to the list.

        Args:
            value: The value to append.
        """
        debug_assert(self._size < capacity, "List is full.")
        self._array[self._size].write(value^)
        self._size += 1
