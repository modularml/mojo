# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

These APIs are imported automatically, just like builtins.
"""


from std.builtin.rebind import downcast
import std.format._utils as fmt
from std.hashlib import Hasher
from std.reflection import reflect
from std.collections import check_bounds, check_slice_bounds
from std.collections._asan_annotations import (
    __sanitizer_annotate_contiguous_container,
)
from std.os import abort
from std.sys import size_of

from std.memory.alloc import (
    alloc,
    dealloc,
    Allocation,
    ThinAllocation,
    Layout,
)
from std.memory import (
    Pointer,
    unsafe_destroy_n,
    unsafe_memcpy,
    unsafe_uninit_copy_n,
    unsafe_uninit_move_n,
)
from std.builtin.builtin_slice import ContiguousSlice, StridedSlice
from .optional import Optional

# ===-----------------------------------------------------------------------===#
# List
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _ListIter[
    mut: Bool,
    //,
    T: Copyable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for List.

    Parameters:
        mut: Whether the reference to the list is mutable.
        T: The type of the elements in the list.
        origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    comptime Element = Self.T  # FIXME(MOCO-2068): shouldn't be needed.

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var _index: Int
    var _data: Pointer[Self.T, Self.origin]
    var _length: Int

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        comptime if Self.forward:
            if self._index >= self._length:
                raise StopIteration()
            self._index += 1
            return self._data[unsafe_offset=self._index - 1]
        else:
            if self._index <= 0:
                raise StopIteration()
            self._index -= 1
            return self._data[unsafe_offset=self._index]

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = self._length - self._index
        else:
            iter_len = self._index

        return (iter_len, {iter_len})


@fieldwise_init
struct _ListIterOwned[T: Movable & Deinitable](
    IterableOwned, Iterator, Movable
):
    """An owning iterator for List.

    Parameters:
        T: The type of the elements in the list.
    """

    comptime Element = Self.T
    comptime IteratorOwnedType = Self

    var _list: List[Self.T]
    var _index: Int

    @always_inline
    def __deinit__(deinit self):
        # Destroy the remaining elements that have not yet been
        # iterated over.
        unsafe_destroy_n(
            self._list.unsafe_ptr().unsafe_offset(self._index),
            count=len(self._list) - self._index,
        )
        self._list._len = 0

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._index >= len(self._list):
            raise StopIteration()
        self._index += 1
        return (
            self._list.unsafe_ptr()
            .unsafe_offset(self._index - 1)
            .unsafe_take_pointee()
        )

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len = len(self._list) - self._index
        return (iter_len, {iter_len})


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `List` of"
    " non-`Deinitable` elements"
)
@stable(since="1.0")
struct List[T: AnyType, /](
    Boolable,
    Copyable where conforms_to(T, Copyable),
    Defaultable,
    Deinitable where conforms_to(T, Deinitable),
    Equatable where conforms_to(T, Equatable),
    Hashable where conforms_to(T, Hashable),
    Iterable,
    IterableOwned where conforms_to(T, Deinitable & Movable),
    Movable,
    Sized,
    Writable where conforms_to(T, Writable),
):
    """A dynamically-allocated and resizable list.

    This is Mojo's primary dynamic array implementation, meaning the list can
    grow and shrink in size at runtime. However, all elements in a `List` must
    be the same type `T`, determined at compile time.

    You can create a `List` in several ways:

    ```mojo
    # Empty list
    var empty_list = List[Int]()

    # With pre-allocated capacity
    var preallocated = List[String](capacity=100)

    # With initial size and fill value
    var filled = List[Float64](length=10, fill=0.0)

    # With initial values and inferred type (Int).
    # The `List` annotation is needed otherwise this creates a fixed-size `Array`.
    var numbers: List = [1, 2, 3, 4, 5]
    ```

    Be aware of the following characteristics:

    - **Type safety**: All elements must be the same type `T`, determined at
      compile time. This is more restrictive than Python's lists but it
      improves performance:

      ```mojo
      var int_list: List = [1, 2, 3]        # List[Int]
      var str_list: List = ["a", "b", "c"]  # List[String]
      # var mixed: List = [1, "hello"]      # Error! All elements must be same type
      ```

      However, you can get around this by defining your list type as
      [`Variant`](/docs/std/utils/variant/Variant/). This is a discriminated
      union type, meaning it can store any number of different types that can
      vary at runtime.

    - **Value semantics:** A `List` is value semantic by default, so
      assignment creates a deep copy of all elements:

      ```mojo
      var list1: List = [1, 2, 3]
      var list2 = list1.copy()        # Deep copy
      list2.append(4)
      print(list1)   # => [1, 2, 3]
      print(list2)   # => [1, 2, 3, 4]
      ```

      This is different from Python, where assignment creates a reference to
      the same list. For more information, read about [value
      semantics](/docs/manual/values/value-semantics).

    - **Reference iteration uses immutable references**: When iterating a list
      by reference, you get immutable references to the actual elements, unless
      you specify `ref`:

      ```mojo
      var numbers: List = [10, 20, 30]

      # Default behavior creates immutable (read-only) references:
      # for num in numbers:
      #     num += 1  # error: expression must be mutable

      # Using `ref` gets mutable (read-write) references
      for ref num in numbers:
          num += 1  # Modifies the original elements
      print(numbers)  # => [11, 21, 31]
      ```

    - **Owned iteration consumes the list**: Using the transfer sigil (`^`)
      moves the list into the loop, yielding each element by value. The
      original list is no longer accessible after the loop:

      ```mojo
      var names: List = ["alice", "bob"]
      for x in names^:
          # `x` is an owned `String` value.
          print(x)
      # `names` is consumed and can no longer be used here
      ```

    - **Out of bounds access**: Accessing elements with invalid indices will
      abort:

      ```mojo
      var my_list: List = [1, 2, 3]
      print(my_list[5])  # Aborts with an Assert Error: index 5 is
                         # out of bounds, valid range is 0 to 2
      ```

      For safe access, you should manually check bounds or use methods that
      handle errors gracefully:

      ```mojo
      var my_list: List = [1, 2, 3]
      if 5 < len(my_list):
          print(my_list[5])  # Safe: check bounds first
      else:
          print("Index out of bounds")

      # Some methods like index() raise exceptions
      try:
          var idx = my_list.index(99)  # Raises ValueError if not found
          print("Found at index:", idx)
      except:
          print("Value not found in list")
      ```

    Examples:

    ```mojo
    var my_list: List = [10, 20, 30]

    # Add elements
    my_list.append(40)           # [10, 20, 30, 40]
    my_list.insert(1, 15)        # [10, 15, 20, 30, 40]
    my_list.extend([50, 60])     # [10, 15, 20, 30, 40, 50, 60]

    # Access elements
    print(my_list[0])                # 10 (first element)
    print(my_list[len(my_list) - 1]) # 60 (last element)
    my_list[1] = 25                  # Modify element: [10, 25, 20, 30, 40, 50, 60]

    # Remove elements
    print(my_list.pop())      # Removes and returns last element (60)
    print(my_list.pop(2))     # Removes element at index 2 (20)

    # List properties
    print('len:', len(my_list))          # Current number of elements
    print('cap:', my_list.capacity())    # Current allocated capacity

    # Multiply a list
    var pair: List = [1, 2]
    var repeated = pair * 3
    print(repeated)    # [1, 2, 1, 2, 1, 2]

    # Iterate over a list:
    var fruits: List = ["apple", "banana", "orange"]

    # Iterate by reference (immutable)
    for fruit in fruits:
        print(fruit)

    # Iterate backwards by reference
    for fruit in reversed(fruits):
        print(fruit)

    # Iterate by index
    for i in range(len(fruits)):
        print(i, fruits[i])

    # Iterate by ownership (consumes the list)
    var temps: List = ["a", "b", "c"]
    for x in temps^:
        print(x)
    # `temps` is no longer accessible here

    # Concatenate with + and +=
    fruits += ["mango"]
    var more_fruits = fruits + ["grape", "kiwi"]
    print(more_fruits)
    ```

    Parameters:
        T: The type of elements stored in the list.
    """

    comptime _PointerType = Pointer[Self.T, MutUntrackedOrigin]

    comptime _InteriorOrigin[origin: Origin] = origin._get_owned_interior[
        "element"
    ]

    # Fields
    var _data: Self._PointerType
    """The underlying storage for the list."""
    var _len: Int
    """The number of elements in the list."""
    var _capacity: Int
    """The amount of elements that can fit in the list without resizing it."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _ListIter[downcast[Self.T, Copyable], iterable_origin, True]
    """The iterator type for this list.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.T, Deinitable & Movable
    ) = _ListIterOwned[Self.T]
    """The owned iterator type for this list."""

    # asan annotation methods
    def _annotate_new(self):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.unsafe_bitcast[NoneType](),
            end=self._data.unsafe_offset(self._capacity).unsafe_bitcast[
                NoneType
            ](),
            old_mid=self._data.unsafe_offset(self._capacity).unsafe_bitcast[
                NoneType
            ](),
            new_mid=self._data.unsafe_offset(self._len).unsafe_bitcast[
                NoneType
            ](),
        )

    def _annotate_delete(self):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.unsafe_bitcast[NoneType](),
            end=self._data.unsafe_offset(self._capacity).unsafe_bitcast[
                NoneType
            ](),
            old_mid=self._data.unsafe_offset(self._len).unsafe_bitcast[
                NoneType
            ](),
            new_mid=self._data.unsafe_offset(self._capacity).unsafe_bitcast[
                NoneType
            ](),
        )

    def _annotate_increase(self, n: Int = 1):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.unsafe_bitcast[NoneType](),
            end=self._data.unsafe_offset(self._capacity).unsafe_bitcast[
                NoneType
            ](),
            old_mid=self._data.unsafe_offset(self._len).unsafe_bitcast[
                NoneType
            ](),
            new_mid=self._data.unsafe_offset(self._len + n).unsafe_bitcast[
                NoneType
            ](),
        )

    def _annotate_shrink(self, old_size: Int):
        __sanitizer_annotate_contiguous_container(
            beg=self._data.unsafe_bitcast[NoneType](),
            end=self._data.unsafe_offset(self._capacity).unsafe_bitcast[
                NoneType
            ](),
            old_mid=self._data.unsafe_offset(old_size).unsafe_bitcast[
                NoneType
            ](),
            new_mid=self._data.unsafe_offset(self._len).unsafe_bitcast[
                NoneType
            ](),
        )

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @stable(since="1.0")
    def __init__(out self):
        """Constructs an empty list."""
        self._data = Self._PointerType.unsafe_dangling()
        self._len = 0
        self._capacity = 0

    @stable(since="1.0")
    def __init__(out self, *, capacity: Int):
        """Constructs a list with the given capacity.

        Args:
            capacity: The requested capacity of the list.
        """
        if capacity:
            self._data = alloc(Layout[Self.T](count=capacity)).unsafe_leak()
        else:
            self._data = Self._PointerType.unsafe_dangling()
        self._len = 0
        self._capacity = capacity
        self._annotate_new()

    @stable(since="1.0")
    def __init__(
        out self, *, length: Int, fill: Self.T
    ) where conforms_to(Self.T, Copyable):
        """Constructs a list with the given length.

        Args:
            length: The requested length of the list.
            fill: The element to fill each element of the list.
        """
        self = Self()
        self._unchecked_grow(length, fill)

    @always_inline
    def __init__(out self, *, length: Int, fill_with: Some[def(Int) -> Self.T]):
        """Constructs a list by calling `fill_with(i)` for each index `i`.

        Args:
            length: The requested length of the list.
            fill_with: A function called with each index in `[0, length)`,
                whose result is written to that position.

        Examples:

        ```mojo
        var squares = List(length=5, fill_with=lambda (i: Int) -> Int: i * i)
        # [0, 1, 4, 9, 16]
        ```
        """
        self = Self(capacity=length)
        self._annotate_increase(length)
        self._len = length

        for i in range(length):
            self._data.unsafe_offset(i).unsafe_write(
                init_with=lambda () {imm} -> Self.T: fill_with(i)
            )

    @always_inline
    def __init__(
        out self, var *values: Self.T, __list_literal__: NoneType
    ) where conforms_to(Self.T, Movable):
        """Constructs a list from the given values.

        Args:
            values: The values to populate the list with.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        var length = len(values)
        self = Self(capacity=length)
        self._annotate_increase(length)

        # Transfer all of the elements into the List.
        def init_elt(idx: Int, var elt: Self.T) {ref}:
            self._data.unsafe_offset(idx).unsafe_write(elt^)

        values^.consume_elements(init_elt)

        # Remember how many values we have.
        self._len = length

    def __init__[
        IterableType: Iterable,
    ](
        ref iterable: IterableType,
        out self: List[IterableType.IteratorType[origin_of(iterable)].Element],
    ) where conforms_to(
        IterableType.IteratorType[origin_of(iterable)].Element, Copyable
    ):
        """Constructs a list from an iterable of values.

        Parameters:
            IterableType: The type of the `iterable` argument.

        Args:
            iterable: The iterable of values to populate the list with.
        """
        var lower, _ = iter(iterable).bounds()
        self = type_of(self)(capacity=lower)
        for var value in iterable:
            self.append(rebind_var[type_of(self).T](value^))

    @always_inline
    def __init__(out self, *, unsafe_uninit_length: Int):
        """Construct a list with the specified length, with uninitialized
        memory. This is unsafe, as it relies on the caller initializing the
        elements with unsafe operations, not assigning over the uninitialized
        data.

        Args:
            unsafe_uninit_length: The number of elements to allocate.
        """
        self = Self(capacity=unsafe_uninit_length)
        self._annotate_increase(unsafe_uninit_length)
        self._len = unsafe_uninit_length

    @stable(since="1.0")
    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        """Creates a deep copy of the given list.

        Args:
            copy: The list to copy.
        """
        self = Self(capacity=copy._capacity)
        self.extend(Span(copy))

    def _unsafe_assume_destroyed_and_deallocate(deinit self):
        """Assumes self's values are already destroyed and deallocate the backing storage.
        """
        if self._capacity > 0:
            self._annotate_delete()
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                    Layout[Self.T](count=self._capacity)
                )
            )

    @stable(since="1.0")
    def __deinit__(deinit self) where conforms_to(Self.T, Deinitable):
        """Destroy all elements in the list and free its memory."""
        unsafe_destroy_n(
            self._data,
            count=len(self),
        )
        self^._unsafe_assume_destroyed_and_deallocate()

    def deinit_with(deinit self, deinit_func: Some[def(var Self.T)], /):
        """Consumes this list and deinitializes its values using the provided closure.

        This can be used to destroy a `List` of non-`Deinitable` values.

        Args:
            deinit_func: The deinitializing closure called on each `List` element.
        """
        for i in range(len(self)):
            # TODO(MOCO-4111): `deinit_func` cannot convert to Pointer.unsafe_deinit_pointee_with
            # `deinit_func` type since UP is bound on `T: AnyType` but List has `T: Movable`.
            deinit_func(
                __get_address_as_owned_value(
                    self._data.unsafe_offset(i)._get_kgen_pointer()
                )
            )
        self^._unsafe_assume_destroyed_and_deallocate()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @stable(since="1.0")
    @always_inline
    def __eq__(
        self, other: Self, /
    ) -> Bool where conforms_to(Self.T, Equatable):
        """Checks if two lists are equal.

        Args:
            other: The list to compare with.

        Returns:
            True if the lists are equal, False otherwise.

        Examples:

        ```mojo
        var x = [1, 2, 3]
        var y = [1, 2, 3]
        print("x and y are equal" if x == y else "x and y are not equal")
        ```
        """
        if len(self) != len(other):
            return False

        var index = 0
        for element in self:
            if element != other[index]:
                return False
            index += 1
        return True

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.T, Hashable):
        """Updates hasher with the hash of each element in the list.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        for element in self:
            element.__hash__(hasher)

    @stable(since="1.0")
    def __contains__(
        self, value: Self.T, /
    ) -> Bool where conforms_to(Self.T, Equatable):
        """Verify if a given value is present in the list.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the list, False otherwise.

        Examples:

        ```mojo
        var x = [1, 2, 3]
        print("x contains 3" if 3 in x else "x does not contain 3")
        ```
        """
        for i in self:
            if i == value:
                return True
        return False

    def __mul__(
        self, x: Int
    ) -> Self where conforms_to(Self.T, Copyable & Deinitable):
        """Multiplies the list by x and returns a new list.

        Args:
            x: The multiplier number.

        Returns:
            The new list.
        """
        # avoid the copy since it would be cleared immediately anyways
        if x == 0:
            return Self()
        var result = self.copy()
        result *= x
        return result^

    def __imul__(
        mut self, x: Int
    ) where conforms_to(Self.T, Deinitable & Copyable) and conforms_to(
        Self, Deinitable & Copyable
    ):
        """Appends the original elements of this list x-1 times or clears it if
        x is <= 0.

        ```mojo
        var a: List = [1, 2]
        a *= 2 # a = [1, 2, 1, 2]
        ```

        Args:
            x: The multiplier number.
        """
        if x <= 0 or len(self) == 0:
            self.clear()
            return
        var orig = self.copy()
        self.reserve(len(self) * x)
        for _ in range(x - 1):
            self.extend(Span(orig))

    def __add__(
        self, var other: Self
    ) -> Self where conforms_to(Self.T, Copyable):
        """Concatenates self with other and returns the result as a new list.

        Args:
            other: List whose elements will be combined with the elements of
                self.

        Returns:
            The newly created list.
        """
        var result = self.copy()
        result.extend(other^)
        return result^

    @stable(since="1.0")
    def __iadd__(
        mut self, var other: Self, /
    ) where conforms_to(Self.T, Copyable):
        """Appends the elements of other into self.

        Args:
            other: List whose elements will be appended to self.
        """
        self.extend(other^)

    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(Self.T, Deinitable & Movable):
        """Consume `self`, returning an owned iterator over its elements.

        Returns:
            An iterator of owned elements.
        """
        return {self^, 0}

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the list, returning immutable references.

        Returns:
            An iterator of immutable references to the list elements.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.T, Copyable
        ), "List iteration requires the element to be `Copyable`."
        return _ListIter(
            0,
            # `_data` points at untracked owned storage; the iterator borrows
            # at this call's origin instead.
            self._data.unsafe_mut_cast[
                origin_of(self).mut
            ]().unsafe_origin_cast[origin_of(self)](),
            self._len,
        )

    def __reversed__(
        ref self,
    ) -> _ListIter[Self.T, origin_of(self), False] where conforms_to(
        Self.T, Copyable
    ):
        """Iterate backwards over the list, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the list elements.
        """
        return _ListIter[forward=False](
            len(self),
            self._data.unsafe_mut_cast[
                origin_of(self).mut
            ]().unsafe_origin_cast[origin_of(self)](),
            self._len,
        )

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Gets the number of elements in the list.

        Returns:
            The number of elements in the list.
        """
        return self._len

    @always_inline("nodebug")
    def __bool__(self) -> Bool:
        """Checks whether the list has any elements or not.

        Returns:
            `False` if the list is empty, `True` if there is at least one
            element.
        """
        return len(self) > 0

    def _write_self_to[
        f: def(Self.T, mut Some[Writer]) thin
    ](self, mut writer: Some[Writer]) where conforms_to(Self.T, Writable):
        var iterator = self.__iter__()

        def iterate(mut w: Some[Writer]) raises StopIteration {mut iterator}:
            f(iterator.__next__(), w)

        fmt.write_sequence_to(writer, iterate)
        _ = iterator^

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this list to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f=fmt.write_to[Self.T]](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write this list to a `Writer`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_self_to[f=fmt.write_repr_to[Self.T]](w)

        fmt.FormatStruct(writer, "List").params(
            fmt.TypeNames[Self.T](),
        ).fields(write_fields)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def byte_length(self) -> Int:
        """Gets the byte length of the List (`len(self) * size_of[T]()`).

        Returns:
            The byte length of the List (`len(self) * size_of[T]()`).
        """
        return len(self) * size_of[Self.T]()

    @always_inline("nodebug")
    def capacity(self) -> Int:
        """Gets the number of elements that can fit in the list without resizing.

        Returns:
            The amount of elements that can fit in the list without resizing it.

        Examples:

        ```mojo
        var my_list: List = [1, 2, 3]
        print(my_list.capacity())  # Current allocated capacity
        ```
        """
        return self._capacity

    @no_inline
    def _realloc(
        mut self, new_capacity: Int
    ) where conforms_to(Self.T, Movable):
        var new_data = alloc(Layout[Self.T](count=new_capacity)).unsafe_leak()

        unsafe_uninit_move_n[overlapping=False](
            dest=new_data, src=self._data, count=len(self)
        )

        if self._capacity > 0:
            self._annotate_delete()
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                    Layout[Self.T](count=self._capacity)
                )
            )
        self._data = new_data
        self._capacity = new_capacity
        self._annotate_new()

    @always_inline
    def _grow_amortized(
        mut self, min_capacity: Int
    ) where conforms_to(Self.T, Movable):
        """Grows the storage to hold at least `min_capacity` elements, at least
        doubling the capacity.

        Args:
            min_capacity: The capacity the caller needs.

        Notes:
            Unlike `reserve`, which honors the requested capacity exactly, this
            never grows by less than a factor of two, so repeated
            single-element or small-batch growth stays amortized O(1) per
            element instead of O(n). Growth routines should reach for this
            rather than `reserve`. `append` is the exception: it open-codes the
            equivalent doubling to keep its hot path as small as possible.
        """
        if self._capacity >= min_capacity:
            return
        self._realloc(max(self._capacity * 2, min_capacity))

    # FIXME: This annotation is needed to support List[Span[x, o]] types with
    # mutable origins.
    @__unsafe_nested_origins_read_only
    @stable(since="1.1")
    def append(
        mut self, var value: Self.T, /
    ) where conforms_to(Self.T, Movable):
        """Appends a value to this list.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to twice the current capacity.
            Except for 0 capacity where it sets 1.

        Examples:

        ```mojo
        var list: List = [1, 2, 3, 4, 5]
        list.append(6)
        print(list) # [1, 2, 3, 4, 5, 6]
        ```
        """
        if self._len >= self._capacity:
            self._realloc(self._capacity * 2 | Int(self._capacity == 0))
        self._annotate_increase()
        # Not `_unsafe_next_uninit_ptr`: its capacity assert survives into the
        # hot path, because the `@no_inline` `_realloc` hides the invariant
        # just established above from the optimizer.
        self._data.unsafe_offset(self._len).unsafe_write(value^)
        self._len += 1

    @always_inline
    def insert(
        mut self, i: Int, var value: Self.T, /
    ) where conforms_to(Self.T, Movable):
        """Inserts a value to the list at the given index.
        `a.insert(len(a), value)` is equivalent to `a.append(value)`.

        Args:
            i: The index for the value. Must be in the range `[0, len(self)]`.
            value: The value to insert.

        Examples:

        ```mojo
        var list: List = ["one", "three"]
        list.insert(1, "two")
        print(list) # ['one', 'two', 'three']
        ```
        """
        # Valid range is `[0, len(self)]` (`len(self)` appends).
        check_bounds(i, len(self) + 1)

        var old_len = self._len
        self._grow_amortized(old_len + 1)
        self._annotate_increase()

        var data = self._data
        unsafe_uninit_move_n[overlapping=True](
            dest=data.unsafe_offset(i + 1),
            src=data.unsafe_offset(i),
            count=old_len - i,
        )
        data.unsafe_offset(i).unsafe_write(value^)
        self._len = old_len + 1

    @stable(since="1.0")
    def extend(mut self, var other: Self) where conforms_to(Self.T, Movable):
        """Extends this list by consuming the elements of `other`.

        Args:
            other: List whose elements will be added in order at the end of this
                list.

        Examples:

        ```mojo
        var list: List = ["one", "two", "three"]
        var more: List = ["four", "five"]
        list.extend(more^) # more's values are consumed
        # print(more)      # Error: use of initialized value
        print(list)        # ['one', 'two', 'three', 'four', 'five']
        ```
        """

        var other_len = len(other)
        var final_size = len(self) + other_len
        self._grow_amortized(final_size)

        var dest_ptr = self._data.unsafe_offset(self._len)
        var src_ptr = other.unsafe_ptr()
        self._annotate_increase(other_len)

        unsafe_uninit_move_n[overlapping=False](
            dest=dest_ptr, src=src_ptr, count=other_len
        )

        # Update the size now since all elements have been moved into this list.
        self._len = final_size
        # `other` only needs to deallocate its underlying buffer and not destroy
        # its elements as they were moved into `self`.
        other^._unsafe_assume_destroyed_and_deallocate()

    def extend(
        mut self, elements: Span[Self.T, _]
    ) where conforms_to(Self.T, Copyable):
        """Extend this list by copying elements from a `Span`.

        The resulting list will have the length `len(self) + len(elements)`.

        Args:
            elements: The elements to copy into this list.

        Examples:

        ```mojo
        var numbers: List = [1, 2, 3]
        var more: List = [4, 5, 6]
        numbers.extend(Span(more))
        print(numbers)   # [1, 2, 3, 4, 5, 6]
        ```
        """
        var elements_len = len(elements)
        var new_num_elts = self._len + elements_len
        self._grow_amortized(new_num_elts)

        self._annotate_increase(elements_len)
        var i = self._len
        self._len = new_num_elts

        unsafe_uninit_copy_n[overlapping=False](
            dest=self._data.unsafe_offset(i),
            src=elements.unsafe_ptr(),
            count=elements_len,
        )

    @__allow_legacy_custom_self_type
    def extend[
        dtype: DType, //
    ](mut self: List[Scalar[dtype]], value: SIMD[dtype, _]):
        """Extends this list with the elements of a vector.

        Parameters:
            dtype: The DType.

        Args:
            value: The value to append.

        Notes:
            If there is no capacity left, resizes to `len(self) + value.length`.

        Examples:

        ```mojo
        from std.collections import List

        var numbers: List[Int64] = [1, 2]
        var more = SIMD[.int64, 2](3, 4)
        numbers.extend(more)
        print(numbers) # [SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2),
                       #  SIMD[DType.int64, 1](3), SIMD[DType.int64, 1](4)]
        ```
        """
        self._grow_amortized(self._len + value.length)
        self._annotate_increase(value.length)
        self._unsafe_next_uninit_ptr().unsafe_store(value)
        self._len += value.length

    @__allow_legacy_custom_self_type
    def extend[
        dtype: DType, //
    ](mut self: List[Scalar[dtype]], value: SIMD[dtype, _], *, count: Int):
        """Extends this list with `count` number of elements from a vector.

        Parameters:
            dtype: The DType.

        Args:
            value: The value to append.
            count: The amount of items to append. Must be less than or equal to
                `value.length`.

        Notes:
            If there is no capacity left, resizes to `len(self) + count`.

        Examples:

        ```mojo
        from std.collections import List

        var numbers: List[Int64] = [1, 2]
        var more = SIMD[.int64, 4](3, 4, 5, 6)
        numbers.extend(more, count=2)
        print(numbers) # [SIMD[DType.int64, 1](1), SIMD[DType.int64, 1](2),
                       #  SIMD[DType.int64, 1](3), SIMD[DType.int64, 1](4)]
        ```
        """
        assert count <= value.length, "count must be <= value.length"
        self._grow_amortized(self._len + count)
        self._annotate_increase(count)
        var v_ptr = Pointer(to=value).unsafe_bitcast[Scalar[dtype]]()
        unsafe_memcpy(
            dest=self._unsafe_next_uninit_ptr(), src=v_ptr, count=count
        )
        self._len += count

    def pop(mut self) -> Self.T where conforms_to(Self.T, Movable):
        """Pops the last value from the list.

        Returns:
            The popped value.

        Examples:

        ```mojo
        var numbers: List = ["1", "2", "3", "4", "5"]
        var value = numbers.pop(); print(value)   # 5
        print("length", len(numbers))             # length 4
        ```
        """
        return self.pop(len(self) - 1)

    @always_inline
    def pop(mut self, i: Int) -> Self.T where conforms_to(Self.T, Movable):
        """Pops a value from the list at the given index.

        Args:
            i: The index of the value to pop.

        Returns:
            The popped value.

        Examples:

        ```mojo
        var numbers: List = ["1", "2", "3", "4", "5"]
        var value = numbers.pop(2); print(value)  # 3
        print(numbers)                            # ['1', '2', '4', '5']
        ```
        """
        check_bounds(i, len(self))
        var ret_val = self._data.unsafe_offset(i).unsafe_take_pointee()
        unsafe_uninit_move_n[overlapping=True](
            dest=self._data.unsafe_offset(i),
            src=self._data.unsafe_offset(i + 1),
            count=len(self) - i - 1,
        )
        self._len -= 1
        self._annotate_shrink(self._len + 1)
        return ret_val^

    @stable(since="1.0")
    def reserve(mut self, capacity: Int) where conforms_to(Self.T, Movable):
        """Reserves the requested capacity.

        Args:
            capacity: The new capacity.

        Notes:
            If the current capacity is greater or equal, this is a no-op.
            Otherwise, the storage is reallocated and the date is moved.
        """
        if self._capacity >= capacity:
            return
        self._realloc(capacity)

    @stable(since="1.0")
    def resize(
        mut self, length: Int, fill: Self.T
    ) where conforms_to(Self.T, Copyable & Deinitable):
        """Resizes the list to the given new length.

        Args:
            length: The new length.
            fill: The value to use to populate new elements.

        Notes:
            If the new length is smaller than the current one, elements at the end
            are discarded. If the new length is larger than the current one, the
            list is appended with new values elements up to the requested length.

        Examples:

        ```mojo
        var list: List = ["z", "y", "x", "w"]
        list.resize(3, "v")
        print(list)                  # ['z', 'y', 'x']
        list.resize(6, "v")
        print(list)                  # ['z', 'y', 'x', 'v', 'v', 'v']
        ```
        """
        if length <= self._len:
            self.shrink(length)
        else:
            self._unchecked_grow(length, fill)

    def _unchecked_grow(
        mut self, new_length: Int, fill: Self.T
    ) where conforms_to(Self.T, Copyable):
        assert new_length >= self._len

        self._grow_amortized(new_length)
        self._annotate_increase(new_length - self._len)
        for i in range(self._len, new_length):
            self._data.unsafe_offset(i).unsafe_write(copy=fill)
        self._len = new_length

    def resize(
        mut self, *, unsafe_uninit_length: Int
    ) where conforms_to(Self.T, Deinitable & Movable):
        """Resizes the list to the given new size leaving any new elements
        uninitialized.

        If the new size is smaller than the current one, elements at the end
        are discarded. If the new size is larger than the current one, the
        list is extended and the new elements are left uninitialized.

        Args:
            unsafe_uninit_length: The new size.

        Examples:

        ```mojo
        var list: List = [1, 2, 3]
        list.resize(unsafe_uninit_length=5) # Indices 3 and 4 are uninitialized memory
        print(len(list))                    # 5
        list[3] = 10; list[4] = 20
        print(list)                         # [1, 2, 3, 10, 20]
        ```
        """
        if unsafe_uninit_length <= self._len:
            self.shrink(unsafe_uninit_length)
        else:
            self._grow_amortized(unsafe_uninit_length)
            self._annotate_increase(unsafe_uninit_length - self._len)
            self._len = unsafe_uninit_length

    def shrink(mut self, new_length: Int) where conforms_to(Self.T, Deinitable):
        """Resizes to the given new length which must be <= the current size.

        Args:
            new_length: The new length.

        Notes:
            With no new value provided, the new length must be smaller than or
            equal to the current one. Elements at the end are discarded.

            Calls abort() if the new length is larger than the current length.

        Examples:

        ```mojo
        var numbers: List = [1, 2, 3, 4, 5, 6]
        numbers.shrink(2); print(numbers) # [1, 2]
        # numbers.shrink(8)               # Error: new size is bigger than current
        ```
        """
        if len(self) < new_length:
            abort(
                "You are calling List.shrink with a new_size bigger than the"
                " current size. If you want to make the List bigger, provide a"
                " value to fill the new slots with. If not, make sure the new"
                " size is smaller than the current size."
            )

        unsafe_destroy_n(
            self._data.unsafe_offset(new_length),
            count=len(self) - new_length,
        )

        var old_length: Int = self._len
        self._len = new_length
        self._annotate_shrink(old_length)

    def reverse(mut self) where conforms_to(Self.T, Movable):
        """Reverses the elements of the list.

        Examples:

        ```mojo
        var list: List = ["o", "l", "l", "e", "H"]
        list.reverse()
        print("".join(list)) # Hello
        ```
        """

        var earlier_idx = 0
        var later_idx = len(self) - 1

        var effective_len = len(self)
        var half_len = effective_len // 2

        for _ in range(half_len):
            var earlier_ptr = self._data.unsafe_offset(earlier_idx)
            var later_ptr = self._data.unsafe_offset(later_idx)

            var tmp = earlier_ptr.unsafe_take_pointee()
            earlier_ptr.unsafe_write_move_from(later_ptr)
            later_ptr.unsafe_write(tmp^)

            earlier_idx += 1
            later_idx -= 1

    def try_index(
        ref self,
        value: Self.T,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) -> Optional[Int] where conforms_to(Self.T, Equatable):
        """Returns the index of the first occurrence of a value in a list
        restricted by the range given the start and stop bounds.

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to None, which means the end of the list).

        Returns:
            The index of the first occurrence of the value in the list or `None` if the value is not found.

        Examples:

        ```mojo
        var my_list: List = [1, 2, 3]
        print(my_list.index(2)) # prints `1`
        ```
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
        return None

    def index(
        ref self,
        value: Self.T,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) raises -> Int where conforms_to(Self.T, Equatable):
        """Returns the index of the first occurrence of a value in a list
        restricted by the range given the start and stop bounds.

        Args:
            value: The value to search for.
            start: The starting index of the search, treated as a slice index
                (defaults to 0).
            stop: The ending index of the search, treated as a slice index
                (defaults to None, which means the end of the list).

        Returns:
            The index of the first occurrence of the value in the list.

        Raises:
            ValueError: If the value is not found in the list.

        Examples:

        ```mojo
        var my_list: List = [1, 2, 3]
        print(my_list.index(2)) # prints `1`
        ```
        """
        var result = self.try_index(value, start=start, stop=stop)
        if result is None:
            raise "ValueError: Given element is not in list"
        return result.value()

    def clear(
        mut self,
    ) where conforms_to(Self.T, Deinitable):
        """Clears the elements in the list.

        Examples:

        ```mojo
        var list: List = ["o", "l", "l", "e", "H"]
        print(len(list))  # 5
        list.clear()
        print(len(list))  # 0
        ```
        """
        unsafe_destroy_n(self._data, count=self._len)
        var old_size: Int = self._len
        self._len = 0
        self._annotate_shrink(old_size)

    def unsafe_take_allocation(mut self) -> Allocation[Self.T]:
        """Take ownership of the underlying storage from the list.

        Returns:
            The `Allocation` that owns the list's backing storage, carrying the
            `Layout` it was allocated with.

        Safety:

        The list's elements are handed over still initialized, and deallocating
        the storage does not run their destructors. Destroy any initialized
        elements yourself (for example with `unsafe_destroy_n`) before
        deallocating if `T` needs it.

        A list that never allocated has no storage to hand over, so the
        returned `Allocation` wraps a dangling pointer and a `Layout` with a
        `count` of zero. Check `layout().count()` before deallocating it.

        Examples:

        ```mojo
        from std.collections import List
        from std.memory.alloc import dealloc

        var list: List[Int64] = [1, 2, 3, 4]
        var allocation = list.unsafe_take_allocation() # list is now empty
        var ptr = allocation.unsafe_ptr()
        for idx in range(4):
            print(ptr[unsafe_offset=idx], end=" ")
        print() # Output: 1 2 3 4
        # Free the storage.
        dealloc(allocation^)
        ```
        """
        self._annotate_delete()
        var layout = Layout[Self.T](count=self._capacity)
        var ptr = self._data
        self._data = Self._PointerType.unsafe_dangling()
        self._len = 0
        self._capacity = 0
        return ThinAllocation(unsafe_owned_ptr=ptr).unsafe_with_layout(layout)

    def __getitem__(
        self, slice: StridedSlice
    ) -> Self where conforms_to(Self.T, Copyable):
        """Gets the sequence of elements at the specified positions.

        Args:
            slice: A slice that specifies positions of the new list.

        Returns:
            A new list containing the list at the specified slice.
        """
        var start, end, step = slice.indices(len(self))
        var r = range(start, end, step)

        if not len(r):
            return Self()

        return Self(
            length=len(r),
            fill_with=lambda (idx: Int) -> Self.T: self[
                start + idx * step
            ].copy(),
        )

    @__unsafe_nested_origins_read_only
    @stable(since="1.0")
    @always_inline
    def __getitem__(
        ref self, slice: ContiguousSlice
    ) -> Span[Self.T, Self._InteriorOrigin[origin_of(self)]]:
        """Gets the sequence of elements at the specified positions.

        Aborts if `slice`'s start or end index is out of bounds (valid range
        is `0` to `len(self)`, inclusive), or if start is greater than end.
        Negative indices are not supported and always abort.

        Args:
            slice: A slice the specifies the positions of the new list.

        Returns:
            A span over the specified slice.
        """
        var start, end = check_slice_bounds(slice, len(self))
        return {
            unsafe_ptr = self._unsafe_interior_ptr().unsafe_offset(start),
            length = end - start,
        }

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, idx: IntLiteral, /
    ) -> ref[Self._InteriorOrigin[origin_of(self)]] Self.T:
        """Gets the list element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        comptime assert (
            IntLiteral[idx.value]() >= 0
        ), "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        check_bounds(idx, len(self))
        return self.unsafe_get(index(idx))

    @stable(since="1.0")
    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, idx: Int, /
    ) -> ref[Self._InteriorOrigin[origin_of(self)]] Self.T:
        """Gets the list element at the given index.

        Unlike when subscripting using slices negative indices are
        considered out of bounds. They will be checked in the same situations
        as "off the end" indexing.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        check_bounds(idx, len(self))
        return self.unsafe_get(index(idx))

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, idx: Some[Indexer]
    ) -> ref[Self._InteriorOrigin[origin_of(self)]] Self.T:
        """Gets the list element at the given index.

        Unlike when subscripting using slices negative indices are
        considered out of bounds. They will be checked in the same situations
        as "off the end" indexing.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        check_bounds(idx, len(self))
        return self.unsafe_get(index(idx))

    @__unsafe_nested_origins_read_only
    @always_inline
    def unsafe_get(
        ref self, idx: Int
    ) -> ref[Self._InteriorOrigin[origin_of(self)]] Self.T:
        """Get a reference to an element of self without checking index bounds.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the given index.

        Notes:
            Users should consider using `__getitem__` instead of this method as
            it is unsafe. If an index is out of bounds, this method will not
            abort, it will be considered undefined behavior.

            Note that there is no wraparound for negative indices, caution is
            advised. Using negative indices is considered undefined behavior.
            Never use `my_list.unsafe_get(-1)` to get the last element of the
            list. Instead, do `my_list.unsafe_get(len(my_list) - 1)`.
        """
        check_bounds[cpu_default=False](idx, len(self))
        return self._unsafe_interior_ptr().unsafe_offset(idx)[]

    @always_inline
    def unsafe_set(
        mut self, idx: Int, var value: Self.T
    ) where conforms_to(Self.T, Deinitable & Movable):
        """Write a value to a given location without checking index bounds.

        Args:
            idx: The index of the element to set.
            value: The value to set.

        Notes:
            Users should consider using `my_list[idx] = value` instead of this
            method as it is unsafe. If an index is out of bounds, this method
            will not abort, it will be considered undefined behavior.

            Note that there is no wraparound for negative indices, caution is
            advised. Using negative indices is considered undefined behavior.
            Never use `my_list.unsafe_set(-1, value)` to set the last element of
            the list. Instead, do `my_list.unsafe_set(len(my_list) - 1, value)`.
        """
        check_bounds[cpu_default=False](idx, len(self))
        var ptr = self._data.unsafe_offset(idx)
        ptr.unsafe_deinit_pointee()
        ptr.unsafe_write(value^)

    def count(self, value: Self.T) -> Int where conforms_to(Self.T, Equatable):
        """Counts the number of occurrences of a value in the list.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the list.

        Examples:

        ```mojo
        var list: List = ["a", "b", "c", "b", "b", "a", "c"]
        print(list.count("b")) # 3
        ```
        """
        var count = 0
        for elem in self:
            if elem == value:
                count += 1
        return count

    def swap_elements(
        mut self, elt_idx_1: Int, elt_idx_2: Int
    ) where conforms_to(Self.T, Movable):
        """Swaps elements at the specified indexes if they are different.

        Args:
            elt_idx_1: The index of one element.
            elt_idx_2: The index of the other element.

        Examples:

        ```mojo
        var my_list: List = [1, 2, 3]
        my_list.swap_elements(0, 2)
        print(my_list) # 3, 2, 1
        ```

        Notes:
            This is useful because `swap(my_list[i], my_list[j])` cannot be
            supported by Mojo, because a mutable alias may be formed.
        """
        assert 0 <= elt_idx_1 < len(self) and 0 <= elt_idx_2 < len(self), (
            "The indices provided to swap_elements must be within the range"
            " [0, len(List)-1]"
        )
        var ptr = self._data
        ptr.unsafe_offset(elt_idx_1).swap_pointees(ptr.unsafe_offset(elt_idx_2))

    @always_inline
    def _unsafe_interior_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> Pointer[
        Self.T, Self._InteriorOrigin[origin], address_space=address_space
    ]:
        return Pointer(
            to=self.unsafe_ptr()._get_ref_with_unsafe_interior_origin[
                "element", origin
            ]()
        )

    def unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> Pointer[
        Self.T, origin, address_space=address_space
    ]:
        """Retrieves a pointer to the underlying memory, or a dangling pointer
        if the `List` has not yet allocated.

        You should use the `len` of this `List` to determine if the pointer
        is valid for reads and writes.

        Parameters:
            origin: The origin of the `List`.
            address_space: The `AddressSpace` of the `List`.

        Returns:
            The pointer to the underlying memory.
        """
        return (
            self._data.unsafe_mut_cast[origin.mut]()
            .unsafe_origin_cast[origin]()
            .unsafe_address_space_cast[address_space]()
        )

    @always_inline
    def _unsafe_next_uninit_ptr(
        ref self,
    ) -> Pointer[Self.T, origin_of(self)]:
        """Retrieves a pointer to the next uninitialized element position.

        Safety:

        - This pointer MUST not be used to read or write memory beyond the
        allocated capacity of this list.
        - This pointer may not be used to initialize non-contiguous elements.
        - Ensure that `List._len` is updated to reflect the new number of
          initialized elements, otherwise elements may be unexpectedly
          overwritten or not destroyed correctly.

        Notes:
            This returns a pointer that points to the element position immediately
            after the last initialized element. This is equivalent to
            `list.unsafe_ptr() + len(list)`.
        """
        assert self._capacity > 0 and self._capacity > self._len, (
            "safety violation: Insufficient capacity to retrieve pointer to"
            " next uninitialized element"
        )

        var length = len(self)
        return self.unsafe_ptr().unsafe_offset(length)


def _clip(value: Int, start: Int, end: Int) -> Int:
    return max(start, min(value, end))
