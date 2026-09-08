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
"""Defines the Deque type.

You can import these APIs from the `collections` package.

Examples:

```mojo
from std.collections import Deque
```
"""


from std.bit import next_power_of_two
from std.builtin.rebind import downcast
import std.format._utils as fmt
from std.hashlib import Hasher
from std.collections import check_bounds
from std.memory.alloc import alloc, dealloc, ThinAllocation, Layout

# ===-----------------------------------------------------------------------===#
# Deque
# ===-----------------------------------------------------------------------===#


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `Deque` of"
    " non-`Deinitable` elements"
)
struct Deque[ElementType: Movable](
    Boolable,
    Copyable where conforms_to(ElementType, Copyable),
    Deinitable where conforms_to(ElementType, Deinitable),
    Equatable where conforms_to(ElementType, Equatable),
    Hashable where conforms_to(ElementType, Hashable),
    Iterable,
    IterableOwned where conforms_to(ElementType, Deinitable),
    Movable,
    Sized,
    Writable where conforms_to(ElementType, Writable),
):
    """Implements a double-ended queue.

    It supports pushing and popping from both ends in O(1) time resizing the
    underlying storage as needed.

    Parameters:
        ElementType: The type of the elements in the deque. Must implement
            `Movable`. A `Deque` is implicitly destructible only when
            `ElementType` is `Deinitable`; otherwise drain it with
            `deinit_with()`.
    """

    # The by-ref iterator still requires `Copyable & Deinitable` (see
    # the `TODO(MSTDL-2390)`s below), while the owned iterator only moves
    # elements out and so requires just `Deinitable`. The `downcast`s
    # restate those bounds explicitly now that `ElementType`'s bound no longer
    # implies them.
    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DequeIter[
        downcast[Self.ElementType, Copyable & Deinitable],
        iterable_origin,
    ]
    """The iterator type for this deque.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.ElementType, Deinitable
    ) = _DequeIterOwned[Self.ElementType]
    """The owned iterator type for this deque."""

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime default_capacity: Int = 64
    """The default capacity of the deque: must be the power of 2."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _data: Pointer[Self.ElementType, MutUntrackedOrigin]
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

    def __init__(
        out self,
        *,
        var elements: Optional[List[Self.ElementType]] = None,
        capacity: Int = Self.default_capacity,
        min_capacity: Int = Self.default_capacity,
        maxlen: Int = -1,
        shrink: Bool = True,
    ) where conforms_to(Self.ElementType, Deinitable):
        """Constructs a deque.

        Args:
            elements: The optional list of initial deque elements.
            capacity: The initial capacity of the deque.
            min_capacity: The minimum allowed capacity of the deque when shrinking.
            maxlen: The maximum allowed capacity of the deque when growing.
            shrink: Should storage be de-allocated when not needed.

        Constraints:
            `ElementType` must be `Deinitable`. A `maxlen`-bounded
            deque can destroy evicted elements, and the `elements` overflow is
            destroyed during the initial fill. To build a deque of
            non-`Deinitable` elements, use the variadic constructor.
        """
        var deque_capacity: Int
        if capacity <= 0:
            deque_capacity = self.default_capacity
        else:
            deque_capacity = next_power_of_two(capacity)

        var min_deque_capacity: Int
        if min_capacity <= 0:
            min_deque_capacity = self.default_capacity
        else:
            min_deque_capacity = next_power_of_two(min_capacity)

        var max_deque_len: Int
        if maxlen <= 0:
            max_deque_len = -1
        else:
            max_deque_len = maxlen
            var max_deque_capacity = next_power_of_two(maxlen)
            if max_deque_capacity == maxlen:
                max_deque_capacity <<= 1
            deque_capacity = min(deque_capacity, max_deque_capacity)

        self._capacity = deque_capacity
        self._data = alloc(
            Layout[Self.ElementType](count=deque_capacity)
        ).unsafe_leak()
        self._head = 0
        self._tail = 0
        self._min_capacity = min_deque_capacity
        self._maxlen = max_deque_len
        self._shrink = shrink

        if elements is not None:
            self.extend(elements.take())

    def __init__(
        out self,
        var *values: Self.ElementType,
        __list_literal__: NoneType = None,
    ):
        """Constructs a deque from the given values.

        Args:
            values: The values to populate the deque with.
            __list_literal__: Tell Mojo to use this method for list literals.
        """
        var args_length = len(values)

        var capacity: Int
        if args_length < self.default_capacity:
            capacity = self.default_capacity
        else:
            capacity = args_length

        # Initialize storage directly (rather than delegating to the
        # keyword constructor, which requires `Deinitable`) so the
        # variadic constructor works for any `Movable` element type.
        var deque_capacity = next_power_of_two(capacity)
        self._capacity = deque_capacity
        self._data = alloc(
            Layout[Self.ElementType](count=deque_capacity)
        ).unsafe_leak()
        self._head = 0
        self._tail = 0
        self._min_capacity = Self.default_capacity
        self._maxlen = -1
        self._shrink = True

        # Transfer all of the values into the deque.
        def init_elt(idx: Int, var elt: Self.ElementType) {ref}:
            (self._data.unsafe_offset(idx)).unsafe_write(elt^)

        values^.consume_elements(init_elt)

        # Remember how many values we have.
        self._tail = args_length

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.ElementType, Copyable):
        """Creates a deep copy of the given deque.

        Args:
            copy: The deque to copy.
        """
        # Initialize storage directly (rather than delegating to the keyword
        # constructor, which requires `Deinitable`) so copying works
        # for any `Copyable` element type. Copying only ever creates elements;
        # it never evicts or otherwise destroys one, so it does not require
        # `Deinitable`. `copy`'s capacities are already powers of two,
        # so no renormalization is needed.
        self._capacity = copy._capacity
        self._data = alloc(
            Layout[Self.ElementType](count=copy._capacity)
        ).unsafe_leak()
        self._head = 0
        self._tail = 0
        self._min_capacity = copy._min_capacity
        self._maxlen = copy._maxlen
        self._shrink = copy._shrink

        for i in range(len(copy)):
            var offset = copy._physical_index(copy._head + i)
            (self._data.unsafe_offset(i)).unsafe_write(
                copy=(copy._data.unsafe_offset(offset))[]
            )

        self._tail = len(copy)

    def _unsafe_assume_destroyed_and_deallocate(deinit self):
        """Assumes self's elements are already destroyed and deallocates the
        backing storage.
        """
        dealloc(
            ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                {count = self._capacity}
            )
        )

    def __deinit__(
        deinit self,
    ) where conforms_to(Self.ElementType, Deinitable):
        """Destroys all elements in the deque and frees its memory."""
        for i in range(len(self)):
            var offset = self._physical_index(self._head + i)
            (self._data.unsafe_offset(offset)).unsafe_deinit_pointee()
        self^._unsafe_assume_destroyed_and_deallocate()

    def deinit_with(
        deinit self, deinit_func: Some[def(var Self.ElementType)], /
    ):
        """Consumes this deque and deinitializes its elements using the provided
        closure.

        This can be used to destroy a `Deque` of non-`Deinitable`
        values.

        Args:
            deinit_func: The deinitializing closure called on each `Deque`
                element.
        """
        for i in range(len(self)):
            var offset = self._physical_index(self._head + i)
            deinit_func(
                __get_address_as_owned_value(
                    (self._data.unsafe_offset(offset))._get_kgen_pointer()
                )
            )
        self^._unsafe_assume_destroyed_and_deallocate()

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __add__(
        self, other: Self
    ) -> Self where conforms_to(Self.ElementType, Copyable & Deinitable):
        """Concatenates self with other and returns the result as a new deque.

        Args:
            other: Deque whose elements will be appended to the elements of self.

        Returns:
            The newly created deque with the properties of `self`.
        """
        var new = self.copy()
        for element in other:
            new.append(element.copy())
        return new^

    def __iadd__(
        mut self, other: Self
    ) where conforms_to(Self.ElementType, Copyable & Deinitable):
        """Appends the elements of other deque into self.

        Args:
            other: Deque whose elements will be appended to self.
        """
        for element in other:
            self.append(element.copy())

    def __mul__(
        self, n: Int
    ) -> Self where conforms_to(Self.ElementType, Copyable & Deinitable):
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
        var new = self.copy()
        for _ in range(n - 1):
            for element in self:
                new.append(element.copy())
        return new^

    def __imul__(
        mut self, n: Int
    ) where conforms_to(Self.ElementType, Copyable & Deinitable):
        """Concatenates self `n` times in place.

        Args:
            n: The multiplier number.
        """
        if n <= 0:
            self.clear()
            return

        var orig = self.copy()
        for _ in range(n - 1):
            for element in orig:
                self.append(element.copy())

    def __eq__(
        self, other: Self
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Checks if two deques are equal.

        Args:
            other: The deque to compare with.

        Returns:
            `True` if the deques are equal, `False` otherwise.
        """
        if len(self) != len(other):
            return False

        for i in range(len(self)):
            var offset_self = self._physical_index(self._head + i)
            var offset_other = other._physical_index(other._head + i)
            ref lhs = (self._data.unsafe_offset(offset_self))[]
            ref rhs = (other._data.unsafe_offset(offset_other))[]
            if lhs != rhs:
                return False
        return True

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.ElementType, Hashable):
        """Updates hasher with the hash of each element in the deque.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        for i in range(len(self)):
            var offset = self._physical_index(self._head + i)
            (self._data.unsafe_offset(offset))[].__hash__(hasher)

    def __contains__(
        self, value: Self.ElementType
    ) -> Bool where conforms_to(Self.ElementType, Equatable):
        """Verify if a given value is present in the deque.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the deque, False otherwise.
        """
        for i in range(len(self)):
            var offset = self._physical_index(self._head + i)
            if (self._data.unsafe_offset(offset))[] == value:
                return True
        return False

    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(Self.ElementType, Deinitable):
        """Consume the deque and return an iterator over its elements.

        Returns:
            An iterator that owns the deque's elements.
        """
        return {self^, 0}

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)]:
        """Iterates over elements of the deque, returning the references.

        Returns:
            An iterator of the references to the deque elements.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.ElementType, Copyable & Deinitable
        ), "Deque iteration requires the element to be `Copyable & Deinitable`."
        return _DequeIter(
            0,
            rebind[
                Pointer[
                    Deque[downcast[Self.ElementType, Copyable & Deinitable]],
                    origin_of(self),
                ]
            ](Pointer(to=self)),
        )

    def __reversed__(
        ref self,
    ) -> _DequeIter[
        Self.ElementType,
        origin_of(self),
        False,
    ] where conforms_to(Self.ElementType, Copyable & Deinitable):
        """Iterate backwards over the deque, returning the references.

        Returns:
            A reversed iterator of the references to the deque elements.
        """
        return _DequeIter[forward=False](
            len(self),
            Pointer(to=self),
        )

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __bool__(self) -> Bool:
        """Checks whether the deque has any elements or not.

        Returns:
            `False` if the deque is empty, `True` if there is at least one element.
        """
        return self._head != self._tail

    @always_inline
    def __len__(self) -> Int:
        """Gets the number of elements in the deque.

        Returns:
            The number of elements in the deque.
        """
        return (self._tail - self._head) & (self._capacity - 1)

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, idx: IntLiteral
    ) -> ref[self._unchecked_get(idx)] Self.ElementType:
        """Gets the deque element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        comptime assert (
            IntLiteral[idx.value]() >= 0
        ), "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        check_bounds(idx, len(self))
        return self._unchecked_get(idx)

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, idx: Int
    ) -> ref[self._unchecked_get(idx)] Self.ElementType:
        """Gets the deque element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            A reference to the element at the given index.
        """
        check_bounds(idx, len(self))
        return self._unchecked_get(idx)

    @__unsafe_nested_origins_read_only
    @always_inline
    def _unchecked_get(
        ref self, idx: Int
    ) -> ref[origin_of(self)._get_owned_interior["element"]] Self.ElementType:
        var offset = self._physical_index(self._head + idx)
        return self._data.unsafe_offset(
            offset
        )._get_ref_with_unsafe_interior_origin["element", origin_of(self)]()

    def _write_self_to[
        f: def(Self.ElementType, mut Some[Writer]) thin
    ](self, mut writer: Some[Writer]) where conforms_to(
        Self.ElementType, Writable
    ):
        var iterator = self.__iter__()

        def iterate(mut w: Some[Writer]) raises StopIteration {mut iterator}:
            f(iterator.__next__(), w)

        fmt.write_sequence_to(writer, iterate)
        _ = iterator^

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.ElementType, Writable):
        """Writes `my_deque.__str__()` to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f=fmt.write_to[Self.ElementType]](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.ElementType, Writable):
        """Writes the repr representation of this deque to a `Writer`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_self_to[f=fmt.write_repr_to[Self.ElementType]](w)

        fmt.FormatStruct(writer, "Deque").params(
            fmt.TypeNames[Self.ElementType](),
        ).fields(write_fields)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def append(
        mut self, var value: Self.ElementType
    ) where conforms_to(Self.ElementType, Deinitable):
        """Appends a value to the right side of the deque.

        Args:
            value: The value to append.

        Constraints:
            `ElementType` must be `Deinitable`, because a bounded
            (`maxlen`) deque destroys the evicted element.
        """
        # checking for positive _maxlen first is important for speed
        if self._maxlen > 0 and len(self) == self._maxlen:
            (self._data.unsafe_offset(self._head)).unsafe_deinit_pointee()
            self._head = self._physical_index(self._head + 1)

        (self._data.unsafe_offset(self._tail)).unsafe_write(value^)
        self._tail = self._physical_index(self._tail + 1)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    def appendleft(
        mut self, var value: Self.ElementType
    ) where conforms_to(Self.ElementType, Deinitable):
        """Appends a value to the left side of the deque.

        Args:
            value: The value to append.

        Constraints:
            `ElementType` must be `Deinitable`, because a bounded
            (`maxlen`) deque destroys the evicted element.
        """
        # checking for positive _maxlen first is important for speed
        if self._maxlen > 0 and len(self) == self._maxlen:
            self._tail = self._physical_index(self._tail - 1)
            (self._data.unsafe_offset(self._tail)).unsafe_deinit_pointee()

        self._head = self._physical_index(self._head - 1)
        (self._data.unsafe_offset(self._head)).unsafe_write(value^)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    def clear(
        mut self,
    ) where conforms_to(Self.ElementType, Deinitable):
        """Removes all elements from the deque leaving it with length 0.

        Resets the underlying storage capacity to `_min_capacity`.
        """
        for i in range(len(self)):
            var offset = self._physical_index(self._head + i)
            (self._data.unsafe_offset(offset)).unsafe_deinit_pointee()
        dealloc(
            ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                {count = self._capacity}
            )
        )
        self._capacity = self._min_capacity
        self._data = alloc(
            Layout[Self.ElementType](count=self._capacity)
        ).unsafe_leak()
        self._head = 0
        self._tail = 0

    def count(
        self, value: Self.ElementType
    ) -> Int where conforms_to(Self.ElementType, Equatable):
        """Counts the number of occurrences of a `value` in the deque.

        Args:
            value: The value to count.

        Returns:
            The number of occurrences of the value in the deque.
        """
        var count = 0
        for i in range(len(self)):
            var offset = self._physical_index(self._head + i)
            if (self._data.unsafe_offset(offset))[] == value:
                count += 1
        return count

    def extend(
        mut self, var values: List[Self.ElementType]
    ) where conforms_to(Self.ElementType, Deinitable):
        """Extends the right side of the deque by consuming elements of the list argument.

        Args:
            values: List whose elements will be added at the right side of the deque.
        """
        var n_move_total, n_move_self, n_move_values, n_pop_self, n_pop_values = self._compute_pop_and_move_counts(
            len(self), len(values)
        )

        # pop excess `self` elements
        for _ in range(n_pop_self):
            (self._data.unsafe_offset(self._head)).unsafe_deinit_pointee()
            self._head = self._physical_index(self._head + 1)

        # move from `self` to new location if we have to re-allocate
        if n_move_total >= self._capacity:
            self._prepare_for_new_elements(n_move_total, n_move_self)

        # we will consume all elements of `values`
        var values_alloc = values.unsafe_take_allocation()
        var values_data = values_alloc.unsafe_ptr()

        # pop excess elements from `values`
        for i in range(n_pop_values):
            (values_data.unsafe_offset(i)).unsafe_deinit_pointee()

        # move remaining elements from `values`
        var src = values_data.unsafe_offset(n_pop_values)
        for i in range(n_move_values):
            (self._data.unsafe_offset(self._tail)).unsafe_write_move_from(
                src.unsafe_offset(i)
            )
            self._tail = self._physical_index(self._tail + 1)

        # free the list backing buffer
        dealloc(values_alloc^)

    def extendleft(
        mut self, var values: List[Self.ElementType]
    ) where conforms_to(Self.ElementType, Deinitable):
        """Extends the left side of the deque by consuming elements from the list argument.

        Acts as series of left appends resulting in reversed order of elements in the list argument.

        Args:
            values: List whose elements will be added at the left side of the deque.
        """
        var n_move_total, n_move_self, n_move_values, n_pop_self, n_pop_values = self._compute_pop_and_move_counts(
            len(self), len(values)
        )

        # pop excess `self` elements
        for _ in range(n_pop_self):
            self._tail = self._physical_index(self._tail - 1)
            (self._data.unsafe_offset(self._tail)).unsafe_deinit_pointee()

        # move from `self` to new location if we have to re-allocate
        if n_move_total >= self._capacity:
            self._prepare_for_new_elements(n_move_total, n_move_self)

        # we will consume all elements of `values`
        var values_alloc = values.unsafe_take_allocation()
        var values_data = values_alloc.unsafe_ptr()

        # pop excess elements from `values`
        for i in range(n_pop_values):
            (values_data.unsafe_offset(i)).unsafe_deinit_pointee()

        # move remaining elements from `values`
        var src = values_data.unsafe_offset(n_pop_values)
        for i in range(n_move_values):
            self._head = self._physical_index(self._head - 1)
            (self._data.unsafe_offset(self._head)).unsafe_write_move_from(
                src.unsafe_offset(i)
            )

        dealloc(values_alloc^)

    def index(
        self,
        value: Self.ElementType,
        start: Int = 0,
        stop: Optional[Int] = None,
    ) raises -> Int where conforms_to(Self.ElementType, Equatable):
        """Returns the index of the first occurrence of a `value` in a deque
        restricted by the range given the `start` and `stop` bounds.

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
        var start_normalized = start

        var stop_normalized = len(self) if stop is None else stop.value()

        if start_normalized < 0:
            start_normalized += len(self)
        if stop_normalized < 0:
            stop_normalized += len(self)

        start_normalized = max(min(start_normalized, len(self)), 0)
        stop_normalized = max(min(stop_normalized, len(self)), 0)

        for idx in range(start_normalized, stop_normalized):
            var offset = self._physical_index(self._head + idx)
            if (self._data.unsafe_offset(offset))[] == value:
                return idx
        raise "ValueError: Given element is not in deque"

    @always_inline
    def insert(
        mut self, idx: Int, var value: Self.ElementType
    ) raises where conforms_to(Self.ElementType, Deinitable):
        """Inserts the `value` into the deque at position `idx`.

        Args:
            idx: The position to insert the value into.
            value: The value to insert.

        Raises:
            IndexError: If deque is already at its maximum size.
        """
        var deque_len = len(self)

        if deque_len == self._maxlen:
            raise "IndexError: Deque is already at its maximum size"

        check_bounds(idx, deque_len + 1)

        if idx <= deque_len // 2:
            for i in range(idx):
                var src = self._physical_index(self._head + i)
                var dst = self._physical_index(src - 1)
                (self._data.unsafe_offset(dst)).unsafe_write_move_from(
                    self._data.unsafe_offset(src)
                )
            self._head = self._physical_index(self._head - 1)
        else:
            for i in range(deque_len - idx):
                var dst = self._physical_index(self._tail - i)
                var src = self._physical_index(dst - 1)
                (self._data.unsafe_offset(dst)).unsafe_write_move_from(
                    self._data.unsafe_offset(src)
                )
            self._tail = self._physical_index(self._tail + 1)

        var offset = self._physical_index(self._head + idx)
        (self._data.unsafe_offset(offset)).unsafe_write(value^)

        if self._head == self._tail:
            self._realloc(self._capacity << 1)

    def remove(
        mut self, value: Self.ElementType
    ) raises where conforms_to(Self.ElementType, Equatable & Deinitable):
        """Removes the first occurrence of the `value`.

        Args:
            value: The value to remove.

        Raises:
            ValueError: If the value is not found in the deque.
        """
        var deque_len = len(self)
        for idx in range(deque_len):
            var offset = self._physical_index(self._head + idx)
            if (self._data.unsafe_offset(offset))[] == value:
                (self._data.unsafe_offset(offset)).unsafe_deinit_pointee()

                if idx < deque_len // 2:
                    for i in reversed(range(idx)):
                        var src = self._physical_index(self._head + i)
                        var dst = self._physical_index(src + 1)
                        (self._data.unsafe_offset(dst)).unsafe_write_move_from(
                            self._data.unsafe_offset(src)
                        )
                    self._head = self._physical_index(self._head + 1)
                else:
                    for i in range(idx + 1, deque_len):
                        var src = self._physical_index(self._head + i)
                        var dst = self._physical_index(src - 1)
                        (self._data.unsafe_offset(dst)).unsafe_write_move_from(
                            self._data.unsafe_offset(src)
                        )
                    self._tail = self._physical_index(self._tail - 1)

                if (
                    self._shrink
                    and self._capacity > self._min_capacity
                    and self._capacity // 4 >= len(self)
                ):
                    self._realloc(self._capacity >> 1)

                return

        raise "ValueError: Given element is not in deque"

    def peek(
        self,
    ) raises -> Self.ElementType where conforms_to(Self.ElementType, Copyable):
        """Inspect the last (rightmost) element of the deque without removing it.

        Returns:
            The last (rightmost) element of the deque.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        return (
            self._data.unsafe_offset(self._physical_index(self._tail - 1))
        )[].copy()

    def peekleft(
        self,
    ) raises -> Self.ElementType where conforms_to(Self.ElementType, Copyable):
        """Inspect the first (leftmost) element of the deque without removing it.

        Returns:
            The first (leftmost) element of the deque.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        return (self._data.unsafe_offset(self._head))[].copy()

    def pop(mut self) raises -> Self.ElementType:
        """Removes and returns the element from the right side of the deque.

        Returns:
            The popped value.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        self._tail = self._physical_index(self._tail - 1)
        var element = (
            self._data.unsafe_offset(self._tail)
        ).unsafe_take_pointee()

        if (
            self._shrink
            and self._capacity > self._min_capacity
            and self._capacity // 4 >= len(self)
        ):
            self._realloc(self._capacity >> 1)

        return element^

    def popleft(mut self) raises -> Self.ElementType:
        """Removes and returns the element from the left side of the deque.

        Returns:
            The popped value.

        Raises:
            IndexError: If the deque is empty.
        """
        if self._head == self._tail:
            raise "IndexError: Deque is empty"

        var element = (
            self._data.unsafe_offset(self._head)
        ).unsafe_take_pointee()
        self._head = self._physical_index(self._head + 1)

        if (
            self._shrink
            and self._capacity > self._min_capacity
            and self._capacity // 4 >= len(self)
        ):
            self._realloc(self._capacity >> 1)

        return element^

    def reverse(mut self):
        """Reverses the elements of the deque in-place."""
        var last = self._head + len(self) - 1
        for i in range(len(self) // 2):
            var src = self._physical_index(self._head + i)
            var dst = self._physical_index(last - i)
            var tmp = (self._data.unsafe_offset(dst)).unsafe_take_pointee()
            (self._data.unsafe_offset(dst)).unsafe_write_move_from(
                self._data.unsafe_offset(src)
            )
            (self._data.unsafe_offset(src)).unsafe_write(tmp^)

    def rotate(mut self, n: Int = 1):
        """Rotates the deque by `n` steps.

        If `n` is positive, rotates to the right.
        If `n` is negative, rotates to the left.

        Args:
            n: Number of steps to rotate the deque
                (defaults to 1).
        """
        if n < 0:
            for _ in range(-n):
                (self._data.unsafe_offset(self._tail)).unsafe_write_move_from(
                    self._data.unsafe_offset(self._head)
                )
                self._tail = self._physical_index(self._tail + 1)
                self._head = self._physical_index(self._head + 1)
        else:
            for _ in range(n):
                self._tail = self._physical_index(self._tail - 1)
                self._head = self._physical_index(self._head - 1)
                (self._data.unsafe_offset(self._head)).unsafe_write_move_from(
                    self._data.unsafe_offset(self._tail)
                )

    def _compute_pop_and_move_counts(
        self, len_self: Int, len_values: Int
    ) -> Tuple[Int, Int, Int, Int, Int]:
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
        var len_total = len_self + len_values

        var n_move_total = (
            min(len_total, self._maxlen) if self._maxlen > 0 else len_total
        )
        var n_move_values = min(len_values, n_move_total)
        var n_move_self = n_move_total - n_move_values

        var n_pop_self = len_self - n_move_self
        var n_pop_values = len_values - n_move_values

        return (
            n_move_total,
            n_move_self,
            n_move_values,
            n_pop_self,
            n_pop_values,
        )

    @always_inline
    def _physical_index(self, logical_index: Int) -> Int:
        """Calculates the physical index in the circular buffer.

        Args:
            logical_index: The logical index, which may fall outside the physical bounds
                of the buffer and needs to be wrapped around.

        The size of the underlying buffer is always a power of two, allowing the use of
        the more efficient bitwise `&` operation instead of the modulo `%` operator.
        """
        return logical_index & (self._capacity - 1)

    def _prepare_for_new_elements(mut self, n_total: Int, n_retain: Int):
        """Prepares the deque’s internal buffer for adding new elements by
        reallocating memory and retaining the specified number of existing elements.

        Args:
            n_total: The total number of elements the new buffer should support.
            n_retain: The number of existing elements to keep in the deque.
        """
        var new_capacity = next_power_of_two(n_total)
        if new_capacity == n_total:
            new_capacity <<= 1

        var new_data = alloc(
            Layout[Self.ElementType](count=new_capacity)
        ).unsafe_leak()

        for i in range(n_retain):
            var offset = self._physical_index(self._head + i)
            (new_data.unsafe_offset(i)).unsafe_write_move_from(
                self._data.unsafe_offset(offset)
            )

        if self._capacity > 0:
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                    {count = self._capacity}
                )
            )

        self._data = new_data
        self._capacity = new_capacity
        self._head = 0
        self._tail = n_retain

    def _realloc(mut self, new_capacity: Int):
        """Relocates data to a new storage buffer.

        Args:
            new_capacity: The new capacity of the buffer.
        """
        # When head == tail, deque is either empty or full (capacity elements).
        # Use new_capacity > _capacity to distinguish: growing means full, shrinking means empty.
        var is_full = not self and new_capacity > self._capacity
        var deque_len = self._capacity if is_full else len(self)

        var tail_len = self._tail
        var head_len = self._capacity - self._head

        if head_len > deque_len:
            head_len = deque_len
            tail_len = 0

        var new_data = alloc(
            Layout[Self.ElementType](count=new_capacity)
        ).unsafe_leak()

        var src = self._data.unsafe_offset(self._head)
        var dsc = new_data
        for i in range(head_len):
            (dsc.unsafe_offset(i)).unsafe_write_move_from(src.unsafe_offset(i))

        src = self._data
        dsc = new_data.unsafe_offset(head_len)
        for i in range(tail_len):
            (dsc.unsafe_offset(i)).unsafe_write_move_from(src.unsafe_offset(i))

        self._head = 0
        self._tail = deque_len

        if self._capacity > 0:
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._data).unsafe_with_layout(
                    {count = self._capacity}
                )
            )
        self._data = new_data
        self._capacity = new_capacity


@fieldwise_init
struct _DequeIter[
    mut: Bool,
    //,
    T: Copyable & Deinitable,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for Deque.

    Parameters:
        mut: Whether the reference to the deque is mutable.
        T: The type of the elements in the deque.
        origin: The lifetime of the Deque.
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Self.T

    var index: Int
    var src: Pointer[Deque[Self.T], Self.origin]

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        # Read the buffer directly with the whole-deque origin rather than
        # through `Deque.__getitem__`, which vends an *interior* origin: an
        # iterator's element reference stays valid for the whole iteration, so
        # it needs the whole-deque origin, and an interior-origin reference
        # cannot be widened to that on return (doing so would silently drop its
        # invalidation-on-mutation guarantee).
        comptime if Self.forward:
            if self.index >= len(self.src[]):
                raise StopIteration()

            var idx = self.index
            self.index += 1
            var offset = self.src[]._physical_index(self.src[]._head + idx)
            return self.src[]._data[unsafe_offset=offset]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            var offset = self.src[]._physical_index(
                self.src[]._head + self.index
            )
            return self.src[]._data[unsafe_offset=offset]

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = len(self.src[]) - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})


@fieldwise_init
struct _DequeIterOwned[T: Movable & Deinitable](
    IterableOwned, Iterator, Movable
):
    """An owning iterator for Deque.

    Parameters:
        T: The type of the elements in the deque.
    """

    comptime Element = Self.T
    comptime IteratorOwnedType = Self

    var _deque: Deque[Self.T]
    var _index: Int

    @always_inline
    def __deinit__(deinit self):
        # Destroy remaining unconsumed elements at their physical positions.
        # Note: `_index` tracks how many elements __next__ has consumed;
        # _head/_tail are never modified, so len(self._deque) stays constant.
        for i in range(self._index, len(self._deque)):
            var phys = self._deque._physical_index(self._deque._head + i)
            (self._deque._data.unsafe_offset(phys)).unsafe_deinit_pointee()
        # Zero out head/tail so Deque.__deinit__ only frees memory.
        self._deque._head = 0
        self._deque._tail = 0

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._index >= len(self._deque):
            raise StopIteration()
        var phys = self._deque._physical_index(self._deque._head + self._index)
        self._index += 1
        return (self._deque._data.unsafe_offset(phys)).unsafe_take_pointee()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = len(self._deque) - self._index
        return (remaining, {remaining})
