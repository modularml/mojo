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
"""Implements the  Set datatype."""

from std.builtin.rebind import downcast
from std.format._utils import (
    write_sequence_to,
    FormatStruct,
    Named,
    TypeNames,
)
from std.hashlib import Hasher, default_hasher

from .dict import (
    Dict,
    DictEntry,
    KeyElement,
    _DictEntryIter,
    _DictEntryIterOwned,
    _DictKeyIter,
    _DictKeyIterOwned,
)


def _dict_capacity_for(n: Int) -> Int:
    """Return the Dict slot count needed to hold n entries without rehashing.

    Dict's 7/8 load factor means `growth_left = capacity * 7 // 8`. Requesting
    `ceil(8n/7)` slots guarantees `growth_left >= n` after power-of-two rounding.

    Args:
        n: The number of entries to accommodate.

    Returns:
        The minimum capacity to pass to `Dict(capacity=...)`.
    """
    return (n * 8 + 6) // 7


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `Set` with a"
    " non-`Deinitable` element type"
)
struct Set[
    T: KeyElement,
    H: Hasher = default_hasher,
](
    Boolable,
    Comparable where conforms_to(T, Copyable) and conforms_to(T, Equatable),
    Copyable where conforms_to(T, Copyable),
    Deinitable where conforms_to(T, Deinitable),
    Equatable where conforms_to(T, Copyable) and conforms_to(T, Equatable),
    Hashable where conforms_to(T, Copyable) and conforms_to(T, Hashable),
    Iterable,
    IterableOwned where conforms_to(T, Deinitable),
    Movable,
    Sized,
    Writable where conforms_to(T, Copyable) and conforms_to(T, Writable),
):
    """A set data type.

    O(1) average-case amortized add, remove, and membership check.

    ```mojo
    from std.collections import Set

    var set = { 1, 2, 3 }
    print(len(set))  # 3
    set.add(4)

    for element in set:
        print(element)

    set -= Set[Int](3, 4, 5)
    print(set == Set[Int](1, 2))  # True
    print(set | Set[Int](0, 1) == Set[Int](0, 1, 2))  # True
    var element = set.pop()
    print(len(set))  # 1
    ```

    Parameters:
        T: The element type of the set. Must implement `KeyElement` (i.e.
            `Movable & Hashable & Equatable`). Methods that fundamentally need
            to copy elements (`union`, `intersection`, `__or__`, iteration,
            ...) are conditionally available via
            `where conforms_to(T, Copyable)` clauses. When `T` is not
            `Deinitable`, the set has no implicit destructor and must
            be torn down with `deinit_with()`.
        H: The type of the hasher used to hash keys.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[
        downcast[Self.T, KeyElement & Copyable],
        NoneType,
        Self.H,
        iterable_origin,
    ]
    """The iterator type for this set.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime IteratorOwnedType: Iterator = _DictKeyIterOwned[
        downcast[Self.T, KeyElement & Deinitable], NoneType, Self.H
    ]
    """The owned iterator type for this set."""

    # Fields
    var _data: Dict[Self.T, NoneType, Self.H]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        """Construct an empty set."""
        self._data = Dict[Self.T, NoneType, Self.H]()

    def __init__(
        out self, *ts: Self.T, __set_literal__: NoneType = None
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """Construct a set from initial elements.

        Args:
            ts: Variadic of elements to add to the set.
            __set_literal__: Tell Mojo to use this method for set literals.
        """
        self._data = Dict[Self.T, NoneType, Self.H](
            capacity=_dict_capacity_for(len(ts))
        )
        for t in ts:
            self.add(t.copy())

    # TODO: Should take the list owned so we can transfer the elements out.
    def __init__(
        out self, elements: List[Self.T]
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """Construct a set from a List of elements.

        Args:
            elements: A vector of elements to add to the set.
        """
        self._data = Dict[Self.T, NoneType, Self.H](
            capacity=_dict_capacity_for(len(elements))
        )
        for e in elements:
            self.add(e.copy())

    def deinit_with(deinit self, deinit_func: Some[def(var Self.T)], /):
        """Consume the set, deinitializing each element with a closure.

        Use this to tear down a `Set` whose element type is not
        `Deinitable`.

        Args:
            deinit_func: A closure called once per element to destroy it.
        """

        # The backing `Dict` maps each element to a `NoneType` value, so wrap
        # the element-only closure to also drop the value slot.
        def forward(var key: Self.T, var value: NoneType) {imm deinit_func}:
            deinit_func(key^)

        self._data^.deinit_with(forward)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    def __contains__(self, t: Self.T) -> Bool:
        """Whether or not the set contains an element.

        Args:
            t: The element to check membership in the set.

        Returns:
            Whether or not the set contains the element.
        """
        return t in self._data

    def __eq__(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Equatable
    ):
        """Set equality.

        Args:
            other: Another Set instance to check equality against.

        Returns:
            True if the sets contain the same elements and False otherwise.
        """
        if len(self) != len(other):
            return False
        # Iterate over dict entries directly to reuse cached hash values,
        # avoiding redundant hash recomputation for each lookup in `other`.
        for entry in self._data.items():
            if not other._data._find_slot(entry._hash, entry.key)[0]:
                return False
        return True

    def __and__(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """The set intersection operator.

        Args:
            other: Another Set instance to intersect with this one.

        Returns:
            A new set containing only the elements which appear in both
            this set and the `other` set.
        """
        return self.intersection(other)

    def __iand__(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """In-place set intersection.

        Updates the set to contain only the elements which are already in
        the set and are also contained in the `other` set.

        Args:
            other: Another Set instance to intersect with this one.
        """
        self.intersection_update(other)

    def __or__(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """The set union operator.

        Args:
            other: Another Set instance to union with this one.

        Returns:
            A new set containing any elements which appear in either
            this set or the `other` set.
        """
        return self.union(other)

    def __ior__(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """In-place set union.

        Updates the set to contain all elements in the `other` set
        as well as keeping all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        self.update(other)

    def __sub__(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """Set subtraction.

        Args:
            other: Another Set instance to subtract from this one.

        Returns:
            A new set containing elements of this set, but not containing
            any elements which were in the `other` set.
        """
        return self.difference(other)

    def __isub__(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """In-place set subtraction.

        Updates the set to remove any elements from the `other` set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        self.difference_update(other)

    def __le__(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Equatable
    ):
        """Overloads the <= operator for sets. Works like as `issubset` method.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a subset of the `other` set, False otherwise.
        """
        return self.issubset(other)

    def __ge__(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Equatable
    ):
        """Overloads the >= operator for sets. Works like as `issuperset` method.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a superset of the `other` set, False otherwise.
        """
        return self.issuperset(other)

    def __gt__(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Equatable
    ):
        """Overloads the > operator for strict superset comparison of sets.

        Args:
            other: The set to compare against for the strict superset relationship.

        Returns:
            True if the set is a strict superset of the `other` set, False otherwise.
        """
        return len(self) > len(other) and other.issubset(self)

    def __lt__(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Equatable
    ):
        """Overloads the < operator for strict subset comparison of sets.

        Args:
            other: The set to compare against for the strict subset relationship.

        Returns:
            True if the set is a strict subset of the `other` set, False otherwise.
        """
        return len(self) < len(other) and self.issubset(other)

    def __xor__(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """Overloads the ^ operator for sets. Works like as `symmetric_difference` method.

        Args:
            other: The set to find the symmetric difference with.

        Returns:
            A new set containing the symmetric difference of the two sets.
        """
        return self.symmetric_difference(other)

    def __ixor__(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """Overloads the ^= operator. Works like as `symmetric_difference_update` method.

        Updates the set with the symmetric difference of itself and another set.

        Args:
            other: The set to find the symmetric difference with.
        """
        self.symmetric_difference_update(other)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    def __bool__(self) -> Bool:
        """Whether the set is non-empty or not.

        Returns:
            True if the set is non-empty, False if it is empty.
        """
        return len(self).__bool__()

    def __len__(self) -> Int:
        """The size of the set.

        Returns:
            The number of elements in the set.
        """
        return len(self._data)

    def __hash__(
        self, mut hasher: Some[Hasher]
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Hashable):
        """Updates hasher with the underlying values.

        The update is order independent, so s1 == s2 -> hash(s1) == hash(s2).

        Args:
            hasher: The hasher instance.
        """
        var hash_value: UInt64 = 0
        # Hash combination needs to be commutative so iteration order
        # doesn't impact the hash value.
        for e in self:
            hash_value ^= hash(e)
        hash_value.__hash__(hasher)

    def _write_self_to[
        *, is_repr: Bool
    ](self, mut writer: Some[Writer]) where conforms_to(
        Self.T, Copyable
    ) and conforms_to(Self.T, Writable):
        var iterator = self.__iter__()

        def iterate(mut w: Some[Writer]) raises StopIteration {mut iterator}:
            ref element = iterator.__next__()

            comptime if is_repr:
                element.write_repr_to(w)
            else:
                element.write_to(w)

        write_sequence_to(writer, iterate, start="{", end="}")
        _ = iterator^

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Writable):
        """Write this set to a `Writer`.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[is_repr=False](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Writable):
        """Write this set to a `Writer`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_self_to[is_repr=True](w)

        FormatStruct(writer, "Set").params(
            TypeNames[Self.T](),
            Named("Hasher", TypeNames[Self.H]()),
        ).fields(write_fields)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def __iter__(
        deinit self,
    ) -> Self.IteratorOwnedType where conforms_to(Self.T, Deinitable):
        """Consume the set and iterate over its elements.

        Constraints:
            `T` must be `Deinitable`; consuming iteration drops the
            backing dictionary's value slots in place.

        Returns:
            An iterator that owns the set's elements.
        """
        return {
            _DictEntryIterOwned(
                rebind_var[
                    Dict[
                        downcast[Self.T, KeyElement & Deinitable],
                        NoneType,
                        Self.H,
                    ]
                ](self._data^),
                0,
            )
        }

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the set, returning immutable references.

        Returns:
            An iterator of immutable references to the set elements.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.T, Copyable
        ), "Set iteration requires the element type to be `Copyable`."
        comptime DictCopyable = Dict[
            downcast[Self.T, KeyElement & Copyable],
            NoneType,
            Self.H,
        ]
        # here we rely on Set being a trivial wrapper of a Dict
        return _DictKeyIter(
            _DictEntryIter(
                0,
                0,
                rebind[Pointer[DictCopyable, origin_of(self)]](
                    Pointer(to=self._data)
                )[],
            )
        )

    def add(mut self, var t: Self.T) where conforms_to(Self.T, Deinitable):
        """Add an element to the set.

        Constraints:
            `T` must be `Deinitable`; adding a duplicate discards the
            incoming element in place.

        Args:
            t: The element to add to the set.
        """
        self._data[t^] = None

    def insert(mut self, var t: Self.T) -> Optional[Self.T]:
        """Insert an element, returning any displaced equal element.

        Unlike `add`, a displaced equal element is moved out and returned
        rather than destroyed in place, so this works when `T` is not
        `Deinitable`. The caller owns the returned element.

        Args:
            t: The element to insert into the set.

        Returns:
            The previously-present equal element if one was displaced,
            otherwise an empty `Optional`.
        """

        def reap(var entry: DictEntry[Self.T, NoneType, Self.H]) -> Self.T:
            return entry^.reap_key()

        return self._data.insert(t^, None).map(reap)

    def remove(
        mut self, t: Self.T
    ) raises where conforms_to(Self.T, Deinitable):
        """Remove an element from the set.

        Constraints:
            `T` must be `Deinitable`; the removed element is destroyed
            in place.

        Args:
            t: The element to remove from the set.

        Raises:
            If the element isn't in the set to remove.
        """
        # TODO(MOCO-4295): the `where` clause narrows `Self.T` to a deletable
        # subtype that no longer unifies with the backing dict's
        # `ref key: Self.K`; rebind the dict to the narrowed key type so the
        # argument matches. Drop this once the narrowing is transparent.
        _ = rebind[
            Pointer[
                Dict[
                    downcast[Self.T, KeyElement & Deinitable], NoneType, Self.H
                ],
                origin_of(self._data),
            ]
        ](Pointer(to=self._data))[].pop(t)

    def pop(mut self) raises -> Self.T:
        """Remove any one item from the set, and return it.

        As an implementation detail this will remove the last item
        according to insertion order.

        Returns:
            The element which was removed from the set.

        Raises:
            If the set is empty.
        """
        try:
            return self._data.popitem().reap_key()
        except:
            raise "Pop on empty set"

    def union(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """Set union.

        Args:
            other: Another Set instance to union with this one.

        Returns:
            A new set containing any elements which appear in either
            this set or the `other` set.
        """
        var result = self.copy()
        for o in other:
            result.add(o.copy())

        return result^

    def intersection(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """Set intersection.

        Args:
            other: Another Set instance to intersect with this one.

        Returns:
            A new set containing only the elements which appear in both
            this set and the `other` set.
        """
        var result = Set[Self.T, Self.H]()
        for v in self:
            if v in other:
                result.add(v.copy())

        return result^

    def difference(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """Set difference.

        Args:
            other: Another Set instance to find the difference with this one.

        Returns:
            A new set containing elements that are in this set but not in
            the `other` set.
        """
        var result = Set[Self.T, Self.H]()
        for e in self:
            if e not in other:
                result.add(e.copy())
        return result^

    def update(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """In-place set update.

        Updates the set to contain all elements in the `other` set
        as well as keeping all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        for e in other:
            self.add(e.copy())

    def intersection_update(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """In-place set intersection update.

        Updates the set by retaining only elements found in both this set and the `other` set,
        removing all other elements. The result is the intersection of this set with `other`.

        Args:
            other: Another Set instance to intersect with this one.
        """
        # When self is larger, it is cheaper to build the intersection
        # from other's side — iterate the smaller collection and do
        # lookups into the larger one.
        if len(self) > len(other):
            var keep = Self()
            for e in other:
                if e in self:
                    keep.add(e.copy())
            self = keep^
        else:
            var to_remove = List[Self.T](capacity=len(self))
            for e in self:
                if e not in other:
                    to_remove.append(e.copy())
            for e in to_remove:
                self.discard(e)

    def difference_update(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """In-place set subtraction.

        Updates the set by removing all elements found in the `other` set,
        effectively keeping only elements that are unique to this set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        for o in other:
            try:
                self.remove(o)
            except:
                pass

    def issubset(self, other: Self) -> Bool where conforms_to(Self.T, Copyable):
        """Check if this set is a subset of another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a subset of the `other` set, False otherwise.
        """
        if len(self) > len(other):
            return False

        for element in self:
            if element not in other:
                return False

        return True

    def isdisjoint(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable):
        """Check if this set is disjoint with another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is disjoint with the `other` set, False otherwise.
        """
        for element in self:
            if element in other:
                return False

        return True

    def issuperset(
        self, other: Self
    ) -> Bool where conforms_to(Self.T, Copyable):
        """Check if this set is a superset of another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a superset of the `other` set, False otherwise.
        """
        if len(self) < len(other):
            return False

        for element in other:
            if element not in self:
                return False

        return True

    def symmetric_difference(
        self, other: Self
    ) -> Self where conforms_to(Self.T, Copyable) and conforms_to(
        Self.T, Deinitable
    ):
        """Returns the symmetric difference of two sets.

        Args:
            other: The set to find the symmetric difference with.

        Returns:
            A new set containing the symmetric difference of the two sets.
        """
        var result = Set[Self.T, Self.H]()

        for element in self:
            if element not in other:
                result.add(element.copy())

        for element in other:
            if element not in self:
                result.add(element.copy())

        return result^

    def symmetric_difference_update(
        mut self, other: Self
    ) where conforms_to(Self.T, Copyable) and conforms_to(Self.T, Deinitable):
        """Updates the set with the symmetric difference of itself and another set.

        Args:
            other: The set to find the symmetric difference with.
        """
        self = self.symmetric_difference(other)

    def discard(mut self, value: Self.T) where conforms_to(Self.T, Deinitable):
        """Remove a value from the set if it exists. Pass otherwise.

        Constraints:
            `T` must be `Deinitable`; a removed element is destroyed
            in place.

        Args:
            value: The element to remove from the set.
        """
        try:
            self._data.pop(value)
        except:
            pass

    def clear(mut self) where conforms_to(Self.T, Deinitable):
        """Removes all elements from the set.

        This method modifies the set in-place, removing all of its elements.
        After calling this method, the set will be empty.

        Constraints:
            `T` must be `Deinitable`, since every element is destroyed
            in place.
        """
        self._data.clear()

    def clear_with(mut self, destroy_func: Some[def(var Self.T)], /):
        """Remove all elements, disposing each with a closure.

        The closure counterpart of `clear`: instead of destroying each element
        in place, it hands each element to `destroy_func`. Use this to clear a
        `Set` whose element type is not `Deinitable`. The set's
        capacity is retained.

        Args:
            destroy_func: A closure called once per element to destroy it.
        """

        # The backing `Dict` maps each element to a `NoneType` value, so wrap
        # the element-only closure to also drop the value slot.
        def forward(var key: Self.T, var value: NoneType) {imm destroy_func}:
            destroy_func(key^)

        self._data.clear_with(forward)
