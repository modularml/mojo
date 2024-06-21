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
"""Implements the  Set datatype."""

from .dict import Dict, KeyElement, _DictEntryIter, _DictKeyIter


struct Set[T: KeyElement](Sized, Comparable, Hashable, Boolable):
    """A set data type.

    O(1) average-case amortized add, remove, and membership check.

    ```mojo
    from collections import Set

    var set = Set[Int](1, 2, 3)
    print(len(set))  # 3
    set.add(4)

    for element in set:
        print(element[])

    set -= Set[Int](3, 4, 5)
    print(set == Set[Int](1, 2))  # True
    print(set | Set[Int](0, 1) == Set[Int](0, 1, 2))  # True
    var element = set.pop()
    print(len(set))  # 1
    ```

    Parameters:
        T: The element type of the set. Must implement KeyElement.
    """

    # Fields
    var _data: Dict[T, NoneType]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(inout self, *ts: T):
        """Construct a set from initial elements.

        Args:
            ts: Variadic of elements to add to the set.
        """
        self._data = Dict[T, NoneType]()
        for t in ts:
            self.add(t[])

    fn __init__(inout self, elements: Self):
        """Explicitly copy another Set instance.

        Args:
            elements: An existing set to copy.
        """
        self.__init__()
        for e in elements:
            self.add(e[])

    fn __init__(inout self, elements: List[T]):
        """Construct a set from a List of elements.

        Args:
            elements: A vector of elements to add to the set.
        """
        self.__init__()
        for e in elements:
            self.add(e[])

    fn __moveinit__(inout self, owned other: Self):
        """Move constructor.

        Args:
            other: The existing Set instance to move from.
        """
        self._data = other._data^

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __contains__(self, t: T) -> Bool:
        """Whether or not the set contains an element.

        Args:
            t: The element to check membership in the set.

        Returns:
            Whether or not the set contains the element.
        """
        return t in self._data

    fn __eq__(self, other: Self) -> Bool:
        """Set equality.

        Args:
            other: Another Set instance to check equality against.

        Returns:
            True if the sets contain the same elements and False otherwise.
        """
        if len(self) != len(other):
            return False
        for e in self:
            if e[] not in other:
                return False
        return True

    fn __ne__(self, other: Self) -> Bool:
        """Set inequality.

        Args:
            other: Another Set instance to check equality against.

        Returns:
            True if the sets are different and False otherwise.
        """
        return not (self == other)

    fn __and__(self, other: Self) -> Self:
        """The set intersection operator.

        Args:
            other: Another Set instance to intersect with this one.

        Returns:
            A new set containing only the elements which appear in both
            this set and the `other` set.
        """
        return self.intersection(other)

    fn __iand__(inout self, other: Self):
        """In-place set intersection.

        Updates the set to contain only the elements which are already in
        the set and are also contained in the `other` set.

        Args:
            other: Another Set instance to intersect with this one.
        """
        self.intersection_update(other)

    fn __or__(self, other: Self) -> Self:
        """The set union operator.

        Args:
            other: Another Set instance to union with this one.

        Returns:
            A new set containing any elements which appear in either
            this set or the `other` set.
        """
        return self.union(other)

    fn __ior__(inout self, other: Self):
        """In-place set union.

        Updates the set to contain all elements in the `other` set
        as well as keeping all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        self.update(other)

    fn __sub__(self, other: Self) -> Self:
        """Set subtraction.

        Args:
            other: Another Set instance to subtract from this one.

        Returns:
            A new set containing elements of this set, but not containing
            any elements which were in the `other` set.
        """
        return self.difference(other)

    fn __isub__(inout self, other: Self):
        """In-place set subtraction.

        Updates the set to remove any elements from the `other` set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        self.difference_update(other)

    fn __le__(self, other: Self) -> Bool:
        """Overloads the <= operator for sets. Works like as `issubset` method.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a subset of the `other` set, False otherwise.
        """
        return self.issubset(other)

    fn __ge__(self, other: Self) -> Bool:
        """Overloads the >= operator for sets. Works like as `issuperset` method.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a superset of the `other` set, False otherwise.
        """
        return self.issuperset(other)

    fn __gt__(self, other: Self) -> Bool:
        """Overloads the > operator for strict superset comparison of sets.

        Args:
            other: The set to compare against for the strict superset relationship.

        Returns:
            True if the set is a strict superset of the `other` set, False otherwise.
        """
        return self >= other and self != other

    fn __lt__(self, other: Self) -> Bool:
        """Overloads the < operator for strict subset comparison of sets.

        Args:
            other: The set to compare against for the strict subset relationship.

        Returns:
            True if the set is a strict subset of the `other` set, False otherwise.
        """
        return self <= other and self != other

    fn __xor__(self, other: Self) -> Self:
        """Overloads the ^ operator for sets. Works like as `symmetric_difference` method.

        Args:
            other: The set to find the symmetric difference with.

        Returns:
            A new set containing the symmetric difference of the two sets.
        """
        return self.symmetric_difference(other)

    fn __ixor__(inout self, other: Self):
        """Overloads the ^= operator. Works like as `symmetric_difference_update` method.

        Updates the set with the symmetric difference of itself and another set.

        Args:
            other: The set to find the symmetric difference with.
        """
        self.symmetric_difference_update(other)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __bool__(self) -> Bool:
        """Whether the set is non-empty or not.

        Returns:
            True if the set is non-empty, False if it is empty.
        """
        return len(self).__bool__()

    fn __len__(self) -> Int:
        """The size of the set.

        Returns:
            The number of elements in the set.
        """
        return len(self._data)

    fn __hash__(self) -> Int:
        """A hash value of the elements in the set.

        The hash value is order independent, so s1 == s2 -> hash(s1) == hash(s2).

        Returns:
            A hash value of the set suitable for non-cryptographic purposes.
        """
        var hash_value = 0
        # Hash combination needs to be commutative so iteration order
        # doesn't impact the hash value.
        for e in self:
            hash_value ^= hash(e[])
        return hash_value

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn __iter__(
        ref [_]self: Self,
    ) -> _DictKeyIter[T, NoneType, __lifetime_of(self)]:
        """Iterate over elements of the set, returning immutable references.

        Returns:
            An iterator of immutable references to the set elements.
        """
        # here we rely on Set being a trivial wrapper of a Dict
        return _DictKeyIter(_DictEntryIter(0, 0, self._data))

    fn add(inout self, t: T):
        """Add an element to the set.

        Args:
            t: The element to add to the set.
        """
        self._data[t] = None

    fn remove(inout self, t: T) raises:
        """Remove an element from the set.

        Args:
            t: The element to remove from the set.

        Raises:
            If the element isn't in the set to remove.
        """
        self._data.pop(t)

    fn pop(inout self) raises -> T:
        """Remove any one item from the set, and return it.

        As an implementation detail this will remove the first item
        according to insertion order. This is practically useful
        for breadth-first search implementations.

        Returns:
            The element which was removed from the set.

        Raises:
            If the set is empty.
        """
        if not self:
            raise "Pop on empty set"
        var iter = self.__iter__()
        var first = iter.__next__()[]
        self.remove(first)
        return first

    fn union(self, other: Self) -> Self:
        """Set union.

        Args:
            other: Another Set instance to union with this one.

        Returns:
            A new set containing any elements which appear in either
            this set or the `other` set.
        """
        var result = Set(self)
        for o in other:
            result.add(o[])

        return result^

    fn intersection(self, other: Self) -> Self:
        """Set intersection.

        Args:
            other: Another Set instance to intersect with this one.

        Returns:
            A new set containing only the elements which appear in both
            this set and the `other` set.
        """
        var result = Set[T]()
        for v in self:
            if v[] in other:
                result.add(v[])

        return result^

    fn difference(self, other: Self) -> Self:
        """Set difference.

        Args:
            other: Another Set instance to find the difference with this one.

        Returns:
            A new set containing elements that are in this set but not in
            the `other` set.
        """
        var result = Set[T]()
        for e in self:
            if e[] not in other:
                result.add(e[])
        return result^

    fn update(inout self, other: Self):
        """In-place set update.

        Updates the set to contain all elements in the `other` set
        as well as keeping all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        for e in other:
            self.add(e[])

    fn intersection_update(inout self, other: Self):
        """In-place set intersection update.

        Updates the set by retaining only elements found in both this set and the `other` set,
        removing all other elements. The result is the intersection of this set with `other`.

        Args:
            other: Another Set instance to intersect with this one.
        """
        # Possible to do this without an extra allocation, but need to be
        # careful about concurrent iteration + mutation
        self.difference_update(self - other)

    fn difference_update(inout self, other: Self):
        """In-place set subtraction.

        Updates the set by removing all elements found in the `other` set,
        effectively keeping only elements that are unique to this set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        for o in other:
            try:
                self.remove(o[])
            except:
                pass

    fn issubset(self, other: Self) -> Bool:
        """Check if this set is a subset of another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a subset of the `other` set, False otherwise.
        """
        if len(self) > len(other):
            return False

        for element in self:
            if element[] not in other:
                return False

        return True

    fn isdisjoint(self, other: Self) -> Bool:
        """Check if this set is disjoint with another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is disjoint with the `other` set, False otherwise.
        """
        for element in self:
            if element[] in other:
                return False

        return True

    fn issuperset(self, other: Self) -> Bool:
        """Check if this set is a superset of another set.

        Args:
            other: Another Set instance to check against.

        Returns:
            True if this set is a superset of the `other` set, False otherwise.
        """
        if len(self) < len(other):
            return False

        for element in other:
            if element[] not in self:
                return False

        return True

    fn symmetric_difference(self, other: Self) -> Self:
        """Returns the symmetric difference of two sets.

        Args:
            other: The set to find the symmetric difference with.

        Returns:
            A new set containing the symmetric difference of the two sets.
        """
        var result = Set[T]()

        for element in self:
            if element[] not in other:
                result.add(element[])

        for element in other:
            if element[] not in self:
                result.add(element[])

        return result^

    fn symmetric_difference_update(inout self, other: Self):
        """Updates the set with the symmetric difference of itself and another set.

        Args:
            other: The set to find the symmetric difference with.
        """
        self = self.symmetric_difference(other)

    fn discard(inout self, value: T):
        """Remove a value from the set if it exists. Pass otherwise.

        Args:
            value: The element to remove from the set.
        """
        try:
            self._data.pop(value)
        except:
            pass

    fn clear(inout self) raises:
        """Removes all elements from the set.

        This method modifies the set in-place, removing all of its elements.
        After calling this method, the set will be empty.
        """
        for _ in range(len(self)):
            var a = self.pop()

        #! This code below (without using range function) won't pass tests
        #! It leaves set with one remaining item. Is this a bug?
        # for _ in self:
        #     var a = self.pop()
