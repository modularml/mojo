# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Implements the  Set datatype."""

from .dict import (
    Dict,
    KeyElement,
    _DictKeyIter,
    _DictEntryIter,
)


struct Set[T: KeyElement](Sized, EqualityComparable, Hashable, Boolable):
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
    let element = set.pop()
    print(len(set))  # 1
    ```

    Parameters:
        T: The element type of the set. Must implement KeyElement.
    """

    var _data: Dict[T, NoneType]

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

    fn __init__(inout self, elements: DynamicVector[T]):
        """Construct a set from a DynamicVector of elements.

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
        self._data = other._data ^

    fn __contains__(self, t: T) -> Bool:
        """Whether or not the set contains an element.

        Args:
            t: The element to check membership in the set.

        Returns:
            Whether or not the set contains the element.
        """
        return t in self._data

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

    fn __eq__(self, other: Self) -> Bool:
        """Set equality.

        Args:
            other: Another Set instance to check equality against.

        Returns:
            True if `other` contains exactly the same elements and no more,
            False if the sets are different.
        """
        if len(self) != len(other):
            return False
        for e in self:
            if e[] not in other:
                return False
        return True

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
        # Possible to do this without an extra allocation, but need to be
        # careful about concurrent iteration + mutation
        self.remove_all(self - other)

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
        as well as all elements it already contained.

        Args:
            other: Another Set instance to union with this one.
        """
        for e in other:
            self.add(e[])

    fn __sub__(self, other: Self) -> Self:
        """Set subtraction.

        Args:
            other: Another Set instance to subtract from this one.

        Returns:
            A new set containing elements of this set, but not containing
            any elements which were in the `other` set.
        """
        var result = Set[T]()
        for e in self:
            if e[] not in other:
                result.add(e[])
        return result ^

    fn __isub__(inout self, other: Self):
        """In-place set subtraction.

        Updates the set to remove any elements from the `other` set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        self.remove_all(other)

    fn __iter__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictKeyIter[T, NoneType, mutability, self_life]:
        """Iterate over elements of the set, returning immutable references.

        Returns:
            An iterator of immutable references to the set elements.
        """
        # self._data has its own lifetime that's not self_lifetime
        # here we rely on Set being a trivial wrapper of a Dict
        return _DictKeyIter(
            _DictEntryIter[
                T,
                NoneType,
                mutability,
                self_life,
            ](0, 0, Reference(self).bitcast_element[Dict[T, NoneType]]())
        )

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
        let first = iter.__next__()[]
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

        return result ^

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

        return result ^

    fn remove_all(inout self, other: Self):
        """In-place set subtraction.

        Updates the set to remove any elements from the `other` set.

        Args:
            other: Another Set instance to subtract from this one.
        """
        for o in other:
            try:
                self.remove(o[])
            except:
                pass
