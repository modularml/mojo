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
"""Defines the `Counter` type.

Import these APIs from the `collections` package:

```mojo
from std.collections import Counter

```

Counters provide convenient tallying objects that use a dictionary to
store keys and their counts. They offer the full functionality of
counted sets, also called bags or multisets, and extend that model by
supporting negative counts.

"""
from std.collections.dict import (
    Dict,
    _DictEntryIter,
    _DictEntryIterOwned,
    _DictKeyIter,
    _DictKeyIterOwned,
    _DictValueIter,
)
import std.format._utils as fmt
from std.hashlib import Hasher, default_hasher

from std.utils import Variant


@fieldwise_init
struct Counter[
    V: KeyElement & Copyable & Deinitable,
    H: Hasher = default_hasher,
](
    Boolable,
    Copyable,
    Defaultable,
    Equatable,
    Iterable,
    IterableOwned,
    Sized,
    Writable where conforms_to(V, Writable),
):
    """A container for counting hashable items.

    In other languages, similar types to counters include bags, counted sets,
    and multisets, although their semantics are normally closer to sets (adding,
    removing, intersecting, unions, etc) rather than increasing and decreasing
    counts. Mojo's `Counter` follows Python's model, and adds math versatility by
    supporting negative counts.

    The value type must implement the `KeyElement` trait and be `Copyable`, as
    its values are stored in a dictionary as keys and the API copies elements
    extensively (e.g., `most_common`, `subtract`, merge ops). The keys' uniform
    value type must be hashable for use in the
    underlying dictionary.

    Example:

    ```mojo
    from std.collections import Counter

    var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
    print(counter["a"]) # prints 3
    print(counter["b"]) # prints 2
    ```

    Parameters:
        V: The value type to be counted. Must be `KeyElement` and `Copyable`.
        H: The type of the hasher in the underlying dictionary.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[Self.V, Int, Self.H, iterable_origin]
    """The iterator type for this counter.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime IteratorOwnedType: Iterator = _DictKeyIterOwned[
        Self.V, Int, Self.H
    ]
    """The owned iterator type for this counter."""

    # Fields
    var _data: Dict[Self.V, Int, Self.H]

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    def __init__(out self):
        """Create a new, empty `Counter` object."""
        self._data = Dict[Self.V, Int, Self.H]()

    def __init__(out self, var *values: Self.V):
        """Create a new `Counter` from a list of values.

        Args:
            values: A list of values to count.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String]("a", "a", "a", "b", "b", "c", "d", "c", "c")
        print(counter["a"])  # print 3
        print(counter["b"])  # print 2
        ```

        Note:
        A counter is not limited to the values used in this initial list.
        You may add new keys as needed or remove them with clear or one
        of the `pop` calls.
        """
        self._data = Dict[Self.V, Int, Self.H]()
        for item in values:
            self._data.setdefault(item.copy(), 0) += 1

    def __init__[
        IterableType: Iterable,
    ](
        out self,
        ref iterable: IterableType,
    ) where (
        IterableType.IteratorType[origin_of(iterable)].Element == Self.V
    ):
        """Create a `Counter` from an iterable of values.

        Each value produced by the iterable is tallied, so the resulting
        `Counter` maps every distinct value to the number of times it appeared.

        Parameters:
            IterableType: The type of the `iterable` argument.

        Args:
            iterable: The iterable of values to count.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter(["a", "a", "a", "b", "b", "c", "d", "c", "c"])
        print(counter["a"]) # prints 3
        print(counter["b"]) # prints 2

        # Any iterable works, e.g. the bytes of a string:
        var by_byte = Counter(String("aaabbaacdd").bytes())
        print(by_byte[Byte(ord("a"))]) # prints 5
        ```
        """
        self = Self()
        for var value in iterable:
            # TODO(MOCO-4355): Drop `rebind_var` once the `where` clause's
            # `Element == Self.V` equality is applied when checking the body.
            self._data.setdefault(rebind_var[Self.V](value^), 0) += 1

    @staticmethod
    def fromkeys(keys: List[Self.V], value: Int) -> Self:
        """Create a new `Counter` from a list of keys and a default value.

        Args:
            keys: The keys to create the `Counter` from.
            value: The default value to associate with each key. Must be non-negative.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter["a"]) # output: 1
        ```

        Returns:
            A new `Counter` with the count of each passed key set to `value`.
        """
        assert value >= 0, "value must be non-negative"
        var result = Counter[Self.V, Self.H]()
        for key in keys:
            result[key] = value
        return result^

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    def __getitem__(self, key: Self.V) -> Int:
        """Get the count of a key.

        Args:
            key: The key to get the count of.

        Returns:
            The count of the key.
        """
        return self.get(key, 0)

    def __setitem__(mut self, value: Self.V, count: Int):
        """Set a value in the keyword `Counter` by key.

        Args:
            value: The value to associate with the specified count.
            count: The count to store in the `Counter`.
        """
        self._data[value.copy()] = count

    def __iter__(deinit self) -> Self.IteratorOwnedType:
        """Consume the counter and iterate over its keys.

        Returns:
            An iterator that owns the counter's keys.
        """
        return {_DictEntryIterOwned(self._data^, 0)}

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the `Counter`'s keys as immutable references.

        Returns:
            An iterator of immutable references to the `Counter` values.
        """
        # TODO(MOCO-4205): origin cast only needed to retarget the origin from
        # `origin_of(self._data)` to `origin_of(self)`.
        return _DictKeyIter(
            _DictEntryIter(
                0,
                0,
                Pointer(to=self._data).unsafe_origin_cast[origin_of(self)]()[],
            )
        )

    def __contains__(self, key: Self.V) -> Bool:
        """Check if a given key is in the `Counter` or not.

        Args:
            key: The key to check.

        Returns:
            `True` if there key exists in the `Counter`, `False` otherwise.
        """
        return key in self._data

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    def __len__(self) -> Int:
        """Returns the number of elements currently stored in the `Counter`.

        Returns:
            The number of elements in the `Counter`.
        """
        return len(self._data)

    def __bool__(self) -> Bool:
        """Check if the `Counter` is empty or not.

        Returns:
            `False` if the `Counter` is empty, `True` otherwise.
        """
        return Bool(len(self))

    def _write_counter_body[
        f_key: def(Self.V, mut Some[Writer]) thin,
        f_val: def(Int, mut Some[Writer]) thin,
    ](self, mut writer: Some[Writer]) where conforms_to(Self.V, Writable):
        """Write the counter's key-value pairs to a writer.

        Parameters:
            f_key: The function to format keys.
            f_val: The function to format values.

        Args:
            writer: The object to write to.
        """
        writer.write_string("{")

        var items = self.most_common(len(self))
        for i in range(len(items)):
            if i > 0:
                writer.write_string(", ")
            ref item = items[i]
            f_key(item._value, writer)
            writer.write_string(": ")
            f_val(item._count, writer)

        writer.write_string("}")

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.V, Writable):
        """Write this `Counter` to a writer.

        Constraints:
            `V` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """
        self._write_counter_body[
            f_key=fmt.write_to[Self.V],
            f_val=fmt.write_to[Int],
        ](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.V, Writable):
        """Write the repr of this `Counter` to a writer.

        Constraints:
            `V` must conform to `Writable`.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_counter_body[
                f_key=fmt.write_repr_to[Self.V],
                f_val=fmt.write_repr_to[Int],
            ](w)

        fmt.FormatStruct(writer, "Counter").params(
            fmt.TypeNames[Self.V](),
        ).fields(write_fields)

    # ===------------------------------------------------------------------=== #
    # Comparison operators
    # ===------------------------------------------------------------------=== #

    def __eq__(self, other: Self) -> Bool:
        """Check if all counts agree. Missing counts are treated as zero.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if the two `Counter`s are equal, `False` otherwise.
        """

        for e in self.keys():
            if self.get(e, 0) != other.get(e, 0):
                return False
        for e in other.keys():
            if self.get(e, 0) != other.get(e, 0):
                return False
        return True

    def le(self, other: Self) -> Bool:
        """Check if all counts are less than or equal to those in the other
        `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are less than or equal to the other `Counter`,
            `False` otherwise.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(counter.le(other)) # output: True
        counter[3] += 20
        print(counter.le(other)) # output: False
        ```
        """

        for e in self.keys():
            if self.get(e, 0) > other.get(e, 0):
                return False
        return True

    def lt(self, other: Self) -> Bool:
        """Check if all counts are less than those in the other `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are less than in the other `Counter`, `False`
            otherwise.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 3)
        print(counter.lt(other)) # output: True
        counter[1] += 1
        print(counter.lt(other)) # output: False
        ```
        """

        for e in self.keys():
            if self.get(e, 0) >= other.get(e, 0):
                return False
        return True

    def gt(self, other: Self) -> Bool:
        """Check if all counts are greater than those in the other `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are greater than in the other `Counter`,
            `False` otherwise.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 3)
        print(other.gt(counter)) # output: True
        counter[1] += 1
        print(other.gt(counter)) # output: False
        ```
        """
        return other.lt(self)

    def ge(self, other: Self) -> Bool:
        """Check if all counts are greater than or equal to those in the other
        `Counter`.

        Note that since we check that _all_ counts satisfy the condition, this
        comparison does not make `Counter`s totally ordered.

        Args:
            other: The other `Counter` to compare to.

        Returns:
            `True` if all counts are greater than or equal to the other
            `Counter`, `False` otherwise.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(other.ge(counter)) # output: True
        counter[3] += 20
        print(other.ge(counter)) # output: False
        ```
        """
        return other.le(self)

    # ===------------------------------------------------------------------=== #
    # Binary operators
    # ===------------------------------------------------------------------=== #

    def __add__(self, other: Self) -> Self:
        """Add counts from two `Counter`s.

        Args:
            other: The other `Counter` to add to this `Counter`.

        Returns:
            A new `Counter` with the counts from both `Counter`s added together.
        """
        var result = Counter[Self.V, Self.H]()

        result.update(self)
        result.update(other)

        return +result^  # Remove zero and negative counts

    def __iadd__(mut self, other: Self):
        """Add counts from another `Counter` to this `Counter`.

        Args:
            other: The other `Counter` to add to this `Counter`.
        """
        self.update(other)
        self._keep_positive()

    def __sub__(self, other: Self) -> Self:
        """Subtract counts, but keep only results with positive counts.

        Args:
            other: The other `Counter` to subtract from this `Counter`.

        Returns:
            A new `Counter` with the counts from the other `Counter` subtracted
            from this `Counter`.
        """
        var result = self.copy()

        result.subtract(other)

        return +result^  # Remove zero and negative counts

    def __isub__(mut self, other: Self):
        """Subtract counts from another `Counter` from this `Counter`, but keep
        only results with positive counts.

        Args:
            other: The other `Counter` to subtract from this `Counter`.
        """
        self.subtract(other)
        self._keep_positive()

    def __and__(self, other: Self) -> Self:
        """Intersection: keep common elements with the minimum count.

        Args:
            other: The other `Counter` to intersect with.

        Returns:
            A new `Counter` with the common elements and the minimum count of
            the two `Counter`s.
        """
        var result = Counter[Self.V, Self.H]()

        for key in self.keys():
            if key in other:
                result[key] = min(self.get(key, 0), other.get(key, 0))

        return result^

    def __iand__(mut self, other: Self):
        """Intersection: keep common elements with the minimum count.

        Args:
            other: The other `Counter` to intersect with.
        """
        for key in self.keys():
            if key not in other:
                try:
                    var key_copy = key.copy()  # Copy due to incorrect origins.
                    _ = self.pop(key_copy)
                except:
                    pass  # this should not happen
            else:
                var key_copy = key.copy()  # Copy due to incorrect origins.
                self[key_copy] = min(self.get(key, 0), other.get(key, 0))

    def __or__(self, other: Self) -> Self:
        """Union: keep all elements with the maximum count.

        Args:
            other: The other `Counter` to union with.

        Returns:
            A new `Counter` with all elements and the maximum count of the two
            `Counter`s.
        """
        var result = Counter[Self.V, Self.H]()

        for key in self.keys():
            var newcount = max(self.get(key, 0), other.get(key, 0))
            if newcount > 0:
                result[key] = newcount

        for key in other.keys():
            if key not in self and other.get(key, 0) > 0:
                result[key] = other.get(key, 0)

        return result^

    def __ior__(mut self, other: Self):
        """Union: keep all elements with the maximum count.

        Args:
            other: The other `Counter` to union with.
        """
        for key in other.keys():
            var newcount = max(self.get(key, 0), other.get(key, 0))
            if newcount > 0:
                self[key] = newcount

    def _keep_positive(mut self):
        """Remove zero and negative counts from the `Counter`."""
        for key in self.keys():
            if self.get(key, 0) <= 0:
                try:
                    var key_copy = key.copy()  # Copy due to incorrect origins.
                    _ = self.pop(key_copy)
                except:
                    pass  # this should not happen

    # ===------------------------------------------------------------------=== #
    # Unary operators
    # ===------------------------------------------------------------------=== #

    def __pos__(self) -> Self:
        """Return a shallow copy of the `Counter`, stripping non-positive
        counts.

        Returns:
            A shallow copy of the `Counter`.
        """
        var result = Counter[Self.V, Self.H]()
        for item in self.items():
            if item.value > 0:
                result[item.key] = item.value
        return result^

    def __neg__(self) -> Self:
        """Subtract from an empty `Counter`. Strips positive and zero counts,
        and flips the sign on negative counts.

        Returns:
            A new `Counter` with stripped counts and negative counts.
        """
        var result = Counter[Self.V, Self.H]()
        for item in self.items():
            if item.value < 0:
                result[item.key] = -item.value
        return result^

    # ===------------------------------------------------------------------=== #
    # Methods
    # ===------------------------------------------------------------------=== #

    def get(self, value: Self.V) -> Optional[Int]:
        """Get a value from the `Counter`.

        Args:
            value: The value to search for in the `Counter`.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty `Optional`.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter.get("a").or_else(0)) # output: 1
        print(counter.get("d").or_else(0)) # output: 0
        ```
        """
        return self._data.get(value)

    def get(self, value: Self.V, default: Int) -> Int:
        """Get a value from the `Counter`.

        Args:
            value: The value to search for in the `Counter`.
            default: Default count to return.

        Returns:
            A copy of the value if it was present, otherwise default.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter.get("a", default=0)) # output: 1
        print(counter.get("d", default=0)) # output: 0
        ```
        """
        return self._data.get(value, default)

    def pop(mut self, value: Self.V) raises -> Int:
        """Remove a value from the `Counter` by value.

        Args:
            value: The value to remove from the `Counter`.

        Returns:
            The value associated with the key, if it was in the `Counter`.

        Raises:
            "KeyError" if the key was not present in the `Counter`.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        print(counter.get("b").or_else(0)) # output: 1
        try:
            var count = counter.pop("b")
            print(count) # output: 1
            print(counter.get("b").or_else(0)) # output: 0
        except e:
            print(e) # KeyError if the key was not in the counter
        ```
        """
        return self._data.pop(value)

    def pop(mut self, value: Self.V, var default: Int) -> Int:
        """Remove a value from the `Counter` by value.

        Args:
            value: The value to remove from the `Counter`.
            default: Optionally provide a default value to return if the value
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the `Counter`.
            If it wasn't, return the provided default value instead.

        Example:

        ```mojo
        from std.collections import Counter


        var counter = Counter[String].fromkeys(["a", "b", "c"], 1)
        var count = counter.pop("b", default=100)
        print(count) # output: 1
        count = counter.pop("not-a-key", default=0)
        print(count) # output 0
        ```
        """
        return self._data.pop(value, default)

    def keys(
        ref self,
    ) -> _DictKeyIter[Self.V, Int, Self.H, origin_of(self._data)]:
        """Iterate over the `Counter`'s keys as immutable references.

        Returns:
            An iterator of immutable references to the `Counter` keys.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String].fromkeys(["d", "b", "a", "c"], 1)
        var key_list = List[String]()
        for key in counter.keys():
            key_list.append(key)
        sort(key_list[:])
        print(key_list) # output: ['a', 'b', 'c', 'd']
        ```
        """
        return _DictKeyIter(
            _DictEntryIter(
                0,
                0,
                self._data,
            )
        )

    def values(
        ref self,
    ) -> _DictValueIter[Self.V, Int, Self.H, origin_of(self._data)]:
        """Iterate over the `Counter`'s values as references.

        Returns:
            An iterator of references to the `Counter` values.

        Example:

        ```mojo
        from std.collections import Counter

        # Construct `counter`
        var counter = Counter[Int]([1, 2, 3, 1, 2, 1, 1, 1, 2, 5, 2, 9])

        # Find most populous key
        var max_count: Int = Int.MIN
        for count in counter.values():
            if count > max_count:
                max_count = count

        # Max count is the five ones
        print(max_count) # output: 5
        ```
        """
        return _DictValueIter(
            _DictEntryIter(
                0,
                0,
                self._data,
            )
        )

    def items(
        ref self,
    ) -> _DictEntryIter[Self.V, Int, Self.H, origin_of(self._data)]:
        """Iterate over the `Counter`'s entries as immutable references.

        Returns:
            An iterator of immutable references to the `Counter` entries.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
        for count in counter.items():
            print(count.key, count.value)
        # output: 1 5
        # output: 2 4
        ```
        """
        return _DictEntryIter(
            0,
            0,
            self._data,
        )

    def clear(mut self):
        """Remove all elements from the `Counter`.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
        print(counter.total()) # output: 9 (5 ones + 4 twos)
        counter.clear() # Removes both entries
        print(counter.total()) # output: 0
        ```
        """
        self._data.clear()

    def popitem(mut self) raises -> CountTuple[Self.V]:
        """Remove and return an arbitrary (key, value) pair from the `Counter`.
        Useful for destructively iterating over the `Counter`.
        Returns in LIFO order.

        Returns:
            A `CountTuple` containing the key and value of the removed item.

        Raises:
            "KeyError" if the `Counter` is empty.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[String].fromkeys(["a", "b", "c"], 5)
        try:
            var tuple = counter.popitem()
            print(tuple._value, tuple._count)
            # output: probably c 5 since that was last in
        except e:
            print(e) # KeyError if the key was not in the counter
        ```
        """
        var item_ref = self._data.popitem()
        return CountTuple[Self.V](item_ref.key, item_ref.value)

    # Special methods for counter

    def total(self) -> Int:
        """Return the total of all counts in the `Counter`.

        Returns:
            The total of all counts in the `Counter`.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 1, 1, 1, 2, 2])
        print(counter.total()) # output: 9 (5 ones + 4 twos)
        counter.clear() # Removes both entries
        print(counter.total()) # output: 0
        ```
        """
        var total = 0
        for count in self.values():
            total += count
        return total

    def most_common(self, n: Int) -> List[CountTuple[Self.V]]:
        """Return a list of the `n` most common elements and their counts from
        the most common to the least.

        Args:
            n: The number of most common elements to return.

        Returns:
            A list of the `n` most common elements and their counts. If the
            counter has fewer than `n` unique elements, all of them are
            returned; if `n` is negative, the list is empty.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
        for tuple in counter.most_common(2):
            print(tuple._value, tuple._count)
            # output: 1 5
            # output: 2 4
        ```
        """
        var items: List[CountTuple[Self.V]] = List[CountTuple[Self.V]]()
        for item in self.items():
            var t = CountTuple[Self.V](item.key, item.value)
            items.append(t^)

        def comparator(a: CountTuple[Self.V], b: CountTuple[Self.V]) -> Bool:
            return a < b

        sort(items, comparator)
        items.shrink(max(0, min(n, len(items))))
        return items^

    def elements(self) -> List[Self.V]:
        """Return an iterator over elements repeating each as many times as its
        count.

        Returns:
            An iterator over the elements in the `Counter`.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3, 1, 1, 1, 6, 6, 2, 2, 7])
        print(counter.elements())
        # output: [1, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 6, 6, 7]
        ```
        """
        var elements: List[Self.V] = List[Self.V]()
        for item in self.items():
            for _ in range(item.value):
                elements.append(item.key.copy())
        return elements^

    def update(mut self, other: Self):
        """Update the `Counter`, like `Dict.update()` but add counts instead of
        replacing them.

        Args:
            other: The `Counter` to update this `Counter` with.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(counter[1]) # output: 2
        counter.update(other)
        print(counter[1]) # output: 12
        ```
        """
        for item in other.items():
            self._data.setdefault(item.key.copy(), 0) += item.value

    def subtract(mut self, other: Self):
        """Subtract counts. Both inputs and outputs may be zero or negative.

        Args:
            other: The `Counter` to subtract from this `Counter`.

        Example:

        ```mojo
        from std.collections import Counter

        var counter = Counter[Int]([1, 2, 1, 2, 3, 3, 3])
        var other = Counter[Int].fromkeys([1, 2, 3], 10)
        print(counter[1]) # output: 2
        counter.subtract(other)
        print(counter[1]) # output: -8
        ```
        """
        for item in other.items():
            self._data.setdefault(item.key.copy(), 0) -= item.value


struct CountTuple[V: KeyElement & Copyable & Deinitable](Comparable, Copyable):
    """A tuple representing a value and its count in a `Counter`.

    Parameters:
        V: The value in the `Counter`.
    """

    # Fields
    var _value: Self.V
    """ The value in the `Counter`."""
    var _count: Int
    """ The count of the value in the `Counter`."""

    # ===------------------------------------------------------------------=== #
    # Life cycle methods
    # ===------------------------------------------------------------------=== #

    def __init__(out self, value: Self.V, count: Int):
        """Create a new `CountTuple`.

        Args:
            value: The value in the `Counter`.
            count: The count of the value in the `Counter`.
        """
        self._value = value.copy()
        self._count = Int(count)

    # ===------------------------------------------------------------------=== #
    # Operator dunders
    # ===------------------------------------------------------------------=== #

    def __lt__(self, other: Self) -> Bool:
        """Compare two `CountTuple`s by count, then by value.

        Args:
            other: The other `CountTuple` to compare to.

        Returns:
            `True` if this `CountTuple` is less than the other, `False`
            otherwise.
        """
        return self._count > other._count

    def __eq__(self, other: Self) -> Bool:
        """Compare two `CountTuple`s for equality.

        Args:
            other: The other `CountTuple` to compare to.

        Returns:
            `True` if the two `CountTuple`s are equal, `False` otherwise.
        """
        return self._count == other._count

    @always_inline
    def __getitem__(self, idx: Int) -> Variant[Self.V, Int]:
        """Get an element in the `CountTuple`.

        Args:
            idx: The element to return.

        Returns:
            The value if `idx` is `0` and the count if `idx` is `1`.
        """
        assert 0 <= idx <= 1, "index must be within bounds"
        if idx == 0:
            return self._value.copy()
        else:
            return self._count
