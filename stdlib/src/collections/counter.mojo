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

from collections.dict import Dict, _DictKeyIter, _DictValueIter, _DictEntryIter


struct Counter[V: KeyElement](
    #    Sized,
    #    CollectionElement,
    #    Boolable
):
    """A container for counting hashable items.

    The value type must be specified statically, unlike a Python
    Counter, which can accept arbitrary value types.

    The value type must implement the `KeyElement` trait, as its values are
    stored in the dictionary as keys. `KeyElement` includes
    `Movable`, `Hashable`, and `EqualityComparable`.

    Usage:

    ```mojo
    from collections import Counter
    var c = Counter[String](["a", "a", "a", "b", "b", "c", "d", "c", "c"])
    print(c["a"]) # prints 3
    print(c["b"]) # prints 2
    ```

    Parameters:
        V: The value type to be counted. Currently must be KeyElement.
    """

    var _data: Dict[V, Int]

    def __init__(inout self):
        """Create a new, empty Counter object.

        Usage:
        ```mojo
        c = Counter()
        ```
        """
        self._data = Dict[V, Int]()

    # TODO: Change List to Iterable when it is supported in Mojo
    def __init__(inout self, items: List[V]):
        """Create a from an input iterable.

        Args:
            items: A list of items to count.

        Usage:
        ```mojo
        c = Counter(['g', 'a', 't', 't', 'a', 'c', 'a'])
        ```
        """
        self._data = Dict[V, Int]()
        for item_ref in items:
            var item = item_ref[]
            self._data[item] = self._data.get(item, 0) + 1

    def __getitem__(self, key: V) -> Int:
        """Get the count of a key.

        Args:
            key: The key to get the count of.

        Returns:
            The count of the key.
        """
        return self._data.get(key, 0)

    fn __setitem__(inout self, value: V, count: Int):
        """Set a value in the keyword Counter by key.

        Args:
            value: The value to associate with the specified count.
            count: The count to store in the Counter.
        """
        self._data[value] = count

    fn __iter__(
        self: Reference[Self, _, _]
    ) -> _DictKeyIter[V, Int, self.is_mutable, self.lifetime]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the Counter values.
        """
        return self[]._data.__iter__()

    fn __len__(self) -> Int:
        """The number of elements currently stored in the Counter."""
        return self._data.size

    fn get(self, value: V) -> Optional[V]:
        """Get a value from the counter.

        Args:
            value: The value to search for in the Counter.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self._data.get(value)

    fn get(self, value: V, default: Int) -> Int:
        """Get a value from the Counter.

        Args:
            value: The value to search for in the counter.
            default: Default count to return.

        Returns:
            A copy of the value if it was present, otherwise default.
        """
        return self._data.find(value).or_else(default)

    fn pop(
        inout self, value: V, owned default: Optional[Int] = None
    ) raises -> Int:
        """Remove a value from the Counter by value.

        Args:
            value: The value to remove from the Counter.
            default: Optionally provide a default value to return if the value
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the Counter.
            If it wasn't, return the provided default value instead.

        Raises:
            "KeyError" if the key was not present in the Counter and no
            default value was provided.
        """
        return self._data.pop(value, default)

    fn keys(
        self: Reference[Self, _, _]
    ) -> _DictKeyIter[V, Int, self.is_mutable, self.lifetime]:
        """Iterate over the Counter's keys as immutable references.

        Returns:
            An iterator of immutable references to the Counter keys.
        """
        return self[]._data.keys()

    fn values(
        self: Reference[Self, _, _]
    ) -> _DictValueIter[V, Int, self.is_mutable, self.lifetime]:
        """Iterate over the Counter's values as references.

        Returns:
            An iterator of references to the Counter values.
        """
        return self[]._data.values()

    fn items(
        self: Reference[Self, _, _]
    ) -> _DictEntryIter[V, Int, self.is_mutable, self.lifetime]:
        """Iterate over the dict's entries as immutable references.

        Returns:
            An iterator of immutable references to the Counter entries.
        """
        return self[]._data.items()

    fn update(inout self, other: Self, /):
        """Update the Counter with the value/count pairs from other, overwriting existing keys.
        The argument must be positional only.

        Args:
            other: The Counter to update from.
        """
        self._data.update(other._data)

    fn clear(inout self):
        """Remove all elements from the Counter."""
        self._data.clear()

    # Special methods for counter

    fn total(self) -> Int:
        """Return the total of all counts in the Counter.

        Returns:
            The total of all counts in the Counter.
        """
        var total = 0
        for count_ref in self._data.values():
            total += count_ref[]
        return total
