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
"""Defines `Dict`, a collection that stores key-value pairs.

Dict provides an efficient, O(1) amortized
average-time complexity for insert, lookup, and removal of dictionary elements.
Its implementation closely mirrors Python's `dict` implementation:

- Performance and size are heavily optimized for small dictionaries, but can
  scale to large dictionaries.

- Insertion order is implicitly preserved. Iteration over keys, values, and
  items have a deterministic order based on insertion.

Key elements must implement the `KeyElement` trait, which encompasses
Movable, Hashable, and EqualityComparable. It also includes CollectionElement
and Copyable until we push references through the standard library types.

Value elements must be CollectionElements for a similar reason. Both key and
value types must always be Movable so we can resize the dictionary as it grows.

See the `Dict` docs for more details.
"""
from sys.ffi import OpaquePointer

from bit import is_power_of_two
from builtin.value import StringableCollectionElement
from memory import UnsafePointer, bitcast, memcpy

from .optional import Optional


trait KeyElement(CollectionElement, Hashable, EqualityComparable):
    """A trait composition for types which implement all requirements of
    dictionary keys. Dict keys must minimally be Movable, Hashable,
    and EqualityComparable for a hash map. Until we have references
    they must also be copyable."""

    pass


trait RepresentableKeyElement(KeyElement, Representable):
    """A trait composition for types which implement all requirements of
    dictionary keys and Stringable."""

    pass


@value
struct _DictEntryIter[
    dict_mutability: Bool, //,
    K: KeyElement,
    V: CollectionElement,
    dict_origin: Origin[dict_mutability],
    forward: Bool = True,
]:
    """Iterator over immutable DictEntry references.

    Parameters:
        dict_mutability: Whether the reference to the dictionary is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        dict_origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    var index: Int
    var seen: Int
    var src: Pointer[Dict[K, V], dict_origin]

    fn __init__(
        mut self, index: Int, seen: Int, ref [dict_origin]dict: Dict[K, V]
    ):
        self.index = index
        self.seen = seen
        self.src = Pointer.address_of(dict)

    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(
        mut self,
    ) -> Pointer[DictEntry[K, V], __origin_of(self.src[]._entries[0].value())]:
        while True:
            var opt_entry_ref = Pointer.address_of(
                self.src[]._entries[self.index]
            )

            @parameter
            if forward:
                self.index += 1
            else:
                self.index -= 1

            if opt_entry_ref[]:
                self.seen += 1
                return Pointer.address_of(opt_entry_ref[].value())

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return len(self.src[]) - self.seen


@value
struct _DictKeyIter[
    dict_mutability: Bool, //,
    K: KeyElement,
    V: CollectionElement,
    dict_origin: Origin[dict_mutability],
    forward: Bool = True,
]:
    """Iterator over immutable Dict key references.

    Parameters:
        dict_mutability: Whether the reference to the vector is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        dict_origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias dict_entry_iter = _DictEntryIter[K, V, dict_origin, forward]

    var iter: Self.dict_entry_iter

    fn __iter__(self) -> Self:
        return self

    fn __next__(
        mut self,
    ) -> Pointer[K, __origin_of(self.iter.__next__()[].key)]:
        return Pointer.address_of(self.iter.__next__()[].key)

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return self.iter.__len__()


@value
struct _DictValueIter[
    dict_mutability: Bool, //,
    K: KeyElement,
    V: CollectionElement,
    dict_origin: Origin[dict_mutability],
    forward: Bool = True,
]:
    """Iterator over Dict value references. These are mutable if the dict
    is mutable.

    Parameters:
        dict_mutability: Whether the reference to the vector is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        dict_origin: The origin of the List
        forward: The iteration direction. `False` is backwards.
    """

    alias ref_type = Pointer[V, dict_origin]

    var iter: _DictEntryIter[K, V, dict_origin, forward]

    fn __iter__(self) -> Self:
        return self

    fn __reversed__(self) -> _DictValueIter[K, V, dict_origin, False]:
        var src = self.iter.src
        return _DictValueIter(
            _DictEntryIter[K, V, dict_origin, False](
                src[]._reserved() - 1, 0, src
            )
        )

    fn __next__(mut self) -> Self.ref_type:
        var entry_ref = self.iter.__next__()
        # Cast through a pointer to grant additional mutability because
        # _DictEntryIter.next erases it.
        return Self.ref_type.address_of(
            UnsafePointer.address_of(entry_ref[].value).bitcast[
                origin=dict_origin
            ]()[]
        )

    @always_inline
    fn __has_next__(self) -> Bool:
        return self.__len__() > 0

    fn __len__(self) -> Int:
        return self.iter.__len__()


@value
struct DictEntry[K: KeyElement, V: CollectionElement](
    CollectionElement, CollectionElementNew
):
    """Store a key-value pair entry inside a dictionary.

    Parameters:
        K: The key type of the dict. Must be Hashable+EqualityComparable.
        V: The value type of the dict.
    """

    var hash: UInt64
    """`key.__hash__()`, stored so hashing isn't re-computed during dict lookup."""
    var key: K
    """The unique key for the entry."""
    var value: V
    """The value associated with the key."""

    fn __init__(out self, owned key: K, owned value: V):
        """Create an entry from a key and value, computing the hash.

        Args:
            key: The key of the entry.
            value: The value of the entry.
        """
        self.hash = hash(key)
        self.key = key^
        self.value = value^

    fn __init__(out self, *, other: Self):
        """Copy an existing entry.

        Args:
            other: The existing entry to copy.
        """
        self.hash = other.hash
        self.key = other.key
        self.value = other.value

    fn reap_value(owned self) -> V as out:
        """Take the value from an owned entry.

        Returns:
            The value of the entry.
        """
        out = self.value^
        __disable_del self


alias _EMPTY = -1
alias _REMOVED = -2


struct _DictIndex:
    """A compact dict-index type. Small dict indices are compressed
    to smaller integer types to use less memory.

    _DictIndex doesn't store its own size, so the size must be passed in to
    its indexing methods.

    Ideally this could be type-parameterized so that the size checks don't
    need to be performed at runtime, but I couldn't find a way to express
    this in the current type system.
    """

    var data: OpaquePointer

    @always_inline
    @implicit
    fn __init__(out self, reserved: Int):
        if reserved <= 128:
            var data = UnsafePointer[Int8].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[NoneType]()
        elif reserved <= 2**16 - 2:
            var data = UnsafePointer[Int16].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[NoneType]()
        elif reserved <= 2**32 - 2:
            var data = UnsafePointer[Int32].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[NoneType]()
        else:
            var data = UnsafePointer[Int64].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[NoneType]()

    fn copy(self, reserved: Int) -> Self:
        var index = Self(reserved)
        if reserved <= 128:
            var data = self.data.bitcast[Int8]()
            var new_data = index.data.bitcast[Int8]()
            memcpy(new_data, data, reserved)
        elif reserved <= 2**16 - 2:
            var data = self.data.bitcast[Int16]()
            var new_data = index.data.bitcast[Int16]()
            memcpy(new_data, data, reserved)
        elif reserved <= 2**32 - 2:
            var data = self.data.bitcast[Int32]()
            var new_data = index.data.bitcast[Int32]()
            memcpy(new_data, data, reserved)
        else:
            var data = self.data.bitcast[Int64]()
            var new_data = index.data.bitcast[Int64]()
            memcpy(new_data, data, reserved)
        return index^

    fn __moveinit__(out self, owned existing: Self):
        self.data = existing.data

    fn get_index(self, reserved: Int, slot: UInt64) -> Int:
        if reserved <= 128:
            var data = self.data.bitcast[Int8]()
            return int(data.load(slot & (reserved - 1)))
        elif reserved <= 2**16 - 2:
            var data = self.data.bitcast[Int16]()
            return int(data.load(slot & (reserved - 1)))
        elif reserved <= 2**32 - 2:
            var data = self.data.bitcast[Int32]()
            return int(data.load(slot & (reserved - 1)))
        else:
            var data = self.data.bitcast[Int64]()
            return int(data.load(slot & (reserved - 1)))

    fn set_index(mut self, reserved: Int, slot: UInt64, value: Int):
        if reserved <= 128:
            var data = self.data.bitcast[Int8]()
            return data.store(slot & (reserved - 1), value)
        elif reserved <= 2**16 - 2:
            var data = self.data.bitcast[Int16]()
            return data.store(slot & (reserved - 1), value)
        elif reserved <= 2**32 - 2:
            var data = self.data.bitcast[Int32]()
            return data.store(slot & (reserved - 1), value)
        else:
            var data = self.data.bitcast[Int64]()
            return data.store(slot & (reserved - 1), value)

    fn __del__(owned self):
        self.data.free()


struct Dict[K: KeyElement, V: CollectionElement](
    Sized, CollectionElement, CollectionElementNew, Boolable
):
    """A container that stores key-value pairs.

    The key type and value type must be specified statically, unlike a Python
    dictionary, which can accept arbitrary key and value types.

    The key type must implement the `KeyElement` trait, which encompasses
    `Movable`, `Hashable`, and `EqualityComparable`. It also includes
    `CollectionElement` and `Copyable` until we have references.

    The value type must implement the `CollectionElement` trait.

    Examples:

    ```mojo
    from collections import Dict
    var d = Dict[String, Int]()
    d["a"] = 1
    d["b"] = 2
    print(len(d))      # prints 2
    print(d["a"])      # prints 1
    print(d.pop("b"))  # prints 2
    print(len(d))      # prints 1
    ```

    Parameters:
        K: The type of the dictionary key. Must be Hashable and EqualityComparable
           so we can find the key in the map.
        V: The value type of the dictionary. Currently must be CollectionElement.
    """

    # Implementation:
    #
    # `Dict` provides an efficient, O(1) amortized average-time complexity for
    # insert, lookup, and removal of dictionary elements.
    #
    # Its implementation closely mirrors Python's `dict` implementation:
    #
    # - Performance and size are heavily optimized for small dictionaries, but can
    #     scale to large dictionaries.
    # - Insertion order is implicitly preserved. Once `__iter__` is implemented
    #     it will return a deterministic order based on insertion.
    # - To achieve this, elements are stored in a dense array. Inserting a new
    #     element will append it to the entry list, and then that index will be stored
    #     in the dict's index hash map. Removing an element updates that index to
    #     a special `REMOVED` value for correctness of the probing sequence, and
    #     the entry in the entry list is marked as removed and the relevant data is freed.
    #     The entry can be re-used to insert a new element, but it can't be reset to
    #     `EMPTY` without compacting or resizing the dictionary.
    # - The index probe sequence is taken directly from Python's dict implementation:
    #
    #     ```mojo
    #     var slot = hash(key) % self._reserved
    #     var perturb = hash(key)
    #     while True:
    #         check_slot(slot)
    #         alias PERTURB_SHIFT = 5
    #         perturb >>= PERTURB_SHIFT
    #         slot = ((5 * slot) + perturb + 1) % self._reserved
    #     ```
    #
    # - Similarly to Python, we aim for a maximum load of 2/3, after which we resize
    #     to a larger dictionary.
    # - In the case where many entries are being added and removed, the dictionary
    #     can fill up with `REMOVED` entries without being resized. In this case
    #     we will eventually "compact" the dictionary and shift entries towards
    #     the beginning to free new space while retaining insertion order.
    #
    # Key elements must implement the `KeyElement` trait, which encompasses
    # Movable, Hashable, and EqualityComparable. It also includes CollectionElement
    # and Copyable until we have references.
    #
    # Value elements must be CollectionElements for a similar reason. Both key and
    # value types must always be Movable so we can resize the dictionary as it grows.
    #
    # Without conditional trait conformance, making a `__str__` representation for
    # Dict is tricky. We'd need to add `Stringable` to the requirements for keys
    # and values. This may be worth it.
    #
    # Invariants:
    #
    # - size = 2^k for integer k:
    #     This allows for faster entry slot lookups, since modulo can be
    #     optimized to a bit shift for powers of 2.
    #
    # - size <= 2/3 * _reserved
    #     If size exceeds this invariant, we double the size of the dictionary.
    #     This is the maximal "load factor" for the dict. Higher load factors
    #     trade off higher memory utilization for more frequent worst-case lookup
    #     performance. Lookup is O(n) in the worst case and O(1) in average case.
    #
    # - _n_entries <= 3/4 * _reserved
    #     If _n_entries exceeds this invariant, we compact the dictionary, retaining
    #     the insertion order while resetting _n_entries = size.
    #     As elements are removed, they retain marker entries for the probe sequence.
    #     The average case miss lookup (ie. `contains` check on a key not in the dict)
    #     is O(_reserved  / (1 + _reserved - _n_entries)). At `(k-1)/k` this
    #     approaches `k` and is therefore O(1) average case. However, we want it to
    #     be _larger_ than the load factor: since `compact` is O(n), we don't
    #     don't churn and compact on repeated insert/delete, and instead amortize
    #     compaction cost to O(1) amortized cost.

    # Fields
    alias EMPTY = _EMPTY
    alias REMOVED = _REMOVED
    alias _initial_reservation = 8

    var size: Int
    """The number of elements currently stored in the dict."""
    var _n_entries: Int
    """The number of entries currently allocated."""

    var _index: _DictIndex

    # We use everything available in the list. Which means that
    # len(self._entries) == self._entries.capacity == self._reserved()
    var _entries: List[Optional[DictEntry[K, V]]]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __init__(out self):
        """Initialize an empty dictiontary."""
        self.size = 0
        self._n_entries = 0
        self._entries = Self._new_entries(Self._initial_reservation)
        self._index = _DictIndex(len(self._entries))

    @always_inline
    fn __init__(out self, *, power_of_two_initial_capacity: Int):
        """Initialize an empty dictiontary with a pre-reserved initial capacity.

        Args:
            power_of_two_initial_capacity: At least 8, has to be a power of two.

        Example usage:

        ```mojo
        from collections import Dict

        var x = Dict[Int, Int](power_of_two_initial_capacity = 1024)
        # Insert (2/3 of 1024) entries without reallocation.
        ```

        """
        debug_assert(
            bit.is_power_of_two(power_of_two_initial_capacity)
            and power_of_two_initial_capacity >= 8,
            "power_of_two_initial_capacity need to be >=8 and a power of two",
        )
        self.size = 0
        self._n_entries = 0
        self._entries = Self._new_entries(power_of_two_initial_capacity)
        self._index = _DictIndex(len(self._entries))

    # TODO: add @property when Mojo supports it to make
    # it possible to do `self._reserved`.
    @always_inline
    fn _reserved(self) -> Int:
        return len(self._entries)

    @always_inline
    fn __init__(out self, *, other: Self):
        """Copy an existing dictiontary.

        Args:
            other: The existing dict.
        """
        self.size = other.size
        self._n_entries = other._n_entries
        self._index = other._index.copy(other._reserved())
        self._entries = other._entries

    @staticmethod
    fn fromkeys(keys: List[K, *_], value: V) -> Self:
        """Create a new dictionary with keys from list and values set to value.

        Args:
            keys: The keys to set.
            value: The value to set.

        Returns:
            The new dictionary.
        """
        var my_dict = Dict[K, V]()
        for key in keys:
            my_dict[key[]] = value
        return my_dict

    @staticmethod
    fn fromkeys(
        keys: List[K, *_], value: Optional[V] = None
    ) -> Dict[K, Optional[V]]:
        """Create a new dictionary with keys from list and values set to value.

        Args:
            keys: The keys to set.
            value: The value to set.

        Returns:
            The new dictionary.
        """
        return Dict[K, Optional[V]].fromkeys(keys, value)

    fn __copyinit__(out self, existing: Self):
        """Copy an existing dictiontary.

        Args:
            existing: The existing dict.
        """
        self.size = existing.size
        self._n_entries = existing._n_entries
        self._index = existing._index.copy(existing._reserved())
        self._entries = existing._entries

    fn __moveinit__(out self, owned existing: Self):
        """Move data of an existing dict into a new one.

        Args:
            existing: The existing dict.
        """
        self.size = existing.size
        self._n_entries = existing._n_entries
        self._index = existing._index^
        self._entries = existing._entries^

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    fn __getitem__(
        self, key: K
    ) raises -> ref [self._entries[0].value().value] Self.V:
        """Retrieve a value out of the dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            "KeyError" if the key isn't present.
        """
        return self._find_ref(key)

    fn __setitem__(mut self, owned key: K, owned value: V):
        """Set a value in the dictionary by key.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._insert(key^, value^)

    fn __contains__(self, key: K) -> Bool:
        """Check if a given key is in the dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if there key exists in the dictionary, False otherwise.
        """
        return self.find(key).__bool__()

    fn __iter__(ref self) -> _DictKeyIter[K, V, __origin_of(self)]:
        """Iterate over the dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return _DictKeyIter(_DictEntryIter(0, 0, self))

    fn __reversed__(ref self) -> _DictKeyIter[K, V, __origin_of(self), False]:
        """Iterate backwards over the dict keys, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the dict keys.
        """
        return _DictKeyIter(
            _DictEntryIter[forward=False](self._reserved() - 1, 0, self)
        )

    fn __or__(self, other: Self) -> Self:
        """Merge self with other and return the result as a new dict.

        Args:
            other: The dictionary to merge with.

        Returns:
            The result of the merge.
        """
        var result = Dict(other=self)
        result.update(other)
        return result^

    fn __ior__(mut self, other: Self):
        """Merge self with other in place.

        Args:
            other: The dictionary to merge with.
        """
        self.update(other)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    fn __len__(self) -> Int:
        """The number of elements currently stored in the dictionary.

        Returns:
            The number of elements currently stored in the dictionary.
        """
        return self.size

    fn __bool__(self) -> Bool:
        """Check if the dictionary is empty or not.

        Returns:
            `False` if the dictionary is empty, `True` if there is at least one element.
        """
        return len(self).__bool__()

    @no_inline
    fn __str__[
        T: RepresentableKeyElement, U: RepresentableCollectionElement, //
    ](self: Dict[T, U]) -> String:
        """Returns a string representation of a `Dict`.

        Note that since we can't condition methods on a trait yet,
        the way to call this method is a bit special. Here is an example below:

        ```mojo
        from collections import Dict

        var my_dict = Dict[Int, Float64]()
        my_dict[1] = 1.1
        my_dict[2] = 2.2
        dict_as_string = my_dict.__str__()
        print(dict_as_string)
        # prints "{1: 1.1, 2: 2.2}"
        ```

        When the compiler supports conditional methods, then a simple `str(my_dict)` will
        be enough.

        Note that both they keys and values' types must implement the `__repr__()` method
        for this to work. See the `Representable` trait for more information.

        Parameters:
            T: The type of the keys in the Dict. Must implement the
              traits `Representable` and `KeyElement`.
            U: The type of the values in the Dict. Must implement the
                traits `Representable` and `CollectionElement`.

        Returns:
            A string representation of the Dict.
        """
        var minimum_capacity = self._minimum_size_of_string_representation()
        var string_buffer = List[UInt8](capacity=minimum_capacity)
        string_buffer.append(0)  # Null terminator
        var result = String(string_buffer^)
        result += "{"

        var i = 0
        for key_value in self.items():
            result += repr(key_value[].key) + ": " + repr(key_value[].value)
            if i < len(self) - 1:
                result += ", "
            i += 1
        result += "}"
        return result

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    fn _minimum_size_of_string_representation(self) -> Int:
        # we do a rough estimation of the minimum number of chars that we'll see
        # in the string representation, we assume that str(key) and str(value)
        # will be both at least one char.
        return (
            2  # '{' and '}'
            + len(self) * 6  # str(key), str(value) ": " and ", "
            - 2  # remove the last ", "
        )

    fn find(self, key: K) -> Optional[V]:
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        try:  # TODO(MOCO-604): push usage through
            return self._find_ref(key)
        except:
            return None

    # TODO(MOCO-604): Return Optional[Pointer] instead of raising
    fn _find_ref(
        ref self, key: K
    ) raises -> ref [self._entries[0].value().value] Self.V:
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a reference to the value if it is
            present, otherwise an empty Optional.
        """
        var hash = hash(key)
        var found: Bool
        var slot: UInt64
        var index: Int
        found, slot, index = self._find_index(hash, key)
        if found:
            var entry = Pointer.address_of(self._entries[index])
            debug_assert(entry[].__bool__(), "entry in index must be full")
            return entry[].value().value
        raise "KeyError"

    fn get(self, key: K) -> Optional[V]:
        """Get a value from the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self.find(key)

    fn get(self, key: K, default: V) -> V:
        """Get a value from the dictionary by key.

        Args:
            key: The key to search for in the dictionary.
            default: Default value to return.

        Returns:
            A copy of the value if it was present, otherwise default.
        """
        return self.find(key).or_else(default)

    fn pop(mut self, key: K, owned default: V) -> V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.
            default: A default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.
        """
        try:
            return self.pop(key)
        except:
            return default

    fn pop(mut self, key: K) raises -> V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.

        Returns:
            The value associated with the key, if it was in the dictionary.
            Raises otherwise.

        Raises:
            "KeyError" if the key was not present in the dictionary.
        """
        var hash = hash(key)
        var found: Bool
        var slot: UInt64
        var index: Int
        found, slot, index = self._find_index(hash, key)
        if found:
            self._set_index(slot, Self.REMOVED)
            var entry = Pointer.address_of(self._entries[index])
            debug_assert(entry[].__bool__(), "entry in index must be full")
            var entry_value = entry[].unsafe_take()
            entry[] = None
            self.size -= 1
            return entry_value^.reap_value()
        raise "KeyError"

    fn popitem(mut self) raises -> DictEntry[K, V]:
        """Remove and return a (key, value) pair from the dictionary. Pairs are returned in LIFO order.
        popitem() is useful to destructively iterate over a dictionary, as often used in set algorithms.
        If the dictionary is empty, calling popitem() raises a KeyError.

        Args: None

        Returns:
            Last dictionary item

        Raises:
            "KeyError" if the dictionary is empty.
        """

        var key = Optional[K](None)
        var val = Optional[V](None)

        for item in reversed(self.items()):
            key = Optional(item[].key)
            val = Optional(item[].value)
            break

        if key:
            _ = self.pop(key.value())
            return DictEntry[K, V](key.value(), val.value())

        raise "KeyError: popitem(): dictionary is empty"

    fn keys(ref self) -> _DictKeyIter[K, V, __origin_of(self)]:
        """Iterate over the dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return Self.__iter__(self)

    fn values(ref self) -> _DictValueIter[K, V, __origin_of(self)]:
        """Iterate over the dict's values as references.

        Returns:
            An iterator of references to the dictionary values.
        """
        return _DictValueIter(_DictEntryIter(0, 0, self))

    fn items(ref self) -> _DictEntryIter[K, V, __origin_of(self)]:
        """Iterate over the dict's entries as immutable references.

        These can't yet be unpacked like Python dict items, but you can
        access the key and value as attributes ie.

        ```mojo
        from collections import Dict

        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for e in my_dict.items():
            print(e[].key, e[].value)
        ```

        Returns:
            An iterator of immutable references to the dictionary entries.
        """
        return _DictEntryIter(0, 0, self)

    fn update(mut self, other: Self, /):
        """Update the dictionary with the key/value pairs from other, overwriting existing keys.
        The argument must be positional only.

        Args:
            other: The dictionary to update from.
        """
        for entry in other.items():
            self[entry[].key] = entry[].value

    fn clear(mut self):
        """Remove all elements from the dictionary."""
        self.size = 0
        self._n_entries = 0
        self._entries = Self._new_entries(Self._initial_reservation)
        self._index = _DictIndex(self._reserved())

    fn setdefault(
        mut self, key: K, owned default: V
    ) raises -> ref [self._find_ref(key)] V:
        """Get a value from the dictionary by key, or set it to a default if it doesn't exist.

        Args:
            key: The key to search for in the dictionary.
            default: The default value to set if the key is not present.

        Returns:
            The value associated with the key, or the default value if it wasn't present.
        """
        try:
            return self._find_ref(key)
        except KeyError:
            self[key] = default^
            return self._find_ref(key)

    @staticmethod
    @always_inline
    fn _new_entries(reserve_at_least: Int) -> List[Optional[DictEntry[K, V]]]:
        var entries = List[Optional[DictEntry[K, V]]](capacity=reserve_at_least)
        # We have memory available, we'll use everything.
        for i in range(entries.capacity):
            entries.append(None)
        return entries

    fn _insert(mut self, owned key: K, owned value: V):
        self._insert(DictEntry[K, V](key^, value^))

    fn _insert[
        safe_context: Bool = False
    ](mut self, owned entry: DictEntry[K, V]):
        @parameter
        if not safe_context:
            self._maybe_resize()
        var found: Bool
        var slot: UInt64
        var index: Int
        found, slot, index = self._find_index(entry.hash, entry.key)

        self._entries[index] = entry^
        if not found:
            self._set_index(slot, index)
            self.size += 1
            self._n_entries += 1

    fn _get_index(self, slot: UInt64) -> Int:
        return self._index.get_index(self._reserved(), slot)

    fn _set_index(mut self, slot: UInt64, index: Int):
        return self._index.set_index(self._reserved(), slot, index)

    fn _next_index_slot(self, mut slot: UInt64, mut perturb: UInt64):
        alias PERTURB_SHIFT = 5
        perturb >>= PERTURB_SHIFT
        slot = ((5 * slot) + int(perturb + 1)) & (self._reserved() - 1)

    fn _find_empty_index(self, hash: UInt64) -> UInt64:
        var slot = hash & (self._reserved() - 1)
        var perturb = hash
        while True:
            var index = self._get_index(slot)
            if index == Self.EMPTY:
                return slot
            self._next_index_slot(slot, perturb)

    fn _find_index(self, hash: UInt64, key: K) -> (Bool, UInt64, Int):
        # Return (found, slot, index)
        var slot = hash & (self._reserved() - 1)
        var perturb = hash
        while True:
            var index = self._get_index(slot)
            if index == Self.EMPTY:
                return (False, slot, self._n_entries)
            elif index == Self.REMOVED:
                pass
            else:
                var entry = self._entries[index]
                debug_assert(entry.__bool__(), "entry in index must be full")
                if hash == entry.value().hash and key == entry.value().key:
                    return (True, slot, index)
            self._next_index_slot(slot, perturb)

    fn _over_load_factor(self) -> Bool:
        return 3 * self.size > 2 * self._reserved()

    fn _over_compact_factor(self) -> Bool:
        return 4 * self._n_entries > 3 * self._reserved()

    fn _maybe_resize(mut self):
        if not self._over_load_factor():
            if self._over_compact_factor():
                self._compact()
            return
        var _reserved = self._reserved() * 2
        self.size = 0
        self._n_entries = 0
        var old_entries = self._entries^
        self._entries = self._new_entries(_reserved)
        self._index = _DictIndex(self._reserved())

        for i in range(len(old_entries)):
            var entry = old_entries[i]
            if entry:
                self._insert[safe_context=True](entry.unsafe_take())

    fn _compact(mut self):
        self._index = _DictIndex(self._reserved())
        var right = 0
        for left in range(self.size):
            while not self._entries[right]:
                right += 1
                debug_assert(right < self._reserved(), "Invalid dict state")
            var entry = self._entries[right]
            debug_assert(entry.__bool__(), "Logic error")
            var slot = self._find_empty_index(entry.value().hash)
            self._set_index(slot, left)
            if left != right:
                self._entries[left] = entry.unsafe_take()
                entry = None
            right += 1

        self._n_entries = self.size


struct OwnedKwargsDict[V: CollectionElement](
    Sized, CollectionElement, CollectionElementNew
):
    """Container used to pass owned variadic keyword arguments to functions.

    This type mimics the interface of a dictionary with `String` keys, and
    should be usable more-or-less like a dictionary. Notably, however, this type
    should not be instantiated directly by users.

    Parameters:
        V: The value type of the dictionary. Currently must be CollectionElement.
    """

    # Fields
    alias key_type = String

    var _dict: Dict[Self.key_type, V]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    fn __init__(out self):
        """Initialize an empty keyword dictionary."""
        self._dict = Dict[Self.key_type, V]()

    fn __init__(out self, *, other: Self):
        """Copy an existing keyword dictionary.

        Args:
            other: The existing keyword dictionary.
        """
        self._dict = other._dict

    fn __copyinit__(out self, existing: Self):
        """Copy an existing keyword dictionary.

        Args:
            existing: The existing keyword dictionary.
        """
        self._dict = existing._dict

    fn __moveinit__(out self, owned existing: Self):
        """Move data of an existing keyword dictionary into a new one.

        Args:
            existing: The existing keyword dictionary.
        """
        self._dict = existing._dict^

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __getitem__(self, key: Self.key_type) raises -> V:
        """Retrieve a value out of the keyword dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            "KeyError" if the key isn't present.
        """
        return self._dict[key]

    @always_inline
    fn __setitem__(mut self, key: Self.key_type, value: V):
        """Set a value in the keyword dictionary by key.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._dict[key] = value

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn __contains__(self, key: Self.key_type) -> Bool:
        """Check if a given key is in the keyword dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if there key exists in the keyword dictionary, False
            otherwise.
        """
        return key in self._dict

    @always_inline
    fn __len__(self) -> Int:
        """The number of elements currently stored in the keyword dictionary.

        Returns:
            The number of elements currently stored in the keyword dictionary.
        """
        return len(self._dict)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    fn find(self, key: Self.key_type) -> Optional[V]:
        """Find a value in the keyword dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self._dict.find(key)

    @always_inline
    fn pop(mut self, key: self.key_type, owned default: V) -> V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.
            default: A default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.
        """
        return self._dict.pop(key, default^)

    @always_inline
    fn pop(mut self, key: self.key_type) raises -> V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.

        Returns:
            The value associated with the key, if it was in the dictionary.
            Raises otherwise.

        Raises:
            "KeyError" if the key was not present in the dictionary.
        """
        return self._dict.pop(key)

    fn __iter__(
        ref self,
    ) -> _DictKeyIter[Self.key_type, V, __origin_of(self._dict)]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return self._dict.keys()

    fn keys(
        ref self,
    ) -> _DictKeyIter[Self.key_type, V, __origin_of(self._dict)]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return self._dict.keys()

    fn values(
        ref self,
    ) -> _DictValueIter[Self.key_type, V, __origin_of(self._dict)]:
        """Iterate over the keyword dict's values as references.

        Returns:
            An iterator of references to the dictionary values.
        """
        return self._dict.values()

    fn items(
        ref self,
    ) -> _DictEntryIter[Self.key_type, V, __origin_of(self._dict)]:
        """Iterate over the keyword dictionary's entries as immutable references.

        These can't yet be unpacked like Python dict items, but you can
        access the key and value as attributes ie.

        ```mojo
        from collections import Dict

        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for e in my_dict.items():
            print(e[].key, e[].value)
        ```

        Returns:
            An iterator of immutable references to the dictionary entries.
        """

        # TODO(#36448): Use this instead of the current workaround
        # return self[]._dict.items()
        return _DictEntryIter(0, 0, self._dict)

    @always_inline
    fn _insert(mut self, owned key: Self.key_type, owned value: V):
        self._dict._insert(key^, value^)

    @always_inline
    fn _insert(mut self, key: StringLiteral, owned value: V):
        self._insert(String(key), value^)
