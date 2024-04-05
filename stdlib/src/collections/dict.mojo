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
from memory.anypointer import AnyPointer

from .optional import Optional


trait KeyElement(CollectionElement, Hashable, EqualityComparable):
    """A trait composition for types which implement all requirements of
    dictionary keys. Dict keys must minimally be Movable, Hashable,
    and EqualityComparable for a hash map. Until we have references
    they must also be copyable."""

    pass


@value
struct _DictEntryIter[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: __mlir_type.`i1`,
    dict_lifetime: AnyLifetime[dict_mutability].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Iterator over immutable DictEntry references.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        dict_mutability: Whether the reference to the dictionary is mutable.
        dict_lifetime: The lifetime of the List
        address_space: the address_space of the list
    """

    alias imm_dict_lifetime = __mlir_attr[
        `#lit.lifetime.mutcast<`, dict_lifetime, `> : !lit.lifetime<1>`
    ]
    alias ref_type = Reference[
        DictEntry[K, V], __mlir_attr.`0: i1`, Self.imm_dict_lifetime
    ]

    var index: Int
    var seen: Int
    var src: Reference[Dict[K, V], dict_mutability, dict_lifetime]

    fn __iter__(self) -> Self:
        return self

    @always_inline
    fn __next__(inout self) -> Self.ref_type:
        while True:
            debug_assert(self.index < self.src[]._reserved, "dict iter bounds")
            if self.src[]._entries.__get_ref(self.index)[]:
                var opt_entry_ref = self.src[]._entries.__get_ref[
                    __mlir_attr.`0: i1`,
                    Self.imm_dict_lifetime,
                ](self.index)
                self.index += 1
                self.seen += 1
                # Super unsafe, but otherwise we have to do a bunch of super
                # unsafe reference lifetime casting.
                return opt_entry_ref.bitcast_element[DictEntry[K, V]]()
            self.index += 1

    fn __len__(self) -> Int:
        return len(self.src[]) - self.seen


@value
struct _DictKeyIter[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: __mlir_type.`i1`,
    dict_lifetime: AnyLifetime[dict_mutability].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Iterator over immutable Dict key references.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        dict_mutability: Whether the reference to the vector is mutable.
        dict_lifetime: The lifetime of the List
        address_space: The address space of the List
    """

    alias imm_dict_lifetime = __mlir_attr[
        `#lit.lifetime.mutcast<`, dict_lifetime, `> : !lit.lifetime<1>`
    ]
    alias ref_type = Reference[
        K, __mlir_attr.`0: i1`, Self.imm_dict_lifetime, address_space
    ]

    alias dict_entry_iter = _DictEntryIter[
        K, V, dict_mutability, dict_lifetime, address_space
    ]

    var iter: Self.dict_entry_iter

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> Self.ref_type:
        var entry_ref = self.iter.__next__()
        var mlir_ptr = __mlir_op.`lit.ref.to_pointer`(
            Reference(entry_ref[].key).value
        )
        var key_ptr = AnyPointer[
            K, address_space = Self.dict_entry_iter.address_space
        ] {
            value: __mlir_op.`pop.pointer.bitcast`[
                _type = AnyPointer[
                    K, address_space = Self.dict_entry_iter.address_space
                ].pointer_type
            ](mlir_ptr)
        }
        return __mlir_op.`lit.ref.from_pointer`[
            _type = Self.ref_type.mlir_ref_type
        ](key_ptr.value)

    fn __len__(self) -> Int:
        return self.iter.__len__()


@value
struct _DictValueIter[
    K: KeyElement,
    V: CollectionElement,
    dict_mutability: __mlir_type.`i1`,
    dict_lifetime: AnyLifetime[dict_mutability].type,
    address_space: AddressSpace = AddressSpace.GENERIC,
]:
    """Iterator over Dict value references. These are mutable if the dict
    is mutable.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        dict_mutability: Whether the reference to the vector is mutable.
        dict_lifetime: The lifetime of the List
        address_space: The address space of the List
    """

    alias ref_type = Reference[V, dict_mutability, dict_lifetime, address_space]

    var iter: _DictEntryIter[
        K, V, dict_mutability, dict_lifetime, address_space
    ]

    fn __iter__(self) -> Self:
        return self

    fn __next__(inout self) -> Self.ref_type:
        var entry_ref = self.iter.__next__()
        var mlir_ptr = __mlir_op.`lit.ref.to_pointer`(
            Reference(entry_ref[].value).value
        )
        var value_ptr = AnyPointer[V, address_space] {
            value: __mlir_op.`pop.pointer.bitcast`[
                _type = AnyPointer[V, address_space].pointer_type
            ](mlir_ptr)
        }
        return __mlir_op.`lit.ref.from_pointer`[
            _type = Self.ref_type.mlir_ref_type
        ](value_ptr.value)

    fn __len__(self) -> Int:
        return self.iter.__len__()


@value
struct DictEntry[K: KeyElement, V: CollectionElement](CollectionElement):
    """Store a key-value pair entry inside a dictionary.

    Parameters:
        K: The key type of the dict. Must be Hashable+EqualityComparable.
        V: The value type of the dict.
    """

    var hash: Int
    """`key.__hash__()`, stored so hashing isn't re-computed during dict lookup."""
    var key: K
    """The unique key for the entry."""
    var value: V
    """The value associated with the key."""

    fn __init__(inout self, owned key: K, owned value: V):
        """Create an entry from a key and value, computing the hash.

        Args:
            key: The key of the entry.
            value: The value of the entry.
        """
        self.hash = hash(key)
        self.key = key^
        self.value = value^


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

    var data: DTypePointer[DType.invalid]

    @always_inline
    fn __init__(inout self, reserved: Int):
        if reserved <= 128:
            var data = DTypePointer[DType.int8].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()
        elif reserved <= 2**16 - 2:
            var data = DTypePointer[DType.int16].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()
        elif reserved <= 2**32 - 2:
            var data = DTypePointer[DType.int32].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()
        else:
            var data = DTypePointer[DType.int64].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()

    fn copy(self, reserved: Int) -> Self:
        var index = Self(reserved)
        if reserved <= 128:
            var data = self.data.bitcast[DType.int8]()
            var new_data = index.data.bitcast[DType.int8]()
            memcpy(new_data, data, reserved)
        elif reserved <= 2**16 - 2:
            var data = self.data.bitcast[DType.int16]()
            var new_data = index.data.bitcast[DType.int16]()
            memcpy(new_data, data, reserved)
        elif reserved <= 2**32 - 2:
            var data = self.data.bitcast[DType.int32]()
            var new_data = index.data.bitcast[DType.int32]()
            memcpy(new_data, data, reserved)
        else:
            var data = self.data.bitcast[DType.int64]()
            var new_data = index.data.bitcast[DType.int64]()
            memcpy(new_data, data, reserved)
        return index^

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data

    fn get_index(self, reserved: Int, slot: Int) -> Int:
        if reserved <= 128:
            var data = self.data.bitcast[DType.int8]()
            return data.load(slot % reserved).to_int()
        elif reserved <= 2**16 - 2:
            var data = self.data.bitcast[DType.int16]()
            return data.load(slot % reserved).to_int()
        elif reserved <= 2**32 - 2:
            var data = self.data.bitcast[DType.int32]()
            return data.load(slot % reserved).to_int()
        else:
            var data = self.data.bitcast[DType.int64]()
            return data.load(slot % reserved).to_int()

    fn set_index(inout self, reserved: Int, slot: Int, value: Int):
        if reserved <= 128:
            var data = self.data.bitcast[DType.int8]()
            return data.store(slot % reserved, value)
        elif reserved <= 2**16 - 2:
            var data = self.data.bitcast[DType.int16]()
            return data.store(slot % reserved, value)
        elif reserved <= 2**32 - 2:
            var data = self.data.bitcast[DType.int32]()
            return data.store(slot % reserved, value)
        else:
            var data = self.data.bitcast[DType.int64]()
            return data.store(slot % reserved, value)

    fn __del__(owned self):
        self.data.free()


struct Dict[K: KeyElement, V: CollectionElement](Sized, CollectionElement):
    """A container that stores key-value pairs.

    The key type and value type must be specified statically, unlike a Python
    dictionary, which can accept arbitrary key and value types.

    The key type must implement the `KeyElement` trait, which encompasses
    `Movable`, `Hashable`, and `EqualityComparable`. It also includes
    `CollectionElement` and `Copyable` until we have references.

    The value type must implement the `CollectionElement` trait.

    Usage:

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

    alias EMPTY = _EMPTY
    alias REMOVED = _REMOVED

    var size: Int
    """The number of elements currently stored in the dict."""
    var _n_entries: Int
    """The number of entries currently allocated."""
    var _reserved: Int
    """The current reserved size of the dictionary."""

    var _index: _DictIndex
    var _entries: List[Optional[DictEntry[K, V]]]

    @always_inline
    fn __init__(inout self):
        """Initialize an empty dictiontary."""
        self.size = 0
        self._n_entries = 0
        self._reserved = 8
        self._index = _DictIndex(self._reserved)
        self._entries = Self._new_entries(self._reserved)

    @always_inline
    fn __init__(inout self, existing: Self):
        """Copy an existing dictiontary.

        Args:
            existing: The existing dict.
        """
        self.size = existing.size
        self._n_entries = existing._n_entries
        self._reserved = existing._reserved
        self._index = existing._index.copy(existing._reserved)
        self._entries = existing._entries

    fn __copyinit__(inout self, existing: Self):
        """Copy an existing dictiontary.

        Args:
            existing: The existing dict.
        """
        self.size = existing.size
        self._n_entries = existing._n_entries
        self._reserved = existing._reserved
        self._index = existing._index.copy(existing._reserved)
        self._entries = existing._entries

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing dict into a new one.

        Args:
            existing: The existing dict.
        """
        self.size = existing.size
        self._n_entries = existing._n_entries
        self._reserved = existing._reserved
        self._index = existing._index^
        self._entries = existing._entries^

    fn __getitem__(self, key: K) raises -> V:
        """Retrieve a value out of the dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            "KeyError" if the key isn't present.
        """
        var value = self.find(key)
        if value:
            return value.value()
        raise "KeyError"

    fn __setitem__(inout self, key: K, value: V):
        """Set a value in the dictionary by key.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._insert(key, value)

    fn __contains__(self, key: K) -> Bool:
        """Check if a given key is in the dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if there key exists in the dictionary, False otherwise.
        """
        return self.find(key).__bool__()

    fn __len__(self) -> Int:
        """The number of elements currenly stored in the dictionary."""
        return self.size

    fn find(self, key: K) -> Optional[V]:
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        var hash = hash(key)
        var found: Bool
        var slot: Int
        var index: Int
        found, slot, index = self._find_index(hash, key)
        if found:
            var ev = self._entries.__get_ref(index)[]
            debug_assert(ev.__bool__(), "entry in index must be full")
            return ev.value().value
        return None

    fn pop(inout self, key: K, owned default: Optional[V] = None) raises -> V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.
            default: Optionally provide a default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.

        Raises:
            "KeyError" if the key was not present in the dictionary and no
            default value was provided.
        """
        var hash = hash(key)
        var found: Bool
        var slot: Int
        var index: Int
        found, slot, index = self._find_index(hash, key)
        if found:
            self._set_index(slot, Self.REMOVED)
            var entry = self._entries.__get_ref(index)[]
            self._entries[index] = None
            self.size -= 1
            debug_assert(entry.__bool__(), "entry in index must be full")
            return entry.value().value
        elif default:
            return default.value()
        raise "KeyError"

    fn __iter__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictKeyIter[K, V, mutability, self_life]:
        """Iterate over the dict's keys as immutable references.

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return _DictKeyIter(
            _DictEntryIter[K, V, mutability, self_life](0, 0, Reference(self))
        )

    fn keys[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictKeyIter[K, V, mutability, self_life]:
        """Iterate over the dict's keys as immutable references.

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return Self.__iter__(self)

    fn values[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictValueIter[K, V, mutability, self_life]:
        """Iterate over the dict's values as references.

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of references to the dictionary values.
        """
        return _DictValueIter(
            _DictEntryIter[K, V, mutability, self_life](0, 0, Reference(self))
        )

    fn items[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictEntryIter[K, V, mutability, self_life]:
        """Iterate over the dict's entries as immutable references.

        These can't yet be unpacked like Python dict items, but you can
        access the key and value as attributes ie.

        ```mojo
        for e in dict.items():
            print(e[].key, e[].value)
        ```

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of immutable references to the dictionary entries.
        """
        return _DictEntryIter[K, V, mutability, self_life](
            0, 0, Reference(self)
        )

    fn update(inout self, other: Self, /):
        """Update the dictionary with the key/value pairs from other, overwriting existing keys.
        The argument must be positional only.

        Args:
            other: The dictionary to update from.
        """
        for entry in other.items():
            self[entry[].key] = entry[].value

    @staticmethod
    @always_inline
    fn _new_entries(reserved: Int) -> List[Optional[DictEntry[K, V]]]:
        var entries = List[Optional[DictEntry[K, V]]](capacity=reserved)
        for i in range(reserved):
            entries.append(None)
        return entries

    fn _insert(inout self, owned key: K, owned value: V):
        self._insert(DictEntry[K, V](key^, value^))

    fn _insert(inout self, owned entry: DictEntry[K, V]):
        self._maybe_resize()
        var found: Bool
        var slot: Int
        var index: Int
        found, slot, index = self._find_index(entry.hash, entry.key)

        self._entries[index] = entry^
        if not found:
            self._set_index(slot, index)
            self.size += 1
            self._n_entries += 1

    fn _get_index(self, slot: Int) -> Int:
        return self._index.get_index(self._reserved, slot)

    fn _set_index(inout self, slot: Int, index: Int):
        return self._index.set_index(self._reserved, slot, index)

    fn _next_index_slot(self, inout slot: Int, inout perturb: Int):
        alias PERTURB_SHIFT = 5
        perturb >>= PERTURB_SHIFT
        slot = ((5 * slot) + perturb + 1) % self._reserved

    fn _find_empty_index(self, hash: Int) -> Int:
        var slot = hash % self._reserved
        var perturb = hash
        for _ in range(self._reserved):
            var index = self._get_index(slot)
            if index == Self.EMPTY:
                return slot
            self._next_index_slot(slot, perturb)
        abort("Dict: no empty index in _find_empty_index")
        return 0

    fn _find_index(self, hash: Int, key: K) -> (Bool, Int, Int):
        # Return (found, slot, index)
        var insert_slot = Optional[Int]()
        var insert_index = Optional[Int]()
        var slot = hash % self._reserved
        var perturb = hash
        for _ in range(self._reserved):
            var index = self._get_index(slot)
            if index == Self.EMPTY:
                return (False, slot, self._n_entries)
            elif index == Self.REMOVED:
                if not insert_slot:
                    insert_slot = slot
                    insert_index = self._n_entries
            else:
                var ev = self._entries.__get_ref(index)[]
                debug_assert(ev.__bool__(), "entry in index must be full")
                var entry = ev.value()
                if hash == entry.hash and key == entry.key:
                    return (True, slot, index)
            self._next_index_slot(slot, perturb)

        debug_assert(insert_slot.__bool__(), "never found a slot")
        debug_assert(insert_index.__bool__(), "slot populated but not index!!")
        return (False, insert_slot.value(), insert_index.value())

    fn _over_load_factor(self) -> Bool:
        return 3 * self.size > 2 * self._reserved

    fn _over_compact_factor(self) -> Bool:
        return 4 * self._n_entries > 3 * self._reserved

    fn _maybe_resize(inout self):
        if not self._over_load_factor():
            if self._over_compact_factor():
                self._compact()
            return
        self._reserved *= 2
        self.size = 0
        self._n_entries = 0
        self._index = _DictIndex(self._reserved)
        var old_entries = self._entries^
        self._entries = self._new_entries(self._reserved)

        for i in range(len(old_entries)):
            var entry = old_entries.__get_ref(i)[]
            if entry:
                self._insert(entry.value())

    fn _compact(inout self):
        self._index = _DictIndex(self._reserved)
        var right = 0
        for left in range(self.size):
            while not self._entries.__get_ref(right)[]:
                right += 1
                debug_assert(right < self._reserved, "Invalid dict state")
            var entry = self._entries.__get_ref(right)[]
            debug_assert(entry.__bool__(), "Logic error")
            var slot = self._find_empty_index(entry.value().hash)
            self._set_index(slot, left)
            if left != right:
                self._entries[left] = entry
                self._entries[right] = None

        self._n_entries = self.size


struct OwnedKwargsDict[V: CollectionElement](Sized, CollectionElement):
    """Container used to pass owned variadic keyword arguments to functions.

    This type mimics the interface of a dictionary with `String` keys, and
    should be usable more-or-less like a dictionary. Notably, however, this type
    should not be instantiated directly by users.

    Parameters:
        V: The value type of the dictionary. Currently must be CollectionElement.
    """

    alias key_type = String

    var _dict: Dict[Self.key_type, V]

    fn __init__(inout self):
        """Initialize an empty keyword dictionary."""
        self._dict = Dict[Self.key_type, V]()

    fn __copyinit__(inout self, existing: Self):
        """Copy an existing keyword dictionary.

        Args:
            existing: The existing keyword dictionary.
        """
        self._dict = existing._dict

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing keyword dictionary into a new one.

        Args:
            existing: The existing keyword dictionary.
        """
        self._dict = existing._dict^

    @always_inline("nodebug")
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

    @always_inline("nodebug")
    fn __setitem__(inout self, key: Self.key_type, value: V):
        """Set a value in the keyword dictionary by key.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._dict[key] = value

    @always_inline("nodebug")
    fn __contains__(self, key: Self.key_type) -> Bool:
        """Check if a given key is in the keyword dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if there key exists in the keyword dictionary, False
            otherwise.
        """
        return key in self._dict

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """The number of elements currenly stored in the keyword dictionary."""
        return len(self._dict)

    @always_inline("nodebug")
    fn find(self, key: Self.key_type) -> Optional[V]:
        """Find a value in the keyword dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self._dict.find(key)

    @always_inline("nodebug")
    fn pop(
        inout self, key: self.key_type, owned default: Optional[V] = None
    ) raises -> V:
        """Remove a value from the keyword dictionary by key.

        Args:
            key: The key to remove from the dictionary.
            default: Optionally provide a default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.

        Raises:
            "KeyError" if the key was not present in the dictionary and no
            default value was provided.
        """
        return self._dict.pop(key, default^)

    fn __iter__[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictKeyIter[Self.key_type, V, mutability, self_life]:
        """Iterate over the keyword dict's keys as immutable references.

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        # TODO(#36448): Use this instead of the current workaround
        # return self._dict.__iter__()
        return _DictKeyIter(
            _DictEntryIter[Self.key_type, V, mutability, self_life](
                0, 0, Reference(self)[]._dict
            )
        )

    fn keys[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictKeyIter[Self.key_type, V, mutability, self_life]:
        """Iterate over the keyword dict's keys as immutable references.

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        # TODO(#36448): Use this instead of the current workaround
        # return self._dict.keys()
        return Self.__iter__(self)

    fn values[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictValueIter[Self.key_type, V, mutability, self_life]:
        """Iterate over the keyword dict's values as references.

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of references to the dictionary values.
        """
        # TODO(#36448): Use this instead of the current workaround
        # return self._dict.values()
        return _DictValueIter(
            _DictEntryIter[Self.key_type, V, mutability, self_life](
                0, 0, Reference(self)[]._dict
            )
        )

    fn items[
        mutability: __mlir_type.`i1`, self_life: AnyLifetime[mutability].type
    ](
        self: Reference[Self, mutability, self_life].mlir_ref_type,
    ) -> _DictEntryIter[Self.key_type, V, mutability, self_life]:
        """Iterate over the keyword dictionary's entries as immutable references.

        These can't yet be unpacked like Python dict items, but you can
        access the key and value as attributes ie.

        ```mojo
        for e in dict.items():
            print(e[].key, e[].value)
        ```

        Parameters:
            mutability: Whether the dict is mutable.
            self_life: The dict's lifetime.

        Returns:
            An iterator of immutable references to the dictionary entries.
        """

        # TODO(#36448): Use this instead of the current workaround
        # return Reference(self)[]._dict.items()
        return _DictEntryIter[Self.key_type, V, mutability, self_life](
            0, 0, Reference(self)[]._dict
        )

    @always_inline("nodebug")
    fn _insert(inout self, owned key: Self.key_type, owned value: V):
        self._dict._insert(key^, value^)

    @always_inline("nodebug")
    fn _insert(inout self, key: StringLiteral, owned value: V):
        self._insert(String(key), value^)
