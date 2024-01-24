# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
"""Defines `Dict`, a collection that stores key-value pairs.

Dict provides an efficient, O(1) amortized
average-time complexity for insert, lookup, and removal of dictionary elements.
Its implementation closely mirrors Python's `dict` implementation:

- Performance and size are heavily optimized for small dictionaries, but can
  scale to large dictionaries.

- Insertion order is implicitly preserved. Once `__iter__` is implemented
  it will return a deterministic order based on insertion.

Key elements must implement the `KeyElement` trait, which encompasses
Movable, Hashable, and EqualityComparable. It also includes CollectionElement
and Copyable until we have references.

Value elements must be CollectionElements for a similar reason. Both key and
value types must always be Movable so we can resize the dictionary as it grows.

See the `Dict` docs for more details.
"""
from memory.anypointer import AnyPointer
from .optional import Optional
from .vector import CollectionElement


trait EqualityComparable:
    """A type which can be compared for equality with other instances of itself.
    """

    fn __eq__(self, other: Self) -> Bool:
        """Define whether two instances of the object are equal to each other.

        Args:
            other: Another instance of the same type.

        Returns:
            True if the instances are equal according to the type's definition
            of equality, False otherwise.
        """
        pass


trait KeyElement(CollectionElement, Hashable, EqualityComparable):
    """A trait composition for types which implement all requirements of
    dictionary keys. Dict keys must minimally be Movable, Hashable,
    and EqualityComparable for a hash map. Until we have references
    they must also be copyable."""

    pass


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
        self.key = key ^
        self.value = value ^


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

    fn __init__(inout self, reserved: Int):
        if reserved <= 128:
            let data = DTypePointer[DType.int8].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()
        elif reserved <= 2**16 - 2:
            let data = DTypePointer[DType.int16].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()
        elif reserved <= 2**32 - 2:
            let data = DTypePointer[DType.int32].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()
        else:
            let data = DTypePointer[DType.int64].alloc(reserved)
            for i in range(reserved):
                data[i] = _EMPTY
            self.data = data.bitcast[DType.invalid]()

    fn __moveinit__(inout self, owned existing: Self):
        self.data = existing.data

    fn get_index(self, reserved: Int, slot: Int) -> Int:
        if reserved <= 128:
            let data = self.data.bitcast[DType.int8]()
            return data.load(slot % reserved).to_int()
        elif reserved <= 2**16 - 2:
            let data = self.data.bitcast[DType.int16]()
            return data.load(slot % reserved).to_int()
        elif reserved <= 2**32 - 2:
            let data = self.data.bitcast[DType.int32]()
            return data.load(slot % reserved).to_int()
        else:
            let data = self.data.bitcast[DType.int64]()
            return data.load(slot % reserved).to_int()

    fn set_index(inout self, reserved: Int, slot: Int, value: Int):
        if reserved <= 128:
            let data = self.data.bitcast[DType.int8]()
            return data.store(slot % reserved, value)
        elif reserved <= 2**16 - 2:
            let data = self.data.bitcast[DType.int16]()
            return data.store(slot % reserved, value)
        elif reserved <= 2**32 - 2:
            let data = self.data.bitcast[DType.int32]()
            return data.store(slot % reserved, value)
        else:
            let data = self.data.bitcast[DType.int64]()
            return data.store(slot % reserved, value)

    fn __del__(owned self):
        self.data.free()


struct Dict[K: KeyElement, V: CollectionElement](Sized):
    """A container that stores key-value pairs.

    The key type and value type must be specified statically, unlike a Python
    dictionary, which can accept arbitrary key and value types.

    The key type must implement the `KeyElement` trait, which encompasses
    `Movable`, `Hashable`, and `EqualityComparable`. It also includes
    `CollectionElement` and `Copyable` until we have references.

    The value type must implemnt the `CollectionElement` trait.

    Usage:

    ```mojo
    from collections import Dict
    var d = Dict[StringKey, Int]()
    d["a"] = 1
    d["b"] = 2
    print(len(d))      # prints 2
    print(d["a"])      # prints 1
    print(d.pop("b"))  # prints 2
    print(len(d))      # prints 1
    ```

    Note that until standard library types implement `KeyElement`, you must
    create custom wrappers to use these as keys. For example, the following
    `StringKey` type wraps a String value and implements the `KeyElement` trait:

    ```mojo
    from collections.dict import Dict, KeyElement

    @value
    struct StringKey(KeyElement):
        var s: String

        fn __init__(inout self, owned s: String):
            self.s = s ^

        fn __init__(inout self, s: StringLiteral):
            self.s = String(s)

        fn __hash__(self) -> Int:
            let ptr = self.s._buffer.data.value
            return hash(DTypePointer[DType.int8](ptr), len(self.s))

        fn __eq__(self, other: Self) -> Bool:
            return self.s == other.s

    ```

    Parameters:
        K: The type of the dictionary key. Must be Hashable and EqualityComparable
           so we can find the key in the map.
        V: The value type of the dictionary. Currently must be CollectionElement
           since we don't have references.
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
    var _entries: DynamicVector[Optional[DictEntry[K, V]]]

    fn __init__(inout self):
        """Initialize an empty dictiontary."""
        self.size = 0
        self._n_entries = 0
        self._reserved = 8
        self._index = _DictIndex(self._reserved)
        self._entries = Self._new_entries(self._reserved)

    fn __moveinit__(inout self, owned existing: Self):
        """Move data of an existing dict into a new one.

        Args:
            existing: The existing dict.
        """
        self.size = existing.size
        self._n_entries = existing._n_entries
        self._reserved = existing._reserved
        self._index = existing._index ^
        self._entries = existing._entries ^

    fn __getitem__(self, key: K) raises -> V:
        """Retrieve a value out of the dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            "KeyError" if the key isn't present.
        """
        let value = self.find(key)
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
        """Check if a given value is in the dictionary or not."""
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
        let hash = hash(key)
        let found: Bool
        let slot: Int
        let index: Int
        found, slot, index = self._find_index(hash, key)
        if found:
            let ev = self._entries[index]
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
        let hash = hash(key)
        let found: Bool
        let slot: Int
        let index: Int
        found, slot, index = self._find_index(hash, key)
        if found:
            self._set_index(slot, Self.REMOVED)
            let entry = self._entries[index]
            self._entries[index] = None
            self.size -= 1
            debug_assert(entry.__bool__(), "entry in index must be full")
            return entry.value().value
        elif default:
            return default.value()
        raise "KeyError"

    @staticmethod
    fn _new_entries(reserved: Int) -> DynamicVector[Optional[DictEntry[K, V]]]:
        var entries = DynamicVector[Optional[DictEntry[K, V]]](reserved)
        for i in range(reserved):
            entries.append(None)
        return entries

    fn _insert(inout self, owned key: K, owned value: V):
        self._insert(DictEntry[K, V](key ^, value ^))

    fn _insert(inout self, owned entry: DictEntry[K, V]):
        self._maybe_resize()
        let found: Bool
        let slot: Int
        let index: Int
        found, slot, index = self._find_index(entry.hash, entry.key)

        self._entries[index] = entry ^
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
            let index = self._get_index(slot)
            if index == Self.EMPTY:
                return slot
            self._next_index_slot(slot, perturb)
        trap("Dict: no empty index in _find_empty_index")
        return 0

    fn _find_index(self, hash: Int, key: K) -> (Bool, Int, Int):
        # Return (found, slot, index)
        var insert_slot = Optional[Int]()
        var insert_index = Optional[Int]()
        var slot = hash % self._reserved
        var perturb = hash
        for _ in range(self._reserved):
            let index = self._get_index(slot)
            if index == Self.EMPTY:
                return (False, slot, self._n_entries)
            elif index == Self.REMOVED:
                if not insert_slot:
                    insert_slot = slot
                    insert_index = self._n_entries
            else:
                let ev = self._entries[index]
                debug_assert(ev.__bool__(), "entry in index must be full")
                let entry = ev.value()
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
        let old_entries = self._entries ^
        self._entries = self._new_entries(self._reserved)

        for i in range(len(old_entries)):
            let entry = old_entries[i]
            if entry:
                self._insert(entry.value())

    fn _compact(inout self):
        self._index = _DictIndex(self._reserved)
        var right = 0
        for left in range(self.size):
            while not self._entries[right]:
                right += 1
                debug_assert(right < self._reserved, "Invalid dict state")
            let entry = self._entries[right]
            debug_assert(entry.__bool__(), "Logic error")
            let slot = self._find_empty_index(entry.value().hash)
            self._set_index(slot, left)
            if left != right:
                self._entries[left] = entry
                self._entries[right] = None

        self._n_entries = self.size
