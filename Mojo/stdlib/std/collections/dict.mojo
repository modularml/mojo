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
"""Defines `Dict`, a collection that stores key-value pairs.

Dict provides an efficient, O(1) amortized
average-time complexity for insert, lookup, and removal of dictionary elements.
It uses a Swiss Table implementation with SIMD group probing for fast lookups:

- Performance and size are heavily optimized for small dictionaries, but can
  scale to large dictionaries.

- Insertion order is implicitly preserved. Iteration over keys, values, and
  items have a deterministic order based on insertion.

- For more information on the Mojo `Dict` type, see the
  [Mojo `Dict` manual](/docs/manual/types/#dict). To learn more about using
  Python dictionaries from Mojo, see
  [Python types in Mojo](/docs/manual/python/types/#python-types-in-mojo).

Key elements only need to be `Movable & Hashable & Equatable`. Methods that
fundamentally need to copy keys (`copy`, `update`, `__or__`, `fromkeys`,
iteration, ...) are conditionally available via
`where conforms_to(K, Copyable)` clauses.

Value elements only need to be `Movable & Deinitable`. Methods that
fundamentally need to copy values (`copy`, `find`, `get`, `update`, `__or__`,
`fromkeys`, iteration, ...) are conditionally available via
`where conforms_to(V, Copyable)` clauses.

See the `Dict` docs for more details.
"""

from std.builtin.rebind import downcast
from std.hashlib import Hasher, default_comp_time_hasher, default_hasher
import std.format._utils as fmt

from std.memory import alloc, dealloc, ThinAllocation, unsafe_memset
from std.memory.alloc import Layout

from ._swisstable import (
    CTRL_DELETED,
    CTRL_EMPTY,
    INITIAL_CAPACITY,
    SwissTable,
    SwissTableEntry,
    h2,
    is_occupied,
)

comptime KeyElement = Movable & Hashable & Equatable
"""A trait composition for types which implement all requirements of
dictionary keys. Dict keys must minimally be `Movable`, `Hashable`,
and `Equatable`. Methods that copy keys (e.g. `update`, `fromkeys`,
iteration) are conditionally available when the key type also
conforms to `Copyable`."""


# ===-----------------------------------------------------------------------===#
# Error types
# ===-----------------------------------------------------------------------===#


struct DictKeyError[K: KeyElement](ImplicitlyCopyable, Writable):
    """A custom error type for Dict lookups that fail.

    Parameters:
        K: The key type of the elements in the dictionary.
    """

    @doc_hidden
    def __init__(out self):
        pass

    def write_to(self, mut writer: Some[Writer]):
        """Write the error and the key to the writer.

        Args:
            writer: The writer to write to.
        """
        self.write_repr_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the error to the writer.

        Args:
            writer: The writer to write to.
        """
        fmt.FormatStruct(writer, "DictKeyError").params(
            fmt.TypeNames[Self.K]()
        ).fields()


@fieldwise_init
struct EmptyDictError(ImplicitlyCopyable, Writable):
    """A custom error type for when a `Dict` is empty."""

    def write_to(self, mut writer: Some[Writer]):
        """Write the error to the writer.

        Args:
            writer: The writer to write to.
        """
        self.write_repr_to(writer)

    def write_repr_to(self, mut writer: Some[Writer]):
        """Write the error to the writer.

        Args:
            writer: The writer to write to.
        """
        fmt.FormatStruct(writer, "EmptyDictError").fields()


# ===-----------------------------------------------------------------------===#
# DictEntry (alias for SwissTableEntry)
# ===-----------------------------------------------------------------------===#


comptime DictEntry = SwissTableEntry
"""Store a key-value pair entry inside a dictionary.

This is a comptime alias for `SwissTableEntry` for backwards compatibility.
"""


# ===-----------------------------------------------------------------------===#
# Iterators
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct _DictEntryIter[
    mut: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator over immutable DictEntry references.

    Parameters:
        mut: Whether the reference to the dictionary is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The origin of the Dict.
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = DictEntry[Self.K, Self.V, Self.H]

    var index: Int
    var seen: Int
    var src: Pointer[Dict[Self.K, Self.V, Self.H], Self.origin]

    def __init__(
        out self,
        index: Int,
        seen: Int,
        ref[Self.origin] dict: Dict[Self.K, Self.V, Self.H],
    ):
        self.index = index
        self.seen = seen
        self.src = Pointer(to=dict)

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        if self.seen >= len(self.src[]):
            raise StopIteration()

        while 0 <= self.index < len(self.src[]._order):
            var idx = self.index

            comptime if Self.forward:
                self.index += 1
            else:
                self.index -= 1

            var slot = Int(self.src[]._order[idx])
            if is_occupied(self.src[]._table._ctrl[unsafe_offset=slot]):
                self.seen += 1
                return (
                    (self.src[]._table._slots.unsafe_offset(slot))
                    .unsafe_mut_cast[Self.mut]()
                    .unsafe_origin_cast[Self.origin]()[]
                )

        assert self.seen == len(
            self.src[]
        ), "_order exhausted but not all entries seen: _len out of sync"
        raise StopIteration()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self.src[]) - self.seen
        return (len, {len})


@fieldwise_init
struct _TakeDictEntryIter[
    K: KeyElement & Deinitable,
    V: Movable & Deinitable,
    H: Hasher,
    origin: MutOrigin,
](Copyable, Iterable, Iterator):
    """Iterator over mutable DictEntry references that moves entries out of the dictionary.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The mutable origin of the Dict
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = DictEntry[Self.K, Self.V, Self.H]

    var index: Int
    var src: Pointer[Dict[Self.K, Self.V, Self.H], Self.origin]

    def __init__(out self, ref[Self.origin] dict: Dict[Self.K, Self.V, Self.H]):
        self.index = 0
        self.src = Pointer(to=dict)

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> Self.Element:
        if len(self.src[]) <= 0:
            raise StopIteration()

        while self.index < len(self.src[]._order):
            var slot = Int(self.src[]._order[self.index])
            self.index += 1

            if is_occupied(self.src[]._table._ctrl[unsafe_offset=slot]):
                var entry = (
                    self.src[]._table._slots.unsafe_offset(slot)
                ).unsafe_take_pointee()
                self.src[]._table.set_ctrl(slot, CTRL_DELETED)
                self.src[]._table._len -= 1
                return entry^

        assert (
            len(self.src[]) == 0
        ), "_order exhausted but _len > 0: ctrl bytes and _len out of sync"
        raise StopIteration()


@fieldwise_init
struct _DictEntryIterOwned[
    K: KeyElement & Deinitable,
    V: Movable & Deinitable,
    H: Hasher,
](IterableOwned, Iterator, Movable):
    """An owning iterator over DictEntry values that consumes the dictionary.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
    """

    comptime Element = DictEntry[Self.K, Self.V, Self.H]
    comptime IteratorOwnedType = Self

    var _dict: Dict[Self.K, Self.V, Self.H]
    var _index: Int

    @always_inline
    def __deinit__(deinit self):
        # Dict.__deinit__ handles destroying remaining occupied slots.
        pass

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        while self._index < len(self._dict._order):
            var slot = Int(self._dict._order[self._index])
            self._index += 1

            if is_occupied(self._dict._table._ctrl[unsafe_offset=slot]):
                var entry = (
                    self._dict._table._slots.unsafe_offset(slot)
                ).unsafe_take_pointee()
                self._dict._table.set_ctrl(slot, CTRL_DELETED)
                self._dict._table._len -= 1
                return entry^

        debug_assert(
            self._dict._table._len == 0,
            "_order exhausted but _len > 0: ctrl bytes and _len out of sync",
        )
        raise StopIteration()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return (self._dict._table._len, {self._dict._table._len})


@fieldwise_init
struct _DictKeyIterOwned[
    K: KeyElement & Deinitable,
    V: Movable & Deinitable,
    H: Hasher,
](IterableOwned, Iterator, Movable):
    """An owning iterator over Dict keys that consumes the dictionary.

    Parameters:
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
    """

    comptime Element = Self.K
    comptime IteratorOwnedType = Self

    var _inner: _DictEntryIterOwned[Self.K, Self.V, Self.H]

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    @always_inline
    def __next__(mut self) raises StopIteration -> Self.Element:
        return self._inner.__next__().reap_key()

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self._inner.bounds()


@fieldwise_init
struct _DictKeyIter[
    mut: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator over immutable Dict key references.

    Parameters:
        mut: Whether the reference to the dictionary is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The origin of the Dict.
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime dict_entry_iter = _DictEntryIter[
        Self.K, Self.V, Self.H, Self.origin, Self.forward
    ]
    comptime Element = Self.K

    var iter: Self.dict_entry_iter

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    @always_inline
    def __next__(
        mut self,
    ) raises StopIteration -> ref[self.iter.__next__().key] Self.Element:
        return self.iter.__next__().key

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self.iter.bounds()


@fieldwise_init
struct _DictValueIter[
    mut: Bool,
    //,
    K: KeyElement & Copyable,
    V: Copyable,
    H: Hasher,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator over Dict value references. These are mutable if the dict
    is mutable.

    Parameters:
        mut: Whether the reference to the dictionary is mutable.
        K: The key type of the elements in the dictionary.
        V: The value type of the elements in the dictionary.
        H: The type of the hasher in the dictionary.
        origin: The origin of the Dict.
        forward: The iteration direction. `False` is backwards.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    var iter: _DictEntryIter[Self.K, Self.V, Self.H, Self.origin, Self.forward]
    comptime Element = Self.V

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __reversed__(
        self,
    ) -> _DictValueIter[Self.K, Self.V, Self.H, Self.origin, False]:
        var src = self.iter.src
        return _DictValueIter(
            _DictEntryIter[Self.K, Self.V, Self.H, Self.origin, False](
                len(src[]._order) - 1, 0, src
            )
        )

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        ref entry_ref = self.iter.__next__()
        # Cast through a pointer to grant additional mutability because
        # _DictEntryIter.next erases it.
        return Pointer(to=entry_ref.value).unsafe_origin_cast[Self.origin]()[]

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        return self.iter.bounds()


# ===-----------------------------------------------------------------------===#
# Dict
# ===-----------------------------------------------------------------------===#


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `Dict` with"
    " non-`Deinitable` keys or values"
)
struct Dict[
    K: KeyElement,
    V: Movable,
    H: Hasher = default_hasher,
](
    Boolable,
    Copyable where conforms_to(K, Copyable) and conforms_to(V, Copyable),
    Defaultable,
    Deinitable where conforms_to(K, Deinitable) and conforms_to(V, Deinitable),
    Equatable where conforms_to(K, Copyable) and conforms_to(V, Equatable),
    Hashable where conforms_to(K, Copyable) and conforms_to(V, Hashable),
    Iterable,
    # TODO(MOCO-4308): Remove redundant 'KeyElement' and 'Movable' constraints
    IterableOwned where conforms_to(K, KeyElement & Deinitable) and conforms_to(
        V, Movable & Deinitable
    ),
    Movable,
    Sized,
    Writable where (
        conforms_to(K, Copyable)
        and conforms_to(K, Writable)
        and conforms_to(V, Writable)
    ),
):
    """A container that stores key-value pairs.

    The `Dict` type is Mojo's primary associative collection, similar to
    Python's `dict` (dictionary). Unlike a `List`, which stores elements by
    index, a `Dict` stores values associated with unique keys, which enables
    fast lookups, insertions, and deletions.

    You can create a `Dict` in several ways:

    ```mojo
    # Empty dictionary
    var empty_dict = Dict[String, Int]()

    # Dictionary literal syntax
    var scores = {"Alice": 95, "Bob": 87, "Charlie": 92}

    # Pre-allocated capacity
    var large_dict = Dict[String, Int](capacity=64)

    # From separate key and value lists
    var keys = ["red", "green", "blue"]
    var values = [255, 128, 64]
    var colors = Dict[String, Int]()
    for key, value in zip(keys, values):
        colors[String(key)] = value # cast list iterator to key-type
    ```

    Be aware of the following characteristics:

    - **Type safety**: Both keys and values must be homogeneous types,
    determined at compile time. This is more restrictive than Python
    dictionaries but provides better performance:

      ```mojo
      var string_to_int = {"count": 42}     # Dict[String, Int]
      var int_to_string = {1: "one"}        # Dict[Int, String]
      # var mixed = {"key": 1, 2: "val"}   # Error! Keys must be same type
      ```

      However, you can get around this by defining your dictionary key and/or
      value type as [`Variant`](/docs/std/utils/variant/Variant/). This is
      a discriminated union type, meaning it can store any number of different
      types that can vary at runtime.

    - **Insertion order**: Iteration over keys, values, and items follows
      insertion order. Updating an existing key's value does not change its
      position. This matches the ordering guarantee of Python's `dict`.

    - **Value semantics**: A `Dict` is value semantic by default. Copying a
      `Dict` creates a deep copy of all key-value pairs. To avoid accidental
      copies, `Dict` is not implicitly copyable—you must explicitly copy it
      using the `.copy()` method.

      ```mojo
      var dict1 = {"a": 1, "b": 2}
      # var dict2 = dict1  # Error: Dict is not implicitly copyable
      var dict2 = dict1.copy()  # Deep copy
      dict2["c"] = 3
      print(dict1)   # => {"a": 1, "b": 2}
      print(dict2)   # => {"a": 1, "b": 2, "c": 3}
      ```

      This is different from Python, where assignment creates a reference to
      the same dictionary. For more information, read about [value
      semantics](/docs/manual/values/value-semantics).

    - **Iteration uses immutable references**: When iterating over keys, values,
      or items, you get immutable references unless you specify `ref` or `var`:

      ```mojo
      var inventory = {"apples": 10, "bananas": 5}

      # Default behavior creates immutable (read-only) references:
      # for value in inventory.values():
      #     value += 1  # error: expression must be mutable

      # Using `ref` gets mutable (read-write) references
      for ref value in inventory.values():
          value += 1  # Modify inventory values in-place
      print(inventory)  # => {"apples": 11, "bananas": 6}

      # Using `var` gets an owned copy of the value
      for var key in inventory.keys():
          inventory[key] += 1  # Modify inventory values in-place
      print(inventory)  # => {"apples": 12, "bananas": 7}
      ```

      Note that indexing into a `Dict` with a key that's a reference to the
      key owned by the `Dict` produces a confusing error related to
      [argument exclusivity](/docs/manual/values/ownership/#argument-exclusivity).
      Using `var key` in the previous example creates an owned copy of the key,
      avoiding the error.

    - **KeyError handling**: Directly accessing values with the `[]` operator
      will raise `DictKeyError` if the key is not found:

      ```mojo
      var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
      # print(phonebook["Charlie"])  # => DictKeyError
      ```

      For safe access, you should instead use `get()`:

      ```mojo
      var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}
      var phone = phonebook.get("Charlie")
      print(phone) if phone else print('phone not found')
      ```


    Examples:

    ```mojo
    var phonebook = {"Alice": "555-0101", "Bob": "555-0102"}

    # Add/update entries
    phonebook["Charlie"] = "555-0103"    # Add new entry
    phonebook["Alice"] = "555-0199"      # Update existing entry

    # Access directly (unsafe and raises DictKeyError if key not found)
    print(phonebook["Alice"])            # => 555-0199

    # Access safely
    var phone = phonebook.get("David")   # Returns Optional type
    print(phone.or_else("phone not found!"))

    # Access safely with default value
    phone = phonebook.get("David", "555-0000")
    print(phone)               # => '555-0000'

    # Check for keys
    if "Bob" in phonebook:
        print("Found Bob")

    # Remove (pop) entries
    print(phonebook.pop("Charlie"))         # Remove and return: "555-0103"
    print(phonebook.pop("Unknown", "N/A"))  # Pop with default

    # Iterate over a dictionary
    for key in phonebook.keys():
        print("Key:", key)

    for value in phonebook.values():
        print("Value:", value)

    for item in phonebook.items():
        print(item.key, "=>", item.value)

    for var key in phonebook:
        print(key, "=>", phonebook[key])

    # Number of key-value pairs
    print('len:', len(phonebook))        # => len: 2

    # Dictionary operations
    var backup = phonebook.copy()        # Explicit copy
    phonebook.clear()                    # Remove all entries

    # Merge dictionaries
    var more_numbers = {"David": "555-0104", "Eve": "555-0105"}
    backup.update(more_numbers)          # Merge in-place
    var combined = backup | more_numbers # Create new merged dict
    print(combined)
    ```

    Parameters:
        K: The type of keys stored in the dictionary.
        V: The type of values stored in the dictionary.
        H: The type of hasher used to hash the keys.
    """

    # Implementation:
    #
    # This Dict uses a Swiss Table design with flat layout + insertion-order
    # side array. The core Swiss Table logic (control bytes, SIMD probing,
    # slot management) lives in _swisstable.SwissTable. Dict adds:
    #
    # - Insertion-order array: A separate List[Int32] tracks the order of
    #   insertion for deterministic iteration.
    # - Ordered find_slot: Uses SwissTable.find_slot which skips DELETED
    #   slots to maintain _order consistency.

    # ===-------------------------------------------------------------------===#
    # Aliases
    # ===-------------------------------------------------------------------===#

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[
        downcast[Self.K, KeyElement & Copyable],
        downcast[Self.V, Copyable],
        Self.H,
        iterable_origin,
    ]
    """The iterator type for this dictionary.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # TODO(MOCO-4308): Remove redundant 'KeyElement' and 'Movable' constraints
    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.K, KeyElement & Deinitable
    ) and conforms_to(Self.V, Movable & Deinitable) = _DictKeyIterOwned[
        Self.K,
        Self.V,
        Self.H,
    ]
    """The owned iterator type for this dictionary."""

    # ===-------------------------------------------------------------------===#
    # Fields
    # ===-------------------------------------------------------------------===#

    var _table: SwissTable[Self.K, Self.V, Self.H]
    """The underlying Swiss Table managing ctrl bytes, slots, and probing."""

    var _order: List[Int32]
    """Insertion-order array of slot indices. Stale entries (from deleted slots)
    are skipped during iteration by checking the ctrl byte.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """Initialize an empty dictionary."""
        self._table = SwissTable[Self.K, Self.V, Self.H]()
        self._order = List[Int32]()

    @always_inline
    def __init__(out self, *, capacity: Int):
        """Initialize an empty dictionary with a pre-reserved capacity.

        The capacity is defined by `next_power_of_two(ceildiv(capacity * 8, 7))`
        (minimum 16) to satisfy internal layout requirements. The usable
        capacity before resizing is `7 // 8` of the rounded value.

        Args:
            capacity: The requested minimum number of slots.

        Examples:

        ```mojo
        var x = Dict[Int, Int](capacity=1000)
        # Actual capacity is 2048; can hold 1792 entries without resizing.
        ```
        """
        self._table = SwissTable[Self.K, Self.V, Self.H](capacity=capacity)
        self._order = List[Int32](capacity=self._table._capacity * 7 // 8)

    @always_inline
    def __init__(
        out self,
        var keys: List[Self.K],
        var values: List[Self.V],
        __dict_literal__: NoneType,
    ) where conforms_to(Self.K, Copyable & Deinitable) and conforms_to(
        Self.V, Copyable & Deinitable
    ):
        """Constructs a dictionary from the given keys and values.

        Args:
            keys: The list of keys to build the dictionary with.
            values: The corresponding values to pair with the keys.
            __dict_literal__: Tell Mojo to use this method for dict literals.
        """
        self = Self(capacity=len(keys))
        assert len(keys) == len(
            values
        ), "keys and values must have the same length"

        # TODO: Should transfer the key/value's from the list to avoid copying
        # the values.
        for key, value in zip(keys, values):
            self._insert(key.copy(), value.copy())

    # TODO: add @property when Mojo supports it to make
    # it possible to do `self._reserved`.
    @always_inline
    def _reserved(self) -> Int:
        return self._table._capacity

    # TODO(MSTDL-2848): Once we have a system for adapting `Iterable` to
    # `IterableOwned`, have this function call the `IterableOwned` version below.
    @staticmethod
    def fromkeys[
        Keys: Iterable
    ](ref keys: Keys, value: Self.V) -> Self where (
        Keys.IteratorType[origin_of(keys)].Element == Self.K
        and conforms_to(Self.K, Deinitable)
        and conforms_to(Self.V, Copyable & Deinitable)
    ):
        """Create a new dictionary with keys from a borrowed iterable and values
        set to value.

        Selected when the iterable is passed by reference, leaving the source
        intact.

        Parameters:
            Keys: The type of the borrowed iterable of keys.

        Args:
            keys: The keys to set.
            value: The value to set.

        Returns:
            The new dictionary.

        Example:

        ```mojo
        var keys: Array[String, 3] = ["a", "b", "c"]
        var dict = Dict.fromkeys(keys, 0)
        print(dict)  # => {"a": 0, "b": 0, "c": 0}
        ```
        """
        var my_dict = Self()
        for var key in keys:
            # TODO(MOCO-4355): Drop `rebind_var` once the `where` clause's
            # `Element == Self.K` equality is applied when checking the body.
            var k = rebind_var[Self.K](key^)
            my_dict[k^] = value.copy()
        return my_dict^

    @staticmethod
    def fromkeys[
        Keys: IterableOwned
    ](var keys: Keys, value: Self.V) -> Self where (
        Keys.IteratorOwnedType.Element == Self.K
        and conforms_to(Self.K, Deinitable)
        and conforms_to(Self.V, Copyable & Deinitable)
    ):
        """Create a new dictionary with keys from an owned iterable and values
        set to value.

        Selected when the iterable is transferred with `keys^`.

        Parameters:
            Keys: The type of the owned iterable of keys.

        Args:
            keys: The keys to set.
            value: The value to set.

        Returns:
            The new dictionary.

        Example:

        ```mojo
        var keys = ["a", "b", "c"]
        var dict = Dict.fromkeys(keys^, 0)
        print(dict)  # => {"a": 0, "b": 0, "c": 0}
        ```
        """
        var my_dict = Self()
        for var key in keys^:
            # TODO(MOCO-4355): Drop `rebind_var` once the `where` clause's
            # `Element == Self.K` equality is applied when checking the body.
            var k = rebind_var[Self.K](key^)
            my_dict[k^] = value.copy()
        return my_dict^

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.K, Copyable) and conforms_to(Self.V, Copyable):
        """Copy an existing dictionary.

        Args:
            copy: The existing dict.
        """
        self._table = SwissTable[Self.K, Self.V, Self.H](copy=copy._table)
        self._order = copy._order.copy()

    def deinit_with(
        deinit self, deinit_func: Some[def(var Self.K, var Self.V)], /
    ):
        """Consume the dictionary, deinitializing each key/value pair with a closure.

        Use this to tear down a `Dict` whose keys or values are not
        `Deinitable`.

        Args:
            deinit_func: A closure called once per entry to destroy its key and
                value.
        """
        self._table^.deinit_with(deinit_func)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    def __getitem__(
        ref self, ref key: Self.K
    ) raises DictKeyError[Self.K] -> ref[
        origin_of(self)._get_owned_interior["value"]
    ] Self.V:
        """Retrieve a value out of the dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            `DictKeyError` if the key isn't present.
        """
        return self._find_ref(key)

    @always_inline
    def __setitem__(
        mut self, var key: Self.K, var value: Self.V
    ) where conforms_to(Self.K, Deinitable) and conforms_to(Self.V, Deinitable):
        """Set a value in the dictionary by key.

        Constraints:
            Both `K` and `V` must be `Deinitable`. Assigning to an
            existing key destroys the displaced entry in place.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._insert(key^, value^)

    def insert(
        mut self, var key: Self.K, var value: Self.V
    ) -> Optional[DictEntry[Self.K, Self.V, Self.H]]:
        """Insert a key/value pair, returning the displaced entry if the key
        was already present.

        Unlike `__setitem__`, the displaced key and value are moved out and
        returned (never destroyed in place), so this works when `K` or `V` is
        linear (non-`Deinitable`). The caller is responsible for
        disposing of the returned entry.

        Args:
            key: The key to associate with the value.
            value: The value to store.

        Returns:
            The previous entry if `key` was already present, otherwise an
            empty `Optional`.
        """
        self._ensure_capacity()
        var entry = DictEntry[Self.K, Self.V, Self.H](key^, value^)
        var found, slot_idx = self._table.find_slot(entry._hash, entry.key)

        if found:
            # Overwrite: move the displaced entry out and return it (never
            # destroyed), then move the new entry into the slot.
            var displaced = (
                self._table._slots.unsafe_offset(slot_idx)
            ).unsafe_take_pointee()
            (self._table._slots.unsafe_offset(slot_idx)).unsafe_write(entry^)
            return displaced^

        # New entry.
        self._place_new_entry(slot_idx, entry^)
        return None

    @always_inline
    def __contains__(self, key: Self.K) -> Bool:
        """Check if a given key is in the dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if the key exists in the dictionary, False otherwise.
        """
        var found, _ = self._table.find_slot(hash[Self.H](key), key)
        return found

    # TODO(MOCO-4308): Remove redundant 'KeyElement' and 'Movable' constraints
    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(
        Self.K, KeyElement & Deinitable
    ) and conforms_to(Self.V, Movable & Deinitable):
        """Consume the dictionary and iterate over its keys.

        Returns:
            An iterator that owns the dictionary's keys.
        """
        return {_DictEntryIterOwned(self^, 0)}

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraints once we have better iter traits.
        comptime assert conforms_to(Self.K, Copyable) and conforms_to(
            Self.V, Copyable
        ), "Dict iteration requires the key and value types to be `Copyable`."
        comptime DictCopyable = Dict[
            downcast[Self.K, KeyElement & Copyable],
            downcast[Self.V, Copyable],
            Self.H,
        ]
        return _DictKeyIter(
            _DictEntryIter(
                0,
                0,
                rebind[Pointer[DictCopyable, origin_of(self)]](
                    Pointer(to=self)
                )[],
            )
        )

    # TODO(MSTDL-2390): Remove `Copyable` constraints once we have better iter traits.
    # TODO(MOCO-4308): Remove redundant `KeyElement` constraints
    def __reversed__(
        ref self,
    ) -> _DictKeyIter[
        Self.K,
        Self.V,
        Self.H,
        origin_of(self),
        False,
    ] where conforms_to(Self.K, KeyElement & Copyable) and conforms_to(
        Self.V, Copyable
    ):
        """Iterate backwards over the dict keys, returning immutable references.

        Returns:
            A reversed iterator of immutable references to the dict keys.
        """
        return _DictKeyIter(
            _DictEntryIter[forward=False](
                len(self._order) - 1,
                0,
                Pointer(to=self),
            )
        )

    def __or__(
        self, other: Self
    ) -> Self where conforms_to(Self.K, Copyable & Deinitable) and conforms_to(
        Self.V, Copyable & Deinitable
    ):
        """Merge self with other and return the result as a new dict.

        Args:
            other: The dictionary to merge with.

        Returns:
            The result of the merge.
        """
        var result = self.copy()
        result.update(other)
        return result^

    def __ior__(
        mut self, other: Self
    ) where conforms_to(Self.K, Copyable & Deinitable) and conforms_to(
        Self.V, Copyable & Deinitable
    ):
        """Merge self with other in place.

        Args:
            other: The dictionary to merge with.
        """
        self.update(other)

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """The number of elements currently stored in the dictionary.

        Returns:
            The number of elements currently stored in the dictionary.
        """
        return self._table._len

    @always_inline("nodebug")
    def __bool__(self) -> Bool:
        """Check if the dictionary is empty or not.

        Returns:
            `False` if the dictionary is empty, `True` if there is at least one
            element.
        """
        return len(self).__bool__()

    def __eq__(
        self, other: Self
    ) -> Bool where conforms_to(Self.K, Copyable) and conforms_to(
        Self.V, Equatable
    ):
        """Checks if two dictionaries are equal.

        Two dictionaries are equal if they contain the same keys and the
        corresponding values are equal.

        Args:
            other: The dictionary to compare with.

        Returns:
            True if the dictionaries are equal, False otherwise.
        """
        if len(self) != len(other):
            return False

        # Probe `other` with the hash cached on the entry rather than through
        # `_find_ref`, which rehashes every key. Both dicts share the hasher
        # `H`, so the cached hash is the one `other`'s table was built with.
        for entry in self.items():
            var found, slot_idx = other._table.find_slot(entry._hash, entry.key)
            if not found:
                return False
            if entry.value != other._table._slots[unsafe_offset=slot_idx].value:
                return False

        return True

    def __hash__[
        H2: Hasher
    ](self, mut hasher: H2) where conforms_to(Self.K, Copyable) and conforms_to(
        Self.V, Hashable
    ):
        """Hashes the dictionary using the given hasher.

        The hash is order-independent: two dictionaries with the same key-value
        pairs will have the same hash regardless of insertion order.

        Parameters:
            H2: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        # XOR with mixing for order-independent hashing (same approach as
        # Python's frozenset). The mixing step provides bit diffusion so that
        # entries with similar hashes don't cancel under XOR.
        var combined = UInt64(0)
        for entry in self.items():
            var entry_hasher = H2()
            entry.key.__hash__(entry_hasher)
            entry.value.__hash__(entry_hasher)
            var h = entry_hasher^.finish()
            h = ((h ^ 89869747) ^ (h << 16)) * 3644798167
            combined ^= h
        hasher._update_with_simd(combined)

    def _write_dict_body[
        f_key: def(Self.K, mut Some[Writer]) thin,
        f_val: def(Self.V, mut Some[Writer]) thin,
    ](self, mut writer: Some[Writer]) where (
        conforms_to(Self.K, Copyable)
        and conforms_to(Self.K, Writable)
        and conforms_to(Self.V, Writable)
    ):
        writer.write_string("{")

        var i = 0
        for entry in self.items():
            if i > 0:
                writer.write_string(", ")
            i += 1

            f_key(entry.key, writer)
            writer.write(": ")
            f_val(entry.value, writer)

        writer.write_string("}")

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where (
        conforms_to(Self.K, Copyable)
        and conforms_to(Self.K, Writable)
        and conforms_to(Self.V, Writable)
    ):
        """Write this `Dict` to the writer.

        Args:
            writer: The value to write to.
        """
        self._write_dict_body[
            f_key=fmt.write_to[Self.K],
            f_val=fmt.write_to[Self.V],
        ](writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where (
        conforms_to(Self.K, Copyable)
        and conforms_to(Self.K, Writable)
        and conforms_to(Self.V, Writable)
    ):
        """Write this `Dict`'s representation to the writer.

        Args:
            writer: The value to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_dict_body[
                f_key=fmt.write_repr_to[Self.K],
                f_val=fmt.write_repr_to[Self.V],
            ](w)

        fmt.FormatStruct(writer, "Dict").params(
            fmt.TypeNames[Self.K, Self.V](),
        ).fields(write_fields)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    def _minimum_size_of_string_representation(self) -> Int:
        # we do a rough estimation of the minimum number of chars that we'll see
        # in the string representation, we assume that String(key) and String(value)
        # will be both at least one char.
        return (
            2  # '{' and '}'
            + len(self) * 6  # String(key), String(value) ": " and ", "
            - 2  # remove the last ", "
        )

    def find(
        self, key: Self.K
    ) -> Optional[Self.V] where conforms_to(Self.V, Copyable):
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        var value = my_dict.find("a")
        print(value)  # => 1
        var missing_value = my_dict.find("c")
        print(missing_value)  # => None
        ```
        """
        # Probes the table directly rather than catching `_find_ref`'s
        # `DictKeyError`: a `try` region around the copy fences the optimizer
        # and the raising path costs an extra copy of the value.
        var found, slot_idx = self._table.find_slot(hash[Self.H](key), key)
        if not found:
            return Optional[Self.V](None)
        return (self._table._slots.unsafe_offset(slot_idx))[].value.copy()

    @__unsafe_nested_origins_read_only
    def _find_ref(
        ref self, ref key: Self.K
    ) raises DictKeyError[Self.K] -> ref[
        origin_of(self)._get_owned_interior["value"]
    ] Self.V:
        """Find a value in the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a reference to the value if it is
            present, otherwise an empty Optional.
        """
        var h = hash[Self.H](key)
        var found, slot_idx = self._table.find_slot(h, key)

        if found:
            assert is_occupied(
                self._table._ctrl[unsafe_offset=slot_idx]
            ), "_find_slot returned found=True but ctrl byte is not occupied"
            return Pointer(
                to=(self._table._slots.unsafe_offset(slot_idx))[].value
            )._get_ref_with_unsafe_interior_origin["value", origin_of(self)]()

        raise DictKeyError[Self.K]()

    @__unsafe_nested_origins_read_only
    def _find_ref_matching(
        ref self, hash: UInt64, key_matches: Some[def(Self.K) -> Bool]
    ) raises DictKeyError[Self.K] -> ref[
        origin_of(self)._get_owned_interior["value"]
    ] Self.V:
        """Find a value by a key predicate and a precomputed hash.

        Like `_find_ref`, but probes with `key_matches` instead of a concrete
        `Self.K`, so callers can look up a heterogeneous key (such as a
        `StringSpan` against a `String` key) without building a `Self.K`.

        Args:
            hash: The hash of the lookup key, from `Self.H`. Any key that
                `key_matches` accepts must hash to this value.
            key_matches: Returns `True` when a stored key equals the lookup key.

        Returns:
            A reference to the value if the key is present.

        Raises:
            `DictKeyError` if no matching key is present.
        """
        var found, slot_idx = self._table.find_slot_matching(hash, key_matches)

        if found:
            assert is_occupied(self._table._ctrl[unsafe_offset=slot_idx]), (
                "find_slot_matching returned found=True but ctrl byte is not"
                " occupied"
            )
            return Pointer(
                to=(self._table._slots.unsafe_offset(slot_idx))[].value
            )._get_ref_with_unsafe_interior_origin["value", origin_of(self)]()

        raise DictKeyError[Self.K]()

    @always_inline
    def get(
        self, key: Self.K
    ) -> Optional[Self.V] where conforms_to(Self.V, Copyable):
        """Get a value from the dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        var value = my_dict.get("a")
        print(value)  # => 1

        var missing_value = my_dict.get("c")
        print(missing_value)  # => -1

        from std.testing import assert_true
        assert_true(my_dict["a"] == my_dict.get("a").or_else(Int.MAX))
        ```
        """
        return self.find(key)

    @always_inline
    def get(
        self, key: Self.K, var default: Self.V
    ) -> Self.V where conforms_to(Self.V, Copyable & Deinitable):
        """Get a value from the dictionary by key.

        Args:
            key: The key to search for in the dictionary.
            default: Default value to return.

        Returns:
            A copy of the value if it was present, otherwise default.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        var value = my_dict.get("a", Int.MAX)
        print(value)  # => 1

        var missing_value = my_dict.get("c", -1)
        print(missing_value)  # => -1

        from std.testing import assert_true
        assert_true(my_dict["a"] == my_dict.get("a", Int.MAX))
        ```
        """
        # Probes the table directly instead of going through `find`, which
        # would build an `Optional` only for `or_else` to unwrap it again.
        var found, slot_idx = self._table.find_slot(hash[Self.H](key), key)
        if not found:
            return default^
        return (self._table._slots.unsafe_offset(slot_idx))[].value.copy()

    def pop(
        mut self, key: Self.K, var default: Self.V
    ) -> Self.V where conforms_to(Self.K, Deinitable) and conforms_to(
        Self.V, Deinitable
    ):
        """Remove a value from the dictionary by key.

        Constraints:
            Both `K` and `V` must be `Deinitable`. Removing an entry
            discards its key and returns only the value; the unused `default` is
            also discarded when the key is found. Use `popitem()` to retrieve
            both key and value.

        Args:
            key: The key to remove from the dictionary.
            default: A default value to return if the key
                was not found instead of raising.

        Returns:
            The value associated with the key, if it was in the dictionary.
            If it wasn't, return the provided default value instead.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        var value = my_dict.pop("a", 99)
        print(value)  # => 1
        var missing_value = my_dict.pop("c", 99)
        print(missing_value)  # => 99
        ```
        """
        # Probes the table directly rather than catching the raising `pop`'s
        # `DictKeyError`, which fences the optimizer around the whole removal.
        var found, slot_idx = self._table.find_slot(hash[Self.H](key), key)
        if not found:
            return default^
        assert is_occupied(
            self._table._ctrl[unsafe_offset=slot_idx]
        ), "find_slot returned found=True but ctrl byte is not occupied"
        var entry = (
            self._table._slots.unsafe_offset(slot_idx)
        ).unsafe_take_pointee()
        self._table.set_ctrl(slot_idx, CTRL_DELETED)
        self._table._len -= 1
        return entry^.reap_value()

    def pop(
        mut self, ref key: Self.K
    ) raises DictKeyError[Self.K] -> Self.V where conforms_to(
        Self.K, Deinitable
    ):
        """Remove a value from the dictionary by key.

        Constraints:
            `K` must be `Deinitable`. Removing an entry discards its
            key and returns only the value. Use `popitem()` to retrieve both.

        Args:
            key: The key to remove from the dictionary.

        Returns:
            The value associated with the key, if it was in the dictionary.
            Raises otherwise.

        Raises:
            `DictKeyError` if the key was not present in the dictionary.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        var value = my_dict.pop("a", 99)
        print(value)  # => 1
        var missing_value = my_dict.pop("c", 99)
        print(missing_value)  # => 99
        ```
        """
        var h = hash[Self.H](key)
        var found, slot_idx = self._table.find_slot(h, key)

        if found:
            assert is_occupied(
                self._table._ctrl[unsafe_offset=slot_idx]
            ), "_find_slot returned found=True but ctrl byte is not occupied"
            var entry = (
                self._table._slots.unsafe_offset(slot_idx)
            ).unsafe_take_pointee()
            self._table.set_ctrl(slot_idx, CTRL_DELETED)
            self._table._len -= 1
            return entry^.reap_value()
        raise DictKeyError[Self.K]()

    def popitem(
        mut self,
    ) raises EmptyDictError -> DictEntry[Self.K, Self.V, Self.H]:
        """Remove and return a (key, value) pair from the dictionary.

        Returns:
            Last dictionary item

        Raises:
            `EmptyDictError` if the dictionary is empty.

        Notes:
            Pairs are returned in LIFO order. popitem() is useful to
            destructively iterate over a dictionary, as often used in set
            algorithms. If the dictionary is empty, calling popitem() raises a
            EmptyDictError.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        print(len(my_dict))  # => 2

        var item = my_dict.popitem()
        print(item.key, item.value)  # => Either "b", 2 or "a", 1
        print(len(my_dict))  # => 1
        ```
        """

        var i = len(self._order) - 1
        while i >= 0:
            var slot = Int(self._order[i])
            if is_occupied(self._table._ctrl[unsafe_offset=slot]):
                var entry = (
                    self._table._slots.unsafe_offset(slot)
                ).unsafe_take_pointee()
                self._table.set_ctrl(slot, CTRL_DELETED)
                self._table._len -= 1
                # Every `_order` entry from `i` up refers to a slot that is now
                # unoccupied, so drop them. Without this a destructive drain
                # rescans the same dead tail on each call, making it quadratic.
                self._order.shrink(i)
                return entry^
            i -= 1

        # Nothing occupied anywhere in `_order`, so all of it is stale.
        self._order.clear()
        raise EmptyDictError()

    def keys(
        ref self,
    ) -> _DictKeyIter[
        downcast[Self.K, KeyElement & Copyable],
        downcast[Self.V, Copyable],
        Self.H,
        origin_of(self),
    ]:
        """Iterate over the dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        for key in my_dict.keys():
            print(key) # prints a then b or b then a
            # All keys will be printed, but order is not guaranteed
        ```
        """
        return Self.__iter__(self)

    # TODO(MOCO-4308): Remove redundant `KeyElement` constraints
    # TODO(MSTDL-2390): Remove `Copyable` constraints once we have better iter traits.
    def values(
        ref self,
    ) -> _DictValueIter[
        Self.K,
        Self.V,
        Self.H,
        origin_of(self),
    ] where conforms_to(Self.K, KeyElement & Copyable) and conforms_to(
        Self.V, Copyable
    ):
        """Iterate over the dict's values as references.

        Returns:
            An iterator of references to the dictionary values.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        for value in my_dict.values():
            print(value) # prints 1 then 2 or 2 then 1
            # All values will be printed, but order is not guaranteed
        ```
        """
        return _DictValueIter(_DictEntryIter(0, 0, Pointer(to=self)))

    def items(
        ref self,
    ) -> _DictEntryIter[
        downcast[Self.K, KeyElement & Copyable],
        downcast[Self.V, Copyable],
        Self.H,
        origin_of(self),
    ]:
        """Iterate over the dict's entries as immutable references.

        Returns:
            An iterator of immutable references to the dictionary entries.

        Examples:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for item in my_dict.items():
            print(item.key, item.value) # prints a 1 then b 2 or b 2 then a 1
            # All entries will be printed, but order is not guaranteed
        ```

        Notes:
            These can't yet be unpacked like Python dict items, but you can
            access the key and value as attributes.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraints once we have better iter traits.
        comptime assert conforms_to(Self.K, Copyable) and conforms_to(
            Self.V, Copyable
        ), "Dict iteration requires the key and value types to be `Copyable`."
        comptime DictCopyable = Dict[
            downcast[Self.K, KeyElement & Copyable],
            downcast[Self.V, Copyable],
            Self.H,
        ]
        return _DictEntryIter(
            0,
            0,
            rebind[Pointer[DictCopyable, origin_of(self)]](Pointer(to=self))[],
        )

    def take_items(
        mut self,
    ) -> _TakeDictEntryIter[
        Self.K,
        Self.V,
        Self.H,
        origin_of(self),
    ] where conforms_to(Self.K, KeyElement & Deinitable) and conforms_to(
        Self.V, Movable & Deinitable
    ):
        """Iterate over the dict's entries and move them out of the dictionary
        effectively draining the dictionary.

        Returns:
            An iterator of mutable references to the dictionary entries that
            moves them out of the dictionary.

        Examples:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for entry in my_dict.take_items():
            print(entry.key, entry.value) # prints a 1 then b 2 or b 2 then a 1
            # All entries will be printed, but order is not guaranteed

        print(len(my_dict))
        # prints 0
        ```
        """
        return _TakeDictEntryIter(Pointer(to=self)[])

    def update(
        mut self, other: Self, /
    ) where conforms_to(Self.K, Copyable & Deinitable) and conforms_to(
        Self.V, Copyable & Deinitable
    ):
        """Update the dictionary with the key/value pairs from other,
        overwriting existing keys.

        Args:
            other: The dictionary to update from.

        Notes:
            The argument must be positional only.

        Example:

        ```mojo
        var dict1 = Dict[String, Int]()
        dict1["a"] = 1
        dict1["b"] = 2
        var dict2 = Dict[String, Int]()
        dict2["b"] = 3
        dict2["c"] = 4
        dict1.update(dict2)
        print(dict1)  # => {"a": 1, "b": 3, "c": 4}
        ```
        """
        # `other` shares the hasher `H`, so reuse each entry's cached hash
        # instead of letting `__setitem__` recompute it.
        for entry in other.items():
            self._insert(
                DictEntry[Self.K, Self.V, Self.H](
                    entry.key.copy(),
                    entry.value.copy(),
                    unsafe_hash=entry._hash,
                )
            )

    def clear(
        mut self,
    ) where conforms_to(Self.K, Deinitable) and conforms_to(Self.V, Deinitable):
        """Remove all elements from the dictionary.

        Constraints:
            Both `K` and `V` must be `Deinitable`, since every entry is
            destroyed in place.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2
        print(len(my_dict))  # => 2
        my_dict.clear()
        print(len(my_dict))  # => 0
        ```
        """
        self._table.clear()
        self._order.clear()

    def clear_with(
        mut self, destroy_func: Some[def(var Self.K, var Self.V)], /
    ):
        """Remove all elements, disposing each entry with a closure.

        The closure counterpart of `clear`: instead of destroying each entry in
        place, it hands the key and value to `destroy_func`. Use this to clear a
        `Dict` whose keys or values are not `Deinitable`. The
        dictionary's capacity is retained, so it can be reused without
        reallocating.

        Args:
            destroy_func: A closure called once per entry to dispose its key and
                value.

        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        var cleared = List[Int]()

        def dispose(var key: String, var value: Int) {mut}:
            cleared.append(value)

        my_dict.clear_with(dispose)
        print(len(my_dict))     # => 0
        print(len(cleared))     # => 2
        ```
        """
        self._table.clear_with(destroy_func)
        self._order.clear()

    @always_inline
    def setdefault(
        mut self, var key: Self.K, var default: Self.V
    ) -> ref[
        origin_of(self)._get_owned_interior["value"]
    ] Self.V where conforms_to(Self.K, Deinitable) and conforms_to(
        Self.V, Deinitable
    ):
        """Get a value from the dictionary by key, or set it to a default if it
        doesn't exist.

        Args:
            key: The key to search for in the dictionary.
            default: The default value to set if the key is not present.

        Returns:
            The value associated with the key, or the default value if it wasn't
            present.


        Example:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        var value1 = my_dict.setdefault("a", 99)
        print(value1)  # => 1

        var value2 = my_dict.setdefault("b", 99)
        print(value2)  # => 99
        print(my_dict)  # => {"a": 1, "b": 99}
        ```
        """
        self._ensure_capacity()
        var h = hash[Self.H](key)
        var found, slot_idx = self._table.find_slot(h, key)
        if not found:
            var entry = DictEntry[Self.K, Self.V, Self.H](
                key^, default^, unsafe_hash=h
            )
            self._table.set_ctrl(slot_idx, h2(h))
            (self._table._slots.unsafe_offset(slot_idx)).unsafe_write(entry^)
            self._order.append(Int32(slot_idx))
            self._table._len += 1
            self._table._growth_left -= 1
        else:
            assert is_occupied(
                self._table._ctrl[unsafe_offset=slot_idx]
            ), "_find_slot returned found=True but ctrl byte is not occupied"
        return Pointer(
            to=(self._table._slots.unsafe_offset(slot_idx))[].value
        )._get_ref_with_unsafe_interior_origin["value", origin_of(self)]()

    # ===-------------------------------------------------------------------===#
    # Internal methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def _find_slot(self, hash: UInt64, key: Self.K) -> Tuple[Bool, Int]:
        """Find a slot matching the given key, or an empty slot for insertion.

        This is a forwarding method to the underlying SwissTable for
        backwards compatibility with internal callers.

        Args:
            hash: The hash of the key.
            key: The key to search for.

        Returns:
            A tuple of (found, slot_index).
        """
        return self._table.find_slot(hash, key)

    @always_inline
    def _set_ctrl(mut self, index: Int, value: UInt8):
        """Set a control byte. Forwards to the underlying SwissTable.

        Args:
            index: The slot index.
            value: The control byte value.
        """
        self._table.set_ctrl(index, value)

    @always_inline
    def _place_new_entry(
        mut self, slot_idx: Int, var entry: DictEntry[Self.K, Self.V, Self.H]
    ):
        """Write a new entry into a known-empty slot.

        The caller must have located `slot_idx` via `find_slot` and confirmed
        the key is absent, so no existing entry is displaced or destroyed. This
        keeps the placement path free of implicit destruction, so it works even
        when `K` or `V` is linear (non-`Deinitable`).

        Args:
            slot_idx: The empty slot to write into, as returned by `find_slot`.
            entry: The new entry to store.
        """
        self._table.set_ctrl(slot_idx, h2(entry._hash))
        (self._table._slots.unsafe_offset(slot_idx)).unsafe_write(entry^)
        self._order.append(Int32(slot_idx))
        self._table._len += 1
        self._table._growth_left -= 1
        assert (
            self._table._growth_left >= 0
        ), "_growth_left went negative after insert"

    @always_inline
    def _insert(
        mut self, var key: Self.K, var value: Self.V
    ) where conforms_to(Self.K, Deinitable) and conforms_to(Self.V, Deinitable):
        self._insert(DictEntry[Self.K, Self.V, Self.H](key^, value^))

    @always_inline
    def _insert[
        safe_context: Bool = False
    ](mut self, var entry: DictEntry[Self.K, Self.V, Self.H]) where conforms_to(
        Self.K, Deinitable
    ) and conforms_to(Self.V, Deinitable):
        comptime if not safe_context:
            self._ensure_capacity()
        var found, slot_idx = self._table.find_slot(entry._hash, entry.key)

        if found:
            # Update existing entry: destroy old, move new in
            (self._table._slots.unsafe_offset(slot_idx)).unsafe_deinit_pointee()
            (self._table._slots.unsafe_offset(slot_idx)).unsafe_write(entry^)
        else:
            # New entry
            self._place_new_entry(slot_idx, entry^)

    @always_inline
    def _ensure_capacity(mut self):
        """Ensures the table has room for one more insertion.

        Handles three cases:
        - Lazy state (`_capacity == 0`): allocates `INITIAL_CAPACITY` slots
          and initializes `_order`.
        - Sparse occupancy with exhausted growth: rehashes in-place to
          reclaim tombstones without growing the buffer.
        - Otherwise, when growth budget is exhausted: doubles capacity and
          rehashes.

        If no growth is needed, opportunistically compacts `_order`.

        Must be called before any insert that may store into a slot
        returned by `find_slot`; in lazy state, `find_slot` returns a
        meaningless slot index that is only safe to use after this call
        has allocated the backing buffer.

        Notes:
            Only the guard is inlined into the caller; the cold resize body
            lives in `_grow_slow` so that inlining an insert does not drag a
            rehash along with it.
        """
        if self._table.needs_resize():
            self._grow_slow()
        else:
            self._maybe_compact_order()

    @no_inline
    def _grow_slow(mut self):
        """Grows or rehashes the table. See `_ensure_capacity`."""
        # First allocation from the lazy empty state: no entries to rehash and
        # no `_order` to rebuild. Allocate INITIAL_CAPACITY directly.
        if self._table._capacity == 0:
            # `resize` returns the relocations list; empty here since _len == 0.
            _ = self._table.resize(INITIAL_CAPACITY)
            self._order = List[Int32](capacity=self._table._capacity * 7 // 8)
            return

        # If table is sparse (occupancy <= 7/16 ~ 44% of capacity), tombstones
        # dominate. Rehash in-place to reclaim them without doubling memory.
        if self._table.is_sparse():
            self._rehash_in_place()
            return

        # Double capacity and rehash
        var old_capacity = self._table._capacity
        var new_capacity = old_capacity * 2
        var old_order = self._order^

        var relocations = self._table.resize(new_capacity)

        # Build old_slot -> new_slot mapping and a set of relocated old slots
        # so we can filter stale _order entries (DELETED slots won't appear
        # in relocations since resize only moves occupied entries).
        var slot_map = alloc(Layout[Int32](count=old_capacity)).unsafe_leak()
        var relocated_set = alloc(
            Layout[UInt8](count=old_capacity)
        ).unsafe_leak()
        unsafe_memset(relocated_set, 0, old_capacity)
        for i in range(len(relocations)):
            slot_map.unsafe_store(relocations[i][0], Int32(relocations[i][1]))
            relocated_set.unsafe_store(relocations[i][0], UInt8(1))

        # Rebuild _order preserving insertion order, skipping stale entries
        self._order = List[Int32](capacity=self._table._len)
        for i in range(len(old_order)):
            var old_slot = Int(old_order[i])
            if relocated_set.unsafe_load(old_slot) != 0:
                self._order.append(slot_map.unsafe_load(old_slot))

        assert (
            len(self._order) == self._table._len
        ), "order length doesn't match _len after resize"

        dealloc(
            ThinAllocation(unsafe_owned_ptr=slot_map).unsafe_with_layout(
                {count = old_capacity}
            )
        )
        dealloc(
            ThinAllocation(unsafe_owned_ptr=relocated_set).unsafe_with_layout(
                {count = old_capacity}
            )
        )

    @no_inline
    def _rehash_in_place(mut self):
        """Rehash the table in place without changing capacity."""
        # Compact _order to remove stale entries before we lose
        # track of which slots are occupied vs deleted.
        var compacted = List[Int32](capacity=self._table._len)
        for j in range(len(self._order)):
            var slot = Int(self._order[j])
            if is_occupied(self._table._ctrl[unsafe_offset=slot]):
                compacted.append(self._order[j])
        self._order = compacted^

        # Delegate the actual rehash to the table, get slot mapping back
        var slot_map = self._table.rehash_in_place()

        # Update _order with new slot indices
        for j in range(len(self._order)):
            self._order[j] = slot_map[unsafe_offset=Int(self._order[j])]

        assert (
            len(self._order) == self._table._len
        ), "order length doesn't match _len after in-place rehash"

        dealloc(
            ThinAllocation(unsafe_owned_ptr=slot_map).unsafe_with_layout(
                {count = self._table._capacity}
            )
        )

    def _maybe_compact_order(mut self):
        """Compact the order array if it has too many stale entries."""
        if len(self._order) <= 2 * self._table._len:
            return
        var new_order = List[Int32](capacity=self._table._len)
        for i in range(len(self._order)):
            var slot = Int(self._order[i])
            if is_occupied(self._table._ctrl[unsafe_offset=slot]):
                new_order.append(self._order[i])
        self._order = new_order^


struct StringDict[V: Movable](
    Copyable where conforms_to(V, Copyable),
    Defaultable,
    Deinitable where conforms_to(V, Deinitable),
    Iterable,
    Movable,
    Sized,
    Writable where conforms_to(V, Writable),
):
    """Container used to pass owned variadic keyword arguments to functions.

    Parameters:
        V: The value type of the dictionary. Must be `Movable`. When `V` is not
            `Deinitable`, the dictionary has no implicit destructor and
            must be torn down with `deinit_with()`.

    This type mimics the interface of a dictionary with `String` keys, and
    should be usable more-or-less like a dictionary. Notably, however, this type
    should not be instantiated directly by users.
    """

    # Fields
    comptime key_type = String
    """The key type for this dictionary (always String)."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _DictKeyIter[
        Self.key_type,
        downcast[Self.V, Copyable],
        default_comp_time_hasher,
        iterable_origin,
    ]
    """The iterator type for this dictionary.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    var _dict: Dict[Self.key_type, Self.V, default_comp_time_hasher]

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    def __init__(out self):
        """Initialize an empty keyword dictionary."""
        self._dict = Dict[Self.key_type, Self.V, default_comp_time_hasher]()

    def deinit_with(
        deinit self, deinit_func: Some[def(var String, var Self.V)], /
    ):
        """Consume the dictionary, deinitializing each key/value pair with a closure.

        Use this to tear down a keyword dictionary whose values are not
        `Deinitable`.

        Args:
            deinit_func: A closure called once per entry to destroy its key and
                value.
        """

        self._dict^.deinit_with(deinit_func)

    # ===-------------------------------------------------------------------===#
    # Operator dunders
    # ===-------------------------------------------------------------------===#

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, ref key: Self.key_type
    ) raises DictKeyError[Self.key_type] -> ref[self._dict[key]] Self.V:
        """Retrieve a value out of the keyword dictionary.

        Args:
            key: The key to retrieve.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            `DictKeyError` if the key isn't present.
        """
        return self._dict[key]

    @__unsafe_nested_origins_read_only
    @always_inline
    def __getitem__(
        ref self, key: ImmStringSpan
    ) raises DictKeyError[Self.key_type] -> ref[
        origin_of(self._dict)._get_owned_interior["value"]
    ] Self.V:
        """Retrieve a value by string view, without allocating a `String`.

        Args:
            key: The key to retrieve, as a string view.

        Returns:
            The value associated with the key, if it's present.

        Raises:
            `DictKeyError` if the key isn't present.
        """

        return self._dict._find_ref_matching(
            hash[default_comp_time_hasher](key),
            lambda (stored_key: Self.key_type) {imm key} -> Bool: (
                stored_key == key
            ),
        )

    @always_inline
    def __setitem__(
        mut self, key: Self.key_type, var value: Self.V
    ) where conforms_to(Self.V, Deinitable):
        """Set a value in the keyword dictionary by key.

        Constraints:
            `V` must be `Deinitable`, since assigning to an existing key
            destroys the displaced value in place. To populate a keyword
            dictionary with a linear `V`, use `insert()` instead.

        Args:
            key: The key to associate with the specified value.
            value: The data to store in the dictionary.
        """
        self._dict[key] = value^

    # ===-------------------------------------------------------------------===#
    # Trait implementations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __contains__(self, key: Self.key_type) -> Bool:
        """Check if a given key is in the keyword dictionary or not.

        Args:
            key: The key to check.

        Returns:
            True if there key exists in the keyword dictionary, False
            otherwise.
        """
        return key in self._dict

    @always_inline
    def __len__(self) -> Int:
        """The number of elements currently stored in the keyword dictionary.

        Returns:
            The number of elements currently stored in the keyword dictionary.
        """
        return len(self._dict)

    @no_inline
    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.V, Writable):
        """Write this `StringDict` to the writer.

        Args:
            writer: The value to write to.
        """
        self._dict.write_to(writer)

    @no_inline
    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.V, Writable):
        """Write this `StringDict`'s representation to the writer.

        Args:
            writer: The value to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._dict._write_dict_body[
                f_key=fmt.write_repr_to[Self.key_type],
                f_val=fmt.write_repr_to[Self.V],
            ](w)

        fmt.FormatStruct(writer, "StringDict").params(
            fmt.TypeNames[Self.V](),
        ).fields(write_fields)

    # ===-------------------------------------------------------------------===#
    # Methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def find(
        self, key: Self.key_type
    ) -> Optional[Self.V] where conforms_to(Self.V, Copyable):
        """Find a value in the keyword dictionary by key.

        Args:
            key: The key to search for in the dictionary.

        Returns:
            An optional value containing a copy of the value if it was present,
            otherwise an empty Optional.
        """
        return self._dict.find(key)

    @always_inline
    def insert(
        mut self, var key: Self.key_type, var value: Self.V
    ) -> Optional[DictEntry[Self.key_type, Self.V, default_comp_time_hasher]]:
        """Insert a key/value pair, returning the displaced entry if the key was
        already present.

        Unlike `__setitem__`, the displaced entry is moved out and returned
        (never destroyed in place), so this works when `V` is linear
        (non-`Deinitable`). The caller is responsible for disposing of
        the returned entry.

        Args:
            key: The key to associate with the value.
            value: The value to store.

        Returns:
            The previous entry if `key` was already present, otherwise an empty
            `Optional`.
        """
        return self._dict.insert(key^, value^)

    @always_inline
    def popitem(
        mut self,
    ) raises EmptyDictError -> DictEntry[
        Self.key_type, Self.V, default_comp_time_hasher
    ]:
        """Remove and return a (key, value) pair from the dictionary.

        The entry is moved out whole (nothing is destroyed in place), so this
        works when `V` is linear. The caller is responsible for disposing of the
        returned entry.

        Returns:
            The removed entry.

        Raises:
            `EmptyDictError` if the dictionary is empty.
        """
        return self._dict.popitem()

    @always_inline
    def pop(
        mut self, key: self.key_type, var default: Self.V
    ) -> Self.V where conforms_to(Self.V, Deinitable):
        """Remove a value from the dictionary by key.

        Constraints:
            `V` must be `Deinitable`, since the unused `default` is
            discarded in place when the key is found. To remove from a keyword
            dictionary with a linear `V`, use `pop(key)` or `popitem()`.

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
    def pop(
        mut self, ref key: self.key_type
    ) raises DictKeyError[Self.key_type] -> Self.V:
        """Remove a value from the dictionary by key.

        Args:
            key: The key to remove from the dictionary.

        Returns:
            The value associated with the key, if it was in the dictionary.
            Raises otherwise.

        Raises:
            `DictKeyError` if the key was not present in the dictionary.
        """
        return self._dict.pop(key)

    def __iter__(
        ref self,
    ) -> Self.IteratorType[origin_of(self)]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return rebind[Self.IteratorType[origin_of(self)]](self._dict.keys())

    def keys(
        ref self,
    ) -> _DictKeyIter[
        Self.key_type,
        downcast[Self.V, Copyable],
        default_comp_time_hasher,
        origin_of(self._dict),
    ]:
        """Iterate over the keyword dict's keys as immutable references.

        Returns:
            An iterator of immutable references to the dictionary keys.
        """
        return self._dict.keys()

    def values(
        ref self,
    ) -> _DictValueIter[
        Self.key_type,
        Self.V,
        default_comp_time_hasher,
        origin_of(self._dict),
    ] where conforms_to(Self.V, Copyable):
        """Iterate over the keyword dict's values as references.

        Returns:
            An iterator of references to the dictionary values.
        """
        return self._dict.values()

    def items(
        ref self,
    ) -> _DictEntryIter[
        Self.key_type,
        Self.V,
        default_comp_time_hasher,
        origin_of(self._dict),
    ] where conforms_to(Self.V, Copyable):
        """Iterate over the keyword dictionary's entries as immutable
        references.

        Returns:
            An iterator of immutable references to the dictionary entries.

        Examples:

        ```mojo
        var my_dict = Dict[String, Int]()
        my_dict["a"] = 1
        my_dict["b"] = 2

        for e in my_dict.items():
            print(e.key, e.value)
        ```

        Notes:
            These can't yet be unpacked like Python dict items, but you can
            access the key and value as attributes.
        """

        # TODO(#36448): Use this instead of the current workaround
        # return self[]._dict.items()
        return self._dict.items()

    @always_inline
    def _insert(
        mut self, var key: Self.key_type, var value: Self.V
    ) where conforms_to(Self.V, Deinitable):
        self._dict._insert(key^, value^)

    @always_inline
    def _insert(
        mut self, key: StringLiteral, var value: Self.V
    ) where conforms_to(Self.V, Deinitable):
        self._insert(String(key), value^)
