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
"""Swiss Table core for hash-based collections.

Provides the low-level Swiss Table implementation shared by `Dict` (ordered)
and `HashMap` (unordered). This module is internal and not part of the public
API.

The Swiss Table design uses SIMD group probing on 16-byte control byte groups
for fast lookups. Each slot has a 1-byte control: EMPTY (0xFF), DELETED (0x80),
or an h2 fingerprint (0x00-0x7F) derived from the top 7 bits of the hash.
"""

from std.bit import count_trailing_zeros, next_power_of_two
from std.hashlib import Hasher, default_hasher
from std.math import ceildiv
from std.memory import (
    alloc,
    dealloc,
    ThinAllocation,
    unsafe_memcpy,
    unsafe_memset,
    pack_bits,
)
from std.memory.alloc import Layout
from std.traits import IsTriviallyDeinitable
from std.sys.intrinsics import likely

# ===-----------------------------------------------------------------------===#
# Swiss Table constants and helpers
# ===-----------------------------------------------------------------------===#

comptime CTRL_EMPTY: UInt8 = 0xFF
"""Control byte for an empty slot."""
comptime CTRL_DELETED: UInt8 = 0x80
"""Control byte for a deleted (tombstone) slot."""
comptime GROUP_WIDTH: Int = 16
"""Number of control bytes processed in one SIMD operation."""
comptime INITIAL_CAPACITY: Int = 16
"""Minimum table capacity. Must be >= GROUP_WIDTH."""


@always_inline
def h2(hash: UInt64) -> UInt8:
    """Extract the top 7 bits of the hash as a fingerprint (0x00-0x7F).

    Args:
        hash: The full hash value.

    Returns:
        The 7-bit fingerprint.
    """
    return UInt8(hash >> 57)


@always_inline
def is_occupied(ctrl: UInt8) -> Bool:
    """Check if a control byte represents an occupied slot.

    Occupied slots have h2 values in range 0x00-0x7F (top bit clear).
    DELETED (0x80) and EMPTY (0xFF) both have top bit set.

    Args:
        ctrl: The control byte to check.

    Returns:
        True if the slot is occupied.
    """
    return ctrl < CTRL_DELETED


# ===-----------------------------------------------------------------------===#
# Group: SIMD group operations on 16 control bytes
# ===-----------------------------------------------------------------------===#


@fieldwise_init
struct Group(Copyable, Movable):
    """A group of control bytes for SIMD probing.

    Loads 16 control bytes at once and performs parallel matching using
    SIMD comparison operations, enabling fast hash table lookups.
    """

    var ctrl: SIMD[.uint8, GROUP_WIDTH]

    @always_inline
    def __init__(out self, ptr: Pointer[UInt8, _]):
        """Load a group of control bytes from memory.

        Args:
            ptr: Pointer to the start of 16 consecutive control bytes.
        """
        self.ctrl = ptr.unsafe_load[width=GROUP_WIDTH]()

    # TODO: Remove `__is_run_in_comptime_interpreter` branches once `pack_bits` is supported
    # by the compile-time interpreter. Currently `pack_bits` uses `pop.bitcast`
    # which the interpreter can't handle, so we fall back to scalar loops for
    # comptime contexts (e.g., Dict used in `comptime` expressions).

    @always_inline
    def match_h2(self, h2_val: UInt8) -> UInt16:
        """Return a bitmask of slots matching the given h2 fingerprint.

        Args:
            h2_val: The h2 fingerprint to match (0x00-0x7F).

        Returns:
            A bitmask where bit i is set if ctrl[i] == h2_val.
        """
        if __is_run_in_comptime_interpreter:
            return Self._scalar_match(self.ctrl, h2_val)
        return pack_bits(self.ctrl.eq(SIMD[.uint8, GROUP_WIDTH](h2_val)))

    @always_inline
    def match_empty(self) -> UInt16:
        """Return a bitmask of empty slots.

        Returns:
            A bitmask where bit i is set if ctrl[i] == EMPTY (0xFF).
        """
        if __is_run_in_comptime_interpreter:
            return Self._scalar_match(self.ctrl, CTRL_EMPTY)
        return pack_bits(self.ctrl.eq(SIMD[.uint8, GROUP_WIDTH](CTRL_EMPTY)))

    @always_inline
    def match_empty_or_deleted(self) -> UInt16:
        """Return a bitmask of empty or deleted slots.

        Both EMPTY (0xFF) and DELETED (0x80) have the top bit set,
        so they are >= 0x80. All occupied h2 values are 0x00-0x7F.

        Returns:
            A bitmask where bit i is set if ctrl[i] is EMPTY or DELETED.
        """
        if __is_run_in_comptime_interpreter:
            var result = UInt16(0)

            comptime for i in range(GROUP_WIDTH):
                if self.ctrl[i] >= CTRL_DELETED:
                    result |= UInt16(1) << UInt16(i)
            return result
        return pack_bits(self.ctrl.ge(SIMD[.uint8, GROUP_WIDTH](CTRL_DELETED)))

    @always_inline
    def convert_special_to_empty_and_full_to_deleted(
        self,
    ) -> SIMD[.uint8, GROUP_WIDTH]:
        """Convert ctrl bytes for in-place rehash preparation.

        EMPTY  (0xFF) -> EMPTY  (0xFF)  (unchanged)
        DELETED(0x80) -> EMPTY  (0xFF)  (reclaim tombstone)
        h2 (0x00-0x7F) -> DELETED(0x80) (mark for relocation)

        Returns:
            Transformed control byte vector.
        """
        if __is_run_in_comptime_interpreter:
            var result = SIMD[.uint8, GROUP_WIDTH](0)

            comptime for i in range(GROUP_WIDTH):
                if self.ctrl[i] < CTRL_DELETED:
                    result[i] = CTRL_DELETED
                else:
                    result[i] = CTRL_EMPTY
            return result
        var is_full = self.ctrl.lt(SIMD[.uint8, GROUP_WIDTH](CTRL_DELETED))
        return is_full.select(
            SIMD[.uint8, GROUP_WIDTH](CTRL_DELETED),
            SIMD[.uint8, GROUP_WIDTH](CTRL_EMPTY),
        )

    @staticmethod
    @always_inline
    def _scalar_match(ctrl: SIMD[.uint8, GROUP_WIDTH], target: UInt8) -> UInt16:
        """Scalar fallback for compile-time evaluation.

        Args:
            ctrl: The control byte vector.
            target: The byte value to match.

        Returns:
            A bitmask where bit i is set if ctrl[i] == target.
        """
        var result = UInt16(0)

        comptime for i in range(GROUP_WIDTH):
            if ctrl[i] == target:
                result |= UInt16(1) << UInt16(i)
        return result


# ===-----------------------------------------------------------------------===#
# SwissTableEntry
# ===-----------------------------------------------------------------------===#


@fieldwise_init
@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `SwissTableEntry` with"
    " non-`Deinitable` keys or values"
)
struct SwissTableEntry[
    K: KeyElement,
    V: Movable,
    H: Hasher,
](
    Copyable where conforms_to(K, Copyable) and conforms_to(V, Copyable),
    Deinitable where conforms_to(K, Deinitable) and conforms_to(V, Deinitable),
    Movable,
):
    """Store a key-value pair entry inside a Swiss Table-based collection.

    Parameters:
        K: The key type. Must be `Movable`, `Hashable`, and `Equatable`.
            `Copyable` is required only for entry copy construction.
        V: The value type. Must be `Movable`. `Copyable` is required only for
            entry copy construction.
        H: The type of the hasher used to hash the key.
    """

    var _hash: UInt64
    """`key.__hash__()`, stored so hashing isn't re-computed during lookup."""
    var key: Self.K
    """The unique key for the entry."""
    var value: Self.V
    """The value associated with the key."""

    def __init__(out self, var key: Self.K, var value: Self.V):
        """Create an entry from a key and value, computing the hash.

        Args:
            key: The key of the entry.
            value: The value of the entry.
        """
        self._hash = hash[Self.H](key)
        self.key = key^
        self.value = value^

    @always_inline
    def __init__(
        out self, var key: Self.K, var value: Self.V, *, unsafe_hash: UInt64
    ):
        """Create an entry from a key and value using a caller-provided hash.

        This skips recomputing the key's hash. Use it when the caller has
        already computed `hash[H](key)`.

        Args:
            key: The key of the entry.
            value: The value of the entry.
            unsafe_hash: The precomputed hash of `key`. Must equal
                `hash[H](key)`; passing any other value corrupts slot lookup.
        """
        self._hash = unsafe_hash
        self.key = key^
        self.value = value^

    def deinit_with(
        deinit self, deinit_func: Some[def(var Self.K, var Self.V)]
    ):
        """Deinitializes the entry's key and value using a caller-provided closure.

        Args:
            deinit_func: A closure that consumes the entry's key and value.
        """
        deinit_func(self.key^, self.value^)

    def reap_key(
        deinit self,
    ) -> Self.K where conforms_to(Self.V, Deinitable):
        """Take the key from an owned entry, discarding hash and value.

        Constraints:
            `V` must be `Deinitable`, since the value is discarded.

        Returns:
            The key of the entry.
        """
        return self.key^

    def reap_value(
        deinit self,
    ) -> Self.V where conforms_to(Self.K, Deinitable):
        """Take the value from an owned entry.

        Constraints:
            `K` must be `Deinitable`, since the key is discarded.

        Returns:
            The value of the entry.
        """
        return self.value^


# ===-----------------------------------------------------------------------===#
# SwissTable
# ===-----------------------------------------------------------------------===#


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy a `SwissTable` with"
    " non-`Deinitable` keys or values"
)
struct SwissTable[
    K: KeyElement,
    V: Movable,
    H: Hasher = default_hasher,
](
    Copyable where conforms_to(K, Copyable) and conforms_to(V, Copyable),
    Deinitable where conforms_to(K, Deinitable) and conforms_to(V, Deinitable),
    Movable,
):
    """Raw Swiss Table providing the hash table core for Dict and HashMap.

    This struct manages the control byte array, slot array, probing, and
    resize logic. Higher-level types (`Dict`, `HashMap`) compose this with
    their own iteration and ordering strategy.

    Parameters:
        K: The key type. Must be `Movable`, `Hashable`, and `Equatable`.
            `Copyable` is required only for table copy construction.
        V: The value type. Must be `Movable`. `Copyable` is required only for
            table copy construction.
        H: The hasher type.
    """

    var _ctrl: Pointer[UInt8, MutUntrackedOrigin]
    """Control byte array. Size is _capacity + GROUP_WIDTH.
    Each byte is EMPTY (0xFF), DELETED (0x80), or h2 fingerprint (0x00-0x7F).
    The last GROUP_WIDTH bytes mirror the first GROUP_WIDTH for SIMD wrapping.
    """

    var _slots: Pointer[
        SwissTableEntry[Self.K, Self.V, Self.H], MutUntrackedOrigin
    ]
    """Flat slot array. Size is _capacity. Only occupied slots are initialized.
    """

    var _len: Int
    """The number of live elements currently stored."""

    var _capacity: Int
    """The number of slots (always a power of 2, >= INITIAL_CAPACITY)."""

    var _growth_left: Int
    """Number of EMPTY slots remaining before a resize is needed.
    Decremented on each new insertion into an EMPTY slot. Reset on resize.
    """

    # ===-------------------------------------------------------------------===#
    # Life cycle methods
    # ===-------------------------------------------------------------------===#

    @always_inline
    def __init__(out self):
        """Initialize an empty Swiss Table."""
        self._capacity = 0
        self._ctrl = type_of(self._ctrl).unsafe_dangling()
        self._slots = type_of(self._slots).unsafe_dangling()
        self._len = 0
        self._growth_left = 0

    @always_inline
    def __init__(out self, *, capacity: Int):
        """Initialize an empty Swiss Table with a pre-reserved capacity.

        The capacity is defined by `next_power_of_two(ceildiv(capacity * 8, 7))`
        (minimum 16) to satisfy internal layout requirements.

        Args:
            capacity: The requested minimum number of slots.
        """
        if capacity == 0:
            self = Self()
            return

        self._capacity = max(
            next_power_of_two(ceildiv(capacity * 8, 7)), INITIAL_CAPACITY
        )
        self._ctrl = alloc(
            Layout[UInt8](count=self._capacity + GROUP_WIDTH)
        ).unsafe_leak()
        unsafe_memset(self._ctrl, CTRL_EMPTY, self._capacity + GROUP_WIDTH)
        self._slots = alloc(
            Layout[SwissTableEntry[Self.K, Self.V, Self.H]](
                count=self._capacity
            )
        ).unsafe_leak()
        self._len = 0
        self._growth_left = self._capacity * 7 // 8

    def __init__(
        out self, *, copy: Self
    ) where conforms_to(Self.K, Copyable) and conforms_to(Self.V, Copyable):
        """Copy an existing Swiss Table.

        Args:
            copy: The existing table to copy.
        """
        if copy._capacity == 0:
            self = Self()
            return

        self._capacity = copy._capacity
        self._len = copy._len
        self._growth_left = copy._growth_left

        self._ctrl = alloc(
            Layout[UInt8](count=self._capacity + GROUP_WIDTH)
        ).unsafe_leak()
        unsafe_memcpy(
            dest=self._ctrl,
            src=copy._ctrl,
            count=self._capacity + GROUP_WIDTH,
        )

        self._slots = alloc(
            Layout[SwissTableEntry[Self.K, Self.V, Self.H]](
                count=self._capacity
            )
        ).unsafe_leak()
        for i in range(self._capacity):
            if is_occupied(self._ctrl[unsafe_offset=i]):
                (self._slots.unsafe_offset(i)).unsafe_write(
                    copy=(copy._slots.unsafe_offset(i))[]
                )

    def __deinit__(
        deinit self,
    ) where conforms_to(Self.K, Deinitable) and conforms_to(Self.V, Deinitable):
        """Destroy all entries and free memory.

        Constraints:
            Both `K` and `V` must be `Deinitable`. When either is not,
            the table has no implicit destructor and must be torn down with
            `deinit_with()`.
        """
        self._delete_occupied_entries()
        self^._deallocate_storage()

    def deinit_with(
        var self, deinit_func: Some[def(var Self.K, var Self.V)], /
    ):
        """Deinitializes all entries with a caller-provided closure, then free memory.

        Use this to tear down a `SwissTable` whose keys or values are not
        `Deinitable`. The closure is called once per occupied entry.

        Args:
            deinit_func: A closure that consumes each entry's key and value.
        """
        self._delete_occupied_entries_with(deinit_func)
        self^._deallocate_storage()

    def _deallocate_storage(deinit self):
        """Free the control and slot arrays without touching entry contents.

        The caller must have already destroyed or moved out every occupied
        entry (see `__deinit__` and `deinit_with`).
        """
        if self._capacity > 0:
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._ctrl).unsafe_with_layout(
                    {count = self._capacity + GROUP_WIDTH}
                )
            )
            dealloc(
                ThinAllocation(unsafe_owned_ptr=self._slots).unsafe_with_layout(
                    {count = self._capacity}
                )
            )

    @always_inline
    def _delete_occupied_entries(
        mut self,
    ) where conforms_to(Self.K, Deinitable) and conforms_to(Self.V, Deinitable):
        """Run destructors on every occupied slot.

        Skips the loop entirely when the entry type is trivially destructible.

        This leaves `_ctrl`, `_len`, and `_capacity` unchanged, so the table is
        in an invalid state afterward: the caller must either reset the control
        bytes (see `clear`) or be about to free the backing storage (see
        `__deinit__`).

        Constraints:
            Both `K` and `V` must be `Deinitable`, since entries are
            destroyed in place.
        """
        comptime if not IsTriviallyDeinitable[
            SwissTableEntry[Self.K, Self.V, Self.H]
        ]:
            for i in range(self._capacity):
                if is_occupied(self._ctrl[unsafe_offset=i]):
                    (self._slots.unsafe_offset(i)).unsafe_deinit_pointee()

    @always_inline
    def _delete_occupied_entries_with(
        mut self, destroy_func: Some[def(var Self.K, var Self.V)]
    ):
        """Destroy every occupied slot using a caller-provided closure.

        The closure counterpart of `_delete_occupied_entries`.
        """
        for i in range(self._capacity):
            if is_occupied(self._ctrl[unsafe_offset=i]):
                (
                    self._slots.unsafe_offset(i)
                ).unsafe_take_pointee().deinit_with(destroy_func)

    # ===-------------------------------------------------------------------===#
    # Core operations
    # ===-------------------------------------------------------------------===#

    @always_inline
    def set_ctrl(mut self, index: Int, value: UInt8):
        """Set a control byte, maintaining the mirror for wrap-around SIMD loads.

        Args:
            index: The slot index.
            value: The control byte value (h2, EMPTY, or DELETED).
        """
        assert 0 <= index < self._capacity, "ctrl index out of bounds"
        self._ctrl[unsafe_offset=index] = value
        if index < GROUP_WIDTH:
            self._ctrl[unsafe_offset=self._capacity + index] = value

    @always_inline
    def find_slot(self, hash: UInt64, key: Self.K) -> Tuple[Bool, Int]:
        """Find a slot matching the given key, or the first EMPTY slot.

        This does NOT return DELETED slots for insertion. This is required
        for ordered collections (Dict) where slot indices must correspond
        to insertion order. Unordered collections (HashMap) should use
        `find_slot_or_deleted` instead for better tombstone reuse.

        Args:
            hash: The hash of the key.
            key: The key to search for.

        Returns:
            A tuple of (found, slot_index). If found, slot_index is the
            matching slot. If not found, slot_index is the first EMPTY slot
            suitable for insertion.
        """

        return self.find_slot_matching(
            hash,
            lambda (stored_key: Self.K) {imm key} -> Bool: (stored_key == key),
        )

    @always_inline
    def find_slot_matching(
        self, hash: UInt64, key_matches: Some[def(Self.K) -> Bool]
    ) -> Tuple[Bool, Int]:
        """Find a slot whose stored key satisfies `key_matches`.

        Like `find_slot`, but probes with a predicate instead of a concrete
        `Self.K`, so callers can look up a heterogeneous key (such as a
        `StringSpan` against `String` keys) without building a `Self.K`.

        Args:
            hash: The hash of the lookup key, from `Self.H`. Any key that
                `key_matches` accepts must hash to this value.
            key_matches: Returns `True` when the stored key equals the lookup
                key.

        Returns:
            A tuple of (found, slot_index). If found, slot_index is the
            matching slot. If not found, slot_index is the first EMPTY slot
            suitable for insertion.
        """
        # Lazy state: no buffers allocated yet. Slot index is meaningless but
        # safe for lookups (callers check `found`). Insert paths must go
        # through `_ensure_capacity` first, which allocates; callers assert
        # that invariant after the call.
        if self._capacity == 0:
            return (False, 0)
        var h2_val = h2(hash)
        var pos = Int(hash) & (self._capacity - 1)

        while True:
            var group = Group(self._ctrl.unsafe_offset(pos))

            var match_mask = group.match_h2(h2_val)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & (self._capacity - 1)
                if (
                    self._slots.unsafe_offset(slot_idx)
                )[]._hash == hash and likely(
                    key_matches((self._slots.unsafe_offset(slot_idx))[].key)
                ):
                    return (True, slot_idx)
                match_mask &= match_mask - 1

            var empty_mask = group.match_empty()
            if empty_mask != 0:
                var bit = count_trailing_zeros(Int(empty_mask))
                return (False, (pos + bit) & (self._capacity - 1))

            pos = (pos + GROUP_WIDTH) & (self._capacity - 1)

    @always_inline
    def find_slot_or_deleted(
        self, hash: UInt64, key: Self.K
    ) -> Tuple[Bool, Int]:
        """Find a slot matching the given key, or the first EMPTY or DELETED slot.

        Unlike `find_slot`, this reuses DELETED (tombstone) slots for
        insertion. This is suitable for unordered collections (HashMap)
        where slot index stability is not required.

        Args:
            hash: The hash of the key.
            key: The key to search for.

        Returns:
            A tuple of (found, slot_index). If found, slot_index is the
            matching slot. If not found, slot_index is the first available
            (EMPTY or DELETED) slot for insertion.

        Notes:
            When not found, the caller must check `self._ctrl[slot_idx]` to
            determine if the slot is EMPTY or DELETED. Only insertions into
            EMPTY slots should decrement `_growth_left`.
        """
        # Mirrors find_slot's lazy-state guard; no in-tree callers exercise
        # this path yet (Dict uses find_slot), but kept for parity with
        # future unordered consumers of SwissTable.
        if self._capacity == 0:
            return (False, 0)
        var h2_val = h2(hash)
        var pos = Int(hash) & (self._capacity - 1)
        var first_deleted = -1

        while True:
            var group = Group(self._ctrl.unsafe_offset(pos))

            var match_mask = group.match_h2(h2_val)
            while match_mask != 0:
                var bit = count_trailing_zeros(Int(match_mask))
                var slot_idx = (pos + bit) & (self._capacity - 1)
                if (
                    self._slots.unsafe_offset(slot_idx)
                )[]._hash == hash and likely(
                    (self._slots.unsafe_offset(slot_idx))[].key == key
                ):
                    return (True, slot_idx)
                match_mask &= match_mask - 1

            # Compute empty mask once for both deleted-tracking and termination
            var empty_mask = group.match_empty()

            # Track first deleted slot we encounter for potential reuse
            if first_deleted == -1:
                var deleted_mask = group.match_empty_or_deleted()
                var pure_deleted = deleted_mask & ~empty_mask
                if pure_deleted != 0:
                    var bit = count_trailing_zeros(Int(pure_deleted))
                    first_deleted = (pos + bit) & (self._capacity - 1)

            if empty_mask != 0:
                if first_deleted != -1:
                    return (False, first_deleted)
                var bit = count_trailing_zeros(Int(empty_mask))
                return (False, (pos + bit) & (self._capacity - 1))

            pos = (pos + GROUP_WIDTH) & (self._capacity - 1)

    @always_inline
    def find_empty_slot(self, hash: UInt64) -> Int:
        """Find the first EMPTY or DELETED slot for the given hash.

        Used during resize and in-place rehash when the key is known to
        be unique.

        Args:
            hash: The hash to determine the starting probe position.

        Returns:
            The index of the first available slot.
        """
        var pos = Int(hash) & (self._capacity - 1)

        while True:
            var group = Group(self._ctrl.unsafe_offset(pos))
            var mask = group.match_empty_or_deleted()
            if mask != 0:
                var bit = count_trailing_zeros(Int(mask))
                return (pos + bit) & (self._capacity - 1)
            pos = (pos + GROUP_WIDTH) & (self._capacity - 1)

    def clear(
        mut self,
    ) where conforms_to(Self.K, Deinitable) and conforms_to(Self.V, Deinitable):
        """Remove all elements, destroying occupied entries.

        Constraints:
            Both `K` and `V` must be `Deinitable`, since every entry is
            destroyed in place.
        """
        if self._capacity == 0:
            return

        self._delete_occupied_entries()

        unsafe_memset(self._ctrl, CTRL_EMPTY, self._capacity + GROUP_WIDTH)
        self._len = 0
        self._growth_left = self._capacity * 7 // 8

    def clear_with(mut self, destroy_func: Some[def(var Self.K, var Self.V)]):
        """Remove all elements, disposing each entry with a caller closure.

        The closure counterpart of `clear`: it hands each entry's key and value
        to `destroy_func` instead of dropping them implicitly.

        Args:
            destroy_func: A closure that consumes each entry's key and value.
        """
        if self._capacity == 0:
            return

        self._delete_occupied_entries_with(destroy_func)

        unsafe_memset(self._ctrl, CTRL_EMPTY, self._capacity + GROUP_WIDTH)
        self._len = 0
        self._growth_left = self._capacity * 7 // 8

    def needs_resize(self) -> Bool:
        """Check whether the table needs a resize.

        Returns:
            True if growth_left has been exhausted.
        """
        return self._growth_left <= 0

    def is_sparse(self) -> Bool:
        """Check whether the table is sparse enough for in-place rehash.

        Returns:
            True if occupancy is <= 7/16 of capacity.
        """
        return self._len <= self._capacity * 7 // 16

    @no_inline
    def resize(mut self, new_capacity: Int) -> List[Tuple[Int, Int]]:
        """Double the table capacity and rehash all entries.

        Returns a list of (old_slot, new_slot) mappings so the caller can
        update any external ordering structures.

        Args:
            new_capacity: The new capacity (must be a power of 2).

        Returns:
            A list of (old_slot, new_slot) pairs for each relocated entry.
        """
        var old_ctrl = self._ctrl
        var old_slots = self._slots
        var old_capacity = self._capacity

        self._ctrl = alloc(
            Layout[UInt8](count=new_capacity + GROUP_WIDTH)
        ).unsafe_leak()
        unsafe_memset(self._ctrl, CTRL_EMPTY, new_capacity + GROUP_WIDTH)
        self._slots = alloc(
            Layout[SwissTableEntry[Self.K, Self.V, Self.H]](count=new_capacity)
        ).unsafe_leak()
        self._capacity = new_capacity
        self._growth_left = new_capacity * 7 // 8 - self._len

        var relocations = List[Tuple[Int, Int]](capacity=self._len)

        for i in range(old_capacity):
            if is_occupied(old_ctrl[unsafe_offset=i]):
                var entry = (old_slots.unsafe_offset(i)).unsafe_take_pointee()
                var h2_val = h2(entry._hash)
                var new_slot = self.find_empty_slot(entry._hash)
                self.set_ctrl(new_slot, h2_val)
                (self._slots.unsafe_offset(new_slot)).unsafe_write(entry^)
                relocations.append((i, new_slot))

        if old_capacity > 0:
            dealloc(
                ThinAllocation(unsafe_owned_ptr=old_ctrl).unsafe_with_layout(
                    {count = old_capacity + GROUP_WIDTH}
                )
            )
            dealloc(
                ThinAllocation(unsafe_owned_ptr=old_slots).unsafe_with_layout(
                    {count = old_capacity}
                )
            )

        return relocations^

    @no_inline
    def rehash_in_place(mut self) -> Pointer[Int32, MutUntrackedOrigin]:
        """Rehash in place without changing capacity (Abseil's drop-deletes).

        Reclaims DELETED tombstones by moving all entries to their ideal
        probe positions. Returns a slot_map array (caller-freed) mapping
        old slot indices to new ones.

        Returns:
            A pointer to a caller-owned Int32 array of size `_capacity`
            mapping old_slot -> new_slot. The caller must free this.
        """
        assert (
            self._len <= self._capacity * 7 // 16
        ), "in-place rehash called when table is too full"

        # Step 1: Rewrite ctrl bytes.
        for pos in range(0, self._capacity, GROUP_WIDTH):
            var group = Group(self._ctrl.unsafe_offset(pos))
            var converted = group.convert_special_to_empty_and_full_to_deleted()
            (self._ctrl.unsafe_offset(pos)).unsafe_store(converted)

        # Step 2: Refresh mirror bytes.
        unsafe_memcpy(
            dest=self._ctrl.unsafe_offset(self._capacity),
            src=self._ctrl,
            count=GROUP_WIDTH,
        )

        # Step 3: Relocate entries.
        var slot_map = alloc(Layout[Int32](count=self._capacity)).unsafe_leak()
        for i in range(self._capacity):
            slot_map[unsafe_offset=i] = Int32(i)

        for i in range(self._capacity):
            if self._ctrl[unsafe_offset=i] != CTRL_DELETED:
                continue

            var entry = (self._slots.unsafe_offset(i)).unsafe_take_pointee()
            self.set_ctrl(i, CTRL_EMPTY)

            var source = i
            var target = self.find_empty_slot(entry._hash)

            while self._ctrl[unsafe_offset=target] == CTRL_DELETED:
                self.set_ctrl(target, h2(entry._hash))
                var displaced = (
                    self._slots.unsafe_offset(target)
                ).unsafe_take_pointee()
                (self._slots.unsafe_offset(target)).unsafe_write(entry^)
                slot_map[unsafe_offset=source] = Int32(target)

                entry = displaced^
                source = target
                target = self.find_empty_slot(entry._hash)

            self.set_ctrl(target, h2(entry._hash))
            (self._slots.unsafe_offset(target)).unsafe_write(entry^)
            slot_map[unsafe_offset=source] = Int32(target)

        # Reset growth_left (all tombstones are now EMPTY).
        self._growth_left = self._capacity * 7 // 8 - self._len

        return slot_map
