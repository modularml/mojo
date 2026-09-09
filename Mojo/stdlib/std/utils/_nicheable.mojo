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

from std.memory import MaybeUninit


struct NicheIndex(Equatable, TrivialRegisterPassable):
    """The result of `UnsafeNicheable.classify_niche`.

    A `NicheIndex` either identifies which niche bit pattern was found in a
    memory location, or indicates that the memory holds a valid (non-niche)
    instance via the sentinel `NicheIndex.NotANiche`.

    Niche indices are zero-based: `NicheIndex(0)` refers to the first niche,
    `NicheIndex(1)` to the second, and so on up to
    `NicheIndex(Self.niche_count() - 1)`.
    """

    var _index: Int

    @always_inline
    def __init__(out self, index: Int):
        """Construct a `NicheIndex` with the given index value. Must be in the
        range (`0` <= index <= `Int.MAX`).

        Args:
            index: The niche index value.
        """
        self._index = index

    @always_inline
    def __eq__(self, other: Self) -> Bool:
        """Returns whether two `NicheIndex` values are equal.

        Args:
            other: The `NicheIndex` to compare against.

        Returns:
            `True` if both indices have the same value, `False` otherwise.
        """
        return self._index == other._index

    comptime NotANiche: Self = Self(-1)
    """Sentinel returned by `classify_niche` when the memory holds a valid
    `Self` instance rather than a niche bit pattern.
    """


comptime NicheStorageTraits = TrivialRegisterPassable
"""Trait requirements for a custom niche backing storage type."""


trait UnsafeCustomNicheStorage:
    """Allows a nicheable type to customize its backing storage representation.

    By default, niche-optimized containers (like `Optional`) store a nicheable
    type's value in a `MaybeUninit[T]`, which lowers to `pop.array<1,T>`.
    This is not always desirable. For example, `Pointer` needs to
    lower directly to `kgen.pointer` to preserve its ABI across exported
    function boundaries.

    A type conforming to this trait provides an associated `NicheStorage` type that
    the container uses instead of `MaybeUninit[T]`. The `NicheStorage` type
    must have the same size and alignment as the nicheable type, and must conform
    to `NicheStorageTraits`.
    """

    comptime NicheStorage: NicheStorageTraits
    """The backing storage type to use instead of `MaybeUninit[Self]`.

    Safety:
        This must have the same alignment and size as `Self`.
    """


trait UnsafeNicheable:
    """A type that exposes known-invalid bit patterns to enable niche optimizations.

    Some types have bit patterns that can never represent a valid instance.
    For example, a non-zero integer can never be zero, and a non-null pointer
    can never have address zero. These "niches" can be exploited to store
    extra information (such as the `None` discriminant of an `Optional`)
    without requiring a separate tag field. The result is that
    `size_of[Optional[T]]() == size_of[T]()` for any `T` that conforms
    to this trait.

    Implementing this trait is **unsafe**. The compiler trusts that
    `write_niche` and `classify_niche` are consistent
    with each other and correctly identify every invalid bit pattern claimed
    by `niche_count`. Violating these contracts can cause
    undefined behavior (e.g. running a destructor on a niche value that is
    treated as a live `Self` instance).

    Example:

    A `NonMaxUInt` that uses `UInt.MAX` as its single niche:

    ```mojo
    from std.utils._nicheable import UnsafeNicheable, NicheIndex
    from std.memory import MaybeUninit

    struct NonMaxUInt(UnsafeNicheable):
        var _n: UInt

        def __init__(out self, var n: UInt) raises:
            if n == UInt.MAX:
                raise Error("NonMaxUInt cannot be constructed from a UInt.MAX!")
            else:
                self._n = n

        @staticmethod
        def niche_count() -> Int:
            # This type only has a single niche value.
            return 1

        @staticmethod
        def write_niche[index: Int](
            memory: MutPointer[MaybeUninit[Self], _]
        ):
            # Write UInt.MAX into the storage, not a valid NonMaxUInt.
            memory.unsafe_bitcast[UInt]().unsafe_store(UInt.MAX)

        @staticmethod
        def classify_niche(
            memory: ImmPointer[MaybeUninit[Self], _]
        ) -> NicheIndex:
            # UInt.MAX is the niche (index 0), anything else is a valid value.
            if memory.unsafe_bitcast[UInt]()[] == UInt.MAX:
                return NicheIndex(0)
            return NicheIndex.NotANiche
    ```
    """

    @staticmethod
    def niche_count() -> Int:
        """Returns the number of invalid bit patterns available as niches.

        Returns:
            The total count of distinct niche values for this type.
        """
        ...

    @staticmethod
    def write_niche[index: Int](memory: MutPointer[MaybeUninit[Self], _]):
        """Writes niche bit pattern `index` into the pointed-to storage.

        On entry, `memory` points to properly aligned, correctly sized storage
        for `Self` that must be treated as **uninitialized**: it does not
        contain a live `Self` value. On return, the bytes at `memory` encode
        niche `index` and are suitable for passing to `classify_niche`.
        The pointer type is `MaybeUninit[Self]` to make it clear that
        the storage does not yet hold a valid `Self` instance.

        Parameters:
            index: Which niche value to write. Guaranteed to be in the range
                `0 <= i < Self.niche_count()`.

        Args:
            memory: A pointer to uninitialized storage, aligned and sized for
                `Self`.
        """
        ...

    @staticmethod
    def classify_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> NicheIndex:
        """Classifies the bit pattern at `memory` as either a niche or a real value.

        The storage is initialized: it either holds a genuine `Self` value or
        bytes previously written by `write_niche`. The pointer
        type is `MaybeUninit[Self]` rather than `Self` to make it clear
        that dereferencing the memory as a live `Self` is not safe — the bytes
        may encode a niche rather than a valid instance.

        Args:
            memory: A pointer to properly aligned, initialized storage that
                contains either a niche bit pattern or a valid `Self` value.

        Returns:
            `NicheIndex.NotANiche` if the memory holds a valid `Self`
            instance, or a `NicheIndex` with value `i`
            (where `0 <= i < Self.niche_count()`) if the memory holds
            niche bit pattern `i`.
        """
        ...


trait UnsafeSingleNicheable(UnsafeNicheable):
    """A simplified form of `UnsafeNicheable` for types with exactly one niche.

    It is common for a nicheable type to have only a single invalid bit
    pattern (e.g. a non-null pointer whose sole niche is address zero).
    This trait reduces the boilerplate of conforming to `UnsafeNicheable` by replacing the
    index-parameterized `write_niche[index]` and `classify_niche` methods
    with simpler alternatives that don't deal with indices at all:

    - `write_niche` — writes the single niche bit pattern (no index needed).
    - `isa_niche` — returns `True` if the memory holds the niche, `False`
      if it holds a valid value.

    The trait automatically provides default implementations of the `UnsafeNicheable`
    methods that delegate to these simpler methods.

    Implementing this trait is **unsafe** for the same reasons as
    `UnsafeNicheable`: `write_niche` and `isa_niche` must be consistent with
    each other. A round-trip of writing the niche and then classifying it
    must return `True`, and classifying a valid `Self` instance must return
    `False`.

    Example:

    A `NonMaxUInt` that uses `UInt.MAX` as its single niche:

    ```mojo
    from std.utils._nicheable import UnsafeSingleNicheable
    from std.memory import MaybeUninit

    struct NonMaxUInt(UnsafeSingleNicheable):
        var _n: UInt

        def __init__(out self, var n: UInt) raises:
            if n == UInt.MAX:
                raise Error("NonMaxUInt cannot be constructed from UInt.MAX!")
            else:
                self._n = n

        @staticmethod
        def write_niche(
            memory: MutPointer[MaybeUninit[Self], _]
        ):
            memory.unsafe_bitcast[UInt]().unsafe_store(UInt.MAX)

        @staticmethod
        def isa_niche(
            memory: ImmPointer[MaybeUninit[Self], _]
        ) -> Bool:
            return memory.unsafe_bitcast[UInt]()[] == UInt.MAX
    ```
    """

    @staticmethod
    def write_niche(memory: MutPointer[MaybeUninit[Self], _]):
        """Writes the single niche bit pattern into the pointed-to storage.

        On entry, `memory` points to properly aligned, correctly sized storage
        for `Self` that must be treated as **uninitialized**: it does not
        contain a live `Self` value. On return, the bytes at `memory` encode
        the niche and `isa_niche` will return `True` for this memory.

        Args:
            memory: A pointer to uninitialized storage, aligned and sized for
                `Self`.
        """
        ...

    @staticmethod
    def isa_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> Bool:
        """Returns whether the bit pattern at `memory` is the niche value.

        Args:
            memory: A pointer to properly aligned, initialized storage that
                contains either the niche bit pattern or a valid `Self` value.

        Returns:
            `True` if the memory holds the niche bit pattern, `False` if it
            holds a valid `Self` instance.
        """
        ...

    @staticmethod
    @always_inline
    @doc_hidden
    def niche_count() -> Int:
        """Returns `1`, since this type has exactly one niche.

        Returns:
            Always returns `1`.
        """
        return 1

    @staticmethod
    @always_inline
    @doc_hidden
    def write_niche[index: Int](memory: MutPointer[MaybeUninit[Self], _]):
        """Implements `UnsafeNicheable.write_niche` by delegating to the
        index-free `write_niche` overload.

        Parameters:
            index: Ignored — there is only one niche (index 0).

        Args:
            memory: A pointer to uninitialized storage, aligned and sized for
                `Self`.
        """
        Self.write_niche(memory)

    @staticmethod
    @always_inline
    @doc_hidden
    def classify_niche(memory: ImmPointer[MaybeUninit[Self], _]) -> NicheIndex:
        """Implements `UnsafeNicheable.classify_niche` by delegating to
        `isa_niche`.

        Args:
            memory: A pointer to properly aligned, initialized storage that
                contains either the niche bit pattern or a valid `Self` value.

        Returns:
            `NicheIndex(0)` if `isa_niche` returns `True`,
            `NicheIndex.NotANiche` otherwise.
        """
        return NicheIndex(0) if Self.isa_niche(memory) else NicheIndex.NotANiche
