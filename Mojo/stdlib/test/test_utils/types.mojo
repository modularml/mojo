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
"""Types for testing object lifecycle events.

* `MoveCounter`
* `CopyCounter`
* `MoveCopyCounter`
* `TriviallyCopyableMoveCounter`
* `DelCounter`
* `CopyCountedStruct`
* `MoveOnly`
* `ExplicitCopyOnly`
* `ImplicitCopyOnly`
* `ObservableMoveOnly`
* `ObservableDel`
* `DelRecorder`
* `AbortOnDel`
* `AbortOnCopy`
* `ExplicitDestroy`
"""

from std.hashlib import Hasher
from std.memory import MaybeUninit
from std.os import abort
from std.reflection import call_location
from std.utils._nicheable import UnsafeNicheable, NicheIndex

# ===----------------------------------------------------------------------=== #
# ExplicitDestroy
# ===----------------------------------------------------------------------=== #


@explicit_destroy("Use .destroy() to consume `ExplicitDestroy`")
@fieldwise_init
struct ExplicitDestroy(Deinitable where False, Movable):
    """Test type that is explicitly-destroyed."""

    var value: Int
    """Int data."""

    def destroy(deinit self):
        """Destroys self."""
        pass


# ===----------------------------------------------------------------------=== #
# ExplicitDestroyKey
# ===----------------------------------------------------------------------=== #


@explicit_destroy("Use .destroy() to consume `ExplicitDestroyKey`")
@fieldwise_init
struct ExplicitDestroyKey(Deinitable where False, Equatable, Hashable, Movable):
    """Linear key type for testing linear-keyed containers: `Movable`,
    `Hashable`, and `Equatable`, but explicitly-destroyed (not
    `Deinitable`)."""

    var value: Int
    """Int data."""

    def __eq__(self, other: Self) -> Bool:
        """Compare two keys on their payload.

        Args:
            other: The other key to compare against.

        Returns:
            `True` if the payloads are equal, `False` otherwise.
        """
        return self.value == other.value

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hash the payload using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        self.value.__hash__(hasher)

    def destroy(deinit self):
        """Destroys self."""
        pass


# ===----------------------------------------------------------------------=== #
# CopyableExplicitDestroyKey
# ===----------------------------------------------------------------------=== #


@explicit_destroy("Use .destroy() to consume `CopyableExplicitDestroyKey`")
@fieldwise_init
struct CopyableExplicitDestroyKey(
    Copyable, Deinitable where False, Equatable, Hashable
):
    """Linear key type that is nonetheless `Copyable`: `Copyable`, `Hashable`,
    and `Equatable`, but explicitly-destroyed (not `Deinitable`).

    Separates bounds that genuinely need `Deinitable` from those that
    only need `Copyable`, which `ExplicitDestroyKey` cannot distinguish because
    it is move-only."""

    var value: Int
    """Int data."""

    def __eq__(self, other: Self) -> Bool:
        """Compare two keys on their payload.

        Args:
            other: The other key to compare against.

        Returns:
            `True` if the payloads are equal, `False` otherwise.
        """
        return self.value == other.value

    def __hash__[H: Hasher](self, mut hasher: H):
        """Hash the payload using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        self.value.__hash__(hasher)

    def destroy(deinit self):
        """Destroys self."""
        pass


# ===----------------------------------------------------------------------=== #
# MoveOnly
# ===----------------------------------------------------------------------=== #


struct MoveOnly[T: Movable & Deinitable](
    Equatable where conforms_to(T, Equatable),
    Hashable where conforms_to(T, Hashable),
    Movable,
    Writable where conforms_to(T, Writable),
):
    """Utility for testing MoveOnly types.

    Parameters:
        T: Can be any type satisfying the Movable trait.
    """

    var data: Self.T
    """Test data payload."""

    @implicit
    def __init__(out self, var i: Self.T):
        """Construct a MoveOnly providing the payload data.

        Args:
            i: The test data payload.
        """
        self.data = i^

    def __eq__(self, other: Self) -> Bool where conforms_to(Self.T, Equatable):
        """Compare two `MoveOnly` instances for equality on their payload.

        Args:
            other: The other instance to compare against.

        Returns:
            `True` if the payloads are equal, `False` otherwise.
        """
        return self.data == other.data

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.T, Hashable):
        """Hash the payload using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        self.data.__hash__(hasher)

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Write the payload to a `Writer`.

        Args:
            writer: The writer to write to.
        """
        writer.write(self.data)


# ===----------------------------------------------------------------------=== #
# ObservableMoveOnly
# ===----------------------------------------------------------------------=== #


struct ObservableMoveOnly[actions_origin: ImmOrigin](Movable):
    """Type for observing move and destruction operations during testing.

    Parameters:
        actions_origin: Origin of the actions list for tracking operations.
    """

    comptime _U = Pointer[List[String], Self.actions_origin]
    var actions: Self._U
    """Pointer to list tracking lifecycle operations."""
    var value: Int
    """Test value payload."""

    def __init__(out self, value: Int, actions: Self._U):
        """Constructs a new instance and records the initialization.

        Args:
            value: The value to store.
            actions: Pointer to list for recording operations.
        """
        self.actions = actions
        self.value = value
        self.actions.unsafe_mut_cast[True]()[].append("__init__")

    def __init__(out self, *, deinit move: Self):
        """Moves from an existing instance and records the operation.

        Args:
            move: The instance being moved from.
        """
        self.actions = move.actions
        self.value = move.value
        self.actions.unsafe_mut_cast[True]()[].append("move ctor")

    def __deinit__(deinit self):
        """Destroys the instance and records the operation."""
        self.actions.unsafe_mut_cast[True]()[].append("__deinit__")


# ===----------------------------------------------------------------------=== #
# ExplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ExplicitCopyOnly(Copyable):
    """Type with explicit copy semantics for testing."""

    var value: Int
    """Integer value payload."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    @implicit
    def __init__(out self, value: Int):
        """Constructs a new instance.

        Args:
            value: The integer value to store.
        """
        self.value = value
        self.copy_count = 0

    def __init__(out self, *, copy: Self):
        """Copies from another instance and increments the count.

        Args:
            copy: The instance being copied from.
        """
        self = Self(copy.value)
        self.copy_count = copy.copy_count + 1


# ===----------------------------------------------------------------------=== #
# ImplicitCopyOnly
# ===----------------------------------------------------------------------=== #


struct ImplicitCopyOnly(ImplicitlyCopyable):
    """Type with implicit copy semantics for testing."""

    var value: Int
    """Integer value payload."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    @implicit
    def __init__(out self, value: Int):
        """Constructs a new instance.

        Args:
            value: The integer value to store.
        """
        self.value = value
        self.copy_count = 0

    def __init__(out self, *, copy: Self):
        """Copies from another instance and increments the count.

        Args:
            copy: The instance being copied from.
        """
        self.value = copy.value
        self.copy_count = copy.copy_count + 1


# ===----------------------------------------------------------------------=== #
# CopyCounter
# ===----------------------------------------------------------------------=== #


struct CopyCounter[
    T: ImplicitlyCopyable & Deinitable & Writable & Defaultable = NoneType,
    *,
    trivial_copy: Bool = False,
](Deinitable, ImplicitlyCopyable, Writable):
    """Counts the number of copies performed on a value.

    Parameters:
        T: The type of value to wrap and count copies for.
        trivial_copy: Weather the copy constructor should be considered trivial.
    """

    comptime __copy_ctor_is_trivial = Self.trivial_copy

    var value: Self.T
    """The wrapped value."""
    var copy_count: Int
    """Number of times this instance has been copied."""

    def __init__(out self):
        """Constructs a new instance with default value."""
        self = Self(Self.T())

    def __init__(out self, s: Self.T):
        """Constructs a new instance with the given value.

        Args:
            s: The value to wrap.
        """
        comptime assert (
            Self.T.__copy_ctor_is_trivial or not Self.trivial_copy
        ), (
            "You cannot override CopyCounter's trivial copy construct when T"
            " does not have one"
        )
        self.value = s
        self.copy_count = 0

    def __init__(out self, *, copy: Self):
        """Copies from another instance and increments the count.

        Args:
            copy: The instance being copied from.
        """
        self.value = copy.value
        self.copy_count = copy.copy_count + 1

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation to the writer.

        Args:
            writer: The writer to output to.
        """
        writer.write("CopyCounter(", self.value, " ", self.copy_count, ")")


# ===----------------------------------------------------------------------=== #
# MoveCounter
# ===----------------------------------------------------------------------=== #


# TODO: This type should not be Copyable, but has to be to satisfy
#       Copyable at the moment.
struct MoveCounter[T: Copyable & Deinitable, *, trivial_move: Bool = False](
    Copyable
):
    """Counts the number of moves performed on a value.

    Parameters:
        T: The type of value to wrap and count moves for.
        trivial_move: Weather the move constructor should be treated as trivial.
    """

    comptime __move_ctor_is_trivial = Self.trivial_move

    var value: Self.T
    """The wrapped value."""
    var move_count: Int
    """Number of times this instance has been moved."""

    @implicit
    def __init__(out self, var value: Self.T):
        """Constructs a new instance of this type. This initial move is not counted.

        Args:
            value: The value to wrap.
        """
        comptime assert (
            Self.T.__move_ctor_is_trivial or not Self.trivial_move
        ), (
            "You cannot override MoveCounter's trivial move construct when T"
            " does not have one"
        )
        self.value = value^
        self.move_count = 0

    def __init__(out self, *, deinit move: Self):
        """Moves from an existing instance and increments the count.

        Args:
            move: The instance being moved from.
        """
        self.value = move.value^
        self.move_count = move.move_count + 1


# ===----------------------------------------------------------------------=== #
# MoveCopyCounter
# ===----------------------------------------------------------------------=== #


struct MoveCopyCounter(ImplicitlyCopyable):
    """Counts both copy and move operations for testing."""

    var copied: Int
    """Number of times this instance has been copied."""
    var moved: Int
    """Number of times this instance has been moved."""

    def __init__(out self):
        """Constructs a new instance with zero counts."""
        self.copied = 0
        self.moved = 0

    def __init__(out self, *, copy: Self):
        """Copies from another instance and increments the copy count.

        Args:
            copy: The instance being copied from.
        """
        self.copied = copy.copied + 1
        self.moved = copy.moved

    def __init__(out self, *, deinit move: Self):
        """Moves from an existing instance and increments the move count.

        Args:
            move: The instance being moved from.
        """
        self.copied = move.copied
        self.moved = move.moved + 1


# ===----------------------------------------------------------------------=== #
# TriviallyCopyableMoveCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct TriviallyCopyableMoveCounter(Copyable):
    """Type used for testing that collections still perform moves and not copies
    when a type has a custom move constructor but is also trivially copyable.

    Types with this property are rare in practice, but its still important to
    get the modeling right. If a type author wants their type to be moved
    with a memcpy, they should mark it as such."""

    var move_count: Int
    """Number of times this instance has been moved."""

    # Copying this type is trivial, it doesn't care to track copies.
    comptime __copy_ctor_is_trivial = True

    def __init__(out self, *, deinit move: Self):
        """Moves from an existing instance and increments the count.

        Args:
            move: The instance being moved from.
        """
        self.move_count = move.move_count + 1


# ===----------------------------------------------------------------------=== #
# DelRecorder
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelRecorder[recorder_origin: ImmOrigin](ImplicitlyCopyable):
    """Records destructor calls for testing.

    Parameters:
        recorder_origin: Origin of the recorder list.
    """

    var value: Int
    """Value to record when destroyed."""
    var destructor_recorder: Pointer[List[Int], Self.recorder_origin]
    """Pointer to list for recording destructor calls."""

    def __deinit__(deinit self):
        """Records this instance's value when destroyed."""
        self.destructor_recorder.unsafe_mut_cast[True]()[].append(self.value)


# ===----------------------------------------------------------------------=== #
# ObservableDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ObservableDel[origin: MutOrigin = MutAnyOrigin](ImplicitlyCopyable):
    """Sets a boolean flag when destroyed.

    Parameters:
        origin: Origin of the target pointer.
    """

    var target: Pointer[Bool, Self.origin]
    """Pointer to boolean flag set on destruction."""

    def __deinit__(deinit self):
        """Sets the target flag to True when destroyed."""
        self.target.unsafe_write(True)


# ===----------------------------------------------------------------------=== #
# ExplicitDelOnly
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ExplicitDelOnly(Deinitable where False, Movable):
    """Utility for testing container support for linear types."""

    var data: Int
    """Test data payload."""

    def destroy(deinit self):
        """Explicitly destroy this linear type."""

        pass


# ===----------------------------------------------------------------------=== #
# PinnedExplicitDelOnly
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct PinnedExplicitDelOnly(Deinitable where False, Movable where False):
    """Utility for testing container support for linear types that also
    cannot be moved."""

    var data: Int
    """Test data payload."""

    def destroy(deinit self):
        """Explicitly destroy this linear type."""

        pass


# ===----------------------------------------------------------------------=== #
# PinnedExplicitDelOnly
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Pinned(Movable where False):
    """Utility testing container which is non-movable."""

    var data: Int
    """Test data payload."""

    def __init__(out self):
        """Default initialize a `Pinned`."""
        self.data = 0


# ===----------------------------------------------------------------------=== #
# DelCounter
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct DelCounter[counter_origin: ImmOrigin, *, trivial_del: Bool = False](
    ImplicitlyCopyable, Writable
):
    """Counts the number of times instances are destroyed.

    Parameters:
        counter_origin: Origin of the counter pointer.
        trivial_del: Whether the destructor is trivial.
    """

    comptime __del__is_trivial = Self.trivial_del

    var counter: Pointer[Int, Self.counter_origin]
    """Pointer to counter incremented on destruction."""

    def __deinit__(deinit self):
        """Increments the counter when destroyed."""
        self.counter.unsafe_mut_cast[True]()[] += 1

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation to the writer.

        Args:
            writer: The writer to output to.
        """
        writer.write("DelCounter(", self.counter[], ")")


# ===----------------------------------------------------------------------=== #
# AbortOnDel
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnDel(Copyable):
    """Type that aborts if its destructor is called.

    Used to test that destructors are not called in certain scenarios.
    """

    var value: Int
    """Test value payload."""

    @always_inline
    def __deinit__(deinit self):
        """Aborts the program if called."""
        abort(
            "We should never call the destructor of AbortOnDel",
            location=call_location(),
        )


# ===----------------------------------------------------------------------=== #
# CopyCountedStruct
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct CopyCountedStruct(ImplicitlyCopyable):
    """Struct that tracks the number of times it has been copied."""

    var counter: CopyCounter[]
    """Counter tracking copy operations."""
    var value: String
    """String value payload."""

    @implicit
    def __init__(out self, value: String):
        """Constructs a new instance with the given value.

        Args:
            value: The string value to store.
        """
        self.counter = CopyCounter()
        self.value = value


# ===----------------------------------------------------------------------=== #
# AbortOnCopy
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct AbortOnCopy(ImplicitlyCopyable):
    """Type that aborts if copied.

    Used to test that implicit copies do not occur in certain scenarios.
    """

    def __init__(out self, *, copy: Self):
        """Aborts the program if called.

        Args:
            copy: The instance being copied from.
        """
        abort("We should never implicitly copy AbortOnCopy")


# ===----------------------------------------------------------------------=== #
# Observable
# ===----------------------------------------------------------------------=== #


struct Observable[
    *,
    CopyOrigin: MutOrigin = MutUntrackedOrigin,
    MoveOrigin: MutOrigin = MutUntrackedOrigin,
    DelOrigin: MutOrigin = MutUntrackedOrigin,
    opt_into_unsafe_niche: Bool = False,
](Copyable, UnsafeNicheable where opt_into_unsafe_niche):
    """A type that tracks the number of times it has been copied, moved, and destroyed.

    Parameters:
        CopyOrigin: Origin of the copies pointer.
        MoveOrigin: Origin of the moves pointer.
        DelOrigin: Origin of the dels pointer.
        opt_into_unsafe_niche: A flag indicating if this type should implement `UnsafeNicheable`.
    """

    var _always_zero: Byte
    var _copies: Optional[Pointer[Int, Self.CopyOrigin]]
    var _moves: Optional[Pointer[Int, Self.MoveOrigin]]
    var _dels: Optional[Pointer[Int, Self.DelOrigin]]

    def __init__(
        out self,
        *,
        var copies: Optional[Pointer[Int, Self.CopyOrigin]] = None,
        var moves: Optional[Pointer[Int, Self.MoveOrigin]] = None,
        var dels: Optional[Pointer[Int, Self.DelOrigin]] = None,
    ):
        """Constructs a new Observable with the given pointers.

        Args:
            copies: Optional pointer to an Int to count copies.
            moves: Optional pointer to an Int to count moves.
            dels: Optional pointer to an Int to count dels.
        """
        self._always_zero = 0
        self._copies = copies^
        self._moves = moves^
        self._dels = dels^

    def __init__(out self, *, copy: Self):
        """Copy initialize the Observable and increment the copy count.

        Args:
            copy: The instance being copied from.
        """
        self._always_zero = 0
        self._copies = copy._copies.copy()
        self._moves = copy._moves.copy()
        self._dels = copy._dels.copy()
        if self._copies:
            self._copies.value()[] += 1

    def __init__(out self, *, deinit move: Self):
        """Move initialize the Observable and increment the move count.

        Args:
            move: The instance being moved from.
        """
        self._always_zero = 0
        self._copies = move._copies^
        self._moves = move._moves^
        self._dels = move._dels^
        if self._moves:
            self._moves.value()[] += 1

    def __deinit__(deinit self):
        """Destroy the Observable and increment the deinit count."""
        if self._dels:
            self._dels.value()[] += 1

    @staticmethod
    def niche_count() -> Int where Self.opt_into_unsafe_niche:
        """Returns the number of niche values available.

        Uses `_always_zero` (which is always `0` in a live instance) to store
        niche indices as non-zero bytes, giving `Byte.MAX - 1` possible niches.

        Returns:
            `Byte.MAX - 1`.
        """
        return Int(Byte.MAX - 1)

    @staticmethod
    def write_niche[
        index: Int
    ](
        memory: MutPointer[MaybeUninit[Self], _]
    ) where Self.opt_into_unsafe_niche:
        """Writes niche bit pattern `index` into storage.

        Parameters:
            index: Which niche to write.

        Args:
            memory: Pointer to uninitialized storage sized and aligned for `Self`.
        """
        comptime niche_offset = reflect[Self].field_offset[
            name="_always_zero"
        ]()
        memory.unsafe_bitcast[Byte]().unsafe_offset(niche_offset).unsafe_store(
            Byte(index + 1)
        )

    @staticmethod
    def classify_niche(
        memory: ImmPointer[MaybeUninit[Self], _]
    ) -> NicheIndex where Self.opt_into_unsafe_niche:
        """Classifies the bit pattern at `memory` as a niche or a live value.

        Args:
            memory: Pointer to initialized storage containing either a live
                `Self` value or a previously written niche.

        Returns:
            The niche index of the memory.
        """
        comptime niche_offset = reflect[Self].field_offset[
            name="_always_zero"
        ]()
        var value = (
            memory.unsafe_bitcast[Byte]()
            .unsafe_offset(niche_offset)
            .unsafe_load()
        )
        if value == 0:
            return NicheIndex.NotANiche
        else:
            return NicheIndex(Int(value - 1))


# ===----------------------------------------------------------------------=== #
# ConfigureTrivial
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct ConfigureTrivial[
    *,
    del_is_trivial: Bool = False,
    copyinit_is_trivial: Bool = False,
    moveinit_is_trivial: Bool = False,
](Copyable):
    """Testing type configurable conditional triviality.

    Parameters:
        del_is_trivial: Whether the destructor is trivial.
        copyinit_is_trivial: Whether the copy initializer is trivial.
        moveinit_is_trivial: Whether the move initializer is trivial.
    """

    comptime __del__is_trivial = Self.del_is_trivial
    comptime __copy_ctor_is_trivial = Self.copyinit_is_trivial
    comptime __move_ctor_is_trivial = Self.moveinit_is_trivial


# ===----------------------------------------------------------------------=== #
# NonMovable
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct NonMovable(Movable where False):
    """A non-movable type."""

    var value: Int
    """Test value payload."""
