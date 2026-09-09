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
"""Provides a fixed-size array implementation with compile-time size checking.

The `Array` type represents a fixed-size sequence of homogeneous elements
where the size is determined at compile time. It provides efficient memory
layout and bounds checking while maintaining type safety.  The `Array`
type is part of the `prelude` module and therefore does not need to be imported
in order to use it.

Examples:

```mojo
# Create an array of 3 integers
var arr = [1, 2, 3]

# Access elements
print(arr[0])  # Prints 1

# Fill with a value
var filled = Array[Int, 5](fill=42)
```
"""


import std.math
import std.memory
from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.builtin.rebind import downcast
from std.collections import check_bounds
import std.format._utils as fmt
from std.reflection import reflect
from std.hashlib.hasher import Hasher
from std.sys import is_gpu, size_of
from std.memory import (
    MaybeUninit,
    forget_deinit,
    unsafe_destroy_n,
    unsafe_uninit_move_n,
    unsafe_uninit_copy_n,
)
from std.traits import (
    IsTriviallyCopyable,
    IsTriviallyDeinitable,
    IsTriviallyMovable,
)

# ===-----------------------------------------------------------------------===#
# Array
# ===-----------------------------------------------------------------------===#


def _array_construction_checks[length: Int]():
    """Checks if the properties in `Array` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
        length: The number of elements.
    """
    comptime assert length >= 0, "number of elements in `Array` must be >= 0"


@fieldwise_init
struct _ArrayIter[
    mut: Bool,
    //,
    T: Copyable,
    length: Int,
    origin: Origin[mut=mut],
    forward: Bool = True,
](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for `Array`.

    Parameters:
        mut: A boolean to indicate if the iterator is mutable.
        T: The type of the elements in the iterator.
        length: The number of elements in the array.
        origin: The origin of the iterator.
        forward: A boolean to indicate if the iterator is forward.
    """

    comptime Element = Self.T

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self

    var index: Int
    var src: Pointer[Array[Self.T, Self.length], Self.origin]

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self.copy()

    def __next__(
        mut self,
    ) raises StopIteration -> ref[Self.origin] Self.Element:
        comptime if Self.forward:
            if self.index >= Self.length:
                raise StopIteration()
            self.index += 1
            return self.src[][self.index - 1]
        else:
            if self.index <= 0:
                raise StopIteration()
            self.index -= 1
            return self.src[][self.index]

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var iter_len: Int

        comptime if Self.forward:
            iter_len = Self.length - self.index
        else:
            iter_len = self.index

        return (iter_len, {iter_len})


struct _ArrayIterOwned[T: Movable & Deinitable, length: Int](
    IterableOwned, Iterator, Movable
):
    """An owning iterator for Array.

    Parameters:
        T: The type of the elements in the array.
        length: The number of elements in the array.
    """

    comptime Element = Self.T
    comptime IteratorOwnedType = Self

    var _array: Array[Self.T, Self.length]
    var _index: Int

    def __init__(out self, var array: Array[Self.T, Self.length]):
        """Consume an array and create an iterator over its elements.

        Args:
            array: The array to consume.
        """
        self._array = array^
        self._index = 0

    def __init__(out self, *, deinit move: Self):
        """Move constructor that handles partially consumed array storage.

        After partial iteration some array slots are uninitialized, so
        the default fieldwise move would be unsound.  This constructor
        moves only the unconsumed elements and marks the source
        destroyed.

        Args:
            move: The iterator to move from.
        """
        self._index = move._index
        self._array = Array[Self.T, Self.length](uninitialized=True)
        unsafe_uninit_move_n[overlapping=False](
            dest=self._array.unsafe_ptr().unsafe_offset(move._index),
            src=move._array.unsafe_ptr().unsafe_offset(move._index),
            count=Self.length - move._index,
        )

    @always_inline
    def __deinit__(deinit self):
        # Move fields out of self so we can manage their lifetimes.
        var idx = self._index
        var array = self._array^

        # Destroy the remaining elements that have not yet been
        # iterated over.
        unsafe_destroy_n(
            array.unsafe_ptr().unsafe_offset(idx), Self.length - idx
        )

        # Mark the array as destroyed so Array.__deinit__ doesn't
        # double-destroy the elements we already handled.
        forget_deinit(array^)

    @always_inline
    def __iter__(var self) -> Self.IteratorOwnedType:
        return self^

    def __next__(mut self) raises StopIteration -> Self.Element:
        if self._index >= Self.length:
            raise StopIteration()
        self._index += 1
        return (
            self._array.unsafe_ptr()
            .unsafe_offset(self._index - 1)
            .unsafe_take_pointee()
        )

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var remaining = Self.length - self._index
        return (remaining, {remaining})


@explicit_destroy(
    "Use `deinit_with()` to explicitly destroy an `Array` of"
    " non-`Deinitable` elements"
)
@stable(since="1.0")
struct Array[T: AnyType, length: Int](
    Comparable where conforms_to(T, Comparable),
    Copyable where conforms_to(T, Copyable),
    Defaultable where conforms_to(T, Defaultable),
    Deinitable where conforms_to(T, Deinitable),
    DevicePassable where conforms_to(T, DevicePassable) and conforms_to(
        T, Copyable
    ),
    Equatable where conforms_to(T, Equatable),
    Hashable where conforms_to(T, Hashable),
    Iterable,
    # TODO(MOCO-4308): Remove redundant 'Movable' constraint
    IterableOwned where conforms_to(T, Movable & Deinitable),
    Movable where conforms_to(T, Movable),
    Sized,
    Writable where conforms_to(T, Writable),
):
    """A fixed-size sequence of homogeneous elements where size is a constant
    expression.

    Array provides a fixed-size array implementation with compile-time
    size checking. The array size is determined at compile time and cannot be
    changed.

    Parameters:
        T: The type of the elements in the array. May be any type (`AnyType`).
            Move construction (including list-literal construction) requires
            `Movable`; copy and `fill=` construction and iteration additionally
            require `Copyable`; these are enforced via conditional `where`
            clauses.
        length: The number of elements in the array. Must be a positive integer
            constant.

    Examples:

    ```mojo
    # Create array of 3 integers
    var arr = [1, 2, 3]

    # Create array filled with value
    var filled = Array[Int, 5](fill=42)

    # Create array of defaulted elements
    var defaulted = Array[Int, 5]()  # [0, 0, 0, 0, 0]

    # Access elements
    print(arr[0])  # Prints 1
    ```

    Notes:
        When the element type is `Comparable`, `Array` is too and orders
        **lexicographically**: elements compare pairwise and the first differing
        pair decides the result.
    """

    comptime __del__is_trivial: Bool = IsTriviallyDeinitable[Self.T]
    comptime __copy_ctor_is_trivial: Bool = IsTriviallyCopyable[Self.T]
    comptime __move_ctor_is_trivial: Bool = IsTriviallyMovable[Self.T]

    # Fields
    comptime type = __mlir_type[
        `!pop.array<`, Self.length.__mlir_index__(), `, `, Self.T, `>`
    ]
    """The underlying MLIR array type."""

    var _array: Self.type
    """The underlying storage for the array."""

    comptime _DeviceElementType: AnyType = downcast[
        Self.T.device_type, Movable
    ] if conforms_to(Self.T, DevicePassable) else Self.T
    """The device-side element type: the element's `device_type` when it is
    `DevicePassable`, otherwise the element type itself."""

    comptime device_type: AnyType = Array[Self._DeviceElementType, Self.length]
    """The device-side type for this array.

    Parametric over the elements' device types, so an array of a `DevicePassable`
    element type encodes to the array of converted elements (and collapses to
    `Self` for identity elements)."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _ArrayIter[
        downcast[Self.T, Copyable],
        Self.length,
        iterable_origin,
        True,
    ]
    """The iterator type for this array.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # TODO(MOCO-4308): Remove redundant 'Movable' constraint
    comptime IteratorOwnedType: Iterator where conforms_to(
        Self.T, Movable & Deinitable
    ) = _ArrayIterOwned[Self.T, Self.length]
    """The owned iterator type for this array."""

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ) where conforms_to(Self.T, DevicePassable) and conforms_to(
        Self.T, Copyable
    ):
        """Convert the host type object to a device_type and store it at the
        target address.

        Args:
            encoder: Target specific device type encoder.
            target: The target address to store the device type.
        """
        # Encode element-wise so a `DevicePassable` element runs its own
        # `_to_device_type` conversion rather than being byte-copied wholesale.
        encoder.encode_array(self, target)

    @staticmethod
    def get_type_name() -> String:
        """Gets the name of the host type (the one implementing this trait).

        Returns:
            The host type's name.
        """
        return String(
            "Array[",
            reflect[Self.T].name(),
            ", ",
            Self.length,
            "]",
        )

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    @always_inline
    def __init__(out self, *, uninitialized: Bool):
        """Create an Array with uninitialized memory.

        Args:
            uninitialized: A boolean to indicate if the array should be
                initialized. Always set to `True` (it's not actually used inside
                the constructor).

        Examples:

        ```mojo
        var uninitialized_array = Array[Int, 10](uninitialized=True)
        ```

        Notes:
            This constructor is unsafe and should be used with caution. The
            array elements will be uninitialized and accessing them before
            initialization is undefined behavior.
        """
        _array_construction_checks[Self.length]()
        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))

    def __init__(
        out self,
        *,
        var unsafe_assume_initialized: Array[MaybeUninit[Self.T], Self.length],
    ) where conforms_to(Self.T, Movable):
        """Constructs an `Array` from an `Array` of
        `MaybeUninit`.

        Args:
            unsafe_assume_initialized: The array of `MaybeUninit`
                elements. All elements must be initialized.

        Warning:
            This is an unsafe constructor. Only use it if you are certain all
            elements are properly initialized.

        Notes:
            This constructor assumes all elements in the input array are
            initialized. Using uninitialized elements results in undefined
            behavior, even for types that are valid for any bit pattern
            (e.g. `Int` or `Float`).
        """

        __mlir_op.`lit.ownership.mark_initialized`(__get_mvalue_as_litref(self))
        for i in range(Self.length):
            self.unsafe_ptr().unsafe_offset(i).unsafe_write_move_from(
                unsafe_assume_initialized[i].unsafe_ptr()
            )
        std.memory.forget_deinit(unsafe_assume_initialized^)

    @always_inline
    def __init__(
        out self,
    ) where conforms_to(Self.T, Defaultable):
        """Constructs an array with each element set to its default value.

        Examples:

        ```mojo
        var arr = Array[Int, 5]()  # [0, 0, 0, 0, 0]
        ```
        """
        _array_construction_checks[Self.length]()
        self = Self(uninitialized=True)
        var ptr = self.unsafe_ptr()
        for i in range(Self.length):
            ptr.unsafe_offset(i).unsafe_write(
                init_with=lambda () -> Self.T: Self.T()
            )

    @always_inline
    def __init__[
        batch_size: Int = 64
    ](out self, *, fill: Self.T) where conforms_to(Self.T, Copyable):
        """Constructs an array where each element is initialized to the supplied
        value.

        Parameters:
            batch_size: The number of elements to unroll for filling the array.
                Default is 64, which optimizes for AVX512 operations on modern
                CPUs. For large arrays (>2k elements), this batched approach
                significantly improves compile times compared to full unrolling
                while maintaining good runtime performance.

        Args:
            fill: The element value to fill each index with.

        Examples:

        ```mojo
        var filled = Array[Int, 5](fill=42)  # [42, 42, 42, 42, 42]

        # For large arrays, consider adjusting batch_size to balance
        # compile time and runtime performance:
        var large = Array[Int, 10000].__init__[batch_size=32](fill=0)
        ```

        Notes:

        - Full unrolling with large arrays (>2k elements) can cause significant
            compiler slowdowns.
        - Using batch_size=64 balances AVX512 efficiency and instruction cache
            usage.
        - For very large arrays, using smaller batch sizes (e.g., 32 or 16) can
            further improve compilation speed while still maintaining good
            runtime performance.
        """
        self = Self.__init__[batch_size=batch_size](
            fill_with=lambda (_i: Int) -> Self.T: fill.copy()
        )

    @always_inline
    def __init__(out self, *, fill_with_unrolled: Some[def[Int]() -> Self.T]):
        """Constructs an array by calling `fill_with_unrolled[i]()` for each
        index `i`.

        Usually prefer the `fill_with` initializer. `fill_with_unrolled` unrolls
        the entire initialization loop, which on a large array hurts compile
        time and doesn't always lead to the best runtime performance.

        Args:
            fill_with_unrolled: A function parameterized on each index in
                `[0, length)`, whose result is written to that position.

        Examples:

        ```mojo
        var simd = SIMD[.uint32, 4](10, 20, 30, 40)
        var extracted = Array[UInt32, 4](
            fill_with_unrolled=lambda[i: Int]() -> UInt32: simd[i]
        )
        # [10, 20, 30, 40]
        ```
        """
        self = {uninitialized = True}
        var ptr = self.unsafe_ptr()

        comptime for i in range(Self.length):
            ptr.unsafe_offset(i).unsafe_write(
                init_with=lambda () {ref} -> Self.T: fill_with_unrolled[i]()
            )

    @always_inline
    def __init__[
        batch_size: Int = 64
    ](out self, *, fill_with: Some[def(Int) -> Self.T]):
        """Constructs an array by calling `fill_with(i)` for each index `i`.

        Parameters:
            batch_size: The number of elements to unroll for filling the array.

        Args:
            fill_with: A function called with each index in `[0, length)`,
                whose result is written to that position.

        Examples:

        ```mojo
        var squares = Array[Int, 5](fill_with=lambda (i: Int) -> Int: i * i)
        # [0, 1, 4, 9, 16]
        ```
        """
        comptime if batch_size >= Self.length:
            self = {
                fill_with_unrolled = lambda [i: Int]() {
                    ref
                } -> Self.T: fill_with(i)
            }
        else:
            comptime unroll_end = std.math.align_down(Self.length, batch_size)

            self = {uninitialized = True}
            var ptr = self.unsafe_ptr()

            # Skip the batched loop entirely when it cannot run. Emitting it for
            # `unroll_end == 0` leaves the inlined iterator's line-table marker
            # behind as an irremovable barrier, even though the loop is dead.
            comptime if unroll_end > 0:
                for batch_start in range(0, unroll_end, batch_size):
                    comptime for i in range(batch_size):
                        var idx = batch_start + i
                        ptr.unsafe_write(
                            init_with=lambda () {ref} -> Self.T: fill_with(idx)
                        )
                        ptr = ptr.unsafe_offset(1)

            # Fill the remainder
            comptime for i in range(unroll_end, Self.length):
                ptr.unsafe_write(
                    init_with=lambda () {ref} -> Self.T: fill_with(i)
                )
                ptr = ptr.unsafe_offset(1)

    def __init__[
        *, __literal_size__: Int
    ](
        out self: Array[Self.T, __literal_size__],
        var *elems: Self.T,
        __list_literal__: NoneType,
    ) where conforms_to(Self.T, Movable):
        """Constructs an array from a variadic list of elements.

        Parameters:
            __literal_size__: Infer the literal size from the list-literal.

        Args:
            elems: The elements to initialize the array with. Must match the
                array size.
            __list_literal__: Tell Mojo to use this method for list literals.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        ```
        """
        self = {uninitialized = True}
        var ptr = self.unsafe_ptr()

        # Move each element into the array storage.
        comptime for i in range(__literal_size__):
            # Safety: We own the elements in the variadic list.
            # The `where conforms_to(Self.T, Movable)` clause narrows the
            # `elems` pack element to `T(Movable)`, but `self.unsafe_ptr()`
            # keeps the struct's `T` bound, so the two views don't unify at
            # `unsafe_write_move_from`. Reconcile the source pointer's element
            # view. (MOCO-4058 fixed the `where`-evidence gap for parametric
            # overloads, but not this pack-element-vs-field-type case.)
            ptr.unsafe_offset(i).unsafe_write_move_from(
                Pointer(to=elems[i]).unsafe_bitcast[Self.T]()
            )

        # Do not destroy the elements when their backing storage goes away.
        # FIXME: Why doesn't consume_elements work here?
        elems^._annihilate()

    @staticmethod
    def _byte_size_favors_field_copy() -> Bool:
        """Returns whether `Self`'s total byte size favors a direct field
        copy over `unsafe_uninit_{copy,move}_n`.

        A field copy wins below ~1024 bytes. Above that, `unsafe_uninit_*_n`
        wins by a growing margin, since it scales with bytes while a field
        copy scales closer to element count once too large for registers.

        Returns:
            `True` if size alone favors a direct field copy.
        """
        comptime TRIVIAL_FAST_PATH_MAX_BYTES = 1024
        return size_of[Self.T]() * Self.length <= TRIVIAL_FAST_PATH_MAX_BYTES

    @stable(since="1.0")
    def __init__(out self, *, copy: Self) where conforms_to(Self.T, Copyable):
        """Copy constructs the array from another array.

        Args:
            copy: The array to copy from.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        var copy = arr.copy()  # Creates new array [1, 2, 3]
        ```
        """
        comptime if IsTriviallyCopyable[
            Self.T
        ] and Self._byte_size_favors_field_copy():
            self._array = copy._array
        else:
            self = Self(uninitialized=True)
            unsafe_uninit_copy_n[overlapping=False](
                dest=self.unsafe_ptr(), src=copy.unsafe_ptr(), count=Self.length
            )

    @stable(since="1.0")
    def __init__(
        out self, *, deinit move: Self
    ) where conforms_to(Self.T, Movable):
        """Move constructs the array from another array.

        Args:
            move: The array to move from.

        Notes:
            Moves the elements from the source array into this array.
        """
        comptime if IsTriviallyMovable[
            Self.T
        ] and Self._byte_size_favors_field_copy():
            self._array = move._array
        else:
            self = Self(uninitialized=True)
            unsafe_uninit_move_n[overlapping=False](
                dest=self.unsafe_ptr(),
                src=move.unsafe_ptr(),
                count=Self.length,
            )

    @stable(since="1.0")
    def __deinit__(
        deinit self,
    ) where conforms_to(Self.T, Deinitable):
        """Destroys the array's elements."""
        unsafe_destroy_n(self.unsafe_ptr(), Self.length)

    def deinit_with(deinit self, deinit_func: Some[def(var Self.T)], /):
        """Consumes this array and deinitializes its elements using the provided
        closure.

        This can be used to deinitialize an `Array` of
        non-`Deinitable` values.

        Args:
            deinit_func: The deinitializing closure called on each array
                element.
        """
        for idx in range(Self.length):
            # TODO(MOCO-4111): `deinit_func` cannot convert to
            # `Pointer.unsafe_deinit_pointee_with` since `Pointer` is
            # bound on `T: AnyType` but `Array` has `T: Movable`.
            deinit_func(
                __get_address_as_owned_value(
                    self.unsafe_ptr().unsafe_offset(idx)._get_kgen_pointer()
                )
            )

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @stable(since="1.0")
    @always_inline
    def __getitem__(ref self, idx: Int, /) -> ref[self] Self.T:
        """Gets a reference to the element at the given index.

        Args:
            idx: The index to access (0 to len-1).

        Returns:
            A reference to the element at the specified index.

        Notes:
            This method provides array-style indexing access to elements in the
            Array. The index is bounds-checked at runtime.
        """
        check_bounds(idx, len(self))
        return self._unchecked_get(idx)

    @always_inline
    def __getitem__(ref self, idx: Some[Indexer]) -> ref[self] Self.T:
        """Gets a reference to the element at the given index.

        Args:
            idx: The index to access (0 to len-1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        print(arr[0])            # Prints 1 - first element
        print(arr[1])            # Prints 2 - second element
        print(arr[len(arr) - 1]) # Prints 3 - last element
        ```

        Notes:
            This method provides array-style indexing access to elements in the
            Array. The index is bounds-checked at runtime.
        """
        check_bounds(idx, len(self))
        return self._unchecked_get(idx)

    @stable(since="1.0")
    @always_inline
    def __getitem_param__[idx: Int, /](ref self) -> ref[self] Self.T:
        """Gets a reference to the element at the given index with compile-time
        bounds checking.

        Parameters:
            idx: The compile-time constant index to access (0 to len-1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        print(arr[0])            # Prints 1 - first element
        print(arr[1])            # Prints 2 - second element
        print(arr[len(arr) - 1]) # Prints 3 - last element
        ```

        Notes:
            This overload provides array-style indexing with compile-time bounds
            checking. The index must be a compile-time constant value.
        """
        comptime assert (
            index(idx) >= 0
        ), "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        comptime assert index(idx) < Self.length, "index is out of bounds"
        return self._unchecked_get(materialize[idx]())

    @always_inline
    def __getitem_param__[
        idx: Some[Indexer & Deinitable]
    ](ref self) -> ref[self] Self.T:
        """Gets a reference to the element at the given index with compile-time
        bounds checking.

        Parameters:
            idx: The compile-time constant index to access (0 to len-1).

        Returns:
            A reference to the element at the specified index.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        print(arr[0])            # Prints 1 - first element
        print(arr[1])            # Prints 2 - second element
        print(arr[len(arr) - 1]) # Prints 3 - last element
        ```

        Notes:
            This overload provides array-style indexing with compile-time bounds
            checking. The index must be a compile-time constant value.
        """
        # Can't construct a String here with the index for the error message, as
        # it causes infinite cycles in gpu compilation tests
        comptime assert (
            index(idx) >= 0
        ), "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        comptime assert index(idx) < Self.length, "index is out of bounds"
        return self._unchecked_get(materialize[idx]())

    @always_inline
    def _unchecked_get(ref self, idx: Some[Indexer]) -> ref[self] Self.T:
        var ptr = __mlir_op.`pop.array.gep`(
            Pointer(to=self._array)._get_kgen_pointer(),
            index(idx).__mlir_index__(),
        )
        return Pointer[_, origin_of(self)](_mlir_value=ptr)[]

    @always_inline
    def concat(
        deinit self,
        deinit rhs: Array[Self.T, _],
        out result: Array[Self.T, Self.length + rhs.length],
    ) where conforms_to(Self.T, Movable):
        """Concatenates this array with another array.

        Both arrays are consumed and their elements are moved into the
        result.

        Args:
            rhs: The array whose elements follow this array's elements in
                the result.

        Returns:
            An array of length `Self.length + rhs.length` containing the
            elements of `self` followed by the elements of `rhs`.

        Examples:

        ```mojo
        var a = [1, 2]
        var b = [3, 4, 5]
        var c = a^.concat(b^)  # [1, 2, 3, 4, 5]
        ```
        """
        result = {uninitialized = True}
        unsafe_uninit_move_n[overlapping=False](
            dest=result.unsafe_ptr(), src=self.unsafe_ptr(), count=Self.length
        )
        unsafe_uninit_move_n[overlapping=False](
            dest=result.unsafe_ptr().unsafe_offset(Self.length),
            src=rhs.unsafe_ptr(),
            count=rhs.length,
        )

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline
    def __len__(self) -> Int:
        """Returns the length of the array.

        Returns:
            The size of the array as an Int.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        print(len(arr))  # Prints 3
        ```

        Notes:
            The length is a compile-time constant value determined by the
            `length` parameter used when creating the array.
        """
        return Self.length

    @stable(since="1.0")
    def __eq__(self, other: Self) -> Bool where conforms_to(Self.T, Equatable):
        """Compares two arrays for equality.

        Args:
            other: The other array to compare against.

        Returns:
            True if all elements are equal, False otherwise.
        """
        # Not a `comptime for` as we can just rely on LLVM unrolling to its
        # own budget. `comptime for` forces large arrays to be fully
        # unrolled, which can be a performance hit.
        for i in range(Self.length):
            if self.unsafe_get(i) != other.unsafe_get(i):
                return False
        return True

    @stable(since="1.0")
    def __ne__(self, other: Self) -> Bool where conforms_to(Self.T, Equatable):
        """Compares two arrays for inequality.

        Args:
            other: The other array to compare against.

        Returns:
            True if any elements are not equal, False otherwise.
        """
        for i in range(Self.length):
            if self.unsafe_get(i) != other.unsafe_get(i):
                return True
        return False

    @always_inline
    def _compare(
        self, other: Self
    ) -> Int where conforms_to(Self.T, Comparable):
        """Lexicographically three-way compares this array to another.

        Both arrays share the same compile-time `length`, so the comparison
        walks the elements pairwise and the first differing pair decides the
        result.

        Args:
            other: The other array to compare against.

        Returns:
            A negative value if `self` sorts before `other`, zero if they are
            equal, and a positive value if `self` sorts after `other`.
        """
        for i in range(Self.length):
            if self.unsafe_get(i) < other.unsafe_get(i):
                return -1
            if other.unsafe_get(i) < self.unsafe_get(i):
                return 1
        return 0

    @always_inline
    def __lt__(self, other: Self) -> Bool where conforms_to(Self.T, Comparable):
        """Compares two arrays lexicographically for less-than ordering.

        Args:
            other: The other array to compare against.

        Returns:
            True if `self` sorts before `other` lexicographically, comparing
            elements pairwise so the first differing pair decides the result.
        """
        return self._compare(other) < 0

    @always_inline
    def __le__(self, other: Self) -> Bool where conforms_to(Self.T, Comparable):
        """Compares two arrays lexicographically for less-than-or-equal
        ordering.

        Args:
            other: The other array to compare against.

        Returns:
            True if `self` sorts before or equal to `other` lexicographically.
        """
        return self._compare(other) <= 0

    @always_inline
    def __gt__(self, other: Self) -> Bool where conforms_to(Self.T, Comparable):
        """Compares two arrays lexicographically for greater-than ordering.

        Args:
            other: The other array to compare against.

        Returns:
            True if `self` sorts after `other` lexicographically, comparing
            elements pairwise so the first differing pair decides the result.
        """
        return self._compare(other) > 0

    @always_inline
    def __ge__(self, other: Self) -> Bool where conforms_to(Self.T, Comparable):
        """Compares two arrays lexicographically for greater-than-or-equal
        ordering.

        Args:
            other: The other array to compare against.

        Returns:
            True if `self` sorts after or equal to `other` lexicographically.
        """
        return self._compare(other) >= 0

    def __hash__[
        H: Hasher
    ](self, mut hasher: H) where conforms_to(Self.T, Hashable):
        """Hashes the elements of the array using the given hasher.

        Parameters:
            H: The hasher type.

        Args:
            hasher: The hasher instance.
        """
        for i in range(Self.length):
            self.unsafe_get(i).__hash__(hasher)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline
    def unsafe_get(ref self, idx: Some[Indexer]) -> ref[self] Self.T:
        """Gets a reference to an element without bounds checking.

        Args:
            idx: The index of the element to get. Must be non-negative and in
                bounds. Using an invalid index will cause undefined behavior.

        Returns:
            A reference to the element at the given index.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        print(arr.unsafe_get(0))  # Prints 1
        ```

        Warning:
            This is an unsafe method. No bounds checking is performed.
            Using an invalid index will cause undefined behavior.
            Negative indices are not supported.

        Notes:
            This is an unsafe method that skips bounds checking for performance.
            Users should prefer `__getitem__` instead for safety.
        """
        check_bounds[cpu_default=False](idx, len(self))
        return self._unchecked_get(idx)

    @always_inline
    @stable(since="1.0")
    def unsafe_ptr[
        origin: Origin, address_space: AddressSpace, //
    ](ref[origin, address_space] self) -> Pointer[
        Self.T, origin, address_space=address_space
    ]:
        """Gets an unsafe pointer to the underlying array storage.

        Parameters:
            origin: The origin of the reference to self.
            address_space: The address space of the array.

        Returns:
            A `Pointer` to the underlying array storage. The pointer's
            mutability matches that of the array reference.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        var ptr = arr.unsafe_ptr()
        print(ptr[0])  # Prints 1
        ```

        Warning:
            This is an unsafe method. The returned pointer:
            - Becomes invalid if the array is moved
            - Must not be used to access memory outside array bounds
            - Must be refreshed after any operation that could move the array

        Notes:
            Returns a raw pointer to the array's memory that can be used for
            direct memory access. The pointer inherits mutability from the array
            reference.
        """
        return (
            Pointer(to=self._array)
            .unsafe_bitcast[Self.T]()
            .unsafe_origin_cast[origin]()
            .unsafe_address_space_cast[address_space]()
        )

    @always_inline
    @stable(since="1.0")
    def __contains__(
        self, value: Self.T
    ) -> Bool where conforms_to(Self.T, Equatable):
        """Tests if a value is present in the array using the `in` operator.

        Args:
            value: The value to search for.

        Returns:
            True if the value is found in any position in the array, False
            otherwise.

        Examples:

        ```mojo
        var arr = [1, 2, 3]
        print(3 in arr)  # Prints True - value exists
        print(4 in arr)  # Prints False - value not found
        ```

        Notes:
            This method enables using the `in` operator to check if a value
            exists in the array. It performs a linear search comparing each
            element for equality with the given value. The element type must
            implement the `Equatable` trait to support equality comparison.
        """
        for i in range(Self.length):
            if self[i] == value:
                return True
        return False

    # ===-------------------------------------------------------------------===#
    # String representation
    # ===-------------------------------------------------------------------===#

    def _write_self_to[
        f: def(Self.T, mut Some[Writer]) thin
    ](self, mut writer: Some[Writer]) where conforms_to(Self.T, Writable):
        var index = 0

        var self_ptr = Pointer(to=self)

        def iterate(
            mut w: Some[Writer],
        ) raises StopIteration {mut index, self_ptr}:
            if index >= Self.length:
                raise StopIteration()
            f(self_ptr[].unsafe_get(index), w)
            index += 1

        fmt.write_sequence_to(writer, iterate)
        _ = index

    def write_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Writes the Array representation to a Writer.

        Args:
            writer: The object to write to.
        """
        self._write_self_to[f=fmt.write_to[Self.T]](writer)

    def write_repr_to(
        self, mut writer: Some[Writer]
    ) where conforms_to(Self.T, Writable):
        """Writes the repr representation of this Array to a Writer.

        Args:
            writer: The object to write to.
        """

        var self_ptr = Pointer(to=self)

        def write_fields(mut w: Some[Writer]) {self_ptr}:
            self_ptr[]._write_self_to[f=fmt.write_repr_to[Self.T]](w)

        fmt.FormatStruct(writer, "Array").params(
            fmt.TypeNames[Self.T](),
            Self.length,
        ).fields(write_fields)

    # TODO(MOCO-4308): Remove redundant 'Movable' constraint
    def __iter__(
        var self,
    ) -> Self.IteratorOwnedType where conforms_to(Self.T, Movable & Deinitable):
        """Consume the array and return an iterator over its elements.

        Returns:
            An iterator that owns the array's elements.
        """
        return Self.IteratorOwnedType(self^)

    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Iterate over elements of the array, returning immutable references.

        Returns:
            An iterator of immutable references to the array elements.
        """
        # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
        comptime assert conforms_to(
            Self.T, Copyable
        ), "Array iteration requires the element to be `Copyable`."
        return {0, Pointer(to=self)}

    # TODO(MSTDL-2390): Remove `Copyable` constraint once we have better iter traits.
    def __reversed__(
        ref self,
    ) -> _ArrayIter[
        Self.T,
        Self.length,
        origin_of(self),
        False,
    ] where conforms_to(Self.T, Copyable):
        """Iterate over elements of the array in reverse order, returning
        immutable references.

        Returns:
            An iterator of immutable references to the array elements in reverse
            order.
        """
        return _ArrayIter[forward=False](
            Self.length,
            Pointer(to=self),
        )

    @always_inline
    def repeat[
        n: Int
    ](
        deinit self,
        out result: Array[Self.T, Self.length * n],
    ) where (
        conforms_to(Self.T, Copyable) and (n > 0),
        "`Array * n` requires n > 0",
    ):
        """Repeats this array's elements the given number of times.


        Parameters:
            n: The number of repetitions. Must be greater than 0.

        Returns:
            An array of length `Self.length * rhs` containing the elements
            of `self` repeated `rhs` times.

        Examples:

        ```mojo
        var a = [1, 2].repeat[3]()  # Array[Int, 6](1, 2, 1, 2, 1, 2)
        ```
        """
        result = {uninitialized = True}
        for rep in range(n - 1):
            unsafe_uninit_copy_n[overlapping=False](
                dest=result.unsafe_ptr().unsafe_offset(rep * Self.length),
                src=self.unsafe_ptr(),
                count=Self.length,
            )
        unsafe_uninit_move_n[overlapping=False](
            dest=result.unsafe_ptr().unsafe_offset((n - 1) * Self.length),
            src=self.unsafe_ptr(),
            count=Self.length,
        )
