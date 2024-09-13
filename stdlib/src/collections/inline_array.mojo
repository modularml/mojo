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
"""Implements InlineArray, a statically-sized uniform container.

You can import these APIs from the `collections` package. For example:

```mojo
from collections import InlineArray
```
"""

from collections._index_normalization import normalize_index
from sys.intrinsics import _type_is_eq
from memory.maybe_uninitialized import UnsafeMaybeUninitialized


# ===----------------------------------------------------------------------===#
# Inline Array
# ===----------------------------------------------------------------------===#


fn _inline_array_construction_checks[size: Int]():
    """Checks if the properties in `InlineArray` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
      size: The number of elements.
    """
    constrained[size >= 0, "number of elements in `InlineArray` must be >= 0"]()


@value
struct InlineArray[
    ElementType: CollectionElementNew,
    size: Int,
    *,
    run_destructors: Bool = False,
](Sized, Movable, Copyable, ExplicitlyCopyable):
    """A fixed-size sequence of size homogeneous elements where size is a constant expression.

    Parameters:
        ElementType: The type of the elements in the array.
        size: The size of the array.
        run_destructors: Whether to run destructors on the elements.  Defaults to False for *backwards compatibility* reasons only.  Eventually this will default to `True` and/or the parameter will be removed to unconditionally run destructors on the elements.
    """

    # Fields
    alias type = __mlir_type[
        `!pop.array<`, size.value, `, `, Self.ElementType, `>`
    ]
    var _array: Self.type
    """The underlying storage for the array."""

    # ===------------------------------------------------------------------===#
    # Life cycle methods
    # ===------------------------------------------------------------------===#

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """
        self.__init__[False]()

        for idx in range(size):
            var ptr = self.unsafe_ptr() + idx

            ptr.init_pointee_explicit_copy(other[idx])

    @always_inline
    fn __init__[
        _T: DefaultableCollectionElementNew
    ](inout self: InlineArray[_T, size, run_destructors=run_destructors]):
        """Create an InlineArray with elements initialized to their default values.

        Parameters:
            _T: A type conformaing to `Defaultable` and `CollectionElement`.
        """
        self.__init__(_T())

    @deprecated(
        "Initialize with either a variadic list of arguments,"
        " a default fill element, or use `array.__init__[False]().`"
    )
    fn __init__[
        _T: CollectionElementNew, overload_1: None = None
    ](inout self: InlineArray[_T, size, run_destructors=run_destructors]):
        self.__init__[False]()
        constrained[
            False,
            (
                "Initialize with either a variadic list of arguments,"
                " a default fill element, or set init_data to False."
            ),
        ]()

    @always_inline
    fn __init__[init_data: Bool](inout self):
        """Used to create an InlineArray with uninitialized memory by setting `init_data` to False.

        Note that this is highly unsafe, and you should initialize the elements manually.

        We recommend to use the `InlineList` instead if all the objects
        are not available when creating the array.

        If you still need an uninitialized array, it is possible with:

        ```mojo
        var uninitialized_array: InlineArray[Int, 10]
        uninitialized_array.__init__[False]()
        # ... manually initialize elements here, or you will get undefined behaviour.
        ```

        Parameters:
            init_data: A boolean to indicate if the array should be initialized.
        """

        @parameter
        if init_data:
            constrained[
                False,
                (
                    "Initialize with either a variadic list of arguments,"
                    " a default fill element, or set init_data to False."
                ),
            ]()
        _inline_array_construction_checks[size]()
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

    fn __init__(
        inout self,
        *,
        owned unsafe_assume_initialized: InlineArray[
            UnsafeMaybeUninitialized[Self.ElementType], Self.size
        ],
    ):
        """Constructs an `InlineArray` from an `InlineArray` of `UnsafeMaybeUninitialized`.

        Calling this function assumes that all elements in the input array are initialized.

        If the elements of the input array are not initialized, the behavior is undefined,
        even  if `ElementType` is valid *for every possible bit pattern* (e.g. `Int` or `Float`).

        Args:
            unsafe_assume_initialized: The array of `UnsafeMaybeUninitialized` elements.
        """

        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

        for i in range(Self.size):
            unsafe_assume_initialized[i].unsafe_ptr().move_pointee_into(
                self.unsafe_ptr() + i
            )

    @always_inline
    fn __init__(inout self, fill: Self.ElementType):
        """Constructs an empty array where each element is the supplied `fill`.

        Args:
            fill: The element to fill each index.
        """
        self.__init__[False]()

        @parameter
        for i in range(size):
            var ptr = UnsafePointer.address_of(self.unsafe_get(i))
            ptr.init_pointee_explicit_copy(fill)

    @always_inline
    fn __init__(inout self, owned *elems: Self.ElementType):
        """Constructs an array given a set of arguments.

        Args:
            elems: The element types.
        """

        self = Self(storage=elems^)

    @always_inline("nodebug")
    fn __init__(
        inout self,
        *,
        owned storage: VariadicListMem[Self.ElementType, _],
    ):
        """Construct an array from a low-level internal representation.

        Args:
            storage: The variadic list storage to construct from.
        """

        debug_assert(len(storage) == size, "Elements must be of length size")
        self.__init__[False]()

        # Move each element into the array storage.
        @parameter
        for i in range(size):
            var eltref: Reference[
                Self.ElementType, __lifetime_of(self)
            ] = self.unsafe_get(i)
            UnsafePointer.address_of(storage[i]).move_pointee_into(
                UnsafePointer[Self.ElementType].address_of(eltref[])
            )

        # Mark the elements as already destroyed.
        storage._is_owned = False

    fn __copyinit__(inout self, other: Self):
        """Copy construct the array.

        Args:
            other: The array to copy.
        """

        self = Self(other=other)

    fn __del__(owned self):
        """Deallocate the array."""

        @parameter
        if Self.run_destructors:

            @parameter
            for idx in range(size):
                var ptr = self.unsafe_ptr() + idx
                ptr.destroy_pointee()

    # ===------------------------------------------------------------------===#
    # Operator dunders
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn __getitem__(
        ref [_]self: Self, idx: Int
    ) -> ref [__lifetime_of(self)] Self.ElementType:
        """Get a `Reference` to the element at the given index.

        Args:
            idx: The index of the item.

        Returns:
            A reference to the item at the given index.
        """
        var normalized_index = normalize_index["InlineArray"](idx, self)

        return self.unsafe_get(normalized_index)

    @always_inline("nodebug")
    fn __getitem__[
        idx: Int,
    ](ref [_]self: Self) -> ref [__lifetime_of(self)] Self.ElementType:
        """Get a `Reference` to the element at the given index.

        Parameters:
            idx: The index of the item.

        Returns:
            A reference to the item at the given index.
        """
        constrained[-size <= idx < size, "Index must be within bounds."]()

        var normalized_idx = idx

        @parameter
        if idx < 0:
            normalized_idx += size

        return self.unsafe_get(normalized_idx)

    # ===------------------------------------------------------------------=== #
    # Trait implementations
    # ===------------------------------------------------------------------=== #

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Returns the length of the array. This is a known constant value.

        Returns:
            The size of the array.
        """
        return size

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    fn unsafe_get(
        ref [_]self: Self, idx: Int
    ) -> ref [__lifetime_of(self)] Self.ElementType:
        """Get a reference to an element of self without checking index bounds.

        Users should opt for `__getitem__` instead of this method as it is
        unsafe.

        Note that there is no wraparound for negative indices. Using negative
        indices is considered undefined behavior.

        Args:
            idx: The index of the element to get.

        Returns:
            A reference to the element at the given index.
        """
        var idx_as_int = index(idx)
        debug_assert(
            0 <= idx_as_int < size,
            (
                "Index must be within bounds when using"
                " `InlineArray.unsafe_get()`."
            ),
        )
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(self._array).address,
            idx_as_int.value,
        )
        return UnsafePointer(ptr)[]

    @always_inline
    fn unsafe_ptr(self) -> UnsafePointer[Self.ElementType]:
        """Get an `UnsafePointer` to the underlying array.

        That pointer is unsafe but can be used to read or write to the array.
        Be careful when using this. As opposed to a pointer to a `List`,
        this pointer becomes invalid when the `InlineArray` is moved.

        Make sure to refresh your pointer every time the `InlineArray` is moved.

        Returns:
            An `UnsafePointer` to the underlying array.
        """
        return UnsafePointer.address_of(self._array).bitcast[Self.ElementType]()

    @always_inline
    fn __contains__[
        T: EqualityComparableCollectionElement, //
    ](self, value: T) -> Bool:
        """Verify if a given value is present in the array.

        ```mojo
        from collections import InlineArray
        var x = InlineArray[Int, 3](1,2,3)
        if 3 in x: print("x contains 3")
        ```

        Parameters:
            T: The type of the elements in the array. Must implement the
              traits `EqualityComparable` and `CollectionElement`.

        Args:
            value: The value to find.

        Returns:
            True if the value is contained in the array, False otherwise.
        """
        constrained[
            _type_is_eq[T, Self.ElementType](),
            "T must be equal to Self.ElementType",
        ]()

        @parameter
        for i in range(size):
            if (
                rebind[Reference[T, __lifetime_of(self)]](Reference(self[i]))[]
                == value
            ):
                return True
        return False
