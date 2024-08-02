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
"""Implements StaticTuple, a statically-sized uniform container.

You can import these APIs from the `utils` package. For example:

```mojo
from utils import StaticTuple
```
"""
from collections._index_normalization import normalize_index
from sys.intrinsics import _type_is_eq

from memory import UnsafePointer

# ===----------------------------------------------------------------------===#
# Utilities
# ===----------------------------------------------------------------------===#


@always_inline
fn _set_array_elem[
    index: Int,
    size: Int,
    type: AnyTrivialRegType,
](
    val: type,
    ref [_]array: __mlir_type[`!pop.array<`, size.value, `, `, type, `>`],
):
    """Sets the array element at position `index` with the value `val`.

    Parameters:
        index: the position to replace the value at.
        size: the size of the array.
        type: the element type of the array

    Args:
        val: the value to set.
        array: the array which is captured by reference.
    """
    var ptr = __mlir_op.`pop.array.gep`(
        UnsafePointer.address_of(array).address, index.value
    )
    UnsafePointer(ptr)[] = val


@always_inline
fn _create_array[
    size: Int, type: AnyTrivialRegType
](lst: VariadicList[type]) -> __mlir_type[
    `!pop.array<`, size.value, `, `, type, `>`
]:
    """Sets the array element at position `index` with the value `val`.

    Parameters:
        size: the size of the array.
        type: the element type of the array

    Args:
        lst: the list of values to set.

    Returns:
        The array with values filled from the input list.
    """
    debug_assert(size == len(lst), "mismatch in the number of elements")

    if len(lst) == 1:
        return __mlir_op.`pop.array.repeat`[
            _type = __mlir_type[`!pop.array<`, size.value, `, `, type, `>`]
        ](lst[0])

    else:
        var array = __mlir_op.`kgen.undef`[
            _type = __mlir_type[`!pop.array<`, size.value, `, `, type, `>`]
        ]()

        @parameter
        for idx in range(size):
            _set_array_elem[idx, size, type](lst[idx], array)

        return array


# ===----------------------------------------------------------------------===#
# StaticTuple
# ===----------------------------------------------------------------------===#


fn _static_tuple_construction_checks[size: Int]():
    """Checks if the properties in `StaticTuple` are valid.

    Validity right now is just ensuring the number of elements is > 0.

    Parameters:
      size: The number of elements.
    """
    constrained[size > 0, "number of elements in `StaticTuple` must be > 0"]()


@value
@register_passable("trivial")
struct StaticTuple[element_type: AnyTrivialRegType, size: Int](
    Sized, CollectionElement
):
    """A statically sized tuple type which contains elements of homogeneous types.

    Parameters:
        element_type: The type of the elements in the tuple.
        size: The size of the tuple.
    """

    alias type = __mlir_type[
        `!pop.array<`, size.value, `, `, Self.element_type, `>`
    ]
    var array: Self.type
    """The underlying storage for the static tuple."""

    @always_inline
    fn __init__(inout self):
        """Constructs an empty (undefined) tuple."""
        _static_tuple_construction_checks[size]()
        self.array = __mlir_op.`kgen.undef`[_type = Self.type]()

    @always_inline
    fn __init__(inout self, *elems: Self.element_type):
        """Constructs a static tuple given a set of arguments.

        Args:
            elems: The element types.
        """
        _static_tuple_construction_checks[size]()
        self.array = _create_array[size](elems)

    @always_inline
    fn __init__(inout self, values: VariadicList[Self.element_type]):
        """Creates a tuple constant using the specified values.

        Args:
            values: The list of values.
        """
        _static_tuple_construction_checks[size]()
        self.array = _create_array[size, Self.element_type](values)

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided StaticTuple.

        Args:
            other: The StaticTuple to copy.
        """
        self.array = other.array

    @always_inline("nodebug")
    fn __len__(self) -> Int:
        """Returns the length of the array. This is a known constant value.

        Returns:
            The size of the list.
        """
        return size

    @always_inline("nodebug")
    fn __getitem__[index: Int](self) -> Self.element_type:
        """Returns the value of the tuple at the given index.

        Parameters:
            index: The index into the tuple.

        Returns:
            The value at the specified position.
        """
        constrained[index < size]()
        var val = __mlir_op.`pop.array.get`[
            _type = Self.element_type,
            index = index.value,
        ](self.array)
        return val

    @always_inline("nodebug")
    fn __setitem__[index: Int](inout self, val: Self.element_type):
        """Stores a single value into the tuple at the specified index.

        Parameters:
            index: The index into the tuple.

        Args:
            val: The value to store.
        """
        constrained[index < size]()
        var tmp = self
        _set_array_elem[index, size, Self.element_type](val, tmp.array)
        self = tmp

    @always_inline("nodebug")
    fn __getitem__(self, idx: Int) -> Self.element_type:
        """Returns the value of the tuple at the given dynamic index.

        Args:
            idx: The index into the tuple.

        Returns:
            The value at the specified position.
        """
        debug_assert(idx < size, "index must be within bounds")
        # Copy the array so we can get its address, because we can't take the
        # address of 'self' in a non-mutating method.
        var arrayCopy = self.array
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(arrayCopy).address, idx.value
        )
        var result = UnsafePointer(ptr)[]
        _ = arrayCopy
        return result

    @always_inline("nodebug")
    fn __setitem__(inout self, idx: Int, val: Self.element_type):
        """Stores a single value into the tuple at the specified dynamic index.

        Args:
            idx: The index into the tuple.
            val: The value to store.
        """
        debug_assert(idx < size, "index must be within bounds")
        var tmp = self
        var ptr = __mlir_op.`pop.array.gep`(
            UnsafePointer.address_of(tmp.array).address, idx.value
        )
        UnsafePointer(ptr)[] = val
        self = tmp


# ===----------------------------------------------------------------------===#
# Array
# ===----------------------------------------------------------------------===#


@value
struct InlineArray[
    ElementType: CollectionElementNew,
    size: Int,
](Sized, Movable, Copyable, ExplicitlyCopyable):
    """A fixed-size sequence of size homogeneous elements where size is a constant expression.

    Parameters:
        ElementType: The type of the elements in the array.
        size: The size of the array.
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

    @always_inline
    fn __init__(inout self):
        """This constructor will always cause a compile time error if used.
        It is used to steer users away from uninitialized memory.
        """
        constrained[
            False,
            (
                "Initialize with either a variadic list of arguments, a default"
                " fill element or pass the keyword argument"
                " 'unsafe_uninitialized'."
            ),
        ]()
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

    @always_inline
    fn __init__(inout self, *, unsafe_uninitialized: Bool):
        """Create an InlineArray with uninitialized memory.

        Note that this is highly unsafe and should be used with caution.

        We recommend to use the `InlineList` instead if all the objects
        are not available when creating the array.

        If despite those workarounds, one still needs an uninitialized array,
        it is possible with:

        ```mojo
        var uninitialized_array = InlineArray[Int, 10](unsafe_uninitialized=True)
        ```

        Args:
            unsafe_uninitialized: A boolean to indicate if the array should be initialized.
                Always set to `True` (it's not actually used inside the constructor).
        """
        _static_tuple_construction_checks[size]()
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

    @always_inline
    fn __init__(inout self, fill: Self.ElementType):
        """Constructs an empty array where each element is the supplied `fill`.

        Args:
            fill: The element to fill each index.
        """
        _static_tuple_construction_checks[size]()
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

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
        owned storage: VariadicListMem[Self.ElementType, _, _],
    ):
        """Construct an array from a low-level internal representation.

        Args:
            storage: The variadic list storage to construct from.
        """

        debug_assert(len(storage) == size, "Elements must be of length size")
        _static_tuple_construction_checks[size]()
        self._array = __mlir_op.`kgen.undef`[_type = Self.type]()

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

    fn __init__(inout self, *, other: Self):
        """Explicitly copy the provided value.

        Args:
            other: The value to copy.
        """

        self = Self(unsafe_uninitialized=True)

        for idx in range(size):
            var ptr = self.unsafe_ptr() + idx

            ptr.init_pointee_explicit_copy(other[idx])

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
        from utils import InlineArray
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
