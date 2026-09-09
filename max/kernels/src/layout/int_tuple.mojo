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

"""Hierarchical integer tuple data structures for high-performance tensor operations.

This module provides a flexible, memory-efficient implementation of nested integer tuples
optimized for tensor shape, stride, and index operations in high-performance computing.
The core data structures support both flat and hierarchical representations with
efficient memory sharing and zero-copy views.

Key components:
- `IntArray`: Low-level register-passable array with direct memory management
- `IntTuple`: Hierarchical nested tuple with efficient memory layout and operations
- Utility functions for tensor shape manipulation, coordinate transformations, and layout operations

Performance features:
- Register-passable data structures for optimal compiler optimizations
- Zero-copy views for efficient memory sharing
- Specialized memory layout for nested structures
- Optimized algorithms for common tensor operations

Common operations:
- Shape manipulation: `flatten`, `to_nest`, `apply`, `product`, `sum`
- Coordinate transformations: `idx2crd`, `crd2idx`
- Layout operations: `compact_order`, `prefix_product`
- Structural comparisons: `congruent`, `compatible`, `weakly_congruent`

Example usage:

```mojo
from layout import IntTuple
from layout.int_tuple import flatten, compact_order, size

# Create nested tuples
var shape = IntTuple(2, IntTuple(3, 4), 5)  # Represents shape (2, (3, 4), 5)

# Flatten a nested tuple
var flat = flatten(shape)  # Results in (2, 3, 4, 5)

# Create compact strides for a given shape and order
var order = IntTuple(1, IntTuple(2, 3), 4)
var strides = compact_order(shape, order)  # Results in (1, (2, 6), 24)

# Calculate total size (product of all elements)
var total_size = size(shape)  # Results in 120
```
"""

from std.os import abort

from std.builtin.range import _StridedRange
from std.memory import dealloc, unsafe_memcpy, ThinAllocation
from std.memory.alloc import Layout as AllocLayout
from std.collections import check_bounds
from std.utils.numerics import max_finite
from std.utils import IndexList
from layout.coord import ComptimeInt, Coord, CoordLike
from .layout import Layout
from . import math as layout_math


def _get_index_type(address_space: AddressSpace) -> DType:
    """Returns int32 for shared/constant GPU memory, index otherwise."""
    if address_space in (
        AddressSpace.SHARED,
        AddressSpace.CONSTANT,
    ):
        return DType.int32
    else:
        return DType.int64


def _get_index_type(layout: Layout) -> DType:
    """Returns int32 if layout size fits in int32 range, int64 otherwise."""
    if layout.cosize() <= Int(max_finite[.int32]()):
        return DType.int32

    return DType.int64


def _get_index_type(layout: Layout, address_space: AddressSpace) -> DType:
    """Selects index type based on layout and address space."""
    if layout.all_dims_known():
        return _get_index_type(layout)
    else:
        return _get_index_type(address_space)


def _get_unsigned_type(layout: Layout, address_space: AddressSpace) -> DType:
    """Returns int32 if layout fits in int32 range or index type is int32, otherwise index.
    """
    if layout.all_dims_known() and layout.cosize() < Int(max_finite[.int32]()):
        return DType.int32
    else:
        var dtype = _get_index_type(address_space)
        return DType.int32 if dtype == .int32 else DType.int64


def _get_layout_type(layout: Layout, address_space: AddressSpace) -> DType:
    """Returns int32 if shape fits in uint32 range or address space is shared/constant, otherwise index.
    """

    if layout.all_dims_known():
        var shape = layout.shape.flatten()

        for i in range(len(shape)):
            if shape[i].value() > Int(max_finite[.int32]()):
                return DType.int64

        return DType.int32
    else:
        return _get_index_type(address_space)


struct IntArray(ImplicitlyCopyable, RegisterPassable):
    """A memory-efficient, register-passable array of integers.

    `IntArray` provides a low-level implementation of a dynamically-sized integer array
    with direct memory management.

    This struct serves as the underlying storage mechanism for `IntTuple` and related
    data structures, optimized for high-performance tensor operations.
    """

    var _data: Optional[MutPointer[Int, MutUntrackedOrigin]]
    var _size: Int

    @always_inline("nodebug")
    def __init__(out self, size: Int = 0):
        """Initialize a new owned `IntArray` with the specified size.

        Args:
            size: Number of integers to allocate space for. Defaults to 0.
        """
        if size > 0:
            self._data = alloc(AllocLayout[Int](count=size)).unsafe_leak()
        else:
            self._data = {}
        self._size = size

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Initialize by copying an existing `IntArray`.

        For owned arrays, this performs a deep copy of the data.

        Args:
            copy: The source array to copy from.
        """
        self._size = copy._size
        if copy.owning():
            var size = copy.size()
            self._data = alloc(AllocLayout[Int](count=size)).unsafe_leak()
            self.copy_from(0, copy, size)
        else:
            self._data = copy._data

    @always_inline("nodebug")
    def __deinit__(deinit self):
        """Destroy the `IntArray` and free its memory if owned.

        Only frees memory for owned arrays (positive _size) to prevent
        double-free errors with views.
        """
        if self.owning() and self._data:
            dealloc(
                ThinAllocation(
                    unsafe_owned_ptr=self._data.unsafe_value()
                ).unsafe_with_layout(AllocLayout[Int](count=self.size()))
            )

    @always_inline("nodebug")
    def __getitem__(self, idx: Int) -> Int:
        """Access an element at the specified index.

        Args:
            idx: Zero-based index of the element to access.

        Returns:
            The integer value at the specified index.
        """
        # TODO(MOCO-3154) - put a bounds check here when the le comparison is fixed.
        # and add below back to the docstring.
        # Note:
        #     Bounds checking is performed when assertions are enabled (e.g., -D ASSERT=all).

        return self._data.unsafe_value()[idx]

    @always_inline("nodebug")
    def __setitem__(mut self, idx: Int, value: Int):
        """Set the value at the specified index.

        Args:
            idx: Zero-based index of the element to modify.
            value: The integer value to store at the specified index.

        Note:
            Bounds checking is performed when assertions are enabled (e.g., -D ASSERT=all).
        """
        debug_assert(
            idx >= 0 and idx < self.size(),
            "IntArray index out of bounds: ",
            idx,
            " (size: ",
            self.size(),
            ")",
        )

        self._data.unsafe_value()[idx] = value

    @always_inline("nodebug")
    def owning(self) -> Bool:
        """Check if this `IntArray` owns its memory.

        Returns:
            True if this array owns its memory (positive _size),
            False if it's a view (negative _size).
        """
        return self._size > 0

    @always_inline("nodebug")
    def size(self) -> Int:
        """Get the number of elements in the array.

        Returns:
            The number of elements in the array, regardless of ownership status.
        """
        return layout_math.abs(self._size)

    @always_inline("nodebug")
    def copy_from(mut self, offset: Int, source: Self, size: Int):
        """Copy elements from another `IntArray`.

        Args:
            offset: Destination offset in this array.
            source: Source array to copy from.
            size: Number of elements to copy.
        """
        if self._data and source._data:
            unsafe_memcpy(
                dest=self._data.unsafe_value() + offset,
                src=source._data.unsafe_value(),
                count=size,
            )

    @always_inline("nodebug")
    def copy_from(
        mut self, dst_offset: Int, source: Self, src_offset: Int, size: Int
    ):
        """Copy elements from another IntArray with source offset.

        Args:
            dst_offset: Destination offset in this array.
            source: Source array to copy from.
            src_offset: Source offset in the source array.
            size: Number of elements to copy.
        """
        if self._data and source._data:
            unsafe_memcpy(
                dest=self._data.unsafe_value() + dst_offset,
                src=source._data.unsafe_value() + src_offset,
                count=size,
            )


comptime UNKNOWN_VALUE = -1
"""Special value indicating an unknown or unspecified dimension.

This constant is used throughout the `IntTuple` system to represent dimensions
that are not known at compile time or have not been specified.
"""


def create_unknown_int_tuple(rank: Int) -> IntTuple:
    """Creates an IntTuple of the given rank with all UNKNOWN_VALUE entries.

    Args:
        rank: The number of dimensions.

    Returns:
        An IntTuple with `rank` elements, all set to UNKNOWN_VALUE.
    """
    var result = IntTuple(rank)
    _to_unknown(result)
    return result


struct _IntTupleIter[origin: ImmOrigin](
    Iterable, Iterator, TrivialRegisterPassable
):
    """Iterator for traversing elements of an IntTuple."""

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    """The iterator type for IntTuple iteration.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    comptime Element = IntTuple

    var src: Pointer[IntTuple, Self.origin]
    """Pointer to the source IntTuple being iterated."""

    var idx: Int
    """Current position in the iteration."""

    @always_inline("nodebug")
    def __init__(out self, src: Pointer[IntTuple, Self.origin], idx: Int):
        """Initialize the iterator with a source IntTuple and starting index."""
        self.src = src
        self.idx = idx

    @always_inline("nodebug")
    def __next__(mut self) raises StopIteration -> IntTuple:
        """Get the next element and advance the iterator.

        Raises:
            `StopIteration` when iteration is complete.
        """
        var idx = self.idx
        self.idx += 1
        if idx >= len(self.src[]):
            raise StopIteration()
        return self.src[][idx]

    @always_inline
    def __has_next__(self) -> Bool:
        return self.idx < len(self.src[])

    @always_inline("nodebug")
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline("nodebug")
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var len = len(self.src[]) - self.idx
        return (len, {len})


struct IntTuple(
    Defaultable,
    Equatable,
    ImplicitlyCopyable,
    Intable,
    Iterable,
    Sized,
    Writable,
):
    """A hierarchical, nested tuple of integers with efficient memory management.

    IntTuple provides a flexible data structure for representing multi-dimensional
    shapes, indices, and other nested integer collections. It supports both flat
    and hierarchical representations with efficient memory sharing.

    This structure is fundamental for tensor operations, layout specifications,
    and dimension handling in high-performance computing contexts.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _IntTupleIter[ImmOrigin(iterable_origin)]
    """The iterator type for IntTuple iteration.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    var _store: IntArray
    """The underlying storage for the `IntTuple`.
    Int values are represented with positive numbers.
    Sub-tuples are represented with a negative offset from the current position."""

    comptime MinimumValue = -0xFFFE
    """Minimum allowed value for integers in an `IntTuple`.

    This constant defines the lower bound for integer values that can be stored
    directly in an `IntTuple`. Values below this threshold are reserved for internal
    use to represent structural information like sub-tuple offsets.
    """

    @staticmethod
    @always_inline("nodebug")
    def elements_size(*elements: IntTuple) -> Int:
        """Calculate the total storage size needed for a list of IntTuples.

        Computes the sum of sizes for all elements, accounting for both direct
        integer values and nested sub-tuples.

        Args:
            elements: List of `IntTuple` elements to measure.

        Returns:
            The total storage size required for all elements.
        """
        var size = 0
        for v in elements:
            # the size of the sub tuple plus the element
            size += v.size() + 1
        return size

    @staticmethod
    @always_inline("nodebug")
    def elements_size[
        _origin: ImmOrigin, n: Int
    ](elements: Array[Pointer[IntTuple, _origin], n], idx: Int) -> Int:
        """Calculate the total storage size needed for IntTuples at a specific index.

        Computes the sum of sizes for all elements at the given index in an array
        of `IntTuple` pointers.

        Parameters:
            _origin: Origin tracking for memory safety.
            n: Size of the inline array.

        Args:
            elements: Array of pointers to `IntTuple`s.
            idx: Index to access in each `IntTuple`.

        Returns:
            The total storage size required for all elements at the specified index.
        """
        var size = 0
        for i in range(n):
            # the size of the sub tuple plus the element
            size += elements[i][][idx].size() + 1
        return size

    @always_inline("nodebug")
    def __init__(out self):
        """Initialize an empty IntTuple.

        Creates an `IntTuple` with zero elements, which can be used as a starting
        point for building tuples incrementally with `append` or `extend`.

        Performance:
            - Minimal allocation (just a single element for length).
            - Structure validation performed when assertions are enabled.
        """
        self._store = IntArray(1)
        self._store[0] = 0
        self.validate_structure()

    @always_inline("nodebug")
    def __init__(out self, *, num_elems: Int):
        """Initialize an `IntTuple` with a specified number of uninitialized elements.

        Creates an `IntTuple` with space for the specified number of elements,
        but does not initialize the elements themselves.

        Args:
            num_elems: The number of elements to allocate space for.

        Note:
            Structure validation performed when assertions are enabled.
        """
        self._store = IntArray(num_elems + 1)
        self._store[0] = num_elems
        self.validate_structure()

    @always_inline("nodebug")
    def __init__(out self, *elements: Int):
        """Initialize an `IntTuple` with a variadic list of integers.

        Creates an `IntTuple` containing the provided integer values.

        Args:
            elements: Variable number of integer values to store in the tuple.

        Notes:
            - Pre-allocates exact memory needed for efficiency.
            - Validates that all values are above `MinimumValue`. If any value is
              less than `MinimumValue`, assertion fails with an error message.
            - Structure validation performed when assertions are enabled.
        """
        var size = len(elements)
        self._store = IntArray(size + 1)
        self._store[0] = size
        for i in range(size):
            var value = elements[i]
            debug_assert(
                value >= Self.MinimumValue,
                "IntTuple value must be >= MinimumValue: ",
                value,
            )
            self._store[i + 1] = value

        self.validate_structure()

    @always_inline
    def __init__[*elements: Int](out self):
        """Initialize an `IntTuple` with a list of integers.

        Creates an `IntTuple` containing the provided integer values.

        Parameters:
            elements: List of integer values to store in the tuple.

        Notes:
            - Pre-allocates exact memory needed for efficiency.
            - Validates that all values are above `MinimumValue`. If any value is
              less than `MinimumValue`, assertion fails with an error message.
            - Structure validation performed when assertions are enabled.
        """
        self._store = IntArray(elements.size + 1)
        self._store[0] = elements.size
        for i in range(elements.size):
            var value = elements[i]
            debug_assert(
                value >= Self.MinimumValue,
                "IntTuple value must be >= MinimumValue: ",
                value,
            )
            self._store[i + 1] = value

        self.validate_structure()

    @implicit
    @always_inline
    def __init__(out self, value: Int):
        """Initialize an `IntTuple` with a single integer value.

        Creates an `IntTuple` containing a single integer element.

        Args:
            value: The integer value to store in the tuple.
        """
        # Skip validation during compile-time interpretation since the comparison
        # may involve complex type witness expressions that can't be evaluated.
        if not __is_run_in_comptime_interpreter:
            debug_assert(
                value >= Self.MinimumValue,
                "IntTuple value must be >= MinimumValue: ",
                value,
            )
        self._store = IntArray(2)
        self._store[0] = 1
        self._store[1] = value
        self.validate_structure()

    @always_inline("nodebug")
    def __init__(
        out self, *elements: IntTuple, __list_literal__: NoneType = None
    ):
        """Initialize an `IntTuple` with nested IntTuples.

        Creates a hierarchical `IntTuple` containing the provided `IntTuple` elements,
        preserving their nested structure.

        Args:
            elements: Variable number of `IntTuple` values to store in the tuple.
            __list_literal__: Specifies that this constructor can be used for
              list literals.
        """
        var size = Self.elements_size(*elements)
        self._store = IntArray(size + 1)
        var num_elems = len(elements)
        self._store[0] = num_elems
        var storage = num_elems + 1
        for i in range(num_elems):
            storage = self._insert(i, storage, elements[i])

        self.validate_structure()

    @always_inline("nodebug")
    def __init__(out self, *, var _owned: IntArray):
        """Initialize an `IntTuple` taking the values of an `IntArray`.

        Args:
            _owned: The `IntArray` to use as storage.
        """
        self._store = _owned^

    @always_inline
    def __init__(out self, existing: Self, rng: _StridedRange[.int]):
        """Initialize an `IntTuple` as a slice of an existing `IntTuple`.

        Creates a new `IntTuple` containing only the elements from the existing
        `IntTuple` that are specified by the range.

        Args:
            existing: The source `IntTuple` to slice from.
            rng: The range of indices to include in the new `IntTuple`.

        Notes:

            - Preserves nested structure of elements in the slice.
            - Structure validation performed when assertions are enabled.
        """
        var size = 0
        var len = 0
        for i in rng:
            size += existing[i].size() + 1
            len += 1
        size += 1

        self._store = IntArray(size)
        self._store[0] = len
        var storage = len + 1

        var pos = 0
        for i in rng:
            storage = self._insert(pos, storage, existing[i])
            pos += 1

        self.validate_structure()

    @always_inline("nodebug")
    def __init__[
        IterableType: Iterable
    ](out self, iterable: IterableType) where (
        IterableType.IteratorType[origin_of(iterable)].Element
        == Tuple[IntTuple, IntTuple]
    ):
        """Initialize an `IntTuple` from a zip iterator.

        Creates an `IntTuple` by appending each element from the zip iterator.

        Parameters:
            IterableType: The type of the iterable.

        Args:
            iterable: An iterable containing pairs of elements to append.

        Note:
            This implementation is not optimized and may be improved in future versions.
        """
        # FIXME: massively inefficient
        self = Self()
        for var elem in iterable:
            var z0, z1 = rebind_var[Tuple[IntTuple, IntTuple]](elem^)
            var tup = IntTuple()
            tup.append(z0)
            tup.append(z1)
            self.append(tup)

    @always_inline("nodebug")
    def __init__(out self, *, copy: Self):
        """Initialize by copying an existing `IntTuple`.

        Creates a deep copy of the provided `IntTuple`, copying all its data
        into newly allocated memory.

        Args:
            copy: The `IntTuple` to copy from.

        Note:
            There is a Mojo bug where this method unnecessarily propagates
            the origin of self to the new copy.
        """
        var size = copy.size()
        self._store = IntArray(size)
        self._store.copy_from(0, copy._store, size)

    @always_inline("nodebug")
    def __lt__(self, rhs: IntTuple) -> Bool:
        """Compare two `IntTuple`s lexicographically.

        This function performs element-wise comparison of two `IntTuple`s and determines
        if the first is lexicographically less than the second. It compares corresponding
        elements until it finds a pair where the elements differ.

        Args:
            rhs: The other `IntTuple` to compare.

        Returns:
            True if `self` is lexicographically less than `rhs`, False otherwise.

        Example:

            ```mojo
            from layout.int_tuple import IntTuple

            var tuple1 = IntTuple(1, 2, 3)
            var tuple2 = IntTuple(1, 2, 4)

            var result = tuple1 < tuple2  # Returns True because 3 < 4
            ```
        """
        for z in zip(self, rhs):
            var z0 = Int(z[0])
            var z1 = Int(z[1])
            if z0 == z1:
                continue
            elif z0 < z1:
                return True
            else:
                return False
        return False

    @always_inline("nodebug")
    def owned_copy(self) -> IntTuple:
        """Create a deep copy of this `IntTuple` with its own memory ownership.

        This method creates a completely independent copy of the `IntTuple` with
        newly allocated memory. Unlike copy init, this method can be called
        on an existing instance to create a separate copy.

        Returns:
            A new `IntTuple` containing the same data as this one but with
            independent memory ownership.

        Example:

            ```mojo
            from layout import IntTuple

            var original = IntTuple(1, 2, 3)
            var copy = original.owned_copy()
            # Modifying copy will not affect original
            ```
        """
        var copy = IntTuple()
        var size = self.size()
        copy._store = IntArray(size)
        copy._store.copy_from(0, self._store, size)
        return copy

    # FIXME: this needs a better name and optimization
    @always_inline
    def replace_entry(self, idx: Int, value: IntTuple) -> IntTuple:
        """Replace an entry in the tuple with another `IntTuple`.

        Creates a new `IntTuple` with the element at the specified index replaced
        by the provided `IntTuple`.

        Args:
            idx: The index of the element to replace.
            value: The `IntTuple` to insert at the specified index.

        Returns:
            A new `IntTuple` with the replacement applied.

        Note:
            If the index is out of bounds, assertion fails with an error message.
        """
        debug_assert(
            idx >= 0 and idx < len(self),
            "IntTuple index out of bounds: ",
            idx,
            " (length: ",
            len(self),
            ")",
        )

        var result = IntTuple()
        for i in range(len(self)):
            if i != idx:
                result.append(self[i])
            else:
                result.append(value)
        return result

    @always_inline("nodebug")
    def replace_entry(mut self, idx: Int, *, int_value: Int):
        """Replace an integer value at the specified index in-place.

        Directly modifies the tuple by replacing the integer value at the given index.
        This is more efficient than creating a new tuple when only a single value
        needs to be changed.

        Args:
            idx: The index of the element to replace.
            int_value: The integer value to insert at the specified index.

        Note:
            If the index is out of bounds, assertion fails with an error message.
        """
        debug_assert(
            idx >= 0 and idx < len(self),
            "IntTuple index out of bounds: ",
            idx,
            " (length: ",
            len(self),
            ")",
        )

        self._store[idx + 1] = int_value

    def count_values(self) -> Int:
        """Count the total number of integer values in this tuple hierarchy.

        Recursively traverses the nested tuple structure and counts all integer values.
        This is useful for determining the size needed for flattened representations.

        Returns:
            The total count of integer values in this tuple and all nested tuples.

        Note:
            For a flat tuple, this will return the same value as `len(self)`.
            For nested tuples, it counts all leaf integer values.
        """
        var count = 0
        for i in range(len(self)):
            if self.is_value(i):
                count += 1
            else:
                count += self[i].count_values()
        return count

    def _fill(mut self, src: IntTuple, var i: Int = 1) -> Int:
        for j in range(len(src)):
            if src.is_value(j):
                self._store[i] = src.value(j)
                i += 1
            else:
                i = self._fill(src[j], i)
        return i

    @always_inline("nodebug")
    def flatten(self) -> IntTuple:
        """Flatten a nested `IntTuple` into a single-level `IntTuple`.

        This function converts a hierarchical `IntTuple` structure into a flat
        sequence of integer values, preserving the order of elements.

        Returns:
            A new `IntTuple` containing all integer values in a flat structure.
        """
        if len(self) == 0 or self.is_value():
            return rebind[IntTuple](self)

        var result = IntTuple(num_elems=self.count_values())
        _ = result._fill(self)
        return result

    def product_flatten(self) -> IntTuple:
        """Coalesces a nested `IntTuple` into a single-level `IntTuple`, by multiplying all the
        values together.

        Returns:
            A new `IntTuple` containing the products of each top level tuple, in a flat structure.
        """

        var rank = len(self)

        var tup = IntTuple(num_elems=rank)

        for i in range(rank):
            var product = product(self[i])
            tup.replace_entry(i, int_value=product)

        return tup

    def all_known(self) -> Bool:
        """Check if all values in this tuple hierarchy are known (not `UNKNOWN_VALUE`).

        Recursively traverses the nested tuple structure and checks if any value
        is equal to `UNKNOWN_VALUE`.

        Returns:
            True if all values in this tuple and nested tuples are known,
            False if any value is `UNKNOWN_VALUE`.
        """
        for i in range(len(self)):
            if self.is_tuple(i):
                if not self[i].all_known():
                    return False
            elif self.value(i) == UNKNOWN_VALUE:
                return False
        return True

    def all_known[start: Int, end: Int](self) -> Bool:
        """Check if all values in this tuple hierarchy are known (not `UNKNOWN_VALUE`).

        Recursively traverses the nested tuple structure and checks if any value
        is equal to `UNKNOWN_VALUE`.

        Parameters:
            start: The starting index (inclusive) for the range to check.
            end: The ending index (exclusive) for the range to check.

        Returns:
            True if all values in this tuple and nested tuples are known,
            False if any value is `UNKNOWN_VALUE`.
        """
        for i in range(start, end):
            if self.is_tuple(i):
                if not self[i].all_known():
                    return False
            elif self.value(i) == UNKNOWN_VALUE:
                return False
        return True

    @always_inline
    def append(mut self, *elements: IntTuple):
        """Append one or more `IntTuple` elements to this tuple.

        This method modifies the tuple in-place by adding the provided elements
        to the end of the tuple. It handles both value tuples and nested tuples.

        Args:
            elements: Variable number of `IntTuple` objects to append to this tuple.

        Notes:

            - This operation requires reallocating the underlying `IntArray` storage to accommodate
            the new elements, which may impact performance for large tuples.
        """
        assert (
            self._store.owning()
        ), "Can't modify a non-owning IntTuple (sub-tuple reference)"

        if len(elements) == 0:
            return

        var old_len = len(self)
        var old_size = self.size()
        var new_size = old_size + Self.elements_size(*elements)
        var new_len = old_len + len(elements)
        var new_store = IntArray(new_size)
        new_store[0] = new_len
        var len_delta = new_len - old_len

        # update old offsets
        for i in range(old_len):
            new_store[i + 1] = (
                self.value(i) if self.is_value(i) else self._store[i + 1]
                - len_delta
            )

        # copy old data
        var new_offset = new_len + 1
        var old_data_size = old_size - old_len - 1
        new_store.copy_from(new_offset, self._store, old_len + 1, old_data_size)

        # append elements
        var storage = new_len + 1 + old_data_size
        for i in range(len(elements)):
            storage = Self._insert(new_store, i + old_len, storage, elements[i])

        # Update store data
        self._store = new_store
        self.validate_structure()

    @always_inline
    def extend(mut self, tuple: IntTuple):
        """
        Extends this tuple by appending all elements from another tuple.

        This method modifies the tuple in-place by adding all elements from the provided
        tuple to the end of this tuple. It efficiently handles both value elements and
        nested tuples.

        Args:
            tuple: The `IntTuple` whose elements will be appended to this tuple.

        Notes:

            - This operation requires reallocating the underlying `IntArray` storage
              to accommodate the new elements, which may impact performance for large tuples.
            - If the input tuple is empty, this method returns without making any changes.
        """
        assert (
            self._store.owning()
        ), "Can't modify a non-owning IntTuple (sub-tuple reference)"

        if len(tuple) == 0:
            return

        var old_len = len(self)
        var old_size = self.size()
        var new_size = old_size + tuple.size() - 1  # FIXME: Yuck
        var new_len = old_len + len(tuple)
        var new_store = IntArray(new_size)
        new_store[0] = new_len
        var storage = new_len + 1

        for i in range(old_len):
            if self.is_value(i):
                new_store[i + 1] = self.value(i)
            else:
                storage = Self._insert(new_store, i, storage, self[i])

        for i in range(len(tuple)):
            if tuple.is_value(i):
                new_store[old_len + i + 1] = tuple.value(i)
            else:
                storage = Self._insert(
                    new_store, i + old_len, storage, tuple[i]
                )

        # Update store data
        self._store = new_store
        self.validate_structure()

    @always_inline
    @staticmethod
    def _insert(
        mut store: IntArray, idx: Int, storage: Int, element: IntTuple
    ) -> Int:
        # Negative offset from current position.
        store[idx + 1] = idx + 1 - storage + Self.MinimumValue
        var size = element.size()
        store.copy_from(storage, element._store, size)
        return storage + size

    @always_inline("nodebug")
    def _insert(mut self, idx: Int, storage: Int, element: IntTuple) -> Int:
        return Self._insert(self._store, idx, storage, element)

    @always_inline("nodebug")
    def size(self) -> Int:
        """
        Returns the total size of the `IntTuple` in memory.

        For owning tuples, returns the size of the underlying `IntArray`.

        Returns:
            The total size in memory units.
        """
        if self._store.owning():
            return self._store.size()
        return Self.tuple_size(self._store)

    @staticmethod
    def tuple_size(data: IntArray) -> Int:
        """
        Recursively calculates the size of a tuple represented by an `IntArray`.

        This method traverses the tuple structure, accounting for both direct values
        and nested sub-tuples to compute the total memory footprint.

        Args:
            data: `IntArray` containing the tuple data.

        Returns:
            The total size of the tuple in memory units.
        """
        return Self._calculate_tuple_size(data, 0)

    @staticmethod
    def _calculate_tuple_size(data: IntArray, offset: Int) -> Int:
        """
        Helper method to calculate the size of a tuple at a given offset without copying.

        Args:
            data: The IntArray containing the tuple data.
            offset: The offset where the tuple starts.

        Returns:
            The size of the tuple starting at the given offset.
        """
        var len = data[offset]
        var size = 1 + len  # Header + all element slots

        # Now add the sizes of nested tuple data
        for i in range(len):
            var val = data[offset + i + 1]
            if val < Self.MinimumValue:
                # For nested tuples, also add the size of the nested tuple data
                # Formula: sub_offset = (offset + i + 1) - (val - MinimumValue)
                var sub_offset = offset + i + 1 - (val - Self.MinimumValue)
                # Ensure sub_offset is valid
                if sub_offset < 0 or sub_offset >= data.size():
                    continue
                var nested_size = Self._calculate_tuple_size(data, sub_offset)
                size += nested_size
        return size

    def validate_structure(self):
        """
        Validates the internal structure of the `IntTuple`.

        Ensures that the actual size of the underlying data matches the computed size
        based on the tuple's structure. This helps detect memory corruption or
        implementation errors.

        Assertion fails with an error message if validation fails.
        """
        # Basic structure validation: ensure size is at least header + elements
        var len = self._store[0]
        debug_assert(
            self._store.size() >= len + 1,
            "Invalid IntTuple structure: size ",
            self._store.size(),
            " is less than required ",
            len + 1,
        )

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """
        Returns the number of elements in the `IntTuple`.

        This is the logical length of the tuple, not its memory size.

        Returns:
            The number of elements in the tuple.
        """
        return self._store[0]

    @always_inline("nodebug")
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """
        Returns an iterator over the elements of the `IntTuple`.

        This enables iteration through the tuple using for-loops.

        Returns:
            An iterator object for this `IntTuple`.
        """
        return _IntTupleIter(Pointer(to=self), 0)

    @always_inline
    def __getitem__(self, idx: IntLiteral) -> IntTuple:
        """Gets the element at the given index.

        Args:
            idx: The index of the element.

        Returns:
            An `IntTuple` containing either a single value or a sub-tuple.
        """
        comptime assert (
            IntLiteral[idx.value]() >= 0
        ), "negative indexing is not supported, use e.g. `x[len(x) - 1]`"
        # This avoids an interpreter unsafe_memcpy error
        if not __is_run_in_comptime_interpreter:
            check_bounds(idx, len(self))
        return self._unchecked_get(Int(idx))

    @always_inline
    def __getitem__(self, idx: Int) -> IntTuple:
        """
        Retrieves an element at the specified index from the `IntTuple`.

        Args:
            idx: The index of the element to retrieve.

        Returns:
            An `IntTuple` containing either a single value or a sub-tuple.
        """
        # This avoids an interpreter unsafe_memcpy error
        if not __is_run_in_comptime_interpreter:
            check_bounds(idx, len(self))
        return self._unchecked_get(idx)

    @always_inline
    def _unchecked_get(self, idx: Int) -> IntTuple:
        # The int value offset to the tuple
        var val = self._store[idx + 1]
        if val >= Self.MinimumValue:
            # Return the Int value
            return IntTuple(val)
        else:
            # Return the sub-tuple with a deep copy
            var offset = idx + 1 - (val - Self.MinimumValue)
            var sub_size = Self._calculate_tuple_size(self._store, offset)
            var sub_data = IntArray(sub_size)
            sub_data.copy_from(0, self._store, offset, sub_size)
            return IntTuple(_owned=sub_data)

    @always_inline("nodebug")
    def __getitem__(self, span: Slice) -> Self:
        """
        Retrieves a slice of elements from the `IntTuple`.

        Creates a new `IntTuple` containing the elements specified by the slice.

        Args:
            span: A slice object specifying the range of elements to retrieve.

        Returns:
            A new `IntTuple` containing the specified elements.
        """
        var start, end, step = span.indices(len(self))
        return Self(self, range(start, end, step))

    @always_inline("nodebug")
    def is_value(self) -> Bool:
        """
        Determines if this `IntTuple` represents a single value rather than a tuple.

        Returns:
            True if this `IntTuple` contains exactly one element that is a value,
            False otherwise.
        """
        return len(self) == 1 and self._store[1] >= Self.MinimumValue

    @always_inline("nodebug")
    def is_tuple(self) -> Bool:
        """
        Determines if this `IntTuple` represents a tuple rather than a single value.

        Returns:
            True if this `IntTuple` is a tuple (not a single value), False otherwise.
        """
        return not self.is_value()

    @always_inline("nodebug")
    def value(self) -> Int:
        """
        Retrieves the value of this `IntTuple` if it represents a single value.

        This method should only be called if `is_value()` returns True.

        Returns:
            The integer value stored in this `IntTuple`.
        """
        return self._store[1]

    @always_inline("nodebug")
    def is_value(self, i: Int) -> Bool:
        """
        Determines if the element at the specified index is a value rather than a tuple.

        Args:
            i: The index of the element to check.

        Returns:
            True if the element at index i is a value, False if it's a tuple.

        Notes:
            If index is out of bounds, assertion fails with an error message.
        """
        debug_assert(
            i >= 0 and i < len(self),
            "IntTuple index out of bounds: ",
            i,
            " (length: ",
            len(self),
            ")",
        )

        return self._store[i + 1] >= Self.MinimumValue

    @always_inline("nodebug")
    def is_tuple(self, i: Int) -> Bool:
        """
        Determines if the element at the specified index is a tuple rather than a value.

        Args:
            i: The index of the element to check.

        Returns:
            True if the element at index i is a tuple, False if it's a value.

        Notes:
            This is the complement of is_value(i).
        """
        return not self.is_value(i)

    @always_inline("nodebug")
    def value(self, i: Int) -> Int:
        """
        Retrieves the value of the element at the specified index.

        This method should only be called if `is_value(i)` returns True.

        Args:
            i: The index of the element to retrieve.

        Returns:
            The integer value stored at the specified index.

        Notes:
            If the element is not a value, the behavior is undefined.
        """
        var val = self._store[i + 1]
        if val >= Self.MinimumValue:
            # Direct value
            return val
        else:
            # It's a tuple reference - extract the value from the sub-tuple
            # This handles the case where elements are stored as single-element tuples
            var offset = i + 1 - (val - Self.MinimumValue)
            # For a single-element tuple containing an int, the value is at offset+1
            return self._store[offset + 1]

    @always_inline("nodebug")
    def tuple(ref self) -> ref[self] Self:
        """
        Returns a reference to this `IntTuple` as a tuple.

        Returns:
            A reference to this `IntTuple` to avoid unnecessary copying.

        Notes:
            This method is used to access the current `IntTuple` as a tuple
            without creating a copy of the data.
        """
        # Avoid making gratuitous copies
        return self

    def write_to(self, mut writer: Some[Writer]):
        """
        Writes a string representation of this `IntTuple` to the provided writer.

        Args:
            writer: The writer to output the string representation to.

        Notes:
            For single values, writes just the value.
            For tuples, writes a comma-separated list of elements enclosed in parentheses.
        """
        if self.is_value():
            writer.write(self.value())
        else:
            writer.write("(")
            var len = len(self)
            for i in range(len):
                if self.is_value(i):
                    writer.write(self.value(i))
                else:
                    writer.write(String(self[i]))
                if i < len - 1:
                    writer.write(", ")
            writer.write(")")

    def write_repr_to(self, mut writer: Some[Writer]):
        """
        Writes a string representation of this `IntTuple` to the provided writer.

        Args:
            writer: The writer to output the string representation to.

        Notes:
            For single values, writes just the value.
            For tuples, writes a comma-separated list of elements enclosed in parentheses.
        """
        self.write_to(writer)

    @staticmethod
    def is_equal(a: IntTuple, b: IntTuple) -> Bool:
        """
        Compares two `IntTuple`s for equality.

        Args:
            a: The first `IntTuple` to compare.
            b: The second `IntTuple` to compare.

        Returns:
            True if the `IntTuple`s are equal in structure and values, False otherwise.

        Notes:
            Handles nested tuples and special cases where a single-element tuple
            is equivalent to its contained value.
        """
        if len(a) != len(b):
            return False

        for i in range(len(a)):
            if a.is_value(i) and b.is_value(i):
                if a.value(i) != b.value(i):
                    return False
            else:
                if a.is_tuple(i) and b.is_tuple(i):
                    if not Self.is_equal(a[i], b[i]):
                        return False
                elif a.is_tuple(i) and len(a[i]) == 1:
                    if a[i].value() != b.value(i):
                        return False
                elif len(b[i]) == 1:  # b.is_tuple(i)
                    if a.value(i) != b[i].value():
                        return False
                else:
                    return False

        return True

    @always_inline("nodebug")
    def __eq__(self, other: Self) -> Bool:
        """
        Equality operator for `IntTuple`.

        Args:
            other: The `IntTuple` to compare with.

        Returns:
            True if the `IntTuple`s are equal, False otherwise.
        """
        return Self.is_equal(self, other)

    @always_inline("nodebug")
    def __ne__(self, other: Self) -> Bool:
        """
        Inequality operator for `IntTuple`.

        Args:
            other: The `IntTuple` to compare with.

        Returns:
            True if the `IntTuple`s are not equal, False otherwise.
        """
        return not Self.is_equal(self, other)

    @always_inline("nodebug")
    def __int__(self) -> Int:
        """
        Converts this `IntTuple` to an integer.

        This method should only be called if `is_value()` returns True.

        Returns:
            The integer value stored in this `IntTuple`.

        Aborts:
            If the `IntTuple` is not a single value.
        """
        if self.is_tuple():
            abort("IntTuple is not a single value. Cannot convert to Int.")

        return self.value()


@always_inline("nodebug")
def signum(a: Int) -> Int:
    """Calculate the sign of an integer.

    This function determines the sign of the input integer and returns a corresponding
    indicator value.

    Args:
        a: The integer value to determine the sign of.

    Returns:
        1 if `a` > 0, -1 if `a` < 0, 0 if `a` == 0.

    Example:

        ```mojo
        from layout.int_tuple import signum

        var result1 = signum(5)    # Returns 1
        var result2 = signum(-10)  # Returns -1
        var result3 = signum(0)    # Returns 0
        ```
    """
    return 1 if (a > 0) else (-1 if (a < 0) else 0)


@always_inline("nodebug")
def is_int(t: IntTuple) -> Bool:
    """Check if an `IntTuple` represents a single integer value.

    This function determines whether the given `IntTuple` contains a single integer value
    rather than a nested tuple structure.

    Args:
        t: The `IntTuple` to check.

    Returns:
        True if the `IntTuple` contains a single integer value,
        False if it's a nested tuple.

    Example:

        ```mojo
        from layout.int_tuple import is_int, IntTuple

        var single_value = IntTuple(5)
        var nested_tuple = IntTuple(1, 2, 3)

        var result1 = is_int(single_value)  # Returns True
        var result2 = is_int(nested_tuple)  # Returns False
        ```
    """
    return t.is_value()


@always_inline("nodebug")
def is_tuple(t: IntTuple) -> Bool:
    """Check if an `IntTuple` represents a nested tuple.

    This function determines whether the given `IntTuple` contains nested elements
    rather than a single integer value. It is the complement of the `is_int` function.

    Args:
        t: The `IntTuple` to check.

    Returns:
        True if the `IntTuple` contains nested elements,
        False if it's a single integer value.

    Example:

        ```mojo
        from layout.int_tuple import is_tuple, IntTuple

        var single_value = IntTuple(5)
        var nested_tuple = IntTuple(1, 2, 3)

        var result1 = is_tuple(single_value)  # Returns False
        var result2 = is_tuple(nested_tuple)  # Returns True
        ```
    """
    return t.is_tuple()


# Python-style reduce


def reduce[
    FuncType: def(a: Int, b: IntTuple) -> Int
](t: IntTuple, initializer: Int, reducer: FuncType) -> Int:
    """Apply a reduction function to an `IntTuple` with an initial value.

    This function iterates through each element of the `IntTuple` and applies
    the provided reduction function cumulatively, starting with the initializer.

    Parameters:
        FuncType: The reduction function type.

    Args:
        t: The `IntTuple` to reduce.
        initializer: The initial value for the reduction operation.
        reducer: A function that combines the accumulated result with the next
            element.

    Returns:
        The final accumulated result after applying the reduction function
        to all elements in the `IntTuple`.
    """
    if is_int(t):
        return Int(t)

    var result = initializer
    for e in t:
        result = reducer(result, e)
    return result


# IntTuple operations


@always_inline("nodebug")
def is_flat(t: IntTuple) -> Bool:
    """Check if an `IntTuple` is flat.

    This function checks if the `IntTuple` is flat, meaning it has no nested
    elements.

    Args:
        t: The `IntTuple` to check.

    Returns:
        True if the `IntTuple` is flat, False otherwise.
    """
    for i in range(len(t)):
        if is_tuple(t[i]):
            return False
    return True


@always_inline("nodebug")
def flatten(t: IntTuple) -> IntTuple:
    """Flatten a nested `IntTuple` into a single-level `IntTuple`.

    This function converts a hierarchical `IntTuple` structure into a flat
    sequence of integer values, preserving the order of elements.

    Args:
        t: The nested `IntTuple` to flatten.

    Returns:
        A new `IntTuple` containing all integer values in a flat structure.
    """
    return t.flatten()


def to_nest(nested: IntTuple, flat: IntTuple) -> IntTuple:
    """Nests a flat `IntTuple` according to the structure of a nested `IntTuple`.

    This function reshapes a flat sequence of values into a hierarchical structure
    that matches the pattern of a template nested `IntTuple`.

    Args:
        nested: The template `IntTuple` defining the desired structure.
        flat: The flat `IntTuple` containing the values to be nested.

    Returns:
        A new `IntTuple` with the values from flat arranged in the structure of nested.

    Example:

        ```mojo
        from layout import IntTuple
        from layout.int_tuple import to_nest

        var result = to_nest(IntTuple(2, IntTuple(3, 4), 5), IntTuple(1, 2, 3, 4))
        # returns IntTuple(1, (2, 3), 4)
        ```
    """
    var result = IntTuple()
    var flat_idx = 0

    for i in range(len(nested)):
        if is_int(nested[i]):
            result.append(flat[flat_idx])
            flat_idx += 1
        else:
            var sub_size = depth(nested[i]) + 1
            result.append(
                to_nest(nested[i], flat[flat_idx : flat_idx + sub_size])
            )
            flat_idx += sub_size

    return result


def _to_unknown(mut t: IntTuple):
    """Recursively replace all values in a tuple with UNKNOWN_VALUE in place.

    This function modifies the tuple's internal storage directly to avoid
    creating unnecessary copies when dealing with nested structures.

    Args:
        t: The tuple to modify in place.
    """
    _to_unknown_impl(t._store, 0)


def _to_unknown_impl(mut data: IntArray, offset: Int):
    """Helper function to recursively replace values with UNKNOWN_VALUE.

    Args:
        data: The IntArray containing tuple data.
        offset: The offset where the current tuple starts.
    """
    var num_elems = data[offset]
    for i in range(num_elems):
        var idx = offset + i + 1
        var val = data[idx]
        if val >= IntTuple.MinimumValue:
            data[idx] = UNKNOWN_VALUE
        else:
            var sub_offset = idx - (val - IntTuple.MinimumValue)
            _to_unknown_impl(data, sub_offset)


# Create a IntTuple with same structure but filled by UNKNOWN_VALUE.
@always_inline("nodebug")
def to_unknown(t: IntTuple) -> IntTuple:
    """Create an `IntTuple` with the same structure but filled with `UNKNOWN_VALUE`.

    This function preserves the hierarchical structure of the input `IntTuple`
    but replaces all integer values with `UNKNOWN_VALUE`.

    Args:
        t: The template `IntTuple` defining the structure.

    Returns:
        A new `IntTuple` with the same structure as t but with all values
        replaced by `UNKNOWN_VALUE`.
    """
    var res = t.owned_copy()
    _to_unknown(res)
    return res


@always_inline
def _merge[
    cmp: def(IntTuple, IntTuple) thin -> Bool,
](left: IntTuple, right: IntTuple) -> IntTuple:
    var result = IntTuple()
    var i = 0
    var j = 0

    while i < len(left) and j < len(right):
        if cmp(left[i], right[j]):
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])

    return result


def sorted[
    cmp: def(IntTuple, IntTuple) thin -> Bool = IntTuple.__lt__,
](tuple: IntTuple) -> IntTuple:
    """Sort an IntTuple using the provided comparison function.

    This function implements a merge sort algorithm to efficiently sort
    the elements of an IntTuple. The sorting is stable and has `O(n log n)`
    time complexity.

    Args:
        tuple: The `IntTuple` to be sorted.

    Parameters:
        cmp: A comparison function that takes two `IntTuple` elements and
             returns True if the first should come before the second.
             Defaults to the `lt` function which performs lexicographical ordering.

    Returns:
        A new `IntTuple` containing the same elements as the input but sorted
        according to the comparison function.
    """
    if len(tuple) <= 1:
        return tuple.owned_copy()  # Avoid propagating a's origin
    var mid = len(tuple) // 2
    return _merge[cmp](sorted[cmp](tuple[:mid]), sorted[cmp](tuple[mid:]))


@always_inline("nodebug")
def sum(t: IntTuple) -> Int:
    """Calculate the sum of all values in an `IntTuple`.

    This function recursively computes the sum of all integer values
    in a potentially nested `IntTuple` structure.

    Args:
        t: The `IntTuple` to sum.

    Returns:
        The sum of all integer values, or `UNKNOWN_VALUE` if any value
        in the tuple is `UNKNOWN_VALUE`.
    """

    @always_inline
    def reducer(a: Int, b: IntTuple) -> Int:
        return UNKNOWN_VALUE if a == UNKNOWN_VALUE else a + (
            Int(b) if is_int(b) else sum(b)
        )

    return reduce(t, 0, reducer)


@always_inline("nodebug")
def product(t: IntTuple) -> Int:
    """Calculate the product of all values in an `IntTuple`.

    This function recursively computes the product of all integer values
    in a potentially nested `IntTuple` structure.

    Args:
        t: The `IntTuple` to multiply.

    Returns:
        The product of all integer values, or `UNKNOWN_VALUE` if any value
        in the tuple is `UNKNOWN_VALUE`.
    """

    @always_inline
    def reducer(a: Int, b: IntTuple) -> Int:
        return UNKNOWN_VALUE if a == UNKNOWN_VALUE else a * (
            Int(b) if is_int(b) else product(b)
        )

    return reduce(t, 1, reducer)


# TODO: Can't call this `max` otherwise the compiler incorrectly
# fails to recurse when calling this local function.
@always_inline("nodebug")
def tuple_max(t: IntTuple) -> Int:
    """Calculate the maximum value in an `IntTuple`.

    This function recursively finds the maximum integer value
    in a potentially nested `IntTuple` structure.

    Args:
        t: The `IntTuple` to search.

    Returns:
        The maximum integer value found in the tuple.
    """

    @always_inline
    def reducer(a: Int, b: IntTuple) -> Int:
        return max(a, Int(b) if is_int(b) else tuple_max(b))

    comptime int_min_val = 0
    return reduce(t, int_min_val, reducer)


def apply[FuncType: def(Int) -> Int](t: IntTuple, func: FuncType) -> IntTuple:
    """Apply a function to each integer value in an `IntTuple`.

    This function recursively applies the given function to each integer value
    in a potentially nested `IntTuple` structure, preserving the structure.

    Parameters:
        FuncType: The transform function type.

    Args:
        t: The `IntTuple` to transform.
        func: Function to apply to each integer value.

    Returns:
        A new `IntTuple` with the same structure but with each integer value
        transformed by the function.
    """
    if is_int(t):
        return func(Int(t))
    var res = IntTuple()
    for e in t:
        res.append(apply(e, func))
    return res


def shallow_apply[func: def(IntTuple) thin -> Int](t: IntTuple) -> IntTuple:
    """Apply a function to each top-level element of an `IntTuple`.

    Unlike `apply()`, this function only operates on the immediate children
    of the input tuple without recursing into nested tuples.

    Parameters:
        func: Function that takes an `IntTuple` and returns an `Int`.

    Args:
        t: The `IntTuple` whose elements will be transformed.

    Returns:
        A new `IntTuple` with the function applied to each top-level element.
    """
    var res = IntTuple()
    for e in t:
        res.append(func(e))
    return res


@always_inline("nodebug")
def apply_zip[
    func: def(IntTuple, IntTuple) thin -> IntTuple
](t1: IntTuple, t2: IntTuple) -> IntTuple:
    """Apply a function to pairs of elements from two `IntTuple`s.

    This function zips two `IntTuple`s together and applies the given function
    to each pair of elements, creating a new `IntTuple` with the results.

    Parameters:
        func: Function that takes two `IntTuple`s and returns an `IntTuple`.

    Args:
        t1: First `IntTuple`.
        t2: Second `IntTuple`.

    Returns:
        A new `IntTuple` containing the results of applying func to each pair.
    """
    var r = IntTuple()
    for z in zip(t1, t2):
        r.append(func(z[0], z[1]))
    return r


@always_inline("nodebug")
def apply_zip[
    FuncType: def(IntTuple, IntTuple) -> IntTuple
](t1: IntTuple, t2: IntTuple, func: FuncType) -> IntTuple:
    """Apply a function to pairs of elements from two `IntTuple`s.

    This overload takes the function as a runtime value so unified closures can
    capture from their environment.

    Parameters:
        FuncType: The zip function type.

    Args:
        t1: First `IntTuple`.
        t2: Second `IntTuple`.
        func: Function that takes two `IntTuple`s and returns an `IntTuple`.

    Returns:
        A new `IntTuple` containing the results of applying func to each pair.
    """
    var r = IntTuple()
    for z in zip(t1, t2):
        r.append(func(z[0], z[1]))
    return r


@always_inline("nodebug")
def apply_zip[
    func: def(IntTuple, IntTuple, IntTuple) thin -> IntTuple
](t1: IntTuple, t2: IntTuple, t3: IntTuple) -> IntTuple:
    """Apply a function to triplets of elements from three `IntTuple`s.

    This function zips three `IntTuple`s together and applies the given function
    to each triplet of elements, creating a new `IntTuple` with the results.

    Parameters:
        func: Function that takes three `IntTuple`s and returns an `IntTuple`.

    Args:
        t1: First `IntTuple`.
        t2: Second `IntTuple`.
        t3: Third `IntTuple`.

    Returns:
        A new `IntTuple` containing the results of applying func to each triplet.
    """
    var r = IntTuple()
    for z in zip(t1, t2, t3):
        r.append(func(z[0], z[1], z[2]))
    return r


@always_inline("nodebug")
def apply_zip[
    FuncType: def(IntTuple, IntTuple, IntTuple) -> IntTuple
](t1: IntTuple, t2: IntTuple, t3: IntTuple, func: FuncType) -> IntTuple:
    """Apply a function to triplets of elements from three `IntTuple`s.

    This overload takes the function as a runtime value so unified closures can
    capture from their environment.

    Parameters:
        FuncType: The zip function type.

    Args:
        t1: First `IntTuple`.
        t2: Second `IntTuple`.
        t3: Third `IntTuple`.
        func: Function that takes three `IntTuple`s and returns an `IntTuple`.

    Returns:
        A new `IntTuple` containing the results of applying func to each triplet.
    """
    var r = IntTuple()
    for z in zip(t1, t2, t3):
        r.append(func(z[0], z[1], z[2]))
    return r


def tuple_min(a: IntTuple, b: IntTuple) -> IntTuple:
    """Compute the element-wise minimum of two `IntTuple`s.

    This function compares corresponding elements of two `IntTuple`s and
    returns a new `IntTuple` containing the minimum value at each position.

    Args:
        a: First `IntTuple`.
        b: Second `IntTuple`.

    Returns:
        A new `IntTuple` with each element being the minimum of the corresponding
        elements in a and b.

    Aborts:
        If the input tuples have different lengths.

    Note:
        If either input contains `UNKNOWN_VALUE`, the result will be `UNKNOWN_VALUE`.
    """
    debug_assert(
        len(a) == len(b),
        "Tuple sizes don't match: ",
        len(a),
        " != ",
        len(b),
    )
    if is_int(a):
        if UNKNOWN_VALUE in (Int(a), Int(b)):
            return UNKNOWN_VALUE
        return min(Int(a), Int(b))
    return apply_zip[tuple_min](a, b)


def inner_product(a: IntTuple, b: IntTuple) -> Int:
    """Compute the inner product of two `IntTuple`s.

    For flat tuples, this is the sum of element-wise products.
    For nested tuples, the function recurses into corresponding nested elements.

    Args:
        a: First `IntTuple`.
        b: Second `IntTuple`.

    Returns:
        The inner product as an `Int`.

    Note:
        If the input tuples have different lengths, assertion fails.
    """
    debug_assert(
        len(a) == len(b),
        "Tuple sizes don't match: ",
        len(a),
        " != ",
        len(b),
    )
    if is_int(a):
        return Int(a) * Int(b)
    var r: Int = 0
    for z in zip(a, b):
        r += inner_product(z[0], z[1])
    return r


@always_inline("nodebug")
def abs(t: IntTuple) -> IntTuple:
    """Compute the absolute value of each element in an `IntTuple`.

    This function applies the absolute value operation to each integer
    in a potentially nested `IntTuple` structure.

    Args:
        t: The `IntTuple` to transform.

    Returns:
        A new `IntTuple` with the same structure but with absolute values.
    """

    def int_abs(x: Int) -> Int:
        return x.__abs__()

    return apply(t, int_abs)


@always_inline("nodebug")
def product_each(t: IntTuple) -> IntTuple:
    """Compute the product of elements in each sub-tuple of an `IntTuple`.

    For each immediate child of the input tuple, this function computes
    the product of all elements within that child.

    Args:
        t: The `IntTuple` containing sub-tuples.

    Returns:
        A new `IntTuple` where each element is the product of the corresponding
        sub-tuple in the input.
    """
    return shallow_apply[product](t)


# Multiply lhs tuple elements by rhs
def _mul(mut lhs: IntTuple, rhs: Int, offset: Int = 0):
    var num_elems = lhs._store[offset]
    for i in range(num_elems):
        var idx = offset + i + 1
        var val = lhs._store[idx]
        if val >= IntTuple.MinimumValue:
            if UNKNOWN_VALUE in (val, rhs):
                lhs._store[idx] = UNKNOWN_VALUE
            else:
                lhs._store[idx] *= rhs
        else:
            var sub_offset = idx - (val - IntTuple.MinimumValue)
            _mul(lhs, rhs, sub_offset)


@always_inline("nodebug")
def mul(lhs: IntTuple, rhs: Int) -> IntTuple:
    """Multiply each element in an `IntTuple` by a scalar value.

    This function creates a new `IntTuple` where each element (at any nesting level)
    is multiplied by the provided integer value.

    Args:
        lhs: The `IntTuple` whose elements will be multiplied.
        rhs: The scalar integer to multiply each element by.

    Returns:
        A new `IntTuple` with the same structure as the input but with all
        elements multiplied by the scalar value.
    """
    var res = lhs.owned_copy()
    _mul(res, rhs)
    return res


@always_inline("nodebug")
def size(a: IntTuple) -> Int:
    """Calculate the total size (product of all elements) of an `IntTuple`.

    This function computes the product of all integer values in the `IntTuple`,
    regardless of nesting level.

    Args:
        a: The `IntTuple` whose elements will be multiplied together.

    Returns:
        The product of all elements in the `IntTuple`.
    """
    return product(a)


def congruent(a: IntTuple, b: IntTuple) -> Bool:
    """Test if two `IntTuple`s have the same hierarchical structure.

    This function checks if two `IntTuple`s have identical nesting patterns,
    regardless of the actual integer values they contain.

    Args:
        a: First `IntTuple` to compare.
        b: Second `IntTuple` to compare.

    Returns:
        True if both `IntTuple`s have the same hierarchical structure,
        False otherwise.
    """
    if is_tuple(a) and is_tuple(b):
        if len(a) != len(b):
            return False
        for z in zip(a, b):
            if not congruent(z[0], z[1]):
                return False
        return True
    elif is_int(a) and is_int(b):
        return True
    return False


def apply_predicate[
    predicate: def(IntTuple, IntTuple) thin -> Bool
](a: IntTuple, b: IntTuple) -> Bool:
    """Apply a predicate function recursively to two `IntTuple`s.

    This function traverses two `IntTuple`s with the same structure and applies
    a predicate function to corresponding elements. The predicate is applied
    only to the leaf nodes (integer values).

    Parameters:
        predicate: A function that takes two `IntTuple`s (containing integer values)
                  and returns a boolean result.

    Args:
        a: First `IntTuple` to compare.
        b: Second `IntTuple` to compare.

    Returns:
        True if the predicate returns True for all corresponding elements and
        the structures match, False otherwise.

    Note:
        If the structures of the two `IntTuple`s don't match (different nesting or length),
        the function returns False without applying the predicate.
    """
    if is_tuple(a) and is_tuple(b):
        if len(a) != len(b):
            return False
        for z in zip(a, b):
            if not apply_predicate[predicate](z[0], z[1]):
                return False
        return True
    if is_int(a):
        return predicate(a, b)
    return False


@always_inline("nodebug")
def weakly_congruent(a: IntTuple, b: IntTuple) -> Bool:
    """Test if two IntTuples have similar hierarchical structures.

    This function establishes a partial order relation between IntTuples
    based on their hierarchical structure. It's less strict than congruent.

    Args:
        a: First IntTuple to compare.
        b: Second IntTuple to compare.

    Returns:
        True if a's structure is compatible with b's structure,
        False otherwise.
    """

    return apply_predicate[lambda (a: IntTuple, b: IntTuple) -> Bool: True](
        a, b
    )


@always_inline("nodebug")
def compatible(a: IntTuple, b: IntTuple) -> Bool:
    """Test if two shapes are compatible for tensor operations.

    This function checks if shape A is compatible with shape B, meaning:
    1. The total size of A and B are the same
    2. Any coordinate into A can also be used as a coordinate into B

    Compatible can also be thought of as a partial order on A and B: A <= B.

    Args:
        a: The first `IntTuple` to compare.
        b: The second `IntTuple` to compare.

    Returns:
        True if shape A is compatible with shape B, False otherwise.
    """

    return apply_predicate[
        lambda (a: IntTuple, b: IntTuple) -> Bool: Int(a) == size(b)
    ](a, b)


@always_inline("nodebug")
def weakly_compatible(a: IntTuple, b: IntTuple) -> Bool:
    """Test if shape A is weakly compatible with shape B.

    A shape A is weakly compatible with shape B if there exists a shape C
    congruent to A such that compatible(elem_scale(A,C), B). This establishes
    a partial order relation between shapes where A <= B.

    Specifically, this checks if the size of B is divisible by the size of A,
    which is a necessary condition for weak compatibility.

    Args:
        a: The first `IntTuple` to compare.
        b: The second `IntTuple` to compare.

    Returns:
        True if shape A is weakly compatible with shape B, False otherwise.
    """

    return apply_predicate[
        lambda (a: IntTuple, b: IntTuple) -> Bool: size(b) % Int(a) == 0
    ](a, b)


@always_inline("nodebug")
def prefix_product(a: IntTuple) -> IntTuple:
    """Compute the exclusive prefix product of an `IntTuple`.

    This is a convenience wrapper that initializes the prefix product with 1.

    Args:
        a: The input `IntTuple` to compute the prefix product for.

    Returns:
        A new `IntTuple` containing the exclusive prefix product of the input.
    """
    return prefix_product(a, 1)


@always_inline("nodebug")
def prefix_product(a: IntTuple, init: Int) -> IntTuple:
    """Compute the exclusive prefix product of an `IntTuple` with an initial value.

    This function delegates to the implementation in prefix_product2.

    Args:
        a: The input `IntTuple` to compute the prefix product for.
        init: The initial value(s) for the prefix product, defaults to 1.

    Returns:
        A new `IntTuple` containing the exclusive prefix product of the input.
    """
    # Short-circuit for empty tuple
    if len(a) == 0:
        return IntTuple()
    # Short-circuit for single integer
    if is_int(a):
        return init

    var init_tuple = IntTuple(init)
    return _prefix_product2(a, init_tuple)


def _prefix_product2(a: IntTuple, init: IntTuple) -> IntTuple:
    """Internal implementation of exclusive prefix product computation.

    Handles four cases:
    1. tuple-tuple: Apply recursively element-wise
    2. tuple-int: Apply to each element with updated init
    3. int-tuple: Not allowed (aborts)
    4. int-int: Return the init value

    Note:
        If dimensions of tuple inputs don't match or if int-tuple case is
        encountered, abort() will be called.
    """
    if is_tuple(a):
        if is_tuple(init):  # tuple tuple
            debug_assert(
                len(a) == len(init),
                "Tuple sizes don't match in prefix_product: len(a)=",
                len(a),
                " len(init)=",
                len(init),
            )
            return apply_zip[_prefix_product2](a, init)
        else:  # tuple "int"
            var v_init = Int(init)
            var r = IntTuple()
            for v in a:
                r.append(_prefix_product2(v, v_init))

                var is_unknown = (
                    v.is_value() and Int(v) == UNKNOWN_VALUE
                ) or v_init == UNKNOWN_VALUE

                v_init = UNKNOWN_VALUE if is_unknown else v_init * product(v)
            return r
    else:
        assert not is_tuple(
            init
        ), "Invalid prefix_product: 'int' tuple case not allowed"

        if is_tuple(init):  # "int" tuple
            return IntTuple()
        else:  # "int" "int"
            return init.owned_copy()


def shape_div[check: Bool = False](a: IntTuple, b: IntTuple) -> IntTuple:
    """Performs division operation between shape tuples.

    Handles four cases:
    1. tuple-tuple: Performs shape_div element-wise when dimensions match
    2. tuple-int: Folds the division of b across each element of a
       Example: `shape_div((4,5,6),40)` -> `shape_div((1,5,6),10)` -> `shape_div((1,1,6),2)` -> `(1,1,3)`
    3. int-tuple: Returns `shape_div(a, product(b))`
    4. int-int: Enforces the divisibility condition `a % b == 0 || b % a == 0` when possible
       Returns `a / b` with rounding away from `0` (that is, `1` or `-1` when `a < b`)

    Parameters:
        check: Whether to check for incompatible shapes.

    Args:
        a: The dividend `IntTuple`.
        b: The divisor `IntTuple`.

    Returns:
        A new `IntTuple` containing the result of the division operation

    Notes:
        - When tuple sizes don't match in the tuple-tuple case, `abort()` will be
          called.
        - When values are incompatible (neither divides the other) in the int-int
          case, `abort()` will be called.
    """
    if is_tuple(a):
        if is_tuple(b):  # tuple tuple
            debug_assert(
                len(a) == len(b),
                "Tuple sizes don't match in shape_div: ",
                len(a),
                " != ",
                len(b),
            )
            return apply_zip[shape_div[check]](a, b)
        else:  # tuple "int"
            var vb = Int(b)
            var r = IntTuple()
            for v in a:
                r.append(shape_div[check](v, vb))
                var prod_v = IntTuple(product(v))
                vb = Int(shape_div[check](vb, prod_v))
            return r
    else:
        if is_tuple(b):  # "int" tuple
            var prod_b = IntTuple(product(b))
            return shape_div[check](a, prod_b)
        else:  # "int" "int"
            var va = Int(a)
            var vb = Int(b)

            if va == UNKNOWN_VALUE or vb == UNKNOWN_VALUE:
                return UNKNOWN_VALUE

            comptime if check:
                debug_assert(
                    va % vb == 0 or vb % va == 0,
                    "Incompatible shape values: ",
                    va,
                    " ",
                    vb,
                )

            return va // vb if va % vb == 0 else signum(va * vb)


# idx2crd(i,s) splits an index into a coordinate within Shape
# via a colexicographical enumeration of coordinates in Shape.
# c0 = (idx / 1) % s0
# c1 = (idx / s0) % s1
# c2 = (idx / (s0 * s1)) % s2
# ...
#


@always_inline("nodebug")
def idx2crd(idx: IntTuple, shape: IntTuple) -> IntTuple:
    """
    Converts a linear index to a coordinate tuple within a given shape.

    This function splits an index into a coordinate within a Shape via a
    colexicographical enumeration of coordinates in Shape.

    Args:
        idx: The linear index to convert.
        shape: The shape of the tensor/array.

    Returns:
        A new `IntTuple` containing the coordinates corresponding to the linear index.
    """
    return idx2crd2(idx, shape, IntTuple())


@always_inline("nodebug")
def idx2crd(idx: IntTuple, shape: IntTuple, _stride: IntTuple) -> IntTuple:
    """
    Converts a linear index to a coordinate tuple within a given shape using custom strides.

    Args:
        idx: The linear index to convert.
        shape: The shape of the tensor/array.
        _stride: Custom strides to use for the conversion.

    Returns:
        A new `IntTuple` containing the coordinates corresponding to the linear index.
    """
    return idx2crd2(idx, shape, _stride)


def idx2crd2(
    idx: IntTuple,
    shape: IntTuple,
    _stride: IntTuple,  # = IntTuple()
) -> IntTuple:
    """
    Convert a linear index to coordinates.

    This function handles the actual conversion logic for different input combinations.

    Args:
        idx: The linear index to convert.
        shape: The shape of the tensor/array.
        _stride: Custom strides to use for the conversion. If empty, strides are computed
                 from the shape using prefix_product.

    Returns:
        A new IntTuple containing the coordinates corresponding to the linear index.

    Notes:

        - Handles four cases: tuple-tuple-tuple, tuple-int-int, int-tuple-tuple, and int-int-int.
        - When input shapes don't match, `abort()` will be called.
    """
    var stride: IntTuple
    if len(_stride) == 0:
        stride = prefix_product(shape).owned_copy()
    else:
        stride = _stride.owned_copy()

    if is_tuple(idx):
        if is_tuple(shape):  # tuple tuple tuple
            debug_assert(
                len(idx) == len(shape) and len(idx) == len(stride),
                "idx2crd input shapes mismatch: len(idx)=",
                len(idx),
                " len(shape)=",
                len(shape),
                " len(stride)=",
                len(stride),
            )

            return apply_zip[idx2crd2](idx, shape, stride)
        else:  # tuple "int" "int"
            abort("Illegal inputs")  # Error
    else:
        if is_tuple(shape):  # "int" tuple tuple
            debug_assert(
                len(shape) == len(stride),
                "idx2crd input shapes mismatch: len(shape)=",
                len(shape),
                " len(stride)=",
                len(stride),
            )

            def idx2crd2(shape: IntTuple, stride: IntTuple) {imm} -> IntTuple:
                return idx2crd(idx, shape, stride)

            return apply_zip(shape, stride, idx2crd2)
        else:  # "int" "int" "int"
            return UNKNOWN_VALUE if (
                Int(idx) == UNKNOWN_VALUE
                or Int(stride) == UNKNOWN_VALUE
                or Int(shape) == UNKNOWN_VALUE
            ) else (Int(idx) // Int(stride)) % Int(shape)


@always_inline("nodebug")
def crd2idx(crd: IntTuple, shape: IntTuple) -> Int:
    """
    Map a logical coordinate to a linear index.

    This function converts a multi-dimensional coordinate to a linear index based on the shape.
    It uses default strides computed from the shape.

    Args:
        crd: The coordinate tuple to convert.
        shape: The shape of the tensor/array.

    Returns:
        The linear index corresponding to the coordinate.
    """
    return crd2idx(crd, shape, IntTuple())


def crd2idx(
    crd: IntTuple,
    shape: IntTuple,
    _stride: IntTuple,  # = IntTuple()
) -> Int:
    """
    Map a logical coordinate to a linear index with custom strides.

    This function converts a multi-dimensional coordinate to a linear index based on the shape
    and stride information. If no stride is provided, it computes default strides from the shape.

    The function handles various input combinations:
    - Tuple coordinates with tuple shapes and strides
    - Single integer coordinate with tuple shapes and strides
    - Single integer coordinate with single integer shape and stride

    Args:
        crd: The coordinate(s) to convert, can be a single value or a tuple of coordinates.
        shape: The shape of the tensor/array, can be a single value or a tuple of dimensions.
        _stride: Optional custom strides, defaults to row-major strides if not provided.

    Returns:
        The linear index corresponding to the coordinate.

    Aborts:

        - If coordinate and shape dimensions don't match.
        - If shape and stride dimensions don't match.
        - If input type combinations are invalid.
    """

    # Quick check for direct values in coordinate tuple
    if len(crd) == 1 and crd.is_value(0):
        var c_val = crd.value(0)

        # Fast path for 1D coordinate into 1D shape with direct values
        if len(shape) == 1 and shape.is_value(0):
            var s_val = shape.value(0)

            # Check for unknown values
            if c_val == UNKNOWN_VALUE or s_val == UNKNOWN_VALUE:
                return UNKNOWN_VALUE

            # If stride is provided with length 1 and is a direct value
            if len(_stride) == 1 and _stride.is_value(0):
                var st_val = _stride.value(0)
                if st_val == UNKNOWN_VALUE:
                    return UNKNOWN_VALUE
                return c_val * st_val
            elif len(_stride) == 0:
                # For a single element tuple with no stride, the stride is 1
                return c_val

        # Fast path for 1D coordinate into 2D shape with direct values
        elif len(shape) == 2 and len(_stride) == 2:
            # Early check for unknown coordinate
            if c_val == UNKNOWN_VALUE:
                return UNKNOWN_VALUE

            # Fast path for non-nested shape and stride (direct integer values)
            if (
                shape.is_value(0)
                and shape.is_value(1)
                and _stride.is_value(0)
                and _stride.is_value(1)
            ):
                var s0_val = shape.value(0)
                var s1_val = shape.value(1)
                var st0_val = _stride.value(0)
                var st1_val = _stride.value(1)

                # Check for unknown values
                if (
                    s0_val == UNKNOWN_VALUE
                    or s1_val == UNKNOWN_VALUE
                    or st0_val == UNKNOWN_VALUE
                    or st1_val == UNKNOWN_VALUE
                ):
                    return UNKNOWN_VALUE

                # Inline coordinate calculation and stride application (single expression)
                return (c_val % s0_val) * st0_val + (
                    (c_val // s0_val) % s1_val
                ) * st1_val

            # Fast path for 2D shape with potentially nested tuples
            else:
                # Compute shape products only once
                var s0_prod = product(shape[0])
                if s0_prod == UNKNOWN_VALUE:
                    return UNKNOWN_VALUE

                # Extract coordinates
                var c1, c0 = divmod(c_val, s0_prod)

                # Handle direct stride values case
                if _stride.is_value(0) and _stride.is_value(1):
                    var st0_val = _stride.value(0)
                    var st1_val = _stride.value(1)

                    if st0_val == UNKNOWN_VALUE or st1_val == UNKNOWN_VALUE:
                        return UNKNOWN_VALUE

                    # Direct calculation with stride values
                    return c0 * st0_val + c1 * st1_val

                # Handle complex nested strides with minimal recursion
                else:
                    # We know len(_stride) == 2, use direct indexing
                    var c0_tuple = IntTuple(c0)
                    var c1_tuple = IntTuple(c1)
                    return crd2idx(c0_tuple, shape[0], _stride[0]) + crd2idx(
                        c1_tuple, shape[1], _stride[1]
                    )

    # Original implementation for all other cases
    var stride: IntTuple
    if len(_stride) == 0:
        stride = prefix_product(shape)
    else:
        stride = _stride.owned_copy()

    if is_tuple(crd):
        if is_tuple(shape):  # tuple tuple tuple
            debug_assert(
                len(crd) == len(shape) and len(crd) == len(stride),
                "crd2idx shape mismatch: len(crd)=",
                len(crd),
                " len(shape)=",
                len(shape),
                " len(stride)=",
                len(stride),
            )
            var r: Int = 0
            for z in zip(crd, shape, stride):
                r += crd2idx(z[0], z[1], z[2])
            return r
        else:  # tuple "int" "int"
            # This case can occur in generic code with composed layouts
            # Return 0 as a fallback.
            return 0
    else:
        var int_crd: Int = 0 if len(crd) == 0 else Int(crd)

        if is_tuple(shape):  # "int" tuple tuple
            debug_assert(
                len(shape) == len(stride),
                "crd2idx shape mismatch: len(shape)=",
                len(shape),
                " len(stride)=",
                len(stride),
            )
            if len(shape) == 0:
                return 0
            var result: Int = 0
            for i in range(len(shape) - 1):
                var remainder: Int
                int_crd, remainder = divmod(int_crd, product(shape[i]))
                result += crd2idx(remainder, shape[i], stride[i])
            return result + crd2idx(
                int_crd, shape[len(shape) - 1], stride[len(stride) - 1]
            )
        else:  # "int" "int" "int"
            return int_crd * Int(stride)


def fill_like(src: IntTuple, val: Int) -> IntTuple:
    """
    Creates an `IntTuple` with the same structure as the source but filled with a specified value.

    This function recursively traverses the source `IntTuple` and creates a new `IntTuple`
    with identical structure, but with all leaf values replaced by the specified value.

    Args:
        src: The source `IntTuple` whose structure will be copied.
        val: The integer value to fill the new `IntTuple` with.

    Returns:
        A new `IntTuple` with the same structure as src but filled with val.
    """
    if is_tuple(src):
        var res = IntTuple()
        for elem in src:
            res.append(fill_like(elem, val))
        return res
    return val


def propagate_unknown(src: IntTuple, target: IntTuple) -> IntTuple:
    """
    Propagates unknown dimensions from the target `IntTuple` to the source `IntTuple`.

    This function creates a new `IntTuple` by combining the source and target `IntTuple`s,
    preserving unknown dimensions (UNKNOWN_VALUE) from the target while using values
    from the source for known dimensions.

    Args:
        src: The source `IntTuple` containing known dimension values.
        target: The target `IntTuple` that may contain unknown dimensions (UNKNOWN_VALUE).

    Returns:
        A new `IntTuple` with unknown dimensions from target and known dimensions from src.
    """
    if is_tuple(target):
        var dim = IntTuple()
        for d in zip(src, target):
            dim.append(propagate_unknown(d[0], d[1]))
        return dim

    if target == UNKNOWN_VALUE:
        return target.owned_copy()
    return src.owned_copy()


def reverse(src: IntTuple) -> IntTuple:
    """
    Reverses the order of elements in an `IntTuple`, recursively.

    This function reverses the top-level elements of the `IntTuple` and
    recursively reverses any nested `IntTuple`s.

    Args:
        src: The source `IntTuple` to reverse.

    Returns:
        A new `IntTuple` with elements in reversed order.

    Example:

        ```mojo
        from layout.int_tuple import IntTuple, reverse
        var t = IntTuple(1, 2, IntTuple(3, 4))
        var reversed = reverse(t) # returns ((4, 3), 2, 1)
        ```
    """
    if src.is_value():
        return IntTuple(src.value())
    var res = IntTuple()
    for i in range(len(src)):
        var idx = len(src) - i - 1
        if src.is_value(idx):
            res.append(src.value(idx))
        else:
            res.append(reverse(src[idx]))
    return res


def depth(src: IntTuple) -> Int:
    """
    Calculates the maximum nesting depth of an `IntTuple`.

    This function recursively traverses the `IntTuple` structure to determine
    its maximum nesting depth. A scalar value has depth 0, a flat tuple has
    depth 1, and nested tuples increase the depth accordingly.

    Args:
        src: The `IntTuple` to measure the depth of.

    Returns:
        An integer representing the maximum nesting depth.

    Example:

        ```mojo
        from layout import IntTuple, depth

        print(depth(IntTuple(1))) # prints 0
        print(depth(IntTuple(1, 2))) # prints 1
        print(depth((IntTuple(1, 2)))) # prints 2
        ```
    """
    if is_int(src):
        return 0
    var res = 1
    for elem in src:
        res += depth(elem)
    return res


comptime IntList = List[Int]
"""
A type alias for a List of integers with ownership.

This alias defines a List that contains Int values and has ownership of its data.
It's used throughout the module for storing and manipulating collections of integers,
particularly for operations like permutations and indices.
"""


def _sorted_perm(tuple: IntTuple) -> IntList:
    """Returns permutation indices that would sort the tuple."""
    var n = len(tuple)
    var indices = IntList(length=n, fill_with=lambda (i: Int) -> Int: i)
    var values = tuple

    # Insertion sort
    for i in range(1, n):
        var key_val = values[i]
        var key_idx = indices[i]
        var j = i - 1

        while j >= 0 and Int(values[j]) > Int(key_val):
            values.replace_entry(j + 1, int_value=Int(values[j]))
            indices[j + 1] = indices[j]
            j -= 1

        values.replace_entry(j + 1, int_value=Int(key_val))
        indices[j + 1] = key_idx

    return indices^


def _flat_apply_perm(tuple: IntTuple, perm: IntList) -> IntTuple:
    """Applies a permutation to an IntTuple."""
    var n = len(tuple)
    var result = IntTuple()
    for i in range(n):
        result.append(tuple[perm[i]])
    return result


def _flat_apply_invperm(tuple: IntTuple, perm: IntList) -> IntTuple:
    """Applies the inverse permutation to an IntTuple."""
    var n = len(tuple)
    var result = IntTuple(num_elems=n)
    for i in range(n):
        result.replace_entry(perm[i], int_value=Int(tuple[i]))
    return result


def _flat_compact_order(shape: IntTuple, order: IntTuple) -> IntTuple:
    """Helper function that computes compact order for flattened inputs."""
    debug_assert(
        len(shape) == len(order),
        "compact_order shape/order mismatch: len(shape)=",
        len(shape),
        " len(order)=",
        len(order),
    )

    var perm = _sorted_perm(order)
    var sorted_shape = _flat_apply_perm(shape, perm)
    var strides = prefix_product(sorted_shape)
    return _flat_apply_invperm(strides, perm)


@always_inline("nodebug")
def compact_order(shape: IntTuple, order: IntTuple) -> IntTuple:
    """Create a compact stride based on shape and order.

    This function generates a stride tuple where lower order numbers imply
    faster varying strides. The resulting shape and stride form a bijective layout.

    Args:
        shape: The shape tuple defining dimensions.
        order: The order tuple defining the relative ordering of dimensions.

    Returns:
        A stride tuple that creates a compact memory layout according to the
        specified order.

    Performance:
        - Always inlined for optimal performance in tight loops.
        - Flattens inputs and re-nests results for consistent behavior.

    Example:

        ```mojo
        from layout import IntTuple
        from layout.int_tuple import compact_order

        # Create a compact layout with dimensions (2,3,4,5) and ordering (1,4,3,5)
        var x = compact_order(IntTuple(2,3,4,5), IntTuple(1,4,3,5))  # returns (1,8,2,24)

        # Create a compact layout with nested dimensions and corresponding ordering
        var y = compact_order(IntTuple(2,IntTuple(3,4),5), IntTuple(1,IntTuple(2,3),4))  # returns (1,(2,6),24)
        ```
    """
    # Flatten both shape and order
    var flat_shape = flatten(shape)
    var flat_order = flatten(order)

    # Call _flat_compact_order on flattened inputs
    var flat_result = _flat_compact_order(flat_shape, flat_order)

    # Re-nest the result according to original shape's structure
    return to_nest(shape, flat_result)


def to_index_list[
    rank: Int, element_type: DType = .int64
](t: IntTuple) -> IndexList[rank, element_type=element_type]:
    """
    Converts an IntTuple to a flattened IndexList with the same values.

    Parameters:
        rank: The rank of the resulting IndexList.
        element_type: Element type, must be integer type.

    Args:
        t: The `IntTuple` defining the values.

    Returns:
        An IndexList filled with the values of t.
    """
    var res = IndexList[rank, element_type=element_type]()
    var flattened_t = t.flatten()
    for i in range(len(t)):
        res[i] = Int(flattened_t[i])

    return res


def coord_to_int_tuple[
    element_types: TypeList[Trait=CoordLike, ...],
    //,
](value: Coord[*element_types]) -> IntTuple:
    """Convert a `Coord` to an `IntTuple`, preserving the nested structure.

    This function recursively traverses the `Coord` and converts each element:
    - Value elements (`ComptimeInt`, `Scalar`) become integer values in the `IntTuple`
    - Tuple elements (nested `Coord`) become nested `IntTuple`s

    Parameters:
        element_types: The list of element types in the `Coord`.

    Args:
        value: The `Coord` to convert.

    Returns:
        An `IntTuple` with the same structure and values as the input `Coord`.
    """
    var result = IntTuple()

    comptime for i in range(type_of(value).__len__()):
        comptime T = element_types[i]

        comptime if T.is_tuple:
            # Recursively convert nested tuples
            result.append(coord_to_int_tuple(value[i].tuple()))
        else:
            # Convert value elements to integers
            result.append(IntTuple(Int(value[i].value())))

    return result


def coord_to_int_tuple[*element_types: CoordLike]() -> IntTuple:
    """Convert a `Coord` to an `IntTuple`, preserving the nested structure.

    This function recursively traverses the `Coord` and converts each element:
    - Value elements (`ComptimeInt`, `Scalar`) become integer values in the `IntTuple`
    - Tuple elements (nested `Coord`) become nested `IntTuple`s

    Parameters:
        element_types: The list of element types in the `Coord`.

    Returns:
        An `IntTuple` with the same structure and values as the input `Coord`.
    """
    var result = IntTuple()

    comptime for i in range(element_types.length):
        comptime T = element_types[i]

        comptime if T.is_tuple:
            # Splat the nested `Coord`'s own element types. Passing `T` itself
            # as the whole pack would re-enter with an identical pack, so the
            # recursion would never shrink.
            result.append(coord_to_int_tuple[*T.ParamListType]())
        else:
            comptime if T.is_static_value:
                result.append(IntTuple(T.static_value))
            else:
                result.append(UNKNOWN_VALUE)

    return result


comptime _IntTupleToCoordLikeTabulator[
    dtype: DType,
    tuple: IntTuple,
    idx: Int,
]: CoordLike = ComptimeInt[Int(tuple[idx])] if Int(
    tuple[idx]
) != UNKNOWN_VALUE else Scalar[
    dtype
]
"""Maps a single IntTuple element to a CoordLike type.

If the value is known, produces ComptimeInt[value].
If UNKNOWN_VALUE, produces Scalar.
"""

comptime _IntTupleToCoordLike[
    dtype: DType, tuple: IntTuple
] = TypeList.tabulate[
    len(tuple),
    _IntTupleToCoordLikeTabulator[dtype, tuple, _],
]()
"""Converts an IntTuple to a variadic of CoordLike types.

Note:
    This transformation is a value-to-type mapper that is meant to be
    used in the parameter domain.

For each element in the IntTuple:
- If the value is known (not UNKNOWN_VALUE), produces `ComptimeInt[value]`
- If the value is UNKNOWN_VALUE, produces `Scalar`

Example:
    ```mojo
    from layout.coord import Coord
    from layout import IntTuple
    from layout.int_tuple import _IntTupleToCoordLike

    # Known values become ComptimeInt, UNKNOWN_VALUE becomes Scalar
    comptime shape = IntTuple(3, -1, 5)
    comptime coord_types = _IntTupleToCoordLike[.int32, shape]
    # coord_types is equivalent to TypeList.of[Trait=CoordLike, ComptimeInt[3], Scalar, ComptimeInt[5]]()

    # Can be used to create a Coord type
    comptime my_coords = Coord[*coord_types]
    ```
"""
