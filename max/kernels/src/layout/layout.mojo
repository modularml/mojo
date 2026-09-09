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
"""Provides a high-performance tensor layout system for memory mapping and indexing.

The layout module implements a comprehensive system for describing memory layouts
of multi-dimensional tensors, enabling efficient mapping between logical tensor
coordinates and physical memory locations. This is a critical component for
high-performance tensor operations in machine learning and scientific computing.
These low-level primitives require careful use to avoid errors.
Understanding the relationship between tensor shapes, strides, and
memory layout is essential for effective use.

Key components:
- `Layout`: Primary struct implementing memory layout with shape and stride information
- Layout algebra: Functions for composing, dividing, and transforming layouts
- Tiling operations: Functions for hierarchical decomposition of layouts

Performance features:
- Zero-cost abstractions for mapping between logical and physical indices
- Support for both compile-time and runtime-determined shapes
- Efficient memory access patterns through layout transformations
- Hierarchical tiling for cache-friendly memory access

Common use cases:
- Defining memory layouts for tensors with different storage formats (row-major, column-major)
- Implementing efficient tensor operations with optimal memory access patterns
- Supporting hardware-specific memory layouts for accelerators
- Enabling zero-copy tensor views and reshaping operations

Example:

```mojo
from layout import Layout, IntTuple
from layout.layout import blocked_product

# Create a 3x4 row-major layout
var layout = Layout.row_major(3, 4)

# Access the memory location for logical coordinates (1, 2)
var memory_idx = layout([1, 2])

# Create a tiled layout for blocked matrix multiplication
var tiled = blocked_product(layout^, Layout([2, 2]))
```
"""

import std.sys
from std.math import ceildiv
from std.collections.string.string import _calc_initial_buffer_size_int32
from std.os import abort


from std.utils import IndexList

from .int_tuple import (
    UNKNOWN_VALUE,
    IntTuple,
    abs,
    compact_order,
    crd2idx,
    flatten,
    idx2crd,
    inner_product,
    is_flat,
    is_int,
    is_tuple,
    mul,
    prefix_product,
    product,
    product_each,
    reverse,
    shape_div,
    sorted,
    to_unknown,
    tuple_min,
)

# ===-----------------------------------------------------------------------===#
# Layout                                                                       #
# ===-----------------------------------------------------------------------===#


def make_layout(*layouts: Layout) -> Layout:
    """Creates a composite layout by concatenating multiple layouts.

    This function combines multiple layouts into a single layout by concatenating
    their shapes and strides. The resulting layout represents a hierarchical
    structure where each input layout becomes a component of the output layout.

    Args:
        layouts: Variable number of `Layout` objects to combine.

    Returns:
        A new Layout with concatenated shapes and strides from the input layouts.

    Example:

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import make_layout

    var layout1 = Layout(IntTuple(2, 3), IntTuple(3, 1))
    var layout2 = Layout(IntTuple(4, 5), IntTuple(5, 1))
    var combined = make_layout(layout1, layout2)
    # Result: Layout with shape ((2, 3), (4, 5)) and stride ((3, 1), (5, 1))
    ```
    """
    var shape = IntTuple()
    var stride = IntTuple()
    for layout in layouts:
        shape.append(layout.shape)
        stride.append(layout.stride)
    return Layout(shape, stride)


def make_layout(layout_a: Layout, layout_b: Layout) -> Layout:
    """Creates a composite layout from two layouts.

    This is a specialized version of make_layout that takes exactly two layouts
    and combines them into a single layout. This function exists as a workaround
    for compiler limitations.

    Args:
        layout_a: The first layout to include in the composite.
        layout_b: The second layout to include in the composite.

    Returns:
        A new `Layout` with concatenated shapes and strides from the input layouts.
    """
    return Layout(
        [layout_a.shape, layout_b.shape],
        [layout_a.stride, layout_b.stride],
    )


def make_ordered_layout(shape: IntTuple, order: IntTuple) -> Layout:
    """Creates a layout with strides ordered according to a specified traversal order.

    This function generates a compact (bijective) layout where the stride values
    follow the traversal order specified by the order parameter. This allows
    creating layouts with custom memory traversal patterns while maintaining
    a compact memory representation.

    Args:
        shape: The shape of the layout.
        order: The traversal order priority (lower values indicate higher priority).

    Returns:
        A `Layout` with the specified shape and strides ordered according to the
        traversal order.

    Example:

    ```mojo
    from layout import IntTuple, Layout
    from layout.layout import make_ordered_layout

    # Create a layout with shape (2,3,4,5) where dimensions are traversed
    # in the order: dim0, dim3, dim2, dim1
    var layout = make_ordered_layout(
        IntTuple(2, 3, 4, 5),
        IntTuple(1, 4, 3, 2)
    )
    # Result: Layout with shape (2,3,4,5) and stride (1,40,10,2)
    ```
    """
    var stride = compact_order(shape, order)
    return Layout(shape, stride)


@fieldwise_init
struct _LayoutIter[origin: ImmOrigin](ImplicitlyCopyable, Iterable, Iterator):
    """Iterator for traversing Layout dimensions.

    This internal iterator allows traversing the dimensions of a Layout object,
    yielding sub-layouts for each dimension. Each sub-layout contains the shape
    and stride for that dimension.

    Parameters:
        origin: The origin type for the `Layout` pointer, must be `ImmOrigin`.

    Attributes:
        index: Current position in the iteration.
        layout: Pointer to the `Layout` being iterated.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = Self
    comptime Element = Layout
    var index: Int
    var layout: Pointer[Layout, Self.origin]

    def __next__(mut self) raises StopIteration -> Self.Element:
        """Returns the next sub-layout in the iteration.

        Advances the iterator and returns a Layout containing the shape and stride
        for the next dimension.

        Returns:
            A Layout representing the next dimension.

        Raises:
            `StopIteration` when iteration is complete.
        """

        if self.__len__() <= 0:
            raise StopIteration()

        var idx = self.index
        self.index += 1
        return Layout(
            self.layout[].shape[idx],
            self.layout[].stride[idx],
        )

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Returns the number of remaining dimensions.

        Returns:
            Number of dimensions left to iterate.
        """
        return len(self.layout[].shape) - self.index

    @always_inline
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        return self

    @always_inline
    def bounds(self) -> Tuple[Int, Optional[Int]]:
        var size = len(self.layout[].shape) - self.index
        return (size, {size})


struct Layout(
    Copyable,
    Defaultable,
    Equatable,
    Iterable,
    Sized,
    Writable,
):
    """Represents a memory layout for multi-dimensional data.

    Layout provides a concrete representation of memory layouts using shape and
    stride information. It maps between logical coordinates and linear
    memory indices, enabling efficient access to multi-dimensional data.

    A Layout consists of:
    - shape: Defines the dimensions of the logical coordinate space
    - stride: Defines the step sizes in memory for each dimension

    The Layout struct supports various operations including:
    - Creation of row-major and column-major layouts
    - Conversion between coordinates and indices
    - Composition with other layouts
    - Iteration over sub-layouts

    Layouts can be hierarchical, with nested shapes and strides, allowing
    for complex memory access patterns like blocked or tiled layouts.
    """

    comptime has_shape = True
    """Indicates whether the layout has a valid shape."""

    var shape: IntTuple
    """The dimensions of the layout.

    This field defines the size of each dimension in the logical coordinate space.
    For example, a shape of (3, 4) represents a 3x4 grid of elements.
    """

    var stride: IntTuple
    """The memory step sizes for each dimension.

    This field defines how many elements to skip in memory when moving one unit
    in each dimension. For example, in a row-major 3x4 layout, the strides might
    be (4, 1), meaning moving one unit in the first dimension requires skipping
    4 elements in memory, while moving one unit in the second dimension requires
    skipping 1 element.
    """

    comptime IteratorType[
        iterable_mut: Bool, //, iterable_origin: Origin[mut=iterable_mut]
    ]: Iterator = _LayoutIter[ImmOrigin(iterable_origin)]
    """The iterator type for Layout iteration.

    Parameters:
        iterable_mut: Whether the iterable is mutable.
        iterable_origin: The origin of the iterable.
    """

    # ===------------------------------------------------------------------===#
    # Initializers
    # ===------------------------------------------------------------------===#

    @always_inline("nodebug")
    def __init__(out self):
        """Initializes an empty layout with no dimensions.

        Creates a layout with empty shape and stride tuples, which can be
        populated later using append operations.
        """
        self.shape = IntTuple()
        self.stride = IntTuple()

    def __init__(out self, shape: IntTuple):
        """Initializes a layout with the given shape and column-major strides.

        Creates a layout with the specified shape and automatically calculates
        column-major strides (where the first dimension varies fastest in memory).

        Args:
            shape: The dimensions of the layout.
        """
        # FIXME: all these owned_copy() calls are annoying, fix copy ctor
        self.shape = shape.owned_copy()
        self.stride = prefix_product(self.shape)

    def __init__(out self, shape: IntTuple, stride: IntTuple):
        """Initializes a layout with the given shape and stride.

        Creates a layout with explicitly specified shape and stride values.
        If an empty stride is provided, column-major strides are calculated.

        Args:
            shape: The dimensions of the layout.
            stride: The memory step size for each dimension, or empty for column-major.
        """
        self.shape = shape.owned_copy()
        if len(stride) == 0:
            self.stride = prefix_product(self.shape)
        else:
            self.stride = stride.owned_copy()

    @always_inline("nodebug")
    def idx2crd(self, idx: IntTuple) -> IntTuple:
        """Converts a linear index to logical coordinates.

        This is the inverse operation of the __call__ method, mapping from
        a memory index back to the corresponding logical coordinates.

        Args:
            idx: The linear index to convert.

        Returns:
            The logical coordinates corresponding to the given index.
        """
        return idx2crd(idx, self.shape, self.stride)

    @staticmethod
    def col_major(*dims: Int) -> Layout:
        """Creates a column-major layout with the specified dimensions.

        In a column-major layout, the first dimension varies fastest in memory,
        which is the default layout in languages like Fortran and MATLAB.

        Args:
            dims: Variable number of dimension sizes.

        Returns:
            A column-major Layout with the specified dimensions

        Example:

        ```mojo
        from layout import Layout

        # Create a 3x4 column-major layout
        var layout = Layout.col_major(3, 4)
        # Result: Layout with shape (3,4) and stride (1,3)
        ```
        """
        var shape = IntTuple(*dims)
        return Self.col_major(shape)

    @staticmethod
    def col_major(shape: IntTuple) -> Layout:
        """Creates a column-major layout with the specified shape.

        In a column-major layout, the first dimension varies fastest in memory,
        which is the default layout in languages like Fortran and MATLAB.

        Args:
            shape: An IntTuple specifying the dimensions.

        Returns:
            A column-major Layout with the specified shape

        Example:

        ```mojo
        from layout import Layout
        from layout.int_tuple import IntTuple

        # Create a 3x4 column-major layout
        var layout = Layout.col_major(IntTuple(3, 4))
        # Result: Layout with shape (3,4) and stride (1,3)
        ```
        """
        return Layout(shape, prefix_product(shape))

    @staticmethod
    def col_major[rank: Int](tuple: IndexList[rank]) -> Layout:
        """Creates a col-major layout from a IndexList with compile-time rank.

        This method creates a col-major layout where the first dimension varies fastest in memory.
        It handles both known and unknown dimensions at compile time, properly calculating
        strides for each dimension. If any dimension is unknown, subsequent strides will
        also be marked as unknown.

        Parameters:
            rank: The compile-time rank (number of dimensions) of the layout.

        Args:
            tuple: An IndexList containing the dimensions of the layout.

        Returns:
            A col-major Layout with the specified dimensions and computed strides.

        Example:

            ```mojo
            from layout import Layout
            from std.utils import IndexList

            # Create a row-major layout with compile-time rank
            var idx_list = IndexList[2](3, 4)
            var layout = Layout.col_major[2](idx_list)
            # Result: Layout with shape (3,4) and stride (1,3)
            ```
        """
        var c_stride = 1
        var shape = IntTuple()
        var stride = IntTuple(c_stride)

        comptime for i in range(rank):
            shape.append(tuple[i])

        var unknown_flag = False

        comptime for i in range(rank - 1):
            var dim = tuple[i]

            if dim == UNKNOWN_VALUE:
                unknown_flag = True

            stride.append(UNKNOWN_VALUE if unknown_flag else dim * c_stride)
            c_stride *= dim

        return Layout(shape, stride)

    @staticmethod
    def row_major(*dims: Int) -> Layout:
        """Creates a row-major layout with the specified dimensions.

        In a row-major layout, the last dimension varies fastest in memory,
        which is the default layout in languages like C, C++, and Python.

        Args:
            dims: Variable number of dimension sizes.

        Returns:
            A row-major Layout with the specified dimensions

        Example:

        ```mojo
        from layout import Layout

        # Create a 3x4 row-major layout
        var layout = Layout.row_major(3, 4)
        # Result: Layout with shape (3,4) and stride (4,1)
        ```
        """
        var shape = IntTuple(*dims)
        return Self.row_major(shape)

    @staticmethod
    def row_major[rank: Int](tuple: IndexList[rank]) -> Layout:
        """Creates a row-major layout from a IndexList with compile-time rank.

        This method creates a row-major layout where the last dimension varies fastest in memory.
        It handles both known and unknown dimensions at compile time, properly calculating
        strides for each dimension. If any dimension is unknown, subsequent strides will
        also be marked as unknown.

        Parameters:
            rank: The compile-time rank (number of dimensions) of the layout.

        Args:
            tuple: An IndexList containing the dimensions of the layout.

        Returns:
            A row-major Layout with the specified dimensions and computed strides.

        Example:

            ```mojo
            from layout import Layout
            from std.utils import IndexList

            # Create a row-major layout with compile-time rank
            var idx_list = IndexList[2](3, 4)
            var layout = Layout.row_major[2](idx_list)
            # Result: Layout with shape (3,4) and stride (4,1)
            ```
        """
        var c_stride = 1
        var shape = IntTuple()
        var stride = IntTuple(c_stride)

        var unknown_flag = False

        comptime for i in range(rank):
            shape.append(tuple[i])

        comptime for i in range(rank - 1):
            var dim = tuple[rank - 1 - i]

            if dim == UNKNOWN_VALUE:
                unknown_flag = True

            stride.append(UNKNOWN_VALUE if unknown_flag else dim * c_stride)
            c_stride *= dim

        return Layout(shape, reverse(stride))

    @staticmethod
    def row_major[rank: Int]() -> Layout:
        """Creates a row-major layout with unknown values for each axis from a compile-time rank.

        Parameters:
            rank: The compile-time rank (number of dimensions) of the layout.

        Returns:
            A row-major Layout with the given rank.

        Example:

            ```mojo
            from layout import Layout

            var layout = Layout.row_major[2]()
            # Result: Layout with shape (UNKNOWN_VALUE, UNKNOWN_VALUE)
            ```
        """
        var shape = IntTuple()

        comptime for i in range(rank):
            shape.append(UNKNOWN_VALUE)

        return Layout.row_major(shape)

    @staticmethod
    def row_major(shape: IntTuple) -> Layout:
        """Creates a row-major layout from an IntTuple of dimensions.

        In a row-major layout, the last dimension varies fastest in memory.
        This method computes the appropriate strides for a row-major layout
        given the input shape.

        Args:
            shape: An IntTuple containing the dimensions of the layout.

        Returns:
            A row-major Layout with the specified shape and computed strides.

        Example:

        ```mojo
        from layout import Layout
        from layout.int_tuple import IntTuple

        # Create a row-major layout from a shape tuple
        var shape = IntTuple(3, 4)
        var layout = Layout.row_major(shape)
        # Result: Layout with shape (3,4) and stride (4,1)
        ```
        """
        return Layout(shape, reverse(prefix_product(reverse(shape))))

    @always_inline
    def make_shape_unknown[axis: Int = UNKNOWN_VALUE](self) -> Layout:
        """Creates a new Layout with unknown shape dimensions.

        This method creates a copy of the current Layout but marks either all dimensions
        or a specific dimension as unknown, while preserving the original strides.
        This is useful for tiling tensors with runtime sizes where the tile's shape
        is unknown but the memory layout (strides) remains constant.

        Parameters:
            axis: The dimension to mark as unknown. If UNKNOWN_VALUE (default),
                 all dimensions are marked as unknown.

        Returns:
            A new Layout with the specified dimension(s) marked as unknown and
            original strides preserved.

        Example:

        ```mojo
        from layout import Layout
        from layout.int_tuple import IntTuple

        # Mark all dimensions as unknown
        var layout = Layout(IntTuple(2, 3))
        var unknown = layout.make_shape_unknown()
        # Result: Layout with shape (?, ?) and original strides

        # Mark only first dimension as unknown
        var partial = layout.make_shape_unknown[0]()
        # Result: Layout with shape (?, 3) and original strides
        ```
        """

        comptime if axis == UNKNOWN_VALUE:
            return Layout(to_unknown(self.shape), self.stride)
        else:
            # var shape_with_unknown = self.shape
            # shape_with_unknown[axis] = to_unknown(self.shape[axis])
            var shape_with_unknown = IntTuple()
            for i in range(len(self.shape)):
                if i != axis:
                    shape_with_unknown.append(self.shape[i])
                else:
                    shape_with_unknown.append(to_unknown(self.shape[i]))
            return Layout(shape_with_unknown, self.stride)

    # ===------------------------------------------------------------------===#
    # Methods
    # ===------------------------------------------------------------------===#

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Writes the layout to the specified writer.

        Formats the layout as "(shape:stride)" and writes it to the provided writer.

        Args:
            writer: The writer to output the layout representation to.
        """
        writer.write("(", self.shape, ":", self.stride, ")")

    @always_inline("nodebug")
    def __eq__(self, other: Layout) -> Bool:
        """Checks if this layout is equal to another layout.

        Two layouts are considered equal if they have identical shape and stride tuples.

        Args:
            other: The layout to compare with.

        Returns:
            True if the layouts are equal, False otherwise.
        """
        return self.shape == other.shape and self.stride == other.stride

    @always_inline("nodebug")
    def __len__(self) -> Int:
        """Returns the number of dimensions in the layout.

        Returns:
            The number of elements in the shape tuple.
        """
        return len(self.shape)

    @always_inline("nodebug")
    def __iter__(ref self) -> Self.IteratorType[origin_of(self)]:
        """Returns an iterator over the layout's dimensions.

        Each iteration yields a Layout containing the shape and stride for one dimension.

        Returns:
            An iterator over the layout's dimensions.
        """
        return _LayoutIter(0, Pointer(to=self))

    def size(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        Calculates the product of all dimensions in the shape.

        Returns:
            The total number of elements in the layout.
        """
        return product(self.shape)

    def cosize(self) -> Int:
        """Returns the size of the memory region spanned by the layout.

        Calculates the maximum memory index plus one, representing the total
        memory footprint required by the layout.

        Returns:
            The size of the memory region required by the layout.
        """
        return self(self.size() - 1) + 1
        # return math.max(1, inner_product(self.shape, self.stride))

    @always_inline("nodebug")
    def __getitem__(self, index: Int) -> Self:
        """Returns a sub-layout for the specified dimension.

        Args:
            index: The dimension index to extract.

        Returns:
            A Layout containing the shape and stride for the specified dimension.
        """
        return Layout(self.shape[index], self.stride[index])

    @always_inline("nodebug")
    def rank(self) -> Int:
        """Returns the number of dimensions in the layout.

        This is equivalent to __len__ and returns the number of elements in the
        shape tuple.

        Returns:
            The number of dimensions in the layout.
        """
        return len(self.shape)

    @always_inline("nodebug")
    def __call__(self, idx: IntTuple) -> Int:
        """Maps logical coordinates to a linear memory index.

        This is the core functionality of a layout, converting multi-dimensional
        coordinates to a linear memory location.

        Args:
            idx: The logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        return crd2idx(idx, self.shape, self.stride)

    @always_inline("nodebug")
    def append(mut self, item: Layout):
        """Appends another layout to this layout.

        This method adds the shape and stride from the provided layout to this layout,
        effectively increasing its dimensionality.

        Args:
            item: The layout to append to this layout.
        """
        self.shape.append(item.shape)
        self.stride.append(item.stride)

    @always_inline("nodebug")
    def all_dims_known(self) -> Bool:
        """Checks if all dimensions in the layout have known values.

        A dimension is considered unknown if its shape or stride is set to the
        special `UNKNOWN_VALUE` constant.

        Returns:
            True if all dimensions have known shape and stride values, False otherwise.
        """
        return self.shape.all_known() and self.stride.all_known()

    @always_inline("nodebug")
    def known_shape(self) -> Bool:
        """Checks if all shape dimensions in the layout have known values.

        A dimension is considered unknown if its shape is set to the special
        `UNKNOWN_VALUE` constant. This method only checks shapes, not strides.

        Returns:
            True if all shape dimensions have known values, False otherwise.
        """
        return self.shape.all_known()

    @always_inline("nodebug")
    def transpose(self) -> Layout:
        """Transposes the layout by reversing the order of dimensions.

        For an n-dimensional layout, this reverses the order of both shapes and strides.
        For nested layouts, only the top-level dimensions are transposed, not the
        hierarchical structure within nested tuples.

        Returns:
            A new Layout with transposed dimensions.

        Example:

        ```mojo
        from layout import Layout
        from layout.int_tuple import IntTuple

        # Simple 2D transpose (row-major to column-major)
        var layout = Layout.row_major(3, 4)  # shape (3,4), stride (4,1)
        var transposed = layout.transpose()  # shape (4,3), stride (1,4)

        # 3D transpose
        var layout3d = Layout.row_major(2, 3, 4)  # shape (2,3,4), stride (12,4,1)
        var trans3d = layout3d.transpose()        # shape (4,3,2), stride (1,4,12)

        # Nested layout - only top level transposed
        var nested = Layout(
            IntTuple(IntTuple(2, 3), 4),
            IntTuple(IntTuple(12, 4), 1)
        )
        var trans_nested = nested.transpose()
        # Result: shape (4, (2,3)), stride (1, (12,4))
        ```
        """
        # Reverse only the top level, not nested tuples
        var reversed_shape = IntTuple()
        var reversed_stride = IntTuple()

        for i in reversed(range(len(self.shape))):
            reversed_shape.append(self.shape[i])
            reversed_stride.append(self.stride[i])

        return Layout(reversed_shape, reversed_stride)


@always_inline("nodebug")
def size(l: Layout) -> Int:
    """Returns the total number of elements in the layout's domain.

    This is a standalone function equivalent to the Layout.size() method.

    Args:
        l: The layout to calculate the size for.

    Returns:
        The total number of elements in the layout.
    """
    return l.size()


@always_inline("nodebug")
def cosize(l: Layout) -> Int:
    """Returns the size of the memory region spanned by the layout.

    This is a standalone function equivalent to the Layout.cosize() method.

    Args:
        l: The layout to calculate the cosize for.

    Returns:
        The size of the memory region required by the layout.
    """
    return l.cosize()


comptime LayoutList = List[Layout]
"""Type alias for a list of Layout objects."""


@always_inline("nodebug")
def MakeLayoutList(var v0: Layout, var v1: Layout) -> LayoutList:
    """Creates a list containing two layouts.

    This is a convenience function for creating a LayoutList with two elements.

    Args:
        v0: The first layout to include in the list.
        v1: The second layout to include in the list.

    Returns:
        A LayoutList containing the two provided layouts.
    """
    return [v0^, v1^]


def MakeTileLayoutList[*tile_sizes: Int]() -> LayoutList:
    """Creates a list of layouts for tiling operations.

    This function creates a list of simple layouts, each with a shape from the
    provided tile_sizes and a stride of 1. These layouts can be used for tiling
    operations.

    Parameters:
        tile_sizes: Variable number of integer tile dimensions.

    Returns:
        A LayoutList containing layouts for each tile size.
    """
    var layout_list = LayoutList(capacity=tile_sizes.size)

    comptime for i in range(tile_sizes.size):
        comptime arg = tile_sizes[i]
        layout_list.append(Layout(arg, 1))

    return layout_list^


# The CUTE version has a second input to specify which modes to coalesce. We simplify
# it to flag to indicate keeping the original rank.
def coalesce(layout: Layout, keep_rank: Bool = False) -> Layout:
    """Simplifies a layout by combining dimensions with contiguous strides.

    This function reduces the rank of a layout by merging dimensions that have
    contiguous memory layouts, resulting in a simpler but equivalent layout.

    Args:
        layout: The layout to coalesce.
        keep_rank: If True, maintains the original rank of the layout. Default is False.

    Returns:
        A simplified layout with reduced rank where possible.

    Example:

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import coalesce

    # A layout with shape (2, (1, 4)) and stride (1, (4, 2)) can be coalesced
    var layout = Layout(IntTuple(2, IntTuple(1, 4)), IntTuple(1, IntTuple(4, 2)))
    var coalesced = coalesce(layout)
    # Result: Layout with shape (8) and stride (1)
    ```
    """
    if keep_rank:
        # Fast path for single-element layouts
        if len(layout) == 1:
            return Layout(
                coalesce(layout[0], False).shape,
                coalesce(layout[0], False).stride,
            )

        # Coalesce each mode and concat the results to preserve rank.
        var shapes = IntTuple()
        var strides = IntTuple()
        for mode in layout:
            var coalesced_mode = coalesce(mode, False)
            shapes.append(coalesced_mode.shape)
            strides.append(coalesced_mode.stride)

        return Layout(shapes, strides)

    var result_shape = IntTuple(1)
    var result_stride = IntTuple(0)

    for z in zip(flatten(layout.shape), flatten(layout.stride)):
        var shape = Int(z[0])
        var stride = Int(z[1])
        var last_shape = len(result_shape) - 1
        var last_stride = len(result_stride) - 1

        # skip their shape-1s
        if shape == 1:
            continue
        # replace our shape-1 with anything
        elif result_shape[last_shape] == 1:
            result_shape.replace_entry(last_shape, int_value=shape)
            result_stride.replace_entry(last_stride, int_value=stride)

        # merge modes if the shape*stride match and computable.
        elif Int(result_shape[last_shape]) * Int(
            result_stride[last_stride]
        ) == stride and UNKNOWN_VALUE not in (
            shape,
            stride,
            Int(result_shape[last_shape]),
            Int(result_stride[last_stride]),
        ):
            result_shape.replace_entry(
                last_shape,
                int_value=Int(result_shape[last_shape]) * shape,
            )

        # append a new mode
        else:
            result_shape.append(shape)
            result_stride.append(stride)

    if len(result_shape) == 1:
        return Layout(result_shape[0], result_stride[0])
    return Layout(result_shape, result_stride)


def composition(var layout_a: Layout, var layout_b: Layout) -> Layout:
    """Composes two layouts to create a new layout.

    This function creates a new layout by composing two layouts, where the first
    layout defines the outer structure and the second layout defines the inner
    structure.

    The new layout is compatible with `layout_b` (that is, it has the same `size`
    and every set of coordinates in `layout_b` has an equivalent in the new
    layout). You can think of `layout_b` as selecting a subset of elements
    from `layout_a`.

    Args:
        layout_a: The outer layout.
        layout_b: The inner layout.

    Returns:
        A new layout representing the composition of the two layouts.

    Example:

    ```mojo
    from layout.layout import Layout, IntTuple
    from layout.layout import composition

    # Compose a row-major layout with a tiling layout
    var base = Layout.row_major(6, 8)
    var tiling = Layout(IntTuple(3, 2), IntTuple(1, 3))
    var composed = composition(base^, tiling^)
    # Result: A layout that represents a 3x2 tile from
    # layout_a
    ```
    """
    if len(layout_b) == 0:
        return layout_a^

    if is_tuple(layout_b.shape):
        var r = Layout()
        for var layoutB_i in layout_b:
            r.append(composition(layout_a.copy(), layoutB_i^))
        return r^

    if layout_b.stride == 0:
        return Layout(layout_b.shape, 0)
    else:
        var result_shape = IntTuple()
        var result_stride = IntTuple()
        var rest_shape = layout_b.shape
        var rest_stride = layout_b.stride
        var flat_stride_a = flatten(layout_a.stride)

        for z in zip(flatten(layout_a.shape)[:-1], flat_stride_a[:-1]):
            var s = Int(z[0])
            var d = Int(z[1])

            var s1 = shape_div(s, rest_stride)
            result_shape.append(tuple_min(s1, rest_shape))
            result_stride.append(mul(rest_stride, d))
            rest_shape = shape_div(rest_shape, abs(s1))
            rest_stride = shape_div(rest_stride, s)

        result_shape.append(rest_shape)
        result_stride.append(
            mul(
                rest_stride,
                Int(flat_stride_a[len(flat_stride_a) - 1]),
            )
        )

        return coalesce(Layout(result_shape, result_stride))


# Tuple of layouts
def composition(var layout_a: Layout, tiler: LayoutList) -> Layout:
    """Composes a layout with a list of layouts to create a hierarchical layout.

    This function creates a new layout by composing each element of the first layout
    with the corresponding element in the tiler list. If the tiler list is shorter
    than the layout, the remaining elements from the layout are appended unchanged.

    Args:
        layout_a: The base layout to compose with the tiler.
        tiler: A list of layouts to compose with the base layout.

    Returns:
        A new layout representing the composition of the base layout with the tiler.

    Example:

    ```mojo
    from layout import Layout, LayoutList, IntTuple
    from layout.layout import composition

    # Compose a layout with a list of tiling layouts
    var base = Layout.row_major(6, 8)
    var tilers = LayoutList()
    tilers.append(Layout(IntTuple(2, 2), IntTuple(1, 2)))
    tilers.append(Layout(IntTuple(3, 3), IntTuple(1, 3)))
    var composed = composition(base^, tilers^)
    # Result: A layout with hierarchical tiling based on the tiler list
    ```
    """
    var result = Layout()
    for layout_item, tiler_item in zip(layout_a, tiler):
        result.append(composition(layout_item.copy(), tiler_item.copy()))

    # Remainder if tiler is shorter.
    for i in range(len(tiler), len(layout_a)):
        result.append(layout_a[i])

    return result^


def complement(layout: Layout, size: Int = 1) -> Layout:
    """Computes the complement layout for a given layout.

    This function creates a layout that represents the "gaps" or complementary
    structure of the input layout. It's useful for creating hierarchical layouts
    where you need to fill in the spaces between existing layout elements.

    Args:
        layout: The input layout to compute the complement for.
        size: The total size of the memory region to consider. Defaults to 1.

    Returns:
        A new layout representing the complement of the input layout.

    Example:

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import complement

    # Compute the complement of a layout
    var base = Layout(IntTuple(2, 3), IntTuple(3, 1))
    var comp = complement(base, 10)
    # Result: A layout that fills the gaps in the original layout
    ```
    """
    var current_idx = 1
    var sorted = sorted(
        IntTuple(zip(flatten(layout.stride), flatten(layout.shape)))
    )
    var sorted_len = len(sorted)

    var result_shape = IntTuple(num_elems=sorted_len + 1)
    var result_stride = IntTuple(num_elems=sorted_len + 1)

    # for z in sorted(zip(flatten(layout.stride), flatten(layout.shape))):
    var i = 0
    for z in sorted:
        var stride = Int(z[0])
        var shape = Int(z[1])

        if UNKNOWN_VALUE in (shape, stride) or stride == 0 or shape == 1:
            continue

        var in_bound = current_idx <= shape * stride
        if not in_bound:
            abort("Complement out of bounds.")

        result_shape.replace_entry(i, int_value=stride // current_idx)
        result_stride.replace_entry(i, int_value=current_idx)
        i += 1
        current_idx = shape * stride

    if size == UNKNOWN_VALUE:
        result_shape.replace_entry(i, int_value=UNKNOWN_VALUE)
    else:
        result_shape.replace_entry(
            i, int_value=ceildiv(size, current_idx)
        )  # ceil_div
    result_stride.replace_entry(i, int_value=current_idx)
    i += 1

    result_shape._store[0] = i
    result_stride._store[0] = i

    return coalesce(Layout(result_shape, result_stride))


@always_inline
def apply_tiler[
    func: def(var Layout, var Layout) thin -> Layout
](var layout_a: Layout, tiler: LayoutList) -> Layout:
    """Applies a layout transformation function to each element of a layout with a tiler.

    This utility function applies the specified transformation function to each
    corresponding pair of elements from the layout and tiler list. It's a generic
    mechanism for implementing various tiling operations.

    Parameters:
        func: A function that takes two layouts and returns a transformed layout.

    Args:
        layout_a: The base layout to transform.
        tiler: A list of layouts to use in the transformation.

    Returns:
        A new layout resulting from applying the transformation function to each pair.

    Example:

    ```mojo
    from layout import Layout, LayoutList, IntTuple
    from layout.layout import apply_tiler, logical_divide

    # Apply logical_divide to each element of a layout with a tiler
    var base = Layout.row_major(6, 8)
    var tilers = LayoutList()
    tilers.append(Layout(IntTuple(2, 2), IntTuple(1, 2)))
    var result = apply_tiler[logical_divide](base^, tilers)
    ```
    """
    if len(tiler) == 0:
        return layout_a^
    var result = Layout()
    for i in range(len(tiler)):
        result.append(func(layout_a[i], tiler[i].copy()))
    return result^


def logical_divide(layout_a: Layout, _layout_b: Layout) -> Layout:
    """Divides a layout into blocks according to another layout.

    This function creates a hierarchical layout by dividing the first layout
    according to the second layout. It's useful for creating blocked or tiled
    representations of tensors.

    Args:
        layout_a: The layout to be divided.
        _layout_b: The layout defining the division pattern.

    Returns:
        A new layout representing the hierarchical division.
    """
    return composition(
        layout_a.copy(),
        make_layout(_layout_b, complement(_layout_b, layout_a.size())),
    )


def logical_divide(layout_a: Layout, tiler: LayoutList) -> Layout:
    """Divides a layout into blocks according to a list of layouts.

    This is a variant of logical_divide that works with a list of layouts
    for more complex tiling patterns.

    Args:
        layout_a: The layout to be divided.
        tiler: A list of layouts defining the division patterns.

    Returns:
        A new layout representing the hierarchical division.
    """
    return apply_tiler[logical_divide](layout_a.copy(), tiler)


def logical_product(var _layout_a: Layout, var layout_b: Layout) -> Layout:
    """Creates a product of two layouts.

    This function creates a hierarchical layout by taking the logical product
    of two layouts. It's a fundamental operation for creating blocked or tiled
    layouts.

    Args:
        _layout_a: The first layout.
        layout_b: The second layout.

    Returns:
        A new layout representing the logical product of the two layouts.
    """
    return make_layout(
        _layout_a,
        composition(
            complement(_layout_a, _layout_a.size() * layout_b.cosize()),
            layout_b.copy(),
        ),
    )


def zip_modes(layout_a: Layout, layout_b: Layout) -> Layout:
    """Combines corresponding modes from two layouts.

    This function creates a new layout by combining corresponding dimensions
    from two layouts. If a dimension in layout_b has a non-positive shape,
    the corresponding dimension from layout_a is used directly.

    Args:
        layout_a: The first layout.
        layout_b: The second layout.

    Returns:
        A new layout with combined dimensions from both input layouts.
    """
    var zipped = Layout()
    for i in range(layout_a.rank()):
        var bi = layout_b[i]
        if is_int(bi.shape) and Int(bi.shape) <= 0:
            zipped.append(layout_a[i])
        else:
            zipped.append(make_layout(layout_a[i], bi))
    return zipped^


# If there is a 0-shape mode in layout_b, then the corresponding mode in
# layout_a is taken as is without adding any additional tiling modes.
def blocked_product(
    var layout_a: Layout,
    var layout_b: Layout,
    coalesce_output: Bool = False,
) -> Layout:
    """Creates a blocked layout by combining two layouts.

    This function creates a hierarchical blocked layout by combining an inner
    (block) and an outer (base) layout. The result is a layout where each
    element of the outer layout is replaced by a block defined by the
    inner layout.

    This is particularly useful for creating tiled layouts for efficient
    cache utilization in tensor operations like matrix multiplication.

    Args:
        layout_a: Inner layout. The layout for an individual block, or tile.
        layout_b: Outer layout. The layout of the tiles in the output layout.
        coalesce_output: Whether to coalesce the output layout. Default is False.

    Returns:
        A new layout representing the blocked structure

    Example:

    ```mojo
    from layout import Layout
    from layout.layout import blocked_product

    # Create a 2x3 matrix layout
    var matrix = Layout.row_major(2, 3)
    # Define 2x2 blocks
    var block = Layout.row_major(2, 2)
    # Create a blocked layout with 2x2 blocks
    var blocked = blocked_product(block^, matrix^)
    ```

    Output:

    ```plaintext
    (((2, 2), (2, 3)):((2, 12), (1, 4)))
          0    1    2    3    4    5
       +----+----+----+----+----+----+
    0  |  0 |  1 |  4 |  5 |  8 |  9 |
       +----+----+----+----+----+----+
    1  |  2 |  3 |  6 |  7 | 10 | 11 |
       +----+----+----+----+----+----+
    2  | 12 | 13 | 16 | 17 | 20 | 21 |
       +----+----+----+----+----+----+
    3  | 14 | 15 | 18 | 19 | 22 | 23 |
       +----+----+----+----+----+----+
    ```
    """
    # ((a_0, a_1, ...), (tile_0, tile_1, ...))
    var lp = logical_product(layout_a^, layout_b^)
    # ((a_0, tile_0), (a_1, tile_1), ...)
    var zipped = zip_modes(lp[0], lp[1])
    if coalesce_output:
        return coalesce(zipped, keep_rank=True)
    else:
        return zipped^


def tile_to_shape(
    var tile: Layout, target_shape: IntTuple, order: Optional[IntTuple] = None
) -> Layout:
    """Creates a layout by tiling a base layout to match a target shape.

    This function creates a hierarchical layout by repeating a tile layout to match
    a target shape. It calculates how many times the tile needs to be repeated in
    each dimension to reach the target shape, and creates a tiler layout with this
    information.

    Args:
        tile: The base layout to be tiled.
        target_shape: The desired final shape to tile to.
        order: Optional memory ordering for the tiler layout. If None, defaults to
            column-major ordering.

    Returns:
        A new layout representing the tiled structure that matches the target shape.

    Example:

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import tile_to_shape

    # Create a 2x2 tile layout
    var tile = Layout.row_major(2, 2)
    # Tile it to create a 6x4 layout
    var tiled = tile_to_shape(tile^, IntTuple(6, 4))
    # Result: A layout with 3x2 tiles of size 2x2 each
    ```
    """
    var flat_tile_shape = product_each(tile.shape)
    var flat_target_shape = product_each(target_shape)
    var tiler_shape = IntTuple()
    for i in range(len(flat_tile_shape)):
        var a = Int(flat_target_shape[i])
        var b = Int(flat_tile_shape[i])
        if a <= 0:
            tiler_shape.append(a)
            continue
        if a % b != 0:
            abort(
                "Tile to shape failed: target shape is not divisible by tile"
                " shape"
            )
        tiler_shape.append(a // b)

    var new_order: IntTuple
    if order:
        new_order = order.value()
    else:
        new_order = prefix_product(tiler_shape)  # default to column major
    var tiler = make_ordered_layout(tiler_shape, new_order)
    return blocked_product(tile^, tiler^)


def logical_product(var layout_a: Layout, tiler: LayoutList) -> Layout:
    """Creates a product of a layout with a list of layouts.

    This is a variant of logical_product that works with a list of layouts
    for more complex tiling patterns. It applies the logical_product operation
    to each element of the layout with the corresponding element in the tiler list.

    Args:
        layout_a: The base layout to create products with.
        tiler: A list of layouts defining the product patterns.

    Returns:
        A new layout representing the logical product with the tiler layouts.

    Example:

    ```mojo
    from layout import Layout, LayoutList, IntTuple
    from layout.layout import logical_product

    # Create a product of a layout with a list of layouts
    var base = Layout.row_major(6, 8)
    var tilers = LayoutList()
    tilers.append(Layout(IntTuple(2, 2)))
    var result = logical_product(base^, tilers)
    ```
    """
    if len(tiler) == 1:
        return logical_product(layout_a^, tiler[0].copy())
    return apply_tiler[logical_product](layout_a^, tiler)


def hierarchical_unzip(layout_a: Layout, tiler: LayoutList) -> Layout:
    """Hierarchically unzips a layout according to a list of layouts.

    This function creates a hierarchical layout by unzipping the first layout
    according to the layouts in the tiler list. It's useful for decomposing
    a layout into hierarchical components for more efficient memory access
    patterns or to enable specialized tensor operations.

    Args:
        layout_a: The layout to be unzipped.
        tiler: A list of layouts defining the unzipping patterns.

    Returns:
        A new layout representing the hierarchical unzipping with components
        from both the original layout and the tiler layouts.

    Example:

    ```mojo
    from layout import Layout, LayoutList, IntTuple
    from layout.layout import hierarchical_unzip

    # Create a layout to unzip
    var base = Layout.row_major(6, 8)
    var tilers = LayoutList()
    tilers.append(Layout(IntTuple(2, 2)))
    var result = hierarchical_unzip(base, tilers)
    ```
    """
    var res_1 = Layout()
    var res_2 = Layout()

    for i in range(len(tiler)):
        var split = hierarchical_unzip(layout_a[i], tiler[i])
        res_1.append(split[0])
        res_2.append(split[1])

    # Remainder if tiler is shorter.
    for i in range(len(tiler), len(layout_a)):
        res_2.append(layout_a[i])

    return make_layout(res_1, res_2)


def hierarchical_unzip(
    layout_a: Layout,
    layout_b: Layout,
) -> Layout:
    """Hierarchically unzips a layout according to another layout.

    This function creates a hierarchical layout by unzipping the first layout
    according to the second layout. It's a fundamental operation for decomposing
    a layout into hierarchical components, which enables more efficient memory
    access patterns for various tensor operations.

    Args:
        layout_a: The layout to be unzipped.
        layout_b: The layout defining the unzipping pattern.

    Returns:
        A new layout representing the hierarchical unzipping of layout_a
        according to the pattern defined by layout_b.

    Example:

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import hierarchical_unzip

    # Create layouts
    var base = Layout.row_major(6, 8)
    var pattern = Layout(IntTuple(2, 2))
    var result = hierarchical_unzip(base, pattern)
    ```
    """
    if len(layout_a) == 1 or len(layout_b) == 1:
        return logical_divide(layout_a, Layout(layout_b.shape))

    var res_1 = Layout()
    var res_2 = Layout()

    for i in range(layout_b.rank()):
        var split = hierarchical_unzip(layout_a[i], layout_b[i])
        res_1.append(split[0])
        res_2.append(split[1])

    # Remainder if tiler is shorter.
    for i in range(len(layout_b), len(layout_a)):
        res_2.append(layout_a[i])

    return make_layout(res_1, res_2)


@always_inline("nodebug")
def zipped_divide(layout_a: Layout, layout_b: Layout) -> Layout:
    """Divides a layout into blocks according to another layout.

    This function creates a hierarchical layout by dividing the first layout
    according to the second layout. It's an alias for hierarchical_unzip that provides a
    more intuitive name for the division operation. This is useful for creating
    blocked or tiled representations of tensors.

    Args:
        layout_a: The layout to be divided.
        layout_b: The layout defining the division pattern.

    Returns:
        A new layout representing the hierarchical division of layout_a according
        to layout_b.

    Example:

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import zipped_divide

    # Create layouts
    var base = Layout.row_major(6, 8)
    var pattern = Layout(IntTuple(2, 2))
    var result = zipped_divide(base, pattern)
    ```
    """
    return hierarchical_unzip(layout_a, layout_b)


@always_inline("nodebug")
def zipped_divide(layout_a: Layout, tiler: LayoutList) -> Layout:
    """Divides a layout into blocks according to a list of layouts.

    This function creates a hierarchical layout by dividing the first layout
    according to the layouts in the tiler list. It's an alias for hierarchical_unzip that
    provides a more intuitive name for the division operation when working with
    multiple tiling patterns.

    Args:
        layout_a: The layout to be divided.
        tiler: A list of layouts defining the division patterns.

    Returns:
        A new layout representing the hierarchical division of layout_a according
        to the patterns in tiler.

    Example:

    ```mojo
    from layout import Layout, LayoutList, IntTuple
    from layout.layout import zipped_divide

    # Create layouts
    var base = Layout.row_major(6, 8)
    var tilers = LayoutList()
    tilers.append(Layout(IntTuple(2, 2)))
    var result = zipped_divide(base, tilers)
    ```
    """
    return hierarchical_unzip(layout_a, tiler)


@no_inline
def print_layout(layout: Layout):
    """Prints a 2D layout to the standard output.

    This function visualizes a 2D layout by printing a formatted table showing
    the memory indices for each logical coordinate.

    Args:
        layout: The 2D layout to print.
    """
    if layout.rank() != 2:
        abort("print_layout only supports 2D layouts")

    print(layout)
    # make stdout mutable
    var stdout = std.sys.stdout
    format_layout(layout, stdout)


@no_inline
def format_layout[W: Writer](layout: Layout, mut writer: W):
    """Formats a 2D layout as a table and writes it to the specified writer.

    This function creates a visual representation of a 2D layout as a table
    showing the memory indices for each logical coordinate.

    Parameters:
        W: Type parameter representing a Writer implementation.

    Args:
        layout: The 2D layout to format.
        writer: The writer to output the formatted layout to.
    """

    @__parameter
    def _write_divider(column_count: Int, cell_width: Int):
        for _ in range(column_count):
            writer.write("+")
            for _ in range(cell_width):
                writer.write("-")
        writer.write("+\n")

    var idx_width = _calc_initial_buffer_size_int32(layout.cosize()) + 2

    # Print column labels
    writer.write("    ")
    for n in range(layout[1].size()):
        writer.write("  ")
        n.write_padded(writer, width=idx_width - 2)

        var is_last_column = n + 1 == layout[1].size()

        if not is_last_column:
            writer.write(" ")

    writer.write("\n")

    for m in range(layout[0].size()):
        writer.write("    ")
        _write_divider(layout[1].size(), idx_width)

        # Print row label
        m.write_padded(writer, width=2)
        writer.write("  ")

        for n in range(layout[1].size()):
            writer.write("| ")
            layout([m, n]).write_padded(
                writer,
                width=idx_width - 2,
            )
            writer.write(" ")
        writer.write("|\n")

    writer.write("    ")

    # Write the final horizontal dividing line
    _write_divider(layout[1].size(), idx_width)


def sublayout(layout: Layout, *modes: Int) -> Layout:
    """Creates a sublayout by selecting specific dimensions from a layout.

    This function extracts a subset of dimensions from a layout to create a new
    layout with lower rank. For example, from a 3D layout, you could extract
    a 2D layout containing only the first and third dimensions.

    Args:
        layout: The source layout to extract dimensions from.
        modes: The indices of dimensions to include in the sublayout.

    Returns:
        A new layout containing only the specified dimensions.

    Example:

    From a layout with shape (3,4,5), sublayout(layout, 0, 2) would
    create a layout with shape (3,5).
    """
    var shape = IntTuple()
    var stride = IntTuple()
    for mode in modes:
        shape.append(layout.shape[mode])
        stride.append(layout.stride[mode])
    return Layout(shape, stride)


def expand_strides(shape: IntTuple, stride: Int) -> IntTuple:
    """Expands a scalar stride into a stride tuple matching a shape tuple.

    This function creates a stride tuple that matches the structure of a shape tuple,
    with each stride value calculated based on the cumulative product of shape
    dimensions.

    Args:
        shape: The shape tuple to match.
        stride: The base stride value to expand.

    Returns:
        A stride tuple matching the structure of the shape tuple.
    """
    var new_stride = IntTuple()
    var cumulative_stride: Int = stride
    for sz in shape:
        if sz.is_tuple():
            new_stride.append(expand_strides(sz, cumulative_stride))
            cumulative_stride *= size(Layout(sz))
        else:
            new_stride.append(cumulative_stride)
            cumulative_stride *= sz.value()
    return new_stride


def expand_modes_alike(
    shape_a: IntTuple, stride_a: IntTuple, shape_b: IntTuple, stride_b: IntTuple
) -> Array[IntTuple, 3]:
    """Aligns two shape-stride pairs to have the same hierarchical structure.

    This function is used to make two layouts compatible for operations by ensuring
    they have the same hierarchical structure, expanding scalar values into tuples
    as needed.

    Args:
        shape_a: The first shape tuple.
        stride_a: The first stride tuple.
        shape_b: The second shape tuple.
        stride_b: The second stride tuple.

    Returns:
        An array containing three tuples: the common shape, the expanded stride_a,
        and the expanded stride_b.
    """
    if shape_a.is_tuple() and shape_b.is_tuple():
        var new_shape = IntTuple()
        var new_stride_a = IntTuple()
        var new_stride_b = IntTuple()
        # TODO: lift this limitation, and instead require
        # that size(shape_a) != size(shape_b):
        # Ideally, we'd be able to call `coalesce` on two layouts prior
        # to `uncoalesce` in functions like `copy_to` to first simplify.
        if len(shape_a) != len(shape_b):
            abort()
        for i in range(len(shape_a)):
            if shape_a[i].is_value() and shape_b[i].is_value():
                new_shape.append(shape_a[i].value())
                new_stride_a.append(stride_a[i].value())
                new_stride_b.append(stride_b[i].value())
            elif shape_a[i].is_value():
                new_shape.append(shape_b[i])
                new_stride_b.append(stride_b[i])
                new_stride_a.append(
                    expand_strides(shape_b[i], stride_a[i].value())
                )
            elif shape_b[i].is_value():
                new_shape.append(shape_a[i])
                new_stride_a.append(stride_a[i])
                new_stride_b.append(
                    expand_strides(shape_a[i], stride_b[i].value())
                )
            else:
                var uc = expand_modes_alike(
                    shape_a[i], stride_a[i], shape_b[i], stride_b[i]
                )
                new_shape.append(uc[0])
                new_stride_a.append(uc[1])
                new_stride_b.append(uc[2])

        return [new_shape, new_stride_a, new_stride_b]
    elif shape_a.is_tuple():
        return [
            shape_a.owned_copy(),
            stride_a.owned_copy(),
            expand_strides(shape_a, stride_b.value()),
        ]
    elif shape_b.is_tuple():
        return [
            shape_b.owned_copy(),
            expand_strides(shape_b.owned_copy(), stride_a.value()),
            stride_b.owned_copy(),
        ]
    else:
        return [
            shape_b.owned_copy(),
            stride_a.owned_copy(),
            stride_b.owned_copy(),
        ]


def expand_modes_alike(layout_a: Layout, layout_b: Layout) -> Array[Layout, 2]:
    """Aligns two layouts to have the same hierarchical structure.

    This function tiles both layouts so they mirror each other's structure,
    making them compatible for operations that require matching hierarchies.

    Args:
        layout_a: The first layout to align.
        layout_b: The second layout to align.

    Returns:
        An array containing two layouts with matching hierarchical structures.

    Example:

    Given layouts with different structures:

    - layout_0: (((3, (5, 2)), 4):((1, (24, 12)), 3))
    - layout_1: ((30, (2, 2)):(2, (60, 1)))

    The result would be two layouts with matching structures:

    - (((3, (5, 2)), (2, 2)):((1, (24, 12)), (3, 6)))
    - (((3, (5, 2)), (2, 2)):((2, (6, 30)), (60, 1)))

    ```mojo
    from layout import Layout, IntTuple
    from layout.layout import expand_modes_alike

    comptime layout_0 = Layout(
        IntTuple(IntTuple(3, IntTuple(5, 2)), 4),
        IntTuple(IntTuple(1, IntTuple(24, 12)), 3),
    )
    comptime layout_1 = Layout(
        IntTuple(30, IntTuple(2, 2)), IntTuple(2, IntTuple(60, 1))
    )
    comptime uc = expand_modes_alike(layout_0, layout_1)
    print(uc[0])
    # (((3, (5, 2)), (2, 2)):((1, (24, 12)), (3, 6)))
    print(uc[1])
    # (((3, (5, 2)), (2, 2)):((2, (6, 30)), (60, 1)))
    ```
    """
    var uc = expand_modes_alike(
        layout_a.shape, layout_a.stride, layout_b.shape, layout_b.stride
    )
    return [Layout(uc[0], uc[1]), Layout(uc[0], uc[2])]


def right_inverse(layout: Layout) -> Layout:
    """Creates a right inverse of a layout.

    The right inverse of a layout maps memory indices back to logical coordinates.
    This is useful for converting between different memory layouts.

    Args:
        layout: The layout to invert.

    Returns:
        A new layout representing the right inverse of the input layout.
    """
    var flat_layout = coalesce(layout)
    var rstride = prefix_product(flat_layout.shape)
    var shape = IntTuple()
    var stride = IntTuple()
    var next_stride = 1
    for _ in range(flat_layout.rank()):
        for j in range(flat_layout.rank()):
            if Int(flat_layout.stride[j]) == next_stride:
                var shape_j = flat_layout.shape[j]
                shape.append(shape_j)
                stride.append(rstride[j])
                next_stride *= Int(shape_j)
                break
    return Layout(shape, stride)


def upcast[check: Bool = True](var layout: Layout, factor: Int) -> Layout:
    """Fuses consecutive elements in a layout to create a coarser layout.

    This function is useful for converting between different data type granularities,
    such as from bytes to larger data types like bfloat16 or tf32.

    Parameters:
        check: Whether to check for incompatible factors.

    Args:
        layout: The layout to upcast.
        factor: The number of consecutive elements to fuse into one.

    Returns:
        A new layout with adjusted shape and stride for the coarser granularity.
    """
    if is_int(layout.shape):
        if layout.stride == 0:
            return layout^
        else:
            var fac = IntTuple(factor)
            var up_shape = shape_div[check](
                layout.shape, shape_div[check](fac, layout.stride)
            )
            var up_stride = shape_div[check](layout.stride, fac)
            return Layout(up_shape, up_stride)
    else:
        var res = Layout()
        for i in range(layout.rank()):
            res.append(upcast(layout[i], factor))
        return res^


def downcast(layout: Layout, factor: Int) -> Layout:
    """Splits elements in a layout to create a finer layout without changing the
    total number of elements so that the alignment is preserved.

    This function is useful for converting between different data type granularities,
    such as from uint128 to bf16.

    Args:
        layout: The layout to downcast.
        factor: The number of elements to split into.

    Returns:
        A new layout with adjusted shape and stride for the finer granularity.
    """
    return Layout(layout.shape, mul(layout.stride, factor))


def is_row_major[rank: Int](layout: Layout) -> Bool:
    """Checks if a layout has row-major ordering for the specified rank.

    A row-major layout has strides that decrease from left to right, with the
    rightmost dimension having a stride of 1.

    Parameters:
        rank: The expected rank of the layout.

    Args:
        layout: The layout to check.

    Returns:
        True if the layout has row-major ordering for the specified rank,
        False otherwise.
    """
    var flat_shape = flatten(layout.shape)
    var flat_rank = len(flat_shape)

    if flat_rank != rank:
        return False

    var flat_stride = flatten(layout.stride)
    if flat_stride[flat_rank - 1].value() != 1:
        return False

    var correct_stride = flat_shape[flat_rank - 1].value()

    for i in reversed(range(flat_rank - 1)):
        var stride_i = flat_stride[i].value()
        if stride_i != correct_stride:
            return False
        correct_stride *= flat_shape[i].value()

    return True


def is_contiguous_dim(layout: Layout, dim: Int) -> Bool:
    """Checks if a flat layout is contiguous in a specific dimension.

    This function checks if a flat layout is contiguous in a specified
    dimension, considering both positive strides and zero strides with a single
    element. The latter case is necessary for coalesced layouts.

    Args:
        layout: The layout to check.
        dim: The dimension to check.

    Returns:
        True if the layout is contiguous in the specified dimension,
        False otherwise.
    """
    if not is_flat(layout.shape):
        abort("layout must be flat")

    return layout.stride[dim] == 1 or (
        layout.stride[dim] == 0 and layout.shape[dim] == 1
    )
