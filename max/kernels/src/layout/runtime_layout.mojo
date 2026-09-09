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
"""
Provides the `RuntimeLayout` type and functions for working with it. You can use
`RuntimeLayout` to define a layout where the dimensions are not known at compile
time.

You can import these APIs from `layout.runtime_layout`.

```mojo
from layout.runtime_layout import RuntimeLayout, make_layout
```
"""


from std.utils import IndexList

from . import IntTuple, Layout
from .int_tuple import UNKNOWN_VALUE, flatten
from .layout import coalesce as coalesce_layout
from .layout import make_layout as make_layout_static
from .runtime_tuple import (
    RuntimeTuple,
    crd2idx,
    idx2crd,
    idx2crd_int_tuple,
    product,
)

# A `Layout` like type that uses RuntimeTuple as its storage instead of
# IntTuple.


struct RuntimeLayout[
    layout: Layout,
    /,
    *,
    element_type: DType = .int64,
    linear_idx_type: DType = .int64,
](Defaultable, TrivialRegisterPassable, Writable):
    """A runtime-configurable layout that uses `RuntimeTuple` for storage.

    This struct provides a layout implementation that can be modified at runtime,
    unlike the static [`Layout`](/api/mojo/layout/layout/Layout) type. It
    uses [`RuntimeTuple`](/api/mojo/layout/runtime_tuple/RuntimeTuple) for
    shape and stride storage.

    Parameters:
        layout: The static `Layout` type to base this runtime layout on.
        element_type: The integer type of the each dimension element. Must be signed.
        linear_idx_type: The integer type of the linear index into memory returned by `crd2idx`. Must be signed.

    The layout must have statically known dimensions at compile time, but the
    actual shape and stride values can be modified during execution.
    """

    comptime ShapeType = RuntimeTuple[
        Self.layout.shape, element_type=Self.element_type
    ]
    """Type alias for the runtime shape tuple."""
    var shape: Self.ShapeType
    """The shape of the layout as a runtime tuple.

    Stores the size of each dimension. Uses the specified bitwidth and is
    unsigned. Must match the static layout's shape dimensions.
    """

    comptime StrideType = RuntimeTuple[
        Self.layout.stride, element_type=Self.linear_idx_type
    ]
    """Type alias for the runtime stride tuple."""
    var stride: Self.StrideType
    """The stride of the layout as a runtime tuple.

    Stores the stride (step size) for each dimension. Uses 64-bit unsigned
    integers since strides can be large values. Must match the static layout's
    stride dimensions.
    """

    @always_inline
    def __init__(out self):
        """Initialize a `RuntimeLayout` with default values.

        Creates a new `RuntimeLayout` instance with default shape and stride
        values. Requires that the static layout has known dimensions at compile
        time.

        Constraints:
            The static layout that this runtime layout is based on must have all
            dimensions known.
        """

        comptime assert (
            Self.layout.all_dims_known()
        ), "Static layout with known dims is required"

        self.shape = {}
        self.stride = {}

    @always_inline
    def __init__(
        out self,
        shape: RuntimeTuple[Self.layout.shape, element_type=Self.element_type],
        stride: RuntimeTuple[
            Self.layout.stride, element_type=Self.linear_idx_type
        ],
    ):
        """Initialize a `RuntimeLayout` with specified shape and stride.

        Args:
            shape: A `RuntimeTuple` containing the dimensions of each axis.
            stride: A `RuntimeTuple` containing the stride values for each axis.
        """

        self.shape = shape
        self.stride = stride

    # FIXME: This should probably better done in the RuntimeTuple constructor
    @always_inline
    def __call__(self, idx: Int) -> Scalar[Self.linear_idx_type]:
        """Convert a single index to a flat linear index.

        Args:
            idx: The one-dimensional index to convert.

        Returns:
            The corresponding flat linear index in the layout.
        """
        return self.__call__(RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx))

    @always_inline
    def __call__[
        t: IntTuple
    ](self, idx: RuntimeTuple[t, ...]) -> Scalar[Self.linear_idx_type]:
        """Convert a multi-dimensional index to a flat linear index.

        Parameters:
            t: The `IntTuple` type for the index.

        Args:
            idx: A `RuntimeTuple` containing the multi-dimensional coordinates.

        Returns:
            The corresponding flat linear index in the layout.
        """
        return crd2idx[out_type=Self.linear_idx_type](
            idx, self.shape, self.stride
        )

    @always_inline("nodebug")
    def idx2crd[
        t: IntTuple
    ](self, idx: RuntimeTuple[t, ...]) -> RuntimeTuple[
        idx2crd_int_tuple(t, Self.layout.shape, Self.layout.stride),
        element_type=Self.element_type,
    ]:
        """Converts a linear index to logical coordinates.

        This is the inverse operation of the __call__ method, mapping from
        a memory index back to the corresponding logical coordinates.

        Parameters:
            t: The `IntTuple` type for the index.

        Args:
            idx: The linear index to convert.

        Returns:
            The logical coordinates corresponding to the given index.
        """
        return idx2crd(idx, self.shape, self.stride)

    @always_inline
    def size(self) -> Int:
        """Calculate the total number of elements in the layout.

        Returns:
            The product of all dimensions in the shape, representing the total
            number of elements that can be addressed by this layout.
        """
        return product(self.shape)

    @always_inline
    def bound_check_required(self) -> Bool:
        """Determine if bounds checking is required for this layout.

        Returns:
            True if any dimension in the shape differs from the static layout's
            shape, False otherwise.
        """

        comptime for i in range(Self.layout.rank()):
            comptime dim_i = Int(Self.layout.shape[i])
            if self.shape.value[i] != dim_i:
                return True
        return False

    @always_inline
    def cast[
        _element_type: DType,
        /,
        *,
        target_linear_idx_type: DType = Self.linear_idx_type,
    ](self) -> RuntimeLayout[
        Self.layout,
        element_type=_element_type,
        linear_idx_type=target_linear_idx_type,
    ]:
        """Cast the layout to use a different element bitwidth.

        Parameters:
            _element_type: The target data type.
            target_linear_idx_type: The target linear idx type.

        Returns:
            A new `RuntimeLayout` with the shape cast to the specified type.
        """
        return {
            self.shape.cast[_element_type](),
            self.stride.cast[target_linear_idx_type](),
        }

    @staticmethod
    def row_major[
        rank: Int, //
    ](
        shape: IndexList[rank, ...],
        out result: RuntimeLayout[
            Self.layout,
            element_type=Self.element_type,
            linear_idx_type=Self.linear_idx_type,
        ],
    ):
        """Create a row-major layout from the given shape.

        In row-major layout, elements with adjacent rightmost indices are
        adjacent in memory.

        Parameters:
            rank: The number of dimensions in the layout.

        Args:
            shape: An `IndexList` containing the dimensions of each axis.

        Returns:
            A `RuntimeLayout` with row-major stride ordering.
        """
        return {
            shape.cast[Self.element_type](),
            shape.get_row_major_strides().cast[Self.linear_idx_type](),
        }

    @staticmethod
    def col_major[
        rank: Int, //
    ](
        shape: IndexList[rank, ...],
        out result: RuntimeLayout[
            Self.layout,
            element_type=Self.element_type,
            linear_idx_type=Self.linear_idx_type,
        ],
    ):
        """Create a column-major layout from the given shape.

        In column-major layout, elements with adjacent leftmost indices are
        adjacent in memory.

        Parameters:
            rank: The number of dimensions in the layout.

        Args:
            shape: An `IndexList` containing the dimensions of each axis.

        Returns:
            A `RuntimeLayout` with column-major stride ordering.
        """

        var stride = IndexList[rank, element_type=Self.linear_idx_type]()
        var c_stride = 1
        stride[0] = c_stride

        comptime for i in range(1, rank):
            var dim = shape[i - 1]
            stride[i] = dim * c_stride
            c_stride *= dim
        return {shape.cast[Self.element_type](), stride}

    @no_inline
    def write_to(self, mut writer: Some[Writer]):
        """Write a string representation of the layout to a writer.

        Args:
            writer: The `Writer` object to write the layout representation to.
        """

        writer.write("(")
        writer.write(self.shape)
        writer.write(":")
        writer.write(self.stride)
        writer.write(")")

    def sublayout[
        i: Int
    ](
        self,
        out result: RuntimeLayout[
            Self.layout[i],
            element_type=Self.element_type,
            linear_idx_type=Self.linear_idx_type,
        ],
    ):
        """Extract a nested sublayout at the specified index.

        Parameters:
            i: The index of the nested layout to extract.

        Returns:
            A `RuntimeLayout` representing the nested layout at index i.
        """
        return {
            rebind[
                RuntimeTuple[
                    Self.layout[i].shape, element_type=Self.element_type
                ]
            ](self.shape[i]),
            rebind[
                RuntimeTuple[
                    Self.layout[i].stride, element_type=Self.linear_idx_type
                ]
            ](self.stride[i]),
        }

    def dim(self, i: Int) -> Int:
        """Get the size of the dimension at the specified index.

        Args:
            i: The index of the dimension to retrieve.

        Returns:
            The size of the dimension at index `i`.
        """
        return self.shape.value[i]

    @staticmethod
    def __len__() -> Int:
        """Get the number of dimensions in the layout.

        Returns:
            The number of dimensions (rank) of the layout.
        """
        return comptime (len(Self.layout))


def coalesce[
    l: Layout,
    keep_rank: Bool = False,
](
    layout: RuntimeLayout[l, ...],
    out result: RuntimeLayout[
        coalesce_layout(l, keep_rank),
        element_type=layout.element_type,
        linear_idx_type=layout.linear_idx_type,
    ],
):
    """Coalesce adjacent dimensions in a runtime layout when possible.

    This optimizes the layout by merging adjacent dimensions when their
    relationship allows it, potentially reducing the number of dimensions.

    Parameters:
        l: The static layout type to coalesce.
        keep_rank: Whether to maintain the original rank (currently unsupported).

    Args:
        layout: The input `RuntimeLayout` to coalesce.

    Returns:
        A new `RuntimeLayout` with coalesced dimensions.
    """

    comptime assert not keep_rank, "Unsupported coalesce mode"

    var res_shape = RuntimeTuple[
        coalesce_layout(l, keep_rank).shape, element_type=layout.element_type
    ]()
    var res_stride = RuntimeTuple[
        coalesce_layout(l, keep_rank).stride,
        element_type=layout.linear_idx_type,
    ]()

    res_shape.value[0] = 1
    res_stride.value[0] = 0

    var idx = 0

    comptime for i in range(len(flatten(l.shape))):
        comptime shape = Int(l.shape[i])
        comptime stride = Int(l.stride[i])

        # If dynamic, append new mode
        if UNKNOWN_VALUE in (shape, stride):
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]
            idx += 1
            continue

        # skip their shape-1s
        if shape == 1:
            continue

        # replace our shape-1 with anything
        if res_shape.value[idx] == 1:
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]

        # merge modes if the shape*stride match
        elif res_shape.value[idx] * res_shape.value[idx] == stride:
            res_shape.value[idx] = res_shape.value[idx] * shape
        # append a new mode
        else:
            res_shape.value[idx] = layout.shape.value[i]
            res_stride.value[idx] = layout.stride.value[i]
            idx += 1

    return {res_shape, res_stride}


def make_layout[
    l1: Layout, l2: Layout, /, *, linear_idx_type: DType = .uint64
](
    a: RuntimeLayout[l1, ...],
    b: RuntimeLayout[l2, ...],
    out result: RuntimeLayout[
        make_layout_static(l1, l2),
        element_type=b.element_type,
        linear_idx_type=linear_idx_type,
    ],
):
    """Combine two runtime layouts into a single composite layout.

    This creates a new layout by concatenating the dimensions and strides of the
    input layouts.

    Parameters:
        l1: The static layout type of `a`.
        l2: The static layout type of `b`.
        linear_idx_type: The integer type of the all index calculated by the returned
                  runtime layout.

    Args:
        a: The first `RuntimeLayout` to combine.
        b: The second `RuntimeLayout` to combine.

    Returns:
        A new `RuntimeLayout` with dimensions from both input layouts.
    """

    var res_shape = RuntimeTuple[
        make_layout_static(l1, l2).shape,
        element_type=b.element_type,
    ]()
    var res_stride = RuntimeTuple[
        make_layout_static(l1, l2).stride,
        element_type=linear_idx_type,
    ]()

    comptime a_length = len(flatten(l1.shape))
    comptime b_length = len(flatten(l2.shape))

    comptime for i in range(a_length):
        res_shape.value[i] = a.shape.value[i]
        res_stride.value[i] = a.stride.value[i]

    comptime for i in range(b_length):
        res_shape.value[a_length + i] = b.shape.value[i]
        res_stride.value[a_length + i] = b.stride.value[i]

    return {res_shape, res_stride}
