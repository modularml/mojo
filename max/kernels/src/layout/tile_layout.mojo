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
"""Provides a mixed compile-time/runtime layout system for tensor memory mapping.

This module provides a layout system where some dimensions can be known at
compile time and others determined at runtime, enabling ergonomic layout
definitions while maintaining performance through compile-time specialization.

Key components:

- [`TensorLayout`](/api/mojo/layout/tile_layout/TensorLayout): Trait
  defining the interface for all mixed layouts.
- [`Layout`](/api/mojo/layout/tile_layout/Layout): Primary struct
  implementing a layout with mixed compile-time and runtime dimensions.
- [`row_major`](/api/mojo/layout/tile_layout/row_major): Create a
  row-major layout from a shape.
- [`col_major`](/api/mojo/layout/tile_layout/col_major): Create a
  column-major layout from a shape.
- [`blocked_product`](/api/mojo/layout/tile_layout/blocked_product):
  Create a hierarchical blocked layout from block and tiler layouts.
- [`zipped_divide`](/api/mojo/layout/tile_layout/zipped_divide): Divide
  a layout into inner and outer components by a tile shape.
- [`coalesce`](/api/mojo/layout/tile_layout/coalesce): Simplify a
  layout by merging dimensions with contiguous strides.

You can import these APIs from the `layout` package:

```mojo
from layout.tile_layout import Layout, TensorLayout, row_major, col_major
```
"""

import std.sys
from std.collections.string.string import _calc_initial_buffer_size_int32
from std.math.uutils import udivmod_unchecked

from layout.coord import (
    ComptimeInt,
    Idx,
    Coord,
    CoordLike,
    DynamicCoord,
    crd2idx,
    idx2crd,
    _IntToComptimeInt,
    _CoordToDynamic,
    _NestedDynamicCoord,
    _Divide,
    _Multiply,
    _MultiplyByScalar,
    _Flattened,
    _IsNotTuplePredicate,
)
from .int_tuple import IntTuple, coord_to_int_tuple

from .layout import Layout as LegacyLayout


@always_inline("nodebug")
def _divide_by_stride[StrideType: CoordLike](idx: Int, stride_val: Int) -> Int:
    """Divide idx by stride, specializing for compile-time known values."""
    comptime if StrideType.is_static_value and StrideType.static_value == 1:
        return idx
    elif StrideType.is_static_value:
        var q, _ = udivmod_unchecked(idx, StrideType.static_value)
        return q
    else:
        var q, _ = udivmod_unchecked(idx, stride_val)
        return q


@always_inline("nodebug")
def _mod_by_shape[ShapeType: CoordLike](val: Int, shape_val: Int) -> Int:
    """Compute val % shape, specializing for compile-time known values.

    When the shape is compile-time known, uses the static value so LLVM can
    constant-fold. Special-cases shape==1 to return 0 directly.
    """
    comptime if ShapeType.is_static_value and ShapeType.static_value == 1:
        return 0
    elif ShapeType.is_static_value:
        _, var r = udivmod_unchecked(val, ShapeType.static_value)
        return r
    else:
        _, var r = udivmod_unchecked(val, shape_val)
        return r


trait TensorLayout(TrivialRegisterPassable):
    """Trait defining the interface for mixed compile-time/runtime layouts.

    Implementors map logical multi-dimensional coordinates to linear memory
    indices, with support for dimensions that are known at compile time or
    determined at runtime.
    """

    comptime rank: Int
    """The number of dimensions in the layout."""
    comptime flat_rank: Int
    """The number of dimensions after flattening nested coordinates."""
    comptime shape_known: Bool
    """Whether all shape dimensions are known at compile time."""
    comptime stride_known: Bool
    """Whether all stride dimensions are known at compile time."""
    comptime all_dims_known: Bool = Self.shape_known and Self.stride_known
    """Whether all shape and stride dimensions are known at compile time."""
    comptime static_shape[i: Int]: Int
    """Returns the compile-time value of the i-th shape dimension.

    Parameters:
        i: The dimension index.
    """
    comptime static_stride[i: Int]: Int
    """Returns the compile-time value of the i-th stride dimension.

    Parameters:
        i: The dimension index.
    """

    comptime static_product: Int
    """The compile-time product of all shape dimensions."""

    comptime static_cosize: Int
    """The compile-time size of the memory region spanned by the layout."""

    comptime __shape_types: TypeList[Trait=CoordLike, _]._mlir_type
    comptime __stride_types: TypeList[Trait=CoordLike, _]._mlir_type

    comptime _shape_types: TypeList[Trait=CoordLike, Self.__shape_types]
    comptime _stride_types: TypeList[Trait=CoordLike, Self.__stride_types]

    def shape[i: Int](self) -> Self._shape_types[i]:
        """Returns the i-th shape dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The shape value for dimension `i`.
        """
        ...

    def stride[i: Int](self) -> Self._stride_types[i]:
        """Returns the i-th stride dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The stride value for dimension `i`.
        """
        ...

    def product(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        Returns:
            The product of all shape dimensions.
        """
        ...

    def size(self) -> Int:
        """Returns the total number of elements. Alias for `product()`.

        Returns:
            The product of all shape dimensions.
        """
        ...

    def __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = .int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        Parameters:
            index_type: The coordinate type.
            linear_idx_type: The data type for the returned linear index.

        Args:
            index: The logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        ...

    def idx2crd[
        *,
        out_dtype: DType = .int64,
    ](self, idx: Int) -> Coord[
        *_NestedDynamicCoord[out_dtype, *Self._shape_types]
    ]:
        """Maps a linear memory index back to logical coordinates.

        This is the inverse of `__call__` (crd2idx). Given a linear index,
        it computes the corresponding multi-dimensional coordinates.
        For hierarchical layouts (e.g. from `zipped_divide`), the result
        preserves the nested coordinate structure.

        Parameters:
            out_dtype: The data type for the output coordinate values.

        Args:
            idx: The linear memory index to convert to coordinates.

        Returns:
            A Coord containing the logical coordinates corresponding to
            the linear index. For nested layouts, the result mirrors the
            shape nesting with Scalar leaves.

        Examples:
            For a layout with shape (3, 4) and row-major strides:
            - layout.idx2crd(0) returns (0, 0).
            - layout.idx2crd(5) returns (1, 1).
            - layout.idx2crd(11) returns (2, 3).
        """
        ...

    def shape_coord(self) -> Coord[*Self._shape_types]:
        """Returns the full shape as a `Coord`.

        Returns:
            A Coord containing all shape dimensions.
        """
        ...

    def stride_coord(self) -> Coord[*Self._stride_types]:
        """Returns the full stride as a `Coord`.

        Returns:
            A Coord containing all stride dimensions.
        """
        ...

    def transpose(
        self,
    ) -> Layout[Self._shape_types.reverse(), Self._stride_types.reverse()]:
        """Transposes the layout by reversing the order of dimensions.

        For an n-dimensional layout, this reverses the order of both shapes
        and strides. For 2D layouts, this swaps rows and columns.

        Returns:
            A new Layout with transposed dimensions.
        """
        ...

    def make_dynamic[
        dtype: DType
    ](self) -> Layout[
        _CoordToDynamic[dtype, Self._shape_types],
        _CoordToDynamic[dtype, Self._stride_types],
    ]:
        """Converts all dimensions to runtime values of the given dtype.

        Parameters:
            dtype: The data type for the resulting `Scalar` values.

        Returns:
            A new Layout with all dimensions as `Scalar[dtype]`.
        """
        ...


comptime RowMajorLayout[*shape_types: CoordLike] = Layout[
    shape_types, _RowMajor[*shape_types]
]
"""A `Layout` with row-major (C-order) strides for a flat shape. For nested
shapes use `RowMajorNestedLayout`.

Parameters:
    shape_types: The types for the shape dimensions.
"""


comptime RowMajorNestedLayout[*shape_types: CoordLike] = Layout[
    shape_types, _RowMajorNested[*shape_types]
]
"""A `Layout` with row-major strides for a nested shape (CuTe semantics).

For shape `((a, b), (c, d))` the strides are `((b*c*d, c*d), (d, 1))`:
row-major over the flattened shape, re-nested.

Parameters:
    shape_types: The types for the shape dimensions.
"""


struct Layout[
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
](ImplicitlyCopyable, TensorLayout, TrivialRegisterPassable, Writable):
    """A layout that supports mixed compile-time and runtime dimensions.

    This layout provides a unified interface for layouts where some dimensions
    are known at compile time and others are determined at runtime. It enables
    more ergonomic layout definitions while maintaining performance.

    A Layout's shape and strides must be non-negative.

    Parameters:
        shape_types: The types for the shape dimensions.
        stride_types: The types for the stride dimensions.
    """

    var _shape: Coord[*Self.shape_types]
    """The shape of the layout as a Coord."""

    var _stride: Coord[*Self.stride_types]
    """The stride of the layout as a Coord."""

    comptime rank = Self.shape_types.length
    """The number of dimensions in the layout."""
    comptime flat_rank = _Flattened[*Self.shape_types].length
    """The number of dimensions after flattening nested coordinates."""
    comptime shape_known = Coord[*Self.shape_types].all_dims_known
    """Whether all shape dimensions are known at compile time."""
    comptime stride_known = Coord[*Self.stride_types].all_dims_known
    """Whether all stride dimensions are known at compile time."""
    comptime _flat_shape_types = _Flattened[*Self.shape_types]
    comptime _flat_stride_types = _Flattened[*Self.stride_types]
    comptime static_shape[i: Int]: Int = Self._flat_shape_types[i].static_value
    """Returns the compile-time value of the i-th flattened shape dimension.

    Parameters:
        i: The dimension index.
    """
    comptime static_stride[i: Int]: Int = Self._flat_stride_types[
        i
    ].static_value
    """Returns the compile-time value of the i-th flattened stride dimension.

    Parameters:
        i: The dimension index.
    """

    comptime _shape_types = Self.shape_types
    comptime _stride_types = Self.stride_types
    comptime __shape_types = Self.shape_types.values
    comptime __stride_types = Self.stride_types.values

    comptime static_product = Coord[*Self._flat_shape_types]().static_product
    """The compile-time product of all shape dimensions (handles nested Coords)."""

    comptime static_cosize = _StaticCosize[
        Self._flat_shape_types, Self._flat_stride_types
    ]
    """The compile-time size of the memory region spanned by the layout."""

    @always_inline("nodebug")
    def __init__(out self):
        """Default-initialize a layout from its compile-time type parameters.

        Each dimension is initialized to its default value: compile-time
        dimensions (`ComptimeInt`) get their static value, runtime dimensions
        (`Scalar`) get 0. This is useful for constructing a fully-static
        layout purely from its type, e.g. ``UpcastLayout[MyLayout, 2]()``.
        """
        self._shape = Coord[*Self.shape_types]()
        self._stride = Coord[*Self.stride_types]()

    @always_inline("nodebug")
    def __init__(
        out self,
        shape: Coord[*Self.shape_types],
        stride: Coord[*Self.stride_types],
    ):
        """Initialize a layout with shape and stride.

        Args:
            shape: The shape as a Coord.
            stride: The stride as a Coord.
        """
        comptime assert (
            type_of(shape).__len__() == type_of(stride).__len__()
        ), String(
            (
                "Shape and stride must have the same length, but got shape"
                " length: "
            ),
            type_of(shape).__len__(),
            " stride length: ",
            type_of(stride).__len__(),
        )
        self._shape = shape
        self._stride = stride

    def __call__[
        index_type: CoordLike,
        *,
        linear_idx_type: DType = .int64,
    ](self, index: index_type) -> Scalar[linear_idx_type]:
        """Maps a logical coordinate to a linear memory index.

        Supports hierarchical indexing where the coordinate structure can
        differ from the shape structure. For a layout with shape (4, (3, 2)):

        - (1, (1, 1)): exact structure match, each element maps directly.
        - (1, 1): rank-matching, the scalar 1 is decomposed within the
          nested (3, 2) sub-dimension.
        - (1): scalar index decomposed across all dimensions.

        Parameters:
            index_type: The coordinate type.
            linear_idx_type: The data type for the returned linear index.

        Args:
            index: The logical coordinates to map.

        Returns:
            The linear memory index corresponding to the given coordinates.
        """
        comptime index_len = index_type.__len__()

        comptime if index_len == Self.rank:
            # Hierarchical: each coord element maps to one shape dimension.
            # If a shape dimension is nested (e.g., (3, 2)) and the
            # corresponding coord element is a scalar, crd2idx decomposes
            # the scalar within that sub-dimension.
            return crd2idx[out_type=linear_idx_type](
                index, self._shape, self._stride
            )
        elif index_type.is_tuple and index_len > Self.rank:
            # More coord elements than shape dimensions: flatten the coord
            # and strides, then compute a direct element-wise dot product.
            var flat_idx = index.tuple().flatten()
            var flat_stride = self._stride.flatten()
            var result: Scalar[linear_idx_type] = 0

            comptime flat_len = type_of(flat_idx).__len__()
            comptime for i in range(flat_len):
                result += Scalar[linear_idx_type](flat_idx[i].value()) * Scalar[
                    linear_idx_type
                ](flat_stride[i].value())

            return result
        else:
            # Scalar or single-element coord: decompose against full shape.
            return crd2idx[out_type=linear_idx_type](
                index, self._shape, self._stride
            )

    @always_inline("nodebug")
    def idx2crd[
        *,
        out_dtype: DType = .int64,
    ](self, idx: Int) -> Coord[
        *_NestedDynamicCoord[out_dtype, *Self.shape_types]
    ]:
        """Maps a linear memory index back to logical coordinates.

        This is the inverse of `__call__` (crd2idx). Given a linear index,
        it computes the corresponding multi-dimensional coordinates using
        the per-element formula: ``coord[i] = (idx // stride[i]) % shape[i]``.
        For hierarchical layouts (e.g. from `zipped_divide`), the result
        preserves the nested coordinate structure.

        Parameters:
            out_dtype: The data type for the output coordinate values.

        Args:
            idx: The linear memory index to convert to coordinates.

        Returns:
            A Coord containing the logical coordinates corresponding to
            the linear index. For nested layouts, the result mirrors the
            shape nesting with Scalar leaves.

        Examples:
            For a layout with shape (3, 4) and row-major strides:
            - layout.idx2crd(0) returns (0, 0).
            - layout.idx2crd(5) returns (1, 1).
            - layout.idx2crd(11) returns (2, 3).
        """
        comptime ResultType = Coord[
            *_NestedDynamicCoord[out_dtype, *Self.shape_types]
        ]
        var result = ResultType()
        var shape_t = self._shape.tuple()
        var stride_t = self._stride.tuple()

        comptime for i in range(Self.rank):
            comptime if Self.shape_types[i].is_tuple:
                # Nested dimension: compute sub-coordinates.
                comptime sub_rank = Self.shape_types[i].__len__()
                comptime SubResultType = DynamicCoord[out_dtype, sub_rank]
                var sub_result = SubResultType()
                var sub_shape = shape_t[i].tuple()
                var sub_stride = stride_t[i].tuple()
                comptime for j in range(sub_rank):
                    var divided = _divide_by_stride[
                        Self.stride_types[i].ParamListType[j]
                    ](idx, Int(sub_stride[j].value()))
                    var coord_val = _mod_by_shape[
                        Self.shape_types[i].ParamListType[j]
                    ](divided, Int(sub_shape[j].value()))
                    Pointer(to=sub_result[j]).write(
                        rebind[SubResultType.element_types[j]](
                            Scalar[out_dtype](coord_val)
                        )
                    )
                Pointer(to=result[i]).write(
                    rebind[ResultType.element_types[i]](sub_result)
                )
            else:
                var divided = _divide_by_stride[Self.stride_types[i]](
                    idx, Int(stride_t[i].value())
                )
                var coord_val = _mod_by_shape[Self.shape_types[i]](
                    divided, Int(shape_t[i].value())
                )
                Pointer(to=result[i]).write(
                    rebind[ResultType.element_types[i]](
                        Scalar[out_dtype](coord_val)
                    )
                )
        return result

    @always_inline("nodebug")
    def product(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        For a layout with shape (m, n), this returns m * n, representing
        the total number of valid coordinates in the layout.

        Returns:
            The total number of elements in the layout.
        """
        return Int(self._shape.product())

    @always_inline("nodebug")
    def size(self) -> Int:
        """Returns the total number of elements in the layout's domain.

        Alias for `product()`. Compatible with the legacy Layout API.

        Returns:
            The total number of elements in the layout.
        """
        return self.product()

    @always_inline("nodebug")
    def cosize[
        linear_idx_type: DType = .int64
    ](self) -> Scalar[linear_idx_type]:
        """Returns the size of the memory region spanned by the layout.

        For a layout with shape `(m, n)` and stride `(r, s)`, this returns
        `(m-1)*r + (n-1)*s + 1`, representing the memory footprint.

        Parameters:
            linear_idx_type: The data type for the returned size value.

        Returns:
            The size of the memory region required by the layout.
        """
        return self[linear_idx_type=linear_idx_type](self.product() - 1) + 1

    @always_inline("nodebug")
    def to_layout(self) -> LegacyLayout:
        """Converts this mixed layout to a legacy `Layout` using `IntTuple`.

        Returns:
            A legacy `Layout` with the same shape and stride.
        """
        return LegacyLayout(
            coord_to_int_tuple(self._shape),
            coord_to_int_tuple(self._stride),
        )

    @staticmethod
    def to_legacy_layout() -> LegacyLayout:
        """Converts this layout type to a legacy `Layout` via type-level extraction.

        All dimensions must be known at compile time. Uses direct `IntTuple`
        construction (no `append`) for fast compile times.

        Returns:
            A legacy `Layout` with the same shape and stride.
        """
        comptime assert (
            Self.all_dims_known
        ), "to_legacy_layout requires all dimensions to be compile-time known"
        return LegacyLayout(
            _types_to_int_tuple[Self._shape_types](),
            _types_to_int_tuple[Self._stride_types](),
        )

    @always_inline("nodebug")
    def reverse(
        self,
    ) -> Layout[Self.shape_types.reverse(), Self.stride_types.reverse(),]:
        """Reverse the order of dimensions in the layout.

        Turns row-major into column-major ordering where the stride-1
        dimension comes first, enabling coalesced scalar iteration.

        Returns:
            A new Layout with shape and stride Coords reversed.
        """
        return Layout(self._shape.reverse(), self._stride.reverse())

    @always_inline("nodebug")
    def transpose(
        self,
    ) -> Layout[Self.shape_types.reverse(), Self.stride_types.reverse()]:
        """Transposes the layout by reversing the order of dimensions.

        For an n-dimensional layout, this reverses the order of both shapes
        and strides. For 2D layouts, this swaps rows and columns, converting
        row-major to column-major and vice versa.

        Returns:
            A new Layout with transposed dimensions.

        Example:

        ```mojo
        from layout.tile_layout import row_major

        var layout = row_major[3, 4]()  # shape (3,4), stride (4,1)
        var transposed = layout.transpose()  # shape (4,3), stride (1,4)
        ```
        """
        return self.reverse()

    @always_inline("nodebug")
    def make_dynamic[
        dtype: DType
    ](self) -> Layout[
        _CoordToDynamic[dtype, Self.shape_types],
        _CoordToDynamic[dtype, Self.stride_types],
    ]:
        """Convert all elements in shape and stride to Scalar[dtype].

        Parameters:
            dtype: The data type for the resulting Scalar values.

        Returns:
            A new Layout where all elements in shape and stride are
            converted to Scalar[dtype].

        Examples:
            ```mojo
            from layout.tile_layout import row_major
            var layout = row_major[3, 4]()  # All compile-time
            var dynamic = layout.make_dynamic[.int64]()
            # dynamic has Int64 for all dimensions
            ```
        """
        return Layout(
            self._shape.make_dynamic[dtype](),
            self._stride.make_dynamic[dtype](),
        )

    @always_inline("nodebug")
    def shape[i: Int](self) -> Self._shape_types[i]:
        """Returns the i-th shape dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The shape value for dimension `i`.
        """
        return self._shape[i]

    @always_inline("nodebug")
    def stride[i: Int](self) -> Self._stride_types[i]:
        """Returns the i-th stride dimension.

        Parameters:
            i: The dimension index.

        Returns:
            The stride value for dimension `i`.
        """
        return self._stride[i]

    @always_inline("nodebug")
    def shape_coord(self) -> Coord[*Self._shape_types]:
        """Returns the full shape as a `Coord`.

        Returns:
            A Coord containing all shape dimensions.
        """
        return self._shape

    @always_inline("nodebug")
    def stride_coord(self) -> Coord[*Self._stride_types]:
        """Returns the full stride as a `Coord`.

        Returns:
            A Coord containing all stride dimensions.
        """
        return self._stride

    @always_inline("nodebug")
    def write_to(self, mut writer: Some[Writer]):
        """Writes the Layout representation to a Writer.

        Args:
            writer: The object to write to.
        """
        writer.write(t"({self.shape_coord()}:{self.stride_coord()})")


# ===----------------------------------------------------------------------=== #
# Type-level Layout Conversion
# ===----------------------------------------------------------------------=== #


def _type_to_int_tuple[T: CoordLike]() -> IntTuple:
    """Convert a CoordLike type to an IntTuple via direct construction.

    For scalar types, returns IntTuple(static_value).
    For tuple types, recursively converts children using direct IntTuple
    construction (no append) for rank <= 2.
    """
    comptime if not T.is_tuple:
        return IntTuple(T.static_value)
    else:
        return _types_to_int_tuple[T.ParamListType]()


def _types_to_int_tuple[Types: TypeList[Trait=CoordLike, ...]]() -> IntTuple:
    """Convert variadic CoordLike types to an IntTuple.

    Uses direct IntTuple construction (no append) for rank 1-2.
    Falls back to append for rank > 2.
    """
    comptime N = Types.length
    comptime if N == 1:
        return _type_to_int_tuple[Types[0]]()
    elif N == 2:
        return IntTuple(
            _type_to_int_tuple[Types[0]](),
            _type_to_int_tuple[Types[1]](),
        )
    else:
        var result = IntTuple()
        comptime for i in range(N):
            result.append(_type_to_int_tuple[Types[i]]())
        return result


# This is passed a pair of [shape, stride] as CoordLike values.
comptime _StaticCosizeTabulator[
    Shapes: TypeList[Trait=CoordLike, ...],
    Strides: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: Int = (Shapes[idx].static_value - 1) * Strides[idx].static_value

comptime _AddReducer[a: Int, b: Int]: Int = a + b

comptime _StaticCosize[
    Shapes: TypeList[Trait=CoordLike, ...],
    Strides: TypeList[Trait=CoordLike, ...],
] = ParameterList.tabulate[
    Shapes.length, _StaticCosizeTabulator[Shapes, Strides, _]
]().reduce[
    1, _AddReducer
]
"""The compile-time size of the memory region spanned by the layout."""


comptime _UnwrapSingleTuple[*element_types: CoordLike] = TypeList[
    element_types[0]._ParamListType if element_types.length == 1
    and element_types[0].is_tuple else element_types.values
]()


# ===-------------------------------------------------------------------=== #
# Nested-shape support for `_RowMajor` / `_ColMajor` (CuTe `local_tile`).
#
# Strides over a nested shape `((a, b), (c, d))` = strides over the
# flattened shape `(a, b, c, d)`, re-nested to mirror the original. The
# helpers below splice the flat-strides list back into the nested shape.
# Depth-1 only; the in-tree kernel suite never goes deeper.
# ===-------------------------------------------------------------------=== #


comptime _ElementFlatRank[T: CoordLike] = (
    _Flattened[*T.ParamListType].length if T.is_tuple else 1
)
"""Number of leaf dims contributed by one (possibly nested) `CoordLike`."""


comptime _CumulativeFlatRankReducer[
    upto: Int,
    Prev: Int,
    Element: CoordLike,
    list_idx: Int,
] = Prev + _ElementFlatRank[Element] if list_idx < upto else Prev


comptime _CumulativeFlatRankAt[
    ShapeTL: TypeList[Trait=CoordLike, ...],
    upto: Int,
] = ShapeTL.reduce_idx[
    0,
    _CumulativeFlatRankReducer[upto, ...],
]
"""Flat-leaf offset of outer-mode `upto` in `ShapeTL`."""


# Re-nesting tabulators form a chain `Leaf → D1 → D2 → D3` mirroring the
# `_FlattenOnce → _Flatten2 → _Flattened` doubling: each Dk handles up to
# `k` levels of nesting under `SubShape[sub_idx]` by delegating to
# `D(k-1)` in its tuple branch. The outer entry point is `D3` (depth-3
# nesting at the entry), which is more than any in-tree kernel needs.
# Mojo comptime aliases cannot self-recurse, hence the layered names.


comptime _NestedStrideLeaf[
    SubShape: TypeList[Trait=CoordLike, ...],
    FlatStrides: TypeList[Trait=CoordLike, ...],
    base: Int,
    sub_idx: Int,
]: CoordLike = FlatStrides[base + _CumulativeFlatRankAt[SubShape, upto=sub_idx]]
"""Scalar-leaf case: pick one entry of `FlatStrides`."""


comptime _NestedStrideD1[
    SubShape: TypeList[Trait=CoordLike, ...],
    FlatStrides: TypeList[Trait=CoordLike, ...],
    base: Int,
    sub_idx: Int,
]: CoordLike = Coord[
    *TypeList.tabulate[
        Trait=CoordLike,
        SubShape[sub_idx].ParamListType.length,
        _NestedStrideLeaf[
            SubShape[sub_idx].ParamListType,
            FlatStrides,
            base + _CumulativeFlatRankAt[SubShape, upto=sub_idx],
            ...,
        ],
    ]()
] if SubShape[
    sub_idx
].is_tuple else _NestedStrideLeaf[
    SubShape, FlatStrides, base, sub_idx
]


comptime _NestedStrideD2[
    SubShape: TypeList[Trait=CoordLike, ...],
    FlatStrides: TypeList[Trait=CoordLike, ...],
    base: Int,
    sub_idx: Int,
]: CoordLike = Coord[
    *TypeList.tabulate[
        Trait=CoordLike,
        SubShape[sub_idx].ParamListType.length,
        _NestedStrideD1[
            SubShape[sub_idx].ParamListType,
            FlatStrides,
            base + _CumulativeFlatRankAt[SubShape, upto=sub_idx],
            ...,
        ],
    ]()
] if SubShape[
    sub_idx
].is_tuple else _NestedStrideLeaf[
    SubShape, FlatStrides, base, sub_idx
]


comptime _NestedStrideTabulator[
    SubShape: TypeList[Trait=CoordLike, ...],
    FlatStrides: TypeList[Trait=CoordLike, ...],
    base: Int,
    sub_idx: Int,
]: CoordLike = Coord[
    *TypeList.tabulate[
        Trait=CoordLike,
        SubShape[sub_idx].ParamListType.length,
        _NestedStrideD2[
            SubShape[sub_idx].ParamListType,
            FlatStrides,
            base + _CumulativeFlatRankAt[SubShape, upto=sub_idx],
            ...,
        ],
    ]()
] if SubShape[
    sub_idx
].is_tuple else _NestedStrideLeaf[
    SubShape, FlatStrides, base, sub_idx
]
"""Entry point: stride type for `sub_idx`-th element of `SubShape`
rooted at flat offset `base`. Handles up to depth-3 nesting under
`SubShape[sub_idx]` (four total levels including the entry's outer)."""


comptime _RowMajorMapperIdx[
    ShapeList: TypeList[Trait=CoordLike, ...],
    Prev: TypeList.of[Trait=CoordLike]._mlir_type,
    element: CoordLike,
    list_idx: Int,
] = TypeList._concat[
    TypeList.of[Trait=CoordLike, ComptimeInt[1]]().values if list_idx
    == 0 else (
        TypeList.of[
            Trait=CoordLike,
            Scalar[
                ShapeList[list_idx - 1]
                .DTYPE if not ShapeList[list_idx - 1]
                .is_static_value else TypeList[Prev]()[0]
                .DTYPE
            ],
        ]()
        .values if not ShapeList[list_idx - 1]
        .is_static_value
        or not TypeList[Prev]()[0]
        .is_static_value else TypeList.of[
            Trait=CoordLike,
            ComptimeInt[
                ShapeList[list_idx - 1].static_value
                * TypeList[Prev]()[0].static_value
            ],
        ]()
        .values
    ),
    Prev,
]().values


comptime _AnyTuple[*element_types: CoordLike] = (
    not element_types.all[_IsNotTuplePredicate]()
)
"""True iff `element_types` contains at least one tuple element."""


comptime _RowMajor[*element_types: CoordLike] = TypeList[
    _UnwrapSingleTuple[*element_types]
    .reverse()
    .reduce_idx[
        TypeList.of[Trait=CoordLike]().values,
        _RowMajorMapperIdx[_UnwrapSingleTuple[*element_types].reverse(), ...],
    ]
]()
"""Row-major (C-order) strides for a flat shape variadic. For nested
shapes use `_RowMajorNested`."""


comptime _RowMajorNested[*element_types: CoordLike] = TypeList.tabulate[
    Trait=CoordLike,
    element_types.length,
    _NestedStrideTabulator[
        TypeList[element_types.values](),
        _RowMajor[*_Flattened[*element_types]],
        0,
        ...,
    ],
]()
"""Row-major strides for a nested shape variadic (CuTe semantics):
row-major over the flattened shape, then re-nested."""


@always_inline
def row_major(var shape: Coord) -> RowMajorLayout[*shape.element_types]:
    """Creates a row-major layout from a shape `Coord`.

    Row-major means the rightmost dimension has stride 1, and each preceding
    dimension has stride equal to the product of all following dimensions.

    For shape (M, N, K):
    - row_major strides: (N*K, K, 1)
    - col_major strides: (1, M, M*N)

    Args:
        shape: The shape as a Coord.

    Returns:
        A Layout with row-major strides.
    """
    comptime RowMajorTypes = _RowMajor[*shape.element_types]
    comptime rank = shape.element_types.length

    var strides = Tuple[*RowMajorTypes]()

    comptime for i in range(rank):
        comptime idx = rank - 1 - i  # Process in reverse order
        var stride_ptr = Pointer(to=strides[idx])

        comptime if i == 0:
            # Rightmost dimension always has stride 1.
            comptime StrideType = RowMajorTypes[idx]
            stride_ptr.write(rebind[StrideType](Idx[1]))
        else:
            # stride[i] = shape[i+1] * stride[i+1]
            comptime StrideType = RowMajorTypes[idx]

            comptime if StrideType.is_static_value:
                comptime stride_val = StrideType.static_value
                stride_ptr.write(rebind[StrideType](Idx[stride_val]))
            else:
                var stride_val = Int(shape[idx + 1].value()) * Int(
                    strides[idx + 1].value()
                )
                stride_ptr.write(
                    rebind[StrideType](
                        Scalar[StrideType.DTYPE](
                            Scalar[StrideType.DTYPE](stride_val)
                        )
                    )
                )

    return {shape, Coord(strides^)}


@always_inline
def row_major[
    *element_types: CoordLike
](var *elements: *element_types) -> RowMajorLayout[*element_types]:
    """Creates a row-major layout from a shape `Coord`.

    Row-major means the rightmost dimension has stride 1, and each preceding
    dimension has stride equal to the product of all following dimensions.

    Parameters:
        element_types: The variadic pack of element types that implement `CoordLike`.

    Args:
        elements: The shape as a Coord.

    Returns:
        A Layout with row-major strides.
    """

    comptime RowMajorTypes = _RowMajor[*element_types]
    comptime rank = element_types.length

    var strides = Tuple[*RowMajorTypes]()

    # Compute row-major strides on the flattened shape (flat-only — the
    # nested case has its own `row_major_nested` constructor).
    comptime for i in range(rank):
        comptime idx = rank - 1 - i  # Process in reverse order
        var stride_ptr = Pointer(to=strides[idx])

        comptime if i == 0:
            # Rightmost dimension always has stride 1.
            comptime StrideType = RowMajorTypes[idx]
            stride_ptr.write(rebind[StrideType](Idx[1]))
        else:
            # stride[i] = shape[i+1] * stride[i+1]
            comptime StrideType = RowMajorTypes[idx]

            comptime if StrideType.is_static_value:
                comptime stride_val = StrideType.static_value
                stride_ptr.write(rebind[StrideType](Idx[stride_val]))
            else:
                var stride_val = Int(elements[idx + 1].value()) * Int(
                    strides[idx + 1].value()
                )
                stride_ptr.write(
                    rebind[StrideType](
                        Scalar[StrideType.DTYPE](
                            Scalar[StrideType.DTYPE](stride_val)
                        )
                    )
                )

    return Layout(
        Coord[*element_types](*elements^), Coord[*RowMajorTypes](strides^)
    )


@always_inline("nodebug")
def row_major[*idxs: Int]() -> RowMajorLayout[*_IntToComptimeInt[*idxs]]:
    """Creates a row-major layout from compile-time shape dimensions.

    Parameters:
        idxs: The shape dimensions as compile-time integers.

    Returns:
        A Layout with row-major strides.
    """
    var shape = Coord[*_IntToComptimeInt[*idxs]]()
    return row_major(shape)


# ===----------------------------------------------------------------------=== #
# Row-major nested-shape constructor (CuTe semantics).
# ===----------------------------------------------------------------------=== #


@always_inline
def row_major_nested(
    var shape: Coord,
) -> RowMajorNestedLayout[*shape.element_types]:
    """Creates a row-major layout from a nested shape `Coord`.

    For a nested shape `((a, b), (c, d))` the result has nested strides
    `((b*c*d, c*d), (d, 1))`: row-major over the flattened shape, re-nested.

    Currently restricted to all-static (compile-time) leaf dimensions
    and one level of nesting. For flat shapes use `row_major`.

    Args:
        shape: The nested shape as a `Coord` whose top-level elements
            are themselves `Coord`s.

    Returns:
        A `Layout` with the matching nested row-major strides.
    """
    comptime RowMajorTypes = _RowMajorNested[*shape.element_types]
    comptime rank = shape.element_types.length

    var strides = Tuple[*RowMajorTypes]()

    comptime for i in range(rank):
        comptime idx = rank - 1 - i
        var stride_ptr = Pointer(to=strides[idx])
        comptime StrideType = RowMajorTypes[idx]
        stride_ptr.write(rebind[StrideType](StrideType()))

    return {shape, Coord(strides^)}


# ===----------------------------------------------------------------------=== #
# Column Major Layout
# ===----------------------------------------------------------------------=== #


comptime ColMajorLayout[shape_types: TypeList[Trait=CoordLike, ...]] = Layout[
    shape_types, _ColMajor[*shape_types]
]
"""A `Layout` with column-major (Fortran-order) strides for a flat shape.
For nested shapes use `ColMajorNestedLayout`.

Parameters:
    shape_types: The types for the shape dimensions.
"""


comptime ColMajorNestedLayout[
    shape_types: TypeList[Trait=CoordLike, ...]
] = Layout[shape_types, _ColMajorNested[*shape_types]]
"""A `Layout` with column-major strides for a nested shape (CuTe semantics).

For shape `((a, b), (c, d))` the strides are `((1, a), (a*b, a*b*c))`:
col-major over the flattened shape, re-nested.

Parameters:
    shape_types: The types for the shape dimensions.
"""

comptime _ColMajor[*element_types: CoordLike] = TypeList[
    _UnwrapSingleTuple[*element_types].reduce_idx[
        TypeList.of[Trait=CoordLike]().values,
        _ColMajorMapperIdx[_UnwrapSingleTuple[*element_types], ...],
    ]
]()
"""Column-major (Fortran-order) strides for a flat shape variadic. For
nested shapes use `_ColMajorNested`."""


comptime _ColMajorNested[*element_types: CoordLike] = TypeList.tabulate[
    Trait=CoordLike,
    element_types.length,
    _NestedStrideTabulator[
        TypeList[element_types.values](),
        _ColMajor[*_Flattened[*element_types]],
        0,
        ...,
    ],
]()
"""Column-major strides for a nested shape variadic (CuTe semantics):
col-major over the flattened shape, then re-nested."""


comptime _ColMajorMapperIdx[
    ShapeList: TypeList[Trait=CoordLike, ...],
    Prev: TypeList.of[Trait=CoordLike]._mlir_type,
    element: CoordLike,
    list_idx: Int,
] = TypeList._concat[
    Prev,
    TypeList.of[Trait=CoordLike, ComptimeInt[1]]().values if list_idx
    == 0 else (
        TypeList.of[
            Trait=CoordLike,
            Scalar[
                ShapeList[list_idx - 1]
                .DTYPE if not ShapeList[list_idx - 1]
                .is_static_value else TypeList[Prev]()[list_idx - 1]
                .DTYPE
            ],
        ]()
        .values if not ShapeList[list_idx - 1]
        .is_static_value
        or not TypeList[Prev]()[list_idx - 1]
        .is_static_value else TypeList.of[
            Trait=CoordLike,
            ComptimeInt[
                ShapeList[list_idx - 1].static_value
                * TypeList[Prev]()[list_idx - 1].static_value
            ],
        ]()
        .values
    ),
]().values


@always_inline
def col_major[
    *element_types: CoordLike
](var *elements: *element_types) -> ColMajorLayout[element_types]:
    """Create a column-major layout from variadic arguments.

    Column-major means the first dimension has stride 1, and each subsequent
    dimension has stride equal to the product of all previous dimensions.

    Parameters:
        element_types: The variadic pack of element types that implement `CoordLike`.

    Args:
        elements: The shape dimensions.

    Returns:
        A Layout with column-major strides.
    """
    return col_major(Coord[*element_types](*elements^))


@always_inline
def col_major(var shape: Coord) -> ColMajorLayout[shape.element_types]:
    """Create a column-major layout from a shape.

    Column-major means the first dimension has stride 1, and each subsequent
    dimension has stride equal to the product of all previous dimensions.

    For shape (M, N, K):
    - row_major strides: (N*K, K, 1)
    - col_major strides: (1, M, M*N)

    Args:
        shape: The shape as a Coord.

    Returns:
        A Layout with column-major strides.
    """
    comptime ColMajorTypes = _ColMajor[*shape.element_types]
    comptime rank = shape.element_types.length

    var strides = Tuple[*ColMajorTypes]()

    # Compute column-major strides (flat-only — the nested case has
    # its own `col_major_nested` constructor).
    comptime for i in range(rank):
        var stride_ptr = Pointer(to=strides[i])

        comptime if i == 0:
            # Leftmost dimension always has stride 1.
            comptime StrideType = ColMajorTypes[i]
            stride_ptr.write(rebind[StrideType](Idx[1]))
        else:
            # stride[i] = shape[i-1] * stride[i-1]
            comptime StrideType = ColMajorTypes[i]

            comptime if StrideType.is_static_value:
                comptime stride_val = StrideType.static_value
                stride_ptr.write(rebind[StrideType](Idx[stride_val]))
            else:
                var stride_val = Int(shape[i - 1].value()) * Int(
                    strides[i - 1].value()
                )
                stride_ptr.write(
                    rebind[StrideType](
                        Scalar[StrideType.DTYPE](
                            Scalar[StrideType.DTYPE](stride_val)
                        )
                    )
                )

    return Layout(shape, Coord[*ColMajorTypes](strides^))


@always_inline("nodebug")
def col_major[*idxs: Int]() -> ColMajorLayout[_IntToComptimeInt[*idxs]]:
    """Create a column-major layout from compile-time shape dimensions.

    Parameters:
        idxs: The shape dimensions as compile-time integers.

    Returns:
        A Layout with column-major strides.

    Example:

    ```mojo
    from layout.tile_layout import col_major

    var layout = col_major[3, 4]()
    # shape: (3, 4), stride: (1, 3)
    ```
    """
    var shape = Coord[*_IntToComptimeInt[*idxs]]()
    return col_major(shape)


@always_inline
def col_major_nested(
    var shape: Coord,
) -> ColMajorNestedLayout[shape.element_types]:
    """Creates a column-major layout from a nested shape `Coord`.

    For a nested shape `((a, b), (c, d))` the result has nested strides
    `((1, a), (a*b, a*b*c))`: col-major over the flattened shape, re-nested.

    Currently restricted to all-static (compile-time) leaf dimensions
    and one level of nesting. For flat shapes use `col_major`.

    Args:
        shape: The nested shape as a `Coord` whose top-level elements
            are themselves `Coord`s.

    Returns:
        A `Layout` with the matching nested column-major strides.
    """
    comptime ColMajorTypes = _ColMajorNested[*shape.element_types]
    comptime rank = shape.element_types.length

    var strides = Tuple[*ColMajorTypes]()

    comptime for i in range(rank):
        var stride_ptr = Pointer(to=strides[i])
        comptime StrideType = ColMajorTypes[i]
        stride_ptr.write(rebind[StrideType](StrideType()))

    return Layout(shape, Coord[*ColMajorTypes](strides^))


@always_inline("nodebug")
def col_major(
    idx: ComptimeInt[...],
) -> Layout[
    shape_types=Coord[type_of(idx)].element_types,
    stride_types=Coord[ComptimeInt[1]].element_types,
]:
    """Creates a 1D column-major layout from a compile-time dimension.

    Args:
        idx: The shape dimension as a `ComptimeInt`.

    Returns:
        A 1D Layout with stride 1.
    """
    return Layout(Coord(idx), Coord(Idx[1]))


@always_inline("nodebug")
def col_major(
    idx: Scalar[...],
) -> Layout[
    shape_types=Coord[type_of(idx)].element_types,
    stride_types=Coord[ComptimeInt[1]].element_types,
]:
    """Creates a 1D column-major layout from a runtime dimension.

    Args:
        idx: The shape dimension as a `Scalar`.

    Returns:
        A 1D Layout with stride 1.
    """
    return Layout(Coord(idx), Coord(Idx[1]))


def zipped_divide[
    LayoutType: TensorLayout, //, tile: Coord
](layout: LayoutType) -> ZippedDivideLayout[LayoutType, tile.element_types]:
    """Divides a layout into inner (tile) and outer (number-of-tiles) parts.

    Given a layout and a tile shape, produces a hierarchical layout where the
    inner component has the tile shape with the original strides, and the outer
    component has shape = original_shape / tile with scaled strides.

    Parameters:
        LayoutType: The type of the input layout.
        tile: The tile shape to divide by.

    Args:
        layout: The layout to divide.

    Returns:
        A `ZippedDivideLayout` with inner and outer components.
    """
    var shape = layout.shape_coord()
    var outer_shape = Coord[
        *_Divide[LayoutType._shape_types, tile.element_types]
    ]()
    var outer_stride = Coord[
        *_Multiply[LayoutType._stride_types, tile.element_types]
    ]()
    var inner_shape = tile
    var inner_stride = layout.stride_coord()

    comptime for i in range(outer_shape.rank):
        comptime if (
            outer_shape.element_types[i].is_value
            and not outer_shape.element_types[i].is_static_value
        ):
            outer_shape[i] = rebind[outer_shape.element_types[i]](
                Scalar[outer_shape.element_types[i].DTYPE](
                    Int(shape[i].value()) // Int(tile[i].value())
                )
            )

        comptime if (
            outer_stride.element_types[i].is_value
            and not outer_stride.element_types[i].is_static_value
        ):
            outer_stride[i] = rebind[outer_stride.element_types[i]](
                Scalar[outer_stride.element_types[i].DTYPE](
                    Int(inner_stride[i].value()) * Int(tile[i].value())
                )
            )
    var out_layout = Layout(
        Coord(inner_shape, outer_shape), Coord(inner_stride, outer_stride)
    )
    return out_layout


comptime ZippedDivideLayout[
    LayoutType: TensorLayout,
    tile: TypeList[Trait=CoordLike, ...],
] = Layout[
    Coord[
        Coord[*tile],  # inner_shape = tile
        Coord[
            *_Divide[LayoutType._shape_types, tile]
        ],  # outer_shape = shape / tile
    ].element_types,
    Coord[
        Coord[*LayoutType._stride_types],  # inner_stride = original stride
        Coord[
            *_Multiply[LayoutType._stride_types, tile]
        ],  # outer_stride = stride * tile
    ].element_types,
]
"""Type alias for the result of `zipped_divide`.

Splits a layout into inner (tile-sized) and outer (number-of-tiles)
components. The result is a 2-level hierarchical layout where:

- ``inner_shape  = tile``
- ``outer_shape  = shape / tile``
- ``inner_stride = original stride``
- ``outer_stride = stride * tile``

For fully-static layouts, this can be used directly at the type level:

```mojo
comptime result = ZippedDivideLayout[type_of(my_layout), tile.element_types]()
```

Parameters:
    LayoutType: The input layout type.
    tile: Shape types of the tile used to divide the layout.
"""


# ===----------------------------------------------------------------------=== #
# Blocked Product
# ===----------------------------------------------------------------------=== #

comptime _BlockedProductShapeTabulator[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    idx: Int,
]: CoordLike = Coord[
    BlockLayoutType._shape_types[idx],
    TilerLayoutType._shape_types[idx],
]

comptime _BlockedProductShapeTypes[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
] = TypeList.tabulate[
    Trait=CoordLike,
    BlockLayoutType._stride_types.length,
    _BlockedProductShapeTabulator[
        BlockLayoutType,
        TilerLayoutType,
        ...,
    ],
]()

comptime _BlockedProductStrideTabulator[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    block_cosize: Int,
    idx: Int,
]: CoordLike = Coord[
    BlockLayoutType._stride_types[idx],
    ComptimeInt[block_cosize * TilerLayoutType._stride_types[idx].static_value],
]

comptime _BlockedProductStrideTypes[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
] = TypeList.tabulate[
    Trait=CoordLike,
    BlockLayoutType._stride_types.length,
    _BlockedProductStrideTabulator[
        BlockLayoutType,
        TilerLayoutType,
        Coord[*BlockLayoutType._shape_types].static_product,
        ...,
    ],
]()

comptime _CoalescedBlockedShapeTabulator[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    block_cosize: Int,
    idx: Int,
]: CoordLike = ComptimeInt[
    # Coalesce: merge into flat ComptimeInt[block_s * tiler_s].
    BlockLayoutType._shape_types[idx].static_value
    * TilerLayoutType._shape_types[idx].static_value
] if _can_coalesce_mode[
    BlockLayoutType._shape_types[idx].static_value,
    BlockLayoutType._stride_types[idx].static_value,
    block_cosize * TilerLayoutType._stride_types[idx].static_value,
] else Coord[  # No coalesce: keep nested Coord[*(block_s, tiler_s)].
    BlockLayoutType._shape_types[idx],
    TilerLayoutType._shape_types[idx],
]


comptime _CoalescedBlockedProductShapeTypes[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
] = TypeList.tabulate[
    Trait=CoordLike,
    BlockLayoutType._shape_types.length,
    _CoalescedBlockedShapeTabulator[
        BlockLayoutType,
        TilerLayoutType,
        Coord[*BlockLayoutType._shape_types].static_product,
        ...,
    ],
]()


comptime _CoalescedBlockedStrideTabulator[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    block_cosize: Int,
    idx: Int,
]: CoordLike = BlockLayoutType._stride_types[idx] if _can_coalesce_mode[
    # Coalesce: stride is the inner (block) stride.
    BlockLayoutType._shape_types[idx].static_value,
    BlockLayoutType._stride_types[idx].static_value,
    block_cosize * TilerLayoutType._stride_types[idx].static_value,
] else Coord[
    # No coalesce: keep nested Coord[*(block_d, outer_d)].
    BlockLayoutType._stride_types[idx],
    ComptimeInt[block_cosize * TilerLayoutType._stride_types[idx].static_value],
]

comptime _CoalescedBlockedProductStrideTypes[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
] = TypeList.tabulate[
    Trait=CoordLike,
    BlockLayoutType._stride_types.length,
    _CoalescedBlockedStrideTabulator[
        BlockLayoutType,
        TilerLayoutType,
        Coord[*BlockLayoutType._shape_types].static_product,
        ...,
    ],
]()

comptime BlockedProductLayout[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    coalesce_output: Bool = False,
] = Layout[
    TypeList[
        _CoalescedBlockedProductShapeTypes[BlockLayoutType, TilerLayoutType]
        .values if coalesce_output else _BlockedProductShapeTypes[
            BlockLayoutType, TilerLayoutType
        ]
        .values
    ](),
    TypeList[
        _CoalescedBlockedProductStrideTypes[BlockLayoutType, TilerLayoutType]
        .values if coalesce_output else _BlockedProductStrideTypes[
            BlockLayoutType, TilerLayoutType
        ]
        .values
    ](),
]
"""Type alias for blocked product layout.

Creates a hierarchical layout by combining a block (inner) layout with a
tiler (outer) layout. The result zips corresponding dimensions so that
each mode ``i`` pairs ``block[i]`` with ``tiler[i]``:

- ``shape[i]  = (block.shape[i],  tiler.shape[i])``
- ``stride[i] = (block.stride[i], block.cosize * tiler.stride[i])``

When ``coalesce_output`` is True, contiguous inner/outer pairs per mode
are merged into flat dimensions (``block_shape[i] * block_stride[i] ==
outer_stride[i]``). This corresponds to the old
``blocked_product(..., coalesce_output=True)`` with ``keep_rank=True``.

For fully-static layouts, this can be used directly at the type level:

```mojo
comptime result = BlockedProductLayout[type_of(block), type_of(tiler)]()
comptime coalesced = BlockedProductLayout[
    type_of(block), type_of(tiler), coalesce_output=True
]()
```

Parameters:
    BlockLayoutType: The inner block layout type.
    TilerLayoutType: The outer tiler layout type.
    coalesce_output: Whether to coalesce contiguous modes. Default is False.
"""


comptime _can_coalesce_mode[
    block_shape: Int, block_stride: Int, outer_stride: Int
] = block_shape * block_stride == outer_stride
"""Check if a blocked-product mode can be coalesced.

A mode can be coalesced when the inner (block) elements are contiguous
with the outer (tiler) elements, i.e., ``block_shape * block_stride ==
outer_stride``.

Args:
    block_shape: The block shape for this mode.
    block_stride: The block stride for this mode.
    outer_stride: The outer stride (``block.cosize * tiler.stride``)
        for this mode.

Returns:
    True if the mode can be merged into a single flat dimension.
"""


def blocked_product[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    //,
](block: BlockLayoutType, tiler: TilerLayoutType) -> BlockedProductLayout[
    BlockLayoutType, TilerLayoutType
]:
    """Creates a blocked layout by combining a block and tiler layout.

    This function creates a hierarchical blocked layout where each element
    of the tiler layout is replaced by a block. This is useful for creating
    tiled layouts for efficient cache utilization.

    Parameters:
        BlockLayoutType: The type of the block layout.
        TilerLayoutType: The type of the tiler layout.

    Args:
        block: The inner layout defining the structure of each tile.
        tiler: The outer layout defining the arrangement of tiles.

    Returns:
        A new layout representing the blocked structure.

    Example:

    ```mojo
    from layout.tile_layout import row_major, blocked_product

    # Create a 2x2 block layout
    var block = row_major[2, 2]()
    # Create a 2x3 tiler (2 rows, 3 cols of blocks)
    var tiler = row_major[2, 3]()
    # Create blocked layout
    var blocked = blocked_product(block, tiler)
    # Result: shape ((2,2), (2,3)), stride ((2,12), (1,4))
    ```
    """
    comptime BlockShape = Coord[*BlockLayoutType._shape_types]
    comptime OuterStrideTypes = _MultiplyByScalar[
        TilerLayoutType._stride_types,
        BlockShape.static_product,
    ]

    # Build inner shape/stride from block layout
    var inner_shape = block.shape_coord()
    var inner_stride = block.stride_coord()

    # Build outer shape from tiler layout
    var outer_shape = tiler.shape_coord()

    # Build outer stride = block.cosize * tiler.stride
    var outer_stride = Coord[*OuterStrideTypes]()

    comptime for i in range(outer_shape.rank):
        comptime if OuterStrideTypes[i].is_static_value:
            Pointer(to=outer_stride[i]).write(
                rebind[OuterStrideTypes[i]](
                    ComptimeInt[OuterStrideTypes[i].static_value]()
                )
            )
        else:
            var block_cosize = Int(block.shape_coord().product())
            Pointer(to=outer_stride[i]).write(
                rebind[OuterStrideTypes[i]](
                    Scalar[OuterStrideTypes[i].DTYPE](
                        Int(tiler.stride_coord()[i].value()) * block_cosize
                    )
                )
            )

    # Zip per dimension: mode i = (block[i], tiler[i])
    comptime ResultType = BlockedProductLayout[BlockLayoutType, TilerLayoutType]
    var result_shape = Coord[*ResultType._shape_types]()
    var result_stride = Coord[*ResultType._stride_types]()

    comptime for i in range(inner_shape.rank):
        Pointer(to=result_shape[i]).write(
            rebind[ResultType._shape_types[i]](
                Coord(inner_shape[i], outer_shape[i])
            )
        )
        Pointer(to=result_stride[i]).write(
            rebind[ResultType._stride_types[i]](
                Coord(inner_stride[i], outer_stride[i])
            )
        )

    return Layout(result_shape, result_stride)


def blocked_product[
    BlockLayoutType: TensorLayout,
    TilerLayoutType: TensorLayout,
    //,
    *,
    coalesce_output: Bool,
](block: BlockLayoutType, tiler: TilerLayoutType) -> BlockedProductLayout[
    BlockLayoutType, TilerLayoutType, coalesce_output
]:
    """Creates a blocked layout with optional output coalescing.

    This overload accepts a ``coalesce_output`` keyword parameter.  When
    True, contiguous inner/outer dimension pairs are merged into flat
    dimensions, reducing the layout rank where possible.

    Parameters:
        BlockLayoutType: The type of the block layout.
        TilerLayoutType: The type of the tiler layout.
        coalesce_output: When True, merge contiguous inner/outer pairs.

    Args:
        block: The inner layout defining the structure of each tile.
        tiler: The outer layout defining the arrangement of tiles.

    Returns:
        A new layout representing the blocked structure, coalesced if
        requested.

    Example:

    ```mojo
    from layout.tile_layout import row_major, blocked_product

    var block = row_major[4]()
    var tiler = row_major[3]()
    # Coalesced: shape (12,), stride (1,) instead of ((4,), (3,))
    var coalesced = blocked_product[coalesce_output=True](block, tiler)
    ```
    """
    return BlockedProductLayout[
        BlockLayoutType, TilerLayoutType, coalesce_output
    ]()


# ===----------------------------------------------------------------------=== #
# Upcast / Downcast
# ===----------------------------------------------------------------------=== #


def _comptime_shape_div(a: Int, b: Int) -> Int:
    """Compile-time shape_div: ``a // b`` if divisible, else ``signum(a * b)``.

    This mirrors the int-int case of the legacy ``shape_div`` function.
    Used by the upcast type-level reducers to compute result types.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        ``a // b`` when ``a`` is evenly divisible by ``b``, otherwise
        1 if ``a * b > 0`` else -1.
    """
    if a % b == 0:
        return a // b
    return 1 if a * b > 0 else -1


comptime _UpcastStrideReducer[
    factor: Int,
    coord: CoordLike,
]: CoordLike = ComptimeInt[
    _comptime_shape_div(coord.static_value, factor)
] if coord.is_static_value else Scalar[
    coord.DTYPE
]
"""Computes the type for each upcast stride dimension.

For a compile-time stride ``d``, the result type is
``ComptimeInt[shape_div(d, factor)]``. For a runtime stride, the result
is ``Scalar``.
"""


comptime _UpcastStrideTypes[
    factor: Int,
    stride_types: TypeList[Trait=CoordLike, ...],
] = stride_types.map[_UpcastStrideReducer[factor, _]]()
"""The stride types after upcast by ``factor``."""


comptime _UpcastShapeTabulator[
    factor: Int,
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = ComptimeInt[
    _comptime_shape_div(
        shape_types[idx].static_value,
        _comptime_shape_div(factor, stride_types[idx].static_value),
    )
] if shape_types[
    idx
].is_static_value and stride_types[
    idx
].is_static_value else Scalar[
    shape_types[idx].DTYPE
]
"""Computes the type for each upcast shape dimension.

For compile-time shape ``s`` and stride ``d``, the result type is
``ComptimeInt[shape_div(s, shape_div(factor, d))]``. When either is
runtime, the result is ``Scalar``.
"""


comptime _UpcastShapeTypes[
    factor: Int,
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    shape_types.length,
    _UpcastShapeTabulator[factor, shape_types, stride_types, ...],
]()
"""The shape types after upcast by ``factor``."""


comptime UpcastLayout[
    LayoutType: TensorLayout,
    factor: Int,
] = Layout[
    _UpcastShapeTypes[
        factor, LayoutType._shape_types, LayoutType._stride_types
    ],
    _UpcastStrideTypes[factor, LayoutType._stride_types],
]
"""Type alias for the result of `upcast`.

Fuses ``factor`` consecutive elements per dimension, producing a layout
with coarser granularity. For fully-static layouts, this can be used
directly at the type level without calling `upcast`:

```mojo
comptime result = UpcastLayout[type_of(my_layout), 2]()
```

Parameters:
    LayoutType: The input layout type.
    factor: The number of consecutive elements to fuse.
"""


@always_inline("nodebug")
def _runtime_shape_div(a: Int, b: Int) -> Int:
    """Runtime shape_div: ``a // b`` if divisible, else ``signum(a * b)``.

    Runtime counterpart of ``_comptime_shape_div``, used in the ``upcast``
    function body for dimensions whose values are only known at runtime.

    Args:
        a: The dividend.
        b: The divisor.

    Returns:
        ``a // b`` when ``a`` is evenly divisible by ``b``, otherwise
        1 if ``a * b > 0`` else -1.
    """
    if a % b == 0:
        return a // b
    return 1 if a * b > 0 else -1


def upcast[
    LayoutType: TensorLayout, factor: Int, //
](layout: LayoutType) -> UpcastLayout[LayoutType, factor]:
    """Fuses consecutive elements in a layout to create a coarser layout.

    This is useful for converting between different data type granularities.
    For example, if a layout describes byte-level offsets and you want to
    treat every 2 bytes as one ``bf16`` element, use ``upcast[2](layout)``.

    For each dimension with shape ``s`` and stride ``d``:
    - ``new_stride = shape_div(d, factor)``
    - ``new_shape  = shape_div(s, shape_div(factor, d))``

    where ``shape_div(a, b)`` returns ``a // b`` if ``a`` is divisible by
    ``b``, otherwise ``signum(a * b)`` (i.e., 1 for positive values).

    Parameters:
        LayoutType: The type of the input layout.
        factor: The number of consecutive elements to fuse into one.

    Args:
        layout: The layout to upcast.

    Returns:
        A new layout with adjusted shape and stride for the coarser
        granularity.

    Example:

    ```mojo
    from layout.tile_layout import row_major, upcast

    # 4x8 row-major, strides (8, 1)
    var layout = row_major[4, 8]()
    # Upcast by 2: treat pairs as single elements
    var coarser = upcast[factor=2](layout)
    # Result: shape (4, 4), strides (4, 1)
    ```
    """
    comptime ResultType = UpcastLayout[LayoutType, factor]
    comptime ResultShapeTypes = ResultType._shape_types
    comptime ResultStrideTypes = ResultType._stride_types

    var new_shape = Coord[*ResultShapeTypes]()
    var new_stride = Coord[*ResultStrideTypes]()

    comptime for i in range(LayoutType.rank):
        comptime s_static = LayoutType._shape_types[i].static_value
        comptime d_static = LayoutType._stride_types[i].static_value

        # Compute new_stride[i] = shape_div(stride[i], factor).
        comptime if ResultStrideTypes[i].is_static_value:
            Pointer(to=new_stride[i]).write(ResultStrideTypes[i]())
        else:
            Pointer(to=new_stride[i]).write(
                rebind[ResultStrideTypes[i]](
                    Scalar[ResultStrideTypes[i].DTYPE](
                        _runtime_shape_div(
                            Int(layout.stride_coord()[i].value()), factor
                        )
                    )
                )
            )

        # Compute new_shape[i] = shape_div(shape[i], shape_div(factor, stride[i])).
        comptime if ResultShapeTypes[i].is_static_value:
            Pointer(to=new_shape[i]).write(ResultShapeTypes[i]())
        else:
            Pointer(to=new_shape[i]).write(
                rebind[ResultShapeTypes[i]](
                    Scalar[ResultShapeTypes[i].DTYPE](
                        _runtime_shape_div(
                            Int(layout.shape_coord()[i].value()),
                            _runtime_shape_div(
                                factor,
                                Int(layout.stride_coord()[i].value()),
                            ),
                        )
                    )
                )
            )

    return Layout(new_shape, new_stride)


# ===----------------------------------------------------------------------=== #
# Coalesce
# ===----------------------------------------------------------------------=== #


comptime _DropLast2[
    types: TypeList.of[Trait=CoordLike]._mlir_type,
] = TypeList[
    types
]().slice[0, TypeList[types].length - 2]()
"""Remove the last two elements from a variadic."""


comptime _CoalesceReducerIdx[
    flat_stride_types: TypeList[Trait=CoordLike, ...],
    Prev: TypeList.of[Trait=CoordLike]._mlir_type,
    element: CoordLike,
    list_idx: Int,
] = Prev if element.static_value == 1 else (
    # prev_shape == 1: replace last pair with current (shape, stride)
    TypeList._concat[
        _DropLast2[Prev].values,
        TypeList.of[
            Trait=CoordLike,
            ComptimeInt[element.static_value],
            ComptimeInt[flat_stride_types[list_idx].static_value],
        ]().values,
    ]()
    .values if TypeList[Prev]()[TypeList[Prev].length - 2]
    .static_value
    == 1 else (
        # Contiguous: merge into previous (prev_shape * cur_shape, prev_stride)
        TypeList._concat[
            _DropLast2[Prev].values,
            TypeList.of[
                Trait=CoordLike,
                ComptimeInt[
                    TypeList[Prev]()[TypeList[Prev].length - 2].static_value
                    * element.static_value
                ],
                TypeList[Prev]()[TypeList[Prev].length - 1],
            ]().values,
        ]()
        .values if TypeList[Prev]()[TypeList[Prev].length - 2]
        .static_value
        * TypeList[Prev]()[TypeList[Prev].length - 1].static_value
        == flat_stride_types[list_idx].static_value else
        # Non-contiguous: append new (shape, stride) pair
        TypeList._concat[
            Prev,
            TypeList.of[
                Trait=CoordLike,
                ComptimeInt[element.static_value],
                ComptimeInt[flat_stride_types[list_idx].static_value],
            ]().values,
        ]().values
    )
)
"""Reducer for coalescing a flattened layout.

Accumulates interleaved (shape, stride) pairs in ``Prev``.  At each step
the current dimension is either skipped (shape == 1), merged into the
previous pair (contiguous strides), or appended as a new pair.

Parameters:
    flat_stride_types: Flattened stride types of the input layout.
    Prev: Accumulated interleaved (shape, stride) pairs so far.
    element: The flattened shape type at ``list_idx``.
    list_idx: Current dimension index.
"""


comptime _CoalescedInterleaved[
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList[
    _Flattened[*shape_types].reduce_idx[
        TypeList.of[Trait=CoordLike, ComptimeInt[1], ComptimeInt[0]]().values,
        _CoalesceReducerIdx[
            _Flattened[*stride_types],
            ...,
        ],
    ]
]()
"""Interleaved (shape0, stride0, shape1, stride1, ...) after coalescing."""


comptime _HalfSizeDriver[
    N: Int,
] = TypeList.splat[Trait=CoordLike, count=N // 2, type=ComptimeInt[0]]()
"""A dummy variadic of size N//2 used to drive even/odd extraction."""


comptime _ExtractEvenTabulator[
    interleaved: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = interleaved[idx * 2]
"""Extracts even-indexed elements from interleaved given as parameter."""


comptime _ExtractOddTabulator[
    interleaved: TypeList[Trait=CoordLike, ...],
    idx: Int,
]: CoordLike = interleaved[idx * 2 + 1]
"""Extracts odd-indexed elements from interleaved given as parameter."""


comptime _CoalescedShapeTypes[
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    _HalfSizeDriver[
        _CoalescedInterleaved[shape_types, stride_types].length
    ].length,
    _ExtractEvenTabulator[_CoalescedInterleaved[shape_types, stride_types], _],
]()
"""Coalesced shape types extracted from the interleaved result."""


comptime _CoalescedStrideTypes[
    shape_types: TypeList[Trait=CoordLike, ...],
    stride_types: TypeList[Trait=CoordLike, ...],
] = TypeList.tabulate[
    Trait=CoordLike,
    _HalfSizeDriver[
        _CoalescedInterleaved[shape_types, stride_types].length
    ].length,
    _ExtractOddTabulator[_CoalescedInterleaved[shape_types, stride_types], _],
]()
"""Coalesced stride types extracted from the interleaved result."""


comptime CoalesceLayout[
    LayoutType: TensorLayout,
] = Layout[
    _CoalescedShapeTypes[LayoutType._shape_types, LayoutType._stride_types],
    _CoalescedStrideTypes[LayoutType._shape_types, LayoutType._stride_types],
]
"""Type alias for the result of `coalesce`.

Simplifies a layout by merging dimensions with contiguous strides.
Adjacent flattened dimensions where ``prev_shape * prev_stride ==
current_stride`` are combined into a single dimension.  Shape-1
dimensions are dropped.

For fully-static layouts, this can be used directly at the type level:

```mojo
comptime result = CoalesceLayout[type_of(my_layout)]()
```

Parameters:
    LayoutType: The input layout type (must have all dimensions known
        at compile time).
"""


@always_inline("nodebug")
def coalesce[
    LayoutType: TensorLayout,
    //,
](layout: LayoutType) -> CoalesceLayout[LayoutType]:
    """Simplifies a layout by merging contiguous dimensions.

    Iterates over the flattened (shape, stride) pairs and:

    1. Skips shape-1 dimensions.
    2. Merges a dimension into the previous one when
       ``prev_shape * prev_stride == current_stride`` (contiguous).
    3. Otherwise starts a new dimension.

    The result is the simplest layout that maps coordinates to the same
    linear offsets as the original.

    Parameters:
        LayoutType: The type of the input layout.

    Args:
        layout: The layout to coalesce.

    Returns:
        A new layout with contiguous dimensions merged.

    Example:

    ```mojo
    from layout.tile_layout import Layout, coalesce, row_major, Idx

    # A row-major 2x4 layout has contiguous strides -> coalesces to 1D
    var layout = row_major[2, 4]()
    var coalesced = coalesce(layout)  # shape (8,), stride (1,)
    ```
    """
    return CoalesceLayout[LayoutType]()


# ===--- Weakly Compatible ---=== #
# Checks structural compatibility between two CoordLike types up to 4 levels
# of nesting.  A scalar coord is always compatible.  A tuple coord requires
# the other type to be a tuple of the same length, with all sub-elements
# recursively compatible.
#
# Because reducers cannot recurse, the check is manually unrolled into four
# depth layers (_WCPair3 → _WCPair2 → _WCPair1 → top-level), each using
# a dedicated reducer to AND-accumulate pair checks over element pairs.
# Returns ``True`` (compatible) or ``False`` (incompatible) as a
# compile-time Bool.


comptime _WCPair3[L: CoordLike, C: CoordLike]: Bool = (
    True if not C.is_tuple else (
        False if not L.is_tuple else (
            L.ParamListType.length == C.ParamListType.length
        )
    )
)
"""Depth-3 pair check (innermost): scalar coords pass, tuple coords only
check length match without descending further."""

comptime _WCPair2[L: CoordLike, C: CoordLike]: Bool = (
    True if not C.is_tuple else (
        False if not L.is_tuple else _AllEltsSatisfy[
            L.ParamListType, C.ParamListType, _WCPair3
        ]
    )
)
"""Depth-2 pair check: delegates sub-element checks to depth-3."""

comptime _WCPair1[L: CoordLike, C: CoordLike]: Bool = (
    True if not C.is_tuple else (
        False if not L.is_tuple else _AllEltsSatisfy[
            L.ParamListType, C.ParamListType, _WCPair2
        ]
    )
)
"""Depth-1 pair check: delegates sub-element checks to depth-2."""


comptime _BoolIsTrue[a: Bool]: Bool = a

comptime _tabulatePredicate[
    a: TypeList[Trait=CoordLike, ...],
    b: TypeList[Trait=CoordLike, ...],
    pred: __generator_type[LHS: CoordLike, RHS: CoordLike] Bool,
    idx: Int,
]: Bool = pred[a[idx], b[idx]]

comptime _AllEltsSatisfy[
    a: TypeList[Trait=CoordLike, ...],
    b: TypeList[Trait=CoordLike, ...],
    pred: __generator_type[LHS: CoordLike, RHS: CoordLike] Bool,
]: Bool = a.length == b.length and ParameterList.tabulate[
    a.length, _tabulatePredicate[a, b, pred, _]
]().all[
    _BoolIsTrue,
]()


comptime _WeaklyCompatible[
    layout_types: TypeList[Trait=CoordLike, ...],
    coord_types: TypeList[Trait=CoordLike, ...],
] = _AllEltsSatisfy[layout_types, coord_types, _WCPair1]
"""Top-level variadic pair check (depth 0): checks that both variadics have
the same length and all element pairs are weakly compatible."""


comptime WeaklyCompatible[
    L: TensorLayout,
    C: TypeList[Trait=CoordLike, ...],
] = _WeaklyCompatible[L._shape_types, C]
"""Check structural compatibility between a layout's shape and coordinate types.

Returns ``True`` if compatible, ``False`` otherwise.  A scalar coordinate
element is always compatible.  A tuple coordinate element requires the
corresponding layout shape element to also be a tuple of the same length,
with all sub-elements recursively compatible.  Handles up to 4 levels of
nesting; beyond that, only length equality is verified.

Parameters:
    L: The layout type whose shape structure is checked.
    C: The coordinate element types to check against.
"""


# ===----------------------------------------------------------------------=== #
# Layout Printing Utilities
# ===----------------------------------------------------------------------=== #


@no_inline
def _print_layout(layout: Layout):
    """Prints a 2D layout to the standard output.

    This function visualizes a 2D layout by printing a formatted table showing
    the memory indices for each logical coordinate.

    Args:
        layout: The 2D layout to print.
    """
    comptime assert layout.rank == 2, "_print_layout only supports 2D layouts"
    print(layout)
    var stdout = std.sys.stdout
    _format_layout(layout, stdout)


@always_inline("nodebug")
def _dim_size[T: CoordLike](dim: T) -> Int:
    """Returns the element count for a single layout dimension.

    For scalar CoordLike types (ComptimeInt, Scalar), returns the value.
    For nested Coord types, returns the product of all sub-elements.
    """
    comptime if T.is_tuple:
        comptime sub_rank = T.__len__()
        var t = dim.tuple()
        var result: Int = 1
        comptime for j in range(sub_rank):
            result *= _dim_size(t[j])
        return result
    else:
        return Int(dim.value())


@no_inline
def _format_layout[W: Writer](layout: Layout, mut writer: W):
    """Formats a 2D layout as a table and writes it to the specified writer.

    This function creates a visual representation of a 2D layout as a table
    showing the memory indices for each logical coordinate.

    Parameters:
        W: Type parameter representing a Writer implementation.

    Args:
        layout: The 2D layout to format.
        writer: The writer to output the formatted layout to.
    """
    comptime assert layout.rank == 2, "_format_layout only supports 2D layouts"

    def _write_divider(column_count: Int, cell_width: Int) {mut writer}:
        for _ in range(column_count):
            writer.write("+")
            for _ in range(cell_width):
                writer.write("-")
        writer.write("+\n")

    var rows = _dim_size(layout.shape[0]())
    var cols = _dim_size(layout.shape[1]())
    # maximum column width is based on the width of the longest label plus 2 spaces
    var idx_width = String(layout.cosize()).byte_length() + 2

    # Print column labels
    writer.write("    ")
    for n in range(cols):
        writer.write("  ")
        n.write_padded(writer, width=idx_width - 2)

        if n + 1 != cols:
            writer.write(" ")

    writer.write("\n")

    for m in range(rows):
        writer.write("    ")
        _write_divider(cols, idx_width)

        # Print row label
        m.write_padded(writer, width=2)
        writer.write("  ")

        for n in range(cols):
            writer.write("| ")
            Int(layout(Coord(m, n))).write_padded(
                writer,
                width=idx_width - 2,
            )
            writer.write(" ")
        writer.write("|\n")

    writer.write("    ")

    # Write the final horizontal dividing line
    _write_divider(cols, idx_width)
