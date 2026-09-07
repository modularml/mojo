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

from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    TileTensor,
    col_major,
    row_major,
)
from layout.tile_layout import (
    Layout,
    ZippedDivideLayout,
    BlockedProductLayout,
    UpcastLayout,
    WeaklyCompatible,
    blocked_product,
    upcast,
)
from test_utils import check_write_to
from std.testing import assert_equal, assert_true, TestSuite


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_size_cosize() raises:
    # Row-major 3x4: last element (2,3) -> 11, cosize = 12
    var layout1 = Layout(
        shape=(Idx[3], Idx[4]),
        stride=(Idx[4], Idx[1]),
    )
    assert_equal(layout1.product(), 12)
    assert_equal(layout1.cosize(), 12)

    # Layout with gaps: last element (1,1) -> 11, cosize = 12
    var layout2 = Layout(shape=(Idx[2], Idx[2]), stride=(Idx[10], Idx[1]))
    assert_equal(layout2.product(), 4)
    assert_equal(layout2.cosize(), 12)


def test_crd2idx() raises:
    var layout = Layout(
        shape=(Idx[4], Idx[2]),
        stride=(Idx[1], Idx[4]),
    )

    # Multi-dimensional coordinates
    assert_equal(layout(Coord(Idx[0], Idx[0])), 0)
    assert_equal(layout(Coord(Idx[1], Idx[0])), 1)
    assert_equal(layout(Coord(Idx[2], Idx[0])), 2)
    assert_equal(layout(Coord(Idx[3], Idx[0])), 3)
    assert_equal(layout(Coord(Idx[0], Idx[1])), 4)
    assert_equal(layout(Coord(Idx[1], Idx[1])), 5)
    assert_equal(layout(Coord(Idx[2], Idx[1])), 6)
    assert_equal(layout(Coord(Idx[3], Idx[1])), 7)

    assert_equal(layout.product(), 8)


def test_row_major() raises:
    var shape = Coord(Idx[3], Idx[4])
    var layout = row_major(shape)
    assert_true(layout.shape_coord() == shape)
    assert_true(layout.stride_coord() == Coord(Idx[4], Idx[1]))

    var shape3 = Coord(Idx[3], Idx[4], Idx[5])
    var layout3 = row_major(shape3)
    assert_true(layout3.shape_coord() == shape3)
    assert_true(layout3.stride_coord() == Coord(Idx[20], Idx[5], Idx[1]))

    var shape3_static = Coord(
        ComptimeInt[3](), ComptimeInt[4](), ComptimeInt[5]()
    )
    var layout3_static = row_major(shape3_static)
    assert_true(layout3_static.shape_coord() == shape3_static)
    assert_true(
        layout3_static.stride_coord()
        == Coord(ComptimeInt[20](), ComptimeInt[5](), ComptimeInt[1]())
    )


def test_row_major_static_constructor_empty() raises:
    var layout = row_major[]()
    assert_equal(len(layout.shape_coord()), 0)
    assert_equal(len(layout.stride_coord()), 0)


def test_row_major_static_constructor_() raises:
    var layout = row_major[1, 2, 3]()
    assert_equal(len(layout.shape_coord()), 3)
    assert_equal(len(layout.stride_coord()), 3)
    assert_equal(layout.shape[0]().value(), 1)
    assert_equal(layout.shape[1]().value(), 2)
    assert_equal(layout.shape[2]().value(), 3)
    assert_equal(layout.stride[0]().value(), 6)
    assert_equal(layout.stride[1]().value(), 3)
    assert_equal(layout.stride[2]().value(), 1)


def test_zipped_divide_layout() raises:
    # row_major[8, 16]() has shape_types = (ComptimeInt[8], ComptimeInt[16])
    # and stride_types = (ComptimeInt[16], ComptimeInt[1])
    comptime a = row_major[8, 16]()
    comptime b = Coord(Idx[2], Idx[4])
    comptime layout = ZippedDivideLayout[type_of(a), b.element_types]
    assert_equal(layout._shape_types[0].ParamListType[0].static_value, 2)
    assert_equal(layout._shape_types[0].ParamListType[1].static_value, 4)
    assert_equal(layout._shape_types[1].ParamListType[0].static_value, 4)
    assert_equal(layout._shape_types[1].ParamListType[1].static_value, 4)
    assert_equal(layout._stride_types[0].ParamListType[0].static_value, 16)
    assert_equal(layout._stride_types[0].ParamListType[1].static_value, 1)
    assert_equal(layout._stride_types[1].ParamListType[0].static_value, 32)
    assert_equal(layout._stride_types[1].ParamListType[1].static_value, 4)


# ===----------------------------------------------------------------------=== #
# TileTensor.reshape tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_reshape_static() raises:
    """Test reshaping a TileTensor with compile-time dimensions."""
    var storage = Array[Float32, 12](fill={})
    var tensor = TileTensor(storage, row_major[3, 4]()).fill(1.0)

    # Verify original shape
    assert_equal(tensor.dim[0](), 3)
    assert_equal(tensor.dim[1](), 4)
    assert_equal(tensor.num_elements(), 12)

    # Reshape to (2, 6)
    var reshaped_2x6 = tensor.reshape[2, 6]()
    assert_equal(reshaped_2x6.dim[0](), 2)
    assert_equal(reshaped_2x6.dim[1](), 6)
    assert_equal(reshaped_2x6.num_elements(), 12)

    # Reshape to (4, 3)
    var reshaped_4x3 = tensor.reshape[4, 3]()
    assert_equal(reshaped_4x3.dim[0](), 4)
    assert_equal(reshaped_4x3.dim[1](), 3)
    assert_equal(reshaped_4x3.num_elements(), 12)

    # Reshape to 1D (equivalent to coalesce)
    var reshaped_1d = tensor.reshape[12]()
    assert_equal(reshaped_1d.dim[0](), 12)
    assert_equal(reshaped_1d.num_elements(), 12)

    # Reshape to 3D
    var reshaped_3d = tensor.reshape[2, 2, 3]()
    assert_equal(reshaped_3d.dim[0](), 2)
    assert_equal(reshaped_3d.dim[1](), 2)
    assert_equal(reshaped_3d.dim[2](), 3)
    assert_equal(reshaped_3d.num_elements(), 12)


def test_tile_tensor_reshape_preserves_data() raises:
    """Test that reshape preserves the underlying data."""
    var storage = Array[Float32, 6](fill={})
    var tensor = TileTensor(storage, row_major[2, 3]())

    # Fill with distinct values
    tensor[0, 0] = 0.0
    tensor[0, 1] = 1.0
    tensor[0, 2] = 2.0
    tensor[1, 0] = 3.0
    tensor[1, 1] = 4.0
    tensor[1, 2] = 5.0

    # Reshape to (3, 2)
    var reshaped = tensor.reshape[3, 2]()

    # Verify data is preserved in row-major order
    assert_equal(reshaped[0, 0], 0.0)
    assert_equal(reshaped[0, 1], 1.0)
    assert_equal(reshaped[1, 0], 2.0)
    assert_equal(reshaped[1, 1], 3.0)
    assert_equal(reshaped[2, 0], 4.0)
    assert_equal(reshaped[2, 1], 5.0)

    # Reshape to 1D and verify
    var reshaped_1d = tensor.reshape[6]()
    assert_equal(reshaped_1d[0], 0.0)
    assert_equal(reshaped_1d[1], 1.0)
    assert_equal(reshaped_1d[2], 2.0)
    assert_equal(reshaped_1d[3], 3.0)
    assert_equal(reshaped_1d[4], 4.0)
    assert_equal(reshaped_1d[5], 5.0)


def test_tile_tensor_reshape_with_coord() raises:
    """Test reshaping with a Coord argument (potentially runtime dims)."""
    var storage = Array[Float32, 12](fill={})
    var tensor = TileTensor(storage, row_major[3, 4]()).fill(2.0)

    # Reshape using Coord with compile-time dimensions
    var reshaped = tensor.reshape(Coord(Idx[2], Idx[6]))
    assert_equal(reshaped.dim[0](), 2)
    assert_equal(reshaped.dim[1](), 6)
    assert_equal(reshaped.num_elements(), 12)

    # Reshape using Coord with runtime dimensions
    var rows = 4
    var cols = 3
    var reshaped_runtime = tensor.reshape(Coord(rows, cols))
    assert_equal(reshaped_runtime.dim[0](), 4)
    assert_equal(reshaped_runtime.dim[1](), 3)
    assert_equal(reshaped_runtime.num_elements(), 12)


def test_tile_tensor_reshape_strides() raises:
    """Test that reshaped tensor has correct row-major strides."""
    var storage = Array[Float32, 24](fill={})
    var tensor = TileTensor(storage, row_major[4, 6]()).fill(0.0)

    # Reshape to (2, 3, 4)
    var reshaped = tensor.reshape[2, 3, 4]()

    # Verify row-major strides: (12, 4, 1) for shape (2, 3, 4)
    assert_equal(reshaped.layout.stride[0]().value(), 12)
    assert_equal(reshaped.layout.stride[1]().value(), 4)
    assert_equal(reshaped.layout.stride[2]().value(), 1)


def test_tile_tensor_reshape_is_view() raises:
    """Test that reshape creates a view, not a copy."""
    var storage = Array[Float32, 6](fill={})
    var tensor = TileTensor(storage, row_major[2, 3]()).fill(0.0)

    var reshaped = tensor.reshape[3, 2]()

    # Modify through reshaped view
    reshaped[1, 0] = 42.0

    # Verify change is visible in original tensor
    # (1, 0) in (3, 2) = index 2 in row-major = (0, 2) in (2, 3)
    assert_equal(tensor[0, 2], 42.0)


# ===----------------------------------------------------------------------=== #
# TileTensor.tile tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_tile_with_int_coords() raises:
    """Test tile method with variadic Int coordinates (LayoutTensor compatible).
    """
    var storage = Array[Float32, 16](fill={})
    var tensor = TileTensor(storage, row_major[4, 4]())

    # Fill with distinct values
    for i in range(4):
        for j in range(4):
            tensor[i, j] = Float32(i * 4 + j)

    # Extract 2x2 tile at position (0, 0)
    var tile_00 = tensor.tile[2, 2](0, 0)
    assert_equal(tile_00.dim[0](), 2)
    assert_equal(tile_00.dim[1](), 2)
    assert_equal(tile_00[0, 0], 0.0)
    assert_equal(tile_00[0, 1], 1.0)
    assert_equal(tile_00[1, 0], 4.0)
    assert_equal(tile_00[1, 1], 5.0)

    # Extract 2x2 tile at position (1, 1)
    var tile_11 = tensor.tile[2, 2](1, 1)
    assert_equal(tile_11[0, 0], 10.0)
    assert_equal(tile_11[0, 1], 11.0)
    assert_equal(tile_11[1, 0], 14.0)
    assert_equal(tile_11[1, 1], 15.0)

    # Extract 2x2 tile at position (0, 1)
    var tile_01 = tensor.tile[2, 2](0, 1)
    assert_equal(tile_01[0, 0], 2.0)
    assert_equal(tile_01[0, 1], 3.0)
    assert_equal(tile_01[1, 0], 6.0)
    assert_equal(tile_01[1, 1], 7.0)


def test_tile_tensor_tile_is_view() raises:
    """Test that tile creates a view, not a copy."""
    var storage = Array[Float32, 16](fill={})
    var tensor = TileTensor(storage, row_major[4, 4]()).fill(0.0)

    var tile = tensor.tile[2, 2](1, 0)

    # Modify through tile view
    tile[0, 0] = 99.0

    # Verify change is visible in original tensor
    # tile (1, 0) with size 2x2 starts at (2, 0) in original
    assert_equal(tensor[2, 0], 99.0)


# ===----------------------------------------------------------------------=== #
# blocked_product tests
# ===----------------------------------------------------------------------=== #


def test_blocked_product_basic() raises:
    """Test blocked_product creates correct hierarchical layout.

    Example from legacy layout docs:
    - block = 2x2 row-major: shape (2,2), stride (2,1)
    - tiler = 2x3 row-major: shape (2,3), stride (3,1)
    - result: shape ((2,2), (2,3)), stride ((2,12), (1,4))
    Each mode i zips block[i] with tiler[i].
    """
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked = blocked_product(block, tiler)

    # Mode 0: (block_shape[0], tiler_shape[0]) = (2, 2)
    assert_equal(blocked.shape[0]()[0].value(), 2)
    assert_equal(blocked.shape[0]()[1].value(), 2)

    # Mode 1: (block_shape[1], tiler_shape[1]) = (2, 3)
    assert_equal(blocked.shape[1]()[0].value(), 2)
    assert_equal(blocked.shape[1]()[1].value(), 3)

    # Mode 0 stride: (block_stride[0], cosize * tiler_stride[0]) = (2, 12)
    assert_equal(blocked.stride[0]()[0].value(), 2)
    assert_equal(blocked.stride[0]()[1].value(), 12)

    # Mode 1 stride: (block_stride[1], cosize * tiler_stride[1]) = (1, 4)
    assert_equal(blocked.stride[1]()[0].value(), 1)
    assert_equal(blocked.stride[1]()[1].value(), 4)


def test_blocked_product_type_alias() raises:
    """Test BlockedProductLayout type alias directly."""
    comptime block = row_major[2, 2]()
    comptime tiler = row_major[2, 3]()
    comptime layout = BlockedProductLayout[type_of(block), type_of(tiler)]

    # Mode 0: (block_shape[0], tiler_shape[0]) = (2, 2)
    assert_equal(layout._shape_types[0].ParamListType[0].static_value, 2)
    assert_equal(layout._shape_types[0].ParamListType[1].static_value, 2)

    # Mode 1: (block_shape[1], tiler_shape[1]) = (2, 3)
    assert_equal(layout._shape_types[1].ParamListType[0].static_value, 2)
    assert_equal(layout._shape_types[1].ParamListType[1].static_value, 3)

    # Mode 0 stride: (block_stride[0], cosize * tiler_stride[0]) = (2, 12)
    # block = row_major[2, 2](): stride = (2, 1), cosize = 4
    # tiler = row_major[2, 3](): stride = (3, 1)
    assert_equal(layout._stride_types[0].ParamListType[0].static_value, 2)
    assert_equal(layout._stride_types[0].ParamListType[1].static_value, 12)

    # Mode 1 stride: (block_stride[1], cosize * tiler_stride[1]) = (1, 4)
    assert_equal(layout._stride_types[1].ParamListType[0].static_value, 1)
    assert_equal(layout._stride_types[1].ParamListType[1].static_value, 4)


# ===----------------------------------------------------------------------=== #
# col_major tests
# ===----------------------------------------------------------------------=== #


def test_col_major_2d() raises:
    """Test column-major layout for 2D shapes.

    For shape (M, N):
    - row_major strides: (N, 1)
    - col_major strides: (1, M)
    """
    var layout = col_major[3, 4]()

    # Shape should be (3, 4)
    assert_equal(layout.shape[0]().value(), 3)
    assert_equal(layout.shape[1]().value(), 4)

    # Strides should be (1, 3) for column-major
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 3)


def test_col_major_3d() raises:
    """Test column-major layout for 3D shapes.

    For shape (M, N, K):
    - row_major strides: (N*K, K, 1)
    - col_major strides: (1, M, M*N)
    """
    var layout = col_major[2, 3, 4]()

    # Shape should be (2, 3, 4)
    assert_equal(layout.shape[0]().value(), 2)
    assert_equal(layout.shape[1]().value(), 3)
    assert_equal(layout.shape[2]().value(), 4)

    # Strides should be (1, 2, 6) for column-major
    # stride[0] = 1
    # stride[1] = shape[0] = 2
    # stride[2] = shape[0] * shape[1] = 2 * 3 = 6
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 2)
    assert_equal(layout.stride[2]().value(), 6)


def test_col_major_vs_row_major() raises:
    """Test that col_major and row_major produce different strides."""
    var row = row_major[3, 4]()
    var col = col_major[3, 4]()

    # Same shape
    assert_equal(row.shape[0]().value(), col.shape[0]().value())
    assert_equal(row.shape[1]().value(), col.shape[1]().value())

    # Different strides
    # row_major: (4, 1)
    assert_equal(row.stride[0]().value(), 4)
    assert_equal(row.stride[1]().value(), 1)

    # col_major: (1, 3)
    assert_equal(col.stride[0]().value(), 1)
    assert_equal(col.stride[1]().value(), 3)


def test_col_major_with_coord() raises:
    """Test col_major with Coord argument."""
    var shape = Coord(Idx[3], Idx[4])
    var layout = col_major(shape)

    assert_equal(layout.shape[0]().value(), 3)
    assert_equal(layout.shape[1]().value(), 4)
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 3)


def test_col_major_with_runtime_dims() raises:
    """Test col_major with runtime dimensions."""
    var m = 3
    var n = 4
    var shape = Coord(m, n)
    var layout = col_major(shape)

    assert_equal(layout.shape[0]().value(), 3)
    assert_equal(layout.shape[1]().value(), 4)
    assert_equal(layout.stride[0]().value(), 1)
    assert_equal(layout.stride[1]().value(), 3)


# ===----------------------------------------------------------------------=== #
# Coord.flatten tests
# ===----------------------------------------------------------------------=== #


def test_coord_flatten_non_nested() raises:
    """Test flatten on a non-nested Coord (should be identity)."""
    var coord = Coord(Idx[1], Idx[2], Idx[3])
    var flat = coord.flatten()

    assert_equal(len(flat), 3)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)


def test_coord_flatten_single_nested() raises:
    """Test flatten on a Coord with one nested Coord."""
    var nested = Coord(Idx[5], Coord(Idx[3], Idx[2]), Idx[7])
    var flat = nested.flatten()

    # Should flatten to (5, 3, 2, 7)
    assert_equal(len(flat), 4)
    assert_equal(flat[0].value(), 5)
    assert_equal(flat[1].value(), 3)
    assert_equal(flat[2].value(), 2)
    assert_equal(flat[3].value(), 7)


def test_coord_flatten_nested_at_start() raises:
    """Test flatten with nested Coord at the beginning."""
    var nested = Coord(Coord(Idx[1], Idx[2]), Idx[3])
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3)
    assert_equal(len(flat), 3)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)


def test_coord_flatten_nested_at_end() raises:
    """Test flatten with nested Coord at the end."""
    var nested = Coord(Idx[1], Coord(Idx[2], Idx[3]))
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3)
    assert_equal(len(flat), 3)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)


def test_coord_flatten_depth2() raises:
    """Test flatten on a depth-2 nested Coord."""
    # Coord(Coord(Coord(1, 2), 3), 4) — depth 2
    var nested = Coord(Coord(Coord(Idx[1], Idx[2]), Idx[3]), Idx[4])
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3, 4)
    assert_equal(len(flat), 4)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)
    assert_equal(flat[3].value(), 4)


def test_coord_flatten_depth3() raises:
    """Test flatten on a depth-3 nested Coord."""
    # Coord(Coord(Coord(Coord(1, 2), 3), 4), 5) — depth 3
    var nested = Coord(
        Coord(Coord(Coord(Idx[1], Idx[2]), Idx[3]), Idx[4]),
        Idx[5],
    )
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3, 4, 5)
    assert_equal(len(flat), 5)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)
    assert_equal(flat[3].value(), 4)
    assert_equal(flat[4].value(), 5)


def test_coord_flatten_depth2_multiple_nested() raises:
    """Test flatten with multiple depth-2 nested subtrees."""
    # Coord(Coord(Coord(1, 2), Coord(3, 4)), Coord(Coord(5, 6), 7))
    var nested = Coord(
        Coord(Coord(Idx[1], Idx[2]), Coord(Idx[3], Idx[4])),
        Coord(Coord(Idx[5], Idx[6]), Idx[7]),
    )
    var flat = nested.flatten()

    # Should flatten to (1, 2, 3, 4, 5, 6, 7)
    assert_equal(len(flat), 7)
    assert_equal(flat[0].value(), 1)
    assert_equal(flat[1].value(), 2)
    assert_equal(flat[2].value(), 3)
    assert_equal(flat[3].value(), 4)
    assert_equal(flat[4].value(), 5)
    assert_equal(flat[5].value(), 6)
    assert_equal(flat[6].value(), 7)


def test_coord_flat_rank_deep_nesting() raises:
    """Test that flat_rank is correct for deeply nested Coords."""
    # Depth 1: Coord(Coord(1, 2), 3) -> flat_rank = 3
    comptime depth1 = Coord[
        ComptimeInt[1],
        Coord[ComptimeInt[2], ComptimeInt[3]],
    ]
    comptime assert depth1.flat_rank == 3

    # Depth 2: Coord(Coord(Coord(1, 2), 3), 4) -> flat_rank = 4
    comptime depth2 = Coord[
        Coord[
            Coord[ComptimeInt[1], ComptimeInt[2]],
            ComptimeInt[3],
        ],
        ComptimeInt[4],
    ]
    comptime assert depth2.flat_rank == 4

    # Depth 3: Coord(Coord(Coord(Coord(1, 2), 3), 4), 5) -> flat_rank = 5
    comptime depth3 = Coord[
        Coord[
            Coord[
                Coord[
                    ComptimeInt[1],
                    ComptimeInt[2],
                ],
                ComptimeInt[3],
            ],
            ComptimeInt[4],
        ],
        ComptimeInt[5],
    ]
    comptime assert depth3.flat_rank == 5


def test_coord_flatten_blocked_product_shape() raises:
    """Test flatten on shape from blocked_product."""
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked = blocked_product(block, tiler)

    # blocked.shape is Coord(Coord(2, 2), Coord(2, 3))
    var flat_shape = blocked.shape_coord().flatten()

    # Should flatten to (2, 2, 2, 3)
    assert_equal(len(flat_shape), 4)
    assert_equal(flat_shape[0].value(), 2)
    assert_equal(flat_shape[1].value(), 2)
    assert_equal(flat_shape[2].value(), 2)
    assert_equal(flat_shape[3].value(), 3)


def test_coord_flatten_blocked_product_stride() raises:
    """Test flatten on stride from blocked_product."""
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked = blocked_product(block, tiler)

    # blocked.stride is Coord(Coord(2, 12), Coord(1, 4))
    var flat_stride = blocked.stride_coord().flatten()

    # Should flatten to (2, 12, 1, 4)
    assert_equal(len(flat_stride), 4)
    assert_equal(flat_stride[0].value(), 2)
    assert_equal(flat_stride[1].value(), 12)
    assert_equal(flat_stride[2].value(), 1)
    assert_equal(flat_stride[3].value(), 4)


# ===----------------------------------------------------------------------=== #
# TileTensor flat indexing tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_flat_rank() raises:
    """Test that flat_rank is computed correctly for nested and non-nested layouts.
    """
    # Non-nested layout: flat_rank == rank
    comptime tensor1 = TileTensor[
        .float32, type_of(row_major[3, 4]()), MutAnyOrigin
    ]
    comptime assert tensor1.rank == 2
    comptime assert tensor1.flat_rank == 2

    # Nested layout from blocked_product: flat_rank > rank
    comptime block = row_major[2, 2]()
    comptime tiler = row_major[2, 3]()
    comptime blocked_layout = blocked_product(block, tiler)

    comptime tensor2 = TileTensor[
        .float32, type_of(blocked_layout), MutAnyOrigin
    ]
    comptime assert tensor2.rank == 2  # Two top-level Coords
    comptime assert tensor2.flat_rank == 4  # Four scalar dimensions


def test_tile_tensor_flat_indexing_blocked() raises:
    """Test flat indexing on a tensor with blocked_product layout."""
    # Create a blocked layout: 2x2 blocks arranged in 2x3 grid
    # Total size: 4 * 6 = 24 elements
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked_layout = blocked_product(block, tiler)

    var storage = Array[Float32, 24](fill={})
    var tensor = TileTensor(storage, blocked_layout)

    # Initialize using flat indices
    # The layout has strides ((2, 1), (12, 4))
    # Flat indexing: tensor[block_row, block_col, tile_row, tile_col]
    for block_row in range(2):
        for block_col in range(2):
            for tile_row in range(2):
                for tile_col in range(3):
                    var val = Float32(
                        block_row * 1000
                        + block_col * 100
                        + tile_row * 10
                        + tile_col
                    )
                    tensor[block_row, block_col, tile_row, tile_col] = val

    # Read back and verify
    assert_equal(tensor[Idx[0], Idx[0], Idx[0], Idx[0]], 0.0)
    assert_equal(tensor[Idx[0], Idx[1], Idx[0], Idx[0]], 100.0)
    assert_equal(tensor[Idx[1], Idx[0], Idx[0], Idx[0]], 1000.0)
    assert_equal(tensor[Idx[1], Idx[1], Idx[0], Idx[0]], 1100.0)
    assert_equal(tensor[Idx[0], Idx[0], Idx[1], Idx[2]], 12.0)
    assert_equal(tensor[Idx[1], Idx[1], Idx[1], Idx[2]], 1112.0)


def test_layout_reverse() raises:
    """Test Layout.reverse() swaps shape and stride dimensions."""
    # row_major[3,4] has shape (3,4), strides (4,1).
    # Reversed: shape (4,3), strides (1,4) — column-major ordering.
    var layout = row_major[3, 4]()
    var rev = layout.reverse()
    assert_equal(rev.shape[0]().value(), 4)
    assert_equal(rev.shape[1]().value(), 3)
    assert_equal(rev.stride[0]().value(), 1)
    assert_equal(rev.stride[1]().value(), 4)

    # Products should be identical.
    assert_equal(rev.product(), layout.product())

    # col_major[2,5] has shape (2,5), strides (1,2).
    # Reversed: shape (5,2), strides (2,1) — row-major ordering.
    var layout2 = col_major[2, 5]()
    var rev2 = layout2.reverse()
    assert_equal(rev2.shape[0]().value(), 5)
    assert_equal(rev2.shape[1]().value(), 2)
    assert_equal(rev2.stride[0]().value(), 2)
    assert_equal(rev2.stride[1]().value(), 1)

    # 1D layout: reverse is identity.
    var layout1d = row_major[7]()
    var rev1d = layout1d.reverse()
    assert_equal(rev1d.shape[0]().value(), 7)
    assert_equal(rev1d.stride[0]().value(), 1)


def test_layout_transpose() raises:
    """Test Layout.transpose() reverses dimensions (same as reverse)."""
    # row_major[3,4] has shape (3,4), strides (4,1).
    # Transposed: shape (4,3), strides (1,4).
    var layout = row_major[3, 4]()
    var trans = layout.transpose()
    assert_equal(trans.shape[0]().value(), 4)
    assert_equal(trans.shape[1]().value(), 3)
    assert_equal(trans.stride[0]().value(), 1)
    assert_equal(trans.stride[1]().value(), 4)

    # Products should be identical.
    assert_equal(trans.product(), layout.product())

    # col_major[2,5] transposed should look row-major.
    var layout2 = col_major[2, 5]()
    var trans2 = layout2.transpose()
    assert_equal(trans2.shape[0]().value(), 5)
    assert_equal(trans2.shape[1]().value(), 2)
    assert_equal(trans2.stride[0]().value(), 2)
    assert_equal(trans2.stride[1]().value(), 1)

    # 3D transpose reverses all dimensions.
    var layout3 = row_major[2, 3, 4]()
    var trans3 = layout3.transpose()
    assert_equal(trans3.shape[0]().value(), 4)
    assert_equal(trans3.shape[1]().value(), 3)
    assert_equal(trans3.shape[2]().value(), 2)
    assert_equal(trans3.stride[0]().value(), 1)
    assert_equal(trans3.stride[1]().value(), 4)
    assert_equal(trans3.stride[2]().value(), 12)

    # 1D transpose is identity.
    var layout1d = row_major[7]()
    var trans1d = layout1d.transpose()
    assert_equal(trans1d.shape[0]().value(), 7)
    assert_equal(trans1d.stride[0]().value(), 1)


def test_tile_tensor_transpose() raises:
    """Test TileTensor.transpose() creates a transposed view."""
    var storage = Array[Float32, 6](fill={})
    var tensor = TileTensor(storage, row_major[2, 3]())

    # Fill with distinct values
    tensor[0, 0] = 0.0
    tensor[0, 1] = 1.0
    tensor[0, 2] = 2.0
    tensor[1, 0] = 3.0
    tensor[1, 1] = 4.0
    tensor[1, 2] = 5.0

    var trans = tensor.transpose()

    # Transposed shape should be (3, 2)
    assert_equal(trans.dim[0](), 3)
    assert_equal(trans.dim[1](), 2)

    # Transposed access: trans[col, row] == tensor[row, col]
    assert_equal(trans[0, 0], 0.0)
    assert_equal(trans[1, 0], 1.0)
    assert_equal(trans[2, 0], 2.0)
    assert_equal(trans[0, 1], 3.0)
    assert_equal(trans[1, 1], 4.0)
    assert_equal(trans[2, 1], 5.0)


def test_tile_tensor_transpose_is_view() raises:
    """Test that transpose creates a view, not a copy."""
    var storage = Array[Float32, 6](fill={})
    var tensor = TileTensor(storage, row_major[2, 3]()).fill(0.0)

    var trans = tensor.transpose()

    # Modify through transposed view
    trans[1, 0] = 42.0

    # Verify change is visible in original tensor
    # trans[1, 0] maps to tensor[0, 1]
    assert_equal(tensor[0, 1], 42.0)


def test_tile_tensor_hierarchical_indexing_with_coord() raises:
    """Test hierarchical indexing using nested Coord on nested layout."""
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked_layout = blocked_product(block, tiler)

    var storage = Array[Float32, 24](fill=0.0)
    var tensor = TileTensor(storage, blocked_layout)

    # Use nested Coord matching layout shape ((2,2),(2,3))
    var coord = Coord(Coord(Idx[1], Idx[0]), Coord(Idx[1], Idx[2]))
    tensor[coord] = 42.0

    # Read back with flat indices
    assert_equal(tensor[1, 0, 1, 2], 42.0)


# ===----------------------------------------------------------------------=== #
# Hierarchical indexing tests (Layout.__call__ with nested shapes)
# ===----------------------------------------------------------------------=== #


def test_layout_call_hierarchical_exact_match() raises:
    """Test __call__ with coord structure matching shape structure exactly.

    Layout shape (4, (3, 2)) with row-major strides (6, (2, 1)).
    Coord (1, (1, 1)) should compute: 1*6 + 1*2 + 1*1 = 9.
    """
    # Build a nested layout: shape (4, (3, 2)), stride (6, (2, 1))
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)

    # (1, (1, 1)) -> 1*6 + 1*2 + 1*1 = 9
    assert_equal(layout(Coord(Idx[1], Coord(Idx[1], Idx[1]))), 9)

    # (0, (0, 0)) -> 0
    assert_equal(layout(Coord(Idx[0], Coord(Idx[0], Idx[0]))), 0)

    # (2, (1, 0)) -> 2*6 + 1*2 + 0*1 = 14
    assert_equal(layout(Coord(Idx[2], Coord(Idx[1], Idx[0]))), 14)

    # (3, (2, 1)) -> 3*6 + 2*2 + 1*1 = 23
    assert_equal(layout(Coord(Idx[3], Coord(Idx[2], Idx[1]))), 23)


def test_layout_call_hierarchical_rank_matching() raises:
    """Test __call__ with coord rank matching shape rank but flat structure.

    Layout shape (4, (3, 2)) with row-major strides (6, (2, 1)).
    Coord (1, 1) has rank 2 matching the layout rank.
    The scalar 1 is decomposed within the (3, 2) sub-dimension.
    """
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)

    # (0, 0) -> 0*6 + crd2idx(0, (3,2), (2,1)) = 0
    assert_equal(layout(Coord(Idx[0], Idx[0])), 0)

    # (1, 0) -> 1*6 + crd2idx(0, (3,2), (2,1)) = 6
    assert_equal(layout(Coord(Idx[1], Idx[0])), 6)

    # (1, 1) -> 1*6 + crd2idx(1, (3,2), (2,1))
    # crd2idx decomposes 1 within (3,2): divmod(1, 3) = (0, 1)
    # -> 1*2 + 0*1 = 2. Total: 6 + 2 = 8
    assert_equal(layout(Coord(Idx[1], Idx[1])), 8)

    # (0, 5) -> crd2idx(5, (3,2), (2,1))
    # divmod(5, 3) = (1, 2) -> 2*2 + 1*1 = 5. Total: 0 + 5 = 5
    assert_equal(layout(Coord(Idx[0], Idx[5])), 5)


def test_layout_call_hierarchical_scalar() raises:
    """Test __call__ with a scalar index on a nested layout.

    Layout shape (4, (3, 2)) with strides (6, (2, 1)).
    Scalar index is decomposed across all dimensions.
    """
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)

    # Scalar 0 -> 0
    assert_equal(layout(Idx[0]), 0)

    # Scalar 1 -> decomposed via divmod against shape elements
    # divmod(1, 4) = (0, 1), so dim0 gets 1, rest gets 0
    # -> 1*6 + 0 = 6
    assert_equal(layout(Idx[1]), 6)


def test_layout_call_hierarchical_flat_coords() raises:
    """Test __call__ with more coord elements than shape dimensions.

    Layout shape (4, (3, 2)) has rank 2, flat_rank 3.
    Coord (1, 1, 1) has rank 3 > layout rank 2.
    This uses flat dot product with flattened strides (6, 2, 1).
    """
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)

    # (1, 1, 1) -> 1*6 + 1*2 + 1*1 = 9 (flat strides)
    assert_equal(layout(Coord(Idx[1], Idx[1], Idx[1])), 9)

    # (0, 0, 0) -> 0
    assert_equal(layout(Coord(Idx[0], Idx[0], Idx[0])), 0)

    # (2, 1, 0) -> 2*6 + 1*2 + 0*1 = 14
    assert_equal(layout(Coord(Idx[2], Idx[1], Idx[0])), 14)

    # (3, 2, 1) -> 3*6 + 2*2 + 1*1 = 23
    assert_equal(layout(Coord(Idx[3], Idx[2], Idx[1])), 23)


def test_layout_call_hierarchical_consistency() raises:
    """Test that all three indexing forms give consistent results.

    For layout shape (4, (3, 2)) strides (6, (2, 1)):
    - (2, (1, 0)): exact match
    - (2, 1): rank-matching with scalar sub-index
    - (2, 1, 0): flat indexing
    All should give the same offset: 2*6 + 1*2 + 0*1 = 14.
    """
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)

    # Exact match: (2, (1, 0))
    var exact = layout(Coord(Idx[2], Coord(Idx[1], Idx[0])))
    # Flat: (2, 1, 0)
    var flat = layout(Coord(Idx[2], Idx[1], Idx[0]))

    assert_equal(exact, 14)
    assert_equal(flat, 14)

    # Rank-matching: (2, 1)
    # Scalar 1 decomposes within (3, 2): divmod(1, 3) = (0, 1)
    # -> coord (1, 0) in the sub-dim -> 1*2 + 0 = 2. Total: 12 + 2 = 14
    var rank_match = layout(Coord(Idx[2], Idx[1]))
    assert_equal(rank_match, 14)


# ===----------------------------------------------------------------------=== #
# TileTensor hierarchical indexing tests
# ===----------------------------------------------------------------------=== #


def test_tile_tensor_hierarchical_load_store() raises:
    """Test TileTensor load/store with coords of varying flat_rank.

    Uses a blocked layout to create a nested shape, then indexes with
    exact-match, rank-matching, and flat coords.
    """
    # blocked_product: block 2x2, tiler 2x3
    # shape: ((2, 2), (2, 3)), strides: ((2, 1), (12, 4))
    # flat_rank = 4
    var block = row_major[2, 2]()
    var tiler = row_major[2, 3]()
    var blocked_layout = blocked_product(block, tiler)

    var storage = Array[Float32, 24](fill=0.0)
    var tensor = TileTensor(storage, blocked_layout)

    # Write using exact-match hierarchical coord: ((1, 0), (0, 2))
    # -> 1*2 + 0*1 + 0*12 + 2*4 = 10
    var exact_coord = Coord(Coord(Idx[1], Idx[0]), Coord(Idx[0], Idx[2]))
    tensor[exact_coord] = 42.0

    # Read back using flat coord: (1, 0, 0, 2) -> same offset
    assert_equal(tensor[1, 0, 0, 2], 42.0)

    # Write using rank-matching coord: (Idx[1], Idx[2])
    # dim 0 is scalar 1, decomposed within shape (2, 2) -> divmod(1, 2) = (0, 1)
    #   -> 0*2 + 1*1 = 1
    # dim 1 is scalar 2, decomposed within shape (2, 3) -> divmod(2, 2) = (1, 0)
    #   -> 1*12 + 0*4 = 12
    # total offset: 1 + 12 = 13
    tensor.store(Coord(Idx[1], Idx[2]), Float32(99.0))

    # Read back at the same offset
    var val = tensor.load(Coord(Idx[1], Idx[2]))
    assert_equal(val, 99.0)


def test_tile_tensor_hierarchical_getitem_setitem() raises:
    """Test TileTensor __getitem__/__setitem__ with hierarchical Coord."""
    var storage = Array[Float32, 24](fill=0.0)

    # Layout shape (4, (3, 2)), stride (6, (2, 1))
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)
    var tensor = TileTensor(storage, layout)

    # Write with exact-match coord (2, (1, 0)) -> offset 14
    tensor[Coord(Idx[2], Coord(Idx[1], Idx[0]))] = 77.0
    assert_equal(storage[14], 77.0)

    # Read with rank-matching coord (2, 1) -> offset 14
    # (scalar 1 in (3,2) sub-dim: divmod(1,3)=(0,1) -> 1*2 = 2, total 12+2=14)
    var val = tensor[Coord(Idx[2], Idx[1])]
    assert_equal(val, 77.0)


def test_tile_tensor_hierarchical_scalar_index() raises:
    """Test TileTensor load with a single scalar coord."""
    var storage = Array[Float32, 24](fill=0.0)

    # Layout shape (4, (3, 2)), stride (6, (2, 1))
    var inner_shape = Coord(Idx[3], Idx[2])
    var inner_stride = Coord(Idx[2], Idx[1])
    var shape = Coord(Idx[4], inner_shape)
    var stride = Coord(Idx[6], inner_stride)
    var layout = Layout(shape=shape, stride=stride)
    var tensor = TileTensor(storage, layout)

    # Write at a known offset via raw pointer
    storage[6] = 55.0

    # Scalar index 1 -> decomposed: divmod(1, 4) = (0, 1)
    # -> 1*6 + 0 = 6
    var val = tensor.load(Coord(Idx[1]))
    assert_equal(val, 55.0)


def test_static_product_flat() raises:
    # row_major[3, 4]() -> product = 12
    comptime flat = row_major[3, 4]()
    assert_equal(flat.static_product, 12)

    # row_major[8, 16]() -> product = 128
    comptime flat2 = row_major[8, 16]()
    assert_equal(flat2.static_product, 128)

    # 1D layout
    comptime one_d = row_major[7]()
    assert_equal(one_d.static_product, 7)


def test_static_product_nested() raises:
    # Nested layout matching tile_layout_k_major_typed structure:
    # shape ((8, 16), (64, 2)) -> product = 8 * 16 * 64 * 2 = 16384
    comptime nested = Layout(
        Coord(Coord(Idx[8], Idx[16]), Coord(Idx[64], Idx[2])),
        Coord(Coord(Idx[64], Idx[512]), Coord(Idx[1], Idx[8192])),
    )
    assert_equal(nested.static_product, 16384)

    # Another nested layout: ((4, 32), (128, 1)) -> 4 * 32 * 128 = 16384
    comptime nested2 = Layout(
        Coord(Coord(Idx[4], Idx[32]), Coord(Idx[128], Idx[1])),
        Coord(Coord(Idx[128], Idx[4096]), Coord(Idx[1], Idx[0])),
    )
    assert_equal(nested2.static_product, 16384)

    # 3-level nesting: ((32, 2), ((4, 4), 1)) -> 32*2*4*4*1 = 1024
    comptime deep = Layout(
        Coord(
            Coord(Idx[32], Idx[2]),
            Coord(Coord(Idx[4], Idx[4]), Idx[1]),
        ),
        Coord(
            Coord(Idx[16], Idx[512]),
            Coord(Coord(Idx[1], Idx[4]), Idx[512]),
        ),
    )
    assert_equal(deep.static_product, 1024)


def test_upcast_row_major_2d() raises:
    """Upcast a row-major (4, 8) layout by factor 2.

    stride (8, 1) → new_stride: (shape_div(8,2)=4, shape_div(1,2)=1)
    shape  (4, 8) → new_shape:  (shape_div(4,shape_div(2,8))=4,
                                  shape_div(8,shape_div(2,1))=4)
    Result: shape (4, 4), stride (4, 1).
    """
    var layout = row_major[4, 8]()
    var up = upcast[factor=2](layout)

    assert_equal(up.shape[0]().value(), 4)
    assert_equal(up.shape[1]().value(), 4)
    assert_equal(up.stride[0]().value(), 4)
    assert_equal(up.stride[1]().value(), 1)

    # Compile-time types should be preserved.
    comptime UpType = UpcastLayout[type_of(layout), 2]
    comptime assert UpType.static_shape[0] == 4
    comptime assert UpType.static_shape[1] == 4
    comptime assert UpType.static_stride[0] == 4
    comptime assert UpType.static_stride[1] == 1


def test_upcast_row_major_factor4() raises:
    """Upcast a row-major (4, 8) layout by factor 4.

    Result: shape (4, 2), stride (2, 1).
    """
    var layout = row_major[4, 8]()
    var up = upcast[factor=4](layout)

    assert_equal(up.shape[0]().value(), 4)
    assert_equal(up.shape[1]().value(), 2)
    assert_equal(up.stride[0]().value(), 2)
    assert_equal(up.stride[1]().value(), 1)


def test_upcast_1d() raises:
    """Upcast a 1D layout (16,) stride (1,) by factor 4 → (4,) stride (1,)."""
    var layout = row_major[16]()
    var up = upcast[factor=4](layout)

    assert_equal(up.shape[0]().value(), 4)
    assert_equal(up.stride[0]().value(), 1)


def test_upcast_col_major() raises:
    """Upcast a column-major (4, 8) layout by factor 2.

    col_major strides: (1, 4).
    Dim 0: stride=1, shape=4 → new_stride=shape_div(1,2)=1,
           new_shape=shape_div(4, shape_div(2,1))=shape_div(4,2)=2
    Dim 1: stride=4, shape=8 → new_stride=shape_div(4,2)=2,
           new_shape=shape_div(8, shape_div(2,4))=shape_div(8,1)=8
    Result: shape (2, 8), stride (1, 2).
    """
    var layout = col_major[4, 8]()
    var up = upcast[factor=2](layout)

    assert_equal(up.shape[0]().value(), 2)
    assert_equal(up.shape[1]().value(), 8)
    assert_equal(up.stride[0]().value(), 1)
    assert_equal(up.stride[1]().value(), 2)


def test_upcast_factor1_identity() raises:
    """Upcast by factor 1 should be the identity operation."""
    var layout = row_major[3, 5]()
    var up = upcast[factor=1](layout)

    assert_equal(up.shape[0]().value(), 3)
    assert_equal(up.shape[1]().value(), 5)
    assert_equal(up.stride[0]().value(), 5)
    assert_equal(up.stride[1]().value(), 1)


def test_upcast_runtime_dims() raises:
    """Upcast with runtime dimensions produces Scalar results."""
    var layout = row_major(Int(4), Idx[8])
    var up = upcast[factor=2](layout)

    assert_equal(up.shape[0]().value(), 4)
    assert_equal(up.shape[1]().value(), 4)
    assert_equal(up.stride[0]().value(), 4)
    assert_equal(up.stride[1]().value(), 1)


def test_upcast_preserves_element_access() raises:
    """Verify that upcast layout correctly maps coarser indices.

    For a row-major (2, 8) tensor upcast by 4: result is (2, 2).
    Coarser element [r, c] should map to linear index r*2 + c, which
    corresponds to original element r*8 + c*4 in the fine-grained space.
    """
    var layout = row_major[2, 8]()
    var up = upcast[factor=4](layout)

    # up: shape (2, 2), stride (2, 1)
    assert_equal(up.shape[0]().value(), 2)
    assert_equal(up.shape[1]().value(), 2)

    # Check that the upcast layout maps indices correctly.
    # up(0, 0) = 0*2 + 0*1 = 0
    assert_equal(Int(up(Coord(Idx[0], Idx[0]))), 0)
    # up(0, 1) = 0*2 + 1*1 = 1
    assert_equal(Int(up(Coord(Idx[0], Idx[1]))), 1)
    # up(1, 0) = 1*2 + 0*1 = 2
    assert_equal(Int(up(Coord(Idx[1], Idx[0]))), 2)
    # up(1, 1) = 1*2 + 1*1 = 3
    assert_equal(Int(up(Coord(Idx[1], Idx[1]))), 3)


def test_upcast_layout_type_level() raises:
    """UpcastLayout can be used purely at the type level for static layouts.

    For row_major[4, 8] (strides (8, 1)) upcast by 4:
    Result shape (4, 2), strides (2, 1).
    Default-initializing the type gives a layout with those values.
    """
    comptime RM = type_of(row_major[4, 8]())
    comptime Up = UpcastLayout[RM, 4]

    # Verify static types are correct.
    comptime assert Up.static_shape[0] == 4
    comptime assert Up.static_shape[1] == 2
    comptime assert Up.static_stride[0] == 2
    comptime assert Up.static_stride[1] == 1

    # Default-init produces a usable layout with the right values.
    var layout = Up()
    assert_equal(layout.shape[0]().value(), 4)
    assert_equal(layout.shape[1]().value(), 2)
    assert_equal(layout.stride[0]().value(), 2)
    assert_equal(layout.stride[1]().value(), 1)

    # It maps coordinates correctly.
    assert_equal(Int(layout(Coord(Idx[0], Idx[0]))), 0)
    assert_equal(Int(layout(Coord(Idx[1], Idx[1]))), 3)
    assert_equal(Int(layout(Coord(Idx[3], Idx[1]))), 7)


def test_upcast_layout_col_major_type_level() raises:
    """UpcastLayout at the type level for a column-major layout.

    col_major[4, 8] has strides (1, 4). Upcast by 2:
    Dim 0: stride=1 < factor=2 → new_stride=1, new_shape=shape_div(4,2)=2.
    Dim 1: stride=4 ≥ factor=2 → new_stride=4/2=2, new_shape=8.
    Result: shape (2, 8), strides (1, 2).
    """
    comptime CM = type_of(col_major[4, 8]())
    comptime Up = UpcastLayout[CM, 2]

    comptime assert Up.static_shape[0] == 2
    comptime assert Up.static_shape[1] == 8
    comptime assert Up.static_stride[0] == 1
    comptime assert Up.static_stride[1] == 2


def test_zipped_divide_layout_type_level() raises:
    """ZippedDivideLayout can be used at the type level for static layouts.

    row_major[8, 16] tiled by (2, 4):
    - inner shape:  (2, 4)    = tile
    - outer shape:  (4, 4)    = (8/2, 16/4)
    - inner stride: (16, 1)   = original strides
    - outer stride: (32, 4)   = (16*2, 1*4)
    """
    comptime RM = type_of(row_major[8, 16]())
    comptime Tile = Coord(Idx[2], Idx[4]).element_types
    comptime ZD = ZippedDivideLayout[RM, Tile]

    # inner_shape = tile
    comptime assert ZD._shape_types[0].ParamListType[0].static_value == 2
    comptime assert ZD._shape_types[0].ParamListType[1].static_value == 4

    # outer_shape = shape / tile
    comptime assert ZD._shape_types[1].ParamListType[0].static_value == 4
    comptime assert ZD._shape_types[1].ParamListType[1].static_value == 4

    # inner_stride = original stride
    comptime assert ZD._stride_types[0].ParamListType[0].static_value == 16
    comptime assert ZD._stride_types[0].ParamListType[1].static_value == 1

    # outer_stride = stride * tile
    comptime assert ZD._stride_types[1].ParamListType[0].static_value == 32
    comptime assert ZD._stride_types[1].ParamListType[1].static_value == 4

    # Default-init produces a usable layout.
    var layout = ZD()
    assert_equal(layout.shape[0]()[0].value(), 2)
    assert_equal(layout.shape[0]()[1].value(), 4)
    assert_equal(layout.shape[1]()[0].value(), 4)
    assert_equal(layout.shape[1]()[1].value(), 4)


def test_upcast_then_zipped_divide_type_level() raises:
    """Chain UpcastLayout and ZippedDivideLayout purely at the type level.

    Start with row_major[4, 16] (strides (16, 1)).
    Upcast by 4: shape (4, 4), strides (4, 1).
    Then zipped_divide by (2, 2): inner (2, 2), outer (2, 2).
    """
    comptime RM = type_of(row_major[4, 16]())
    comptime Up = UpcastLayout[RM, 4]

    comptime assert Up.static_shape[0] == 4
    comptime assert Up.static_shape[1] == 4
    comptime assert Up.static_stride[0] == 4
    comptime assert Up.static_stride[1] == 1

    comptime Tile = Coord(Idx[2], Idx[2]).element_types
    comptime ZD = ZippedDivideLayout[Up, Tile]

    # inner shape = tile = (2, 2)
    comptime assert ZD._shape_types[0].ParamListType[0].static_value == 2
    comptime assert ZD._shape_types[0].ParamListType[1].static_value == 2

    # outer shape = (4/2, 4/2) = (2, 2)
    comptime assert ZD._shape_types[1].ParamListType[0].static_value == 2
    comptime assert ZD._shape_types[1].ParamListType[1].static_value == 2

    # inner stride = upcast stride = (4, 1)
    comptime assert ZD._stride_types[0].ParamListType[0].static_value == 4
    comptime assert ZD._stride_types[0].ParamListType[1].static_value == 1

    # outer stride = upcast stride * tile = (4*2, 1*2) = (8, 2)
    comptime assert ZD._stride_types[1].ParamListType[0].static_value == 8
    comptime assert ZD._stride_types[1].ParamListType[1].static_value == 2


def test_coalesced_blocked_product_1d() raises:
    """CoalescedBlockedProductLayout on 1D layouts fully coalesces.

    block = (4,) stride (1,), tiler = (3,) stride (1,), cosize = 4.
    Mode 0: 4 * 1 == 4 * 1 = 4. Coalesces to shape=12, stride=1.
    """
    comptime Block = type_of(row_major[4]())
    comptime Tiler = type_of(row_major[3]())
    comptime C = BlockedProductLayout[Block, Tiler, coalesce_output=True]

    # Should be flat: single ComptimeInt[12] shape, ComptimeInt[1] stride.
    comptime assert C.static_shape[0] == 12
    comptime assert C.static_stride[0] == 1

    var layout = C()
    assert_equal(layout.shape[0]().value(), 12)
    assert_equal(layout.stride[0]().value(), 1)


def test_coalesced_blocked_product_no_coalesce() raises:
    """CoalescedBlockedProductLayout when no modes coalesce.

    block = row_major[2, 2] strides (2, 1), tiler = row_major[2, 3] strides (3, 1).
    cosize = 4.
    Mode 0: 2 * 2 = 4 != 4 * 3 = 12. No coalesce.
    Mode 1: 2 * 1 = 2 != 4 * 1 = 4.  No coalesce.
    Result is the same as BlockedProductLayout (nested).
    """
    comptime Block = type_of(row_major[2, 2]())
    comptime Tiler = type_of(row_major[2, 3]())
    comptime C = BlockedProductLayout[Block, Tiler, coalesce_output=True]

    # Mode 0: nested shape (2, 2), stride (2, 12)
    comptime assert C._shape_types[0].ParamListType[0].static_value == 2
    comptime assert C._shape_types[0].ParamListType[1].static_value == 2
    comptime assert C._stride_types[0].ParamListType[0].static_value == 2
    comptime assert C._stride_types[0].ParamListType[1].static_value == 12

    # Mode 1: nested shape (2, 3), stride (1, 4)
    comptime assert C._shape_types[1].ParamListType[0].static_value == 2
    comptime assert C._shape_types[1].ParamListType[1].static_value == 3
    comptime assert C._stride_types[1].ParamListType[0].static_value == 1
    comptime assert C._stride_types[1].ParamListType[1].static_value == 4


def test_coalesced_blocked_product_partial() raises:
    """CoalescedBlockedProductLayout with partial coalescing.

    block = row_major[2, 4] strides (4, 1),
    tiler = col_major[3, 2] strides (1, 3). cosize = 8.
    Mode 0: 2 * 4 = 8 == 8 * 1 = 8. Coalesces to shape=6, stride=4.
    Mode 1: 4 * 1 = 4 != 8 * 3 = 24. No coalesce — nested.
    """
    comptime Block = type_of(row_major[2, 4]())
    comptime Tiler = type_of(col_major[3, 2]())
    comptime C = BlockedProductLayout[Block, Tiler, coalesce_output=True]

    # Mode 0: coalesced flat. shape=2*3=6, stride=4 (block stride).
    comptime assert C._shape_types[0].static_value == 6
    comptime assert C._stride_types[0].static_value == 4

    # Mode 1: not coalesced — nested Coord.
    comptime assert C._shape_types[1].ParamListType[0].static_value == 4
    comptime assert C._shape_types[1].ParamListType[1].static_value == 2
    comptime assert C._stride_types[1].ParamListType[0].static_value == 1
    comptime assert C._stride_types[1].ParamListType[1].static_value == 24


def test_write_to_static() raises:
    var layout = row_major(Idx[3], Idx[4])
    check_write_to(layout, expected="((3, 4):(4, 1))", is_repr=False)


def test_write_to_dynamic() raises:
    var layout = row_major(Int(3), Idx[4])
    check_write_to(layout, expected="((3, 4):(4, 1))", is_repr=False)


def test_weakly_compatible_scalar_coord() raises:
    """Scalar coord elements are always compatible with any layout."""
    comptime L = type_of(row_major[3, 4]())
    comptime assert WeaklyCompatible[
        L,
        Coord[ComptimeInt[5], ComptimeInt[7]].element_types,
    ]
    comptime assert WeaklyCompatible[
        L,
        Coord[Int32, Int32].element_types,
    ]


def test_weakly_compatible_flat_match() raises:
    """Flat coord types with matching rank is compatible."""
    comptime L = type_of(row_major[3, 4]())
    comptime assert WeaklyCompatible[
        L,
        Coord[ComptimeInt[2], ComptimeInt[3]].element_types,
    ]


def test_weakly_compatible_flat_mismatch() raises:
    """Flat coord types with wrong rank is incompatible."""
    comptime L = type_of(row_major[3, 4]())
    comptime assert not WeaklyCompatible[
        L,
        Coord[ComptimeInt[1], ComptimeInt[2], ComptimeInt[3]].element_types,
    ]


def test_weakly_compatible_1d_layout() raises:
    """1D layout vs 1-element and 2-element coord types."""
    comptime L = type_of(row_major[8]())
    comptime assert WeaklyCompatible[L, Coord[ComptimeInt[4]].element_types]
    comptime assert not WeaklyCompatible[
        L,
        Coord[ComptimeInt[2], ComptimeInt[4]].element_types,
    ]


def test_weakly_compatible_nested_depth2() raises:
    """Depth-2 nesting: nested coord types match nested layout shape."""
    # blocked_product produces nested shapes like ((2, 3), (4, 2))
    comptime Block = type_of(row_major[2, 4]())
    comptime Tiler = type_of(row_major[3, 2]())
    comptime L = BlockedProductLayout[Block, Tiler]

    # Matching nested structure: ((_, _), (_, _))
    comptime assert WeaklyCompatible[
        L,
        Coord[
            Coord[ComptimeInt[1], ComptimeInt[1]],
            Coord[ComptimeInt[1], ComptimeInt[1]],
        ].element_types,
    ]

    # Wrong inner rank in first mode: ((_, _, _), (_, _))
    comptime assert not WeaklyCompatible[
        L,
        Coord[
            Coord[
                ComptimeInt[1],
                ComptimeInt[1],
                ComptimeInt[1],
            ],
            Coord[ComptimeInt[1], ComptimeInt[1]],
        ].element_types,
    ]


def test_weakly_compatible_mixed_scalar_and_tuple() raises:
    """Scalar sub-coord types are always compatible, even when layout has tuples.
    """
    comptime Block = type_of(row_major[2, 4]())
    comptime Tiler = type_of(row_major[3, 2]())
    comptime L = BlockedProductLayout[Block, Tiler]

    # Outer rank matches (2 modes), but sub-coords are scalar — compatible.
    comptime assert WeaklyCompatible[
        L,
        Coord[ComptimeInt[5], ComptimeInt[7]].element_types,
    ]


def test_weakly_compatible_nested_inner_mismatch() raises:
    """Inner rank mismatch at depth 2 is caught."""
    comptime Block = type_of(row_major[2, 4]())
    comptime Tiler = type_of(row_major[3, 2]())
    comptime L = BlockedProductLayout[Block, Tiler]

    # First mode matches (2 elements), second mode wrong inner rank (3 vs 2)
    comptime assert not WeaklyCompatible[
        L,
        Coord[
            Coord[ComptimeInt[1], ComptimeInt[1]],
            Coord[ComptimeInt[1], ComptimeInt[1], ComptimeInt[1]],
        ].element_types,
    ]


def test_weakly_compatible_coord_tuple_vs_layout_scalar() raises:
    """Tuple coord type against a scalar layout shape element is incompatible.
    """
    comptime L = type_of(row_major[3, 4]())

    # Layout shape is (3, 4) — flat scalars. Coord has a nested tuple in
    # first position — incompatible because scalar 3 can't match a tuple.
    comptime assert not WeaklyCompatible[
        L,
        Coord[
            Coord[ComptimeInt[1], ComptimeInt[2]],
            ComptimeInt[3],
        ].element_types,
    ]


def test_weakly_compatible_not_symmetric() raises:
    """WeaklyCompatible is not symmetric: swapping layout and coord can flip
    the result.

    Coord ((1, 1), 1) is NOT compatible with flat layout (1, 1):(1, 1),
    because the nested tuple in the coord can't match a scalar shape element.

    But coord (1, 1) IS compatible with nested layout ((1, 1), 1):((1, 1), 1),
    because scalar coord elements are always compatible with any layout mode.
    """
    comptime flat_L = type_of(row_major[1, 1]())
    comptime nested_L = type_of(row_major(Coord(Coord(Idx[1], Idx[1]), Idx[1])))

    # Nested coord vs flat layout — incompatible.
    comptime assert not WeaklyCompatible[
        flat_L,
        Coord[
            Coord[ComptimeInt[1], ComptimeInt[1]],
            ComptimeInt[1],
        ].element_types,
    ]

    # Flat coord vs nested layout — compatible (scalars match anything).
    comptime assert WeaklyCompatible[
        nested_L,
        Coord[ComptimeInt[1], ComptimeInt[1]].element_types,
    ]
