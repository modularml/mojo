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

from std.utils.index import IndexList
from layout import (
    All,
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    RowMajorLayout,
    TileTensor,
    row_major,
    Layout,
    UNKNOWN_VALUE,
)
from layout.tile_layout import Layout as TileLayout
from layout.swizzle import Swizzle
from layout.tensor_engine import TensorOps
from std.math import exp
from std.testing import (
    TestSuite,
    assert_almost_equal,
    assert_equal,
    assert_true,
)


def _storage_add(
    dst: TileTensor,
    lhs: TileTensor[dst.dtype, ...],
    rhs: TileTensor[dst.dtype, ...],
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    """Engine-level out-of-place `TensorOps.add`.

    `LhsEngine`/`RhsEngine` are only inferable from symbolic operand
    storages, so calls go through this generic helper rather than a direct
    call with concrete handles.
    """
    type_of(dst).Engine.add(
        dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
        lhs=(lhs._storage, lhs.layout),
        rhs=(rhs._storage, rhs.layout),
    )


def _storage_abs(
    dst: TileTensor, src: TileTensor[dst.dtype, ...]
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    """Engine-level out-of-place `TensorOps.abs`."""
    type_of(dst).Engine.abs(
        dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
        src=(src._storage, src.layout),
    )


def _storage_exp[
    scale_dtype: DType, //, scale: Scalar[scale_dtype] = 1
](
    dst: TileTensor, src: TileTensor[dst.dtype, ...]
) where dst.mut and conforms_to(dst.Engine, TensorOps):
    """Engine-level out-of-place `TensorOps.exp`."""
    type_of(dst).Engine.exp[scale](
        dst=(dst._unsafe_storage_cast[to_mut=True](), dst.layout),
        src=(src._storage, src.layout),
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()


def test_distribute() raises:
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var array = Array[UInt32, 16](fill=-1)
    var ptr: MutPointer[UInt32, origin_of(array)] = array.unsafe_ptr()

    comptime data_layout_shape = Coord[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = Coord[ComptimeInt[4], ComptimeInt[1]]
    var layout_tensor = TileTensor(
        ptr=ptr,
        layout=TileLayout(
            shape=data_layout_shape(Idx[4], Idx[4]),
            stride=data_layout_stride(Idx[4], Idx[1]),
        ),
    )

    var counter = 0
    for th_id in range(4):
        var frag = layout_tensor.distribute[thread_layout=thread_layout,](th_id)

        # Fill the fragment positions with the thread id (0..3)
        for i in range(2):
            for j in range(2):
                frag[i, j] = UInt32(counter)
                counter += 1

    var expected = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    for i in range(16):
        assert_equal(ptr[i], UInt32(expected[i]))


def test_distribute_with_swizzle() raises:
    """Test distribute with swizzle parameter.

    This test verifies that the swizzle parameter correctly transforms
    the memory access pattern. We use a simple swizzle that XORs bits
    to remap thread offsets.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])

    # Use Swizzle(1, 0, 2) which XORs bit 2 with bit 0
    # yyy_mask = 1 << 2 = 4 (binary: 100)
    # swizzle(x) = x ^ ((x & 4) >> 2)
    # For offset 0: swizzle(0) = 0 ^ ((0 & 4) >> 2) = 0 ^ 0 = 0
    # For offset 1: swizzle(1) = 1 ^ ((1 & 4) >> 2) = 1 ^ 0 = 1
    # For offset 4: swizzle(4) = 4 ^ ((4 & 4) >> 2) = 4 ^ 1 = 5
    # For offset 5: swizzle(5) = 5 ^ ((5 & 4) >> 2) = 5 ^ 1 = 4
    comptime swizzle = Swizzle(1, 0, 2)

    var array = Array[UInt32, 16](fill=-1)
    var ptr: MutPointer[UInt32, origin_of(array)] = array.unsafe_ptr()

    comptime data_layout_shape = Coord[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = Coord[ComptimeInt[4], ComptimeInt[1]]
    var layout_tensor = TileTensor[.uint32](
        ptr=ptr,
        layout=TileLayout(
            shape=data_layout_shape(Idx[4], Idx[4]),
            stride=data_layout_stride(Idx[4], Idx[1]),
        ),
    )

    # Assign thread IDs to positions with swizzle
    for th_id in range(4):
        var frag = layout_tensor.distribute[
            thread_layout=thread_layout, swizzle=swizzle
        ](th_id)
        # Write thread ID to each position in the fragment
        for i in range(2):
            for j in range(2):
                frag[i, j] = UInt32(th_id)

    # Thread layout row_major[2, 2] has strides [2, 1]
    # Thread 0: coord (0, 0) -> base offset 0*4 + 0*1 = 0, swizzle(0) = 0
    # Thread 1: coord (0, 1) -> base offset 0*4 + 1*1 = 1, swizzle(1) = 1
    # Thread 2: coord (1, 0) -> base offset 1*4 + 0*1 = 4, swizzle(4) = 5
    # Thread 3: coord (1, 1) -> base offset 1*4 + 1*1 = 5, swizzle(5) = 4

    # Verify that thread assignments are swizzled correctly
    # Thread 0 writes starting at offset 0
    assert_equal(ptr[0], 0)

    # Thread 1 writes starting at offset 1
    assert_equal(ptr[1], 1)

    # Thread 2 writes starting at swizzled offset 5 (from base 4)
    assert_equal(ptr[5], 2)

    # Thread 3 writes starting at swizzled offset 4 (from base 5)
    assert_equal(ptr[4], 3)


def test_distribute_swizzle_vs_no_swizzle() raises:
    """Test that swizzle actually changes the memory access pattern.

    Compare the results of distribute with and without swizzle to verify
    that swizzling produces different memory layouts.
    """
    comptime thread_layout = row_major(Idx[2], Idx[2])
    comptime swizzle = Swizzle(1, 0, 2)

    # Array without swizzle
    var array_no_swizzle = Array[UInt32, 16](fill=0)
    var ptr_no_swizzle: MutPointer[
        UInt32, origin_of(array_no_swizzle)
    ] = array_no_swizzle.unsafe_ptr()

    # Array with swizzle
    var array_with_swizzle = Array[UInt32, 16](fill=0)
    var ptr_with_swizzle: MutPointer[
        UInt32, origin_of(array_with_swizzle)
    ] = array_with_swizzle.unsafe_ptr()

    comptime data_layout_shape = Coord[ComptimeInt[4], ComptimeInt[4]]
    comptime data_layout_stride = Coord[ComptimeInt[4], ComptimeInt[1]]

    var tensor_no_swizzle = TileTensor[.uint32](
        ptr=ptr_no_swizzle,
        layout=TileLayout(
            shape=data_layout_shape(Idx[4], Idx[4]),
            stride=data_layout_stride(Idx[4], Idx[1]),
        ),
    )

    var tensor_with_swizzle = TileTensor[.uint32](
        ptr=ptr_with_swizzle,
        layout=TileLayout(
            shape=data_layout_shape(Idx[4], Idx[4]),
            stride=data_layout_stride(Idx[4], Idx[1]),
        ),
    )

    # Fill both tensors with thread IDs
    for th_id in range(4):
        var frag_no_swizzle = tensor_no_swizzle.distribute[
            thread_layout=thread_layout
        ](th_id)
        var frag_with_swizzle = tensor_with_swizzle.distribute[
            thread_layout=thread_layout, swizzle=swizzle
        ](th_id)

        for i in range(2):
            for j in range(2):
                frag_no_swizzle[i, j] = UInt32(th_id)
                frag_with_swizzle[i, j] = UInt32(th_id)

    # Verify that the two arrays are different (swizzle changes layout)
    var differ = False
    for i in range(16):
        if ptr_no_swizzle[i] != ptr_with_swizzle[i]:
            differ = True
            break
    assert_true(differ, "Swizzle should produce different memory layout")


def test_tile() raises:
    # Create a 4x4 tensor with row-major layout
    var data = Array[UInt32, 16](fill=0)

    var layout_tensor = TileTensor(data, row_major(Idx[4], Idx[4]))

    var counter = 0

    comptime for tile_i in range(2):
        comptime for tile_j in range(2):
            var current_tile = layout_tensor.tile[2, 2](
                (tile_i, tile_j),
            )

            for i in range(2):
                for j in range(2):
                    current_tile[i, j] = UInt32(counter)
                    counter += 1

    # Expected layout after tiling:
    # Tile (0,0): values 0,1,2,3   -> positions [0,1], [4,5]
    # Tile (0,1): values 4,5,6,7   -> positions [2,3], [6,7]
    # Tile (1,0): values 8,9,10,11 -> positions [8,9], [12,13]
    # Tile (1,1): values 12,13,14,15 -> positions [10,11], [14,15]
    var expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    for i in range(16):
        assert_equal(data[i], UInt32(expected[i]))


def test_tile_with_coord_shape() raises:
    """Test tile() with a mixed static/dynamic tile shape Coord.

    The tile shape has one ComptimeInt (rows) and one Scalar (cols),
    exercising the mixed compile-time/runtime Coord path.
    """
    # Create a 4x4 tensor with values 0..15 in row-major order
    var data = Array[UInt32, 16](fill=0)
    var layout_tensor = TileTensor(data, row_major(Idx[4], Idx[4]))

    # Mixed tile shape: static rows, dynamic cols
    var tile_shape = Coord(Idx[2], Idx[2])

    var counter = 0

    # Tile with Coord shape — same logic as test_tile but using the
    # tile(tile_shape, tile_coord) overload instead of tile[2, 2](coord).
    comptime for tile_i in range(2):
        comptime for tile_j in range(2):
            var current_tile = layout_tensor.tile(
                tile_shape,
                (tile_i, tile_j),
            )

            for i in range(2):
                for j in range(2):
                    current_tile[i, j] = UInt32(counter)
                    counter += 1

    # Must match the same memory layout as the parametric tile[] version.
    var expected = [0, 1, 4, 5, 2, 3, 6, 7, 8, 9, 12, 13, 10, 11, 14, 15]
    for i in range(16):
        assert_equal(data[i], UInt32(expected[i]))

    # Also verify reading back through the mixed Coord-based tile.
    var t00 = layout_tensor.tile(tile_shape, (Idx[0], Idx[0]))
    assert_equal(t00[Idx[0], Idx[0]], UInt32(0))
    assert_equal(t00[Idx[0], Idx[1]], UInt32(1))
    assert_equal(t00[Idx[1], Idx[0]], UInt32(2))
    assert_equal(t00[Idx[1], Idx[1]], UInt32(3))

    var t11 = layout_tensor.tile(tile_shape, (Idx[1], Idx[1]))
    assert_equal(t11[Idx[0], Idx[0]], UInt32(12))
    assert_equal(t11[Idx[0], Idx[1]], UInt32(13))
    assert_equal(t11[Idx[1], Idx[0]], UInt32(14))
    assert_equal(t11[Idx[1], Idx[1]], UInt32(15))


def test_tensor_span_constructor() raises:
    var bytes: List[UInt8] = [0, 1, 2, 3]
    var _tensor = TileTensor(
        bytes,
        row_major(Idx[2], Idx[2]),
    )


def test_fill() raises:
    var stack = Array[UInt32, 16](fill=0)
    var tensor = TileTensor(stack, row_major[4, 4]()).fill(1)
    for i in range(Int(tensor.layout.shape[0]().value())):
        for j in range(Int(tensor.layout.shape[1]().value())):
            assert_equal(tensor[i, j], 1)


def test_fill_large() raises:
    # layout._fillers.BATCH_SIZE is 2048, so we do 4096
    var stack = Array[UInt32, 4096](fill=0)
    var tensor = TileTensor(stack, row_major[2048, 2]()).fill(1)
    for i in range(Int(tensor.layout.shape[0]().value())):
        for j in range(Int(tensor.layout.shape[1]().value())):
            assert_equal(tensor[i, j], 1)


def test_slice() raises:
    """Test tensor slicing functionality."""
    # Test 2D slice (most common case)
    # Initialize with values 0-15 in row-major order
    var data_2d = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 4x4 tensor:
    # [0  1  2  3]
    # [4  5  6  7]
    # [8  9  10 11]
    # [12 13 14 15]
    var tensor_2d = TileTensor(data_2d, row_major[4, 4]())

    # Slice to extract middle 2x2 region [1:3, 1:3]:
    # [5  6]
    # [9  10]
    var sliced = tensor_2d.slice[1:3, 1:3]()

    # Verify slice dimensions
    assert_equal(sliced.layout.shape[0]().value(), 2)
    assert_equal(sliced.layout.shape[1]().value(), 2)

    # Verify slice values - use runtime indices since slice returns runtime shapes
    assert_equal(sliced[0, 0], 5)
    assert_equal(sliced[0, 1], 6)
    assert_equal(sliced[1, 0], 9)
    assert_equal(sliced[1, 1], 10)

    # Test that slice is a view (modifying slice affects original)
    sliced[Coord(Idx[0], Idx[0])] = 99
    assert_equal(tensor_2d[Idx[1], Idx[1]], 99)

    # Test different slice ranges
    var top_left = tensor_2d.slice[0:2, 0:2]()
    assert_equal(top_left[0, 0], 0)
    assert_equal(top_left[0, 1], 1)
    assert_equal(top_left[1, 0], 4)
    assert_equal(top_left[1, 1], 99)  # Modified earlier

    # Test slice with start=0 (should work with default)
    var first_row = tensor_2d.slice[0:1, 0:4]()
    assert_equal(first_row.layout.shape[0]().value(), 1)
    assert_equal(first_row.layout.shape[1]().value(), 4)
    assert_equal(first_row[0, 0], 0)
    assert_equal(first_row[0, 3], 3)


def test_slice_3d() raises:
    """Test 3D tensor slicing."""
    # Create a 4x4x4 tensor
    var data_3d = Array[Int32, 64](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor_3d = TileTensor(data_3d, row_major[4, 4, 4]())

    # Slice [1:3, 1:3, 1:3] to get a 2x2x2 cube from the middle
    var sliced_3d = tensor_3d.slice[1:3, 1:3, 1:3]()

    # Verify dimensions
    assert_equal(sliced_3d.layout.shape[0]().value(), 2)
    assert_equal(sliced_3d.layout.shape[1]().value(), 2)
    assert_equal(sliced_3d.layout.shape[2]().value(), 2)

    # Verify some values - use runtime indices
    # Original tensor[1][1][1] = 1*16 + 1*4 + 1 = 21
    assert_equal(sliced_3d[0, 0, 0], 21)

    # Original tensor[1][1][2] = 1*16 + 1*4 + 2 = 22
    assert_equal(sliced_3d[0, 0, 1], 22)

    # Original tensor[2][2][2] = 2*16 + 2*4 + 2 = 42
    assert_equal(sliced_3d[1, 1, 1], 42)

    # Test that it's a view
    sliced_3d[Coord(Idx[0], Idx[0], Idx[0])] = 999
    assert_equal(tensor_3d[Coord(Idx[1], Idx[1], Idx[1])], 999)


# def test_slice_runtime_shapes() raises:
#     """Test slicing with runtime-shaped tensors."""
#     var data = Array[Float32, 12](uninitialized=True)
#
#     for i in range(12):
#         data[i] = Float32(i)
#
#     # Create tensor with runtime shapes
#     var shape = Coord(
#         Int32(3), Int32(4)
#     )
#     var stride = Coord(
#         Int32(4), Int32(1)
#     )
#     var layout = MixedLayout(shape^, stride^)
#
#     var tensor_runtime = TileTensor(
#         data.unsafe_ptr(), layout^
#     )
#
#     # Slice with compile-time slice bounds
#     var sliced = tensor_runtime.slice[1:3, 1:3]()
#
#     # Verify dimensions (result should be runtime too)
#     assert_equal(sliced.layout.shape[0].value(), 2)
#     assert_equal(sliced.layout.shape[1].value(), 2)
#
#     # Verify values - use runtime indices
#     # Original[1][1] = 1 * 4 + 1 = 5
#     assert_equal(sliced[(Idx[0], Idx[0])], Float32(5))
#
#     # Original[2][2] = 2*4 + 2 = 10
#     assert_equal(sliced[(Idx[1], Idx[1])], Float32(10))


def test_slice_dynamic() raises:
    """Test slice with runtime (start, end) tuples."""
    var data_2d = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # 4x4 row-major:
    # [0  1  2  3]
    # [4  5  6  7]
    # [8  9  10 11]
    # [12 13 14 15]
    var tensor_2d = TileTensor(data_2d, row_major[4, 4]())

    # Slice middle 2x2: rows [1:3], cols [1:3] -> [5,6],[9,10]
    var sliced = tensor_2d.slice((1, 3), (1, 3))
    assert_equal(sliced.layout.shape[0]().value(), 2)
    assert_equal(sliced.layout.shape[1]().value(), 2)
    assert_equal(sliced[0, 0], 5)
    assert_equal(sliced[0, 1], 6)
    assert_equal(sliced[1, 0], 9)
    assert_equal(sliced[1, 1], 10)

    # Top-left 2x2
    var top_left = tensor_2d.slice((0, 2), (0, 2))
    assert_equal(top_left[0, 0], 0)
    assert_equal(top_left[0, 1], 1)
    assert_equal(top_left[1, 0], 4)
    assert_equal(top_left[1, 1], 5)

    # Single row: rows [2:3], all cols
    var row2 = tensor_2d.slice((2, 3), (0, 4))
    assert_equal(row2.layout.shape[0]().value(), 1)
    assert_equal(row2.layout.shape[1]().value(), 4)
    assert_equal(row2[0, 0], 8)
    assert_equal(row2[0, 3], 11)

    # Verify it's a view
    sliced[Coord(Idx[0], Idx[0])] = 99
    assert_equal(tensor_2d[Coord(Idx[1], Idx[1])], 99)


def test_vectorize() raises:
    """Test tensor vectorization functionality."""
    # Create a 16x16 tensor with row-major layout
    # Initialize with sequential values
    var data = Array[Int32, 256](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[16, 16]())

    # Vectorize with 4x4 blocks
    var vectorized = tensor.vectorize[4, 4]()

    # Verify vectorized tensor shape: 16/4 x 16/4 = 4x4
    assert_equal(vectorized.layout.shape[0]().value(), 4)
    assert_equal(vectorized.layout.shape[1]().value(), 4)

    # Verify vectorized tensor strides: original_stride * vector_shape
    # Original row-major 16x16 has strides [16, 1]
    # Vectorized strides should be [16*4, 1*4] = [64, 4]
    assert_equal(vectorized.layout.stride[0]().value(), 64)
    assert_equal(vectorized.layout.stride[1]().value(), 4)

    # Verify that vectorized[i, j] returns a SIMD vector starting at the (i,j) block
    # Block (0, 0) starts at element 0 - check first element of the SIMD vector
    assert_equal(vectorized[Coord(Idx[0], Idx[0])][0], 0)

    # Block (0, 1) starts at element 4 (column offset by vector width)
    assert_equal(vectorized[Coord(Idx[0], Idx[1])][0], 4)

    # Block (1, 0) starts at element 64 (row offset by vector height * row stride)
    assert_equal(vectorized[Coord(Idx[1], Idx[0])][0], 64)

    # Block (1, 1) starts at element 68
    assert_equal(vectorized[Coord(Idx[1], Idx[1])][0], 68)

    # Block (3, 3) is the last block, starts at element 3*64 + 3*4 = 204
    assert_equal(vectorized[Coord(Idx[3], Idx[3])][0], 204)


def test_vectorize_non_square() raises:
    """Test vectorization with non-square vector shapes."""
    var data = Array[Int32, 64](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 8x8 tensor
    var tensor = TileTensor(data, row_major[8, 8]())

    # Vectorize with 2x4 blocks (different dimensions)
    var vectorized = tensor.vectorize[2, 4]()

    # Shape should be 8/2 x 8/4 = 4x2
    assert_equal(vectorized.layout.shape[0]().value(), 4)
    assert_equal(vectorized.layout.shape[1]().value(), 2)

    # Strides should be [8*2, 1*4] = [16, 4]
    assert_equal(vectorized.layout.stride[0]().value(), 16)
    assert_equal(vectorized.layout.stride[1]().value(), 4)

    # Verify block positions - check first element of each SIMD vector
    assert_equal(vectorized[Idx[0], Idx[0]][0], 0)  # Block (0,0) at element 0
    assert_equal(vectorized[Idx[0], Idx[1]][0], 4)  # Block (0,1) at element 4
    assert_equal(vectorized[Idx[1], Idx[0]][0], 16)  # Block (1,0) at element 16
    assert_equal(vectorized[Idx[3], Idx[1]][0], 52)  # Block (3,1) at element 52


def test_vectorize_1d() raises:
    """Test vectorization of 1D tensor."""
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 16-element 1D tensor
    var tensor = TileTensor(data, row_major[16]())

    # Vectorize with width 4
    var vectorized = tensor.vectorize[4]()

    # Shape should be 16/4 = 4
    assert_equal(vectorized.layout.shape[0]().value(), 4)

    # Stride should be 1*4 = 4
    assert_equal(vectorized.layout.stride[0]().value(), 4)

    # Verify block positions - check first element of each SIMD vector
    assert_equal(vectorized[Idx[0]][0], 0)
    assert_equal(vectorized[Idx[1]][0], 4)
    assert_equal(vectorized[Idx[2]][0], 8)
    assert_equal(vectorized[Idx[3]][0], 12)


def test_vectorize_runtime_dims() raises:
    """Test vectorize works when tensor has runtime dimensions."""
    var data = Array[Int32, 64](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 8x8 tensor with runtime first dimension, static second.
    var tensor = TileTensor(data, row_major(Int(8), Idx[8]))

    # Vectorize with 2x4 blocks.
    var vectorized = tensor.vectorize[2, 4]()

    # Shape: 8/2 x 8/4 = 4x2
    assert_equal(vectorized.layout.shape[0]().value(), 4)
    assert_equal(vectorized.layout.shape[1]().value(), 2)

    # Strides: original row-major 8x8 has strides [8, 1]
    # Vectorized: [8*2, 1*4] = [16, 4]
    assert_equal(vectorized.layout.stride[0]().value(), 16)
    assert_equal(vectorized.layout.stride[1]().value(), 4)

    # Verify block positions.
    assert_equal(vectorized[Idx[0], Idx[0]][0], 0)
    assert_equal(vectorized[Idx[0], Idx[1]][0], 4)
    assert_equal(vectorized[Idx[1], Idx[0]][0], 16)
    assert_equal(vectorized[Idx[3], Idx[1]][0], 52)


def test_vectorize_fully_runtime_dims() raises:
    """Test vectorize with all dimensions runtime."""
    var data = Array[Int32, 256](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Both dims runtime.
    var tensor = TileTensor(data, row_major(Int(16), Int(16)))

    var vectorized = tensor.vectorize[4, 4]()

    # Shape: 16/4 x 16/4 = 4x4
    assert_equal(vectorized.layout.shape[0]().value(), 4)
    assert_equal(vectorized.layout.shape[1]().value(), 4)

    # Strides: [16*4, 16*4] wait no — [16*4, 1*4] = [64, 4]
    # But stride[0] is runtime (16 is runtime) so 16*4=64
    # stride[1] is runtime (1 is runtime from row_major with runtime dims)
    assert_equal(vectorized.layout.stride[0]().value(), 64)
    assert_equal(vectorized.layout.stride[1]().value(), 4)

    assert_equal(vectorized[Coord(Idx[0], Idx[0])][0], 0)
    assert_equal(vectorized[Coord(Idx[0], Idx[1])][0], 4)
    assert_equal(vectorized[Coord(Idx[1], Idx[0])][0], 64)
    assert_equal(vectorized[Coord(Idx[3], Idx[3])][0], 204)


def test_distribute_runtime_dims() raises:
    """Test distribute works when tensor has runtime dimensions."""
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var array = Array[UInt32, 16](fill=-1)
    var ptr: MutPointer[UInt32, origin_of(array)] = array.unsafe_ptr()

    # Create 4x4 tensor with runtime first dim.
    var layout_tensor = TileTensor(
        ptr=ptr,
        layout=TileLayout(
            shape=Coord(Int(4), Idx[4]),
            stride=Coord(Int(4), Idx[1]),
        ),
    )

    var counter = 0
    for th_id in range(4):
        var frag = layout_tensor.distribute[thread_layout=thread_layout](th_id)

        for i in range(2):
            for j in range(2):
                frag[i, j] = UInt32(counter)
                counter += 1

    # Same expected layout as the all-static test.
    var expected = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    for i in range(16):
        assert_equal(ptr[i], UInt32(expected[i]))


def test_distribute_with_offset_runtime_dims() raises:
    """Test distribute_with_offset works when tensor has runtime dimensions."""
    comptime thread_layout = row_major(Idx[2], Idx[2])

    var array = Array[UInt32, 16](fill=-1)
    var ptr: MutPointer[UInt32, origin_of(array)] = array.unsafe_ptr()

    # Create 4x4 tensor with runtime dims.
    var layout_tensor = TileTensor(
        ptr=ptr,
        layout=TileLayout(
            shape=Coord(Int(4), Idx[4]),
            stride=Coord(Int(4), Idx[1]),
        ),
    )

    var counter = 0
    for th_id in range(4):
        var result = layout_tensor.distribute_with_offset[
            thread_layout=thread_layout
        ](th_id)
        var frag = result[0]
        # result[1] is thread_coords, result[2] is offset

        for i in range(2):
            for j in range(2):
                frag[i, j] = UInt32(counter)
                counter += 1

    var expected = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
    for i in range(16):
        assert_equal(ptr[i], UInt32(expected[i]))


def test_indexing() raises:
    var stack: Array[UInt8, 4] = [1, 2, 3, 4]
    var tensor = TileTensor(stack, row_major[2, 2]())
    assert_equal(tensor[Int32(0), Int64(0)], 1)
    assert_equal(tensor[Int(1), Int64(0)], 3)


def test_to_layout_tensor_square() raises:
    var stack: Array[UInt8, 4] = [1, 2, 3, 4]
    var tensor = TileTensor(stack, row_major[2, 2]()).to_layout_tensor()
    assert_equal(materialize[tensor.layout](), Layout.row_major(2, 2))
    assert_equal(tensor.rank, 2)
    assert_equal(
        rebind[IndexList[2]](tensor.runtime_layout.shape.value.canonicalize()),
        IndexList[2](2, 2),
    )


def test_to_layout_tensor_3d() raises:
    var stack = Array[UInt8, 64 * 8 * 4](fill=0)
    var tensor = TileTensor(stack, row_major[64, 8, 4]())
    var lt = tensor.to_layout_tensor()
    assert_equal(materialize[lt.layout](), Layout.row_major(64, 8, 4))
    assert_equal(lt.rank, 3)
    assert_equal(
        rebind[IndexList[3]](lt.runtime_layout.shape.value.canonicalize()),
        IndexList[3](64, 8, 4),
    )


def test_to_layout_tensor_3d_dynamic() raises:
    var stack = Array[UInt8, 64 * 8 * 4](fill=0)
    var tensor = TileTensor(stack, row_major(Idx[64], Idx[8], Int(4)))
    var lt = tensor.to_layout_tensor()
    assert_equal(
        materialize[lt.layout](),
        Layout.row_major(64, 8, UNKNOWN_VALUE),
    )
    assert_equal(lt.rank, 3)
    assert_equal(
        rebind[IndexList[3]](lt.runtime_layout.shape.value.canonicalize()),
        IndexList[3](64, 8, 4),
    )


def test_coalesce_2d() raises:
    """Test coalescing a 2D tensor to rank-1."""
    # Initialize with sequential values
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 4x4 tensor
    var tensor = TileTensor(data, row_major[4, 4]())

    # Coalesce to rank-1
    var coalesced = tensor.coalesce()

    # Verify coalesced tensor shape: 4*4 = 16
    assert_equal(coalesced.layout.shape[0]().value(), 16)

    # Verify coalesced tensor stride: 1
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify elements are accessible in order
    for i in range(16):
        assert_equal(coalesced[i], Int32(i))


def test_coalesce_3d() raises:
    """Test coalescing a 3D tensor to rank-1."""
    var data = Array[Int32, 24](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 2x3x4 tensor
    var tensor = TileTensor(data, row_major[2, 3, 4]())

    # Coalesce to rank-1
    var coalesced = tensor.coalesce()

    # Verify coalesced tensor shape: 2*3*4 = 24
    assert_equal(coalesced.layout.shape[0]().value(), 24)

    # Verify coalesced tensor stride: 1
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify elements are accessible in order
    for i in range(24):
        assert_equal(coalesced[i], Int32(i))


def test_coalesce_1d() raises:
    """Test coalescing a 1D tensor (should be no-op effectively)."""
    var data = Array[Int32, 8](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 8-element 1D tensor
    var tensor = TileTensor(data, row_major[8]())

    # Coalesce (should maintain rank-1)
    var coalesced = tensor.coalesce()

    # Verify shape and stride unchanged
    assert_equal(coalesced.layout.shape[0]().value(), 8)
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify elements
    for i in range(8):
        assert_equal(coalesced[i], Int32(i))


def test_coalesce_element_size() raises:
    """Test that coalesce properly tracks element_size."""
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # Create 4x4 tensor
    var tensor = TileTensor(data, row_major[4, 4]())

    # Verify element_size is 1 for non-vectorized tensor
    assert_equal(tensor.element_size, 1)

    # Coalesce the tensor
    var coalesced = tensor.coalesce()

    # Verify coalesced shape: 4*4 = 16
    assert_equal(coalesced.layout.shape[0]().value(), 16)
    assert_equal(coalesced.layout.stride[0]().value(), 1)

    # Verify element_size is still 1 (coalesced from 1)
    assert_equal(coalesced.element_size, 1)

    # Verify all elements accessible
    for i in range(16):
        assert_equal(coalesced[i], Int32(i))


def test_load_store_linear_row_major() raises:
    # 3x4 row-major: strides are (4, 1)
    var data = Array[Int32, 12](fill=0)
    for i in range(12):
        data[i] = Int32(i * 10)

    var tensor = TileTensor(data, row_major(Idx[3], Idx[4]))

    # Verify load_linear at known positions.
    assert_equal(Int(tensor.load_linear[1](IndexList[2](0, 0))), 0)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](0, 3))), 30)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](1, 0))), 40)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](2, 3))), 110)

    # Verify vectorized load (width=2).
    var vec = tensor.load_linear[2](IndexList[2](1, 0))
    assert_equal(Int(vec[0]), 40)
    assert_equal(Int(vec[1]), 50)

    # Verify store_linear.
    tensor.store_linear(IndexList[2](0, 1), Int32(999))
    assert_equal(Int(tensor.load_linear[1](IndexList[2](0, 1))), 999)

    # Verify vectorized store (width=2).
    tensor.store_linear(IndexList[2](2, 0), SIMD[.int32, 2](77, 88))
    assert_equal(Int(tensor.load_linear[1](IndexList[2](2, 0))), 77)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](2, 1))), 88)


def test_load_store_linear_non_trivial_stride() raises:
    # 2x3 column-major: shape (2,3), strides (1,2) — non-contiguous access
    var data = Array[Int32, 6](fill=0)
    for i in range(6):
        data[i] = Int32(i)

    # Column-major layout: stride[0]=1, stride[1]=2
    comptime col_major_shape = Coord[ComptimeInt[2], ComptimeInt[3]]
    comptime col_major_stride = Coord[ComptimeInt[1], ComptimeInt[2]]
    var data_ptr: MutPointer[Int32, origin_of(data)] = data.unsafe_ptr()
    var tensor = TileTensor(
        ptr=data_ptr,
        layout=TileLayout(
            shape=col_major_shape(Idx[2], Idx[3]),
            stride=col_major_stride(Idx[1], Idx[2]),
        ),
    )

    # In column-major with strides (1,2), linear offset = row*1 + col*2
    # data[0]=0, data[1]=1, data[2]=2, data[3]=3, data[4]=4, data[5]=5
    # (0,0) -> offset 0 -> data[0] = 0
    # (1,0) -> offset 1 -> data[1] = 1
    # (0,1) -> offset 2 -> data[2] = 2
    # (1,1) -> offset 3 -> data[3] = 3
    # (0,2) -> offset 4 -> data[4] = 4
    # (1,2) -> offset 5 -> data[5] = 5
    assert_equal(Int(tensor.load_linear[1](IndexList[2](0, 0))), 0)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](1, 0))), 1)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](0, 1))), 2)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](1, 1))), 3)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](0, 2))), 4)
    assert_equal(Int(tensor.load_linear[1](IndexList[2](1, 2))), 5)

    # Store and verify.
    tensor.store_linear(IndexList[2](1, 1), Int32(42))
    assert_equal(Int(tensor.load_linear[1](IndexList[2](1, 1))), 42)
    # Verify underlying data: offset 3 should be 42.
    assert_equal(Int(data[3]), 42)


def test_linear_idx_type_small_static_layout() raises:
    """Small fully-static layouts use int32 for linear_idx_type."""
    # Cosize = (4-1)*4 + (4-1)*1 + 1 = 16, fits in int32
    comptime TensorType = TileTensor[
        .float32, RowMajorLayout[ComptimeInt[4], ComptimeInt[4]], MutAnyOrigin
    ]
    comptime assert TensorType.linear_idx_type == .int32


def test_linear_idx_type_dynamic_layout_generic() raises:
    """Dynamic layouts in GENERIC address space use int64."""
    comptime TensorType = TileTensor[
        .float32, RowMajorLayout[Int, ComptimeInt[4]], MutAnyOrigin
    ]
    # Not all dims known -> falls through to address_space check -> GENERIC -> int64
    comptime assert TensorType.linear_idx_type == .int64


def test_linear_idx_type_shared_address_space() raises:
    """Shared memory address space always uses int32."""
    comptime TensorType = TileTensor[
        .float32,
        RowMajorLayout[ComptimeInt[4], ComptimeInt[4]],
        MutAnyOrigin,
        address_space=.SHARED,
    ]
    comptime assert TensorType.linear_idx_type == .int32


def test_linear_idx_type_recomputed_after_tile() raises:
    """After tile(), linear_idx_type is recomputed from the new layout."""
    var stack = Array[Int32, 256](fill=0)
    var tensor = TileTensor(stack, row_major[16, 16]())
    # Original: cosize=256, int32
    assert_equal(type_of(tensor).linear_idx_type, DType.int32)

    # After tiling: new layout has smaller cosize, still int32
    var tiled = tensor.tile[4, 4]((Idx[0], Idx[0]))
    assert_equal(type_of(tiled).linear_idx_type, DType.int32)
    _ = tiled


def test_linear_idx_type_recomputed_after_distribute() raises:
    """After distribute(), linear_idx_type is recomputed from the new layout."""
    var stack = Array[Int32, 16](fill=0)
    var tensor = TileTensor(stack, row_major[4, 4]())
    assert_equal(type_of(tensor).linear_idx_type, DType.int32)

    comptime thread_layout = row_major(Idx[2], Idx[2])
    var frag = tensor.distribute[thread_layout=thread_layout](0)
    # Distributed fragment: shape [2,2], strides [8,2] -> cosize = (2-1)*8 + (2-1)*2 + 1 = 11
    assert_equal(type_of(frag).linear_idx_type, DType.int32)
    _ = frag


def test_linear_idx_type_recomputed_after_vectorize() raises:
    """After vectorize(), linear_idx_type is recomputed from the new layout."""
    var stack = Array[Int32, 256](fill=0)
    var tensor = TileTensor(stack, row_major[16, 16]())
    assert_equal(type_of(tensor).linear_idx_type, DType.int32)

    var vectorized = tensor.vectorize[4, 4]()
    # Vectorized: shape [4,4], strides [64,4] -> cosize = (4-1)*64 + (4-1)*4 + 1 = 205
    assert_equal(type_of(vectorized).linear_idx_type, DType.int32)
    _ = vectorized


def test_transpose_2d() raises:
    """Test transpose on a 2D tensor swaps rows and columns."""
    var data = Array[Int32, 12](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # 3x4 row-major:
    # [0  1  2  3]
    # [4  5  6  7]
    # [8  9  10 11]
    var tensor = TileTensor(data, row_major[3, 4]())
    var trans = tensor.transpose()

    # Transposed shape should be (4, 3)
    assert_equal(trans.dim[0](), 4)
    assert_equal(trans.dim[1](), 3)

    # Transposed strides: original (4, 1) -> reversed (1, 4)
    assert_equal(trans.layout.stride[0]().value(), 1)
    assert_equal(trans.layout.stride[1]().value(), 4)

    # Verify element access: trans[col, row] == tensor[row, col]
    assert_equal(trans[0, 0], 0)
    assert_equal(trans[1, 0], 1)
    assert_equal(trans[2, 0], 2)
    assert_equal(trans[3, 0], 3)
    assert_equal(trans[0, 1], 4)
    assert_equal(trans[0, 2], 8)
    assert_equal(trans[3, 2], 11)


def test_transpose_is_view() raises:
    """Test that transpose creates a view sharing memory with the original."""
    var data = Array[Int32, 6](fill={})
    var tensor = TileTensor(data, row_major[2, 3]()).fill(0)

    var trans = tensor.transpose()

    # Modify through transposed view: trans[1, 0] -> tensor[0, 1]
    trans[Coord(Idx[1], Idx[0])] = 42
    assert_equal(tensor[Coord(Idx[0], Idx[1])], 42)

    # Modify through original: tensor[1, 2] -> trans[2, 1]
    tensor[Coord(Idx[1], Idx[2])] = 99
    assert_equal(trans[Coord(Idx[2], Idx[1])], 99)


def test_transpose_square() raises:
    """Test transpose on a square tensor."""
    var data = Array[Int32, 9](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # 3x3 row-major:
    # [0 1 2]
    # [3 4 5]
    # [6 7 8]
    var tensor = TileTensor(data, row_major[3, 3]())
    var trans = tensor.transpose()

    assert_equal(trans.dim[0](), 3)
    assert_equal(trans.dim[1](), 3)

    # Diagonal unchanged
    assert_equal(trans[0, 0], 0)
    assert_equal(trans[1, 1], 4)
    assert_equal(trans[2, 2], 8)

    # Off-diagonal swapped
    assert_equal(trans[0, 1], 3)  # was tensor[1, 0]
    assert_equal(trans[1, 0], 1)  # was tensor[0, 1]
    assert_equal(trans[0, 2], 6)  # was tensor[2, 0]
    assert_equal(trans[2, 0], 2)  # was tensor[0, 2]


def test_transpose_1d() raises:
    """Test transpose on a 1D tensor (identity operation)."""
    var data = Array[Int32, 4](
        fill_with=lambda (i: Int) -> Int32: Int32(i * 10)
    )

    var tensor = TileTensor(data, row_major[4]())
    var trans = tensor.transpose()

    assert_equal(trans.dim[0](), 4)
    assert_equal(trans.layout.stride[0]().value(), 1)

    for i in range(4):
        assert_equal(trans[i], Int32(i * 10))


def test_transpose_preserves_element_count() raises:
    """Test that transpose preserves the total number of elements."""
    var data = Array[Int32, 20](fill={})
    var tensor = TileTensor(data, row_major[4, 5]()).fill(1)
    var trans = tensor.transpose()

    assert_equal(trans.num_elements(), tensor.num_elements())
    assert_equal(trans.num_elements(), 20)


def test_select_4d_to_2d() raises:
    """Test selecting from a 4D tensor to a 2D tensor (CuTE-style)."""
    # 4D tensor: (batch=2, N=3, heads=4, head_dim=2)
    comptime total = 2 * 3 * 4 * 2
    var data = Array[Int32, total](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[2, 3, 4, 2]())

    # Fix batch=1 and heads=2, keep N and head_dim → 2D (3, 2)
    var selected = tensor.slice(Idx[1], All, Idx[2], All)

    # Output should be rank 2 with shape (3, 2)
    assert_equal(selected.layout.shape[0]().value(), 3)
    assert_equal(selected.layout.shape[1]().value(), 2)

    # Verify values:
    # batch=1 offset: 1 * (3*4*2) = 24
    # heads=2 offset: 2 * 2 = 4
    # So base = 24 + 4 = 28
    # selected[n, d] = tensor[1, n, 2, d] = 28 + n*8 + d
    for n in range(3):
        for d in range(2):
            assert_equal(selected[n, d], Int32(28 + n * 8 + d))

    # Test that it's a view (modifying selected affects original)
    selected[Coord(Idx[0], Idx[0])] = 999
    assert_equal(tensor[Idx[1], Idx[0], Idx[2], Idx[0]], 999)


def test_select_preserves_comptime_dims() raises:
    """Test that select preserves compile-time shape and stride info."""
    var data = Array[Int32, 48](fill={})
    var tensor = TileTensor(data, row_major[2, 3, 4, 2]())

    _ = tensor.slice(Idx[0], All, Idx[1], All)

    # Shape should be ComptimeInt[3] and ComptimeInt[2]
    comptime SelectedType = type_of(
        TileTensor(data, row_major[2, 3, 4, 2]()).slice(
            Idx[0], All, Idx[1], All
        )
    )
    comptime assert SelectedType.LayoutType.static_shape[0] == 3
    comptime assert SelectedType.LayoutType.static_shape[1] == 2

    # Strides should be ComptimeInt[8] and ComptimeInt[1]
    comptime assert SelectedType.LayoutType.static_stride[0] == 8
    comptime assert SelectedType.LayoutType.static_stride[1] == 1


def test_select_3d_to_1d() raises:
    """Test selecting from a 3D tensor to a 1D tensor."""
    var data = Array[Int32, 24](fill_with=lambda (i: Int) -> Int32: Int32(i))

    # (2, 3, 4) tensor
    var tensor = TileTensor(data, row_major[2, 3, 4]())

    # Fix dims 0 and 1, keep dim 2 → 1D (4,)
    var selected = tensor.slice(Idx[1], Idx[2], All)

    assert_equal(selected.layout.shape[0]().value(), 4)

    # tensor[1, 2, d] = 1*12 + 2*4 + d = 20 + d
    for d in range(4):
        assert_equal(selected[d], Int32(20 + d))


def test_select_keep_all() raises:
    """Test selecting with all dimensions kept (identity)."""
    var data = Array[Int32, 12](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[3, 4]())
    var selected = tensor.slice(All, All)

    assert_equal(selected.layout.shape[0]().value(), 3)
    assert_equal(selected.layout.shape[1]().value(), 4)

    for i in range(3):
        for j in range(4):
            assert_equal(selected[i, j], Int32(i * 4 + j))


def test_write_to_1d() raises:
    comptime layout = row_major[4]()
    var storage: Array[Float32, layout.static_product] = [1, 2, 3, 4]
    var tensor = TileTensor(storage, layout)
    assert_equal(String(tensor), "[1.0, 2.0, 3.0, 4.0]")


def test_write_to_1d_single_element() raises:
    var storage: Array[Float32, 1] = [42]
    var tensor = TileTensor(storage, row_major[1]())
    assert_equal(String(tensor), "[42.0]")


def test_write_to_2d() raises:
    var storage: Array[Float32, 6] = [1, 2, 3, 4, 5, 6]
    var tensor = TileTensor(storage, row_major[2, 3]())
    assert_equal(String(tensor), "[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]")


def test_write_to_2d_dynamic() raises:
    """A 2D tensor with a runtime dimension falls through to the elementwise
    printer because `static_shape` is unknown."""
    var storage: Array[Float32, 6] = [1, 2, 3, 4, 5, 6]
    var tensor = TileTensor(storage, row_major(Idx[2], Int(3)))
    # Elementwise printer iterates in column-major coordinate order.
    assert_equal(String(tensor), "[1.0, 4.0, 2.0, 5.0, 3.0, 6.0]")


def test_write_to_3d() raises:
    """Test that printing a 3D TileTensor produces output via the generic
    fallback."""
    var storage: Array[Float32, 8] = [1, 2, 3, 4, 5, 6, 7, 8]
    var tensor = TileTensor(storage, row_major[2, 2, 2]())
    # Elements are printed in column-major coordinate order.
    assert_equal(String(tensor), "[1.0, 5.0, 3.0, 7.0, 2.0, 6.0, 4.0, 8.0]")


def test_copy_from_roundtrip_scalar() raises:
    """Roundtrip `TileTensor.copy_from` with element_size == 1 restores original
    data."""
    var src_data = Array[Int32, 16](
        fill_with=lambda (i: Int) -> Int32: Int32(i + 1)
    )

    var mid_data = Array[Int32, 16](fill=0)
    var dst_data = Array[Int32, 16](fill=0)

    var src = TileTensor(src_data, row_major[4, 4]())
    var mid = TileTensor(mid_data, row_major[4, 4]())
    var dst = TileTensor(dst_data, row_major[4, 4]())

    mid.copy_from(src)
    dst.copy_from(mid)

    for i in range(16):
        assert_equal(dst_data[i], src_data[i])


def test_copy_from_roundtrip_vectorized() raises:
    """Roundtrip `TileTensor.copy_from` when both sides have element_size > 1.

    Regression test for the element_size-vs-scalar-count bug: previously
    this would only touch 1/element_size of the underlying scalars.
    """
    var src_data = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i + 1)
    )

    var mid_data = Array[Float32, 16](fill=0)
    var dst_data = Array[Float32, 16](fill=0)

    # Vectorize 4x4 -> 4x1 with element_size == 4.
    var src = TileTensor(src_data, row_major[4, 4]()).vectorize[1, 4]()
    var mid = TileTensor(mid_data, row_major[4, 4]()).vectorize[1, 4]()
    var dst = TileTensor(dst_data, row_major[4, 4]()).vectorize[1, 4]()

    mid.copy_from(src)
    dst.copy_from(mid)

    for i in range(16):
        assert_equal(dst_data[i], src_data[i])


def test_copy_from_roundtrip_tile_by_tile() raises:
    """Roundtrip a full tensor tile-by-tile through an intermediate.

    Exercises `TileTensor.copy_from` on sub-tiles (so the layout strides don't
    equal the full-tensor strides) and verifies a full tile-by-tile
    round-trip restores every element of the original tensor.
    """
    var src_data = Array[Int32, 16](
        fill_with=lambda (i: Int) -> Int32: Int32(i + 1)
    )

    var mid_data = Array[Int32, 16](fill=0)
    var dst_data = Array[Int32, 16](fill=0)

    var src = TileTensor(src_data, row_major[4, 4]())
    var mid = TileTensor(mid_data, row_major[4, 4]())
    var dst = TileTensor(dst_data, row_major[4, 4]())

    comptime for tile_i in range(2):
        comptime for tile_j in range(2):
            var s = src.tile[2, 2]((tile_i, tile_j))
            var m = mid.tile[2, 2]((tile_i, tile_j))
            m.copy_from(s)

    comptime for tile_i in range(2):
        comptime for tile_j in range(2):
            var m = mid.tile[2, 2]((tile_i, tile_j))
            var d = dst.tile[2, 2]((tile_i, tile_j))
            d.copy_from(m)

    for i in range(16):
        assert_equal(dst_data[i], src_data[i])


def test_copy_from_respects_non_contiguous_layout() raises:
    """Copy into a padded layout without touching padding slots."""
    var src_data = Array[Int32, 6](
        fill_with=lambda (i: Int) -> Int32: Int32(i + 1)
    )

    var dst_data = Array[Int32, 8](fill=-1)

    var src = TileTensor(src_data, row_major[2, 3]())

    comptime padded_shape = Coord[ComptimeInt[2], ComptimeInt[3]]
    comptime padded_stride = Coord[ComptimeInt[4], ComptimeInt[1]]
    var dst_data_ptr: MutPointer[
        Int32, origin_of(dst_data)
    ] = dst_data.unsafe_ptr()
    var dst = TileTensor(
        ptr=dst_data_ptr,
        layout=TileLayout(
            shape=padded_shape(Idx[2], Idx[3]),
            stride=padded_stride(Idx[4], Idx[1]),
        ),
    )

    dst.copy_from(src)

    var expected = [1, 2, 3, -1, 4, 5, 6, -1]
    for i in range(8):
        assert_equal(dst_data[i], Int32(expected[i]))


def test_copy_from_converts_dtype() raises:
    """Copy between different dtypes by casting each logical element."""
    var src_data: Array[Int32, 4] = [1, 2, 3, 4]
    var dst_data = Array[Float32, 4](fill=0)

    var src = TileTensor(src_data, row_major[2, 2]())
    var dst = TileTensor(dst_data, row_major[2, 2]())

    dst.copy_from(src)

    for i in range(4):
        assert_equal(dst_data[i], Float32(i + 1))


def test_copy_from_contiguous_converts_dtype() raises:
    """Copy contiguous data between different dtypes."""
    var src_data = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i + 1)
    )

    var dst_data = Array[BFloat16, 16](fill=0)

    var src = TileTensor(src_data, row_major[4, 4]())
    var dst = TileTensor(dst_data, row_major[4, 4]())

    dst.copy_from(src)

    for i in range(16):
        assert_equal(dst_data[i], BFloat16(i + 1))


def test_copy_from_vectorized_converts_dtype() raises:
    """Copy vectorized elements between different dtypes."""
    var src_data = Array[Float32, 16](
        fill_with=lambda (i: Int) -> Float32: Float32(i + 1)
    )

    var dst_data = Array[BFloat16, 16](fill=0)

    var src = TileTensor(src_data, row_major[4, 4]()).vectorize[1, 4]()
    var dst = TileTensor(dst_data, row_major[4, 4]()).vectorize[1, 4]()

    dst.copy_from(src)

    for i in range(16):
        assert_equal(dst_data[i], BFloat16(i + 1))


def test_split_static_axis0() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[4, 4]())
    var tiles = tensor.as_immut().split[2]()

    assert_equal(tiles[0][0, 0], 0)
    assert_equal(tiles[0][1, 3], 7)
    assert_equal(tiles[1][0, 0], 8)
    assert_equal(tiles[1][1, 3], 15)


def test_split_static_axis1() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[4, 4]())
    var tiles = tensor.as_immut().split[2, axis=1]()

    assert_equal(tiles[0][0, 0], 0)
    assert_equal(tiles[0][3, 1], 13)
    assert_equal(tiles[1][0, 0], 2)
    assert_equal(tiles[1][3, 1], 15)


def test_split_static_count_one() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[4, 4]())
    var tiles = tensor.as_immut().split[1]()

    assert_equal(tiles[0][0, 0], tensor[0, 0])
    assert_equal(tiles[0][2, 1], tensor[2, 1])
    assert_equal(tiles[0][3, 3], tensor[3, 3])


def test_split_static_vectorized() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[4, 4]()).vectorize[1, 2]()
    var tiles = tensor.as_immut().split[2]()

    assert_equal(tiles[0][0, 0][0], 0)
    assert_equal(tiles[0][1, 1][0], 6)
    assert_equal(tiles[1][0, 0][0], 8)
    assert_equal(tiles[1][1, 1][0], 14)


def test_split_static_after_tile_non_contiguous() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[4, 4]())
    var tile = tensor.as_immut().tile[4, 2]((Idx[0], Idx[1]))
    var splits = tile.split[2]()

    assert_equal(splits[0][0, 0], 2)
    assert_equal(splits[0][1, 1], 7)
    assert_equal(splits[1][0, 0], 10)
    assert_equal(splits[1][1, 1], 15)


def test_tile_after_split_static_non_contiguous() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[4, 4]())
    var splits = tensor.as_immut().split[2, axis=1]()
    var tile = splits[1].tile[2, 2]((Idx[1], Idx[0]))

    assert_equal(tile[0, 0], 10)
    assert_equal(tile[0, 1], 11)
    assert_equal(tile[1, 0], 14)
    assert_equal(tile[1, 1], 15)


def test_split_dynamic_axis0_without_alignment() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[8, 2]())
    var split0 = tensor.as_immut().split[axis=0](4, 0)
    var split1 = tensor.as_immut().split[axis=0](4, 1)
    var split3 = tensor.as_immut().split[axis=0](4, 3)

    assert_equal(split0.layout.shape[0]().value(), 2)
    assert_equal(split1.layout.shape[0]().value(), 2)
    assert_equal(split3.layout.shape[0]().value(), 2)

    assert_equal(split0[0, 0], 0)
    assert_equal(split0[1, 1], 3)
    assert_equal(split1[0, 0], 4)
    assert_equal(split1[1, 1], 7)
    assert_equal(split3[0, 0], 12)
    assert_equal(split3[1, 1], 15)


def test_split_dynamic_non_divisible_without_alignment() raises:
    var data = Array[Int32, 10](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[5, 2]())
    var split0 = tensor.as_immut().split[axis=0](2, 0)
    var split1 = tensor.as_immut().split[axis=0](2, 1)

    assert_equal(split0.layout.shape[0]().value(), 3)
    assert_equal(split1.layout.shape[0]().value(), 2)

    assert_equal(split0[0, 0], 0)
    assert_equal(split0[2, 1], 5)
    assert_equal(split1[0, 0], 6)
    assert_equal(split1[1, 1], 9)


def test_split_dynamic_axis1_with_alignment() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[2, 8]())
    var split0 = tensor.as_immut().split[1, split_alignment=3](3, 0)
    var split1 = tensor.as_immut().split[1, split_alignment=3](3, 1)
    var split2 = tensor.as_immut().split[1, split_alignment=3](3, 2)

    assert_equal(split0.layout.shape[1]().value(), 3)
    assert_equal(split1.layout.shape[1]().value(), 3)
    assert_equal(split2.layout.shape[1]().value(), 2)

    assert_equal(split0[0, 0], 0)
    assert_equal(split0[1, 2], 10)
    assert_equal(split1[0, 0], 3)
    assert_equal(split1[1, 2], 13)
    assert_equal(split2[0, 0], 6)
    assert_equal(split2[1, 1], 15)


def test_split_dynamic_alignment_trailing_zero_partition() raises:
    var data = Array[Int32, 16](fill_with=lambda (i: Int) -> Int32: Int32(i))

    var tensor = TileTensor(data, row_major[2, 8]())
    var split0 = tensor.as_immut().split[1, split_alignment=4](3, 0)
    var split1 = tensor.as_immut().split[1, split_alignment=4](3, 1)
    var split2 = tensor.as_immut().split[1, split_alignment=4](3, 2)

    assert_equal(split0.layout.shape[1]().value(), 4)
    assert_equal(split1.layout.shape[1]().value(), 4)
    assert_equal(split2.layout.shape[1]().value(), 0)

    assert_equal(split0[0, 0], 0)
    assert_equal(split0[1, 3], 11)
    assert_equal(split1[0, 0], 4)
    assert_equal(split1[1, 3], 15)


def test_iadd_same_shape() raises:
    """In-place elementwise add of two same-shape tensors."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 1
    a[0, 1] = 2
    a[1, 0] = 3
    a[1, 1] = 4
    b[0, 0] = 10
    b[0, 1] = 20
    b[1, 0] = 30
    b[1, 1] = 40

    a += b

    assert_equal(a[0, 0], 11)
    assert_equal(a[0, 1], 22)
    assert_equal(a[1, 0], 33)
    assert_equal(a[1, 1], 44)


def test_imul_same_shape() raises:
    """In-place elementwise multiply of two same-shape tensors."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 1
    a[0, 1] = 2
    a[1, 0] = 3
    a[1, 1] = 4
    b[0, 0] = 2
    b[0, 1] = 3
    b[1, 0] = 4
    b[1, 1] = 5

    a *= b

    assert_equal(a[0, 0], 2)
    assert_equal(a[0, 1], 6)
    assert_equal(a[1, 0], 12)
    assert_equal(a[1, 1], 20)


def test_isub_same_shape() raises:
    """In-place elementwise subtract of two same-shape tensors."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 10
    a[0, 1] = 20
    a[1, 0] = 30
    a[1, 1] = 40
    b[0, 0] = 1
    b[0, 1] = 2
    b[1, 0] = 3
    b[1, 1] = 4

    a -= b

    assert_equal(a[0, 0], 9)
    assert_equal(a[0, 1], 18)
    assert_equal(a[1, 0], 27)
    assert_equal(a[1, 1], 36)


def test_ifloordiv_same_shape() raises:
    """In-place elementwise floor-divide of two same-shape tensors."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 10
    a[0, 1] = 20
    a[1, 0] = 30
    a[1, 1] = 40
    b[0, 0] = 3
    b[0, 1] = 7
    b[1, 0] = 4
    b[1, 1] = 9

    a //= b

    assert_equal(a[0, 0], 3)
    assert_equal(a[0, 1], 2)
    assert_equal(a[1, 0], 7)
    assert_equal(a[1, 1], 4)


def test_itruediv_same_shape() raises:
    """In-place elementwise true-divide of two same-shape tensors."""
    var a_data = Array[Float32, 4](fill=0)
    var b_data = Array[Float32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 10.0
    a[0, 1] = 20.0
    a[1, 0] = 30.0
    a[1, 1] = 40.0
    b[0, 0] = 4.0
    b[0, 1] = 5.0
    b[1, 0] = 8.0
    b[1, 1] = 16.0

    a /= b

    assert_equal(a[0, 0], 2.5)
    assert_equal(a[0, 1], 4.0)
    assert_equal(a[1, 0], 3.75)
    assert_equal(a[1, 1], 2.5)


def test_min_same_shape() raises:
    """In-place elementwise minimum of two same-shape tensors."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 1
    a[0, 1] = 20
    a[1, 0] = 3
    a[1, 1] = 40
    b[0, 0] = 10
    b[0, 1] = 2
    b[1, 0] = 30
    b[1, 1] = 4

    a.min(b)

    assert_equal(a[0, 0], 1)
    assert_equal(a[0, 1], 2)
    assert_equal(a[1, 0], 3)
    assert_equal(a[1, 1], 4)


def test_max_same_shape() raises:
    """In-place elementwise maximum of two same-shape tensors."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())

    a[0, 0] = 1
    a[0, 1] = 20
    a[1, 0] = 3
    a[1, 1] = 40
    b[0, 0] = 10
    b[0, 1] = 2
    b[1, 0] = 30
    b[1, 1] = 4

    a.max(b)

    assert_equal(a[0, 0], 10)
    assert_equal(a[0, 1], 20)
    assert_equal(a[1, 0], 30)
    assert_equal(a[1, 1], 40)


def test_abs() raises:
    """In-place elementwise absolute value."""
    var data = Array[Int32, 4](fill=0)
    var a = TileTensor(data, row_major[2, 2]())

    a[0, 0] = -1
    a[0, 1] = 2
    a[1, 0] = -3
    a[1, 1] = 0

    a.abs()

    assert_equal(a[0, 0], 1)
    assert_equal(a[0, 1], 2)
    assert_equal(a[1, 0], 3)
    assert_equal(a[1, 1], 0)


def test_abs_float() raises:
    """In-place elementwise absolute value on a floating-point tensor."""
    var data = Array[Float32, 4](fill=0)
    var a = TileTensor(data, row_major[2, 2]())

    a[0, 0] = -1.5
    a[0, 1] = 2.5
    a[1, 0] = -0.0
    a[1, 1] = -4.25

    a.abs()

    assert_equal(a[0, 0], 1.5)
    assert_equal(a[0, 1], 2.5)
    assert_equal(a[1, 0], 0.0)
    assert_equal(a[1, 1], 4.25)


def test_recip() raises:
    """In-place elementwise reciprocal."""
    var data = Array[Float32, 4](fill=0)
    var a = TileTensor(data, row_major[2, 2]())

    a[0, 0] = 2.0
    a[0, 1] = 4.0
    a[1, 0] = -8.0
    a[1, 1] = 16.0

    a.recip()

    assert_equal(a[0, 0], 0.5)
    assert_equal(a[0, 1], 0.25)
    assert_equal(a[1, 0], -0.125)
    assert_equal(a[1, 1], 0.0625)


def test_exp() raises:
    """In-place elementwise exponential with the default scale of 1."""
    var data = Array[Float32, 4](fill=0)
    var a = TileTensor(data, row_major[2, 2]())

    a[0, 0] = 0.0
    a[0, 1] = 1.0
    a[1, 0] = -1.0
    a[1, 1] = 2.0

    a.exp()

    assert_almost_equal(a[0, 0], 1.0)
    assert_almost_equal(a[0, 1], exp(Float32(1.0)))
    assert_almost_equal(a[1, 0], exp(Float32(-1.0)))
    assert_almost_equal(a[1, 1], exp(Float32(2.0)))


def test_exp_scale() raises:
    """In-place elementwise `exp(scale * x)` with a non-unit scale."""
    var data = Array[Float32, 4](fill=0)
    var a = TileTensor(data, row_major[2, 2]())

    a[0, 0] = 0.0
    a[0, 1] = 1.0
    a[1, 0] = -1.0
    a[1, 1] = 0.5

    a.exp[scale=Float32(2.0)]()

    assert_almost_equal(a[0, 0], 1.0)
    assert_almost_equal(a[0, 1], exp(Float32(2.0)))
    assert_almost_equal(a[1, 0], exp(Float32(-2.0)))
    assert_almost_equal(a[1, 1], exp(Float32(1.0)))


def test_storage_add_out_of_place() raises:
    """Out-of-place `TensorOps.add` writes `lhs + rhs` into `dst`."""
    var a_data = Array[Int32, 4](fill=0)
    var b_data = Array[Int32, 4](fill=0)
    var out_data = Array[Int32, 4](fill=-1)
    var a = TileTensor(a_data, row_major[2, 2]())
    var b = TileTensor(b_data, row_major[2, 2]())
    var out = TileTensor(out_data, row_major[2, 2]())

    a[0, 0] = 1
    a[0, 1] = 2
    a[1, 0] = 3
    a[1, 1] = 4
    b[0, 0] = 10
    b[0, 1] = 20
    b[1, 0] = 30
    b[1, 1] = 40

    _storage_add(out, a, b)

    assert_equal(out[0, 0], 11)
    assert_equal(out[0, 1], 22)
    assert_equal(out[1, 0], 33)
    assert_equal(out[1, 1], 44)
    # Inputs are unchanged.
    assert_equal(a[0, 0], 1)
    assert_equal(b[1, 1], 40)


def test_storage_add_out_of_place_broadcast() raises:
    """Out-of-place `TensorOps.add` broadcasts a rank-1 rhs into `dst`."""
    var a_data = Array[Int32, 4](fill=0)
    var bias_data = Array[Int32, 2](fill=0)
    var out_data = Array[Int32, 4](fill=-1)
    var a = TileTensor(a_data, row_major[2, 2]())
    var bias = TileTensor(bias_data, row_major[2]())
    var out = TileTensor(out_data, row_major[2, 2]())

    a[0, 0] = 1
    a[0, 1] = 2
    a[1, 0] = 3
    a[1, 1] = 4
    bias[0] = 100
    bias[1] = 200

    _storage_add(out, a, bias)

    assert_equal(out[0, 0], 101)
    assert_equal(out[0, 1], 102)
    assert_equal(out[1, 0], 203)
    assert_equal(out[1, 1], 204)


def test_storage_exp_out_of_place() raises:
    """Out-of-place `TensorOps.exp` writes `exp(scale * src)` into `dst`."""
    var src_data = Array[Float32, 4](fill=0)
    var out_data = Array[Float32, 4](fill=-1)
    var src = TileTensor(src_data, row_major[2, 2]())
    var out = TileTensor(out_data, row_major[2, 2]())

    src[0, 0] = 0.0
    src[0, 1] = 1.0
    src[1, 0] = -1.0
    src[1, 1] = 0.5

    _storage_exp[scale=Float32(2.0)](out, src)

    assert_almost_equal(out[0, 0], 1.0)
    assert_almost_equal(out[0, 1], exp(Float32(2.0)))
    assert_almost_equal(out[1, 0], exp(Float32(-2.0)))
    assert_almost_equal(out[1, 1], exp(Float32(1.0)))
    assert_equal(src[0, 1], 1.0)


def test_storage_abs_out_of_place() raises:
    """Out-of-place `TensorOps.abs` writes `|src|` into `dst`."""
    var src_data = Array[Int32, 4](fill=0)
    var out_data = Array[Int32, 4](fill=-1)
    var src = TileTensor(src_data, row_major[2, 2]())
    var out = TileTensor(out_data, row_major[2, 2]())

    src[0, 0] = -1
    src[0, 1] = 2
    src[1, 0] = -3
    src[1, 1] = 0

    _storage_abs(out, src)

    assert_equal(out[0, 0], 1)
    assert_equal(out[0, 1], 2)
    assert_equal(out[1, 0], 3)
    assert_equal(out[1, 1], 0)
    assert_equal(src[0, 0], -1)


def test_tuple_getter() raises:
    var data = Array[Float32, 4](fill=0)
    var a = TileTensor(data, row_major[2, 2]())
    comptime for i in range(data.length):
        data[i] = Float32(i)

    assert_equal(a[(1, 1)], 3)
    var a10 = (1, 0)
    assert_equal(a[a10], 2)
