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
"""Tests for ManagedTensorSlice - a tensor view type for custom graph operations.
"""

from extensibility import get_row_major_tensor_spec_static
from extensibility import ManagedTensorSlice, IOSpec
from extensibility.managed_tensor_slice import (
    StaticTensorSpec,
    _IndexListToTileLayout,
)
from layout import coord_to_index_list
from std.memory import AddressSpace
from std.sys import align_of
from std.testing import assert_equal, TestSuite

from std.utils import IndexList


def test_basic_construction() raises:
    """Test basic ManagedTensorSlice construction from pointer and shape."""
    var storage = Array[Float32, 3 * 4](fill={})
    # Shape-only constructor computes row-major strides automatically
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    assert_equal(tensor.rank, 2)
    assert_equal(tensor.size(), 12)


def test_shape_and_strides() raises:
    """Test shape() and strides() methods."""
    var storage = Array[Float32, 2 * 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[
        DType.float32, 3, 2, 3, 4
    ]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[3](2, 3, 4)
    )

    var shape = tensor.shape()
    assert_equal(shape[0], 2)
    assert_equal(shape[1], 3)
    assert_equal(shape[2], 4)

    var strides = tensor.strides()
    assert_equal(strides[0], 12)
    assert_equal(strides[1], 4)
    assert_equal(strides[2], 1)


def test_dim_size() raises:
    """Test dim_size methods (compile-time and runtime)."""
    var storage = Array[Float32, 5 * 7](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 5, 7]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](5, 7)
    )

    # Test compile-time dim_size
    assert_equal(tensor.dim_size[0](), 5)
    assert_equal(tensor.dim_size[1](), 7)

    # Test runtime dim_size
    assert_equal(tensor.dim_size(0), 5)
    assert_equal(tensor.dim_size(1), 7)


def test_getitem_setitem() raises:
    """Test __getitem__ and __setitem__ operations."""
    var storage = Array[Float32, 2 * 3](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 2, 3]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](2, 3))

    # Set values
    tensor[0, 0] = 1.0
    tensor[0, 1] = 2.0
    tensor[0, 2] = 3.0
    tensor[1, 0] = 4.0
    tensor[1, 1] = 5.0
    tensor[1, 2] = 6.0

    # Get values using variadic indices
    assert_equal(tensor[0, 0], 1.0)
    assert_equal(tensor[0, 1], 2.0)
    assert_equal(tensor[0, 2], 3.0)
    assert_equal(tensor[1, 0], 4.0)
    assert_equal(tensor[1, 1], 5.0)
    assert_equal(tensor[1, 2], 6.0)

    # Get values using IndexList
    assert_equal(tensor[IndexList[2](0, 2)], 3.0)
    assert_equal(tensor[IndexList[2](1, 0)], 4.0)


def test_simd_load_store() raises:
    """Test SIMD load and store operations."""
    var storage = Array[Float32, 8](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 1, 8]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[1](8))

    # Store a SIMD vector
    var vec = SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0)
    tensor.store(IndexList[1](0), vec)

    var vec2 = SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0)
    tensor.store(IndexList[1](4), vec2)

    # Load and verify
    var loaded = tensor.load[4](IndexList[1](0))
    assert_equal(loaded, SIMD[.float32, 4](1.0, 2.0, 3.0, 4.0))

    var loaded2 = tensor.load[4](IndexList[1](4))
    assert_equal(loaded2, SIMD[.float32, 4](5.0, 6.0, 7.0, 8.0))


def test_to_layout_tensor() raises:
    """Test to_layout_tensor() conversion."""
    var storage = Array[Float32, 3 * 4](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](3, 4))

    # Convert to LayoutTensor
    var layout_tensor = tensor.to_layout_tensor()

    # Verify the layout tensor has the same data
    assert_equal(layout_tensor[0, 0], 0.0)
    assert_equal(layout_tensor[1, 1], 5.0)
    assert_equal(layout_tensor[2, 3], 11.0)

    # Verify dimensions
    assert_equal(Int(layout_tensor.runtime_layout.shape[0]), 3)
    assert_equal(Int(layout_tensor.runtime_layout.shape[1]), 4)

    # TODO(GEX-4147): ManagedTensorSlice needs to carry the Array's origin
    # `tensor` holds an untracked pointer into `storage`; keep `storage` alive
    # until the last read through it.
    _ = storage^


def test_stride_length() raises:
    """Test stride_length methods."""
    var storage = Array[Float32, 3 * 5](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 5]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 5)
    )

    # Test compile-time stride_length
    assert_equal(tensor.stride_length[0](), 5)
    assert_equal(tensor.stride_length[1](), 1)

    # Test runtime stride_length
    assert_equal(tensor.stride_length(0), 5)
    assert_equal(tensor.stride_length(1), 1)


def test_simd_load_store_2d() raises:
    """Test SIMD load and store operations on 2D tensor."""
    var storage = Array[Float32, 4 * 8](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 4, 8]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](4, 8))

    # Store vectors in each row
    for i in range(4):
        var vec = SIMD[.float32, 4](
            Float32(i * 10),
            Float32(i * 10 + 1),
            Float32(i * 10 + 2),
            Float32(i * 10 + 3),
        )
        tensor.store(IndexList[2](i, 0), vec)

        var vec2 = SIMD[.float32, 4](
            Float32(i * 10 + 4),
            Float32(i * 10 + 5),
            Float32(i * 10 + 6),
            Float32(i * 10 + 7),
        )
        tensor.store(IndexList[2](i, 4), vec2)

    # Load and verify
    var loaded_row0 = tensor.load[4](IndexList[2](0, 0))
    assert_equal(loaded_row0, SIMD[.float32, 4](0.0, 1.0, 2.0, 3.0))

    var loaded_row2 = tensor.load[4](IndexList[2](2, 4))
    assert_equal(loaded_row2, SIMD[.float32, 4](24.0, 25.0, 26.0, 27.0))

    var loaded_row3 = tensor.load[4](IndexList[2](3, 0))
    assert_equal(loaded_row3, SIMD[.float32, 4](30.0, 31.0, 32.0, 33.0))


def test_to_tile_tensor() raises:
    """Test to_tile_tensor() conversion."""
    var storage = Array[Float32, 3 * 4](
        fill_with=lambda (i: Int) -> Float32: Float32(i)
    )
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[
        mut=True, io_spec=IOSpec.Unknown, static_spec=spec
    ](storage.unsafe_ptr(), IndexList[2](3, 4))

    # Convert to TileTensor
    var tile_tensor = tensor.to_tile_tensor[.int64]()

    # Verify the layout tensor has the same data
    comptime assert tile_tensor.flat_rank == 2
    assert_equal(tile_tensor[0, 0], 0.0)
    assert_equal(tile_tensor[1, 1], 5.0)
    assert_equal(tile_tensor[2, 3], 11.0)

    # Verify dimensions
    assert_equal(tile_tensor.layout.shape[0]().value(), 3)
    assert_equal(tile_tensor.layout.shape[1]().value(), 4)

    # TODO(GEX-4147): ManagedTensorSlice needs to carry the Array's origin
    # `tensor` holds an untracked pointer into `storage`; keep `storage` alive
    # until the last read through it.
    _ = storage^


def test_shape_coord_static() raises:
    """Test shape_coord() preserves fully-static shape information."""
    var storage = Array[Float32, 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    var shape = tensor.shape_coord()

    # Every dimension is statically known, so this is a compile-time fact.
    comptime assert shape.all_dims_known
    comptime assert shape.element_types[0].static_value == 3
    comptime assert shape.element_types[1].static_value == 4

    # The runtime values still round-trip correctly.
    var index_list = coord_to_index_list(shape)
    assert_equal(index_list[0], 3)
    assert_equal(index_list[1], 4)


def test_shape_coord_mixed() raises:
    """Test shape_coord() encodes static dims while filling dynamic ones."""
    var storage = Array[Float32, 2 * 4](fill={})
    # dim 0 is dynamic (-1), dim 1 is static (4); strides are row-major.
    comptime mixed_layout = _IndexListToTileLayout[
        IndexList[2](-1, 4), IndexList[2](4, 1)
    ]
    comptime mixed_spec = StaticTensorSpec[
        DType.float32, 2, static_layout=mixed_layout
    ](align_of[DType.float32](), AddressSpace.GENERIC)
    var tensor = ManagedTensorSlice[
        io_spec=IOSpec.Unknown, static_spec=mixed_spec
    ](storage.unsafe_ptr(), IndexList[2](2, 4))

    var shape = tensor.shape_coord()

    # The static/dynamic structure is preserved in the Coord's type.
    comptime assert not shape.all_dims_known
    comptime assert not shape.element_types[0].is_static_value
    comptime assert shape.element_types[1].is_static_value
    comptime assert shape.element_types[1].static_value == 4

    # The dynamic dimension is filled from the runtime shape.
    var index_list = coord_to_index_list(shape)
    assert_equal(index_list[0], 2)
    assert_equal(index_list[1], 4)


def test_strides_coord_static() raises:
    """Test strides_coord() preserves fully-static stride information."""
    var storage = Array[Float32, 3 * 4](fill={})
    comptime spec = get_row_major_tensor_spec_static[.float32, 2, 3, 4]()
    var tensor = ManagedTensorSlice[io_spec=IOSpec.Unknown, static_spec=spec](
        storage.unsafe_ptr(), IndexList[2](3, 4)
    )

    var strides = tensor.strides_coord()

    # Row-major strides are statically known, so this is a compile-time fact.
    comptime assert strides.all_dims_known
    comptime assert strides.element_types[0].static_value == 4
    comptime assert strides.element_types[1].static_value == 1

    # The runtime values still round-trip correctly.
    var index_list = coord_to_index_list(strides)
    assert_equal(index_list[0], 4)
    assert_equal(index_list[1], 1)


def test_strides_coord_mixed() raises:
    """Test strides_coord() encodes static strides while filling dynamic ones.
    """
    var storage = Array[Float32, 2 * 4](fill={})
    # Shape is static (2, 4); stride 0 is dynamic (-1) and stride 1 is static.
    comptime mixed_layout = _IndexListToTileLayout[
        IndexList[2](2, 4), IndexList[2](-1, 1)
    ]
    comptime mixed_spec = StaticTensorSpec[
        DType.float32, 2, static_layout=mixed_layout
    ](align_of[DType.float32](), AddressSpace.GENERIC)
    var tensor = ManagedTensorSlice[
        io_spec=IOSpec.Unknown, static_spec=mixed_spec
    ](storage.unsafe_ptr(), IndexList[2](2, 4), IndexList[2](4, 1))

    var strides = tensor.strides_coord()

    # The static/dynamic structure is preserved in the Coord's type.
    comptime assert not strides.all_dims_known
    comptime assert not strides.element_types[0].is_static_value
    comptime assert strides.element_types[1].is_static_value
    comptime assert strides.element_types[1].static_value == 1

    # The dynamic stride is filled from the runtime strides.
    var index_list = coord_to_index_list(strides)
    assert_equal(index_list[0], 4)
    assert_equal(index_list[1], 1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
