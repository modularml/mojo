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

from std.itertools import product
from layout import (
    CoordLike,
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    row_major,
)
from layout.layout import blocked_product
from layout._fillers import arange
from std.testing import assert_equal

from std.utils.index import IndexList


def test_runtime_and_compile_time_dim_and_stride[
    MType: CoordLike, KType: CoordLike, //
](m: MType, k: KType) raises:
    var shape = Coord(k, m)
    var tt = TileTensor(
        MutPointer[Float32, MutAnyOrigin].unsafe_dangling(), row_major(shape)
    )
    var tensor = tt.to_layout_tensor()

    var K = Int(k.value())
    var M = Int(m.value())

    assert_equal(tensor.dim(0), K)
    assert_equal(tensor.dim(1), M)
    assert_equal(tensor.stride(0), M)
    assert_equal(tensor.stride(1), 1)

    assert_equal(tensor.dim[0](), K)
    assert_equal(tensor.dim[1](), M)
    assert_equal(tensor.stride[0](), M)
    assert_equal(tensor.stride[1](), 1)


def test_nested_layout_shape() raises:
    """Test that shape[idx]() works correctly for nested layouts."""
    # Test case 1: blocked_product creates nested layout
    comptime tiler_layout = Layout.row_major(2, 4)
    comptime base_layout = Layout.row_major(32, 32)
    comptime smem_layout = blocked_product(base_layout, tiler_layout)

    var tensor = LayoutTensor[.float32, smem_layout, MutAnyOrigin](None)

    # Shape should be (64, 128) because:
    # - First dimension: 32 * 2 = 64
    # - Second dimension: 32 * 4 = 128
    comptime shape0 = tensor.shape[0]()
    comptime shape1 = tensor.shape[1]()

    assert_equal(shape0, 64, "Shape[0] should be 64 for nested layout")
    assert_equal(shape1, 128, "Shape[1] should be 128 for nested layout")

    # Total size should be 64 * 128 = 8192
    var total_size = tensor.size()
    assert_equal(total_size, 8192, "Total size should be 8192")

    # Test case 2: Ensure non-nested layouts still work (regression test)
    comptime simple_layout = Layout.row_major(16, 32)
    comptime simple_shape0 = LayoutTensor[
        .float32, simple_layout, MutAnyOrigin
    ].shape[0]()
    comptime simple_shape1 = LayoutTensor[
        .float32, simple_layout, MutAnyOrigin
    ].shape[1]()

    assert_equal(simple_shape0, 16, "Non-nested shape[0] should still work")
    assert_equal(simple_shape1, 32, "Non-nested shape[1] should still work")


def _create_tensor_2x2[
    dtype: DType
]() -> LayoutTensor[dtype, Layout.row_major(2, 2), MutAnyOrigin]:
    """Helper to create a 2x2 row-major tensor on the stack."""
    return LayoutTensor[
        dtype,
        Layout.row_major(2, 2),
        MutAnyOrigin,
        address_space=.GENERIC,
    ].stack_allocation()


def _copy_transpose[
    dtype: DType
](
    src: LayoutTensor[dtype, Layout.row_major(2, 2), MutAnyOrigin],
    mut dst: LayoutTensor[dtype, Layout.row_major(2, 2), MutAnyOrigin],
):
    """Copy tensor src into dst with transposed indices."""
    for i, j in product(range(2), range(2)):
        dst[j, i] = src[i, j]


def test_transpose_arithmetic() raises:
    """Test all arithmetic operations with transposed tensors.

    This test verifies that arithmetic operations (+, -, *, /) work correctly
    when one operand is a transposed view of a tensor. This ensures the
    transpose operation properly maintains stride information for arithmetic.
    """
    # Test with arange values: a = [[0, 1], [2, 3]]
    var a = _create_tensor_2x2[.float32]()
    arange(a)

    var b = _create_tensor_2x2[.float32]()
    _copy_transpose(a, b)

    # After transpose, a.transpose() = [[0, 2], [1, 3]] = b
    # Test subtraction: should be all zeros
    var sub_result = a.transpose() - b
    assert_equal(sub_result[0, 0], 0.0)
    assert_equal(sub_result[0, 1], 0.0)
    assert_equal(sub_result[1, 0], 0.0)
    assert_equal(sub_result[1, 1], 0.0)

    # Test addition: a.transpose() + b = 2 * [[0, 2], [1, 3]]
    var add_result = a.transpose() + b
    assert_equal(add_result[0, 0], 0.0)
    assert_equal(add_result[0, 1], 4.0)
    assert_equal(add_result[1, 0], 2.0)
    assert_equal(add_result[1, 1], 6.0)

    # Test multiplication: element-wise product
    var mul_result = a.transpose() * b
    assert_equal(mul_result[0, 0], 0.0)  # 0 * 0
    assert_equal(mul_result[0, 1], 4.0)  # 2 * 2
    assert_equal(mul_result[1, 0], 1.0)  # 1 * 1
    assert_equal(mul_result[1, 1], 9.0)  # 3 * 3

    # Test division with non-zero values: c = [[2, 4], [6, 8]]
    var c = _create_tensor_2x2[.float32]()
    for i, j in product(range(2), range(2)):
        c[i, j] = Float32((i * 2 + j + 1) * 2)

    var d = _create_tensor_2x2[.float32]()
    _copy_transpose(c, d)

    # c.transpose() / d should be all ones
    var div_result = c.transpose() / d
    assert_equal(div_result[0, 0], 1.0)
    assert_equal(div_result[0, 1], 1.0)
    assert_equal(div_result[1, 0], 1.0)
    assert_equal(div_result[1, 1], 1.0)


def test_different_layouts_arithmetic() raises:
    """Test arithmetic between row-major and column-major tensors.

    This verifies that tensors with different memory layouts can still
    perform arithmetic operations correctly based on their logical indices.
    """
    var a = _create_tensor_2x2[.float32]()
    arange(a)

    # Create column-major tensor with same logical values
    var b = LayoutTensor[
        .float32,
        Layout.col_major(2, 2),
        MutAnyOrigin,
        address_space=.GENERIC,
    ].stack_allocation()
    for i, j in product(range(2), range(2)):
        b[i, j] = a[i, j]

    # Subtraction should yield zeros despite different memory layouts
    var result = a - b
    assert_equal(result[0, 0], 0.0)
    assert_equal(result[0, 1], 0.0)
    assert_equal(result[1, 0], 0.0)
    assert_equal(result[1, 1], 0.0)


def test_flatten() raises:
    var stack = Array[Int8, 16](fill=0)
    var tensor = LayoutTensor[.int8, Layout.row_major(4, 4)](stack).flatten()
    assert_equal(tensor.size(), 16)
    assert_equal(tensor.rank, 1)
    assert_equal(tensor.stride[0](), 1)


def test_get_shape() raises:
    var stack = Array[Int8, 16](fill=0)
    var tensor = LayoutTensor[.int8, Layout.row_major(4, 4)](stack)
    assert_equal(4, tensor.get_shape()[0])
    assert_equal(4, tensor.get_shape()[1])


def test_reshape() raises:
    var stack = Array[Int8, 16](fill=0)
    var tensor = LayoutTensor[.int8, Layout(16)](stack).reshape[
        Layout.row_major[2]()
    ](RuntimeLayout[Layout.row_major[2]()].row_major(IndexList[2](4, 4)))
    assert_equal(tensor.size(), 16)
    assert_equal(tensor.get_shape()[0], 4)
    assert_equal(tensor.get_shape()[1], 4)


def test_aligned_load() raises:
    """Tests aligned_load with both index types."""
    # Use a 4x7 tensor so we can load 4 elements starting at columns 0,1,2,3
    # without going out of bounds (column 3 + width 4 = 7)
    var storage = Array[Float32, 4 * 7](fill=0.0)
    var tensor = LayoutTensor[
        .float32,
        Layout([4, 7]),
    ](storage)

    tensor.store[4](0, 0, 1.0)  # Store 4 elements starting at column 0
    tensor.store[4](0, 1, 2.0)  # Store 4 elements starting at column 1
    tensor.store[4](0, 2, 3.0)  # Store 4 elements starting at column 2
    tensor.store[4](0, 3, 4.0)  # Store 4 elements starting at column 3

    # Load 4 elements starting at column 0
    var a0 = tensor.aligned_load[4](0, 0)
    var b0 = tensor.aligned_load[4](IndexList[2](0, 0))
    assert_equal(a0, b0)

    # Load 4 elements starting at column 1
    var a1 = tensor.aligned_load[4](0, 1)
    var b1 = tensor.aligned_load[4](IndexList[2](0, 1))
    assert_equal(a1, b1)

    # Load 4 elements starting at column 2
    var a2 = tensor.aligned_load[4](0, 2)
    var b2 = tensor.aligned_load[4](IndexList[2](0, 2))
    assert_equal(a2, b2)

    # Load 4 elements starting at column 3
    var a3 = tensor.aligned_load[4](0, 3)
    var b3 = tensor.aligned_load[4](IndexList[2](0, 3))
    assert_equal(a3, b3)


def main() raises:
    test_runtime_and_compile_time_dim_and_stride(Idx[120], Idx[512])
    test_nested_layout_shape()
    test_transpose_arithmetic()
    test_different_layouts_arithmetic()
    test_aligned_load()
    test_flatten()
    test_get_shape()
    test_reshape()
