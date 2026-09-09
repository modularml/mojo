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
"""Test LayoutTensor._offset(IndexList) per-dimension stride selection.

This test verifies that LayoutTensor._offset(IndexList) correctly uses:
- Compile-time strides for dimensions where the stride is known (not UNKNOWN_VALUE)
- Runtime strides for dimensions where the stride is UNKNOWN_VALUE

This per-dimension approach enables constant folding for known strides while
correctly handling view tensors where some strides depend on runtime values.

The key scenarios tested:
1. Fully static layouts - all strides are known, should produce correct offsets
2. Partially static layouts - some strides unknown (like BHSD tensors with dynamic B)
3. View tensors - stride[0] is UNKNOWN because actual stride depends on parent tensor
"""

from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from std.testing import TestSuite, assert_equal
from std.utils import Index, IndexList


def test_ptr_at_offset_static_2d() raises:
    """Test that ptr_at_offset(IndexList) produces correct pointers for 2D layouts.
    """
    comptime layout = Layout.row_major(10, 20)
    comptime total_elems = 10 * 20

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = LayoutTensor[.int32, layout](data.unsafe_ptr())

    # Test pointer at (2, 3) -> offset = 2 * 20 + 3 = 43
    var ptr = tensor.ptr_at_offset(Index(2, 3))
    assert_equal(ptr[], 43)

    # Test pointer at (5, 15) -> offset = 5 * 20 + 15 = 115
    ptr = tensor.ptr_at_offset(Index(5, 15))
    assert_equal(ptr[], 115)


def test_ptr_at_offset_static_3d() raises:
    """Test that ptr_at_offset(IndexList) produces correct pointers for 3D layouts.
    """
    comptime layout = Layout.row_major(5, 10, 20)
    comptime total_elems = 5 * 10 * 20

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = LayoutTensor[.int32, layout](data.unsafe_ptr())

    # Test pointer at (1, 2, 3) -> offset = 1*200 + 2*20 + 3 = 243
    var ptr = tensor.ptr_at_offset(Index(1, 2, 3))
    assert_equal(ptr[], 243)


def test_ptr_at_offset_static_4d() raises:
    """Test that ptr_at_offset(IndexList) produces correct pointers for 4D layouts.
    """
    comptime layout = Layout.row_major(2, 4, 8, 16)
    comptime total_elems = 2 * 4 * 8 * 16

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = LayoutTensor[.int32, layout](data.unsafe_ptr())

    # Test pointer at (1, 2, 3, 4) -> offset = 1*512 + 2*128 + 3*16 + 4 = 820
    var ptr = tensor.ptr_at_offset(Index(1, 2, 3, 4))
    assert_equal(ptr[], 820)


def test_ptr_at_offset_col_major() raises:
    """Test that ptr_at_offset(IndexList) produces correct pointers for col-major layouts.
    """
    comptime layout = Layout.col_major(10, 20)
    comptime total_elems = 10 * 20

    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    var tensor = LayoutTensor[.int32, layout](data.unsafe_ptr())

    # Col-major: stride = (1, 10), offset = 2*1 + 3*10 = 32
    var ptr = tensor.ptr_at_offset(Index(2, 3))
    assert_equal(ptr[], 32)


def test_ptr_at_offset_with_unknown_stride() raises:
    """Test that ptr_at_offset uses runtime stride for UNKNOWN dimensions.

    This tests the per-dimension approach: when a stride is UNKNOWN_VALUE,
    the runtime stride should be used. When a stride is known, the compile-time
    stride should be used.
    """
    # Create a 3D layout with unknown first stride, but known inner strides
    comptime d1 = 8
    comptime d2 = 16
    comptime shape = IntTuple(UNKNOWN_VALUE, d1, d2)
    # stride[0] = UNKNOWN, stride[1] = d2 = 16, stride[2] = 1
    comptime strides = IntTuple(UNKNOWN_VALUE, d2, 1)
    comptime layout = Layout(shape, strides)

    # Allocate test data
    comptime total_elems = 4 * d1 * d2  # 4 * 8 * 16 = 512
    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    # Create tensor with runtime stride[0] = 128
    var runtime_stride_0 = d1 * d2  # = 128
    var runtime_shape = IndexList[3](4, d1, d2)
    var runtime_strides = IndexList[3](runtime_stride_0, d2, 1)

    var tensor = LayoutTensor[.int32, layout](
        data.unsafe_ptr(),
        RuntimeLayout[layout](runtime_shape, runtime_strides),
    )

    # Test pointer at (2, 3, 5) -> offset = 2*128 + 3*16 + 5 = 309
    # stride[0]=128 from runtime, stride[1]=16 and stride[2]=1 from compile-time
    var ptr = tensor.ptr_at_offset(Index(2, 3, 5))
    var expected_offset = 2 * runtime_stride_0 + 3 * d2 + 5  # = 309
    assert_equal(ptr[], Int32(expected_offset))


def test_ptr_at_offset_view_tensor() raises:
    """Test that view tensors use correct runtime strides via ptr_at_offset.

    This simulates PagedKVCache-like scenarios where a 4D view's stride[0]
    depends on dimensions in the parent tensor that aren't in the view's shape.
    """
    # Child layout with UNKNOWN stride[0]
    comptime child_shape = IntTuple(UNKNOWN_VALUE, 4)
    comptime child_strides = IntTuple(UNKNOWN_VALUE, 1)
    comptime child_layout = Layout(child_shape, child_strides)

    comptime total_elems = 24
    var data = Array[Int32, total_elems](
        fill_with=lambda (i: Int) -> Int32: Int32(i)
    )

    # Create view with runtime stride[0] = 8 (different from shape[1] = 4)
    var runtime_stride_0 = 8
    var runtime_shape = IndexList[2](3, 4)
    var runtime_strides = IndexList[2](runtime_stride_0, 1)

    var child = LayoutTensor[.int32, child_layout](
        data.unsafe_ptr(),
        RuntimeLayout[child_layout](runtime_shape, runtime_strides),
    )

    # Test pointer at (1, 2) with runtime stride -> offset = 1*8 + 2*1 = 10
    # If wrong compile-time stride were used (4 instead of 8),
    # we'd read from offset 6 instead of 10
    var ptr = child.ptr_at_offset(Index(1, 2))
    var expected_offset = 1 * runtime_stride_0 + 2  # = 10
    assert_equal(ptr[], Int32(expected_offset))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
