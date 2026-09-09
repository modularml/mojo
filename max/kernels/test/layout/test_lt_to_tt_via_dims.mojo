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
"""Tests for lt_to_tt (LayoutTensor to TileTensor conversion)."""

from layout import (
    IntTuple,
    LTToTTLayout,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
)
from std.utils.index import Index
from std.testing import assert_equal


def test_lt_to_tt_2d_static() raises:
    """Test lt_to_tt with a fully static 2D layout."""
    comptime static_2d = Layout.row_major(4, 8)
    var arr = Array[Float32, 32](fill=1.0)
    var lt = LayoutTensor[.float32, static_2d](arr.unsafe_ptr())
    var tt = lt_to_tt(lt)
    assert_equal(tt.rank, 2)
    assert_equal(Int(tt.dim[0]()), 4)
    assert_equal(Int(tt.dim[1]()), 8)


def test_lt_to_tt_2d_dynamic() raises:
    """Test lt_to_tt with a partially dynamic 2D layout."""
    comptime shape = IntTuple(UNKNOWN_VALUE, 8)
    comptime dynamic_2d = Layout.row_major(shape)
    var arr = Array[Float32, 24](fill=2.0)
    var lt = LayoutTensor[.float32, dynamic_2d](
        arr.unsafe_ptr(),
        RuntimeLayout[dynamic_2d].row_major(Index(3, 8)),
    )
    var tt = lt_to_tt(lt)
    assert_equal(tt.rank, 2)
    assert_equal(Int(tt.dim[0]()), 3)
    assert_equal(Int(tt.dim[1]()), 8)


def test_lt_to_tt_1d_dynamic() raises:
    """Test lt_to_tt with a fully dynamic 1D layout."""
    comptime dynamic_1d = Layout(UNKNOWN_VALUE)
    var arr = Array[UInt32, 5](fill=42)
    var lt = LayoutTensor[.uint32, dynamic_1d](
        arr.unsafe_ptr(),
        RuntimeLayout[dynamic_1d](Index(5), Index(1)),
    )
    var tt = lt_to_tt(lt)
    assert_equal(tt.rank, 1)
    assert_equal(Int(tt.dim[0]()), 5)


def test_lt_to_tt_4d_mixed() raises:
    """Test lt_to_tt with a 4D layout matching the kv_cache pattern."""
    # Shape: (UNKNOWN, UNKNOWN, 8, 128), strides row-major from known dims
    comptime shape_4d = IntTuple(UNKNOWN_VALUE, UNKNOWN_VALUE, 8, 128)
    comptime layout_4d = Layout.row_major(shape_4d)
    # Verify the strides we expect from row_major with unknown leading dims:
    #   stride[3] = 1, stride[2] = 128, stride[1] = 1024, stride[0] = UNKNOWN
    comptime assert layout_4d.stride[3].value() == 1
    comptime assert layout_4d.stride[2].value() == 128
    comptime assert layout_4d.stride[1].value() == 1024

    # Verify the TileTensor layout types
    comptime TTLayout4D = LTToTTLayout[layout_4d]
    comptime assert TTLayout4D.rank == 4
    # dim 0 and 1 are dynamic (UNKNOWN_VALUE in shape)
    comptime assert not TTLayout4D._shape_types[0].is_static_value
    comptime assert not TTLayout4D._shape_types[1].is_static_value
    # dim 2 and 3 are static
    comptime assert TTLayout4D._shape_types[2].is_static_value
    comptime assert TTLayout4D._shape_types[3].is_static_value
    comptime assert TTLayout4D._shape_types[2].static_value == 8
    comptime assert TTLayout4D._shape_types[3].static_value == 128
    # stride 0 is dynamic, strides 1-3 are static
    comptime assert not TTLayout4D._stride_types[0].is_static_value
    comptime assert TTLayout4D._stride_types[1].is_static_value
    comptime assert TTLayout4D._stride_types[1].static_value == 1024
    comptime assert TTLayout4D._stride_types[2].static_value == 128
    comptime assert TTLayout4D._stride_types[3].static_value == 1

    # Construct a 4D LayoutTensor and convert
    comptime num_blocks = 2
    comptime max_seq_len = 16
    comptime num_elements = num_blocks * max_seq_len * 8 * 128
    var arr = Array[Float32, num_elements](fill=3.0)
    var lt = LayoutTensor[.float32, layout_4d](
        arr.unsafe_ptr(),
        RuntimeLayout[layout_4d](
            Index(num_blocks, max_seq_len, 8, 128),
            Index(max_seq_len * 8 * 128, 8 * 128, 128, 1),
        ),
    )
    var tt = lt_to_tt(lt)
    assert_equal(tt.rank, 4)
    assert_equal(Int(tt.dim[0]()), num_blocks)
    assert_equal(Int(tt.dim[1]()), max_seq_len)
    assert_equal(Int(tt.dim[2]()), 8)
    assert_equal(Int(tt.dim[3]()), 128)


def main() raises:
    test_lt_to_tt_2d_static()
    test_lt_to_tt_2d_dynamic()
    test_lt_to_tt_1d_dynamic()
    test_lt_to_tt_4d_mixed()
