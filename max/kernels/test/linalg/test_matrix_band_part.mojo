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

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Layout,
    LayoutTensor,
    TileTensor,
    row_major,
)
from layout.int_tuple import to_index_list
from linalg.matrix_band_part import matrix_band_part as _matrix_band_part

from std.testing import assert_equal

from std.utils import IndexList


def matrix_band_part[
    output_layout: Layout,
    dtype: DType,
](
    input: LayoutTensor[dtype, output_layout, ImmutAnyOrigin],
    output: LayoutTensor[dtype, output_layout, MutAnyOrigin],
    num_lower: Int,
    num_upper: Int,
    exclude: Bool,
) raises:
    comptime int_type = DType.int
    comptime cond_type = DType.bool

    var num_lower_buf = LayoutTensor[
        int_type, Layout.row_major(1), MutAnyOrigin
    ].stack_allocation()
    var num_upper_buf = LayoutTensor[
        int_type, Layout.row_major(1), MutAnyOrigin
    ].stack_allocation()
    var exclude_buf = LayoutTensor[
        cond_type, Layout.row_major(1), MutAnyOrigin
    ].stack_allocation()

    num_lower_buf[0] = Int(num_lower)
    num_upper_buf[0] = Int(num_upper)
    exclude_buf[0] = exclude
    comptime rank = input.rank
    var input_shape: IndexList[rank] = to_index_list[rank](input.layout.shape)

    def input_fn[
        width: Int,
        _rank: Int,
    ](coords: IndexList[_rank]) {var input} -> SIMD[dtype, width]:
        return input.load[width=width](rebind[IndexList[rank]](coords))

    # Create TileTensors for scalar parameters.
    var num_lower_shape = Coord(Int64(1))
    var num_lower_tt = TileTensor(num_lower_buf.ptr, row_major(num_lower_shape))
    var num_upper_shape = Coord(Int64(1))
    var num_upper_tt = TileTensor(num_upper_buf.ptr, row_major(num_upper_shape))
    var exclude_shape = Coord(Int64(1))
    var exclude_tt = TileTensor(exclude_buf.ptr, row_major(exclude_shape))

    # Create TileTensor for output.
    comptime m = output_layout.shape[0].value()
    comptime n = output_layout.shape[1].value()
    var output_shape = Coord(Int64(m), Int64(n))
    var output_tt = TileTensor(output.ptr, row_major(output_shape))

    _matrix_band_part[
        dtype,
        int_type,
        cond_type,
        rank,
        simd_width=1,
    ](
        input_fn,
        input_shape,
        num_lower_tt,
        num_upper_tt,
        exclude_tt,
        output_tt,
        DeviceContext(api="cpu"),
    )


def test_matrix_band_part() raises:
    comptime layout = Layout.row_major(3, 3)
    comptime dtype = DType.float32

    var input = LayoutTensor[dtype, layout, MutAnyOrigin].stack_allocation()
    var output = LayoutTensor[dtype, layout, MutAnyOrigin].stack_allocation()

    input[0, 0] = 1
    input[0, 1] = 2
    input[0, 2] = 3
    input[1, 0] = 4
    input[1, 1] = 5
    input[1, 2] = 6
    input[2, 0] = 7
    input[2, 1] = 8
    input[2, 2] = 9

    matrix_band_part(
        input.as_imm(),
        output,
        num_lower=0,
        num_upper=-1,
        exclude=False,
    )

    assert_equal(output[0, 0], 1)
    assert_equal(output[0, 1], 2)
    assert_equal(output[0, 2], 3)
    assert_equal(output[1, 0], 0)
    assert_equal(output[1, 1], 5)
    assert_equal(output[1, 2], 6)
    assert_equal(output[2, 0], 0)
    assert_equal(output[2, 1], 0)
    assert_equal(output[2, 2], 9)

    matrix_band_part(
        input.as_imm(),
        output,
        num_lower=0,
        num_upper=-1,
        exclude=True,
    )

    assert_equal(output[0, 0], 0)
    assert_equal(output[0, 1], 0)
    assert_equal(output[0, 2], 0)
    assert_equal(output[1, 0], 4)
    assert_equal(output[1, 1], 0)
    assert_equal(output[1, 2], 0)
    assert_equal(output[2, 0], 7)
    assert_equal(output[2, 1], 8)
    assert_equal(output[2, 2], 0)


def main() raises:
    test_matrix_band_part()
