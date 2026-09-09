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

from std.sys import simd_width_of

from layout._fillers import arange
from layout import TileTensor, row_major
from nn.pool import PoolMethod, avg_pool, max_pool, pool_shape_impl
from std.testing import assert_almost_equal, assert_equal


def pool[
    count_boundary: Bool = False
](
    pool_method: PoolMethod,
    output_tensor: TileTensor[mut=True, .float32, ...],
) raises:
    var in_stack = Array[Float32, 2 * 5 * 7 * 2](fill={})
    var input_tensor = TileTensor(in_stack, row_major[2, 5, 7, 2]())
    arange(input_tensor)

    var paddings_stack = Array[Int32, 4](fill=0)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack: Array[Int32, _] = [3, 2]
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack: Array[Int32, _] = [2, 3]
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    comptime simd_width = simd_width_of[DType.float32]()

    if pool_method == PoolMethod.MAX:
        max_pool[int_type=.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            output_tensor,
        )
    else:
        avg_pool[int_type=.int32, count_boundary=count_boundary](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            output_tensor,
        )


# CHECK-LABEL: test_max_pool_2d
def test_max_pool_2d() raises:
    print("== test_max_pool_2d")

    # output should have form
    # ([[[[ 30.,  31.],
    #    [ 36.,  37.]],
    #   [[ 58.,  59.],
    #    [ 64.,  65.]]],
    #  [[[ 100.,  101.],
    #    [ 106., 107.]],
    #   [[128., 129.],
    #    [134., 135.]]]])

    var out_stack = Array[Float32, 2 * 2 * 2 * 2](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[2, 2, 2, 2]())
    pool(PoolMethod.MAX, output_tensor)

    assert_equal(output_tensor[0, 0, 0, 0], 30)
    assert_equal(output_tensor[0, 0, 0, 1], 31)
    assert_equal(output_tensor[0, 0, 1, 0], 36)
    assert_equal(output_tensor[0, 0, 1, 1], 37)
    assert_equal(output_tensor[0, 1, 0, 0], 58)
    assert_equal(output_tensor[0, 1, 0, 1], 59)
    assert_equal(output_tensor[0, 1, 1, 0], 64)
    assert_equal(output_tensor[0, 1, 1, 1], 65)
    assert_equal(output_tensor[1, 0, 0, 0], 100)
    assert_equal(output_tensor[1, 0, 0, 1], 101)
    assert_equal(output_tensor[1, 0, 1, 0], 106)
    assert_equal(output_tensor[1, 0, 1, 1], 107)
    assert_equal(output_tensor[1, 1, 0, 0], 128)
    assert_equal(output_tensor[1, 1, 0, 1], 129)
    assert_equal(output_tensor[1, 1, 1, 0], 134)
    assert_equal(output_tensor[1, 1, 1, 1], 135)


# CHECK-LABEL: test_avg_pool_2d
def test_avg_pool_2d() raises:
    print("== test_avg_pool_2d")

    # output should have form
    # ([[[[  15.0,  16.0],
    #    [ 21.0,  22.0]],
    #   [[ 43.0,  44.0],
    #    [ 49.0,  50.0]]],
    #  [[[ 85.0,  86.0],
    #    [ 91.0,  92.0]],
    #   [[113.0, 114.0],
    #    [119.0, 120.0]]]])

    var out_stack = Array[Float32, 2 * 2 * 2 * 2](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[2, 2, 2, 2]())
    pool(PoolMethod.AVG, output_tensor)

    assert_equal(output_tensor[0, 0, 0, 0], 15.0)
    assert_equal(output_tensor[0, 0, 0, 1], 16)
    assert_equal(output_tensor[0, 0, 1, 0], 21)
    assert_equal(output_tensor[0, 0, 1, 1], 22)
    assert_equal(output_tensor[0, 1, 0, 0], 43)
    assert_equal(output_tensor[0, 1, 0, 1], 44)
    assert_equal(output_tensor[0, 1, 1, 0], 49)
    assert_equal(output_tensor[0, 1, 1, 1], 50)
    assert_equal(output_tensor[1, 0, 0, 0], 85)
    assert_equal(output_tensor[1, 0, 0, 1], 86)
    assert_equal(output_tensor[1, 0, 1, 0], 91)
    assert_equal(output_tensor[1, 0, 1, 1], 92)
    assert_equal(output_tensor[1, 1, 0, 0], 113)
    assert_equal(output_tensor[1, 1, 0, 1], 114)
    assert_equal(output_tensor[1, 1, 1, 0], 119)
    assert_equal(output_tensor[1, 1, 1, 1], 120)


def test_avg_pool_2d_with_padding[
    count_boundary: Bool = False
](output_tensor: TileTensor[mut=True, .float32, ...]) raises:
    var in_stack = Array[Float32, 1 * 7 * 7 * 1](fill={})
    var input_tensor = TileTensor(in_stack, row_major[1, 7, 7, 1]())
    arange(input_tensor)

    var paddings_stack = Array[Int32, 4](fill=1)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=3)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=1)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    comptime simd_width = simd_width_of[DType.float32]()

    avg_pool[int_type=.int32, count_boundary=count_boundary](
        input_tensor,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        output_tensor,
    )


# CHECK-LABEL: test_avg_pool_2d_count_boundary: True
def test_avg_pool_2d_with_padding_true() raises:
    print("== test_avg_pool_2d_count_boundary: True")
    var out_stack = Array[Float32, 1 * 7 * 7 * 1](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[1, 7, 7, 1]())
    test_avg_pool_2d_with_padding[True](output_tensor)

    assert_almost_equal(output_tensor[0, 0, 0, 0], 1.7778, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 1, 0], 3.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 2, 0], 3.6667, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 3, 0], 4.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 4, 0], 5.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 5, 0], 5.6667, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 6, 0], 4.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 0, 0], 5.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 1, 0], 8.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 2, 0], 9.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 3, 0], 10.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 4, 0], 11.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 5, 0], 12.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 6, 0], 8.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 0, 0], 9.6667, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 1, 0], 15.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 2, 0], 16.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 3, 0], 17.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 4, 0], 18.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 5, 0], 19.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 6, 0], 13.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 0, 0], 14.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 1, 0], 22.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 2, 0], 23.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 3, 0], 24.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 4, 0], 25.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 5, 0], 26.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 6, 0], 17.6667, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 0, 0], 19.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 1, 0], 29.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 2, 0], 30.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 3, 0], 31.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 4, 0], 32.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 5, 0], 33.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 6, 0], 22.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 0, 0], 23.6667, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 1, 0], 36.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 2, 0], 37.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 3, 0], 38.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 4, 0], 39.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 5, 0], 40.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 6, 0], 27.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 0, 0], 17.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 1, 0], 26.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 2, 0], 27.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 3, 0], 27.6667, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 4, 0], 28.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 5, 0], 29.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 6, 0], 19.5556, atol=1e-4)


# CHECK-LABEL: test_avg_pool_2d_count_boundary: False
def test_avg_pool_2d_with_padding_false() raises:
    print("== test_avg_pool_2d_count_boundary: False")
    var out_stack = Array[Float32, 1 * 7 * 7 * 1](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[1, 7, 7, 1]())
    test_avg_pool_2d_with_padding[False](output_tensor)

    # Replace filecheck lines with assert_almost_equal
    assert_almost_equal(output_tensor[0, 0, 0, 0], 4.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 1, 0], 4.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 2, 0], 5.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 3, 0], 6.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 4, 0], 7.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 5, 0], 8.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 6, 0], 9.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 0, 0], 7.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 1, 0], 8.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 2, 0], 9.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 3, 0], 10.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 4, 0], 11.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 5, 0], 12.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 6, 0], 12.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 0, 0], 14.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 1, 0], 15.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 2, 0], 16.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 3, 0], 17.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 4, 0], 18.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 5, 0], 19.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 2, 6, 0], 19.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 0, 0], 21.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 1, 0], 22.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 2, 0], 23.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 3, 0], 24.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 4, 0], 25.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 5, 0], 26.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 3, 6, 0], 26.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 0, 0], 28.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 1, 0], 29.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 2, 0], 30.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 3, 0], 31.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 4, 0], 32.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 5, 0], 33.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 4, 6, 0], 33.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 0, 0], 35.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 1, 0], 36.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 2, 0], 37.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 3, 0], 38.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 4, 0], 39.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 5, 0], 40.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 5, 6, 0], 40.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 0, 0], 39.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 1, 0], 39.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 2, 0], 40.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 3, 0], 41.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 4, 0], 42.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 5, 0], 43.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 6, 6, 0], 44.0000, atol=1e-4)


def pool_ceil_test[
    count_boundary: Bool = False, ceil_mode: Bool = True
](
    pool_method: PoolMethod,
    output_tensor: TileTensor[mut=True, .float32, ...],
) raises:
    var in_stack = Array[Float32, 1 * 4 * 4 * 1](fill={})
    var input_tensor = TileTensor(in_stack, row_major[1, 4, 4, 1]())
    arange(input_tensor)

    var paddings_stack = Array[Int32, 4](fill=0)
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=3)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=2)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=1)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    comptime simd_width = simd_width_of[DType.float32]()

    var output_shape = pool_shape_impl[
        .float32,
        .int32,
        .int32,
        .int32,
        .int32,
        ceil_mode,
    ](
        input_tensor,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
    )

    if pool_method == PoolMethod.MAX:
        max_pool[int_type=.int32](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            output_tensor,
            ceil_mode,
        )
    else:
        avg_pool[int_type=.int32, count_boundary=count_boundary](
            input_tensor,
            filter_tensor,
            stride_tensor,
            dilation_tensor,
            paddings_tensor,
            output_tensor,
            ceil_mode,
        )


# CHECK-LABEL: test_max_pool_2d_ceil
def test_maxpool_2d_ceil() raises:
    print("== test_max_pool_2d_ceil")
    var out_stack = Array[Float32, 1 * 2 * 2 * 1](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[1, 2, 2, 1]())
    pool_ceil_test(PoolMethod.MAX, output_tensor)
    assert_almost_equal(output_tensor[0, 0, 0, 0], 10.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 1, 0], 11.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 0, 0], 14.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 1, 0], 15.0000, atol=1e-4)


# CHECK-LABEL: test_average_pool_2d_ceil_exclude_bound
def test_average_pool_2d_ceil_exclude_bound() raises:
    print("== test_average_pool_2d_ceil_exclude_bound")
    var out_stack = Array[Float32, 1 * 2 * 2 * 1](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[1, 2, 2, 1]())
    pool_ceil_test(PoolMethod.AVG, output_tensor)
    assert_almost_equal(output_tensor[0, 0, 0, 0], 5.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 1, 0], 6.5000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 0, 0], 11.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 1, 0], 12.5000, atol=1e-4)


# CHECK-LABEL: test_average_pool_2d_ceil_include_bound
def test_average_pool_2d_ceil_include_bound() raises:
    print("== test_average_pool_2d_ceil_include_bound")
    var out_stack = Array[Float32, 1 * 2 * 2 * 1](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[1, 2, 2, 1]())
    pool_ceil_test[True, True](PoolMethod.AVG, output_tensor)
    assert_almost_equal(output_tensor[0, 0, 0, 0], 5.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 1, 0], 4.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 0, 0], 7.3333, atol=1e-4)
    assert_almost_equal(output_tensor[0, 1, 1, 0], 5.5556, atol=1e-4)


# CHECK-LABEL: test_max_pool_pad_dilation_2d
def test_max_pool_pad_dilation_2d() raises:
    print("== test_max_pool_pad_dilation_2d")

    var in_stack = Array[Float32, 1 * 4 * 4 * 1](fill={})
    var input_tensor = TileTensor(in_stack, row_major[1, 4, 4, 1]())
    arange(input_tensor)

    var out_stack = Array[Float32, 1 * 1 * 3 * 1](fill=0)
    var output_tensor = TileTensor(out_stack, row_major[1, 1, 3, 1]())

    var paddings_stack: Array[Int32, _] = [0, 0, 2, 0]
    var paddings_tensor = TileTensor(paddings_stack, row_major[4]())

    var filter_stack = Array[Int32, 2](fill=2)
    var filter_tensor = TileTensor(filter_stack, row_major[2]())

    var stride_stack = Array[Int32, 2](fill=1)
    var stride_tensor = TileTensor(stride_stack, row_major[2]())

    var dilation_stack = Array[Int32, 2](fill=3)
    var dilation_tensor = TileTensor(dilation_stack, row_major[2]())

    comptime simd_width = simd_width_of[DType.float32]()

    max_pool[int_type=.int32](
        input_tensor,
        filter_tensor,
        stride_tensor,
        dilation_tensor,
        paddings_tensor,
        output_tensor,
    )

    assert_almost_equal(output_tensor[0, 0, 0, 0], 13.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 1, 0], 14.0000, atol=1e-4)
    assert_almost_equal(output_tensor[0, 0, 2, 0], 15.0000, atol=1e-4)


def main() raises:
    test_max_pool_2d()
    test_avg_pool_2d()
    test_avg_pool_2d_with_padding_true()
    test_avg_pool_2d_with_padding_false()
    test_maxpool_2d_ceil()
    test_average_pool_2d_ceil_exclude_bound()
    test_average_pool_2d_ceil_include_bound()
    test_max_pool_pad_dilation_2d()
