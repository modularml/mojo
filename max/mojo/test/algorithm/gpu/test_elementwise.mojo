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

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from std.testing import assert_equal, TestSuite

from std.utils import IndexList
from std.utils.coord import Coord, coord_to_index_list
from std.utils.index import Index


def _linear_index[
    rank: Int
](coords: IndexList[rank], shape: IndexList[rank]) -> Int:
    """Convert multi-dimensional coordinates to linear index (row-major)."""
    var linear_idx = 0
    var stride = 1

    comptime for i in reversed(range(rank)):
        linear_idx += coords[i] * stride
        stride *= shape[i]
    return linear_idx


def _strided_index[
    rank: Int
](coords: IndexList[rank], strides: IndexList[rank]) -> Int:
    """Convert multi-dimensional coordinates to linear index using explicit strides.
    """
    var linear_idx = 0

    comptime for i in range(rank):
        linear_idx += coords[i] * strides[i]
    return linear_idx


def run_elementwise[dtype: DType](ctx: DeviceContext) raises:
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    var in_host_stack = Array[Scalar[dtype], 16](fill=0)
    var in_host = Span(in_host_stack)
    var out_host_stack = Array[Scalar[dtype], 16](fill=0)
    var out_host = Span(out_host_stack)

    var flattened_length = len(in_host)
    for i in range(2):
        for j in range(8):
            in_host[_linear_index(Index(i, j), Index(2, 8))] = Scalar[dtype](
                i + j
            )

    var in_device = ctx.enqueue_create_buffer[dtype](flattened_length)
    var out_device = ctx.enqueue_create_buffer[dtype](flattened_length)

    in_device.enqueue_copy_from(in_host)

    var shape = IndexList[2](2, 8)
    var in_buffer = Span(
        unsafe_ptr=in_device.unsafe_ptr(), length=flattened_length
    )
    var out_buffer = Span(
        unsafe_ptr=out_device.unsafe_ptr(), length=flattened_length
    )

    @always_inline
    @__copy_capture(in_buffer, out_buffer, shape)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = rebind[IndexList[2]](coord_to_index_list(idx0))
        var linear_idx = _linear_index(idx, shape)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            linear_idx,
            in_buffer.unsafe_ptr().unsafe_load[width=simd_width](linear_idx)
            + 42,
        )

    elementwise[func, pack_size, target="gpu"](
        (2, 8),
        ctx,
    )

    out_device.enqueue_copy_to(out_host)

    ctx.synchronize()

    var expected_vals: List[Scalar[dtype]] = [
        42.0,
        43.0,
        44.0,
        45.0,
        46.0,
        47.0,
        48.0,
        49.0,
        43.0,
        44.0,
        45.0,
        46.0,
        47.0,
        48.0,
        49.0,
        50.0,
    ]
    for i in range(2):
        for j in range(8):
            assert_equal(
                out_host[_linear_index(Index(i, j), Index(2, 8))],
                expected_vals[i * 8 + j],
            )

    _ = in_device
    _ = out_device


def run_elementwise_uneven_simd[dtype: DType](ctx: DeviceContext) raises:
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()
    var in_host_stack = Array[Scalar[dtype], 9](fill=0)
    var in_host = Span(in_host_stack)
    var out_host_stack = Array[Scalar[dtype], 9](fill=0)
    var out_host = Span(out_host_stack)

    var flattened_length = len(in_host)
    for i in range(3):
        for j in range(3):
            in_host[_linear_index(Index(i, j), Index(3, 3))] = Scalar[dtype](
                i + j
            )

    var in_device = ctx.enqueue_create_buffer[dtype](flattened_length)
    var out_device = ctx.enqueue_create_buffer[dtype](flattened_length)

    in_device.enqueue_copy_from(in_host)

    var shape = IndexList[2](3, 3)
    var in_buffer = Span(
        unsafe_ptr=in_device.unsafe_ptr(), length=flattened_length
    )
    var out_buffer = Span(
        unsafe_ptr=out_device.unsafe_ptr(), length=flattened_length
    )

    @always_inline
    @__copy_capture(in_buffer, out_buffer, shape)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = rebind[IndexList[2]](coord_to_index_list(idx0))
        var linear_idx = _linear_index(idx, shape)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            linear_idx,
            in_buffer.unsafe_ptr().unsafe_load[width=simd_width](linear_idx)
            + 42,
        )

    elementwise[func, pack_size, target="gpu"](
        (3, 3),
        ctx,
    )
    out_device.enqueue_copy_to(out_host)
    ctx.synchronize()

    var expected_vals: List[Scalar[dtype]] = [
        42.0,
        43.0,
        44.0,
        43.0,
        44.0,
        45.0,
        44.0,
        45.0,
        46.0,
    ]
    for i in range(3):
        for j in range(3):
            assert_equal(
                out_host[_linear_index(Index(i, j), Index(3, 3))],
                expected_vals[i * 3 + j],
            )

    _ = in_device
    _ = out_device


def run_elementwise_exact_boundary_uses_simd[
    dtype: DType
](ctx: DeviceContext) raises:
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()
    comptime if pack_size == 1:
        return

    comptime flattened_length = pack_size * (pack_size + 1)
    var out_host_stack = Array[Scalar[dtype], flattened_length](fill=0)
    var out_host = Span(out_host_stack)

    var out_device = ctx.enqueue_create_buffer[dtype](flattened_length)
    var out_buffer = Span(
        unsafe_ptr=out_device.unsafe_ptr(), length=flattened_length
    )
    var shape = IndexList[2](pack_size, pack_size + 1)

    @always_inline
    @__copy_capture(out_buffer, shape)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = rebind[IndexList[2]](coord_to_index_list(idx0))
        var linear_idx = _linear_index(idx, shape)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            linear_idx, SIMD[dtype, simd_width](simd_width)
        )

    elementwise[func, pack_size, target="gpu"](Coord(shape), ctx)
    out_device.enqueue_copy_to(out_host.unsafe_ptr())
    ctx.synchronize()

    var last_row = pack_size - 1
    assert_equal(
        out_host[_linear_index(Index(last_row, 0), shape)], Scalar[dtype](1)
    )
    for j in range(1, pack_size + 1):
        assert_equal(
            out_host[_linear_index(Index(last_row, j), shape)],
            Scalar[dtype](pack_size),
        )

    _ = out_device


def run_elementwise_transpose_copy[dtype: DType](ctx: DeviceContext) raises:
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()
    var in_host_stack = Array[Scalar[dtype], 2 * 4 * 5](fill=0)
    var in_host = Span(in_host_stack)
    var out_host_stack = Array[Scalar[dtype], 2 * 4 * 5](fill=0)
    var out_host = Span(out_host_stack)

    var flattened_length = len(in_host)
    for i in range(2):
        for j in range(4):
            for k in range(5):
                in_host[_linear_index(Index(i, j, k), Index(2, 4, 5))] = Scalar[
                    dtype
                ](i * 4 * 5 + j * 5 + k)

    var in_device = ctx.enqueue_create_buffer[dtype](flattened_length)
    var out_device = ctx.enqueue_create_buffer[dtype](flattened_length)

    in_device.enqueue_copy_from(in_host)

    # Transposed view: logical shape (4, 2, 5) with strides (5, 20, 1)
    var in_strides = IndexList[3](5, 20, 1)
    var out_shape = IndexList[3](4, 2, 5)
    var in_buffer = Span(
        unsafe_ptr=in_device.unsafe_ptr(), length=flattened_length
    )
    var out_buffer = Span(
        unsafe_ptr=out_device.unsafe_ptr(), length=flattened_length
    )

    @always_inline
    @__copy_capture(in_buffer, out_buffer, in_strides, out_shape)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = rebind[IndexList[3]](coord_to_index_list(idx0))

        # We need to perform unaligned loads because the non-uniform strides
        # being used for in_buffer.
        var in_idx = _strided_index(idx, in_strides)
        var out_idx = _linear_index(idx, out_shape)
        out_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            out_idx,
            in_buffer.unsafe_ptr().unsafe_load[width=simd_width, alignment=1](
                in_idx
            ),
        )

    elementwise[func, 1, target="gpu"](Coord(out_shape), ctx)

    out_device.enqueue_copy_to(out_host)
    ctx.synchronize()

    var expected_vals: List[Scalar[dtype]] = [
        0.0,
        1.0,
        2.0,
        3.0,
        4.0,
        20.0,
        21.0,
        22.0,
        23.0,
        24.0,
        5.0,
        6.0,
        7.0,
        8.0,
        9.0,
        25.0,
        26.0,
        27.0,
        28.0,
        29.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        30.0,
        31.0,
        32.0,
        33.0,
        34.0,
        15.0,
        16.0,
        17.0,
        18.0,
        19.0,
        35.0,
        36.0,
        37.0,
        38.0,
        39.0,
    ]
    for i in range(4):
        for j in range(2):
            for k in range(5):
                assert_equal(
                    out_host[_linear_index(Index(i, j, k), out_shape)],
                    expected_vals[i * 2 * 5 + j * 5 + k],
                )

    _ = in_device
    _ = out_device


def _test_elementwise_zero_dimension_3d(ctx: DeviceContext) raises:
    """Test elementwise operations with zero dimension in 3D tensor."""
    comptime dtype = DType.float32
    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    var input_device_ptr = ctx.enqueue_create_buffer[dtype](1)
    var output_device_ptr = ctx.enqueue_create_buffer[dtype](1)

    var input_buffer = Span(unsafe_ptr=input_device_ptr.unsafe_ptr(), length=1)
    var output_buffer = Span(
        unsafe_ptr=output_device_ptr.unsafe_ptr(), length=1
    )

    # Test with zero in first dimension
    var shape = IndexList[3](0, 4, 4)

    @always_inline
    @__copy_capture(input_buffer, output_buffer, shape)
    @__parameter
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord):
        var idx = rebind[IndexList[3]](coord_to_index_list(idx0))
        var linear_idx = _linear_index(idx, shape)
        output_buffer.unsafe_ptr().unsafe_store[width=simd_width](
            linear_idx,
            input_buffer.unsafe_ptr().unsafe_load[width=simd_width](linear_idx),
        )

    elementwise[func, pack_size, target="gpu"](
        (0, 4, 4),
        ctx,
    )
    ctx.synchronize()

    # Test with zero in second dimension
    elementwise[func, pack_size, target="gpu"](
        (2, 0, 4),
        ctx,
    )
    ctx.synchronize()

    # Test with zero in third dimension
    elementwise[func, pack_size, target="gpu"](
        (2, 4, 0),
        ctx,
    )
    ctx.synchronize()

    _ = input_device_ptr
    _ = output_device_ptr


def test_elementwise_gpu() raises:
    with DeviceContext() as ctx:
        run_elementwise[.float32](ctx)
        run_elementwise_uneven_simd[.float32](ctx)
        run_elementwise_exact_boundary_uses_simd[.float32](ctx)
        run_elementwise_transpose_copy[.float32](ctx)
        run_elementwise[.bfloat16](ctx)
        run_elementwise_uneven_simd[.bfloat16](ctx)
        run_elementwise_exact_boundary_uses_simd[.bfloat16](ctx)
        run_elementwise_transpose_copy[.bfloat16](ctx)
        run_elementwise[.float16](ctx)
        run_elementwise_uneven_simd[.float16](ctx)
        run_elementwise_exact_boundary_uses_simd[.float16](ctx)
        run_elementwise_transpose_copy[.float16](ctx)
        _test_elementwise_zero_dimension_3d(ctx)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
