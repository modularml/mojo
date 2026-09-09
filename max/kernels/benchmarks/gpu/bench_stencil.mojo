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

from std.sys import get_defined_dtype, get_defined_int, get_defined_string

from max.algorithm.functional import stencil, stencil_gpu
from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from layout import (
    Coord,
    Idx,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from max.gpu.host import DeviceContext
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.numerics import min_or_neg_inf


def assert_allclose[
    dtype: DType
](
    h_output_ref: TileTensor[dtype, ...],
    h_output_gpu: TileTensor[dtype, ...],
) raises:
    for i in range(h_output_ref.num_elements()):
        assert_almost_equal(h_output_ref.raw_load(i), h_output_gpu.raw_load(i))


def bench_stencil_avg_pool[
    dtype: DType,
    batch_size: Int,
    input_height: Int,
    input_width: Int,
    pool_window_h: Int,
    pool_window_w: Int,
    num_channels: Int,
](ctx: DeviceContext, mut m: Bench) raises:
    comptime rank = 4
    comptime dilation = 1
    comptime stencil_rank = 2
    comptime simd_width = 1

    comptime output_height = input_height - pool_window_h + 1
    comptime output_width = input_width - pool_window_w + 1
    comptime input_size = 1 * input_height * input_width * num_channels
    comptime output_size = 1 * output_height * output_width * num_channels
    comptime dynamic_input_shape = IndexList[4](
        1, input_height, input_width, num_channels
    )
    comptime dynamic_output_shape = IndexList[4](
        1, output_height, output_width, num_channels
    )

    # Create host buffers
    var h_input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var h_input = TileTensor(
        h_input_ptr,
        row_major(
            Coord(
                Idx[1],
                Idx[input_height],
                Idx[input_width],
                Idx[num_channels],
            )
        ),
    )
    var h_output_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var h_output = TileTensor(
        h_output_ptr,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[num_channels],
            )
        ),
    )
    var h_output_ref_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var h_output_ref = TileTensor(
        h_output_ref_ptr,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[num_channels],
            )
        ),
    )

    # Initialize input data
    for i in range(h_input.num_elements()):
        h_input.raw_store(i, Scalar[dtype](i + 1))

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](input_size)
    var d_input = TileTensor(
        d_input_buf,
        row_major(
            Coord(
                Idx[1],
                Idx[input_height],
                Idx[input_width],
                Idx[num_channels],
            )
        ),
    )
    var d_output_buf = ctx.enqueue_create_buffer[dtype](output_size)
    var d_output = TileTensor(
        d_output_buf,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[num_channels],
            )
        ),
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.ptr)
    ctx.enqueue_copy(d_output_buf, h_output.ptr)

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        var res = val / Scalar[dtype](pool_window_h * pool_window_w)
        d_output.store_linear(point, res)

    var out_shape = rebind[IndexList[rank]](
        coord_to_index_list(d_output.layout.shape_coord())
    )
    var in_shape = rebind[IndexList[rank]](
        coord_to_index_list(d_input.layout.shape_coord())
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {imm}:
        comptime stencil_axis = IndexList[stencil_rank](1, 2)
        stencil_gpu[
            rank,
            stencil_rank,
            stencil_axis,
            simd_width,
            dtype,
        ](
            ctx,
            out_shape,
            in_shape,
            map_fn_gpu,
            dilation_fn_gpu,
            load_fn_gpu,
            avg_pool_compute_init_gpu,
            avg_pool_compute_gpu,
            avg_pool_compute_finalize_gpu,
        )

    @always_inline
    def bench_gpu(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_cpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_cpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        var res = val / Scalar[dtype](pool_window_h * pool_window_w)
        h_output_ref.store_linear(point, res)

    @always_inline
    def bench_cpu(mut b: Bencher) {imm}:
        @always_inline
        def kernel_launch() {imm}:
            comptime stencil_axis = IndexList[stencil_rank](1, 2)
            stencil[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
            ](
                dynamic_output_shape,
                dynamic_input_shape,
                map_fn_cpu,
                dilation_fn_cpu,
                load_fn_cpu,
                avg_pool_compute_init_cpu,
                avg_pool_compute_cpu,
                avg_pool_compute_finalize_cpu,
            )

        b.iter(kernel_launch)

    # Calculate FLOPs for throughput measurement
    def compute_flops() -> Int:
        return (
            input_height * input_width * pool_window_h * pool_window_w * 2
        )  # One add, one divide per window element

    # Ensure correctness
    assert_allclose(h_output_ref, h_output)

    # Run benchmarks
    var bench_name = String(
        "stencil_avg_pool_",
        batch_size,
        "x",
        input_height,
        "x",
        input_width,
        "x",
        num_channels,
    )
    var flops = ThroughputMeasure(BenchMetric.flops, compute_flops())
    m.bench_function(
        bench_gpu,
        BenchId(bench_name + "_gpu"),
        [flops],
    )

    m.bench_function(
        bench_cpu,
        BenchId(bench_name + "_cpu"),
        [flops],
    )

    # Ensure correctness
    ctx.enqueue_copy(h_output.ptr, d_output_buf)
    ctx.synchronize()
    assert_allclose(h_output_ref, h_output)

    _ = d_input_buf^
    _ = d_output_buf^
    _ = h_output_ref_ptr^
    _ = h_output_ptr^
    _ = h_input_ptr^


def bench_stencil_max_pool[
    dtype: DType,
    batch_size: Int,
    input_height: Int,
    input_width: Int,
    pool_window_h: Int,
    pool_window_w: Int,
    num_channels: Int,
](ctx: DeviceContext, mut m: Bench) raises:
    comptime rank = 4
    comptime dilation = 1
    comptime stencil_rank = 2
    comptime simd_width = 1

    comptime output_height = input_height - pool_window_h + 1
    comptime output_width = input_width - pool_window_w + 1
    comptime input_size = 1 * input_height * input_width * num_channels
    comptime output_size = 1 * output_height * output_width * num_channels
    comptime dynamic_input_shape = IndexList[4](
        1, input_height, input_width, num_channels
    )
    comptime dynamic_output_shape = IndexList[4](
        1, output_height, output_width, num_channels
    )

    # Create host buffers
    var h_input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var h_input = TileTensor(
        h_input_ptr,
        row_major(
            Coord(
                Idx[1],
                Idx[input_height],
                Idx[input_width],
                Idx[num_channels],
            )
        ),
    )
    var h_output_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var h_output = TileTensor(
        h_output_ptr,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[num_channels],
            )
        ),
    )
    var h_output_ref_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var h_output_ref = TileTensor(
        h_output_ref_ptr,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[num_channels],
            )
        ),
    )

    # Initialize input data
    for i in range(h_input.num_elements()):
        h_input.raw_store(i, Scalar[dtype](i + 1))

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](input_size)
    var d_input = TileTensor(
        d_input_buf,
        row_major(
            Coord(
                Idx[1],
                Idx[input_height],
                Idx[input_width],
                Idx[num_channels],
            )
        ),
    )
    var d_output_buf = ctx.enqueue_create_buffer[dtype](output_size)
    var d_output = TileTensor(
        d_output_buf,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[num_channels],
            )
        ),
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.ptr)
    ctx.enqueue_copy(d_output_buf, h_output.ptr)

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def max_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    def max_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        d_output.store_linear(point, val)

    var out_shape = rebind[IndexList[rank]](
        coord_to_index_list(d_output.layout.shape_coord())
    )
    var in_shape = rebind[IndexList[rank]](
        coord_to_index_list(d_input.layout.shape_coord())
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {imm}:
        comptime stencil_axis = IndexList[stencil_rank](1, 2)
        stencil_gpu[
            rank,
            stencil_rank,
            stencil_axis,
            simd_width,
            dtype,
        ](
            ctx,
            out_shape,
            in_shape,
            map_fn_gpu,
            dilation_fn_gpu,
            load_fn_gpu,
            max_pool_compute_init_gpu,
            max_pool_compute_gpu,
            max_pool_compute_finalize_gpu,
        )

    @always_inline
    def bench_gpu(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](point[0], point[1])
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h, point[1] + pool_window_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_cpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def max_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return min_or_neg_inf[dtype]()

    def max_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return max(val, result)

    @always_inline
    def max_pool_compute_finalize_cpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        h_output_ref.store_linear(point, val)

    @always_inline
    def bench_cpu(mut b: Bencher) {imm}:
        @always_inline
        def kernel_launch() {imm}:
            comptime stencil_axis = IndexList[stencil_rank](1, 2)
            stencil[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
            ](
                dynamic_output_shape,
                dynamic_input_shape,
                map_fn_cpu,
                dilation_fn_cpu,
                load_fn_cpu,
                max_pool_compute_init_cpu,
                max_pool_compute_cpu,
                max_pool_compute_finalize_cpu,
            )

        b.iter(kernel_launch)

    # Calculate FLOPs for throughput measurement
    def compute_flops() -> Int:
        return (
            input_height * input_width * pool_window_h * pool_window_w
        )  # One comparison per window element

    # Run benchmarks
    var bench_name = String(
        "stencil_max_pool_",
        batch_size,
        "x",
        input_height,
        "x",
        input_width,
        "x",
        num_channels,
    )
    var flops = ThroughputMeasure(BenchMetric.flops, compute_flops())
    m.bench_function(
        bench_gpu,
        BenchId(bench_name + "_gpu"),
        [flops],
    )

    m.bench_function(
        bench_cpu,
        BenchId(bench_name + "_cpu"),
        [flops],
    )

    # Ensure correctness
    ctx.enqueue_copy(h_output.ptr, d_output_buf)
    ctx.synchronize()
    assert_allclose(h_output_ref, h_output)

    _ = d_input_buf^
    _ = d_output_buf^
    _ = h_output_ref_ptr^
    _ = h_output_ptr^
    _ = h_input_ptr^


def bench_stencil_avg_pool_padded[
    dtype: DType,
    batch_size: Int,
    input_height: Int,
    input_width: Int,
    pool_window_h: Int,
    pool_window_w: Int,
    pad_h: Int,
    pad_w: Int,
](ctx: DeviceContext, mut m: Bench) raises:
    comptime rank = 4
    comptime stencil_rank = 2
    comptime simd_width = 1
    comptime dilation = 1

    comptime output_height = input_height - pool_window_h + pad_h * 2 + 1
    comptime output_width = input_width - pool_window_w + pad_w * 2 + 1
    comptime input_size = 1 * input_height * input_width * 1
    comptime output_size = 1 * output_height * output_width * 1
    var dynamic_input_shape = IndexList[4](1, input_height, input_width, 1)
    comptime dynamic_output_shape = IndexList[4](
        1, output_height, output_width, 1
    )

    # Create host buffers
    var h_input_ptr = List(length=input_size, fill=Scalar[dtype](0))
    var h_input = TileTensor(
        h_input_ptr,
        row_major(Coord(Idx[1], Idx[input_height], Idx[input_width], Idx[1])),
    )
    var h_output_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var h_output = TileTensor(
        h_output_ptr,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[1],
            )
        ),
    )
    var h_output_ref_ptr = List(length=output_size, fill=Scalar[dtype](0))
    var h_output_ref = TileTensor(
        h_output_ref_ptr,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[1],
            )
        ),
    )

    # Initialize input data
    for i in range(h_input.num_elements()):
        h_input.raw_store(i, Scalar[dtype](i + 1))

    # Create device buffers
    var d_input_buf = ctx.enqueue_create_buffer[dtype](input_size)
    var d_input = TileTensor(
        d_input_buf,
        row_major(
            Coord(
                Idx[1],
                Idx[input_height],
                Idx[input_width],
                Idx[1],
            )
        ),
    )
    var d_output_buf = ctx.enqueue_create_buffer[dtype](output_size)
    var d_output = TileTensor(
        d_output_buf,
        row_major(
            Coord(
                Idx[1],
                Int64(output_height),
                Int64(output_width),
                Idx[1],
            )
        ),
    )

    # Copy to device
    ctx.enqueue_copy(d_input_buf, h_input.ptr)
    ctx.enqueue_copy(d_output_buf, h_output.ptr)

    def map_fn_gpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    def dilation_fn_gpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_gpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var d_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            d_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_gpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_gpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_gpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var d_output,
    }:
        var res = val / Scalar[dtype](pool_window_h * pool_window_w)
        d_output.store_linear(point, res)

    var out_shape = rebind[IndexList[rank]](
        coord_to_index_list(d_output.layout.shape_coord())
    )
    var in_shape = rebind[IndexList[rank]](
        coord_to_index_list(d_input.layout.shape_coord())
    )

    @always_inline
    def kernel_launch(
        ctx: DeviceContext,
    ) raises {imm}:
        comptime stencil_axis = IndexList[stencil_rank](1, 2)
        stencil_gpu[
            rank,
            stencil_rank,
            stencil_axis,
            simd_width,
            dtype,
        ](
            ctx,
            out_shape,
            in_shape,
            map_fn_gpu,
            dilation_fn_gpu,
            load_fn_gpu,
            avg_pool_compute_init_gpu,
            avg_pool_compute_gpu,
            avg_pool_compute_finalize_gpu,
        )

    @always_inline
    def bench_gpu(mut b: Bencher) raises {imm}:
        bencher_iter_custom(b, kernel_launch, ctx)

    def map_fn_cpu(
        point: IndexList[stencil_rank, ...],
    ) {} -> Tuple[IndexList[stencil_rank], IndexList[stencil_rank]]:
        var lower_bound = IndexList[stencil_rank](
            point[0] - pad_h, point[1] - pad_w
        )
        var upper_bound = IndexList[stencil_rank](
            point[0] + pool_window_h - pad_h, point[1] + pool_window_w - pad_w
        )
        return lower_bound, upper_bound

    def dilation_fn_cpu(dim: Int) {} -> Int:
        return dilation

    @always_inline
    def load_fn_cpu[
        simd_width: Int, dtype: DType
    ](point: IndexList[rank, ...]) {var h_input,} -> SIMD[dtype, simd_width]:
        return rebind[SIMD[dtype, simd_width]](
            h_input.load_linear[width=simd_width](point)
        )

    def avg_pool_compute_init_cpu[
        simd_width: Int
    ]() {} -> SIMD[dtype, simd_width]:
        return SIMD[dtype, simd_width](0)

    def avg_pool_compute_cpu[
        simd_width: SIMDLength
    ](
        point: IndexList[rank, ...],
        val: SIMD[dtype, simd_width],
        result: SIMD[dtype, simd_width],
    ) {} -> SIMD[dtype, simd_width]:
        return val + result

    @always_inline
    def avg_pool_compute_finalize_cpu[
        simd_width: SIMDLength
    ](point: IndexList[rank, ...], val: SIMD[dtype, simd_width]) {
        var h_output_ref,
    }:
        var res = val / Scalar[dtype](pool_window_h * pool_window_w)
        h_output_ref.store_linear(point, res)

    @always_inline
    def bench_cpu(mut b: Bencher) {imm}:
        @always_inline
        def kernel_launch() {imm}:
            comptime stencil_axis = IndexList[stencil_rank](1, 2)
            stencil[
                rank,
                stencil_rank,
                stencil_axis,
                simd_width,
                dtype,
            ](
                dynamic_output_shape,
                dynamic_input_shape,
                map_fn_cpu,
                dilation_fn_cpu,
                load_fn_cpu,
                avg_pool_compute_init_cpu,
                avg_pool_compute_cpu,
                avg_pool_compute_finalize_cpu,
            )

        b.iter(kernel_launch)

    # Calculate FLOPs for throughput measurement
    def compute_flops() -> Int:
        return (
            input_height * input_width * pool_window_h * pool_window_w * 2
        )  # One add, one divide per window element

    # Ensure correctness
    assert_allclose(h_output_ref, h_output)

    # Run benchmarks
    var bench_name = String(
        "stencil_avg_pool_padded_",
        batch_size,
        "x",
        input_height,
        "x",
        input_width,
        "_pad",
        pad_h,
        "x",
        pad_w,
    )

    var flops = ThroughputMeasure(BenchMetric.flops, compute_flops())
    m.bench_function(
        bench_gpu,
        BenchId(bench_name + "_gpu"),
        [flops],
    )

    m.bench_function(
        bench_cpu,
        BenchId(bench_name + "_cpu"),
        [flops],
    )

    # Ensure correctness
    ctx.enqueue_copy(h_output.ptr, d_output_buf)
    ctx.synchronize()
    assert_allclose(h_output_ref, h_output)

    _ = d_input_buf^
    _ = d_output_buf^
    _ = h_output_ref_ptr^
    _ = h_output_ptr^
    _ = h_input_ptr^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime batch_size = get_defined_int["batch_size", 128]()
    comptime input_height = get_defined_int["input_height", 1024]()
    comptime input_width = get_defined_int["input_width", 1024]()
    comptime num_channels = get_defined_int["num_channels", 3]()
    comptime pool_window_h = get_defined_int["pool_window_h", 3]()
    comptime pool_window_w = get_defined_int["pool_window_w", 3]()

    comptime pad_h = get_defined_int["pad_h", 0]()
    comptime pad_w = get_defined_int["pad_w", 0]()
    comptime method = get_defined_string["method", "max_pool"]()

    var m = Bench()
    with DeviceContext() as ctx:
        comptime if method == "avg_pool":
            bench_stencil_avg_pool[
                dtype,
                batch_size,
                input_height,
                input_width,
                pool_window_h,
                pool_window_w,
                num_channels,
            ](ctx, m)
        elif method == "max_pool":
            bench_stencil_max_pool[
                dtype,
                batch_size,
                input_height,
                input_width,
                pool_window_h,
                pool_window_w,
                num_channels,
            ](ctx, m)
        elif method == "avg_pool_padded":
            bench_stencil_avg_pool_padded[
                dtype,
                batch_size,
                input_height,
                input_width,
                pool_window_h,
                pool_window_w,
                pad_h,
                pad_w,
            ](ctx, m)

    m.dump_report()
