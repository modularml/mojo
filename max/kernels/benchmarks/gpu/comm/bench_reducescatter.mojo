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

from std.collections import Array
from std.sys import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    size_of,
    simd_width_of,
)
from std.utils.numerics import get_accum_type

from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.benchmark import (
    bench_multicontext,
    bencher_iter_custom,
)
from comm.sync import enable_p2p
from comm.reducescatter import reducescatter, ReduceScatterConfig
from layout import Idx, TileTensor, row_major
from comm import MAX_GPUS, Signal
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
    human_readable_size,
)

from std.testing import assert_almost_equal, assert_true


@always_inline
@__parameter
def _per_gpu_value[
    dtype: DType,
](gpu_rank: Int, j: Int) -> Scalar[dtype]:
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](Scalar[dtype](gpu_rank + 1) + Scalar[dtype](j % 251))


def _get_test_str[
    dtype: DType,
    use_multimem: Bool,
    cache_busting: Bool,
](ngpus: Int, num_bytes: Int, ragged: Bool) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    var cache_tag = "-cachebust" if cache_busting else ""
    var ragged_tag = "-ragged" if ragged else ""
    return String(
        "reducescatter-",
        dtype,
        "-",
        ngpus,
        multimem_tag,
        cache_tag,
        ragged_tag,
        "-",
        human_readable_size(num_bytes),
    )


def bench_reducescatter_2d[
    dtype: DType,
    axis: Int,
    ngpus: Int,
    *,
    use_multimem: Bool,
    cache_busting: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    M: Int,
    D: Int,
    max_num_blocks: Optional[Int],
) raises:
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime assert axis == 0 or axis == 1
    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

    var multimem_tag = "-multimem" if use_multimem else ""
    var cache_tag = "-cachebust" if cache_busting else ""
    var name = String(
        "reducescatter-axis",
        axis,
        "-",
        dtype,
        "-",
        ngpus,
        multimem_tag,
        cache_tag,
        "-(",
        M,
        "x",
        D,
        ")",
    )
    print("Running " + name)

    var num_elements = M * D
    var num_bytes = num_elements * size_of[dtype]()

    # Compute axis-aware partitioning.
    var axis_size: Int
    var unit_numel: Int
    if axis == 0:
        axis_size = M
        unit_numel = D
    else:
        axis_size = D // simd_size
        unit_numel = M * simd_size

    var rs_config = ReduceScatterConfig[dtype, ngpus](axis_size, unit_numel, 0)

    comptime num_buffers = 1 if use_multimem else ngpus

    # Create cache busting input buffers for each GPU.
    var cb_inputs = List[CacheBustingBuffer[dtype]]()
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[List[Scalar[dtype]]](capacity=ngpus)

    # Create signal buffers for synchronization.
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # Initialize buffers for each GPU.
    comptime for gpu_idx in range(ngpus):
        cb_inputs.append(
            CacheBustingBuffer[dtype](
                num_elements,
                simd_size,
                list_of_ctx[gpu_idx],
                cache_busting,
            )
        )
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                rs_config.rank_num_elements(gpu_idx)
            )
        )

        # Create and initialize host buffers.
        var host_buffer = List[Scalar[dtype]](
            unsafe_uninit_length=cb_inputs[0].alloc_size()
        )

        # Fill with repeated GPU-specific values for cache busting.
        for i in range(cb_inputs[0].alloc_size() // cb_inputs[0].stride):
            for j in range(num_elements):
                host_buffer[i * cb_inputs[0].stride + j] = _per_gpu_value[
                    dtype
                ](gpu_idx, j)

        # Copy to device.
        list_of_ctx[gpu_idx].enqueue_copy(
            cb_inputs[gpu_idx].device_buffer(), host_buffer
        )

        host_buffers.append(host_buffer^)

        # Create and initialize signal buffers.
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[.uint8](size_of[Signal]())
        )
        list_of_ctx[gpu_idx].enqueue_memset[.uint8](signal_buffers[gpu_idx], 0)
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    # Create 2D input and output TileTensors.
    comptime OutputTileType = TileTensor[
        dtype, type_of(row_major(M, D)), MutAnyOrigin
    ]
    comptime InputTileType = TileTensor[
        dtype, type_of(row_major(M, D)), ImmutAnyOrigin
    ]
    var in_bufs = Array[InputTileType, num_buffers](uninitialized=True)
    var out_bufs = Array[OutputTileType, ngpus](uninitialized=True)

    comptime for i in range(ngpus):
        in_bufs[i if not use_multimem else 0] = InputTileType(
            cb_inputs[i].device_buffer(), row_major(M, D)
        )
        if axis == 0:
            var my_rows = rs_config.rank_units(i)
            out_bufs[i] = OutputTileType(
                out_bufs_list[i],
                row_major(my_rows, D),
            )
        else:
            var my_cols = rs_config.rank_units(i) * simd_size
            out_bufs[i] = OutputTileType(
                out_bufs_list[i],
                row_major(M, my_cols),
            )
        list_of_ctx[i].synchronize()

    @always_inline
    def bench_iter_2d(
        mut b: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_bufs, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_bufs, imm}:
            comptime for i in range(num_buffers):
                in_bufs[i] = InputTileType(
                    cb_inputs[i].offset_ptr(cache_iter).as_unsafe_any_origin(),
                    row_major(M, D),
                )

            reducescatter[
                dtype=dtype,
                ngpus=ngpus,
                use_multimem=use_multimem,
                axis=axis,
            ](
                in_bufs,
                out_bufs[ctx_idx],
                rank_sigs,
                ctx_inner,
                max_num_blocks,
            )

        bencher_iter_custom(b, call_fn, ctx)

    bench_multicontext(
        b,
        bench_iter_2d,
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    b.dump_report()

    # Copy results back and verify.
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results.
    comptime for i in range(ngpus):
        if axis == 0:
            var row_start = rs_config.rank_unit_start(i)
            var my_rows = rs_config.rank_units(i)
            for r in range(my_rows):
                for c in range(D):
                    var global_idx = (row_start + r) * D + c
                    comptime accum_t = get_accum_type[dtype]()
                    var accum = Scalar[accum_t](0)
                    comptime for k in range(ngpus):
                        accum += Scalar[accum_t](
                            _per_gpu_value[dtype](k, global_idx)
                        )
                    var expected_sum = Scalar[dtype](accum)
                    try:
                        var rtol, atol = pytorch_like_tolerances_for[dtype]()
                        assert_almost_equal(
                            host_buffers[i][r * D + c],
                            expected_sum,
                            atol=atol,
                            rtol=rtol,
                        )
                    except e:
                        print(
                            "Verification failed at GPU",
                            i,
                            "row",
                            r,
                            "col",
                            c,
                        )
                        print("Value:", host_buffers[i][r * D + c])
                        print("Expected:", expected_sum)
                        raise e^
        else:
            var col_start = rs_config.rank_unit_start(i) * simd_size
            var my_cols = rs_config.rank_units(i) * simd_size
            for r in range(M):
                for c in range(my_cols):
                    var global_idx = r * D + (col_start + c)
                    comptime accum_t = get_accum_type[dtype]()
                    var accum = Scalar[accum_t](0)
                    comptime for k in range(ngpus):
                        accum += Scalar[accum_t](
                            _per_gpu_value[dtype](k, global_idx)
                        )
                    var expected_sum = Scalar[dtype](accum)
                    try:
                        var rtol, atol = pytorch_like_tolerances_for[dtype]()
                        assert_almost_equal(
                            host_buffers[i][r * my_cols + c],
                            expected_sum,
                            atol=atol,
                            rtol=rtol,
                        )
                    except e:
                        print(
                            "Verification failed at GPU",
                            i,
                            "row",
                            r,
                            "col",
                            c,
                        )
                        print("Value:", host_buffers[i][r * my_cols + c])
                        print("Expected:", expected_sum)
                        raise e^

    # Clean up
    _ = host_buffers^


def bench_reducescatter[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    use_multimem: Bool,
    cache_busting: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
    ragged: Bool,
) raises:
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime assert rank == 1, "this test code currently assumes rank 1"

    var name = String(
        _get_test_str[dtype, use_multimem, cache_busting](
            ngpus, num_bytes, ragged
        )
    )
    print("Running " + name)

    # Total input size per GPU
    var input_length = num_bytes // size_of[dtype]()

    # Use ReduceScatterConfig to compute per-GPU partition info (w/ dummy nthreads)
    var rs_config = ReduceScatterConfig[dtype, ngpus](input_length, 0)
    var output_lengths = List[Int](capacity=ngpus)
    var rank_starts = List[Int](capacity=ngpus)
    for gpu_idx in range(ngpus):
        rank_starts.append(rs_config.rank_start(gpu_idx))
        output_lengths.append(rs_config.rank_part(gpu_idx))

    comptime num_buffers = 1 if use_multimem else ngpus

    # Create cache busting input buffers for each GPU
    var cb_inputs = List[CacheBustingBuffer[dtype]]()
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[List[Scalar[dtype]]](capacity=ngpus)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # Initialize buffers for each GPU
    for gpu_idx in range(ngpus):
        # Create input (cache busting) and output device buffers
        cb_inputs.append(
            CacheBustingBuffer[dtype](
                input_length,
                rs_config.simd_width,
                list_of_ctx[gpu_idx],
                cache_busting,
            )
        )
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                output_lengths[gpu_idx]
            )
        )

        # Create and initialize host buffers
        var host_buffer = List[Scalar[dtype]](
            unsafe_uninit_length=cb_inputs[0].alloc_size()
        )

        # Fill with repeated GPU-specific values for cache busting
        for i in range(cb_inputs[0].alloc_size() // cb_inputs[0].stride):
            for j in range(input_length):
                host_buffer[i * cb_inputs[0].stride + j] = _per_gpu_value[
                    dtype
                ](gpu_idx, j)

        # Copy to device
        list_of_ctx[gpu_idx].enqueue_copy(
            cb_inputs[gpu_idx].device_buffer(), host_buffer
        )

        host_buffers.append(host_buffer^)

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[.uint8](size_of[Signal]())
        )
        list_of_ctx[gpu_idx].enqueue_memset[.uint8](signal_buffers[gpu_idx], 0)
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    # Create input and output TileTensors
    comptime OutputTileType = TileTensor[
        dtype, type_of(row_major(output_lengths[0])), MutAnyOrigin
    ]
    comptime InputTileType = TileTensor[
        dtype, type_of(row_major(output_lengths[0])), ImmutAnyOrigin
    ]
    var in_bufs = Array[InputTileType, num_buffers](uninitialized=True)
    var out_bufs = Array[OutputTileType, ngpus](uninitialized=True)

    for i in range(ngpus):
        in_bufs[i if not use_multimem else 0] = InputTileType(
            cb_inputs[i].device_buffer(), row_major(input_length)
        )
        out_bufs[i] = OutputTileType(
            out_bufs_list[i], row_major(output_lengths[i])
        )
        list_of_ctx[i].synchronize()

    @always_inline
    def bench_iter(
        mut b: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_bufs, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_bufs, imm}:
            comptime for i in range(num_buffers):
                in_bufs[i] = InputTileType(
                    cb_inputs[i].offset_ptr(cache_iter).as_unsafe_any_origin(),
                    row_major(input_length),
                )

            reducescatter[dtype=dtype, ngpus=ngpus, use_multimem=use_multimem](
                in_bufs,
                out_bufs[ctx_idx],
                rank_sigs,
                ctx_inner,
                max_num_blocks,
            )

        bencher_iter_custom(b, call_fn, ctx)

    bench_multicontext(
        b,
        bench_iter,
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, num_bytes)],
    )
    b.dump_report()

    # Copy results back and verify
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    comptime for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Verify results
    # For low-precision dtypes (e.g., bfloat16), inputs were quantized to `dtype`
    # before reduction on device. Mirror the device path here by:
    #  - quantizing each per-GPU term to `dtype` by calling _per_gpu_value[dtype](...)
    #  - accumulating in Float32
    #  - finally casting to `dtype` for the expected value
    comptime for i in range(ngpus):
        for j in range(output_lengths[i]):
            comptime accum_t = get_accum_type[dtype]()
            var accum = Scalar[accum_t](0)
            var global_idx = rank_starts[i] + j

            comptime for k in range(ngpus):
                var term_dtype = _per_gpu_value[dtype](k, global_idx)
                accum += Scalar[accum_t](term_dtype)
            var expected_sum = Scalar[dtype](accum)
            try:
                var rtol, atol = pytorch_like_tolerances_for[dtype]()
                assert_almost_equal(
                    host_buffers[i][j], expected_sum, atol=atol, rtol=rtol
                )
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", expected_sum)
                raise e^

    # Clean up
    _ = host_buffers^


def main() raises:
    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
    comptime axis = get_defined_int["axis", -1]()
    comptime use_multimem = get_defined_bool["multimem", False]()
    comptime cache_busting = True

    var max_nb = get_defined_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )

    # Create GPU contexts
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    var m = Bench()

    comptime if axis == -1:
        # 1D flat reduce-scatter
        var num_bytes = arg_parse("num_bytes", 16 * 1024)
        comptime rank = get_defined_int["rank", 1]()
        comptime ragged = get_defined_bool["ragged", False]()
        comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

        comptime if ragged:
            num_bytes += (num_gpus // 2) * simd_size * size_of[dtype]()

        assert_true(num_bytes % size_of[dtype]() == 0)

        bench_reducescatter[
            dtype=dtype,
            rank=rank,
            ngpus=num_gpus,
            use_multimem=use_multimem,
            cache_busting=cache_busting,
        ](m, ctx, num_bytes, max_num_blocks, ragged)
    else:
        # 2D axis-aware reduce-scatter
        var M = arg_parse("M", 16)
        var D = arg_parse("D", 128)

        bench_reducescatter_2d[
            dtype=dtype,
            axis=axis,
            ngpus=num_gpus,
            use_multimem=use_multimem,
            cache_busting=cache_busting,
        ](m, ctx, M, D, max_num_blocks)
