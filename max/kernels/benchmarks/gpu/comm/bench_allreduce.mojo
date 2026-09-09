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
from layout import Idx, TileTensor, row_major
from comm.sync import enable_p2p, init_signal_buffer
from comm.allreduce import allreduce
from comm import MAX_GPUS, Signal
import comm.vendor.ccl as vendor_ccl
from max.gpu.host import (
    DeviceBuffer,
    DeviceContext,
    DeviceMulticastBuffer,
    get_gpu_target,
)
from internal_utils import (
    CacheBustingBuffer,
    arg_parse,
    pytorch_like_tolerances_for,
    human_readable_size,
)

from std.testing import assert_almost_equal, assert_true

from std.utils.index import StaticTuple


@always_inline
@__parameter
def _per_gpu_value[
    dtype: DType,
](gpu_rank: Int, j: Int) -> Scalar[dtype]:
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](Scalar[dtype](gpu_rank + 1) + Scalar[dtype](j % 251))


# TODO: convert 'ngpus' to runtime variable
def bench_reduce[
    dtype: DType,
    ngpus: Int,
    *,
    use_multimem: Bool,
    cache_busting: Bool,
    use_vendor_ccl: Bool = False,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    num_bytes: Int,
    max_num_blocks: Optional[Int],
    ragged: Bool,
) raises:
    comptime assert ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"

    var name = String(
        _get_test_str[dtype, use_multimem, use_vendor_ccl, cache_busting](
            ngpus, num_bytes, max_num_blocks.or_else(0), ragged
        )
    )

    # Create cache busting template on GPU 0 for metadata (stride, alloc_size,
    # offset). In non-multimem path, also serves as GPU 0's input buffer.
    var out_dev = List[DeviceBuffer[dtype]](capacity=ngpus)
    var host_buffers = List[List[Scalar[dtype]]](capacity=ngpus)

    comptime num_buffers = 1 if use_multimem else ngpus

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * num_bytes
    var length = num_bytes // size_of[dtype]()

    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()
    var cb_template = CacheBustingBuffer[dtype](
        length, simd_size, list_of_ctx[0], cache_busting
    )

    # Per-GPU input CacheBustingBuffers (non-multimem path)
    var cb_inputs = List[CacheBustingBuffer[dtype]]()

    # Initialize buffers for each GPU
    comptime for gpu_idx in range(ngpus):
        # Create and store device buffers (outputs)
        out_dev.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](length)
        )

        # Create and initialize host buffers
        var host_buffer = List[Scalar[dtype]](
            unsafe_uninit_length=cb_template.alloc_size()
        )

        for i in range(cb_template.alloc_size() // cb_template.stride):
            for j in range(length):
                host_buffer[i * cb_template.stride + j] = _per_gpu_value[dtype](
                    gpu_idx, j
                )

        comptime if not use_multimem:
            # Create per-GPU input buffers on device and copy from host
            if gpu_idx == 0:
                cb_inputs.append(cb_template)
            else:
                cb_inputs.append(
                    CacheBustingBuffer[dtype](
                        length, simd_size, list_of_ctx[gpu_idx], cache_busting
                    )
                )
            list_of_ctx[gpu_idx].enqueue_copy(
                cb_inputs[gpu_idx].device_buffer(), host_buffer
            )

        host_buffers.append(host_buffer^)

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[gpu_idx].create_buffer_sync[.uint8](
                size_of[Signal]() + temp_buffer_num_bytes
            )
        )
        init_signal_buffer(signal_buffers[gpu_idx], list_of_ctx[gpu_idx])
        rank_sigs[gpu_idx] = (
            signal_buffers[gpu_idx]
            .unsafe_ptr()
            .bitcast[Signal]()
            .as_unsafe_any_origin()
        )

    # Create and initialize input and output buffers.
    comptime InTensorType = TileTensor[
        dtype, type_of(row_major(length)), ImmutAnyOrigin
    ]
    comptime OutTensorType = TileTensor[
        dtype, type_of(row_major(length)), MutAnyOrigin
    ]
    var in_tensors = Array[InTensorType, num_buffers](uninitialized=True)
    var out_tensors = Array[OutTensorType, ngpus](uninitialized=True)

    var multi_ptr = Optional[MutPointer[Scalar[dtype], MutAnyOrigin]]()

    comptime if use_multimem:
        var multicast_buf = DeviceMulticastBuffer[dtype](
            list_of_ctx.copy(), cb_template.alloc_size()
        )

        comptime for i in range(ngpus):
            var unicast_buf = multicast_buf.unicast_buffer_for(list_of_ctx[i])
            list_of_ctx[i].enqueue_copy(unicast_buf, host_buffers[i])

        # All GPUs use the same multicast pointer
        multi_ptr = (
            multicast_buf.multicast_buffer_for(list_of_ctx[0])
            .unsafe_ptr()
            .unsafe_mut_cast[True]()
            .as_unsafe_any_origin()
        )
        in_tensors[0] = TileTensor(
            rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                multi_ptr.unsafe_value()
            ),
            row_major(length),
        )
    else:
        comptime for i in range(ngpus):
            in_tensors[i] = TileTensor(
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    cb_inputs[i].unsafe_ptr()
                ),
                row_major(length),
            )

    for i in range(ngpus):
        out_tensors[i] = TileTensor(out_dev[i], row_major(length))
        # Ensure setup has propagated.
        list_of_ctx[i].synchronize()

    # Zero device output buffers once before benchmarking so verification isn't
    # affected by any stale data in case a kernel path doesn't overwrite fully.
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_memset(out_dev[i], 0)

    # Copy-capture in registers since the lambda will be used on GPU.
    var out_tensors_capture = StaticTuple[OutTensorType, ngpus]()

    comptime for i in range(ngpus):
        out_tensors_capture[i] = TileTensor(out_dev[i], row_major(length))

    # Pre-initialize vendor CCL communicators from the main thread.
    # ncclCommInitAll is not thread-safe, so we must initialize before
    # spawning worker threads.
    comptime if use_vendor_ccl:
        if not vendor_ccl.is_allreduce_available():
            raise "Vendor CCL not available; skipping vendor path."
        vendor_ccl.init_comms(ngpus)

    @always_inline
    def bench_iter(
        mut b: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut in_tensors, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut in_tensors, imm}:
            comptime if not use_multimem:
                comptime for i in range(ngpus):
                    in_tensors[i] = TileTensor(
                        rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                            cb_inputs[i].offset_ptr(cache_iter)
                        ),
                        row_major(length),
                    )
            else:
                # multi_ptr is set when use_multimem == True
                in_tensors[0] = TileTensor(
                    rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                        multi_ptr.unsafe_value()
                        + cb_template.offset(cache_iter)
                    ),
                    row_major(length),
                )
            # Run allreduce
            comptime if use_vendor_ccl:
                vendor_ccl.allreduce[
                    ngpus=ngpus,
                    use_multimem=use_multimem,
                ](
                    in_tensors,
                    out_tensors[ctx_idx],
                    rank_sigs,
                    ctx_inner,
                    max_num_blocks,
                )
            else:
                allreduce[
                    ngpus=ngpus,
                    use_multimem=use_multimem,
                ](
                    in_tensors,
                    out_tensors[ctx_idx],
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

    var max_time = b.info_vec[0].result.mean(unit="ms")
    var gbps = Float64(num_bytes) / (max_time * 1000 * 1000)
    # algbw and busbw are explain in the following link:
    # https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md#allreduce
    var busbw = 2 * gbps * Float64((ngpus - 1)) / Float64(ngpus)
    print(
        "|",
        name,
        "| slowest mean time",
        max_time,
        "ms |",
        "algbw:",
        gbps,
        "GB/s |",
        "busbw:",
        busbw,
        "GB/s |",
    )

    # Copy results back and verify
    comptime for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_dev[i])

    # Verify results
    # For low-precision dtypes (e.g., bfloat16), inputs were quantized to `dtype`
    # before reduction on device. Mirror the device path here by:
    #  - quantizing each per-GPU term to `dtype` by calling _per_gpu_value[dtype](...)
    #  - accumulating in Float32
    #  - finally casting to `dtype` for the expected value
    comptime for i in range(ngpus):
        for j in range(length):
            comptime accum_t = get_accum_type[dtype]()
            var accum = Scalar[accum_t](0)

            comptime for k in range(ngpus):
                var term_dtype = _per_gpu_value[dtype](k, j)
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

    # Cleanup
    _ = host_buffers^
    _ = signal_buffers^


def _get_test_str[
    dtype: DType,
    use_multimem: Bool,
    use_vendorccl: Bool,
    cache_busting: Bool,
](ngpus: Int, num_bytes: Int, max_num_blocks: Int, ragged: Bool) -> String:
    var multimem_tag = "-multimem" if use_multimem else ""
    var vendorccl_tag = "-vendorccl" if use_vendorccl else ""
    var cache_tag = "-cachebust" if cache_busting else ""
    var ragged_tag = "-ragged" if ragged else ""
    return String(
        "allreduce-",
        dtype,
        "-",
        ngpus,
        "-max_num_blocks-",
        max_num_blocks,
        multimem_tag,
        vendorccl_tag,
        cache_tag,
        ragged_tag,
        "-",
        human_readable_size(num_bytes),
    )


def main() raises:
    var num_bytes = arg_parse("num_bytes", 16 * 1024)

    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
    comptime ragged = get_defined_bool["ragged", False]()
    # Force passing `max_num_blocks` explicitly.
    var max_nb = get_defined_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)
    comptime use_multimem = get_defined_bool["use_multimem", False]()
    comptime use_vendor_ccl = get_defined_bool["use_vendor_ccl", False]()
    comptime cache_busting = True

    # When ragged, add (ngpus/2) * simd_width elements to create uneven partitions
    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

    comptime if ragged:
        num_bytes += (num_gpus // 2) * simd_size * size_of[dtype]()

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(num_bytes % size_of[dtype]() == 0)

    # Create GPU context.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not enable_p2p():
        # Don't benchmark the naive allreduce.
        print("P2P not enabled, skipping benchmark.")
        return

    bench_reduce[
        dtype=dtype,
        ngpus=num_gpus,
        use_multimem=use_multimem,
        cache_busting=cache_busting,
        use_vendor_ccl=use_vendor_ccl,
    ](m, ctx, num_bytes, max_num_blocks, ragged)
