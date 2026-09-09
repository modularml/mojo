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
from std.sys.defines import (
    get_defined_bool,
    get_defined_dtype,
    get_defined_int,
    get_defined_string,
)
from std.sys import size_of, simd_width_of

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
from comm.sync import enable_p2p
from comm.allgather import allgather
from comm import MAX_GPUS, Signal
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from internal_utils import arg_parse, human_readable_size, CacheBustingBuffer

from std.testing import assert_true


@always_inline
def _per_gpu_value[dtype: DType](gpu_rank: Int, j: Int) -> Scalar[dtype]:
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](Scalar[dtype](gpu_rank + 1) + Scalar[dtype](j % 251))


def _compute_lengths[
    ngpus: Int, length_mode: StaticString
](num_bytes: Int, elem_size: Int) -> Array[Int, ngpus]:
    """Compute per-GPU element counts based on the length distribution mode.

    Modes:
        uniform:      All GPUs get the same length.
        halved_last:  Last GPU gets half the elements.
        doubled_first: First GPU gets 2x, rest get num_bytes worth.
        staircase:    Linear ramp from num_bytes/ngpus to num_bytes.
        uneven:       First half gets length+1, second half gets length
                      (mimics off-by-one splits like 37919/37918).
    """
    var base_length = num_bytes // elem_size
    var lengths = Array[Int, ngpus](fill=base_length)

    comptime if length_mode == "uniform":
        pass
    elif length_mode == "halved_last":
        lengths[ngpus - 1] = base_length // 2
    elif length_mode == "doubled_first":
        lengths[0] = base_length * 2
    elif length_mode == "staircase":
        for i in range(ngpus):
            lengths[i] = base_length * (i + 1) // ngpus
    elif length_mode == "uneven":
        # First ceil(ngpus/2) GPUs get base_length, rest get base_length - 1.
        # If base_length is 0 the shorter ones clamp to 0.
        var half = (ngpus + 1) // 2
        for i in range(half, ngpus):
            lengths[i] = max(base_length - 1, 0)
    else:
        comptime assert False, "unknown length_mode: " + length_mode

    return lengths^


def _get_test_str[
    dtype: DType,
    cache_busting: Bool,
    length_mode: StaticString,
](ngpus: Int, total_bytes: Int) -> String:
    var cache_tag = "-cachebust" if cache_busting else ""
    var mode_tag = String("")
    comptime if length_mode != "uniform":
        mode_tag = String("-", length_mode)
    return String(
        "allgather-",
        dtype,
        "-",
        ngpus,
        "gpus",
        mode_tag,
        cache_tag,
        "-",
        human_readable_size(total_bytes),
    )


def bench_allgather[
    dtype: DType,
    ngpus: Int,
    length_mode: StaticString,
    *,
    cache_busting: Bool,
](
    mut b: Bench,
    list_of_ctx: List[DeviceContext],
    lengths: Array[Int, ngpus],
    max_num_blocks: Optional[Int],
) raises:
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"

    var total_bytes = 0
    for i in range(ngpus):
        total_bytes += lengths[i] * size_of[dtype]()

    var name = String(
        _get_test_str[dtype, cache_busting, length_mode](ngpus, total_bytes)
    )
    print("Running " + name)

    comptime simd_size = simd_width_of[dtype, target=get_gpu_target()]()

    # Create cache-busting input buffers for each GPU.
    var cb_inputs = List[CacheBustingBuffer[dtype]]()
    var host_buffers = List[List[Scalar[dtype]]](capacity=ngpus)

    # Create output device buffers: ngpus outputs per GPU (one per source).
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus * ngpus)

    # Create signal buffers for synchronization.
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for gpu_idx in range(ngpus):
        var length = lengths[gpu_idx]

        # Input buffer with cache busting.
        cb_inputs.append(
            CacheBustingBuffer[dtype](
                length,
                simd_size,
                list_of_ctx[gpu_idx],
                cache_busting,
            )
        )

        # Output buffers: one per source GPU on this device.
        for src_idx in range(ngpus):
            out_bufs_list.append(
                list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](
                    lengths[src_idx]
                )
            )

        # Host buffer for verification.
        var host_buffer = List[Scalar[dtype]](
            unsafe_uninit_length=cb_inputs[gpu_idx].alloc_size()
        )

        # Fill with GPU-specific values for cache busting.
        for i in range(
            cb_inputs[gpu_idx].alloc_size() // cb_inputs[gpu_idx].stride
        ):
            for j in range(length):
                host_buffer[i * cb_inputs[gpu_idx].stride + j] = _per_gpu_value[
                    dtype
                ](gpu_idx, j)

        # Copy to device.
        list_of_ctx[gpu_idx].enqueue_copy(
            cb_inputs[gpu_idx].device_buffer(), host_buffer
        )

        host_buffers.append(host_buffer^)

        # Signal buffers.
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

    # Build TileTensor arrays for allgather.
    comptime InTileType = TileTensor[
        dtype, type_of(row_major(lengths[0])), ImmutAnyOrigin
    ]
    var tt_in = Array[InTileType, ngpus](uninitialized=True)

    comptime OutTileType = TileTensor[
        dtype, type_of(row_major(lengths[0])), MutAnyOrigin
    ]
    var tt_out = Array[OutTileType, ngpus * ngpus](uninitialized=True)

    for gpu_idx in range(ngpus):
        comptime for src_idx in range(ngpus):
            var flat_idx = gpu_idx * ngpus + src_idx
            tt_out[flat_idx] = TileTensor(
                out_bufs_list[flat_idx],
                row_major(lengths[src_idx]),
            )
        list_of_ctx[gpu_idx].synchronize()

    @always_inline
    def bench_iter(
        mut bencher: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut tt_in, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut tt_in, imm}:
            # Update input pointers to the cache-busted offset.
            comptime for i in range(ngpus):
                tt_in[i] = TileTensor(
                    cb_inputs[i].offset_ptr(cache_iter),
                    row_major(lengths[i]),
                ).as_immut()

            var device_out = Array[OutTileType, ngpus](
                fill_with_unrolled=lambda [
                    src_idx: Int
                ]() -> OutTileType: tt_out[ctx_idx * ngpus + src_idx]
            )

            allgather(
                tt_in,
                device_out,
                rank_sigs,
                ctx_inner,
                ctx_idx,
                max_num_blocks,
            )

        bencher_iter_custom(bencher, call_fn, ctx)

    bench_multicontext(
        b,
        bench_iter,
        list_of_ctx,
        BenchId(name),
        [ThroughputMeasure(BenchMetric.bytes, total_bytes)],
    )
    b.dump_report()

    var max_time = b.info_vec[0].result.mean(unit="ms")
    var gbps = Float64(total_bytes) / (max_time * 1000 * 1000)
    # For allgather, busbw = algbw * (n-1)/n.
    # See: https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md
    var busbw = gbps * Float64(ngpus - 1) / Float64(ngpus)
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

    # Copy results back and verify the benchmarked outputs directly.
    var max_length = 0
    for i in range(ngpus):
        max_length = max(max_length, lengths[i])
    var verify_host = List(length=max_length, fill=Scalar[dtype](0))

    for gpu_idx in range(ngpus):
        for src_idx in range(ngpus):
            var src_length = lengths[src_idx]
            var flat_idx = gpu_idx * ngpus + src_idx
            list_of_ctx[gpu_idx].enqueue_copy(
                verify_host, out_bufs_list[flat_idx]
            )
            list_of_ctx[gpu_idx].synchronize()

            for j in range(src_length):
                var expected = _per_gpu_value[dtype](src_idx, j)
                if verify_host[j] != expected:
                    print(
                        "Verification failed at GPU",
                        gpu_idx,
                        "source",
                        src_idx,
                        "index",
                        j,
                    )
                    print("Value:", verify_host[j])
                    print("Expected:", expected)
                    raise Error("Verification failed")

    # Cleanup.
    _ = host_buffers^
    _ = signal_buffers^
    _ = cb_inputs^
    _ = verify_host^


def main() raises:
    var num_bytes = arg_parse("num_bytes", 64 * 1024 * 1024)

    comptime dtype = get_defined_dtype["dtype", .bfloat16]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()
    comptime length_mode = get_defined_string["length_mode", "uniform"]()

    var max_nb = get_defined_int["TUNE_MAX_NUM_BLOCKS", -1]()
    var max_num_blocks: Optional[Int] = Optional[Int]()
    if max_nb > 0:
        max_num_blocks = Optional[Int](max_nb)

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(num_bytes % size_of[dtype]() == 0)

    var lengths = _compute_lengths[num_gpus, length_mode](
        num_bytes, size_of[dtype]()
    )

    # Create GPU contexts.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    bench_allgather[
        dtype=dtype,
        ngpus=num_gpus,
        length_mode=length_mode,
        cache_busting=cache_busting,
    ](m, ctx, lengths, max_num_blocks)
