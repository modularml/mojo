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
"""Benchmark for scatter+broadcast kernel.

Distributes different data chunks from a root GPU to multiple device groups.
Each group (DP replica) gets a different chunk, and all devices within a group
(TP devices) get the same chunk.

Focus is on small sizes typical of row_offsets distribution.
"""

from std.collections import Array
from std.math import ceildiv
from std.sys import size_of, simd_width_of
from std.sys.defines import get_defined_bool, get_defined_dtype, get_defined_int

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
from comm.scatter import scatter
from layout import Idx, TileTensor, row_major
from comm import MAX_GPUS, Signal
from max.gpu.host import DeviceBuffer, DeviceContext
from internal_utils import arg_parse, CacheBustingBuffer

from std.testing import assert_true


@always_inline
@__parameter
def _chunk_value[dtype: DType](dp_idx: Int, j: Int) -> Scalar[dtype]:
    """Generate position-based value that includes the DP replica index.

    Each element has a unique value based on position and replica,
    allowing verification that the correct chunk reached each GPU.
    """
    # 251 is the largest prime < 256; using a prime avoids power-of-two aliasing.
    return Scalar[dtype](
        Scalar[dtype](dp_idx + 1) * 1000 + Scalar[dtype](j % 251)
    )


def _get_test_str[
    dtype: DType,
    cache_busting: Bool,
](ngpus: Int, dp_size: Int, num_elems: Int) -> String:
    var cache_tag = "-cachebust" if cache_busting else ""
    return String(
        "scatter-",
        dtype,
        "-",
        ngpus,
        "gpus-dp",
        dp_size,
        cache_tag,
        "-",
        num_elems,
        "elems",
    )


def bench_scatter[
    dtype: DType,
    ngpus: Int,
    dp_size: Int,
    *,
    cache_busting: Bool,
](mut b: Bench, list_of_ctx: List[DeviceContext], num_elems: Int) raises:
    comptime assert ngpus in (2, 4, 8), "ngpus must be 2, 4, or 8"
    comptime assert ngpus >= dp_size, "ngpus must be >= dp_size"
    comptime tp_size = ceildiv(ngpus, dp_size)

    var name = String(
        _get_test_str[dtype, cache_busting](ngpus, dp_size, num_elems)
    )
    print("Running " + name)

    var num_bytes = num_elems * size_of[dtype]()

    # Create cache-busting input buffers for each DP chunk on GPU 0 (root).
    # Use a conservative alignment that works across all GPU architectures.
    comptime alignment = max(16, size_of[dtype]())
    var cb_inputs = List[CacheBustingBuffer[dtype]]()
    var host_buffers = List[List[Scalar[dtype]]](capacity=dp_size)

    for dp_idx in range(dp_size):
        cb_inputs.append(
            CacheBustingBuffer[dtype](
                num_elems,
                alignment,
                list_of_ctx[0],
                cache_busting,
            )
        )

        var host_buffer = List[Scalar[dtype]](
            unsafe_uninit_length=cb_inputs[0].alloc_size()
        )

        for i in range(cb_inputs[0].alloc_size() // cb_inputs[0].stride):
            for j in range(num_elems):
                host_buffer[i * cb_inputs[0].stride + j] = _chunk_value[dtype](
                    dp_idx, j
                )

        list_of_ctx[0].enqueue_copy(
            cb_inputs[dp_idx].device_buffer(), host_buffer
        )

        host_buffers.append(host_buffer^)

    # Create output device buffers for each GPU.
    var out_bufs_list = List[DeviceBuffer[dtype]](capacity=ngpus)
    for gpu_idx in range(ngpus):
        out_bufs_list.append(
            list_of_ctx[gpu_idx].enqueue_create_buffer[dtype](num_elems)
        )
        list_of_ctx[gpu_idx].enqueue_memset(out_bufs_list[gpu_idx], 0)

    # Create signal buffers for synchronization.
    var signal_buffers = List[DeviceBuffer[.uint8]](capacity=ngpus)
    var rank_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )

    for gpu_idx in range(ngpus):
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

    # Build TileTensor arrays for the scatter API.
    comptime InputTileType = TileTensor[
        dtype, type_of(row_major(num_elems)), ImmutAnyOrigin
    ]
    var tt_in_bufs = Array[InputTileType, dp_size](
        fill_with=lambda (dp_idx: Int) -> InputTileType: TileTensor(
            cb_inputs[dp_idx].device_buffer(), row_major(num_elems)
        ).as_immut()
    )

    comptime OutputTileType = TileTensor[
        dtype, type_of(row_major(num_elems)), MutAnyOrigin
    ]
    var out_tiles = Array[OutputTileType, ngpus](uninitialized=True)
    for gpu_idx in range(ngpus):
        out_tiles[gpu_idx] = OutputTileType(
            out_bufs_list[gpu_idx], row_major(num_elems)
        )
        list_of_ctx[gpu_idx].synchronize()

    @always_inline
    def bench_iter(
        mut bencher: Bencher, ctx: DeviceContext, ctx_idx: Int
    ) raises {mut tt_in_bufs, imm}:
        @always_inline
        def call_fn(
            ctx_inner: DeviceContext, cache_iter: Int
        ) raises {mut tt_in_bufs, imm}:
            # Update input pointers to the cache-busted offset.
            comptime for dp_idx in range(dp_size):
                tt_in_bufs[dp_idx] = TileTensor(
                    cb_inputs[dp_idx].offset_ptr(cache_iter),
                    row_major(num_elems),
                ).as_immut()

            scatter[ngpus=ngpus, dp_size=dp_size](
                tt_in_bufs,
                out_tiles[ctx_idx],
                rank_sigs,
                ctx_inner,
            )

        bencher_iter_custom(bencher, call_fn, ctx)

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
    # For scatter, busbw = algbw (data leaves root once per replica).
    var busbw = gbps
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
    var verify_host = List(length=num_elems, fill=Scalar[dtype](0))
    for gpu_idx in range(ngpus):
        var dp_idx = gpu_idx // tp_size
        list_of_ctx[gpu_idx].enqueue_copy(verify_host, out_bufs_list[gpu_idx])
        list_of_ctx[gpu_idx].synchronize()

        for j in range(num_elems):
            var expected = _chunk_value[dtype](dp_idx, j)
            if verify_host[j] != expected:
                print(
                    "Verification failed at GPU",
                    gpu_idx,
                    "(replica",
                    dp_idx,
                    ") index",
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
    var num_elems = arg_parse("num_elems", 16)

    comptime dtype = get_defined_dtype["dtype", .uint32]()
    comptime num_gpus = get_defined_int["num_gpus", 2]()
    comptime dp_size = get_defined_int["dp_size", 2]()
    comptime cache_busting = get_defined_bool["cache_busting", True]()

    var m = Bench()

    var num_gpus_found = DeviceContext.number_of_devices()
    assert_true(
        num_gpus_found >= num_gpus,
        String(num_gpus_found) + " devices found, expected " + String(num_gpus),
    )
    assert_true(
        num_gpus >= dp_size,
        "num_gpus must be >= dp_size",
    )

    # Create GPU contexts.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    if not enable_p2p():
        print("P2P not enabled, skipping benchmark.")
        return

    bench_scatter[
        dtype=dtype,
        ngpus=num_gpus,
        dp_size=dp_size,
        cache_busting=cache_busting,
    ](m, ctx, num_elems)
