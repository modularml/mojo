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
"""Benchmark for cp.async.bulk (1D TMA) memcpy throughput.

Measures global->shared->global copy bandwidth using cp.async.bulk,
parameterized by:
  - NUM_THREADS (compile-time): block_dim, controls number of warps issuing TMA
  - BYTES_PER_COPY (compile-time): bytes each warp copies per TMA instruction
  - PIPELINE_NUM_STAGES (compile-time): pipeline depth per warp (S slots + mbars)
  - grid_dim (runtime): number of thread blocks
  - total_bytes (runtime): total bytes to copy

Pipeline design (per warp, S = PIPELINE_NUM_STAGES):

  Prologue:  issue g2s for slots 0..S-2  (S-1 loads in flight)
  Steady:    wait_group[0]  -> g2s(new_slot)  -> wait_mbar(drain_slot)
             -> s2g(drain_slot) + commit
  Epilogue:  drain remaining S-1 slots

The g2s issued in steady state overlaps with the mbar_wait + s2g of a
different slot, hiding g2s latency behind S-1 iterations of work.
"""

from std.sys import get_defined_bool, get_defined_int, size_of

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu import (
    block_idx,
    grid_dim as gpu_grid_dim,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, FuncAttribute
from max.gpu.memory import (
    cp_async_bulk_global_shared_cta,
    cp_async_bulk_prefetch,
    cp_async_bulk_shared_cluster_global,
    external_memory,
    fence_mbarrier_init,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.sync import (
    cp_async_bulk_commit_group,
    cp_async_bulk_wait_group,
)
from internal_utils import arg_parse

from layout.tma_async import SharedMemBarrier


@always_inline
def _smem_ptr[
    BYTES_PER_COPY: Int, S: Int
](
    base: MutPointer[UInt8, _, address_space=.SHARED],
    warp: Int,
    slot: Int,
) -> type_of(base):
    return base + (warp * S + slot) * BYTES_PER_COPY


@always_inline
def _mbar_ref[
    S: Int
](
    base: MutPointer[SharedMemBarrier, _, address_space=.SHARED],
    warp: Int,
    slot: Int,
) -> type_of(base):
    return base + warp * S + slot


def bulk_memcpy_kernel[
    NUM_THREADS: Int, BYTES_PER_COPY: Int, S: Int, PREFETCH: Bool
](
    src: ImmPointer[UInt8, ImmutAnyOrigin],
    dst: MutPointer[UInt8, MutAnyOrigin],
    total_chunks: Int32,
):
    comptime NUM_WARPS = NUM_THREADS // 32
    comptime DATA_BYTES = NUM_WARPS * S * BYTES_PER_COPY

    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var mbar_base = (smem_base + DATA_BYTES).bitcast[SharedMemBarrier]()

    var w = Int(warp_id())
    var is_leader = elect_one_sync()

    if is_leader:
        for s in range(S):
            _mbar_ref[S](mbar_base, w, s)[].init()
    fence_mbarrier_init()
    barrier()

    var src_g = src.address_space_cast[.GLOBAL]()
    var dst_g = dst.address_space_cast[.GLOBAL]()

    var first = Int(block_idx.x) * NUM_WARPS + w
    var stride = Int(gpu_grid_dim.x) * NUM_WARPS
    var my_total = max(0, (Int(total_chunks) - first + stride - 1) // stride)

    if not is_leader:
        return

    var prologue = min(S - 1, my_total)

    # --- Prologue: issue S-1 g2s loads (or fewer if my_total < S) ---
    for i in range(prologue):
        var slot = i % S
        var smem = _smem_ptr[BYTES_PER_COPY, S](smem_base, w, slot)
        var mbar = _mbar_ref[S](mbar_base, w, slot)

        comptime if PREFETCH:
            var next_i = i + 1
            if next_i < my_total:
                cp_async_bulk_prefetch(
                    src_g + (first + next_i * stride) * BYTES_PER_COPY,
                    Int32(BYTES_PER_COPY),
                )

        mbar[].expect_bytes(Int32(BYTES_PER_COPY))
        cp_async_bulk_shared_cluster_global(
            smem,
            src_g + (first + i * stride) * BYTES_PER_COPY,
            Int32(BYTES_PER_COPY),
            mbar[].unsafe_ptr(),
        )

    # --- Steady state ---
    for i in range(S - 1, my_total):
        var new_slot = i % S
        var drain_idx = i - (S - 1)
        var drain_slot = drain_idx % S

        comptime if PREFETCH:
            var next_i = i + 1
            if next_i < my_total:
                cp_async_bulk_prefetch(
                    src_g + (first + next_i * stride) * BYTES_PER_COPY,
                    Int32(BYTES_PER_COPY),
                )

        cp_async_bulk_wait_group[Int32(S - 1)]()

        var smem_new = _smem_ptr[BYTES_PER_COPY, S](smem_base, w, new_slot)
        var mbar_new = _mbar_ref[S](mbar_base, w, new_slot)
        mbar_new[].expect_bytes(Int32(BYTES_PER_COPY))
        cp_async_bulk_shared_cluster_global(
            smem_new,
            src_g + (first + i * stride) * BYTES_PER_COPY,
            Int32(BYTES_PER_COPY),
            mbar_new[].unsafe_ptr(),
        )

        var smem_drain = _smem_ptr[BYTES_PER_COPY, S](smem_base, w, drain_slot)
        var mbar_drain = _mbar_ref[S](mbar_base, w, drain_slot)
        mbar_drain[].wait(phase=UInt32((drain_idx // S) & 1))

        cp_async_bulk_global_shared_cta(
            dst_g + (first + drain_idx * stride) * BYTES_PER_COPY,
            smem_drain,
            Int32(BYTES_PER_COPY),
        )
        cp_async_bulk_commit_group()

    # --- Epilogue: drain remaining prologue slots ---
    for j in range(prologue):
        var drain_idx = my_total - prologue + j
        var drain_slot = drain_idx % S

        var smem_drain = _smem_ptr[BYTES_PER_COPY, S](smem_base, w, drain_slot)
        var mbar_drain = _mbar_ref[S](mbar_base, w, drain_slot)
        mbar_drain[].wait(phase=UInt32((drain_idx // S) & 1))

        cp_async_bulk_global_shared_cta(
            dst_g + (first + drain_idx * stride) * BYTES_PER_COPY,
            smem_drain,
            Int32(BYTES_PER_COPY),
        )
        cp_async_bulk_commit_group()

    cp_async_bulk_wait_group[0]()


def main() raises:
    comptime NUM_THREADS = get_defined_int["NUM_THREADS", 128]()
    comptime BYTES_PER_COPY = get_defined_int["BYTES_PER_COPY", 1024]()
    comptime S = get_defined_int["PIPELINE_NUM_STAGES", 1]()
    comptime PREFETCH = get_defined_bool["PREFETCH", False]()
    comptime NUM_WARPS = NUM_THREADS // 32

    var grid = arg_parse("grid_dim", 128)
    var total_bytes = arg_parse("total_bytes", 256 * 1024 * 1024)

    debug_assert(
        total_bytes % BYTES_PER_COPY == 0,
        "total_bytes must be divisible by BYTES_PER_COPY",
    )
    var total_chunks = total_bytes // BYTES_PER_COPY

    comptime smem_bytes = (
        NUM_WARPS * S * BYTES_PER_COPY
        + NUM_WARPS * S * size_of[SharedMemBarrier]()
    )

    var m = Bench()

    with DeviceContext() as ctx:
        var src_dev = ctx.enqueue_create_buffer[.uint8](total_bytes)
        var dst_dev = ctx.enqueue_create_buffer[.uint8](total_bytes)

        @always_inline
        def bench_func(mut b: Bencher) {imm}:
            @always_inline
            def kernel_launch(ctx: DeviceContext) raises {imm}:
                ctx.enqueue_function[
                    bulk_memcpy_kernel[NUM_THREADS, BYTES_PER_COPY, S, PREFETCH]
                ](
                    src_dev,
                    dst_dev,
                    Int32(total_chunks),
                    grid_dim=(grid,),
                    block_dim=(NUM_THREADS,),
                    shared_mem_bytes=smem_bytes,
                )

            bencher_iter_custom(b, kernel_launch, ctx)

        m.bench_function(
            bench_func,
            BenchId(
                "cp_async_bulk",
                input_id=String(
                    "threads=",
                    NUM_THREADS,
                    "/bytes_per_copy=",
                    BYTES_PER_COPY,
                    "/stages=",
                    S,
                    "/prefetch=",
                    PREFETCH,
                    "/grid=",
                    grid,
                    "/total=",
                    total_bytes,
                ),
            ),
            # A factor of 2 accounts for the read and write.
            [ThroughputMeasure(BenchMetric.bytes, total_bytes * 2)],
        )

        ctx.synchronize()
        _ = src_dev
        _ = dst_dev

    m.dump_report()
