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
"""Fused one-shot Lamport allreduce + RMSNorm (high-perf protocol).

Grafts the barrier-free Lamport comm protocol from `comm/lamport.mojo` +
`comm/allreduce.mojo` (in-register local seed, poll-all-peers arrival, 3-way
generation rotation, fused clear, the device-resident `Signal.lamport_region` /
`lamport_state`) onto a row-blocked RMSNorm epilogue so the whole TP
allreduce+RMSNorm producer is a single kernel.

Layout: each block owns whole rows (tokens). With the 128-bit Lamport pack
(`atomic_width` = 16B / sizeof(dtype) = 8 for bf16) one thread owns exactly one
pack = one column group, so a row's `cols/atomic_width` packs map to that many
threads in one block. After the allreduce reduces each pack in-register, the
block reduces the row's sum-of-squares (`block.sum`) and normalizes.

PDL is on by default: wait for the producer, then trigger the consumer
immediately (early trigger) so a following GEMM overlaps this kernel; consumer
correctness is its own `wait_on_dependent_grids`. The signal buffers must be
sentinel-initialized once (`lamport_init`) before the first call.
"""

from std.math import align_down, rsqrt
from std.sys import align_of, size_of

from std.atomic import Atomic
from std.collections import Array

from max.gpu import WARP_SIZE, block_idx, grid_dim, thread_idx
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext, get_gpu_target
from max.gpu.primitives import block
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    pdl_launch_attributes,
    wait_on_dependent_grids,
)
from std.utils.numerics import get_accum_type

from .sync import MAX_GPUS, MAX_NUM_BLOCKS_UPPER_BOUND, Signal, circular_add
from .lamport import (
    Lamport,
    LamportGeneration,
    has_neg_zero,
    remove_neg_zero,
    set_neg_zero,
)


@__name(t"allreduce_lamport_rmsnorm_{dtype}_{ngpus}")
def _allreduce_lamport_rmsnorm_kernel[
    dtype: DType,
    ngpus: Int,
    BLOCK_SIZE: Int,
    *,
    pdl: Bool = True,
    early_launch: Bool = True,
](
    result: MutPointer[Scalar[dtype], MutAnyOrigin],
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    rows_dev: Int32,
    cols_dev: Int32,
    epsilon: Scalar[dtype],
    my_rank_dev: Int32,
):
    """Row-blocked fused Lamport allreduce + RMSNorm; one 128-bit pack per thread.
    """
    var rows = Int(rows_dev)
    var cols = Int(cols_dev)
    var my_rank = Int(my_rank_dev)
    comptime accum_type = get_accum_type[dtype]()
    # Pin to the 128-bit single-copy-atomic pack the sentinel protocol needs.
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    comptime assert (
        atomic_width * size_of[dtype]() == 16
    ), "Lamport pack must be exactly 16 bytes (the 128-bit atomic width)."
    comptime alignment = align_of[SIMD[dtype, atomic_width]]()

    var tid = Int(thread_idx.x)
    var col = tid * atomic_width
    var is_valid = col < cols
    var packs_per_row = cols // atomic_width

    # PDL: wait for the producer, then (if early_launch) release the consumer
    # immediately so it overlaps the whole AR+RMSNorm. Called by all threads in
    # the block before any divergence. With `early_launch=False` the trigger is
    # instead moved to the END of the kernel (after the output is written), so
    # the consumer overlaps only the tail bookkeeping + grid teardown -- partial
    # overlap rather than full.
    comptime if pdl:
        wait_on_dependent_grids()
        comptime if early_launch:
            launch_dependent_grids()

    # This rank's own region (polled) + peer regions in round-robin order.
    var my_region = rank_sigs[my_rank][].lamport_region_ptr[dtype]()
    comptime PtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var peer_regions = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> PtrType: rank_sigs[
            circular_add[ngpus](my_rank, i)
        ][].lamport_region_ptr[dtype]()
    )
    var sentinel = set_neg_zero[dtype, atomic_width]()

    # Generation geometry (read the device-resident counter once per call).
    var state = rank_sigs[my_rank][].lamport_state_ptr()
    var flag = Int(state.load[width=1, volatile=True](Lamport.STATE_FLAG))
    var clear_size = Int(
        state.load[width=1, volatile=True](Lamport.STATE_PREV_ELEMS)
    )
    var clear_packs = clear_size // atomic_width
    comptime gen_stride_packs = ngpus * Lamport.MAX_PACKS
    var data_gen_off = LamportGeneration.data_index(flag) * gen_stride_packs
    var clear_gen_off = LamportGeneration.clear_index(flag) * gen_stride_packs

    # Preload this thread's gamma columns once (reused across all rows).
    var gamma_vec = SIMD[accum_type, atomic_width](0)
    if is_valid:
        gamma_vec = gamma.load[width=atomic_width, alignment=alignment](
            col
        ).cast[accum_type]()

    for row in range(Int(block_idx.x), rows, Int(grid_dim.x)):
        var base = row * cols + col
        var global_pack = row * packs_per_row + tid

        var accum = SIMD[accum_type, atomic_width](0)
        if is_valid:
            # (a) Local input pack -> producer-sanitize; kept in-register as the
            # reduction seed, so this rank only touches the ngpus-1 remote slots.
            var local = remove_neg_zero[dtype, atomic_width](
                src.load[width=atomic_width, alignment=alignment](base)
            )

            # (b) Push the sanitized pack into each PEER's slot (skip self).
            var push_off = (
                data_gen_off + my_rank * Lamport.MAX_PACKS + global_pack
            )
            comptime for i in range(1, ngpus):
                peer_regions[i].store[
                    width=atomic_width, alignment=alignment, volatile=True
                ](push_off * atomic_width, local)

            # (d) Poll all ngpus-1 remote slots until none hold the sentinel
            # (observes parallel arrival rather than per-peer spinning).
            var peer_packs = Array[SIMD[dtype, atomic_width], ngpus](
                uninitialized=True
            )
            var done = False
            while not done:
                done = True
                comptime for i in range(1, ngpus):
                    var peer = circular_add[ngpus](my_rank, i)
                    var slot_off = (
                        data_gen_off + peer * Lamport.MAX_PACKS + global_pack
                    )
                    var p = my_region.load[
                        width=atomic_width, alignment=alignment, volatile=True
                    ](slot_off * atomic_width)
                    peer_packs[i] = p
                    if has_neg_zero[dtype, atomic_width](p):
                        done = False

            # (e) Reduce: in-register local seed + arrived peers, in accum_type.
            accum = local.cast[accum_type]()
            comptime for i in range(1, ngpus):
                accum += peer_packs[i].cast[accum_type]()

        # RMSNorm over the row (block reduction; invalid threads add 0).
        var thread_sumsq = (accum * accum).reduce_add()
        var row_sumsq = block.sum[block_size=BLOCK_SIZE, broadcast=True](
            thread_sumsq
        )
        var norm = rsqrt(
            row_sumsq / Scalar[accum_type](cols) + epsilon.cast[accum_type]()
        )
        if is_valid:
            var normalized = (accum * norm) * gamma_vec
            result.store[width=atomic_width, alignment=alignment](
                base, normalized.cast[dtype]()
            )

        # (f) Fused clear: reset this pack's remote slots in the generation
        # reused two calls from now, within the previous writer's extent.
        if global_pack < clear_packs:
            comptime for i in range(1, ngpus):
                var peer = circular_add[ngpus](my_rank, i)
                var clr_off = (
                    clear_gen_off + peer * Lamport.MAX_PACKS + global_pack
                )
                my_region.store[
                    width=atomic_width, alignment=alignment, volatile=True
                ](clr_off * atomic_width, sentinel)

    # When the early trigger is disabled, release the consumer HERE instead --
    # after this block's `result` writes are done, but before the tail
    # bookkeeping (leftover clear + generation advance) and grid teardown. The
    # next GEMM thus still overlaps that tail rather than waiting for full grid
    # completion (partial overlap). launch_dependent_grids makes this block's
    # prior writes visible to the dependent grid, so triggering after the output
    # store is correct (strictly safer than the start trigger, which fires before
    # any output is written).
    comptime if pdl and not early_launch:
        launch_dependent_grids()

    # (f-tail) If the previous call was LARGER, clear the leftover packs of the
    # reused generation here (empty in the common same-size case).
    var cur_packs = rows * packs_per_row
    var gtid = Int(block_idx.x) * BLOCK_SIZE + tid
    var nthreads = Int(grid_dim.x) * BLOCK_SIZE
    for cp in range(cur_packs + gtid, clear_packs, nthreads):
        comptime for i in range(1, ngpus):
            var peer = circular_add[ngpus](my_rank, i)
            var clr_off = clear_gen_off + peer * Lamport.MAX_PACKS + cp
            my_region.store[
                width=atomic_width, alignment=alignment, volatile=True
            ](clr_off * atomic_width, sentinel)

    # (g) Advance this rank's generation flag exactly once per call via an
    # intra-GPU block-arrival counter (device-side state -> CUDA-graph-safe).
    barrier()
    if tid == 0:
        var arrived = Atomic.fetch_add(state + Lamport.STATE_ARRIVAL, UInt32(1))
        if Int(arrived) == Int(grid_dim.x) - 1:
            (state + Lamport.STATE_FLAG).store[volatile=True](UInt32(flag + 1))
            (state + Lamport.STATE_PREV_ELEMS).store[volatile=True](
                UInt32(rows * cols)
            )
            (state + Lamport.STATE_ARRIVAL).store[volatile=True](UInt32(0))


def lamport_allreduce_rmsnorm[
    dtype: DType,
    ngpus: Int,
    *,
    pdl: Bool = True,
    early_launch: Bool = True,
](
    my_rank: Int,
    src: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    dst: MutPointer[Scalar[dtype], MutAnyOrigin],
    gamma: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    rows: Int,
    cols: Int,
    epsilon: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """Per-device fused Lamport allreduce (sum) + RMSNorm (high-perf protocol).

    `src`/`dst` are `[rows, cols]` (tokens x hidden); RMSNorm runs over `cols`.
    Uses the device-resident `Signal.lamport_region` / `lamport_state` (so the
    signal buffers must be `lamport_init`-ed once before the first call, then one
    host sync). The generation flag advances in-kernel, so repeated calls --
    including CUDA-graph replay -- rotate the buffers correctly with no per-call
    fill or phase argument.

    Parameters:
        dtype: Element type (bf16/fp16/fp32 transport).
        ngpus: Number of participating GPUs.
        pdl: If True (default) launch with PDL and early-trigger the consumer.
        early_launch: If True (default) and `pdl`, call launch_dependent_grids()
            at kernel start so the consumer overlaps the whole kernel. If False,
            the trigger moves to the END (after the output is written), so the
            consumer overlaps only the tail bookkeeping + grid teardown.
    """
    comptime assert ngpus >= 2 and ngpus <= MAX_GPUS, "ngpus must be in [2, 8]"
    comptime assert (
        dtype.is_floating_point()
    ), "lamport_allreduce_rmsnorm requires a floating-point dtype"
    comptime atomic_width = Lamport.ATOMIC_BYTES // size_of[dtype]()
    comptime max_tpb = ctx.default_device_info.max_thread_block_size
    comptime BLOCK_SIZE = align_down(max_tpb, WARP_SIZE)

    if cols % atomic_width != 0:
        raise Error(
            "lamport_allreduce_rmsnorm requires cols % atomic_width == 0 (whole"
            " 128-bit packs)"
        )
    if cols // atomic_width > BLOCK_SIZE:
        raise Error(
            "lamport_allreduce_rmsnorm: cols too large for one pack/thread"
        )
    var num_bytes = rows * cols * size_of[dtype]()
    if num_bytes > Lamport.MAX_SMALL_MESSAGE_BYTES:
        raise Error(
            "lamport_allreduce_rmsnorm message exceeds the reserved Lamport"
            " workspace"
        )

    var grid = min(MAX_NUM_BLOCKS_UPPER_BOUND, rows)
    comptime pdl_level = PDLLevel.ON if pdl else PDLLevel.OFF
    comptime kernel = _allreduce_lamport_rmsnorm_kernel[
        dtype, ngpus, BLOCK_SIZE, pdl=pdl, early_launch=early_launch
    ]
    ctx.enqueue_function[kernel](
        dst,
        src,
        gamma,
        rank_sigs,
        Int32(rows),
        Int32(cols),
        epsilon,
        Int32(my_rank),
        grid_dim=grid,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(pdl_level),
    )
