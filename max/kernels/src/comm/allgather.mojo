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
"""Multi-GPU allgather implementation that gathers values from multiple GPUs
into an output buffer.

This module provides an optimized implementation of allgather operations across
multiple GPUs, supporting both peer-to-peer (P2P) and non-P2P communication
patterns. The implementation automatically selects between approaches based on
hardware capabilities:

1. P2P-based implementation (when P2P access is available):
   - Uses direct GPU-to-GPU memory access for better performance.
   - Optimized for NVLink and xGMI bandwidth utilization.
   - Uses vectorized memory access.

2. Non-P2P fallback implementation:
   - Copies data through device memory when direct GPU access isn't possible.
   - Simple but functional approach for systems without P2P support.
"""

from std.collections import Array
from std.math.uutils import ualign_down
from std.math import ceildiv
from std.sys import simd_width_of, align_of, size_of

from layout import TensorEngine, TileTensor
from layout.tile_layout import TensorLayout
from layout.tma_async import SharedMemBarrier
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    global_idx,
    grid_dim,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.memory import (
    cp_async_bulk_global_shared_cta,
    cp_async_bulk_shared_cluster_global,
    external_memory,
    fence_mbarrier_init,
)
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from max.gpu.host import DeviceBuffer, DeviceContext, get_gpu_target
from max.gpu.host.info import _is_sm10x_gpu

from std.utils import StaticTuple

from .device_query import (
    DefaultCommTuningConfig,
    GB,
    KB,
    dispatch_select_comm_config,
)
from .relay import (
    RELAY_ARCH,
    RelayTuningConfig,
    _relay_pairs,
    _relay_slice_vectors,
)
from internal_utils import Table

from .reducescatter import _target_address_space
from .sync import (
    MAX_GPUS,
    Signal,
    _multi_gpu_barrier,
    is_p2p_enabled,
    circular_add,
)


# Tuning table to get num_blocks for allgather.
# Arch-specific defaults use ngpus=-1, num_bytes=-1 with the arch's sm_version.
# The global default (sm_version="default") is the ultimate fallback for
# unknown architectures -- dispatch_select_comm_config prefers arch-specific
# defaults when available.
comptime allgather_tuning_table = Table(
    [
        # default for sm90 (encoded with ngpus=-1, num_bytes=-1)
        DefaultCommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="sm_90a", num_blocks=216
        ),
        # default for sm100 (encoded with ngpus=-1, num_bytes=-1)
        DefaultCommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="sm_100a", num_blocks=512
        ),
        # default for sm103 (B300, encoded with ngpus=-1, num_bytes=-1)
        DefaultCommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="sm_103a", num_blocks=512
        ),
        # default for CDNA4 (MI355X, encoded with ngpus=-1, num_bytes=-1).
        # Interleaved peer copy saturates the PCIe fabric at ~128 blocks
        # (measured peak, TP4); more only adds barrier/scheduling overhead.
        DefaultCommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="CDNA4", num_blocks=128
        ),
        # global default for unknown architectures
        DefaultCommTuningConfig(
            ngpus=-1, num_bytes=-1, sm_version="default", num_blocks=512
        ),
    ],
    "allgather_table",
)


comptime allgather_relay_tuning_table = Table(
    [
        # Default for group widths with no rows of their own. These are the
        # measured optima at a group width of 4, which is also the shape this
        # kernel exists for.
        RelayTuningConfig(
            group_size=-1,
            num_bytes=-1,
            num_blocks=72,
            num_relay_blocks=8,
            relay_percent=46,
        ),
        # Small shards are barrier-bound rather than link-bound, so the extra
        # links buy less than the wider barrier costs: measured on MI355X the
        # relay path ties the plain one at a 256 KB shard and loses about 1.4x
        # at 128 KB. A zero share declines the relay path outright. Above the
        # cutoff a width falls through to the default unless it has a row of
        # its own, which is why only width 2 needs one.
        RelayTuningConfig(
            group_size=4,
            num_bytes=(256 * KB),
            num_blocks=0,
            num_relay_blocks=0,
            relay_percent=0,
        ),
        RelayTuningConfig(
            group_size=2,
            num_bytes=(256 * KB),
            num_blocks=0,
            num_relay_blocks=0,
            relay_percent=0,
        ),
        # A group width of 2 gives every relay a fan-out of one, so it is pure
        # store-and-forward with no multicast amplification to pay for wide
        # teams. Leaner blocks and a smaller relayed share are worth up to 11%
        # over the default at mid sizes.
        RelayTuningConfig(
            group_size=2,
            num_bytes=(2 * GB),
            num_blocks=32,
            num_relay_blocks=4,
            relay_percent=38,
        ),
    ],
    "allgather_relay_table",
)


@inline(.always)
def _allgather_naive[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    in_engine: TensorEngine,
    out_engine: TensorEngine,
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin, Engine=out_engine],
        ngpus,
    ],
    ctx: DeviceContext,
) raises:
    """Per-device allgather fallback when P2P access is not available.

    One instance runs per GPU. Each instance copies data from all GPUs
    into its own output buffers using device-to-device memory copies.
    """
    var device_buffers = List[DeviceBuffer[dtype]](capacity=ngpus)

    for i in range(ngpus):
        var rctx = DeviceContext(device_id=i)
        device_buffers.append(
            DeviceBuffer(
                rctx,
                rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
                    input_buffers[i].ptr
                ),
                input_buffers[i].num_elements(),
                owning=False,
            )
        )

    for input_idx in range(ngpus):
        var output_device_buffer = DeviceBuffer(
            ctx,
            rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
                output_buffers[input_idx].ptr
            ),
            output_buffers[input_idx].num_elements(),
            owning=False,
        )

        ctx.enqueue_copy(
            output_device_buffer,
            device_buffers[input_idx],
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allgather_p2p_{dtype}")
def _allgather_p2p_kernel[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    domain_id: Int = 0,
](
    outputs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    src_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    lengths: StaticTuple[Int32, ngpus],
    max_num_blocks: Int32,
    my_rank: Int32,
):
    """P2P kernel for allgather operation.

    Each GPU directly reads from all other GPUs and writes to its output buffers.
    Uses round-robin access pattern to balance NVLink traffic.
    """
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var global_tid = global_idx.x
    var stride = grid_dim.x * BLOCK_SIZE
    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]

    comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
    var src_ptrs_rr = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> SrcPtrType: src_ptrs[
            circular_add[ngpus](_my_rank, i)
        ]
    )

    comptime OutPtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
    var out_ptrs_rr = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> OutPtrType: outputs[
            circular_add[ngpus](_my_rank, i)
        ]
    )
    var lengths_rr = Array[_, ngpus](
        fill_with_unrolled=lambda [i: Int]() -> Int: Int(
            lengths[circular_add[ngpus](_my_rank, i)]
        )
    )

    with PDL():
        # Synchronize before reading.
        _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )

        # Copy each source GPU's data to its output buffer (outputs[i] holds
        # GPU i). Peer copies are interleaved (all `ngpus` loads issued before
        # any store) for `ngpus`-way memory-level parallelism.
        var num_simd_vectors = Array[Int, ngpus](uninitialized=True)
        var max_num_simd_vectors = 0
        comptime for gpu_idx in range(ngpus):
            var nsv = lengths_rr[gpu_idx] // simd_width
            num_simd_vectors[gpu_idx] = nsv
            max_num_simd_vectors = max(max_num_simd_vectors, nsv)

        # Grid-strided loop over the longest source; per-peer guards skip
        # shorter sources.
        for idx in range(global_tid, max_num_simd_vectors, stride):
            var elem_idx = idx * simd_width
            var data = Array[SIMD[dtype, simd_width], ngpus](uninitialized=True)
            # Issue all peer reads first (memory-level parallelism).
            comptime for gpu_idx in range(ngpus):
                if idx < num_simd_vectors[gpu_idx]:
                    data[gpu_idx] = (
                        src_ptrs_rr[gpu_idx]
                        .address_space_cast[_target_address_space]()
                        .load[width=simd_width, alignment=alignment](elem_idx)
                    )
            # Then store each peer's data.
            comptime for gpu_idx in range(ngpus):
                if idx < num_simd_vectors[gpu_idx]:
                    out_ptrs_rr[gpu_idx].address_space_cast[
                        _target_address_space
                    ]().store[width=simd_width, alignment=alignment](
                        elem_idx, data[gpu_idx]
                    )

        # Scalar remainder per source.
        comptime for gpu_idx in range(ngpus):
            var nsv = num_simd_vectors[gpu_idx]
            var remainder = lengths_rr[gpu_idx] - nsv * simd_width
            if remainder > 0:
                var tail_start = nsv * simd_width
                # Use first warp to handle tail to minimize divergence.
                if global_tid < WARP_SIZE:
                    for i in range(global_tid, remainder, WARP_SIZE):
                        var elem_idx = tail_start + i
                        out_ptrs_rr[gpu_idx][elem_idx] = src_ptrs_rr[gpu_idx][
                            elem_idx
                        ]

        # Synchronize after writing.
        _multi_gpu_barrier[ngpus, is_start=False, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allgather_p2p_tma_{dtype}")
def _allgather_tma_kernel[
    dtype: DType,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    BYTES_PER_COPY: Int,
    domain_id: Int = 0,
](
    outputs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    src_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    lengths: StaticTuple[Int32, ngpus],
    my_rank: Int32,
):
    """Allgather using cp.async.bulk TMA instructions.

    Each warp is assigned to one source GPU (warp % ngpus). The warp leader
    copies its source in BYTES_PER_COPY chunks: async g2s, wait, async s2g,
    wait. Multiple blocks distribute chunks across warps via grid-strided
    indexing. The last chunk uses the remaining byte count (<= BYTES_PER_COPY).

    Shared memory layout per warp: one BYTES_PER_COPY data slot + one mbar.
    """
    comptime NUM_WARPS = BLOCK_SIZE // WARP_SIZE

    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]

    var smem_base = external_memory[
        UInt8, address_space=.SHARED, alignment=128
    ]()
    var mbar_base = (smem_base + NUM_WARPS * BYTES_PER_COPY).bitcast[
        SharedMemBarrier
    ]()

    var warp = warp_id()
    var is_leader = elect_one_sync()

    if is_leader:
        mbar_base[warp].init()
    fence_mbarrier_init()
    barrier()

    # Warp-to-source mapping.
    var my_src_idx = warp % ngpus

    var src_g = (
        src_ptrs[my_src_idx].bitcast[UInt8]().address_space_cast[.GLOBAL]()
    )
    var dst_g = (
        outputs[my_src_idx].bitcast[UInt8]().address_space_cast[.GLOBAL]()
    )
    var nbytes = Int(lengths[my_src_idx]) * size_of[dtype]()
    var smem = smem_base + warp * BYTES_PER_COPY
    var mbar = mbar_base + warp

    # Grid-strided chunk distribution across warps handling the same source.
    var warps_per_src_per_block = NUM_WARPS // ngpus
    var src_local_warp = warp // ngpus
    var first = Int(block_idx.x) * warps_per_src_per_block + src_local_warp
    var warp_stride = Int(grid_dim.x) * warps_per_src_per_block

    var total_chunks = ceildiv(nbytes, BYTES_PER_COPY)

    with PDL():
        _multi_gpu_barrier[ngpus, is_start=True, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )

        if is_leader:
            var phase = UInt32(0)
            for chunk_idx in range(first, total_chunks, warp_stride):
                var offset = chunk_idx * BYTES_PER_COPY
                var copy_bytes = min(BYTES_PER_COPY, nbytes - offset)

                # Async NVLink read: global → shared.
                mbar[].expect_bytes(Int32(copy_bytes))
                cp_async_bulk_shared_cluster_global(
                    smem, src_g + offset, Int32(copy_bytes), mbar[].unsafe_ptr()
                )
                mbar[].wait(phase=phase)
                phase ^= 1

                # Async local write: shared → global.
                cp_async_bulk_global_shared_cta(
                    dst_g + offset, smem, Int32(copy_bytes)
                )
                cp_async_bulk_commit_group()
                cp_async_bulk_wait_group[0]()

        _multi_gpu_barrier[ngpus, is_start=False, domain_id=domain_id](
            rank_sigs, my_sig, _my_rank
        )


@inline(.always)
def _allgather_p2p_tma[
    dtype: DType,
    ngpus: Int,
    domain_id: Int = 0,
](
    output_ptrs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    list_of_in_ptrs: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    lengths: StaticTuple[Int32, ngpus],
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """P2P kernel for allgather operation using cp.async.bulk TMA instructions.
    """
    comptime TMA_BLOCK_SIZE = 256
    comptime TMA_BYTES_PER_COPY = 16384
    comptime NUM_WARPS = TMA_BLOCK_SIZE // WARP_SIZE
    comptime tma_smem = (
        NUM_WARPS * TMA_BYTES_PER_COPY + NUM_WARPS * size_of[SharedMemBarrier]()
    )
    comptime warps_per_src_per_block = NUM_WARPS // ngpus
    comptime assert (
        warps_per_src_per_block > 0
    ), "warps_per_src_per_block must be greater than 0"

    var max_length = 0
    for i in range(ngpus):
        max_length = max(max_length, Int(lengths[i]))

    # Dynamic grid: at least 1 block, scale with data volume,
    var total_chunks = ceildiv(
        max_length * size_of[dtype](), TMA_BYTES_PER_COPY
    )
    var tma_grid = min(
        32,  # 32 CTAs are more than enough to saturate the NVLink.
        max(1, ceildiv(total_chunks, warps_per_src_per_block)),
    )

    comptime tma_kernel = _allgather_tma_kernel[
        dtype,
        ngpus,
        BLOCK_SIZE=TMA_BLOCK_SIZE,
        BYTES_PER_COPY=TMA_BYTES_PER_COPY,
        domain_id=domain_id,
    ]
    ctx.enqueue_function[tma_kernel](
        output_ptrs,
        list_of_in_ptrs,
        rank_sigs,
        lengths,
        Int32(my_rank),
        grid_dim=tma_grid,
        block_dim=TMA_BLOCK_SIZE,
        shared_mem_bytes=tma_smem,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )
    return


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"allgather_relay_{dtype}")
def _allgather_relay_kernel[
    dtype: DType,
    ngpus: Int,
    *,
    BLOCK_SIZE: Int,
    domain_id: Int = 0,
](
    outputs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    src_ptrs: StaticTuple[ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus],
    peer_src_ptrs: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    peer_out_ptrs: StaticTuple[
        MutPointer[Scalar[dtype], MutAnyOrigin], ngpus * ngpus
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    lengths: StaticTuple[Int32, ngpus],
    peer_lengths: StaticTuple[Int32, ngpus],
    relay_percent: Int32,
    num_direct_blocks: Int32,
    my_rank: Int32,
):
    """Relay-assisted P2P kernel for a grouped allgather.

    Blocks below `num_direct_blocks` take the direct role and the rest take the
    relay role; every block joins both barriers, since the barrier pairs blocks
    by id across GPUs and an early return would hang the node.
    """
    comptime world_size = 2 * ngpus
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var _my_rank = Int(my_rank)
    var my_sig = rank_sigs[_my_rank]
    # Groups occupy contiguous rank ranges, so the group-local rank is just the
    # world rank folded by the group width.
    var group_rank = _my_rank % ngpus

    var bid = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var _relay_percent = Int(relay_percent)

    # Synchronize before reading. The domain spans both groups because a
    # relay reads and writes GPUs outside its own group.
    _multi_gpu_barrier[world_size, is_start=True, domain_id=domain_id](
        rank_sigs, my_sig, _my_rank
    )

    if bid < Int(num_direct_blocks):
        # Direct role: one stream per group source. Stream 0 copies my own
        # shard, which is never relayed, in full; the other streams pull
        # the leading `num_direct_vectors` of a peer's shard over the
        # intra-group link. Blocks of a stream split its range contiguously
        # so each link carries one long sequential stream.
        var stream = bid % ngpus
        var blocks_per_stream = Int(num_direct_blocks) // ngpus
        var stream_block = bid // ngpus
        var src_idx = circular_add[ngpus](group_rank, stream)
        var src_ptr = src_ptrs[src_idx].address_space_cast[
            _target_address_space
        ]()
        var out_ptr = outputs[src_idx].address_space_cast[
            _target_address_space
        ]()
        # Each shard is split on its own length, so the group may be
        # ragged. The self-copy owns its whole shard; a peer pull stops
        # where that peer's relayed slices begin.
        var length = Int(lengths[src_idx])
        var num_simd_vectors = length // simd_width
        var span = num_simd_vectors
        if stream != 0:
            span -= (
                _relay_slice_vectors[ngpus](num_simd_vectors, _relay_percent)
                * ngpus
            )

        for idx in range(
            stream_block * span // blocks_per_stream + tid,
            (stream_block + 1) * span // blocks_per_stream,
            BLOCK_SIZE,
        ):
            var elem_idx = idx * simd_width
            out_ptr.store[width=simd_width, alignment=alignment](
                elem_idx,
                src_ptr.load[
                    width=simd_width, alignment=alignment, invariant=True
                ](elem_idx),
            )

        # Scalar remainder. It sits past the last whole vector, so it is
        # outside every relayed slice and the direct streams own it. One
        # block per stream handles it to minimize divergence.
        if stream_block == 0 and tid < WARP_SIZE:
            for i in range(
                num_simd_vectors * simd_width + tid, length, WARP_SIZE
            ):
                out_ptr[i] = src_ptr[i]
    else:
        # Relay role: forward the trailing slices of the *other* group's
        # shards over links that a grouped collective leaves idle. This GPU
        # ingests slice `group_rank` of one peer-group shard once, then
        # remote-writes it into the `ngpus - 1` peer-group GPUs that still
        # need it, so one inbound link feeds `ngpus - 1` outbound ones.
        #
        # The forwarded bytes are final data written at their final
        # address, so no scratch buffer, staging copy or release/acquire
        # chain is needed: the paired end barrier already orders the relay
        # stores before any consumer reads the output.
        var relay_bid = bid - Int(num_direct_blocks)
        var src_idx = relay_bid % ngpus
        var blocks_per_src = (Int(grid_dim.x) - Int(num_direct_blocks)) // ngpus
        var src_block = relay_bid // ngpus
        var src_ptr = peer_src_ptrs[src_idx].address_space_cast[
            _target_address_space
        ]()

        # Hoist this source's destination pointers. Indexing the flat
        # `peer_out_ptrs` table by a runtime source inside the loop makes
        # the compiler stage the whole table through scratch memory; a
        # comptime-indexed Array of the `ngpus - 1` live destinations stays
        # in registers and drops the skip-self test from the inner loop.
        comptime OutPtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
        var dst_ptrs = Array[_, ngpus - 1](
            fill_with_unrolled=lambda [k: Int]() -> OutPtrType: peer_out_ptrs[
                circular_add[ngpus](src_idx, k + 1) * ngpus + src_idx
            ]
        )

        # The relayed shard is split on its own length too, so a peer
        # group with a short shard simply leaves its relays with less to
        # do; a shard too small to split gives an empty range.
        var peer_simd_vectors = Int(peer_lengths[src_idx]) // simd_width
        var slice_vectors = _relay_slice_vectors[ngpus](
            peer_simd_vectors, _relay_percent
        )
        var slice_start = (
            peer_simd_vectors - slice_vectors * ngpus
        ) + group_rank * slice_vectors
        var start = slice_start + src_block * slice_vectors // blocks_per_src
        var end = (
            slice_start + (src_block + 1) * slice_vectors // blocks_per_src
        )

        # Inbound loads a thread batches before issuing its outbound
        # stores: the read link and the `ngpus - 1` write links only
        # overlap while several remote loads -- which are latency-bound per
        # thread -- are in flight. Four beat one, two and eight on MI355X.
        comptime UNROLL = 4

        for idx in range(start + tid, end, BLOCK_SIZE * UNROLL):
            var data = Array[SIMD[dtype, simd_width], UNROLL](
                uninitialized=True
            )
            comptime for u in range(UNROLL):
                if idx + u * BLOCK_SIZE < end:
                    data[u] = src_ptr.load[
                        width=simd_width,
                        alignment=alignment,
                        invariant=True,
                    ]((idx + u * BLOCK_SIZE) * simd_width)

            comptime for u in range(UNROLL):
                var elem_idx = (idx + u * BLOCK_SIZE) * simd_width
                if idx + u * BLOCK_SIZE < end:
                    comptime for k in range(ngpus - 1):
                        dst_ptrs[k].address_space_cast[
                            _target_address_space
                        ]().store[width=simd_width, alignment=alignment](
                            elem_idx, data[u]
                        )

    # Synchronize after writing. This is also what publishes the relay
    # stores to their destination GPUs.
    _multi_gpu_barrier[world_size, is_start=False, domain_id=domain_id](
        rank_sigs, my_sig, _my_rank
    )


@inline(.always)
def _allgather_p2p_relay[
    dtype: DType,
    ngpus: Int,
](
    output_ptrs: StaticTuple[MutPointer[Scalar[dtype], MutAnyOrigin], ngpus],
    list_of_in_ptrs: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    lengths: StaticTuple[Int32, ngpus],
    peer_input_ptrs: StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ],
    peer_lengths: StaticTuple[Int32, ngpus],
    peer_output_ptrs: StaticTuple[
        MutPointer[Scalar[dtype], MutAnyOrigin], ngpus * ngpus
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    recipe: RelayTuningConfig,
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Per-device allgather over one of two groups, assisted by the other.

    Two groups of `ngpus` GPUs running a grouped allgather concurrently use
    only their intra-group links; every link between the groups sits idle. This
    path routes a trailing fraction of every shard over those idle links: a GPU
    in the other group ingests a slice of a shard once and multicasts it to the
    `ngpus - 1` GPUs that need it. Direct and relay links then carry about half
    a shard each instead of a full shard on the direct links alone, which
    roughly halves the topology floor.

    Each shard is split on its own length, so the groups may be ragged; what
    all `2 * ngpus` GPUs must agree on is the recipe and the block counts,
    since they run one kernel and one barrier domain and the barrier pairs
    blocks by id. Both counts are therefore derived from the whole relay
    world's shapes, which every rank sees. A node running more than two groups
    pairs them up, and each pair is its own relay world.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs in one group; the relay world holds `2 * ngpus`.

    Args:
        output_ptrs: This GPU's output buffers, by group-local source rank.
        list_of_in_ptrs: This group's input shards, by group-local rank.
        lengths: Elements in each of this group's shards.
        peer_input_ptrs: The other group's input shards, by that group's local
            rank. This GPU reads them when it acts as a relay.
        peer_lengths: Elements in each of those shards, which is what tells a
            relay how much of each to forward.
        peer_output_ptrs: The other group's output buffers, with
            `peer_output_ptrs[d * ngpus + s]` pointing at the buffer on that
            group's rank `d` that receives its rank `s`. This GPU writes them
            when it acts as a relay.
        rank_sigs: Signals for the `2 * ngpus` GPUs of this relay world, packed
            in relay-world rank order.
        recipe: Block counts and relayed fraction, from
            `allgather_relay_tuning_table`.
        ctx: Device context for THIS GPU.
        my_rank: This GPU's rank within the relay world.
    """
    # The barrier bank follows the file's convention of keying domains by
    # collective width, and this barrier is `2 * ngpus` wide -- NOT the width
    # the caller's own `domain_id` describes. Reusing that one would let an
    # `ngpus`-wide grouped barrier and this one advance the same counters at
    # different rates and hang the node.
    comptime relay_domain_id = 0 if 2 * ngpus == MAX_GPUS else 2 * ngpus
    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()

    # A relay block only runs if some shard in the relay world is long enough
    # to split. Both groups see both length sets, so they agree on the grid --
    # deciding from one group's shards alone would let the two launch different
    # grids and hang the barrier.
    var any_relayed = False
    comptime for i in range(ngpus):
        if (
            _relay_slice_vectors[ngpus](
                Int(lengths[i]) // simd_width, recipe.relay_percent
            )
            > 0
            or _relay_slice_vectors[ngpus](
                Int(peer_lengths[i]) // simd_width, recipe.relay_percent
            )
            > 0
        ):
            any_relayed = True

    # Both roles fan their blocks out over `ngpus` streams, so each count is
    # rounded down to a nonzero multiple of the group width.
    var num_direct_blocks = ngpus * max(1, recipe.num_blocks // ngpus)
    var num_relay_blocks = 0
    if any_relayed:
        num_relay_blocks = ngpus * max(1, recipe.num_relay_blocks // ngpus)

    comptime BLOCK_SIZE = 256
    comptime relay_kernel = _allgather_relay_kernel[
        dtype, ngpus, BLOCK_SIZE=BLOCK_SIZE, domain_id=relay_domain_id
    ]
    ctx.enqueue_function[relay_kernel](
        output_ptrs,
        list_of_in_ptrs,
        peer_input_ptrs,
        peer_output_ptrs,
        rank_sigs,
        lengths,
        peer_lengths,
        Int32(recipe.relay_percent),
        Int32(num_direct_blocks),
        Int32(my_rank),
        grid_dim=num_direct_blocks + num_relay_blocks,
        block_dim=BLOCK_SIZE,
    )


@inline(.always)
def _allgather_p2p[
    dtype: DType,
    rank: Int,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    in_engine: TensorEngine,
    out_engine: TensorEngine,
    group_size: Int = ngpus,
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin, Engine=out_engine],
        ngpus * group_size,
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    _max_num_blocks: Optional[Int],
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Per-device P2P allgather: each GPU reads from all peers directly.

    World view vs. group: `ngpus` is the total number of devices in the
    world; `input_buffers` and `rank_sigs` carry every device's data,
    indexed by GLOBAL device rank. `output_buffers` holds EVERY device's
    own `group_size` outputs, laid out `[device * group_size +
    group_local_source]`. `group_size` (defaults to `ngpus`) is the number
    of devices that actually cooperate on this all-gather; it must evenly
    divide `ngpus`. `my_rank` is this device's GLOBAL rank in `[0, ngpus)`.
    """
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide ngpus"
    comptime domain_id = 0 if group_size == ngpus else group_size

    # This device's group. `group_start` is 0 for a full-world collective, so
    # every group_start-relative read below is byte-identical to the
    # pre-grouping code path in that case.
    var group_start = ualign_down(my_rank, group_size)
    var loc_rank = my_rank - group_start

    var list_of_in_ptrs = StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], group_size
    ]()
    var lengths = StaticTuple[Int, group_size]()

    comptime for i in range(group_size):
        list_of_in_ptrs[i] = rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            input_buffers[group_start + i].ptr
        )
        lengths[i] = input_buffers[group_start + i].num_elements()

    # Prepare output pointers: this device's own slice of the world outputs.
    var output_ptrs = StaticTuple[
        MutPointer[Scalar[dtype], MutAnyOrigin], group_size
    ]()

    comptime for src_idx in range(group_size):
        output_ptrs[src_idx] = rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
            output_buffers[my_rank * group_size + src_idx].ptr
        )

    # This device's GROUP's signal pointers, re-indexed to [0, group_size).
    # Byte-identical to `rank_sigs` for a full-world collective.
    var group_sigs = Array[UnsafePointer[Signal, MutAnyOrigin], MAX_GPUS](
        uninitialized=True
    )
    comptime for i in range(group_size):
        group_sigs[i] = rank_sigs[group_start + i]

    # Build Int32 versions for passing to GPU kernels.
    var lengths_i32 = StaticTuple[Int32, group_size]()
    comptime for i in range(group_size):
        lengths_i32[i] = Int32(lengths[i])

    # Relay path: with an even number of groups, adjacent groups pair up and
    # relay for each other, so part of every shard can travel over the
    # inter-group links a grouped collective leaves idle. Gated on an arch it
    # has been measured on -- the transport is generic but the win is not --
    # on the pair fitting the barrier's rank space, and on a size the tuning
    # table thinks is worth relaying.
    comptime _use_relay = _relay_pairs[ngpus, group_size](
        ctx.default_device_info.version
    )
    comptime if _use_relay:
        # The pair this device belongs to, and the group inside it that relays
        # for this one.
        var pair_base = ualign_down(my_rank, 2 * group_size)
        var peer_start = (
            pair_base + group_size if group_start == pair_base else pair_base
        )

        comptime SrcPtrType = ImmPointer[Scalar[dtype], ImmutAnyOrigin]
        comptime OutPtrType = MutPointer[Scalar[dtype], MutAnyOrigin]
        var peer_input_ptrs = StaticTuple[SrcPtrType, group_size]()
        var peer_lengths = StaticTuple[Int32, group_size]()
        var peer_output_ptrs = StaticTuple[
            OutPtrType, group_size * group_size
        ]()

        # Every rank in the pair must reach the same verdict, so the lookup is
        # keyed on the largest shard anywhere in it rather than on this group's
        # shards, which the two groups see differently.
        var pair_max_length = 0
        comptime for i in range(group_size):
            peer_input_ptrs[i] = rebind[SrcPtrType](
                input_buffers[peer_start + i].ptr
            )
            var peer_length = input_buffers[peer_start + i].num_elements()
            peer_lengths[i] = Int32(peer_length)
            pair_max_length = max(pair_max_length, peer_length)
            pair_max_length = max(pair_max_length, lengths[i])

            comptime for src_idx in range(group_size):
                peer_output_ptrs[i * group_size + src_idx] = rebind[OutPtrType](
                    output_buffers[(peer_start + i) * group_size + src_idx].ptr
                )

        comptime relay_sm_version = ctx.default_device_info.version
        var recipe = dispatch_select_comm_config[
            group_size, relay_sm_version, allgather_relay_tuning_table
        ](pair_max_length * size_of[dtype]())

        if pair_max_length > 0 and recipe.relay_percent > 0:
            var relay_sigs = Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS](
                uninitialized=True
            )
            for i in range(2 * group_size):
                relay_sigs[i] = rank_sigs[pair_base + i]

            return _allgather_p2p_relay(
                output_ptrs,
                list_of_in_ptrs,
                lengths_i32,
                peer_input_ptrs,
                peer_lengths,
                peer_output_ptrs,
                relay_sigs,
                recipe,
                ctx,
                my_rank - pair_base,
            )

    # TMA path: NVIDIA sm100+ with 16-byte-aligned (possibly zero) inputs.
    # Uses cp.async.bulk DMA for both NVLink reads and local HBM writes.
    comptime _use_tma = _is_sm10x_gpu(ctx.default_device_info)
    comptime if _use_tma:
        var tma_ok = True
        comptime for i in range(group_size):
            if (lengths[i] * size_of[dtype]()) % 16 != 0:
                tma_ok = False

        if tma_ok:
            return _allgather_p2p_tma[domain_id=domain_id](
                output_ptrs,
                list_of_in_ptrs,
                group_sigs,
                lengths_i32,
                ctx,
                loc_rank,
            )

    comptime BLOCK_SIZE = 256

    # Calculate grid size.
    var max_length = 0
    for i in range(group_size):
        max_length = max(max_length, lengths[i])

    # Only reachable for a relay pair whose partner carried the work: this
    # group had to get this far to relay, but has nothing of its own to gather.
    if max_length == 0:
        return

    comptime sm_version = ctx.default_device_info.version
    var max_num_blocks = _max_num_blocks.or_else(
        dispatch_select_comm_config[
            group_size, sm_version, allgather_tuning_table
        ](max_length * size_of[dtype]()).get_num_blocks()
    )

    comptime simd_width = simd_width_of[dtype, target=get_gpu_target()]()
    # Use ceildiv for max_length to ensure we have enough threads.
    var grid_size = min(
        max_num_blocks,
        ceildiv(ceildiv(max_length, simd_width), BLOCK_SIZE),
    )

    # Launch kernel.
    comptime allgather_p2p_kernel = _allgather_p2p_kernel[
        dtype,
        rank,
        group_size,
        BLOCK_SIZE=BLOCK_SIZE,
        domain_id=domain_id,
    ]
    ctx.enqueue_function[allgather_p2p_kernel](
        output_ptrs,
        list_of_in_ptrs,
        group_sigs,
        lengths_i32,
        Int32(max_num_blocks),
        Int32(loc_rank),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@inline(.always)
def allgather[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    in_engine: TensorEngine,
    out_engine: TensorEngine,
    *,
    group_size: Int = ngpus,
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin, Engine=out_engine],
        ngpus * group_size,
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    my_rank: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device all-gather: one instance per GPU builds its own outputs.

    Each instance reads all input buffers and writes to its own group_size
    output buffers. The caller is responsible for launching one instance per
    device in parallel (e.g. via _launch_device_collective).

    The implementation automatically selects between P2P and non-P2P paths
    based on hardware capabilities (non-P2P is full-world only; see Raises).

    World view vs. group
    - `ngpus` is the TOTAL number of devices in the world; `input_buffers` and
      `rank_sigs` carry every device's data, indexed by GLOBAL device rank.
    - `group_size` (defaults to `ngpus`) is the number of devices that
      actually cooperate on one all-gather. It must evenly divide `ngpus`.
      Devices `[g*group_size, (g+1)*group_size)` form group `g`; this call's
      group is derived from `my_rank`.
    - `output_buffers` holds EVERY device's own `group_size` outputs, laid out
      `[device * group_size + group_local_source]`. `my_rank` selects this
      device's own slice; the source dimension within that slice is the
      GROUP-local rank, not the global one.
    - `my_rank` is this device's GLOBAL rank in `[0, ngpus)`, not its rank
      within the group -- the group-local rank is derived internally.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Total number of devices in the world.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensors.
        out_origin: Origin of the output TileTensors.
        in_engine: Engine of the input TileTensors.
        out_engine: Engine of the output TileTensors.
        group_size: Number of devices per independent all-gather group. Must
            evenly divide `ngpus`. Defaults to `ngpus` (one full-world group,
            byte-identical to the pre-grouping behavior).

    Args:
        input_buffers: Input buffers from ALL `ngpus` devices as TileTensors,
            indexed by GLOBAL device rank.
        output_buffers: Output buffers for EVERY device (`ngpus * group_size`
            TileTensors); `output_buffers[my_rank * group_size + i]` receives
            the data from group-local source `i`. Every slot must name a real
            buffer: a grouped allgather may route part of a shard through the
            paired group, whose GPUs then write these buffers directly, so
            peer slots are not spare.
        rank_sigs: All `ngpus` devices' Signal pointers, indexed by GLOBAL
            device rank.
        ctx: Device context for THIS GPU.
        my_rank: GLOBAL rank of this GPU in `[0, ngpus)`.
        _max_num_blocks: Maximum number of blocks for kernel launch (optional).

    Raises:
        Error: `group_size != ngpus` (a grouped collective) and P2P access is
            not available -- the non-P2P fallback assumes a full-world,
            contiguous `0..ngpus-1` device layout and cannot be grouped.
    """
    comptime assert (
        group_size >= 2
    ), "allgather requires at least 2 GPUs per group"
    comptime assert (
        ngpus % group_size == 0
    ), "group_size must evenly divide ngpus"

    # This device's group. `group_start` is 0 for a full-world collective
    # (group_size == ngpus), so the emptiness check below is byte-identical
    # to the pre-grouping `range(ngpus)` scan from world index 0 in that case.
    var group_start = ualign_down(my_rank, group_size)

    # Return early if all of THIS DEVICE'S GROUP's input buffers are empty --
    # not the whole world's; sibling groups may legitimately be non-empty
    # while this one is. The exception is a relay pair: an empty group still
    # has to launch, because its GPUs relay for the partner that is not empty,
    # and the pair shares one barrier.
    comptime _empty_scan = (
        2
        * group_size if _relay_pairs[ngpus, group_size](
            ctx.default_device_info.version
        ) else group_size
    )
    var scan_start = ualign_down(my_rank, _empty_scan)
    var all_empty = True
    comptime for i in range(_empty_scan):
        if input_buffers[scan_start + i].num_elements() > 0:
            all_empty = False
            break
    if all_empty:
        return

    # Non-P2P fallback: full-world only (it assumes a contiguous 0..ngpus-1
    # device layout, so it cannot be grouped).
    if not is_p2p_enabled():
        comptime if group_size != ngpus:
            raise Error(
                "grouped allgather (group_size != ngpus) requires P2P access"
                " between GPUs"
            )
        comptime OutputTensorType = type_of(output_buffers[0])
        var my_outputs = Array[OutputTensorType, ngpus](uninitialized=True)
        comptime for i in range(ngpus):
            my_outputs[i] = output_buffers[my_rank * ngpus + i]
        return _allgather_naive(input_buffers, my_outputs, ctx)

    # P2P path: hand the collective the whole world plus the group width, and
    # let it derive the group-local slice, rank, and barrier domain itself.
    return _allgather_p2p[rank=1, group_size=group_size](
        input_buffers,
        output_buffers,
        rank_sigs,
        _max_num_blocks,
        ctx,
        my_rank,
    )
