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

from .device_query import dispatch_select_comm_config, DefaultCommTuningConfig
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


@always_inline
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


@always_inline
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


@always_inline
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
    domain_id: Int = 0,
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin, Engine=out_engine],
        ngpus,
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    _max_num_blocks: Optional[Int],
    ctx: DeviceContext,
    my_rank: Int,
) raises:
    """Per-device P2P allgather: each GPU reads from all peers directly."""

    var list_of_in_ptrs = StaticTuple[
        ImmPointer[Scalar[dtype], ImmutAnyOrigin], ngpus
    ]()
    var lengths = StaticTuple[Int, ngpus]()

    comptime for i in range(ngpus):
        list_of_in_ptrs[i] = rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
            input_buffers[i].ptr
        )
        lengths[i] = input_buffers[i].num_elements()

    # Prepare output pointers.
    var output_ptrs = StaticTuple[
        MutPointer[Scalar[dtype], MutAnyOrigin], ngpus
    ]()

    comptime for src_idx in range(ngpus):
        output_ptrs[src_idx] = rebind[MutPointer[Scalar[dtype], MutAnyOrigin]](
            output_buffers[src_idx].ptr
        )

    # Build Int32 versions for passing to GPU kernels.
    var lengths_i32 = StaticTuple[Int32, ngpus]()
    comptime for i in range(ngpus):
        lengths_i32[i] = Int32(lengths[i])

    # TMA path: NVIDIA sm100+ with 16-byte-aligned (possibly zero) inputs.
    # Uses cp.async.bulk DMA for both NVLink reads and local HBM writes.
    comptime _use_tma = _is_sm10x_gpu(ctx.default_device_info)
    comptime if _use_tma:
        var tma_ok = True
        comptime for i in range(ngpus):
            if (lengths[i] * size_of[dtype]()) % 16 != 0:
                tma_ok = False

        if tma_ok:
            return _allgather_p2p_tma[domain_id=domain_id](
                output_ptrs,
                list_of_in_ptrs,
                rank_sigs,
                lengths_i32,
                ctx,
                my_rank,
            )

    comptime BLOCK_SIZE = 256

    # Calculate grid size.
    var max_length = 0
    for i in range(ngpus):
        max_length = max(max_length, lengths[i])

    comptime sm_version = ctx.default_device_info.version
    var max_num_blocks = _max_num_blocks.or_else(
        dispatch_select_comm_config[ngpus, sm_version, allgather_tuning_table](
            max_length * size_of[dtype]()
        ).get_num_blocks()
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
        ngpus,
        BLOCK_SIZE=BLOCK_SIZE,
        domain_id=domain_id,
    ]
    ctx.enqueue_function[allgather_p2p_kernel](
        output_ptrs,
        list_of_in_ptrs,
        rank_sigs,
        lengths_i32,
        Int32(max_num_blocks),
        Int32(my_rank),
        grid_dim=grid_size,
        block_dim=BLOCK_SIZE,
        attributes=pdl_launch_attributes(PDLLevel.ON),
    )


@always_inline
def allgather[
    dtype: DType,
    ngpus: Int,
    in_layout: TensorLayout,
    in_origin: Origin,
    out_layout: TensorLayout,
    out_origin: MutOrigin,
    in_engine: TensorEngine,
    out_engine: TensorEngine,
    domain_id: Int = 0,
](
    input_buffers: Array[
        TileTensor[dtype, in_layout, in_origin, Engine=in_engine], ngpus
    ],
    output_buffers: Array[
        TileTensor[mut=True, dtype, out_layout, out_origin, Engine=out_engine],
        ngpus,
    ],
    rank_sigs: Array[MutPointer[Signal, MutAnyOrigin], MAX_GPUS],
    ctx: DeviceContext,
    my_rank: Int,
    _max_num_blocks: Optional[Int] = None,
) raises:
    """Per-device all-gather: one instance per GPU builds its own outputs.

    Each instance reads all input buffers and writes to its own ngpus output
    buffers. The caller is responsible for launching one instance per device
    in parallel (e.g. via _launch_device_collective).

    The implementation automatically selects between P2P and non-P2P paths
    based on hardware capabilities.

    Parameters:
        dtype: Data type of the tensor elements.
        ngpus: Number of GPUs participating in all-gather.
        in_layout: Layout of the input TileTensors.
        in_origin: Origin of the input TileTensors.
        out_layout: Layout of the output TileTensors.
        out_origin: Origin of the output TileTensors.
        in_engine: Engine of the input TileTensors.
        out_engine: Engine of the output TileTensors.
        domain_id: Barrier counter bank to use (0 for full-world; a distinct
            nonzero value for grouped collectives sharing the same Signal
            buffers). See `_multi_gpu_barrier`.

    Args:
        input_buffers: Input buffers from ALL GPUs as TileTensors.
        output_buffers: Output buffers for THIS GPU (ngpus TileTensors).
                       output_buffers[i] receives the data from GPU i.
        rank_sigs: Per-GPU Signal pointers for P2P synchronization.
        ctx: Device context for THIS GPU.
        my_rank: Index of this GPU among the participants.
        _max_num_blocks: Maximum number of blocks for kernel launch (optional).
    """
    comptime assert ngpus >= 2, "allgather requires at least 2 GPUs"

    # Return early if all input buffers are empty.
    var all_empty = True

    comptime for i in range(ngpus):
        if input_buffers[i].num_elements() > 0:
            all_empty = False
            break
    if all_empty:
        return

    # Check P2P availability.
    if not is_p2p_enabled():
        return _allgather_naive(input_buffers, output_buffers, ctx)
    else:
        return _allgather_p2p[rank=1, domain_id=domain_id](
            input_buffers,
            output_buffers,
            rank_sigs,
            _max_num_blocks,
            ctx,
            my_rank,
        )
