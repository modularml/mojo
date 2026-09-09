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
"""Shared memory layout for grouped 1D-1D block-scaled SM100 matmul.

This is a simplified SMEM structure for the 1D-1D kernel variant that uses
offset-based addressing instead of pointer-per-group. Key differences from
the standard GroupedBlockScaledSmem:

1. No tensormap descriptors - TMAs are grid-constant (not updated per-group)
2. No CLC pipeline storage - uses 3-warp specialization (no scheduler warp)
3. Simpler barrier structure optimized for the 1D-1D workload

Tile storage is shared via BlockScaledTileCore from block_scaled_smem.mojo.
"""

from std.sys import size_of

from ..block_scaled.block_scaled_smem import BlockScaledTileCore
from ..structured_kernels.config import BlockScaledMatmulConfig
from structured_kernels.pipeline_storage import (
    BarrierPair,
    RawBarrierStorage,
    SmemPipelineBundleNoClc,
    MbarPtr,
)
from structured_kernels.smem_types import SMemArray, SMemPtr


# ===----------------------------------------------------------------------=== #
# Scheduler warp SMEM: double-buffered batch of tile descriptors
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct SchedulerSlot(TrivialRegisterPassable):
    """One precomputed tile descriptor in SMEM (32 bytes).

    Scheduler warp writes; every consumer warp reads. All the fields
    consumers need are precomputed here so they avoid GMEM reads per tile.
    """

    var m: UInt32
    var n: UInt32
    var group_idx: UInt32
    var expert_id: Int32  # -1 = sentinel (end of tiles)
    var m_start: UInt32
    var m_end: UInt32
    var expert_scale: Float32
    # Pad to 32 bytes so the 2-slot SMEM array lines up on a 64-B boundary.
    var _pad: UInt32


struct SchedulerSmem:
    """Scheduler-warp SMEM: 2-slot ProducerConsumer ring + active-group cache.

    The scheduler warp writes a tile descriptor into `slots[i % 2]` then
    arrives on `full[i % 2]`; consumers wait on that mbarrier, read the slot,
    and arrive on `empty[i % 2]` to free the slot for the next producer write.
    """

    comptime NUM_SLOTS = 2  # 2-slot ProducerConsumer
    comptime MAX_TILES = Self.NUM_SLOTS  # backward compat alias
    # Group cache sized to fit the B200 SMEM limit under worst-case MXFP8
    # 2SM configs (base SMEM is already ~232.5 KB, leaving <1 KB for us).
    # 32 covers the common MoE decode regime and the larger Kimi-style 49
    # expert workloads fall back to GMEM reads per tile without correctness
    # impact.
    comptime GROUP_CACHE_CAP = 32
    comptime SlotArray = SMemArray[SchedulerSlot, Self.NUM_SLOTS]
    comptime GroupOffsetsArray = SMemArray[UInt32, Self.GROUP_CACHE_CAP + 1]
    comptime ExpertIdsArray = SMemArray[Int32, Self.GROUP_CACHE_CAP]
    comptime ExpertScalesArray = SMemArray[Float32, Self.GROUP_CACHE_CAP]

    # 4 mbarriers: full[0], full[1], empty[0], empty[1].
    var barriers: RawBarrierStorage[4]
    var slot_storage: Self.SlotArray.Storage
    # Active-group cache: avoids GMEM re-reads on the always-on scheduler
    # path when num_active_experts <= GROUP_CACHE_CAP.
    var group_offsets_storage: Self.GroupOffsetsArray.Storage
    var expert_ids_storage: Self.ExpertIdsArray.Storage
    var expert_scales_storage: Self.ExpertScalesArray.Storage

    @always_inline
    def slots(ref[AddressSpace.SHARED] self) -> SMemPtr[SchedulerSlot]:
        return Self.SlotArray(self.slot_storage).ptr

    @always_inline
    def group_offsets_ptr(ref[AddressSpace.SHARED] self) -> SMemPtr[UInt32]:
        return Self.GroupOffsetsArray(self.group_offsets_storage).ptr

    @always_inline
    def expert_ids_ptr(ref[AddressSpace.SHARED] self) -> SMemPtr[Int32]:
        return Self.ExpertIdsArray(self.expert_ids_storage).ptr

    @always_inline
    def expert_scales_ptr(ref[AddressSpace.SHARED] self) -> SMemPtr[Float32]:
        return Self.ExpertScalesArray(self.expert_scales_storage).ptr

    # ── 2-slot ProducerConsumer accessors ──
    # Reuse mbar[0]/mbar[1] as full, mbar[2]/mbar[3] as empty.
    @always_inline
    def full_mbar(ref[AddressSpace.SHARED] self) -> MbarPtr:
        return self.barriers.ptr()

    @always_inline
    def empty_mbar(ref[AddressSpace.SHARED] self) -> MbarPtr:
        return self.barriers.ptr() + 2


struct Grouped1D1DSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """SMEM struct for grouped 1D-1D block-scaled GEMM without CLC scheduler.

    Thin wrapper over BlockScaledTileCore + SmemPipelineBundleNoClc.
    Uses 3-warp specialization (Load, MMA, Epilogue) without a scheduler warp.

    Parameters:
        a_type: Element type of the A operand matrix.
        b_type: Element type of the B operand matrix.
        c_type: Element type of the C output matrix.
        sfa_dtype: Element type of the A operand scaling factors.
        sfb_dtype: Element type of the B operand scaling factors.
        transpose_b: Whether the B operand is stored in transposed layout.
        config: Block-scaled matmul configuration providing tile shapes,
            pipeline stage counts, and scaling factor layout parameters.
    """

    # ========== Core (tile storage + constants) ==========
    comptime Core = BlockScaledTileCore[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config=Self.config,
    ]
    comptime SCHED_GROUP_CACHE_CAP = SchedulerSmem.GROUP_CACHE_CAP

    # ========== Storage Fields ==========
    var core: Self.Core

    # ========== Pipeline Storage (no CLC) ==========
    comptime Pipelines = SmemPipelineBundleNoClc[
        Self.Core.num_group_pipeline_stages,
        Self.Core.num_accum_pipeline_stages,
        Self.Core.Payload,
    ]
    var pipelines: Self.Pipelines

    # ========== SFB Load Barriers (MMA_N < 64) ==========
    # SFB TMEM load warps arrive after writing SFB to TMEM via tcgen05_st.
    # MMA warp waits before issuing UMMA.
    var sfb_load_barriers: RawBarrierStorage[
        Int(Self.Core.num_group_pipeline_stages)
    ]

    # ========== SFB TMA Pipeline Barriers (MMA_N < 64) ==========
    # Producer-consumer pipeline between SfbTMALoad warp and SfbTMEMLoad/MMA.
    # Producer (SfbTMALoad): signals FULL after TMA copy, waits EMPTY from MMA.
    # Consumer (SfbTMEMLoad): waits FULL, signals sfb_load_barriers after TMEM write.
    # Consumer (MMA): signals EMPTY after consuming TMEM data.
    var sfb_tma_barriers: BarrierPair[Int(Self.Core.num_group_pipeline_stages)]

    # ========== Scheduler Warp Buffers ==========
    # Double-buffered precomputed tile descriptors from the scheduler warp.
    var scheduler: SchedulerSmem

    # ========== SFB Barrier Accessors ==========
    @always_inline
    def sfb_load_mbars_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get pointer to SFB-load mbarrier array (SFB Load→MMA)."""
        return self.sfb_load_barriers.ptr()

    @always_inline
    def sfb_tma_mbars_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get pointer to SFB TMA pipeline mbarrier array (SfbTMALoad↔MMA)."""
        return self.sfb_tma_barriers.ptr()

    # ========== Scheduler Accessors ==========
    @always_inline
    def sched_slots(
        ref[AddressSpace.SHARED] self,
    ) -> SMemPtr[SchedulerSlot]:
        return self.scheduler.slots()

    @always_inline
    def sched_full_mbar(ref[AddressSpace.SHARED] self) -> MbarPtr:
        return self.scheduler.full_mbar()

    @always_inline
    def sched_empty_mbar(ref[AddressSpace.SHARED] self) -> MbarPtr:
        return self.scheduler.empty_mbar()

    @always_inline
    def sched_group_offsets(
        ref[AddressSpace.SHARED] self,
    ) -> SMemPtr[UInt32]:
        return self.scheduler.group_offsets_ptr()

    @always_inline
    def sched_expert_ids(
        ref[AddressSpace.SHARED] self,
    ) -> SMemPtr[Int32]:
        return self.scheduler.expert_ids_ptr()

    @always_inline
    def sched_expert_scales(
        ref[AddressSpace.SHARED] self,
    ) -> SMemPtr[Float32]:
        return self.scheduler.expert_scales_ptr()

    # ========== Tile Accessors (forwarding) ==========
    @always_inline
    def a_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.ATileArray:
        """Get A tile array accessor."""
        return self.core.a_tiles()

    @always_inline
    def b_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.BTileArray:
        """Get B tile array accessor."""
        return self.core.b_tiles()

    @always_inline
    def c_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.CTileArray:
        """Get C tile array accessor."""
        return self.core.c_tiles()

    @always_inline
    def sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFATileArray:
        """Get SFA tile array accessor."""
        return self.core.sfa_tiles()

    @always_inline
    def sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.Core.SFBTileArray:
        """Get SFB tile array accessor."""
        return self.core.sfb_tiles()

    # ========== Size Utilities (forwarding) ==========
    @staticmethod
    @always_inline
    def ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.Core.ab_pipeline_size()

    @staticmethod
    @always_inline
    def sf_pipeline_size() -> Int:
        """Total size of SFA+SFB tiles for all pipeline stages (in elements)."""
        return Self.Core.sf_pipeline_size()

    @staticmethod
    @always_inline
    def c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.Core.c_output_size()

    @staticmethod
    @always_inline
    def total_tile_size() -> Int:
        """Total tile storage size (A+B+SFA+SFB+C) in elements."""
        return Self.Core.total_tile_size()
