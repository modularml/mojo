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
"""Grouped 1D-1D block-scaled SM100 matmul kernel.

This kernel implements grouped GEMM for Mixture of Experts (MoE) layers using
the 1D-1D tensor layout with offset-based addressing.

Key characteristics:
- Warp specialization (Load, MMA, Epilogue, optional SFB Load)
- Grid-constant TMA descriptors (no runtime tensormap updates)
- Offset-based addressing via a_offsets for contiguous token buffers
- Per-expert output scaling via expert_scales tensor

Architecture (MMA_N >= 64, 192 threads):
- TMA warp: Loads A, B, SFA, SFB tiles using grid-constant TMAs
- MMA warp: Executes block-scaled matrix multiply (SFB via tcgen05_cp)
- Epilogue warps: Stores results with expert_scale applied

Architecture (MMA_N < 64, 352 threads):
- TMA warp: Loads A, B, SFA tiles using grid-constant TMAs
- MMA warp: Executes block-scaled matrix multiply
- Epilogue warps: Stores results with expert_scale applied
- SFB TMA Load warp: Loads SFB from GMEM to SMEM via TMA (reduced tile)
- SFB TMEM Load warps: Reads SFB from SMEM, writes to TMEM via tcgen05_st

This is a port of grouped_matmul_sm100_1d1d.mojo to the structured kernels
architecture.
"""

from std.builtin.device_passable import DevicePassable, DeviceTypeEncoder
from std.collections import Optional
from std.math import align_up, ceildiv
from std.memory import Pointer, UnsafePointer, bitcast
from std.math.uutils import ufloordiv, umod
from std.sys import align_of, size_of

from max.gpu import (
    WARP_SIZE,
    block_id_in_cluster,
    block_idx,
    grid_dim,
    thread_idx,
    lane_id,
    warp_id as get_warp_id,
)
from max.gpu.memory import (
    async_copy,
    external_memory,
    fence_mbarrier_init,
)
from max.gpu.primitives.grid_controls import (
    PDLLevel,
    launch_dependent_grids,
    wait_on_dependent_grids,
)
import max.gpu.primitives.warp as warp
from max.gpu.primitives.cluster import (
    block_rank_in_cluster,
    cluster_sync,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from max.gpu.sync import async_copy_arrive, syncwarp
from max.gpu.compute.arch.tcgen05 import (
    tcgen05_fence_before,
    tcgen05_st,
    tcgen05_store_wait,
)
from layout.tma_async import PipelineState
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Coord,
    Idx,
    Layout,
    DefaultEngine,
    RowMajorLayout,
    TensorLayout,
    TensorEngine,
    TileTensor,
    row_major,
)
from layout.swizzle import Swizzle
from structured_kernels.tile_types import (
    GMEMLayout1D,
    TmaOpType,
    static_row_major,
)
from layout.tile_layout import Layout as TileLayout, _IntToComptimeInt

from std.utils.index import IndexList
from std.utils.static_tuple import StaticTuple

from std.math import exp, recip
from std.time import global_perf_counter_ns

from linalg.arch.sm100 import MmaOpSM100_BlockScaled_SS
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    _is_packed_fp4,
    block_scaled_operands_compatible,
    cast_fp32_to_fp4e2m1,
    set_scale_factor,
)
from linalg.utils import elementwise_compute_lambda_type

from ..structured_kernels.config import (
    BlockScaledMatmulConfig,
    OutputPipelineConfig,
)
from structured_kernels.kernel_common import (
    MbarPtr,
    WarpRole1D1D,
    compute_tma_tile_dims,
    compute_accum_barrier_counts,
    compute_input_consumer_count,
    init_core_barriers,
)
from ..structured_kernels.tile_pipeline import (
    InputTilePipeline,
    ProducerTiles,
    ConsumerTiles,
    OutputTilePipeline,
    BlockScaledTilePayload,
)
from structured_kernels.pipeline import ProducerConsumerPipeline
from ..structured_kernels.epilogue_components import AccumBarrier
from ..structured_kernels.tmem import (
    BlockScaledTmem,
    TmemAllocation,
    TmemArrayType,
    TmemDeallocBarrier,
)
from structured_kernels.barriers import WarpGroupBarrier
from structured_kernels.trace_buf import GmemTrace, NullTrace, TraceBuf
from ..structured_kernels.warp_context import (
    MmaWarpContext,
    EpilogueWarpContext,
)

from .grouped_1d1d_smem import Grouped1D1DSmem, SchedulerSlot
from .grouped_1d1d_tile_scheduler import (
    GroupedWorkIterator1D1D,
    GroupedWorkContext1D1D,
)
from ..structured_kernels.output_writer import TileWriter


comptime SWIGLU_MAX_TRACED_TILES = 64
"""Maximum number of consecutive output tiles whose per-tile pipeline
events are recorded per CTA. Tiles past this count are not traced
(the per-warp counters increment but the gated record-sites become
no-ops). 64 covers a large-prefill cta_group=2 cluster (~47 output
tiles on the leader CTA) with headroom; the trace buffer lives in
global memory, so raising this only grows the debug-only GmemTrace
allocation, not SMEM."""

comptime GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK = 16 * SWIGLU_MAX_TRACED_TILES
"""Number of `UInt64` timestamp slots reserved per CTA.

Schema is per-tile pipeline events plus per-tile epi sub-phase events for
the first epi `loop_stage` only (sub-phases of stages 1..N-1 are not
recorded; stage 0 is representative since stages have uniform cost).

  9 base events × N tiles          (offset 9*i)
  5 sub-phase events × N tiles     (offset 9*N + 5*i)
  2 MMA-acquire events × N tiles   (offset 14*N + i and 15*N + i)
  total = 16*N, N = SWIGLU_MAX_TRACED_TILES (the per-field offsets
  below are shown for the N=8 layout; the actual bases are the derived
  constants _GROUPED_TRACE_SUBPHASE_BASE / _MMA_OUTPUT_ACQ_BASE /
  _MMA_INPUT_ACQ_BASE just below this docstring)

Slot encoding for output tile i (i in [0, SWIGLU_MAX_TRACED_TILES)):

  Base events (offset 9*i + 0..8):
    9*i + 0 = T_LOAD_DISPATCH (load warp, top of outer tile loop:
                                BEFORE producer-pipeline acquire;
                                marks when the warp is dispatched to
                                attempt this tile)
    9*i + 1 = T_LOAD_START    (AFTER producer.acquire returns:
                                marks real-work begin; gap to
                                DISPATCH = pipeline-hazard wait)
    9*i + 2 = T_LOAD_END      (after the LAST k-tile's TMA issue)
    9*i + 3 = T_MMA_DISPATCH  (MMA warp, top of outer tile loop;
                                leader CTA only)
    9*i + 4 = T_MMA_START     (AFTER both output- and input-pipeline
                                acquires; first mma is about to fire)
    9*i + 5 = T_MMA_END       (after the LAST mma issue + commit)
    9*i + 6 = T_EPI_DISPATCH  (epi warp, top of outer tile loop)
    9*i + 7 = T_EPI_START     (AFTER consumer-pipeline acquire:
                                accumulator ready, body about to start)
    9*i + 8 = T_EPI_END       (after Self.epilogue returns)

  Stage-0 sub-phase events (offset 72 + 5*i + 0..4):
    72 + 5*i + 0 = T_EPI_S0_TMEM_DONE     (stage 0: after wait_load
                                              returns, fragments resident)
    72 + 5*i + 1 = T_EPI_S0_SCATTER_DONE  (stage 0: after the per-thread
                                              scatter loop emits all
                                              fp32→bf16 SMEM stores)
    72 + 5*i + 2 = T_EPI_S0_BAR1_DONE     (stage 0: after the first
                                              `WarpGroupBarrier.sync()`
                                              returns; SMEM scratchpad
                                              visible to all epi warps)
    72 + 5*i + 3 = T_EPI_S0_COOP_DONE     (stage 0: after the cooperative
                                              SwiGLU + quant + GMEM store
                                              loop completes, before the
                                              second sync)
    72 + 5*i + 4 = T_EPI_S0_FINAL_DONE    (stage 0: after the second
                                              `WarpGroupBarrier.sync()`
                                              returns; stage 0 fully done)

  MMA per-tile acquire-split events (offset 112 + i, 120 + i):
    112 + i = T_MMA_OUTPUT_ACQ  (AFTER `output_pipeline.producer()`
                                  returns the slot; BEFORE input acquire
                                  and any MMA work. Lets us split
                                  M_S - M_D into output-pipeline wait
                                  vs input-pipeline + SFB wait.)
    120 + i = T_MMA_INPUT_ACQ   (AFTER `input_pipeline.consumer()` data
                                  is acquired for k_tile=0; BEFORE the
                                  SFB-load barrier when MMA_N<64.
                                  T_MMA_INPUT_ACQ - T_MMA_OUTPUT_ACQ
                                  isolates the input-pipeline wait;
                                  M_S - T_MMA_INPUT_ACQ isolates the
                                  SFB-load barrier wait.)

Issue latency = T_X_START − T_X_DISPATCH for X in {LOAD, MMA, EPI}.
This is the wait at the acquire: the warp wanted to begin tile i but
had to block on a pipeline slot. A near-zero issue latency means the
pipeline is well-fed; a large one means the warp is starved.

MMA wait decomposition (per tile i):
    output_wait = T_MMA_OUTPUT_ACQ - T_MMA_DISPATCH  (wait on EPI to
                  release output-pipeline slot via AccumBarrier.arrive)
    input_wait  = T_MMA_INPUT_ACQ  - T_MMA_OUTPUT_ACQ (wait on TMA load
                  data to land for k_tile=0)
    sfb_wait    = T_MMA_START      - T_MMA_INPUT_ACQ (wait on SFB-load
                  warp to write SFB to TMEM; only if MMA_N<64, else 0)
The dep with the longest sub-wait is the binding constraint for tile i.

Sub-phase durations for stage 0 of tile i:
    tmem    = T_EPI_S0_TMEM_DONE - T_EPI_START
    scatter = T_EPI_S0_SCATTER_DONE - T_EPI_S0_TMEM_DONE
    bar1    = T_EPI_S0_BAR1_DONE - T_EPI_S0_SCATTER_DONE
    coop    = T_EPI_S0_COOP_DONE - T_EPI_S0_BAR1_DONE
    bar2    = T_EPI_S0_FINAL_DONE - T_EPI_S0_COOP_DONE

Inter-tile overlap is still directly visible:
  T_LOAD_START[i+1] < T_MMA_END[i]    → Load[i+1] real-work overlaps
                                         with MMA[i] real-work."""


comptime _GROUPED_TRACE_SUBPHASE_BASE = 9 * SWIGLU_MAX_TRACED_TILES
"""Base trace slot for the per-tile stage-0 sub-phase events (5/tile);
follows the 9*N base events."""
comptime _GROUPED_TRACE_MMA_OUTPUT_ACQ_BASE = 14 * SWIGLU_MAX_TRACED_TILES
"""Base trace slot for the per-tile T_MMA_OUTPUT_ACQ events (1/tile);
follows the 9*N base + 5*N sub-phase events."""
comptime _GROUPED_TRACE_MMA_INPUT_ACQ_BASE = 15 * SWIGLU_MAX_TRACED_TILES
"""Base trace slot for the per-tile T_MMA_INPUT_ACQ events (1/tile)."""


# =============================================================================
# SwiGLUOutput trait — zero-cost-when-unused fused-output destinations
# =============================================================================
#
# Modeled on `TraceBuf` in `structured_kernels/trace_buf.mojo`. The
# kernel takes `SwiGLUOutputT: SwiGLUOutput` as a comptime parameter and
# `swiglu_out: SwiGLUOutputT` as a runtime arg. When fusion is off, the
# caller passes `NullSwiGLUOutput()` — a zero-sized struct whose method
# bodies compile to no-ops, contributing 0 bytes to the kernel ABI.
#
# When fusion is on, the caller passes `RealSwiGLUOutput(c_packed,
# c_swiglu_scales, c_input_scales)`, which carries the three GMEM
# tensors needed by the in-tile fused epilogue.
#
# Trace observability is layered on top via the kernel's `swiglu_enable_trace`
# comptime parameter and `trace_buf: TraceBufT` arg, so the SwiGLU output
# concern stays separate from instrumentation.
# =============================================================================


trait SwiGLUOutput(DevicePassable, TrivialRegisterPassable):
    """Trait for fused SwiGLU + block-scaled output destinations.

    Parametric on the per-block scale dtype and vector size: NVFP4
    (E4M3, 16) and MXFP8 (E8M0, 32). The packed output is a byte
    stream where each NVFP4 byte holds 2 nibbles and each MXFP8 byte
    holds 1 fp8_e4m3fn element; byte-stride math lives in the kernel.

    Implementations:
      - `NullSwiGLUOutput`: zero-sized no-op for the non-fused path.
      - `RealSwiGLUOutput`: carries packed output, 5D FP8 scale tile,
        and per-active-expert input scales.
    """

    comptime SfDtype: DType
    comptime SfVectorSize: Int
    # `True` enables clamped SwiGLU (`swigluoai`):
    #   g' = min(g, L); u' = clamp(u, -L, L); (u'+1)·g'·σ(g'·α).
    # `clamp_alpha()` and `clamp_limit()` carry the runtime constants.
    comptime ClampActivation: Bool

    def store_packed_byte(self, m: Int, byte_pos: Int, val: UInt8):
        """Store one output byte at GMEM (m, byte_pos).

        Args:
            m: Row index in the packed output tensor.
            byte_pos: Byte offset within the row.
            val: Byte value to store.
        """
        ...

    def store_packed_word(self, m: Int, byte_pos: Int, val: UInt32):
        """Store one 4-byte word at GMEM (m, byte_pos). `byte_pos` and
        the row stride must both be 4-byte aligned so this emits one
        ST.GLOBAL.B32 vs four scalar byte stores.

        Args:
            m: Row index in the packed output tensor.
            byte_pos: Byte offset within the row; must be a multiple of
                4.
            val: 4-byte word value to store.
        """
        ...

    def set_sf(
        self,
        m: Int,
        post_col: Int,
        sf: Scalar[Self.SfDtype],
    ):
        """Set the per-(m, post_col) FP8 scale factor.

        Args:
            m: Row index in the output (M dimension).
            post_col: Column index in the post-SwiGLU output (N
                dimension).
            sf: Scale factor value to store.
        """
        ...

    def input_scale(self, active_expert_idx: Int) -> Float32:
        """Read per-active-expert input scale (`tensor_sf` in ep_comm).
        Unused for MXFP8 (gated out by the kernel).

        Args:
            active_expert_idx: Index into the per-active-expert input
                scales array.
        """
        ...

    def clamp_alpha(self) -> Float32:
        """Runtime α for the clamped activation. Unused when
        `ClampActivation=False`."""
        ...

    def clamp_limit(self) -> Float32:
        """Runtime limit L for the clamped activation. Unused when
        `ClampActivation=False`."""
        ...

    def pad_sf_zero_block(
        self,
        sf_block_base: Int,
        tokens_e: Int,
        tid: Int,
        stride: Int,
    ):
        """Zero-fill the per-expert SF tail-pad rows in
        `[tokens_e, ceildiv(tokens_e, 128) * 128)` across all post-SwiGLU
        channels, distributed over `stride` threads keyed by `tid`.

        Called once per expert, on the CTA that processed the last live
        tile, so the host doesn't need to memset the SF buffer.

        Args:
            sf_block_base: Row offset of this expert's SF block within
                the global SF buffer.
            tokens_e: Number of live (non-padded) token rows for this
                expert.
            tid: Thread id used to distribute the zero-writes across
                threads.
            stride: Number of threads participating in the zero-fill.
        """
        ...


struct NullSwiGLUOutput[
    _sf_dtype: DType = NVFP4_SF_DTYPE,
    _sf_vector_size: Int = NVFP4_SF_VECTOR_SIZE,
    _clamp_activation: Bool = False,
](SwiGLUOutput):
    """Zero-sized no-op SwiGLU output for `fuse_swiglu=False`.

    Parametric so the kernel's comptime trait bound matches the
    dispatch's wiring; defaults are NVFP4 + plain SwiGLU.

    Parameters:
        _sf_dtype: `DType` of the per-block scale factors; `NVFP4_SF_DTYPE`
            for NVFP4 or `MXFP8_SF_DTYPE` for MXFP8 (defaults to
            `NVFP4_SF_DTYPE`).
        _sf_vector_size: Number of output elements covered by each scale
            factor; 16 for NVFP4, 32 for MXFP8 (defaults to
            `NVFP4_SF_VECTOR_SIZE`).
        _clamp_activation: When `True`, enables the clamped SwiGLU
            activation; unused on this no-op struct but kept so the
            trait bound matches the dispatch wiring (defaults to
            `False`).
    """

    comptime SfDtype = Self._sf_dtype
    comptime SfVectorSize = Self._sf_vector_size
    comptime ClampActivation = Self._clamp_activation

    comptime device_type: AnyType = Self

    @always_inline
    def __init__(out self):
        pass

    @always_inline
    def store_packed_byte(self, m: Int, byte_pos: Int, val: UInt8):
        pass

    @always_inline
    def store_packed_word(self, m: Int, byte_pos: Int, val: UInt32):
        pass

    @always_inline
    def set_sf(self, m: Int, post_col: Int, sf: Scalar[Self.SfDtype]):
        pass

    @always_inline
    def input_scale(self, active_expert_idx: Int) -> Float32:
        return Float32(0.0)

    @always_inline
    def clamp_alpha(self) -> Float32:
        return Float32(0.0)

    @always_inline
    def clamp_limit(self) -> Float32:
        return Float32(0.0)

    @always_inline
    def pad_sf_zero_block(
        self,
        sf_block_base: Int,
        tokens_e: Int,
        tid: Int,
        stride: Int,
    ):
        pass

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        pass

    @staticmethod
    def get_type_name() -> String:
        return "NullSwiGLUOutput"


struct RealSwiGLUOutput[
    # Output row stride in bytes. NVFP4: H/2. MXFP8: H.
    c_packed_row_stride: Int,
    # SF tile dim 1 (n_blocks_per_row). The 5D SF tile shape is
    # (m_groups, sf_dim1, 32, 4, 4); `set_scale_factor` indexes it via
    # this dim count.
    sf_dim1: Int,
    _sf_dtype: DType = NVFP4_SF_DTYPE,
    _sf_vector_size: Int = NVFP4_SF_VECTOR_SIZE,
    _clamp_activation: Bool = False,
](SwiGLUOutput):
    """Carries the three GMEM destinations (packed output, 5D SF tile,
    per-expert input scales) as raw pointers + comptime shape info.

    Raw `Pointer`s rather than `TileTensor`s sidestep a callsite
    type mismatch (TileTensor has many implicit parameters). `set_sf`
    inlines the 5D SF index. Layout is the same for NVFP4 and MXFP8,
    only the SF_VECTOR_SIZE divisor differs.

    Parameters:
        c_packed_row_stride: Output row stride in bytes; `H/2` for
            NVFP4 (two nibbles per byte) and `H` for MXFP8 (one
            element per byte).
        sf_dim1: Second dimension of the 5D SF tile
            `(m_groups, sf_dim1, 32, 4, 4)`, equal to the number of
            scale-factor blocks per output row; used by `set_sf` to
            index the linearized SF buffer.
        _sf_dtype: `DType` of the per-block scale factors; `NVFP4_SF_DTYPE`
            for NVFP4 or `MXFP8_SF_DTYPE` for MXFP8 (defaults to
            `NVFP4_SF_DTYPE`).
        _sf_vector_size: Number of output elements covered by each scale
            factor; 16 for NVFP4, 32 for MXFP8 (defaults to
            `NVFP4_SF_VECTOR_SIZE`).
        _clamp_activation: When `True`, enables the clamped SwiGLU
            activation (`swigluoai`); the runtime constants are carried
            by `clamp_alpha` and `clamp_limit` (defaults to `False`).
    """

    comptime SfDtype = Self._sf_dtype
    comptime SfVectorSize = Self._sf_vector_size
    comptime ClampActivation = Self._clamp_activation

    comptime device_type: AnyType = Self

    @__allow_legacy_any_origin_fields
    var c_packed_ptr: UnsafePointer[UInt8, MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var c_swiglu_scales_ptr: UnsafePointer[Scalar[Self._sf_dtype], MutAnyOrigin]

    @__allow_legacy_any_origin_fields
    var c_input_scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin]
    # Runtime alpha and L for the clamped activation. Always stored;
    # the 8 bytes keep the struct layout uniform across activations.
    var _clamp_alpha: Float32
    var _clamp_limit: Float32

    @always_inline
    def __init__(
        out self,
        c_packed_ptr: UnsafePointer[UInt8, MutAnyOrigin],
        c_swiglu_scales_ptr: UnsafePointer[
            Scalar[Self._sf_dtype], MutAnyOrigin
        ],
        c_input_scales_ptr: UnsafePointer[Float32, ImmutAnyOrigin],
        clamp_alpha: Float32 = Float32(0.0),
        clamp_limit: Float32 = Float32(0.0),
    ):
        self.c_packed_ptr = c_packed_ptr
        self.c_swiglu_scales_ptr = c_swiglu_scales_ptr
        self.c_input_scales_ptr = c_input_scales_ptr
        self._clamp_alpha = clamp_alpha
        self._clamp_limit = clamp_limit

    @always_inline
    def store_packed_byte(self, m: Int, byte_pos: Int, val: UInt8):
        # c_packed shape (M_total, c_packed_row_stride), row-major.
        self.c_packed_ptr.store(m * Self.c_packed_row_stride + byte_pos, val)

    @always_inline
    def store_packed_word(self, m: Int, byte_pos: Int, val: UInt32):
        # Caller guarantees `byte_pos` and `c_packed_row_stride` are
        # multiples of 4 so PTX lowers to one ST.GLOBAL.B32.
        self.c_packed_ptr.bitcast[UInt32]().store(
            (m * Self.c_packed_row_stride + byte_pos) // 4, val
        )

    @always_inline
    def set_sf(
        self,
        m: Int,
        post_col: Int,
        sf: Scalar[Self.SfDtype],
    ):
        # Mirrors `set_scale_factor[SF_VECTOR_SIZE=Self.SfVectorSize]`
        # in `linalg/fp4_utils.mojo`; the 5D SF tile is shaped
        # (n_blocks, sf_dim1, SF_ATOM_M[0], SF_ATOM_M[1], SF_ATOM_K).
        var dim2 = SF_ATOM_M[0]
        var dim3 = SF_ATOM_M[1]
        var dim4 = SF_ATOM_K
        var i0 = m // SF_MN_GROUP_SIZE
        var i1 = post_col // (Self.SfVectorSize * SF_ATOM_K)
        var i2 = m % dim2
        var i3 = (m % SF_MN_GROUP_SIZE) // dim2
        var i4 = (post_col // Self.SfVectorSize) % dim4
        var linear_idx = (
            ((i0 * Self.sf_dim1 + i1) * dim2 + i2) * dim3 + i3
        ) * dim4 + i4
        self.c_swiglu_scales_ptr.store(linear_idx, sf)

    @always_inline
    def input_scale(self, active_expert_idx: Int) -> Float32:
        return self.c_input_scales_ptr[active_expert_idx]

    @always_inline
    def clamp_alpha(self) -> Float32:
        return self._clamp_alpha

    @always_inline
    def clamp_limit(self) -> Float32:
        return self._clamp_limit

    @always_inline
    def pad_sf_zero_block(
        self,
        sf_block_base: Int,
        tokens_e: Int,
        tid: Int,
        stride: Int,
    ):
        # Per-expert SF block reserves `ceildiv(tokens_e, 128) * 128`
        # rows; rows in `[tokens_e, pad_end)` must be zero so the down-
        # projection's SF reader sees clean values for masked tokens.
        # Live tiles only cover up to `MMA_N` rows per tile in the M
        # direction, so pad rows past that need explicit zeroing here.
        #
        # Optimization: 4 consecutive `i4` slots at fixed (i0, i1, i2,
        # i3) are contiguous in linear memory, so each (row, i1) pair
        # zeroes 4 bytes via a single uint32 store instead of 4
        # separate `set_sf` byte writes.
        var pad_end_local = align_up(tokens_e, SF_MN_GROUP_SIZE)
        var pad_total = pad_end_local - tokens_e
        if pad_total <= 0:
            return
        var total_writes = pad_total * Self.sf_dim1
        var ptr_u32 = self.c_swiglu_scales_ptr.bitcast[UInt32]()
        for i in range(tid, total_writes, stride):
            var pad_idx = i // Self.sf_dim1
            var i1 = i % Self.sf_dim1
            var m = tokens_e + pad_idx + sf_block_base
            var i0 = m // SF_MN_GROUP_SIZE
            var i2 = m % SF_ATOM_M[0]
            var i3 = (m % SF_MN_GROUP_SIZE) // SF_ATOM_M[0]
            var byte_idx = (
                ((i0 * Self.sf_dim1 + i1) * SF_ATOM_M[0] + i2) * SF_ATOM_M[1]
                + i3
            ) * SF_ATOM_K
            ptr_u32.store(byte_idx // 4, UInt32(0))

    def _to_device_type(
        self, mut encoder: Some[DeviceTypeEncoder], target: MutOpaquePointer[_]
    ):
        encoder.encode(self, target)

    @staticmethod
    def get_type_name() -> String:
        return "RealSwiGLUOutput"


# =============================================================================
# Grouped1D1DMatmulKernel - Main kernel struct
# =============================================================================


struct Grouped1D1DMatmulKernel[
    # Core types
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    # C device layout (TensorLayout from caller's TileTensor)
    c_device_layout: TensorLayout,
    # Configuration
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
    # Static N dimension (expert output size)
    static_N: Int,
    # Cluster shape
    cluster_shape: StaticTuple[Int32, 3] = StaticTuple[Int32, 3](1),
    # Epilogue fusion
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    # Programmatic dependent launch level.
    pdl_level: PDLLevel = PDLLevel(),
    # When True, treats adjacent matmul-N column pairs (2i, 2i+1) as
    # (gate, up) and emits packed FP4/FP8 plus a 5D scale tile from the
    # epilogue instead of the BF16 GMEM store. Caller pre-permutes W on
    # N with `sigma(2i)=i, sigma(2i+1)=D+i` (D = output hidden dim,
    # N = 2D). Default False keeps every code path bit-identical to the
    # original BF16-output kernel.
    fuse_swiglu: Bool = False,
    # SwiGLUOutput trait impl. Default `NullSwiGLUOutput[]` is
    # zero-sized so the kernel ABI is unchanged when fusion is off.
    SwiGLUOutputT: SwiGLUOutput = NullSwiGLUOutput[],
    # When True (default), the SMEM scatter rounds through bf16 to
    # match the chained reference path byte-exact. False preserves fp32
    # across the scatter, slightly more accurate but not byte-identical.
    swiglu_match_bf16: Bool = True,
    # Diagnostic: strip the cooperative compute body while keeping the
    # TMEM load, SMEM scatter, and barrier structure, so the EPI cost
    # decomposes into structure vs compute. Output is invalid when True.
    swiglu_disable_compute: Bool = False,
    # When True, gates per-CTA trace records on `comptime if`. When
    # False (default) every record site strips at compile time and the
    # kernel emits byte-identical PTX to an uninstrumented build.
    swiglu_enable_trace: Bool = False,
    TraceBufT: TraceBuf = NullTrace,
    # When True, the epilogue uses an in-place register path: each
    # thread shuffles its (gate, up) fragments across lanes and stores
    # quantized output directly, skipping the bf16 SMEM scratchpad.
    # MXFP8 additionally exchanges per-warp half-block amaxes through
    # SMEM (its 32-elt SF block spans a warp pair) and requires the
    # cta_group=1 layout. False (default) keeps the cooperative
    # scatter path.
    swiglu_use_inplace: Bool = False,
    c_device_engine: TensorEngine = DefaultEngine[element_width=1],
    offsets_engine: TensorEngine = DefaultEngine[element_width=1],
    a_scale_offsets_engine: TensorEngine = DefaultEngine[element_width=1],
    expert_ids_engine: TensorEngine = DefaultEngine[element_width=1],
    expert_scales_engine: TensorEngine = DefaultEngine[element_width=1],
]:
    """Grouped 1D-1D block-scaled matmul kernel.

    Uses 3-warp specialization (Load, MMA, Epilogue) with grid-constant TMAs.
    Work distribution via GroupedWorkIterator1D1D using offset-based
    addressing.

    Parameters:
        a_type: Element `DType` of the A (left) input tensor.
        b_type: Element `DType` of the B (right) input tensor.
        c_type: Element `DType` of the C output tensor.
        sfa_dtype: `DType` of the A-side block scale factors.
        sfb_dtype: `DType` of the B-side block scale factors.
        c_device_layout: `TensorLayout` of the caller's C `TileTensor`,
            used for bounds-checked GMEM stores.
        transpose_b: When `True`, B is stored transposed; the only
            layout this kernel supports.
        config: `BlockScaledMatmulConfig` carrying tile shapes, pipeline
            stage counts, swizzle modes, and cluster geometry.
        static_N: Expert output size (the N dimension), known at compile
            time for TMA descriptor construction.
        cluster_shape: 3D `(x, y, z)` cluster shape passed to the
            `nvvm.cluster_dim` metadata (defaults to `(1, 1, 1)`).
        elementwise_compute_lambda_fn: Optional fused elementwise lambda
            applied in the epilogue (defaults to `None`).
        pdl_level: Programmatic dependent launch level controlling
            cross-grid ordering fences (defaults to `PDLLevel()`).
        fuse_swiglu: When `True`, treats adjacent matmul-N column pairs
            `(2i, 2i+1)` as `(gate, up)` and emits packed FP4/FP8 plus a
            5D scale tile instead of the BF16 GMEM store (defaults to
            `False`).
        SwiGLUOutputT: `SwiGLUOutput` trait impl carrying the fused
            output destinations; `NullSwiGLUOutput[]` is zero-sized so
            the kernel ABI is unchanged when fusion is off (defaults to
            `NullSwiGLUOutput[]`).
        swiglu_match_bf16: When `True` (default), the SMEM scatter rounds
            through `bfloat16` to match the chained reference path
            byte-exact; `False` preserves `float32` for slightly more
            accuracy.
        swiglu_disable_compute: Diagnostic flag that strips the
            cooperative compute body while keeping the TMEM load, SMEM
            scatter, and barrier structure; output is invalid when
            `True` (defaults to `False`).
        swiglu_enable_trace: When `True`, gates per-CTA trace records on
            `comptime if`; when `False` (default) every record site
            strips at compile time and the kernel emits byte-identical
            PTX to an uninstrumented build.
        TraceBufT: `TraceBuf` trait impl for diagnostic timing;
            `NullTrace` is zero-sized so the kernel ABI is unchanged
            when tracing is off (defaults to `NullTrace`).
        swiglu_use_inplace: When `True`, the epilogue uses an in-place
            register path that skips the `bfloat16` SMEM scratchpad via
            cross-lane shuffles; `False` (default) keeps the cooperative
            scatter path.
        c_device_engine: Engine of the C `TileTensor`.
        offsets_engine: Engine of the per-expert offsets
            `TileTensor`.
        a_scale_offsets_engine: Engine of the per-expert A-scale
            offsets `TileTensor`.
        expert_ids_engine: Engine of the expert-IDs `TileTensor`.
        expert_scales_engine: Engine of the expert-scales
            `TileTensor`.
    """

    # ========== Derived Constants ==========

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    comptime accum_type = DType.float32
    comptime cta_group = Self.config.cta_group

    comptime CLUSTER_M: Int = Self.config.cluster_shape[0]
    comptime CLUSTER_N: Int = Self.config.cluster_shape[1]
    comptime CLUSTER_SIZE = Self.CLUSTER_M * Self.CLUSTER_N

    # ========== Thread/Warp Organization ==========

    comptime num_output_warps = 4
    # SFB warps are only launched on the decode (MMA_N < 64) path; on the
    # prefill / 2SM path (MMA_N >= 64) they are compile-time elided and the
    # scheduler warp takes warp 6 instead of warp 11, saving 160 idle threads.
    comptime WarpRole = WarpRole1D1D[
        Self.MMA_N < 64, num_epi_warps=Self.num_output_warps
    ]
    comptime NUM_THREADS = Self.WarpRole.TOTAL_THREADS

    # ========== Pipeline Configuration ==========

    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_output_stages: Int = Self.config.num_output_stages

    # SFB N dimension aligned up to SF_MN_GROUP_SIZE (e.g. 64 → 128).
    # Used for TMA descriptors and SMEM layout (both need the full SF group).
    comptime SFB_N_ALIGNED = align_up(Self.MMA_N, SF_MN_GROUP_SIZE)

    # TMEM configuration — stride matches MMA output width for scaled kernels.
    # SFB TMEM width must be SFB_N_ALIGNED (not MMA_N) because
    # _copy_sf_to_tmem_tt writes SF_MN_GROUP_SIZE//32 = 4 columns per
    # SF group regardless of MMA_N. Matches the sm100/block_scaled kernel.
    comptime NUM_TMEM_COLS = 512
    comptime SFA_NUM_COLS = Self.config.num_sf_k_tiles * (Self.BM // 32)
    comptime SFB_NUM_COLS = Self.config.num_sf_k_tiles * (
        Self.SFB_N_ALIGNED // 32
    )
    comptime stage_stride_cols = Self.MMA_N

    # Output pipeline config (bundles accum stages, stride, and cta_group)
    comptime opc = OutputPipelineConfig(
        Self.num_accum_pipeline_stages,
        Self.stage_stride_cols,
        Self.cta_group,
    )

    # ========== Barrier Arrival Counts ==========

    comptime _accum_barrier_counts = compute_accum_barrier_counts[
        Self.WarpRole.NUM_EPILOGUE_THREADS, Self.cta_group
    ]()
    comptime accum_pipeline_producer_arv_count = Self._accum_barrier_counts[0]
    comptime accum_pipeline_consumer_arv_count = Self._accum_barrier_counts[1]

    # ========== Shared Memory Type ==========

    comptime SmemType = Grouped1D1DSmem[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.transpose_b,
        config=Self.config,
    ]

    # ========== MMA Operation Type ==========

    comptime MmaOp = MmaOpSM100_BlockScaled_SS[
        Self.c_type,
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.config.scaling_kind,
        Self.config.block_tile_shape,
        Self.config.mma_shape,
        accum_type=Self.accum_type,
        cta_group=Self.cta_group,
        cluster_shape=Self.config.cluster_shape,
        a_swizzle=Self.config.a_swizzle,
        b_swizzle=Self.config.b_swizzle,
        transpose_b=Self.transpose_b,
    ]

    # ========== Tile Pipeline Types ==========
    # TileTensor-native payload - passed directly to TMA/MMA

    comptime TilePayload = BlockScaledTilePayload[
        Self.a_type,
        Self.b_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        IndexList[2](
            Self.SmemType.Core.BM, Self.SmemType.Core.BK
        ),  # A tile shape
        IndexList[2](
            Self.SmemType.Core.BN, Self.SmemType.Core.BK
        ),  # B tile shape
        IndexList[2](
            Self.SmemType.Core.SFA_DIM0, Self.SmemType.Core.SFA_DIM1
        ),  # SFA shape
        IndexList[2](
            Self.SmemType.Core.SFB_DIM0, Self.SmemType.Core.SFB_DIM1
        ),  # SFB shape
        Self.SmemType.Core.num_pipeline_stages,
    ]

    comptime InputTilePipelineType = InputTilePipeline[
        Self.TilePayload,
        Self.SmemType.Core.num_group_pipeline_stages,
        Self.config.k_group_size,
    ]

    # ========== TMEM and Output Pipeline Types ==========

    comptime Tmem = TmemAllocation[Self.opc.cta_group]

    comptime TmemRegion = BlockScaledTmem[
        Self.accum_type,
        Self.MMA_M,
        Self.MMA_N,
        Self.num_accum_pipeline_stages,
        Self.sfa_dtype,
        Self.BM,
        Self.num_pipeline_stages,
        cta_group=Self.cta_group,
        num_sf_k_tiles=Self.config.num_sf_k_tiles,
        SFB_N=Self.SFB_N_ALIGNED,
    ]

    comptime OutputPipeline = OutputTilePipeline[Self.opc]

    comptime TmemDealloc = TmemDeallocBarrier[Self.opc.cta_group]

    # ========== Warp Context Types ==========

    comptime MmaEpilogueSync = WarpGroupBarrier[
        Self.WarpRole.NUM_MMA_THREADS + Self.WarpRole.NUM_EPILOGUE_THREADS, 1
    ]

    # Barrier for MMA+SFB sync (MMA_N < 64 only, barrier_id=2)
    # SFB load warps wait for MMA to allocate TMEM before reading the address.
    comptime MmaSfbSync = WarpGroupBarrier[
        Self.WarpRole.NUM_MMA_THREADS + Self.WarpRole.NUM_SFB_LOAD_THREADS, 2
    ]

    comptime MmaCtx = MmaWarpContext[
        Self.opc,
        Self.WarpRole.NUM_MMA_THREADS,
        Self.WarpRole.NUM_EPILOGUE_THREADS,
    ]

    comptime EpilogueCtx = EpilogueWarpContext[
        Self.opc,
        Self.WarpRole.NUM_MMA_THREADS,
        Self.WarpRole.NUM_EPILOGUE_THREADS,
    ]

    # ========== Tile Writer Type ==========

    comptime TileWriterType = TileWriter[
        a_type=Self.a_type,
        accum_type=Self.accum_type,
        block_tile_shape=Self.config.block_tile_shape,
        mma_shape=Self.config.mma_shape,
        opc=Self.opc,
        c_swizzle=Self.config.c_swizzle,
        transpose_c=Self.config.AB_swapped,
        c_smem_dim0=Self.SmemType.Core.OutputM,
        c_smem_dim1=Self.SmemType.Core.OutputN,
        num_output_stages=Self.config.num_output_stages,
        num_output_warps=Self.num_output_warps,
        batched=False,  # 1D-1D uses 2D coordinates with bounds checking
        problem_n=Self.static_N,
    ]

    # ========== Work Iterator Type ==========

    comptime WorkIterator = GroupedWorkIterator1D1D[
        static_N=Self.static_N,
        tile_shape=Self.config.block_tile_shape,
        cluster=Self.config.cluster_shape,
        cta_group=Self.cta_group,
        AB_swapped=Self.config.AB_swapped,
        OffsetsEngine=Self.offsets_engine,
        ExpertIdsEngine=Self.expert_ids_engine,
        ExpertScalesEngine=Self.expert_scales_engine,
    ]

    # ========== TMA Load Size Constants ==========

    # TMA transaction sizes count the bytes the copy engine READS from global
    # memory. An unpacked-FP4 operand occupies BK shared-memory bytes but is
    # sourced from half as many packed ones, and a barrier told to expect the
    # shared-memory figure never completes.
    comptime a_gmem_bytes_per_elem_recip = 2 if _is_packed_fp4[
        Self.a_type, Self.b_type
    ]() else 1
    comptime b_gmem_bytes_per_elem_recip = 2 if _is_packed_fp4[
        Self.b_type, Self.a_type
    ]() else 1
    comptime a_expected_bytes = Self.BM * Self.BK * size_of[
        Self.a_type
    ]() // Self.a_gmem_bytes_per_elem_recip
    comptime b_expected_bytes = Self.BN * Self.BK * size_of[
        Self.b_type
    ]() // Self.b_gmem_bytes_per_elem_recip
    comptime sfa_expected_bytes = Self.SmemType.Core.sfa_smem_layout.size() * size_of[
        Self.sfa_dtype
    ]()
    comptime sfb_expected_bytes = Self.SmemType.Core.sfb_smem_layout.size() * size_of[
        Self.sfb_dtype
    ]()

    # For MMA_N < 64, SFB is loaded by the dedicated SfbTMALoad warp
    # on a separate pipeline, so exclude it from the input pipeline bytes.
    comptime input_expected_bytes = Self.cta_group * (
        Self.a_expected_bytes
        + Self.b_expected_bytes
        + Self.sfa_expected_bytes
        + (Self.sfb_expected_bytes if Self.MMA_N >= 64 else 0)
    ) * Self.config.k_group_size

    # ========== TMA Layouts (computed from config, new Layout types) ==========

    comptime _tma_tile_dims = compute_tma_tile_dims[
        Self.BM,
        Self.BN,
        Self.MMA_M,
        Self.OutputM,
        Self.CLUSTER_M,
        Self.CLUSTER_N,
        Self.cta_group,
        AB_swapped=Self.config.AB_swapped,
    ]()
    comptime a_tile_dim0 = Self._tma_tile_dims[0]
    comptime b_tile_dim0 = Self._tma_tile_dims[1]
    comptime a_swizzle_elems = Self.config.a_swizzle.bytes() // size_of[
        Self.a_type
    ]()
    comptime b_swizzle_elems = Self.config.b_swizzle.bytes() // size_of[
        Self.b_type
    ]()
    comptime c_swizzle_elems = Self.config.c_swizzle.bytes() // size_of[
        Self.c_type
    ]()

    # C tile shape -- same logic as default/block_scaled kernels
    comptime c_tile_dim0 = Self._tma_tile_dims[2]
    comptime c_tile_dim1 = Self.c_swizzle_elems if (
        Self.config.AB_swapped
    ) else Self.OutputN

    # A, B, C: 2D TMA layouts
    comptime ATileLayout = static_row_major[Self.a_tile_dim0, Self.BK]
    comptime ADescLayout = static_row_major[
        Self.a_tile_dim0, Self.a_swizzle_elems
    ]
    comptime BTileLayout = static_row_major[Self.b_tile_dim0, Self.BK]
    comptime BDescLayout = static_row_major[
        Self.b_tile_dim0, Self.b_swizzle_elems
    ]
    comptime CTileLayout = static_row_major[Self.c_tile_dim0, Self.c_tile_dim1]
    # When c_swizzle is SWIZZLE_NONE (MMA_N=8), c_swizzle_elems is 0.
    # The TMA descriptor dim1 must equal the tile dim1 in that case.
    comptime c_desc_dim1 = Self.c_tile_dim1 if Self.c_swizzle_elems == 0 else Self.c_swizzle_elems
    comptime CDescLayout = static_row_major[Self.c_tile_dim0, Self.c_desc_dim1]

    # SFA, SFB: 4D uint16 TMA layouts (batch=1 prefix) to avoid 2× TMA overfetch.
    # SM100 TMA rounds boxDim[0] to 32B min; old innermost=16B caused 2× fetch.
    # Reinterpret as uint16, merge SF_ATOM_M[0] and SF_ATOM_M[1]*SF_ATOM_K into
    # sf_atom_u16 = 256 uint16 = 512B innermost, well above 32B minimum.
    comptime sf_tma_dtype = DType.uint16
    comptime sf_atom_u16 = (
        SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    ) // 2  # 256 uint16 = 512 bytes

    comptime SFATileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            1,
            Self.BM // SF_MN_GROUP_SIZE,
            Self.config.num_sf_k_tiles,
            Self.sf_atom_u16,
        ]
    ]
    comptime SFADescLayout = Self.SFATileLayout

    # SFB TMA tile: for MMA_N < 64 load 1 k-atom at a time with only
    # MMA_N rows; for MMA_N >= 64 unchanged (full atom, all k-atoms).
    comptime SFB_TMA_ROWS = Self.MMA_N if Self.MMA_N < SF_ATOM_M[
        0
    ] else SF_ATOM_M[0]
    comptime SFB_TMA_K_ATOMS = 1 if Self.MMA_N < 64 else Self.config.num_sf_k_tiles
    comptime sfb_atom_u16 = (Self.SFB_TMA_ROWS * SF_ATOM_M[1] * SF_ATOM_K) // 2
    comptime SFBTileLayout = RowMajorLayout[
        *_IntToComptimeInt[
            1,
            Self.SFB_N_ALIGNED // SF_MN_GROUP_SIZE,
            Self.SFB_TMA_K_ATOMS,
            Self.sfb_atom_u16,
        ]
    ]
    comptime SFBDescLayout = Self.SFBTileLayout

    # TMA operation types
    comptime ATmaOp = TmaOpType[Self.a_type, Self.ATileLayout, Self.ADescLayout]
    comptime BTmaOp = TmaOpType[Self.b_type, Self.BTileLayout, Self.BDescLayout]
    comptime CTmaOp = TmaOpType[Self.c_type, Self.CTileLayout, Self.CDescLayout]
    comptime SFATmaOp = TmaOpType[
        Self.sf_tma_dtype, Self.SFATileLayout, Self.SFADescLayout
    ]
    comptime SFBTmaOp = TmaOpType[
        Self.sf_tma_dtype, Self.SFBTileLayout, Self.SFBDescLayout
    ]

    # 1D data TileTensor types (offsets, expert IDs, scales)
    comptime OffsetsTile = TileTensor[
        .uint32, GMEMLayout1D, MutAnyOrigin, Engine=Self.offsets_engine
    ]
    comptime AScaleOffsetsTile = TileTensor[
        .uint32,
        GMEMLayout1D,
        MutAnyOrigin,
        Engine=Self.a_scale_offsets_engine,
    ]
    comptime ExpertIdsTile = TileTensor[
        .int32, GMEMLayout1D, MutAnyOrigin, Engine=Self.expert_ids_engine
    ]
    comptime ExpertScalesTile = TileTensor[
        .float32, GMEMLayout1D, MutAnyOrigin, Engine=Self.expert_scales_engine
    ]

    # C device tensor type (for bounds-checked stores)
    comptime CDeviceTile = TileTensor[
        Self.c_type,
        Self.c_device_layout,
        MutAnyOrigin,
        Engine=Self.c_device_engine,
    ]

    # TMA load size constants (from desc layout dimensions)
    comptime a_tma_load_size = Self.a_tile_dim0 * Self.a_swizzle_elems
    comptime b_tma_load_size = Self.b_tile_dim0 * Self.b_swizzle_elems
    comptime a_tma_rows = Self.a_tile_dim0
    comptime b_tma_rows = Self.b_tile_dim0

    # ========== Validation ==========

    @staticmethod
    def validate_config():
        """Compile-time validation of kernel configuration."""
        comptime assert block_scaled_operands_compatible[
            Self.a_type, Self.b_type
        ](), "A and B types must match for block-scaled GEMM, or be the W4A8 pair"
        comptime assert (
            Self.sfa_dtype == Self.sfb_dtype
        ), "SFA and SFB types must match"
        comptime assert Self.cta_group in (
            1,
            2,
        ), "Only support cta_group == 1 or 2"
        comptime assert Self.transpose_b, "Only support transposed B"
        comptime if Self.MMA_N < 64:
            comptime assert (
                Self.cta_group == 1
            ), "MMA_N < 64 cooperative SFB loading requires cta_group=1"

    # ========== Static Helper Methods ==========

    @staticmethod
    @always_inline
    def init_barriers(
        elect_one_warp: Bool,
        elect_one_thread: Bool,
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c_tma_op: Self.CTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        input_barriers: Self.SmemType.Pipelines.InputBarriers,
        accum_barriers: Self.SmemType.Pipelines.AccumBarriers,
        tmem_dealloc: Self.SmemType.Pipelines.TmemDealloc,
    ):
        """Initialize barriers and prefetch TMA descriptors.

        Args:
            elect_one_warp: When `True`, this warp is elected to run the
                one-time setup.
            elect_one_thread: When `True`, this thread is elected to run
                the one-time setup; both flags must be `True` for the
                prefetch and barrier init to execute.
            a_tma_op: TMA operation descriptor for the A tensor, prefetched
                to the constant cache.
            b_tma_op: TMA operation descriptor for the B tensor, prefetched
                to the constant cache.
            c_tma_op: TMA operation descriptor for the C output tensor,
                prefetched unless `fuse_swiglu` is set.
            sfa_tma_op: TMA operation descriptor for the A-side block scale
                factors, prefetched to the constant cache.
            sfb_tma_op: TMA operation descriptor for the B-side block scale
                factors, prefetched to the constant cache.
            input_barriers: Input pipeline `mbarrier` array to initialize
                for producer-consumer synchronization.
            accum_barriers: Accumulator pipeline `mbarrier` array to
                initialize for MMA-to-epilogue handoff.
            tmem_dealloc: TMEM deallocation barrier to initialize for
                epilogue-to-MMA TMEM recycling.
        """
        if elect_one_warp and elect_one_thread:
            a_tma_op.prefetch_descriptor()
            b_tma_op.prefetch_descriptor()
            # On the fused path the C TMA op is an empty placeholder (results
            # are written through `swiglu_out`), so there is no valid
            # descriptor to prefetch.
            comptime if not Self.fuse_swiglu:
                c_tma_op.prefetch_descriptor()
            sfa_tma_op.prefetch_descriptor()
            sfb_tma_op.prefetch_descriptor()

            init_core_barriers[
                Self.num_group_pipeline_stages,
                Self.num_accum_pipeline_stages,
            ](
                input_barriers.ptr,
                Int32(
                    compute_input_consumer_count[
                        Self.CLUSTER_M, Self.CLUSTER_N, Self.cta_group
                    ]()
                ),
                accum_barriers.ptr,
                Int32(Self.accum_pipeline_producer_arv_count),
                Int32(Self.accum_pipeline_consumer_arv_count),
                tmem_dealloc.ptr,
                Int32(Self.WarpRole.NUM_EPILOGUE_THREADS * Self.cta_group),
            )

        fence_mbarrier_init()
        cluster_sync()

    @staticmethod
    @always_inline
    def _load_sched_ctx(
        ref[AddressSpace.SHARED] smem: Self.SmemType, slot_idx: Int
    ) -> GroupedWorkContext1D1D:
        var s_m: UInt32 = 0
        var s_n: UInt32 = 0
        var s_gidx: UInt32 = 0
        var s_eid: Int32 = -1
        var s_ms: UInt32 = 0
        var s_me: UInt32 = 0
        var s_scale: Float32 = 1.0

        if lane_id() == 0:
            var slot = smem.sched_slots()[slot_idx]
            s_m = slot.m
            s_n = slot.n
            s_gidx = slot.group_idx
            s_eid = slot.expert_id
            s_ms = slot.m_start
            s_me = slot.m_end
            s_scale = slot.expert_scale

        return GroupedWorkContext1D1D(
            warp.broadcast(s_m),
            warp.broadcast(s_n),
            warp.broadcast(s_gidx),
            warp.broadcast(s_eid),
            warp.broadcast(s_ms),
            warp.broadcast(s_scale),
            warp.broadcast(s_me),
        )

    @staticmethod
    @always_inline
    def _consume_sched_ctx(
        ref[AddressSpace.SHARED] smem: Self.SmemType,
        mut sched_ci: Int,
        mut sched_phase: UInt32,
    ) -> GroupedWorkContext1D1D:
        var slot_idx = sched_ci % 2
        smem.sched_full_mbar()[slot_idx].wait(sched_phase)
        if slot_idx == 1:
            sched_phase ^= 1
        var ctx = Self._load_sched_ctx(smem, slot_idx)
        if lane_id() == 0:
            _ = smem.sched_empty_mbar()[slot_idx].arrive()
        sched_ci += 1
        return ctx

    @staticmethod
    @always_inline
    def _compute_iter0_ctx(
        num_active_experts: Int,
        a_offsets: Self.OffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
    ) -> GroupedWorkContext1D1D:
        """Compute this CTA's first tile inline (no scheduler, no mbarrier).

        Each consumer warp calls this independently at kernel start,
        eliminating the latency of waiting for the scheduler warp to
        publish slot 0. Only lane 0 runs the GMEM scan; results are
        broadcast to all lanes via warp.broadcast (which also provides
        the implicit __syncwarp memory fence).
        """
        var s_m: UInt32 = 0
        var s_n: UInt32 = 0
        var s_gidx: UInt32 = 0
        var s_eid: Int32 = -1
        var s_ms: UInt32 = 0
        var s_me: UInt32 = 0
        var s_scale: Float32 = 1.0

        if lane_id() == 0:
            var work_iter = Self.WorkIterator(
                num_active_experts, a_offsets, expert_ids, expert_scales
            )
            var ctx = work_iter.next()
            if not ctx.is_done():
                s_m = ctx.m()
                s_n = ctx.n()
                s_gidx = ctx.group_idx()
                s_eid = ctx.expert_id()
                s_ms = ctx.m_start()
                s_me = ctx.m_end
                s_scale = ctx.expert_scale

        return GroupedWorkContext1D1D(
            warp.broadcast(s_m),
            warp.broadcast(s_n),
            warp.broadcast(s_gidx),
            warp.broadcast(s_eid),
            warp.broadcast(s_ms),
            warp.broadcast(s_scale),
            warp.broadcast(s_me),
        )

    @staticmethod
    @always_inline
    def _sched_terminal_slot() -> SchedulerSlot:
        return SchedulerSlot(
            UInt32(0),
            UInt32(0),
            UInt32(0),
            Int32(-1),
            UInt32(0),
            UInt32(0),
            Float32(1.0),
            UInt32(0),
        )

    @staticmethod
    @always_inline
    def _compute_sched_slot(
        ref[AddressSpace.SHARED] smem: Self.SmemType,
        num_active_experts: Int,
        a_offsets: Self.OffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
        use_group_cache: Bool,
        nbi: UInt32,
        mut grp: UInt32,
        mut cumsum: UInt32,
        mut bstart: UInt32,
    ) -> SchedulerSlot:
        comptime _cta_m = UInt32(Self.WorkIterator.cta_group_tile_shape[0])
        comptime _cta_n = UInt32(Self.WorkIterator.cta_group_tile_shape[1])
        comptime _num_n_blks = Self.WorkIterator.num_static_dim_blocks

        if grp >= UInt32(num_active_experts):
            return Self._sched_terminal_slot()

        var sched_group_offsets = smem.sched_group_offsets()
        var sched_expert_ids = smem.sched_expert_ids()
        var sched_expert_scales = smem.sched_expert_scales()

        var si: UInt32 = 0
        if use_group_cache:
            si = sched_group_offsets[Int(grp)]
        else:
            si = a_offsets[Int(grp)][0]

        var found = False
        var s_m: UInt32 = 0
        var s_n: UInt32 = 0
        var s_gidx: UInt32 = 0
        var s_eid: Int32 = -1
        var s_ms: UInt32 = 0
        var s_me: UInt32 = 0
        var s_scale: Float32 = 1.0

        while grp < UInt32(num_active_experts):
            var ei: UInt32 = 0
            var eid: Int32 = 0
            if use_group_cache:
                ei = sched_group_offsets[Int(grp + 1)]
                eid = sched_expert_ids[Int(grp)]
            else:
                ei = a_offsets[Int(grp + 1)][0]
                eid = expert_ids[Int(grp)][0]
            var gs = ei - si
            if eid < 0 or gs <= 0:
                grp += 1
                si = ei
                continue
            var mb = ceildiv(gs, _cta_m)
            var cum = cumsum + mb
            var bs = cum * _num_n_blks
            if nbi < bs:
                var loc = nbi - bstart
                s_m = (loc % mb) * _cta_m + si
                s_n = (loc / mb) * _cta_n
                s_gidx = grp
                s_eid = eid
                s_ms = si
                s_me = ei
                if use_group_cache:
                    s_scale = sched_expert_scales[Int(grp)]
                else:
                    s_scale = expert_scales[Int(eid)][0]
                found = True
                break
            grp += 1
            cumsum = cum
            bstart = bs
            si = ei

        if not found:
            return Self._sched_terminal_slot()

        return SchedulerSlot(
            s_m,
            s_n,
            s_gidx,
            s_eid,
            s_ms,
            s_me,
            s_scale,
            UInt32(0),
        )

    @staticmethod
    @always_inline
    def _publish_sched_slot(
        ref[AddressSpace.SHARED] smem: Self.SmemType,
        slot_idx: Int,
        sched_slot: SchedulerSlot,
    ):
        smem.sched_slots()[slot_idx] = sched_slot
        _ = smem.sched_full_mbar()[slot_idx].arrive()

    # ========== Kernel Entry Point ==========

    @staticmethod
    @always_inline
    @__llvm_metadata(`nvvm.cluster_dim`=Self.cluster_shape)
    @__llvm_arg_metadata(a_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(b_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(c_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfa_tma_op, `nvvm.grid_constant`)
    @__llvm_arg_metadata(sfb_tma_op, `nvvm.grid_constant`)
    @__name(
        StaticString(Self.config.get_kernel_name())
        + StaticString(
            "_fused_compute_epi" if Self.elementwise_compute_lambda_fn
            is not None else ""
        ),
    )
    def run(
        # Grid-constant TMA descriptors
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        c_tma_op: Self.CTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        # Offset tensors for 1D-1D addressing (TileTensor)
        a_offsets: Self.OffsetsTile,
        a_scale_offsets: Self.AScaleOffsetsTile,
        expert_ids: Self.ExpertIdsTile,
        expert_scales: Self.ExpertScalesTile,
        # C tensor for bounds-checked stores (TileTensor)
        c_device: Self.CDeviceTile,
        # Number of active experts
        num_active_experts: Int32,
        # K dimension for iteration
        K: UInt32,
        # Raw SFB pointer and strides for cp.async path (MMA_N < 64 only).
        # When group_size < SF_MN_GROUP_SIZE, cp.async replaces TMA for SFB.
        sfb_global_ptr: UnsafePointer[Scalar[Self.sfb_dtype], ImmutAnyOrigin],
        sfb_n_stride: Int32,
        sfb_k_tiles: Int32,
        # Fused-SwiGLU+quant output sink. Pass `NullSwiGLUOutput[]()` for
        # non-fused callers — zero-sized, contributes 0 bytes to the
        # kernel ABI when `SwiGLUOutputT=NullSwiGLUOutput`.
        swiglu_out: Self.SwiGLUOutputT,
        # Trace buffer (see `structured_kernels/trace_buf.mojo`). Default
        # `NullTrace()` is zero-sized — 0 bytes of kernel ABI when
        # `swiglu_enable_trace=False`.
        trace_buf: Self.TraceBufT,
    ):
        """Grouped 1D-1D block-scaled GEMM kernel entry point.

        Uses grid-constant TMAs with offset-based addressing for 1D-1D
        layout.

        Args:
            a_tma_op: Grid-constant TMA operation descriptor for
                loading A (left input) tiles from GMEM to SMEM.
            b_tma_op: Grid-constant TMA operation descriptor for
                loading B (right weight) tiles from GMEM to SMEM.
            c_tma_op: Grid-constant TMA operation descriptor for
                storing C output tiles from SMEM to GMEM; an empty
                placeholder on the fused-SwiGLU path.
            sfa_tma_op: Grid-constant TMA operation descriptor for
                loading the A-side block scale factors from GMEM to
                SMEM.
            sfb_tma_op: Grid-constant TMA operation descriptor for
                loading the B-side block scale factors from GMEM to
                SMEM, used on the `MMA_N >= 64` path; the `MMA_N < 64`
                path uses `sfb_global_ptr` with cp.async instead.
            a_offsets: 1D `uint32` `TileTensor` of group start offsets
                into the packed token buffer; `a_offsets[i]` is the
                start of group `i` and `a_offsets[i+1]` is its end.
            a_scale_offsets: 1D `uint32` `TileTensor` of per-group
                A-side scale-factor offsets used to locate the SFA
                tile for each expert group.
            expert_ids: 1D `int32` `TileTensor` of expert IDs indexed
                by group; a negative value marks an empty or terminal
                group.
            expert_scales: 1D `float32` `TileTensor` of per-expert
                output scales applied to the accumulator in the
                epilogue.
            c_device: C output `TileTensor` on device used for
                bounds-checked GMEM stores in the non-fused epilogue.
            num_active_experts: Number of active expert groups in the
                `a_offsets` and `expert_ids` arrays for this launch.
            K: K contraction dimension shared by all experts, in
                elements; drives the k-tile iteration count via
                `ceildiv(K, BK)`.
            sfb_global_ptr: Raw GMEM pointer to the SFB scale-factor
                tensor, used by the cp.async path when `MMA_N < 64`
                and `group_size < SF_MN_GROUP_SIZE`; unused on the TMA
                path.
            sfb_n_stride: Element stride between SFB N-groups in the
                SFB GMEM tensor, used by the cp.async path to index
                rows.
            sfb_k_tiles: Number of valid SFB k-tiles along K; cp.async
                loads past this count zero-fill so out-of-bounds
                k-tiles produce clean scales.
            swiglu_out: Fused SwiGLU plus quantization output sink;
                pass `NullSwiGLUOutput[]()` for the non-fused path
                (zero-sized, 0 bytes of kernel ABI).
            trace_buf: Diagnostic trace buffer for per-tile pipeline
                timing; `NullTrace()` is zero-sized when
                `swiglu_enable_trace=False`.
        """
        var _num_active_experts = Int(num_active_experts)
        var _sfb_n_stride = Int(sfb_n_stride)
        var _sfb_k_tiles = Int(sfb_k_tiles)
        Self.validate_config()

        # ===== Shared Memory Setup =====
        ref smem = external_memory[
            UInt8,
            address_space=.SHARED,
            alignment=128,
        ]().bitcast[Self.SmemType]()[]

        # Get typed tile arrays from SMEM
        var a_tiles = smem.a_tiles()
        var b_tiles = smem.b_tiles()
        var c_tiles = smem.c_tiles()
        var sfa_tiles = smem.sfa_tiles()
        var sfb_tiles = smem.sfb_tiles()

        # Get typed barrier arrays
        var input_barriers = smem.pipelines.input_barriers()
        var accum_barriers = smem.pipelines.accum_barriers()
        var tmem_addr_storage = smem.pipelines.tmem_addr().ptr

        # Create input pipeline with tile payload
        var tile_payload = Self.TilePayload(
            a_tiles, b_tiles, sfa_tiles, sfb_tiles
        )
        var input_pipeline = Self.InputTilePipelineType(
            input_barriers, tile_payload
        )

        # ===== Warp/Thread Election =====
        var elect_one_warp = ufloordiv(thread_idx.x, WARP_SIZE) == 0
        var elect_one_thread = elect_one_sync_with_mask()
        var elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )

        # CTA coordinates in cluster (matches KernelContext pattern)
        var rank_m = block_id_in_cluster.x
        var rank_n = block_id_in_cluster.y

        # Peer CTA coordinates: (peer_id, mma_coord_m, mma_coord_n)
        # Following KernelContext convention:
        #   [0] = rank_m % cta_group  (peer ID within CTA group)
        #   [1] = rank_m // cta_group (MMA coordinate in M)
        #   [2] = rank_n              (MMA coordinate in N)
        var peer_cta_coord = (
            umod(rank_m, Self.cta_group),
            ufloordiv(rank_m, Self.cta_group),
            rank_n,
        )

        # Per-CTA multicast masks (following KernelContext)
        var a_multicast_mask = UInt16(0)

        comptime for i in range(Self.CLUSTER_N):
            a_multicast_mask |= UInt16(1 << (i * Self.CLUSTER_M))
        a_multicast_mask <<= UInt16(rank_m)

        var b_multicast_mask = UInt16(0)

        comptime for i in range(Self.CLUSTER_M // Self.cta_group):
            b_multicast_mask |= UInt16(1 << (i * Self.cta_group))
        b_multicast_mask <<= UInt16(umod(rank_m, Self.cta_group))
        b_multicast_mask <<= UInt16(rank_n * Self.CLUSTER_M)

        var mma_complete_mask = UInt16((1 << Self.cta_group) - 1)

        # K iteration count
        var num_k_iters = ceildiv(Int(K), Self.BK)

        # ===== Barrier Initialization =====
        Self.init_barriers(
            elect_one_warp,
            elect_one_thread,
            a_tma_op,
            b_tma_op,
            c_tma_op,
            sfa_tma_op,
            sfb_tma_op,
            input_barriers,
            accum_barriers,
            smem.pipelines.tmem_dealloc(),
        )

        # Init SFB load barriers (MMA_N < 64 only).
        comptime if Self.MMA_N < 64:
            if elect_one_warp and elect_one_thread:
                var sfb_mbars_ptr = smem.sfb_load_mbars_ptr()
                comptime for i in range(Self.num_group_pipeline_stages):
                    sfb_mbars_ptr[i].init(
                        Int32(Self.WarpRole.NUM_SFB_LOAD_THREADS)
                    )
                ProducerConsumerPipeline[Self.num_group_pipeline_stages](
                    smem.sfb_tma_mbars_ptr()
                ).init_mbars(Int32(Self.MMA_N), Int32(1))

        # Init scheduler: 4 mbarriers (full[0,1] + empty[0,1]).
        if elect_one_warp and elect_one_thread:
            smem.sched_full_mbar()[0].init(Int32(1))
            smem.sched_full_mbar()[1].init(Int32(1))
            # One arrival per consumer warp (lane 0) per slot drain:
            # MMA_N < 64: 4 Epilogue + 1 Load + 1 MMA + 1 SfbTMA + 4 SfbTMEM
            # MMA_N >= 64: 4 Epilogue + 1 Load + 1 MMA (SFB warps idle).
            comptime num_sched_empty_arrivals = 11 if Self.MMA_N < 64 else 6
            smem.sched_empty_mbar()[0].init(Int32(num_sched_empty_arrivals))
            smem.sched_empty_mbar()[1].init(Int32(num_sched_empty_arrivals))

        fence_mbarrier_init()
        cluster_sync()

        # SAFE (PDLLevel.ON) tier: block-wide PDL fence fires after all SMEM
        # / barrier init but before any warp touches GMEM. Orders every
        # warp's subsequent _compute_iter0_ctx read of a_offsets /
        # expert_ids / expert_scales, the scheduler's direct reads, and all
        # TMA loads vs the previous grid. AGGRESSIVE (PDLLevel(2)) skips
        # this fence and lets the Load warp's split-then-wait handle PDL
        # ordering, at the cost of letting the scheduler and iter-0 ctx
        # reads race ahead of the previous grid's writes.
        comptime if Self.pdl_level == PDLLevel.ON:
            wait_on_dependent_grids()

        var mma_op = Self.MmaOp()

        # ===== TMA LOAD WARP =====
        # For cta_group=2: BOTH CTAs run the production loop to keep
        # pipeline state in sync. UMMA multicast arrives on both
        # CTAs' EMPTY barriers, so both must advance through stages
        # to match. Inside load_input_tiles, elect_one_cta gates
        # expect_bytes and the cta_group parameter on TMA ops ensures
        # only the leader CTA issues loads.
        if Self.WarpRole.is_load():
            with input_pipeline.producer() as producer:
                var sched_ci: Int = 0
                var sched_phase = UInt32(0)
                # Iter 0: compute inline (no scheduler, no mbarrier).
                var ctx = Self._compute_iter0_ctx(
                    _num_active_experts,
                    a_offsets,
                    expert_ids,
                    expert_scales,
                )
                var prefetched_ctx = ctx
                var has_prefetched_ctx = False
                # PDL: hide `wait_on_dependent_grids` behind the first tile's
                # weight loads. Weights are static across grids so their TMAs
                # can issue before the wait; activations can only issue after.
                # Flag flips False after the first tile so all later tiles
                # take the unified path.
                var pdl_first_tile = True
                # Per-tile counter for the load warp's pipeline trace.
                # DCEs to nothing when `swiglu_enable_trace=False`.
                var tile_idx_load: Int = 0
                while True:
                    if ctx.expert_id() < 0:
                        break

                    # T_LOAD_DISPATCH tile i: top of outer per-tile
                    # iteration, BEFORE producer-pipeline acquire.
                    # Pairs with T_LOAD_START to expose acquire-wait.
                    comptime if Self.swiglu_enable_trace:
                        if (
                            tile_idx_load < SWIGLU_MAX_TRACED_TILES
                            and lane_id() == 0
                        ):
                            trace_buf.store(
                                Int(block_idx.x)
                                * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                + 9 * tile_idx_load
                                + 0,
                                UInt64(global_perf_counter_ns()),
                            )

                    var next_ready = True
                    if num_k_iters > 0:
                        next_ready = producer.try_acquire()

                    for k_tile in range(num_k_iters):
                        with producer.acquire_if_needed(next_ready) as tiles:
                            # T_LOAD_START tile i: AFTER producer-slot
                            # acquire, just before the first actual TMA
                            # issue. Real-work start.
                            comptime if Self.swiglu_enable_trace:
                                if (
                                    k_tile == 0
                                    and tile_idx_load < SWIGLU_MAX_TRACED_TILES
                                    and lane_id() == 0
                                ):
                                    trace_buf.store(
                                        Int(block_idx.x)
                                        * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                        + 9 * tile_idx_load
                                        + 1,
                                        UInt64(global_perf_counter_ns()),
                                    )

                            var did_split = False
                            comptime if Self.pdl_level == PDLLevel(2):
                                if pdl_first_tile:
                                    Self.load_input_tiles(
                                        a_tma_op,
                                        b_tma_op,
                                        sfa_tma_op,
                                        sfb_tma_op,
                                        tiles,
                                        peer_cta_coord,
                                        ctx,
                                        a_scale_offsets,
                                        UInt32(k_tile),
                                        elect_one_cta,
                                        a_multicast_mask,
                                        b_multicast_mask,
                                        load_weights=True,
                                        load_activations=False,
                                    )
                                    wait_on_dependent_grids()
                                    Self.load_input_tiles(
                                        a_tma_op,
                                        b_tma_op,
                                        sfa_tma_op,
                                        sfb_tma_op,
                                        tiles,
                                        peer_cta_coord,
                                        ctx,
                                        a_scale_offsets,
                                        UInt32(k_tile),
                                        elect_one_cta,
                                        a_multicast_mask,
                                        b_multicast_mask,
                                        load_weights=False,
                                        load_activations=True,
                                    )
                                    pdl_first_tile = False
                                    did_split = True

                            if not did_split:
                                Self.load_input_tiles(
                                    a_tma_op,
                                    b_tma_op,
                                    sfa_tma_op,
                                    sfb_tma_op,
                                    tiles,
                                    peer_cta_coord,
                                    ctx,
                                    a_scale_offsets,
                                    UInt32(k_tile),
                                    elect_one_cta,
                                    a_multicast_mask,
                                    b_multicast_mask,
                                )

                            # T_LOAD_END tile i: after the LAST k-tile's
                            # actual TMA issue (still inside the producer
                            # acquire scope). Real-work end.
                            comptime if Self.swiglu_enable_trace:
                                if (
                                    k_tile == num_k_iters - 1
                                    and tile_idx_load < SWIGLU_MAX_TRACED_TILES
                                    and lane_id() == 0
                                ):
                                    trace_buf.store(
                                        Int(block_idx.x)
                                        * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                        + 9 * tile_idx_load
                                        + 2,
                                        UInt64(global_perf_counter_ns()),
                                    )
                        next_ready = True
                        if k_tile + 1 < num_k_iters:
                            next_ready = producer.try_acquire()
                            # Steal the next-tile scheduler slot read only when
                            # the input pipeline is full — hides the handoff
                            # wait behind an unavoidable stall.
                            if not has_prefetched_ctx and not next_ready:
                                prefetched_ctx = Self._consume_sched_ctx(
                                    smem, sched_ci, sched_phase
                                )
                                has_prefetched_ctx = True
                    syncwarp()
                    tile_idx_load += 1

                    if has_prefetched_ctx:
                        ctx = prefetched_ctx
                        has_prefetched_ctx = False
                    else:
                        ctx = Self._consume_sched_ctx(
                            smem, sched_ci, sched_phase
                        )

                producer.drain()

        # ===== MMA WARP =====
        if Self.WarpRole.is_mma():
            var tmem = Self.Tmem.allocate(smem.pipelines.tmem_addr())
            var mma_ctx = Self.MmaCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, mma_complete_mask
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            var tmem_region = Self.TmemRegion(tmem)

            # Signal SFB load warps that TMEM is allocated
            comptime if Self.MMA_N < 64:
                Self.MmaSfbSync.arrive()

            # SFB pipeline state: tracks sfb_load_mbar phase for MMA_N<64.
            # For MMA_N>=64, SFB is loaded via tcgen05_cp inside mma_op.mma().
            var sfb_mbars = smem.sfb_load_mbars_ptr()
            var sfb_pipe_state = PipelineState[Self.num_group_pipeline_stages]()

            # SFB TMA pipeline: MMA is the consumer that signals the
            # SfbTMALoad warp when SMEM is free for reuse.
            var sfb_tma_pipeline = ProducerConsumerPipeline[
                Self.num_group_pipeline_stages
            ](smem.sfb_tma_mbars_ptr())

            with mma_ctx:
                # Iter 0: compute inline (no scheduler, no mbarrier).
                var ctx = Self._compute_iter0_ctx(
                    _num_active_experts,
                    a_offsets,
                    expert_ids,
                    expert_scales,
                )
                var sched_ci: Int = 0
                var sched_phase = UInt32(0)
                # Per-tile counter for the MMA warp's pipeline trace.
                var tile_idx_mma: Int = 0

                while ctx.expert_id() >= 0:
                    if elect_one_cta:
                        # T_MMA_DISPATCH tile i: top of outer per-tile
                        # iteration, BEFORE both pipeline acquires.
                        # Leader CTA only.
                        comptime if Self.swiglu_enable_trace:
                            if (
                                tile_idx_mma < SWIGLU_MAX_TRACED_TILES
                                and lane_id() == 0
                            ):
                                trace_buf.store(
                                    Int(block_idx.x)
                                    * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                    + 9 * tile_idx_mma
                                    + 3,
                                    UInt64(global_perf_counter_ns()),
                                )

                        with mma_ctx.output_pipeline.producer() as output_stage:
                            # T_MMA_OUTPUT_ACQ tile i: AFTER output-pipeline
                            # producer slot has been acquired (i.e. EPI[i-2]
                            # has called AccumBarrier.arrive on its
                            # consumer_mbar); BEFORE input acquire or any
                            # MMA work. Slot 112 + i.
                            comptime if Self.swiglu_enable_trace:
                                if (
                                    tile_idx_mma < SWIGLU_MAX_TRACED_TILES
                                    and lane_id() == 0
                                ):
                                    trace_buf.store(
                                        Int(block_idx.x)
                                        * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                        + _GROUPED_TRACE_MMA_OUTPUT_ACQ_BASE
                                        + tile_idx_mma,
                                        UInt64(global_perf_counter_ns()),
                                    )

                            var tmem_offset = UInt32(output_stage.tmem.offset())

                            var sfb_tmem_adj = Self._compute_sfb_tmem_adj(
                                ctx.m(), ctx.n(), ctx.m_start()
                            )

                            with input_pipeline.consumer() as consumer:
                                var next_ready = True
                                if num_k_iters > 0:
                                    next_ready = consumer.try_acquire()

                                for k_tile in range(num_k_iters):
                                    with consumer.acquire_if_needed(
                                        next_ready
                                    ) as input_tiles:
                                        # T_MMA_INPUT_ACQ tile i: AFTER input-
                                        # pipeline data acquired for k_tile=0;
                                        # BEFORE the SFB-load barrier (if
                                        # MMA_N<64) so the wait between this
                                        # event and T_MMA_START isolates the
                                        # SFB barrier cost. Slot 120 + i.
                                        comptime if Self.swiglu_enable_trace:
                                            if (
                                                k_tile == 0
                                                and tile_idx_mma
                                                < SWIGLU_MAX_TRACED_TILES
                                                and lane_id() == 0
                                            ):
                                                trace_buf.store(
                                                    Int(block_idx.x)
                                                    * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                                    + _GROUPED_TRACE_MMA_INPUT_ACQ_BASE
                                                    + tile_idx_mma,
                                                    UInt64(
                                                        global_perf_counter_ns()
                                                    ),
                                                )

                                        # Wait for SFB load warps to
                                        # finish writing SFB to TMEM.
                                        comptime if Self.MMA_N < 64:
                                            sfb_mbars[
                                                sfb_pipe_state.index()
                                            ].wait(sfb_pipe_state.phase())

                                        # T_MMA_START tile i: AFTER both
                                        # output- and input-pipeline
                                        # acquires, just before the first
                                        # mma issue. Real-work start.
                                        comptime if Self.swiglu_enable_trace:
                                            if (
                                                k_tile == 0
                                                and tile_idx_mma
                                                < SWIGLU_MAX_TRACED_TILES
                                                and lane_id() == 0
                                            ):
                                                trace_buf.store(
                                                    Int(block_idx.x)
                                                    * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                                    + 9 * tile_idx_mma
                                                    + 4,
                                                    UInt64(
                                                        global_perf_counter_ns()
                                                    ),
                                                )

                                        Self.mma(
                                            input_tiles,
                                            mma_op,
                                            tmem_offset,
                                            tmem_region,
                                            UInt32(k_tile),
                                            0,
                                            sfb_tmem_adj,
                                        )

                                        comptime if Self.MMA_N < 64:
                                            sfb_pipe_state.step()

                                            # Signal SfbTMALoad that SMEM
                                            # slot is consumed and can be
                                            # reused.
                                            if elect_one_sync():
                                                _ = sfb_tma_pipeline.consumer_mbar(
                                                    sfb_tma_pipeline.consumer_stage()
                                                )[
                                                    0
                                                ].arrive()
                                            sfb_tma_pipeline.consumer_step()

                                        # T_MMA_END tile i: after the LAST
                                        # mma issue + commit (still inside
                                        # the consumer scope). Real-work
                                        # end.
                                        comptime if Self.swiglu_enable_trace:
                                            if (
                                                k_tile == num_k_iters - 1
                                                and tile_idx_mma
                                                < SWIGLU_MAX_TRACED_TILES
                                                and lane_id() == 0
                                            ):
                                                trace_buf.store(
                                                    Int(block_idx.x)
                                                    * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                                    + 9 * tile_idx_mma
                                                    + 5,
                                                    UInt64(
                                                        global_perf_counter_ns()
                                                    ),
                                                )
                                    next_ready = True
                                    if k_tile + 1 < num_k_iters:
                                        next_ready = consumer.try_acquire()

                    tile_idx_mma += 1
                    ctx = Self._consume_sched_ctx(smem, sched_ci, sched_phase)

        # ===== EPILOGUE WARPS =====
        if Self.WarpRole.is_epilogue():
            Self.MmaEpilogueSync.wait()

            var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
            var epi_ctx = Self.EpilogueCtx(
                tmem,
                Self.OutputPipeline(
                    accum_barriers.ptr, tmem, mma_complete_mask
                ),
                Self.TmemDealloc(smem.pipelines.tmem_dealloc()),
            )

            with epi_ctx:
                # Iter 0: compute inline (no scheduler, no mbarrier).
                var ctx = Self._compute_iter0_ctx(
                    _num_active_experts,
                    a_offsets,
                    expert_ids,
                    expert_scales,
                )
                var sched_ci: Int = 0
                var sched_phase = UInt32(0)
                # Per-tile counter for the epilogue warp's pipeline trace.
                var tile_idx_epi: Int = 0

                while ctx.expert_id() >= 0:
                    # T_EPI_DISPATCH tile i: top of outer per-tile
                    # iteration, BEFORE consumer-pipeline acquire.
                    comptime if Self.swiglu_enable_trace:
                        if (
                            tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                            and lane_id() == 0
                        ):
                            trace_buf.store(
                                Int(block_idx.x)
                                * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                + 9 * tile_idx_epi
                                + 6,
                                UInt64(global_perf_counter_ns()),
                            )

                    with epi_ctx.output_pipeline.consumer() as output_stage:
                        # T_EPI_START tile i: AFTER consumer-pipeline
                        # acquire (accumulator is ready), just before
                        # the actual epilogue body. Real-work start.
                        comptime if Self.swiglu_enable_trace:
                            if (
                                tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                                and lane_id() == 0
                            ):
                                trace_buf.store(
                                    Int(block_idx.x)
                                    * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                    + 9 * tile_idx_epi
                                    + 7,
                                    UInt64(global_perf_counter_ns()),
                                )

                        Self.epilogue(
                            c_tiles,
                            c_tma_op,
                            c_device,
                            output_stage,
                            ctx,
                            a_scale_offsets,
                            swiglu_out,
                            trace_buf,
                            tile_idx_epi,
                        )

                        # T_EPI_END tile i: after Self.epilogue returns,
                        # still inside the consumer scope (before the
                        # output_pipeline release).
                        comptime if Self.swiglu_enable_trace:
                            if (
                                tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                                and lane_id() == 0
                            ):
                                trace_buf.store(
                                    Int(block_idx.x)
                                    * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                                    + 9 * tile_idx_epi
                                    + 8,
                                    UInt64(global_perf_counter_ns()),
                                )
                    tile_idx_epi += 1

                    ctx = Self._consume_sched_ctx(smem, sched_ci, sched_phase)

                comptime if Self.pdl_level > PDLLevel.OFF:
                    launch_dependent_grids()

        # ===== SFB TMA LOAD WARP (MMA_N < 64 only) =====
        # Dedicated warp that loads SFB scale factors from GMEM to SMEM.
        # Dynamically chooses TMA or cp.async based on group_size:
        #   group_size >= SF_MN_GROUP_SIZE (128): TMA (full atom, efficient)
        #   group_size <  SF_MN_GROUP_SIZE (128): cp.async (exact rows, no waste)
        comptime if Self.MMA_N < 64:
            if Self.WarpRole.is_sfb_tma_load():
                # PDL: when AB_swapped, SFB covers activation scales so
                # reads depend on the previous grid's output. The main Load
                # warp's wait covers A/B/SFA but this warp issues SFB TMAs /
                # cp.async independently, so it needs its own wait before
                # touching GMEM. SfbTMEMLoad only reads SMEM and inherits
                # ordering via sfb_tma_pipeline. Gated on AB_swapped for
                # robustness: if a future config has MMA_N<64 with
                # AB_swapped=False, SFB would be weight scales and the
                # wait would be wasteful.
                comptime if (
                    Self.pdl_level == PDLLevel(2) and Self.config.AB_swapped
                ):
                    wait_on_dependent_grids()

                var sfb_tma_pipeline = ProducerConsumerPipeline[
                    Self.num_group_pipeline_stages
                ](smem.sfb_tma_mbars_ptr())

                # Bytes per k-group iteration for TMA path.
                comptime ROW_STRIDE = SF_ATOM_M[1] * SF_ATOM_K
                comptime K_TILE_ELEMS = SF_ATOM_M[0] * ROW_STRIDE
                comptime sfb_single_copy_bytes = (
                    (Self.SFB_N_ALIGNED // SF_MN_GROUP_SIZE)
                    * Self.SFB_TMA_ROWS
                    * ROW_STRIDE
                    * size_of[Self.sfb_dtype]()
                )
                comptime sfb_tma_expected_bytes = (
                    Self.config.num_sf_k_tiles
                    * sfb_single_copy_bytes
                    * Self.config.k_group_size
                )

                # Layout mapping (k_atom, row) → flat SMEM offset within atom.
                comptime sfb_atom_layout = TileLayout(
                    Coord(
                        Idx[Self.config.num_sf_k_tiles],
                        Idx[SF_ATOM_M[0]],
                    ),
                    Coord(
                        Idx[SF_ATOM_M[0] * ROW_STRIDE],
                        Idx[ROW_STRIDE],
                    ),
                )

                # Global memory strides for cp.async path:
                # flat offset = k_tile * K_TILE_ELEMS + row * ROW_STRIDE + col * SF_ATOM_K
                # (K_TILE_ELEMS, ROW_STRIDE, SF_ATOM_K already defined above.)

                # Iter 0: compute inline (no scheduler, no mbarrier).
                var ctx = Self._compute_iter0_ctx(
                    _num_active_experts,
                    a_offsets,
                    expert_ids,
                    expert_scales,
                )
                var sched_ci: Int = 0
                var sched_phase = UInt32(0)

                while ctx.expert_id() >= 0:
                    # Decide TMA vs cp.async based on group_size.
                    var group_size = Int(ctx.m_end) - Int(ctx.m_start())
                    var use_cpasync = group_size < SF_MN_GROUP_SIZE
                    # AB_swapped: B = tokens, SFB covers token dimension.
                    # Only lanes with real tokens do cp.async; rest zero-fill.
                    # Non-swapped: B = weights, SFB covers feature dimension
                    # (always large), so all MMA_N lanes are always needed.
                    var sfb_active_lanes: Int
                    comptime if Self.config.AB_swapped:
                        sfb_active_lanes = min(group_size, Int(Self.MMA_N))
                    else:
                        sfb_active_lanes = Int(Self.MMA_N)

                    # Hoist loop-invariant SF coords outside k_tile loop.
                    # sfb_n_coord must be visible to ALL lanes (cp.async
                    # needs it per-lane), so compute outside elect_one_sync.
                    var a_scale_offset = a_scale_offsets[Int(ctx.group_idx())][
                        0
                    ]
                    var _sfa_coord: Int
                    var sfb_n_coord: Int
                    _sfa_coord, sfb_n_coord = Self._get_sf_coords(
                        ctx.m(),
                        ctx.n(),
                        ctx.expert_id(),
                        a_scale_offset,
                        ctx.m_start(),
                    )

                    # row_in_atom only needed by TMA path (single lane).
                    var row_in_atom: Int = 0
                    if elect_one_sync():
                        comptime if Self.config.AB_swapped:
                            row_in_atom = (
                                Int(ctx.m()) - Int(ctx.m_start())
                            ) % SF_ATOM_M[0]
                        else:
                            row_in_atom = Int(ctx.n()) % SF_ATOM_M[0]

                    # cp.async per-lane addressing (computed by all lanes).
                    # outer = position within SF_MN_GROUP_SIZE (128),
                    # matching the non-grouped cp.async formula.
                    var cp_outer: UInt
                    comptime if Self.config.AB_swapped:
                        cp_outer = (UInt(ctx.m()) - UInt(ctx.m_start())) % UInt(
                            SF_MN_GROUP_SIZE
                        ) + UInt(lane_id())
                    else:
                        cp_outer = UInt(ctx.n()) % UInt(
                            SF_MN_GROUP_SIZE
                        ) + UInt(lane_id())
                    var cp_row_in_atom = cp_outer % UInt(SF_ATOM_M[0])
                    var cp_sub_column = cp_outer / UInt(SF_ATOM_M[0])

                    for k_tile in range(num_k_iters):
                        sfb_tma_pipeline.wait_consumer()
                        var stage = sfb_tma_pipeline.producer_stage()
                        var sfb_tma_mbar = sfb_tma_pipeline.producer_mbar(stage)

                        if use_cpasync:
                            # === cp.async path ===
                            # Each of MMA_N lanes loads SF_ATOM_K bytes per
                            # k-atom directly from GMEM to SMEM.
                            comptime for kg in range(Self.config.k_group_size):
                                var offset = stage * UInt32(
                                    Self.config.k_group_size
                                ) + UInt32(kg)
                                var sfb_smem_tile = sfb_tiles[offset]

                                comptime for k_atom in range(
                                    Self.config.num_sf_k_tiles
                                ):
                                    if lane_id() < Self.MMA_N:
                                        # SMEM offset must match SfbTMEMLoad read pattern:
                                        # outer%32 * ROW_STRIDE + outer/32 * SF_ATOM_K
                                        var smem_offset = (
                                            k_atom * K_TILE_ELEMS
                                            + Int(cp_row_in_atom) * ROW_STRIDE
                                            + Int(cp_sub_column) * SF_ATOM_K
                                        )
                                        var k_tile_base = (
                                            Int(
                                                UInt32(k_tile)
                                                * UInt32(
                                                    Self.config.k_group_size
                                                )
                                                + UInt32(kg)
                                            )
                                            * Self.config.num_sf_k_tiles
                                        )

                                        var global_offset = (
                                            sfb_n_coord * _sfb_n_stride
                                            + (k_tile_base + k_atom)
                                            * K_TILE_ELEMS
                                            + Int(cp_row_in_atom) * ROW_STRIDE
                                            + Int(cp_sub_column) * SF_ATOM_K
                                        )
                                        # cp.async with src_size masking: when
                                        # src_size=0 (OOB k-tile or lane beyond
                                        # group_size), hardware fills SMEM with
                                        # zero. No branch, no overhead.
                                        comptime copy_size = (
                                            SF_ATOM_K
                                            * size_of[Self.sfb_dtype]()
                                        )
                                        var is_valid = (
                                            lane_id() < sfb_active_lanes
                                            and k_tile_base + k_atom
                                            < _sfb_k_tiles
                                        )
                                        async_copy[
                                            size=copy_size,
                                            fill=Scalar[Self.sfb_dtype](0),
                                        ](
                                            (
                                                sfb_global_ptr + global_offset
                                            ).address_space_cast[.GLOBAL](),
                                            (
                                                sfb_smem_tile._storage
                                                + smem_offset
                                            ).address_space_cast[.SHARED](),
                                            src_size=Int32(
                                                copy_size
                                            ) if is_valid else Int32(0),
                                        )

                            # Barrier: all MMA_N lanes arrive uniformly.
                            # Lanes that did zero-fill (not cp.async) commit
                            # an empty async group — 0 bytes, harmless.
                            if lane_id() < Self.MMA_N:
                                async_copy_arrive(sfb_tma_mbar[0].unsafe_ptr())
                                _ = sfb_tma_mbar[0].arrive()
                        else:
                            # === TMA path (adapted for MMA_N arrive count) ===
                            # Lane 0: expect_bytes (1 arrive) + TMA copies.
                            # Lanes 1..MMA_N-1: bare arrive().
                            if lane_id() == 0:
                                sfb_tma_mbar[0].expect_bytes(
                                    Int32(sfb_tma_expected_bytes)
                                )

                                comptime for kg in range(
                                    Self.config.k_group_size
                                ):
                                    var offset = stage * UInt32(
                                        Self.config.k_group_size
                                    ) + UInt32(kg)
                                    var sfb_smem_tile = sfb_tiles[offset]

                                    comptime for k_atom in range(
                                        Self.config.num_sf_k_tiles
                                    ):
                                        var smem_offset = Int(
                                            sfb_atom_layout(
                                                Coord(
                                                    Idx[k_atom],
                                                    Int64(row_in_atom),
                                                )
                                            )
                                        )

                                        var atom_dst = TileTensor(
                                            sfb_smem_tile._storage
                                            + smem_offset,
                                            row_major[
                                                Self.SFB_TMA_ROWS, ROW_STRIDE
                                            ](),
                                        )

                                        var k_tile_base = (
                                            Int(
                                                UInt32(k_tile)
                                                * UInt32(
                                                    Self.config.k_group_size
                                                )
                                                + UInt32(kg)
                                            )
                                            * Self.config.num_sf_k_tiles
                                        )

                                        var atom_dst_u16 = TileTensor[
                                            Self.sf_tma_dtype,
                                            type_of(atom_dst).LayoutType,
                                            MutAnyOrigin,
                                            address_space=.SHARED,
                                        ](
                                            rebind[
                                                UnsafePointer[
                                                    Scalar[Self.sf_tma_dtype],
                                                    MutAnyOrigin,
                                                    address_space=.SHARED,
                                                ]
                                            ](atom_dst._storage),
                                            atom_dst.layout,
                                        )
                                        sfb_tma_op.async_copy_4d[
                                            Self.cta_group
                                        ](
                                            atom_dst_u16,
                                            sfb_tma_mbar[0],
                                            (
                                                row_in_atom
                                                * (
                                                    SF_ATOM_M[1]
                                                    * SF_ATOM_K
                                                    // 2
                                                ),
                                                k_tile_base + k_atom,
                                                sfb_n_coord,
                                                0,
                                            ),
                                        )

                            elif lane_id() < Self.MMA_N:
                                _ = sfb_tma_mbar[0].arrive()

                        sfb_tma_pipeline.producer_step()

                    ctx = Self._consume_sched_ctx(smem, sched_ci, sched_phase)

                # Drain: prevent exit while SfbTMEMLoad is still working.
                comptime for i in range(Self.num_group_pipeline_stages):
                    sfb_tma_pipeline.wait_consumer()
                    sfb_tma_pipeline.producer_step()

        # ===== SFB TMEM LOAD WARPS (MMA_N < 64 only) =====
        # Dedicated warps that read SFB scale factors from SMEM and
        # write them to TMEM via tcgen05_st. Wait on the sfb_tma_pipeline
        # for TMA loads to complete before reading SMEM.
        comptime if Self.MMA_N < 64:
            if Self.WarpRole.is_sfb_load():
                # Wait for MMA warp to allocate TMEM
                Self.MmaSfbSync.wait()

                var tmem = Self.Tmem.from_shared(smem.pipelines.tmem_addr())
                var tmem_region = Self.TmemRegion(tmem)

                # Track the sfb_tma_pipeline (SfbTMALoad→SfbTMEMLoad).
                var sfb_input_pipeline = ProducerConsumerPipeline[
                    Self.num_group_pipeline_stages
                ](smem.sfb_tma_mbars_ptr())
                var sfb_pipe_state = PipelineState[
                    Self.num_group_pipeline_stages
                ]()
                var sfb_mbars = smem.sfb_load_mbars_ptr()

                # Iter 0: compute inline (no scheduler, no mbarrier).
                var ctx = Self._compute_iter0_ctx(
                    _num_active_experts,
                    a_offsets,
                    expert_ids,
                    expert_scales,
                )
                var sched_ci: Int = 0
                var sched_phase = UInt32(0)

                while ctx.expert_id() >= 0:
                    for k_tile in range(num_k_iters):
                        var stage = sfb_input_pipeline.consumer_stage()
                        sfb_input_pipeline.wait_producer()

                        Self._sfb_load_to_tmem(
                            sfb_tiles,
                            tmem_region,
                            stage,
                            UInt32(k_tile),
                            ctx,
                        )

                        _ = sfb_mbars[sfb_pipe_state.index()].arrive()
                        sfb_input_pipeline.consumer_step()
                        sfb_pipe_state.step()

                    ctx = Self._consume_sched_ctx(smem, sched_ci, sched_phase)

        # ===== SCHEDULER WARP =====
        # Sequential single-lane producer: 2-slot ProducerConsumer.
        # Consumers compute iter 0 inline, so the scheduler starts at iter 1:
        # bootstrap publishes iters 1,2 to slots 0,1, then steady-state uses
        # slot = (it-1) % 2 to stay aligned with consumers reading slot = ci % 2.
        if Self.WarpRole.is_scheduler():
            var use_group_cache = (
                _num_active_experts <= Self.SmemType.SCHED_GROUP_CACHE_CAP
            )
            var cta_stride = UInt32(
                ufloordiv(grid_dim.x, Self.config.cta_group)
            )
            var cta_offset = UInt32(
                ufloordiv(block_idx.x, Self.config.cta_group)
            )

            var grp: UInt32 = 0
            var cumsum: UInt32 = 0
            var bstart: UInt32 = 0
            var has_steady_state = Int32(0)

            # --- Bootstrap: iters 1,2 from GMEM (cache not primed yet) ---
            if lane_id() == 0:
                var slot0 = Self._compute_sched_slot(
                    smem,
                    _num_active_experts,
                    a_offsets,
                    expert_ids,
                    expert_scales,
                    False,  # GMEM — cache not primed yet
                    cta_stride + cta_offset,  # iter 1
                    grp,
                    cumsum,
                    bstart,
                )
                Self._publish_sched_slot(smem, 0, slot0)

                if slot0.expert_id >= 0:
                    var slot1 = Self._compute_sched_slot(
                        smem,
                        _num_active_experts,
                        a_offsets,
                        expert_ids,
                        expert_scales,
                        False,  # GMEM — cache not primed yet
                        UInt32(2) * cta_stride + cta_offset,  # iter 2
                        grp,
                        cumsum,
                        bstart,
                    )
                    Self._publish_sched_slot(smem, 1, slot1)

                    if slot1.expert_id >= 0:
                        has_steady_state = Int32(1)

            # --- Prime SMEM group cache (all 32 lanes, fire-and-forget) ---
            # Steady-state empty_mbar.wait() below fences these stores before
            # any cache reads.
            if use_group_cache:
                var sched_group_offsets = smem.sched_group_offsets()
                var sched_expert_ids = smem.sched_expert_ids()
                var sched_expert_scales = smem.sched_expert_scales()
                var lane = Int(lane_id())
                for i in range(lane, _num_active_experts + 1, WARP_SIZE):
                    sched_group_offsets[i] = a_offsets[i][0]
                for i in range(lane, _num_active_experts, WARP_SIZE):
                    var eid = expert_ids[i][0]
                    sched_expert_ids[i] = eid
                    sched_expert_scales[i] = expert_scales[Int(eid)][
                        0
                    ] if eid >= 0 else Float32(1.0)

            # --- Steady-state: use SMEM cache (fast) ---
            # warp.broadcast() is shuffle_idx with full mask — it includes an
            # implicit __syncwarp that fences the cache priming stores above.
            if warp.broadcast(has_steady_state) > 0:
                if lane_id() == 0:
                    var it = Int32(3)
                    var prod_phase = UInt32(0)
                    while True:
                        # Align with consumer's `ci % 2`: iter=ci+1, so the
                        # slot the producer writes is (iter-1)%2.
                        var slot = Int(it - 1) % 2
                        smem.sched_empty_mbar()[slot].wait(prod_phase)
                        if slot == 1:
                            prod_phase ^= 1

                        var sched_slot = Self._compute_sched_slot(
                            smem,
                            _num_active_experts,
                            a_offsets,
                            expert_ids,
                            expert_scales,
                            use_group_cache,
                            UInt32(it) * cta_stride + cta_offset,
                            grp,
                            cumsum,
                            bstart,
                        )
                        Self._publish_sched_slot(smem, slot, sched_slot)

                        if sched_slot.expert_id < 0:
                            break
                        it += 1

    # ========== SFB Load to TMEM (MMA_N < 64) ==========

    @staticmethod
    @always_inline
    def _sfb_load_to_tmem(
        sfb_tiles: Self.SmemType.Core.SFBTileArray,
        tmem_region: Self.TmemRegion,
        stage: UInt32,
        k_tile: UInt32,
        work_ctx: GroupedWorkContext1D1D,
    ):
        """Load SFB scale factors from SMEM to TMEM via tcgen05_st.

        Matches the SFB load pattern from block_scaled_matmul_small_bn.mojo.
        Each of the 4 SFB load warps (128 threads) covers 32 datapaths via
        tcgen05_st[datapaths=32]. Only lanes 0..MMA_N-1 read valid data;
        others write zero (harmless: UMMA only reads dp 0..MMA_N-1).
        """
        comptime k_group_size = Self.config.k_group_size

        comptime for kg in range(k_group_size):
            var offset = stage * UInt32(k_group_size) + UInt32(kg)

            # SFB SMEM tile at this pipeline offset
            var sfb_smem_tile = sfb_tiles[offset]
            var sfb_tmem_offset = UInt32(tmem_region.sfb(Int(offset)).col_addr)

            comptime SFB_TILE_BYTES = SF_ATOM_M[0] * SF_ATOM_M[
                1
            ] * SF_ATOM_K * size_of[Self.sfb_dtype]()

            # Staging values for ALL the k-atoms' stores, scoped past
            # the single store_wait below: tcgen05_st is asynchronous,
            # so its source registers must stay live until the wait —
            # a loop-scoped staging buffer would let the compiler reuse
            # the register while the store is still in flight. Same
            # batched-stores-then-one-wait shape as
            # TmemFragments.store() + wait_store().
            var _sfb_st_vals = Array[
                Array[UInt32, 1],
                Self.config.num_sf_k_tiles,
            ](uninitialized=True)

            comptime for sf_idx in range(Self.config.num_sf_k_tiles):
                var sfb_scales = SIMD[Self.sfb_dtype, SF_ATOM_K]()
                if lane_id() < Self.MMA_N:
                    # Compute N-position within SF group.
                    # work_ctx.n() is in element space (not tile index),
                    # so no multiplication by MMA_N needed.
                    var outer: UInt
                    comptime if Self.config.AB_swapped:
                        # AB_swapped: N position is derived from M
                        var m_in_group = UInt(work_ctx.m()) - UInt(
                            work_ctx.m_start()
                        )
                        outer = m_in_group % UInt(SF_MN_GROUP_SIZE) + UInt(
                            lane_id()
                        )
                    else:
                        outer = UInt(work_ctx.n()) % UInt(
                            SF_MN_GROUP_SIZE
                        ) + UInt(lane_id())

                    var scales_offset = (
                        UInt(sf_idx) * UInt(SFB_TILE_BYTES)
                        + (outer % UInt(SF_ATOM_M[0]))
                        * (UInt(SF_ATOM_M[1]) * UInt(SF_ATOM_K))
                        + (outer / UInt(SF_ATOM_M[0])) * UInt(SF_ATOM_K)
                    )
                    sfb_scales = sfb_smem_tile.raw_load[
                        width=SF_ATOM_K,
                        alignment=align_of[SIMD[Self.sfb_dtype, SF_ATOM_K]](),
                    ](scales_offset)

                # Reconverge the divergent load above before the
                # warp-collective store.
                syncwarp()

                _sfb_st_vals[sf_idx][0] = bitcast[.uint32, 1](sfb_scales)[0]
                tcgen05_st[
                    datapaths=32,
                    bits=32,
                    repeat=1,
                    pack=False,
                ](
                    sfb_tmem_offset + UInt32(sf_idx * (SF_MN_GROUP_SIZE // 32)),
                    _sfb_st_vals[sf_idx],
                )
                # No per-store wait: the stores write disjoint TMEM
                # columns (stride SF_MN_GROUP_SIZE//32, so repeat=N
                # cannot fuse them) and the only consumer-visible
                # ordering point is the caller's mbarrier arrive after
                # this function returns.

            # One wait + fence for the whole batch, ordering every
            # store above before the caller's arrive on the SFB-load
            # mbarrier (which the MMA warp waits on before reading the
            # scales from TMEM).
            tcgen05_store_wait()
            tcgen05_fence_before()
            # Pin the staging registers' liveness past the wait (the
            # in-flight stores read them).
            _ = _sfb_st_vals

    @staticmethod
    @always_inline
    def _get_sf_coords(
        m_coord: UInt32,
        n_coord: UInt32,
        expert_id: Int32,
        a_scale_offset: UInt32,
        m_start: UInt32,
    ) -> Tuple[Int, Int]:
        """Return (sfa_m_coord, sfb_n_coord), swapped when AB_swapped."""
        var expert_sf_coord = (
            Int(expert_id) * ceildiv(Self.static_N, SF_MN_GROUP_SIZE)
            + Int(n_coord) // SF_MN_GROUP_SIZE
        )

        # Use expert-relative position for the SF group index to avoid
        # incorrect floor-division when MMA_N < SF_MN_GROUP_SIZE and the
        # expert starts at a non-SF_MN_GROUP_SIZE-aligned token offset.
        # (m_coord - m_start) // G gives the correct per-expert group;
        # m_start // G + a_scale_offset restores the absolute SF row.
        var token_sf_coord = (
            Int(m_coord - m_start) // SF_MN_GROUP_SIZE
            + Int(m_start) // SF_MN_GROUP_SIZE
            + Int(a_scale_offset)
        )

        comptime if Self.config.AB_swapped:
            return (expert_sf_coord, token_sf_coord)
        else:
            return (token_sf_coord, expert_sf_coord)

    # ========== Load Input Tiles ==========

    @staticmethod
    @always_inline
    def load_input_tiles[
        tiles_origin: MutOrigin,
        //,
        split_txn: Bool = False,
        weight_txn_bytes: Int = 0,
        activation_txn_bytes: Int = 0,
    ](
        a_tma_op: Self.ATmaOp,
        b_tma_op: Self.BTmaOp,
        sfa_tma_op: Self.SFATmaOp,
        sfb_tma_op: Self.SFBTmaOp,
        tiles: ProducerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        peer_cta_coord: Tuple[Int, Int, Int],
        work_ctx: GroupedWorkContext1D1D,
        a_scale_offsets: Self.AScaleOffsetsTile,
        iter_idx: UInt32,
        elect_one_cta: Bool,
        a_multicast_mask: UInt16,
        b_multicast_mask: UInt16,
        load_weights: Bool = True,
        load_activations: Bool = True,
        wr_mbar: Optional[MbarPtr] = None,
    ):
        """Load A, B, SFA, SFB tiles using TMA.

        When PDL splits loads around `wait_on_dependent_grids`, the weight
        loads (A + SFA if AB_swapped, B + SFB otherwise) can issue before
        the wait since weights are static across grids. The activation
        loads (the other pair) must come after. Set `load_weights=True,
        load_activations=False` for the pre-wait call, and the inverse
        for the post-wait call. `expect_bytes` is issued during the
        weight phase (which covers the first call when both phases
        happen, or the only call when the split is not used).

        The phase flags are runtime so a single instantiation of this
        function covers all three use-cases (both, weights-only,
        activations-only). Branches inside are cheap compared to the
        TMA ops and the warp-level election that gates them.

        `split_txn` splits the stage's single transaction count in two:
        the weight side (A + SFA when `AB_swapped`) completes a
        caller-supplied weight-ready barrier carrying
        `weight_txn_bytes`, and the stage's own FULL barrier is reduced
        to `activation_txn_bytes`. Both counts are armed during the
        weight phase, so an activation-only follow-up call for the same
        stage re-arms neither. A consumer that reads A must then join
        the weight-ready barrier as well as the stage barrier. With
        `split_txn=False` (the default) nothing about the emitted code
        changes.

        Parameters:
            tiles_origin: SMEM `MutOrigin` of the producer tiles (inferred
                from `tiles`).
            split_txn: When `True`, account the weight bytes on `wr_mbar`
                and the activation bytes on the stage barrier instead of
                arming the stage barrier with `input_expected_bytes`.
            weight_txn_bytes: Weight-side expected bytes (`split_txn`).
            activation_txn_bytes: Activation-side expected bytes
                (`split_txn`).

        Args:
            a_tma_op: TMA operation descriptor for A tile loads.
            b_tma_op: TMA operation descriptor for B tile loads.
            sfa_tma_op: TMA operation descriptor for A-side block scale
                factor loads.
            sfb_tma_op: TMA operation descriptor for B-side block scale
                factor loads.
            tiles: `ProducerTiles` pipeline handle providing the barrier
                and payload slot for the current stage.
            peer_cta_coord: `(peer_rank_n, peer_rank_m, peer_m_rank)`
                cluster peer ranks used to slice A/B tiles across paired
                CTAs in 2SM mode.
            work_ctx: `GroupedWorkContext1D1D` carrying the current tile's
                m/n coordinates, expert id, group idx, and m bounds.
            a_scale_offsets: Per-group A-side scale factor offsets;
                `a_scale_offsets[group_idx]` gives the SFA row offset.
            iter_idx: Outer k-tile iteration index; the base k coordinate
                is `(iter_idx + j) * BK`.
            elect_one_cta: When `True`, this CTA is the cluster leader and
                issues `expect_bytes` for the input pipeline.
            a_multicast_mask: TMA multicast bitmask selecting which cluster
                CTAs receive the A tile.
            b_multicast_mask: TMA multicast bitmask selecting which cluster
                CTAs receive the B tile.
            load_weights: When `True`, load the weight side (A+SFA if
                `AB_swapped`, else B+SFB) (defaults to `True`).
            load_activations: When `True`, load the activation side
                (defaults to `True`).
            wr_mbar: Weight-ready barrier the weight-side TMAs signal
                under `split_txn`. Must be present whenever `split_txn`
                is set; ignored otherwise.
        """
        comptime assert (
            not split_txn
            or weight_txn_bytes + activation_txn_bytes
            == Self.input_expected_bytes
        ), (
            "split_txn byte counts must reconstruct input_expected_bytes"
            " exactly, or the stage's transactions never balance"
        )
        var peer_rank_n = peer_cta_coord[0]
        var peer_rank_m = peer_cta_coord[1]
        var peer_m_rank = peer_cta_coord[2]

        # M coordinate in contiguous token space
        var m_coord = work_ctx.m()
        var n_coord = work_ctx.n()
        var expert_id = work_ctx.expert_id()
        var group_idx = work_ctx.group_idx()

        var a_gmem_m_coord: Int
        var b_gmem_n_coord: Int

        comptime if Self.config.AB_swapped:
            # A loads weights (b_device): per-CTA weight row offset.
            # peer_rank_n differentiates CTAs in the weight dimension
            # (0 for CTA0, 1 for CTA1 in a 2SM cluster).
            # Each CTA provides BM weight rows to the UMMA.
            a_gmem_m_coord = (
                peer_rank_n * Self.BM
                + Int(n_coord)
                + Int(expert_id) * Self.static_N
            )
            # B loads tokens (a_device): per-CTA token offset.
            # The UMMA combines B rows from both CTAs, so each CTA
            # loads a different portion of the token range.
            b_gmem_n_coord = (
                peer_rank_m * Self.b_tma_rows
                + peer_rank_n * Self.BN
                + Int(m_coord)
            )
        else:
            # Normal: A loads tokens, B loads weights
            a_gmem_m_coord = peer_m_rank * Self.a_tma_rows + Int(m_coord)
            b_gmem_n_coord = (
                peer_rank_m * Self.b_tma_rows
                + peer_rank_n * Self.BN
                + Int(n_coord)
                + Int(expert_id) * Self.static_N
            )

        # A-side TMAs (a_tma_op + sfa_tma_op) load weights when AB_swapped,
        # activations otherwise. B-side (b_tma_op + sfb_tma_op) is the inverse.
        # AB_swapped is comptime so each side resolves to one of the two
        # runtime flags with no extra select cost.
        var load_a_side: Bool
        var load_b_side: Bool
        comptime if Self.config.AB_swapped:
            load_a_side = load_weights
            load_b_side = load_activations
        else:
            load_a_side = load_activations
            load_b_side = load_weights

        if elect_one_sync():
            # expect_bytes is issued once per barrier: tie it to the weight
            # phase so activation-only calls (post-PDL-wait) don't re-expect.
            if load_weights:
                if elect_one_cta:
                    comptime if split_txn:
                        # BOTH counts are armed in the weight phase, so the
                        # `load_weights=False, load_activations=True`
                        # follow-up call for the same stage re-arms
                        # neither and no transaction is left outstanding.
                        wr_mbar.unsafe_value()[0].expect_bytes(
                            Int32(weight_txn_bytes)
                        )
                        tiles.expect_bytes(activation_txn_bytes)
                    else:
                        tiles.expect_bytes(Self.input_expected_bytes)

            var barrier = tiles.barrier()
            # Per-side completion barriers. Both are the stage barrier
            # unless the caller split the transaction, in which case the
            # WEIGHT side (A+SFA when AB_swapped, B+SFB otherwise) moves
            # onto the caller's weight-ready barrier.
            var a_side_barrier = barrier
            var b_side_barrier = barrier
            comptime if split_txn:
                comptime if Self.config.AB_swapped:
                    a_side_barrier = wr_mbar.unsafe_value()
                else:
                    b_side_barrier = wr_mbar.unsafe_value()

            comptime for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)

                # Get tiles as TileTensor
                var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)

                # Peer CTA slice using TileTensor pattern (ptr + layout)
                var a_peer_tt = type_of(a_tt)(
                    a_tt._storage + peer_m_rank * Self.a_tma_load_size,
                    a_tt.layout,
                )
                var b_peer_tt = type_of(b_tt)(
                    b_tt._storage + peer_rank_m * Self.b_tma_load_size,
                    b_tt.layout,
                )

                var k_coord = Int(iter_idx + j) * Self.BK

                # TileTensor directly to TMA (uses TileTensor overload)
                if load_a_side:
                    a_tma_op.async_multicast_load[Self.cta_group](
                        a_peer_tt,
                        a_side_barrier[0],
                        (k_coord, a_gmem_m_coord),
                        a_multicast_mask,
                    )
                if load_b_side:
                    b_tma_op.async_multicast_load[Self.cta_group](
                        b_peer_tt,
                        b_side_barrier[0],
                        (k_coord, b_gmem_n_coord),
                        b_multicast_mask,
                    )

                # Scale factor load with offset
                # TMA 4D now has TileTensor overload - pass tiles directly
                var a_scale_offset = a_scale_offsets[Int(group_idx)][0]

                var sfa_m_coord: Int
                var sfb_n_coord: Int
                # For AB_swapped 2SM, each CTA needs scale factors
                # for its own weight rows. Add per-CTA offset (BM)
                # to the weight coordinate used for SFA lookup.
                var n_coord_sf = n_coord
                comptime if Self.config.AB_swapped:
                    n_coord_sf = UInt32(Int(n_coord) + peer_rank_n * Self.BM)

                sfa_m_coord, sfb_n_coord = Self._get_sf_coords(
                    m_coord,
                    n_coord_sf,
                    expert_id,
                    a_scale_offset,
                    work_ctx.m_start(),
                )

                if load_a_side:
                    # Cast SMEM tile pointers to uint16 for TMA (4D uint16 descriptor).
                    var sfa_tt_u16 = TileTensor[
                        Self.sf_tma_dtype,
                        type_of(sfa_tt).LayoutType,
                        MutAnyOrigin,
                        address_space=.SHARED,
                    ](
                        rebind[
                            UnsafePointer[
                                Scalar[Self.sf_tma_dtype],
                                MutAnyOrigin,
                                address_space=.SHARED,
                            ]
                        ](sfa_tt._storage),
                        sfa_tt.layout,
                    )
                    sfa_tma_op.async_copy_4d[Self.cta_group](
                        sfa_tt_u16,
                        a_side_barrier[0],
                        (
                            0,
                            Int(
                                (iter_idx + j)
                                * UInt32(Self.config.num_sf_k_tiles)
                            ),
                            sfa_m_coord,
                            0,
                        ),
                    )

                # For MMA_N < 64, SFB is loaded by the SfbTMALoad warp.
                comptime if Self.MMA_N >= 64:
                    if load_b_side:
                        var sfb_tt_u16 = TileTensor[
                            Self.sf_tma_dtype,
                            type_of(sfb_tt).LayoutType,
                            MutAnyOrigin,
                            address_space=.SHARED,
                        ](
                            rebind[
                                UnsafePointer[
                                    Scalar[Self.sf_tma_dtype],
                                    MutAnyOrigin,
                                    address_space=.SHARED,
                                ]
                            ](sfb_tt._storage),
                            sfb_tt.layout,
                        )
                        sfb_tma_op.async_copy_4d[Self.cta_group](
                            sfb_tt_u16,
                            b_side_barrier[0],
                            (
                                0,
                                Int(
                                    (iter_idx + j)
                                    * UInt32(Self.config.num_sf_k_tiles)
                                ),
                                sfb_n_coord,
                                0,
                            ),
                        )

    # ========== MMA Operation ==========

    @staticmethod
    @always_inline
    def _compute_sfb_tmem_adj(
        m_coord: UInt32, n_coord: UInt32, m_start: UInt32
    ) -> UInt32:
        """Compute SFB TMEM column adjustment for MMA_N < SF_MN_GROUP_SIZE.

        When MMA_N reads exactly 2 TMEM columns (MMA_N=64 or 192), one SF
        group (128 elements = 4 TMEM columns) spans two adjacent tiles.
        The adjustment selects the correct half.

        For MMA_N < 64, each SF atom covers 32 N positions in one TMEM
        column. The adj selects which column (atom) within the 128-N SF
        group to read: adj = n_in_sf_group // SF_ATOM_M[0].

        Divides by MMA_N (not BN): in 2SM mode BN = MMA_N/2, but both CTAs
        must supply the same adj because the paired UMMA distributes SF data
        internally.
        """
        comptime if Self.MMA_N < 64:
            # SFB is loaded externally to TMEM via dedicated SFB load
            # warps using tcgen05_st. Data is placed at dp 0..MMA_N-1
            # of the base TMEM column, so no adjustment is needed.
            return UInt32(0)
        elif Self.MMA_N % SF_MN_GROUP_SIZE != 0:
            var effective_n: Int
            comptime if Self.config.AB_swapped:
                effective_n = Int(m_coord) - Int(m_start)
            else:
                effective_n = Int(n_coord)
            return UInt32(effective_n // Self.MMA_N % 2) * 2
        else:
            return UInt32(0)

    @staticmethod
    @always_inline
    def mma[
        tiles_origin: MutOrigin,
        //,
    ](
        tiles: ConsumerTiles[
            tiles_origin,
            Self.TilePayload,
            Self.SmemType.Core.num_group_pipeline_stages,
            Self.config.k_group_size,
        ],
        mma_op: Self.MmaOp,
        tmem_addr: UInt32,
        tmem_region: Self.TmemRegion,
        iter_idx: UInt32,
        k_start: UInt32,
        sfb_tmem_adj: UInt32,
    ):
        """Execute MMA operations.

        For MMA_N >= 64: SFB is loaded to TMEM via tcgen05_cp inside
        mma_op.mma().
        For MMA_N < 64: SFB is pre-loaded by dedicated SFB load warps
        via tcgen05_st. The MMA warp waits on sfb_load_mbars before
        entering this function.

        Parameters:
            tiles_origin: Memory origin of the consumer tiles payload
                (inferred).

        Args:
            tiles: Consumer tiles holding the A, B, SFA, and SFB
                payloads for the current k-group iteration.
            mma_op: Block-scaled MMA operation object that issues
                `tcgen05` MMAs and commits the accumulator.
            tmem_addr: TMEM column address of the output accumulator
                for this tile.
            tmem_region: TMEM region allocation tracking the SFA and
                SFB accumulator column addresses.
            iter_idx: Global k-tile iteration index (`k_tile *
                k_group_size + jj`) used to detect the first k
                iteration for accumulator initialization.
            k_start: K-tile index at which accumulation begins for
                this output tile; `is_first_k` is `True` when
                `iter_idx + j` equals this value.
            sfb_tmem_adj: SFB TMEM column adjustment for the current
                tile, computed by `_compute_sfb_tmem_adj`, that
                accounts for the SFB layout offset at this `(m, n,
                m_start)` position.
        """
        if elect_one_sync():
            comptime for jj in range(Self.config.k_group_size):
                var j = UInt32(jj)
                var a_tt, b_tt, sfa_tt, sfb_tt = tiles.payload().get_tile[
                    Self.config.k_group_size
                ](tiles.stage(), jj)
                var tile_idx = (
                    Int(tiles.stage()) * Self.config.k_group_size + jj
                )
                var sfa_tmem_offset = UInt32(tmem_region.sfa(tile_idx).col_addr)
                var sfb_tmem_offset = UInt32(tmem_region.sfb(tile_idx).col_addr)
                var is_first_k = (iter_idx + j) == k_start
                mma_op.mma(
                    a_tt,
                    b_tt,
                    sfa_tt,
                    sfb_tt,
                    tmem_addr,
                    sfa_tmem_offset,
                    sfb_tmem_offset,
                    init_c=is_first_k,
                    sfb_tmem_adj=sfb_tmem_adj,
                )
            mma_op.commit(tiles.mbar())

    # ========== Epilogue ==========

    @staticmethod
    @always_inline
    def _swiglu_quant_epilogue_body(
        c_tiles: Self.SmemType.Core.CTileArray,
        output_stage: Self.TileWriterType.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        m_start: UInt32,
        expert_scale: Float32,
        active_expert_idx: UInt32,
        a_scale_offsets: Self.AScaleOffsetsTile,
        swiglu_out: Self.SwiGLUOutputT,
        trace_buf: Self.TraceBufT,
        tile_idx_epi: Int,
    ):
        """Fused SwiGLU + block-scaled quantization epilogue body.

        Caller must pre-permute `W` on the N axis with `σ(2i)=i, σ(2i+1)=H+i`
        so adjacent output-N positions hold `(gate, up)` pairs.

        Parametric on the carrier's `(SfDtype, SfVectorSize)`:
          - NVFP4: `SfVectorSize=16, SfDtype=NVFP4_SF_DTYPE` (E4M3).
            One thread owns one full 16-element SF block; quant cast is
            `cast_fp32_to_fp4e2m1` (2 elements packed per byte); SF
            scale folds in a per-expert `tensor_sf`.
          - MXFP8: `SfVectorSize=32, SfDtype=MXFP8_SF_DTYPE` (E8M0).
            Two lanes cooperate per 32-element SF block via
            `lane_group_max[num_lanes=2]` (paired-lane amax); quant
            cast is `cast[float8_e4m3fn]` (1 byte per element); SF
            scale is `block_max / 448` rounded to a power of two by
            the E8M0 cast (no per-expert `tensor_sf` because E8M0
            cannot represent non-power-of-2 multipliers losslessly).

        Numerics mirror the standalone reference kernels in
        `ep_comm.mojo`: `fused_silu_nvfp4_interleaved_kernel` for
        NVFP4 and `fused_silu_mxfp8_interleaved_kernel` for MXFP8.
        The c_tiles SMEM region is re-interpreted as fp32 and used as
        a transposing scratchpad (TMEM → fp32 SMEM → cooperative
        SwiGLU+quant → GMEM).
        """
        # TileWriter's comptime aliases would resolve through unbound
        # `tma_origin`, so we recompute the same EpilogueConfig values from
        # kernel-level params.
        comptime transpose_c = Self.config.AB_swapped
        comptime stageN = (Self.OutputM if transpose_c else Self.OutputN)
        comptime BM = Self.BM
        comptime repeats = stageN // 8
        comptime threads_per_row = stageN // repeats // 2
        comptime num_output_warps = Self.num_output_warps
        comptime is_lower_frag_required = not (
            Self.cta_group == 1 and Self.BM == 64
        )
        comptime num_stages = (
            (
                Self.MMA_N // stageN if Self.MMA_M
                == 256 else Self.MMA_N // stageN // 2
            ) if Self.cta_group
            == 2 else Self.MMA_N // stageN
        )
        comptime rep_frag_size = 4 * repeats

        comptime assert BM % 32 == 0, (
            "fused SwiGLU+quant requires BM % 32 == 0 (one whole SF block"
            " per warp slice post-SwiGLU)"
        )
        comptime assert threads_per_row == 4, (
            "fused SwiGLU+quant expects threads_per_row=4 (FragmentCoords);"
            " got "
            + String(threads_per_row)
        )
        comptime assert (
            Self.config.AB_swapped
        ), "fused SwiGLU+quant currently only supports AB_swapped=True"

        # Quant format is picked from the carrier's trait params. MXFP4
        # collides with MXFP8 on (SfVectorSize=32, SfDtype=E8M0) so the
        # assert below explicitly rejects it; a future `QuantKind` on
        # the trait would discriminate cleanly.
        comptime SfDtype = Self.SwiGLUOutputT.SfDtype
        comptime SfVectorSize = Self.SwiGLUOutputT.SfVectorSize
        comptime assert (SfVectorSize == 16 and SfDtype == NVFP4_SF_DTYPE) or (
            SfVectorSize == 32 and SfDtype == MXFP8_SF_DTYPE
        ), (
            "fused SwiGLU+quant only supports NVFP4 (SF=16, E4M3) or"
            " MXFP8 (SF=32, E8M0); MXFP4 is not yet implemented."
        )
        comptime is_mxfp8 = SfVectorSize == 32
        comptime ClampActivation = Self.SwiGLUOutputT.ClampActivation
        # NUM_THREADS_PER_SF=1 for NVFP4, =2 for MXFP8; MXFP8 lanes
        # cooperate via lane_group_max to derive the per-block scale.
        comptime src_width = 16
        comptime NUM_THREADS_PER_SF = SfVectorSize // src_width
        comptime n_threads_per_token = (BM // 2) // src_width
        comptime total_threads = num_output_warps * WARP_SIZE
        comptime work_per_stage = n_threads_per_token * stageN
        # When `swiglu_disable_compute=True` (diagnostic only), zero out
        # iters_per_stage so the cooperative `comptime for it in range(...)`
        # body strips entirely. The TMEM load, SMEM scatter, and both
        # WarpGroupBarrier syncs still execute — isolates "structural epi
        # cost" from "cooperative compute cost". OUTPUT IS INVALID.
        comptime iters_per_stage = 0 if Self.swiglu_disable_compute else ceildiv(
            work_per_stage, total_threads
        )

        # Per-tile pipeline START/END events are recorded by the outer
        # epilogue warp (T_EPI_START / T_EPI_END at 6*tile_idx + 4 / +5).
        # Intra-body sub-phase events were dropped in favor of the
        # per-tile schema; take a separate trace pass for sub-phase
        # timings.

        # tensor_sf is gated out on MXFP8: E8M0 cannot represent
        # non-power-of-2 multipliers, and the MXFP8 dispatch reuses
        # expert_scales as a placeholder pointer the kernel must not
        # dereference.
        var tensor_sf: Float32
        comptime if is_mxfp8:
            tensor_sf = Float32(1.0)
        else:
            tensor_sf = swiglu_out.input_scale(Int(active_expert_idx))
        # Fold recip(6.0) and recip(448.0) at comptime so the per-iter
        # SF math emits a mul, not rcp.approx, on the cooperative
        # critical path (4 epilogue warps = 1 warp/scheduler).
        comptime SIXTH = Float32(1.0 / 6.0)
        comptime E4M3_MAXABS_RECIP = Float32(1.0 / 448.0)

        # Per-expert SF-block base in the 5D scale tile. Computed once per
        # epilogue call; mirrors `ep_comm.mojo:4068-4070`.
        var scales_offset_blocks = Int(a_scale_offsets[Int(active_expert_idx)])
        var sf_block_base = (
            Int(m_start) // SF_MN_GROUP_SIZE + scales_offset_blocks
        ) * SF_MN_GROUP_SIZE
        var m_local_base = Int(m_start)

        comptime _accum_tile_layout = Layout.row_major(BM, stageN)
        comptime AccumTmemArrayLocal = TmemArrayType[
            Self.accum_type,
            _accum_tile_layout,
            num_stages,
            cta_group=Self.cta_group,
        ]
        var accum_tiles = AccumTmemArrayLocal(output_stage.tmem.offset())
        var warp_id_v = get_warp_id()
        var lane_v = lane_id()
        var scale = expert_scale.cast[Self.accum_type]()

        var lane_row = UInt32(lane_v) // UInt32(threads_per_row)
        var lane_col = (UInt32(lane_v) % UInt32(threads_per_row)) * UInt32(2)

        # Layout A/D/F per `epilogue_components.mojo:721-731`.
        var warp_row_offset: UInt32
        comptime if Self.MMA_M == 256 or (
            Self.MMA_M == 128 and Self.cta_group == 1
        ):
            warp_row_offset = UInt32(warp_id_v) * UInt32(32)
        elif Self.MMA_M == 64 and Self.cta_group == 1:
            warp_row_offset = UInt32(warp_id_v) * UInt32(16)
        else:
            warp_row_offset = (UInt32(warp_id_v) % UInt32(2)) * UInt32(32)

        # In-place register-only path (gated comptime; default off).
        # Skips the bf16 SMEM scratchpad: 4 useful lanes per warp
        # (lane_row in {0,2,4,6} at one lane_col) cover all 16 swiglu-N
        # values for the warp's token range. Cross-lane gathers via
        # `warp.shuffle_xor(v, m)` (XOR the source lane id with mask m)
        # pull each thread's partner-frag values:
        #   m=4   pairs (lane_row 0,1), (2,3), (4,5), (6,7)
        #   m=8   pairs lane_row 0 with 2 (after the 4-XOR)
        #   m=16  pairs lane_row 0 with 4
        #   m=24  pairs lane_row 0 with 6
        # Lane row=0 is the writer: it packs the quantized bytes into
        # uint32 words and issues `store_packed_word`.
        #
        # SF ownership: each warp's 16 values are one full NVFP4 SF
        # block (per-warp amax, no exchange). For MXFP8 they are half
        # of a 32-elt SF block whose other half lives in the paired
        # warp (warp_id ^ 1, owning the adjacent 32 raw columns), so
        # the half-maxes are exchanged through SMEM and only the even
        # warp of the pair writes the SF.
        comptime if Self.swiglu_use_inplace:
            comptime if is_mxfp8:
                comptime assert Self.cta_group == 1, (
                    "MXFP8 in-place epilogue requires the cta_group=1"
                    " decode layout (warp_row_offset = warp_id * 32);"
                    " its warp pairing assumes adjacent warps tile one"
                    " 32-elt SF block"
                )
            # MXFP8 cross-warp amax staging: 32 fp32 slots indexed
            # warp * 8 + token. Reuses the c_tiles SMEM, idle in this
            # path (no bf16 scatter).
            var amax_smem = c_tiles[0]._storage.bitcast[Float32]()

            comptime PartialType = Array[Scalar[Self.accum_type], rep_frag_size]
            var lane_row_is_even = (lane_row & UInt32(1)) == UInt32(0)

            comptime for loop_stage in range(num_stages):
                var frags_ip = accum_tiles[loop_stage].load_fragments[repeats]()
                AccumTmemArrayLocal.Tile.wait_load()

                comptime if loop_stage == num_stages - 1:
                    AccumBarrier[Self.cta_group].arrive(
                        output_stage.pipeline, output_stage.index
                    )

                var upper_ip = rebind[PartialType](frags_ip.upper).copy()
                var lower_ip = Array[Scalar[Self.accum_type], rep_frag_size](
                    uninitialized=True
                )
                comptime if is_lower_frag_required:
                    lower_ip = rebind[PartialType](frags_ip.lower).copy()

                # Apply expert_scale + bf16 round-trip per fragment.
                # Batch fp32 -> bf16 in SIMD-2 chunks (`cvt.rn.bf16x2.f32`)
                # to match `tile_writer`'s cast width and stay byte-
                # identical with the standalone-matmul BF16 GMEM output.
                comptime SIMD_CAST_W = 2
                comptime _n_pairs = rep_frag_size // SIMD_CAST_W
                comptime for r0 in range(repeats):
                    comptime for _pair in range(_n_pairs):
                        comptime _off = _pair * SIMD_CAST_W
                        var src_u = SIMD[Self.accum_type, SIMD_CAST_W]()
                        comptime for _j in range(SIMD_CAST_W):
                            src_u[_j] = upper_ip[r0 * 4 + _off + _j]
                        var dst_u = (
                            (src_u * scale)
                            .cast[.bfloat16]()
                            .cast[Self.accum_type]()
                        )
                        comptime for _j in range(SIMD_CAST_W):
                            upper_ip[r0 * 4 + _off + _j] = dst_u[_j]

                        comptime if is_lower_frag_required:
                            var src_l = SIMD[Self.accum_type, SIMD_CAST_W]()
                            comptime for _j in range(SIMD_CAST_W):
                                src_l[_j] = lower_ip[r0 * 4 + _off + _j]
                            var dst_l = (
                                (src_l * scale)
                                .cast[.bfloat16]()
                                .cast[Self.accum_type]()
                            )
                            comptime for _j in range(SIMD_CAST_W):
                                lower_ip[r0 * 4 + _off + _j] = dst_l[_j]

                # SwiGLU compute via `shuffle_xor(v, 4)` partner pair
                # (pairs lanes 0<->4, 1<->5, 2<->6, 3<->7 within each
                # group of 8). Even lane_row treats its frag as gate,
                # the partner as up; both lanes in the pair compute the
                # same `sw_ip` value, so lane_row even is picked as the
                # writer downstream.
                # Storage: per repeat, 8 SwiGLU values per thread.
                comptime n_swiglu_per_repeat = 8 if is_lower_frag_required else 4
                var sw_ip = Array[
                    Scalar[Self.accum_type],
                    n_swiglu_per_repeat * repeats,
                ](uninitialized=True)

                comptime for r1 in range(repeats):
                    comptime for s in range(n_swiglu_per_repeat):
                        comptime is_lower_s = s >= 4
                        comptime f_s = s % 4
                        var local: Scalar[Self.accum_type]
                        comptime if is_lower_s:
                            local = lower_ip[r1 * 4 + f_s]
                        else:
                            local = upper_ip[r1 * 4 + f_s]
                        var partner = warp.shuffle_xor(local, UInt32(4))
                        var gate: Scalar[Self.accum_type]
                        var up: Scalar[Self.accum_type]
                        if lane_row_is_even:
                            gate = local
                            up = partner
                        else:
                            gate = partner
                            up = local
                        comptime if ClampActivation:
                            var alpha_ip = swiglu_out.clamp_alpha()
                            var limit_ip = swiglu_out.clamp_limit()
                            var g_c = min(gate, limit_ip)
                            var u_c = max(min(up, limit_ip), -limit_ip)
                            var sigmoid_ip = recip(
                                Float32(1.0) + exp(-(g_c * alpha_ip))
                            )
                            sw_ip[r1 * n_swiglu_per_repeat + s] = (
                                (u_c + Float32(1.0)) * g_c * sigmoid_ip
                            )
                        else:
                            var sigmoid = recip(Float32(1.0) + exp(-gate))
                            sw_ip[r1 * n_swiglu_per_repeat + s] = (
                                gate * sigmoid * up
                            )

                # ---- Store path: per (warp, token, half) ----
                # The writer lane is lane_row==0 at lane_col=token&~1
                # (lane_col is even; both tokens t=lane_col, t=lane_col+1
                # share the same writer lane via f=0 vs f=1 fragments).
                #
                # For one (token=t, half h) pair we need 8 fp32 values
                # ordered by swiglu-N (within the half-block of 8). The
                # source per swiglu-N is:
                #   sN 0/4 (upper) or 8/12 (lower) -> lane row=0
                #   sN 1/5         or 9/13         -> lane row=2
                #   sN 2/6         or 10/14        -> lane row=4
                #   sN 3/7         or 11/15        -> lane row=6
                # Lane row=0 issues the gather with shuffle_xor masks
                # 8/16/24 (each XORs lane row=0 with row=2/4/6 within
                # one warp's row dimension).

                # Token parity selects the fragment column. f_col_off=0 → f=0,2,4,6 slot pairs;
                # f_col_off=1 → f=1,3,5,7. Both tokens (t=lane_col and lane_col+1) use the same
                # writer lane (row=0); the half h selects upper (slots 0,2) vs lower (slots 4,6).

                # Compute SF and writer-lane gather + store.
                # Per (token, full SF block of 16 swiglu-N): gather all 16
                # fp32 values, compute amax over all 16 (matches the
                # cooperative path), derive ONE SF, quantize lo/hi halves,
                # store both halves + SF.
                comptime for r2 in range(repeats):
                    var stage_m_off = m_abs + UInt32(loop_stage) * UInt32(
                        stageN
                    )

                    comptime for t_par in range(2):
                        # Slot indexing per t_par (token = lane_col + t_par):
                        #   h=0 (sN 0..7): s_a=t_par, s_b=t_par+2
                        #   h=1 (sN 8..15): s_a=t_par+4, s_b=t_par+6
                        comptime s_a_h0 = t_par
                        comptime s_b_h0 = t_par + 2
                        comptime s_a_h1 = t_par + 4
                        comptime s_b_h1 = t_par + 6

                        var own_a_h0 = sw_ip[r2 * n_swiglu_per_repeat + s_a_h0]
                        var own_b_h0 = sw_ip[r2 * n_swiglu_per_repeat + s_b_h0]
                        var own_a_h1 = sw_ip[r2 * n_swiglu_per_repeat + s_a_h1]
                        var own_b_h1 = sw_ip[r2 * n_swiglu_per_repeat + s_b_h1]

                        # 12 cross-lane shuffles: 4 slot values * 3
                        # shuffle_xor masks (8, 16, 24). Row=0 lane
                        # collects values from rows 2/4/6 at the same
                        # lane_col.
                        var p8_a_h0 = warp.shuffle_xor(own_a_h0, UInt32(8))
                        var p8_b_h0 = warp.shuffle_xor(own_b_h0, UInt32(8))
                        var p8_a_h1 = warp.shuffle_xor(own_a_h1, UInt32(8))
                        var p8_b_h1 = warp.shuffle_xor(own_b_h1, UInt32(8))
                        var p16_a_h0 = warp.shuffle_xor(own_a_h0, UInt32(16))
                        var p16_b_h0 = warp.shuffle_xor(own_b_h0, UInt32(16))
                        var p16_a_h1 = warp.shuffle_xor(own_a_h1, UInt32(16))
                        var p16_b_h1 = warp.shuffle_xor(own_b_h1, UInt32(16))
                        var p24_a_h0 = warp.shuffle_xor(own_a_h0, UInt32(24))
                        var p24_b_h0 = warp.shuffle_xor(own_b_h0, UInt32(24))
                        var p24_a_h1 = warp.shuffle_xor(own_a_h1, UInt32(24))
                        var p24_b_h1 = warp.shuffle_xor(own_b_h1, UInt32(24))

                        # 16 fp32 values in swiglu-N order [0..15] for one
                        # (token, SF block).
                        var pack16 = SIMD[Self.accum_type, 16]()
                        pack16[0] = own_a_h0
                        pack16[1] = p8_a_h0
                        pack16[2] = p16_a_h0
                        pack16[3] = p24_a_h0
                        pack16[4] = own_b_h0
                        pack16[5] = p8_b_h0
                        pack16[6] = p16_b_h0
                        pack16[7] = p24_b_h0
                        pack16[8] = own_a_h1
                        pack16[9] = p8_a_h1
                        pack16[10] = p16_a_h1
                        pack16[11] = p24_a_h1
                        pack16[12] = own_b_h1
                        pack16[13] = p8_b_h1
                        pack16[14] = p16_b_h1
                        pack16[15] = p24_b_h1

                        # Per-block amax. NVFP4: the 16 gathered values
                        # are one full SF block, so the per-warp reduce
                        # is final. MXFP8: they are half of a 32-elt
                        # block; exchange half-maxes with the paired
                        # warp through SMEM (shuffles cannot cross
                        # warps). All threads hit both barriers; only
                        # writer lanes touch the slots.
                        var half_max = abs(pack16).reduce_max()
                        var block_max = half_max
                        comptime if is_mxfp8:
                            var slot = Int(
                                UInt32(warp_id_v) * UInt32(8)
                                + lane_col
                                + UInt32(t_par)
                            )
                            if lane_row == UInt32(0):
                                amax_smem[slot] = half_max
                            WarpGroupBarrier[
                                num_output_warps * WARP_SIZE
                            ].sync()
                            if lane_row == UInt32(0):
                                var partner_slot = Int(
                                    (UInt32(warp_id_v) ^ UInt32(1)) * UInt32(8)
                                    + lane_col
                                    + UInt32(t_par)
                                )
                                block_max = max(
                                    half_max, amax_smem[partner_slot]
                                )
                            # Slots are reused on the next (r2, t_par)
                            # round; resync before they are rewritten.
                            WarpGroupBarrier[
                                num_output_warps * WARP_SIZE
                            ].sync()

                        var sf_f32: Float32
                        comptime if is_mxfp8:
                            sf_f32 = block_max * E4M3_MAXABS_RECIP
                        else:
                            sf_f32 = tensor_sf * (block_max * SIXTH)
                        var sf_byte = sf_f32.cast[SfDtype]()
                        var output_scale = Float32(0.0)
                        if block_max != Float32(0.0):
                            comptime if is_mxfp8:
                                output_scale = recip(sf_byte.cast[.float32]())
                            else:
                                output_scale = tensor_sf * recip(
                                    sf_byte.cast[.float32]()
                                )

                        var p_scaled = pack16 * output_scale

                        var token_g = (
                            stage_m_off
                            + UInt32(lane_col)
                            + UInt32(t_par)
                            + UInt32(r2 * 8)
                        )

                        comptime if is_mxfp8:
                            # 16 fp32 -> 16 fp8 bytes -> 4 word stores.
                            var packed16_ip = p_scaled.cast[
                                DType.float8_e4m3fn
                            ]()
                            var byte_base = (
                                Int(n_abs) // 2 + Int(warp_row_offset) // 2
                            )
                            if lane_row == UInt32(0) and token_g < m_end:
                                var packed_u32 = bitcast[.uint32, 4](
                                    packed16_ip
                                )
                                swiglu_out.store_packed_word(
                                    Int(token_g), byte_base, packed_u32[0]
                                )
                                swiglu_out.store_packed_word(
                                    Int(token_g),
                                    byte_base + 4,
                                    packed_u32[1],
                                )
                                swiglu_out.store_packed_word(
                                    Int(token_g),
                                    byte_base + 8,
                                    packed_u32[2],
                                )
                                swiglu_out.store_packed_word(
                                    Int(token_g),
                                    byte_base + 12,
                                    packed_u32[3],
                                )
                                # The even warp of the pair owns the
                                # 32-elt SF block start (its post_col
                                # is 32-aligned).
                                if UInt32(warp_id_v) % UInt32(2) == UInt32(0):
                                    var effective_m = (
                                        Int(token_g)
                                        - m_local_base
                                        + sf_block_base
                                    )
                                    swiglu_out.set_sf(
                                        effective_m,
                                        Int(n_abs) // 2
                                        + Int(warp_row_offset) // 2,
                                        sf_byte,
                                    )
                        else:
                            # 16 fp32 -> 8 nibble-packed bytes -> 2 word
                            # stores. The SF block spans 8 bytes at
                            # (n_abs + warp_row_offset) / 4.
                            var packed_lo = cast_fp32_to_fp4e2m1(
                                p_scaled.slice[8, offset=0]()
                            )
                            var packed_hi = cast_fp32_to_fp4e2m1(
                                p_scaled.slice[8, offset=8]()
                            )
                            var byte_base = (
                                Int(n_abs) // 4 + Int(warp_row_offset) // 4
                            )
                            if lane_row == UInt32(0) and token_g < m_end:
                                swiglu_out.store_packed_word(
                                    Int(token_g), byte_base, packed_lo
                                )
                                swiglu_out.store_packed_word(
                                    Int(token_g), byte_base + 4, packed_hi
                                )
                                var effective_m = (
                                    Int(token_g) - m_local_base + sf_block_base
                                )
                                swiglu_out.set_sf(
                                    effective_m,
                                    Int(n_abs) // 2 + Int(warp_row_offset) // 2,
                                    sf_byte,
                                )

            # Per-expert SF tail-pad zero-fill (in-place path).
            # Triggers on (last m-tile, first n-tile) of the expert
            # so the pad-fill runs exactly once per expert (each
            # expert is processed by multiple (m, n) tiles, and the
            # SF tile is shared across n-tiles).
            if Int(m_abs) + Int(Self.MMA_N) >= Int(m_end) and n_abs == UInt32(
                0
            ):
                var tid_within_epi_ip = UInt32(warp_id_v) * UInt32(
                    WARP_SIZE
                ) + UInt32(lane_v)
                swiglu_out.pad_sf_zero_block(
                    sf_block_base,
                    Int(m_end) - Int(m_start),
                    Int(tid_within_epi_ip),
                    num_output_warps * WARP_SIZE,
                )

            return

        # The epilogue mirrors `fused_silu_nvfp4_interleaved_kernel` in
        # `shmem/ep_comm.mojo` (TMEM → reg → bf16 SMEM → reg → SwiGLU+quant →
        # GMEM); the GMEM round trip is replaced by a SMEM round trip. bf16
        # SMEM matches the unfused kernel's BF16-from-GMEM read pattern
        # exactly, so swiglu_match_bf16 precision is automatic.
        var smem_bf16_ptr = c_tiles[0]._storage.bitcast[BFloat16]()

        comptime PartialType = Array[Scalar[Self.accum_type], rep_frag_size]

        var tid_within_epi = UInt32(warp_id_v) * UInt32(WARP_SIZE) + UInt32(
            lane_v
        )

        # XOR-swizzle on bf16 indices: XOR bits 3..5 with bits 7..9.
        # `Swizzle(bits=3, base=3, shift=4)` realizes the same mapping
        # as `idx ^ (((idx >> 7) & 7) << 3)`. Each ld.shared.b128 reads
        # 8 contiguous bf16; leaving bits 0..2 alone keeps width-aligned
        # loads/stores contiguous, while the bank slot rotates by 4
        # banks per 128 bf16 of row offset. Source range is bits 7..9
        # (not 6..8) because that lifts WRITE bank efficiency from 25%
        # to 50% on this access pattern (READ stays at 33%).
        comptime SWIZZLE_BF = Swizzle(3, 3, 4)

        # SIMD-2 fp32 -> bf16 batched scatter store. Matches the
        # `cvt.rn.bf16x2.f32` cast width used by `tile_writer` so the
        # bf16 SMEM scratchpad is byte-identical to the standalone
        # matmul's BF16 GMEM output (chain reference).
        @always_inline
        @__parameter
        def store_scaled_pair(
            smem_idx_a: UInt32,
            smem_idx_b: UInt32,
            pair_fp32: SIMD[Self.accum_type, 2],
        ):
            var pair_bf = pair_fp32.cast[.bfloat16]()
            smem_bf16_ptr.store(Int(SWIZZLE_BF(smem_idx_a)), pair_bf[0])
            smem_bf16_ptr.store(Int(SWIZZLE_BF(smem_idx_b)), pair_bf[1])

        comptime for loop_stage in range(num_stages):
            var frags = accum_tiles[loop_stage].load_fragments[repeats]()
            AccumTmemArrayLocal.Tile.wait_load()

            # Sub-phase trace: stage 0 only. Records at offset 72 + 5*tile +
            # j (j=0..4). Stages > 0 have uniform structure so we sample
            # stage 0 to keep the trace buffer small. tid_within_epi == 0
            # ensures a single writer; gated comptime so the record DCEs
            # entirely when swiglu_enable_trace is False.
            comptime if Self.swiglu_enable_trace:
                comptime if loop_stage == 0:
                    if (
                        tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                        and tid_within_epi == 0
                    ):
                        trace_buf.store(
                            Int(block_idx.x)
                            * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                            + _GROUPED_TRACE_SUBPHASE_BASE
                            + 5 * tile_idx_epi
                            + 0,
                            UInt64(global_perf_counter_ns()),
                        )

            comptime if loop_stage == num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            var upper_partial = rebind[PartialType](frags.upper).copy()
            var lower_partial = Array[Scalar[Self.accum_type], rep_frag_size](
                uninitialized=True
            )
            comptime if is_lower_frag_required:
                lower_partial = rebind[PartialType](frags.lower).copy()

            # Scatter (apply expert_scale + bf16 round-trip).
            # transpose_c maps (kernel-M, kernel-N) -> (output-N, output-M),
            # so out_m=k_n (token) and out_n=k_m+warp_offset (H position).
            # Batch fragment slots into SIMD-2 chunks so the cast emits
            # `cvt.rn.bf16x2.f32` matching `tile_writer`'s cast width.
            comptime for r in range(repeats):
                comptime for _pair in range(2):  # pair f=0,1 and f=2,3
                    comptime f0 = _pair * 2
                    comptime f1 = f0 + 1
                    comptime frag_row_off = (f0 >> 1) * 8  # same for f0,f1
                    var k_m = lane_row + UInt32(frag_row_off)
                    var k_n_a = lane_col + UInt32(r * 8) + UInt32(0)  # f0
                    var k_n_b = lane_col + UInt32(r * 8) + UInt32(1)  # f1
                    var smem_idx_a = k_n_a * UInt32(BM) + k_m + warp_row_offset
                    var smem_idx_b = k_n_b * UInt32(BM) + k_m + warp_row_offset
                    var pair_u = SIMD[Self.accum_type, 2](
                        upper_partial[r * 4 + f0],
                        upper_partial[r * 4 + f1],
                    )
                    store_scaled_pair(smem_idx_a, smem_idx_b, pair_u * scale)
                    comptime if is_lower_frag_required:
                        var smem_idx_a_l = (
                            k_n_a * UInt32(BM)
                            + k_m
                            + UInt32(16)
                            + warp_row_offset
                        )
                        var smem_idx_b_l = (
                            k_n_b * UInt32(BM)
                            + k_m
                            + UInt32(16)
                            + warp_row_offset
                        )
                        var pair_l = SIMD[Self.accum_type, 2](
                            lower_partial[r * 4 + f0],
                            lower_partial[r * 4 + f1],
                        )
                        store_scaled_pair(
                            smem_idx_a_l, smem_idx_b_l, pair_l * scale
                        )

            # Sub-phase trace: SCATTER_DONE (stage 0 only).
            comptime if Self.swiglu_enable_trace:
                comptime if loop_stage == 0:
                    if (
                        tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                        and tid_within_epi == 0
                    ):
                        trace_buf.store(
                            Int(block_idx.x)
                            * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                            + _GROUPED_TRACE_SUBPHASE_BASE
                            + 5 * tile_idx_epi
                            + 1,
                            UInt64(global_perf_counter_ns()),
                        )

            WarpGroupBarrier[num_output_warps * WARP_SIZE].sync()

            # Sub-phase trace: BAR1_DONE (stage 0 only).
            comptime if Self.swiglu_enable_trace:
                comptime if loop_stage == 0:
                    if (
                        tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                        and tid_within_epi == 0
                    ):
                        trace_buf.store(
                            Int(block_idx.x)
                            * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                            + _GROUPED_TRACE_SUBPHASE_BASE
                            + 5 * tile_idx_epi
                            + 2,
                            UInt64(global_perf_counter_ns()),
                        )

            # Cooperative read + SwiGLU + quant + GMEM store. Threads
            # execute a uniform iteration count (comptime for); the per-store
            # bounds check skips out-of-range tokens.
            var stage_m_offset = m_abs + UInt32(loop_stage) * UInt32(stageN)

            comptime for it in range(iters_per_stage):
                var linear_var = Int(tid_within_epi) + it * total_threads
                var in_bounds = linear_var < work_per_stage
                var token_idx = UInt32(linear_var // n_threads_per_token)
                var hid_idx = UInt32(linear_var % n_threads_per_token)
                var m_global = stage_m_offset + token_idx
                var k_post = hid_idx * UInt32(src_width)
                var k_raw = 2 * k_post

                # The bf16 XOR swizzle preserves only 8-bf16 contiguity
                # (bits 0..2 untouched, bit 3 XORed with row bits), so a
                # wider single load would cross a swizzle boundary. Use
                # width-8 chunks (one ld.shared.b128 each, conflict-free)
                # then deinterleave for the (gate, up) stride-2 split.
                comptime BF16_LOAD_W = 8
                comptime n_loads = (2 * src_width) // BF16_LOAD_W
                var smem_base = UInt32(token_idx) * UInt32(BM) + UInt32(k_raw)
                var pair_bf = SIMD[.bfloat16, 2 * src_width]()
                comptime for li in range(n_loads):
                    var chunk = smem_bf16_ptr.load[width=BF16_LOAD_W](
                        Int(SWIZZLE_BF(smem_base + UInt32(li * BF16_LOAD_W)))
                    )
                    comptime for ci in range(BF16_LOAD_W):
                        pair_bf[li * BF16_LOAD_W + ci] = chunk[ci]
                var pair = pair_bf.cast[.float32]()
                var gate, up = pair.deinterleave()

                # silu(g) and the final SF reciprocal use `recip()`
                # (`rcp.approx.ftz.f32`) rather than fp32 `/`: with only
                # 4 epilogue warps (1 warp/scheduler) IEEE `div.f32`
                # latency does not hide.
                #
                # Clamped variant: g' = min(g, L); u' = clamp(u, -L, L);
                #                  z  = (u' + 1) · g' · σ(g'·α).
                var z: SIMD[Self.accum_type, src_width]
                comptime if ClampActivation:
                    var alpha = swiglu_out.clamp_alpha()
                    var limit = swiglu_out.clamp_limit()
                    var g_clamped = min(gate, limit)
                    var u_clamped = max(min(up, limit), -limit)
                    var sigmoid = recip(
                        Float32(1.0) + exp(-(g_clamped * alpha))
                    )
                    z = (u_clamped + Float32(1.0)) * g_clamped * sigmoid
                else:
                    var sigmoid = recip(Float32(1.0) + exp(-gate))
                    z = gate * sigmoid * up

                # MXFP8 needs cross-lane amax across the 2 lanes that
                # share one 32-elt SF block; NVFP4's 16-elt block fits
                # inside one lane's fragment.
                var thread_max = abs(z).reduce_max()
                var block_max = thread_max
                comptime if NUM_THREADS_PER_SF > 1:
                    block_max = warp.lane_group_max[
                        num_lanes=NUM_THREADS_PER_SF
                    ](thread_max)

                # NVFP4 folds tensor_sf into the SF; MXFP8 (E8M0,
                # exponent-only) cannot, and uses block_max / 448
                # (FP8-E4M3 maxabs) rounded to a power of two.
                var sf_f32: Float32
                comptime if is_mxfp8:
                    sf_f32 = block_max * E4M3_MAXABS_RECIP
                else:
                    sf_f32 = tensor_sf * (block_max * SIXTH)
                var sf_byte = sf_f32.cast[SfDtype]()

                # Algebraic simplification for NVFP4:
                #   recip(sf.fp32() * recip(tensor_sf))
                # = tensor_sf * recip(sf.fp32())
                var output_scale = Float32(0.0)
                if block_max != Float32(0.0):
                    comptime if is_mxfp8:
                        output_scale = recip(sf_byte.cast[.float32]())
                    else:
                        output_scale = tensor_sf * recip(
                            sf_byte.cast[.float32]()
                        )

                var z_scaled = z * output_scale

                # NVFP4 packs 16 fp32 → 8 bytes (2 cvt + 2 word stores);
                # MXFP8 emits 16 fp32 → 16 fp8_e4m3fn bytes (1 cast +
                # 4 word stores). byte_base_global is a multiple of 4
                # under the kernel's MMA geometry, so word stores
                # coalesce on the L2 sector.
                comptime if is_mxfp8:
                    var packed16 = z_scaled.cast[.float8_e4m3fn]()
                    if in_bounds and m_global < m_end:
                        var byte_base_global = Int(n_abs // 2) + Int(k_post)
                        var packed_u32 = bitcast[.uint32, 4](packed16)
                        swiglu_out.store_packed_word(
                            Int(m_global), byte_base_global, packed_u32[0]
                        )
                        swiglu_out.store_packed_word(
                            Int(m_global), byte_base_global + 4, packed_u32[1]
                        )
                        swiglu_out.store_packed_word(
                            Int(m_global), byte_base_global + 8, packed_u32[2]
                        )
                        swiglu_out.store_packed_word(
                            Int(m_global), byte_base_global + 12, packed_u32[3]
                        )
                else:
                    var packed_lo = cast_fp32_to_fp4e2m1(
                        z_scaled.slice[8, offset=0]()
                    )
                    var packed_hi = cast_fp32_to_fp4e2m1(
                        z_scaled.slice[8, offset=8]()
                    )
                    if in_bounds and m_global < m_end:
                        var byte_base_global = Int(n_abs // 4) + Int(
                            k_post // 2
                        )
                        swiglu_out.store_packed_word(
                            Int(m_global), byte_base_global, packed_lo
                        )
                        swiglu_out.store_packed_word(
                            Int(m_global), byte_base_global + 4, packed_hi
                        )

                # SF store: only the SF-owner lane writes. NVFP4 owns
                # its 16-elt block alone; MXFP8's 32-elt block is
                # shared between paired lanes, so only the even lane in
                # the pair (lane_id % 2 == 0) issues the store. set_sf
                # indexes the 5D tile via effective_m so i0 lands in
                # the active expert's reserved block.
                var sf_owns: Bool
                comptime if NUM_THREADS_PER_SF == 1:
                    sf_owns = True
                else:
                    sf_owns = hid_idx % UInt32(NUM_THREADS_PER_SF) == UInt32(0)
                if in_bounds and m_global < m_end and sf_owns:
                    var effective_m = (
                        Int(m_global) - m_local_base + sf_block_base
                    )
                    swiglu_out.set_sf(
                        effective_m,
                        Int(n_abs // 2) + Int(k_post),
                        sf_byte,
                    )

            # Sub-phase trace: COOP_DONE (stage 0 only).
            comptime if Self.swiglu_enable_trace:
                comptime if loop_stage == 0:
                    if (
                        tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                        and tid_within_epi == 0
                    ):
                        trace_buf.store(
                            Int(block_idx.x)
                            * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                            + _GROUPED_TRACE_SUBPHASE_BASE
                            + 5 * tile_idx_epi
                            + 3,
                            UInt64(global_perf_counter_ns()),
                        )

            WarpGroupBarrier[num_output_warps * WARP_SIZE].sync()

            # Sub-phase trace: FINAL_DONE (stage 0 only).
            comptime if Self.swiglu_enable_trace:
                comptime if loop_stage == 0:
                    if (
                        tile_idx_epi < SWIGLU_MAX_TRACED_TILES
                        and tid_within_epi == 0
                    ):
                        trace_buf.store(
                            Int(block_idx.x)
                            * GROUPED_SWIGLU_TRACE_EVENTS_PER_BLOCK
                            + _GROUPED_TRACE_SUBPHASE_BASE
                            + 5 * tile_idx_epi
                            + 4,
                            UInt64(global_perf_counter_ns()),
                        )

        # Per-expert SF tail-pad zero-fill (cooperative path).
        # Triggers on (last m-tile, first n-tile) of the expert so the
        # pad-fill runs exactly once per expert (each expert is
        # processed by multiple (m, n) tiles; the SF tile is shared
        # across n-tiles, so we only need to zero pad rows once).
        if Int(m_abs) + Int(Self.MMA_N) >= Int(m_end) and n_abs == UInt32(0):
            swiglu_out.pad_sf_zero_block(
                sf_block_base,
                Int(m_end) - Int(m_start),
                Int(tid_within_epi),
                num_output_warps * WARP_SIZE,
            )

    @staticmethod
    @always_inline
    def epilogue(
        c_tiles: Self.SmemType.Core.CTileArray,
        c_tma_op: Self.CTmaOp,
        c_device: Self.CDeviceTile,
        stage: Self.TileWriterType.Stage,
        work_ctx: GroupedWorkContext1D1D,
        a_scale_offsets: Self.AScaleOffsetsTile,
        swiglu_out: Self.SwiGLUOutputT,
        trace_buf: Self.TraceBufT,
        tile_idx_epi: Int = 0,
    ):
        """Execute epilogue to store accumulated results with expert_scale.

        `tile_idx_epi` is forwarded to the SwiGLU+quant fused body for the
        per-tile sub-phase trace events (`T_EPI_S0_*` slots at offset
        `72 + 5*tile_idx_epi`). When `swiglu_enable_trace=False` the
        sub-phase records DCE; the parameter is otherwise unused.

        Args:
            c_tiles: SMEM C tile array used as the output staging buffer
                (or reinterpreted as fp32/bf16 scratchpad on the fused
                SwiGLU path).
            c_tma_op: C TMA store operation for the non-fused BF16 output
                path; an empty placeholder when `fuse_swiglu=True`.
            c_device: C device `TileTensor` used for bounds-checked
                GMEM stores on the non-fused path.
            stage: `TileWriterType.Stage` from the output pipeline
                consumer, holding the TMEM accumulator slot.
            work_ctx: `GroupedWorkContext1D1D` carrying the tile's
                coordinates, expert ID, expert scale, and token bounds.
            a_scale_offsets: Per-group A-side scale factor offsets used
                to compute the SF block base in the fused path.
            swiglu_out: `SwiGLUOutputT` fused output sink; pass
                `NullSwiGLUOutput[]()` for non-fused callers.
            trace_buf: `TraceBufT` for diagnostic per-tile timing
                records; zero-sized when `swiglu_enable_trace=False`.
            tile_idx_epi: Per-tile epilogue counter for trace event
                indexing (defaults to 0).
        """

        # For 1D-1D, pass absolute coordinates directly (not tile indices)
        # to handle unaligned expert offsets correctly.
        # m_abs = token offset, n_abs = weight offset, m_end = token boundary.
        # When transpose_c (AB_swapped), the writer handles the coordinate
        # swap internally.

        # For AB_swapped 2SM, each CTA computes different weight rows.
        # Add per-CTA offset (BM) to the N (weight) coordinate so each CTA
        # writes its portion: CTA0 writes n..n+BM-1, CTA1 writes n+BM..n+2*BM-1.
        var n_abs = work_ctx.n()
        comptime if Self.config.AB_swapped and Self.cta_group > 1:
            var rank_m = block_id_in_cluster.x
            var cta_n_offset = umod(rank_m, Self.cta_group) * Self.BM
            n_abs = UInt32(Int(n_abs) + cta_n_offset)

        # Fused-SwiGLU+quant epilogue branch (gated comptime; bit-identical
        # to the BF16 path when fuse_swiglu is False, the default).
        comptime if Self.fuse_swiglu:
            Self._swiglu_quant_epilogue_body(
                c_tiles,
                stage,
                work_ctx.m(),
                n_abs,
                work_ctx.m_end,
                work_ctx.m_start(),
                work_ctx.expert_scale,
                work_ctx.group_idx(),
                a_scale_offsets,
                swiglu_out,
                trace_buf,
                tile_idx_epi,
            )
        else:
            var tile_writer = Self.TileWriterType(Pointer(to=c_tma_op))
            tile_writer.write_absolute_with_bounds_check[Self.c_device_layout](
                c_tiles,
                stage,
                work_ctx.m(),  # Absolute M in contiguous token space
                n_abs,  # Absolute N in output space (per-CTA for 2SM)
                work_ctx.m_end,  # Token dim end for bounds checking
                work_ctx.expert_scale,
                c_device,
            )
