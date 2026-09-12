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

"""TileWriter for SM100 matmul output pipeline.

Writes accumulated results from TMEM → Registers → SMEM → GMEM (via TMA).

Usage:
    var writer = TileWriter[config=..., ...](Pointer(to=c_tma_op))
    writer.write(smem.c_tiles(), stage, coord, shape, elect)
"""

from std.collections import Array, Optional
from std.utils.static_tuple import StaticTuple
from std.memory import Pointer, UnsafePointer
from std.sys import (
    simd_width_of,
    size_of,
    align_of,
    get_defined_bool,
    get_defined_int,
)

from max.gpu import WARP_SIZE, block_idx, thread_idx
from max.gpu import lane_id, warp_id as get_warp_id
from max.gpu.memory import (
    cp_async_bulk_global_shared_cta,
    fence_async_view_proxy,
)
from max.gpu.sync import cp_async_bulk_commit_group, cp_async_bulk_wait_group
from std.memory import AddressSpace
from std.time import global_perf_counter_ns
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    RuntimeTuple,
    TensorEngine,
    TensorLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.layout import zipped_divide
from layout.layout_tensor import upcast
from layout.runtime_tuple import crd2idx as rt_crd2idx
from layout.swizzle import make_swizzle
from layout.tma_async import TMATensorTile
from max.gpu.compute.mma import ld_matrix
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)

from std.utils.index import IndexList

# TileTensor-based types for C tiles
from structured_kernels.tile_types import SMemTileArray2DRowMajor

from structured_kernels.barriers import WarpGroupBarrier
from structured_kernels.kernel_common import WarpRole1D1D
from .config import OutputPipelineConfig
from .tile_pipeline import OutputStage
from .output_writer_trait import OutputWriter
from .tile_scheduler_splitk import TileScheduler, WorkInfo
from .epilogue_components import (
    AccumBarrier,
    AccumTile,
    EpilogueApplier,
    EpilogueConfig,
    SMemEpilogueWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    TMEMToSMemWriter,
    tma_wait_pipelined,
)
from structured_kernels.pipeline import ProducerConsumerPipeline
from .tmem import TmemArrayType

# Fixed upper bound on the P2P rank count, so `P3PeerSendConfig.recv_buf_ptrs`
# can be a plain `Array` (`DevicePassable` across `enqueue_function`) instead of
# a pointer-to-pointer needing its own device-passability story. 8 covers any
# realistic P2P/NVLink domain size.
comptime P3_MAX_RANKS = 8

# SMEM mailbox slot count for the dedicated send warp class (`-D
# P5_SEND_WARPS=N`). One 4-`Int32` entry per epilogue `loop_stage` (m_abs,
# n_abs, m_end, expert_id); 32 Int32 covers `num_stages <= 8`, which
# `TileWriter` asserts. The kernel owns the single allocation site -- both the
# producing epilogue warps and the consuming send warps must see the SAME
# storage, and an `@always_inline` helper that allocated it locally would give
# each call site its own `.shared` object.
comptime P5_SEND_MBX_INTS = 32
comptime P5_SEND_MBX_STRIDE = 4

comptime P5SendMailboxPtr = UnsafePointer[
    Int32, MutUntrackedOrigin, address_space=AddressSpace.SHARED
]


@fieldwise_init
struct P3PeerSendConfig(ImplicitlyCopyable, Movable):
    """Runtime state for the in-epilogue EP-combine peer scatter-send.

    Bundles everything needed to resolve and issue a per-row peer send
    directly from the L2 epilogue's SMEM. One struct instead of ~8 separate
    new params: `write_absolute_with_bounds_check` has 3 callers spanning 2
    kernel families unrelated to EP-comm (`blockwise_fp8_1d2d`,
    `blockwise_fp8_output_writer`) that must stay byte-identical, so this
    adds exactly one new defaulted param to their call sites, not eight.

    `recv_count_layout((e, rk)) = e*n_ranks + rk` and
    `recv_buf_layout((src_idx, src_topk, byte_off)) =
    (src_idx*top_k + src_topk)*msg_bytes + byte_off` are the row-major
    linear-index formulas `EPCombineKernel` computes for those coordinate
    tuples -- copied here as plain arithmetic rather than threading the full
    `EPCombineKernel[...]` comptime specialization
    (num_threads/n_sms/n_experts/max_tokens_per_rank/p2p_world_size), none of
    which either formula depends on. Same precedent as this file's own
    `_stage2_recv_offset_in_bounds`, a documented copy of
    `EPCombineKernel._recv_offset_in_bounds` for the identical reason.

    `disabled()` is the default: `.unsafe_dangling()` pointers (never
    dereferenced -- gated behind `p3_control >= 0`) and 0-valued scalars.
    """

    var atomic_counter: UnsafePointer[Int32, MutUntrackedOrigin]
    var src_info_ptr: UnsafePointer[Int32, ImmUntrackedOrigin]
    # Per-ROW resolved destination cache, one UInt64 per absolute row, zeroed
    # once per launch by the caller.
    #
    # A row's destination is a property of the ROW -- `(dst_rank, src_idx,
    # src_topk)` -> peer base address -- and does NOT depend on which column
    # block is being written. Without the cache the epilogue resolves it once
    # per (row, TILE), and every row is touched by every n-block tile, so the
    # resolve runs an n-block factor more often than it changes. That repeated
    # resolve, not the stores, dominates the in-epilogue send: its cost is flat
    # in bytes sent.
    #
    # Sentinels: 0 = not yet resolved, 1 = resolved to NO destination (row not
    # owned by any rank, or `src_info` out of bounds), anything else = the peer
    # base pointer. A real base pointer is never 0 or 1. Two tiles racing on
    # the same row both compute the SAME value, so the store is idempotent and
    # needs no ordering.
    var row_base_ptr: UnsafePointer[UInt64, MutUntrackedOrigin]
    var recv_buf_ptrs: StaticTuple[
        UnsafePointer[UInt8, MutUntrackedOrigin], P3_MAX_RANKS
    ]
    var n_ranks: Int
    var p2p_world_size: Int
    var top_k: Int
    var msg_bytes: Int
    var max_tokens_per_rank: Int
    # This rank's own index into `recv_buf_ptrs`, i.e. the one entry that is a
    # LOCAL device allocation rather than a peer mapping.
    var my_rank: Int
    # SMEM mailbox through which the epilogue warps hand each finished output
    # tile's coordinates to the send warp class. Only dereferenced when
    # `P5_SEND_WARPS > 0` AND the send is runtime-enabled.
    var send_mbx: P5SendMailboxPtr
    # `-D P5_STAMP_WINDOW=true` (default OFF): the send's issue window, 2
    # UInt64 per physical CTA -- [2*cta] = first peer store issued, [2*cta+1] =
    # last. Zeroed once per launch by the caller; `first` is written only while
    # still 0, `last` on every tile.
    var send_stamp_ptr: UnsafePointer[UInt64, MutUntrackedOrigin]
    # `send_stamp_ptr` is `unsafe_dangling()` in every arm that does not supply
    # a buffer, and a dangling pointer is NOT a valid "uninitialized" sentinel,
    # so the writes need their own runtime enable -- the same shape as
    # `p3_control`. 0 = no stamp buffer; the send warps must not touch the
    # pointer. Without this the stamp faults with CUDA_ERROR_ILLEGAL_ADDRESS in
    # the arms that share this instantiation but pass no buffer.
    var send_stamp_on: Int
    # `-D P5_TRACE_EPI=true` (default OFF): per-CTA accumulator for the send
    # warps' own busy time, which nothing outside this file can see.
    var trace_ptr: UnsafePointer[UInt64, MutUntrackedOrigin]
    # Own runtime enable, for the same reason `send_stamp_on` has one: a
    # dangling pointer is not a valid "uninitialized" sentinel, and the arms
    # that share this instantiation pass no buffer.
    var trace_on: Int
    # The send warp class runs OUTSIDE `_write_absolute_with_bounds_check`, so
    # it cannot take a per-call slot argument the way the epilogue does. It
    # accumulates into one per-CTA slot instead, set once per launch.
    var send_trace_slot: Int

    @staticmethod
    def disabled() -> Self:
        return Self(
            UnsafePointer[Int32, MutUntrackedOrigin].unsafe_dangling(),
            UnsafePointer[Int32, ImmUntrackedOrigin].unsafe_dangling(),
            UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling(),
            StaticTuple[UnsafePointer[UInt8, MutUntrackedOrigin], P3_MAX_RANKS](
                UnsafePointer[UInt8, MutUntrackedOrigin].unsafe_dangling()
            ),
            0,
            1,
            1,
            0,
            0,
            0,
            P5SendMailboxPtr.unsafe_dangling(),
            UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling(),
            0,
            UnsafePointer[UInt64, MutUntrackedOrigin].unsafe_dangling(),
            0,
            -1,
        )


trait EpiloguePeerSink:
    """A second destination for the epilogue's SMEM output tile.

    The epilogue normally stores its tile to the local C tensor. A sink can
    take that tile somewhere else instead -- for expert parallelism, straight
    to the peers that own the rows -- in which case the local store is dead and
    the caller need not allocate C at all.

    A sink is a STATELESS comptime policy: every method is a `@staticmethod`
    and the destination's runtime state arrives as `p3_cfg`, exactly as it did
    when this epilogue owned the send. So a sink is never a value, contributes
    no kernel-argument bytes whether enabled or not, and needs no construction
    threading through the writer.

    Implementations:
      - `NullPeerSink`: `Enabled` is `False`; every body is `pass` and every
        call site sits inside a `comptime if`, so the no-sink build emits
        nothing for it.
      - `EpPeerSink` (closed tree): the EP-combine peer send.
    """

    comptime Enabled: Bool
    """Whether this sink owns the output. When `True` the local C store is
    dead, and the epilogue skips it along with its TMA descriptor."""

    comptime SendWarps: Int
    """Warps the sink needs above the scheduler, or 0 to run inside the output
    warps. The warp-role geometry reserves exactly this many."""

    @staticmethod
    def send_tile[
        stage: Int, num_threads: Int, TileT: AnyType
    ](
        tile: TileT,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_id: Int32,
        p3_cfg: P3PeerSendConfig,
        p3_control: Int,
        tid: Int,
    ):
        """Takes one finished output tile in place of the local store.

        Parameters:
            stage: Output pipeline stage the tile belongs to.
            num_threads: Threads cooperating on the tile.
            TileT: The epilogue's SMEM tile view type.

        Args:
            tile: The finished, barrier-visible output tile in SMEM.
            m_abs: Absolute row of the tile's first row.
            n_abs: Absolute column of the tile's first column.
            m_end: Exclusive row bound for the tile's group.
            expert_id: Logical per-rank expert slot of this tile.
            p3_cfg: The sink's runtime state.
            p3_control: Runtime gate; negative disables the sink.
            tid: Calling thread's index within `num_threads`.
        """
        ...

    @staticmethod
    def service[
        TilesT: AnyType
    ](tiles: TilesT, send_tid: Int, p3_control: Int, p3_cfg: P3PeerSendConfig,):
        """Runs the sink's own warps until the epilogue terminates them.

        Parameters:
            TilesT: The epilogue's SMEM tile array type.

        Args:
            tiles: The epilogue's SMEM output tile array.
            send_tid: Calling thread's index within `SendWarps` warps.
            p3_control: Runtime gate; negative leaves these warps idle.
            p3_cfg: The sink's runtime state.
        """
        ...

    @staticmethod
    def drain(p3_control: Int, p3_cfg: P3PeerSendConfig):
        """Collects work deferred from the previous tile.

        Args:
            p3_control: Runtime gate; negative means the warps never ran.
            p3_cfg: The sink's runtime state.
        """
        ...

    @staticmethod
    def shutdown(
        p3_control: Int, p3_cfg: P3PeerSendConfig, drain_pending: Bool
    ):
        """Retires the sink's warps with arrivals matched.

        Args:
            p3_control: Runtime gate; negative means the warps never ran.
            p3_cfg: The sink's runtime state.
            drain_pending: Whether a deferred drain is outstanding.
        """
        ...


struct NullPeerSink(EpiloguePeerSink):
    """No-op sink: the epilogue keeps its local store.

    Stateless like every sink, and `Enabled` is `False`, so each call site's
    `comptime if` strips the call outright. This is the default, which is what
    keeps the standalone matmuls byte-identical to a build with no sink
    parameter at all.
    """

    comptime Enabled = False
    comptime SendWarps = 0

    @staticmethod
    @always_inline
    def send_tile[
        stage: Int, num_threads: Int, TileT: AnyType
    ](
        tile: TileT,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_id: Int32,
        p3_cfg: P3PeerSendConfig,
        p3_control: Int,
        tid: Int,
    ):
        """No-op; the epilogue stores locally instead."""
        pass

    @staticmethod
    @always_inline
    def service[
        TilesT: AnyType
    ](tiles: TilesT, send_tid: Int, p3_control: Int, p3_cfg: P3PeerSendConfig,):
        """No-op; there are no sink warps."""
        pass

    @staticmethod
    @always_inline
    def drain(p3_control: Int, p3_cfg: P3PeerSendConfig):
        """No-op; nothing is ever deferred."""
        pass

    @staticmethod
    @always_inline
    def shutdown(
        p3_control: Int, p3_cfg: P3PeerSendConfig, drain_pending: Bool
    ):
        """No-op; there are no sink warps to retire."""
        pass


struct TileWriter[
    # Inferred from constructor arg
    tma_origin: ImmOrigin,
    c_type: DType,
    c_rank: Int,
    c_tile_shape: IndexList[c_rank],
    c_desc_shape: IndexList[c_rank],
    //,
    # Explicit config parameters (works with any config type)
    a_type: DType,
    accum_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    opc: OutputPipelineConfig,
    c_swizzle: TensorMapSwizzle,
    transpose_c: Bool,
    # Kernel-level parameters - dimensions replace c_smem_layout
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    num_output_stages: Int,
    num_output_warps: Int,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    register_based_epilogue: Bool = True,
    batched: Bool = False,
    problem_n: Int = 0,
    num_peers: Int = 1,  # this is a local epilogue
    c_store_dead: Bool = False,
    # The in-epilogue peer send and its row cache. Parameters rather than
    # build defines so a caller states them explicitly and a test can enable
    # them from source; the defaults reproduce the define for every existing
    # caller of this shared epilogue.
    p5_direct_scatter: Bool = False,
    p5_row_cache: Bool = False,
    # A second consumer of the output tile. The default keeps the local store
    # and costs nothing: `NullPeerSink` is zero-sized and every call site is
    # comptime-guarded on its `Enabled`.
    SinkT: EpiloguePeerSink = NullPeerSink,
](TrivialRegisterPassable):
    """Output tile writer for SM100 matmul epilogue.

    Stores pointer to TMA descriptor. SMEM tiles passed per-call.

    Parameters are passed explicitly to work with both MatmulConfig
    and BlockScaledMatmulConfig.

    The opc (OutputPipelineConfig) parameter must match the config used
    when constructing the OutputTilePipeline that provides OutputStage
    instances to the write() method.

    Parameters:
        tma_origin: Memory origin of the TMA descriptor pointer
            (inferred).
        c_type: Element dtype of the C output tensor (inferred).
        c_rank: Rank of the C output tensor (inferred).
        c_tile_shape: Per-tile shape of the C output (inferred).
        c_desc_shape: TMA descriptor shape for C (inferred).
        a_type: Element dtype of the A input matrix.
        accum_type: Accumulator dtype stored in TMEM.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        opc: Output pipeline config bundling accumulator stages, stage
            stride, and CTA group.
        c_swizzle: TMA swizzle pattern for the C SMEM layout.
        transpose_c: Whether C is stored transposed.
        c_smem_dim0: Row dimension of the C SMEM tile.
        c_smem_dim1: Column dimension of the C SMEM tile.
        num_output_stages: Number of C SMEM pipeline stages.
        num_output_warps: Number of warps driving the output pipeline.
        elementwise_lambda_fn: Optional elementwise epilogue applied to
            fragments before the store (defaults to None).
        elementwise_compute_lambda_fn: Optional compute epilogue fused
            into the register path (defaults to None).
        register_based_epilogue: Whether the compute epilogue runs in
            registers (true) or SMEM (false) (defaults to True).
        batched: Whether the output uses 3D batched coordinates with a
            batch index (defaults to False).
        problem_n: Logical N dimension used for row-major bounds checking
            in the slow path; 0 disables the N check (defaults to 0).
        num_peers: Number of TMA store descriptors in the array; 1 for a
            local epilogue (defaults to 1).
        c_store_dead: Whether nothing reads the local C output, so the
            store is dead for the whole launch. The caller then supplies an
            empty C descriptor and an unbacked C tensor, so neither may be
            touched here (defaults to False).
        p5_direct_scatter: Whether to compile in the in-epilogue EP-combine
            peer scatter-send. Still gated at runtime by `p3_control`
            (defaults to False).
        p5_row_cache: Whether the peer send resolves each row's destination
            once per launch through `P3PeerSendConfig.row_base_ptr` instead of
            once per (row, tile) (defaults to False).
        SinkT: A second destination for the output tile, replacing the local
            store when its `Enabled` is set. Stateless, so it costs nothing to
            carry; defaults to `NullPeerSink`, which keeps the local store.
    """

    # Local aliases from OutputPipelineConfig
    comptime cta_group = Self.opc.cta_group
    comptime num_accum_pipeline_stages = Self.opc.num_stages
    comptime stage_stride_cols = Self.opc.stage_stride_cols

    # Create internal layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    # Type aliases
    comptime TmaOp = TMATensorTile[
        Self.c_type, Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
    ]
    comptime TmaOpPtr = Pointer[Self.TmaOp, Self.tma_origin]
    # Whole-array pointer accepted by the `TileWriterLike` ctor (one descriptor
    # for the standard store; the ctor uses element [0]).
    comptime TmaOpArray = Array[Self.TmaOp, Self.num_peers]
    comptime TmaOpArrayPtr = Pointer[Self.TmaOpArray, Self.tma_origin]

    # No cross-GPU synchronization for a local TMA store.
    comptime needs_sync = False
    # C tile array (output and source tiles)
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.num_output_stages,
        128,
    ]
    comptime Stage = OutputStage[Self.opc]

    # Derived constants
    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    # FP8 uses float32 epilogue (GEX-2630), bf16/fp4 uses native type.
    comptime epilogue_dtype = Self.get_epilogue_dtype()

    # Stage dimensions - now use direct dimension access
    comptime N_dim = 0 if Self.transpose_c else 1
    comptime stageN = Self.c_smem_dim0 if Self.transpose_c else Self.c_smem_dim1
    comptime stage_contiguous_size = Self.c_smem_dim1

    # Warp-role table used only for `epilogue_role_index()`, which depends
    # on `EPILOGUE_WARP_START` alone — identical under either `has_sfb`
    # setting, so the default instantiation is exact for every caller.
    comptime WarpRole = WarpRole1D1D[num_epi_warps=Self.num_output_warps]

    # EpilogueConfig bundles common epilogue parameters
    comptime epc = EpilogueConfig.create(
        MMA_M=Self.MMA_M,
        MMA_N=Self.MMA_N,
        stageN=Self.stageN,
        cta_group=Self.cta_group,
        transpose_c=Self.transpose_c,
        BM=Self.BM,
        BN=Self.BN,
    )

    # Fragment layout constants
    comptime data_paths = 16
    comptime bits = 256
    comptime rep = Self.stageN // (Self.bits // 32)
    comptime fragment_size = (Self.data_paths * (Self.bits // 32)) // WARP_SIZE
    comptime rep_frag_size = Self.fragment_size * Self.rep

    # Aliases from EpilogueConfig
    comptime is_lower_frag_required = Self.epc.is_lower_frag_required
    comptime num_stages = Self.epc.num_stages

    # EP-combine peer send on a dedicated warp class.
    #
    # `-D P5_SEND_WARPS=N` moves the peer scatter-send OFF the epilogue warps
    # and onto N warps appended above the scheduler. The in-epilogue send's
    # cost is FLAT against every property of the work -- bytes, store
    # coalescing, store address space, resolve count and the resolve's global
    # loads -- while registers stay well inside budget and SMEM alone already
    # pins 1 CTA/SM. What is left is PLACEMENT: the epilogue warps are one half
    # of an MMA<->epilogue ping-pong, so work added after their wait extends
    # the cycle PERIOD once per stage per tile. That "idle" wait is the MMA
    # half of the period, not spare capacity, which is why nothing done WITHIN
    # the epilogue moves the number.
    #
    # The send becomes a second CONSUMER of the SMEM output tile. The epilogue
    # produces `c_smem_tile` exactly as before and pays one extra named-barrier
    # arrive; the send warps wait on it, then resolve and store.
    #
    # SLOT LIFETIME (`c_tiles[loop_stage % 2]` is double-buffered): the two
    # barriers are placed at the epilogue's OWN existing producer/consumer
    # points -- release right after the post-STSM `WarpGroupBarrier`, drain
    # under the exact same `loop_stage > 0 or loop_stage == num_stages - 1`
    # gate that already guards double-buffer reuse. That makes the pairing
    # 1:1 by construction (no credits, no cross-call state) and, because the
    # last stage always drains, it also covers the L1 epilogue's later reuse
    # of the same tiles.
    comptime p5_send_warps = get_defined_int["P5_SEND_WARPS", 0]()
    # `-D P5_TRACE_EPI=true` accumulates the send warps' own busy time per CTA.
    # Default OFF, so an untraced build stays byte-identical.
    comptime p5_trace_epi = get_defined_bool["P5_TRACE_EPI", False]()
    comptime p5_send_enabled = (
        Self.p5_send_warps > 0 and Self.p5_direct_scatter
    )
    comptime P5_SEND_THREADS = Self.p5_send_warps * WARP_SIZE
    # Ids 0-4 are live in the fused MegaFFN kernel (0 EpiSyncBarrier /
    # in-epilogue WarpGroupBarrier, 1 MmaEpilogueSync, 2 MmaSfbSync, 3 EP send
    # join, 4 EP/FFN init join), and the fused MegaFFN kernel's own comm and
    # setup barriers sit above this pair at 7 and 8.
    comptime P5_SEND_RELEASE_BARRIER_ID = 5
    comptime P5_SEND_DRAIN_BARRIER_ID = 6
    comptime P5SendRelease = WarpGroupBarrier[
        Self.num_output_warps * WARP_SIZE + Self.P5_SEND_THREADS,
        Self.P5_SEND_RELEASE_BARRIER_ID,
    ]
    comptime P5SendDrain = WarpGroupBarrier[
        Self.num_output_warps * WARP_SIZE + Self.P5_SEND_THREADS,
        Self.P5_SEND_DRAIN_BARRIER_ID,
    ]

    # `-D P5_STAMP_WINDOW=true` (default OFF): stamp the send's ISSUE WINDOW --
    # first and last peer store issued, per CTA. Two `global_perf_counter_ns()`
    # reads per CTA on the send warp only. Keep it off the timed path: a device
    # stamp there costs more than the interval it measures.
    comptime p5_stamp_window = get_defined_bool["P5_STAMP_WINDOW", False]()

    # TMEM array type for accumulator tiles
    comptime accum_tile_layout = Layout.row_major(Self.BM, Self.stageN)
    comptime AccumTmemArray = TmemArrayType[
        Self.accum_type,
        Self.accum_tile_layout,
        Self.num_stages,
        cta_group=Self.cta_group,
    ]

    var c_tma_op: Self.TmaOpPtr

    @always_inline
    def __init__(out self, c_tma_op: Self.TmaOpPtr):
        """Initialize with pointer to TMA descriptor.

        Args:
            c_tma_op: Pointer to the TMA store descriptor for C.
        """
        comptime assert (
            Self.stage_stride_cols > 0
        ), "stage_stride_cols must be positive"
        self.c_tma_op = c_tma_op

    @always_inline
    def __init__(out self, c_tma_ops: Self.TmaOpArrayPtr):
        """Initialize from the `c_tma_ops` array pointer (`TileWriterLike`).

        The standard local store targets a single descriptor, so this uses
        element `[0]` of the array. Unifies construction with the
        reduce-scatter writer, which retains all `num_peers` descriptors.

        Args:
            c_tma_ops: Pointer to the array of TMA store descriptors for
                C; element `[0]` is used for the local store.
        """
        comptime assert (
            Self.stage_stride_cols > 0
        ), "stage_stride_cols must be positive"
        self.c_tma_op = Pointer(to=c_tma_ops[][0])

    @always_inline
    @staticmethod
    def get_epilogue_dtype() -> DType:
        if (Self.a_type == Self.c_type == .bfloat16) or (
            Self.a_type == .uint8 and Self.c_type == .bfloat16
        ):
            return DType.bfloat16
        else:
            return DType.float32

    # ========== Public Write Methods ==========

    @always_inline
    def write(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write accumulated results to global memory (2D coords).

        Args:
            c_tiles: SMEM tile array for the C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile) tile coordinates.
            shape: (M, N) problem dimensions.
            elect_one_warp: Whether this warp is elected for coordination.
        """
        self._copy_to_gmem(c_tiles, stage, tile_coord, shape)

    @always_inline
    def write_batched(
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
    ):
        """Write accumulated results to global memory (3D batched coords).

        Args:
            c_tiles: TileTensor-based SMEM tile array for C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile, batch) coordinates.
            shape: (M, N) problem dimensions.
            alpha: Tensor scale factor (scalar).
        """
        self._copy_to_gmem_batched(c_tiles, stage, tile_coord, shape, alpha)

    @always_inline
    def write_splitk[
        reduction_layout: TensorLayout,
        reduction_engine: TensorEngine,
    ](
        self,
        c_tiles: Self.CTileArray,
        stage: Self.Stage,
        scheduler: TileScheduler,
        reduction_tensor: TileTensor[
            Self.accum_type,
            reduction_layout,
            MutAnyOrigin,
            Engine=reduction_engine,
        ],
        work_info: WorkInfo,
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write with split-K reduction. Only last split writes to GMEM."""
        var epilogue_thread_idx = thread_idx.x

        # Perform reduction and check if this is the last split
        var is_last_split = scheduler.reduction(
            reduction_tensor,
            stage.tmem.address(),
            epilogue_thread_idx,
            work_info,
        )

        # If not last split, signal and exit early
        if not is_last_split:
            AccumBarrier[Self.cta_group].arrive(stage.pipeline, stage.index)
            return

        self._copy_to_gmem(c_tiles, stage, (work_info.m, work_info.n), shape)

    @always_inline
    def write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
        # Peer scatter-send direct from SMEM, bytes only; the arrival protocol
        # is the caller's. Default -1/disabled() leaves every existing caller
        # byte-identical. See `P3PeerSendConfig`'s docstring above for the
        # param-bundling rationale.
        p3_control: Int = -1,
        p3_expert_id: Int32 = 0,
        p3_cfg: P3PeerSendConfig = P3PeerSendConfig.disabled(),
    ):
        """Write with absolute coordinates and bounds checking.

        For 1D-1D grouped kernels where M coordinate is absolute.

        Parameters:
            c_tensor_layout: Layout of the C tensor in GMEM (inferred).

        Args:
            c_tiles: SMEM tile array for the C output.
            output_stage: OutputStage with pipeline, index, and TMEM
                handle.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM for bounds-checked stores.
            p3_control: Peer-send gate; `-1` disables the send.
            p3_expert_id: Local expert id of the tile being written; the
                destination resolve keys its counter lookup on it.
            p3_cfg: Bundled peer-send configuration (buffers, geometry and
                the destination-resolve tables).
        """
        self._write_absolute_with_bounds_check[c_tensor_layout](
            c_tiles,
            output_stage,
            m_abs,
            n_abs,
            m_end,
            expert_scale,
            c_tensor,
            p3_control=p3_control,
            p3_expert_id=p3_expert_id,
            p3_cfg=p3_cfg,
        )

    @always_inline
    def _copy_to_gmem(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → SMEM → GMEM pipeline (2D coords)."""
        comptime if Self.elementwise_lambda_fn:
            self._copy_to_gmem_with_elementwise_epilogue_impl(
                c_tiles, output_stage, c_coord, c_shape
            )
        else:
            self._copy_to_gmem_impl(c_tiles, output_stage, c_coord, c_shape)

    @always_inline
    def _copy_to_gmem_batched(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32,
    ):
        """TMEM → Registers → GMEM (elementwise epilogue) pipeline (3D batched coords).
           TMEM → Registers → SMEM → GMEM (compute epilogue) pipeline (3D batched coords).

        If elementwise epilogue function is provided, it will be used to write the results to global memory.
        Otherwise, the results will be written to global memory using the standard TMA based pipeline.
        """
        comptime if Self.elementwise_lambda_fn:
            self._copy_to_gmem_with_elementwise_epilogue_impl(
                c_tiles,
                output_stage,
                (c_coord[0], c_coord[1]),
                c_shape,
                alpha,
                c_coord[2],
            )
        else:
            self._copy_to_gmem_impl(
                c_tiles,
                output_stage,
                (c_coord[0], c_coord[1]),
                c_shape,
                alpha,
                c_coord[2],
            )

    @always_inline
    def _copy_to_gmem_with_elementwise_epilogue_impl(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
        batch_idx: UInt32 = 0,
    ):
        """Unified TMEM → Registers → GMEM (elementwise epilogue) pipeline.

        Handles both standard (2D) and batched (3D) output paths.
        Alpha scaling is applied to fragments (defaults to 1.0 = no-op).
        Batch index is used for TMA store coordinates when batched=True.

        In contrast to compute epilogue, elementwise epilogue input is casted to c_type, not epilogue_dtype.
        This is because elementwise epilogue writes directly to global memory, not registers.
        Therefore, we need to cast the input to c_type to match the output type.
        """

        comptime assert (
            Self.elementwise_lambda_fn is not None
        ), "Elementwise epilogue function is not provided"

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        # Warp-uniform, computed once per tile: lets the direct-GMEM epilogue
        # skip per-position bounds checks (and their branches) for tiles fully
        # inside (M, N). transpose_c swaps the row/col → user-M/user-N mapping.
        # Mirrors _copy_to_gmem_impl's tile_in_bounds.
        var tile_in_bounds: Bool
        comptime if Self.transpose_c:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[1]
                and c_col + UInt32(Self.MMA_N) <= c_shape[0]
            )
        else:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[0]
                and c_col + UInt32(Self.MMA_N) <= c_shape[1]
            )

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # Extract fragments (rebind bridges symbolic size mismatch
            # between TmemTensor.frag_size*rep and Self.fragment_size*rep)
            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale by alpha and cast to c_type in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions
            # (e.g., cvt.rn.bf16x2.f32 for fp32→bf16).
            var alpha_val = alpha.cast[Self.accum_type]()
            comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
            var upper_simd = SIMD[Self.c_type, Self.rep_frag_size]()
            var lower_simd = SIMD[Self.c_type, Self.rep_frag_size]()

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * alpha_val).cast[Self.c_type]()
                comptime for _j in range(cast_width):
                    upper_simd[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * alpha_val).cast[Self.c_type]()
                    comptime for _j in range(cast_width):
                        lower_simd[offset + _j] = dst[_j]

            if tile_in_bounds:
                epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                    Self.c_type,
                    Self.rep_frag_size,
                    Self.elementwise_lambda_fn.value(),
                    Self.is_lower_frag_required,
                    is_in_bounds=True,
                ](
                    upper_simd,
                    lower_simd,
                    UInt32(stage),
                    c_row,
                    c_col,
                )
            else:
                epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                    Self.c_type,
                    Self.rep_frag_size,
                    Self.elementwise_lambda_fn.value(),
                    Self.is_lower_frag_required,
                    is_in_bounds=False,
                ](
                    upper_simd,
                    lower_simd,
                    UInt32(stage),
                    c_row,
                    c_col,
                )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    # ========== Shared Output Helpers ==========

    @always_inline
    def _cast_frags_and_write_to_smem[
        c_tile_layout: TensorLayout,
    ](
        self,
        upper_frag_casted: Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ],
        lower_frag_casted: Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ],
        c_smem_tile: TileTensor[
            Self.c_type, c_tile_layout, MutAnyOrigin, address_space=.SHARED
        ],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Cast fragments from epilogue dtype to c_type and write to SMEM."""
        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(warp_id, lane)

        comptime expected_size = Self.epc.fragment_size * Self.rep
        var upper_c = Array[Scalar[Self.c_type], expected_size](
            uninitialized=True
        )
        var lower_c = Array[Scalar[Self.c_type], expected_size](
            uninitialized=True
        )

        comptime cast_width_c = (4 // size_of[Scalar[Self.c_type]]())
        comptime for _chunk in range(Self.rep_frag_size // cast_width_c):
            comptime offset = _chunk * cast_width_c
            var src_u = SIMD[Self.epilogue_dtype, cast_width_c]()
            var src_l = SIMD[Self.epilogue_dtype, cast_width_c]()
            comptime for _j in range(cast_width_c):
                src_u[_j] = upper_frag_casted[offset + _j]
                src_l[_j] = lower_frag_casted[offset + _j]
            var dst_u = src_u.cast[Self.c_type]()
            var dst_l = src_l.cast[Self.c_type]()
            comptime for _j in range(cast_width_c):
                upper_c[offset + _j] = dst_u[_j]
                lower_c[offset + _j] = dst_l[_j]
        smem_writer.write_fragments[Self.rep](
            rebind[Array[Scalar[Self.c_type], expected_size]](upper_c),
            rebind[Array[Scalar[Self.c_type], expected_size]](lower_c),
            c_smem_tile,
        )
        WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @always_inline
    def _tma_store_to_gmem[
        stage: Int,
        c_tile_layout: TensorLayout,
    ](
        self,
        c_smem_tile: TileTensor[
            Self.c_type, c_tile_layout, MutAnyOrigin, address_space=.SHARED
        ],
        c_coord: Tuple[UInt32, UInt32],
        batch_idx: UInt32,
        warp_id: UInt32,
        lane: UInt32,
    ):
        """TMA store from SMEM to GMEM with pipelined wait and barrier sync."""
        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime StoreCoords = TMAStoreCoords[
            Self.epc,
            Self.c_smem_dim0,
            stage,
            batched=Self.batched,
        ]

        var store_coords = StoreCoords(
            (c_coord[0], c_coord[1], batch_idx if Self.batched else 0),
            warp_id,
        )
        StoreExecutor.execute[
            Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
        ](
            c_smem_tile,
            store_coords,
            self.c_tma_op[],
            warp_id,
            lane,
        )

        tma_wait_pipelined[
            Self.c_type,
            Self.c_rank,
            Self.c_tile_shape,
            Self.c_desc_shape,
            stage == Self.num_stages - 1,
        ](self.c_tma_op[])

        comptime if stage > 0 or stage == Self.num_stages - 1:
            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @always_inline
    def _copy_to_gmem_impl(
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
        batch_idx: UInt32 = 0,
    ):
        """Unified TMEM → Registers → SMEM → GMEM pipeline.

        Handles both standard (2D) and batched (3D) output paths.
        Alpha scaling is applied to fragments (defaults to 1.0 = no-op).
        Batch index is used for TMA store coordinates when batched=True.
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        # Warp-uniform: lets apply_to_both_fragments skip per-position
        # bounds checks for fully-in-bounds tiles. transpose_c swaps the
        # row/col → user-M/user-N mapping.
        var tile_in_bounds: Bool

        comptime if Self.transpose_c:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[1]
                and c_col + UInt32(Self.MMA_N) <= c_shape[0]
            )
        else:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[0]
                and c_col + UInt32(Self.MMA_N) <= c_shape[1]
            )

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            var frags = accum_tiles[stage].load_fragments[Self.rep]()

            # Extract fragments (rebind bridges symbolic size mismatch
            # between TmemTensor.frag_size*rep and Self.fragment_size*rep)
            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                Self.AccumTmemArray.Tile.wait_load()
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale by alpha and cast to epilogue dtype in SIMD chunks
            # of at least 4 bytes for efficient hardware cast
            # instructions (e.g., cvt.rn.bf16x2.f32 for fp32→bf16).
            var alpha_val = alpha.cast[Self.accum_type]()
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * alpha_val).cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * alpha_val).cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # Apply epilogue lambda if provided
            comptime if Self.elementwise_compute_lambda_fn:
                comptime if Self.register_based_epilogue:
                    if tile_in_bounds:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=True,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()
                    else:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=False,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()

            var c_smem_tile = c_tiles[stage % 2]

            comptime if (
                Self.register_based_epilogue
                or not Self.elementwise_compute_lambda_fn
            ):
                self._cast_frags_and_write_to_smem(
                    upper_frag_casted,
                    lower_frag_casted,
                    c_smem_tile,
                    UInt32(warp_id),
                    UInt32(lane),
                )
            else:
                var writer = SMemEpilogueWriter[
                    Self.c_smem_dim0,
                    Self.c_smem_dim1,
                    Self.epilogue_dtype,
                    Self.epc,
                    Self.num_output_warps,
                    Self.c_swizzle,
                    simd_size,
                    stage,
                    Self.rep_frag_size,
                    Self.elementwise_compute_lambda_fn.value(),
                ](UInt32(warp_id), c_tiles, c_shape, c_coord)
                writer.write_tile(
                    AccumTile(upper_frag_casted, lower_frag_casted)
                )

            self._tma_store_to_gmem[stage](
                c_smem_tile,
                (c_coord[0], c_coord[1]),
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )

    @always_inline
    def _write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
        p3_control: Int = -1,
        p3_expert_id: Int32 = 0,
        p3_cfg: P3PeerSendConfig = P3PeerSendConfig.disabled(),
    ):
        """Internal implementation of write with absolute coordinates and bounds checking.

        For 1D-1D grouped kernels where M coordinate is absolute (not tile index).
        Handles partial tiles that cross expert boundaries by using element-by-element
        stores for rows that would exceed m_end.

        Args:
            c_tiles: SMEM tile array for C output (TileTensor-based).
            output_stage: OutputStage with pipeline, index, and TMEM handle.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM (for bounds-checked stores).
            p3_control: Peer-send gate; `-1` disables the send.
            p3_expert_id: Local expert id of the tile being written; the
                destination resolve keys its counter lookup on it.
            p3_cfg: Bundled peer-send configuration (buffers, geometry and
                the destination-resolve tables).
        """
        # Dropping the local store leaves the epilogue's peer send as the
        # only output path, so a build without it would publish nothing.
        comptime assert (
            not Self.c_store_dead or Self.p5_direct_scatter
        ), "c_store_dead needs the in-epilogue peer send compiled in"
        # A disabled sink must be free: it is the default every standalone
        # matmul takes, and a sink with fields would put bytes in their
        # kernel ABI for a feature they do not use.
        comptime assert (
            Self.SinkT.Enabled or size_of[Self.SinkT]() == 0
        ), "a disabled EpiloguePeerSink must be zero-sized"
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())
        # Role-relative: this path's `warp_id == 0` single-writer election
        # (TMAStoreCoords, commit_group) means "first EPILOGUE warp", not
        # "first warp in the block". Identical today (the pool starts at
        # thread 0) and correct if the pool ever moves.
        var warp_id = Self.WarpRole.epilogue_role_index()
        var lane = lane_id()
        var scale = expert_scale.cast[Self.accum_type]()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutorLocal = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=False,  # Always 2D for absolute coords
        ]

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        # `-D P0_GEOMETRY_DUMP=true` (default OFF): print the geometry
        # constants the epilogue derives, once per (block 0, thread 0) call.
        comptime if get_defined_bool["P0_GEOMETRY_DUMP", False]():
            if block_idx.x == 0 and thread_idx.x == 0:
                print(
                    "[P0] num_stages=",
                    Self.num_stages,
                    " stageN=",
                    Self.stageN,
                    " MMA_M=",
                    Self.MMA_M,
                    " MMA_N=",
                    Self.MMA_N,
                    " cta_group=",
                    Self.cta_group,
                    " transpose_c=",
                    Self.transpose_c,
                    " c_smem_dim0=",
                    Self.c_smem_dim0,
                    " c_smem_dim1=",
                    Self.c_smem_dim1,
                    " rep=",
                    Self.rep,
                    " rep_frag_size=",
                    Self.rep_frag_size,
                    " is_lower_frag_required=",
                    Self.is_lower_frag_required,
                    " c_swizzle_bytes=",
                    Self.c_swizzle.bytes(),
                    " stage_contiguous_size=",
                    Self.stage_contiguous_size,
                )

        comptime for loop_stage in range(Self.num_stages):
            # Phase 1: TMEM Load
            var frags = accum_tiles[loop_stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            # rebind bridges symbolic size mismatch between
            # TmemTensor.frag_size*rep and Self.fragment_size*rep
            comptime PartialType2 = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType2](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType2](frags.lower).copy()

            # Phase 2: Barrier Arrive
            comptime if loop_stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # Scale and cast to epilogue dtype in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions.
            comptime cast_width_e = (
                4 // size_of[Scalar[Self.epilogue_dtype]]()
            )

            comptime for _chunk in range(Self.rep_frag_size // cast_width_e):
                comptime offset = _chunk * cast_width_e
                var src = SIMD[Self.accum_type, cast_width_e]()
                comptime for _j in range(cast_width_e):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = (src * scale).cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width_e):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(
                    Self.rep_frag_size // cast_width_e
                ):
                    comptime offset = _chunk * cast_width_e
                    var src = SIMD[Self.accum_type, cast_width_e]()
                    comptime for _j in range(cast_width_e):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = (src * scale).cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width_e):
                        lower_frag_casted[offset + _j] = dst[_j]

            # Phase 3: SMEM Write
            var c_smem_tile = c_tiles[loop_stage % 2]

            comptime expected_size = Self.epc.fragment_size * Self.rep
            # Cast from epilogue_dtype to c_type in SIMD chunks
            # of at least 4 bytes.
            var upper_c2 = Array[Scalar[Self.c_type], expected_size](
                uninitialized=True
            )
            var lower_c2 = Array[Scalar[Self.c_type], expected_size](
                uninitialized=True
            )

            comptime cast_width_c2 = (4 // size_of[Scalar[Self.c_type]]())
            comptime for _chunk in range(Self.rep_frag_size // cast_width_c2):
                comptime offset = _chunk * cast_width_c2
                var src_u = SIMD[Self.epilogue_dtype, cast_width_c2]()
                var src_l = SIMD[Self.epilogue_dtype, cast_width_c2]()
                comptime for _j in range(cast_width_c2):
                    src_u[_j] = upper_frag_casted[offset + _j]
                    src_l[_j] = lower_frag_casted[offset + _j]
                var dst_u = src_u.cast[Self.c_type]()
                var dst_l = src_l.cast[Self.c_type]()
                comptime for _j in range(cast_width_c2):
                    upper_c2[offset + _j] = dst_u[_j]
                    lower_c2[offset + _j] = dst_l[_j]
            smem_writer.write_fragments[Self.rep](
                rebind[Array[Scalar[Self.c_type], expected_size]](upper_c2),
                rebind[Array[Scalar[Self.c_type], expected_size]](lower_c2),
                c_smem_tile,
            )

            WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

            # Hand this stage's tile to the send warp class. The
            # coordinates go into the SMEM mailbox first; the barrier arrive
            # that follows is what publishes them (`barrier.cta.arrive` is the
            # release, the send warps' `barrier.cta.sync` the acquire -- the
            # same producer/consumer named-barrier shape the kernel already
            # uses for `MmaEpilogueSync`). The arrive does NOT block, so the
            # epilogue's critical path pays one instruction, not the send.
            #
            # The previous stage's DRAIN is collected right here, immediately
            # before this stage's release. That placement is forced, not
            # stylistic: a hardware named barrier holds ONE generation at a
            # time, so two consecutive `bar.arrive` from the producer with no
            # intervening `bar.sync` let the second generation's arrivals
            # complete the first, and the consumer's own arrival then lands in
            # the wrong generation. Draining here is what throttles the
            # producer to one outstanding generation. It also satisfies slot
            # lifetime with a stage to spare: stage k's drain lands inside
            # stage k+1, and the next write to k's physical slot is stage k+2.
            # The LAST stage's drain has no stage k+1 to sit in and is deferred
            # to the caller (the sink's `drain`, after the next tile's accumulator
            # acquire) -- which is where the overlap actually comes from, since
            # that wait is behind the MMA.
            comptime if Self.p5_send_enabled and loop_stage > 0:
                if p3_control >= 0 and p3_cfg.n_ranks > 0:
                    Self.P5SendDrain.wait()

            comptime if Self.p5_send_enabled:
                if p3_control >= 0 and p3_cfg.n_ranks > 0:
                    if warp_id == 0 and lane == 0:
                        var mbx_off = loop_stage * P5_SEND_MBX_STRIDE
                        p3_cfg.send_mbx[mbx_off + 0] = Int32(m_abs)
                        p3_cfg.send_mbx[mbx_off + 1] = Int32(n_abs)
                        p3_cfg.send_mbx[mbx_off + 2] = Int32(m_end)
                        p3_cfg.send_mbx[mbx_off + 3] = p3_expert_id
                    Self.P5SendRelease.arrive()

            # `-D P3_INLINE_SEND=true` (default OFF): peer scatter-send direct
            # from SMEM, bytes only -- the arrival protocol is the caller's.
            # DECODE-ONLY: it borrows `c_tiles[1 - loop_stage % 2]` as scratch,
            # and at prefill that slot is genuinely double-buffered by the real
            # epilogue. `p5_direct_scatter` below is the prefill-safe form.
            comptime if get_defined_bool["P3_INLINE_SEND", False]():
                if p3_control >= 0 and p3_cfg.n_ranks > 0:
                    # Opt-in (`-D P3_ENABLED_PRINT=true`), default OFF: a
                    # device-side print here runs once per L2 tile, on the
                    # timed path, and costs far more than the send it reports.
                    # Prefer an artifact the harness can compare (arrived bytes
                    # against a mechanism-off control) over a print.
                    comptime if get_defined_bool["P3_ENABLED_PRINT", False]():
                        if block_idx.x == 0 and thread_idx.x == 0:
                            print(
                                "[P3_ENABLED] loop_stage=", loop_stage, sep=""
                            )
                    # Step 1: unswizzle-gather. `cp_async_bulk_global_
                    # shared_cta` (step 2) needs a physically-contiguous
                    # per-row span; a logical row is NOT contiguous in the
                    # swizzled `c_smem_tile` (P0 measured c_swizzle_bytes=
                    # 32, and `_store_with_bounds_check_transpose`'s own
                    # gather loop shows one logical row is assembled from
                    # `chunk_num` disjoint swizzle-permuted fragments).
                    # Reuses that function's comptime swizzle derivation
                    # and inner-loop math VERBATIM (never re-derive the
                    # swizzle formula), retargeting the store from
                    # `c_tensor` (GMEM) to `c_tiles[1 - loop_stage % 2]`
                    # (plain row-major SMEM, decode-only-safe per P2).
                    var p3_unswiz = c_tiles[1 - (loop_stage % 2)]
                    comptime p3_simd_size = simd_width_of[Self.c_type]()
                    comptime p3_swizzle_width = Self.c_swizzle.bytes() // size_of[
                        Self.c_type
                    ]()
                    comptime p3_chunkM = p3_swizzle_width
                    comptime p3_vec_chunkM = p3_chunkM // p3_simd_size
                    comptime p3_chunk_num = Self.stage_contiguous_size // p3_chunkM
                    comptime p3_logical_size = p3_chunk_num * Self.stageN * p3_vec_chunkM
                    comptime p3_output_threads = Self.num_output_warps * WARP_SIZE
                    comptime p3_value_shape = p3_logical_size // p3_output_threads
                    comptime p3_smem_alignment = align_of[
                        SIMD[Self.c_type, p3_simd_size]
                    ]()
                    comptime p3_swizzle = make_swizzle[
                        Self.c_type, Self.c_swizzle
                    ]()

                    # `m_abs` is this TILE's own start, constant across every
                    # `loop_stage` unroll, so the rows THIS stage covers start
                    # at `m_abs + loop_stage*stageN` -- exactly like Phase 4's
                    # `stage_token_start` below. Inert at `num_stages == 1`.
                    # Without the term, every `loop_stage >= 1` re-derives
                    # stage 0's row range: an in-bounds-looking but wrong
                    # `p3_abs_row` that corrupts the destination-rank lookup
                    # and the `src_info_ptr` index it feeds.
                    var p3_n_inbound = (
                        Int32(m_end)
                        - Int32(m_abs)
                        - Int32(loop_stage * Self.stageN)
                    )

                    comptime for v in range(p3_value_shape):
                        comptime p3_thread_offset = v * p3_output_threads
                        var p3_tidx = UInt32(thread_idx.x) + UInt32(
                            p3_thread_offset
                        )
                        var p3_rest, p3_vec_chunkM_idx = divmod(
                            p3_tidx, UInt32(p3_vec_chunkM)
                        )
                        var p3_n_idx = p3_rest % UInt32(Self.stageN)
                        if Int32(p3_n_idx) >= min(
                            p3_n_inbound, Int32(Self.stageN)
                        ):
                            continue
                        var p3_src_idx = UInt32(p3_simd_size) * p3_tidx
                        var p3_smem_idx = p3_swizzle(p3_src_idx)
                        var p3_val_vec = (c_smem_tile.ptr + p3_smem_idx).load[
                            width=p3_simd_size,
                            alignment=p3_smem_alignment,
                        ]()
                        var p3_chunk_idx = p3_rest // UInt32(Self.stageN)
                        var p3_local_col = (
                            p3_chunk_idx * UInt32(p3_vec_chunkM)
                            + p3_vec_chunkM_idx
                        ) * UInt32(p3_simd_size)
                        (
                            p3_unswiz.ptr
                            + p3_n_idx * UInt32(Self.stage_contiguous_size)
                            + p3_local_col
                        ).store[alignment=p3_smem_alignment](p3_val_vec)

                    WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

                    # Step 2: per-row resolve + send. A per-tile rank-range
                    # cache from `atomic_counter`, then per-row `src_info` +
                    # bounds check + `recv_buf_layout` offset (both formulas
                    # copied as plain arithmetic -- see `P3PeerSendConfig`'s
                    # docstring). `P3_MAX_RANKS` also sizes
                    # `P3PeerSendConfig.recv_buf_ptrs`.
                    var p3_rank_lo = Array[Int32, P3_MAX_RANKS](fill=0)
                    var p3_rank_hi = Array[Int32, P3_MAX_RANKS](fill=0)
                    for rk in range(p3_cfg.n_ranks):
                        var p3_cnt_off = Int(p3_expert_id) * p3_cfg.n_ranks + rk
                        var p3_packed = p3_cfg.atomic_counter.load[
                            width=2,
                            alignment=align_of[SIMD[DType.int32, 2]](),
                        ](2 * p3_cnt_off)
                        # EP_DATA_READY_FLAG = 1 << 10 (`shmem/ep_comm.mojo:
                        # 110`), copied as a literal for the same reason as
                        # `_stage2_recv_offset_in_bounds` above.
                        var p3_t_end = p3_packed[0] - Int32(1 << 10)
                        p3_rank_hi[rk] = p3_t_end
                        p3_rank_lo[rk] = p3_t_end - p3_packed[1]

                    var p3_valid_rows = min(Int(p3_n_inbound), Int(Self.stageN))
                    comptime if get_defined_bool["P3_STAGE_DIAG", False]():
                        if thread_idx.x == 0 and block_idx.x < 3:
                            print(
                                "  [P3_STAGE_DIAG_RAW] block=",
                                block_idx.x,
                                " loop_stage=",
                                loop_stage,
                                " m_abs=",
                                Int(m_abs),
                                " m_end=",
                                Int(m_end),
                                " p3_n_inbound=",
                                Int(p3_n_inbound),
                                " p3_valid_rows=",
                                p3_valid_rows,
                                sep="",
                            )
                    if p3_valid_rows > 0:
                        var p3_tid = Int(thread_idx.x)
                        if p3_tid < p3_valid_rows:
                            # See `p3_n_inbound` above: this thread's
                            # absolute row is THIS STAGE's start
                            # (`m_abs + loop_stage*stageN`) plus its position
                            # within the stage, not `m_abs + p3_tid`.
                            var p3_abs_row = (
                                Int(m_abs)
                                + Int(loop_stage * Self.stageN)
                                + p3_tid
                            )
                            var p3_dst_rank = -1
                            for rk in range(p3_cfg.n_ranks):
                                if p3_abs_row >= Int(
                                    p3_rank_lo[rk]
                                ) and p3_abs_row < Int(p3_rank_hi[rk]):
                                    p3_dst_rank = rk
                                    break
                            comptime if get_defined_bool[
                                "P3_STAGE_DIAG", False
                            ]():
                                if block_idx.x == 0 and p3_tid < 4:
                                    print(
                                        "  [P3_STAGE_DIAG_EARLY] loop_stage=",
                                        loop_stage,
                                        " tid=",
                                        p3_tid,
                                        " m_abs=",
                                        Int(m_abs),
                                        " p3_valid_rows=",
                                        p3_valid_rows,
                                        " p3_abs_row=",
                                        p3_abs_row,
                                        " rank_lo0=",
                                        Int(p3_rank_lo[0]),
                                        " rank_hi0=",
                                        Int(p3_rank_hi[0]),
                                        " rank_lo1=",
                                        Int(p3_rank_lo[1]),
                                        " rank_hi1=",
                                        Int(p3_rank_hi[1]),
                                        " p3_dst_rank=",
                                        p3_dst_rank,
                                        sep="",
                                    )
                            if p3_dst_rank >= 0:
                                var p3_dst_p2p_rank = (
                                    p3_dst_rank % p3_cfg.p2p_world_size
                                )
                                var p3_st = p3_cfg.src_info_ptr.load[
                                    width=2,
                                    alignment=align_of[SIMD[DType.int32, 2]](),
                                ](p3_abs_row * 2)
                                var p3_src_idx = p3_st[0]
                                var p3_src_topk = p3_st[1]
                                var p3_in_bounds = (
                                    p3_src_idx >= 0
                                    and Int(p3_src_idx)
                                    < p3_cfg.max_tokens_per_rank
                                    and p3_src_topk >= 0
                                    and Int(p3_src_topk) < p3_cfg.top_k
                                )
                                if p3_in_bounds:
                                    var p3_byte_off = (
                                        Int(n_abs) * size_of[Self.c_type]()
                                    )
                                    var p3_recv_off = (
                                        Int(p3_src_idx) * p3_cfg.top_k
                                        + Int(p3_src_topk)
                                    ) * p3_cfg.msg_bytes + p3_byte_off
                                    var p3_dst_ptr = (
                                        p3_cfg.recv_buf_ptrs[p3_dst_p2p_rank]
                                        + p3_recv_off
                                    )
                                    comptime p3_row_bytes = (
                                        Self.stage_contiguous_size
                                        * size_of[Self.c_type]()
                                    )
                                    comptime if get_defined_bool[
                                        "P3_STAGE_DIAG", False
                                    ]():
                                        print(
                                            "  [P3_STAGE_DIAG] loop_stage=",
                                            loop_stage,
                                            " tid=",
                                            p3_tid,
                                            " m_abs=",
                                            Int(m_abs),
                                            " n_abs=",
                                            Int(n_abs),
                                            " p3_abs_row=",
                                            p3_abs_row,
                                            " p3_dst_rank=",
                                            p3_dst_rank,
                                            " p3_dst_p2p_rank=",
                                            p3_dst_p2p_rank,
                                            " p3_src_idx=",
                                            Int(p3_src_idx),
                                            " p3_src_topk=",
                                            Int(p3_src_topk),
                                            " p3_recv_off=",
                                            p3_recv_off,
                                            " p3_row_bytes=",
                                            p3_row_bytes,
                                            " recv_buf_ptr=",
                                            Int(
                                                p3_cfg.recv_buf_ptrs[
                                                    p3_dst_p2p_rank
                                                ]
                                            ),
                                            " p3_dst_ptr=",
                                            Int(p3_dst_ptr),
                                            sep="",
                                        )
                                    cp_async_bulk_global_shared_cta(
                                        p3_dst_ptr,
                                        (
                                            p3_unswiz.ptr
                                            + UInt32(p3_tid)
                                            * UInt32(Self.stage_contiguous_size)
                                        ).bitcast[UInt8](),
                                        Int32(p3_row_bytes),
                                    )
                                    cp_async_bulk_commit_group()
                        # Drain and async-proxy promotion are PER-THREAD by
                        # PTX definition (ISA 9.1 s9.7.9.27.2.1); called
                        # unconditionally by every thread in the output-
                        # warp group (not just `p3_tid < p3_valid_rows`) to
                        # avoid warp divergence at the barrier that
                        # follows. Matches both `stage2_send_kernel` and
                        # `test_ep_combine_send_scheduler.mojo`'s
                        # `send_kernel`. NVIDIA's `cp.async.bulk` STORE
                        # direction (SMEM->GMEM) has no `.mbarrier::
                        # complete_tx` completion variant anywhere in this
                        # codebase (verified: grep across `memory.mojo` --
                        # every `.mbarrier::complete_tx::bytes` site is a
                        # LOAD or a tensor-descriptor S2G store's sibling
                        # function documents `.bulk_group` instead, e.g.
                        # `cp_async_bulk_tensor_global_shared_cta_elect`'s
                        # own docstring). `WarpGroupBarrier` (a named
                        # barrier, reused verbatim rather than a novel,
                        # zero-precedent-in-tree raw mbarrier rendezvous of
                        # equivalent semantic value) is the block-wide
                        # rendezvous after the per-thread drain.
                        #
                        # THIS COMMENT IS THE GUARD. Per-thread drain
                        # cannot be made structurally impossible to get
                        # wrong for this direction -- there is no
                        # complete_tx-style mechanism the hardware itself
                        # enforces (see above), and a mutation sweep
                        # confirmed removing this discipline is INVISIBLE
                        # to every test that exercises this hardware. If
                        # you are about to gate the two calls below on
                        # `p3_tid < p3_valid_rows` (looks like a harmless
                        # optimization -- only issuing threads need to
                        # wait, right?), read PTX ISA 9.1 s9.7.9.27.2.1
                        # first: promotion is per-thread by definition, a
                        # barrier does not perform it, and no test here
                        # will tell you that you broke it.
                        cp_async_bulk_wait_group[0]()
                        fence_async_view_proxy()
                        WarpGroupBarrier[
                            Self.num_output_warps * WARP_SIZE
                        ].sync()

            # The prefill-safe peer scatter-send (`p5_direct_scatter`):
            # gather-to-register plus a direct scattered store, with NO
            # borrowed `c_tiles` scratch slot at all. The `P3_INLINE_SEND`
            # path above is decode-only precisely because it borrows a slot,
            # and at prefill BOTH physical slots are genuinely double-buffered
            # by the real epilogue. A dedicated SMEM staging buffer would fit,
            # but costs occupancy on every launch; a direct store to a
            # peer-mapped pointer costs no SMEM and reuses a primitive this
            # file already relies on for its peer atomics.
            #
            # Every output thread walks the SAME flat logical index space
            # `_store_with_bounds_check_transpose` walks, decomposed by its
            # formula (never re-derive the swizzle math): one `divmod` gives
            # the column-chunk, `rest % stageN` the row. Each unit loads one
            # `simd_size`-wide vector from the swizzled `c_smem_tile` and
            # stores it straight to that row's already-resolved peer address,
            # so no intermediate buffer ever holds more than one vector. That
            # keeps every output thread busy; a row-per-thread partition here
            # would leave most of them idle at the decode tile.
            #
            # `-D P5_SEND_WARPS=N` relocates this whole block to the send warp
            # class (the sink's own `service`), so it compiles out here.
            comptime if Self.SinkT.Enabled and Self.SinkT.SendWarps == 0:
                # Opt-in (`-D P5_ENABLED_PRINT=true`), default OFF: a
                # device-side print here runs once per L2 tile, on the
                # timed path, and costs far more than the send it reports.
                comptime if get_defined_bool["P5_ENABLED_PRINT", False]():
                    if block_idx.x == 0 and thread_idx.x == 0:
                        print("[P5_ENABLED] loop_stage=", loop_stage, sep="")
                Self.SinkT.send_tile[
                    loop_stage, Self.num_output_warps * WARP_SIZE
                ](
                    c_smem_tile,
                    m_abs,
                    n_abs,
                    m_end,
                    p3_expert_id,
                    p3_cfg,
                    p3_control,
                    Int(thread_idx.x),
                )

            # Phase 4: TMA Store with bounds checking
            comptime CG2_TMA_BM = Self.c_smem_dim0 if Self.MMA_M == 256 else Self.BM
            comptime CG1_TMA_BM = Self.c_smem_dim0
            comptime TMA_BM = CG2_TMA_BM if Self.cta_group == 2 else CG1_TMA_BM
            comptime StoreCoordsLocal = TMAStoreCoords[
                Self.epc,
                Self.c_smem_dim0,
                loop_stage,
                batched=False,
            ]

            # Transpose and non-transpose have different bounds checks
            # and coordinate conventions:
            # - Non-transpose: check m_abs (token dim), store_non_transpose
            #   swaps coords so coord_m→tc1(tokens), coord_n→tc0(weights)
            # - Transpose: check per-stage token dim (in n_abs position),
            #   store_transpose_lt passes coord_m→tc0(weights),
            #   coord_n→tc1(tokens)
            var tile_needs_bounds_check: Bool

            comptime if Self.transpose_c:
                # Per-stage bounds check on token dimension (passed as
                # n_abs). Each stage writes stageN token rows.
                var stage_token_start = m_abs + UInt32(loop_stage * Self.stageN)
                tile_needs_bounds_check = (
                    stage_token_start + UInt32(Self.stageN) > m_end
                )
            else:
                tile_needs_bounds_check = m_abs + UInt32(TMA_BM) > m_end

            # Negative control: force the FAST (unclamped) TMA-store path
            # even for a tile that genuinely crosses the expert boundary, so a
            # ragged distribution's correctness gate has something to fail
            # against. Default False = byte-identical.
            comptime if get_defined_bool["NEGCTL_DISABLE_ROW_CLAMP", False]():
                tile_needs_bounds_check = False

            # `-D P5_SKIP_C_STORE=true`: in the fused path the local C tensor
            # is DEAD for every row the peer send covers -- nothing reads it
            # once the combine reduces out of `recv_buf`. Without this the
            # epilogue writes the whole output twice, and the two writes
            # compete for the same store path.
            #
            # PIPELINE SAFETY, which a first attempt got wrong: an earlier
            # version `return`ed out of Phase 4 and DEADLOCKED, because it
            # skipped the group commit that `tma_wait_pipelined` below counts
            # on. Committing an EMPTY group keeps the accounting exact -- the
            # wait finds a group that is already complete -- and it is the same
            # idiom the bounds-check path already uses for its own non-TMA
            # store. No traffic, no missing group, no stall.
            #
            # `Self.c_store_dead` is the same removal declared per call site
            # instead of per build. That is what lets the launcher drop the C
            # TMA encode and the caller drop the C allocation: the `-D` form
            # cannot, because it also covers the arms in the same binary that
            # do read C and are saved only by `p3_control < 0` at runtime.
            var p5_c_store_dead = Self.c_store_dead
            comptime if get_defined_bool["P5_SKIP_C_STORE", False]():
                p5_c_store_dead = p5_c_store_dead or p3_control >= 0
            # `-D P5_ELIDE_DEAD_TMA` (default TRUE): when the store is dead,
            # do not commit an empty group and do not wait on one either.
            #
            # The empty-group idiom above exists so `tma_wait_pipelined` finds
            # the group it counts on -- correct, but it leaves the fused arm
            # issuing a commit + wait pair per stage per CTA for stores that
            # were never made.
            #
            # Skipping BOTH sides keeps the accounting exact for the same
            # reason committing an empty group did: `p5_c_store_dead` is
            # derived from `p3_control`, a LAUNCH parameter, so it is uniform
            # across every stage and every tile of the launch. Either every
            # stage commits and waits, or none does -- there is no mixed state
            # in which a wait could go looking for a group that was never
            # committed. The SMEM-reuse ordering the wait also provided is
            # carried by the warpgroup barrier below, which is unconditional.
            comptime p5_elide_dead_tma = get_defined_bool[
                "P5_ELIDE_DEAD_TMA", True
            ]()
            if p5_c_store_dead:
                comptime if not p5_elide_dead_tma:
                    if warp_id == 0 and lane == 0:
                        self.c_tma_op[].commit_group()
            elif tile_needs_bounds_check:
                comptime if Self.transpose_c:
                    # CUDA core fallback for unaligned group boundaries
                    Self._store_with_bounds_check_transpose[c_tensor_layout](
                        c_smem_tile._storage,
                        c_tensor,
                        m_abs + UInt32(loop_stage * Self.stageN),
                        n_abs,
                        m_end,
                    )
                else:
                    # Slow path: element-by-element stores with bounds check
                    Self._store_with_bounds_check[c_tensor_layout](
                        c_smem_tile,
                        c_tensor,
                        m_abs,
                        n_abs + UInt32(loop_stage * Self.stageN),
                        m_end,
                        UInt32(warp_id),
                        UInt32(lane),
                    )

                # Advance TMA group counter so wait_group[1] properly
                # drains the previous stage's in-flight TMA store.
                # Without this, the group count stays stale and
                # wait_group[1] returns immediately, allowing the next
                # double-buffer stage to overwrite SMEM while TMA reads
                # it.
                if warp_id == 0 and lane == 0:
                    self.c_tma_op[].commit_group()
            else:
                # Fast path: TMA store for tiles fully within bounds
                var n_tile = n_abs / UInt32(Self.MMA_N)
                var dummy_m_tile = UInt32(0)
                var store_coords = StoreCoordsLocal(
                    (dummy_m_tile, n_tile), UInt32(warp_id)
                )

                comptime if Self.transpose_c:
                    # Transpose: coord_m→tc0→N(weights),
                    # coord_n→tc1→M(tokens)
                    store_coords.coord_m = Int(n_abs)
                    store_coords.coord_n = Int(m_abs) + loop_stage * Self.stageN
                else:
                    # Non-transpose: coord_m→tc1→M(tokens),
                    # coord_n→tc0→N(weights)
                    store_coords.coord_m = Int(m_abs)
                    store_coords.coord_n = Int(n_abs) + loop_stage * Self.stageN

                StoreExecutorLocal.execute[
                    Self.c_rank, Self.c_tile_shape, Self.c_desc_shape
                ](
                    c_smem_tile,
                    store_coords,
                    self.c_tma_op[],
                    UInt32(warp_id),
                    UInt32(lane),
                )

            # Phase 5: TMA Wait — drains outstanding TMA stores from earlier
            # stages when the slow path skips TMA, preventing SMEM races with
            # double-buffered tiles. Skipped only when the store is dead for
            # the WHOLE launch, in which case there is nothing outstanding to
            # drain; see the `P5_ELIDE_DEAD_TMA` note above.
            if not (p5_c_store_dead and p5_elide_dead_tma):
                tma_wait_pipelined[
                    Self.c_type,
                    Self.c_rank,
                    Self.c_tile_shape,
                    Self.c_desc_shape,
                    loop_stage == Self.num_stages - 1,
                ](self.c_tma_op[])

            comptime if loop_stage > 0 or loop_stage == Self.num_stages - 1:
                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

    @staticmethod
    @always_inline
    def _store_with_bounds_check[
        c_tensor_layout: TensorLayout,
        c_smem_layout: TensorLayout,
    ](
        c_smem_tile: TileTensor[
            Self.c_type, c_smem_layout, MutAnyOrigin, address_space=.SHARED
        ],
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Store SMEM tile to GMEM with per-element bounds checking.

        Used when the tile crosses the expert boundary (m_abs + TMA_BM > m_end).
        Uses element-by-element stores to avoid writing past m_end.

        Args:
            c_smem_tile: SMEM tile to store (TileTensor).
            c_tensor: C tensor in global memory (TileTensor).
            m_abs: Absolute M coordinate (start of tile).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            warp_id: Current warp ID.
            lane: Current lane ID.
        """
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime c_smem_M = Self.c_smem_dim0
        comptime TMA_BM = 64 if Self.cta_group == 1 else 128
        # Ensure enough work for all threads: need thread_rows <= TMA_BM,
        # i.e., output_threads / thread_n <= TMA_BM,
        # i.e., simd_size <= stageN * TMA_BM / output_threads.
        comptime max_simd = simd_width_of[Self.c_type]()
        comptime max_allowed_simd = Self.stageN * TMA_BM // output_threads
        comptime simd_size = min(max_simd, max(1, max_allowed_simd))
        comptime alignment = align_of[SIMD[Self.c_type, simd_size]]()
        comptime thread_n = Self.stageN // simd_size
        comptime thread_layout = Layout.row_major(
            output_threads // thread_n, thread_n
        )

        # Precompute the split layout from known dimensions
        comptime split_layout = Layout(
            IntTuple(TMA_BM, Self.stageN), IntTuple(Self.c_smem_dim1, 1)
        )

        # Swizzle function
        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        # Ensure fence before reading from SMEM
        if warp_id == 0 and lane == 0:
            fence_async_view_proxy()

        # Synchronize all epilogue threads
        WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

        # Iterate over SMEM chunks
        comptime for i in range(c_smem_M // TMA_BM):
            var c_smem_split = c_smem_tile.tile[TMA_BM, Self.stageN](i, 0)
            comptime zipped = zipped_divide(
                upcast(split_layout, simd_size), thread_layout
            )
            # Use new Layout for idx2crd
            comptime split_layout_new = row_major[TMA_BM, Self.stageN]()

            comptime for j in range(zipped.shape[1][0].value()):
                var input_crd = RuntimeTuple[
                    IntTuple(UNKNOWN_VALUE, j),
                    element_type=.uint32,
                ](thread_idx.x, j)
                var linear_idx = rt_crd2idx[
                    IntTuple(UNKNOWN_VALUE, j),
                    zipped.shape,
                    zipped.stride,
                    DType.uint32,
                ](
                    input_crd,
                    RuntimeTuple[zipped.shape](),
                    RuntimeTuple[zipped.stride](),
                ) * UInt32(
                    simd_size
                )
                var cmem_crd = split_layout_new.idx2crd[out_dtype=DType.uint32](
                    Int(linear_idx)
                )
                var local_i = cmem_crd[0].value()
                var local_j = cmem_crd[1].value()
                var coord_m = m_abs + UInt32(i * TMA_BM)
                var global_i = coord_m + UInt32(local_i)
                var global_j = n_abs + UInt32(local_j)

                # Bounds check: only store if within M and N boundaries.
                # The N check prevents row-major wrap-around when the
                # last N-tile extends past the logical output width.
                var in_bounds = global_i < m_end
                comptime if Self.problem_n > 0:
                    in_bounds = in_bounds and (
                        global_j + UInt32(simd_size) <= UInt32(Self.problem_n)
                    )
                if in_bounds:
                    comptime if size_of[Self.c_type]() == 2:
                        var src_ptr = c_smem_split._storage + swizzle(
                            linear_idx
                        )
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        c_tensor.store[width=simd_size, alignment=alignment](
                            Coord(Int(global_i), Int(global_j)),
                            src,
                        )
                    else:
                        var src_ptr = c_smem_split._storage + linear_idx
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        c_tensor.store[width=simd_size, alignment=alignment](
                            Coord(Int(global_i), Int(global_j)),
                            src,
                        )

    @staticmethod
    @always_inline
    def _store_with_bounds_check_transpose[
        c_tensor_layout: TensorLayout,
    ](
        c_smem_ptr: UnsafePointer[
            Scalar[Self.c_type], _, address_space=.SHARED
        ],
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
    ):
        """CUDA core fallback for unaligned group boundaries (transpose).

        With SWIZZLE_32B, each swizzle_width chunk (16 bf16 elements) in
        SMEM maps to one TMA row. We read flat SMEM with swizzle(simd_size
        * tidx) and decompose tidx into (vec_chunkM_idx, n_idx, chunk_idx)
        to compute global coordinates.

        This matches the reference implementation in grouped_matmul_sm100_1d1d.

        Args:
            c_smem_ptr: Raw pointer to SMEM tile.
            c_tensor: C tensor in global memory (TileTensor).
            m_abs: Token start for this stage.
            n_abs: Weight start (absolute N coordinate).
            m_end: Token boundary (exclusive).
        """
        comptime simd_size = simd_width_of[Self.c_type]()
        comptime swizzle_width = Self.c_swizzle.bytes() // size_of[
            Self.c_type
        ]()
        comptime chunkM = swizzle_width
        comptime vec_chunkM = chunkM // simd_size
        comptime chunk_num = Self.stage_contiguous_size // chunkM
        comptime logical_size = chunk_num * Self.stageN * vec_chunkM
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime assert (
            logical_size % output_threads == 0
        ), "logical_size must be divisible by output_threads"
        comptime value_shape = logical_size // output_threads
        comptime cN = c_tensor.static_shape[1]
        comptime smem_alignment = align_of[SIMD[Self.c_type, simd_size]]()

        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        var n_inbound = Int32(m_end) - Int32(m_abs)

        comptime for v in range(value_shape):
            comptime thread_offset = v * output_threads
            var tidx = UInt32(thread_idx.x) + UInt32(thread_offset)
            var rest, vec_chunkM_idx = divmod(tidx, UInt32(vec_chunkM))
            var n_idx = rest % UInt32(Self.stageN)
            if Int32(n_idx) >= min(n_inbound, Int32(Self.stageN)):
                continue
            var src_idx = UInt32(simd_size) * tidx
            var c_smem_idx = swizzle(src_idx)
            var val_vec = (c_smem_ptr + c_smem_idx).load[
                width=simd_size,
                alignment=smem_alignment,
            ]()
            var chunk_idx = rest // UInt32(Self.stageN)
            # m_abs = token index, n_abs = weight index
            var global_token = m_abs + n_idx
            var global_weight = n_abs + (
                chunk_idx * UInt32(vec_chunkM) + vec_chunkM_idx
            ) * UInt32(simd_size)
            if global_token < m_end and global_weight < UInt32(cN):
                c_tensor.store[width=simd_size, alignment=smem_alignment](
                    Coord(Int(global_token), Int(global_weight)),
                    val_vec,
                )

    # ========== Residual Add Support ==========
    # Methods for D = lambda(accum) + beta * C residual operations

    @always_inline
    def write_with_residual[
        pipeline_origin: MutOrigin,
        //,
        num_src_stages: Int,
    ](
        self,
        out_tiles: Self.CTileArray,
        stage: Self.Stage,
        src_tile: SMemTileArray2DRowMajor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            num_src_stages,
            128,
        ],
        src_pipeline: Pointer[
            ProducerConsumerPipeline[num_src_stages], pipeline_origin
        ],
        beta: Scalar[Self.c_type],
        tile_coord: Tuple[UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        elect_one_warp: Bool,
    ):
        """Write with residual: D = lambda(accum) + beta * C.

        Matches the CUTLASS `sm100_epilogue_tma_warpspecialized` lockstep
        pattern: the epilogue load warp pre-fetches one source sub-tile per
        inner epilogue stage into a `num_src_stages`-deep SMEM pipeline; this
        method drives one `wait_producer / use / consumer_release / step`
        cycle on `src_pipeline` per inner stage. The buffer index is read from
        the pipeline's `consumer_stage()` rather than computed offline, so
        producer and consumer stay synchronized exactly as in CUTLASS's
        `consumer_wait → copy(sC) → consumer_release` per epi sub-tile.

        Pipeline per inner stage:
        1. Load accum from TMEM to registers (epilogue dtype).
        2. Apply `elementwise_compute_lambda_fn` (pre-residual fusion).
        3. Wait for source[k] via `src_pipeline.consume()`; compute
           `D = accum + beta * C` reading from the SMEM buffer at the
           pipeline's current stage index; release source[k] on context exit.
        4. Apply `elementwise_lambda_fn` (post-residual, owns the GMEM store)
           OR stage to output SMEM and TMA-store to GMEM.

        Parameters:
            pipeline_origin: Mutability origin of the source pipeline ref.
            num_src_stages: Number of source SMEM buffers; must equal the
                epi-load pipeline's stage count in the kernel.

        Args:
            out_tiles: Output SMEM tile array (for D output).
            stage: OutputStage with pipeline, index, and TMEM handle.
            src_tile: Source C SMEM tile array (num_src_stages buffers).
            src_pipeline: Pointer to the source producer/consumer pipeline.
                One acquire/release cycle is driven per inner epilogue stage.
            beta: Residual scale factor.
            tile_coord: (m_tile, n_tile) coordinates.
            shape: (M, N) problem dimensions.
            elect_one_warp: Whether this warp is elected for coordination.
        """
        self._copy_to_gmem_with_residual[num_src_stages](
            out_tiles,
            stage,
            src_tile,
            src_pipeline,
            beta,
            tile_coord,
            shape,
        )

    @always_inline
    def _copy_to_gmem_with_residual[
        pipeline_origin: MutOrigin,
        //,
        num_src_stages: Int,
    ](
        self,
        out_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        src_tiles: SMemTileArray2DRowMajor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            num_src_stages,
            128,
        ],
        src_pipeline: Pointer[
            ProducerConsumerPipeline[num_src_stages], pipeline_origin
        ],
        beta: Scalar[Self.c_type],
        c_coord: Tuple[UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """TMEM → Registers → (+ beta*C) → SMEM → GMEM pipeline with residual.

        Per-inner-stage CUTLASS lockstep: wait the source pipeline, read the
        SMEM tile at `consumer_stage()`, do the residual add, release on
        context exit. The buffer index is provided by the pipeline rather
        than computed locally.
        """
        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        comptime simd_size = simd_width_of[Self.c_type]()
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )
        var c_row = c_coord[0] * UInt32(Self.BM)
        var c_col = c_coord[1] * UInt32(Self.MMA_N)

        # Warp-uniform: lets apply_to_both_fragments skip per-position
        # bounds checks for fully-in-bounds tiles. transpose_c swaps the
        # row/col → user-M/user-N mapping.
        var tile_in_bounds: Bool

        comptime if Self.transpose_c:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[1]
                and c_col + UInt32(Self.MMA_N) <= c_shape[0]
            )
        else:
            tile_in_bounds = (
                c_row + UInt32(Self.BM) <= c_shape[0]
                and c_col + UInt32(Self.MMA_N) <= c_shape[1]
            )

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM tile
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()
            var casted = frags.cast[Self.epilogue_dtype]()

            var upper_frag_casted = Array[_, Self.rep_frag_size](
                fill_with_unrolled=lambda [i: Int]() -> Scalar[
                    Self.epilogue_dtype
                ]: casted.upper[i]
            )
            var lower_frag_casted = Array[_, Self.rep_frag_size](
                fill_with_unrolled=lambda [i: Int]() -> Scalar[
                    Self.epilogue_dtype
                ]: casted.lower[i]
            )

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Apply epilogue lambda (if present)
            comptime if Self.elementwise_compute_lambda_fn:
                comptime if Self.register_based_epilogue:
                    if tile_in_bounds:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=True,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()
                    else:
                        var _result = epilogue_applier.apply_to_both_fragments[
                            Self.epilogue_dtype,
                            Self.rep_frag_size,
                            Self.elementwise_compute_lambda_fn.value(),
                            Self.is_lower_frag_required,
                            is_in_bounds=False,
                        ](
                            upper_frag_casted,
                            lower_frag_casted,
                            UInt32(stage),
                            c_row,
                            c_col,
                        )
                        upper_frag_casted = _result[0].copy()
                        lower_frag_casted = _result[1].copy()

            # 3. Apply residual: D = accum + beta * C in registers.
            # CUTLASS-style lockstep: wait, read SMEM at the buffer the
            # producer just filled, do the add, release the stage, advance.
            # Each per-stage SMEM buffer holds exactly one inner stage's
            # source (BM x OutputN), so we pass `stage=0` to the residual
            # helper — its internal `stage * stageN` column offset would
            # otherwise index past the buffer.
            comptime residual_swizzle = make_swizzle[
                Self.c_type, Self.c_swizzle
            ]()
            src_pipeline[].wait_producer()
            var _src_idx = src_pipeline[].consumer_stage()
            var src_smem_tile = src_tiles[Int(_src_idx)]
            var _residual_result = (
                epilogue_applier.add_residual_to_both_fragments[
                    Self.epilogue_dtype,
                    Self.rep_frag_size,
                    Self.is_lower_frag_required,
                    Self.c_type,
                    Self.c_smem_dim1,
                    residual_swizzle,
                ](
                    upper_frag_casted,
                    lower_frag_casted,
                    UInt32(0),
                    src_smem_tile._storage,
                    beta.cast[Self.epilogue_dtype](),
                )
            )
            upper_frag_casted = _residual_result[0].copy()
            lower_frag_casted = _residual_result[1].copy()
            # Release this source stage and advance — every epilogue thread
            # arrives on the consumer mbar (arv_count = 128).
            _ = src_pipeline[].consumer_mbar(_src_idx)[0].arrive()
            src_pipeline[].consumer_step()

            # 4. Final store. When a void `elementwise_lambda_fn` is set the
            # lambda owns the GMEM write (post-residual contract — see
            # `nn/conv/gpu/amd/amd_4wave_conv_residual.mojo` for the matching
            # AMD path); otherwise stage to SMEM and TMA-store.
            comptime if Self.elementwise_lambda_fn:
                # Cast fragments from epilogue_dtype to c_type. Chunked by
                # 4 bytes to match the hardware cvt instruction width.
                comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
                var upper_simd = SIMD[Self.c_type, Self.rep_frag_size]()
                var lower_simd = SIMD[Self.c_type, Self.rep_frag_size]()

                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src_u = SIMD[Self.epilogue_dtype, cast_width]()
                    var src_l = SIMD[Self.epilogue_dtype, cast_width]()
                    comptime for _j in range(cast_width):
                        src_u[_j] = upper_frag_casted[offset + _j]
                        src_l[_j] = lower_frag_casted[offset + _j]
                    var dst_u = src_u.cast[Self.c_type]()
                    var dst_l = src_l.cast[Self.c_type]()
                    comptime for _j in range(cast_width):
                        upper_simd[offset + _j] = dst_u[_j]
                        lower_simd[offset + _j] = dst_l[_j]

                epilogue_applier.apply_elementwise_epilogue_to_both_fragments[
                    Self.c_type,
                    Self.rep_frag_size,
                    Self.elementwise_lambda_fn.value(),
                    Self.is_lower_frag_required,
                ](
                    upper_simd,
                    lower_simd,
                    UInt32(stage),
                    c_row,
                    c_col,
                )

                WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()
            else:
                # Write to output SMEM
                var c_smem_tile = out_tiles[stage % 2]

                comptime if (
                    Self.register_based_epilogue
                    or not Self.elementwise_compute_lambda_fn
                ):
                    self._cast_frags_and_write_to_smem(
                        upper_frag_casted,
                        lower_frag_casted,
                        c_smem_tile,
                        UInt32(warp_id),
                        UInt32(lane),
                    )
                else:
                    var writer = SMemEpilogueWriter[
                        Self.c_smem_dim0,
                        Self.c_smem_dim1,
                        Self.epilogue_dtype,
                        Self.epc,
                        Self.num_output_warps,
                        Self.c_swizzle,
                        simd_size,
                        stage,
                        Self.rep_frag_size,
                        Self.elementwise_compute_lambda_fn.value(),
                    ](UInt32(warp_id), out_tiles, c_shape, c_coord)
                    writer.write_tile(
                        AccumTile(upper_frag_casted, lower_frag_casted)
                    )

                self._tma_store_to_gmem[stage](
                    c_smem_tile,
                    c_coord,
                    UInt32(0),
                    UInt32(warp_id),
                    UInt32(lane),
                )

    @always_inline
    def write_batched_with_tma_epilogue_load[
        epi_load_swizzle: TensorMapSwizzle,
        epilogue_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        epilogue_tile: TileTensor[
            Self.c_type, epilogue_layout, MutAnyOrigin, address_space=.SHARED
        ],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results with epilogue tensor addition to global memory.

        Pipeline: TMEM → Registers → (+epilogue from SMEM) → SMEM → GMEM (TMA).
        """
        var c_coord = (tile_coord[0], tile_coord[1])
        var batch_idx = tile_coord[2]

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )

        comptime sub_tile_n = epi_load_swizzle.bytes() // size_of[Self.c_type]()
        comptime epi_rows = Self.MMA_N if Self.transpose_c else Self.BM
        comptime sub_tile_elems = epi_rows * sub_tile_n
        var epi_load_sw = make_swizzle[Self.c_type, epi_load_swizzle]()

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Cast to epilogue dtype
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = src.cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = src.cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # 3. Add epilogue tensor from swizzled SMEM.
            # Epilogue SMEM is row-major [M, N] with swizzle. Fragment coords
            # from compute_staged_coords are in TMEM space:
            #   non-transpose: row→M, col→N
            #   transpose:     row→N, col→M
            # So for transpose we swap which TMEM dim maps to epilogue row/col.
            var local_row, local_col = epilogue_applier.compute_staged_coords(
                UInt32(stage), 0, 0
            )
            comptime for rep in range(Self.rep):
                comptime inc = rep * 8
                comptime frag_offset = rep * 4

                comptime if Self.transpose_c:
                    # N position: lanes 0-7 → N, lanes 8-15 → N+8
                    var n_pos = local_row + UInt32(lane & 8)
                    var ldm_tidx = n_pos // UInt32(sub_tile_n)
                    var ldm_nloc = n_pos % UInt32(sub_tile_n)
                    var ldm_tile_base = ldm_tidx * UInt32(sub_tile_elems)

                    var ldm_m_row = local_col + UInt32(inc) + UInt32(lane & 7)

                    # Top + bottom via ldmatrix.trans .x2
                    var ldm_off = ldm_tile_base + epi_load_sw(
                        ldm_m_row * UInt32(sub_tile_n) + ldm_nloc
                    )
                    var epi_vals = ld_matrix[simd_width=4, transpose=True](
                        epilogue_tile._storage + Int(ldm_off)
                    )

                    upper_frag_casted[frag_offset] += epi_vals[0].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 1] += epi_vals[1].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 2] += epi_vals[2].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 3] += epi_vals[3].cast[
                        Self.epilogue_dtype
                    ]()

                else:
                    var col_base = local_col + UInt32(inc)
                    var ldm_tidx = col_base // UInt32(sub_tile_n)
                    var ldm_nloc = col_base % UInt32(sub_tile_n)
                    var ldm_tile_base = ldm_tidx * UInt32(sub_tile_elems)

                    # Rows 0-15 via ldmatrix .x2
                    var ldm_row = local_row + UInt32(lane & 15)
                    var ldm_off = ldm_tile_base + epi_load_sw(
                        ldm_row * UInt32(sub_tile_n) + ldm_nloc
                    )
                    var epi_vals = ld_matrix[simd_width=4](
                        epilogue_tile._storage + Int(ldm_off)
                    )

                    upper_frag_casted[frag_offset] += epi_vals[0].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 1] += epi_vals[1].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 2] += epi_vals[2].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 3] += epi_vals[3].cast[
                        Self.epilogue_dtype
                    ]()

            comptime if Self.is_lower_frag_required:
                comptime for rep in range(Self.rep):
                    comptime inc = rep * 8
                    comptime frag_offset = rep * 4

                    comptime if Self.transpose_c:
                        # N position: lanes 0-7 → N+16, lanes 8-15 → N+24
                        var n_pos_l = local_row + 16 + UInt32(lane & 8)
                        var ldm_tidx_l = n_pos_l // UInt32(sub_tile_n)
                        var ldm_nloc_l = n_pos_l % UInt32(sub_tile_n)
                        var ldm_tile_base_l = ldm_tidx_l * UInt32(
                            sub_tile_elems
                        )

                        var ldm_m_row_l = (
                            local_col + UInt32(inc) + UInt32(lane & 7)
                        )

                        # Rows 16-31 via ldmatrix.trans .x2
                        var ldm_off_l = ldm_tile_base_l + epi_load_sw(
                            ldm_m_row_l * UInt32(sub_tile_n) + ldm_nloc_l
                        )
                        var epi_vals_l = ld_matrix[
                            simd_width=4, transpose=True
                        ](epilogue_tile._storage + Int(ldm_off_l))

                        lower_frag_casted[frag_offset] += epi_vals_l[0].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset + 1] += epi_vals_l[
                            1
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 2] += epi_vals_l[
                            2
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 3] += epi_vals_l[
                            3
                        ].cast[Self.epilogue_dtype]()
                    else:
                        var col_base_l = local_col + UInt32(inc)
                        var ldm_tidx_l = col_base_l // UInt32(sub_tile_n)
                        var ldm_nloc_l = col_base_l % UInt32(sub_tile_n)
                        var ldm_tile_base_l = ldm_tidx_l * UInt32(
                            sub_tile_elems
                        )

                        # Rows 16-31 via ldmatrix .x2
                        var ldm_row_l = local_row + 16 + UInt32(lane & 15)
                        var ldm_off_l = ldm_tile_base_l + epi_load_sw(
                            ldm_row_l * UInt32(sub_tile_n) + ldm_nloc_l
                        )
                        var epi_vals_l = ld_matrix[simd_width=4](
                            epilogue_tile._storage + Int(ldm_off_l)
                        )

                        lower_frag_casted[frag_offset] += epi_vals_l[0].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset + 1] += epi_vals_l[
                            1
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 2] += epi_vals_l[
                            2
                        ].cast[Self.epilogue_dtype]()
                        lower_frag_casted[frag_offset + 3] += epi_vals_l[
                            3
                        ].cast[Self.epilogue_dtype]()

            # 4. Cast to c_type, write to output SMEM, and TMA store to GMEM
            var c_smem_tile = c_tiles[stage % 2]
            self._cast_frags_and_write_to_smem(
                upper_frag_casted,
                lower_frag_casted,
                c_smem_tile,
                UInt32(warp_id),
                UInt32(lane),
            )
            self._tma_store_to_gmem[stage](
                c_smem_tile,
                c_coord,
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )

    @always_inline
    def write_batched_with_1d_bias[
        epilogue_layout: TensorLayout,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        epilogue_tile: TileTensor[
            Self.c_type, epilogue_layout, MutAnyOrigin, address_space=.SHARED
        ],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results with 1D bias addition to global memory.

        Pipeline: TMEM -> Registers -> (+1D bias broadcast from SMEM) -> SMEM -> GMEM (TMA).

        The bias SMEM tile is 1×MMA_N loaded via cp.async (linear layout,
        no swizzle) and then broadcast across all M rows.
        """
        var c_coord = (tile_coord[0], tile_coord[1])
        var batch_idx = tile_coord[2]

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())

        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime SMEMWriter = TMEMToSMemWriter[
            Self.c_type,
            Self.accum_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.num_output_warps,
            Self.c_swizzle,
        ]
        var smem_writer = SMEMWriter(UInt32(warp_id), UInt32(lane))

        comptime StoreExecutor = TMAStoreExecutor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.epc,
            Self.stage_contiguous_size,
            Self.c_swizzle,
            batched=Self.batched,
        ]

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id),
            UInt32(lane),
            c_shape,
        )

        var upper_frag_partial: Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ]
        var lower_frag_partial = Array[
            Scalar[Self.accum_type], Self.rep_frag_size
        ](uninitialized=True)
        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            upper_frag_partial = rebind[PartialType](frags.upper).copy()

            comptime if Self.is_lower_frag_required:
                lower_frag_partial = rebind[PartialType](frags.lower).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Cast to epilogue dtype
            comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())

            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime offset = _chunk * cast_width
                var src = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src[_j] = upper_frag_partial[offset + _j]
                var dst = src.cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[offset + _j] = dst[_j]

            comptime if Self.is_lower_frag_required:
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime offset = _chunk * cast_width
                    var src = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src[_j] = lower_frag_partial[offset + _j]
                    var dst = src.cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[offset + _j] = dst[_j]

            # 3. Add 1D bias from SMEM.
            # Non-transpose: vectorized 32-bit load (2 consecutive bf16).
            # Transpose: scalar load + broadcast (non-contiguous addresses).
            var local_row, local_col = epilogue_applier.compute_staged_coords(
                UInt32(stage), 0, 0
            )
            var top_u = epilogue_applier.coords.top_upper
            var bot_u = epilogue_applier.coords.bottom_upper

            comptime for rep in range(Self.rep):
                comptime inc = rep * 8
                comptime frag_offset = rep * 4

                comptime if Self.transpose_c:
                    var n_top = local_row + top_u[0]
                    var n_bot = local_row + bot_u[0]
                    var b_top = epilogue_tile._storage[Int(n_top)].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset] += b_top
                    upper_frag_casted[frag_offset + 1] += b_top
                    var b_bot = epilogue_tile._storage[Int(n_bot)].cast[
                        Self.epilogue_dtype
                    ]()
                    upper_frag_casted[frag_offset + 2] += b_bot
                    upper_frag_casted[frag_offset + 3] += b_bot
                else:
                    var n_col = local_col + top_u[1] + UInt32(inc)
                    var bias = (
                        (epilogue_tile._storage + Int(n_col))
                        .load[width=2]()
                        .cast[Self.epilogue_dtype]()
                    )
                    upper_frag_casted[frag_offset] += bias[0]
                    upper_frag_casted[frag_offset + 1] += bias[1]
                    upper_frag_casted[frag_offset + 2] += bias[0]
                    upper_frag_casted[frag_offset + 3] += bias[1]

            comptime if Self.is_lower_frag_required:
                var top_l = epilogue_applier.coords.top_lower
                var bot_l = epilogue_applier.coords.bottom_lower

                comptime for rep in range(Self.rep):
                    comptime inc = rep * 8
                    comptime frag_offset = rep * 4

                    comptime if Self.transpose_c:
                        var n_top_l = local_row + top_l[0]
                        var n_bot_l = local_row + bot_l[0]
                        var b_top_l = epilogue_tile._storage[Int(n_top_l)].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset] += b_top_l
                        lower_frag_casted[frag_offset + 1] += b_top_l
                        var b_bot_l = epilogue_tile._storage[Int(n_bot_l)].cast[
                            Self.epilogue_dtype
                        ]()
                        lower_frag_casted[frag_offset + 2] += b_bot_l
                        lower_frag_casted[frag_offset + 3] += b_bot_l
                    else:
                        var n_col_l = local_col + top_l[1] + UInt32(inc)
                        var bias = (
                            (epilogue_tile._storage + Int(n_col_l))
                            .load[width=2]()
                            .cast[Self.epilogue_dtype]()
                        )
                        lower_frag_casted[frag_offset] += bias[0]
                        lower_frag_casted[frag_offset + 1] += bias[1]
                        lower_frag_casted[frag_offset + 2] += bias[0]
                        lower_frag_casted[frag_offset + 3] += bias[1]

            # 4. Cast to c_type, write to output SMEM, and TMA store to GMEM
            var c_smem_tile = c_tiles[stage % 2]
            self._cast_frags_and_write_to_smem(
                upper_frag_casted,
                lower_frag_casted,
                c_smem_tile,
                UInt32(warp_id),
                UInt32(lane),
            )
            self._tma_store_to_gmem[stage](
                c_smem_tile,
                c_coord,
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )

    @always_inline
    def write_batched_with_tma_epilogue_load_strips[
        epi_load_swizzle: TensorMapSwizzle,
        num_epi_stages: Int,
    ](
        self,
        c_tiles: Self.CTileArray,
        output_stage: Self.Stage,
        mut epilogue_pipeline: ProducerConsumerPipeline[num_epi_stages],
        epilogue_tiles_base: UnsafePointer[
            Scalar[Self.c_type], MutAnyOrigin, address_space=.SHARED
        ],
        epilogue_tile_elems: Int,
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        c_shape: Tuple[UInt32, UInt32],
    ):
        """Write accumulated results with BM×stageN pipelined epilogue addition.

        For non-AB_swapped configs. Each epilogue pipeline stage is one BM×stageN tile.
        Producer sends tiles in stage-outer / col_wg-inner order; consumer mirrors that
        structure so each TMEM stage is fully processed (load → add epilogue → write)
        before advancing to the next.
        """
        var c_coord = (tile_coord[0], tile_coord[1])
        var batch_idx = tile_coord[2]

        var accum_tiles = Self.AccumTmemArray(output_stage.tmem.offset())
        var warp_id = get_warp_id()
        var lane = lane_id()

        comptime EpilogueApplierType = EpilogueApplier[
            Self.MMA_M,
            Self.stageN,
            Self.num_stages,
            Self.rep,
            Self.cta_group,
            Self.transpose_c,
        ]
        var epilogue_applier = EpilogueApplierType(
            UInt32(warp_id), UInt32(lane), c_shape
        )

        # Each epilogue SMEM stage is BM×stageN. sub_tile_n equals stageN in all modes:
        #   SWIZZLE_NONE  (stageN= 8): sub_tile_n= 8
        #   SWIZZLE_32B   (stageN=16): sub_tile_n=16  (32 bytes / 2 bytes per bf16)
        #   SWIZZLE_64B   (stageN=32): sub_tile_n=32
        #   SWIZZLE_128B  (stageN=64): sub_tile_n=64
        comptime sub_tile_n = Self.stageN
        var epi_load_sw = make_swizzle[Self.c_type, epi_load_swizzle]()

        comptime cast_width = (4 // size_of[Scalar[Self.epilogue_dtype]]())
        comptime num_col_warp_groups = Self.MMA_N // (
            Self.num_stages * Self.stageN
        )

        var upper_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)
        var lower_frag_casted = Array[
            Scalar[Self.epilogue_dtype], Self.rep_frag_size
        ](uninitialized=True)

        comptime for stage in range(Self.num_stages):
            # 1. Load fragments from TMEM
            var frags = accum_tiles[stage].load_fragments[Self.rep]()
            Self.AccumTmemArray.Tile.wait_load()

            comptime PartialType = Array[
                Scalar[Self.accum_type], Self.rep_frag_size
            ]
            var upper_partial = rebind[PartialType](frags.upper).copy()

            comptime if stage == Self.num_stages - 1:
                AccumBarrier[Self.cta_group].arrive(
                    output_stage.pipeline, output_stage.index
                )

            # 2. Cast upper (and lower) accumulator fragments to epilogue dtype
            comptime for _chunk in range(Self.rep_frag_size // cast_width):
                comptime off = _chunk * cast_width
                var src_u = SIMD[Self.accum_type, cast_width]()
                comptime for _j in range(cast_width):
                    src_u[_j] = upper_partial[off + _j]
                var dst_u = src_u.cast[Self.epilogue_dtype]()
                comptime for _j in range(cast_width):
                    upper_frag_casted[off + _j] = dst_u[_j]

            comptime if Self.is_lower_frag_required:
                var lower_partial = rebind[PartialType](frags.lower).copy()
                comptime for _chunk in range(Self.rep_frag_size // cast_width):
                    comptime off = _chunk * cast_width
                    var src_l = SIMD[Self.accum_type, cast_width]()
                    comptime for _j in range(cast_width):
                        src_l[_j] = lower_partial[off + _j]
                    var dst_l = src_l.cast[Self.epilogue_dtype]()
                    comptime for _j in range(cast_width):
                        lower_frag_casted[off + _j] = dst_l[_j]

            # 3. Consume epilogue pipeline stages and add to register fragments.
            # Producer sends tiles in stage-outer / col_wg-inner order so the
            # (stage, col_wg)-th tile covers N columns
            #   [col_wg*num_stages*stageN + stage*stageN,
            #    col_wg*num_stages*stageN + (stage+1)*stageN).
            # For cta_group=2/MMA_M=128 (num_col_warp_groups=2): warps 0+1 own
            # col_wg=0 columns, warps 2+3 own col_wg=1 columns. All warps
            # participate in every pipeline step for correct barrier counts; only
            # the matching warp group adds the loaded values.
            var local_row, local_col = epilogue_applier.compute_staged_coords(
                UInt32(stage), 0, 0
            )

            comptime for col_wg in range(num_col_warp_groups):
                epilogue_pipeline.wait_producer()
                var epi_stage_idx = Int(epilogue_pipeline.consumer_stage())
                var tile_ptr = (
                    epilogue_tiles_base + epi_stage_idx * epilogue_tile_elems
                )

                comptime expected_col = (
                    col_wg * Self.num_stages * Self.stageN + stage * Self.stageN
                )
                var is_my_col_group = local_col == UInt32(expected_col)

                comptime for rep in range(Self.rep):
                    # In-tile column = rep*8 (always < stageN, so ldm_tidx = 0)
                    comptime col_in_tile = rep * 8
                    var ldm_row = local_row + UInt32(lane & 15)
                    var ldm_off = epi_load_sw(
                        ldm_row * UInt32(sub_tile_n) + UInt32(col_in_tile)
                    )
                    var epi_vals = ld_matrix[simd_width=4](
                        tile_ptr + Int(ldm_off)
                    )

                    if is_my_col_group:
                        upper_frag_casted[rep * 4] += epi_vals[0].cast[
                            Self.epilogue_dtype
                        ]()
                        upper_frag_casted[rep * 4 + 1] += epi_vals[1].cast[
                            Self.epilogue_dtype
                        ]()
                        upper_frag_casted[rep * 4 + 2] += epi_vals[2].cast[
                            Self.epilogue_dtype
                        ]()
                        upper_frag_casted[rep * 4 + 3] += epi_vals[3].cast[
                            Self.epilogue_dtype
                        ]()

                    comptime if Self.is_lower_frag_required:
                        var ldm_row_l = local_row + 16 + UInt32(lane & 15)
                        var ldm_off_l = epi_load_sw(
                            ldm_row_l * UInt32(sub_tile_n) + UInt32(col_in_tile)
                        )
                        var epi_vals_l = ld_matrix[simd_width=4](
                            tile_ptr + Int(ldm_off_l)
                        )

                        if is_my_col_group:
                            lower_frag_casted[rep * 4] += epi_vals_l[0].cast[
                                Self.epilogue_dtype
                            ]()
                            lower_frag_casted[rep * 4 + 1] += epi_vals_l[
                                1
                            ].cast[Self.epilogue_dtype]()
                            lower_frag_casted[rep * 4 + 2] += epi_vals_l[
                                2
                            ].cast[Self.epilogue_dtype]()
                            lower_frag_casted[rep * 4 + 3] += epi_vals_l[
                                3
                            ].cast[Self.epilogue_dtype]()

                _ = epilogue_pipeline.consumer_mbar(UInt32(epi_stage_idx))[
                    0
                ].arrive()
                epilogue_pipeline.consumer_step()

            # 4. Write to output SMEM and TMA store to GMEM
            var c_smem_tile = c_tiles[stage % 2]
            self._cast_frags_and_write_to_smem(
                upper_frag_casted,
                lower_frag_casted,
                c_smem_tile,
                UInt32(warp_id),
                UInt32(lane),
            )
            self._tma_store_to_gmem[stage](
                c_smem_tile,
                c_coord,
                batch_idx,
                UInt32(warp_id),
                UInt32(lane),
            )


# ===----------------------------------------------------------------------=== #
# StandardOutputWriter - default OutputWriter policy (local TMA store)
# ===----------------------------------------------------------------------=== #


struct StandardOutputWriter(OutputWriter):
    """Default `OutputWriter` policy: local TMA store via `TileWriter`.

    One peer, no cross-GPU synchronization. This is the writer policy
    `BlackwellMatmulSM100Kernel` uses unless a reduce-scatter policy is
    injected. Target hardware: SM100 (B200).
    """

    comptime needs_sync = False
    comptime num_peers = 1

    @staticmethod
    @always_inline
    def write_batched[
        tma_origin: ImmOrigin,
        c_type: DType,
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
        a_type: DType,
        accum_type: DType,
        block_tile_shape: IndexList[3],
        mma_shape: IndexList[3],
        opc: OutputPipelineConfig,
        c_swizzle: TensorMapSwizzle,
        transpose_c: Bool,
        c_smem_dim0: Int,
        c_smem_dim1: Int,
        num_output_stages: Int,
        num_output_warps: Int,
        elementwise_lambda_fn: Optional[elementwise_epilogue_type],
        elementwise_compute_lambda_fn: Optional[
            elementwise_compute_lambda_type
        ],
        register_based_epilogue: Bool,
    ](
        c_tma_ops: Pointer[
            Array[
                TMATensorTile[c_type, c_rank, c_tile_shape, c_desc_shape],
                Self.num_peers,
            ],
            tma_origin,
        ],
        c_tiles: SMemTileArray2DRowMajor[
            c_type, c_smem_dim0, c_smem_dim1, num_output_stages
        ],
        stage: OutputStage[opc],
        tile_coord: Tuple[UInt32, UInt32, UInt32],
        shape: Tuple[UInt32, UInt32],
        alpha: Float32 = Float32(1.0),
    ):
        """Local TMA store of one batched output tile (uses descriptor [0]).

        Parameters:
            tma_origin: Memory origin of the TMA descriptor pointer
                (inferred).
            c_type: Element dtype of the C output tensor (inferred).
            c_rank: Rank of the C output tensor (inferred).
            c_tile_shape: Per-tile shape of the C output (inferred).
            c_desc_shape: TMA descriptor shape for C (inferred).
            a_type: Element dtype of the A input matrix.
            accum_type: Accumulator dtype stored in TMEM.
            block_tile_shape: Block tile shape as (BM, BN, BK).
            mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
            opc: Output pipeline config bundling accumulator stages,
                stage stride, and CTA group.
            c_swizzle: TMA swizzle pattern for the C SMEM layout.
            transpose_c: Whether C is stored transposed.
            c_smem_dim0: Row dimension of the C SMEM tile.
            c_smem_dim1: Column dimension of the C SMEM tile.
            num_output_stages: Number of C SMEM pipeline stages.
            num_output_warps: Number of warps driving the output
                pipeline.
            elementwise_lambda_fn: Optional elementwise epilogue applied
                to fragments before the store.
            elementwise_compute_lambda_fn: Optional compute epilogue
                fused into the register path.
            register_based_epilogue: Whether the compute epilogue runs
                in registers (true) or SMEM (false).

        Args:
            c_tma_ops: Pointer to the array of TMA store descriptors for
                C.
            c_tiles: SMEM tile array for the C output.
            stage: OutputStage with pipeline, index, and TMEM handle.
            tile_coord: (m_tile, n_tile, batch) tile coordinates.
            shape: (M, N) problem dimensions.
            alpha: Scalar applied to fragments before the store
                (defaults to 1.0).
        """
        # The descriptor params (tma_origin, c_type, c_rank, c_tile_shape,
        # c_desc_shape) are inferred from the `c_tma_ops` ctor arg.
        var writer = TileWriter[
            a_type=a_type,
            accum_type=accum_type,
            block_tile_shape=block_tile_shape,
            mma_shape=mma_shape,
            opc=opc,
            c_swizzle=c_swizzle,
            transpose_c=transpose_c,
            c_smem_dim0=c_smem_dim0,
            c_smem_dim1=c_smem_dim1,
            num_output_stages=num_output_stages,
            num_output_warps=num_output_warps,
            elementwise_lambda_fn=elementwise_lambda_fn,
            elementwise_compute_lambda_fn=elementwise_compute_lambda_fn,
            register_based_epilogue=register_based_epilogue,
            batched=True,
            num_peers=1,
        ](c_tma_ops)
        writer.write_batched(c_tiles, stage, tile_coord, shape, alpha)
