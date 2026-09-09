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
"""Pure TileTensor structured AMD matmul kernel.

Uses RegTileLoader for DRAM to regs, blocked-product SMEM with
Swizzle(3,0,1), StructuredMmaOp for per-k-tile MMA, and RegTileWriter
for output. Schedule-driven pipeline via build_default_matmul_schedule.

Entry point: AMDMatmul.run()
"""

from std.bit import log2_floor
from std.collections import Optional
from std.sys import align_of, simd_width_of

from .._multistage_gemm_gpu import (
    warp_split_k_reduction,
    WarpSplitKReductionSMem,
)

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.sync import barrier
from max.gpu.sync import (
    AMDScheduleBarrierMask,
    schedule_barrier,
    schedule_group_barrier,
)
from layout import TensorEngine, TensorLayout, TileTensor
from layout.swizzle import Swizzle
from layout.tile_layout import row_major, col_major
from layout.tile_tensor import stack_allocation

from std.utils import IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from ....utils import elementwise_epilogue_type
from ....utils_gpu import MatmulConfig

from .matmul_mma import MmaOp
from .amd_matmul_schedule import build_default_matmul_schedule
from .amd_matmul_schedule import (
    DefaultMatmulOps,
    COMPUTE,
    LOAD_DRAM,
    LOAD_FRAG,
    STORE_SMEM,
)
from pipeline.pipeline_dsl import ScheduleEntry
from structured_kernels.amd_tile_io import (
    RegTileLoader,
    RegTileWriter,
    RegTileWriterLDS,
)

comptime SCHED_MASK_DS_READ = 0
comptime SCHED_MASK_DS_WRITE = 1
comptime SCHED_MASK_VMEM_READ = 2
comptime SCHED_MASK_MFMA = 3


struct AMDMatmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Pure TileTensor structured matmul for AMD GPUs.

    Schedule-driven single-buffer pipeline. All data movement uses
    TileTensor: no LayoutTensor anywhere.

    Parameters:
        a_type: Element type of the A input matrix.
        b_type: Element type of the B input matrix.
        c_type: Element type of the C output matrix.
        transpose_b: Whether B is stored transposed as `[N, K]`; must be
            `True`.
        config: Tile and warp shapes, MMA shape, and thread count for the
            kernel.
        elementwise_lambda_fn: Optional epilogue applied elementwise to the
            output (defaults to `None`).
    """

    comptime accum_type = get_accum_type[Self.a_type]()

    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2] * Int(
        Self.config.num_warp_k_partitions
    )
    comptime WM = Self.config.warp_tile_shape[0]
    comptime WN = Self.config.warp_tile_shape[1]
    comptime WK = Self.config.warp_tile_shape[2]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime simd_width = simd_width_of[Self.a_type]()

    comptime num_warps_m = Self.BM // Self.WM
    comptime num_warps_n = Self.BN // Self.WN
    comptime num_warps_k = Self.BK // Self.WK

    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N

    comptime frag_size = Self.MMA_M * Self.MMA_K // WARP_SIZE
    comptime c_frag_size = Self.MMA_M * Self.MMA_N // WARP_SIZE
    comptime k_group_size = Self.simd_width // Self.frag_size
    comptime k_tile_size = Self.MMA_K * Self.k_group_size
    comptime num_k_tiles = Self.WK // Self.k_tile_size
    comptime num_k_mmas = Self.WK // Self.MMA_K

    @staticmethod
    def make_mma_swizzle() -> Swizzle:
        """Swizzle for blocked-product SMEM layout (LDS bank conflict avoidance).

        The blocked-product layout stores k-tile elements in contiguous blocks.
        The MMA distribute reads these in col_major[MMA_M, WARP_SIZE/MMA_M]
        order, giving WARP_SIZE/MMA_M vector columns per block. The swizzle
        XORs enough bits to spread those column groups across LDS banks.

        Unlike the ping-pong make_mma_swizzle (element-space for row-major
        SMEM with base/shift derived from fragment bytes), this operates in
        the vector-index space of each blocked-product chunk (base=0, shift=1).

        Returns:
            Swizzle for bank-conflict-free blocked-product LDS access.
        """
        comptime cols_per_blk = WARP_SIZE // Self.MMA_M
        comptime bits = log2_floor(cols_per_blk) + 1
        return Swizzle(bits, 0, 1)

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads())
        )
    )
    @__name(
        t"amd_matmul_{Self.a_type}_{Self.b_type}_{Self.c_type}_BM{Self.BM}_BN{Self.BN}_BK{Self.BK}_WM{Self.WM}_WN{Self.WN}"
    )
    @staticmethod
    def run[
        c_layout: TensorLayout,
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_engine: TensorEngine,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
    ](
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
        a: TileTensor[Self.a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
    ):
        """TileTensor GEMM matching original kernel config exactly.

        Uses StructuredMmaOp with per-k-tile load_frag/mma dispatch,
        original warp index order, and schedule-driven pipeline.

        Parameters:
            c_layout: Tensor layout of the output C tile.
            a_layout: Tensor layout of the input A tile.
            b_layout: Tensor layout of the input B tile.
            c_engine: Engine of the output C tile.
            a_engine: Engine of the input A tile.
            b_engine: Engine of the input B tile.

        Args:
            c: Output tile of shape `[M, N]` accumulating the matmul
                result.
            a: Input A tile of shape `[M, K]` in row-major block layout.
            b: Input B tile of shape `[N, K]` (transposed, `transpose_b`
                is `True`).
        """
        comptime assert Self.transpose_b, "transpose_b must be True"
        comptime assert Self.a_type == Self.b_type, "a/b must match"

        comptime BM = Self.BM
        comptime BN = Self.BN
        comptime BK = Self.BK
        comptime WM = Self.WM
        comptime WN = Self.WN
        comptime MMA_M = Self.MMA_M
        comptime MMA_N = Self.MMA_N
        comptime num_m_mmas = Self.num_m_mmas
        comptime num_n_mmas = Self.num_n_mmas
        comptime c_frag_size = Self.c_frag_size
        comptime simd_width = Self.simd_width
        comptime num_threads = Int(Self.config.num_threads())

        comptime swizzle = Optional[Swizzle](Self.make_mma_swizzle())

        var M = Int(a.dim[0]())
        comptime K = type_of(a).static_shape[1]
        comptime N = type_of(b).static_shape[0]
        comptime assert N > 0, "N must be known at compile time"

        # Warp index: match original divmod(warp_id, num_warps_n) order.
        var _warp_id = warp_id()
        var warp_km, warp_n = divmod(_warp_id, Self.num_warps_n)
        var warp_k, warp_m = divmod(warp_km, Self.num_warps_m)

        # === GMEM views ===
        var a_gmem = a.bitcast[Self.a_type]()
        var b_gmem = b.bitcast[Self.a_type]()

        # === SMEM: row_major tiles for the full BK-wide block ===
        comptime k_tile_size = Self.MMA_K * Self.k_group_size

        var a_smem = stack_allocation[Self.a_type, address_space=.SHARED](
            row_major[BM, BK]()
        )
        var b_smem = stack_allocation[Self.a_type, address_space=.SHARED](
            row_major[BN, BK]()
        )

        # === DRAM→regs loaders ===
        comptime load_thread_cols = BK // simd_width
        comptime load_thread_rows = num_threads // load_thread_cols
        comptime a_reg_elems = BM * BK // num_threads
        comptime b_reg_elems = BN * BK // num_threads
        var a_load_reg = stack_allocation[Self.a_type, address_space=.LOCAL](
            row_major[1, a_reg_elems]()
        )
        var b_load_reg = stack_allocation[Self.a_type, address_space=.LOCAL](
            row_major[1, b_reg_elems]()
        )

        # Block-row tiles with OOB clamping from the full tensors.
        var a_blockrow = a_gmem.tile[BM, K](block_idx.y, 0)
        var b_blockrow = b_gmem.tile[BN, K](block_idx.x, 0)
        comptime load_layout = row_major[load_thread_rows, load_thread_cols]()
        var a_loader = RegTileLoader[Self.a_type, load_layout](
            a_blockrow,
            bounds_from=a_gmem,
        )
        var b_loader = RegTileLoader[Self.a_type, load_layout](
            b_blockrow,
            bounds_from=b_gmem,
        )

        # === MMA operator: MmaOp with per-k-tile API ===
        var mma_op = MmaOp[
            out_type=Self.accum_type,
            in_type=Self.a_type,
            shape=Self.config.mma_shape,
            k_group_size=Self.k_group_size,
            num_k_tiles=Self.num_k_tiles,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            swizzle=swizzle,
        ]()

        # === Output writer ===
        comptime output_thread_layout = col_major[MMA_M, WARP_SIZE // MMA_M]()
        var c_writer = RegTileWriter[Self.c_type, MMA_M, WARP_SIZE // MMA_M](c)

        # === Helpers ===
        var k_counter = 0

        @always_inline
        @__parameter
        def load_tiles_from_dram():
            var a_block = a_blockrow.tile[BM, BK](0, k_counter)
            var b_block = b_blockrow.tile[BN, BK](0, k_counter)
            a_loader.load(a_load_reg, a_block.vectorize[1, simd_width]())
            b_loader.load(b_load_reg, b_block.vectorize[1, simd_width]())
            k_counter += 1

        @always_inline
        @__parameter
        def copy_tiles_to_smem():
            comptime thread_layout = row_major[
                load_thread_rows, load_thread_cols
            ]()
            RegTileWriterLDS[
                thread_layout,
                swizzle=swizzle,
                num_threads=num_threads,
            ].copy_blocked[k_tile_size](a_smem, a_load_reg)
            RegTileWriterLDS[
                thread_layout,
                swizzle=swizzle,
                num_threads=num_threads,
            ].copy_blocked[k_tile_size](b_smem, b_load_reg)

        # === Warp-K SMEM offsets for split-K ===
        # Each warp-K partition reads its own WK-wide slice of the BK-wide
        # SMEM tile. Offset is zero when num_warps_k == 1.
        var a_warp_k_off = warp_k * (Self.num_k_tiles * BM * k_tile_size)
        var b_warp_k_off = warp_k * (Self.num_k_tiles * BN * k_tile_size)

        # === Schedule ===
        comptime threads_per_row = BK // simd_width
        comptime rows_per_thread_block = num_threads // threads_per_row
        comptime a_loads_per_thread = BM // rows_per_thread_block
        comptime b_loads_per_thread = BN // rows_per_thread_block

        comptime schedule = build_default_matmul_schedule[
            num_k_tiles=Self.num_k_tiles,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            num_k_mmas=Self.num_k_mmas,
            MMA_M=MMA_M,
            MMA_N=MMA_N,
            a_loads_per_thread=a_loads_per_thread,
            b_loads_per_thread=b_loads_per_thread,
        ]()

        @__parameter
        @always_inline
        def _bind[entry: ScheduleEntry]():
            comptime if entry.op.tag == LOAD_DRAM:
                load_tiles_from_dram()
            elif entry.op.tag == STORE_SMEM:
                copy_tiles_to_smem()
            elif entry.op.tag == LOAD_FRAG:
                # Block-local SMEM views for this k-tile within this
                # warp-K partition's slice of the BK-wide SMEM buffer.
                # Offset is in blocked-product element order, not row-major.
                comptime k = entry.op.subtile
                var a_blk = TileTensor(
                    a_smem.ptr + a_warp_k_off + k * BM * k_tile_size,
                    row_major[BM, k_tile_size](),
                )
                var b_blk = TileTensor(
                    b_smem.ptr + b_warp_k_off + k * BN * k_tile_size,
                    row_major[BN, k_tile_size](),
                )
                mma_op.load_frag[k](
                    a_blk.tile[WM, k_tile_size](warp_m, 0),
                    b_blk.tile[WN, k_tile_size](warp_n, 0),
                )
            elif entry.op.tag == COMPUTE:
                mma_op.mma[entry.op.subtile]()
            elif entry.op.tag == DefaultMatmulOps.BARRIER.value:
                barrier()
            elif entry.op.tag == DefaultMatmulOps.SCHEDULE_BARRIER.value:
                schedule_barrier()
            elif entry.op.tag == DefaultMatmulOps.SCHED_GROUP_BARRIER.value:
                comptime sub = entry.op.subtile
                comptime wait = entry.op.wait_value
                comptime if sub == SCHED_MASK_DS_READ:
                    schedule_group_barrier(
                        AMDScheduleBarrierMask.DS_READ, Int32(wait), 0
                    )
                elif sub == SCHED_MASK_DS_WRITE:
                    schedule_group_barrier(
                        AMDScheduleBarrierMask.DS_WRITE, Int32(wait), 0
                    )
                elif sub == SCHED_MASK_VMEM_READ:
                    schedule_group_barrier(
                        AMDScheduleBarrierMask.VMEM_READ, Int32(wait), 0
                    )
                elif sub == SCHED_MASK_MFMA:
                    schedule_group_barrier(
                        AMDScheduleBarrierMask.MFMA, Int32(wait), 0
                    )

        # Prologue.
        comptime for i in range(len(schedule.prologue)):
            _bind[schedule.prologue[i]]()

        # Main K-loop.
        for _ in range(2, K // BK):
            comptime for i in range(len(schedule.kernel)):
                _bind[schedule.kernel[i]]()

        # Epilogue.
        comptime for i in range(len(schedule.epilogue)):
            _bind[schedule.epilogue[i]]()

        # === Split-K reduction ===
        # When multiple warps partition the K dimension, each warp-K
        # partition accumulates a partial result. Tree-reduce them via
        # SMEM so warp_k=0 holds the final sum.
        var c_reg = mma_op.accum_tile()

        comptime if Self.num_warps_k > 1:
            var reduction_smem = WarpSplitKReductionSMem[
                Self.accum_type, BM, BN, Self.num_warps_k
            ].stack_allocation()
            warp_split_k_reduction[
                BM, BN, num_threads // Self.num_warps_k, Self.num_warps_k
            ](warp_k, c_reg.to_layout_tensor(), reduction_smem.ptr)

            if warp_k != 0:
                return

        # === Output ===
        comptime if Bool(Self.elementwise_lambda_fn):
            # Epilogue path with OOB masking.
            comptime epilogue_fn = Self.elementwise_lambda_fn.value()
            var lane_group, thread_m = divmod(Int(lane_id()), MMA_M)
            var warp_tile_m = Int(block_idx.y) * BM + warp_m * WM
            var warp_tile_n = Int(block_idx.x) * BN + warp_n * WN

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    var m = warp_tile_m + m_mma * MMA_M + thread_m

                    if m < M:
                        comptime src_off = (
                            m_mma * num_n_mmas * c_frag_size
                            + n_mma * c_frag_size
                        )
                        var v = c_reg.raw_load[width=c_frag_size](src_off).cast[
                            Self.c_type
                        ]()

                        comptime if MMA_M == 32:
                            for e in range(c_frag_size):
                                var col = (
                                    warp_tile_n
                                    + n_mma * MMA_N
                                    + (e // 4) * 8
                                    + lane_group * 4
                                    + (e % 4)
                                )
                                if col < N:
                                    epilogue_fn[
                                        alignment=align_of[
                                            Scalar[Self.c_type]
                                        ]()
                                    ](
                                        IndexList[2](m, col),
                                        SIMD[Self.c_type, 1](v[e]),
                                    )
                        else:
                            var n = (
                                warp_tile_n
                                + n_mma * MMA_N
                                + lane_group * c_frag_size
                            )
                            if n < N:
                                epilogue_fn[
                                    alignment=align_of[
                                        SIMD[Self.c_type, c_frag_size]
                                    ]()
                                ](IndexList[2](m, n), v)
        elif N % BN != 0:
            # Boundary path: N not block-aligned, per-element OOB store.
            var lane_group, thread_m = divmod(Int(lane_id()), MMA_M)
            var warp_tile_m = Int(block_idx.y) * BM + warp_m * WM
            var warp_tile_n = Int(block_idx.x) * BN + warp_n * WN

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    var m = warp_tile_m + m_mma * MMA_M + thread_m

                    if m < M:
                        comptime src_off = (
                            m_mma * num_n_mmas * c_frag_size
                            + n_mma * c_frag_size
                        )
                        var v = c_reg.raw_load[width=c_frag_size](src_off).cast[
                            Self.c_type
                        ]()

                        comptime if MMA_M == 32:
                            for e in range(c_frag_size):
                                var col = (
                                    warp_tile_n
                                    + n_mma * MMA_N
                                    + (e // 4) * 8
                                    + lane_group * 4
                                    + (e % 4)
                                )
                                if col < N:
                                    c.raw_store(m * N + col, v[e])
                        else:
                            var n = (
                                warp_tile_n
                                + n_mma * MMA_N
                                + lane_group * c_frag_size
                            )
                            if n < N:
                                c.raw_store[width=c_frag_size](m * N + n, v)
        else:
            # Fast path: N is block-aligned, no OOB checks needed.
            var c_block = c.tile[BM, BN](block_idx.y, block_idx.x)
            var c_warp = c_block.tile[WM, WN](warp_m, warp_n)

            comptime vec_width = 4 if MMA_M == 32 else c_frag_size

            comptime for m_mma in range(num_m_mmas):
                comptime for n_mma in range(num_n_mmas):
                    c_writer.store(
                        c_warp.tile[MMA_M, MMA_N](m_mma, n_mma).vectorize[
                            1, vec_width
                        ](),
                        c_reg.tile[1, c_frag_size](m_mma, n_mma),
                    )
