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
"""Structured ping-pong matmul for AMD MI355X (CDNA4).

Entry point: AMDPingPongMatmul.run()
Host launcher: amd_ping_pong_matmul()
"""

from std.math import ceildiv
from std.sys import align_of, simd_width_of, size_of
from layout.tile_tensor import stack_allocation

from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.host import DeviceContext
from max.gpu.sync import schedule_barrier, s_waitcnt
from std.sys import llvm_intrinsic
from std.sys.intrinsics import readfirstlane

from std.utils import Index, IndexList, StaticTuple
from std.utils.numerics import get_accum_type

from layout import TensorLayout, TensorEngine, TileTensor
from layout.swizzle import Swizzle
from layout.tile_layout import row_major, col_major
from layout.tensor_core import num_matrix_reg

from std.bit import log2_floor

from ....utils import elementwise_epilogue_type

from pipeline.config import ScheduleConfig, SchedulingStrategy
from pipeline.pipeline_dsl import ScheduleEntry
from pipeline.program_builder import derive_safe_max_globals
from .amd_target import mi355x_target
from .amd_ping_pong_schedule import (
    build_schedule,
    PingPongOps,
    LOAD_A,
    LOAD_B,
    MMA_LOAD_A,
    MMA_LOAD_B,
    MMA,
)

from .matmul_mma import QuadrantMmaOp
from structured_kernels.amd_tile_io import RegTileWriter, TileLoaderLDS


# ===----------------------------------------------------------------------=== #
# KernelConfig: Ping-pong kernel shape configuration
# ===----------------------------------------------------------------------=== #


struct KernelConfig(ImplicitlyCopyable, Movable, Writable):
    """Block/warp/MMA shape configuration for ping-pong kernels."""

    var block_shape: IndexList[3]
    var warp_shape: IndexList[3]
    var mma_shape: IndexList[3]

    def __init__(
        out self,
        *,
        block_shape: IndexList[3],
        warp_shape: IndexList[3],
        mma_shape: IndexList[3],
    ):
        self.block_shape = block_shape
        self.warp_shape = warp_shape
        self.mma_shape = mma_shape

    @staticmethod
    def _write_index_list(
        mut writer: Some[Writer], list: IndexList, sep: StaticString
    ):
        comptime for i in range(list.size):
            if i != 0:
                writer.write(sep)
            writer.write(list[i])

    @always_inline
    def num_threads(self) -> Int:
        var num_warps = self.block_shape // self.warp_shape
        return num_warps.flattened_length() * WARP_SIZE

    def write_to(self, mut writer: Some[Writer]):
        writer.write("config_")
        Self._write_index_list(writer, self.block_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.warp_shape, "x")
        writer.write("_")
        Self._write_index_list(writer, self.mma_shape, "x")

    def write_repr_to(self, mut writer: Some[Writer]):
        self.write_to(writer)


struct AMDPingPongMatmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    config: KernelConfig,
    /,
    enable_swizzle: Bool,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
]:
    """Structured ping-pong matmul for AMD MI355X.

    8-warp double-buffered kernel with register-based DRAM→SMEM path.

    Parameters:
        a_type: Input A element type.
        b_type: Input B element type.
        c_type: Output C element type.
        config: KernelConfig with block/warp/mma shapes.
        enable_swizzle: Enable LDS bank conflict avoidance.
        elementwise_lambda_fn: Optional epilogue.
    """

    comptime BM = Self.config.block_shape[0]
    comptime BN = Self.config.block_shape[1]
    comptime BK = Self.config.block_shape[2]

    comptime WM = Self.config.warp_shape[0]
    comptime WN = Self.config.warp_shape[1]

    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]
    comptime MMA_K = Self.config.mma_shape[2]

    comptime num_warps_m = Self.BM // Self.WM
    comptime num_warps_n = Self.BN // Self.WN
    comptime total_warps = Self.num_warps_m * Self.num_warps_n

    comptime num_m_mmas = Self.WM // Self.MMA_M
    comptime num_n_mmas = Self.WN // Self.MMA_N
    comptime num_k_mmas = Self.BK // Self.MMA_K

    comptime in_type = Self.a_type

    comptime simd_width = simd_width_of[Self.in_type]()
    comptime accum_dtype = get_accum_type[Self.c_type]()
    comptime accum_width = (Self.MMA_M * Self.MMA_N) // WARP_SIZE

    comptime c_frag_size = Self.MMA_M * Self.MMA_N // WARP_SIZE

    # Half-tile dimensions.
    comptime half_BM = Self.WM
    comptime half_BN = Self.BN // 2
    comptime mma_tile_m = Self.WM // 2
    comptime mma_tile_n = Self.WN // 2

    # Quadrant counts.
    comptime quadrant_m_mmas = Self.num_m_mmas // 2
    comptime quadrant_n_mmas = Self.num_n_mmas // 2

    # Producer thread layout for RegTileLoader (warp-scope).
    comptime _thread_cols = Self.BK // Self.simd_width
    comptime _thread_rows = WARP_SIZE // Self._thread_cols

    @staticmethod
    def make_mma_swizzle() -> Swizzle:
        """Consumer swizzle for MMA LDS reads (element-space).

        AMD MI355X have 64 LDS banks x 4 bytes each. Without swizzling,
        the MMA thread access pattern causes 4-way bank conflicts. The
        swizzle XORs high-order address bits into the bank selection bits
        to distribute accesses across banks.

        Swizzle parameters:
        - log_tile: Number of bits to XOR, scales with MMA_K.
        - base: Log2 of read granularity in bytes (lds_frag_width * elem_size).
        - shift: Fixed at 4 for AMD LDS bank geometry.

        Configuration examples:
            BF16 16x16x32:  lds_frag=8  bytes=16  -> Swizzle(1, 4, 4)
            FP8  16x16x128: lds_frag=16 bytes=16  -> Swizzle(3, 4, 4)
            FP8  32x32x64:  lds_frag=32 bytes=32  -> Swizzle(2, 5, 4)

        Returns:
            Swizzle pattern for bank-conflict-free LDS access.
        """
        comptime mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
        comptime use_split_k = (
            Self.in_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
        )
        comptime lds_frag_width = 16 if use_split_k else mma_frag_width

        comptime log_tile = log2_floor(Self.MMA_K // 32) + 1
        comptime frag_bytes = lds_frag_width * size_of[Self.in_type]()
        comptime base = log2_floor(frag_bytes)

        return Swizzle(log_tile, base, 4)

    # Consumer swizzle (element-space, for MMA LDS reads).
    comptime mma_swizzle = Optional(
        Self.make_mma_swizzle()
    ) if Self.enable_swizzle else Optional[Swizzle]()

    # Producer swizzle (byte-space, for load_to_lds writes).
    # For N-byte elements, byte base = element base + log2(N).
    # BF16: element Swizzle(1,4,4) -> byte Swizzle(1,5,4)
    # FP8:  element Swizzle(3,4,4) -> byte Swizzle(3,4,4) (1-byte, no shift)
    comptime _elem_size = size_of[Self.in_type]()
    comptime _mma_frag_w = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime _use_split = (
        Self.in_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
    )
    comptime _lds_frag_w = 16 if Self._use_split else Self._mma_frag_w
    comptime _swizzle_log_tile = log2_floor(Self.MMA_K // 32) + 1
    comptime _swizzle_subtile_cols = 4 * Self.simd_width
    comptime _frag_bytes = Self._lds_frag_w * Self._elem_size
    comptime _swizzle_base = (
        log2_floor(Self._frag_bytes) if Self.in_type.is_float8() else (
            log2_floor(Self._swizzle_subtile_cols // 2)
            + log2_floor(Self._elem_size)
        )
    )
    comptime byte_swizzle = Optional(
        Swizzle(Self._swizzle_log_tile, Self._swizzle_base, 4)
    ) if Self.enable_swizzle else Optional[Swizzle]()

    # LDS lgkm counts for scheduling.
    comptime _mma_frag_width = (Self.MMA_M * Self.MMA_K) // WARP_SIZE
    comptime _use_split_lds = (
        Self.in_type.is_float8() and Self.MMA_M == 16 and Self.MMA_K == 128
    )
    comptime _lds_frag_width = (
        16 if Self._use_split_lds else Self._mma_frag_width
    )
    comptime _k_loads_per_mma = Self._mma_frag_width // Self._lds_frag_width
    comptime _ds_reads_per_frag = ceildiv(
        Self._lds_frag_width * size_of[Self.in_type](), 16
    )
    comptime LGKM_PER_LOAD_A = (
        Self.quadrant_m_mmas
        * Self.num_k_mmas
        * Self._k_loads_per_mma
        * Self._ds_reads_per_frag
    )
    comptime LGKM_PER_LOAD_B = (
        Self.quadrant_n_mmas
        * Self.num_k_mmas
        * Self._k_loads_per_mma
        * Self._ds_reads_per_frag
    )

    # Global vmcnt counts.
    comptime loads_per_row = Self.BK // Self.simd_width
    comptime rows_per_iter_8warp = (8 * WARP_SIZE) // Self.loads_per_row
    comptime VMCNT_PER_LOAD_A = Self.half_BM // Self.rows_per_iter_8warp
    comptime VMCNT_PER_LOAD_B = Self.half_BN // Self.rows_per_iter_8warp

    @staticmethod
    def validate_config():
        comptime assert (
            Self.BM % Self.WM == 0
        ), "Block M must be divisible by Warp M"
        comptime assert (
            Self.BN % Self.WN == 0
        ), "Block N must be divisible by Warp N"
        comptime assert (
            Self.BK % Self.MMA_K == 0
        ), "Block K must be divisible by MMA K"
        comptime assert (
            Self.WM % Self.MMA_M == 0
        ), "Warp M must be divisible by MMA M"
        comptime assert (
            Self.WN % Self.MMA_N == 0
        ), "Warp N must be divisible by MMA N"
        comptime assert (
            Self.total_warps == 8
        ), "Ping-pong kernel requires exactly 8 warps"
        comptime assert (
            Self.num_warps_m == 2
        ), "Ping-pong kernel requires 2 warps in M dimension"
        comptime assert Self.a_type == Self.b_type, "A/B must match"

    @__llvm_metadata(
        MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
            Int32(Self.config.num_threads())
        )
    )
    @__name(
        t"amd_ping_pong_matmul_{Self.a_type}_{Self.b_type}_{Self.c_type}_BM{Self.BM}_BN{Self.BN}_BK{Self.BK}_WM{Self.WM}_WN{Self.WN}"
    )
    @staticmethod
    def run[
        a_layout: TensorLayout,
        b_layout: TensorLayout,
        c_layout: TensorLayout,
        a_engine: TensorEngine,
        b_engine: TensorEngine,
        c_engine: TensorEngine,
    ](
        a: TileTensor[Self.a_type, a_layout, ImmutAnyOrigin, Engine=a_engine],
        b: TileTensor[Self.b_type, b_layout, ImmutAnyOrigin, Engine=b_engine],
        c: TileTensor[Self.c_type, c_layout, MutAnyOrigin, Engine=c_engine],
    ):
        """Structured ping-pong GEMM kernel entry point.

        Parameters:
            a_layout: Memory layout of the `a` input tile (inferred).
            b_layout: Memory layout of the `b` input tile (inferred).
            c_layout: Memory layout of the `c` output tile (inferred).
            a_engine: Engine of the `a` input tile (inferred).
            b_engine: Engine of the `b` input tile (inferred).
            c_engine: Engine of the `c` output tile (inferred).

        Args:
            a: LHS input matrix tile of shape `M` x `K` in `a_type`.
            b: RHS input matrix tile of shape `N` x `K` in `b_type`.
            c: Output matrix tile of shape `M` x `N` in `c_type`.
        """
        Self.validate_config()

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
        comptime consumer_swizzle = Self.mma_swizzle
        comptime producer_swizzle = Self.byte_swizzle
        comptime half_BM = Self.half_BM
        comptime half_BN = Self.half_BN
        comptime mma_tile_m = Self.mma_tile_m
        comptime mma_tile_n = Self.mma_tile_n

        var M = Int(a.dim[0]())
        comptime N = type_of(b).static_shape[0]
        comptime K = type_of(a).static_shape[1]

        var _lane_id = lane_id()
        var _warp_id = readfirstlane(warp_id())
        var m = block_idx.y * BM
        var n = block_idx.x * BN
        var warp_id_m, warp_id_n = divmod(_warp_id, Self.num_warps_n)
        var warp_group_id = _warp_id // 4

        # === Unified-dtype GMEM views ===
        var a_gmem = a.bitcast[Self.a_type]()
        var b_gmem = b.bitcast[Self.in_type]()

        # === SMEM: double-buffered half-tiles ===
        # A: 2 stages x 2 warp_m groups
        comptime a_half_layout = row_major[half_BM, BK]()
        var a_s0_g0 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s0_g1 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s1_g0 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )
        var a_s1_g1 = stack_allocation[Self.in_type, address_space=.SHARED](
            a_half_layout
        )

        # B: 2 stages x 2 warp_n groups
        comptime b_half_layout = row_major[half_BN, BK]()
        var b_s0_h0 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s0_h1 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s1_h0 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )
        var b_s1_h1 = stack_allocation[Self.in_type, address_space=.SHARED](
            b_half_layout
        )

        # Select this warp's tiles.
        var a_s0 = a_s0_g0 if warp_id_m == 0 else a_s0_g1
        var a_s1 = a_s1_g0 if warp_id_m == 0 else a_s1_g1

        comptime num_warps_n = Self.num_warps_n
        comptime warps_per_b_half = num_warps_n // 2
        var b_half_idx, b_local_n = divmod(warp_id_n, warps_per_b_half)
        var b_s0 = b_s0_h0 if b_half_idx == 0 else b_s0_h1
        var b_s1 = b_s1_h0 if b_half_idx == 0 else b_s1_h1

        # MMA sub-tiles: [stage][quadrant] views into half-tiles.
        var a_mma_s0_0 = a_s0.tile[mma_tile_m, BK](0, 0)
        var a_mma_s0_1 = a_s0.tile[mma_tile_m, BK](1, 0)
        var a_mma_s1_0 = a_s1.tile[mma_tile_m, BK](0, 0)
        var a_mma_s1_1 = a_s1.tile[mma_tile_m, BK](1, 0)

        var b_warp_s0 = b_s0.tile[WN, BK](b_local_n, 0)
        var b_warp_s1 = b_s1.tile[WN, BK](b_local_n, 0)
        var b_mma_s0_0 = b_warp_s0.tile[mma_tile_n, BK](0, 0)
        var b_mma_s0_1 = b_warp_s0.tile[mma_tile_n, BK](1, 0)
        var b_mma_s1_0 = b_warp_s1.tile[mma_tile_n, BK](0, 0)
        var b_mma_s1_1 = b_warp_s1.tile[mma_tile_n, BK](1, 0)

        # === DRAM→LDS loaders (direct DMA via load_to_lds) ===
        comptime _is_fp8 = Self.a_type.is_float8()
        comptime use_fp8_row_major = _is_fp8
        comptime byte_swizzle = Self.byte_swizzle

        # Pre-slice the per-block view: the SRD base advances to the
        # block's origin, so `load_tile(m_offset, k_offset)` works in
        # block-local coords. Avoids the +2 SGPR-uniform `m_anchor +
        # m_offset` and `k_anchor + k_offset` add ops at every
        # `load_tile` call site — these compound across the epilogue's
        # post-loop drain (+6 waitcnts, +0.85% FP8 8K³ regression
        # observed otherwise). The conv `TileLoaderLDSIm2col` sibling
        # still uses the anchor form because conv's im2col address
        # math can't fold the block origin into the SRD base.
        var a_block_gmem = a_gmem.tile[BM, K](Int(block_idx.y), 0)
        var b_block_gmem = b_gmem.tile[BN, K](Int(block_idx.x), 0)
        var a_loader = TileLoaderLDS[
            Self.in_type,
            half_BM,
            BK,
            stride=type_of(a_gmem).static_stride[0],
            num_loading_warps=8,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](a_block_gmem, _warp_id, Int(_lane_id))
        var b_loader = TileLoaderLDS[
            Self.in_type,
            half_BN,
            BK,
            stride=type_of(b_gmem).static_stride[0],
            num_loading_warps=8,
            swizzle=byte_swizzle,
            load_width=simd_width,
            use_full_tile_width=use_fp8_row_major,
        ](b_block_gmem, _warp_id, Int(_lane_id))

        # === MMA operator (full warp tile, quadrant accessed via methods) ===
        var mma_op = QuadrantMmaOp[
            out_type=Self.accum_dtype,
            in_type=Self.in_type,
            shape=Self.config.mma_shape,
            k_group_size=1,
            num_k_groups=Self.num_k_mmas,
            num_m_mmas=num_m_mmas,
            num_n_mmas=num_n_mmas,
            swizzle=consumer_swizzle,
        ]()

        # === Output writer ===
        comptime output_thread_layout = col_major[MMA_M, WARP_SIZE // MMA_M]()
        var c_writer = RegTileWriter[
            Self.c_type, Self.MMA_M, WARP_SIZE // Self.MMA_M
        ](c)

        # === Sync helpers ===
        @always_inline
        def s_barrier():
            llvm_intrinsic["llvm.amdgcn.s.barrier", NoneType]()

        @always_inline
        def s_setprio[priority: Int16]():
            llvm_intrinsic["llvm.amdgcn.s.setprio", NoneType](priority)

        # === Load helpers (DRAM→LDS via load_to_lds) ===
        # Load tiles indexed as [stage][which] from the stored half-tile tuples.
        var a_load_tiles = (
            (a_s0_g0, a_s0_g1),
            (a_s1_g0, a_s1_g1),
        )
        var b_load_tiles = (
            (b_s0_h0, b_s0_h1),
            (b_s1_h0, b_s1_h1),
        )

        @always_inline
        @__parameter
        def load_a[stage: Int, which: Int](k: Int):
            a_loader.load_tile(
                rebind[type_of(a_load_tiles[0][0])](
                    rebind[type_of(a_load_tiles[0])](a_load_tiles[stage])[which]
                ),
                m_offset=which * half_BM,
                k_offset=k,
            )

        @always_inline
        @__parameter
        def load_b[stage: Int, which: Int](k: Int):
            b_loader.load_tile(
                rebind[type_of(b_load_tiles[0][0])](
                    rebind[type_of(b_load_tiles[0])](b_load_tiles[stage])[which]
                ),
                m_offset=which * half_BN,
                k_offset=k,
            )

        # === MMA tile lookup: [stage][subtile] -> SMEM tile ===
        # Use tuples for comptime stage/subtile dispatch.
        var a_mma_tiles = (
            (a_mma_s0_0, a_mma_s0_1),
            (a_mma_s1_0, a_mma_s1_1),
        )
        var b_mma_tiles = (
            (b_mma_s0_0, b_mma_s0_1),
            (b_mma_s1_0, b_mma_s1_1),
        )

        # === Schedule ===
        comptime is_fp8 = Self.in_type.is_float8()
        comptime sched_config = ScheduleConfig(
            scheduling=SchedulingStrategy.CSP, auto_waits=True
        )
        comptime target = mi355x_target(
            vm_per_load_a=Self.VMCNT_PER_LOAD_A,
            vm_per_load_b=Self.VMCNT_PER_LOAD_B,
            # Uniform global distribution (max_globals=1) requires enough
            # MMA latency per k-tile to cover async LDS writes.
            # derive_safe_max_globals checks num_k_mmas >= 2. Additionally,
            # configs with very few total ops per quadrant block (e.g.
            # 32×32×64: 2*2*1=4) are unsafe even with num_k_mmas=2.
            max_globals=0 if (
                Self.num_k_mmas * Self.quadrant_m_mmas * Self.quadrant_n_mmas
                < 8
            ) else derive_safe_max_globals(Self.num_k_mmas),
        )
        comptime schedule = build_schedule[
            is_fp8,
            Self.LGKM_PER_LOAD_A,
            Self.LGKM_PER_LOAD_B,
        ](sched_config, target)

        @__parameter
        @always_inline
        def _bind[entry: ScheduleEntry](k_base: Int):
            comptime k_off = entry.op.k_offset.signed_bk_multiple()
            var k = k_base + k_off * BK
            comptime if entry.op.tag == LOAD_A:
                load_a[entry.op.stage, entry.op.subtile](k)
            elif entry.op.tag == LOAD_B:
                load_b[entry.op.stage, entry.op.subtile](k)
            elif entry.op.tag == MMA_LOAD_A:
                mma_op.load_a_quadrant[entry.op.subtile](
                    rebind[type_of(a_mma_tiles[0][0])](
                        rebind[type_of(a_mma_tiles[0])](
                            a_mma_tiles[entry.op.stage]
                        )[entry.op.subtile]
                    )
                )
            elif entry.op.tag == MMA_LOAD_B:
                mma_op.load_b_quadrant[entry.op.subtile](
                    rebind[type_of(b_mma_tiles[0][0])](
                        rebind[type_of(b_mma_tiles[0])](
                            b_mma_tiles[entry.op.stage]
                        )[entry.op.subtile]
                    )
                )
            elif entry.op.tag == MMA:
                mma_op.mma_quadrant[entry.op.stage, entry.op.subtile]()
            elif entry.op.tag == PingPongOps.BARRIER.value:
                s_barrier()
            elif entry.op.tag == PingPongOps.WAIT_VM.value:
                s_waitcnt[vmcnt=UInt32(entry.op.wait_value)]()
            elif entry.op.tag == PingPongOps.WAIT_LGKM.value:
                s_waitcnt[lgkmcnt=UInt32(entry.op.wait_value)]()
            elif entry.op.tag == PingPongOps.SET_PRIO.value:
                s_setprio[Int16(entry.op.wait_value)]()
            elif entry.op.tag == PingPongOps.SCHEDULE_BARRIER.value:
                schedule_barrier()

        # Prologue.
        comptime for i in range(schedule.warp_stagger_index):
            _bind[schedule.prologue[i]](0)

        if warp_group_id == 1:
            s_barrier()

        comptime for i in range(
            schedule.warp_stagger_index, len(schedule.prologue)
        ):
            _bind[schedule.prologue[i]](0)
        s_barrier()

        # Main loop.
        for k in range(BK * 2, K, BK * 2):
            comptime for i in range(len(schedule.kernel)):
                _bind[schedule.kernel[i]](k)

        # Epilogue.
        comptime for i in range(len(schedule.epilogue)):
            _bind[schedule.epilogue[i]](K)

        if warp_group_id == 0:
            s_barrier()

        # === Output Store ===
        var warp_tile_m = m + WM * warp_id_m
        var warp_tile_n = n + WN * warp_id_n

        if warp_tile_m < M and warp_tile_n < N:
            var c_reg = mma_op.accum_tile()

            comptime if Bool(Self.elementwise_lambda_fn):
                comptime epilogue_fn = Self.elementwise_lambda_fn.value()
                var lane_group, thread_m = divmod(Int(lane_id()), MMA_M)

                comptime for m_mma in range(num_m_mmas):
                    comptime for n_mma in range(num_n_mmas):
                        var m = warp_tile_m + m_mma * MMA_M + thread_m

                        if m < M:
                            comptime src_off = (
                                m_mma * num_n_mmas * c_frag_size
                                + n_mma * c_frag_size
                            )
                            var v = c_reg.raw_load[width=c_frag_size](
                                src_off
                            ).cast[Self.c_type]()

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
            else:
                var c_block = c.tile[BM, BN](block_idx.y, block_idx.x)
                var c_warp = c_block.tile[WM, WN](warp_id_m, warp_id_n)

                comptime vec_width = 4 if MMA_M == 32 else c_frag_size

                comptime for m_mma in range(num_m_mmas):
                    comptime for n_mma in range(num_n_mmas):
                        c_writer.store(
                            c_warp.tile[MMA_M, MMA_N](m_mma, n_mma).vectorize[
                                1, vec_width
                            ](),
                            c_reg.tile[1, c_frag_size](m_mma, n_mma),
                        )


@always_inline
def amd_ping_pong_matmul[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    //,
    enable_swizzle: Bool = True,
](
    a: TileTensor[mut=False, a_type, ...],
    b: TileTensor[mut=False, b_type, ...],
    c: TileTensor[mut=True, c_type, ...],
    ctx: DeviceContext,
) raises:
    """Host launcher for the structured ping-pong matmul.

    Parameters:
        a_type: Element type of the `a` input matrix (inferred).
        b_type: Element type of the `b` input matrix (inferred).
        c_type: Element type of the `c` output matrix (inferred).
        enable_swizzle: Enable LDS bank-conflict avoidance swizzle
            (defaults to `True`).

    Args:
        a: LHS input matrix tile of shape `M` x `K`.
        b: RHS input matrix tile of shape `N` x `K`.
        c: Output matrix tile of shape `M` x `N`.
        ctx: Device context used to enqueue the kernel launch.
    """
    comptime assert a_type == b_type, "A and B must have the same type"
    comptime assert (
        a_type == .bfloat16 or a_type.is_float8()
    ), "A must be bfloat16 or float8_e4m3fn"

    comptime is_fp8 = a_type.is_float8()

    var N = Int(c.dim[1]())
    var M = Int(c.dim[0]())

    @always_inline
    @__parameter
    def run_kernel[config: KernelConfig]() raises:
        comptime kernel = AMDPingPongMatmul[
            a_type,
            b_type,
            c_type,
            config,
            enable_swizzle,
        ].run[
            type_of(a).LayoutType,
            type_of(b).LayoutType,
            type_of(c).LayoutType,
            type_of(a).Engine,
            type_of(b).Engine,
            type_of(c).Engine,
        ]

        ctx.enqueue_function[kernel](
            a,
            b,
            c,
            grid_dim=(
                ceildiv(N, config.block_shape[1]),
                ceildiv(M, config.block_shape[0]),
            ),
            block_dim=config.num_threads(),
        )

    # Dispatch: FP8 uses 16x16x128 MMA with BK=128.
    # BF16 uses 16x16x32 with BK=64.
    #
    # Skinny 128x256 wins in a mid-M band whose boundaries depend on N:
    #   Large N (>=4096): skinny wins for 150 < M <= 512
    #   Small N (<4096):  skinny wins for 512 < M <= 2048
    # Outside those bands, baseline 256x256 is faster.
    comptime if is_fp8:
        comptime BM = 256
        comptime BN = 256
        comptime BK = 128

        # Standard 256x256 config
        comptime config_16x16 = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(16, 16, 128),
        )

        # Skinny 128x256 config (170 FLOP/B, 96 KB LDS)
        comptime skinny_config = KernelConfig(
            block_shape=Index(128, BN, BK),
            warp_shape=Index(64, BN // 4, BK),
            mma_shape=Index(16, 16, 128),
        )

        # Alternative 256x256 config with 32x32x64 MMA
        comptime config_32x32 = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(32, 32, 64),
        )

        # Small K configuration with 32x32x64 MMA and BK=64
        comptime config_32x32x64 = KernelConfig(
            block_shape=Index(BM, BN, 64),
            warp_shape=Index(BM // 2, BN // 4, 64),
            mma_shape=Index(32, 32, 64),
        )

        # Skinny wins in a mid-M band whose boundaries depend on N:
        #   Large N (>=4096): skinny wins for 150 < M <= 512
        #   Small N (<4096):  skinny wins for 512 < M <= 2048
        if (N >= 4096 and 150 < M <= 512) or (N < 4096 and 512 < M <= 2048):
            run_kernel[skinny_config]()
        else:
            run_kernel[config_16x16]()
            # run_kernel[config_32x32]()
    else:
        # BF16: 16x16x32 MMA with BK=64
        comptime BM = 256
        comptime BN = 256
        comptime BK = 64
        comptime config = KernelConfig(
            block_shape=Index(BM, BN, BK),
            warp_shape=Index(BM // 2, BN // 4, BK),
            mma_shape=Index(16, 16, 32),
        )
        run_kernel[config]()
