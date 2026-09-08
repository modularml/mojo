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

"""Low-level epilogue components for SM100 matrix multiplication.

This module provides modular building blocks for the output pipeline:

1. **store_fragment_to_smem**: Register to shared memory via st.matrix instructions
2. **TMEMToSMemWriter**: Write TMEM accumulators to shared memory
3. **TMAStoreExecutor**: Execute TMA stores with proper SMEM tiling
4. **EpilogueApplier**: Apply element-wise operations on fragments

The SM100 epilogue pipeline flows as:
    TMEM (accumulators) → Registers → SMEM → GMEM (via TMA)
"""

from std.sys import align_of, size_of, simd_width_of

from max.gpu import WARP_SIZE, lane_id, warp_id
from max.gpu.primitives.cluster import elect_one_sync
from max.gpu.memory import (
    fence_async_view_proxy,
    cp_async_bulk_tensor_reduce_global_shared_cta,
    ReduceOp,
)
from max.gpu.sync import cp_async_bulk_commit_group
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from structured_kernels.barriers import WarpGroupBarrier
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    DefaultEngine,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from layout.tile_layout import Layout as InternalLayout
from layout.layout import blocked_product, upcast, zipped_divide
from layout.runtime_tuple import idx2crd, crd2idx as rt_crd2idx
from layout.swizzle import Swizzle, make_swizzle as _make_swizzle
from layout.tma_async import TMATensorTile
from std.utils.index import Index, IndexList
from linalg.utils import (
    elementwise_compute_lambda_type,
    elementwise_epilogue_type,
)
from std.utils.fast_div import FastDiv
from std.utils.static_tuple import StaticTuple
from std.math.uutils import udivmod

# TileTensor-based types for SMemEpilogueWriter constructor
from structured_kernels.tile_types import SMemTileArray2DRowMajor


# =============================================================================
# tma_wait_pipelined - Pipelined TMA wait helper
# =============================================================================


@always_inline
def tma_wait_pipelined[
    c_type: DType,
    tma_rank: Int,
    tile_shape: IndexList[tma_rank],
    desc_shape: IndexList[tma_rank],
    is_last_stage: Bool,
](c_tma_op: TMATensorTile[c_type, tma_rank, tile_shape, desc_shape]):
    """Wait for TMA stores with pipelining.

    For SM100 output pipeline:
    - Non-last stages: Keep 1 store in flight for pipelining
    - Last stage: Wait for all stores to complete
    """

    comptime if is_last_stage:
        c_tma_op.wait_group[0]()  # Wait for all stores
    else:
        c_tma_op.wait_group[1]()  # Keep 1 store in flight


# =============================================================================
# AccumTile - Accumulator tile (upper + lower fragments) for writing
# =============================================================================


struct AccumTile[dtype: DType, size: Int](Copyable, Movable):
    """Upper + lower TMEM fragments (16 rows each) for SM100 output."""

    var upper: Array[Scalar[Self.dtype], Self.size]
    var lower: Array[Scalar[Self.dtype], Self.size]

    @always_inline
    def __init__(
        out self,
        upper: Array[Scalar[Self.dtype], Self.size],
        lower: Array[Scalar[Self.dtype], Self.size],
    ):
        self.upper = upper.copy()
        self.lower = lower.copy()


# =============================================================================
# AccumBarrier - Accumulator pipeline barrier helper
# =============================================================================


struct AccumBarrier[cta_group: Int](TrivialRegisterPassable):
    """Pipeline barrier helper for single-CTA vs 2-CTA arrival patterns."""

    @staticmethod
    @always_inline
    def arrive(pipeline: ProducerConsumerPipeline, stage: UInt32):
        """Signal accumulator arrival on pipeline barrier."""

        comptime if Self.cta_group == 1:
            from max.gpu.sync import mbarrier_arrive

            _ = mbarrier_arrive(rebind[MbarPtr](pipeline.consumer_mbar(stage)))
        else:
            from max.gpu.sync import umma_arrive_leader_cta

            umma_arrive_leader_cta(
                rebind[MbarPtr](pipeline.consumer_mbar(stage))
            )


# Import for AccumBarrier
from structured_kernels.pipeline import ProducerConsumerPipeline
from structured_kernels.pipeline_backend import MbarPtr


# =============================================================================
# store_fragment_to_smem - Static helper for st.matrix operations
# =============================================================================


@always_inline
def st_shared_frag_to_smem[
    swizzle: Swizzle,
    stageN: Int,
    transpose_c: Bool,
    vec_dtype: DType,
    vec_size: Int,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: TileTensor[
        address_space=.SHARED, Engine=DefaultEngine[element_width=1], ...
    ],
    warp_offset: UInt32 = 0,
):
    """Store a register fragment to SMEM via plain st.shared, applying the
    SMEM swizzle to each offset.

    Used for output dtypes that cannot go through `st_matrix` (which is
    `.b16`-only): `float8_e4m3fn` (SWIZZLE_NONE, so `swizzle` is the identity
    and `warp_offset` is 0 -- reproducing the original un-swizzled fp8 store)
    and `float32` (32B/64B/128B swizzle). The fp32 stores stay swizzle-safe
    because the swizzle keeps the low 16 bytes (4 fp32) contiguous and the
    per-store offsets are even.
    """
    comptime assert (
        dst.dtype == .float32 or stageN == 16
    ), "stageN must be 16 for FP8 output type"
    comptime assert vec_dtype == dst.dtype and dst.dtype in (
        DType.float8_e4m3fn,
        DType.float32,
    ), "vec_dtype and dst.dtype must match and be float8_e4m3fn or float32"

    comptime load_width = 2
    comptime repeats = stageN // 8
    comptime assert (
        vec_size // 4
    ) == repeats, "vec_size must be divisible by 4 and equal to repeats * 4"

    comptime stride0 = UInt32(dst.static_stride[0])
    comptime stride1 = UInt32(dst.static_stride[1])

    var coords = FragmentCoords[stageN, repeats](UInt32(lane_id()))
    var top = coords.top_upper
    var bot = coords.bottom_upper

    comptime for rep in range(repeats):
        comptime inc = rep * 8
        comptime offset = rep * 4

        var top_row = top[0]
        var top_col = top[1] + UInt32(inc)
        var bot_row = bot[0]
        var bot_col = bot[1] + UInt32(inc)

        var elem0 = vec[offset]
        var elem1 = vec[offset + 1]
        var elem2 = vec[offset + 2]
        var elem3 = vec[offset + 3]

        var dst_ptr = dst._storage.unsafe_mut_cast[True]()

        comptime if transpose_c:
            var m0n0 = (
                swizzle(top_col * stride0 + top_row * stride1 + warp_offset)
                - warp_offset
            )
            var m0n1 = (
                swizzle(
                    (top_col + 1) * stride0 + top_row * stride1 + warp_offset
                )
                - warp_offset
            )
            var m1n0 = (
                swizzle(bot_col * stride0 + bot_row * stride1 + warp_offset)
                - warp_offset
            )
            var m1n1 = (
                swizzle(
                    (bot_col + 1) * stride0 + bot_row * stride1 + warp_offset
                )
                - warp_offset
            )

            dst_ptr.store[alignment=align_of[SIMD[dst.dtype, 1]]()](
                m0n0, SIMD[dst.dtype, 1](elem0)
            )
            dst_ptr.store[alignment=align_of[SIMD[dst.dtype, 1]]()](
                m0n1, SIMD[dst.dtype, 1](elem1)
            )
            dst_ptr.store[alignment=align_of[SIMD[dst.dtype, 1]]()](
                m1n0, SIMD[dst.dtype, 1](elem2)
            )
            dst_ptr.store[alignment=align_of[SIMD[dst.dtype, 1]]()](
                m1n1, SIMD[dst.dtype, 1](elem3)
            )

        else:
            var top_elems = SIMD[vec_dtype, load_width](elem0, elem1).cast[
                dst.dtype
            ]()
            var bot_elems = SIMD[vec_dtype, load_width](elem2, elem3).cast[
                dst.dtype
            ]()

            var top_ptr_offset = swizzle(top_row * stride0 + top_col * stride1)
            var bot_ptr_offset = swizzle(bot_row * stride0 + bot_col * stride1)

            dst_ptr.store[alignment=align_of[type_of(top_elems)]()](
                top_ptr_offset, top_elems
            )
            dst_ptr.store[alignment=align_of[type_of(bot_elems)]()](
                bot_ptr_offset, bot_elems
            )


@always_inline
def store_fragment_to_smem[
    vec_dtype: DType,
    vec_size: Int,
    //,
    swizzle: Swizzle,
    stageN: Int,
    transpose_c: Bool = False,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](
    vec: Array[Scalar[vec_dtype], vec_size],
    dst: TileTensor[
        address_space=.SHARED, Engine=DefaultEngine[element_width=1], ...
    ],
    warp_offset: UInt32 = 0,
):
    """Store fragment to SMEM via st.matrix instruction for bf16 output type and st.shared instruction for FP8 output type.
    """

    # st_matrix is .b16-only, so fp8 and fp32 use plain swizzled st.shared.
    comptime if dst.dtype in (DType.float8_e4m3fn, DType.float32):
        return st_shared_frag_to_smem[swizzle, stageN, transpose_c](
            vec, dst, warp_offset
        )

    from max.gpu.compute.mma import st_matrix
    from std.memory import bitcast

    comptime c_type = dst.dtype
    comptime stsmx_row_size = 32 // size_of[
        c_type
    ]() if stageN % 16 == 0 else 16 // size_of[c_type]()
    comptime stsmx_lane_size = 16 // size_of[c_type]()
    comptime stmtx_simd_width = 4 if stageN % 16 == 0 else 2
    comptime stride0 = dst.static_stride[0]
    comptime stride1 = dst.static_stride[1]
    comptime shape0 = dst.static_shape[
        1
    ] if not transpose_c else dst.static_shape[0]
    comptime stsmx_tile_offset = (
        stride0 if transpose_c else stride1
    ) * stsmx_row_size

    var lane_id = lane_id()
    var stsm_lane_offset: UInt32

    comptime if transpose_c:
        comptime trans_layout = InternalLayout(
            Coord(Idx[8], Idx[2], Idx[2]),
            Coord(Idx[stride0], Idx[8 * stride1], Idx[8 * stride0]),
        )
        stsm_lane_offset = UInt32(trans_layout(lane_id))
    else:
        stsm_lane_offset = (
            UInt32(lane_id & 15) * UInt32(stride0) + UInt32(lane_id >> 4) * 8
        )

    comptime for i in range(shape0 // stsmx_row_size):
        comptime n_offset = i * stsmx_tile_offset
        var offset: UInt32

        comptime if transpose_c:
            offset = (
                swizzle(stsm_lane_offset + UInt32(n_offset) + warp_offset)
                - warp_offset
            )
        else:
            offset = swizzle(stsm_lane_offset + UInt32(n_offset))

        # Build a small hardware-sized SIMD from Array elements
        comptime cast_width = 4 // size_of[Scalar[c_type]]()
        var v = SIMD[c_type, stmtx_simd_width * cast_width]()
        comptime for k in range(stmtx_simd_width):
            var src = SIMD[vec_dtype, cast_width]()
            comptime for _j in range(cast_width):
                src[_j] = vec[i * stsmx_lane_size + k * cast_width + _j]
            var casted = src.cast[c_type]()
            comptime for _j in range(cast_width):
                v[k * cast_width + _j] = casted[_j]

        st_matrix[simd_width=stmtx_simd_width, transpose=transpose_c](
            dst._storage.unsafe_mut_cast[True]() + offset,
            bitcast[.float32, stmtx_simd_width](v),
        )


# =============================================================================
# EpilogueConfig - Configuration for epilogue operations
# =============================================================================


@fieldwise_init
struct EpilogueConfig(Copyable, Equatable, TrivialRegisterPassable):
    """Computed epilogue parameters based on MMA and CTA configuration.

    Bundles the 7 input parameters shared by all epilogue component structs
    (TMAStoreCoords, TMAStoreExecutor, TMEMToSMemWriter, SMemEpilogueWriter)
    plus 3 derived fields (is_lower_frag_required, num_stages, fragment_size).

    Constructed once per TileWriter/BlockwiseFP8TileWriter and propagated to
    all epilogue component types.
    """

    # Input parameters
    var MMA_M: Int
    var MMA_N: Int
    var stageN: Int
    var cta_group: Int
    var transpose_c: Bool
    var BM: Int
    var BN: Int

    # Computed fields (set by caller via create() factory)
    var is_lower_frag_required: Bool
    var num_stages: Int
    var fragment_size: Int

    @staticmethod
    def create(
        *,
        MMA_M: Int,
        MMA_N: Int,
        stageN: Int,
        cta_group: Int,
        transpose_c: Bool,
        BM: Int,
        BN: Int,
    ) -> EpilogueConfig:
        """Construct EpilogueConfig with derived fields computed automatically.
        """
        var is_lower = not (cta_group == 1 and BM == 64)
        var cg2_ns = MMA_N // stageN if MMA_M == 256 else MMA_N // stageN // 2
        var cg1_ns = MMA_N // stageN
        var ns = cg2_ns if cta_group == 2 else cg1_ns
        # fragment_size = (data_paths * (bits // 32)) // WARP_SIZE
        # = (16 * 8) // 32 = 4
        var frag_size = (16 * (256 // 32)) // 32
        return EpilogueConfig(
            MMA_M,
            MMA_N,
            stageN,
            cta_group,
            transpose_c,
            BM,
            BN,
            is_lower,
            ns,
            frag_size,
        )


# =============================================================================
# TMAStoreCoords - Compute coordinates for TMA store operations
# =============================================================================


struct TMAStoreCoords[
    epc: EpilogueConfig,
    c_smem_shape0: Int,
    stage: Int,
    batched: Bool = False,
](TrivialRegisterPassable):
    """TMA store coordinates and warp election for SM100 epilogue.

    When batched=True, includes a batch coordinate for 3D TMA stores.
    """

    # Local aliases from EpilogueConfig
    comptime BM = Self.epc.BM
    comptime BN = Self.epc.BN
    comptime MMA_M = Self.epc.MMA_M
    comptime MMA_N = Self.epc.MMA_N
    comptime stageN = Self.epc.stageN
    comptime cta_group = Self.epc.cta_group

    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM
    comptime stage_n_offset = Self.stage * Self.stageN

    var coord_m: Int
    var coord_n: Int
    var coord_b: Int  # Batch coordinate (only used when batched=True)
    var elect_one_warp: Bool
    var c_smem_coord_m: Int

    @always_inline
    def __init__(out self, c_coord: Tuple[UInt32, UInt32], warp_id: UInt32):
        """Compute TMA store coordinates from 2D tile coords and warp ID."""
        self = Self((c_coord[0], c_coord[1], UInt32(0)), warp_id)

    @always_inline
    def __init__(
        out self, c_coord: Tuple[UInt32, UInt32, UInt32], warp_id: UInt32
    ):
        """Compute TMA store coordinates from 3D tile coords and warp ID."""
        # Warp election
        var cg2_elect = warp_id == 0 if Self.MMA_M == 256 else warp_id % 2 == 0
        var cg1_elect = warp_id == 0
        self.elect_one_warp = cg2_elect if Self.cta_group == 2 else cg1_elect

        # N coordinate
        var n_base = c_coord[1] * UInt32(Self.MMA_N) + UInt32(
            Self.stage_n_offset
        )
        var n_mma128 = n_base + UInt32(Self.BN * Int(warp_id // 2))
        var cg2_n = n_base if Self.MMA_M == 256 else n_mma128
        self.coord_n = Int(cg2_n if Self.cta_group == 2 else n_base)

        # M coordinate
        self.coord_m = Int(c_coord[0]) * Self.BM

        # Batch coordinate
        self.coord_b = Int(c_coord[2])

        # SMEM tile offset
        var cg2_smem_m: Int

        comptime if Self.MMA_M == 256:
            cg2_smem_m = 0
        else:
            cg2_smem_m = Int(warp_id // 2)
        self.c_smem_coord_m = cg2_smem_m if Self.cta_group == 2 else 0


# =============================================================================
# TMAStoreExecutor - Execute TMA stores with proper SMEM tiling
# =============================================================================


struct TMAStoreExecutor[
    c_type: DType,
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    epc: EpilogueConfig,
    stage_contiguous_size: Int,
    c_swizzle: TensorMapSwizzle,
    batched: Bool = False,
](TrivialRegisterPassable):
    """Execute TMA store from SMEM to GMEM with proper tiling.

    Handles 3 paths: transpose+cta_group2+MMA128, transpose+other, non-transpose.
    When batched=True, uses 3D coordinates (M, N, Batch) for TMA stores.
    """

    # Local aliases from EpilogueConfig
    comptime BM = Self.epc.BM
    comptime BN = Self.epc.BN
    comptime MMA_M = Self.epc.MMA_M
    comptime MMA_N = Self.epc.MMA_N
    comptime stageN = Self.epc.stageN
    comptime cta_group = Self.epc.cta_group
    comptime transpose_c = Self.epc.transpose_c
    comptime is_lower_frag_required = Self.epc.is_lower_frag_required

    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime num_c_smem_tiles = Self.BM // Self.swizzle_width
    comptime c_smem_shape0 = Self.c_smem_dim0
    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.MMA_M == 256 else Self.BM
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM

    @staticmethod
    @always_inline
    def _store_non_transpose[
        tma_rank: Int,
        tile_shape: IndexList[tma_rank],
        desc_shape: IndexList[tma_rank],
        //,
    ](
        c_smem_tile: TileTensor[Self.c_type, address_space=.SHARED, ...],
        store_coords: TMAStoreCoords[
            Self.epc,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, tma_rank, tile_shape, desc_shape],
    ):
        """Handle non-transpose TMA store path."""
        # Path C: Simple tile selection by TMA_BM
        # Note: coords are (coord_n, coord_m) - swapped for non-transpose!
        var c_smem_split = c_smem_tile.tile[Self.TMA_BM, Self.stageN](
            Coord(store_coords.c_smem_coord_m, Idx[0])
        )

        comptime if Self.batched:
            c_tma_op.async_store(
                c_smem_split,
                StaticTuple[UInt32, 3](
                    UInt32(store_coords.coord_n),
                    UInt32(store_coords.coord_m),
                    UInt32(store_coords.coord_b),
                ),
            )
        else:
            c_tma_op.async_store(
                c_smem_split,
                (store_coords.coord_n, store_coords.coord_m),
            )

    @staticmethod
    @always_inline
    def execute[
        tma_rank: Int,
        tile_shape: IndexList[tma_rank],
        desc_shape: IndexList[tma_rank],
    ](
        c_smem_tile: TileTensor[Self.c_type, address_space=.SHARED, ...],
        store_coords: TMAStoreCoords[
            Self.epc,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, tma_rank, tile_shape, desc_shape],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Execute TMA store."""
        if store_coords.elect_one_warp and elect_one_sync():
            fence_async_view_proxy()

            comptime if Self.transpose_c:
                Self._store_transpose[tma_rank, tile_shape, desc_shape](
                    c_smem_tile, store_coords, c_tma_op, warp_id
                )
            else:
                Self._store_non_transpose(c_smem_tile, store_coords, c_tma_op)

            c_tma_op.commit_group()

    @staticmethod
    @always_inline
    def _store_transpose[
        tma_rank: Int,
        tile_shape: IndexList[tma_rank],
        desc_shape: IndexList[tma_rank],
    ](
        c_smem_tile: TileTensor[Self.c_type, address_space=.SHARED, ...],
        store_coords: TMAStoreCoords[
            Self.epc,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, tma_rank, tile_shape, desc_shape],
        warp_id: UInt32,
    ):
        """Transpose TMA store using reshape."""

        comptime if Self.cta_group == 2 and Self.MMA_M == 128:
            # Path A: reshape to (2*stageN, sc_size//2), tile by warp
            comptime reshaped = row_major[
                2 * Self.stageN, Self.stage_contiguous_size // 2
            ]()
            var c_reshaped = c_smem_tile.reshape(reshaped)
            var c_split = c_reshaped.tile[
                Self.stageN, Self.stage_contiguous_size // 2
            ](Coord(Int(warp_id // 2), Idx[0]))

            comptime for i in range(Self.num_c_smem_tiles):
                var c_split_tile = c_split.tile[
                    Self.stageN // Self.num_c_smem_tiles,
                    Self.stage_contiguous_size // 2,
                ](Coord(i, Idx[0]))

                comptime if Self.batched:
                    c_tma_op.async_store(
                        c_split_tile,
                        StaticTuple[UInt32, 3](
                            UInt32(
                                store_coords.coord_m + i * Self.swizzle_width
                            ),
                            UInt32(store_coords.coord_n),
                            UInt32(store_coords.coord_b),
                        ),
                    )
                else:
                    c_tma_op.async_store(
                        c_split_tile,
                        (
                            store_coords.coord_m + i * Self.swizzle_width,
                            store_coords.coord_n,
                        ),
                    )
        else:
            # Path B: loop over swizzle tiles
            comptime tile_dim0 = (
                Self.stageN * Self.swizzle_width // Self.stage_contiguous_size
            )

            comptime for i in range(Self.num_c_smem_tiles):
                var tiled = c_smem_tile.tile[
                    tile_dim0, Self.stage_contiguous_size
                ](Coord(i, Idx[0]))

                comptime reshaped = row_major[Self.stageN, Self.swizzle_width]()
                var c_warp_tile = tiled.reshape(reshaped)

                comptime if Self.batched:
                    c_tma_op.async_store(
                        c_warp_tile,
                        StaticTuple[UInt32, 3](
                            UInt32(
                                store_coords.coord_m + i * Self.swizzle_width
                            ),
                            UInt32(store_coords.coord_n),
                            UInt32(store_coords.coord_b),
                        ),
                    )
                else:
                    c_tma_op.async_store(
                        c_warp_tile,
                        (
                            store_coords.coord_m + i * Self.swizzle_width,
                            store_coords.coord_n,
                        ),
                    )


# =============================================================================
# TMAReduceExecutor - Execute TMA reduce-add stores with proper SMEM tiling
# =============================================================================


struct TMAReduceExecutor[
    c_type: DType,
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    epc: EpilogueConfig,
    stage_contiguous_size: Int,
    c_swizzle: TensorMapSwizzle,
    batched: Bool = False,
](TrivialRegisterPassable):
    """Execute TMA reduce-add from SMEM to GMEM.

    Mirrors TMAStoreExecutor but uses cp_async_bulk_tensor_reduce_global_shared_cta (add)
    instead of cp_async_bulk_tensor_global_shared_cta (store).
    Takes a typed TMATensorTile value (not a raw pointer) so the
    descriptor keeps its grid_constant provenance end-to-end -- otherwise
    the compiler drops the constant-memory optimization and each TMA
    issue refetches the descriptor.
    Only supports non-transpose path.
    """

    comptime stageN = Self.epc.stageN
    comptime cta_group = Self.epc.cta_group
    comptime c_smem_shape0 = Self.c_smem_dim0
    comptime CG1_TMA_BM = Self.c_smem_shape0
    comptime CG2_TMA_BM = Self.c_smem_shape0 if Self.epc.MMA_M == 256 else Self.epc.BM
    comptime TMA_BM = Self.CG2_TMA_BM if Self.cta_group == 2 else Self.CG1_TMA_BM

    @staticmethod
    @always_inline
    def execute[
        tma_rank: Int,
        tile_shape: IndexList[tma_rank],
        desc_shape: IndexList[tma_rank],
    ](
        c_smem_tile: TileTensor[
            Self.c_type,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
        store_coords: TMAStoreCoords[
            Self.epc,
            Self.c_smem_shape0,
            _,
            Self.batched,
        ],
        c_tma_op: TMATensorTile[Self.c_type, tma_rank, tile_shape, desc_shape],
        warp_id: UInt32,
        lane: UInt32,
    ):
        """Execute TMA reduce-add from SMEM to GMEM via typed descriptor."""
        comptime assert (
            not Self.epc.transpose_c
        ), "TMAReduceExecutor only supports non-transpose path"
        if store_coords.elect_one_warp and elect_one_sync():
            fence_async_view_proxy()

            var c_smem_split = c_smem_tile.tile[Self.TMA_BM, Self.stageN](
                Coord(Int(store_coords.c_smem_coord_m), Idx[0])
            )

            comptime if Self.batched:
                cp_async_bulk_tensor_reduce_global_shared_cta[
                    reduction_kind=ReduceOp.ADD
                ](
                    c_smem_split._storage,
                    UnsafePointer(to=c_tma_op.descriptor).bitcast[NoneType](),
                    Index(
                        store_coords.coord_n,
                        store_coords.coord_m,
                        store_coords.coord_b,
                    ),
                )
            else:
                cp_async_bulk_tensor_reduce_global_shared_cta[
                    reduction_kind=ReduceOp.ADD
                ](
                    c_smem_split._storage,
                    UnsafePointer(to=c_tma_op.descriptor).bitcast[NoneType](),
                    Index(store_coords.coord_n, store_coords.coord_m),
                )

            cp_async_bulk_commit_group()


# =============================================================================
# FragmentCoords - Coordinate tracking for fragment elements
# =============================================================================


struct FragmentCoords[stageN: Int, repeats: Int](TrivialRegisterPassable):
    """Fragment element coordinates for tcgen05 16x256b matrix layout."""

    comptime load_width = 2
    comptime threads_per_row = Self.stageN // Self.repeats // Self.load_width

    var top_upper: StaticTuple[UInt32, 2]
    var bottom_upper: StaticTuple[UInt32, 2]
    var top_lower: StaticTuple[UInt32, 2]
    var bottom_lower: StaticTuple[UInt32, 2]

    @always_inline
    def __init__(out self, lane_id: UInt32):
        """Compute (row, col) for each fragment position from lane ID."""
        var row = lane_id // UInt32(Self.threads_per_row)
        var col = (lane_id % UInt32(Self.threads_per_row)) * UInt32(
            Self.load_width
        )

        self.top_upper = StaticTuple[UInt32, 2](row, col)
        self.bottom_upper = StaticTuple[UInt32, 2](row + 8, col)
        self.top_lower = StaticTuple[UInt32, 2](row + 16, col)
        self.bottom_lower = StaticTuple[UInt32, 2](row + 24, col)


# =============================================================================
# EpilogueApplier - Apply element-wise operations on fragments
# =============================================================================


struct EpilogueApplier[
    MMA_M: Int,
    stageN: Int,
    num_stages: Int,
    repeats: Int,
    cta_group: Int,
    transpose_c: Bool,
](TrivialRegisterPassable):
    """Apply element-wise epilogue lambda to register fragments."""

    comptime Coords = FragmentCoords[Self.stageN, Self.repeats]

    var coords: Self.Coords
    var warp_id: UInt32
    var lane_id: UInt32
    var M: UInt32
    var N: UInt32

    @always_inline
    def __init__(
        out self,
        warp_id: UInt32,
        lane_id: UInt32,
        c_shape: Tuple[UInt32, UInt32],
    ):
        self.coords = Self.Coords(lane_id)
        self.warp_id = warp_id
        self.lane_id = lane_id
        self.M = c_shape[0]
        self.N = c_shape[1]

    @always_inline
    def compute_staged_coords(
        self, stage: UInt32, c_row: UInt32, c_col: UInt32
    ) -> Tuple[UInt32, UInt32]:
        """Compute global coords with warp and stage offsets (layout-dependent).
        """
        var staged_col = c_col + stage * UInt32(Self.stageN)
        var staged_row = c_row

        comptime if Self.MMA_M == 256 or (
            Self.MMA_M == 128 and Self.cta_group == 1
        ):
            staged_row += self.warp_id * 32  # Layout A/D
        elif Self.MMA_M == 64 and Self.cta_group == 1:
            staged_row += self.warp_id * 16  # Layout F
        else:
            staged_row += (self.warp_id % 2) * 32  # Layout B
            staged_col += (self.warp_id // 2) * UInt32(
                Self.num_stages * Self.stageN
            )

        return (staged_row, staged_col)

    @always_inline
    def apply_to_fragment[
        epilogue_dtype: DType,
        frag_size: Int,
        compute_lambda_fn: elementwise_compute_lambda_type,
        is_in_bounds: Bool = False,
    ](
        self,
        mut frag: Array[Scalar[epilogue_dtype], frag_size],
        staged_row: UInt32,
        staged_col: UInt32,
        is_upper: Bool,
    ):
        """Apply epilogue lambda to fragment elements with global coords.

        ``is_in_bounds=True``: caller asserts the whole tile fits in
        ``(self.M, self.N)``; per-position checks are elided. Default
        ``False`` keeps them — TMA masks the OOB write but the lambda
        may dereference operands at the supplied (row, col).

        transpose_c=True: per-position — top.row/bot.row in a (top, bot)
        pair are 8 apart (upper +0/+8 or lower +16/+24) and can straddle
        ``self.N``. transpose_c=False: early return on top_col is safe
        (``top.col == bot.col`` and ``self.N`` is alignment-bound).
        """
        var top = self.coords.top_upper if is_upper else self.coords.top_lower
        var bot = (
            self.coords.bottom_upper if is_upper else self.coords.bottom_lower
        )

        comptime for rep in range(Self.repeats):
            comptime inc = rep * 8
            comptime offset = rep * 4

            var top_row = staged_row + top[0]
            var top_col = staged_col + top[1] + UInt32(inc)
            var bot_row = staged_row + bot[0]
            var bot_col = staged_col + bot[1] + UInt32(inc)

            var elem0 = frag[offset]
            var elem1 = frag[offset + 1]
            var elem2 = frag[offset + 2]
            var elem3 = frag[offset + 3]

            comptime if Self.transpose_c:
                comptime if is_in_bounds:
                    frag[offset] = compute_lambda_fn[epilogue_dtype, 1](
                        IndexList[2](Int(top_col), Int(top_row)), elem0
                    )
                    frag[offset + 1] = compute_lambda_fn[epilogue_dtype, 1](
                        IndexList[2](Int(top_col + 1), Int(top_row)), elem1
                    )
                    frag[offset + 2] = compute_lambda_fn[epilogue_dtype, 1](
                        IndexList[2](Int(bot_col), Int(bot_row)), elem2
                    )
                    frag[offset + 3] = compute_lambda_fn[epilogue_dtype, 1](
                        IndexList[2](Int(bot_col + 1), Int(bot_row)), elem3
                    )
                else:
                    var valid_top_row = top_row < self.N
                    var valid_bot_row = bot_row < self.N

                    if valid_top_row and top_col < self.M:
                        frag[offset] = compute_lambda_fn[epilogue_dtype, 1](
                            IndexList[2](Int(top_col), Int(top_row)), elem0
                        )
                    if valid_bot_row and top_col < self.M:
                        frag[offset + 2] = compute_lambda_fn[epilogue_dtype, 1](
                            IndexList[2](Int(bot_col), Int(bot_row)), elem2
                        )

                    if valid_top_row and (top_col + 1) < self.M:
                        frag[offset + 1] = compute_lambda_fn[epilogue_dtype, 1](
                            IndexList[2](Int(top_col + 1), Int(top_row)), elem1
                        )
                    if valid_bot_row and (top_col + 1) < self.M:
                        frag[offset + 3] = compute_lambda_fn[epilogue_dtype, 1](
                            IndexList[2](Int(bot_col + 1), Int(bot_row)), elem3
                        )
            else:
                comptime if is_in_bounds:
                    var elem01 = compute_lambda_fn[
                        epilogue_dtype, 2, alignment=2
                    ](
                        IndexList[2](Int(top_row), Int(top_col)),
                        SIMD[epilogue_dtype, 2](
                            elem0,
                            elem1,
                        ),
                    )
                    var elem23 = compute_lambda_fn[
                        epilogue_dtype, 2, alignment=2
                    ](
                        IndexList[2](Int(bot_row), Int(bot_col)),
                        SIMD[epilogue_dtype, 2](
                            elem2,
                            elem3,
                        ),
                    )
                    frag[offset] = elem01[0]
                    frag[offset + 1] = elem01[1]
                    frag[offset + 2] = elem23[0]
                    frag[offset + 3] = elem23[1]
                else:
                    if top_col >= self.N:
                        return

                    var valid_top_row = top_row < self.M
                    var valid_bot_row = bot_row < self.M

                    if valid_top_row:
                        var elem01 = compute_lambda_fn[
                            epilogue_dtype, 2, alignment=2
                        ](
                            IndexList[2](Int(top_row), Int(top_col)),
                            SIMD[epilogue_dtype, 2](
                                elem0,
                                elem1,
                            ),
                        )
                        frag[offset] = elem01[0]
                        frag[offset + 1] = elem01[1]

                    if valid_bot_row:
                        var elem23 = compute_lambda_fn[
                            epilogue_dtype, 2, alignment=2
                        ](
                            IndexList[2](Int(bot_row), Int(bot_col)),
                            SIMD[epilogue_dtype, 2](
                                elem2,
                                elem3,
                            ),
                        )
                        frag[offset + 2] = elem23[0]
                        frag[offset + 3] = elem23[1]

    @always_inline
    def apply_to_both_fragments[
        epilogue_dtype: DType,
        frag_size: Int,
        compute_lambda_fn: elementwise_compute_lambda_type,
        is_lower_frag_required: Bool,
        is_in_bounds: Bool = False,
    ](
        self,
        mut upper_frag: Array[Scalar[epilogue_dtype], frag_size],
        mut lower_frag: Array[Scalar[epilogue_dtype], frag_size],
        stage: UInt32,
        c_row: UInt32,
        c_col: UInt32,
    ) -> Tuple[
        Array[Scalar[epilogue_dtype], frag_size],
        Array[Scalar[epilogue_dtype], frag_size],
    ]:
        """Apply epilogue to both fragments (main entry point)."""
        var staged_row, staged_col = self.compute_staged_coords(
            stage, c_row, c_col
        )

        self.apply_to_fragment[
            epilogue_dtype,
            frag_size,
            compute_lambda_fn,
            is_in_bounds=is_in_bounds,
        ](upper_frag, staged_row, staged_col, is_upper=True)

        comptime if is_lower_frag_required:
            self.apply_to_fragment[
                epilogue_dtype,
                frag_size,
                compute_lambda_fn,
                is_in_bounds=is_in_bounds,
            ](lower_frag, staged_row, staged_col, is_upper=False)

        return (upper_frag.copy(), lower_frag.copy())

    @always_inline
    def apply_elementwise_epilogue_to_fragment[
        epilogue_dtype: DType,
        frag_size: Int,
        elementwise_lambda_fn: elementwise_epilogue_type,
        is_in_bounds: Bool = False,
    ](
        self,
        frag: SIMD[epilogue_dtype, frag_size],
        staged_row: UInt32,
        staged_col: UInt32,
        is_upper: Bool,
    ):
        """Apply elementwise epilogue lambda to fragment elements with global coords.

        Unlike apply_to_fragment which uses a compute lambda that returns modified
        values, this calls an elementwise epilogue (returns None) that stores
        directly to global memory.

        ``is_in_bounds=True``: caller asserts the whole tile fits in
        ``(self.M, self.N)``; the per-position row/column checks are elided and
        the lambda is called for every element. Default ``False`` keeps the
        checks (needed for border tiles, where an unchecked direct store would
        write out of bounds). Mirrors ``apply_to_fragment``'s ``is_in_bounds``.
        """
        var top = self.coords.top_upper if is_upper else self.coords.top_lower
        var bot = (
            self.coords.bottom_upper if is_upper else self.coords.bottom_lower
        )

        comptime for rep in range(Self.repeats):
            comptime inc = rep * 8
            comptime offset = rep * 4

            var top_col = staged_col + top[1] + UInt32(inc)
            var bot_col = staged_col + bot[1] + UInt32(inc)
            var top_row = staged_row + top[0]
            var bot_row = staged_row + bot[0]

            var elems = frag.slice[4, offset=offset]()

            comptime if Self.transpose_c:
                comptime if is_in_bounds:
                    # Whole tile in bounds: store all elements, no checks.
                    elementwise_lambda_fn[epilogue_dtype](
                        IndexList[2](Int(top_col), Int(top_row)), elems[0]
                    )
                    elementwise_lambda_fn[epilogue_dtype](
                        IndexList[2](Int(top_col + 1), Int(top_row)), elems[1]
                    )
                    elementwise_lambda_fn[epilogue_dtype](
                        IndexList[2](Int(bot_col), Int(bot_row)), elems[2]
                    )
                    elementwise_lambda_fn[epilogue_dtype](
                        IndexList[2](Int(bot_col + 1), Int(bot_row)), elems[3]
                    )
                else:
                    # The pair's rows sit 8 apart along N, so each needs
                    # its own bound check.
                    var valid_top_row = top_row < self.N
                    var valid_bot_row = bot_row < self.N

                    if valid_top_row and top_col < self.M:
                        elementwise_lambda_fn[epilogue_dtype](
                            IndexList[2](Int(top_col), Int(top_row)), elems[0]
                        )
                    if valid_bot_row and top_col < self.M:
                        elementwise_lambda_fn[epilogue_dtype](
                            IndexList[2](Int(bot_col), Int(bot_row)), elems[2]
                        )

                    if valid_top_row and (top_col + 1) < self.M:
                        elementwise_lambda_fn[epilogue_dtype](
                            IndexList[2](Int(top_col + 1), Int(top_row)),
                            elems[1],
                        )
                    if valid_bot_row and (top_col + 1) < self.M:
                        elementwise_lambda_fn[epilogue_dtype](
                            IndexList[2](Int(bot_col + 1), Int(bot_row)),
                            elems[3],
                        )
            else:
                comptime if is_in_bounds:
                    # Whole tile in bounds: store all elements, no checks.
                    elementwise_lambda_fn[epilogue_dtype, 2](
                        IndexList[2](Int(top_row), Int(top_col)),
                        SIMD[epilogue_dtype, 2](
                            elems[0],
                            elems[1],
                        ),
                    )
                    elementwise_lambda_fn[epilogue_dtype, 2](
                        IndexList[2](Int(bot_row), Int(bot_col)),
                        SIMD[epilogue_dtype, 2](
                            elems[2],
                            elems[3],
                        ),
                    )
                else:
                    # self.N is alignment-bound: static_N * size_of[c_type]()
                    # % 16 == 0 makes it even, and column pairs start on an
                    # even column, so top_col and top_col + 1 cross it
                    # together.
                    if top_col >= self.N:
                        return

                    var valid_top_row = top_row < self.M
                    var valid_bot_row = bot_row < self.M

                    if valid_top_row:
                        elementwise_lambda_fn[epilogue_dtype, 2](
                            IndexList[2](Int(top_row), Int(top_col)),
                            SIMD[epilogue_dtype, 2](
                                elems[0],
                                elems[1],
                            ),
                        )

                    if valid_bot_row:
                        elementwise_lambda_fn[epilogue_dtype, 2](
                            IndexList[2](Int(bot_row), Int(bot_col)),
                            SIMD[epilogue_dtype, 2](
                                elems[2],
                                elems[3],
                            ),
                        )

    @always_inline
    def apply_elementwise_epilogue_to_both_fragments[
        epilogue_dtype: DType,
        frag_size: Int,
        elementwise_lambda_fn: elementwise_epilogue_type,
        is_lower_frag_required: Bool,
        is_in_bounds: Bool = False,
    ](
        self,
        upper_frag: SIMD[epilogue_dtype, frag_size],
        lower_frag: SIMD[epilogue_dtype, frag_size],
        stage: UInt32,
        c_row: UInt32,
        c_col: UInt32,
    ):
        """Apply elementwise epilogue to both fragments.

        Similar to apply_to_both_fragments but uses elementwise_epilogue_type
        which writes directly to global memory and returns None.

        ``is_in_bounds`` is threaded to the per-fragment path: when ``True`` the
        caller has asserted the whole tile fits in ``(M, N)`` and the
        per-position bounds checks are elided.
        """
        var staged_row, staged_col = self.compute_staged_coords(
            stage, c_row, c_col
        )

        self.apply_elementwise_epilogue_to_fragment[
            epilogue_dtype,
            frag_size,
            elementwise_lambda_fn,
            is_in_bounds=is_in_bounds,
        ](upper_frag, staged_row, staged_col, is_upper=True)

        comptime if is_lower_frag_required:
            self.apply_elementwise_epilogue_to_fragment[
                epilogue_dtype,
                frag_size,
                elementwise_lambda_fn,
                is_in_bounds=is_in_bounds,
            ](
                lower_frag,
                staged_row,
                staged_col,
                is_upper=False,
            )

    # =========================================================================
    # Residual Add - Load C from SMEM and add beta*C to fragment registers
    # =========================================================================

    @always_inline
    def add_residual_to_fragment[
        epilogue_dtype: DType,
        frag_size: Int,
        c_type: DType,
        c_smem_stride: Int,
        swizzle: Swizzle,
    ](
        self,
        mut frag: Array[Scalar[epilogue_dtype], frag_size],
        local_row: UInt32,
        local_col: UInt32,
        is_upper: Bool,
        src_ptr: UnsafePointer[Scalar[c_type], _, address_space=.SHARED],
        beta: Scalar[epilogue_dtype],
    ):
        """Add beta * C to fragment elements by loading C from swizzled SMEM.

        Uses the same per-lane coordinate mapping as apply_to_fragment, but
        instead of applying a lambda, loads source C values from SMEM at the
        matching swizzled addresses and adds beta * C to each element.

        Args:
            frag: Fragment register values to modify in-place.
            local_row: Tile-local row offset (warp offset within tile).
            local_col: Tile-local column offset (stage offset within tile).
            is_upper: Whether this is the upper (rows 0-15) or lower (16-31)
                fragment half.
            src_ptr: Pointer to source C SMEM tile (same TMA swizzle as output).
            beta: Residual scale factor.
        """
        var top = self.coords.top_upper if is_upper else self.coords.top_lower
        var bot = (
            self.coords.bottom_upper if is_upper else self.coords.bottom_lower
        )

        comptime for rep in range(Self.repeats):
            comptime inc = rep * 8
            comptime offset = rep * 4

            var top_row = local_row + top[0]
            var top_col = local_col + top[1] + UInt32(inc)
            var bot_row = local_row + bot[0]
            var bot_col = local_col + bot[1] + UInt32(inc)

            # Load C from swizzled SMEM at matching fragment coordinates
            var c0 = src_ptr.load(
                Int(swizzle(top_row * UInt32(c_smem_stride) + top_col))
            ).cast[epilogue_dtype]()
            var c1 = src_ptr.load(
                Int(swizzle(top_row * UInt32(c_smem_stride) + top_col + 1))
            ).cast[epilogue_dtype]()
            var c2 = src_ptr.load(
                Int(swizzle(bot_row * UInt32(c_smem_stride) + bot_col))
            ).cast[epilogue_dtype]()
            var c3 = src_ptr.load(
                Int(swizzle(bot_row * UInt32(c_smem_stride) + bot_col + 1))
            ).cast[epilogue_dtype]()

            # FMA: frag += beta * C
            frag[offset] += beta * c0
            frag[offset + 1] += beta * c1
            frag[offset + 2] += beta * c2
            frag[offset + 3] += beta * c3

    @always_inline
    def add_residual_to_both_fragments[
        epilogue_dtype: DType,
        frag_size: Int,
        is_lower_frag_required: Bool,
        c_type: DType,
        c_smem_stride: Int,
        swizzle: Swizzle,
    ](
        self,
        mut upper_frag: Array[Scalar[epilogue_dtype], frag_size],
        mut lower_frag: Array[Scalar[epilogue_dtype], frag_size],
        stage: UInt32,
        src_ptr: UnsafePointer[Scalar[c_type], _, address_space=.SHARED],
        beta: Scalar[epilogue_dtype],
    ) -> Tuple[
        Array[Scalar[epilogue_dtype], frag_size],
        Array[Scalar[epilogue_dtype], frag_size],
    ]:
        """Add beta * C to both fragment halves from swizzled SMEM.

        Computes tile-local coordinates from stage and warp ID, then loads
        source C from SMEM and adds beta * C to each fragment element.

        Args:
            upper_frag: Upper fragment (rows 0-15 within warp tile).
            lower_frag: Lower fragment (rows 16-31 within warp tile).
            stage: Output stage index (for column offset computation).
            src_ptr: Pointer to source C SMEM tile.
            beta: Residual scale factor.

        Returns:
            Updated (upper_frag, lower_frag) tuple.
        """
        # Tile-local coords: pass (0, 0) as origin since we're indexing
        # within the SMEM tile, not in global coordinates.
        var local_row, local_col = self.compute_staged_coords(stage, 0, 0)

        self.add_residual_to_fragment[
            epilogue_dtype, frag_size, c_type, c_smem_stride, swizzle
        ](upper_frag, local_row, local_col, True, src_ptr, beta)

        comptime if is_lower_frag_required:
            self.add_residual_to_fragment[
                epilogue_dtype, frag_size, c_type, c_smem_stride, swizzle
            ](lower_frag, local_row, local_col, False, src_ptr, beta)

        return (upper_frag.copy(), lower_frag.copy())


# =============================================================================
# TMEMToSMemWriter - Write TMEM accumulators to shared memory (SM100-specific)
# =============================================================================


struct TMEMToSMemWriter[
    c_type: DType,
    accum_type: DType,
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    epc: EpilogueConfig,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](TrivialRegisterPassable):
    """Write TMEM accumulators to SMEM via st.matrix (SM100-specific)."""

    # Local aliases from EpilogueConfig
    comptime BM = Self.epc.BM
    comptime stageN = Self.epc.stageN
    comptime cta_group = Self.epc.cta_group
    comptime transpose_c = Self.epc.transpose_c

    # Create internal layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    # Alias for fragment_size access (used by callers via SMEMWriter.Config)
    comptime Config = Self.epc

    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime stage_contiguous_size = Self.c_smem_dim1
    comptime data_paths = 16
    comptime swizzle = _make_swizzle[Self.c_type, Self.c_swizzle]()

    var warp_id: UInt32
    var lane_id: UInt32

    @always_inline
    def __init__(out self, warp_id: UInt32, lane_id: UInt32):
        self.warp_id = warp_id
        self.lane_id = lane_id

    @always_inline
    def write_fragments[
        repeat: Int
    ](
        self,
        upper_frag: Array[
            Scalar[Self.c_type], Self.Config.fragment_size * repeat
        ],
        lower_frag: Array[
            Scalar[Self.c_type], Self.Config.fragment_size * repeat
        ],
        c_smem_tile: TileTensor[
            address_space=.SHARED, Engine=DefaultEngine[element_width=1], ...
        ],
    ):
        """Write pre-loaded fragments to SMEM."""
        comptime is_lower_required = Self.Config.is_lower_frag_required

        comptime if Self.transpose_c:
            self._write_transpose[repeat, is_lower_required](
                upper_frag, lower_frag, c_smem_tile
            )
        else:
            self._write_non_transpose[repeat, is_lower_required](
                upper_frag, lower_frag, c_smem_tile
            )

    @always_inline
    def _write_transpose[
        repeat: Int, is_lower_required: Bool
    ](
        self,
        upper_casted: Array[
            Scalar[Self.c_type], Self.Config.fragment_size * repeat
        ],
        lower_casted: Array[
            Scalar[Self.c_type], Self.Config.fragment_size * repeat
        ],
        c_smem_tile: TileTensor[
            address_space=.SHARED, Engine=DefaultEngine[element_width=1], ...
        ],
    ):
        """Transposed output using reshape.

        Two paths based on swizzle width:
        - Large swizzle (SWIZZLE_128B): multiple warps share a swizzle block,
          requiring warp_offset for correct swizzle/deswizzle addressing.
        - Small swizzle (SWIZZLE_32B): each warp fragment fits within its own
          swizzle block, using simple tiles_per_frag tiling.
        """

        # SWIZZLE_128B: swizzle_width=64 > data_paths=16 → True
        # SWIZZLE_32B:  swizzle_width=16 == data_paths=16 → False
        comptime _large_swizzle = Self.swizzle_width > Self.data_paths

        comptime if _large_swizzle:
            comptime if is_lower_required:
                comptime tile_width = 32
                comptime num_swblocks = Self.stage_contiguous_size // Self.swizzle_width
                comptime warps_per_swblock = Self.swizzle_width // tile_width

                comptime logical_layout = InternalLayout(
                    Coord(
                        Idx[num_swblocks],
                        Idx[Self.stageN],
                        Idx[warps_per_swblock],
                        Idx[tile_width],
                    ),
                    Coord(
                        Idx[Self.stageN * Self.swizzle_width],
                        Idx[Self.swizzle_width],
                        Idx[tile_width],
                        Idx[1],
                    ),
                )
                var new_smem = c_smem_tile.reshape(logical_layout)

                var warp_j, warp_i = divmod(
                    Int(self.warp_id), warps_per_swblock
                )
                var tiled = new_smem.tile[1, Self.stageN, 1, tile_width](
                    Coord(warp_j, Idx[0], warp_i, Idx[0])
                )

                # Coalesce: (1, stageN, 1, 32) -> (stageN, 32)
                comptime coalesced = InternalLayout(
                    Coord(Idx[Self.stageN], Idx[tile_width]),
                    Coord(Idx[Self.swizzle_width], Idx[1]),
                )
                var c_smem_warp_tile = tiled.reshape(coalesced)

                var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                    Self.stageN, Self.data_paths
                ](Coord(Idx[0], Idx[0]))
                var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                    Self.stageN, Self.data_paths
                ](Coord(Idx[0], Idx[1]))

                var warp_offset = warp_i * tile_width
                store_fragment_to_smem[
                    Self.swizzle,
                    Self.stageN,
                    transpose_c=Self.transpose_c,
                    c_swizzle=Self.c_swizzle,
                ](upper_casted, c_smem_warp_tile_upper, UInt32(warp_offset))

                warp_offset += tile_width // 2
                store_fragment_to_smem[
                    Self.swizzle,
                    Self.stageN,
                    transpose_c=Self.transpose_c,
                    c_swizzle=Self.c_swizzle,
                ](lower_casted, c_smem_warp_tile_lower, UInt32(warp_offset))
            else:
                comptime tile_width = 16
                comptime num_swblocks = Self.stage_contiguous_size // Self.swizzle_width
                comptime warps_per_swblock = Self.swizzle_width // tile_width

                comptime logical_layout = InternalLayout(
                    Coord(
                        Idx[num_swblocks],
                        Idx[Self.stageN],
                        Idx[warps_per_swblock],
                        Idx[tile_width],
                    ),
                    Coord(
                        Idx[Self.stageN * Self.swizzle_width],
                        Idx[Self.swizzle_width],
                        Idx[tile_width],
                        Idx[1],
                    ),
                )
                var new_smem = c_smem_tile.reshape(logical_layout)

                var warp_j, warp_i = divmod(
                    Int(self.warp_id), warps_per_swblock
                )
                var tiled = new_smem.tile[1, Self.stageN, 1, tile_width](
                    Coord(warp_j, Idx[0], warp_i, Idx[0])
                )

                # Coalesce: (1, stageN, 1, tile_width) -> (stageN, tile_width)
                # with the in-block stageN-row stride = swizzle_width.
                comptime coalesced = InternalLayout(
                    Coord(Idx[Self.stageN], Idx[tile_width]),
                    Coord(Idx[Self.swizzle_width], Idx[1]),
                )
                var c_smem_warp_tile = tiled.reshape(coalesced)

                var warp_offset = warp_i * tile_width
                store_fragment_to_smem[
                    Self.swizzle,
                    Self.stageN,
                    transpose_c=Self.transpose_c,
                    c_swizzle=Self.c_swizzle,
                ](upper_casted, c_smem_warp_tile, UInt32(warp_offset))
        else:
            comptime tiles_per_frag = (
                Self.stageN * Self.swizzle_width // Self.stage_contiguous_size
            )
            comptime reshaped = row_major[Self.stageN, Self.swizzle_width]()

            comptime if is_lower_required:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Coord(2 * Int(self.warp_id), Idx[0])).reshape(reshaped)
                var c_smem_warp_tile_lower = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Coord(2 * Int(self.warp_id) + 1, Idx[0])).reshape(reshaped)

                store_fragment_to_smem[
                    Self.swizzle,
                    Self.stageN,
                    transpose_c=Self.transpose_c,
                    c_swizzle=Self.c_swizzle,
                ](upper_casted, c_smem_warp_tile_upper)
                store_fragment_to_smem[
                    Self.swizzle,
                    Self.stageN,
                    transpose_c=Self.transpose_c,
                    c_swizzle=Self.c_swizzle,
                ](lower_casted, c_smem_warp_tile_lower)
            else:
                var c_smem_warp_tile_upper = c_smem_tile.tile[
                    tiles_per_frag, Self.stage_contiguous_size
                ](Coord(Int(self.warp_id), Idx[0])).reshape(reshaped)

                store_fragment_to_smem[
                    Self.swizzle,
                    Self.stageN,
                    transpose_c=Self.transpose_c,
                    c_swizzle=Self.c_swizzle,
                ](upper_casted, c_smem_warp_tile_upper)

    @always_inline
    def _write_non_transpose[
        repeat: Int, is_lower_required: Bool
    ](
        self,
        upper_casted: Array[
            Scalar[Self.c_type], Self.Config.fragment_size * repeat
        ],
        lower_casted: Array[
            Scalar[Self.c_type], Self.Config.fragment_size * repeat
        ],
        c_smem_tile: TileTensor[
            address_space=.SHARED, Engine=DefaultEngine[element_width=1], ...
        ],
    ):
        """Non-transposed output."""

        comptime c_smem_tile_m = 32 if Self.cta_group == 2 else Self.BM // Self.num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
            Coord(Int(self.warp_id), Idx[0])
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](Coord(Idx[0], Idx[0]))
        store_fragment_to_smem[
            Self.swizzle,
            Self.stageN,
            transpose_c=Self.transpose_c,
            c_swizzle=Self.c_swizzle,
        ](upper_casted, c_smem_warp_tile_upper)

        comptime if is_lower_required:
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.data_paths, Self.stageN
            ](Coord(Idx[1], Idx[0]))
            store_fragment_to_smem[
                Self.swizzle,
                Self.stageN,
                transpose_c=Self.transpose_c,
                c_swizzle=Self.c_swizzle,
            ](lower_casted, c_smem_warp_tile_lower)


# =============================================================================
# Imports for IndexList
# =============================================================================
from std.utils.index import IndexList
from layout.layout import flatten


# =============================================================================
# SMemEpilogueWriter - SMEM-based epilogue writer with element-wise compute
# =============================================================================


struct SMemEpilogueWriter[
    # Infer-only: deduced from c_tiles argument type
    c_type: DType,
    num_output_stages: Int,
    //,
    # Configuration parameters (must be explicit)
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    epilogue_dtype: DType,
    epc: EpilogueConfig,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle,
    simd_size: Int,
    stage: Int,
    rep_frag_size: Int,
    compute_lambda_fn: elementwise_compute_lambda_type,
](TrivialRegisterPassable):
    """SMEM-based epilogue: write accumulators and apply lambda in SMEM."""

    # Local aliases from EpilogueConfig
    comptime BM = Self.epc.BM
    comptime BN = Self.epc.BN
    comptime MMA_M = Self.epc.MMA_M
    comptime MMA_N = Self.epc.MMA_N
    comptime cta_group = Self.epc.cta_group
    comptime transpose_c = Self.epc.transpose_c
    comptime is_lower_frag_required = Self.epc.is_lower_frag_required
    comptime num_stages = Self.epc.num_stages

    # Create layout from dimensions
    comptime c_smem_layout = Layout.row_major(
        Self.c_smem_dim0, Self.c_smem_dim1
    )

    comptime stageN = Self.c_smem_dim0 if Self.transpose_c else Self.c_smem_dim1
    comptime stage_contiguous_size = Self.c_smem_dim1

    comptime swizzle = _make_swizzle[Self.c_type, Self.c_swizzle]()
    comptime swizzle_width = Self.c_swizzle.bytes() // size_of[Self.c_type]()
    comptime data_paths = 16
    comptime barrier_threads = Self.num_output_warps * WARP_SIZE
    comptime OutputSyncBarrier = WarpGroupBarrier[Self.barrier_threads]
    comptime Tile = AccumTile[Self.epilogue_dtype, Self.rep_frag_size]
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.num_output_stages,
        128,
    ]

    var warp_id: UInt32
    var c_tiles: Self.CTileArray
    var M: UInt32
    var N: UInt32
    var c_row: UInt32
    var c_col: UInt32

    @always_inline
    def __init__(
        out self,
        warp_id: UInt32,
        c_tiles: SMemTileArray2DRowMajor[
            Self.c_type,
            Self.c_smem_dim0,
            Self.c_smem_dim1,
            Self.num_output_stages,
            128,
        ],
        c_shape: Tuple[UInt32, UInt32],
        c_coord: Tuple[UInt32, UInt32],
    ):
        """Initialize from TileTensor array."""
        self.warp_id = warp_id
        self.c_tiles = c_tiles
        self.M = c_shape[0]
        self.N = c_shape[1]
        self.c_row = c_coord[0] * UInt32(Self.BM)
        self.c_col = c_coord[1] * UInt32(Self.MMA_N)

    @always_inline
    def write_tile(self, tile: Self.Tile):
        """Write accumulator tile to SMEM and apply epilogue lambda."""
        # Double-buffer tile selection
        var c_smem_tile = self.c_tiles[Self.stage % Self.num_output_stages]

        comptime if Self.transpose_c:
            self._write_transpose(tile.upper, tile.lower, c_smem_tile)
        else:
            self._write_non_transpose(tile.upper, tile.lower, c_smem_tile)

    @always_inline
    def _write_transpose(
        self,
        upper_frag: Array[Scalar[Self.epilogue_dtype], Self.rep_frag_size],
        lower_frag: Array[Scalar[Self.epilogue_dtype], Self.rep_frag_size],
        c_smem_tile: TileTensor[
            Self.c_type,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
    ):
        """Transpose path: reshape tiles and apply epilogue."""

        comptime if Self.is_lower_frag_required:
            # cta_group=2 path with both upper and lower fragments
            comptime tile_width = 32
            comptime num_swblocks = Self.stage_contiguous_size // Self.swizzle_width

            # Old Layout for shared_memory_epilogue_transpose coordinate computation
            comptime smem_swblock_layout = Layout.row_major(
                Self.stageN, 2, tile_width
            )
            comptime smem_logical_layout = Layout(
                flatten([num_swblocks, smem_swblock_layout.shape]),
                flatten(
                    [
                        Self.stageN * Self.swizzle_width,
                        smem_swblock_layout.stride,
                    ]
                ),
            )

            # 4D logical layout: (num_swblocks, stageN, 2, tile_width)
            comptime logical_layout = InternalLayout(
                Coord(
                    Idx[num_swblocks],
                    Idx[Self.stageN],
                    Idx[2],
                    Idx[tile_width],
                ),
                Coord(
                    Idx[Self.stageN * Self.swizzle_width],
                    Idx[2 * tile_width],
                    Idx[tile_width],
                    Idx[1],
                ),
            )
            var new_smem = c_smem_tile.reshape(logical_layout)

            var warp_j, warp_i = divmod(Int(self.warp_id), 2)
            var tiled = new_smem.tile[1, Self.stageN, 1, tile_width](
                Coord(warp_j, Idx[0], warp_i, Idx[0])
            )

            # Coalesce: (1, stageN, 1, 32) -> (stageN, 32)
            comptime coalesced = InternalLayout(
                Coord(Idx[Self.stageN], Idx[tile_width]),
                Coord(Idx[2 * tile_width], Idx[1]),
            )
            var c_smem_warp_tile = tiled.reshape(coalesced)

            var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
                Self.stageN, Self.data_paths
            ](Coord(Idx[0], Idx[0]))
            var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
                Self.stageN, Self.data_paths
            ](Coord(Idx[0], Idx[1]))

            var warp_offset = warp_i * tile_width
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, transpose_c=Self.transpose_c
            ](upper_frag, c_smem_warp_tile_upper, UInt32(warp_offset))
            warp_offset += tile_width // 2
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, transpose_c=Self.transpose_c
            ](lower_frag, c_smem_warp_tile_lower, UInt32(warp_offset))

            Self.OutputSyncBarrier.sync()

            shared_memory_epilogue_transpose[
                Self.stage,
                Self.stageN,
                Self.c_type,
                smem_logical_layout,
                Self.swizzle,
                Self.compute_lambda_fn,
                Self.num_output_warps,
                2,  # warp_dim
                Self.MMA_M,
                Self.BN,
                Self.cta_group,
            ](
                self.M,
                self.N,
                Int(self.c_col),
                Int(self.c_row),
                new_smem,
                warp_i,
                warp_j,
            )
        else:
            # cta_group=1 path with only upper fragment
            comptime tile_width = 16

            # Old Layout for shared_memory_epilogue_transpose coordinate computation
            comptime smem_logical_layout = Layout.row_major(
                Self.stageN, 4, tile_width
            )

            comptime logical = row_major[Self.stageN, 4, tile_width]()
            var new_smem = c_smem_tile.reshape(logical)

            var tiled = new_smem.tile[Self.stageN, 1, tile_width](
                Coord(Idx[0], Int(self.warp_id), Idx[0])
            )

            # Coalesce: (stageN, 1, 16) -> (stageN, 16)
            comptime coalesced = InternalLayout(
                Coord(Idx[Self.stageN], Idx[tile_width]),
                Coord(Idx[4 * tile_width], Idx[1]),
            )
            var c_smem_warp_tile = tiled.reshape(coalesced)

            var warp_offset = Int(self.warp_id) * tile_width
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, transpose_c=Self.transpose_c
            ](upper_frag, c_smem_warp_tile, UInt32(warp_offset))

            Self.OutputSyncBarrier.sync()

            shared_memory_epilogue_transpose[
                Self.stage,
                Self.stageN,
                Self.c_type,
                smem_logical_layout,
                Self.swizzle,
                Self.compute_lambda_fn,
                Self.num_output_warps,
                1,  # warp_dim
                Self.MMA_M,
                Self.BN,
                Self.cta_group,
            ](
                self.M,
                self.N,
                Int(self.c_col),
                Int(self.c_row),
                new_smem,
                Int(self.warp_id),
                0,
            )

    @always_inline
    def _write_non_transpose(
        self,
        upper_frag: Array[Scalar[Self.epilogue_dtype], Self.rep_frag_size],
        lower_frag: Array[Scalar[Self.epilogue_dtype], Self.rep_frag_size],
        c_smem_tile: TileTensor[
            Self.c_type,
            address_space=.SHARED,
            Engine=DefaultEngine[element_width=1],
            ...,
        ],
    ):
        """Non-transpose path: tile per warp and apply epilogue."""
        comptime c_smem_tile_m = 32 if Self.cta_group == 2 else Self.BM // Self.num_output_warps
        var c_smem_warp_tile = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
            Coord(Int(self.warp_id), Idx[0])
        )

        var c_smem_warp_tile_upper = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](Coord(Idx[0], Idx[0]))
        store_fragment_to_smem[
            Self.swizzle, Self.stageN, transpose_c=Self.transpose_c
        ](upper_frag, c_smem_warp_tile_upper)

        var c_smem_warp_tile_lower = c_smem_warp_tile.tile[
            Self.data_paths, Self.stageN
        ](Coord(Idx[1], Idx[0]))

        comptime if Self.is_lower_frag_required:
            store_fragment_to_smem[
                Self.swizzle, Self.stageN, transpose_c=Self.transpose_c
            ](lower_frag, c_smem_warp_tile_lower)

        Self.OutputSyncBarrier.sync()

        # Precompute old Layout for warp tiles (needed for coordinate algebra)
        comptime c_smem_warp_layout = Layout(
            IntTuple(Self.data_paths, Self.stageN),
            IntTuple(Self.c_smem_dim1, 1),
        )

        # Construct fresh TileTensors from raw pointers to break origin
        # aliasing (upper and lower share the same c_smem_tile origin).
        comptime warp_tile_layout = row_major[Self.data_paths, Self.stageN]()
        comptime SMemPtr = UnsafePointer[
            Scalar[Self.c_type], MutUntrackedOrigin, address_space=.SHARED
        ]
        var upper_tile = TileTensor(
            rebind[SMemPtr](c_smem_warp_tile_upper._storage),
            warp_tile_layout,
        )
        var lower_tile = TileTensor(
            rebind[SMemPtr](c_smem_warp_tile_lower._storage),
            warp_tile_layout,
        )

        shared_memory_epilogue[
            Self.MMA_M,
            Self.data_paths,
            Self.num_stages,
            Self.stage,
            Self.stageN,
            Self.c_type,
            Self.c_smem_dim1,
            Self.simd_size,
            c_smem_warp_layout,
            c_smem_warp_layout,
            Self.swizzle,
            Self.compute_lambda_fn,
            Self.num_output_warps,
        ](
            self.M,
            self.N,
            Int(self.c_col),
            Int(self.c_row),
            upper_tile,
            lower_tile,
        )


# =============================================================================
# Shared Memory Epilogue Functions
# =============================================================================
# These functions apply element-wise compute lambdas to data in shared memory.
# Used when register_based_epilogue=False.


@always_inline
def shared_memory_epilogue_transpose[
    stage: Int,
    stageN: Int,
    c_type: DType,
    c_smem_layout: Layout,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: Int,
    warp_dim: Int,
    MMA_M: Int,
    BN: Int,
    cta_group: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: Int,
    c_row: Int,
    c_smem: TileTensor[
        c_type,
        address_space=.SHARED,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    warp_i: Int,
    warp_j: Int,
):
    """Apply element-wise epilogue to transposed SMEM tile.

    Supports warp_dim=1 (stageN, warp_i, U) or warp_dim=2 (warp_j, stageN, warp_i, UL).
    """
    var gmem_col = UInt32(c_col + stage * stageN)
    var gmem_row = UInt32(c_row)

    comptime simd_size = simd_width_of[c_type]()
    comptime alignment = align_of[SIMD[c_type, simd_size]]()
    comptime swizzle_dim = 64

    comptime if warp_dim == 2:
        # Use new Layout for idx2crd operations
        comptime layout_3d = row_major[2, stageN, swizzle_dim]()
        comptime assert c_smem_layout.rank() == 4, "c_smem_layout must be 4D"
        comptime thread_layout = Layout.row_major(1, 8, 1, 4)
        comptime result = zipped_divide(
            upcast(c_smem_layout, simd_size), thread_layout
        )
        comptime thread_layout_new = row_major[1, 8, 1, 4]()
        var lane = lane_id()
        var crd = thread_layout_new.idx2crd[out_dtype=DType.uint32](lane)
        comptime thread_shape = IntTuple(0, UNKNOWN_VALUE, 0, UNKNOWN_VALUE)

        comptime for iter_i in range(result.shape[1][3].value()):
            comptime for iter_j in range(result.shape[1][1].value()):
                comptime rest_shape = IntTuple(
                    UNKNOWN_VALUE, iter_j, UNKNOWN_VALUE, iter_i
                )
                var coord = RuntimeTuple[
                    [thread_shape, rest_shape], element_type=.uint32
                ](
                    Int(0),
                    Int(crd[1].value()),
                    Int(0),
                    Int(crd[3].value()),
                    warp_j,
                    iter_j,
                    warp_i,
                    iter_i,
                )
                var offset = UInt32(simd_size) * rt_crd2idx[
                    [thread_shape, rest_shape],
                    result.shape,
                    result.stride,
                    DType.uint32,
                ](
                    coord,
                    RuntimeTuple[result.shape](),
                    RuntimeTuple[result.stride](),
                )
                var logical_crd = layout_3d.idx2crd[out_dtype=DType.uint32](
                    Int(offset)
                )
                var local_i: UInt32
                var local_j: UInt32

                var ci = logical_crd[0].value()
                var cj = logical_crd[1].value()
                var ck = logical_crd[2].value()

                comptime if cta_group == 2 and MMA_M == 128:
                    # logical shared memory -> global layout Layout B:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-b
                    local_i = UInt32(cj) + UInt32(ci) * UInt32(BN)
                    local_j = UInt32(ck)
                else:
                    # logical shared memory -> global layout Layout A:
                    # https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-a
                    local_i = UInt32(cj)
                    local_j = UInt32(ci * swizzle_dim + ck)

                # undo swizzle to get logical `c_smem[logical_crd]` value.
                var ptr = (
                    c_smem._storage.unsafe_mut_cast[True]()
                    + swizzle(cj * swizzle_dim + ck)
                    + UInt32(ci * swizzle_dim) * UInt32(stageN)
                )
                var row = local_i + gmem_col
                var col = local_j + gmem_row
                if row < UInt32(Int(M)) and col < UInt32(Int(N)):
                    var val = ptr.load[width=simd_size, alignment=alignment]()
                    ptr.store[width=simd_size, alignment=alignment](
                        compute_lambda_fn[alignment=simd_size](
                            (Int(row), Int(col)), val
                        )
                    )
    else:
        # Layout F: https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-f
        comptime assert c_smem_layout.rank() == 3, "c_smem_layout must be 3D"
        comptime thread_layout = Layout.row_major(min(16, stageN), 1, 2)
        comptime thread_bound = thread_layout.cosize()
        var lane = lane_id()
        if lane < thread_bound:
            comptime result = zipped_divide(
                upcast(c_smem_layout, simd_size), thread_layout
            )
            # Use new Layout for idx2crd operations
            comptime thread_layout_new = row_major[min(16, stageN), 1, 2]()
            var crd = thread_layout_new.idx2crd[out_dtype=DType.uint32](lane)
            comptime thread_shape = IntTuple(UNKNOWN_VALUE, 0, UNKNOWN_VALUE)
            comptime layout_2d_new = row_major[stageN, swizzle_dim]()

            comptime for iter_i in range(result.shape[1][2].value()):
                comptime for iter_j in range(result.shape[1][0].value()):
                    comptime rest_shape = IntTuple(
                        iter_j,
                        UNKNOWN_VALUE,
                        iter_i,
                    )
                    var coord = RuntimeTuple[
                        [thread_shape, rest_shape], element_type=.uint32
                    ](
                        Int(crd[0].value()),
                        Int(0),
                        Int(crd[2].value()),
                        iter_j,
                        warp_i,
                        iter_i,
                    )
                    var offset = UInt32(simd_size) * rt_crd2idx[
                        [thread_shape, rest_shape],
                        result.shape,
                        result.stride,
                        DType.uint32,
                    ](
                        coord,
                        RuntimeTuple[result.shape](),
                        RuntimeTuple[result.stride](),
                    )
                    var logical_crd = layout_2d_new.idx2crd[
                        out_dtype=DType.uint32
                    ](Int(offset))

                    var local_i = logical_crd[0].value()
                    var local_j = logical_crd[1].value()

                    # Undo swizzle to get logical value
                    var ptr = c_smem._storage.unsafe_mut_cast[True]() + swizzle(
                        offset
                    )
                    var row = UInt32(local_i) + gmem_col
                    var col = UInt32(local_j) + gmem_row
                    if row < UInt32(Int(M)) and col < UInt32(Int(N)):
                        var val = ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        ptr.store[width=simd_size, alignment=alignment](
                            compute_lambda_fn[alignment=simd_size](
                                (Int(row), Int(col)), val
                            )
                        )

    WarpGroupBarrier[num_output_warps * WARP_SIZE].sync()


@always_inline
def shared_memory_epilogue[
    MMA_M: Int,
    data_paths: Int,
    num_stages: Int,
    stage: Int,
    stageN: Int,
    c_type: DType,
    shared_n: Int,
    simd_size: Int,
    c_smem_upper_layout: Layout,
    c_smem_lower_layout: Layout,
    swizzle: Swizzle,
    compute_lambda_fn: elementwise_compute_lambda_type,
    num_output_warps: Int,
](
    M: UInt32,
    N: UInt32,
    c_col: Int,
    c_row: Int,
    c_smem_warp_tile_upper: TileTensor[
        c_type,
        address_space=.SHARED,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
    c_smem_warp_tile_lower: TileTensor[
        c_type,
        address_space=.SHARED,
        Engine=DefaultEngine[element_width=1],
        ...,
    ],
):
    """Apply element-wise epilogue to non-transposed SMEM tile.

    Each warp processes upper (rows 0-15) and lower (rows 16-31) fragments.
    Uses distribute layout to map SIMD vectors to threads within each warp.
    """
    # Global column with stage offset
    var gmem_col = Int64(c_col + stage * stageN)

    # Each warp owns 32 rows: upper half (0-15) and lower half (16-31)
    var warp_base_row = warp_id() * 32
    var upper_row = warp_base_row
    var lower_row = warp_base_row + 16

    # Distribute layout: maps stageN elements across warp threads
    # e.g., stageN=32 → 8x4 (8 rows × 4 cols), stageN=16 → 16x2, stageN=8 → 16x1
    # Cap distribute_rows at data_paths to avoid exceeding the tile's row count
    # when stageN is small (e.g., stageN=8 gives only 1 SIMD vector per row,
    # so 16 threads suffice to cover all 16 data_path rows).
    comptime distribute_cols = stageN // simd_size
    comptime distribute_rows = min(data_paths, WARP_SIZE // distribute_cols)
    comptime active_lanes = distribute_rows * distribute_cols

    comptime distribute_layout = row_major[distribute_rows, distribute_cols]()

    if lane_id() < active_lanes:
        # Construct TileTensors with known layout to enable type resolution
        # through .vectorize().distribute() chain.
        comptime tile_layout = row_major[data_paths, stageN]()
        var upper_tt = TileTensor(
            c_smem_warp_tile_upper._storage.unsafe_mut_cast[True](),
            tile_layout,
        )
        var lower_tt = TileTensor(
            c_smem_warp_tile_lower._storage.unsafe_mut_cast[True](),
            tile_layout,
        )

        var c_smem_upper_frag = upper_tt.vectorize[1, simd_size]().distribute[
            distribute_layout, swizzle=swizzle
        ](lane_id())

        var c_smem_lower_frag = lower_tt.vectorize[1, simd_size]().distribute[
            distribute_layout, swizzle=swizzle
        ](lane_id())

        comptime fragment_size = c_smem_upper_frag.LayoutType.static_product

        var lane_row, lane_col = udivmod(lane_id(), distribute_cols)
        var col = lane_col * simd_size
        upper_row += lane_row
        lower_row += lane_row

        comptime for i in range(fragment_size):
            # Compute swizzled SMEM offsets, then un-swizzle to get logical coords
            var swz_offset_upper = upper_row * shared_n + col
            var swz_offset_lower = lower_row * shared_n + col

            var offset_upper = swizzle(swz_offset_upper)
            var offset_lower = swizzle(swz_offset_lower)

            var local_upper_row: Int64
            var local_upper_col: Int64
            var local_lower_row: Int64
            var local_lower_col: Int64

            # Convert SMEM offset to logical (row, col) - layout differs by MMA_M size
            comptime if MMA_M != 256:
                comptime blocked_m_128_layout = blocked_product(
                    Layout.row_major(data_paths * 2, stageN),
                    Layout.col_major(2, 2),
                    coalesce_output=True,
                )

                var upper_coord = idx2crd(
                    RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset_upper),
                    RuntimeTuple[
                        blocked_m_128_layout.shape,
                        element_type=.int64,
                    ](),
                    RuntimeTuple[
                        blocked_m_128_layout.stride,
                        element_type=.int64,
                    ](),
                )

                var lower_coord = idx2crd(
                    RuntimeTuple[IntTuple(UNKNOWN_VALUE)](offset_lower),
                    RuntimeTuple[
                        blocked_m_128_layout.shape,
                        element_type=.int64,
                    ](),
                    RuntimeTuple[
                        blocked_m_128_layout.stride,
                        element_type=.int64,
                    ](),
                )

                local_upper_row = upper_coord[0].get_int()
                local_lower_row = lower_coord[0].get_int()

                var section_offset_upper = upper_coord[1][1].get_int()
                var col_offset_upper = upper_coord[1][0].get_int()
                var section_offset_lower = lower_coord[1][1].get_int()
                var col_offset_lower = lower_coord[1][0].get_int()

                local_upper_col = (
                    section_offset_upper * Int64(num_stages * stageN)
                    + col_offset_upper
                )
                local_lower_col = (
                    section_offset_lower * Int64(num_stages * stageN)
                    + col_offset_lower
                )

            else:
                # MMA_M=256: simple row-major indexing
                comptime fast_div = FastDiv[.uint32](shared_n)

                local_upper_row = (
                    Int(offset_upper).cast[fast_div.uint_type]() / fast_div
                ).cast[.int64]()
                local_upper_col = Int64(offset_upper % shared_n)

                local_lower_row = (
                    Int(offset_lower).cast[fast_div.uint_type]() / fast_div
                ).cast[.int64]()
                local_lower_col = Int64(offset_lower % shared_n)

            # Convert local SMEM coords to global memory coords
            var gmem_upper_row = local_upper_row + Int64(c_row)
            var gmem_upper_col = local_upper_col + gmem_col
            var gmem_lower_row = local_lower_row + Int64(c_row)
            var gmem_lower_col = local_lower_col + gmem_col

            # Apply epilogue if within bounds
            if gmem_upper_row < Int64(Int(M)) and gmem_upper_col < Int64(
                Int(N)
            ):
                c_smem_upper_frag[i, 0] = compute_lambda_fn[
                    alignment=simd_size
                ](
                    (Int(gmem_upper_row), Int(gmem_upper_col)),
                    c_smem_upper_frag[i, 0],
                )

            if gmem_lower_row < Int64(Int(M)) and gmem_lower_col < Int64(
                Int(N)
            ):
                c_smem_lower_frag[i, 0] = compute_lambda_fn[
                    alignment=simd_size
                ](
                    (Int(gmem_lower_row), Int(gmem_lower_col)),
                    c_smem_lower_frag[i, 0],
                )

            # Advance to next chunk (spaced distribute_rows apart)
            upper_row += distribute_rows
            lower_row += distribute_rows

    WarpGroupBarrier[num_output_warps * WARP_SIZE].sync()
