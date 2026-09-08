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

"""Output writer for blockwise FP8 SM100 matmul.

Handles Register → SMEM → GMEM (via TMA) flow. Unlike standard matmul which
reads from TMEM, blockwise FP8 accumulators are already in registers.

Supports two write modes:
- write(): TMA store for standard non-grouped matmul
- write_absolute_with_bounds_check(): Element-by-element store for 1D2D
  grouped matmul with expert boundary bounds checking
"""

from std.sys import align_of, simd_width_of, size_of

from max.gpu import WARP_SIZE, thread_idx, lane_id, warp_id as get_warp_id
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.memory import fence_async_view_proxy
from max.gpu.sync import named_barrier
from layout import (
    ComptimeInt,
    Coord,
    CoordLike,
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.swizzle import make_swizzle
from layout.tile_layout import Layout, UpcastLayout, ZippedDivideLayout
from layout.tma_async import TMATensorTile
from std.utils.index import IndexList

from .blockwise_fp8_accumulator import BlockwiseFP8Accumulator
from ..structured_kernels.epilogue_components import (
    EpilogueConfig,
    TMEMToSMemWriter,
    TMAStoreCoords,
    TMAStoreExecutor,
    tma_wait_pipelined,
)
from structured_kernels.barriers import WarpGroupBarrier
from linalg.matmul.gpu.sm100.matmul import stsm_helper

# TileTensor-based types for C tiles
from structured_kernels.tile_types import SMemTileArray2DRowMajor


# =============================================================================
# BlockwiseFP8TileWriter - Write register accumulators to GMEM
# =============================================================================


struct BlockwiseFP8TileWriter[
    c_type: DType,
    c_smem_dim0: Int,
    c_smem_dim1: Int,
    accum_type: DType,
    accum_num_stages: Int,
    accum_num_elements: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    is_lower_frag_required: Bool,
    cta_group: Int,
    num_output_stages: Int,
    num_output_warps: Int,
    c_swizzle: TensorMapSwizzle,
]:
    """Write register accumulators to GMEM via SMEM and TMA.

    Parameters:
        c_type: Element type of the C output tensor.
        c_smem_dim0: M dimension of the C shared-memory tile.
        c_smem_dim1: N dimension of the C shared-memory tile.
        accum_type: Element type of the accumulator registers.
        accum_num_stages: Number of accumulator pipeline stages to drain.
        accum_num_elements: Number of elements per accumulator fragment set.
        block_tile_shape: Block tile shape as (BM, BN, BK).
        mma_shape: MMA instruction shape as (MMA_M, MMA_N, MMA_K).
        is_lower_frag_required: Whether the lower register fragment is populated.
        cta_group: Number of CTAs cooperating per output tile.
        num_output_stages: Number of SMEM buffer stages for the output epilogue.
        num_output_warps: Number of warps participating in the output epilogue.
        c_swizzle: TMA swizzle pattern applied to the C shared-memory tile.
    """

    # ========== Layout from dimensions ==========
    comptime c_smem_layout = row_major[Self.c_smem_dim0, Self.c_smem_dim1]()

    # ========== Tile Array Types ==========
    comptime CTileArray = SMemTileArray2DRowMajor[
        Self.c_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.num_output_stages,
        128,
    ]

    comptime BM = Self.block_tile_shape[0]
    comptime BN = Self.block_tile_shape[1]
    comptime MMA_M = Self.mma_shape[0]
    comptime MMA_N = Self.mma_shape[1]

    comptime num_stages = Self.accum_num_stages
    comptime num_elements = Self.accum_num_elements

    comptime data_paths = 16
    comptime bits = 256
    comptime num_elements_per_load = Self.bits // 32
    comptime fragment_size = (
        Self.data_paths * Self.num_elements_per_load
    ) // WARP_SIZE
    comptime repeats = Self.num_elements // Self.fragment_size
    comptime stageN = Self.repeats * (Self.bits // 32)
    comptime fragments_per_stage = Self.fragment_size * Self.repeats

    # EpilogueConfig bundles common epilogue parameters
    comptime epc = EpilogueConfig.create(
        MMA_M=Self.MMA_M,
        MMA_N=Self.MMA_N,
        stageN=Self.stageN,
        cta_group=Self.cta_group,
        transpose_c=False,  # blockwise FP8 never transposes
        BM=Self.BM,
        BN=Self.BN,
    )

    # Reuse TMEMToSMemWriter for fragment → SMEM path
    comptime SMEMWriter = TMEMToSMemWriter[
        Self.c_type,
        Self.accum_type,
        Self.c_smem_dim0,
        Self.c_smem_dim1,
        Self.epc,
        Self.num_output_warps,
        Self.c_swizzle,
    ]

    # ========== Public Write Method ==========

    @staticmethod
    @always_inline
    def write[
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_num_stages,
            Self.accum_num_elements,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
            _,
        ],
        c_tiles: Self.CTileArray,
        c_tma_op: TMATensorTile[
            Self.c_type, c_rank, c_tile_shape, c_desc_shape
        ],
        c_coord: Tuple[Int, Int],
    ):
        """Write accumulated register tiles to GMEM via double-buffered SMEM.

        Parameters:
            c_rank: Rank of the C output tensor.
            c_tile_shape: Tile shape of the C output tensor.
            c_desc_shape: Descriptor shape of the C output tensor.
            cluster_size: Size of the threadblock cluster for the matmul.

        Args:
            accum: Blockwise FP8 accumulator holding upper and lower register tiles.
            c_tiles: Double-buffered SMEM tile array for C output.
            c_tma_op: TMA tensor tile descriptor for the C store.
            c_coord: (M, N) tile coordinate of this C tile in the output tensor.
        """
        Self._write_impl[c_rank, c_tile_shape, c_desc_shape, cluster_size](
            accum, c_tiles, c_tma_op, c_coord
        )

    # ========== Internal Implementation ==========

    @staticmethod
    @always_inline
    def _write_impl[
        c_rank: Int,
        c_tile_shape: IndexList[c_rank],
        c_desc_shape: IndexList[c_rank],
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_num_stages,
            Self.accum_num_elements,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
            _,
        ],
        c_tiles: Self.CTileArray,
        c_tma_op: TMATensorTile[
            Self.c_type, c_rank, c_tile_shape, c_desc_shape
        ],
        c_coord: Tuple[Int, Int],
    ):
        """Internal implementation for writing accumulated register tiles."""
        var warp_id = get_warp_id()
        var smem_writer = Self.SMEMWriter(UInt32(warp_id), UInt32(lane_id()))

        comptime for stage in range(Self.num_stages):
            var upper_frag = accum.upper.load[Self.fragments_per_stage](
                Coord(stage, Idx[0])
            )
            var lower_frag = accum.lower.load[Self.fragments_per_stage](
                Coord(stage, Idx[0])
            )

            var c_smem_tile = c_tiles[stage % 2]  # double-buffer

            # Cast from accum_type to c_type in SIMD chunks of at
            # least 4 bytes for efficient hardware cast instructions.
            comptime frag_size = Self.epc.fragment_size * Self.repeats
            var upper_st = Array[Scalar[Self.c_type], frag_size](
                uninitialized=True
            )
            var lower_st = Array[Scalar[Self.c_type], frag_size](
                uninitialized=True
            )

            comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
            comptime for _chunk in range(
                Self.fragments_per_stage // cast_width
            ):
                comptime offset = _chunk * cast_width
                var casted_u = upper_frag.slice[
                    cast_width, offset=offset
                ]().cast[Self.c_type]()
                var casted_l = lower_frag.slice[
                    cast_width, offset=offset
                ]().cast[Self.c_type]()
                comptime for _j in range(cast_width):
                    upper_st[offset + _j] = casted_u[_j]
                    lower_st[offset + _j] = casted_l[_j]
            smem_writer.write_fragments[Self.repeats](
                rebind[Array[Scalar[Self.c_type], frag_size]](upper_st),
                rebind[Array[Scalar[Self.c_type], frag_size]](lower_st),
                c_smem_tile,
            )

            named_barrier[Int32(Self.num_output_warps * WARP_SIZE)]()

            var lane = lane_id()

            # Use shared TMA store components from epilogue_components
            comptime StoreCoords = TMAStoreCoords[
                Self.epc,
                Self.c_smem_dim0,
                stage,
            ]
            var store_coords = StoreCoords(
                (UInt32(c_coord[0]), UInt32(c_coord[1])), UInt32(warp_id)
            )

            comptime StoreExec = TMAStoreExecutor[
                Self.c_type,
                Self.c_smem_dim0,
                Self.c_smem_dim1,
                Self.epc,
                Self.stageN,  # stage_contiguous_size
                Self.c_swizzle,
            ]
            StoreExec.execute[c_rank, c_tile_shape, c_desc_shape](
                c_smem_tile,
                store_coords,
                c_tma_op,
                UInt32(warp_id),
                UInt32(lane),
            )
            tma_wait_pipelined[
                Self.c_type,
                c_rank,
                c_tile_shape,
                c_desc_shape,
                stage == Self.num_stages - 1,
            ](c_tma_op)

            comptime if stage > 0 and stage < Self.num_stages - 1:
                named_barrier[Int32(Self.num_output_warps * WARP_SIZE)]()

    # ========== Bounds-Checked Write for 1D2D Grouped Matmul ==========

    @staticmethod
    @always_inline
    def write_absolute_with_bounds_check[
        c_tensor_layout: TensorLayout,
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_num_stages,
            Self.accum_num_elements,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
            _,
        ],
        c_tiles: Self.CTileArray,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
    ):
        """Write accumulated register tiles to GMEM with bounds checking.

        Parameters:
            c_tensor_layout: Layout of the C output tensor in GMEM used for
                bounds-checked element stores.
            cluster_size: Number of CTAs in the threadblock cluster for the
                matmul.

        Args:
            accum: Blockwise FP8 accumulator with upper/lower register tiles.
            c_tiles: SMEM tile array for C output.
            m_abs: Absolute M coordinate (start of tile in token space).
            n_abs: Absolute N coordinate (start of tile).
            m_end: End offset for bounds checking (exclusive).
            expert_scale: Per-expert output scaling factor.
            c_tensor: C tensor in GMEM (TileTensor for bounds-checked stores).
        """
        Self._write_absolute_impl[c_tensor_layout, cluster_size](
            accum, c_tiles, m_abs, n_abs, m_end, expert_scale, c_tensor
        )

    @staticmethod
    @always_inline
    def _write_absolute_impl[
        c_tensor_layout: TensorLayout,
        cluster_size: Int,
    ](
        accum: BlockwiseFP8Accumulator[
            Self.accum_type,
            Self.accum_num_stages,
            Self.accum_num_elements,
            Self.is_lower_frag_required,
            Self.block_tile_shape,
            Self.mma_shape,
            cluster_size,
            _,
        ],
        c_tiles: Self.CTileArray,
        m_abs: UInt32,
        n_abs: UInt32,
        m_end: UInt32,
        expert_scale: Float32,
        c_tensor: TileTensor[
            mut=True, Self.c_type, LayoutType=c_tensor_layout, ...
        ],
    ):
        """Internal implementation for bounds-checked register-to-GMEM write.

        Uses stsm_helper for register -> SMEM path to support all output
        tile shapes (including narrow c_smem_dim1=32 with bf16).
        """
        var warp_id = get_warp_id()
        var scale = expert_scale.cast[Self.accum_type]()

        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()
        comptime c_smem_tile_m = Self.BM // Self.num_output_warps

        comptime for stage in range(Self.num_stages):
            var upper_frag = accum.upper.load[Self.fragments_per_stage](
                Coord(stage, Idx[0])
            )
            var lower_frag = accum.lower.load[Self.fragments_per_stage](
                Coord(stage, Idx[0])
            )

            # Apply expert scale
            upper_frag = upper_frag * scale

            comptime if Self.is_lower_frag_required:
                lower_frag = lower_frag * scale

            var c_smem_tile = c_tiles[stage % 2]  # double-buffer

            # Write register fragments to SMEM using stsm_helper
            # (handles bf16 correctly with stsmx instead of stmtx)
            var c_smem_warp_tt = c_smem_tile.tile[c_smem_tile_m, Self.stageN](
                warp_id, 0
            )
            var c_smem_warp_tile_upper = c_smem_warp_tt.tile[
                Self.data_paths, Self.stageN
            ](0, 0)
            # Cast in SIMD chunks of at least 4 bytes for efficient
            # hardware cast instructions.
            var upper_st = Array[Scalar[Self.c_type], Self.fragments_per_stage](
                uninitialized=True
            )

            comptime cast_width = 4 // size_of[Scalar[Self.c_type]]()
            comptime for _chunk in range(
                Self.fragments_per_stage // cast_width
            ):
                comptime offset = _chunk * cast_width
                var casted = upper_frag.slice[cast_width, offset=offset]().cast[
                    Self.c_type
                ]()
                comptime for _j in range(cast_width):
                    upper_st[offset + _j] = casted[_j]
            stsm_helper[swizzle, Self.stageN, swizzle_mode=Self.c_swizzle](
                upper_st, c_smem_warp_tile_upper
            )

            var c_smem_warp_tile_lower = c_smem_warp_tt.tile[
                Self.data_paths, Self.stageN
            ](1, 0)

            comptime if Self.is_lower_frag_required:
                var lower_st = Array[
                    Scalar[Self.c_type], Self.fragments_per_stage
                ](uninitialized=True)

                comptime for _chunk in range(
                    Self.fragments_per_stage // cast_width
                ):
                    comptime offset = _chunk * cast_width
                    var casted = lower_frag.slice[
                        cast_width, offset=offset
                    ]().cast[Self.c_type]()
                    comptime for _j in range(cast_width):
                        lower_st[offset + _j] = casted[_j]
                stsm_helper[swizzle, Self.stageN, swizzle_mode=Self.c_swizzle](
                    lower_st, c_smem_warp_tile_lower
                )

            named_barrier[Int32(Self.num_output_warps * WARP_SIZE)]()

            # Bounds-checked element stores from SMEM to GMEM
            Self._store_with_bounds_check[c_tensor_layout](
                c_smem_tile,
                c_tensor,
                m_abs,
                n_abs + UInt32(stage * Self.stageN),
                m_end,
                UInt32(warp_id),
                UInt32(lane_id()),
            )

            comptime if stage > 0 and stage < Self.num_stages - 1:
                named_barrier[Int32(Self.num_output_warps * WARP_SIZE)]()

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
        """Store SMEM tile to GMEM with per-row bounds checking.

        Used when the tile crosses the expert boundary.
        Uses element-by-element stores to avoid writing past m_end.
        """
        comptime output_threads = Self.num_output_warps * WARP_SIZE
        comptime c_smem_M = Self.c_smem_dim0
        comptime TMA_BM = c_smem_M  # blockwise FP8 uses cta_group==1 always
        comptime simd_size = simd_width_of[Self.c_type]()
        comptime alignment = align_of[SIMD[Self.c_type, simd_size]]()
        comptime thread_n = Self.stageN // simd_size
        comptime thread_layout = row_major[
            output_threads // thread_n, thread_n
        ]()

        # Swizzle function
        comptime swizzle = make_swizzle[Self.c_type, Self.c_swizzle]()

        # Ensure fence before reading from SMEM
        if warp_id == 0 and lane == 0:
            fence_async_view_proxy()

        # Synchronize all epilogue threads
        WarpGroupBarrier[Self.num_output_warps * WARP_SIZE].sync()

        # Precompute the split layout from known dimensions (stride
        # inherited from the row_major parent tile).
        comptime split_layout = Layout[
            shape_types=Coord[
                ComptimeInt[TMA_BM],
                ComptimeInt[Self.stageN],
            ].element_types,
            stride_types=Coord[
                ComptimeInt[Self.c_smem_dim1],
                ComptimeInt[1],
            ].element_types,
        ]

        # Iterate over SMEM chunks
        comptime for i in range(c_smem_M // TMA_BM):
            var c_smem_split = c_smem_tile.tile[TMA_BM, Self.stageN](i, 0)
            comptime zipped = ZippedDivideLayout[
                UpcastLayout[split_layout, simd_size],
                thread_layout.shape_types,
            ]()
            comptime split_layout_new = row_major[TMA_BM, Self.stageN]()

            comptime for j in range(
                zipped.shape_types[1].element_types[0].static_value
            ):
                var input_crd = Coord(UInt32(thread_idx.x), Idx[j])
                var linear_idx = zipped[linear_idx_type=DType.uint32](
                    input_crd
                ) * UInt32(simd_size)
                var cmem_crd = split_layout_new.idx2crd[out_dtype=DType.uint32](
                    Int(linear_idx)
                )
                var local_i = cmem_crd[0].value()
                var local_j = cmem_crd[1].value()
                var coord_m = m_abs + UInt32(i * TMA_BM)
                var global_i = coord_m + UInt32(local_i)
                var global_j = n_abs + UInt32(local_j)

                # Bounds check: only store if within expert boundary
                if global_i < m_end:
                    var dst_crd = Coord(Int(global_i), Int(global_j))

                    comptime if size_of[Self.c_type]() == 2:
                        var src_ptr = c_smem_split._storage + swizzle(
                            linear_idx
                        )
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        c_tensor.store[width=simd_size, alignment=alignment](
                            dst_crd, src
                        )
                    else:
                        var src_ptr = c_smem_split._storage + linear_idx
                        var src = src_ptr.load[
                            width=simd_size, alignment=alignment
                        ]()
                        c_tensor.store[width=simd_size, alignment=alignment](
                            dst_crd, src
                        )
