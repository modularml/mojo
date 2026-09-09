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

"""Provides the output writer for SM90 warp-group matrix multiply-accumulate kernels.

Defines `MatmulTileWriter`, which stores WGMMA accumulator register tiles to
global memory using either TMA async stores or thread-wise stores, with support
for swizzled shared memory layouts, optional elementwise epilogues, and the
swapAB small-M strategy.
"""

from std.math import ceildiv
from std.sys import simd_width_of, size_of

from max.gpu.globals import WARPGROUP_SIZE
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.sync import named_barrier
from layout import Coord, Idx, Layout, TensorLayout, TileTensor, row_major
from layout.swizzle import make_ldmatrix_swizzle
from layout.tensor_engine import TensorEngine
from layout.tma_async import TMATensorTile

from std.utils.index import IndexList

from ....utils import elementwise_compute_lambda_type, elementwise_epilogue_type
from std.collections import OptionalReg
from ....structuring import (
    RegTile,
)
from .tile_writer import (
    TileWriterTMA,
    TileWriterThreadwise,
    FragmentToSMemWriter,
    RegisterToGMemWriter,
    TileCoordinates,
)
import std.itertools


struct MatmulTileWriter[
    dtype: DType,
    tensor_layout: TensorLayout,
    output_engine: TensorEngine,
    linear_idx_type: DType,
    smem_tile_layout: TensorLayout,
    //,
    /,
    *,
    BM: Int,
    BN: Int,
    swizzle: TensorMapSwizzle,
    wgmma_shape: IndexList[3],
    num_consumer: Int = 1,
    use_tma_store: Bool = False,
    elementwise_lambda_fn: Optional[elementwise_epilogue_type] = None,
    elementwise_compute_lambda_fn: Optional[
        elementwise_compute_lambda_type
    ] = None,
    swapAB: Bool = False,
](TrivialRegisterPassable):
    """Writes WGMMA accumulator register tiles to global memory for SM90 matmul.

    Provides two write paths: TMA store (shared memory → global via hardware
    async copies) and thread-wise store (register → global directly). Handles
    swizzled shared memory layouts, optional elementwise epilogues, and A/B swap
    when the problem uses the swapAB small-M strategy.

    Parameters:
        dtype: Element type of the output tensor and shared memory tile
            (inferred).
        tensor_layout: Memory layout of the output tensor in global memory
            (inferred).
        output_engine: Engine backing the output tensor in global memory
            (inferred).
        linear_idx_type: Integer type used for linear index arithmetic
            into the output tensor (inferred).
        smem_tile_layout: Memory layout of the shared memory tile used to
            stage output before storing to global memory; also defines the
            workgroup tile dimensions `WG_BM` and `WG_BN` (inferred).
        BM: Row (M) dimension of the matmul output tile per block.
        BN: Column (N) dimension of the matmul output tile per block.
        swizzle: TMA swizzle mode applied to shared memory for TMA async
            stores.
        wgmma_shape: Shape of each WGMMA instruction as a 3-element index
            list `(M, N, K)`.
        num_consumer: Number of consumer warp groups sharing the output
            tile (defaults to 1).
        use_tma_store: Whether to use TMA async stores for shared memory
            to global memory copies (defaults to `False`).
        elementwise_lambda_fn: Optional epilogue lambda invoked with
            each output element's global coordinates and value,
            writing directly to global memory
            (defaults to `None`).
        elementwise_compute_lambda_fn: Optional compute lambda that
            transforms each output element value before it is written
            back to shared memory and then to global memory (defaults
            to `None`).
        swapAB: Whether to swap the A and B operand roles for the small-M
            strategy, transposing the tile and block coordinate mapping
            (defaults to `False`).
    """

    comptime N = Self.tensor_layout.static_shape[1]
    comptime frag_size = Self.wgmma_shape[0] * Self.wgmma_shape[
        1
    ] // WARPGROUP_SIZE
    comptime num_m_mmas = Self.BM // Self.wgmma_shape[0] // Self.num_consumer
    comptime num_n_mmas = Self.BN // Self.wgmma_shape[1]
    comptime num_consumer_threads = Self.num_consumer * WARPGROUP_SIZE
    comptime simd_size = simd_width_of[Self.dtype]()

    # Layout dimensions
    comptime WG_BM = Self.smem_tile_layout.static_shape[0]
    comptime WG_BN = Self.smem_tile_layout.static_shape[1]

    comptime CTensorType = TileTensor[
        mut=True,
        Self.dtype,
        LayoutType=Self.tensor_layout,
        origin=MutAnyOrigin,
        Engine=Self.output_engine,
        address_space=.GENERIC,
        linear_idx_type=Self.linear_idx_type,
    ]
    comptime lambda_type = def[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](IndexList[2], mut SIMD[dtype, width]) -> None

    # Instance fields
    @__allow_legacy_any_origin_fields
    var tensor: Self.CTensorType

    @__allow_legacy_any_origin_fields
    var smem_tile: TileTensor[
        mut=True,
        Self.dtype,
        LayoutType=Self.smem_tile_layout,
        origin=MutAnyOrigin,
        address_space=.SHARED,
    ]
    var warp_group_thread_idx: Int
    var local_warp_group_idx: Int
    var local_thread_idx: Int
    var block_y: Int
    var block_x: Int

    @always_inline
    def __init__(
        out self,
        tensor: Self.CTensorType,
        smem_tile: TileTensor[
            mut=True,
            Self.dtype,
            LayoutType=Self.smem_tile_layout,
            origin=MutAnyOrigin,
            address_space=.SHARED,
        ],
        warp_group_thread_idx: Int,
        local_warp_group_idx: Int,
        local_thread_idx: Int,
        block_y: Int,
        block_x: Int,
    ):
        self.tensor = tensor
        self.smem_tile = smem_tile
        self.warp_group_thread_idx = warp_group_thread_idx
        self.local_warp_group_idx = local_warp_group_idx
        self.local_thread_idx = local_thread_idx
        self.block_y = block_y
        self.block_x = block_x

    @always_inline
    def _calculate_output_bounds(self) -> Tuple[UInt32, UInt32]:
        """Calculate valid output bounds for the current block's tile."""
        var rows = self.tensor.dim[0]()
        var max_row: UInt32
        var max_col: UInt32

        comptime if Self.swapAB:
            # swapAB: tile covers rows [block_x * BN, ...] and cols [block_y * BM, ...]
            max_row = min(UInt32((self.block_x + 1) * Self.BN), UInt32(rows))
            max_col = min(UInt32((self.block_y + 1) * Self.BM), UInt32(Self.N))
        else:
            # Normal: tile covers rows [block_y * BM, ...] and cols [block_x * BN, ...]
            max_row = min(UInt32((self.block_y + 1) * Self.BM), UInt32(rows))
            max_col = min(UInt32((self.block_x + 1) * Self.BN), UInt32(Self.N))

        return max_row, max_col

    @always_inline
    def _apply_epilogue[
        F: ImplicitlyCopyable & RegisterPassable & Self.lambda_type
    ](
        self,
        epilogue: F,
        output_tile: TileTensor[mut=True, Self.dtype, ...],
        tile_row_offset: Int,
        tile_col_offset: Int,
        max_row: UInt32,
        max_col: UInt32,
    ):
        """Apply epilogue operations (bias, activation) to shared memory data.
        """
        comptime smem_swizzle = make_ldmatrix_swizzle[Self.dtype, Self.WG_BN]()
        comptime thread_layout = row_major[
            Self.num_consumer_threads // (Self.WG_BN // Self.simd_size),
            Self.WG_BN // Self.simd_size,
        ]()

        var output_fragment, fragment_offsets, _ = output_tile.vectorize[
            1, Self.simd_size
        ]().distribute_with_offset[thread_layout](self.local_thread_idx)
        var row_coord = tile_row_offset + fragment_offsets[0]
        var col_coord = tile_col_offset + fragment_offsets[1] * Self.simd_size

        var shared_fragment = self.smem_tile.vectorize[
            1, Self.simd_size
        ]().distribute[thread_layout, swizzle=smem_swizzle](
            self.local_thread_idx
        )

        comptime num_elements_per_thread = (
            type_of(output_fragment).LayoutType.static_product
        )

        comptime for i in range(num_elements_per_thread):
            var output_idx = Int(
                type_of(output_fragment).LayoutType()(Coord(Idx[i], Idx[0]))
            )
            var row_offset, col_offset = divmod(output_idx, Self.N)
            var row = UInt32(row_coord + row_offset)
            var col = UInt32(col_coord + col_offset)

            if row < max_row and col < max_col:
                var shared_value = shared_fragment.load(Coord(Idx[i], Idx[0]))
                epilogue[
                    dtype=type_of(shared_value).dtype,
                    width=type_of(shared_value).length,
                ](IndexList[2](Int(row), Int(col)), shared_value)
                shared_fragment.store(Coord(Idx[i], Idx[0]), shared_value)

    @always_inline
    def _write_tile_to_gmem[
        accum_type: DType,
        reg_tile_layout: Layout,
        //,
        check_runtime_bounds: Bool = False,
    ](self, reg_tile: RegTile[accum_type, reg_tile_layout]):
        """Write from registers to global memory."""

        comptime out_tile_size_m = Self.BM if not Self.swapAB else Self.BN
        comptime out_tile_size_n = Self.BN if not Self.swapAB else Self.BM

        var m_block = self.block_y if not Self.swapAB else self.block_x
        var n_block = self.block_x if not Self.swapAB else self.block_y

        var output_tile, tile_origin, _ = self.tensor.tile_with_offset[
            out_tile_size_m, out_tile_size_n
        ](Coord(m_block, n_block))

        # For normal: M is divided by num_consumer, N stays full
        # For swapAB: M stays full (BN), N is divided by num_consumer
        comptime tile_slice_m_regular = Self.BM // Self.num_consumer
        comptime tile_slice_n_regular = Self.BN
        comptime tile_slice_m_swapAB = Self.BN
        comptime tile_slice_n_swapAB = Self.BM // Self.num_consumer

        comptime tile_slice_m = tile_slice_m_regular if not Self.swapAB else tile_slice_m_swapAB
        comptime tile_slice_n = tile_slice_n_regular if not Self.swapAB else tile_slice_n_swapAB

        var coord_m = self.local_warp_group_idx if not Self.swapAB else 0
        var coord_n = 0 if not Self.swapAB else self.local_warp_group_idx

        var consumer_tile, consumer_coords, _ = output_tile.tile_with_offset[
            tile_slice_m, tile_slice_n
        ](Coord(coord_m, coord_n))

        var tile_coords: OptionalReg[TileCoordinates] = None
        var max_row: OptionalReg[UInt32] = None

        comptime if (
            Self.elementwise_lambda_fn is not None
            or Self.elementwise_compute_lambda_fn is not None
        ):
            tile_coords = TileCoordinates(
                IndexList[2](tile_origin[0], tile_origin[1]),
                IndexList[2](consumer_coords[0], consumer_coords[1]),
            )
            max_row = UInt32(self.tensor.dim[0]())

        var reg_writer = RegisterToGMemWriter[
            wgmma_shape=Self.wgmma_shape,
            num_consumer=Self.num_consumer,
            N=Self.N,
            epilogue_fn=Self.elementwise_lambda_fn,
            compute_lambda_fn=Self.elementwise_compute_lambda_fn,
            check_runtime_bounds=check_runtime_bounds,
            swapAB=Self.swapAB,
        ](
            consumer_tile,
            self.warp_group_thread_idx,
            Self.num_m_mmas,
            tile_coords,
            max_row,
        )

        comptime for row_tile, col_tile in std.itertools.product(
            range(Self.num_m_mmas), range(Self.num_n_mmas)
        ):
            reg_writer.write_tile(
                reg_tile,
                (row_tile, col_tile),
            )

    @always_inline
    def _write_tile_stmatrix[
        tma_rank: Int,
        tma_tile_shape: IndexList[tma_rank],
        tma_desc_shape: IndexList[tma_rank],
        accum_type: DType,
        reg_tile_layout: Layout,
        //,
    ](
        self,
        tma_op: TMATensorTile[
            Self.dtype, tma_rank, tma_tile_shape, tma_desc_shape
        ],
        reg_tile: RegTile[accum_type, reg_tile_layout],
        output_tile: TileTensor[mut=True, Self.dtype, ...],
        tile_origin: IndexList[2],
    ):
        """Use st.matrix instructions for optimized bf16 output."""
        var max_row, max_col = self._calculate_output_bounds()

        comptime TMA_BN_regular = tma_tile_shape[
            1
        ] if Self.use_tma_store else Self.WG_BN

        comptime TMA_BN_swapAB = tma_tile_shape[
            0
        ] if Self.use_tma_store else Self.WG_BM

        comptime TMA_BN = TMA_BN_swapAB if Self.swapAB else TMA_BN_regular

        comptime needs_x2_regular = Self.BN % Self.WG_BN != 0
        comptime needs_x2_swapAB = Self.BN % Self.WG_BM != 0

        comptime needs_x2 = needs_x2_swapAB if Self.swapAB else needs_x2_regular

        comptime assert needs_x2 == (
            Self.frag_size % 4 == 0 and Self.frag_size % 8 != 0
        ), (
            "stmatrix and wgmma register count conflict: needs_x2 = "
            + String(needs_x2)
            + " frag_size ="
            + String(Self.frag_size)
        )

        comptime fragment_writer_type[
            sub_wg_id: Int, half_tile: Bool
        ] = FragmentToSMemWriter[
            tile_n_size=TMA_BN,
            num_m_mmas=Self.num_m_mmas,
            num_consumer=Self.num_consumer,
            half_tile=half_tile,
            WG_BM=Self.WG_BM,
            WG_BN=Self.WG_BN,
            sub_wg_id=sub_wg_id,
            swapAB=Self.swapAB,
        ]

        comptime num_column_tiles = ceildiv(Self.BN, Self.WG_BN)
        comptime num_row_tile = ceildiv(Self.BN, Self.WG_BM)

        comptime num_tile = num_column_tiles if not Self.swapAB else num_row_tile
        comptime last_tile = Self.BN // Self.WG_BN if not Self.swapAB else Self.BN // Self.WG_BM

        comptime for tile_idx in range(num_tile):
            comptime is_partial_tile = needs_x2 and tile_idx == last_tile

            # Write fragments to shared memory
            var fragment_writer = fragment_writer_type[
                tile_idx, is_partial_tile
            ](
                self.smem_tile,
                self.warp_group_thread_idx,
                self.local_warp_group_idx,
            )

            comptime for tma_chunk in range(
                (Self.WG_BN if not Self.swapAB else Self.WG_BM) // TMA_BN
            ):
                fragment_writer.write_tile(reg_tile, (0, tma_chunk))

            named_barrier[Int32(Self.num_consumer_threads)](10)

            # swapAB: swap tile shape and position
            comptime tile_rows = Self.WG_BM if Self.swapAB else Self.BM
            comptime tile_cols = Self.WG_BN
            var pos_row = tile_idx if Self.swapAB else 0
            var pos_col = 0 if Self.swapAB else tile_idx

            var workgroup_tile, tile_coords, _ = output_tile.tile_with_offset[
                tile_rows, tile_cols
            ](Coord(pos_row, pos_col))

            var global_coords = rebind[IndexList[2]](tile_coords) + tile_origin

            def apply_epilogue[
                F: ImplicitlyCopyable & RegisterPassable & Self.lambda_type
            ](epilogue_fn: F):
                self._apply_epilogue(
                    epilogue_fn,
                    workgroup_tile,
                    global_coords[0],
                    global_coords[1],
                    max_row,
                    max_col,
                )

            comptime if Self.elementwise_compute_lambda_fn:
                comptime compute_fn = Self.elementwise_compute_lambda_fn.value()

                def _compute[
                    dtype: DType, width: SIMDLength, *, alignment: Int = 1
                ](index: IndexList[2], mut val: SIMD[dtype, width]) {}:
                    val = compute_fn[alignment=alignment](index, val)

                apply_epilogue(_compute)
                named_barrier[Int32(Self.num_consumer_threads)](10)

            comptime if Self.elementwise_lambda_fn:
                comptime epilogue_fn = Self.elementwise_lambda_fn.value()

                def _epilogue[
                    dtype: DType, width: SIMDLength, *, alignment: Int = 1
                ](index: IndexList[2], mut val: SIMD[dtype, width]) {}:
                    _ = epilogue_fn[alignment=alignment](index, val)

                apply_epilogue(_epilogue)
            else:
                comptime if Self.use_tma_store and not is_partial_tile:
                    var tma_writer = TileWriterTMA(Pointer(to=tma_op))

                    if self.local_thread_idx < (Self.WG_BN // TMA_BN):
                        var smem_offset = self.smem_tile._storage + (
                            Self.WG_BM * TMA_BN * self.local_thread_idx
                        )
                        comptime tma_smem_layout = row_major[
                            tma_tile_shape[0], tma_tile_shape[1]
                        ]()
                        var tma_tile = TileTensor[
                            mut=True,
                            Self.dtype,
                            LayoutType=type_of(tma_smem_layout),
                            origin=MutAnyOrigin,
                            address_space=.SHARED,
                        ](smem_offset, tma_smem_layout)

                        var tma_coords = (
                            self.block_x * Self.BN
                            + tile_idx * Self.WG_BN
                            + self.local_thread_idx * TMA_BN,
                            self.block_y * Self.BM,
                        )

                        tma_writer.write_tile(tma_tile, tma_coords)
                else:
                    comptime thread_layout = row_major[
                        Self.num_consumer_threads
                        // (Self.WG_BN // Self.simd_size),
                        Self.WG_BN // Self.simd_size,
                    ]()

                    var threadwise_writer = TileWriterThreadwise[
                        thread_layout=thread_layout,
                        simd_size=Self.simd_size,
                        half_tile=is_partial_tile,
                        swapAB=Self.swapAB,
                    ](
                        workgroup_tile.address_space_cast[.GENERIC](),
                        self.local_thread_idx,
                    )

                    threadwise_writer.write_tile(self.smem_tile, (0, 0))

            named_barrier[Int32(Self.num_consumer_threads)](10)

    @always_inline
    def write_tile[
        tma_rank: Int,
        tma_tile_shape: IndexList[tma_rank],
        tma_desc_shape: IndexList[tma_rank],
        accum_type: DType,
        reg_tile_layout: Layout,
        //,
    ](
        self,
        tma_op: TMATensorTile[
            Self.dtype, tma_rank, tma_tile_shape, tma_desc_shape
        ],
        reg_tile: RegTile[accum_type, reg_tile_layout],
    ):
        """Write output from registers to global memory.

        Selects optimized st.matrix path for bf16 when constraints are met,
        otherwise uses general register-to-global path.

        Parameters:
            tma_rank: Number of dimensions in the TMA tensor descriptor.
            tma_tile_shape: Shape of each TMA store tile per async copy, as
                an index list of length `tma_rank`.
            tma_desc_shape: Full shape of the TMA tensor descriptor as an
                index list of length `tma_rank`.
            accum_type: Data type of the WGMMA accumulator register tile.
            reg_tile_layout: Memory layout of the accumulator register tile.

        Args:
            tma_op: TMA tensor tile descriptor used for async stores from
                shared memory to global memory.
            reg_tile: WGMMA accumulator register tile containing the matmul
                result to write.
        """
        # Output tile dimensions and block coordinates
        # For normal: tile is BM x BN, positioned at (block_y, block_x)
        # For swapAB: tile is BN x BM, positioned at (block_x, block_y)
        comptime tile_m = Self.BM if not Self.swapAB else Self.BN
        comptime tile_n = Self.BN if not Self.swapAB else Self.BM
        var block_row = self.block_y if not Self.swapAB else self.block_x
        var block_col = self.block_x if not Self.swapAB else self.block_y

        var output_tile, tile_origin, _ = self.tensor.tile_with_offset[
            tile_m, tile_n
        ](Coord(block_row, block_col))

        comptime TMA_BN = tma_tile_shape[
            1
        ] if Self.use_tma_store else Self.WG_BN
        comptime row_size_aligned = Self.N * size_of[Self.dtype]() % 16 == 0

        # Check if st.matrix optimization can be used
        # fmt: off
        comptime can_use_stmatrix_normal = (
            accum_type == .float32 and Self.dtype == .bfloat16  # F32→BF16
            and Self.frag_size % 4 == 0                               # Register count
            and Self.BM % Self.wgmma_shape[0] == 0                              # M alignment
            and Self.WG_BN % 16 == 0                                  # Shared memory
            and Self.num_consumer <= 2                                     # Thread limit
            and Self.BN == Self.wgmma_shape[1]                                  # Tile size
            and Self.BM == Self.WG_BM                                      # Block size
            and row_size_aligned                                      # Row alignment
        )

        comptime can_use_stmatrix_swapAB = (
            accum_type == .float32 and Self.dtype == .bfloat16             # F32→BF16
            and Self.frag_size % 4 == 0                                              # Register count (at least stmatrix x2 can be used)
            and Self.BM % Self.wgmma_shape[0] == 0                                   # each consumer should get one wgmma tile
            and Self.WG_BM % 8 == 0                                                  # Shared memory, must have at least 8 rows for swapAB
            and Self.num_consumer <= 2                                               # Thread limit
            and Self.BN == Self.wgmma_shape[1]                                       # Tile size
            and self.BM == Self.WG_BN                                                # Block size (we load by chunks of BM (this checks that this aligns with it))
            and row_size_aligned                                                     # Row alignment
        )

        # fmt: on
        comptime can_use_stmatrix = can_use_stmatrix_swapAB if Self.swapAB else can_use_stmatrix_normal

        comptime if can_use_stmatrix:
            self._write_tile_stmatrix(
                tma_op,
                reg_tile,
                output_tile,
                tile_origin,
            )
        else:
            comptime check_bounds = (
                Self.N % Self.BN != 0
            ) if not Self.swapAB else (Self.N % Self.BM != 0)
            self._write_tile_to_gmem[check_runtime_bounds=check_bounds](
                reg_tile
            )
