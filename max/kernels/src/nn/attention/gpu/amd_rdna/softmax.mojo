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
"""Online softmax for RDNA Wave32 attention kernels.

Warp lane layout is `col_major(16, 2)` (lane_row = l % 16,
lane_col = l // 16). Per-lane C/D fragment is `row_major(1, 8)`: 8
fp32 elements stored as a row vector. `full()` runs one online-softmax
iteration end-to-end (max → exp → sum → correction → output update).
"""

import max.gpu.primitives.warp as warp
from std.math.uutils import umod
from max.gpu import lane_id, warp_id as get_warp_id
from max.gpu.sync import barrier
from layout import TileTensor
from layout.tile_layout import col_major, row_major
from layout.tile_tensor import stack_allocation as tt_stack_allocation
from nn.softmax import _exp2_concrete, _exp_concrete


struct SoftmaxRDNA[
    dtype: DType,
    num_m_mmas: Int,
    num_n_mmas: Int,
    num_warps_m: Int,
    num_warps_n: Int,
    use_exp2: Bool = False,
]:
    """Online softmax accumulator for RDNA Wave32 attention kernels.

    Tracks per-row running maximum and sum across online softmax iterations
    so that attention scores can be normalized incrementally as new key/value
    tiles arrive. The `full()` method runs one complete iteration (max → exp →
    sum → correction → output update) end-to-end.

    Parameters:
        dtype: Element type of the score and output fragments.
        num_m_mmas: Number of MMA tiles along the query (row) dimension.
        num_n_mmas: Number of MMA tiles along the key (column) dimension.
        num_warps_m: Number of warps assigned to the row dimension.
        num_warps_n: Number of warps assigned to the column dimension.
        use_exp2: Selects `exp2` instead of natural `exp` for the softmax kernel.
    """

    comptime _warp_rows = 16
    comptime _warp_cols = 2
    comptime WarpLayoutT = type_of(
        col_major[Self._warp_rows, Self._warp_cols]()
    )

    comptime _frag_size = 8
    comptime FragmentLayoutT = type_of(row_major[1, Self._frag_size]())

    comptime warp_rows = Self.WarpLayoutT.static_shape[0]
    comptime warp_cols = Self.WarpLayoutT.static_shape[1]

    comptime num_rowwise_lanes = UInt32(Self.warp_cols)
    comptime num_colwise_lanes = UInt32(Self.warp_rows)
    # Stride between adjacent lane-columns (= stride[1] of col-major warp layout).
    comptime rowwise_lanes_stride = UInt32(Self.WarpLayoutT.static_stride[1])

    comptime exp_function = _exp2_concrete if Self.use_exp2 else _exp_concrete
    comptime num_colwise_warps = Self.num_warps_m
    comptime num_rowwise_warps = Self.num_warps_n

    comptime frag_num_rows = Self.FragmentLayoutT.static_shape[0]  # = 1
    comptime frag_size = Self.FragmentLayoutT.static_product  # = 8
    comptime frag_is_row_vector = Self.frag_num_rows == 1

    comptime num_colwise_tiles = Self.num_m_mmas
    comptime num_rowwise_tiles = Self.num_n_mmas

    comptime row_layout = row_major[Self.num_m_mmas, Self.frag_num_rows]()
    comptime RowMaxTensorType = TileTensor[
        Self.dtype,
        type_of(Self.row_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]
    comptime RowSumTensorType = Self.RowMaxTensorType

    var rowmax_tensor: Self.RowMaxTensorType
    var rowsum_tensor: Self.RowSumTensorType

    comptime score_frag_layout = row_major[
        Self.num_colwise_tiles, Self.frag_num_rows
    ]()
    comptime ScoreFragTensorType = TileTensor[
        Self.dtype,
        type_of(Self.score_frag_layout),
        MutUntrackedOrigin,
        address_space=.LOCAL,
    ]

    var score_frag_rowmax: Self.ScoreFragTensorType
    var score_frag_rowsum: Self.ScoreFragTensorType
    var correction: Self.ScoreFragTensorType

    @always_inline
    def __init__(out self):
        self.rowmax_tensor = tt_stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.row_layout)
        self.rowsum_tensor = tt_stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.row_layout)
        self.score_frag_rowmax = tt_stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.score_frag_layout)
        self.score_frag_rowsum = tt_stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.score_frag_layout).fill(0)
        self.correction = tt_stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.score_frag_layout).fill(1)

    @always_inline
    def _score_row_idx[col_tile: Int, row: Int](self, lane_row: Int) -> UInt32:
        """Map (col_tile, lane_row, row) to a linear score matrix row index."""
        return (
            UInt32(col_tile)
            * Self.num_colwise_lanes
            * UInt32(Self.frag_num_rows)
            + UInt32(lane_row * Self.frag_num_rows)
            + UInt32(row)
        )

    @always_inline
    def _reduce_rows[
        is_max: Bool
    ](
        self,
        score: TileTensor[Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        """Reduce score rows to per-thread scalars (max or sum).

        Vectorize the score tile, reduce within each warp via lane
        shuffles, then (if rows span multiple warps) communicate via
        shared memory to produce a single value per row.
        """
        comptime assert score.flat_rank == 2
        comptime assert warp_scratch.flat_rank == 2
        var score_reg_tile = score.vectorize[1, Self.frag_size]()
        comptime assert score_reg_tile.flat_rank == 2

        # Init accumulator.
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                comptime if is_max:
                    self.score_frag_rowmax[col_tile, row] = self.rowmax_tensor[
                        col_tile, row
                    ]
                else:
                    self.score_frag_rowsum[col_tile, row] = 0

        var warp_x = umod(get_warp_id[broadcast=True](), Self.num_rowwise_warps)
        var lane_coord = Self.WarpLayoutT().idx2crd(Int(lane_id()))
        var lane_row = Int(lane_coord[0].value())
        var lane_col = Int(lane_coord[1].value())
        var lane_contains_first_column = lane_col == 0

        # Per-thread fragment reduction + warp lane shuffle.
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(Self.num_rowwise_tiles):
                comptime tile_id = (
                    col_tile + row_tile * Self.num_colwise_tiles
                )
                var frag = score_reg_tile[tile_id, 0]

                comptime assert Self.frag_is_row_vector
                comptime for row in range(Self.frag_num_rows):
                    comptime for col in range(Self.frag_size):
                        comptime if is_max:
                            self.score_frag_rowmax[col_tile, row] = max(
                                self.score_frag_rowmax[col_tile, row],
                                frag[col],
                            )
                        else:
                            self.score_frag_rowsum[col_tile, row] += frag[col]

            comptime for row in range(Self.frag_num_rows):
                comptime if is_max:
                    self.score_frag_rowmax[col_tile, row] = warp.lane_group_max[
                        Int(Self.num_rowwise_lanes),
                        stride=Int(Self.rowwise_lanes_stride),
                    ](self.score_frag_rowmax[col_tile, row])
                else:
                    self.score_frag_rowsum[col_tile, row] = warp.lane_group_sum[
                        Int(Self.num_rowwise_lanes),
                        stride=Int(Self.rowwise_lanes_stride),
                    ](self.score_frag_rowsum[col_tile, row])

        # Cross-warp reduce via shared memory.
        comptime if Self.num_rowwise_warps > 1:
            comptime smem_row_offset = 0 if is_max else Self.num_rowwise_warps

            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var sri = self._score_row_idx[col_tile, row](lane_row)
                        comptime if is_max:
                            warp_scratch[
                                warp_x, Int(sri)
                            ] = self.score_frag_rowmax[col_tile, row][0]
                        else:
                            warp_scratch[
                                warp_x + smem_row_offset, Int(sri)
                            ] = self.score_frag_rowsum[col_tile, row][0]

            barrier()

            if lane_contains_first_column:
                comptime for col_tile in range(Self.num_colwise_tiles):
                    comptime for row in range(Self.frag_num_rows):
                        var sri = self._score_row_idx[col_tile, row](lane_row)

                        comptime if is_max:
                            comptime for rw in range(Self.num_rowwise_warps):
                                self.score_frag_rowmax[col_tile, row] = max(
                                    rebind[Scalar[Self.dtype]](
                                        self.score_frag_rowmax[col_tile, row]
                                    ),
                                    rebind[Scalar[Self.dtype]](
                                        warp_scratch[Int(rw), Int(sri)]
                                    ),
                                )
                        else:
                            self.score_frag_rowsum[col_tile, row] = 0
                            comptime for rw in range(Self.num_rowwise_warps):
                                self.score_frag_rowsum[col_tile, row] += rebind[
                                    Scalar[Self.dtype]
                                ](
                                    warp_scratch[
                                        Int(rw) + smem_row_offset, Int(sri)
                                    ]
                                )

            # Broadcast reduced value to all lanes in the row.
            comptime for col_tile in range(Self.num_colwise_tiles):
                comptime for row in range(Self.frag_num_rows):
                    comptime if is_max:
                        self.score_frag_rowmax[
                            col_tile, row
                        ] = warp.lane_group_max[
                            Int(Self.num_rowwise_lanes),
                            stride=Int(Self.rowwise_lanes_stride),
                        ](
                            self.score_frag_rowmax[col_tile, row]
                        )
                    else:
                        # lane_group_max acts as broadcast (all lanes hold
                        # the same value after reduction).
                        self.score_frag_rowsum[
                            col_tile, row
                        ] = warp.lane_group_max[
                            Int(Self.num_rowwise_lanes),
                            stride=Int(Self.rowwise_lanes_stride),
                        ](
                            self.score_frag_rowsum[col_tile, row]
                        )

    @always_inline
    def calculate_qk_max(
        self,
        score: TileTensor[Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        self._reduce_rows[is_max=True](score, warp_scratch)

    @always_inline
    def calculate_qk_sum(
        self,
        score: TileTensor[Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        self._reduce_rows[is_max=False](score, warp_scratch)

    @always_inline
    def exp(self, score: TileTensor[mut=True, Self.dtype, ...]):
        comptime assert score.flat_rank == 2
        comptime assert Self.frag_is_row_vector
        var score_reg_tile = score.vectorize[1, Self.frag_size]()
        comptime assert score_reg_tile.flat_rank == 2

        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(Self.num_rowwise_tiles):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile

                score_reg_tile[tile_id, 0] = Self.exp_function(
                    score_reg_tile[tile_id, 0]
                    - SIMD[Self.dtype, Self.frag_size](
                        self.score_frag_rowmax[col_tile, 0][0]
                    )
                )

    @always_inline
    def calculate_correction(self):
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.correction[col_tile, row] = Self.exp_function(
                    self.rowmax_tensor[col_tile, row]
                    - self.score_frag_rowmax[col_tile, row]
                )

    @always_inline
    def update_output(self, output: TileTensor[mut=True, Self.dtype, ...]):
        comptime assert output.flat_rank == 2
        comptime assert Self.frag_is_row_vector
        var output_reg_tile = output.vectorize[1, Self.frag_size]()
        comptime assert output_reg_tile.flat_rank == 2
        comptime num_output_tiles = output_reg_tile.static_shape[0]

        # Apply per-row correction to all output tiles. The correction is
        # indexed by col_tile (the m_mma dimension), which cycles with period
        # num_colwise_tiles across tile_id.
        comptime for tile_id in range(num_output_tiles):
            comptime col_tile = tile_id % Self.num_colwise_tiles
            output_reg_tile[tile_id, 0] = output_reg_tile[tile_id, 0] * SIMD[
                Self.dtype, Self.frag_size
            ](self.correction[col_tile, 0][0])

    @always_inline
    def update_sum(self):
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
                    * self.correction[col_tile, row]
                    + self.score_frag_rowsum[col_tile, row]
                )

    @always_inline
    def update_max(self):
        comptime for i in range(Self.num_colwise_tiles):
            comptime for j in range(Self.frag_num_rows):
                self.rowmax_tensor[i, j] = self.score_frag_rowmax[i, j]

    @always_inline
    def full(
        self,
        output: TileTensor[mut=True, Self.dtype, ...],
        score: TileTensor[mut=True, Self.dtype, ...],
        warp_scratch: TileTensor[mut=True, Self.dtype, ...],
    ):
        """Single-pass online softmax iteration: max -> exp -> sum ->
        correction -> update output -> update max/sum.

        Args:
            output: Accumulator tile of partial softmax outputs to correct
                in place with the per-row renormalization factor.
            score: Attention score tile for this iteration; overwritten in
                place with exponentiated, max-subtracted values.
            warp_scratch: Shared-memory scratch tile used for cross-warp row
                reductions of the running maximum and sum.
        """
        self.calculate_qk_max(score, warp_scratch)
        self.exp(score)
        self.calculate_qk_sum(score, warp_scratch)
        self.calculate_correction()
        self.update_output(output)
        self.update_max()
        self.update_sum()
