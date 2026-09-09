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
"""Online softmax for gfx950 attention kernels.

Score, warp_scratch, and output are TileTensors. Warp + fragment geometry is
expressed via TileLayout / `Coord`: `WarpLayoutT` is the col-major
(warp_rows, warp_cols) lane layout, and `FragmentLayoutT` describes the
per-lane MMA fragment shape. All lane decomposition / stride queries go
through `Layout` methods rather than hand-rolled integer arithmetic.
"""

import max.gpu.primitives.warp as warp
from std.math import fma
from std.math.uutils import umod
from std.bit import log2_floor
from max.gpu import lane_id, warp_id as get_warp_id
from max.gpu.sync import barrier
from layout import TileTensor, row_major, stack_allocation
from layout.tile_layout import TensorLayout, col_major
from nn.softmax import _exp2_concrete, _exp_concrete


struct Softmax[
    dtype: DType,
    # Score tile is tiled by MMA units of shape (num_m_mmas, num_n_mmas).
    num_m_mmas: Int,
    num_n_mmas: Int,
    # Block's warp grid: num_warps_m colwise × num_warps_n rowwise.
    num_warps_m: Int,
    num_warps_n: Int,
    # MMA M extent (32 or 16 on gfx950) — selects lane + fragment geometry.
    mma_m: Int,
    use_exp2: Bool = False,
]:
    """Maintains online-softmax state and operations for gfx950 attention kernels.

    Keeps per-thread running row-max and row-sum statistics across MMA tiles so
    that attention can accumulate softmax numerators and rescale previously
    accumulated outputs as new score tiles arrive. Lane and fragment geometry is
    derived from `WarpLayoutT` and `FragmentLayoutT`.

    Parameters:
        dtype: Element type of the score and output tiles.
        num_m_mmas: Number of MMA unit tiles along the M (colwise)
            dimension of the score matrix.
        num_n_mmas: Number of MMA unit tiles along the N (rowwise)
            dimension of the score matrix.
        num_warps_m: Number of warps along the colwise (M) dimension of
            the block's warp grid.
        num_warps_n: Number of warps along the rowwise (N) dimension of
            the block's warp grid.
        mma_m: MMA M extent, 32 or 16 on gfx950, selecting lane and
            fragment geometry.
        use_exp2: Use `exp2` instead of `exp` for the exponential
            (defaults to False).
    """

    # Warp lane layout (col-major (warp_rows, warp_cols)) as a proper
    # TileLayout. `static_shape` / `static_stride` expose geometry; `idx2crd`
    # decomposes a lane index into (lane_row, lane_col).
    comptime _warp_rows = 32 if Self.mma_m == 32 else 16
    comptime _warp_cols = 2 if Self.mma_m == 32 else 4
    comptime WarpLayoutT = type_of(
        col_major[Self._warp_rows, Self._warp_cols]()
    )

    # Per-lane fragment layout (flat `(1, regs_per_lane)`, row-major). 32x32
    # MFMA produces 16 regs/lane, 16x16 produces 4.
    comptime _frag_size = 16 if Self.mma_m == 32 else 4
    comptime FragmentLayoutT = type_of(col_major[1, Self._frag_size]())

    comptime warp_rows = Self.WarpLayoutT.static_shape[0]
    comptime warp_cols = Self.WarpLayoutT.static_shape[1]

    comptime num_shuffles_per_row = log2_floor(Self.warp_cols)
    comptime num_rowwise_lanes = UInt32(Self.warp_cols)
    comptime num_colwise_lanes = UInt32(Self.warp_rows)
    # Stride between adjacent lane-columns (= stride[1] of col-major warp layout).
    comptime rowwise_lanes_stride = UInt32(Self.WarpLayoutT.static_stride[1])

    comptime exp_function = _exp2_concrete if Self.use_exp2 else _exp_concrete
    comptime num_colwise_warps = Self.num_warps_m
    comptime num_rowwise_warps = Self.num_warps_n

    # Per-lane fragment geometry. For a flattened fragment layout, the first
    # dim is rows and the second is (possibly nested) cols; static_shape[0]
    # gives the row count, static_product gives total regs per lane.
    comptime frag_num_rows = Self.FragmentLayoutT.static_shape[0]
    comptime frag_size = Self.FragmentLayoutT.static_product
    comptime frag_is_row_vector = Self.frag_num_rows == 1

    # Number of mma unit tiles in the score matrix.
    comptime num_colwise_tiles = Self.num_m_mmas
    comptime num_rowwise_tiles = Self.num_n_mmas
    # The online softmax attributes for each thread's elements (fragments).
    comptime num_rows_per_thread = Self.num_colwise_tiles * Self.frag_num_rows

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
        self.rowmax_tensor = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.row_layout)
        self.rowsum_tensor = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.row_layout)
        self.score_frag_rowmax = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.score_frag_layout)
        self.score_frag_rowsum = stack_allocation[
            dtype=Self.dtype, address_space=.LOCAL
        ](Self.score_frag_layout).fill(0)
        self.correction = stack_allocation[
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

        Vectorizes the score tile, reduces within each warp via lane
        shuffles, then (if rows span multiple warps) communicates via
        shared memory to produce a single value per row.

        Parameters:
            is_max: True for rowwise max, False for rowwise sum.

        Args:
            score: Score tile in registers.
            warp_scratch: Shared memory scratch for cross-warp reduce.
        """
        comptime assert score.flat_rank == 2
        comptime assert warp_scratch.flat_rank == 2
        var score_reg_tile = score.vectorize[1, Self.frag_size]()
        comptime assert score_reg_tile.flat_rank == 2

        # Init accumulator: copy current max (for max) or zero (for sum).
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                comptime if is_max:
                    self.score_frag_rowmax[col_tile, row] = self.rowmax_tensor[
                        col_tile, row
                    ]
                else:
                    self.score_frag_rowsum[col_tile, row] = 0

        var warp_x = umod(get_warp_id[broadcast=True](), Self.num_rowwise_warps)
        # Decompose lane via the warp layout: col-major (warp_rows, warp_cols).
        var lane_coord = Self.WarpLayoutT().idx2crd(Int(lane_id()))
        var lane_row = Int(lane_coord[0].value())
        var lane_col = Int(lane_coord[1].value())
        var lane_contains_first_column = lane_col == 0

        # Per-thread fragment reduction + warp-level lane shuffle.
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(Self.num_rowwise_tiles):
                comptime tile_id = (
                    col_tile + row_tile * Self.num_colwise_tiles
                )
                var frag = score_reg_tile[tile_id, 0]

                # gfx950 MFMA fragments are always row-vectors (shape[0]=1);
                # the outer loop iterates once with row=0, then `col` walks
                # the frag_size register lanes contiguously.
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

        # Cross-warp reduce via shared memory (if rows span multiple warps).
        comptime if Self.num_rowwise_warps > 1:
            # SMEM offset: max uses rows [0, N), sum uses rows [N, 2N).
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
    def exp[
        start: Int = 0, stride: Int = 1
    ](self, score: TileTensor[mut=True, Self.dtype, ...]):
        # gfx950 MFMA fragments are always row-vectors (shape[0]=1).
        comptime assert score.flat_rank == 2
        comptime assert Self.frag_is_row_vector
        var score_reg_tile = score.vectorize[1, Self.frag_size]()
        comptime assert score_reg_tile.flat_rank == 2

        comptime for col_tile in range(Self.num_colwise_tiles):
            # Softmax numerator based on mma results.
            comptime for row_tile in range(
                start, Self.num_rowwise_tiles, stride
            ):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile

                score_reg_tile[tile_id, 0] = Self.exp_function(
                    score_reg_tile[tile_id, 0]
                    - SIMD[Self.dtype, Self.frag_size](
                        self.score_frag_rowmax[col_tile, 0][0]
                    )
                )

    @always_inline
    def scale_rowmax(self, scale: Scalar[Self.dtype]):
        """Scale score_frag_rowmax by scale factor (e.g. scale * log2e).

        Must be called after exp_scaled so that score_frag_rowmax is in the
        same units as rowmax_tensor for calculate_correction.

        Args:
            scale: Scalar multiplier applied to every per-thread row-max
                value.
        """
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.score_frag_rowmax[col_tile, row] *= scale

    @always_inline
    def exp_scaled[
        start: Int = 0, stride: Int = 1
    ](
        self,
        score: TileTensor[mut=True, Self.dtype, ...],
        scale: Scalar[Self.dtype],
    ):
        """Numerically stable scaled exp: exp2((score - max) * scale).

        Subtracts the unscaled max before scaling, so the subtraction is exact
        for the maximum element (IEEE 754 guarantees a - a == 0). This avoids
        the precision gap in exp_fma where fma(score, scale, -scaled_max) can
        produce nonzero results when score == max due to independent rounding
        of scaled_max.

        Parameters:
            start: Starting row-tile index of the iteration range
                (defaults to 0).
            stride: Step between consecutive row-tile indices
                (defaults to 1).

        Args:
            score: Score tile in registers to exponentiate in place.
            scale: Scale factor applied to `(score - max)` before the
                exponential; use `scale * log2e` for `exp2`-based softmax.
        """
        # gfx950 MFMA fragments are always row-vectors (shape[0]=1).
        comptime assert score.flat_rank == 2
        comptime assert Self.frag_is_row_vector
        var score_reg_tile = score.vectorize[1, Self.frag_size]()
        comptime assert score_reg_tile.flat_rank == 2

        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row_tile in range(
                start, Self.num_rowwise_tiles, stride
            ):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile

                var neg_max = SIMD[Self.dtype, Self.frag_size](
                    -self.score_frag_rowmax[col_tile, 0][0]
                )
                var scale_vec = SIMD[Self.dtype, Self.frag_size](scale)
                score_reg_tile[tile_id, 0] = Self.exp_function(
                    (score_reg_tile[tile_id, 0] + neg_max) * scale_vec
                )

    @always_inline
    def exp_pkfma[
        start: Int = 0, stride: Int = 1
    ](
        self,
        score: TileTensor[mut=True, Self.dtype, ...],
        scale: Scalar[Self.dtype],
    ):
        """Scaled exp using fused `score * scale + (-max * scale)` form
        so the compiler emits `v_pk_fma_f32` instead of separate add+mul.

        Mirrors `exp_scaled` but pre-multiplies `-max` by `scale` once per row
        (1 packed mul per row) and uses `fma(score, scale, neg_scaled_max)`
        for every score pair (matches aiter's softmax inner loop).

        Precision: `score == max` produces `fma(max, scale, -scaled_max)`
        where `scaled_max` is pre-rounded; the FMA's internal `max*scale` may
        round to a slightly different value, yielding a tiny epsilon instead
        of exactly 0. `exp2(epsilon) ≈ 1 + epsilon·ln(2)`, well within the
        FP8 softmax tolerance budget (the row-sum normalization absorbs it).

        Parameters:
            start: Starting row-tile index of the iteration range
                (defaults to 0).
            stride: Step between consecutive row-tile indices
                (defaults to 1).

        Args:
            score: Score tile in registers to exponentiate in place.
            scale: Scale factor applied to `score` and `-max` inside the
                fused multiply-add; use `scale * log2e` for `exp2`-based
                softmax.
        """
        comptime assert score.flat_rank == 2
        comptime assert Self.frag_is_row_vector
        var score_reg_tile = score.vectorize[1, Self.frag_size]()
        comptime assert score_reg_tile.flat_rank == 2

        var scale_vec = SIMD[Self.dtype, Self.frag_size](scale)

        comptime for col_tile in range(Self.num_colwise_tiles):
            var neg_scaled_max = SIMD[Self.dtype, Self.frag_size](
                -self.score_frag_rowmax[col_tile, 0][0] * scale
            )
            comptime for row_tile in range(
                start, Self.num_rowwise_tiles, stride
            ):
                comptime tile_id = col_tile + Self.num_colwise_tiles * row_tile

                score_reg_tile[tile_id, 0] = Self.exp_function(
                    fma(score_reg_tile[tile_id, 0], scale_vec, neg_scaled_max)
                )

    @always_inline
    def calculate_correction(self):
        comptime for col_tile in range(Self.num_colwise_tiles):
            # Correction since previous max may be updated.
            comptime for row in range(Self.frag_num_rows):
                self.correction[col_tile, row] = Self.exp_function(
                    self.rowmax_tensor[col_tile, row]
                    - self.score_frag_rowmax[col_tile, row]
                )

    @always_inline
    def update_output(self, output: TileTensor[mut=True, Self.dtype, ...]):
        # gfx950 MFMA fragments are always row-vectors (shape[0]=1).
        comptime assert output.flat_rank == 2
        comptime assert Self.frag_is_row_vector
        var output_reg_tile = output.vectorize[1, Self.frag_size]()
        comptime assert output_reg_tile.flat_rank == 2
        comptime num_output_tiles = output_reg_tile.static_shape[0]

        # Apply per-row correction to all output tiles. The correction is
        # indexed by col_tile (the m_mma dimension), which cycles with period
        # num_colwise_tiles across tile_id. The flat loop handles both MHA
        # (num_output_tiles == num_colwise*num_rowwise) and MLA decode
        # (output_depth != depth => extra replications along N).
        comptime for tile_id in range(num_output_tiles):
            comptime col_tile = tile_id % Self.num_colwise_tiles

            output_reg_tile[tile_id, 0] = output_reg_tile[tile_id, 0] * SIMD[
                Self.dtype, Self.frag_size
            ](self.correction[col_tile, 0][0])

    @always_inline
    def update_sum(self):
        # Save current rowmax and rowsum
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
                    * self.correction[col_tile, row]
                    + self.score_frag_rowsum[col_tile, row]
                )

    @always_inline
    def apply_sum_correction(self):
        """Apply rowsum *= correction (deferred sum rescale pattern)."""
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
                    * self.correction[col_tile, row]
                )

    @always_inline
    def update_sum_additive(self):
        """Additive rowsum update: rowsum += new_sum (no correction)."""
        comptime for col_tile in range(Self.num_colwise_tiles):
            comptime for row in range(Self.frag_num_rows):
                self.rowsum_tensor[col_tile, row] = (
                    self.rowsum_tensor[col_tile, row]
                    + self.score_frag_rowsum[col_tile, row]
                )

    @always_inline
    def update_max(self):
        # Save current rowmax and rowsum
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
        self.calculate_qk_max(score, warp_scratch)
        self.exp(score)
        self.calculate_qk_sum(score, warp_scratch)
        self.calculate_correction()
        self.update_output(output)
        self.update_max()
        self.update_sum()
