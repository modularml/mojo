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

"""Implements the ARM I8MM (8-bit integer matrix multiply) CPU microkernel.

Provides `Inner_matmul_i8mm`, an `InnerMatmulKernel` conforming struct that
performs 8-bit integer matrix multiplication using the `_neon_matmul` NEON
intrinsic. It accumulates `int32` or `uint32` results from `uint8`/`uint8`,
`uint8`/`int8`, or `int8`/`int8` operand pairs (i8mm does not support a
signed-`A`, unsigned-`B` pair) through `LoadStore_i8mm`.
"""

from std.math import align_up
from std.sys import prefetch
from std.sys.info import align_of
from std.sys.intrinsics import PrefetchOptions

from linalg.utils import partial_simd_load, partial_simd_store
from layout import Coord, Idx, TileTensor

from std.utils.index import Index, IndexList

from ...accumulate import _Accumulator
from ...arch.cpu.neon_intrinsics import _neon_matmul
from ...utils import GemmShape, get_matmul_prefetch_b_distance_k
from .impl import InnerMatmulKernel


struct LoadStore_i8mm[
    dtype: DType,
    simd_size: Int,
    single_row: Bool,
    tile_rows: Int,
    tile_columns: Int,
]:
    """Handles C-tile load and store operations for the I8MM microkernel.

    Manages a local accumulator tile of shape `(tile_rows, tile_columns //
    simd_size)` and provides helpers to initialize it, load an existing C
    sub-tile from memory, and store results back with optional boundary checks.

    Parameters:
        dtype: Accumulator data type.
        simd_size: SIMD lane count (must be 4 for I8MM).
        single_row: Whether the tile has only one effective row (M=1 path).
        tile_rows: Number of accumulator rows (half of kernel_rows for I8MM pairs).
        tile_columns: Number of accumulator columns (kernel_cols).
    """

    comptime num_simd_cols = Self.tile_columns // Self.simd_size
    """Number of SIMD-width column groups in the output tile."""
    var output_tile: _Accumulator[
        Self.dtype, Self.tile_rows, Self.num_simd_cols, Self.simd_size
    ]
    """Accumulation buffer holding the partial sums for the output tile."""
    var skip_boundary_check: Bool
    """Whether to skip partial-tile boundary handling on load and store."""

    @always_inline
    def __init__(out self, skip_boundary_check: Bool):
        """Initializes the tile buffer with a boundary-check setting.

        Args:
            skip_boundary_check: Whether to skip partial-tile boundary
                handling on load and store.
        """
        self.output_tile = _Accumulator[
            Self.dtype, Self.tile_rows, Self.num_simd_cols, Self.simd_size
        ]()
        self.skip_boundary_check = skip_boundary_check

    @always_inline
    def _initialize_c_tile(mut self):
        self.output_tile.init(0)

    @always_inline
    def _load_c_tile(
        mut self,
        c_ptr: UnsafePointer[Scalar[Self.dtype], ...],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
    ):
        var c_ptr_loc = c_ptr + tile_n_idx

        comptime for idx0 in range(Self.tile_rows):
            comptime for idx1 in range(Self.tile_columns // Self.simd_size):
                var c_data: SIMD[Self.dtype, Self.simd_size] = 0
                if self.skip_boundary_check or (
                    idx1 * 2 + 2 <= c_bound[1] - tile_n_idx
                ):
                    var t0 = c_ptr_loc.load[width=2](
                        c_stride * (2 * idx0) + 2 * idx1
                    )
                    var t1 = c_ptr_loc.load[width=2](
                        c_stride * (2 * idx0 + 1) + 2 * idx1
                    ) if not Self.single_row else SIMD[Self.dtype, 2](0)
                    c_data = rebind[SIMD[Self.dtype, Self.simd_size]](
                        t0.join(t1)
                    )
                elif idx1 * 2 <= c_bound[1]:
                    var t0 = partial_simd_load[2](
                        c_ptr_loc + (c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    )
                    var t1 = partial_simd_load[2](
                        c_ptr_loc + (c_stride * (2 * idx0 + 1) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        0,
                    ) if not Self.single_row else SIMD[Self.dtype, 2](0)
                    c_data = rebind[SIMD[Self.dtype, Self.simd_size]](
                        t0.join(t1)
                    )

                self.output_tile[idx0, idx1] = c_data

    @always_inline
    def _store_c_tile(
        mut self,
        c_ptr: UnsafePointer[mut=True, Scalar[Self.dtype], ...],
        c_stride: Int,
        tile_n_idx: Int,
        c_bound: IndexList[2],
    ):
        var c_ptr_loc = c_ptr + tile_n_idx

        comptime for idx0 in range(Self.tile_rows):
            comptime for idx1 in range(Self.tile_columns // Self.simd_size):
                var c_data = self.output_tile[idx0, idx1]
                if self.skip_boundary_check or (
                    idx1 * 2 + 2 <= c_bound[1] - tile_n_idx
                ):
                    (c_ptr_loc + (c_stride * (2 * idx0 + 0) + 2 * idx1)).store(
                        c_data.slice[2](),
                    )

                    comptime if not Self.single_row:
                        (
                            c_ptr_loc + (c_stride * (2 * idx0 + 1) + 2 * idx1)
                        ).store(
                            c_data.slice[2, offset=2](),
                        )
                elif idx1 * 2 <= c_bound[1]:
                    partial_simd_store(
                        c_ptr_loc + (c_stride * (2 * idx0 + 0) + 2 * idx1),
                        0,
                        c_bound[1] - tile_n_idx - idx1 * 2,
                        c_data.slice[2](),
                    )

                    comptime if not Self.single_row:
                        partial_simd_store(
                            c_ptr_loc + (c_stride * (2 * idx0 + 1) + 2 * idx1),
                            0,
                            c_bound[1] - tile_n_idx - idx1 * 2,
                            c_data.slice[2, offset=2](),
                        )


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the I8MM microkernel.
@fieldwise_init
struct Inner_matmul_i8mm(InnerMatmulKernel, Movable):
    """ARM I8MM (8-bit integer matrix multiply) microkernel for CPU matmul.

    Implements `InnerMatmulKernel` using the `_neon_matmul` intrinsic to
    compute 8-bit integer dot products in pairs of two rows. Operates on a
    pre-packed A buffer (packed by `packA_i8mm`) and a packed B tile, and
    accumulates `int32` or `uint32` results from `uint8`/`uint8`,
    `uint8`/`int8`, or `int8`/`int8` operands (not signed-`A`, unsigned-`B`)
    in `LoadStore_i8mm`.
    """

    # Parameters for global reference.

    @always_inline
    def _accumulate[
        simd_size: Int, kernel_rows: Int, kernel_cols: Int
    ](
        self,
        a: TileTensor,
        b_packed: TileTensor,
        mut c_local: _Accumulator[
            _, kernel_rows, kernel_cols // simd_size, simd_size
        ],
        global_offset: GemmShape,
        tile_n_k_idx: IndexList[2],
    ):
        """Utility function on the inner loop. Launch one tile of fma on the
        local accumulation buffer while processing a single column of A.

        Args:
            a: Input A matrix tile being processed.
            b_packed: Packed B matrix tile in cache-friendly layout.
            c_local: Pre-allocated local buffer for c partial sums.
            global_offset: Global (M, N, K) coordinate offset for this tile.
            tile_n_k_idx: Index tuple with (n, k) coordinates within the current
                processing tile to index the packed B matrix.
        """
        comptime assert b_packed.flat_rank == 3, "b_packed must be rank 3"

        var n_outer_idx = tile_n_k_idx[0] // (kernel_cols // 2)
        var kl = tile_n_k_idx[1]

        var b_ptr = b_packed.ptr_at_offset(Coord(n_outer_idx, kl // 8, Idx[0]))

        # This inner kernels works with non-transposed A.
        var K = Int(a.dim[1]())
        var a_ptr = a.ptr + (global_offset.M * K + 2 * global_offset.K + 2 * kl)

        # Prefetch B matrix.
        comptime prefetch_distance = get_matmul_prefetch_b_distance_k()
        comptime assert simd_size == 4

        comptime if prefetch_distance > 0:
            comptime prefetch_offset = prefetch_distance * kernel_cols

            comptime for idx in range(kernel_cols // simd_size):
                prefetch[
                    PrefetchOptions().for_read().high_locality().to_data_cache()
                ](b_ptr + (prefetch_offset + idx * simd_size))

        # Loop over local accumulator tiles.
        comptime for idx0 in range(kernel_rows):
            comptime for idx1 in range(kernel_cols // simd_size):
                comptime alignment = align_of[SIMD[c_local.dtype, simd_size]]()
                var a_val = a_ptr.load[width=SIMDLength(simd_size) * 4](
                    2 * idx0 * K
                )
                var b_val = (b_ptr + 16 * idx1).load[
                    width=SIMDLength(simd_size) * 4, alignment=alignment
                ]()
                var c_val = c_local[idx0, idx1]
                c_val = _neon_matmul(c_val, a_val, b_val)
                c_local[idx0, idx1] = c_val

    @always_inline
    def __inner_matmul__[
        kernel_rows: Int,
        kernel_cols: Int,
        simd_size: Int,
    ](
        self,
        c: TileTensor[mut=True, ...],
        a: TileTensor,
        b_packed: TileTensor,
        global_offset: GemmShape,
        global_bound: GemmShape,
        tile_n_k: IndexList[2],
        skip_boundary_check: Bool,
    ):
        """Utility function on the inner loop. Run the inner kernel on the whole
        (kernel_rows2, TileN, TileK) tile.

        Parameters:
            kernel_rows: Number of rows in the inner kernel tile. Halved
                internally for I8MM row pairing unless equal to 1.
            kernel_cols: Number of columns in the inner kernel tile, also
                the N-dimension step size.
            simd_size: SIMD lane count for the I8MM dot-product intrinsic
                (must be 4).

        Args:
            c: Output C matrix tile where accumulated results are stored.
            a: Input A matrix tile in pre-packed layout, read non-transposed.
            b_packed: Packed B matrix tile in cache-friendly rank-3 layout.
            global_offset: Global (M, N, K) coordinate offset of this tile
                within the full matmul problem space.
            global_bound: Global (M, N, K) upper bound of the full matmul
                problem space, used for boundary checking.
            tile_n_k: Index list with the (N, K) range to process within
                this tile.
            skip_boundary_check: Whether to skip boundary checks when loading
                and storing the C tile.
        """
        comptime assert b_packed.flat_rank == 3, "b_packed must be rank 3"

        comptime kernel_rows2 = kernel_rows // 2 if kernel_rows != 1 else kernel_rows
        comptime single_row = (kernel_rows == 1)

        var c_stride = Int(c.dim[1]())

        var c_ptr = c.ptr + (global_offset.M * c_stride + global_offset.N)

        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        var acc = LoadStore_i8mm[
            c.dtype,
            simd_size,
            single_row,
            kernel_rows2,
            kernel_cols,
        ](skip_boundary_check)

        for idx_n in range(0, tile_n_k[0], kernel_cols // 2):
            if global_offset.K == 0:
                acc._initialize_c_tile()
            else:
                acc._load_c_tile(
                    c_ptr,
                    c_stride,
                    idx_n,
                    c_bound,
                )
            var kl = align_up(tile_n_k[1], 8)
            for idx_k in range(0, kl, 8):
                self._accumulate[simd_size, kernel_rows2, kernel_cols](
                    a,
                    b_packed,
                    acc.output_tile,
                    global_offset,
                    Index(idx_n, idx_k),
                )
            acc._store_c_tile(
                c_ptr,
                c_stride,
                idx_n,
                c_bound,
            )
