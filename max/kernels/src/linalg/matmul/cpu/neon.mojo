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

"""Implements the ARM NEON lane-FMA microkernel for CPU matmul.

Defines `Inner_matmul_neon`, a struct conforming to the `InnerMatmulKernel`
trait that accumulates partial products using vectorized lane-broadcast FMA
against a packed B matrix. The dispatcher selects this kernel on ARM NEON
hardware for operand types that take neither the integer dot-product
(`dotprod`) nor the `i8mm` path, which in practice means floating-point
operands.
"""

from std.math import fma

from layout import Coord, Idx, TileTensor

from std.utils.index import Index, IndexList

from ...accumulate import _Accumulator
from ...utils import GemmShape
from .impl import InnerMatmulKernel


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the Neon microkernel.
@fieldwise_init
struct Inner_matmul_neon(InnerMatmulKernel, Movable):
    """ARM NEON lane-FMA microkernel for CPU matmul.

    Implements `InnerMatmulKernel` using vectorized lane-broadcast FMA to
    accumulate partial products. Loads `simd_size` elements of A, broadcasts
    each lane across all kernel rows, and computes dot products against packed
    B columns. Handles tail elements in the K dimension with scalar cleanup.
    """

    @always_inline
    def _accumulate_lane[
        simd_size: Int,
        a_col_size: Int,
        kernel_rows: Int,
        kernel_cols: Int,
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

        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // kernel_cols

        # Global K index.
        var global_k = global_offset.K + tile_n_k_idx[1]

        var b_ptr = b_packed.ptr_at_offset(
            Coord(n_outer_idx, tile_n_k_idx[1], Idx[0])
        )

        var a_vals = Array[_, kernel_rows](
            fill_with_unrolled=lambda [row: Int]() -> SIMD[
                c_local.dtype, a_col_size
            ]: a.load[width=a_col_size](
                Coord(global_offset.M + row, global_k)
            ).cast[
                c_local.dtype
            ]()
        )

        comptime for lane in range(a_col_size):
            comptime for col in range(kernel_cols // simd_size):
                var b_val = (
                    (b_ptr + col * simd_size)
                    .load[width=simd_size]()
                    .cast[c_local.dtype]()
                )

                comptime for row in range(kernel_rows):
                    var a_val = a_vals[row]
                    var c_val = c_local[row, col]
                    c_val = fma[dtype=c_local.dtype, width=simd_size](
                        a_val[lane], b_val, c_val
                    )
                    c_local[row, col] = c_val

            b_ptr = b_ptr + kernel_cols

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
        (kernel_rows, TileN, TileK) tile.

        Parameters:
            kernel_rows: Number of rows in the M-dimension tile processed per
                invocation.
            kernel_cols: Number of columns in the N-dimension tile processed
                per invocation; must be a multiple of `simd_size`.
            simd_size: SIMD vector width used for lane-broadcast FMA on packed
                B elements.

        Args:
            c: Output C matrix tile accumulating the partial products.
            a: Input A matrix tile read in row-major order.
            b_packed: Packed B matrix tile in cache-friendly layout; must be
                rank 3.
            global_offset: Global (M, N, K) coordinate offset of this tile
                within the whole matmul problem space.
            global_bound: Global (M, N, K) extent of the full matmul problem
                space, used for boundary clipping on the C store.
            tile_n_k: (n, k) extents of the tile to process in the N and K
                dimensions.
            skip_boundary_check: Whether the tile lies fully within
                `global_bound` so boundary clipping can be skipped.
        """
        comptime assert b_packed.flat_rank == 3, "b_packed must be rank 3"

        var c_stride = Int(c.dim[1]())

        var c_ptr = c.ptr + (global_offset.M * c_stride + global_offset.N)
        var c_bound = Index(global_bound.M, global_bound.N) - Index(
            global_offset.M, global_offset.N
        )

        var acc = _Accumulator[
            c.dtype, kernel_rows, kernel_cols // simd_size, simd_size
        ]()

        for idx_n in range(0, tile_n_k[0], kernel_cols):
            # Initialize accumulation buffer
            #  either zero filling or load existing value.
            if global_offset.K == 0:
                acc.init(0)
            else:
                acc.load(
                    rebind[UnsafePointer[Scalar[c.dtype], MutAnyOrigin]](c_ptr),
                    c_stride,
                    idx_n,
                    c_bound,
                )

            var partition_end = simd_size * (tile_n_k[1] // simd_size)
            for idx_k0 in range(0, partition_end, simd_size):
                self._accumulate_lane[
                    simd_size, simd_size, kernel_rows, kernel_cols
                ](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k0),
                )

            for idx_k1 in range(partition_end, tile_n_k[1]):
                # accumulate data for this (n, k) index
                self._accumulate_lane[simd_size, 1, kernel_rows, kernel_cols](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k1),
                )
            acc.store(
                rebind[UnsafePointer[Scalar[c.dtype], MutAnyOrigin]](c_ptr),
                c_stride,
                idx_n,
                c_bound,
            )
