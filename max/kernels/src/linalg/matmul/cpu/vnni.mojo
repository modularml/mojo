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

"""Implements the integer CPU matmul microkernel (x86 VNNI or ARM NEON).

On x86 it accumulates `int32` from a `uint8` `A` and an `int8` `B`; on ARM it
accumulates `int32` from `int8`/`int8` or `uint8`/`uint8` operands.
"""

from std.math import align_down
from std.sys import prefetch
from std.sys.info import CompilationTarget, align_of
from std.sys.intrinsics import PrefetchOptions

from linalg.utils import partial_simd_load
from layout import Coord, Idx, TileTensor, stack_allocation
from layout.tile_layout import row_major
from std.memory.unsafe import bitcast

from std.utils.index import Index, IndexList

from ...accumulate import _Accumulator
from ...arch.cpu.neon_intrinsics import _neon_dotprod
from ...arch.cpu.vnni_intrinsics import (
    dot_i8_to_i32_saturated_x86,
    dot_i8_to_i32_x86,
)
from ...utils import GemmShape, get_matmul_prefetch_b_distance_k
from .impl import InnerMatmulKernel


# Define a struct that conforms to the InnerMatmulKernel trait that
# implements the VNNI microkernel.
@fieldwise_init
struct Inner_matmul_vnni[saturated_vnni: Bool](InnerMatmulKernel, Movable):
    """Integer CPU matmul microkernel using x86 VNNI or ARM NEON dot-product.

    Implements `InnerMatmulKernel` using 4-element 8-bit integer dot-product
    instructions to accumulate `int32` partial products. On x86 it accumulates
    from a `uint8` `A` and an `int8` `B`, emitting the VNNI `vpdpbusd`
    instruction when the target has VNNI (AVX-512 VNNI or AVX-VNNI) and falling
    back to an AVX2 emulation otherwise (AVX2 is the minimum, not AVX-512). On
    ARM it accumulates from `int8`/`int8` or `uint8`/`uint8` operands using the
    NEON dot-product (`_neon_dotprod`). Handles K tails (remainder < 4) by
    packing A into a local buffer or using AVX-512 masked loads.

    Parameters:
        saturated_vnni: When `True`, uses the saturating x86 variant
            (`dot_i8_to_i32_saturated_x86`), which requires the `A` operand to
            lie in `[0, 127]`. This saves instructions on the AVX2 emulation
            path used when the target lacks VNNI; on VNNI hardware it emits the
            same instruction as the default.
    """

    # Parameters for global reference.

    @always_inline
    def _accumulate[
        is_tail: Bool,
        simd_size: Int,
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
        tile_n_k: IndexList[2],
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
            tile_n_k: Dynamic tile sizes along the N and K dimensions.
        """
        comptime assert b_packed.flat_rank == 3, "b_packed must be rank 3"

        comptime c_type = c_local.dtype
        # Seek outer indices in packed layout.
        var n_outer_idx = tile_n_k_idx[0] // kernel_cols
        var kl = tile_n_k_idx[1]

        # Global K index.
        var global_k = global_offset.K + kl

        var b_ptr = b_packed.ptr_at_offset(
            Coord(n_outer_idx, kl // 4, Idx[0])
        ).bitcast[Scalar[c_type]]()

        comptime if not is_tail:
            # Prefetch B matrix.
            comptime prefetch_distance = get_matmul_prefetch_b_distance_k()

            comptime if prefetch_distance > 0:
                comptime prefetch_offset = prefetch_distance * kernel_cols

                comptime for idx in range(kernel_cols // simd_size):
                    prefetch[
                        PrefetchOptions()
                        .for_read()
                        .high_locality()
                        .to_data_cache()
                    ](b_ptr + (prefetch_offset + idx * simd_size))

        # This inner kernels works with non-transposed A.
        var K = Int(a.dim[1]())

        # Stack allocation for local A buffer
        var a_local = stack_allocation[
            dtype=a.dtype, address_space=a.address_space
        ](row_major[kernel_rows * 4]())
        var a_base_ptr = a.ptr + (global_offset.M * K + global_k)
        var a_ptr = a_local.ptr if (
            is_tail
            and not CompilationTarget.has_avx512f()
            # This origin cast is not ideal since we give up
            # exclusivity checking, but it is safe in the sense that
            # `a` will be guaranteed to remain alive because
            # it is an argument to the function.
        ) else a_base_ptr.unsafe_mut_cast[True]().as_unsafe_any_origin()
        var a_ptr_stride = 4 if (
            is_tail and not CompilationTarget.has_avx512f()
        ) else K

        var tail_length = tile_n_k[1] - kl

        # pack A if (tile_n_k_idx[1] - kl) is 1, 2, or 3
        comptime if is_tail and not CompilationTarget.has_avx512f():
            for idx0 in range(kernel_rows):
                for idx_k in range(tail_length):
                    a_local[4 * idx0 + idx_k] = a_base_ptr[idx0 * K + idx_k]

        # Loop over local accumulator tiles.
        comptime for idx0 in range(kernel_rows):
            comptime for idx1 in range(kernel_cols // simd_size):
                # width K bytes or K/4 ints, a_ptr is pointer to ints
                var a_val = (
                    bitcast[c_type, 1](
                        partial_simd_load[4](
                            a_ptr + idx0 * a_ptr_stride, 0, tail_length, 0
                        )
                    ) if (is_tail and CompilationTarget.has_avx512f()) else (
                        a_ptr + idx0 * a_ptr_stride
                    )
                    .bitcast[Scalar[c_type]]()
                    .load()
                )

                comptime alignment = align_of[SIMD[c_type, simd_size]]()
                # var c_idx = Index(idx0, idx1 * simd_size)
                var c_val = c_local[idx0, idx1]
                var b_val = (b_ptr + idx1 * simd_size).load[
                    width=simd_size, alignment=alignment
                ]()

                comptime if CompilationTarget.has_neon_int8_dotprod():
                    var a_val2 = SIMD[c_type, simd_size](a_val)
                    c_val = _neon_dotprod(
                        c_val,
                        bitcast[a.dtype, SIMDLength(simd_size) * 4](a_val2),
                        bitcast[b_packed.dtype, SIMDLength(simd_size) * 4](
                            b_val
                        ),
                    )
                elif Self.saturated_vnni:
                    c_val = dot_i8_to_i32_saturated_x86[simd_size](
                        c_val, a_val, b_val
                    )
                else:
                    c_val = dot_i8_to_i32_x86[simd_size](c_val, a_val, b_val)
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
        (kernel_rows, TileN, TileK) tile.

        Parameters:
            kernel_rows: Number of rows processed per inner kernel iteration
                along the M dimension.
            kernel_cols: Number of columns processed per inner kernel
                iteration along the N dimension.
            simd_size: SIMD vector width used for the int8 dot-product
                instructions.

        Args:
            c: Output matrix tile accumulating the matmul partial sums.
            a: Input A matrix tile being processed.
            b_packed: Packed B matrix tile in cache-friendly layout.
            global_offset: Global (M, N, K) coordinate offset for this tile.
            global_bound: Global (M, N, K) bound of the matrices, used for
                boundary checks.
            tile_n_k: Dynamic tile sizes along the N and K dimensions.
            skip_boundary_check: Whether to skip boundary checks when storing
                results.
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
                    skip_boundary_check,
                )

            # Iterate on tile K dimension.
            # Not unrolled on K path.
            var kl = align_down(tile_n_k[1], 4)
            for idx_k in range(0, kl, 4):
                # accumulate data for this (n, k) index
                self._accumulate[False, simd_size, kernel_rows, kernel_cols](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, idx_k),
                    tile_n_k,
                )
            if kl != tile_n_k[1]:
                self._accumulate[True, simd_size, kernel_rows, kernel_cols](
                    a,
                    b_packed,
                    acc,
                    global_offset,
                    Index(idx_n, kl),
                    tile_n_k,
                )
            acc.store(
                rebind[UnsafePointer[Scalar[c.dtype], MutAnyOrigin]](c_ptr),
                c_stride,
                idx_n,
                c_bound,
                skip_boundary_check,
            )
