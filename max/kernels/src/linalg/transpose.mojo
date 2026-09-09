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
"""The module implements Transpose functions."""

from std.math import ceildiv
from std.sys.info import simd_width_of, size_of
from std.sys.intrinsics import strided_load, strided_store

from std.algorithm import (
    tile,
    vectorize,
)

from max.algorithm import (
    sync_parallelize,
    unsafe_parallel_memcpy,
)
from max.gpu.host import DeviceContext
from layout import TileTensor
from std.memory import unsafe_memcpy
from max.runtime.asyncrt import parallelism_level

from std.utils.index import IndexList, StaticTuple


def _transpose_inplace_4x4[
    rows: Int,
    cols: Int,
    dtype: DType,
](bufloat0: TileTensor[mut=True, dtype, ...]):
    comptime assert rows == 4
    comptime assert cols == 4
    comptime assert bufloat0.flat_rank == 2

    # Contiguous row-major 4x4: row i starts at offset i * 4.
    var row0 = bufloat0.raw_load[width=4](0)
    var row1 = bufloat0.raw_load[width=4](4)
    var row2 = bufloat0.raw_load[width=4](8)
    var row3 = bufloat0.raw_load[width=4](12)

    var tmp0 = row0.shuffle[0, 1, 4, 5](row1)
    var tmp1 = row2.shuffle[0, 1, 4, 5](row3)
    var tmp2 = row0.shuffle[2, 3, 6, 7](row1)
    var tmp3 = row2.shuffle[2, 3, 6, 7](row3)

    var r0 = tmp0.shuffle[0, 2, 4, 6](tmp1)
    var r1 = tmp0.shuffle[1, 3, 5, 7](tmp1)
    var r2 = tmp2.shuffle[0, 2, 4, 6](tmp3)
    var r3 = tmp2.shuffle[1, 3, 5, 7](tmp3)

    bufloat0.raw_store[width=4](0, r0)
    bufloat0.raw_store[width=4](4, r1)
    bufloat0.raw_store[width=4](8, r2)
    bufloat0.raw_store[width=4](12, r3)


def _transpose_inplace_8x8[
    rows: Int,
    cols: Int,
    dtype: DType,
](bufloat0: TileTensor[mut=True, dtype, ...]):
    comptime assert rows == 8
    comptime assert cols == 8
    comptime assert bufloat0.flat_rank == 2

    # Contiguous row-major 8x8: row i starts at offset i * 8.
    var row0 = bufloat0.raw_load[width=8](0)
    var row1 = bufloat0.raw_load[width=8](8)
    var row2 = bufloat0.raw_load[width=8](16)
    var row3 = bufloat0.raw_load[width=8](24)
    var row4 = bufloat0.raw_load[width=8](32)
    var row5 = bufloat0.raw_load[width=8](40)
    var row6 = bufloat0.raw_load[width=8](48)
    var row7 = bufloat0.raw_load[width=8](56)

    @__parameter
    def _apply_permute_0(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 8, 1, 9, 4, 12, 5, 13](other)

    @__parameter
    def _apply_permute_1(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[2, 10, 3, 11, 6, 14, 7, 15](other)

    var k0 = _apply_permute_0(row0, row1)
    var k1 = _apply_permute_1(row0, row1)
    var k2 = _apply_permute_0(row2, row3)
    var k3 = _apply_permute_1(row2, row3)
    var k4 = _apply_permute_0(row4, row5)
    var k5 = _apply_permute_1(row4, row5)
    var k6 = _apply_permute_0(row6, row7)
    var k7 = _apply_permute_1(row6, row7)

    @__parameter
    def _apply_permute_2(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 1, 8, 9, 4, 5, 12, 13](other)

    @__parameter
    def _apply_permute_3(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[2, 3, 10, 11, 6, 7, 14, 15](other)

    var k020 = _apply_permute_2(k0, k2)
    var k021 = _apply_permute_3(k0, k2)
    var k130 = _apply_permute_2(k1, k3)
    var k131 = _apply_permute_3(k1, k3)
    var k460 = _apply_permute_2(k4, k6)
    var k461 = _apply_permute_3(k4, k6)
    var k570 = _apply_permute_2(k5, k7)
    var k571 = _apply_permute_3(k5, k7)

    @__parameter
    def _apply_permute_4(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[0, 1, 2, 3, 8, 9, 10, 11](other)

    @__parameter
    def _apply_permute_5(
        vec: SIMD[dtype, 8], other: SIMD[dtype, 8]
    ) -> SIMD[dtype, 8]:
        return vec.shuffle[4, 5, 6, 7, 12, 13, 14, 15](other)

    var r0 = _apply_permute_4(k020, k460)
    var r1 = _apply_permute_4(k021, k461)
    var r2 = _apply_permute_4(k130, k570)
    var r3 = _apply_permute_4(k131, k571)
    var r4 = _apply_permute_5(k020, k460)
    var r5 = _apply_permute_5(k021, k461)
    var r6 = _apply_permute_5(k130, k570)
    var r7 = _apply_permute_5(k131, k571)

    bufloat0.raw_store[width=8](0, r0)
    bufloat0.raw_store[width=8](8, r1)
    bufloat0.raw_store[width=8](16, r2)
    bufloat0.raw_store[width=8](24, r3)
    bufloat0.raw_store[width=8](32, r4)
    bufloat0.raw_store[width=8](40, r5)
    bufloat0.raw_store[width=8](48, r6)
    bufloat0.raw_store[width=8](56, r7)


def _transpose_inplace_16x16[
    rows: Int,
    cols: Int,
    dtype: DType,
](bufloat0: TileTensor[mut=True, dtype, ...]):
    comptime assert rows == 16
    comptime assert cols == 16
    comptime assert bufloat0.flat_rank == 2

    @__parameter
    def _apply_permute_0(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 16, 1, 17, 4, 20, 5, 21, 8, 24, 9, 25, 12, 28, 13, 29
        ](other)

    @__parameter
    def _apply_permute_1(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            2, 18, 3, 19, 6, 22, 7, 23, 10, 26, 11, 27, 14, 30, 15, 31
        ](other)

    @__parameter
    def _apply_permute_2(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 16, 17, 4, 5, 20, 21, 8, 9, 24, 25, 12, 13, 28, 29
        ](other)

    @__parameter
    def _apply_permute_3(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            2, 3, 18, 19, 6, 7, 22, 23, 10, 11, 26, 27, 14, 15, 30, 31
        ](other)

    @__parameter
    def _apply_permute_4(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
        ](other)

    @__parameter
    def _apply_permute_5(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        ](other)

    @__parameter
    def _apply_permute_6(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27
        ](other)

    @__parameter
    def _apply_permute_7(
        vec: SIMD[dtype, 16], other: SIMD[dtype, 16]
    ) -> SIMD[dtype, 16]:
        return vec.shuffle[
            4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23, 28, 29, 30, 31
        ](other)

    # Contiguous row-major 16x16: row i starts at offset i * 16.
    var row00 = bufloat0.raw_load[width=16](0)
    var row01 = bufloat0.raw_load[width=16](16)
    var row02 = bufloat0.raw_load[width=16](32)
    var row03 = bufloat0.raw_load[width=16](48)
    var row04 = bufloat0.raw_load[width=16](64)
    var row05 = bufloat0.raw_load[width=16](80)
    var row06 = bufloat0.raw_load[width=16](96)
    var row07 = bufloat0.raw_load[width=16](112)
    var row08 = bufloat0.raw_load[width=16](128)
    var row09 = bufloat0.raw_load[width=16](144)
    var row10 = bufloat0.raw_load[width=16](160)
    var row11 = bufloat0.raw_load[width=16](176)
    var row12 = bufloat0.raw_load[width=16](192)
    var row13 = bufloat0.raw_load[width=16](208)
    var row14 = bufloat0.raw_load[width=16](224)
    var row15 = bufloat0.raw_load[width=16](240)

    var k00 = _apply_permute_0(row00, row01)
    var k01 = _apply_permute_1(row00, row01)
    var k02 = _apply_permute_0(row02, row03)
    var k03 = _apply_permute_1(row02, row03)
    var k04 = _apply_permute_0(row04, row05)
    var k05 = _apply_permute_1(row04, row05)
    var k06 = _apply_permute_0(row06, row07)
    var k07 = _apply_permute_1(row06, row07)
    var k08 = _apply_permute_0(row08, row09)
    var k09 = _apply_permute_1(row08, row09)
    var k10 = _apply_permute_0(row10, row11)
    var k11 = _apply_permute_1(row10, row11)
    var k12 = _apply_permute_0(row12, row13)
    var k13 = _apply_permute_1(row12, row13)
    var k14 = _apply_permute_0(row14, row15)
    var k15 = _apply_permute_1(row14, row15)

    var j00 = _apply_permute_2(k00, k02)
    var j01 = _apply_permute_3(k00, k02)
    var j02 = _apply_permute_2(k01, k03)
    var j03 = _apply_permute_3(k01, k03)
    var j04 = _apply_permute_2(k04, k06)
    var j05 = _apply_permute_3(k04, k06)
    var j06 = _apply_permute_2(k05, k07)
    var j07 = _apply_permute_3(k05, k07)
    var j08 = _apply_permute_2(k08, k10)
    var j09 = _apply_permute_3(k08, k10)
    var j10 = _apply_permute_2(k09, k11)
    var j11 = _apply_permute_3(k09, k11)
    var j12 = _apply_permute_2(k12, k14)
    var j13 = _apply_permute_3(k12, k14)
    var j14 = _apply_permute_2(k13, k15)
    var j15 = _apply_permute_3(k13, k15)

    var t00 = _apply_permute_4(j00, j04)
    var t01 = _apply_permute_4(j01, j05)
    var t02 = _apply_permute_4(j02, j06)
    var t03 = _apply_permute_4(j03, j07)
    var t04 = _apply_permute_5(j00, j04)
    var t05 = _apply_permute_5(j01, j05)
    var t06 = _apply_permute_5(j02, j06)
    var t07 = _apply_permute_5(j03, j07)
    var t08 = _apply_permute_4(j08, j12)
    var t09 = _apply_permute_4(j09, j13)
    var t10 = _apply_permute_4(j10, j14)
    var t11 = _apply_permute_4(j11, j15)
    var t12 = _apply_permute_5(j08, j12)
    var t13 = _apply_permute_5(j09, j13)
    var t14 = _apply_permute_5(j10, j14)
    var t15 = _apply_permute_5(j11, j15)

    var r00 = _apply_permute_6(t00, t08)
    var r01 = _apply_permute_6(t01, t09)
    var r02 = _apply_permute_6(t02, t10)
    var r03 = _apply_permute_6(t03, t11)
    var r04 = _apply_permute_6(t04, t12)
    var r05 = _apply_permute_6(t05, t13)
    var r06 = _apply_permute_6(t06, t14)
    var r07 = _apply_permute_6(t07, t15)
    var r08 = _apply_permute_7(t00, t08)
    var r09 = _apply_permute_7(t01, t09)
    var r10 = _apply_permute_7(t02, t10)
    var r11 = _apply_permute_7(t03, t11)
    var r12 = _apply_permute_7(t04, t12)
    var r13 = _apply_permute_7(t05, t13)
    var r14 = _apply_permute_7(t06, t14)
    var r15 = _apply_permute_7(t07, t15)

    bufloat0.raw_store[width=16](0, r00)
    bufloat0.raw_store[width=16](16, r01)
    bufloat0.raw_store[width=16](32, r02)
    bufloat0.raw_store[width=16](48, r03)
    bufloat0.raw_store[width=16](64, r04)
    bufloat0.raw_store[width=16](80, r05)
    bufloat0.raw_store[width=16](96, r06)
    bufloat0.raw_store[width=16](112, r07)
    bufloat0.raw_store[width=16](128, r08)
    bufloat0.raw_store[width=16](144, r09)
    bufloat0.raw_store[width=16](160, r10)
    bufloat0.raw_store[width=16](176, r11)
    bufloat0.raw_store[width=16](192, r12)
    bufloat0.raw_store[width=16](208, r13)
    bufloat0.raw_store[width=16](224, r14)
    bufloat0.raw_store[width=16](240, r15)


def _transpose_inplace_naive[
    rows: Int,
    cols: Int,
    dtype: DType,
](buf: TileTensor[mut=True, dtype, ...]):
    comptime assert buf.flat_rank == 2

    for i in range(rows):
        for j in range(i + 1, cols):
            var tmp = buf[i, j]
            buf[i, j] = buf[j, i]
            buf[j, i] = tmp


def transpose_inplace[
    rows: Int,
    cols: Int,
    dtype: DType,
](buf: TileTensor[mut=True, dtype, ...]):
    """Transposes a square `rows` x `cols` tile in place.

    Dispatches to a specialized SIMD shuffle kernel for 4x4, 8x8, and 16x16
    tiles, falling back to an element-swap loop for other sizes.

    Parameters:
        rows: Number of rows in the tile, equal to `cols`.
        cols: Number of columns in the tile, equal to `rows`.
        dtype: Element type of the tile.

    Args:
        buf: The mutable square tile to transpose in place.
    """
    comptime assert buf.flat_rank == 2
    comptime assert rows == cols
    comptime assert rows == buf.static_shape[0]
    comptime assert cols == buf.static_shape[1]

    comptime if rows == 4:
        _transpose_inplace_4x4[rows, cols, dtype](buf)
    elif rows == 8:
        _transpose_inplace_8x8[rows, cols, dtype](buf)
    elif rows == 16:
        _transpose_inplace_16x16[rows, cols, dtype](buf)
    else:
        _transpose_inplace_naive[rows, cols, dtype](buf)


def _permute_data[
    size: Int,
    dtype: DType,
](
    input: UnsafePointer[mut=False, Scalar[dtype], _],
    output: UnsafePointer[mut=True, Scalar[dtype], _],
    perms: UnsafePointer[mut=False, Int, _],
):
    """
    Ensures that output[i] = input[perms[i]] for i ∈ [0, size)
    """

    comptime for idx in range(size):
        var perm_axis = perms.load(idx)[0]
        var perm_data = input.load(perm_axis)
        output[idx] = perm_data


def _fill_strides[
    dtype: DType,
](
    buf: TileTensor[mut=False, dtype, ...],
    strides: TileTensor[mut=True, .int, ...],
):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    comptime assert buf.rank > 0
    comptime assert strides.rank == 1 and strides.static_shape[0] == buf.rank
    # Provide evidence for flat indexing constraint
    comptime assert strides.flat_rank == 1
    strides[buf.rank - 1] = 1

    comptime for idx in range(buf.rank - 1):
        comptime axis = buf.rank - idx - 2
        var next_axis_stride = strides[axis + 1]
        var next_axis_dim = buf.dim[axis + 1]()
        var curr_axis_stride = next_axis_stride * type_of(next_axis_stride)(
            next_axis_dim
        )
        strides[axis] = curr_axis_stride


# ===------------------------------------------------------------------=== #
# Transpose Permutation simplification
# ===------------------------------------------------------------------=== #
@always_inline
def _collapse_unpermuted_dims[
    rank: Int, tuple_size: Int
](
    mut simplified_shape: IndexList[tuple_size],
    mut simplified_perms: IndexList[tuple_size],
    dim: Int,
):
    var merged_dim = simplified_perms[dim]
    simplified_shape[merged_dim] = (
        simplified_shape[merged_dim] * simplified_shape[merged_dim + 1]
    )

    for j in range(merged_dim + 1, rank - 1):
        simplified_shape[j] = simplified_shape[j + 1]

    for i in range(rank):
        if simplified_perms[i] > merged_dim:
            simplified_perms[i] -= 1
    for k in range(dim + 1, rank - 1):
        simplified_perms[k] = simplified_perms[k + 1]
    simplified_shape[rank - 1] = 0
    simplified_perms[rank - 1] = 0


@always_inline
def _devare_size_1_dim[
    rank: Int, tuple_size: Int
](
    mut simplified_shape: IndexList[tuple_size],
    mut simplified_perms: IndexList[tuple_size],
    dim: Int,
):
    for i in range(dim, rank - 1):
        simplified_shape[i] = simplified_shape[i + 1]

    var found_devared: Bool = False
    for i in range(rank - 1):
        if simplified_perms[i] == dim:
            found_devared = True
        if found_devared:
            simplified_perms[i] = simplified_perms[i + 1]
        if simplified_perms[i] > dim:
            simplified_perms[i] -= 1

    simplified_shape[rank - 1] = 0
    simplified_perms[rank - 1] = 0


@always_inline
def _simplify_transpose_perms_impl[
    rank: Int, tuple_size: Int
](
    mut simplified_rank: Int,
    mut simplified_shape: IndexList[tuple_size],
    mut simplified_perms: IndexList[tuple_size],
):
    comptime if rank < 2:
        return

    else:
        for i in range(rank - 1):
            if simplified_perms[i] + 1 == simplified_perms[i + 1]:
                _collapse_unpermuted_dims[rank](
                    simplified_shape, simplified_perms, i
                )
                simplified_rank -= 1
                _simplify_transpose_perms_impl[rank - 1, tuple_size](
                    simplified_rank, simplified_shape, simplified_perms
                )
                return
            if simplified_shape[i] == 1:
                _devare_size_1_dim[rank](simplified_shape, simplified_perms, i)
                simplified_rank -= 1
                _simplify_transpose_perms_impl[rank - 1, tuple_size](
                    simplified_rank, simplified_shape, simplified_perms
                )
                return


@always_inline
def _simplify_transpose_perms[
    rank: Int
](
    mut simplified_rank: Int,
    mut simplified_shape: IndexList[rank],
    mut simplified_perms: IndexList[rank],
):
    """Simplify the given permutation pattern.

    In some cases a permutation can be modeled by another permutation of a smaller rank.
    For instance, if we have
        shape=[1,3,200,200], perm = [0, 2, 3, 1]
    Then it is equivalent to:
        shape=[1,3,40000], perm = [0, 2, 1]
    Which in its turn is equivalent to:
        shape=[3,40000], perm = [1, 0]

    This function takes the original shape, permutation, and rank by reference,
    and updates their values to simplified ones.
    """
    _simplify_transpose_perms_impl[rank, rank](
        simplified_rank, simplified_shape, simplified_perms
    )


@always_inline
def _convert_transpose_perms_to_static_int_tuple[
    rank: Int
](perms: UnsafePointer[mut=False, Int, _]) -> IndexList[rank]:
    var simplified_perms = IndexList[rank]()
    # TODO: unroll
    for j in range(rank):
        simplified_perms[j] = Int(perms.load(j)[0])
    return simplified_perms


# ===------------------------------------------------------------------=== #
#  Transpose special cases
# ===------------------------------------------------------------------=== #
@always_inline
def _process_tile[
    tile_size_m: Int, tile_size_n: Int, dtype: DType
](
    m: Int,
    n: Int,
    M: Int,
    N: Int,
    out_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    in_ptr: UnsafePointer[Scalar[dtype], _],
):
    var input_tile_offset = M * n + m
    var output_tile_offset = N * m + n

    var input_vals = StaticTuple[SIMD[dtype, tile_size_m], tile_size_n]()
    var output_vals = StaticTuple[SIMD[dtype, tile_size_n], tile_size_m]()

    comptime for i in range(tile_size_n):
        input_vals[i] = in_ptr.load[width=tile_size_m](
            input_tile_offset + M * i
        )

    comptime for m in range(tile_size_m):
        comptime for n in range(tile_size_n):
            output_vals[m][n] = input_vals[n][m]

    comptime for i in range(tile_size_m):
        out_ptr.store(output_tile_offset + N * i, output_vals[i])


def _transpose_2d_serial_tiled[
    rank: Int, dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    offset: Int,
):
    comptime simd_width = simd_width_of[dtype]()

    comptime if rank < 2:
        return
    # The input tile is MxN, the output tile is NxM.
    # We want to do:
    #   output[m, n] = input[n, m]
    # This is equivalent to:
    #   output[n*M + m] = input[m*N + n]
    # And we also have a global offset which needs to be added to both output
    # and input pointers.
    var N = simplified_input_shape[simplified_rank - 2]
    var M = simplified_input_shape[simplified_rank - 1]

    @always_inline
    def process_tile[
        tile_size_m: Int, tile_size_n: Int
    ](m: Int, n: Int) {var N, var M, imm}:
        _process_tile[tile_size_m, tile_size_n, dtype](
            m, n, M, N, output.ptr + offset, input.ptr + offset
        )

    comptime tile_size = simd_width if simd_width <= 16 else 1
    tile[[tile_size, 1], [tile_size, 1]](0, 0, M, N, process_tile)


@always_inline
def _should_run_parallel(
    M: Int, N: Int, simd_width: Int, min_work_per_task: Int
) -> Bool:
    if N == 1:
        return False

    # Check if we can tile the space evenly
    if (N % simd_width) != 0 or (M % simd_width) != 0:
        return False

    var work_per_row = M * simd_width
    if min_work_per_task > work_per_row:
        # We will have to process several rows in each thread
        if (min_work_per_task % work_per_row) != 0:
            return False
        var rows_per_worker = ceildiv(min_work_per_task, work_per_row)
        if N // rows_per_worker < 4:
            return False

    return True


def _transpose_2d_parallel_tiled[
    rank: Int, dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    offset: Int,
    ctx: Optional[DeviceContext] = None,
):
    comptime if rank < 2:
        return

    comptime simd_width = simd_width_of[dtype]()
    var N = simplified_input_shape[simplified_rank - 2]
    var M = simplified_input_shape[simplified_rank - 1]
    comptime min_work_per_task = 1024
    comptime tile_size_m = simd_width if simd_width <= 16 else 1
    comptime tile_size_n = simd_width if simd_width <= 16 else 1

    var n_unit_size = simd_width
    var m_unit_size = simd_width

    var n_tiles = N // n_unit_size
    var m_tiles = M // m_unit_size

    var rows_per_worker = (
        1  # Row in terms of tiles, i.e. we still take simd_width elements
    )
    if min_work_per_task > M * simd_width:
        rows_per_worker = min_work_per_task // (M * simd_width)

    var work = ceildiv(n_tiles, rows_per_worker)

    var num_threads = parallelism_level(ctx)

    var num_tasks = min(work, num_threads)

    var work_block_size = ceildiv(work, num_tasks)

    @always_inline
    def _parallel_tile(
        thread_id: Int,
    ) {var work_block_size, var m_tiles, var N, var M, imm}:
        var n_tile_begin = work_block_size * thread_id
        var n_tile_end = min(work_block_size * (thread_id + 1), work)

        for n_tile in range(n_tile_begin, n_tile_end):
            for m_tile in range(m_tiles):
                var m = tile_size_m * m_tile
                var n = tile_size_n * n_tile
                _process_tile[tile_size_m, tile_size_n, dtype](
                    m,
                    n,
                    M,
                    N,
                    output.ptr + offset,
                    input.ptr + offset,
                )

    sync_parallelize(_parallel_tile, num_tasks, ctx)


def transpose_2d[
    rank: Int, dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    offset: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Transposes the inner two dimensions of a simplified rank-2 tensor.

    Selects a serial tiled implementation or a parallel tiled implementation
    based on the problem size and available parallelism.

    Parameters:
        rank: Number of dimensions in the `simplified_input_shape` array
            (inferred).
        dtype: Element type of the `input` and `output` buffers (inferred).

    Args:
        output: The output buffer with the inner two dimensions swapped.
        input: The input buffer.
        perms: Permutation of the input axes.
        simplified_input_shape: Shape of the tensor after simplification.
        simplified_rank: Effective rank after simplification.
        offset: Flat offset added to both input and output pointers.
        ctx: The context to execute the work on.
    """
    comptime if rank < 2:
        return

    comptime simd_width = simd_width_of[dtype]()
    var N = simplified_input_shape[simplified_rank - 2]
    var M = simplified_input_shape[simplified_rank - 1]
    comptime min_work_per_task = 1024

    if _should_run_parallel(M, N, simd_width, min_work_per_task):
        _transpose_2d_parallel_tiled(
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
            ctx,
        )
    else:
        _transpose_2d_serial_tiled(
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
        )

        return


def _transpose_4d_swap_middle_helper[
    dtype: DType, //
](
    dst_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    src_ptr: UnsafePointer[Scalar[dtype], _],
    L: Int,
    M: Int,
    N: Int,
    K: Int,
    ctx: Optional[DeviceContext] = None,
):
    var work = L * M * N
    var total_size = L * M * N * K

    comptime KB = 1024

    # TODO: These parameters might be tuned
    comptime min_work_per_task = 1 * KB
    comptime min_work_for_parallel = 4 * min_work_per_task

    # TODO: take into account dimension K for parallelization.
    #
    # E.g. if we're transposing 2x3x8192 -> 3x2x8192, then parallelizing just
    # on dimensions M and N is not enough.
    if total_size <= min_work_for_parallel:
        for l in range(L):
            for m in range(M):
                for n in range(N):
                    # We want to do:
                    #   output[l, n, m, k] = input[l, m, n, k]
                    var in_off = l * M * N * K + m * N * K + n * K
                    var out_off = l * M * N * K + n * M * K + m * K
                    unsafe_memcpy(
                        dest=dst_ptr + out_off,
                        src=src_ptr + in_off,
                        count=K,
                    )
        return
    else:
        var num_threads = parallelism_level(ctx)

        var num_tasks = min(work, num_threads)

        var work_block_size = ceildiv(work, num_tasks)

        @always_inline
        def _parallel_copy(thread_id: Int) {var work, var work_block_size, imm}:
            var begin = work_block_size * thread_id
            var end = min(work_block_size * (thread_id + 1), work)
            for block_idx in range(begin, end):
                var l, block_idx_mn = divmod(block_idx, M * N)
                var m, n = divmod(block_idx_mn, N)

                var in_off = l * M * N * K + m * N * K + n * K
                var out_off = l * M * N * K + n * M * K + m * K
                unsafe_memcpy(
                    dest=dst_ptr + out_off,
                    src=src_ptr + in_off,
                    count=K,
                )

        sync_parallelize(_parallel_copy, num_tasks, ctx)


def transpose_4d_swap_middle[
    rank: Int, dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
    ctx: Optional[DeviceContext] = None,
):
    """Transposes the middle two axes of a rank-4 tensor.

    Maps an `LxMxNxK` input to an `LxNxMxK` output by swapping the `M` and `N`
    axes while copying contiguous `K`-sized slices.

    Parameters:
        rank: Number of dimensions in the `simplified_input_shape` array
            (inferred).
        dtype: Element type of the `input` and `output` buffers (inferred).

    Args:
        output: The output buffer with the middle two axes swapped.
        input: The input buffer.
        perms: Permutation of the input axes.
        simplified_input_shape: Shape of the tensor after simplification.
        simplified_rank: Effective rank after simplification.
        ctx: The context to execute the work on.
    """
    comptime if rank < 4:
        return
    # The input tile is LxMxNxK, the output tile is LxNxMxK.
    # We want to do:
    #   output[l, n, m, k] = input[l, m, n, k]
    var L = simplified_input_shape[simplified_rank - 4]
    var M = simplified_input_shape[simplified_rank - 3]
    var N = simplified_input_shape[simplified_rank - 2]
    var K = simplified_input_shape[simplified_rank - 1]
    var src_ptr = input.ptr
    var dst_ptr = output.ptr
    _transpose_4d_swap_middle_helper(dst_ptr, src_ptr, L, M, N, K, ctx)


def transpose_3d_swap_outer[
    rank: Int, dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
):
    """Transposes the outer two axes of a rank-3 tensor.

    Maps an `MxNxK` input to an `NxMxK` output by delegating to the rank-4
    middle-swap helper with an implicit outer dimension of size 1.

    Parameters:
        rank: Number of dimensions in the `simplified_input_shape` array
            (inferred).
        dtype: Element type of the `input` and `output` buffers (inferred).

    Args:
        output: The output buffer with the outer two axes swapped.
        input: The input buffer.
        perms: Permutation of the input axes.
        simplified_input_shape: Shape of the tensor after simplification.
        simplified_rank: Effective rank after simplification.
    """
    comptime if rank < 3:
        return
    # The input tile is MxNxK, the output tile is NxMxK.
    # We want to do:
    #   output[n, m, k] = input[m, n, k]
    # We use a 4d helper function for this, pretending that we have an outer
    # dimensions L=1.
    var M = simplified_input_shape[simplified_rank - 3]
    var N = simplified_input_shape[simplified_rank - 2]
    var K = simplified_input_shape[simplified_rank - 1]
    var src_ptr = input.ptr
    var dst_ptr = output.ptr
    _transpose_4d_swap_middle_helper(dst_ptr, src_ptr, 1, M, N, K)


def transpose_3d_swap_inner[
    rank: Int, dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    simplified_input_shape: IndexList[rank],
    simplified_rank: Int,
):
    """Transposes the inner two axes of a batched rank-3 tensor.

    Iterates over the leading axis and applies the serial tiled 2D transpose
    to each `MxN` slice, advancing the flat offset by the slice size each
    iteration.

    Parameters:
        rank: Number of dimensions in the `simplified_input_shape` array
            (inferred).
        dtype: Element type of the `input` and `output` buffers (inferred).

    Args:
        output: The output buffer with the inner two axes swapped.
        input: The input buffer.
        perms: Permutation of the input axes.
        simplified_input_shape: Shape of the tensor after simplification.
        simplified_rank: Effective rank after simplification.
    """
    comptime if rank < 3:
        return
    # simplified perms must be 0, 2, 1
    var offset = 0
    var step = (
        simplified_input_shape[simplified_rank - 2]
        * simplified_input_shape[simplified_rank - 1]
    )
    # TODO: parallelize this loop
    for _i in range(simplified_input_shape[0]):
        _transpose_2d_serial_tiled(
            output,
            input,
            perms,
            simplified_input_shape,
            simplified_rank,
            offset,
        )
        offset += step


def transpose_trivial_memcpy[
    dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    ctx: Optional[DeviceContext] = None,
):
    """Copies the input buffer to the output buffer as a trivial transpose.

    Uses a single `memcpy` for small transfers and a parallel `memcpy` for
    larger ones.

    Parameters:
        dtype: Element type of the `input` and `output` buffers (inferred).

    Args:
        output: The output buffer.
        input: The input buffer.
        ctx: The context to execute the work on.
    """
    var src_ptr = input.ptr
    var dst_ptr = output.ptr

    comptime KB = 1024
    comptime min_work_per_task = 1 * KB
    comptime min_work_for_parallel = 4 * min_work_per_task

    var total_size = Int(output.num_elements())

    if total_size <= min_work_for_parallel:
        unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=total_size)

    else:
        var work_units = ceildiv(total_size, min_work_per_task)
        var num_tasks = min(work_units, parallelism_level(ctx))
        var work_block_size = ceildiv(work_units, num_tasks)

        unsafe_parallel_memcpy(
            dest=dst_ptr,
            src=src_ptr,
            count=total_size,
            count_per_task=work_block_size * min_work_per_task,
            num_tasks=num_tasks,
        )


# ===------------------------------------------------------------------=== #
#  Transpose generic strided implementation
# ===------------------------------------------------------------------=== #
def _copy_with_strides[
    rank: Int, dtype: DType, //
](
    axis: Int,
    output_ptr: UnsafePointer[mut=True, Scalar[dtype], _],
    output_shape: IndexList[rank],
    output_bytecount: Int,
    input_ptr: UnsafePointer[mut=False, Scalar[dtype], _],
    input_strides: UnsafePointer[mut=False, Int, _],
    output_strides: UnsafePointer[mut=False, Int, _],
    input_offset: Int,
    output_offset: Int,
    ctx: Optional[DeviceContext] = None,
) raises:
    """
    Copy data from `input_ptr` to `output_ptr`, starting at corresponding
    offsets, based on given strides.

    Args:
        axis: The axis value.
        output_ptr: Pointer to the output data.
        output_shape: Shape of the output tensor.
        output_bytecount: Total byte count of the output tensor.
        input_ptr: Pointer to the input data.
        input_strides: The stride at each input axis.
        output_strides: The stride at each output axis.
        input_offset: The offset at which input data starts.
        output_offset: The offset at which output data starts.
        ctx: The context to execute the work on.
    """
    if axis + 1 > rank:
        raise Error("out of range")

    var axis_dim = output_shape[axis]
    var input_axis_stride: Int = Int(input_strides.load(axis)[0])
    var output_axis_stride: Int = Int(output_strides.load(axis)[0])

    if axis + 1 == rank:
        var src_ptr = input_ptr + input_offset
        var dst_ptr = output_ptr + output_offset
        if input_axis_stride == 1 and output_axis_stride == 1:
            unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=axis_dim)
        else:

            @always_inline
            def _copy[
                simd_width: Int
            ](offset: Int) {
                var input_axis_stride,
                var output_axis_stride,
                mut dst_ptr,
                mut src_ptr,
            }:
                strided_store(
                    strided_load[simd_width](src_ptr, input_axis_stride),
                    dst_ptr,
                    output_axis_stride,
                )
                src_ptr = src_ptr + simd_width * input_axis_stride
                dst_ptr = dst_ptr + simd_width * output_axis_stride

            vectorize[simd_width_of[dtype]()](axis_dim, _copy)

        return

    var next_axis = axis + 1

    comptime KB = 1024

    # TODO: These parameters might be tuned
    comptime min_work_per_task = 1 * KB
    comptime min_work_for_parallel = 4 * min_work_per_task

    if output_bytecount <= min_work_for_parallel or axis_dim == 1:
        var next_input_offset = input_offset
        var next_output_offset = output_offset
        for _ in range(axis_dim):
            _copy_with_strides(
                next_axis,
                output_ptr,
                output_shape,
                output_bytecount,
                input_ptr,
                input_strides,
                output_strides,
                next_input_offset,
                next_output_offset,
                ctx,
            )
            next_input_offset += input_axis_stride
            next_output_offset += output_axis_stride

    else:
        var num_threads = parallelism_level(ctx)
        var num_tasks = min(
            ceildiv(output_bytecount, min_work_per_task), num_threads
        )

        var work = axis_dim
        var work_block_size = ceildiv(work, num_tasks)

        @always_inline
        def _parallel_copy(
            thread_id: Int,
        ) raises {
            var work_block_size,
            var work,
            var next_axis,
            var input_axis_stride,
            var output_axis_stride,
            imm,
        }:
            var next_input_offset = (
                thread_id * work_block_size * input_axis_stride + input_offset
            )
            var next_output_offset = (
                thread_id * work_block_size * output_axis_stride + output_offset
            )

            for _ in range(
                work_block_size * thread_id,
                min(work_block_size * (thread_id + 1), work),
            ):
                _copy_with_strides(
                    next_axis,
                    output_ptr,
                    output_shape,
                    output_bytecount,
                    input_ptr,
                    input_strides,
                    output_strides,
                    next_input_offset,
                    next_output_offset,
                    ctx,
                )
                next_input_offset += input_axis_stride
                next_output_offset += output_axis_stride

        # TODO: transpose_strided is using stack allocated structures and
        # so depends on us being synchronous. We need a better way to do this.
        sync_parallelize(_parallel_copy, num_tasks, ctx)


def transpose_strided[
    rank: Int,
    dtype: DType,
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """Transposes a tensor with arbitrary strides via a generic strided copy.

    Computes row-major strides for the input, permutes them according to
    `perms`, and recursively copies data into the row-major output buffer.

    Parameters:
        rank: Number of dimensions in the tensor shape.
        dtype: Element type of the `input` and `output` buffers.

    Args:
        output: The output buffer.
        input: The input buffer.
        perms: Permutation of the input axes.
        ctx: The context to execute the work on.
    """
    # Compute row-major strides for input.
    var input_strides_arr = Array[Int, rank](uninitialized=True)
    input_strides_arr[rank - 1] = 1
    comptime for idx in range(rank - 1):
        comptime axis = rank - idx - 2
        input_strides_arr[axis] = input_strides_arr[axis + 1] * Scalar[
            DType.int
        ](Int(input.dim[axis + 1]()))

    # Permute input strides.
    var permuted_strides_arr = Array[Int, rank](uninitialized=True)
    _permute_data[rank, DType.int](
        input_strides_arr.unsafe_ptr(),
        permuted_strides_arr.unsafe_ptr(),
        perms,
    )

    # Compute row-major strides for output.
    var output_strides_arr = Array[Int, rank](uninitialized=True)
    output_strides_arr[rank - 1] = 1
    comptime for idx in range(rank - 1):
        comptime axis = rank - idx - 2
        output_strides_arr[axis] = output_strides_arr[axis + 1] * Scalar[
            DType.int
        ](Int(output.dim[axis + 1]()))

    # Build output shape for _copy_with_strides.
    var output_shape = IndexList[rank]()
    comptime for i in range(rank):
        output_shape[i] = Int(output.dim[i]())

    var output_bytecount = Int(output.num_elements()) * size_of[dtype]()

    # Kickoff; for intuition on permuted input strides, note that
    #   transpose(output, input, [2, 0, 1])
    # guarantees
    #   output[x, y, z] = input[z, x, y]
    comptime init_axis = 0
    # NOTE: Synchronous, so the stack allocated strides are safe to use.
    _copy_with_strides(
        init_axis,
        output.ptr,
        output_shape,
        output_bytecount,
        input.ptr,
        permuted_strides_arr.unsafe_ptr(),
        output_strides_arr.unsafe_ptr(),
        0,  # input_offset
        0,  # output_offset
        ctx,
    )


# ===------------------------------------------------------------------=== #
#  Transpose entry points
# ===------------------------------------------------------------------=== #
def transpose[
    dtype: DType, //
](
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    input: TileTensor[mut=False, dtype, address_space=.GENERIC, ...],
    perms: UnsafePointer[Int, _],
    ctx: Optional[DeviceContext] = None,
) raises:
    """
    Permute the axis of `input` based on `perms`, and place the result in
    `output`.

    Example:
        ```mojo
        transpose(output, input, [2, 0, 1])
        # guarantees output[x, y, z] = input[z, x, y]
        ```

    Parameters:
        dtype: The dtype of buffer elements.

    Args:
        output: The output buffer.
        input: The input buffer.
        perms: Permutation of the input axes.
        ctx: The context to execute the work on.
    """
    comptime assert (
        output.rank == input.rank
    ), "output and input must have the same rank"
    comptime assert (
        output.flat_rank == output.rank
    ), "output must have a non-nested layout"
    comptime assert (
        input.flat_rank == input.rank
    ), "input must have a non-nested layout"

    comptime rank = output.rank

    # Build the input shape as an IndexList for simplification logic.
    var input_shape = IndexList[rank]()
    comptime for i in range(rank):
        input_shape[i] = Int(input.dim[i]())

    # Try to recognize common special cases in the desired permutation.
    # E.g.
    #   shape=[1,3,200,200], perm = [0, 2, 3, 1]
    # is equivalent to
    #   shape=[1,3,40000], perm = [0, 2, 1]
    #
    # And that just swaps two inner dimensions.
    var simplified_perms = _convert_transpose_perms_to_static_int_tuple[rank](
        perms
    )
    var simplified_shape = input_shape
    var simplified_rank = rank
    _simplify_transpose_perms[rank](
        simplified_rank, simplified_shape, simplified_perms
    )

    if simplified_rank == 1:
        # unsafe_memcpy
        return transpose_trivial_memcpy(output, input, ctx)
    # TODO: Re-enable once #15947 is fixed.
    # elif simplified_rank == 2:
    #     # tiled transpose
    #     return transpose_2d(
    #         output,
    #         input,
    #         perms,
    #         simplified_shape,
    #         simplified_rank,
    #         0,
    #     )
    elif rank >= 3 and simplified_rank == 3:
        if (
            simplified_perms[0] == 0
            and simplified_perms[1] == 2
            and simplified_perms[2] == 1
        ):
            # batched tiled transpose
            return transpose_3d_swap_inner(
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
            )
        elif (
            simplified_perms[0] == 1
            and simplified_perms[1] == 0
            and simplified_perms[2] == 2
        ):
            return transpose_3d_swap_outer(
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
            )
    elif rank >= 4 and simplified_rank == 4:
        if (
            simplified_perms[0] == 0
            and simplified_perms[1] == 2
            and simplified_perms[2] == 1
            and simplified_perms[3] == 3
        ):
            return transpose_4d_swap_middle(
                output,
                input,
                perms,
                simplified_shape,
                simplified_rank,
                ctx,
            )

    # Fall back to the generic strided implementation.
    transpose_strided[rank](output, input, perms, ctx)
