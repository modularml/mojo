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

import std.math
from std.collections.string import StaticString
from std.random import rand
from std.sys import align_of, simd_width_of

import std.benchmark
from std.algorithm import Static2DTileUnitFunc as Tile2DFunc
from std.algorithm import vectorize

from max.algorithm import sync_parallelize
from layout import *
from std.memory import (
    Allocation,
    ThinAllocation,
    alloc,
    dealloc,
    unsafe_memset_zero,
)
from std.python import Python

comptime M = 512  # rows of A and C
comptime N = 4096  # cols of B and C
comptime K = 512  # cols of A and rows of B

comptime dtype = DType.float32


struct Matrix[rows: Int, cols: Int]:
    var data: ThinAllocation[Scalar[dtype]]

    # Initialize zeroeing all values
    def __init__(out self):
        self.data = alloc[Scalar[dtype]](
            {count = Self.rows * Self.cols}
        ).into_thin()
        unsafe_memset_zero(self.data.unsafe_ptr(), Self.rows * Self.cols)

    # Initialize taking a pointer, don't set any elements
    def __init__(out self, var data: Allocation[Scalar[dtype]]):
        assert data.layout().count() == Self.rows * Self.cols
        self.data = data^.into_thin()

    def __deinit__(deinit self):
        dealloc(self.data^.unsafe_with_layout({count = Self.rows * Self.cols}))

    ## Initialize with random values
    @staticmethod
    def rand() -> Self:
        var data = alloc[Scalar[dtype]]({count = Self.rows * Self.cols})
        rand(data.unsafe_span())
        return Self(data^)

    def __getitem__(self, y: Int, x: Int) -> Scalar[dtype]:
        return self.load(y, x)

    def __setitem__(mut self, y: Int, x: Int, val: Scalar[dtype]):
        self.store(y, x, val)

    def load[nelts: Int = 1](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.unsafe_ptr().unsafe_load[width=nelts](
            y * self.cols + x
        )

    def store[
        nelts: Int = 1
    ](mut self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.unsafe_ptr().unsafe_store(y * self.cols + x, val)


def matmul_naive(mut C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Perform 2D tiling on the iteration space defined by end_x and end_y
def tile[
    tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int, tiled_fn: Some[Tile2DFunc]):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Unroll the vectorized loop by a constant factor
def matmul_unrolled(mut C: Matrix, A: Matrix, B: Matrix):
    # simdwidth of = amount of `dtype` elements that fit into a single SIMD register
    # 2x multiplier will use multiple SIMD registers in parallel where possible
    comptime nelts = simd_width_of[dtype]() * 2
    comptime tile_m = 8  # M must be a multiple of this
    comptime tile_n = 64  # N must be a multiple of this
    comptime tile_k = 4  # K must be a multiple of this

    comptime assert M % tile_m == 0, "M must be a multiple of tile_m"
    comptime assert N % tile_n == 0, "N must be a multiple of tile_n"
    comptime assert K % tile_k == 0, "K must be a multiple of tile_k"

    def calc_row(m0: Int) {mut C, imm}:
        for m in range(tile_m * m0, tile_m * m0 + tile_m):
            _ = m  # FIXME: param closures not noticing access.

            def calc_tile[
                tile_x: Int, tile_y: Int
            ](x: Int, y: Int) {mut C, imm}:
                comptime for _k in range(tile_y):
                    var k = _k + y
                    var A_val = A[m, k]

                    def dot[
                        simd_size: Int
                    ](n: Int) {x, mut C, mut A_val, imm B, imm m, mut k}:
                        var idx = n + x
                        C.store(
                            m,
                            idx,
                            C.load[simd_size](m, idx)
                            + A_val * B.load[simd_size](k, idx),
                        )

                    comptime unroll_factor = tile_x // nelts
                    vectorize[
                        nelts,
                        size=tile_x,
                        unroll_factor=unroll_factor,
                    ](dot)

            tile[tile_n, tile_k](C.cols, B.rows, calc_tile)

    sync_parallelize(calc_row, C.rows // tile_m)


def matmul_tiled_layout(mut C: Matrix, A: Matrix, B: Matrix):
    var dst = LayoutTensor[dtype, Layout.row_major(M, N)](
        C.data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var lhs = LayoutTensor[dtype, Layout.row_major(M, K)](
        A.data.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var rhs = LayoutTensor[dtype, Layout.row_major(K, N)](
        B.data.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )

    comptime vec_size = simd_width_of[dtype]() * 2

    comptime tile_m = 2
    comptime tile_n = 64
    comptime tile_k = 4

    comptime assert M % tile_m == 0, "N must be a multiple of tile_m"
    comptime assert N % tile_n == 0, "N must be a multiple of tile_n"
    comptime assert K % tile_k == 0, "K must be a multiple of tile_k"

    def calc_row(m_1: Int) {imm}:
        for k_1 in range(K // tile_k):
            for n_1 in range(N // tile_n):
                var lhs_view = lhs.tile[tile_m, tile_k](m_1, k_1)
                var dst_view = dst.tile[tile_m, tile_n](m_1, n_1)
                var rhs_view = rhs.tile[tile_k, tile_n](k_1, n_1)

                comptime for m in range(tile_m):
                    comptime for k in range(tile_k):
                        var lhs_val = rebind[Scalar[dtype]](lhs_view[m, k])

                        def dot[simd_size: Int](n: Int) {mut}:
                            comptime assert (
                                type_of(dst_view).layout.stride[1] == 1
                            ), "elements of dst should be contiguous"
                            comptime assert (
                                type_of(rhs_view).layout.stride[1] == 1
                            ), "elements of rhs should be contiguous"

                            dst_view.store[simd_size](
                                m,
                                n,
                                dst_view.load[simd_size](m, n)
                                + lhs_val * rhs_view.load[simd_size](k, n),
                            )

                        comptime unroll_factor = tile_n // vec_size
                        vectorize[
                            vec_size,
                            size=tile_n,
                            unroll_factor=unroll_factor,
                        ](dot)

    sync_parallelize(calc_row, M // tile_m)


def matmul_tiled_layout_cache(mut C: Matrix, A: Matrix, B: Matrix):
    var dst = LayoutTensor[dtype, Layout.row_major(M, N)](
        C.data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var lhs = LayoutTensor[dtype, Layout.row_major(M, K)](
        A.data.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var rhs = LayoutTensor[dtype, Layout.row_major(K, N)](
        B.data.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )

    comptime vec_size = simd_width_of[dtype]() * 2

    comptime tile_m = 8
    comptime tile_n = 64
    comptime tile_k = 4

    comptime assert M % tile_m == 0, "N must be a multiple of tile_m"
    comptime assert N % tile_n == 0, "N must be a multiple of tile_n"
    comptime assert K % tile_k == 0, "K must be a multiple of tile_k"

    def calc_row(m_1: Int) {imm}:
        var rhs_cache = LayoutTensor[
            dtype, Layout.row_major(tile_k, tile_n), MutAnyOrigin
        ].stack_allocation()

        for k_1 in range(K // tile_k):
            for n_1 in range(N // tile_n):
                var lhs_view = lhs.tile[tile_m, tile_k](m_1, k_1)
                var dst_view = dst.tile[tile_m, tile_n](m_1, n_1)
                var rhs_view = rhs.tile[tile_k, tile_n](k_1, n_1)

                rhs_cache.copy_from(rhs_view)

                comptime for m in range(tile_m):
                    comptime for k in range(tile_k):
                        var lhs_val = rebind[Scalar[dtype]](lhs_view[m, k])

                        def dot[simd_size: Int](n: Int) {mut}:
                            comptime assert (
                                type_of(dst_view).layout.stride[1] == 1
                            ), "elements of dst should be contiguous"

                            dst_view.store[simd_size](
                                m,
                                n,
                                dst_view.load[simd_size](m, n)
                                + lhs_val
                                * rhs_cache.aligned_load[simd_size](k, n),
                            )

                        comptime unroll_factor = tile_n // vec_size
                        vectorize[
                            vec_size,
                            size=tile_n,
                            unroll_factor=unroll_factor,
                        ](dot)

    sync_parallelize(calc_row, M // tile_m)


def matmul_layout_transposed(mut C: Matrix, A: Matrix, B: Matrix):
    var dst = LayoutTensor[dtype, Layout.row_major(M, N)](
        C.data.unsafe_ptr().unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var lhs = LayoutTensor[dtype, Layout.row_major(M, K)](
        A.data.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )
    var rhs = LayoutTensor[dtype, Layout.row_major(K, N)](
        B.data.unsafe_ptr()
        .unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
    )

    comptime vec_size = 4 * simd_width_of[dtype]()

    comptime tile_m = 16
    comptime tile_n = 16
    comptime tile_k = 128

    comptime assert M % tile_m == 0, "N must be a multiple of tile_m"
    comptime assert N % tile_n == 0, "N must be a multiple of tile_n"
    comptime assert K % tile_k == 0, "K must be a multiple of tile_k"

    comptime assert (
        tile_k % vec_size == 0
    ), "tile_k must be a multiple of vec_size"

    def calc_row(m_1: Int) {imm}:
        var rhs_cache = LayoutTensor[
            dtype, Layout.row_major(tile_n, tile_k), MutAnyOrigin
        ].stack_allocation()
        var lhs_cache = LayoutTensor[
            dtype, Layout.row_major(tile_m, tile_k), MutAnyOrigin
        ].stack_allocation()

        for k_1 in range(K // tile_k):
            var lhs_view = lhs.tile[tile_m, tile_k](m_1, k_1)
            lhs_cache.copy_from(lhs_view)

            for n_1 in range(N // tile_n):
                var dst_view = dst.tile[tile_m, tile_n](m_1, n_1)
                var rhs_view = rhs.tile[tile_k, tile_n](k_1, n_1).transpose()
                rhs_cache.copy_from(rhs_view)

                for var m in range(tile_m):
                    for var n in range(tile_n):
                        var sum = SIMD[dtype, vec_size](0)

                        def dot[simd_size: Int](k: Int) {mut}:
                            sum = std.math.fma(
                                lhs_cache.load[vec_size](m, k),
                                rhs_cache.aligned_load[vec_size](n, k),
                                sum,
                            )

                        comptime unroll_factor = tile_k // vec_size
                        vectorize[
                            vec_size,
                            size=tile_k,
                            unroll_factor=unroll_factor,
                        ](dot)

                        dst_view[m, n] += sum.reduce_add()

    sync_parallelize(calc_row, M // tile_m)


@always_inline
def bench[
    func: def(mut Matrix, Matrix, Matrix) thin -> None, name: StaticString
]() raises:
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    @always_inline
    def test_fn() {mut C, imm A, imm B}:
        _ = func(C, A, B)

    var secs = std.benchmark.run(test_fn, max_runtime_secs=0.5).mean()

    _ = A^
    _ = B^
    _ = C^

    var gflops = ((2 * M * N * K) / secs) / 1e9

    var py = Python.import_module("builtins")
    _ = py.print(py.str("{:<13}{:>8.3f} GFLOPS").format(name, gflops))


@always_inline
def test_matrix_equal[
    func: def(mut Matrix, Matrix, Matrix) thin -> None
](mut C: Matrix, A: Matrix, B: Matrix) raises -> Bool:
    """Runs a matmul function on A and B and tests the result for equality with
    C on every element.
    """
    var result = Matrix[M, N]()
    _ = func(result, A, B)
    for i in range(C.rows):
        for j in range(C.cols):
            # if C[i, j] != result[i, j]:
            if abs(C[i, j] - result[i, j]) > 1e-3:
                return False
    return True


def test_all() raises:
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    matmul_naive(C, A, B)

    if not test_matrix_equal[matmul_unrolled](C, A, B):
        raise Error("Unroll output does not match naive implementation")
    if not test_matrix_equal[matmul_tiled_layout](C, A, B):
        raise Error(
            "Layout Tiled Parallel Vectorized output does not match naive"
            " implementation"
        )
    if not test_matrix_equal[matmul_tiled_layout_cache](C, A, B):
        raise Error(
            "Layout Tiled Parallel Vectorized output does not match naive"
            " implementation"
        )
    if not test_matrix_equal[matmul_layout_transposed](C, A, B):
        raise Error(
            "Layout Transposed output does not match naive implementation"
        )


def main() raises:
    test_all()
    print("CPU Results\n")

    bench[matmul_naive, "Naive:"]()
    bench[matmul_unrolled, "Unrolled:"]()
    bench[matmul_tiled_layout, "LayoutTensor:"]()
    bench[matmul_tiled_layout_cache, "LayoutTensor Cached:"]()
    bench[matmul_layout_transposed, "LayoutTensor Transposed:"]()
