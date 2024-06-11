# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
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
# RUN: %mojo %s | FileCheck %s

# This sample demonstrates how various systems optimizations can be applied to a
# naive matmul implementation in Mojo to gain significant performance speedups

from random import rand

import benchmark
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import parallelize, vectorize
from sys import info
from memory import memset_zero
from python import Python
from os.env import getenv

alias M = 512  # rows of A and C
alias N = 4096  # cols of B and C
alias K = 512  # cols of A and rows of B
alias type = DType.float32

# simdwidth of = amount of `type` elements that fit into a single SIMD register
# 2x multiplier will use multiple SIMD registers in parallel where possible
alias nelts = simdwidthof[type]() * 2
alias tile_n = 64  # N must be a multiple of this
alias tile_k = 4  # K must be a multiple of this


struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[type]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[type]):
        self.data = data

    ## Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> Scalar[type]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: Scalar[type]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return SIMD[size=nelts].load(self.data, y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        SIMD[size=nelts].store(self.data, y * self.cols + x, val)


def run_matmul_python() -> Float64:
    var pymatmul: PythonObject = Python.import_module("pymatmul")
    var py = Python.import_module("builtins")

    var gflops = pymatmul.benchmark_matmul_python(128, 128, 128).to_float64()
    py.print(py.str("{:<18}{:>8.3f} GFLOPS").format("Python:", gflops))

    return gflops


def run_matmul_numpy():
    var pymatmul: PythonObject = Python.import_module("pymatmul")
    var py = Python.import_module("builtins")

    var gflops = pymatmul.benchmark_matmul_numpy(M, N, K).to_float64()
    py.print(py.str("{:<18}{:>8.3f} GFLOPS").format("Numpy:", gflops))


fn matmul_naive(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Using stdlib vectorize function
fn matmul_vectorized(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.cols]()


# Parallelize the code by using the builtin parallelize function
# num_workers is the number of worker threads to use in parallalize
fn matmul_parallelized(inout C: Matrix, A: Matrix, B: Matrix):
    var num_workers = C.rows

    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts, size = C.cols]()

    parallelize[calc_row](C.rows, num_workers)


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul
# Also parallelize with num_workers threads
fn matmul_tiled(inout C: Matrix, A: Matrix, B: Matrix):
    var num_workers = C.rows

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[dot, nelts, size=tile_x]()

        tile[calc_tile, tile_n, tile_k](C.cols, B.rows)

    parallelize[calc_row](C.rows, num_workers)


# Unroll the vectorized loop by a constant factor
# Also parallelize with num_workers threads
fn matmul_unrolled[mode: Int](inout C: Matrix, A: Matrix, B: Matrix):
    var num_workers: Int
    if mode == 1:
        num_workers = info.num_physical_cores()
    elif mode == 2:
        num_workers = info.num_logical_cores()
    elif mode == 3:
        num_workers = info.num_performance_cores()
    else:
        num_workers = C.rows

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            @parameter
            for _k in range(tile_y):
                var k = _k + y

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[
                    dot, nelts, size=tile_x, unroll_factor = tile_x // nelts
                ]()

        tile[calc_tile, tile_n, tile_k](C.cols, B.rows)

    parallelize[calc_row](C.rows, num_workers)


@always_inline
fn bench[
    func: fn (inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
](base_gflops: Float64) raises:
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9
    var speedup: Float64 = gflops / base_gflops

    var py = Python.import_module("builtins")
    _ = py.print(
        py.str("{:<18}{:>8.3f} GFLOPS {:>9.2f}x Python").format(
            name, gflops, speedup
        )
    )


@always_inline
fn test_matrix_equal[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](inout C: Matrix, A: Matrix, B: Matrix) raises -> Bool:
    """Runs a matmul function on A and B and tests the result for equality with
    C on every element.
    """
    var result = Matrix[M, N]()
    _ = func(result, A, B)
    for i in range(C.rows):
        for j in range(C.cols):
            if C[i, j] != result[i, j]:
                return False
    return True


def test_all():
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    matmul_naive(C, A, B)

    if not test_matrix_equal[matmul_vectorized](C, A, B):
        raise Error("Vectorize output does not match naive implementation")
    if not test_matrix_equal[matmul_parallelized](C, A, B):
        raise Error("Parallelize output does not match naive implementation")
    if not test_matrix_equal[matmul_tiled](C, A, B):
        raise Error("Tiled output does not match naive implementation")
    if not test_matrix_equal[matmul_unrolled[0]](C, A, B):
        raise Error("Unroll output does not match naive implementation")
    if not test_matrix_equal[matmul_unrolled[1]](C, A, B):
        raise Error("Unroll/physical cores does not match naive implementation")
    if not test_matrix_equal[matmul_unrolled[2]](C, A, B):
        raise Error("Unroll/logical cores does not match naive implementation")
    if not test_matrix_equal[matmul_unrolled[3]](C, A, B):
        raise Error("Unroll/perf cores does not match naive implementation")

    A.data.free()
    B.data.free()
    C.data.free()


def main():
    constrained[N % tile_n == 0, "N must be a multiple of tile_n"]()
    constrained[K % tile_k == 0, "K must be a multiple of tile_k"]()

    test_all()
    print("CPU Results\n")
    var python_gflops = run_matmul_python()
    run_matmul_numpy()

    # Don't run all these benchmarks in CI, too resource intensive
    if not getenv("CI"):
        bench[matmul_naive, "Naive:"](python_gflops)
        bench[matmul_vectorized, "Vectorized: "](python_gflops)
        bench[matmul_parallelized, "Parallelized:"](python_gflops)
        bench[matmul_tiled, "Tiled:"](python_gflops)
        bench[matmul_unrolled[0], "Unrolled:"](python_gflops)
        bench[matmul_unrolled[1], "Physical Cores:"](python_gflops)
        bench[matmul_unrolled[2], "Logical Cores:"](python_gflops)
    # CHECK: Performance Cores
    bench[matmul_unrolled[3], "Performance Cores:"](python_gflops)
