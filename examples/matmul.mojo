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

# This sample demonstrates how various systems optimizations can be
# applied to a naive matmul implementation in Mojo to gain significant
# performance speedups

import benchmark
from memory import memset_zero, stack_allocation
from random import rand
from algorithm import vectorize, parallelize
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python import Python
from tensor import Tensor
from utils.index import Index
from memory.buffer import NDBuffer

alias M = 512
alias N = 512
alias K = 4096
alias type = DType.float32


struct Matrix:
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize zeroeing all values
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[type]):
        self.data = data
        self.rows = rows
        self.cols = cols

    ## Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(self, y: Int, x: Int) -> SIMD[type, 1]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[type, 1]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[type, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[type, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


def run_matmul_python() -> Float64:
    Python.add_to_path(".")
    let pymatmul: PythonObject = Python.import_module("pymatmul")
    let py = Python.import_module("builtins")

    let gflops = pymatmul.benchmark_matmul_python(128, 128, 128).to_float64()
    py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Python:", gflops))

    return gflops


def run_matmul_numpy() -> Float64:
    let pymatmul: PythonObject = Python.import_module("pymatmul")
    let py = Python.import_module("builtins")

    let gflops = pymatmul.benchmark_matmul_numpy(M, N, K).to_float64()
    py.print(py.str("{:<13}{:>8.3f} GFLOPS").format("Numpy:", gflops))

    return gflops


fn matmul_naive(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = simdwidthof[type]()  # The SIMD vector width.


# Using stdlib vectorize function
fn matmul_vectorized(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts](C.cols)


# Parallelize the code by using the builtin parallelize function
fn matmul_parallelized(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[dot, nelts](C.cols)

    parallelize[calc_row](C.rows, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul.
fn matmul_tiled(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[dot, nelts](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


# Unroll the vectorized loop by a constant factor.
fn matmul_unrolled(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_size = 4

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, tile_x, unroll_factor]()

        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


@always_inline
fn bench[
    func: fn (inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
](base_gflops: Float64, numpy_gflops: Float64) raises:
    var A = Matrix.rand(M, K)
    var B = Matrix.rand(K, N)
    var C = Matrix(M, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    # Prevent the matrices from being freed before the benchmark run
    A.data.free()
    B.data.free()
    C.data.free()
    let gflops = ((2 * M * N * K) / secs) / 1e9
    let speedup: Float64 = gflops / base_gflops
    let numpy_speedup: Float64 = gflops / numpy_gflops

    let py = Python.import_module("builtins")
    _ = py.print(
        py.str("{:<13}{:>8.3f} GFLOPS {:>9.2f}x Python {:>5.2f}x Numpy").format(
            name, gflops, speedup, numpy_speedup
        )
    )


@always_inline
fn test_matrix_equal[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](inout C: Matrix, A: Matrix, B: Matrix) raises -> SIMD[type, 1]:
    """Runs a matmul function on A and B and tests the result for equality with
    C on every element.
    """
    var result = Matrix(M, N)
    _ = func(result, A, B)
    for i in range(C.rows):
        for j in range(C.cols):
            if C[i, j] != result[i, j]:
                return False
    return True


fn test_all() raises:
    constrained[M == N, "M and N must be equal for matrix multiplication"]()

    let A = Matrix.rand(M, K)
    let B = Matrix.rand(K, N)
    var C = Matrix(M, N)

    matmul_naive(C, A, B)

    if not test_matrix_equal[matmul_vectorized](C, A, B):
        raise Error("Vectorize output does not match naive implementation")
    if not test_matrix_equal[matmul_parallelized](C, A, B):
        raise Error("Parallelize output does not match naive implementation")
    if not test_matrix_equal[matmul_tiled](C, A, B):
        raise Error("Tiled output does not match naive implementation")
    if not test_matrix_equal[matmul_unrolled](C, A, B):
        raise Error("Unroll output does not match naive implementation")

    A.data.free()
    B.data.free()
    C.data.free()


fn main() raises:
    test_all()
    print("CPU Results\n")
    let python_gflops = run_matmul_python()
    let numpy_gflops = run_matmul_numpy()

    bench[matmul_naive, "Naive:"](python_gflops, numpy_gflops)
    bench[matmul_vectorized, "Vectorized: "](python_gflops, numpy_gflops)
    bench[matmul_parallelized, "Parallelized:"](python_gflops, numpy_gflops)
    bench[matmul_tiled, "Tiled:"](python_gflops, numpy_gflops)
    bench[matmul_unrolled, "Unrolled:"](python_gflops, numpy_gflops)
