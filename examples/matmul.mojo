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

from benchmark import Benchmark
from memory import memset_zero, stack_allocation
from random import rand
from algorithm import vectorize, parallelize, vectorize_unroll
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
    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[DType.float32]):
        self.data = data
        self.rows = rows
        self.cols = cols

    ## Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
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

            vectorize[nelts, dot](C.cols)


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

            vectorize[nelts, dot](C.cols)

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
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


# Unroll the vectorized loop by a constant factor.
# from Functional import vectorize_unroll
fn matmul_unroll(inout C: Matrix, A: Matrix, B: Matrix):
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
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y, parallelizing over y.
fn tile_parallel[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    @parameter
    fn row(yo: Int):
        let y = tile_y * yo
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

    parallelize[row](end_y // tile_y, M)


# Tile the output and accumulate in registers. This strategy means we can
# compute tile_i * tile_j values of output for only reading tile_i + tile_j input values.
fn accumulate_registers(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
        # Allocate the tile of accumulators on the stack.
        var accumulators = Matrix(
            tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]()
        )

        for k in range(0, A.cols):

            @parameter
            fn calc_tile_row[i: Int]():
                @parameter
                fn calc_tile_cols[nelts: Int](j: Int):
                    accumulators.store[nelts](
                        i,
                        j,
                        accumulators.load[nelts](i, j)
                        + A[io + i, k] * B.load[nelts](k, jo + j),
                    )

                vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

            unroll[tile_i, calc_tile_row]()

        # Copy the local tile to the output
        for i in range(tile_i):
            for j in range(tile_j):
                C[io + i, jo + j] = accumulators[i, j]

    alias tile_i = 4
    alias tile_j = nelts * 4
    tile_parallel[calc_tile, tile_j, tile_i](C.cols, C.rows)


@always_inline
fn benchmark[
    func: fn (inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
](base_gflops: Float64, numpy_gflops: Float64) raises:
    var A = Matrix.rand(M, K)
    var B = Matrix.rand(K, N)
    var C = Matrix(M, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000

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
fn test[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](A: Matrix, B: Matrix) raises -> SIMD[type, 1]:
    var C = Matrix(M, N)
    _ = func(C, A, B)
    var result = SIMD[type, 1]()
    for i in range(C.rows):
        for j in range(C.cols):
            result += C[i, j]
    return result


fn test_all() raises:
    constrained[M == N, "M and N must be equal for matrix multiplication"]()

    let A = Matrix.rand(M, K)
    let B = Matrix.rand(K, N)

    let result = test[matmul_naive](A, B)

    if test[matmul_vectorized](A, B) != result:
        raise Error("Vectorize output does not match")
    if test[matmul_parallelized](A, B) != result:
        raise Error("Parallelize output incorrect")
    if test[matmul_tiled](A, B) != result:
        raise Error("Tiled output incorrect")
    if test[matmul_unroll](A, B) != result:
        raise Error("Unroll output incorrect")
    if test[accumulate_registers](A, B) != result:
        raise Error("Accumulate output incorrect")

    A.data.free()
    B.data.free()


fn main() raises:
    # Uncomment below to test correctness of Matmuls
    # test_all()
    print("CPU Results\n")
    let python_gflops = run_matmul_python()
    let numpy_gflops = run_matmul_numpy()

    benchmark[matmul_naive, "Naive:"](python_gflops, numpy_gflops)
    benchmark[matmul_vectorized, "Vectorized: "](python_gflops, numpy_gflops)
    benchmark[matmul_parallelized, "Parallelized:"](python_gflops, numpy_gflops)
    benchmark[matmul_tiled, "Tiled:"](python_gflops, numpy_gflops)
    benchmark[matmul_unroll, "Unrolled:"](python_gflops, numpy_gflops)
    benchmark[accumulate_registers, "Accumulated:"](python_gflops, numpy_gflops)
