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
from sys.intrinsics import strided_load
from utils.list import VariadicList
from math import div_ceil, min
from memory import memset_zero
from random import rand, random_float64
from sys.info import simdwidthof
from time import now
from algorithm import vectorize, parallelize, vectorize_unroll
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python.object import PythonObject
from python.python import Python, _destroy_python, _init_python
from runtime.llcl import Runtime


struct Matrix:
    var data: DTypePointer[DType.float32]
    var rows: Int
    var cols: Int

    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[DType.float32].alloc(rows * cols)
        rand(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    fn __del__(owned self):
        self.data.free()

    fn zero(inout self):
        memset_zero(self.data, self.rows * self.cols)

    @always_inline
    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    @always_inline
    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    @always_inline
    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    @always_inline
    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


fn run_matmul_python(M: Int, N: Int, K: Int) -> Float64:
    var gflops: Float64 = 0.0
    let python = Python()
    try:
        Python.add_to_path(".")
        Python.add_to_path("./examples")
        let pymatmul_module: PythonObject = Python.import_module("pymatmul")
        if pymatmul_module:
            gflops = pymatmul_module.benchmark_matmul_python(
                M, N, K
            ).to_float64()
        else:
            print("pymatmul module not found")
    except e:
        print(e.value)
        pass
    return gflops


fn matmul_naive(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = simdwidthof[DType.float32]()  # The SIMD vector width.


fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols, nelts):
                C.store[nelts](
                    m, nv, C.load[nelts](m, nv) + A[m, k] * B.load[nelts](k, nv)
                )

            # Handle remaining elements with scalars.
            for n in range(nelts * (C.cols // nelts), C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Simplify the code by using the builtin vectorize function
# from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix, _rt: Runtime):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)


# Parallelize the code by using the builtin parallelize function
# from Functional import parallelize
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

    parallelize[calc_row](rt, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul.
fn matmul_tiled_parallelized(C: Matrix, A: Matrix, B: Matrix, rt: Runtime):
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

                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](rt, C.rows)


# Unroll the vectorized loop by a constant factor.
# from Functional import vectorize_unroll
fn matmul_tiled_unrolled_parallelized(
    C: Matrix, A: Matrix, B: Matrix, rt: Runtime
):
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

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](rt, C.rows)


@always_inline
fn benchmark[
    func: fn (Matrix, Matrix, Matrix, Runtime) -> None
](M: Int, N: Int, K: Int, base_gflops: Float64, str: String):
    var C = Matrix(M, N)
    C.zero()
    var A = Matrix(M, K)
    var B = Matrix(K, N)

    with Runtime() as rt:

        @always_inline
        @parameter
        fn test_fn():
            _ = func(C, A, B, rt)

        let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
        # Prevent the matrices from being freed before the benchmark run
        _ = (A, B, C)
        let gflops = ((2 * M * N * K) / secs) / 1e9
        let speedup: Float64 = gflops / base_gflops
        # print(gflops, "GFLOP/s", speedup, " speedup")
        print(str)
        print(gflops, "GFLOP/s <>", speedup.to_int(), "x speedup over Python")


fn main():
    # Python
    print("Throughput of a 128x128 matrix multiplication in Python: ")
    let python_gflops = run_matmul_python(128, 128, 128)
    alias M = 512
    # Mojo variants
    benchmark[matmul_naive](
        M,
        M,
        M,
        python_gflops,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using a"
            " naive algorithm: "
        ),
    )
    benchmark[matmul_vectorized_0](
        M,
        M,
        M,
        python_gflops,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using"
            " vectorization: "
        ),
    )
    benchmark[matmul_vectorized_1](
        M,
        M,
        M,
        python_gflops,
        (
            "Throughput of a 512x512 matrix multiplication in Mojo using the"
            " stdlib `vectorize`: "
        ),
    )
    benchmark[matmul_parallelized](
        M,
        M,
        M,
        python_gflops,
        (
            "Throughput of a 512x512 {vectorized + parallelized} matrix"
            " multiplication in Mojo: "
        ),
    )
    benchmark[matmul_tiled_parallelized](
        M,
        M,
        M,
        python_gflops,
        (
            "Throughput of a 512x512 {tiled + vectorized + parallelized} matrix"
            " multiplication in Mojo: "
        ),
    )
    benchmark[matmul_tiled_unrolled_parallelized](
        M,
        M,
        M,
        python_gflops,
        (
            "Throughput of a 512x512 {tiled + unrolled + vectorized +"
            " parallelized} matrix multiplication in Mojo: "
        ),
    )
