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

import benchmark
from math import iota
from sys import num_physical_cores
from algorithm import parallelize, vectorize
from complex import ComplexFloat64, ComplexSIMD

alias float_type = DType.float32
alias int_type = DType.int32
alias simd_width = 2 * simdwidthof[float_type]()
alias unit = benchmark.Unit.ms

alias cols = 960
alias rows = 960
alias MAX_ITERS = 200

alias min_x = -2.0
alias max_x = 0.6
alias min_y = -1.5
alias max_y = 1.5


struct Matrix[type: DType, rows: Int, cols: Int]:
    var data: DTypePointer[type]

    fn __init__(inout self):
        self.data = DTypePointer[type].alloc(rows * cols)

    fn store[nelts: Int](self, row: Int, col: Int, val: SIMD[type, nelts]):
        SIMD[size=nelts].store(self.data, row * cols + col, val)


fn mandelbrot_kernel_SIMD[
    simd_width: Int
](c: ComplexSIMD[float_type, simd_width]) -> SIMD[int_type, simd_width]:
    """A vectorized implementation of the inner mandelbrot computation."""
    var cx = c.re
    var cy = c.im
    var x = SIMD[float_type, simd_width](0)
    var y = SIMD[float_type, simd_width](0)
    var y2 = SIMD[float_type, simd_width](0)
    var iters = SIMD[int_type, simd_width](0)
    var t: SIMD[DType.bool, simd_width] = True

    for _ in range(MAX_ITERS):
        if not any(t):
            break
        y2 = y * y
        y = x.fma(y + y, cy)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, cx - y2)
        iters = t.select(iters + 1, iters)
    return iters


fn main() raises:
    var matrix = Matrix[int_type, rows, cols]()

    @parameter
    fn worker(row: Int):
        var scale_x = (max_x - min_x) / cols
        var scale_y = (max_y - min_y) / rows

        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we operate on a `simd_width` vector of pixels."""
            var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            var cy = min_y + row * scale_y
            var c = ComplexSIMD[float_type, simd_width](cx, cy)
            matrix.store(row, col, mandelbrot_kernel_SIMD(c))

        # Vectorize the call to compute_vector with a chunk of pixels.
        vectorize[compute_vector, simd_width, size=cols]()

    @parameter
    fn bench():
        for row in range(rows):
            worker(row)

    @parameter
    fn bench_parallel():
        parallelize[worker](rows, rows)

    print("Number of physical cores:", num_physical_cores())

    var vectorized = benchmark.run[bench]().mean(unit)
    print("Vectorized:", vectorized, unit)
    var parallelized = benchmark.run[bench_parallel]().mean(unit)
    print("Parallelized:", parallelized, unit)

    # CHECK: Parallel speedup
    print("Parallel speedup:", vectorized / parallelized)

    matrix.data.free()
