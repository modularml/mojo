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

from math import iota
from sys import num_logical_cores

import benchmark
from algorithm import parallelize, vectorize
from complex import ComplexFloat64, ComplexSIMD
from python import Python
from runtime.llcl import Runtime
from tensor import Tensor

from utils import Index

alias float_type = DType.float64
alias int_type = DType.int64
alias simd_width = 2 * simdwidthof[float_type]()

alias width = 960
alias height = 960
alias MAX_ITERS = 200

alias min_x = -2.0
alias max_x = 0.6
alias min_y = -1.5
alias max_y = 1.5


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
    for i in range(MAX_ITERS):
        if not t.reduce_or():
            break
        y2 = y * y
        y = x.fma(y + y, cy)
        t = x.fma(x, y2) <= 4
        x = x.fma(x, cx - y2)
        iters = t.select(iters + 1, iters)
    return iters


fn main() raises:
    var t = Tensor[int_type](height, width)

    @parameter
    fn worker(row: Int):
        var scale_x = (max_x - min_x) / width
        var scale_y = (max_y - min_y) / height

        @__copy_capture(scale_x, scale_y)
        @parameter
        fn compute_vector[simd_width: Int](col: Int):
            """Each time we operate on a `simd_width` vector of pixels."""
            var cx = min_x + (col + iota[float_type, simd_width]()) * scale_x
            var cy = min_y + row * scale_y
            var c = ComplexSIMD[float_type, simd_width](cx, cy)
            t.data().store[width=simd_width](
                row * width + col, mandelbrot_kernel_SIMD[simd_width](c)
            )

        # Vectorize the call to compute_vector where call gets a chunk of pixels.
        vectorize[compute_vector, simd_width, size=width]()

    @parameter
    fn bench[simd_width: Int]():
        for row in range(height):
            worker(row)

    var vectorized = benchmark.run[bench[simd_width]](
        max_runtime_secs=0.5
    ).mean()
    print("Number of threads:", num_logical_cores())
    print("Vectorized:", vectorized, "s")

    with Runtime(num_logical_cores()) as rt:
        # Parallelized
        @parameter
        fn bench_parallel[simd_width: Int]():
            parallelize[worker](height, height)

        var parallelized = benchmark.run[bench_parallel[simd_width]](
            max_runtime_secs=0.5
        ).mean()
        print("Parallelized:", parallelized, "s")
        # CHECK: Parallel speedup
        print("Parallel speedup:", vectorized / parallelized)

    _ = t  # Make sure tensor isn't destroyed before benchmark is finished
