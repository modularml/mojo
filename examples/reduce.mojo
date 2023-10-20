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

# This sample implements a simple reduction operation on a
# large array of values to produce a single result.
# Reductions and scans are common algorithm patterns in parallel computing.

from benchmark import Benchmark
from time import now
from algorithm import sum
from random import rand
from memory.buffer import Buffer
from python import Python

# Change these numbers to reduce on different sizes
alias size_small: Int = 1 << 21
alias size_large: Int = 1 << 29

# Simple array struct
struct ArrayInput:
    var data: DTypePointer[DType.float32]

    fn __init__(inout self, size: Int):
        self.data = DTypePointer[DType.float32].alloc(size)
        rand(self.data, size)

    fn __del__(owned self):
        self.data.free()

    @always_inline
    fn __getitem__(self, x: Int) -> Float32:
        return self.data.load(x)


# Use the https://en.wikipedia.org/wiki/Kahan_summation_algorithm
# Simple summation of the array elements
fn reduce_sum_naive(data: ArrayInput, size: Int) -> Float32:
    var my_sum = data[0]
    var c: Float32 = 0.0
    for i in range(size):
        let y = data[i] - c
        let t = my_sum + y
        c = (t - my_sum) - y
        my_sum = t
    return my_sum


fn benchmark_naive_reduce_sum[size: Int]() -> Float32:
    var A = ArrayInput(size)
    # Prevent DCE
    var my_sum: Float32 = 0.0

    @always_inline
    @parameter
    fn test_fn():
        _ = reduce_sum_naive(A, size)

    let bench_time = Float64(Benchmark().run[test_fn]())
    return my_sum


fn benchmark_stdlib_reduce_sum[size: Int]() -> Float32:
    # Allocate a Buffer and then use the Mojo stdlib Reduction class
    var B = DTypePointer[DType.float32].alloc(size)
    var A = Buffer[size, DType.float32](B)

    # initialize buffer
    for i in range(size):
        A[i] = Float32(i)

    # Prevent DCE
    var my_sum: Float32 = 0.0

    @always_inline
    @parameter
    fn test_fn():
        my_sum = sum[size, DType.float32](A)

    let bench_time = Float64(Benchmark().run[test_fn]())
    return my_sum

fn pretty_print(str: StringLiteral, elements: Int, time: Float64) raises:
    let py = Python.import_module("builtins")
    _ = py.print(
        py.str("{:<16} {:>11,} {:>8.2f}ms").format(
            str, elements, time
        )
    )

fn benchmark[func: fn[size: Int]() -> Float32, size: Int, name: StringLiteral]() raises:
    let eval_begin: Float64 = now()
    let sum = func[size]()
    let eval_end: Float64 = now()
    let execution_time = Float64((eval_end - eval_begin)) / 1e6
    pretty_print("naive elements:", size, execution_time)

fn main() raises:
    print("Reduction sum across a large array, shows better scaling using stdlib\n")

    benchmark[benchmark_naive_reduce_sum, size_small, "naive"]()
    benchmark[benchmark_naive_reduce_sum, size_large, "naive"]()

    benchmark[benchmark_stdlib_reduce_sum, size_small, "stdlib"]()
    benchmark[benchmark_stdlib_reduce_sum, size_large, "stdlib"]()
