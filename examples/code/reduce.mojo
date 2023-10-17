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
    var sum = data[0]
    var c: Float32 = 0.0
    for i in range(size):
        let y = data[i] - c
        let t = sum + y
        c = (t - sum) - y
        sum = t
    return sum


fn benchmark_naive_reduce_sum(size: Int) -> Float32:
    print("Computing reduction sum for array num elements: ", size)
    var A = ArrayInput(size)
    # Prevent DCE
    var mySum: Float32 = 0.0

    @always_inline
    @parameter
    fn test_fn():
        _ = reduce_sum_naive(A, size)

    let bench_time = Float64(Benchmark().run[test_fn]())
    return mySum


fn benchmark_stdlib_reduce_sum(size: Int) -> Float32:
    # Allocate a Buffer and then use the Mojo stdlib Reduction class
    # TODO: Use globals
    # alias numElem = size
    alias numElem = 1 << 30
    # Can use either stack allocation or heap
    # see stackalloc
    # var A = Buffer[numElem, DType.float32].stack_allocation()
    # see heapalloc
    var B = DTypePointer[DType.float32].alloc(numElem)
    var A = Buffer[numElem, DType.float32](B)

    # initialize buffer
    for i in range(numElem):
        A[i] = Float32(i)

    # Prevent DCE
    var mySum: Float32 = 0.0
    print("Computing reduction sum for array num elements: ", size)

    @always_inline
    @parameter
    fn test_fn():
        mySum = sum[numElem, DType.float32](A)

    let bench_time = Float64(Benchmark().run[test_fn]())
    return mySum


fn main():
    # Number of array elements
    let size = 1 << 21
    print("# Reduction sum across a large array. The naive algorithm's ")
    print("# computation time scales with the size of the array; while Mojo ")
    print("# exhibits significantly better scaling...")
    var eval_begin: Float64 = now()
    var sum = benchmark_naive_reduce_sum(size)
    var eval_end: Float64 = now()
    var execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed naive reduction sum: ", sum, " in ", execution_time, "ms")

    eval_begin = now()
    sum = benchmark_stdlib_reduce_sum(size)
    eval_end = now()
    execution_time = Float64((eval_end - eval_begin)) / 1e6
    print("Completed stdlib reduction sum: ", sum, " in ", execution_time, "ms")
