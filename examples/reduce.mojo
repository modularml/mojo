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

# This sample implements a simple reduction operation on a
# large array of values to produce a single result.
# Reductions and scans are common algorithm patterns in parallel computing.

from time import now
from algorithm import sum
from benchmark import Unit, benchmark, keep
from buffer import Buffer
from python import Python
from random import rand

# Change these numbers to reduce on different sizes
alias size_small: Int = 1 << 21
alias size_large: Int = 1 << 27

# Datatype for Tensor/Array
alias type = DType.float32
alias scalar = Scalar[type]


# Use the https://en.wikipedia.org/wiki/Kahan_summation_algorithm
# Simple summation of the array elements
fn naive_reduce_sum[size: Int](buffer: Buffer[type, size]) -> scalar:
    var my_sum: scalar = 0
    var c: scalar = 0
    for i in range(buffer.size):
        var y = buffer[i] - c
        var t = my_sum + y
        c = (t - my_sum) - y
        my_sum = t
    return my_sum


fn stdlib_reduce_sum[size: Int](array: Buffer[type, size]) -> scalar:
    var my_sum = sum(array)
    return my_sum


def pretty_print(name: String, elements: Int, time: Float64):
    py = Python.import_module("builtins")
    py.print(
        py.str("{:<16} {:>11,} {:>8.2f}ms").format(
            name + " elements:", elements, time
        )
    )


fn bench[
    func: fn[size: Int] (buffer: Buffer[type, size]) -> scalar,
    size: Int,
    name: String,
](buffer: Buffer[type, size]) raises:
    @parameter
    fn runner():
        var result = func[size](buffer)
        keep(result)

    var ms = benchmark.run[runner](max_runtime_secs=0.5).mean(Unit.ms)
    pretty_print(name, size, ms)


fn main() raises:
    print(
        "Sum all values in a small array and large array\n"
        "Shows algorithm.sum from stdlib with much better performance\n"
    )
    # Allocate and randomize data, then create two buffers
    var ptr_small = DTypePointer[type].alloc(size_small)
    var ptr_large = DTypePointer[type].alloc(size_large)

    rand(ptr_small, size_small)
    rand(ptr_large, size_large)

    var buffer_small = Buffer[type, size_small](ptr_small)
    var buffer_large = Buffer[type, size_large](ptr_large)

    bench[naive_reduce_sum, size_small, "naive"](buffer_small)
    bench[naive_reduce_sum, size_large, "naive"](buffer_large)
    bench[stdlib_reduce_sum, size_small, "stdlib"](buffer_small)
    # CHECK: stdlib elements
    bench[stdlib_reduce_sum, size_large, "stdlib"](buffer_large)

    ptr_small.free()
    ptr_large.free()
