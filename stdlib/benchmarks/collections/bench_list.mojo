# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
# RUN: %mojo-no-debug %s -t
# NOTE: to test changes on the current branch using run-benchmarks.sh, remove
# the -t flag. Remember to replace it again before pushing any code.

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from memory import UnsafePointer
from random import seed
from collections import List
from collections import Dict
from time import now


def benchmark_list_hint_trivial_type_int[length: Int, iterations: Int]() -> Int:
    var start = now()
    var stop = now()
    alias size = length

    var items = List[Int, True]()
    for i in range(size):
        items.append(i)

    start = now()
    for iter in range(iterations):
        var items2 = items
        keep(items2.data)
    stop = now()
    keep(items.data)
    return stop - start


def benchmark_string_copyinit__[length: Int, iterations: Int]() -> Int:
    var start = now()
    var stop = now()

    var x: String = ""
    for l in range(length):
        x += str(l)[0]

    start = now()
    for iter in range(iterations):
        var y: String
        String.__copyinit__(y, x)
        keep(y._buffer.data)
    stop = now()
    keep(x._buffer.data)
    return stop - start


def main():
    seed()

    alias iterations = 1 << 10

    alias result_type = Dict[String, Int]
    var results = Dict[String, result_type]()
    results["list_hint_trivial_type"] = result_type()
    results["string_copyinit"] = result_type()

    alias lengths = (1, 2, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096)

    @parameter
    for i in range(len(lengths)):
        alias length = lengths.get[i, Int]()
        results["list_hint_trivial_type"][
            str(length)
        ] = benchmark_list_hint_trivial_type_int[length, iterations]()
        results["string_copyinit"][str(length)] = benchmark_string_copyinit__[
            length, iterations
        ]()

    print("iterations: ", iterations)
    for benchmark in results:
        print(benchmark[])
        for result in results[benchmark[]]:
            print("\t", result[], "\t", results[benchmark[]][result[]])
        print()
