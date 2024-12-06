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

from collections import Dict, Optional
from collections.dict import DictEntry
from math import ceil
from random import *
from sys import sizeof

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from bit import bit_ceil


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_dict[size: Int]() -> Dict[Int, Int]:
    var d = Dict[Int, Int]()
    for i in range(0, size):
        d[i] = random.random_si64(0, size).value
    return d


# ===-----------------------------------------------------------------------===#
# Benchmark Dict init
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_init(mut b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(1000):
            var d = Dict[Int, Int]()
            keep(d._entries.data)
            keep(d._index.data)

    b.iter[call_fn]()


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Insert
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_insert[size: Int](mut b: Bencher) raises:
    """Insert 100 new items."""
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(size, size + 100):
            items[key] = random.random_si64(0, size).value

    b.iter[call_fn]()
    keep(bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Lookup
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_dict_lookup[size: Int](mut b: Bencher) raises:
    """Lookup 100 items."""
    var items = make_dict[size]()
    var closest_divisor = ceil(100 / size)

    @__copy_capture(closest_divisor)
    @always_inline
    @parameter
    fn call_fn() raises:
        @parameter
        if size < 100:
            for _ in range(closest_divisor):
                for key in range(int(100 // closest_divisor)):
                    var res = items[key]
                    keep(res)
        else:
            for key in range(100):
                var res = items[key]
                keep(res)

    b.iter[call_fn]()
    keep(bool(items))


# ===-----------------------------------------------------------------------===#
# Benchmark Dict Memory Footprint
# ===-----------------------------------------------------------------------===#


fn total_bytes_used(items: Dict[Int, Int]) -> Int:
    # the allocated memory by entries:
    var entry_size = sizeof[Optional[DictEntry[Int, Int]]]()
    var amnt_bytes = items._entries.capacity * entry_size
    amnt_bytes += sizeof[Dict[Int, Int]]()

    # the allocated memory by index table:
    var reserved = items._reserved()
    if reserved <= 128:
        amnt_bytes += sizeof[Int8]() * reserved
    elif reserved <= 2**16 - 2:
        amnt_bytes += sizeof[Int16]() * reserved
    elif reserved <= 2**32 - 2:
        amnt_bytes += sizeof[Int32]() * reserved
    else:
        amnt_bytes += sizeof[Int64]() * reserved

    return amnt_bytes


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function[bench_dict_init](BenchId("bench_dict_init"))
    alias sizes = (10, 30, 50, 100, 1000, 10_000, 100_000, 1_000_000)

    @parameter
    for i in range(len(sizes)):
        alias size = sizes.get[i, Int]()
        m.bench_function[bench_dict_insert[size]](
            BenchId("bench_dict_insert[" + str(size) + "]")
        )
        m.bench_function[bench_dict_lookup[size]](
            BenchId("bench_dict_lookup[" + str(size) + "]")
        )

    m.dump_report()

    @parameter
    for i in range(len(sizes)):
        alias size = sizes.get[i, Int]()
        var mem_s = total_bytes_used(make_dict[size]())
        print(
            '"bench_dict_memory_size[' + str(size) + ']",' + str(mem_s) + ",0"
        )
