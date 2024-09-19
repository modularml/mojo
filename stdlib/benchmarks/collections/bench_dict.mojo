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

from random import *

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from memory.memory import sizeof
from bit import bit_ceil
from math import ceil
from stdlib.collections.dict import Dict, DictEntry


# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
fn make_dict[size: Int]() -> Dict[Int, Int]:
    var d = Dict[Int, Int]()
    for i in range(0, size):
        d[i] = random.random_si64(0, size).value
    return d


# ===----------------------------------------------------------------------===#
# Benchmark Dict init
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_init(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        for _ in range(1000):
            var d = Dict[Int, Int]()
            keep(d._entries.data)
            keep(d._index.data)

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Dict Insert
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_insert[size: Int](inout b: Bencher) raises:
    """Insert 100 new items."""
    var items = make_dict[size]()

    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(size, size + 100):
            items[key] = random.random_si64(0, size).value

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark Dict Lookup
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_lookup[size: Int](inout b: Bencher) raises:
    """Lookup 100 items."""
    var items = make_dict[size]()
    var closest_divisor = ceil(100 / size)

    @always_inline
    @parameter
    fn call_fn() raises:
        @parameter
        if size < 100:
            for _ in range(closest_divisor):
                for key in range(100 // closest_divisor):
                    var res = items[key]
                    keep(res)
        else:
            for key in range(100):
                var res = items[key]
                keep(res)

    b.iter[call_fn]()
    keep(bool(items))


# ===----------------------------------------------------------------------===#
# Benchmark Dict Memory Footprint
# ===----------------------------------------------------------------------===#


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


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=100))
    m.bench_function[bench_dict_init](BenchId("bench_dict_init"))
    alias sizes = (
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
        200,
        300,
        400,
        500,
        600,
        700,
        800,
        900,
        1000,
        2000,
        3000,
        4000,
        5000,
        6000,
        7000,
        8000,
        9000,
        10_000,
        20_000,
        30_000,
        40_000,
        50_000,
        60_000,
        70_000,
        80_000,
        90_000,
        100_000,
        200_000,
        300_000,
        400_000,
        500_000,
        600_000,
        700_000,
        800_000,
        900_000,
        1_000_000,
    )

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
