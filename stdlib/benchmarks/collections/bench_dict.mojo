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
# RUN: %mojo %s -t

from benchmark import Bench, Bencher, BenchId, keep, BenchConfig, Unit, run
from stdlib.collections import Dict
from random import *


# ===----------------------------------------------------------------------===#
# Benchmark Data
# ===----------------------------------------------------------------------===#
fn make_dict(n: Int) -> Dict[Int, Int]:
    var dict = Dict[Int, Int]()
    for i in range(0, n):
        dict[i] = random.random_si64(0, n).value
    return dict


alias small_n = 10_000
alias large_n = 500_000
alias insert_n = small_n // 2
alias partial_n = small_n // 4

var small = make_dict(small_n)
var large = make_dict(large_n)


# ===----------------------------------------------------------------------===#
# Benchmark Dict Ctor
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_ctor(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn():
        var _d: Dict[Int, Int] = Dict[Int, Int]()

    b.iter[call_fn]()


# ===----------------------------------------------------------------------===#
# Benchmark Dict Small Insert
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_small_insert(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(small_n, small_n + insert_n):
            small[key] = random.random_si64(0, small_n).value

    b.iter[call_fn]()
    keep(small)


# ===----------------------------------------------------------------------===#
# Benchmark Dict Large Insert
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_large_insert(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(large_n, large_n + insert_n):
            large[key] = random.random_si64(0, large_n).value

    b.iter[call_fn]()
    keep(large)


# ===----------------------------------------------------------------------===#
# Benchmark Dict Small Lookup
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_small_lookup(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(0, partial_n):
            _ = small[key]

    b.iter[call_fn]()
    keep(small)


# ===----------------------------------------------------------------------===#
# Benchmark Dict Large Lookup
# ===----------------------------------------------------------------------===#
@parameter
fn bench_dict_large_lookup(inout b: Bencher) raises:
    @always_inline
    @parameter
    fn call_fn() raises:
        for key in range(0, partial_n):
            _ = large[key]

    b.iter[call_fn]()
    keep(large)


# ===----------------------------------------------------------------------===#
# Benchmark Main
# ===----------------------------------------------------------------------===#
def main():
    seed()
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=100))
    m.bench_function[bench_dict_ctor](BenchId("bench_dict_ctor"))
    m.bench_function[bench_dict_small_insert](
        BenchId("bench_dict_small_insert")
    )
    m.bench_function[bench_dict_large_insert](
        BenchId("bench_dict_large_insert")
    )
    m.bench_function[bench_dict_small_lookup](
        BenchId("bench_dict_small_lookup")
    )
    m.bench_function[bench_dict_large_lookup](
        BenchId("bench_dict_large_lookup")
    )
    m.dump_report()
