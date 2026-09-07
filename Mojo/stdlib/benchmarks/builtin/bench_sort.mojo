# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from std.random import *

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.builtin.sort import _heap_sort, _insertion_sort, _small_sort, sort

# ===-----------------------------------------------------------------------===#
# Benchmark Utils
# ===-----------------------------------------------------------------------===#


@always_inline
def randomize_list[
    dt: DType
](mut list: List[Scalar[dt]], size: Int, max: Scalar[dt] = Scalar[dt].MAX):
    comptime if dt.is_integral():
        randint(list.unsafe_ptr(), size, 0, Int(max))
    else:
        for i in range(size):
            var res = random_float64()
            # GCC doesn't support cast from float64 to float16
            list[i] = res.cast[.float32]().cast[dt]()


@always_inline
def insertion_sort[dtype: DType](mut list: List[Scalar[dtype]]):
    def _less_than(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
        return lhs < rhs

    _insertion_sort(list, _less_than)


@always_inline
def small_sort[size: Int, dtype: DType](mut list: List[Scalar[dtype]]):
    def _less_than(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
        return lhs < rhs

    _small_sort[size](list, _less_than)


@always_inline
def heap_sort[dtype: DType](mut list: List[Scalar[dtype]]):
    def _less_than(lhs: Scalar[dtype], rhs: Scalar[dtype]) -> Bool:
        return lhs < rhs

    _heap_sort(list, _less_than)


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a tiny list size
# ===-----------------------------------------------------------------------===#


def bench_tiny_list_sort[dtype: DType](mut m: Bench) raises:
    comptime small_list_size = 5

    comptime for count in range(2, small_list_size + 1):

        def bench_sort_list(mut b: Bencher) raises {}:
            seed(1)
            var list = List(length=count, fill=Scalar[dtype]())

            @always_inline
            def preproc(mut list: List[Scalar[dtype]]):
                randomize_list(list, count)

            @always_inline
            def call_fn(mut list: List[Scalar[dtype]]):
                sort(list)

            b.iter_preproc(list, call_fn, preproc)

        def bench_small_sort(mut b: Bencher) raises {}:
            seed(1)
            var list = List(length=count, fill=Scalar[dtype]())

            @always_inline
            def preproc(mut list: List[Scalar[dtype]]):
                randomize_list(list, count)

            @always_inline
            def call_fn(mut list: List[Scalar[dtype]]):
                small_sort[count](list)

            b.iter_preproc(list, call_fn, preproc)

        def bench_insertion_sort(mut b: Bencher) raises {}:
            seed(1)
            var list = List(length=count, fill=Scalar[dtype]())

            @always_inline
            def preproc(mut list: List[Scalar[dtype]]):
                randomize_list(list, count)

            @always_inline
            def call_fn(mut list: List[Scalar[dtype]]):
                insertion_sort(list)

            b.iter_preproc(list, call_fn, preproc)

        m.bench_function(
            bench_sort_list,
            BenchId(String("std_sort_random_", count, "_", dtype)),
        )
        m.bench_function(
            bench_small_sort,
            BenchId(String("sml_sort_random_", count, "_", dtype)),
        )
        m.bench_function(
            bench_insertion_sort,
            BenchId(String("ins_sort_random_", count, "_", dtype)),
        )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a small list size
# ===-----------------------------------------------------------------------===#


def bench_small_list_sort[dtype: DType](mut m: Bench, count: Int) raises:
    def bench_sort_list(mut b: Bencher) raises {var count}:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        def preproc(mut list: List[Scalar[dtype]]) {var count}:
            randomize_list(list, count)

        @always_inline
        def call_fn(mut list: List[Scalar[dtype]]):
            sort(list)

        b.iter_preproc(list, call_fn, preproc)

    def bench_insertion_sort(mut b: Bencher) raises {var count}:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        def preproc(mut list: List[Scalar[dtype]]) {var count}:
            randomize_list(list, count)

        @always_inline
        def call_fn(mut list: List[Scalar[dtype]]):
            insertion_sort(list)

        b.iter_preproc(list, call_fn, preproc)

    m.bench_function(
        bench_sort_list,
        BenchId(String("std_sort_random_", count, "_", dtype)),
    )
    m.bench_function(
        bench_insertion_sort,
        BenchId(String("ins_sort_random_", count, "_", dtype)),
    )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a large list size
# ===-----------------------------------------------------------------------===#


def bench_large_list_sort[dtype: DType](mut m: Bench, count: Int) raises:
    def bench_sort_list(mut b: Bencher) raises {var count}:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        def preproc(mut list: List[Scalar[dtype]]) {var count}:
            randomize_list(list, count)

        @always_inline
        def call_fn(mut list: List[Scalar[dtype]]):
            sort(list)

        b.iter_preproc(list, call_fn, preproc)

    def bench_heap_sort(mut b: Bencher) raises {var count}:
        seed(1)
        var list = List(length=count, fill=Scalar[dtype]())

        @always_inline
        def preproc(mut list: List[Scalar[dtype]]) {var count}:
            randomize_list(list, count)

        @always_inline
        def call_fn(mut list: List[Scalar[dtype]]):
            heap_sort(list)

        b.iter_preproc(list, call_fn, preproc)

    m.bench_function(
        bench_sort_list,
        BenchId(String("std_sort_random_", count, "_", dtype)),
    )

    m.bench_function(
        bench_heap_sort,
        BenchId(String("heap_sort_random_", count, "_", dtype)),
    )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with low delta lists
# ===-----------------------------------------------------------------------===#


def bench_low_cardinality_list_sort(
    mut m: Bench, count: Int, delta: Int
) raises:
    def bench_sort_list(mut b: Bencher) raises {var count, var delta}:
        seed(1)
        var list = List(length=count, fill=UInt8())

        @always_inline
        def preproc(mut list: List[UInt8]) {var count, var delta}:
            randomize_list(list, count, UInt8(delta))

        @always_inline
        def call_fn(mut list: List[UInt8]):
            sort(list)

        b.iter_preproc(list, call_fn, preproc)

    def bench_heap_sort(mut b: Bencher) raises {var count, var delta}:
        seed(1)
        var list = List(length=count, fill=UInt8())

        @always_inline
        def preproc(mut list: List[UInt8]) {var count, var delta}:
            randomize_list(list, count, UInt8(delta))

        @always_inline
        def call_fn(mut list: List[UInt8]):
            heap_sort(list)

        b.iter_preproc(list, call_fn, preproc)

    m.bench_function(
        bench_sort_list,
        BenchId(String("std_sort_low_card_", count, "_delta_", delta)),
    )
    m.bench_function(
        bench_heap_sort,
        BenchId(String("heap_sort_low_card_", count, "_delta_", delta)),
    )


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    var m = Bench(BenchConfig(max_runtime_secs=0.1))

    comptime dtypes = [
        DType.uint8,
        DType.uint16,
        DType.float16,
        DType.uint32,
        DType.float32,
        DType.uint64,
        DType.float64,
    ]
    var small_counts = [10, 20, 32, 64, 100]
    var large_counts = [2**12, 2**16, 2**20]
    var deltas = [0, 2, 5, 20, 100]

    comptime for dtype in dtypes:
        bench_tiny_list_sort[dtype](m)

    comptime for dtype in dtypes:
        for count1 in small_counts:
            bench_small_list_sort[dtype](m, count1)

    comptime for i in range(len(dtypes)):
        comptime dtype = dtypes[i]
        for count2 in large_counts:
            bench_large_list_sort[dtype](m, count2)

    for count3 in large_counts:
        for delta2 in deltas:
            bench_low_cardinality_list_sort(m, count3, delta2)

    m.dump_report()
