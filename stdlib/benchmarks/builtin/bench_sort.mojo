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

from random import *

from benchmark import Bench, BenchConfig, Bencher, BenchId, Unit, keep, run
from memory import UnsafePointer
from stdlib.builtin.sort import (
    _heap_sort,
    _insertion_sort,
    _small_sort,
    _SortWrapper,
    sort,
)

# ===-----------------------------------------------------------------------===#
# Benchmark Utils
# ===-----------------------------------------------------------------------===#


@always_inline
fn randomize_list[
    dt: DType
](mut list: List[Scalar[dt]], size: Int, max: Scalar[dt] = Scalar[dt].MAX):
    @parameter
    if dt.is_integral():
        randint(list.data, size, 0, int(max))
    else:
        for i in range(size):
            var res = random_float64()
            # GCC doesn't support cast from float64 to float16
            list[i] = res.cast[DType.float32]().cast[dt]()


@always_inline
fn insertion_sort[type: DType](mut list: List[Scalar[type]]):
    @parameter
    fn _less_than(
        lhs: _SortWrapper[Scalar[type]], rhs: _SortWrapper[Scalar[type]]
    ) -> Bool:
        return lhs.data < rhs.data

    _insertion_sort[_less_than](list)


@always_inline
fn small_sort[size: Int, type: DType](mut list: List[Scalar[type]]):
    @parameter
    fn _less_than(
        lhs: _SortWrapper[Scalar[type]], rhs: _SortWrapper[Scalar[type]]
    ) -> Bool:
        return lhs.data < rhs.data

    _small_sort[size, Scalar[type], _less_than](list.data)


@always_inline
fn heap_sort[type: DType](mut list: List[Scalar[type]]):
    @parameter
    fn _less_than(
        lhs: _SortWrapper[Scalar[type]], rhs: _SortWrapper[Scalar[type]]
    ) -> Bool:
        return lhs.data < rhs.data

    _heap_sort[_less_than](list)


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a tiny list size
# ===-----------------------------------------------------------------------===#


fn bench_tiny_list_sort[type: DType](mut m: Bench) raises:
    alias small_list_size = 5

    @parameter
    for count in range(2, small_list_size + 1):

        @parameter
        fn bench_sort_list(mut b: Bencher) raises:
            seed(1)
            var ptr = UnsafePointer[Scalar[type]].alloc(count)
            var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

            @always_inline
            @parameter
            fn preproc():
                randomize_list(list, count)

            @always_inline
            @parameter
            fn call_fn():
                sort(list)

            b.iter_preproc[call_fn, preproc]()
            _ = list^

        @parameter
        fn bench_small_sort(mut b: Bencher) raises:
            seed(1)
            var ptr = UnsafePointer[Scalar[type]].alloc(count)
            var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

            @always_inline
            @parameter
            fn preproc():
                randomize_list(list, count)

            @always_inline
            @parameter
            fn call_fn():
                small_sort[count](list)

            b.iter_preproc[call_fn, preproc]()
            _ = list^

        @parameter
        fn bench_insertion_sort(mut b: Bencher) raises:
            seed(1)
            var ptr = UnsafePointer[Scalar[type]].alloc(count)
            var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

            @always_inline
            @parameter
            fn preproc():
                randomize_list(list, count)

            @always_inline
            @parameter
            fn call_fn():
                insertion_sort(list)

            b.iter_preproc[call_fn, preproc]()
            _ = list^

        m.bench_function[bench_sort_list](
            BenchId("std_sort_random_" + str(count) + "_" + str(type))
        )
        m.bench_function[bench_small_sort](
            BenchId("sml_sort_random_" + str(count) + "_" + str(type))
        )
        m.bench_function[bench_insertion_sort](
            BenchId("ins_sort_random_" + str(count) + "_" + str(type))
        )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a small list size
# ===-----------------------------------------------------------------------===#


fn bench_small_list_sort[type: DType](mut m: Bench, count: Int) raises:
    @parameter
    fn bench_sort_list(mut b: Bencher) raises:
        seed(1)
        var ptr = UnsafePointer[Scalar[type]].alloc(count)
        var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    @parameter
    fn bench_insertion_sort(mut b: Bencher) raises:
        seed(1)
        var ptr = UnsafePointer[Scalar[type]].alloc(count)
        var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            insertion_sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    m.bench_function[bench_sort_list](
        BenchId("std_sort_random_" + str(count) + "_" + str(type))
    )
    m.bench_function[bench_insertion_sort](
        BenchId("ins_sort_random_" + str(count) + "_" + str(type))
    )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with a large list size
# ===-----------------------------------------------------------------------===#


fn bench_large_list_sort[type: DType](mut m: Bench, count: Int) raises:
    @parameter
    fn bench_sort_list(mut b: Bencher) raises:
        seed(1)
        var ptr = UnsafePointer[Scalar[type]].alloc(count)
        var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    @parameter
    fn bench_heap_sort(mut b: Bencher) raises:
        seed(1)
        var ptr = UnsafePointer[Scalar[type]].alloc(count)
        var list = List[Scalar[type]](ptr=ptr, length=count, capacity=count)

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count)

        @always_inline
        @parameter
        fn call_fn():
            heap_sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    m.bench_function[bench_sort_list](
        BenchId("std_sort_random_" + str(count) + "_" + str(type))
    )

    m.bench_function[bench_heap_sort](
        BenchId("heap_sort_random_" + str(count) + "_" + str(type))
    )


# ===-----------------------------------------------------------------------===#
# Benchmark sort functions with low delta lists
# ===-----------------------------------------------------------------------===#


fn bench_low_cardinality_list_sort(mut m: Bench, count: Int, delta: Int) raises:
    @parameter
    fn bench_sort_list(mut b: Bencher) raises:
        seed(1)
        var ptr = UnsafePointer[UInt8].alloc(count)
        var list = List[UInt8](ptr=ptr, length=count, capacity=count)

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count, delta)

        @always_inline
        @parameter
        fn call_fn():
            sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    @parameter
    fn bench_heap_sort(mut b: Bencher) raises:
        seed(1)
        var ptr = UnsafePointer[UInt8].alloc(count)
        var list = List[UInt8](ptr=ptr, length=count, capacity=count)

        @always_inline
        @parameter
        fn preproc():
            randomize_list(list, count, delta)

        @always_inline
        @parameter
        fn call_fn():
            heap_sort(list)

        b.iter_preproc[call_fn, preproc]()
        _ = list^

    m.bench_function[bench_sort_list](
        BenchId("std_sort_low_card_" + str(count) + "_delta_" + str(delta))
    )
    m.bench_function[bench_heap_sort](
        BenchId("heap_sort_low_card_" + str(count) + "_delta_" + str(delta))
    )


# ===-----------------------------------------------------------------------===#
# Benchmark Main
# ===-----------------------------------------------------------------------===#


def main():
    var m = Bench(BenchConfig(max_runtime_secs=0.1))

    alias dtypes = List(
        DType.uint8,
        DType.uint16,
        DType.float16,
        DType.uint32,
        DType.float32,
        DType.uint64,
        DType.float64,
    )
    var small_counts = List(10, 20, 32, 64, 100)
    var large_counts = List(2**12, 2**16, 2**20)
    var deltas = List(0, 2, 5, 20, 100)

    @parameter
    for i in range(len(dtypes)):
        alias type = dtypes[i]
        bench_tiny_list_sort[type](m)

    @parameter
    for i in range(len(dtypes)):
        alias type = dtypes[i]
        for count in small_counts:
            bench_small_list_sort[type](m, count[])

    @parameter
    for i in range(len(dtypes)):
        alias type = dtypes[i]
        for count in large_counts:
            bench_large_list_sort[type](m, count[])

    for count in large_counts:
        for delta in deltas:
            bench_low_cardinality_list_sort(m, count[], delta[])

    m.dump_report()
