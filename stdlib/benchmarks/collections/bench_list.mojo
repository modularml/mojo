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
from random import seed
from collections import List
from random import *


# ===-----------------------------------------------------------------------===#
# Benchmark Data
# ===-----------------------------------------------------------------------===#
fn make_list[
    size: Int, DT: DType, is_trivial: Bool
]() -> List[Scalar[DT], is_trivial]:
    alias scalar_t = Scalar[DT]
    var d = List[Scalar[DT], is_trivial](capacity=size)
    rand[DT](
        d.unsafe_ptr(),
        size,
        min=scalar_t.MIN.cast[DType.float64](),
        max=scalar_t.MAX.cast[DType.float64](),
    )
    d.size = size
    return d


# ===-----------------------------------------------------------------------===#
# Benchmark `List[DT, True].__copyinit__`
# ===-----------------------------------------------------------------------===#


@parameter
fn bench_list_copyinit[
    size: Int, DT: DType, is_trivial: Bool
](mut b: Bencher) raises:
    var items = make_list[size, DT, is_trivial]()
    var result = List[Scalar[DT], is_trivial]()
    var res = 0

    @always_inline
    @parameter
    fn call_fn() raises:
        result = items
        res += len(result)
        keep(result.data)
        keep(items.data)

    b.iter[call_fn]()
    print(res)
    keep(bool(items))
    keep(bool(result))
    keep(result.data)
    keep(items.data)


def main():
    var m = Bench(
        BenchConfig(
            num_repetitions=1,
            max_runtime_secs=0.5,
            min_runtime_secs=0.25,
            min_warmuptime_secs=0,  # FIXME: adjust the values
        )
    )
    alias lengths = (1, 2, 4, 8, 16, 32, 128, 256, 512, 1024, 2048, 4096)

    @parameter
    for i in range(len(lengths)):
        alias length = lengths.get[i, Int]()
        m.bench_function[bench_list_copyinit[length, DType.uint8, True]](
            BenchId("List[Scalar[DT], True].__copyinit__ [" + str(length) + "]")
        )
        m.bench_function[bench_list_copyinit[length, DType.uint8, False]](
            BenchId(
                "List[Scalar[DT], False].__copyinit__ [" + str(length) + "]"
            )
        )

    m.dump_report()
