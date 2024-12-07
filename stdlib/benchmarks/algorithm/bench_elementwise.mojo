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

from sys import simdwidthof

from algorithm import elementwise
from benchmark import Bench, BenchConfig, Bencher, BenchId
from buffer import Buffer

from utils.index import Index, IndexList


# ===-----------------------------------------------------------------------===#
# Benchmark elementwise
# ===-----------------------------------------------------------------------===#
@parameter
fn bench_elementwise[n: Int](mut b: Bencher) raises:
    var vector = Buffer[DType.index, n].stack_allocation()

    for i in range(len(vector)):
        vector[i] = -1

    @always_inline
    @parameter
    fn call_fn() raises:
        @always_inline
        @parameter
        fn func[simd_width: Int, rank: Int](idx: IndexList[rank]):
            vector[idx[0]] = 42

        elementwise[func, 1](Index(n))
        elementwise[func=func, simd_width = simdwidthof[DType.index]()](
            Index(n)
        )

    b.iter[call_fn]()
    _ = vector


fn main() raises:
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function[bench_elementwise[32]](BenchId("bench_elementwise_32"))
    m.bench_function[bench_elementwise[128]](BenchId("bench_elementwise_128"))
    m.bench_function[bench_elementwise[1024]](BenchId("bench_elementwise_1024"))
    m.bench_function[bench_elementwise[8192]](BenchId("bench_elementwise_8192"))
    m.bench_function[bench_elementwise[32768]](
        BenchId("bench_elementwise_32768")
    )
    m.bench_function[bench_elementwise[131072]](
        BenchId("bench_elementwise_131072")
    )
    m.dump_report()
