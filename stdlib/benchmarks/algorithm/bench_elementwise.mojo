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

from algorithm import elementwise
from benchmark import Bench, Bencher, BenchId, BenchConfig
from buffer import Buffer
from utils.index import Index, StaticIntTuple


# ===----------------------------------------------------------------------===#
# Benchmark elementwise
# ===----------------------------------------------------------------------===#
@parameter
fn bench_elementwise[n: Int](inout b: Bencher) raises:
    var vector = Buffer[DType.index, n].stack_allocation()

    for i in range(len(vector)):
        vector[i] = -1

    @always_inline
    @parameter
    fn call_fn() raises:
        @always_inline
        @parameter
        fn func[simd_width: Int, rank: Int](idx: StaticIntTuple[rank]):
            vector[idx[0]] = 42

        elementwise[func, 1, 1](Index(n))
        elementwise[func=func, simd_width = simdwidthof[DType.index](), rank=1](
            Index(n)
        )

    b.iter[call_fn]()


fn main() raises:
    var m = Bench(BenchConfig(num_repetitions=1, warmup_iters=10000))
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
