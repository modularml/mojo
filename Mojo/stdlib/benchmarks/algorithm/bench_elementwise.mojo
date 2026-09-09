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

from std.sys import simd_width_of

from max.algorithm import elementwise
from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from max.gpu.host import DeviceContext

from std.utils.coord import Coord
from std.utils.index import IndexList


# ===-----------------------------------------------------------------------===#
# Benchmark elementwise
# ===-----------------------------------------------------------------------===#
def bench_elementwise[n: Int](mut b: Bencher) raises:
    var vector = Array[Int, n](fill=-1)

    @always_inline
    def call_fn() raises {mut vector}:
        @always_inline
        @__parameter
        def func[simd_width: Int, alignment: Int = 1](idx: Coord):
            vector[Int(idx[0].value())] = 42

        elementwise[func=func, simd_width=simd_width_of[DType.int]()](
            Coord(IndexList[1](n)), DeviceContext(api="cpu")
        )

    b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig(num_repetitions=1))
    m.bench_function(bench_elementwise[32], BenchId("bench_elementwise_32"))
    m.bench_function(bench_elementwise[128], BenchId("bench_elementwise_128"))
    m.bench_function(bench_elementwise[1024], BenchId("bench_elementwise_1024"))
    m.bench_function(bench_elementwise[8192], BenchId("bench_elementwise_8192"))
    m.bench_function(
        bench_elementwise[32768], BenchId("bench_elementwise_32768")
    )
    m.bench_function(
        bench_elementwise[131072], BenchId("bench_elementwise_131072")
    )
    m.dump_report()
