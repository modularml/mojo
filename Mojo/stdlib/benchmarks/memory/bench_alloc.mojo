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

from std.benchmark import Bench, BenchConfig, Bencher, BenchId, keep


def bench_allocation(mut b: Bencher) raises:
    @always_inline
    def call_fn():
        for _ in range(10000):
            var allocation = alloc[Int]({count = 100}).into_managed()
            var a = allocation.unsafe_ptr()
            keep(a)

    b.iter(call_fn)


def main() raises:
    var m = Bench(BenchConfig())
    m.bench_function(bench_allocation, BenchId("bench_allocation"))
    m.dump_report()
