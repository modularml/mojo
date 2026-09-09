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

from std.sys.info import num_physical_cores

from max.algorithm import parallelize, sync_parallelize
from std.benchmark import Bench, Bencher, BenchId, keep
from std.testing import assert_true


def bench_empty_sync_parallelize(mut b: Bencher) raises:
    @always_inline
    def parallel_fn(thread_id: Int):
        keep(thread_id)

    sync_parallelize(parallel_fn, num_physical_cores())


def bench_empty_parallelize(mut b: Bencher) raises:
    @always_inline
    def parallel_fn(thread_id: Int):
        keep(thread_id)

    parallelize(parallel_fn, num_physical_cores())


def main() raises:
    var m = Bench()
    m.bench_function(bench_empty_sync_parallelize, BenchId("sync_parallelize"))
    m.bench_function(bench_empty_parallelize, BenchId("parallelize"))
    assert_true("No benchmarks recorded..." in String(m))
