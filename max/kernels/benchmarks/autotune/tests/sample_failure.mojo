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
"""Test fixture for kbench shared-lib failure/recovery tests.

Runtime args (passed via $-prefixed YAML params):
  $sleep_secs  – how long each benchmark iteration sleeps (default 0.01).
  $should_crash – if True, raise before benchmarking (default False).
"""

from std.benchmark import Bench, BenchConfig, Bencher, BenchId
from std.sys import stderr
from std.time import sleep
from internal_utils import arg_parse, update_bench_config_args


def bench_func(mut m: Bench, pe_rank: Int, sleep_secs: Float64) raises:
    @always_inline
    def bench_iter(mut b: Bencher) {var sleep_secs}:
        @always_inline
        def call_fn() {var sleep_secs}:
            sleep(sleep_secs)

        b.iter(call_fn)

    m.bench_function(
        bench_iter, BenchId("sleep_test", input_id=String("pe_rank=", pe_rank))
    )


def main() raises:
    var sleep_secs = arg_parse("sleep_secs", 0.01)
    var should_crash = arg_parse("should_crash", False)

    if should_crash:
        print("CRASH: intentional crash for testing", file=stderr)
        raise "intentional crash for testing"

    var m = Bench(BenchConfig(max_iters=1, max_batch_size=1))
    var pe_rank = m.check_mpirun()
    update_bench_config_args(m)
    bench_func(m, pe_rank, sleep_secs)
    m.dump_report()
