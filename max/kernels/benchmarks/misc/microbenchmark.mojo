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

import std.time


trait Benchmarkable:
    def global_pre_run(self):
        """Function that runs once during the start of the entire Benchmark trace.
        """
        ...

    def pre_run(self):
        """Function that runs before the Target Function during every benchmark iteration.
        """
        ...

    def run(self):
        """The target Function that is to be benchmarked."""
        ...

    def post_run(self):
        """Function that runs after the Target Function during every benchmark iteration.
        """
        ...

    def global_post_run(self):
        """Function that runs once during the end of the entire Benchmark trace.
        """
        ...


@inline(.always)
def run[
    T: Benchmarkable
](benchmark_obj: T, name: String, num_iters: Int = 10) -> None:
    benchmark_obj.global_pre_run()

    var total_time = 0.0
    for _ in range(num_iters):
        benchmark_obj.pre_run()
        var start_time = time.perf_counter_ns()
        benchmark_obj.run()
        var end_time = time.perf_counter_ns()
        benchmark_obj.post_run()
        total_time += end_time - start_time

    var average_execution_time = total_time / num_iters / 1e3

    benchmark_obj.global_post_run()
    print(name, average_execution_time)
