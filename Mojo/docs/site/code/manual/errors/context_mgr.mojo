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
import std.sys
import std.time


@fieldwise_init
struct Timer(ImplicitlyCopyable):
    var start_time: Int

    def __init__(out self):
        self.start_time = 0

    def __enter__(mut self) -> Self:
        self.start_time = std.time.perf_counter_ns()
        return self

    def __exit__(mut self):
        var end_time = std.time.perf_counter_ns()
        var elapsed_time_ms = round(
            Float64(end_time - self.start_time) / 1e6, 3
        )
        print("Elapsed time:", elapsed_time_ms, "milliseconds")


def main() raises:
    with Timer():
        print("Beginning execution")
        std.time.sleep(1.0)
        if len(std.sys.argv()) > 1:
            raise "simulated error"
        std.time.sleep(1.0)
        print("Ending execution")
