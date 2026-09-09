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


@fieldwise_init
struct ConditionalTimer(ImplicitlyCopyable):
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

    def __exit__(mut self, e: Error) -> Bool:
        if String(e) == "just a warning":
            print("Suppressing error:", e)
            self.__exit__()
            return True
        else:
            print("Propagating error")
            self.__exit__()
            return False


def flaky_identity(n: Int) raises -> Int:
    if (n % 4) == 0:
        raise "really bad"
    elif (n % 2) == 0:
        raise "just a warning"
    else:
        return n


def main() raises:
    for i in range(1, 9):
        with ConditionalTimer():
            print("\nBeginning execution")

            print("i =", i)
            std.time.sleep(0.1)

            if i == 3:
                print("continue executed")
                continue

            var j = flaky_identity(i)
            print("j =", j)

            print("Ending execution")
