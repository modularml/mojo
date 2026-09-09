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

from std.atomic import Atomic
from std.time import time_function

from std.runtime._asyncrt import TaskGroup
from std.testing import assert_equal, TestSuite

from std.utils.lock import BlockingScopedLock, BlockingSpinLock


def test_basic_lock() raises:
    var lock = BlockingSpinLock()
    var rawCounter = 0
    var counter = Atomic[Int64](0)
    comptime maxI = 100
    comptime maxJ = 100

    async def inc(mut blockingLock: BlockingSpinLock) {mut}:
        with BlockingScopedLock(blockingLock):
            rawCounter += 1
            _ = counter.fetch_add(1)

    # CHECK: PRE::Atomic counter is 0 , and raw counter, 0
    print(
        "PRE::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )

    def test_atomic() {mut} -> None:
        var tg = TaskGroup()
        for _ in range(0, maxI):
            for _ in range(0, maxJ):
                tg.create_task(inc(lock))
        tg.wait[origin_of(lock)._mlir_origin]()

    _ = time_function(test_atomic)
    _ = lock^
    # print("Total time taken ", time_ns / (1_000_000_000), " s")

    # CHECK: POST::Atomic counter is 10000 , and raw counter, 10000
    print(
        "POST::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )
    assert_equal(counter.load(), Int64(rawCounter), "atomic stress test failed")

    return


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
