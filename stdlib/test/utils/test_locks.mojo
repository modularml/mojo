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
# RUN: %mojo %s | FileCheck %s


from os import Atomic
from time import now, sleep, time_function

from utils.lock import SpinWaiter, BlockingSpinLock, BlockingScopedLock
from runtime.llcl import TaskGroup
from testing import assert_equal, assert_true


# CHECK-LABEL: test_spin_waiter
def test_spin_waiter():
    print("== test_spin_waiter")
    var waiter = SpinWaiter()
    alias RUNS = 1000
    for i in range(RUNS):
        waiter.wait()
    assert_true(True)


fn test_basic_lock() raises:
    var lock = BlockingSpinLock()
    var rawCounter = 0
    var counter = Atomic[DType.int64](False)
    alias maxI = 100
    alias maxJ = 100

    @parameter
    async fn inc() capturing:
        with BlockingScopedLock(lock):
            rawCounter += 1
            _ = counter.fetch_add(1)

    # CHECK: PRE::Atomic counter is 0 , and raw counter, 0
    print(
        "PRE::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )

    @parameter
    fn test_atomic() capturing -> None:
        var tg = TaskGroup[__lifetime_of()]()
        for i in range(0, maxI):
            for j in range(0, maxJ):
                tg.create_task(inc())
        tg.wait()

    var time_ns = time_function[test_atomic]()
    _ = lock^
    # print("Total time taken ", time_ns / (1_000_000_000), " s")

    # CHECK: POST::Atomic counter is 10000 , and raw counter, 10000
    print(
        "POST::Atomic counter is ",
        counter.load(),
        ", and raw counter, ",
        rawCounter,
    )
    assert_equal(counter.load(), rawCounter, "atomic stress test failed")

    return


def main():
    test_spin_waiter()
    test_basic_lock()
