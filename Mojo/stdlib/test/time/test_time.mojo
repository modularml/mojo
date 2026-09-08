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

from std.time import (
    monotonic,
    perf_counter,
    perf_counter_ns,
    sleep,
    time_function,
)

from std.time.time import _CTimeSpec
from std.testing import assert_equal, assert_true, TestSuite


@always_inline
def time_me():
    sleep(1.0)


@always_inline
def time_me_templated[
    dtype: DType,
]():
    time_me()
    return


# Check that time_function works on templated function
def time_templated_function[
    dtype: DType,
]() -> Int:
    return Int(time_function(time_me_templated[dtype]))


def time_capturing_function(iters: Int) -> Int:
    def time_fn():
        sleep(1.0)

    return Int(time_function(time_fn))


def test_time() raises:
    comptime ns_per_sec = 1_000_000_000

    assert_true(perf_counter() > 0)
    assert_true(perf_counter_ns() > 0)
    assert_true(monotonic() > 0)

    var t1 = time_function(time_me)
    assert_true(t1 > 1 * ns_per_sec)
    assert_true(t1 < 10 * ns_per_sec)

    var t2 = time_templated_function[.float32]()
    assert_true(t2 > 1 * ns_per_sec)
    assert_true(t2 < 10 * ns_per_sec)

    var t3 = time_capturing_function(42)
    assert_true(t3 > 1 * ns_per_sec)
    assert_true(t3 < 10 * ns_per_sec)

    # test perf_counter_ns() directly since time_function doesn't use now on windows
    var t4 = perf_counter_ns()
    time_me()
    var t5 = perf_counter_ns()
    assert_true((t5 - t4) > 1 * ns_per_sec)
    assert_true((t5 - t4) < 10 * ns_per_sec)


def test_ctimespec_as_nanoseconds() raises:
    assert_equal(_CTimeSpec().as_nanoseconds(), 0)
    assert_equal(_CTimeSpec(0, 1).as_nanoseconds(), 1)
    assert_equal(_CTimeSpec(1, 0).as_nanoseconds(), 1_000_000_000)
    assert_equal(_CTimeSpec(1, 500_000_000).as_nanoseconds(), 1_500_000_000)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
