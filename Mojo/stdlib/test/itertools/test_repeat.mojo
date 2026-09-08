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

from std.itertools import repeat
from std.testing import (
    TestSuite,
    assert_equal,
    assert_false,
    assert_true,
    assert_raises,
)


def test_repeat_finite() raises:
    """Tests repeat with a finite number of times."""
    var it = repeat(42, times=3)

    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    assert_equal(next(it), 42)
    with assert_raises():
        _ = next(it)  # raises StopIteration


def test_repeat_string() raises:
    """Tests repeat with a string value."""
    var it = repeat("hello", times=5)

    var count = 0
    for val in it:
        assert_equal(val, "hello")
        count += 1

    assert_equal(count, 5)


def test_repeat_zero_times() raises:
    """Tests repeat with zero repetitions."""
    var it = repeat(99, times=0)
    with assert_raises():
        _ = next(it)  # raises StopIteration


def test_repeat_one_time() raises:
    """Tests repeat with a single repetition."""
    var it = repeat(7, times=1)

    assert_equal(next(it), 7)
    with assert_raises():
        _ = next(it)  # raises StopIteration


def test_repeat_large_count() raises:
    """Tests repeat with a large number of repetitions."""
    var it = repeat(123, times=1000)

    var count = 0
    for val in it:
        assert_equal(val, 123)
        count += 1

    assert_equal(count, 1000)


def test_repeat_in_for_loop() raises:
    """Tests repeat iterator in a for loop."""
    var sum = 0
    for val in repeat(10, times=5):
        sum += val

    assert_equal(sum, 50)


def test_repeat_param() raises:
    """Tests repeat with parameter for loop."""
    var trip_count = 0

    comptime for val in repeat(42, times=3):
        assert_equal(val, 42)
        trip_count += 1

    assert_equal(trip_count, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
