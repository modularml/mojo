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

from test_utils import CopyCounter
from std.testing import TestSuite, assert_equal, assert_raises


def test_map() raises:
    var l = [1, 2, 3]

    def add_one(x: Int) -> Int:
        return x + 1

    var m = map[add_one](l)
    assert_equal(next(m), 2)
    assert_equal(next(m), 3)
    assert_equal(next(m), 4)
    with assert_raises():
        _ = m.__next__()  # raises StopIteration

    def to_str(x: Int) -> String:
        return String(x)

    var m2 = map[to_str](l)
    assert_equal(next(m2), "1")
    assert_equal(next(m2), "2")
    assert_equal(next(m2), "3")
    with assert_raises():
        _ = m2.__next__()  # raises StopIteration


def test_map_function_can_take_owned_value() raises:
    def report_copies_owned(var counter: CopyCounter[NoneType]) -> Int:
        return counter.copy_count

    def report_copies_ref(counter: CopyCounter[NoneType]) -> Int:
        return counter.copy_count

    # FIXME: fix compiler, just to make progress.
    var list = [CopyCounter[NoneType](None)]

    # ensure the number of copies are equal between an "owned" and
    # "borrowed" mapping function.
    var m1 = map[report_copies_owned](list)
    var m2 = map[report_copies_ref](list)

    assert_equal(next(m1), next(m2))


def test_map_owned() raises:
    def double(x: Int) -> Int:
        return x * 2

    var l: List[Int] = [1, 2, 3]
    var m = map[double](l^)
    assert_equal(next(m), 2)
    assert_equal(next(m), 4)
    assert_equal(next(m), 6)
    with assert_raises():
        _ = next(m)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
