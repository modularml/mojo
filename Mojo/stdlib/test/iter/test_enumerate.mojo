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

from std.testing import TestSuite, assert_equal, assert_true, assert_raises


def test_enumerate() raises:
    var l = ["hey", "hi", "hello"]
    var it = enumerate(l)
    var elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], "hey")
    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], "hi")
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], "hello")
    with assert_raises():
        _ = next(it)  # raises StopIteration


def test_enumerate_with_start() raises:
    var l = ["hey", "hi", "hello"]
    var it = enumerate(l, start=1)
    var elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], "hey")
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], "hi")
    elem = next(it)
    assert_equal(elem[0], 3)
    assert_equal(elem[1], "hello")
    with assert_raises():
        _ = next(it)  # raises StopIteration

    # Check negative start
    it = enumerate(l, start=-1)
    elem = next(it)
    assert_equal(elem[0], -1)
    assert_equal(elem[1], "hey")
    elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], "hi")


def test_enumerate_destructure() raises:
    var l = ["hey", "hi", "hello"]
    var count = 0
    for i, elem in enumerate(l):
        assert_equal(i, count)
        assert_equal(elem, l[count])
        count += 1


def test_enumerate_bounds() raises:
    var list = [1, 2, 3]
    var e = enumerate(list)

    assert_equal(iter(list).bounds()[0], e.bounds()[0])
    assert_equal(iter(list).bounds()[1].value(), e.bounds()[1].value())


def test_enumerate_owned() raises:
    var l: List[Int] = [10, 20, 30]
    var it = enumerate(l^)
    var elem = next(it)
    assert_equal(elem[0], 0)
    assert_equal(elem[1], 10)
    elem = next(it)
    assert_equal(elem[0], 1)
    assert_equal(elem[1], 20)
    elem = next(it)
    assert_equal(elem[0], 2)
    assert_equal(elem[1], 30)
    with assert_raises():
        _ = next(it)


def test_enumerate_owned_for_loop() raises:
    var count = 0
    var l: List[Int] = [10, 20, 30]
    for i, _ in enumerate(l^):
        assert_equal(i, count)
        count += 1
    assert_equal(count, 3)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
