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

from std.iter import chain


def test_chain_ref() raises:
    var l1 = [1, 2, 3]
    var l2 = [1, 2, 3]  # different origins

    var it = chain(l1, l2)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises():
        _ = next(it)  # raises StopIteration

    var s1 = ["hi", "hey", "hello"]
    var s2 = ["hi", "hey", "hello"]  # different origins

    var it2 = chain(s1, s2)
    assert_equal(next(it2), "hi")
    assert_equal(next(it2), "hey")
    assert_equal(next(it2), "hello")
    assert_equal(next(it2), "hi")
    assert_equal(next(it2), "hey")
    assert_equal(next(it2), "hello")
    with assert_raises():
        _ = next(it2)  # raises StopIteration


def test_chain_owned() raises:
    var it = chain(List([1, 2, 3]), List([1, 2, 3]))
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    assert_equal(next(it), 1)
    assert_equal(next(it), 2)
    assert_equal(next(it), 3)
    with assert_raises():
        _ = next(it)  # raises StopIteration

    var it2 = chain(List(["hi", "hey", "hello"]), List(["hi", "hey", "hello"]))
    assert_equal(next(it2), "hi")
    assert_equal(next(it2), "hey")
    assert_equal(next(it2), "hello")
    assert_equal(next(it2), "hi")
    assert_equal(next(it2), "hey")
    assert_equal(next(it2), "hello")
    with assert_raises():
        _ = next(it2)  # raises StopIteration


def test_nested_chain_ref() raises:
    var l1 = [0, 1, 2]
    var l2 = [0, 1, 2]  # different origins
    var s1 = ["0", "1", "2"]
    var s2 = ["0", "1", "2"]  # different origins

    var cl = chain(l1, l2)
    var i = 0
    for elem in chain(cl, l1, l2):
        assert_equal(i % 3, elem)
        i += 1

    var cs = chain(s1, s2)
    var s_i = 0
    for elem in chain(cs, s1, s2):
        assert_equal(String(s_i % 3), elem)
        s_i += 1


def test_nested_chain_owned() raises:
    var cl = chain(List([0, 1, 2]), List([0, 1, 2]))
    var i = 0
    for elem in chain(cl^, List([0, 1, 2]), List([0, 1, 2])):
        assert_equal(i % 3, elem)
        i += 1

    var cs = chain(List(["0", "1", "2"]), List(["0", "1", "2"]))
    var s_i = 0
    for elem in chain(cs^, List(["0", "1", "2"]), List(["0", "1", "2"])):
        assert_equal(String(s_i % 3), elem)
        s_i += 1


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
