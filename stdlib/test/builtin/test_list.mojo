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
# RUN: %mojo %s

from testing import assert_equal, assert_true, assert_false


fn test_list() raises:
    assert_equal(len([1, 2.0, 3.14, [-1, -2]]), 4)


fn test_variadic_list() raises:
    @parameter
    fn check_list(*nums: Int) raises:
        assert_equal(nums[0], 5)
        assert_equal(nums[1], 8)
        assert_equal(nums[2], 6)
        assert_equal(nums[True], 8)

        assert_equal(len(nums), 3)

    check_list(5, 8, 6)


fn test_repr_list() raises:
    var l = List(1, 2, 3)
    assert_equal(l.__repr__(), "[1, 2, 3]")
    var empty = List[Int]()
    assert_equal(empty.__repr__(), "[]")


fn test_contains() raises:
    # List
    var l = List[String]("Hello", ",", "World", "!")
    assert_true("Hello" in l)
    assert_true(l.__contains__(String(",")))
    assert_true("World" in l)
    assert_true("!" in l)
    assert_false("Mojo" in l)
    assert_false(l.__contains__("hello"))
    assert_false("" in l or l.__contains__(""))

    # ListLiteral
    var h = [1, False, String("Mojo")]
    assert_true(1 in h)
    assert_true(h.__contains__(1))
    assert_false(True in h)
    assert_false(h.__contains__(True))
    assert_false(0 in h)
    assert_true(False in h)
    assert_false("Mojo" in h)
    assert_true(String("Mojo") in h)
    assert_false(String("") in h)
    assert_false("" in h)

    # TODO:
    # Reevaluate the strict type checking behaviour in ListLiteral.__contains__.
    # Consider aligning with the behavior with List.__contains__ when possible
    # or a feasible workaround is identified.
    # For instance, consider the following:
    assert_true("Hello" in l and String("Hello") in l)
    assert_true("Mojo" not in h and String("Mojo") in h)


def main():
    test_list()
    test_variadic_list()
    test_repr_list()
    test_contains()
