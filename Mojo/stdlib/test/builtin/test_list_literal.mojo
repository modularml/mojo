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

from std.testing import assert_equal, assert_false, assert_true, TestSuite


def test_variadic_list() raises:
    def check_list(*nums: Int) raises:
        assert_equal(nums[0], 5)
        assert_equal(nums[1], 8)
        assert_equal(nums[2], 6)

        assert_equal(len(nums), 3)

    check_list(5, 8, 6)


def test_contains() raises:
    # There are additional tests for `List.__contains__` in the `test_list.mojo` file.
    var l = ["Hello", ",", "World", "!"]
    assert_true("Hello" in l)
    assert_true(l.__contains__(","))
    assert_true("World" in l)
    assert_true("!" in l)
    assert_false("Mojo" in l)
    assert_false(l.__contains__("hello"))
    assert_false("" in l or l.__contains__(""))
    assert_true("Hello" in l and "Hello" in l)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
