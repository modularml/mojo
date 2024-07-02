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

from testing import assert_equal


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


fn get_last_value(*values: Int) -> Int:
    return values[-1]


fn get_last_string(*values: String) -> String:
    return values[-1]


fn test_variadic_negative_index() raises:
    var last_value = get_last_value(3, 4, 5)
    # print(last_value)
    assert_equal(5, last_value)

    last_value = get_last_value(6, 7)
    assert_equal(7, last_value)

    last_value = get_last_value(8)
    assert_equal(8, last_value)

    var last_string = get_last_string("abc", "def")
    assert_equal("def", last_string)

    last_string = get_last_string("abc")
    assert_equal("abc", last_string)


def main():
    test_list()
    test_variadic_list()
    test_repr_list()
    test_variadic_negative_index()
