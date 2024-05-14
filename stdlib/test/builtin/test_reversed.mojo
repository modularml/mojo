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


def test_reversed_list():
    var list = List[Int](1, 2, 3, 4, 5, 6)
    var check: Int = 6

    for item in reversed(list):
        assert_equal(item[], check, "item[], check")
        check -= 1


def test_reversed_dict():
    var dict = Dict[String, Int]()
    dict["a"] = 1
    dict["b"] = 2
    dict["c"] = 3
    dict["d"] = 4
    dict["a"] = 1

    var keys = String("")
    for key in reversed(dict):
        keys += key[]

    assert_equal(keys, "dcba")

    var check: Int = 4
    for val in reversed(dict.values()):
        assert_equal(val[], check)
        check -= 1

    keys = String("")
    check = 4
    for item in reversed(dict.items()):
        keys += item[].key
        assert_equal(item[].value, check)
        check -= 1

    assert_equal(keys, "dcba")

    # Order preserved

    _ = dict.pop("a")
    _ = dict.pop("c")

    # dict: {'b': 2, 'd': 4}

    keys = String("")
    for key in dict:
        keys += key[]

    assert_equal(keys, "bd")

    keys = String("")
    for key in reversed(dict):
        keys += key[]

    assert_equal(keys, "db")

    # got 4 and 2
    check = 4
    for val in reversed(dict.values()):
        assert_equal(val[], check)
        check -= 2

    keys = String("")
    check = 4
    for item in reversed(dict.items()):
        keys += item[].key
        assert_equal(item[].value, check)
        check -= 2

    assert_equal(keys, "db")

    # Refill dict
    dict["c"] = 2
    dict["a"] = 1
    dict["b"] = 4
    dict["d"] = 3

    # dict: {'b': 4, 'd': 3, 'c': 2, 'a': 1}

    keys = String("")
    check = 1
    for item in reversed(dict.items()):
        keys += item[].key
        assert_equal(item[].value, check)
        check += 1

    assert_equal(keys, "acdb")

    # Empty dict is iterable

    var empty_dict = Dict[String, Int]()

    keys = String("")
    for key in reversed(empty_dict):
        keys += key[]

    assert_equal(keys, "")

    check = 0
    for val in reversed(empty_dict.values()):
        # values is empty, should not reach here
        check += 1

    assert_equal(check, 0)

    keys = String("")
    check = 0
    for item in reversed(empty_dict.items()):
        keys += item[].key
        check += item[].value

    assert_equal(keys, "")
    assert_equal(check, 0)


def main():
    test_reversed_dict()
    test_reversed_list()
