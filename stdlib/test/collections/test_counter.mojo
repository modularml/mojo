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

from collections.counter import Counter

from testing import assert_equal, assert_false, assert_raises, assert_true


def test_counter_construction():
    _ = Counter[Int]()
    _ = Counter[Int](List[Int]())
    _ = Counter[String](List[String]())


def test_counter_getitem():
    c = Counter[Int](List[Int](1, 2, 2, 3, 3, 3, 4))
    assert_equal(c[1], 1)
    assert_equal(c[2], 2)
    assert_equal(c[3], 3)
    assert_equal(c[4], 1)
    assert_equal(c[5], 0)


def test_iter():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    for key in c:
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_keys():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    for key in c.keys():
        keys += key[]

    assert_equal(keys, "ab")


def test_iter_values():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var sum = 0
    for value in c.values():
        sum += value[]

    assert_equal(sum, 3)


def test_iter_values_mut():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    for value in c.values():
        value[] += 1

    assert_equal(2, c["a"])
    assert_equal(3, c["b"])
    assert_equal(2, len(c))


def test_iter_items():
    var c = Counter[String]()
    c["a"] = 1
    c["b"] = 2

    var keys = String("")
    var sum = 0
    for entry in c.items():
        keys += entry[].key
        sum += entry[].value

    assert_equal(keys, "ab")
    assert_equal(sum, 3)


def main():
    test_counter_construction()
    test_counter_getitem()
