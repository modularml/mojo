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


def test_counter_setitem():
    c = Counter[Int]()
    c[1] = 1
    c[2] = 2
    assert_equal(c[1], 1)
    assert_equal(c[2], 2)
    assert_equal(c[3], 0)


def main():
    test_counter_construction()
    test_counter_getitem()
    test_counter_setitem()
