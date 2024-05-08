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

from collections import InlineList, Set
from testing import assert_equal, assert_false, assert_true, assert_raises


def test_mojo_issue_698():
    var list = InlineList[Float64]()
    for i in range(5):
        list.append(i)

    assert_equal(0.0, list[0])
    assert_equal(1.0, list[1])
    assert_equal(2.0, list[2])
    assert_equal(3.0, list[3])
    assert_equal(4.0, list[4])


def test_list():
    var list = InlineList[Int]()

    for i in range(5):
        list.append(i)

    assert_equal(5, len(list))
    assert_equal(0, list[0])
    assert_equal(1, list[1])
    assert_equal(2, list[2])
    assert_equal(3, list[3])
    assert_equal(4, list[4])

    assert_equal(0, list[-5])
    assert_equal(3, list[-2])
    assert_equal(4, list[-1])

    list[2] = -2
    assert_equal(-2, list[2])

    list[-5] = 5
    assert_equal(5, list[-5])
    list[-2] = 3
    assert_equal(3, list[-2])
    list[-1] = 7
    assert_equal(7, list[-1])


@value
struct ValueToCountDestructor(CollectionElement):
    var value: Int
    var destructor_counter: UnsafePointer[List[Int]]

    fn __del__(owned self):
        self.destructor_counter[].append(self.value)


def test_append_and_destructor():
    var destructor_counter = List[Int]()
    var inline_list = InlineList[ValueToCountDestructor, capacity=32]()

    var nb_elements_to_add = 8
    for index in range(nb_elements_to_add):
        inline_list.append(
            ValueToCountDestructor(index, UnsafePointer(destructor_counter))
        )

    # Using .append() should trigger a move and not a copy+delete.
    assert_equal(len(destructor_counter), 0)

    assert_equal(len(inline_list), nb_elements_to_add)

    # this is the last use of the inline list, so it should be destroyed here, along with each element.
    assert_equal(
        len(destructor_counter), nb_elements_to_add
    )  # It's important that it's not 32, which is the capacity.
    for i in range(nb_elements_to_add):
        assert_equal(destructor_counter[i], i)


def main():
    test_mojo_issue_698()
    test_list()
    test_append_and_destructor()
