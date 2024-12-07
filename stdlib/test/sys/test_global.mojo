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

from sys.ffi import _Global
from testing import assert_equal, assert_false, assert_true


struct TestCounter(CollectionElement):
    var copied: Int
    var moved: Int

    fn __init__(out self):
        self.copied = 0
        self.moved = 0

    fn __init__(out self, *, other: Self):
        self = other

    fn __copyinit__(out self, other: Self):
        self.copied = other.copied + 1
        self.moved = other.moved

    fn __moveinit__(out self, owned other: Self):
        self.copied = other.copied
        self.moved = other.moved + 1


alias TEST_GLOBAL = _Global["_TEST_GLOBAL", TestCounter, _initialize_counter]


fn _initialize_counter() -> TestCounter:
    return TestCounter()


def test_global():
    assert_equal(0, TEST_GLOBAL.get_or_create_ptr()[].moved)
    assert_equal(0, TEST_GLOBAL.get_or_create_ptr()[].copied)
    b = TEST_GLOBAL.get_or_create_ptr()[]
    assert_equal(1, b.copied)
    assert_equal(0, b.moved)


struct NonMovableStruct:
    var value: Int

    fn __init__(out self, arg: Int):
        self.value = arg

    @staticmethod
    fn initialize() -> Self:
        return Self(0)


alias TEST_GLOBAL_NON_MOVABLE = _Global[
    "_TEST_GLOBAL_NON_MOVABLE", NonMovableStruct, NonMovableStruct.initialize
]


def test_global_non_movable():
    assert_equal(0, TEST_GLOBAL_NON_MOVABLE.get_or_create_ptr()[].value)


def main():
    test_global()
    test_global_non_movable()
