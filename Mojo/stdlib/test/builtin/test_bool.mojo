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

from std.python import PythonObject
from std.testing import assert_equal, assert_false, assert_true, TestSuite


def test_default() raises:
    assert_equal(Bool(), False)


def test_min_max() raises:
    assert_equal(Bool.MIN, False)
    assert_equal(Bool.MAX, True)


def test_bool_cast_to_int() raises:
    assert_equal(False.__int__(), 0)
    assert_equal(True.__int__(), 1)

    assert_equal(Int(False), 0)
    assert_equal(Int(True), 1)


def test_bool_none() raises:
    var test = None
    assert_equal(Bool(None), False)
    assert_equal(Bool(test), False)


@fieldwise_init
struct MyTrue(Boolable):
    def __bool__(self) -> Bool:
        return True


def test_convert_from_boolable() raises:
    assert_true(Bool(MyTrue()))


def test_bool_representation() raises:
    assert_equal(repr(True), "True")
    assert_equal(repr(False), "False")


def test_bitwise() raises:
    var value: Bool

    # and
    value = False
    value &= False
    assert_false(value)
    value = False
    value &= True
    assert_false(value)
    value = True
    value &= False
    assert_false(value)
    value = True
    value &= True
    assert_true(value)

    # or
    value = False
    value |= False
    assert_false(value)
    value = False
    value |= True
    assert_true(value)
    value = True
    value |= False
    assert_true(value)
    value = True
    value |= True
    assert_true(value)

    # xor
    value = False
    value ^= False
    assert_false(value)
    value = False
    value ^= True
    assert_true(value)
    value = True
    value ^= False
    assert_true(value)
    value = True
    value ^= True
    assert_false(value)


def test_comparisons() raises:
    assert_true(False == False)
    assert_true(True == True)
    assert_false(False == True)
    assert_false(True == False)

    assert_true(False != True)
    assert_true(True != False)
    assert_false(False != False)
    assert_false(True != True)

    assert_true(True > False)
    assert_false(False > True)
    assert_false(False > False)
    assert_false(True > True)

    assert_true(True >= False)
    assert_false(False >= True)
    assert_true(False >= False)
    assert_true(True >= True)

    assert_false(True < False)
    assert_true(False < True)
    assert_false(False < False)
    assert_false(True < True)

    assert_false(True <= False)
    assert_true(False <= True)
    assert_true(False <= False)
    assert_true(True <= True)


def test_float_conversion() raises:
    assert_equal((True).__float__(), 1.0)
    assert_equal((False).__float__(), 0.0)


def test_all() raises:
    assert_true(all([True, True, True]))
    assert_false(all({True, False, True}))
    # empty
    assert_true(all(List[Int]()))

    def gt0(x: Int) -> Bool:
        return x > 0

    var l = [1, 2, 3]
    assert_true(all(map[gt0](l)))
    var l2 = [-1, 2, 3]
    assert_false(all(map[gt0](l2)))


def test_any() raises:
    assert_true(any([True, True, True]))
    assert_false(any([False, False, False]))
    assert_false(any({False}))
    # empty
    assert_false(any(List[Int]()))

    def gt0(x: Int) -> Bool:
        return x > 0

    var l = [1, 2, 3]
    assert_true(any(map[gt0](l)))
    var l2 = [-1, -2, -3]
    assert_false(any(map[gt0](l2)))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
