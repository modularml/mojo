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

from collections import Optional, OptionalReg

from testing import assert_equal, assert_false, assert_true


def test_basic():
    var a = Optional(1)
    var b = Optional[Int](None)

    assert_true(a)
    assert_false(b)

    assert_true(a and True)
    assert_true(True and a)
    assert_false(a and False)

    assert_false(b and True)
    assert_false(b and False)

    assert_true(a or True)
    assert_true(a or False)

    assert_true(b or True)
    assert_false(b or False)

    assert_equal(1, a.value())

    # Test invert operator
    assert_false(~a)
    assert_true(~b)

    # TODO(27776): can't inline these, they need to be mutable lvalues
    var a1 = a.or_else(2)
    var b1 = b.or_else(2)

    assert_equal(1, a1)
    assert_equal(2, b1)

    assert_equal(1, a.unsafe_take())

    # TODO: this currently only checks for mutable references.
    # We may want to come back and add an immutable test once
    # there are the language features to do so.
    var a2 = Optional(1)
    a2.value() = 2
    assert_equal(a2.value(), 2)


def test_optional_reg_basic():
    var val: OptionalReg[Int] = None
    assert_false(val.__bool__())

    val = 15
    assert_true(val.__bool__())

    assert_equal(val.value(), 15)

    assert_true(val or False)
    assert_true(val and True)

    assert_true(False or val)
    assert_true(True and val)


def test_optional_is():
    a = Optional(1)
    assert_false(a is None)

    a = Optional[Int](None)
    assert_true(a is None)


def test_optional_isnot():
    a = Optional(1)
    assert_true(a is not None)

    a = Optional[Int](None)
    assert_false(a is not None)


def test_optional_reg_is():
    a = OptionalReg(1)
    assert_false(a is None)

    a = OptionalReg[Int](None)
    assert_true(a is None)


def test_optional_reg_isnot():
    a = OptionalReg(1)
    assert_true(a is not None)

    a = OptionalReg[Int](None)
    assert_false(a is not None)


def test_optional_take_mutates():
    var opt1 = Optional[Int](5)

    assert_true(opt1)

    var value: Int = opt1.take()

    assert_equal(value, 5)
    # The optional should now be empty
    assert_false(opt1)


def main():
    test_basic()
    test_optional_reg_basic()
    test_optional_is()
    test_optional_isnot()
    test_optional_reg_is()
    test_optional_reg_isnot()
    test_optional_take_mutates()
