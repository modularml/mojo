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

from collections import Result, ResultReg, Optional, OptionalReg

from testing import assert_true, assert_false, assert_equal


fn _returning_err[T: AnyRegType](value: T) -> ResultReg[T]:
    var result = Result[T](ErroReg("something"))
    if not result:
        return result.err


fn _returning_ok[T: AnyRegType](value: T) -> ResultReg[T]:
    var result = ResultReg[T](value)
    if result:
        return result


fn _returning_err[T: AnyType](value: T) -> Result[T]:
    var result = Result[T](Error("something"))
    if not result:
        return result


fn _returning_ok[T: AnyType](value: T) -> Result[T]:
    var result = Result[T](value)
    if result:
        return result.take()


def test_returning_err():
    var item = _returning_err(Int())
    assert_true(not item and item.err)
    item = _returning_err(Int64())
    assert_true(not item and item.err)
    item = _returning_err(Float64())
    assert_true(not item and item.err)
    item = _returning_err(String())
    assert_true(not item and item.err)
    item = _returning_err(StringLiteral())
    assert_true(not item and item.err)
    item = _returning_err(Tuple[Int]())
    assert_true(not item and item.err)
    item = _returning_err(Tuple[String]())
    assert_true(not item and item.err)
    item = _returning_err(List[Int]())
    assert_true(not item and item.err)
    item = _returning_err(List[String]())
    assert_true(not item and item.err)
    item = _returning_err(Dict[Int, Int]())
    assert_true(not item and item.err)
    item = _returning_err(Dict[String, String]())
    assert_true(not item and item.err)
    item = _returning_err(Optional[Int]())
    assert_true(not item and item.err)
    item = _returning_err(Optional[String]())
    assert_true(not item and item.err)
    item = _returning_err(OptionalReg[UInt64]())
    assert_true(not item and item.err)
    item = _returning_err(OptionalReg[StringLiteral]())
    assert_true(not item and item.err)


def test_returning_ok():
    var item = _returning_ok(Int())
    assert_true(item and not item.err)
    item = _returning_ok(Int64())
    assert_true(item and not item.err)
    item = _returning_ok(Float64())
    assert_true(item and not item.err)
    item = _returning_ok(String())
    assert_true(item and not item.err)
    item = _returning_ok(StringLiteral())
    assert_true(item and not item.err)
    item = _returning_ok(Tuple[Int]())
    assert_true(item and not item.err)
    item = _returning_ok(Tuple[String]())
    assert_true(item and not item.err)
    item = _returning_ok(List[Int]())
    assert_true(item and not item.err)
    item = _returning_ok(List[String]())
    assert_true(item and not item.err)
    item = _returning_ok(Dict[Int, Int]())
    assert_true(item and not item.err)
    item = _returning_ok(Dict[String, String]())
    assert_true(item and not item.err)
    item = _returning_ok(Optional[Int]())
    assert_true(item and not item.err)
    item = _returning_ok(Optional[String]())
    assert_true(item and not item.err)
    item = _returning_ok(OptionalReg[UInt64]())
    assert_true(item and not item.err)
    item = _returning_ok(OptionalReg[StringLiteral]())
    assert_true(item and not item.err)


def test_basic():
    var a = Result(1)
    var b = Result[Int]()

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

    assert_equal(1, a.value()[])

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
    var a2 = Result(1)
    a2.value()[] = 2
    assert_equal(a2.value()[], 2)


def test_optional_reg_basic():
    var val: ResultReg[Int] = ErrorReg("something")
    var val2: Result[Int] = Error("something")
    assert_false(val and val2)

    val = 15
    assert_true(val)

    assert_equal(val.value(), 15)

    assert_true(val or False)
    assert_true(val and True)

    assert_true(False or val)
    assert_true(True and val)


def test_optional_is():
    a = Result(1)
    assert_false(a is None)

    a = Result[Int]()
    assert_true(a is None)


def test_optional_isnot():
    a = Result(1)
    assert_true(a is not None)

    a = Result[Int]()
    assert_false(a is not None)


def test_optional_reg_is():
    a = ResultReg(1)
    assert_false(a is None)

    a = ResultReg[Int]()
    assert_true(a is None)


def test_optional_reg_isnot():
    a = ResultReg(1)
    assert_true(a is not None)

    a = ResultReg[Int]()
    assert_false(a is not None)


def main():
    test_basic()
    test_optional_reg_basic()
    test_optional_is()
    test_optional_isnot()
    test_optional_reg_is()
    test_optional_reg_isnot()
    test_returning_ok()
    test_returning_err()
