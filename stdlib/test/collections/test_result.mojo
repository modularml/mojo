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


fn _returning_err_reg[T: AnyRegType](value: T) raises -> ResultReg[T]:
    var result = ResultReg[T](ErrorReg("something"))
    if not result:
        return result.err
    raise Error("shouldn't get here")


fn _returning_ok_reg[T: AnyRegType](value: T) raises -> ResultReg[T]:
    var result = ResultReg[T](value)
    if result:
        return result
    raise Error("shouldn't get here")


fn _returning_err[T: CollectionElement](value: T) raises -> Result[T]:
    var result = Result[T](Error("something"))
    if not result:
        return result
    raise Error("shouldn't get here")


fn _returning_ok[T: CollectionElement](value: T) raises -> Result[T]:
    var result = Result[T](value)
    if result:
        return result.take()
    raise Error("shouldn't get here")


def test_returning_err():
    var item_i = _returning_err_reg(Int())
    assert_true(not item_i and item_i.err and item_i.err == "something")
    var item_i64 = _returning_err_reg(Int64())
    assert_true(not item_i64 and item_i64.err and item_i64.err == "something")
    var item_f = _returning_err_reg(Float64())
    assert_true(not item_f and item_f.err and item_f.err == "something")
    var item_sl = _returning_err_reg("stringliteral")
    assert_true(not item_sl and item_sl.err and item_sl.err == "something")
    var item_s = _returning_err(String("string"))
    assert_true(not item_s and item_s.err and item_s.err == "something")
    # var item_ti = _returning_err(Tuple[Int]())
    # assert_true(not item_ti and item_ti.err and item_ti.err == "something")
    # var item_ts = _returning_err(Tuple[String]())
    # assert_true(not item_ts and item_ts.err and item_ts.err == "something")
    var item_li = _returning_err(List[Int]())
    assert_true(not item_li and item_li.err and item_li.err == "something")
    var item_ls = _returning_err(List[String]())
    assert_true(not item_ls and item_ls.err and item_ls.err == "something")
    var item_dii = _returning_err(Dict[Int, Int]())
    assert_true(not item_dii and item_dii.err and item_dii.err == "something")
    var item_dss = _returning_err(Dict[String, String]())
    assert_true(not item_dss and item_dss.err and item_dss.err == "something")
    var item_oi = _returning_err(Optional[Int]())
    assert_true(not item_oi and item_oi.err and item_oi.err == "something")
    var item_os = _returning_err(Optional[String]())
    assert_true(not item_os and item_os.err and item_os.err == "something")
    var item_oi64 = _returning_err(OptionalReg[UInt64]())
    assert_true(
        not item_oi64 and item_oi64.err and item_oi64.err == "something"
    )
    var item_osl = _returning_err(OptionalReg[StringLiteral]())
    assert_true(not item_osl and item_osl.err and item_osl.err == "something")


def test_returning_ok():
    var item_i = _returning_ok_reg(Int())
    assert_true(item_i.value() == _returning_ok(Int()).value()[])
    assert_true(item_i and not item_i.err and item_i.err == "")
    var item_i64 = _returning_ok_reg(Int64())
    assert_true(item_i64.value() == _returning_ok(Int64()).value()[])
    assert_true(item_i64 and not item_i64.err and item_i64.err == "")
    var item_f = _returning_ok_reg(Float64())
    assert_true(item_f.value() == _returning_ok(Float64()).value()[])
    assert_true(item_f and not item_f.err and item_f.err == "")
    var item_sl = _returning_ok_reg("stringliteral")
    assert_true(item_sl.value() == _returning_ok("stringliteral").value()[])
    assert_true(item_sl and not item_sl.err and item_sl.err == "")
    var item_s = _returning_ok(String("string"))
    assert_true(item_s and not item_s.err and item_s.err == "")
    # var item_ti = _returning_ok(Tuple[Int]())
    # assert_true(item_ti and not item_ti.err and item_ti.err == "")
    # var item_ts = _returning_ok(Tuple[String]())
    # assert_true(item_ts and not item_ts.err and item_ts.err == "")
    var item_li = _returning_ok(List[Int]())
    assert_true(item_li and not item_li.err and item_li.err == "")
    var item_ls = _returning_ok(List[String]())
    assert_true(item_ls and not item_ls.err and item_ls.err == "")
    var item_dii = _returning_ok(Dict[Int, Int]())
    assert_true(item_dii and not item_dii.err and item_dii.err == "")
    var item_dss = _returning_ok(Dict[String, String]())
    assert_true(item_dss and not item_dss.err and item_dss.err == "")
    var item_oi = _returning_ok(Optional[Int]())
    assert_true(item_oi and not item_oi.err and item_oi.err == "")
    var item_os = _returning_ok(Optional[String]())
    assert_true(item_os and not item_os.err and item_os.err == "")
    var item_oi64 = _returning_ok(OptionalReg[UInt64]())
    assert_true(item_oi64 and not item_oi64.err and item_oi64.err == "")
    var item_osl = _returning_ok(OptionalReg[StringLiteral]())
    assert_true(item_osl and not item_osl.err and item_osl.err == "")


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

    assert_equal(1, a.value()[])

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
