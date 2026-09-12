# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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


from testing import TestSuite, assert_true, assert_false, assert_equal
from collections.result import Result


fn _returning_err[T: Copyable & Movable](value: T) raises -> Result[T]:
    var result = Result[T](err=Error("something"))
    if not result:
        return result
    raise Error("shouldn't get here")


fn _returning_ok[T: Copyable & Movable](var value: T) raises -> Result[T]:
    var result = Result[T](value^)
    if result:
        return result
    raise Error("shouldn't get here")


fn _returning_transferred_err[
    T: Copyable & Movable
](value: T) raises -> Result[T]:
    var res1 = Result[String](err=Error("some error"))
    if not res1:
        return res1
    raise Error("shouldn't get here")


fn _returning_none_err[T: Copyable & Movable](value: T) raises -> Result[T]:
    var res1 = Result[String](err=Error("some error"))
    if not res1:
        return None, res1.err_take()
    raise Error("shouldn't get here")


def test_none_err_constructor():
    var res1 = _returning_none_err(String("some string"))
    assert_true(
        not res1
        and res1.err_value()
        and String(res1.err_value()) == "some error"
    )
    var res2 = _returning_none_err[String]("some string")
    assert_true(
        not res2
        and res2.err_value()
        and String(res2.err_value()) == "some error"
    )
    var res3 = _returning_none_err[StaticString]("some string")
    assert_true(
        not res3
        and res3.err_value()
        and String(res3.err_value()) == "some error"
    )
    var res4 = _returning_none_err("some string")
    assert_true(
        not res4
        and res4.err_value()
        and String(res4.err_value()) == "some error"
    )


def test_error_transfer():
    var res1 = _returning_transferred_err(String("some string"))
    assert_true(res1 is None and String(res1.err_value()) == "some error")
    var res2 = _returning_transferred_err[String]("some string")
    assert_true(res2 is None and String(res2.err_value()) == "some error")
    var res3 = _returning_transferred_err[StaticString]("some string")
    assert_true(res3 is None and String(res3.err_value()) == "some error")
    var res4 = _returning_transferred_err("some string")
    assert_true(res4 is None and String(res4.err_value()) == "some error")


def test_returning_err():
    var item_i = _returning_err(Int())
    assert_true(
        not item_i
        and item_i.err_value()
        and String(item_i.err_value()) == "something"
    )
    var item_i64 = _returning_err(Int64())
    assert_true(
        not item_i64
        and item_i64.err_value()
        and String(item_i64.err_value()) == "something"
    )
    var item_f = _returning_err(Float64())
    assert_true(
        not item_f
        and item_f.err_value()
        and String(item_f.err_value()) == "something"
    )
    var item_sl = _returning_err("StaticString")
    assert_true(
        not item_sl
        and item_sl.err_value()
        and String(item_sl.err_value()) == "something"
    )
    var item_s = _returning_err(String("string"))
    assert_true(
        not item_s
        and item_s.err_value()
        and String(item_s.err_value()) == "something"
    )
    var item_ti = _returning_err(Tuple[Int]())
    assert_true(
        not item_ti
        and item_ti.err_value()
        and String(item_ti.err_value()) == "something"
    )
    var item_ts = _returning_err(Tuple[String]())
    assert_true(
        not item_ts
        and item_ts.err_value()
        and String(item_ts.err_value()) == "something"
    )
    var item_li = _returning_err(List[Int]())
    assert_true(
        not item_li
        and item_li.err_value()
        and String(item_li.err_value()) == "something"
    )
    var item_ls = _returning_err(List[String]())
    assert_true(
        not item_ls
        and item_ls.err_value()
        and String(item_ls.err_value()) == "something"
    )
    var item_dii = _returning_err(Dict[Int, Int]())
    assert_true(
        not item_dii
        and item_dii.err_value()
        and String(item_dii.err_value()) == "something"
    )
    var item_dss = _returning_err(Dict[String, String]())
    assert_true(
        not item_dss
        and item_dss.err_value()
        and String(item_dss.err_value()) == "something"
    )
    var item_oi = _returning_err(Result[Int]())
    assert_true(
        not item_oi
        and item_oi.err_value()
        and String(item_oi.err_value()) == "something"
    )
    var item_os = _returning_err(Result[String]())
    assert_true(
        not item_os
        and item_os.err_value()
        and String(item_os.err_value()) == "something"
    )
    var item_oi64 = _returning_err(Result[UInt64]())
    assert_true(
        not item_oi64
        and item_oi64.err_value()
        and String(item_oi64.err_value()) == "something"
    )
    var item_osl = _returning_err(Result[StaticString]())
    assert_true(
        not item_osl
        and item_osl.err_value()
        and String(item_osl.err_value()) == "something"
    )


def test_returning_ok():
    assert_true(_returning_ok(Int()))
    assert_true(_returning_ok(Int64()))
    assert_true(_returning_ok(Float64()))
    assert_true(_returning_ok("StaticString"))
    # this one would fail if the String gets implicitly cast to Error(src: String)
    assert_true(_returning_ok(String("string")))
    assert_true(_returning_ok(Tuple[Int]()))
    assert_true(_returning_ok(Tuple[String]()))
    assert_true(_returning_ok(List[Int]()))
    assert_true(_returning_ok(List[String]()))
    assert_true(_returning_ok(Dict[Int, Int]()))
    assert_true(_returning_ok(Dict[String, String]()))
    assert_true(_returning_ok(Result[Int]()))
    assert_true(_returning_ok(Result[String]()))
    assert_true(_returning_ok(Result[UInt64]()))
    assert_true(_returning_ok(Result[StaticString]()))


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

    assert_equal(1, a.value())

    # Test invert operator
    assert_false(~a)
    assert_true(~b)

    # TODO(27776): can't inline these, they need to be mutable lvalues
    var a1 = a.or_else(2)
    var b1 = b.or_else(2)

    assert_equal(1, a1)
    assert_equal(2, b1)

    assert_equal(1, a.value())

    # TODO: this currently only checks for mutable references.
    # We may want to come back and add an immutable test once
    # there are the language features to do so.
    var a2 = Result(1)
    a2.value() = 2
    assert_equal(a2.value(), 2)


def test_result_is():
    var a = Result(1)
    assert_false(a is None)

    a = Result[Int]()
    assert_true(a is None)


def test_result_isnot():
    var a = Result(1)
    assert_true(a is not None)

    a = Result[Int]()
    assert_false(a is not None)


def test_result_with_same_value_and_err_types():
    assert_true(Result[String, String](value="correct"))
    assert_false(Result[String, String](value="incorrect"))
    assert_equal(String(Result[String, String](value="incorrect")), "incorrect")
    assert_equal(String(Result[String, String](value="correct")), "correct")


def main():
    TestSuite.discover_tests[__functions_in_module()]().run()
