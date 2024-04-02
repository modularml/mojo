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
# XFAIL: asan && !system-darwin
# RUN: %mojo %s | FileCheck %s

from memory.unsafe import Pointer
from python._cpython import CPython, PyObjectPtr
from python.object import PythonObject
from python.python import Python
from testing import assert_false, assert_raises, assert_true, assert_equal

from utils import StringRef


fn test_dunder_methods(inout python: Python):
    try:
        var a = PythonObject(34)
        var b = PythonObject(10)

        # __add__
        var c = a + b
        assert_true(c, 44)

        # __add__
        c = a + 100
        assert_true(c, 134)

        # __iadd__
        c += 100
        assert_equal(c, 234)

        # __radd__
        c = 100 + a
        assert_equal(c, 134)

        # __sub__
        c = a - b
        assert_equal(c, 24)

        # __isub__
        c -= 100
        assert_equal(c, -76)

        # __sub__
        c = a - 100
        assert_equal(c, -66)

        # __rsub__
        c = 100 - a
        assert_equal(c, 66)

        # __mul__
        c = a * b
        assert_equal(c, 340)

        # __imul__
        c *= 10
        assert_equal(c, 3400)

        # __mul__
        c = a * 10
        assert_equal(c, 340)

        # __rmul__
        c = 34 * b
        assert_equal(c, 340)

        # __floordiv__
        c = a // b
        assert_equal(c, 3)

        # __ifloordiv__
        c //= 2
        assert_equal(c, 1)

        # __floordiv__
        c = a // 10
        assert_equal(c, 3)

        # __rfloordiv__
        c = 34 // b
        assert_equal(c, 3)

        # __truediv__
        c = a / b
        assert_equal(c, 3.4)

        # __itruediv__
        c /= 2
        assert_equal(c, 1.7)

        # __truediv__
        c = a / 10
        assert_equal(c, 3.4)

        # __rtruediv__
        c = 34 / b
        assert_equal(c, 3.4)

        # __mod__
        c = a % b
        assert_equal(c, 4)

        # __imod__
        c %= 3
        assert_equal(c, 1)

        # __mod__
        c = a % 10
        assert_equal(c, 4)

        # __rmod__
        c = 34 % b
        assert_equal(c, 4)

        # __xor__
        c = a ^ b
        assert_equal(c, 40)

        # __ixor__
        c ^= 15
        assert_equal(c, 39)

        # __xor__
        c = a ^ 10
        assert_equal(c, 40)

        # __rxor__
        c = 34 ^ b
        assert_equal(c, 40)

        # __or__
        c = a | b
        assert_equal(c, 42)

        # __ior__
        c |= 9
        assert_equal(c, 43)

        # __or__
        c = a | 10
        assert_equal(c, 42)

        # __ror__
        c = 34 | b
        assert_equal(c, 42)

        # __and__
        c = a & b
        assert_equal(c, 2)

        # __iand__
        c &= 6
        assert_equal(c, 2)

        # __and__
        c = a & 10
        assert_equal(c, 2)

        # __rand__
        c = 34 & b
        assert_equal(c, 2)

        # __rshift__
        var d = PythonObject(2)
        c = a >> d
        assert_equal(c, 8)

        # __irshift__
        c >>= 2
        assert_equal(c, 2)

        # __rshift__
        c = a >> 2
        assert_equal(c, 8)

        # __rrshift__
        c = 34 >> d
        assert_equal(c, 8)

        # __lshift__
        c = a << d
        assert_equal(c, 136)

        # __ilshift__
        c <<= 1
        assert_equal(c, 272)

        # __lshift__
        c = a << 2
        assert_equal(c, 136)

        # __rlshift__
        c = 34 << d
        assert_equal(c, 136)

        # __pow__
        c = a**d
        assert_equal(c, 1156)

        # __ipow__
        c = 3
        c **= 4
        assert_equal(c, 81)

        # __pow__
        c = a**2
        assert_equal(c, 1156)

        # __rpow__
        c = 34**d
        assert_equal(c, 1156)

        # __lt__
        c = a < b
        assert_false(c)

        # __le__
        c = a <= b
        assert_false(c)

        # __gt__
        c = a > b
        assert_true(c)

        # __ge__
        c = a >= b
        assert_true(c)

        # __eq__
        c = a == b
        assert_false(c)

        # __ne__
        c = a != b
        assert_true(c)

        # __pos__
        c = +a
        assert_equal(c, 34)

        # __neg__
        c = -a
        assert_equal(c, -34)

        # __invert__
        c = ~a
        assert_equal(c, -35)
    except e:
        pass


def test_bool_conversion() -> None:
    var x: PythonObject = 1
    assert_true(x == 0 or x == 1)


fn test_string_conversions() raises -> None:
    fn test_string_literal() -> None:
        try:
            var mojo_str: StringLiteral = "mojo"
            var py_str = PythonObject(mojo_str)
            var py_capitalized = py_str.capitalize()
            var py = Python()
            var mojo_capitalized = py.__str__(py_capitalized)
            assert_equal(mojo_capitalized, "Mojo")
        except e:
            print("Error occurred")

    fn test_string_ref() -> None:
        try:
            var mojo_str: StringLiteral = "mojo"
            var mojo_strref = StringRef(mojo_str)
            var py_str = PythonObject(mojo_strref)
            var py_capitalized = py_str.capitalize()
            var py = Python()
            var mojo_capitalized = py.__str__(py_capitalized)
            assert_equal(mojo_capitalized, "Mojo")
        except e:
            print("Error occurred")

    fn test_string() -> None:
        try:
            var mo_str = String("mo")
            var jo_str = String("jo")
            var mojo_str = mo_str + jo_str
            var py_str = PythonObject(mojo_str)
            var py_capitalized = py_str.capitalize()
            var py = Python()
            var mojo_capitalized = py.__str__(py_capitalized)
            assert_equal(mojo_capitalized, "Mojo")
        except e:
            print("Error occurred")

    fn test_type_object() raises -> None:
        var py = Python()
        var py_float = PythonObject(3.14)
        var type_obj = py.type(py_float)
        assert_equal(str(type_obj), "<class 'float'>")

    test_string_literal()
    test_string_ref()
    test_string()
    test_type_object()


def test_len():
    var empty_list = Python.evaluate("[]")
    assert_equal(len(empty_list), 0)

    var l1 = Python.evaluate("[1,2,3]")
    assert_equal(len(l1), 3)

    var l2 = Python.evaluate("[42,42.0]")
    assert_equal(len(l2), 2)


# CHECK-LABEL: test_is
def test_is():
    print("=== test_is ===")
    var x = PythonObject(500)
    var y = PythonObject(500)
    assert_false(x is y)
    assert_true(x is not y)

    # Assign to a new variable but this still holds
    # the same object and same memory location
    var z = x
    assert_true(z is x)
    assert_false(z is not x)

    # Two separate lists/objects, and therefore are not the "same".
    # as told by the `__is__` function. They point to different addresses.
    var l1 = Python.evaluate("[1,2,3]")
    var l2 = Python.evaluate("[1,2,3]")
    assert_false(l1 is l2)
    assert_true(l1 is not l2)


# CHECK-LABEL: test_iter
fn test_iter() raises:
    print("=== test_iter ===")

    var list_obj: PythonObject = ["apple", "orange", "banana"]
    # CHECK: I like to eat apple
    # CHECK: I like to eat orange
    # CHECK: I like to eat banana
    for fruit in list_obj:
        print("I like to eat", fruit)

    var list2: PythonObject = []
    # CHECK-NOT: I have eaten
    for fruit in list2:
        print("I have eaten", fruit)

    var not_iterable: PythonObject = 3
    with assert_raises():
        for x in not_iterable:
            # CHECK-NOT: I should not exist
            print("I should not exist", x)
            assert_false(True)


fn test_dict() raises:
    var d = Dict[PythonObject, PythonObject]()
    d["food"] = 123
    d["fries"] = "yes"

    var dd = PythonObject(d)
    # CHECK: {'food': 123, 'fries': 'yes'}
    print(dd)


def main():
    # initializing Python instance calls init_python
    var python = Python()

    test_dunder_methods(python)
    test_bool_conversion()
    test_string_conversions()
    test_len()
    test_is()
    test_iter()
    test_dict()
