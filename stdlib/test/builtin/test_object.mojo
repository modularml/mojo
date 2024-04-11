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
# RUN: %mojo %s | FileCheck %s

from random import random_float64

from testing import assert_equal, assert_false, assert_true


def test_object_ctors():
    a = object()
    assert_true(a._value.is_none())
    a = 5
    assert_true(a._value.is_int())
    assert_equal(a._value.get_as_int(), 5)
    a = 6.5
    assert_true(a._value.is_float())
    assert_equal(a._value.get_as_float(), 6.5)
    a = False
    assert_true(a._value.is_bool())
    assert_false(a._value.get_as_bool())

    a = "foobar"
    assert_true(a)
    a = ""
    assert_false(a)
    a = []
    assert_false(a)
    a = [1, 2]
    assert_true(a)
    b = object([2, 4])
    assert_true(a < b)


def test_comparison_ops():
    assert_true(object(False) < True)
    assert_false(object(True) < True)
    assert_true(object(1) > False)
    assert_true(object(2) == 2)
    assert_false(object(True) != 1)
    assert_true(object(True) <= 1.0)
    assert_false(object(False) >= 0.5)

    lhs = object("aaa")
    rhs = object("bbb")
    assert_false(lhs == rhs)
    assert_true(lhs != rhs)
    assert_true(lhs < rhs)
    assert_true(lhs <= rhs)
    assert_false(lhs > rhs)
    assert_false(lhs >= rhs)


def test_arithmetic_ops():
    a = object(False)
    a += True
    assert_true(a == True)

    a = object(1)
    a -= 5.5
    assert_true(a == -4.5)

    a = object(2.5)
    a *= 2

    assert_true(a == 5)
    assert_true(-object(True) == -1)
    assert_true(~object(5) == -6)
    assert_true((object(True) + True) == 2)
    assert_true(5 - object(6) == -1)

    assert_false(object(False) and True)
    assert_true(object(False) or True)

    assert_true(object(5) ** 2 == 25)
    assert_true(5 ** object(2) == 25)
    assert_true(object(5) ** object(2) == 25)
    assert_true(object(4.5) ** 2 == 20.25)

    a = 5
    a **= 2
    assert_true(a == 25)

    lhs = object("foo")
    rhs = object("bar")
    concatted = lhs + rhs
    lhs += rhs
    assert_true(lhs == concatted)


def test_function(borrowed lhs, borrowed rhs) -> object:
    return lhs + rhs


def test_function_raises(borrowed a) -> object:
    raise Error("Error from function type")


def test_object_function():
    var a: object = test_function
    print(a)
    print(a(1, 2))
    a = test_function_raises
    try:
        a(1)
    except e:
        print(e)


def test_non_object_getattr():
    var a: object = [2, 3, 4]
    try:
        a.foo(2)
    except e:
        print(e)


def matrix_getitem(borrowed self, borrowed i) -> object:
    return self.value[i]


def matrix_setitem(borrowed self, borrowed i, borrowed value) -> object:
    self.value[i] = value
    return None


def matrix_append(borrowed self, borrowed value) -> object:
    var impl = self.value
    impl.append(value)
    return None


def matrix_init(rows: Int, cols: Int) -> object:
    value = object([])
    return object(
        Attr("value", value),
        Attr("__getitem__", matrix_getitem),
        Attr("__setitem__", matrix_setitem),
        Attr("rows", rows),
        Attr("cols", cols),
        Attr("append", matrix_append),
    )


def matmul_untyped(C, A, B):
    for m in range(C.rows):
        for n in range(C.cols):
            for k in range(A.rows):
                C[m, n] += A[m, k] * B[k, n]


def test_matrix():
    alias size = 3
    A = matrix_init(size, size)
    B = matrix_init(size, size)
    C = matrix_init(size, size)
    for i in range(size):
        row = object([])
        row_zero = object([])
        for j in range(size):
            row_zero.append(0)
            row.append(i + j)
        A.append(row)
        B.append(row)
        C.append(row_zero)

    matmul_untyped(C, A, B)
    for k in range(size):
        C[k].print()
        print()


def main():
    # CHECK-LABEL: == test_object
    print("== test_object")
    try:
        test_object_ctors()
        test_comparison_ops()
        test_arithmetic_ops()
        # CHECK: Function at address 0x{{[a-float0-9]+}}
        # CHECK-NEXT: 3
        # CHECK-NEXT: Error from function type
        test_object_function()
        # CHECK: Type 'list' does not have attribute 'foo'
        test_non_object_getattr()
        # CHECK: [5, 8, 11]
        # CHECK: [8, 14, 20]
        # CHECK: [11, 20, 29]
        test_matrix()
    except e0:
        print(e0)
        # CHECK-NOT: TEST FAILED
        print("TEST FAILED")

    try:
        # CHECK-LABEL: Printing Tests
        print("Printing Tests")
        var a: object = True
        # CHECK-NEXT: True
        print(a)
        a = 42
        # CHECK-NEXT: 42
        print(a)
        a = 2.5
        # CHECK-NEXT: 2.5
        print(a)
        a = "hello"
        # CHECK-NEXT: 'hello'
        print(a)
        a = []
        # CHECK-NEXT: []
        print(a)
        a.append(3)
        a.append(False)
        a.append(5.5)
        var b: object = []
        b.append("foo")
        b.append("baz")
        a.append(b)
        # CHECK: [3, False, 5.5{{.*}}, ['foo', 'baz']]
        print(a)
        # CHECK: 'baz'
        print(a[3, 1])
        a[3, 1] = "bar"
        # CHECK: 'bar'
        print(a[3, 1])
        var c = a + b
        # CHECK: [3, False, 5.5{{.*}}, ['foo', 'bar'], 'foo', 'bar']
        print(c)
        b.append(False)
        # CHECK: [3, {{.*}}, ['foo', 'bar', False], 'foo', 'bar']
        print(c)
        # CHECK: [3, {{.*}}, ['foo', 'bar', False]]
        print(a)
        # CHECK: ['foo', 'bar', False]
        print(c[3])
        b[1] = object()
        # CHECK: [3, False, 5.5{{.*}}, ['foo', None, False]]
        print(a)
        a = "abc"
        b = a[True]
        # CHECK: 'b'
        print(b)
        b = a[2]
        # CHECK: 'c'
        print(b)
        a = [1, 1.2, False, "true"]
        # CHECK: [1, 1.2, False, 'true']
        print(a)

        a = object(Attr("foo", 5), Attr("bar", "hello"), Attr("baz", False))
        # CHECK: 'hello'
        print(a.bar)
        a.bar = [1, 2]
        # CHECK: {'foo' = 5, 'bar' = [1, 2], 'baz' = False}
        print(a)
    except e1:
        print(e1)
        # CHECK-NOT: TEST FAILED
        print("TEST FAILED")
