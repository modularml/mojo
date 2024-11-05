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

from random import random_float64

from testing import assert_equal, assert_false, assert_raises, assert_true


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

    a = String("hello world")
    assert_true(a == object("hello world"))
    a += "!"
    assert_true(a == "hello world!")

    b = object.dict()
    b["one"] = 1
    b[2] = 2
    b[3.0] = "three"
    assert_equal(len(b), 3)
    assert_equal(b["one"], 1)
    assert_equal(b[2], 2)
    assert_equal(b[3.0], "three")

    a = (0, True, 2.0, "three")
    assert_true(bool(a))
    assert_equal(len(a), 4)
    assert_equal(a[0], 0)
    assert_equal(a[1], True)
    assert_equal(a[2], 2.0)
    assert_equal(a[3], "three")

    b["tuple"] = a
    assert_equal(b["tuple"], a)
    assert_equal(b._value.ref_count(), 1)
    assert_equal(a._value.ref_count(), 2)
    _ = b^
    assert_equal(a._value.ref_count(), 1)


def method_obj_gt(self, rhs):
    return self.value > rhs.value


def method_obj_lt(self, rhs):
    return self.value < rhs.value


def method_obj_ge(self, rhs):
    return self.value >= rhs.value


def method_obj_le(self, rhs):
    return self.value <= rhs.value


def test_comparison_ops():
    assert_true(object(False) < True)
    assert_false(object(True) < True)
    assert_true(object(False) < object(True))
    assert_false(object(True) < object(True))
    assert_false(object(False) < object(False))
    assert_false(object(True) < object(True))
    assert_false(object(False) > object(True))
    assert_true(object(True) > object(False))
    assert_false(object(True) > object(True))
    assert_false(object(False) > object(False))
    assert_true(object(False) <= object(True))
    assert_false(object(True) <= object(False))
    assert_true(object(True) <= object(True))
    assert_true(object(False) <= object(False))
    assert_false(object(False) >= object(True))
    assert_true(object(True) >= object(False))
    assert_true(object(True) >= object(True))
    assert_true(object(False) >= object(False))
    assert_true(object(1) > False)
    assert_true(object(1) > False)
    assert_true(object(2) == 2)
    assert_false(object(True) != 1)
    assert_true(object(True) <= 1.0)
    assert_false(object(False) >= 0.5)
    assert_true(object(True) == object(True))
    assert_false(object(True) == object(False))
    assert_false(object(False) == object(True))
    assert_true(object(False) == object(False))

    lhs = object("aaa")
    rhs = object("bbb")
    assert_false(lhs == rhs)
    assert_true(lhs != rhs)
    assert_true(lhs < rhs)
    assert_true(lhs <= rhs)
    assert_false(lhs > rhs)
    assert_false(lhs >= rhs)

    lhs = [False, 1, "two", 3.0]
    rhs = [False, 1, "two", 3.0]
    assert_true(lhs == rhs)
    lhs.append(4)
    assert_false(lhs == rhs)

    lhs = (False, 1, "two", 3.0)
    rhs = (False, 1, "two", 3.0)
    assert_true(lhs == rhs)
    assert_false(lhs != rhs)

    lhs = object.dict()
    rhs = object.dict()
    lhs["one"] = [2, 3.0]
    rhs["one"] = [2, 3.0]
    assert_true(lhs == rhs)
    rhs["one"].append(4)
    assert_false(lhs == rhs)
    assert_true(lhs != rhs)

    lhs = object(Attr("value", [1, 2.0]))
    rhs = object(Attr("value", [1, 2.0]))
    assert_true(lhs == rhs)
    rhs.value = 1
    assert_false(lhs == rhs)
    assert_true(lhs != rhs)

    lhs = object.dict()
    rhs = object([1, 2])
    assert_false(lhs == rhs)
    lhs = 1
    assert_false(lhs == rhs)
    lhs = object(Attr("value", 1))
    assert_false(lhs == rhs)
    rhs = object.dict()
    assert_false(lhs == rhs)

    lhs = []
    lhs.append(object.dict())
    lhs[0]["one"] = 1
    assert_equal(lhs[0]["one"], 1)
    rhs = []
    assert_false(lhs == rhs)
    rhs.append(object.dict())
    assert_false(lhs == rhs)
    rhs[0]["one"] = 1
    assert_true(lhs == rhs)

    lhs = (0, True, 2.0, "three")
    rhs = (0, True, 2.0, "three")
    assert_true(lhs == rhs)
    assert_equal(lhs[2], rhs[2])
    assert_equal(lhs[2], 2.0)
    rhs = (0, 1, 2)
    assert_false(lhs == rhs)

    lhs = [1.0, 0.0, -1.0]
    rhs = [1.0, 0.0, -2.0]
    assert_true(rhs < lhs)
    assert_false(rhs > lhs)
    assert_false(rhs == lhs)
    lhs = [1, 0, -1]
    rhs = [1, 0, -2]
    assert_true(rhs < lhs)
    assert_false(rhs > lhs)
    assert_false(rhs == lhs)
    lhs = [1.0, 0.0, -1.0]
    rhs = [1, 0, -2]
    assert_true(rhs < lhs)
    assert_false(rhs > lhs)
    assert_false(rhs == lhs)
    lhs = [True, True]
    rhs = [True, False]
    assert_true(rhs < lhs)
    assert_false(rhs > lhs)
    assert_false(rhs == lhs)

    lhs = (1.0, 0.0, -1.0)
    rhs = (1.0, 0.0, -2.0)
    assert_true(rhs < lhs)
    assert_false(rhs > lhs)

    lhs = (1.0, 0.0, -1.0)
    rhs = (1.0, 0.0, -2.0)
    assert_true(rhs <= lhs)
    assert_false(rhs >= lhs)

    lhs = (1.0, 0.0, -1.0)
    rhs = (1.0, 0.0, -1.0)
    assert_true(rhs <= lhs)
    assert_true(rhs >= lhs)

    lhs = object(
        Attr("value", 0),
        Attr("__le__", method_obj_le),
        Attr("__ge__", method_obj_ge),
        Attr("__lt__", method_obj_lt),
        Attr("__gt__", method_obj_gt),
    )
    rhs = object(
        Attr("value", 1),
        Attr("__le__", method_obj_le),
        Attr("__ge__", method_obj_ge),
        Attr("__lt__", method_obj_lt),
        Attr("__gt__", method_obj_gt),
    )
    assert_true(lhs < rhs)
    assert_false(lhs > rhs)
    assert_true(lhs <= rhs)
    assert_false(lhs >= rhs)
    lhs.value = 10
    rhs.value = 10
    assert_true(lhs <= rhs)
    assert_true(lhs >= rhs)


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


def test_arithmetic_ops_div():
    # test mod
    lhs = object(5.5)
    rhs = object(2.0)
    assert_true((lhs % rhs) == 1.5)
    lhs %= rhs
    assert_true(lhs == 1.5)
    assert_true(5.5 % object(2.0) == 1.5)

    lhs = object(5)
    rhs = object(2)
    assert_true((lhs % rhs) == 1)
    lhs %= rhs
    assert_true(lhs == 1)
    assert_true(5 % object(2) == 1)

    # truediv
    lhs = object(5.5)
    rhs = object(2.0)
    assert_true(lhs / rhs == 2.75)
    lhs /= rhs
    assert_true(lhs == 2.75)
    assert_true(5.5 / object(2.0) == 2.75)

    lhs = object(5)
    rhs = object(2)
    assert_true(lhs / rhs == 2)
    lhs /= rhs
    assert_true(lhs == 2)
    assert_true(5 / object(2) == 2)

    # floor div
    lhs = object(5.5)
    rhs = object(2.0)
    assert_true(lhs // rhs == 2)
    lhs //= rhs
    assert_true(lhs == 2)
    assert_true(5.5 // object(2.0) == 2)

    lhs = object(5)
    rhs = object(2)
    assert_true(lhs // rhs == 2)
    lhs //= rhs
    assert_true(lhs == 2)
    assert_true(5 // object(2) == 2)


def test_object_bitwise():
    a = object(1)
    b = object(2)
    assert_true(a << b == 4)
    assert_true(b >> a == 1)

    b <<= a
    assert_true(b == 4)
    b >>= a
    assert_true(b == 2)

    assert_true(2 << object(1) == 4)
    assert_true(2 >> object(1) == 1)

    assert_true(object(15) & object(7) == 7)
    assert_true(object(15) | object(7) == 15)
    assert_true(object(15) ^ object(7) == 8)

    a = object(15)
    b = object(7)
    a &= b
    assert_true(a == 7)
    a = object(15)
    a |= b
    assert_true(a == 15)
    a = object(15)
    a ^= b
    assert_true(a == 8)

    assert_true(15 & object(7) == 7)
    assert_true(15 | object(7) == 15)
    assert_true(15 ^ object(7) == 8)


def test_function(lhs, rhs) -> object:
    return lhs + rhs


# These are all marked borrowed because 'object' doesn't support function
# types with owned arguments.
def test_function_raises(a) -> object:
    raise Error("Error from function type")


def test_object_function():
    var a: object = test_function
    assert_true(str(a).startswith("Function at address 0x"))
    assert_equal(str(a(1, 2)), str(3))
    a = test_function_raises
    with assert_raises(contains="Error from function type"):
        a(1)


def test_non_object_getattr():
    var a: object = [2, 3, 4]
    with assert_raises(contains="Type 'list' does not have attribute 'foo'"):
        a.foo(2)


def matrix_getitem(self, i) -> object:
    return self.value[i[0]][i[1]]


def matrix_setitem(self, i, value) -> object:
    self.value[i[0]][i[1]] = value
    return None


def matrix_append(self, value) -> object:
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
    assert_equal(str(C.value[0]), "[5, 8, 11]")
    assert_equal(str(C.value[1]), "[8, 14, 20]")
    assert_equal(str(C.value[2]), "[11, 20, 29]")


def test_convert_to_string():
    var a: object = True
    assert_equal(str(a), "True")
    a = 42
    assert_equal(str(a), "42")
    a = 2.5
    assert_equal(str(a), "2.5")
    a = "hello"
    assert_equal(str(a), "hello")
    a = []
    assert_equal(str(a), "[]")
    a.append(3)
    a.append(False)
    a.append(5.5)
    var b: object = []
    b.append("foo")
    b.append("baz")
    a.append(b)
    assert_equal(str(a), "[3, False, 5.5, ['foo', 'baz']]")
    assert_equal(str(a[3][1]), "baz")
    a[3][1] = "bar"
    assert_equal(str(a[3][1]), "bar")
    var c = a + b
    assert_equal(str(c), "[3, False, 5.5, ['foo', 'bar'], 'foo', 'bar']")
    b.append(False)
    assert_equal(str(c), "[3, False, 5.5, ['foo', 'bar', False], 'foo', 'bar']")
    assert_equal(str(a), "[3, False, 5.5, ['foo', 'bar', False]]")
    assert_equal(str(c[3]), "['foo', 'bar', False]")
    b[1] = object()
    assert_equal(str(a), "[3, False, 5.5, ['foo', None, False]]")
    a = "abc"
    b = a[True]
    assert_equal(str(b), "b")
    b = a[2]
    assert_equal(str(b), "c")
    a = [1, 1.2, False, "true"]
    assert_equal(str(a), "[1, 1.2, False, 'true']")

    a = object(Attr("foo", 5), Attr("bar", "hello"), Attr("baz", False))
    assert_equal(str(a.bar), "hello")
    a.bar = [1, 2]
    assert_equal(str(a), "{'foo' = 5, 'bar' = [1, 2], 'baz' = False}")
    assert_equal(repr(a), "{'foo' = 5, 'bar' = [1, 2], 'baz' = False}")

    a = object.dict()
    a["one"] = 1
    a[2] = "two"
    assert_equal(str(a["one"]), "1")
    assert_equal(str(a[2]), "two")
    assert_equal(str(a), "{'one' = 1, 2 = 'two'}")
    b = object.dict()
    b["three"] = 1
    a["nested"] = b
    assert_equal(str(a), "{'one' = 1, 2 = 'two', 'nested' = {'three' = 1}}")
    b["three"] = True
    assert_equal(str(a), "{'one' = 1, 2 = 'two', 'nested' = {'three' = True}}")

    a = object(Attr("value", object.dict()))
    b = object(Attr("value", object.dict()))
    a.value["function"] = matrix_append
    b.value["function"] = matrix_append
    assert_equal(repr(a), repr(b))

    a = (0, True, 1.0, "Three")
    assert_equal(str(a), "(0, True, 1.0, 'Three')")
    b = []
    b.append(a)
    b.append(4)
    assert_equal(str(b), "[(0, True, 1.0, 'Three'), 4]")


def test_object_dict():
    a = object.dict()
    a["one"] = 1
    a[2] = "two"
    assert_equal(a["one"], 1)
    assert_equal(a[2], "two")
    assert_equal(str(a[2]), "two")
    b = a
    assert_equal(a._value.ref_count(), 2)
    # asap __del__ of a
    assert_equal(b._value.ref_count(), 1)

    ref_counted_list = object([1, 2, 3])
    assert_equal(ref_counted_list._value.ref_count(), 1)
    b["ref_counted_list"] = ref_counted_list
    assert_equal(ref_counted_list._value.ref_count(), 2)
    ref_counted_list.append(4)
    assert_equal(b["ref_counted_list"], [1, 2, 3, 4])
    # asap __del__ of b
    assert_equal(ref_counted_list._value.ref_count(), 1)


def test_object_dict_contains():
    a = object.dict()
    a["one"] = 1
    a["twothree"] = [2, 3]
    a[4] = "four"
    a[5.5] = 6
    assert_equal("twothree" in a, True)
    assert_equal("one" in a, True)
    assert_equal("two" in a, False)
    assert_equal(4 in a, True)
    assert_equal(5 in a, False)
    assert_equal(5.5 in a, True)
    assert_equal(6.5 in a, False)


def test_object_dict_pop():
    a = object.dict()
    a["one"] = 1
    a["twothree"] = [2, 3]
    a[4] = "four"
    a[5.5] = 6
    assert_equal(len(a), 4)
    tmp_element = a.pop(4)
    assert_equal(tmp_element, "four")
    assert_equal(len(a), 3)
    tmp_element = a.pop("twothree")
    assert_equal(tmp_element, [2, 3])
    assert_equal(len(a), 2)
    assert_equal(tmp_element._value.ref_count(), 1)

    with assert_raises(contains="usage: .pop(key) for dictionaries"):
        a.pop()

    with assert_raises(contains="KeyError"):
        a.pop("key")


def test_object_cast():
    a = object()
    a = "1"
    assert_equal(int(a) + 1, 2)


def test_object_init_list_attr():
    attrs = List[Attr]()
    attrs.append(Attr("val", [1, 2]))
    attrs.append(Attr("add", test_function))
    y = object(attrs)
    assert_equal(y.val, [1, 2])
    assert_equal(y.add(10, 20), 30)


def test_object_list_contains():
    a = object([1, "two", True, 1.5])
    assert_equal(1 in a, True)
    assert_equal(2 in a, False)
    assert_equal("two" in a, True)
    assert_equal("three" in a, False)
    assert_equal(1.5 in a, True)
    assert_equal(2.0 in a, False)
    assert_equal(True in a, True)
    assert_equal(False in a, False)


def test_object_list_pop():
    a = object([1, "two", 3.0])
    assert_equal(len(a), 3)
    tmp_element = a.pop(2)
    assert_equal(len(a), 2)
    assert_equal(tmp_element, 3.0)
    tmp_element = a.pop(0)
    assert_equal(len(a), 1)
    assert_equal(tmp_element, 1)
    res = a.pop()
    assert_equal(res, "two")
    with assert_raises(contains="List is empty"):
        a.pop()

    a = object([1, "two", 3.0])
    with assert_raises(contains="pop index out of range"):
        a.pop(3)
    assert_equal(len(a), 3)
    b = a.pop(-1)
    assert_equal(b, 3.0)
    with assert_raises(contains="List uses non float numbers as indexes"):
        a.pop(3.5)


def test_object_hash():
    a = Int(1)
    b = Float64(2.5)
    c = String("hello world")
    assert_equal(hash(a), hash(object(a)))
    assert_equal(hash(b), hash(object(b)))
    assert_equal(hash(c), hash(object(c)))

    abc = object([a, b, c])
    abc_repr = repr(abc)
    assert_equal(hash(abc), hash("[1, 2.5, 'hello world']"))


def test_object_RefCountedCowString():
    a = object(String("Hello world"))
    b = a
    assert_equal(a._value.ref_count(), 2)
    assert_equal(b._value.ref_count(), 2)
    # asap del of b
    assert_equal(a._value.ref_count(), 1)
    c = a
    assert_equal(a._value.ref_count(), 2)
    assert_equal(c._value.ref_count(), 2)
    c += "!"
    assert_equal(a._value.ref_count(), 1)
    assert_equal(c._value.ref_count(), 1)
    assert_equal(str(a), "Hello world")
    assert_equal(str(c), "Hello world!")

    a = object.dict()
    c = object("hello world")
    a[1] = c
    assert_equal(c._value.ref_count(), 2)
    a[1] += "!"
    assert_equal(c, "hello world")
    assert_equal(c._value.ref_count(), 1)
    assert_equal(a[1], "hello world!")
    a = a.pop(1)
    assert_equal(a._value.ref_count(), 1)


def test_object_tuple_contains():
    a = object((1, "two", True, 1.5))
    assert_equal(1 in a, True)
    assert_equal(2 in a, False)
    assert_equal("two" in a, True)
    assert_equal("three" in a, False)
    assert_equal(1.5 in a, True)
    assert_equal(2.0 in a, False)
    assert_equal(True in a, True)
    assert_equal(False in a, False)


def test_object_tuple_add():
    a = object((0, 1))
    b = object(("two", "three"))
    c = a + b
    assert_equal(len(c), 4)
    assert_equal(len(a), 2)
    assert_equal(len(b), 2)
    assert_equal(c[2], "two")


def test_object_get_type_id():
    var x = object()
    assert_equal(x._value.get_type_id(), x._value.none)
    x = 1
    assert_equal(x._value.get_type_id(), x._value.int)
    x = 1.0
    assert_equal(x._value.get_type_id(), x._value.float)
    x = "hello world"
    assert_equal(x._value.get_type_id(), x._value.str)
    x = object(Attr("value", 1))
    assert_equal(x._value.get_type_id(), x._value.obj)
    x = [1, 2.0, "three"]
    assert_equal(x._value.get_type_id(), x._value.list)
    x = (1, 2.0, "three")
    assert_equal(x._value.get_type_id(), x._value.tuple)
    x = object.dict()
    x["one"] = 1
    assert_equal(x._value.get_type_id(), x._value.dict)


def test_object_getattr():
    # test cow 1:
    x = object(Attr("value", "hello world"))
    y = x.value
    assert_equal(x.value, y)
    x.value = "hello world!"
    assert_equal(x.value, "hello world!")
    assert_equal(y, "hello world")

    # test cow 2:
    x = object(Attr("value", "hello world"))
    y = object(Attr("value", x.value))
    assert_equal(y.value, "hello world")
    x.value = "hello world!"
    assert_equal(x.value, "hello world!")
    assert_equal(y.value, "hello world")

    with assert_raises(contains="does not have an attribute of name 'value2'"):
        _ = x.value2
    with assert_raises(
        contains="does not have an attribute of name 'new_attr'"
    ):
        x.new_attr = 1


def main():
    test_object_ctors()
    test_comparison_ops()
    test_arithmetic_ops()
    test_arithmetic_ops_div()
    test_object_bitwise()
    test_object_function()
    test_non_object_getattr()
    test_matrix()
    test_convert_to_string()
    test_object_hash()
    test_object_dict()
    test_object_dict_contains()
    test_object_dict_pop()
    test_object_cast()
    test_object_init_list_attr()
    test_object_list_contains()
    test_object_list_pop()
    test_object_RefCountedCowString()
    test_object_tuple_contains()
    test_object_tuple_add()
    test_object_get_type_id()
    test_object_getattr()
