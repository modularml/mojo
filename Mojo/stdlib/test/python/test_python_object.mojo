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

from std.python import Python, PythonObject
from std.python._cpython import Py_ssize_t, PyObjectPtr
from std.python.bindings import PythonModuleBuilder
from std.testing import (
    assert_equal,
    assert_equal_pyobj,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def _test_dunder_methods(mut python: Python) raises:
    var a = PythonObject(34)
    var b = PythonObject(10)
    var d = PythonObject(2)

    # Arithmetic operators (binary, in-place, reverse)
    var c = a + b  # __add__
    assert_equal_pyobj(c, 44)
    c += 100  # __iadd__
    assert_equal_pyobj(c, 144)
    c = 100 + a  # __radd__
    assert_equal_pyobj(c, 134)

    c = a - b  # __sub__
    assert_equal_pyobj(c, 24)
    c -= 100  # __isub__
    assert_equal_pyobj(c, -76)
    c = 100 - a  # __rsub__
    assert_equal_pyobj(c, 66)

    c = a * b  # __mul__
    assert_equal_pyobj(c, 340)
    c *= 10  # __imul__
    assert_equal_pyobj(c, 3400)
    c = 34 * b  # __rmul__
    assert_equal_pyobj(c, 340)

    c = a // b  # __floordiv__
    assert_equal_pyobj(c, 3)
    c //= 2  # __ifloordiv__
    assert_equal_pyobj(c, 1)
    c = 34 // b  # __rfloordiv__
    assert_equal_pyobj(c, 3)

    c = a / b  # __truediv__
    assert_equal_pyobj(c, 3.4)
    c /= 2  # __itruediv__
    assert_equal_pyobj(c, 1.7)
    c = 34 / b  # __rtruediv__
    assert_equal_pyobj(c, 3.4)

    c = a % b  # __mod__
    assert_equal_pyobj(c, 4)
    c %= 3  # __imod__
    assert_equal_pyobj(c, 1)
    c = 34 % b  # __rmod__
    assert_equal_pyobj(c, 4)

    c = a**d  # __pow__
    assert_equal_pyobj(c, 1156)
    c = 3  # __ipow__
    c **= 4
    assert_equal_pyobj(c, 81)
    c = 34**d  # __rpow__
    assert_equal_pyobj(c, 1156)

    # Bitwise operators
    c = a ^ b  # __xor__
    assert_equal_pyobj(c, 40)
    c ^= 15  # __ixor__
    assert_equal_pyobj(c, 39)
    c = 34 ^ b  # __rxor__
    assert_equal_pyobj(c, 40)

    c = a | b  # __or__
    assert_equal_pyobj(c, 42)
    c |= 9  # __ior__
    assert_equal_pyobj(c, 43)
    c = 34 | b  # __ror__
    assert_equal_pyobj(c, 42)

    c = a & b  # __and__
    assert_equal_pyobj(c, 2)
    c &= 6  # __iand__
    assert_equal_pyobj(c, 2)
    c = 34 & b  # __rand__
    assert_equal_pyobj(c, 2)

    c = a >> d  # __rshift__
    assert_equal_pyobj(c, 8)
    c >>= 2  # __irshift__
    assert_equal_pyobj(c, 2)
    c = 34 >> d  # __rrshift__
    assert_equal_pyobj(c, 8)

    c = a << d  # __lshift__
    assert_equal_pyobj(c, 136)
    c <<= 1  # __ilshift__
    assert_equal_pyobj(c, 272)
    c = 34 << d  # __rlshift__
    assert_equal_pyobj(c, 136)

    # Comparison operators
    c = a < b  # __lt__
    assert_equal_pyobj(c, PythonObject(False))
    c = a <= b  # __le__
    assert_equal_pyobj(c, PythonObject(False))
    c = a > b  # __gt__
    assert_equal_pyobj(c, PythonObject(True))
    c = a >= b  # __ge__
    assert_equal_pyobj(c, PythonObject(True))
    c = a == b  # __eq__
    assert_equal_pyobj(c, PythonObject(False))
    c = a != b  # __ne__
    assert_equal_pyobj(c, PythonObject(True))

    # Unary operators
    c = +a  # __pos__
    assert_equal_pyobj(c, 34)
    c = -a  # __neg__
    assert_equal_pyobj(c, -34)
    c = ~a  # __invert__
    assert_equal_pyobj(c, -35)


def _test_inplace_dunder_methods(mut python: Python) raises:
    # test dunder methods that don't fall back to their non-inplace counterparts
    var list_obj: PythonObject = [1, 2]

    # Note that there is no @implicit
    var to_be_added: PythonObject = [3, 4]
    list_obj += to_be_added

    assert_equal(String(list_obj), "[1, 2, 3, 4]")

    list_obj *= 2
    assert_equal(String(py=list_obj), "[1, 2, 3, 4, 1, 2, 3, 4]")

    _ = python.eval("class A:\n  def __iadd__(self, other):\n    return 1")
    var a = python.evaluate("A()")
    a += 1
    assert_equal_pyobj(a, 1)

    # Both operands the same object: the mutating case must not read the
    # operand after rebinding the target.
    var self_aliased: PythonObject = [1, 2]
    self_aliased += self_aliased
    assert_equal(String(py=self_aliased), "[1, 2, 1, 2]")

    var immutable = PythonObject(21)
    immutable += immutable
    assert_equal_pyobj(immutable, 42)


def test_num_conversion() raises:
    comptime n = UInt64(0xFEDC_BA09_8765_4321)
    comptime n_str = String(n)
    assert_equal(n_str, String(py=PythonObject(n)))


def test_boolean_operations() raises:
    # Test boolean conversion and context
    var x: PythonObject = 1
    assert_true(x == 1)
    assert_false(x == 0)
    assert_true(x == 0 or x == 1)

    # Test __bool__ method on various objects
    assert_true(PythonObject(1).__bool__())
    assert_false(PythonObject(0).__bool__())
    assert_true(PythonObject("hello").__bool__())
    assert_false(PythonObject("").__bool__())

    var list_obj: PythonObject = [1, 2, 3]
    assert_true(list_obj.__bool__())

    var empty_list: PythonObject = []
    assert_false(empty_list.__bool__())

    assert_false(Python.none().__bool__())


def _test_string_conversions(mut python: Python) raises -> None:
    # static string
    var static_str: StaticString = "mojo"
    var py_str = PythonObject(static_str)
    var py_capitalized = py_str.capitalize()
    var mojo_capitalized = python.as_string_slice(py_capitalized)
    assert_true(mojo_capitalized == "Mojo")

    # string object
    var mo_str = "mo"
    var jo_str = "jo"
    var mojo_str = mo_str + jo_str
    py_str = PythonObject(mojo_str)
    py_capitalized = py_str.capitalize()
    mojo_capitalized = python.as_string_slice(py_capitalized)
    assert_true(mojo_capitalized == "Mojo")

    # type object
    var py_float = PythonObject(3.14)
    var type_obj = python.type(py_float)
    assert_equal(String(py=type_obj), "<class 'float'>")


def test_len() raises:
    var empty_list: PythonObject = []
    assert_equal(len(empty_list), 0)

    var l1 = Python.evaluate("[1,2,3]")
    assert_equal(len(l1), 3)

    var l2 = Python.evaluate("[42,42.0]")
    assert_equal(len(l2), 2)

    var x = PythonObject(42)
    with assert_raises(contains="object of type 'int' has no len()"):
        _ = len(x)


def test_is() raises:
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


def test_nested_object() raises:
    var a: PythonObject = [1, 2, 3]
    var b: PythonObject = [4, 5, 6]
    var nested_list: PythonObject = [a, b]
    var nested_tuple = Python.tuple(a, b)

    assert_equal(String(py=nested_list), "[[1, 2, 3], [4, 5, 6]]")
    assert_equal(String(py=nested_tuple), "([1, 2, 3], [4, 5, 6])")


def test_iter() raises:
    var list_obj: PythonObject = ["apple", "orange", "banana"]
    var i = 0
    for fruit in list_obj:
        if i == 0:
            assert_equal_pyobj(fruit, "apple")
        elif i == 1:
            assert_equal_pyobj(fruit, "orange")
        elif i == 2:
            assert_equal_pyobj(fruit, "banana")
        i += 1

    var list2: PythonObject = []
    for _ in list2:
        raise Error("This should not be reachable as the list is empty.")

    var not_iterable: PythonObject = 3
    with assert_raises():
        for _ in not_iterable:
            assert_false(
                True,
                msg=(
                    "This should not be reachable as the object is not"
                    " iterable."
                ),
            )

    # test Iterator APIs
    var it = enumerate(list_obj.__iter__())
    var val = next(it)
    assert_equal_pyobj(val[0], 0)
    assert_equal_pyobj(val[1], "apple")
    val = next(it)
    assert_equal_pyobj(val[0], 1)
    assert_equal_pyobj(val[1], "orange")
    val = next(it)
    assert_equal_pyobj(val[0], 2)
    assert_equal_pyobj(val[1], "banana")
    with assert_raises():
        _ = it.__next__()  # raises StopIteration


def test_setitem() raises:
    var ll: PythonObject = [1, 2, 3, "food"]
    assert_equal(String(py=ll), "[1, 2, 3, 'food']")
    ll[1] = "nomnomnom"
    assert_equal(String(py=ll), "[1, 'nomnomnom', 3, 'food']")


def test_dict() raises:
    # Test Python.dict from keyword arguments.
    var dd = Python.dict(food=123, fries="yes")
    assert_equal(String(py=dd), "{'food': 123, 'fries': 'yes'}")

    var dd2: PythonObject = {"food": 123, "fries": "yes"}
    assert_equal(String(py=dd2), "{'food': 123, 'fries': 'yes'}")

    dd["food"] = "salad"
    dd[42] = Python.list(4, 2)
    assert_equal(String(py=dd), "{'food': 'salad', 'fries': 'yes', 42: [4, 2]}")

    # Test Python.dict from a Span of tuples.
    var tuples = [
        (PythonObject(123), PythonObject("food")),
        (PythonObject(42), PythonObject("42")),
    ]
    dd = Python.dict(tuples)
    assert_equal(String(py=dd), "{123: 'food', 42: '42'}")

    # Also test that Python.dict() creates the right object.
    var empty = Python.dict()
    assert_equal(String(py=empty), "{}")

    var empty2: PythonObject = {}
    assert_equal(String(py=empty2), "{}")

    # Test that Python.dict uses RC correctly.
    ref cpy = Python().cpython()

    # large integer so it's RC'd
    var n = PythonObject(1000)
    var d = Python.dict(num=n)

    var _pos: Py_ssize_t = 0
    var key: PyObjectPtr = {}
    var val: PyObjectPtr = {}
    _ = cpy.PyDict_Next(
        d._obj_ptr,
        Pointer(to=_pos).as_unsafe_any_origin(),
        Pointer(to=key).as_unsafe_any_origin(),
        Pointer(to=val).as_unsafe_any_origin(),
    )

    assert_equal(cpy._Py_REFCNT(key), 1)
    assert_equal(cpy._Py_REFCNT(val), 1)
    _ = d


def test_set() raises:
    # Test Python set literals.
    var dd: PythonObject = {123, "yes"}
    var dd2 = Python.evaluate("{123, 'yes'}")
    # Be care about instability of set ordering across platforms.
    assert_equal(String(py=dd), String(py=dd2))

    assert_true(123 in dd)
    assert_true("yes" in dd)
    assert_false(42 in dd)


def test_none() raises:
    var n = Python.none()
    assert_equal(String(py=n), "None")
    assert_true(n is PythonObject(None))


def test_none_implicit_conversion() raises:
    # Test implicit conversion from None literal to PythonObject.

    # Direct assignment.
    var a: PythonObject = None
    assert_equal(String(a), "None")
    assert_true(a is Python.none())

    # Reassignment.
    var b = PythonObject(42)
    assert_equal(String(b), "42")
    b = None
    assert_equal(String(b), "None")

    # Function argument.
    def takes_python_object(obj: PythonObject) raises -> String:
        return String(obj)

    assert_equal(takes_python_object(None), "None")

    # Return value.
    def returns_none() -> PythonObject:
        return None

    assert_true(returns_none() is Python.none())

    # In a list.
    var list = Python.evaluate("[]")
    var none_val: PythonObject = None
    list.append(none_val)
    assert_equal(String(list), "[None]")


def test_getitem_raises() raises:
    var custom_indexable = Python.import_module("custom_indexable")

    var a = PythonObject(2)
    with assert_raises(contains="'int' object is not subscriptable"):
        _ = a[0]
    with assert_raises(contains="'int' object is not subscriptable"):
        _ = a[0, 0]

    var b = PythonObject(2.2)
    with assert_raises(contains="'float' object is not subscriptable"):
        _ = b[0]
    with assert_raises(contains="'float' object is not subscriptable"):
        _ = b[0, 0]

    var c = PythonObject(True)
    with assert_raises(contains="'bool' object is not subscriptable"):
        _ = c[0]
    with assert_raises(contains="'bool' object is not subscriptable"):
        _ = c[0, 0]

    var d = PythonObject(None)
    with assert_raises(contains="'NoneType' object is not subscriptable"):
        _ = d[0]
    with assert_raises(contains="'NoneType' object is not subscriptable"):
        _ = d[0, 0]

    var with_get = custom_indexable.WithGetItem()
    assert_equal("Key: 0", String(py=with_get[0]))
    assert_equal("Keys: 0, 0", String(py=with_get[0, 0]))
    assert_equal("Keys: 0, 0, 0", String(py=with_get[0, 0, 0]))

    var without_get = custom_indexable.Simple()
    with assert_raises(contains="'Simple' object is not subscriptable"):
        _ = without_get[0]

    with assert_raises(contains="'Simple' object is not subscriptable"):
        _ = without_get[0, 0]

    var with_get_exception = custom_indexable.WithGetItemException()
    with assert_raises(contains="Custom error"):
        _ = with_get_exception[1]

    var with_2d = custom_indexable.With2DGetItem()
    assert_equal("[1, 2, 3]", String(py=with_2d[0]))
    assert_equal(2, Int(py=with_2d[0, 1]))
    assert_equal(6, Int(py=with_2d[1, 2]))

    with assert_raises(contains="list index out of range"):
        _ = with_2d[0, 4]

    with assert_raises(contains="list index out of range"):
        _ = with_2d[3, 0]

    with assert_raises(contains="list index out of range"):
        _ = with_2d[3]


def test_setitem_raises() raises:
    var custom_indexable = Python.import_module("custom_indexable")
    var t = Python.evaluate("(1,2,3)")
    with assert_raises(
        contains="'tuple' object does not support item assignment"
    ):
        t[0] = 0

    var lst = Python.evaluate("[1, 2, 3]")
    with assert_raises(contains="list assignment index out of range"):
        lst[10] = 4

    var s = Python.evaluate('"hello"')
    with assert_raises(
        contains="'str' object does not support item assignment"
    ):
        s[3] = "xy"

    var with_out = custom_indexable.Simple()
    with assert_raises(
        contains="'Simple' object does not support item assignment"
    ):
        with_out[0] = 0

    var d = Python.evaluate("{}")
    with assert_raises(contains="unhashable type: 'list'"):
        d[Python.list(1, 2, 3)] = 5


def test_py_slice() raises:
    var custom_indexable = Python.import_module("custom_indexable")
    var a: PythonObject = [1, 2, 3, 4, 5]
    assert_equal("[2, 3]", String(py=a[1:3]))
    assert_equal("[1, 2, 3, 4, 5]", String(py=a[:]))
    assert_equal("[1, 2, 3]", String(py=a[:3]))
    assert_equal("[3, 4, 5]", String(py=a[2:]))
    assert_equal("[1, 3, 5]", String(py=a[::2]))
    assert_equal("[2, 4]", String(py=a[1::2]))
    assert_equal("[4, 5]", String(py=a[-2:]))
    assert_equal("[1, 2, 3]", String(py=a[:-2]))
    assert_equal("[5, 4, 3, 2, 1]", String(py=a[::-1]))
    assert_equal("[1, 2, 3, 4, 5]", String(py=a[-10:10]))  # out of bounds
    assert_equal("[1, 2, 3, 4, 5]", String(py=a[::]))
    assert_equal("[1, 2, 3, 4, 5]", String(py=a[:100]))
    assert_equal("[]", String(py=a[5:]))
    assert_equal("[5, 4, 3, 2]", String(py=a[:-5:-1]))

    var b = Python.evaluate("[i for i in range(1000)]")
    assert_equal("[0, 250, 500, 750]", String(py=b[::250]))
    with assert_raises(contains="slice step cannot be zero"):
        _ = b[::0]
    # Negative cases such as `b[1.3:10]` or `b["1":10]` are handled by parser
    # which would normally throw a TypeError in Python

    var s = PythonObject("Hello, World!")
    assert_equal("Hello", String(py=s[:5]))
    assert_equal("World!", String(py=s[7:]))
    assert_equal("!dlroW ,olleH", String(py=s[::-1]))
    assert_equal("Hello, World!", String(py=s[:]))
    assert_equal("Hlo ol!", String(py=s[::2]))
    assert_equal("Hlo ol!", String(py=s[None:None:2]))

    var t = Python.tuple(1, 2, 3, 4, 5)
    assert_equal("(2, 3, 4)", String(py=t[1:4]))
    assert_equal("(4, 3, 2)", String(py=t[3:0:-1]))

    var empty: PythonObject = []
    assert_equal("[]", String(py=empty[:]))
    assert_equal("[]", String(py=empty[1:2:3]))

    # TODO: enable this test.  Currently it fails with error: unhashable type: 'slice'
    # var d = Python.dict()
    # d["a"] = 1
    # d["b"] = 2
    # with assert_raises(contains="slice(1, 3, None)"):
    #     _ = d[1:3]

    var custom = custom_indexable.Sliceable()
    assert_equal("slice(1, 3, None)", String(py=custom[1:3]))

    var i = PythonObject(1)
    with assert_raises(contains="'int' object is not subscriptable"):
        _ = i[0:1]

    var with_2d = custom_indexable.With2DGetItem()
    assert_equal("[1, 2]", String(py=with_2d[0, PythonObject(Slice(0, 2))]))
    assert_equal("[1, 2]", String(py=with_2d[0][0:2]))

    assert_equal("[4, 5, 6]", String(py=with_2d[PythonObject(Slice(0, 2)), 1]))
    assert_equal("[4, 5, 6]", String(py=with_2d[0:2][1]))

    assert_equal(
        "[[1, 2, 3], [4, 5, 6]]", String(py=with_2d[PythonObject(Slice(0, 2))])
    )
    assert_equal("[[1, 2, 3], [4, 5, 6]]", String(py=with_2d[0:2]))
    assert_equal("[[1, 3], [4, 6]]", String(py=with_2d[0:2, ::2]))

    assert_equal(
        "[6, 5, 4]", String(py=with_2d[1, PythonObject(Slice(None, None, -1))])
    )
    assert_equal("[6, 5, 4]", String(py=with_2d[1][::-1]))

    assert_equal("[7, 9]", String(py=with_2d[2][::2]))

    with assert_raises(contains="list index out of range"):
        _ = with_2d[0:1][4]


def test_contains_dunder() raises:
    # `in` now routes through CPython's `PySequence_Contains`, which raises the
    # standard membership-test message (wording varies across Python versions,
    # e.g. "... is not iterable" vs "... is not a container or iterable").
    with assert_raises(contains="argument of type 'int' is not"):
        var z = PythonObject(0)
        _ = 5 in z

    var x: PythonObject = [1.1, 2.2]
    assert_true(1.1 in x)
    assert_false(3.3 in x)

    x = Python.list("Hello", "World")
    assert_true("World" in x)

    x = Python.tuple(1.5, 2)
    assert_true(1.5 in x)
    assert_false(3.5 in x)

    var y = Python.dict(A="A", B=5)
    assert_true("A" in y)
    assert_false("C" in y)
    assert_true("B" in y)

    # An object that is iterable but defines no `__contains__` is searched by
    # iteration.
    var p = Python()
    _ = p.eval(
        "class OnlyIter:\n  def __iter__(self):\n    return iter([7, 8])"
    )
    var only_iter = p.evaluate("OnlyIter()")
    assert_true(7 in only_iter)
    assert_false(9 in only_iter)


@fieldwise_init
struct Person(Defaultable, Movable, Writable):
    var name: String
    var age: Int

    def __init__(out self):
        self.name = ""
        self.age = 0


def test_python_mojo_object_operations() raises:
    # Type registration
    var b = PythonModuleBuilder("fake_module")
    _ = b.add_type[Person]("Person")
    _ = b.finalize()

    # Alloc
    var person_obj = PythonObject(alloc=Person("John Smith", 42))

    # Downcast
    var person_ptr = person_obj.downcast_value_ptr[Person]()

    assert_equal(person_ptr[].name, "John Smith")


def test_conversion_to_simd() raises:
    var py_float = PythonObject(0.123456789121212)
    var py_int = PythonObject(256)

    assert_equal(Float64(py=py_float), 0.123456789121212)
    assert_equal(Float32(py=py_float), 0.12345679)
    assert_equal(Float16(py=py_float), 0.12345679)
    assert_equal(Float64(py=py_int), 256.0)

    var py_str = PythonObject("inf")
    with assert_raises(contains="must be real number, not str"):
        _ = Float64(py=py_str)

    assert_equal(Int64(py=py_int), Int64(256))
    assert_equal(Int32(py=py_int), Int32(256))
    assert_equal(Int16(py=py_int), Int16(256))
    assert_equal(Int8(py=py_int), Int8(0))

    assert_equal(UInt64(py=py_int), UInt64(256))
    assert_equal(UInt32(py=py_int), UInt32(256))
    assert_equal(UInt16(py=py_int), UInt16(256))
    assert_equal(UInt8(py=py_int), UInt8(0))


def test_hash() raises:
    # Test __hash__ method
    var obj1 = PythonObject(42)
    var obj2 = PythonObject(42)
    var obj3 = PythonObject("hello")

    # Same integers should have same hash
    assert_equal(obj1.__hash__(), obj2.__hash__())

    # Different objects should typically have different hashes
    assert_true(obj1.__hash__() != obj3.__hash__())

    # Test that unhashable types raise appropriate errors
    var list_obj: PythonObject = [1, 2, 3]
    with assert_raises(contains="unhashable type"):
        _ = list_obj.__hash__()


def test_call_with_kwargs() raises:
    # Test calling Python functions with keyword arguments
    var print_func = Python.import_module("builtins").print

    # Test calling with positional and keyword args
    var io = Python.import_module("io")
    var string_io = io.StringIO()
    _ = print_func("test", file=string_io)

    var output = string_io.getvalue()
    assert_equal(String(py=output).strip(), "test")


def test_attribute_access() raises:
    # Test __getattr__ and __setattr__
    var test_dict: PythonObject = {"attr": "value"}

    # Test getting attributes that exist
    var attr_value = test_dict.__getattr__("get")
    assert_true(attr_value is not PythonObject(None))

    # Test setting attributes on objects that support it
    var custom_obj = Python.evaluate("type('TestClass', (), {})()")
    custom_obj.__setattr__("new_attr", "new_value")
    var retrieved = custom_obj.__getattr__("new_attr")
    assert_equal_pyobj(retrieved, "new_value")

    # Test getting non-existent attributes raises an error
    with assert_raises(contains="no attribute"):
        _ = test_dict.__getattr__("nonexistent")


def test_copy() raises:
    # Test that copy constructor works correctly
    var original = PythonObject(42)
    var copied = original

    # They should be equal but not the same object
    assert_equal_pyobj(original, copied)

    # For immutable objects like integers, they might be the same
    # but let's test with a mutable object
    var list_original: PythonObject = [1, 2, 3]
    var list_copied = list_original

    # They should reference the same object
    assert_true(list_original is list_copied)


def _test_python_eval_and_evaluate(mut python: Python) raises:
    # Test Python.eval() method
    var success = python.eval("x = 42")
    assert_true(success)

    # Test Python.evaluate() method
    var result = Python.evaluate("2 + 3")
    assert_equal_pyobj(result, 5)


def test_python_module_operations() raises:
    # Test Python.import_module()
    var math_module = Python.import_module("math")
    var pi_value = math_module.pi
    assert_true(Float64(py=pi_value) > 3.1 and Float64(py=pi_value) < 3.2)

    # Test Python.add_to_path() and importing custom modules
    # Note: This test might need a custom module file to be truly effective
    # For now, just test that the function doesn't crash
    Python.add_to_path(".")


def test_python_type_functions() raises:
    # Test Python.type()
    var int_obj = PythonObject(42)
    var int_type = Python.type(int_obj)
    assert_equal(String(py=int_type), "<class 'int'>")

    var str_obj = PythonObject("hello")
    var str_type = Python.type(str_obj)
    assert_equal(String(py=str_type), "<class 'str'>")

    # Test Python.str(), Python.int(), Python.float()
    var str_result = Python.str(int_obj)
    assert_equal_pyobj(str_result, "42")

    var int_result = Python.int(str_obj.__getattr__("__len__")())
    assert_equal_pyobj(int_result, 5)

    var float_result = Python.float(int_obj)
    assert_equal_pyobj(float_result, 42.0)


def test_advanced_slicing() raises:
    # Test more complex slicing scenarios
    var data: PythonObject = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    # Test step slicing with negative indices
    assert_equal("[9, 7, 5, 3, 1]", String(py=data[-1::-2]))

    # Test empty slices
    assert_equal("[]", String(py=data[5:5]))
    assert_equal("[]", String(py=data[10:20]))

    # Test slice assignment (if supported)
    var mutable_list: PythonObject = [1, 2, 3, 4, 5]
    # Note: slice assignment would be: mutable_list[1:3] = [10, 20]
    # but this might not be supported in current implementation


def test_error_handling() raises:
    # Test various error conditions
    var zero = PythonObject(0)
    var one = PythonObject(1)

    # Test division by zero
    with assert_raises(contains="division by zero"):
        _ = one / zero

    # Test invalid operations on None
    var none_obj = Python.none()
    with assert_raises():
        _ = none_obj + one


def test_with_python_dunder_methods() raises:
    var python = Python()
    _test_dunder_methods(python)


def test_with_python_inplace_dunder_methods() raises:
    var python = Python()
    _test_inplace_dunder_methods(python)


def test_reflected_operand_fallback() raises:
    # A left operand that returns `NotImplemented` must fall back to the right
    # operand's reflected dunder. Each dunder returns its own name, so these
    # also pin which operator maps to which protocol function.
    var mod = Python.import_module("custom_operators")
    var r = mod.OnlyReflected()
    var i = PythonObject(1)

    # Reached through this object's forward operators.
    assert_equal_pyobj(i + r, "radd")
    assert_equal_pyobj(i - r, "rsub")
    assert_equal_pyobj(i * r, "rmul")
    assert_equal_pyobj(i / r, "rtruediv")
    assert_equal_pyobj(i // r, "rfloordiv")
    assert_equal_pyobj(i % r, "rmod")
    assert_equal_pyobj(i**r, "rpow")
    assert_equal_pyobj(i << r, "rlshift")
    assert_equal_pyobj(i >> r, "rrshift")
    assert_equal_pyobj(i & r, "rand")
    assert_equal_pyobj(i | r, "ror")
    assert_equal_pyobj(i ^ r, "rxor")

    # Reached through this object's reverse operators, which evaluate the whole
    # operation with the operands swapped.
    assert_equal_pyobj(1 + r, "radd")
    assert_equal_pyobj(1 - r, "rsub")
    assert_equal_pyobj(1 * r, "rmul")
    assert_equal_pyobj(1 / r, "rtruediv")
    assert_equal_pyobj(1 // r, "rfloordiv")
    assert_equal_pyobj(1 % r, "rmod")
    assert_equal_pyobj(1**r, "rpow")
    assert_equal_pyobj(1 << r, "rlshift")
    assert_equal_pyobj(1 >> r, "rrshift")
    assert_equal_pyobj(1 & r, "rand")
    assert_equal_pyobj(1 | r, "ror")
    assert_equal_pyobj(1 ^ r, "rxor")


def test_inplace_operator_fallback() raises:
    # An augmented assignment on a type with no in-place dunder falls back to
    # the plain operator. A fresh operand per case, since each assignment
    # rebinds the name to the result.
    var mod = Python.import_module("custom_operators")

    var add = mod.OnlyForward()
    add += 1
    assert_equal_pyobj(add, "add")

    var sub = mod.OnlyForward()
    sub -= 1
    assert_equal_pyobj(sub, "sub")

    var mul = mod.OnlyForward()
    mul *= 1
    assert_equal_pyobj(mul, "mul")

    var truediv = mod.OnlyForward()
    truediv /= 1
    assert_equal_pyobj(truediv, "truediv")

    var floordiv = mod.OnlyForward()
    floordiv //= 1
    assert_equal_pyobj(floordiv, "floordiv")

    var mod_ = mod.OnlyForward()
    mod_ %= 1
    assert_equal_pyobj(mod_, "mod")

    var pow_ = mod.OnlyForward()
    pow_ **= 1
    assert_equal_pyobj(pow_, "pow")

    var lshift = mod.OnlyForward()
    lshift <<= 1
    assert_equal_pyobj(lshift, "lshift")

    var rshift = mod.OnlyForward()
    rshift >>= 1
    assert_equal_pyobj(rshift, "rshift")

    var and_ = mod.OnlyForward()
    and_ &= 1
    assert_equal_pyobj(and_, "and")

    var or_ = mod.OnlyForward()
    or_ |= 1
    assert_equal_pyobj(or_, "or")

    var xor = mod.OnlyForward()
    xor ^= 1
    assert_equal_pyobj(xor, "xor")


def test_unsupported_operands() raises:
    # An operation that no operand supports raises, rather than yielding the
    # `NotImplemented` object as a value.
    with assert_raises(contains="unsupported operand type(s) for +"):
        _ = PythonObject(1) + PythonObject("two")

    with assert_raises(contains="not supported between instances"):
        _ = PythonObject(1) < PythonObject("two")

    # Equality is the exception: mismatched types compare unequal rather than
    # raising, so this must be `False`, not a truthy `NotImplemented`.
    assert_false(PythonObject(1) == PythonObject("two"))
    assert_true(PythonObject(1) != PythonObject("two"))


def test_with_python_string_conversions() raises:
    var python = Python()
    _test_string_conversions(python)


def test_with_python_eval_and_evaluate() raises:
    var python = Python()
    _test_python_eval_and_evaluate(python)


def test_python_object_string() raises:
    var s = String(py=PythonObject("hello"))
    assert_equal(s, "hello")

    var p = Python()
    _ = p.eval("class A:\n  def __str__(self): pass")
    var a = p.evaluate("A()")
    with assert_raises(contains="__str__ returned non-string"):
        _ = String(py=a)


def test_python_object_write_repr_to() raises:
    var obj = PythonObject(42)
    var s = String()
    obj.write_repr_to(s)
    assert_equal(s, "PythonObject(42)")

    var str_obj = PythonObject("hello")
    s = String()
    str_obj.write_repr_to(s)
    assert_equal(s, "PythonObject('hello')")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
