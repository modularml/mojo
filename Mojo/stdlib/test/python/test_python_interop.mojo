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

from std.python.python import Python, PythonObject
from std.testing import assert_equal, assert_raises, assert_true, TestSuite


def _test_execute_python_string(mut python: Python) -> String:
    try:
        _ = Python.evaluate("print('evaluated by PyRunString')")
        return String(Python.evaluate("'a' + 'b'"))
    except e:
        return String(e)


def _test_local_import(mut python: Python) -> String:
    try:
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var foo = my_module.Foo("apple")
            foo.bar = "orange"
            return String(foo.bar)
        return "no module, no fruit"
    except e:
        return String(e)


def _test_dynamic_import(mut python: Python, times: Int = 1) -> String:
    comptime INLINE_MODULE = """
called_already = False
def hello(name):
    global called_already
    if not called_already:
        called_already = True
        return f"Hello {name}!"
    return "Again?"
"""
    try:
        var mod = Python.evaluate(INLINE_MODULE, file=True)
        for _ in range(times - 1):
            mod.hello("world")
        return String(mod.hello("world"))
    except e:
        return String(e)


def _test_call(mut python: Python) -> String:
    try:
        var my_module: PythonObject = Python.import_module("my_module")
        return String(
            my_module.eat_it_all(
                "carrot",
                "bread",
                "rice",
                fruit="pear",
                protein="fish",
                cake="yes",
            )
        )
    except e:
        return String(e)


def test_int_conversion() raises:
    var py_int = Python.int(PythonObject("123"))
    # TODO: use assert_equal once we have parametric raises in __eq__.
    assert_true(py_int == PythonObject(123))

    with assert_raises(contains="invalid literal for int()"):
        _ = Python.int(PythonObject("foo"))


def test_float_conversion() raises:
    var math = Python.import_module("math")

    var f = Python.float(PythonObject("123.45"))
    assert_true(f == PythonObject(123.45))

    f = Python.float(PythonObject("inf"))
    assert_true(f == math.inf)

    with assert_raises(contains="could not convert string to float"):
        _ = Python.float(PythonObject("foo"))


def test_str_conversion() raises:
    var py_str = Python.str(PythonObject(123))
    assert_true(py_str == PythonObject("123"))


def test_imports() raises:
    var python = Python()
    assert_equal(_test_local_import(python), "orange")

    # Test twice to ensure that the module state is fresh.
    assert_equal(_test_dynamic_import(python), "Hello world!")
    assert_equal(_test_dynamic_import(python), "Hello world!")

    # Test with two calls to ensure that the state is persistent.
    assert_equal(_test_dynamic_import(python, times=2), "Again?")


def test_call() raises:
    var python = Python()
    assert_equal(
        _test_call(python),
        (
            "carrot ('bread', 'rice') fruit=pear {'protein': 'fish', 'cake':"
            " 'yes'}"
        ),
    )


def test_object_properties() raises:
    var python = Python()
    var obj: PythonObject = [1, 2.4, True, "False"]
    assert_equal(String(obj), "[1, 2.4, True, 'False']")

    obj = Python.tuple(1, 2.4, True, "False")
    assert_equal(String(obj), "(1, 2.4, True, 'False')")

    obj = PythonObject(None)
    assert_equal(String(obj), "None")

    assert_equal(_test_execute_python_string(python), "ab")


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
