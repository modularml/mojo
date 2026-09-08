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
from std.testing import (
    assert_equal,
    assert_equal_pyobj,
    assert_false,
    assert_raises,
    assert_true,
    TestSuite,
)


def test_string() raises:
    var py_string = PythonObject("mojo")
    var py_string_capitalized = py_string.capitalize()

    var cap_mojo_string = String(py_string_capitalized)
    assert_equal(cap_mojo_string, "Mojo")
    assert_equal_pyobj(cap_mojo_string, PythonObject("Mojo"))

    var os = Python.import_module("os")
    assert_true(String(os.environ).startswith("environ({"))


def test_int() raises:
    assert_equal(Int(py=PythonObject(5)), 5)
    assert_equal(Int(py=PythonObject(-1)), -1)

    # Test error trying conversion from Python '"str"'
    with assert_raises(contains="invalid literal for int()"):
        _ = Int(py=PythonObject("str"))

    assert_equal_pyobj(Int(5), PythonObject(5))
    assert_equal_pyobj(Int(-1), PythonObject(-1))


def test_float() raises:
    var py_float = PythonObject(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py=py_float), mojo_float)


def test_bool() raises:
    assert_true(Bool(PythonObject(True)))
    assert_false(Bool(PythonObject(False)))

    assert_equal_pyobj(Bool(True), PythonObject(True))
    assert_equal_pyobj(Bool(False), PythonObject(False))


def test_numpy_int() raises:
    var np = Python.import_module("numpy")
    var py_numpy_int = np.int64(1)
    var mojo_int = Int(1)
    assert_equal(Int(py=py_numpy_int), mojo_int)


def test_int_subclass_override_takes_fallback() raises:
    # An `int` subclass with overridden `__int__` must observe the
    # override, matching CPython's `int(x)` semantics. `PyLong_CheckExact`
    # rejects subclasses so the fast path is skipped and `__int__()` runs.
    var mod = Python.evaluate(
        "class _MyInt(int):\n    def __int__(self):\n        return 99\n",
        file=True,
        name="_int_subclass_test_mod",
    )
    var my_int = mod._MyInt(1)
    assert_equal(Int(py=my_int), 99)


def test_numpy_float() raises:
    var np = Python.import_module("numpy")
    var py_numpy_float = np.float64(1.0)
    var mojo_float = Float64(1.0)
    assert_equal(Float64(py=py_numpy_float), mojo_float)


def test_scalar_from_python_unsigned() raises:
    # Values in [2**63, 2**64) survived the Mojo -> Python leg (via the
    # unsigned `PyLong_FromSize_t`) and must survive the read-back too.
    assert_equal(UInt64(py=PythonObject(UInt64.MAX)), UInt64.MAX)
    var two_63 = UInt64(1) << 63
    assert_equal(UInt64(py=PythonObject(two_63)), two_63)
    assert_equal(UInt32(py=PythonObject(UInt32.MAX)), UInt32.MAX)

    # A negative Python int must raise for an unsigned scalar, matching
    # CPython's `PyLong_AsSize_t` (`OverflowError` for negatives), rather than
    # wrapping to `UInt64.MAX`.
    with assert_raises():
        _ = UInt64(py=PythonObject(-1))

    # Signed scalars still accept negatives.
    assert_equal(Int64(py=PythonObject(-1)), -1)


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
