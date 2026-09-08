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

from std.os import abort

from std.python import Python, PythonObject
from std.python._cpython import PyObjectPtr, Py_ssize_t
from std.python.bindings import PythonModuleBuilder, raise_python_exception


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    try:
        var b = PythonModuleBuilder("mojo_module")

        # def_function with return, raising
        b.def_function[takes_zero_raises_returns]("takes_zero_raises_returns")
        b.def_function[takes_one_raises_returns]("takes_one_raises_returns")
        b.def_function[takes_two_raises_returns]("takes_two_raises_returns")
        b.def_function[takes_three_raises_returns]("takes_three_raises_returns")
        b.def_function[takes_seven_raises_returns]("takes_seven_raises_returns")
        b.def_function[takes_eight_raises_returns]("takes_eight_raises_returns")

        # def_function with return, not raising
        b.def_function[takes_zero_returns]("takes_zero_returns")
        b.def_function[takes_one_returns]("takes_one_returns")
        b.def_function[takes_two_returns]("takes_two_returns")
        b.def_function[takes_three_returns]("takes_three_returns")

        # def_function with no return, raising
        b.def_function[takes_zero_raises]("takes_zero_raises")
        b.def_function[takes_one_raises]("takes_one_raises")
        b.def_function[takes_two_raises]("takes_two_raises")
        b.def_function[takes_three_raises]("takes_three_raises")

        # def_function with no return, not raising
        b.def_function[takes_zero]("takes_zero")
        b.def_function[takes_one]("takes_one")
        b.def_function[takes_two]("takes_two")
        b.def_function[takes_three]("takes_three")

        # kwargs test functions
        b.def_function[sum_kwargs_ints]("sum_kwargs_ints")
        b.def_function[sum_pos_arg_and_kwargs]("sum_pos_arg_and_kwargs")
        b.def_function[append_kwarg]("append_kwarg")
        b.def_function[ignore_kwargs]("ignore_kwargs")

        # Direct METH_FASTCALL registration via def_py_c_function overload.
        b.def_py_c_function(fastcall_concat, "fastcall_concat")

        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


def takes_zero_raises_returns() raises -> PythonObject:
    var s = Python().evaluate("getattr(sys.modules['test_module'], 's')")
    if s != "just a python string":
        raise Error("`s` must be 'just a python string'")

    return PythonObject("just another python string")


def takes_one_raises_returns(
    a: PythonObject,
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise Error("input must be 'foo'")
    return a


def takes_two_raises_returns(
    a: PythonObject, b: PythonObject
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise Error("first input must be 'foo'")
    return a + b


def takes_three_raises_returns(
    a: PythonObject, b: PythonObject, c: PythonObject
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise Error("first input must be 'foo'")
    return a + b + c


def takes_seven_raises_returns(
    a: PythonObject,
    b: PythonObject,
    c: PythonObject,
    d: PythonObject,
    e: PythonObject,
    f: PythonObject,
    g: PythonObject,
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise Error("first input must be 'foo'")
    return a + b + c + d + e + f + g


def takes_eight_raises_returns(
    a: PythonObject,
    b: PythonObject,
    c: PythonObject,
    d: PythonObject,
    e: PythonObject,
    f: PythonObject,
    g: PythonObject,
    h: PythonObject,
) raises -> PythonObject:
    if a != PythonObject("foo"):
        raise Error("first input must be 'foo'")
    return a + b + c + d + e + f + g + h


def takes_zero_returns() -> PythonObject:
    try:
        return takes_zero_raises_returns()
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_one_returns(a: PythonObject) -> PythonObject:
    try:
        return takes_one_raises_returns(a)
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_two_returns(a: PythonObject, b: PythonObject) -> PythonObject:
    try:
        return takes_two_raises_returns(a, b)
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_three_returns(
    a: PythonObject, b: PythonObject, c: PythonObject
) -> PythonObject:
    try:
        return takes_three_raises_returns(a, b, c)
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_zero_raises() raises:
    var s = Python().evaluate("getattr(sys.modules['test_module'], 's')")
    if s != "just a python string":
        raise Error("`s` must be 'just a python string'")

    _ = Python().eval(
        "setattr(sys.modules['test_module'], 's', 'Hark! A mojo function"
        " calling into Python, called from Python!')"
    )


def takes_one_raises(list_obj: PythonObject) raises:
    if len(list_obj) != 3:
        raise Error("list_obj must have length 3")
    list_obj[PythonObject(0)] = PythonObject("baz")


def takes_two_raises(list_obj: PythonObject, obj: PythonObject) raises:
    if len(list_obj) != 3:
        raise Error("list_obj must have length 3")
    list_obj[PythonObject(0)] = obj


def takes_three_raises(
    list_obj: PythonObject, obj: PythonObject, obj2: PythonObject
) raises:
    if len(list_obj) != 3:
        raise Error("list_obj must have length 3")
    list_obj[PythonObject(0)] = obj + obj2


def takes_zero():
    try:
        takes_zero_raises()
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_one(list_obj: PythonObject):
    try:
        takes_one_raises(list_obj)
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_two(list_obj: PythonObject, obj: PythonObject):
    try:
        takes_two_raises(list_obj, obj)
    except e:
        abort(String("Unexpected Python error: ", e))


def takes_three(list_obj: PythonObject, obj: PythonObject, obj2: PythonObject):
    try:
        takes_three_raises(list_obj, obj, obj2)
    except e:
        abort(String("Unexpected Python error: ", e))


# ===----------------------------------------------------------------------=== #
# Kwargs Test Functions
# ===----------------------------------------------------------------------=== #


def sum_kwargs_ints(var **kwargs: PythonObject) raises -> PythonObject:
    """Test function that takes kwargs, converts them to Ints, adds them together and returns the sum.
    """
    var total = 0
    for entry in kwargs.items():
        var value = entry.value
        total += Int(py=value)

    return PythonObject(total)


def sum_pos_arg_and_kwargs(
    arg1: PythonObject, var **kwargs: PythonObject
) raises -> PythonObject:
    return PythonObject(Int(py=arg1) + Int(py=sum_kwargs_ints(**kwargs^)))


def append_kwarg(list_obj: PythonObject, var **kwargs: PythonObject) raises:
    list_obj.append(kwargs["value"])


def ignore_kwargs(var **kwargs: PythonObject):
    pass


def fastcall_concat(
    py_self: PyObjectPtr,
    args: Pointer[PyObjectPtr, MutUntrackedOrigin],
    nargs: Py_ssize_t,
) abi("C") -> PyObjectPtr:
    """Hand-written METH_FASTCALL wrapper that concatenates `nargs` strings.

    Exercises the `def_py_c_function(PyCFunctionFast, ...)` overload directly:
    CPython hands us a borrowed `PyObject *const *` array plus the arg count,
    we read each argument out of the array without any tuple packing.
    """
    try:
        var result = PythonObject("")
        # `args` is non-null per the vectorcall protocol (PEP 590), even when
        # `nargs == 0`; read each argument straight out of the borrowed array.
        for i in range(Int(nargs)):
            result = result + PythonObject(from_borrowed=args[unsafe_offset=i])
        return result.steal_data()
    except e:
        return raise_python_exception(e)
