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
from std.python._cpython import PyObjectPtr
from std.python.bindings import (
    PythonModuleBuilder,
    check_and_get_arg,
    check_and_get_or_convert_arg,
    check_arguments_arity,
)


@export
def PyInit_mojo_module() abi("C") -> PythonObject:
    # ----------------------------------
    # Create a Python module
    # ----------------------------------

    # This will initialize the Python interpreter and create
    # an extension module with the provided name.

    try:
        var b = PythonModuleBuilder("mojo_module")
        b.def_py_function[case_return_arg_tuple]("case_return_arg_tuple")
        b.def_function[case_raise_empty_error]("case_raise_empty_error")
        b.def_function[case_raise_string_error]("case_raise_string_error")
        b.def_function[case_mojo_raise]("case_mojo_raise")
        b.def_function[case_mojo_mutate]("case_mojo_mutate")
        b.def_function[case_downcast_unbound_type]("case_downcast_unbound_type")
        b.def_py_function[incr_int__wrapper]("incr_int")
        b.def_py_function[add_to_int__wrapper]("add_to_int")
        b.def_function[create_string]("create_string")

        _ = (
            b.add_type[Person]("Person")
            .def_init_defaultable[Person]()
            .def_method[Person.obj_name]("name")
            .def_method[Person.change_name]("change_name")
        )
        _ = b.add_type[Int]("Int").def_init_defaultable[Int]()
        _ = b.add_type[String]("String").def_init_defaultable[String]()
        _ = b.add_type[FailToInitialize](
            "FailToInitialize"
        ).def_init_defaultable[FailToInitialize]()
        return b.finalize()
    except e:
        abort(String("failed to create Python module: ", e))


# ===----------------------------------------------------------------------=== #
# Functions
# ===----------------------------------------------------------------------=== #


def case_return_arg_tuple(
    py_self: PythonObject, args: PythonObject
) -> PythonObject:
    return args


def case_raise_empty_error() -> PythonObject:
    ref cpython = Python().cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetNone(error_type)

    return PythonObject(from_owned=PyObjectPtr())


def case_raise_string_error() -> PythonObject:
    ref cpython = Python().cpython()

    var error_type = cpython.get_error_global("PyExc_ValueError")

    cpython.PyErr_SetString(
        error_type,
        "sample value error".as_c_string_slice().ptr().as_unsafe_any_origin(),
    )

    return PythonObject(from_owned=PyObjectPtr())


# Returning New Mojo Values
def create_string() raises -> PythonObject:
    var result = "Hello"

    return PythonObject(alloc=result^)


def case_mojo_raise() raises -> PythonObject:
    raise Error("Mojo error")


def case_mojo_mutate(list: PythonObject) raises -> PythonObject:
    # this would work even if args was `read`, but we want just to test that
    # the binding API accepts a function that mutates the argument.
    list[0] += 1

    return PythonObject(None)


struct NonBoundType:
    pass


def case_downcast_unbound_type(value: PythonObject) raises:
    var _ptr = value.downcast_value_ptr[NonBoundType]()


# ===----------------------------------------------------------------------=== #
# Custom Types
# ===----------------------------------------------------------------------=== #


@fieldwise_init
struct Person(Defaultable, ImplicitlyCopyable, Writable):
    var name: String
    var age: Int

    def __init__(out self):
        self.name = "John Smith"
        self.age = 123

    @staticmethod
    def obj_name(self_: PythonObject) raises -> PythonObject:
        var self0 = self_.downcast_value_ptr[Self]()

        return PythonObject(self0[].name)

    @staticmethod
    def change_name(
        self_: PythonObject, new_name: PythonObject
    ) raises -> PythonObject:
        var self0 = Pointer[Self, ...](
            unchecked_downcast_value=self_
        ).unsafe_mut_cast[True]()

        if len(new_name) > len(self0[].name.codepoints()):
            raise Error("cannot make name longer than current name")

        self0[].name = String(py=new_name)

        return PythonObject(None)


# ===----------------------------------------------------------------------=== #
# Test: Object Creation Behavior
# ===----------------------------------------------------------------------=== #


struct FailToInitialize(Defaultable, Movable, Writable):
    def __init__(out self):
        pass

    def __deinit__(deinit self):
        abort("FailToInitialize should never be deinitialized.")


# ===----------------------------------------------------------------------=== #
# Recipe Book
# ===----------------------------------------------------------------------=== #

# ====================================
# Recipe: Argument: Arity and argument type checking
# ====================================


def incr_int(mut arg: Int):
    arg += 1


def add_to_int(mut arg: Int, var value: Int):
    arg += value


#
# Manual Wrappers
#


def incr_int__wrapper(
    py_self: PythonObject, py_args: PythonObject
) raises -> PythonObject:
    check_arguments_arity(1, py_args, "incr_int")

    var arg_0: Pointer[Int, MutAnyOrigin] = check_and_get_arg[Int](
        "incr_int", py_args, 0
    )

    # Note: Pass an `mut` reference to the wrapped function
    incr_int(arg_0[])

    return PythonObject(None)


def add_to_int__wrapper(
    py_self: PythonObject, py_args: PythonObject
) raises -> PythonObject:
    check_arguments_arity(2, py_args, "add_to_int")

    var arg_0: Pointer[Int, MutAnyOrigin] = check_and_get_arg[Int](
        "add_to_int", py_args, 0
    )

    var arg_1: Pointer[Int, MutAnyOrigin] = check_and_get_or_convert_arg[Int](
        "add_to_int",
        py_args,
        1,
    )

    # Note: Pass an `mut` reference to the wrapped function
    add_to_int(arg_0[], arg_1[])

    return PythonObject(None)
