# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin
# RUN: %mojo -D TEST_DIR=%S -I %py_interop_bin_dir -I %test_py_interop_bin_dir %s | FileCheck %s

from sys.param_env import env_get_string

from memory.unsafe import Pointer
from python._cpython import CPython, PyObjectPtr
from python.object import PythonObject
from python.python import Python

alias TEST_DIR = env_get_string["TEST_DIR"]()


fn test_import(inout python: Python) raises -> String:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        var py_string = my_module.my_function("Hello")
        var str = String(python.__str__(py_string))
        return str
    except e:
        return e.__str__()


fn test_list(inout python: Python) raises -> String:
    try:
        var b: PythonObject = Python.import_module("builtins")
        var my_list = PythonObject([1, 2.34, "False"])
        var py_string = my_list.__str__()
        return String(python.__str__(py_string))
    except e:
        return e.__str__()


fn test_tuple(inout python: Python) raises -> String:
    try:
        var b: PythonObject = Python.import_module("builtins")
        var my_tuple = PythonObject((1, 2.34, "False"))
        var py_string = my_tuple.__str__()
        return String(python.__str__(py_string))
    except e:
        return e.__str__()


fn test_call_ownership(inout python: Python) raises -> String:
    try:
        var obj: PythonObject = [1, "5"]
        var py_string = obj.__str__()
        var string = python.__str__(py_string)
        return String(string)
    except e:
        return e.__str__()


fn test_getitem_ownership(inout python: Python) raises -> String:
    try:
        var obj: PythonObject = [1, "5"]
        var py_string = obj[1].__str__()
        var string = python.__str__(py_string)
        return String(string)
    except e:
        return e.__str__()


fn test_getattr_ownership(inout python: Python) raises -> String:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        var obj = my_module.Foo(4)
        var py_string = obj.bar.__str__()
        var string = python.__str__(py_string)
        return String(string)
    except e:
        return e.__str__()


def main():
    # initializing Python instance calls init_python
    var python = Python()

    # CHECK: [1, 2.34, 'False']
    print(test_list(python))

    # CHECK: (1, 2.34, 'False')
    print(test_tuple(python))

    # CHECK: [1, '5']
    print(test_call_ownership(python))

    # CHECK: 5
    print(test_getitem_ownership(python))

    # CHECK: 4
    print(test_getattr_ownership(python))

    # CHECK: Formatting the string from Lit with Python: Hello
    print(test_import(python))
