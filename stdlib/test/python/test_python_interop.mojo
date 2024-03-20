# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin
# RUN: %mojo -D TEST_DIR=%S %s | FileCheck %s

from sys.param_env import env_get_string

from python._cpython import CPython, PyObjectPtr
from python.object import PythonObject
from python.python import Python, _get_global_python_itf

alias TEST_DIR = env_get_string["TEST_DIR"]()


fn test_execute_python_string(inout python: Python) -> String:
    try:
        _ = Python.evaluate("print('evaluated by PyRunString')")
        return Python.evaluate("'a' + 'b'")
    except e:
        return e


fn test_local_import(inout python: Python) -> String:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            var foo = my_module.Foo("apple")
            foo.bar = "orange"
            return foo.bar
        return "no module, no fruit"
    except e:
        return e


fn test_call(inout python: Python) -> String:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        return str(
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
        return e


def main():
    var python = Python()
    # CHECK: orange
    print(test_local_import(python))

    # CHECK: carrot ('bread', 'rice') fruit=pear {'protein': 'fish', 'cake': 'yes'}
    print(test_call(python))

    # CHECK: [1, 2.4, True, 'False']
    var obj: PythonObject = [1, 2.4, True, "False"]
    print(obj)

    # CHECK: (1, 2.4, True, 'False')
    obj = (1, 2.4, True, "False")
    print(obj)

    # CHECK: None
    obj = None
    print(obj)

    # CHECK: ab
    print(test_execute_python_string(python))
