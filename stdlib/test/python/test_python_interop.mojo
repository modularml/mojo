# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin
# RUN: %mojo -D TEST_DIR=%S -I %py_interop_bin_dir -I %test_py_interop_bin_dir %s | FileCheck %s

from sys.param_env import env_get_string

from python._cpython import CPython, PyObjectPtr
from python.object import PythonObject
from python.python import Python, _get_global_python_itf

alias TEST_DIR = env_get_string["TEST_DIR"]()


fn test_execute_python_string(inout python: Python) -> String:
    try:
        _ = Python.evaluate("print('evaluated by PyRunString')")
        var py_string = Python.evaluate("'a' + 'b'")
        var str = python.__str__(py_string.__str__())
        return String(str)
    except e:
        return e.__str__()


fn test_local_import(inout python: Python) raises:
    try:
        Python.add_to_path(TEST_DIR)
        var my_module: PythonObject = Python.import_module("my_module")
        if my_module:
            _ = my_module.my_function("Mojo")
            var foo = my_module.Foo("apple")
            foo.bar = "orange"
            # CHECK: orange
            print(python.__str__(foo.bar.__str__()))
    except e:
        print(e)


fn test_is_dict():
    var dict = Python.dict()
    var list: PythonObject = [1, 2, 3]
    var cpython = _get_global_python_itf().cpython()
    var should_be_true = cpython.PyDict_Check(dict.py_object)
    print(should_be_true)
    var should_be_false = cpython.PyDict_Check(list.py_object)
    print(should_be_false)


def main():
    var python = Python()
    test_local_import(python)

    # CHECK: [1, 2.4, True, 'False']
    var obj: PythonObject = [1, 2.4, True, "False"]
    print(obj)

    # CHECK: (1, 2.4, True, 'False')
    obj = (1, 2.4, True, "False")
    print(obj)

    # CHECK: None
    obj = None
    print(obj)

    # CHECK: 189
    var my_dictionary = Python.dict()
    my_dictionary["a"] = 189
    var needle = my_dictionary["a"]
    var one_eight_nine = needle.__str__()
    var mojo_one_eight_nine = one_eight_nine.__str__()
    print(mojo_one_eight_nine)

    # Verify key failure
    var menu_dict = Python.dict()
    var key: PythonObject = "starch"
    menu_dict["protein"] = "seitan"

    # CHECK: no starch on the menu
    if Python.is_type(menu_dict.get(key), Python.none()):
        print("no starch on the menu")

    # CHECK: we will eat protein
    if not Python.is_type(menu_dict.get("protein"), Python.none()):
        print("we will eat protein")

    # CHECK: ab
    print(test_execute_python_string(python))

    # CHECK: True
    # CHECK-NEXT: False
    test_is_dict()
