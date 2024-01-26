# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin
# RUN: %mojo -I %py_interop_bin_dir %s | FileCheck %s
from python.object import PythonObject
from python.python import Python


fn test_string_to_python_to_mojo(inout python: Python) raises:
    let py_string = PythonObject("mojo")
    let py_string_capitalized = py_string.capitalize()

    # CHECK: Mojo
    let cap_mojo_string = py_string_capitalized.__str__()
    print(cap_mojo_string)


fn test_range() raises:
    let array_size: PythonObject = 2
    # CHECK: 0
    # CHECK-NEXT: 1
    for i in range(array_size):
        print(i)

    let start: PythonObject = 0
    let end: PythonObject = 4
    # CHECK: 0
    # CHECK-NEXT: 1
    # CHECK-NEXT: 2
    # CHECK-NEXT: 3
    for i in range(start, end):
        print(i)

    let start2: PythonObject = 5
    let end2: PythonObject = 10
    let step: PythonObject = 2
    # CHECK: 5
    # CHECK-NEXT: 7
    # CHECK-NEXT: 9
    for i in range(start2, end2, step):
        print(i)


fn test_python_to_string() raises:
    # CHECK: environ({
    let os = Python.import_module("os")
    print(os.environ)


fn main():
    var python = Python()
    try:
        test_string_to_python_to_mojo(python)
        test_range()
        test_python_to_string()
    except:
        pass
