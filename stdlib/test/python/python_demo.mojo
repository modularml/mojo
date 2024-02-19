# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo -I %py_interop_bin_dir %s | FileCheck %s

from python._cpython import CPython, PyObjectPtr
from python.python import Python


fn main():
    var python = Python()
    try:
        var np = Python.import_module("numpy")
        var cpython = python.impl.cpython()
        var size = 3
        var a = np.random.rand(size, size)
        var b = np.random.rand(size, size)
        var c = np.matmul(a, b)

        # CHECK: [[a2:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a2:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a3:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a4:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a5:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a6:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a7:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a8:[0-9]+.[0-9]+]]
        # CHECK-NEXT: [[a9:[0-9]+.[0-9]+]]
        for i in range(size):
            var row = c[i]
            for j in range(size):
                var w = row[j]
                var x = cpython.PyFloat_AsDouble(w.py_object.value)
                print(x)
    except:
        print("Python failed")
