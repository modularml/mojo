# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo -I %py_interop_bin_dir %s | FileCheck %s


from CPython import PythonVersion
from python import Python


fn test_python_version(inout python: Python):
    var version = "3.10.8 (main, Nov 24 2022, 08:08:27) [Clang 14.0.6 ]"
    var pythonVersion = PythonVersion(version)
    # CHECK: 3
    print(pythonVersion.major)
    # CHECK: 10
    print(pythonVersion.minor)
    # CHECK: 8
    print(pythonVersion.patch)


fn main():
    var python = Python()
    test_python_version(python)
