# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# XFAIL: asan && !system-darwin
# RUN: %mojo %s

from python.object import PythonObject
from python.python import Python
from testing import assert_raises


def test_invalid_len():
    var python = Python()
    var x = PythonObject(42)
    with assert_raises(contains="object has no len()"):
        _ = len(x)


def main():
    test_invalid_len()
