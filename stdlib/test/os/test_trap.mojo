# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_not
# RUN: not --crash mojo --debug-level full %s 2>&1 | FileCheck %s

from os import abort


fn main():
    # CHECK: hello world
    abort("hello world")
