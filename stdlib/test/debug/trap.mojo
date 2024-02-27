# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not --crash mojo --debug-level full %s 2>&1 | FileCheck %s

from debug import trap


fn main():
    # CHECK: hello world
    trap("hello world")
