# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not not mojo --debug-level full %s 2>&1 | FileCheck %s -check-prefix=CHECK

from debug import trap


fn main():
    # CHECK: 123
    trap(123)
