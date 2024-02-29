# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: not --crash mojo --debug-level full %s 2>&1 | FileCheck %s -check-prefix=CHECK

from debug import abort


fn main():
    # CHECK: 123
    abort(123)
