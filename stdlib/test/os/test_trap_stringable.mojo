# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_not
# RUN: not --crash mojo --debug-level full %s 2>&1 | FileCheck %s -check-prefix=CHECK

from os import abort


fn main():
    # CHECK: 123
    abort(123)
