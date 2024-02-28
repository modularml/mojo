# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s -an_argument | FileCheck %s
# We pass an_argument here to avoid the compiler from optimizing the code
# away.

from sys import argv


# CHECK-LABEL: OK
fn main():
    if len(argv()) == 0:
        trap()
    print("== OK")
