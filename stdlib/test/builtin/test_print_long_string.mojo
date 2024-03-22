# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s > %t
# RUN: wc -c %t | FileCheck %s


fn main():
    # CHECK: 536870913
    print(String("*") * 0x2000_0000)
