# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: aarch64
# COM: TODO (17471): Not all aarch64 have neon, so we need to guard against that,
# for now just require apple-m1.
# REQUIRES: apple-m1
# RUN: %mojo -debug-level %s | FileCheck %s

from sys.info import alignof, has_avx512f, has_neon, simdbitwidth


# CHECK-LABEL: test_arch_query
fn test_arch_query():
    print("== test_arch_query")

    # CHECK: True
    print(has_neon())

    # CHECK: 128
    print(simdbitwidth())

    # CHECK: False
    print(has_avx512f())


fn main():
    test_arch_query()
