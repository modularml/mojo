# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


# CHECK-LABEL: test_range_len
fn test_range_len():
    print("== test_range_len")

    # CHECK: 10
    print(range(10).__len__())

    # CHECK: 10
    print(range(0, 10).__len__())

    # CHECK: 5
    print(range(5, 10).__len__())

    # CHECK: 10
    print(range(10, 0, -1).__len__())

    # CHECK: 5
    print(range(0, 10, 2).__len__())

    # CHECK: 3
    print(range(38, -13, -23).__len__())


# CHECK-LABEL: test_range_getitem
fn test_range_getitem():
    print("== test_range_getitem")

    # CHECK: 5
    print(range(10)[5])

    # CHECK: 3
    print(range(0, 10)[3])

    # CHECK: 8
    print(range(5, 10)[3])

    # CHECK: 8
    print(range(10, 0, -1)[2])

    # CHECK: 8
    print(range(0, 10, 2)[4])


fn main():
    test_range_len()
    test_range_getitem()
