# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


# CHECK-LABEL: test_list
fn test_list():
    print("== test_list")
    # CHECK: 4
    print(len([1, 2.0, 3.14, [-1, -2]]))


# CHECK-LABEL: test_variadic_list
fn test_variadic_list():
    print("== test_variadic_list")

    @parameter
    fn print_list(*nums: Int):
        # CHECK: 5
        # CHECK: 8
        # CHECK: 6
        for num in nums:
            print(num)

        # CHECK: 3
        print(len(nums))

    print_list(5, 8, 6)


fn main():
    test_list()
    test_variadic_list()
