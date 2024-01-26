# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


# CHECK-LABEL: test_none_end_folds
fn test_none_end_folds():
    print("== test_none_end_folds")
    alias all_def_slice = slice(0, None, 1)
    #      CHECK: 0
    # CHECK-SAME: 1
    print(all_def_slice.start, all_def_slice.end, all_def_slice.step)


fn main():
    test_none_end_folds()
