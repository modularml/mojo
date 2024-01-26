# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from memory.unsafe import bitcast


# CHECK-LABEL: test_bitcast
fn test_bitcast():
    print("== test_bitcast")

    # CHECK: [1, 0, 2, 0, 3, 0, 4, 0]
    print(bitcast[DType.int8, 8](SIMD[DType.int16, 4](1, 2, 3, 4)))

    # CHECK: 1442775295
    print(bitcast[DType.int32, 1](SIMD[DType.int8, 4](0xFF, 0x00, 0xFF, 0x55)))


fn main():
    test_bitcast()
