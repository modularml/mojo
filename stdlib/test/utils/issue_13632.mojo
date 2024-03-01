# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from collections import List


fn sum_items(data: List[Int8]) -> Int:
    var sum: Int = 0
    for i in range(len(data)):
        sum += int(data[i])
    return sum


fn make_abcd_vector() -> List[Int8]:
    var v = List[Int8]()
    v.push_back(97)
    v.push_back(98)
    v.push_back(99)
    v.push_back(100)
    return v


fn main():
    var vec = make_abcd_vector()
    # CHECK: 394
    print(sum_items(vec))
