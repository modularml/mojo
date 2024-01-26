# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from random import randn_float64, random_float64, random_si64, random_ui64, seed


# CHECK-LABEL: test_random
fn test_random():
    print("== test_random")

    # CHECK-LABEL: random_float64 =
    print("random_float64 = ", random_float64(0, 1))

    # CHECK-LABEL: random_si64 =
    print("random_si64 = ", random_si64(-255, 255))

    # CHECK-LABEL: random_ui64 =
    print("random_ui64 = ", random_ui64(0, 255))

    # CHECK-LABEL: randn_float64 =
    print("randn_float64 = ", randn_float64(0, 1))


# CHECK-LABEL: test_seed
fn test_seed():
    print("== test_seed")

    seed(5)

    # CHECK: random_seed_float64 = [[FLOAT64:.*]]
    print("random_seed_float64 = ", random_float64(0, 1))

    # CHECK: random_seed_si64 = [[SI64:.*]]
    print("random_seed_si64 = ", random_si64(-255, 255))

    # CHECK: random_seed_ui64 = [[UI64:.*]]
    print("random_seed_ui64 = ", random_ui64(0, 255))

    seed(5)

    # CHECK: random_seed_float64 = [[FLOAT64]]
    print("random_seed_float64 = ", random_float64(0, 1))

    # CHECK: random_seed_si64 = [[SI64]]
    print("random_seed_si64 = ", random_si64(-255, 255))

    # CHECK: random_seed_ui64 = [[UI64]]
    print("random_seed_ui64 = ", random_ui64(0, 255))


fn main():
    test_random()
    test_seed()
