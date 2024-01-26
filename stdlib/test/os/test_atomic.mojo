# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s

from os.atomic import Atomic


# CHECK-LABEL: test_atomic
fn test_atomic():
    print("== test_atomic")

    var atom: Atomic[DType.index] = 3

    # CHECK: 3
    print(atom.value)

    atom += 4

    # CHECK: 7
    print(atom.value)

    atom -= 4

    # CHECK: 3
    print(atom.value)

    # CHECK: 3
    atom.max(0)
    print(atom.value)

    # CHECK: 42
    atom.max(42)
    print(atom.value)

    # CHECK: 3
    atom.min(3)
    print(atom.value)

    # CHECK: 0
    atom.min(0)
    print(atom.value)


# CHECK-LABEL: test_atomic_floating_point
fn test_atomic_floating_poInt__():
    print("== test_atomic_floating_point")

    var atom: Atomic[DType.float32] = Float32(3.0)

    # CHECK: 3.0
    print(atom.value)

    atom += 4

    # CHECK: 7.0
    print(atom.value)

    atom -= 4

    # CHECK: 3.0
    print(atom.value)

    # CHECK: 3.0
    atom.max(0)
    print(atom.value)

    # CHECK: 42.0
    atom.max(42)
    print(atom.value)

    # CHECK: 3.0
    atom.min(3)
    print(atom.value)

    # CHECK: 0.0
    atom.min(0)
    print(atom.value)


fn main():
    test_atomic()
    test_atomic_floating_poInt__()
