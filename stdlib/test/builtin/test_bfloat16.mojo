# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: !arm
# RUN: %mojo -debug-level full %s | FileCheck %s
from math import *
from testing import *


def test_methods():
    assert_equal(BFloat16(4.4), 4.4)
    assert_equal(BFloat16(4.4) * 0.5, 2.2)
    assert_equal(BFloat16(4.4) / 0.5, 8.8)

    assert_equal(int(BFloat16(3.0)), 3)
    assert_equal(int(BFloat16(3.5)), 3)

    assert_equal(floor(BFloat16(3.0)), 3)
    assert_equal(ceil(BFloat16(3.5)), 4)

    assert_almost_equal(exp(BFloat16(2.0)), 7.375)

    assert_almost_equal(cos(BFloat16(2.0)), -0.416015625)

    assert_almost_equal(BFloat16(4.4).cast[DType.float32](), 4.40625)
    assert_almost_equal(Float32(4.4).cast[DType.bfloat16](), 4.4)


def main():
    # CHECK: 33.0
    print(
        __mlir_op.`pop.cast`[_type = __mlir_type[`!pop.scalar<f64>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"33"> : !pop.scalar<bf16>`],
            ]()
        )
    )

    # CHECK: nan
    print(
        __mlir_op.`pop.cast`[_type = __mlir_type[`!pop.scalar<f64>`]](
            __mlir_op.`kgen.param.constant`[
                _type = __mlir_type[`!pop.scalar<bf16>`],
                value = __mlir_attr[`#pop.simd<"nan"> : !pop.scalar<bf16>`],
            ]()
        )
    )

    test_methods()
