# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s
from math import *
from testing import *
from sys.info import has_neon
from random import randn_float64


def test_methods():
    assert_equal(BFloat16(4.4), 4.4)
    assert_equal(BFloat16(4.4) * 0.5, 2.2)
    assert_equal(BFloat16(4.4) / 0.5, 8.8)

    assert_equal(int(BFloat16(3.0)), 3)
    assert_equal(int(BFloat16(3.5)), 3)

    assert_almost_equal(BFloat16(4.4).cast[DType.float32](), 4.40625)
    assert_almost_equal(Float32(4.4).cast[DType.bfloat16](), 4.4)
    assert_almost_equal(BFloat16(2.0), 2.0)


def test_math():
    assert_equal(exp(BFloat16(2.0)), 7.375)
    assert_equal(cos(BFloat16(2.0)), -0.416015625)

    assert_equal(floor(BFloat16(2.5)), 2.0)
    assert_equal(ceil(BFloat16(2.0)), 2.0)

    assert_equal(min(BFloat16(2.0), BFloat16(3.0)), 2.0)
    assert_equal(max(BFloat16(2.0), BFloat16(3.0)), 3.0)
    assert_true(BFloat16(2.0) > BFloat16(-2.0))
    assert_false(BFloat16(2.0) < BFloat16(-3.0))
    assert_true(BFloat16(2.0) <= BFloat16(2.0))
    assert_true(BFloat16(2.0) >= BFloat16(2.0))
    assert_true(BFloat16(2.0) != BFloat16(3.0))
    assert_false(BFloat16(2.0) != BFloat16(2.0))
    assert_false(BFloat16(2.0) == BFloat16(3.0))
    assert_true(BFloat16(2.0) == BFloat16(2.0))


fn test_bf_primitives():
    # we have to use dynamic values, otherwise these get evaled at compile time.
    let a = randn_float64().cast[DType.bfloat16]()
    let b = randn_float64().cast[DType.bfloat16]()

    print(a + b)
    print(a - b)
    print(a / b)
    print(a * b)
    print(a == b)
    print(a != b)
    print(a <= b)
    print(a >= b)

    print("DONE!")


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

    # TODO re-enable this test when we sort out BF16 support for graviton3 #30525
    @parameter
    if not has_neon():
        test_methods()
        test_math()

        # CHECK: DONE!
        test_bf_primitives()
