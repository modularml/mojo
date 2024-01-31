# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo -debug-level full %s | FileCheck %s


fn main():
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
