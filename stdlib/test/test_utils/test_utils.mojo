# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math.math import _simd_apply, abs, max
from sys import external_call, llvm_intrinsic


@always_inline
fn libm_call[
    type: DType, simd_width: Int, fn_fp32: StringLiteral, fn_fp64: StringLiteral
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
    # TODO: add two strings as parameters for FP32 and FP64 function names in libm, like 'tanhf' and 'tanh'
    @always_inline("nodebug")
    @parameter
    fn _float32_dispatch[
        input_type: DType, result_type: DType
    ](arg: SIMD[input_type, 1]) -> SIMD[result_type, 1]:
        return external_call[fn_fp32, SIMD[result_type, 1]](arg)

    @always_inline("nodebug")
    @parameter
    fn _float64_dispatch[
        input_type: DType, result_type: DType
    ](arg: SIMD[input_type, 1]) -> SIMD[result_type, 1]:
        return external_call[fn_fp64, SIMD[result_type, 1]](arg)

    constrained[type.is_floating_point(), "input type must be floating point"]()

    @parameter
    if type == DType.float64:
        return _simd_apply[simd_width, type, type, _float64_dispatch](arg)
    return _simd_apply[
        simd_width, DType.float32, DType.float32, _float32_dispatch
    ](arg.cast[DType.float32]()).cast[type]()
