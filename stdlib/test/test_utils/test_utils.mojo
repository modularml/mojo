# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from builtin.simd import _simd_apply
from sys import external_call


@always_inline
fn libm_call[
    type: DType, simd_width: Int, fn_fp32: StringLiteral, fn_fp64: StringLiteral
](arg: SIMD[type, simd_width]) -> SIMD[type, simd_width]:
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
        return _simd_apply[_float64_dispatch, type, simd_width](arg)
    return _simd_apply[_float32_dispatch, DType.float32, simd_width](
        arg.cast[DType.float32]()
    ).cast[type]()
