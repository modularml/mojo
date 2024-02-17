# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math.math import _simd_apply, abs, max
from sys import external_call, llvm_intrinsic

from tensor import Tensor, TensorShape


fn linear_fill[
    type: DType
](inout t: Tensor[type], elems: VariadicList[SIMD[type, 1]]) raises:
    if t.num_elements() != len(elems):
        raise Error("must fill all elements of tensor")

    let buf = t._to_buffer()
    for i in range(t.num_elements()):
        buf[i] = elems[i]


fn linear_fill[
    type: DType
](inout t: Tensor[type], *elems: SIMD[type, 1]) raises:
    linear_fill(t, elems)


fn get_minmax[dtype: DType](x: Tensor[dtype], N: Int) -> Tensor[dtype]:
    var max_val = x[0]
    var min_val = x[0]
    for i in range(1, N):
        if x[i] > max_val:
            max_val = x[i]
        if x[i] < min_val:
            min_val = x[i]
    return Tensor[dtype](TensorShape(2), min_val, max_val)


fn compare[_dtype: DType, N: Int](x: Tensor, y: Tensor, label: String):
    var atol = Tensor[_dtype](TensorShape(N))
    var rtol = Tensor[_dtype](TensorShape(N))

    for i in range(N):
        let xx = x[i].cast[_dtype]()
        let yy = y[i].cast[_dtype]()

        let d = abs[_dtype, 1](xx - yy)
        let e = abs[_dtype, 1](d / yy)
        atol[i] = d
        rtol[i] = e

    print(label)
    let atol_minmax = get_minmax[_dtype](atol, N)
    let rtol_minmax = get_minmax[_dtype](rtol, N)
    print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
    print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
    print("==========================================================")


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
