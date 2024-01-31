# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from tensor import Tensor


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
