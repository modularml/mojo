# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
"""Implements math methods that work on layout tensors."""

import std.math
from std.sys.info import simd_width_of

import max.algorithm.reduction as reduction
from std.algorithm import vectorize
from std.math.math import max as b_max
from layout import (
    Coord,
    Idx,
    IntTuple,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    RuntimeTuple,
    TileTensor,
    UNKNOWN_VALUE,
)

from std.utils.index import IndexList


@always_inline
def outer_product_acc(
    res: LayoutTensor[mut=True, ...],
    lhs: LayoutTensor,
    rhs: LayoutTensor,
):
    """Updates result tensor with the outer product of two vectors.

    Computes `res += outer(lhs, rhs)` where `lhs` and `rhs` are vectors and
    `res` is a matrix.

    Args:
        res: The result matrix to accumulate into, shape (M, N).
        lhs: The left-hand side vector, shape (M,).
        rhs: The right-hand side vector, shape (N,).

    Constraints:

        All tensors must have statically known shapes.
        `res` must be rank 2.
        `lhs` and `rhs` must be rank 1.
        `res.shape[0]` `==` `lhs.shape[0]` and `res.shape[1]` `==` `rhs.shape[0]`.
    """

    comptime assert (
        res.layout.known_shape()
        and lhs.layout.known_shape()
        and rhs.layout.known_shape()
    ), "outer_product_acc expects inputs with statically known shapes"
    comptime assert res.rank == 2, "Only rank 2 res is allowed."
    comptime assert lhs.rank == 1, "Only rank 1 lhs is allowed."
    comptime assert rhs.rank == 1, "Only rank 1 rhs is allowed."

    comptime dtype = res.dtype

    comptime M = res.shape[0]()
    comptime N = res.shape[1]()

    comptime assert lhs.shape[0]() == M, "lhs shape mismatch"
    comptime assert rhs.shape[0]() == N, "rhs shape mismatch"

    comptime for i in range(M):
        comptime for j in range(N):
            res[i, j] += rebind[res.element_type](
                lhs[i].cast[dtype]()
            ) * rebind[res.element_type](rhs[j].cast[dtype]())


@always_inline
def _reduce[
    axis: Int,
    init_func: def[dtype: DType, width: Int]() thin -> SIMD[dtype, width],
    func: def[dtype: DType, width: SIMDLength](
        SIMD[dtype, width], SIMD[dtype, width]
    ) thin -> (SIMD[dtype, width]),
](inp: LayoutTensor, outp: LayoutTensor[mut=True, ...]):
    comptime assert (
        inp.layout.known_shape() and outp.layout.known_shape()
    ), "_reduce expects inputs with statically know shapes"
    comptime assert (
        inp.rank - 1 == outp.rank
    ), "_reduce expects output of rank = inp.rank - 1"

    comptime for dim in range(axis):
        comptime if dim != axis:
            comptime assert dim != UNKNOWN_VALUE
            comptime assert (
                inp.shape[dim]() == outp.shape[dim]()
            ), "_reduce expects none reduction dims to be the same"

    comptime for dim in range(axis + 1, inp.rank):
        comptime if dim != axis:
            comptime assert dim != UNKNOWN_VALUE
            comptime assert (dim - 1) != UNKNOWN_VALUE
            comptime assert (
                inp.shape[dim]() == outp.shape[dim - 1]()
            ), "_reduce expects none reduction dims to be the same"

    # TODO(KERN-777): We need to relax this constraine.
    comptime assert inp.rank == 2, "Only rank-2 _reduce is supported"

    comptime if inp.rank == 2 and axis == 1:
        comptime for i in range(inp.shape[0]()):
            var reduce_val = init_func[outp.dtype, outp.element_size]()

            comptime for j in range(inp.shape[1]()):
                reduce_val = func(
                    reduce_val,
                    rebind[outp.element_type](inp[i, j].cast[outp.dtype]()),
                )

            outp[i] = reduce_val

    elif inp.rank == 2 and axis == 0:
        comptime for j in range(inp.shape[1]()):
            var reduce_val = init_func[outp.dtype, outp.element_size]()

            comptime for i in range(inp.shape[0]()):
                reduce_val = func(
                    reduce_val,
                    rebind[outp.element_type](inp[i, j].cast[outp.dtype]()),
                )

            outp[j] = reduce_val


@always_inline
def sum[axis: Int](inp: LayoutTensor, outp: LayoutTensor[mut=True, ...]):
    """Computes sum reduction along specified axis.

    Reduces the input tensor by summing elements along the specified axis
    and stores the result in the output tensor.

    Parameters:
        axis: The axis to sum along.

    Args:
        inp: The input tensor to sum.
        outp: The output tensor to store sum results.

    Constraints:
        All tensors must have statically known shapes.
        `outp.rank` must equal `inp.rank - 1`.
        Non-reduction dimensions must match between inp and outp.
        Currently only supports rank-2 inputs.

    Example:

    ```mojo
    from layout import LayoutTensor, Layout
    from layout.math import sum

    data: Array[Int32, 6] = [0, 1, 2, 3, 4, 5]
    tensor = LayoutTensor[.int32, Layout.row_major(2, 3)](data)
    print(tensor)
    print("-----")
    print(sum[0](tensor))
    ```

    Output:

    ```plaintext
    0 1 2
    3 4 5
    -----
    3 5 7
    ```
    """

    def sum_init[dtype: DType, width: Int]() -> SIMD[dtype, width]:
        return 0

    def sum_func[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return a + b

    _reduce[axis, sum_init, sum_func](inp, outp)


@always_inline
def max[axis: Int](inp: LayoutTensor, outp: LayoutTensor[mut=True, ...]):
    """Computes maximum reduction along specified axis.

    Reduces the input tensor by taking maximum elements along the specified
    axis and stores the result in the output tensor.

    Parameters:
        axis: The axis to take maximum along.

    Args:
        inp: The input tensor to reduce.
        outp: The output tensor to store maximum results.

    Constraints:
        All tensors must have statically known shapes.
        `outp.rank` must equal `inp.rank - 1`.
        Non-reduction dimensions must match between `inp` and `outp`.
        Currently only supports rank-2 inputs.
    """

    def max_init[dtype: DType, width: Int]() -> SIMD[dtype, width]:
        return SIMD[dtype, width].MIN

    def max_func[
        dtype: DType, width: SIMDLength
    ](a: SIMD[dtype, width], b: SIMD[dtype, width]) -> SIMD[dtype, width]:
        return b_max(a, b)

    _reduce[axis, max_init, max_func](inp, outp)


def _reduce_res_row_major_shape(axis: Int, in_layout: Layout) -> Layout:
    var res_shape = IntTuple()
    for dim in range(0, axis):
        res_shape.append(Int(in_layout.shape[dim]))
    for dim in range(axis + 1, in_layout.rank()):
        res_shape.append(Int(in_layout.shape[dim]))
    return Layout.row_major(res_shape)


@always_inline
def max[
    axis: Int
](
    inp: LayoutTensor,
    out res: LayoutTensor[
        inp.dtype,
        _reduce_res_row_major_shape(axis, inp.layout),
        MutAnyOrigin,
        address_space=inp.address_space,
        element_layout=inp.element_layout,
        layout_int_type=inp.layout_int_type,
        linear_idx_type=inp.linear_idx_type,
    ],
):
    """Computes maximum reduction along specified axis, returning a new tensor.

    Reduces the input tensor by taking maximum elements along the specified
    axis and returns a new tensor with the results.

    Parameters:
        axis: The axis to take maximum along.

    Args:
        inp: The input tensor to reduce.

    Returns:
        A new tensor containing the maximum values along the specified axis.

    Constraints:
        All tensors must have statically known shapes.
        Result will have rank equal to `inp.rank` - 1.
        Non-reduction dimensions in the result match the input.
        Currently only supports rank-2 inputs.
    """

    var res_tensor = type_of(res).stack_allocation()
    max[axis](inp, res_tensor)
    return res_tensor


@always_inline
def max[
    dtype: DType, layout: Layout
](
    x: LayoutTensor[dtype, layout, ...], y: LayoutTensor[dtype, layout, ...]
) -> type_of(x).MutableAnyType:
    """Computes element-wise maximum of two tensors.

    Returns a new tensor containing the element-wise maximum between the
    input tensors.

    Parameters:
        dtype: The data type of the input tensors.
        layout: The layout of the input tensors.

    Args:
        x: First input tensor.
        y: Second input tensor.

    Returns:
        A new tensor containing the element-wise maximum.

    Constraints:
        Input tensors must have statically known shapes and matching layouts.
    """

    comptime assert (
        x.layout.all_dims_known()
    ), "max expects tensor of statically know shape"
    var res_tensor = type_of(x).stack_allocation()

    comptime for i in range(res_tensor.layout.size()):
        comptime idx = x.layout(i)
        res_tensor.ptr[idx] = b_max(x.ptr[idx], y.ptr[idx])
    return res_tensor


@always_inline
def sum[
    axis: Int,
](
    inp: LayoutTensor,
    out res: LayoutTensor[
        inp.dtype,
        _reduce_res_row_major_shape(axis, inp.layout),
        MutAnyOrigin,
        address_space=inp.address_space,
        element_layout=inp.element_layout,
        layout_int_type=inp.layout_int_type,
        linear_idx_type=inp.linear_idx_type,
    ],
):
    """Computes sum reduction along specified axis, returning a new tensor.

    Reduces the input tensor by summing elements along the specified axis
    and returns a new tensor with the results.

    Parameters:
        axis: The axis to sum along.

    Args:
        inp: The input tensor to sum.

    Returns:
        A new tensor containing the sum values along the specified axis.

    Constraints:
        All tensors must have statically known shapes.
        Result will have rank equal to `inp.rank` - 1.
        Non-reduction dimensions in the result match the input.
        Currently only supports rank-2 inputs.
    """

    var res_tensor = type_of(res).stack_allocation()
    sum[axis](inp, res_tensor)
    return res_tensor


def mean(src: LayoutTensor) raises -> Scalar[src.dtype]:
    """Computes the mean value of the elements in a buffer.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """
    comptime assert src.rank == 1, "src must be of rank 1"

    assert src.size() != 0, "input must not be empty"

    @__parameter
    @always_inline
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        var src_idx = src.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
        )
        return rebind[SIMD[dtype_, width]](src.ptr.load[width=width](src_idx))

    return reduction.mean[src.dtype, input_fn_1d](src.size())


def mean[
    reduce_axis: Int
](src: LayoutTensor, dst: LayoutTensor[mut=True, src.dtype, ...]) raises:
    """Computes the mean across reduce_axis of a LayoutTensor.

    Parameters:
        reduce_axis: The axis to reduce across.

    Args:
        src: The input buffer.
        dst: The output buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """
    comptime simd_width = simd_width_of[dst.dtype]()
    sum[reduce_axis](src, dst)

    var n = src.dim[reduce_axis]()
    var dst_1d = LayoutTensor[
        dst.dtype,
        Layout.row_major(UNKNOWN_VALUE),
        address_space=dst.address_space,
    ](
        dst.ptr,
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](dst.size())
        ),
    )

    comptime src_dtype = src.dtype

    comptime if dst.dtype.is_integral():

        @always_inline
        def normalize_integral[simd_width: Int](idx: Int) {var dst_1d, var n}:
            var idx_1d = dst_1d.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
            )
            var elem = dst_1d.ptr.load[width=simd_width](idx_1d)
            var to_store = elem // SIMD[src_dtype, simd_width](n)
            dst_1d.ptr.store(idx_1d, to_store)

        vectorize[simd_width](dst_1d.size(), normalize_integral)
    else:
        var n_recip = Scalar[dst.dtype](1) / Scalar[src.dtype](n)

        @always_inline
        def normalize_floating[
            simd_width: Int
        ](idx: Int) {var dst_1d, var n, var n_recip}:
            var idx_1d = dst_1d.runtime_layout(
                RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
            )
            var elem = dst_1d.ptr.load[width=simd_width](idx_1d)
            var to_store = elem * n_recip
            dst_1d.ptr.store(idx_1d, to_store)

        vectorize[simd_width](dst_1d.size(), normalize_floating)


def variance(
    src: LayoutTensor, correction: Int = 1
) raises -> Scalar[src.dtype]:
    """Computes the variance value of the elements in a buffer.

    ```
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Args:
        src: The buffer.
        correction: Normalize variance by size - correction (Default=1).

    Returns:
        The variance value of the elements in a buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """

    @always_inline
    @__parameter
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        var src_idx = src.runtime_layout(
            RuntimeTuple[IntTuple(UNKNOWN_VALUE)](idx)
        )
        return rebind[SIMD[dtype_, width]](src.ptr.load[width=width](src_idx))

    return reduction.variance[src.dtype, input_fn_1d](src.size(), correction)


def variance(src: TileTensor, correction: Int = 1) raises -> Scalar[src.dtype]:
    """Computes the variance value of the elements in a buffer.

    ```
    variance(x) = sum((x - E(x))^2) / (size - correction)
    ```

    Args:
        src: The buffer.
        correction: Normalize variance by size - correction (Default=1).

    Returns:
        The variance value of the elements in a buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """

    @always_inline
    @__parameter
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        var src_idx = src.layout(idx)
        return rebind[SIMD[dtype_, width]](src.raw_load[width=width](src_idx))

    return reduction.variance[src.dtype, input_fn_1d](
        src.num_elements(), correction
    )


def mean(src: TileTensor) raises -> Scalar[src.dtype]:
    """Computes the mean value of the elements in a buffer.

    Args:
        src: The buffer of elements for which the mean is computed.

    Returns:
        The mean value of the elements in the given buffer.

    Raises:
        May raise on GPU targets when a device error occurs.
    """
    comptime assert src.rank == 1, "src must be of rank 1"

    assert src.num_elements() != 0, "input must not be empty"

    @__parameter
    @always_inline
    def input_fn_1d[
        dtype_: DType, width: Int
    ](idx: Int) capturing -> SIMD[dtype_, width]:
        var src_idx = src.layout(idx)
        return rebind[SIMD[dtype_, width]](src.raw_load[width=width](src_idx))

    return reduction.mean[src.dtype, input_fn_1d](src.num_elements())
