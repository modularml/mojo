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
# GENERATED FILE, DO NOT EDIT MANUALLY!
# ===----------------------------------------------------------------------=== #

from collections.abc import Sequence
from typing import overload

import max._core
import max._core.dialects.builtin
import max._core.dialects.kgen
import max._core.dialects.mo
import max._core.dialects.mosh
import max._core.dtype
from max.mlir import Location

from . import passes as passes

# C++ overloads on different int types look the same in Python, ignore these
# mypy: disable-error-code="overload-cannot-match"

class MoReduceArgMaxOp(max._core.Operation):
    """
    This op is equivalent to reduce_max, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical maximum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %1 = rmo.mo.reduce.arg_max(%0) {axis = 1 : index} : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceArgMinOp(max._core.Operation):
    """
    This op is equivalent to reduce_min, but returns indices instead of values.

    The axis attribute specifies the reduction axis.

    Like reductions, the output shape is the same as the input shape, except for
    the reduced axis which is set to 1. Moreover, the value of `axis` follows
    numpy semantics, e.g., -1 represents the last axis.

    For identical minimum values, the lowest index is returned.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 3, 2> : tensor<2x2xsi32>
      } : !mo.tensor<[2, 2], si32>
      %axis = mo.constant {
        value = #M.dense_array<1> : tensor<si32>
      } : !mo.tensor<[], si32>
      %1 = rmo.mo.reduce.arg_min(%0, %axis) : (!mo.tensor<[2, 2], si32>) -> !mo.tensor<[2, 1], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceMaxOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their maximum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.max(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceMinOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their minimum, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.min(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoAtanhOp(max._core.Operation):
    """
    Computes `atanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoAbsOp(max._core.Operation):
    """
    Returns `abs(x)`, where `x` is the input tensors.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.abs(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoAddOp(max._core.Operation):
    """Does a non-broadcasted elementwise addition."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoAndOp(max._core.Operation):
    """
    Returns `x and y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.and(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                      !mo.tensor<[2, 3], bool>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoArgNonzeroOp(max._core.Operation):
    """
    Returns a tensor of coordinates of the nonzero values in the given tensor.
    The return value is a 2D tensor of shape [nnz x rank_in], where nnz is the
    number of nonzero elements in the input tensor, and rank_in is the rank of
    the input tensor. Coordinates are generated in row-major order.

    Example:

    ```mlir
      %0 = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8> : tensor<3x3xsi32>
      } : !mo.tensor<[3, 3], si32>
      %1 = rmo.mo.arg_nonzero(%0) : (!mo.tensor<[3, 3], si32>) -> !mo.tensor<[?, 2], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoAvgPoolCeilModeTrueOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2d avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.mo.avg_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.Value[max._core.dialects.mo.TensorType],
        dilations: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        count_boundary: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def dilations(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoAvgPoolOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    The op supports 2D avg pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. `strides`, `dilations`, `padding`) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.mo.avg_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.Value[max._core.dialects.mo.TensorType],
        dilations: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        count_boundary: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def dilations(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoLinalgBandPartOp(max._core.Operation):
    """
    Copies a tensor setting everything outside central (diagonal) band of the
    matrices to zero, where all but the last two axes are effectively batches,
    and the last two axes define sub matrices.

    Assumes the input has dimensions [I, J, ..., M, N], then the output tensor
    has the same shape as the input, and the values values are given by

    out[i, j, ..., m, n] = in_band(m, n) * input[i, j,  ..., m, n].

    With the indicator function

    in_band(m, n) = ((num_lower < 0 || (m - n) <= num_lower)) &&
                     (num_upper < 0 || (n - m) <= num_upper))

    If `exclude` is set, the selection is reverted: The elements in band are set
    to zero while the elements outside the band are copied to the output tensor.

    Please explicitly note that with negative values, this kernel returns the
    entire lower or upper triangle of the matrix, and otherwise returns
    a diagonal band around the main diagonal of the matrix.

    Example:

    ```mlir
      %arg: !mo.tensor<[3, 2, 3], f32>
      %num_lower = mo.constant {
        value = #M.dense_array<-1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %num_upper = mo.constant {
        value = #M.dense_array<1> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %exclude = mo.constant {
        value = #M.dense_array<0> : tensor<1xui8>} : !mo.tensor<[], bool>
      %res = rmo.mo.linalg.band_part(%arg, %num_lower, %num_upper, %exclude) : (
        !mo.tensor<[3, 2, 3], f32>, !mo.tensor<[], si64>, !mo.tensor<[], si64>,
        !mo.tensor<[], bool>
        ) -> !mo.tensor<[3, 2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        num_lower: max._core.Value[max._core.dialects.mo.TensorType],
        num_upper: max._core.Value[max._core.dialects.mo.TensorType],
        exclude: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def num_lower(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def num_upper(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def exclude(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoBottomKOp(max._core.Operation):
    """
    Computes the bottom (lowest) values and their corresponding indices in a
    tensor along a specified axis. Returned values along the axis are always
    sorted (stable).

    Example:
    ```mlir
      %in = mo.constant {
        value = #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11> : tensor<2x6xsi64>
      } : !mo.tensor<[2, 6], si64>
      %k = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<3> : tensor<si64> } : !mo.tensor<[], si64>
      %axis = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64> } : !mo.tensor<[], si64>
      %sorted = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<1xi1> } : !mo.tensor<[], bool>
      %values, %indices = rmo.mo.bottom_k(%in, %k, %axis, %sorted) : (
        !mo.tensor<[2, 6], si64>, !mo.tensor<[], si64>, !mo.tensor<[], si64>, !mo.tensor<[], bool>
      ) -> (
        !mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>
      )
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: max._core.dialects.mo.TensorType,
        indices: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        _k: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.Value[max._core.dialects.mo.TensorType],
        sorted: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def _k(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def sorted(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoBroadcastShapeOp(max._core.Operation):
    """
    Given two tensors representing shapes, calculate the result of broadcasting
    the shapes.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoBroadcastToOp(max._core.Operation):
    """Broadcast the given `input` to the shape represented in `newShape`."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def new_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoCastOp(max._core.Operation):
    """Casts the given data, changing from one dtype to another."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        dtype: max._core.dtype.DType,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoCeilOp(max._core.Operation):
    """
    Returns the elementwise smallest integer greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.ceil(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoConvOp(max._core.Operation):
    """
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout RSCF, i.e.,
      (height, width, in_channels / num_groups, out_channels)

    `strides`, `dilations`, and `padding` must be of rank 1, or unranked.
    If the input has static rank, all hyperparameters with static shape must
    have sizes of `input_rank - 2`, except padding, which must have
    size `2 * (input_rank - 2)`. Individual elements in the hyperparameters
    apply to corresponding dimensions of the input (after ignoring the batch
    and channel dimensions), with padding representing a before/after pair for
    each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here represents H and dim2 represents W. In python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter tensors' ranks must match if statically known.

    `num_groups` must be a ranked scalar. The number of input and output
    channels must both be divisible by the number of groups `num_groups`.

    This op currently only supports strides and padding on the input.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %ng = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1> : tensor<si64>}
        : %!mo.tensor<[], si64>
      %res = rmo.mo.conv(%input, %filter) [strides = %st, dilations = %di, paddings = %pa, num_groups = %ng] : (
        !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>, !mo.tensor<[], si64>
      ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.Value[max._core.dialects.mo.TensorType],
        dilations: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        num_groups: max._core.Value[max._core.dialects.mo.TensorType],
        input_layout: max._core.dialects.builtin.StringAttr,
        filter_layout: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def dilations(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def num_groups(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def filter_layout(self) -> str: ...
    @filter_layout.setter
    def filter_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoConvTransposeOp(max._core.Operation):
    """
    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The op supports 1D-3D spatial dimensions, with the following layout
    assumptions (note the `out_channel` is w.r.t. the original convolution):
    - input has channel last layout.For 2D, that's NHWC, i.e.,
      (batch_size, height, width, out_channels)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters (i.e. strides, dilations, padding, output_paddings) must
    be of rank 1, or unranked. If the input has static rank, all hyperparameters
    with static shape must have sizes of `input_rank - 2`, except padding, which
    must have size `2 * (input_rank - 2)`. Individual elements in the
    hyperparameters applies to corresponding dimensions of the input (after
    ignoring the batch and channel dimensions), with padding representing a
    before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    The `output_paddings` argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.

    The input, output and filter tensors' ranks must match if statically known.

    Example:

    ```mlir
      %st = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %di = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1, 1> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %pa = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>}
        : !mo.tensor<[4], si64>
      %op = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0> : tensor<2xsi64>}
        : !mo.tensor<[2], si64>
      %res = rmo.mo.conv_transpose(%input, %filter)
        [strides = %st, dilations = %di, paddings = %pa, output_paddings = %op] : (
        !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.Value[max._core.dialects.mo.TensorType],
        dilations: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        output_paddings: max._core.Value[max._core.dialects.mo.TensorType],
        input_layout: max._core.dialects.builtin.StringAttr,
        filter_layout: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def dilations(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_paddings(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def filter_layout(self) -> str: ...
    @filter_layout.setter
    def filter_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoCosOp(max._core.Operation):
    """
    Returns `cos(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.mo.cos(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoCumsumOp(max._core.Operation):
    """
    Returns the cumulative summation of input tensors along an axis. By default,
    it copies the first element as is. If the `exclusive` attribute is set to 1,
    then the first element is excluded. The `reverse` attribute causes the
    summation to be done in the opposite direction of the axis.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example of outputs:

    ```
      input_x = [1, 2, 3]
      axis=0
      output = [1, 3, 6]
      exclusive=1
      output = [0, 1, 3]
      exclusive=0
      reverse=1
      output = [6, 5, 3]
      exclusive=1
      reverse=1
      output = [5, 3, 0]
    ```

    Example:

    ```mlir
    %arg: !mo.tensor<[2, 3], f32>
    %res = rmo.mo.cumsum(%arg) {axis = 0 : index, exclusive = 1 : index, reverse = 0 : index} : (
      !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        exclusive: max._core.dialects.builtin.IntegerAttr,
        reverse: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def exclusive(self) -> int: ...
    @exclusive.setter
    def exclusive(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def reverse(self) -> int: ...
    @reverse.setter
    def reverse(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoDivOp(max._core.Operation):
    """Does a non-broadcasted elementwise division."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoEqualOp(max._core.Operation):
    """
    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                        !mo.tensor<[2, 3], f32>
                                        ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoErfOp(max._core.Operation):
    """
    Computes the Gauss error function of the input tensor elements.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.erf(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoExpOp(max._core.Operation):
    """
    Returns `exp(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.exp(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoFloorOp(max._core.Operation):
    """
    Returns the elementwise largest integer not greater than `x`, where `x` is
    input tensor.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.floor(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGatherNdOp(max._core.Operation):
    """
    Variant of `mo.gather` that accepts multi-dimensional indices.

    The last dimension stores the index whereas
    the outer dimensions act like batch dimensions. The size of the last
    dimension is at most the rank of the input. When the dimension size is less
    than the rank of the input, slices of the input are gathered, starting from
    the leftmost dimension.

    ```
    output_shape = (
        batch_dims_shape + list(indices.shape)[batch_dims:-1]
        if (indices.shape[-1] == data_rank - batch_dims)
        else batch_dims_shape
        + list(indices.shape)[batch_dims:-1]
        + list(data.shape)[batch_dims + indices.shape[-1] :]
    )
    ```

    ```mlir
      %input = mo.constant {device = #M.device_ref<"cpu", 0>, value =
        #M.dense_array<0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15> :
        tensor<2x2x4xsi64>} : !mo.tensor<[2, 2, 4], si64>
      %indices = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<0, 0, 0> : tensor<3xsi64>} :
        !mo.tensor<[3], si64>

      %result = rmo.mo.gather_nd(%input, %indices) {batchDims = 0} :
        (!mo.tensor<[2, 2, 4], si64>, !mo.tensor<[3], si64>) ->
        !mo.tensor<[], si64>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        batch_dims: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def batch_dims(self) -> int: ...
    @batch_dims.setter
    def batch_dims(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGatherOp(max._core.Operation):
    """
    Gathers slices from input's axis according to indices.

    If input and indices are statically ranked, the output rank must be
    `inputRank - 1 + indicesRank`. In general, the output satisfies the
    following:
    ```
    output[a_0, ..., a_n, i, ..., j, b_0, ... b_n] =
      input[a_0, ..., a_n, indices[i, ..., j], b_0, ..., b_n]
    ```
    where `indices` appears at given axis of input.

    The values of `axis` and `indices` follows numpy semantics, e.g., -1
    represents the last axis. `axis` is a compile-time `index` attribute.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 2], f32>
      %indices: !mo.tensor<[2], si64>
      %res = rmo.mo.gather(%input, %indices) {axis = 0 : index} : (
        !mo.tensor<[2, 2], f32>, !mo.tensor<[2], si64>
      ) -> !mo.tensor<[2, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGeluOp(max._core.Operation):
    """
    Returns the exact GELU activation `0.5 * x * (1 + erf(x / sqrt(2)))`, where
    `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGeluQuickOp(max._core.Operation):
    """
    Returns the quick approximation of GELU `x * sigmoid(1.702 * x)`, where `x`
    is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu_quick(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGeluTanhOp(max._core.Operation):
    """
    Returns the tanh approximation of GELU
    `0.5 * x * (1 + tanh(0.7978845608028654 * (x + 0.044715 * x^3)))`, where
    `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.gelu_tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGreaterEqualOp(max._core.Operation):
    """
    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                                !mo.tensor<[2, 3], f32>
                                                ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoGreaterOp(max._core.Operation):
    """
    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.greater(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                          !mo.tensor<[2, 3], f32>
                                          ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoIsInfOp(max._core.Operation):
    """
    Returns true if `x` represents a floating point Inf, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.is_inf(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoIsNanOp(max._core.Operation):
    """
    Returns true if `x` represents a floating point NaN, where `x` is input
    tensor.

    Example:

    ```mlir
      %x: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.is_nan(%x) : (!mo.tensor<[2, 3], f32>
                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoLog1pOp(max._core.Operation):
    """
    Returns `log(1 + x)`, maintaining accuracy for small `x` that could
    otherwise lead to floating-point roundings of the kind `1 + x = 1`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.log1p(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoLogOp(max._core.Operation):
    """
    Returns the natural logarithm, `log(x)`.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.log(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceLogsoftmaxOp(max._core.Operation):
    """
    Returns `log(softmax(x))`, where `x` is input tensor.

    The softmax is applied along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.logsoftmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoMaxOp(max._core.Operation):
    """Does a non-broadcasted elementwise maximum."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoMaxPoolCeilModeTrueOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    The op supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<3, 3> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.mo.max_pool_ceil_mode_true(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[1, 4, 4, 1], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[1, 2, 2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.Value[max._core.dialects.mo.TensorType],
        dilations: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def dilations(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoMaxPoolOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input and filter must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    This op currently only supports strides and dilations on the filter.

    Example:

    ```mlir
      %fs = mo.constant {
        value = #M.dense_array<2, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %st = mo.constant {
        value = #M.dense_array<1, 2> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %di = mo.constant {
        value = #M.dense_array<1, 1> : tensor<2xsi64>} : !mo.tensor<[2], si64>
      %pa = mo.constant {
        value = #M.dense_array<0, 0, 0, 0> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.mo.max_pool(%arg) [
          filter_shape = %fs, strides = %st, dilations = %di, paddings = %pa
      ] : (
        !mo.tensor<[20, 10, 10, 32], f32>, !mo.tensor<[2], si64>,
        !mo.tensor<[2], si64>, !mo.tensor<[2], si64>, !mo.tensor<[4], si64>
      ) -> !mo.tensor<[20, 9, 5, 32], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.Value[max._core.dialects.mo.TensorType],
        dilations: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def dilations(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceMeanOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their mean value, changng that
    axis's dimension to 1.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.mean(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoMinOp(max._core.Operation):
    """Does a non-broadcasted elementwise minimum."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoModOp(max._core.Operation):
    """Does a non-broadcasted elementwise modulus."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoMulOp(max._core.Operation):
    """Does a non-broadcasted elementwise multiplication."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoMutableLoadOp(max._core.Operation):
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_tensor: max._core.dialects.mo.TensorType,
        out_chain: max._core.dialects.mo.ChainType,
        in_buffer: max._core.Value[max._core.dialects.mo.BufferType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        in_chain: max._core.Value[max._core.dialects.mo.ChainType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def in_buffer(
        self,
    ) -> max._core.Value[max._core.dialects.mo.BufferType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[max._core.dialects.mo.ChainType]: ...

class MoMutableStoreOp(max._core.Operation):
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: max._core.dialects.mo.ChainType,
        in_buffer: max._core.Value[max._core.dialects.mo.BufferType],
        in_tensor: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        in_chain: max._core.Value[max._core.dialects.mo.ChainType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def in_buffer(
        self,
    ) -> max._core.Value[max._core.dialects.mo.BufferType]: ...
    @property
    def in_tensor(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[max._core.dialects.mo.ChainType]: ...

class MoMutableStoreSliceOp(max._core.Operation):
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        out_chain: max._core.dialects.mo.ChainType,
        in_buffer: max._core.Value[max._core.dialects.mo.BufferType],
        slice: max._core.Value[max._core.dialects.mo.TensorType],
        start: max._core.Value[max._core.dialects.mo.TensorType],
        stop: max._core.Value[max._core.dialects.mo.TensorType],
        step: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        in_chain: max._core.Value[max._core.dialects.mo.ChainType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def in_buffer(
        self,
    ) -> max._core.Value[max._core.dialects.mo.BufferType]: ...
    @property
    def slice(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def start(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def stop(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def step(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def in_chain(self) -> max._core.Value[max._core.dialects.mo.ChainType]: ...

class MoNegativeOp(max._core.Operation):
    """
    Returns `-x`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.negative(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoNonMaximumSuppressionOp(max._core.Operation):
    """
    Filters out boxes that have high intersection-over-union (IOU).

    `boxes` is supplied as [y1, x1, y2, x2] where (y1, x1) and (y2, x2) are the
    coordinates of any diagonal pair of box corners and the coordinates can be
    provided as normalized (i.e., lying in the interval [0, 1]) or absolute.

     Example:

     ```mlir
       %boxes : !mo.tensor<[1, 6, 4], f32>
       %scores : !mo.tensor<[1, 1, 6], f32>
       %maxOutputBoxesPerClass : !mo.tensor<[], si64>
       %iouThreshold : !mo.tensor<[], f32>
       %scoreThreshold : !mo.tensor<[], f32>
       %res = rmo.mo.non_maximum_suppression(%boxes, %scores, %maxOutputBoxesPerClass, %iouThreshold, %scoreThreshold)
         : (!mo.tensor<[1, 6, 4], f32>, !mo.tensor<[1, 1, 6], f32>, !mo.tensor<[], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>)
         -> !mo.tensor<[?, ?], si64>
     ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: max._core.dialects.mo.TensorType,
        boxes: max._core.Value[max._core.dialects.mo.TensorType],
        scores: max._core.Value[max._core.dialects.mo.TensorType],
        max_output_boxes_per_class: max._core.Value[
            max._core.dialects.mo.TensorType
        ],
        iou_threshold: max._core.Value[max._core.dialects.mo.TensorType],
        score_threshold: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def boxes(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def scores(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def max_output_boxes_per_class(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def iou_threshold(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def score_threshold(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoNotEqualOp(max._core.Operation):
    """
    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 3], f32>,
                                            !mo.tensor<[2, 3], f32>
                                            ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoNotOp(max._core.Operation):
    """
    Returns `not x` on given input, where input is a boolean tensor.

    Example:

    ```mlir
      %in: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.not(%in) : (!mo.tensor<[2, 3], bool>) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoOrOp(max._core.Operation):
    """
    Returns `x or y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.or(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                     !mo.tensor<[2, 3], bool>
                                     ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoPadConstantOp(max._core.Operation):
    """
    Pads the `input` tensor with a scalar tensor `constant` according to the
    `paddings`. Assumes input has rank `N`, the `paddings` tensor should have
    shape `(2 * N)`, where each consecutive pair of elements has the form
    `[before, after]`, indicating how many of `constant` to add before and after
    the contents of `input` in that dimension. The size of each dimension D of
    the padded output is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %constant = mo.constant {
        value = #M.dense_array<1.0> : tensor<f32>} : !mo.tensor<[], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output =   mo.pad.constant(%input, %paddings, %constant) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[3, 5], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        constant: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def constant(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoPadReflectOp(max._core.Operation):
    """
    Pads the `input` tensor by reflecting it according to the `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.reflect(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoPadRepeatOp(max._core.Operation):
    """
    Pads the `input` tensor by repeating border values according to `paddings`.
    Assumes input has rank `N`, the `paddings` tensor should have shape `(2 *
    N)`, where each consecutive pair of elements has the form `[before, after]`,
    indicating how many of `constant` to add before and after the contents of
    `input` in that dimension. The size of each dimension D of the padded output
    is: `paddings[2*D] + input.dim(D) + paddings[2*D+1]`.

    `paddings[D, 0] + input.dim(D) + paddings[D, 1]`.

    Example:

    ```mlir
      %input: !mo.tensor<[2, 3], f32>
      %paddings = mo.constant {
        value = #M.dense_array<1, 0, 1, 1> : tensor<4xsi64>
      } : !mo.tensor<[4], si64>
      %output   = mo.pad.repeat(%input, %paddings) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[4], si64>) ->
        !mo.tensor<[3, 5], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        paddings: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def paddings(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoPowOp(max._core.Operation):
    """
    Does a non-broadcasted elementwise power. Allows mixed precision operands.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoRandomNormalOp(max._core.Operation):
    """
    Returns a tensor with shape `shape` populated with random values from a
      normal distribution, with the mean of the distribution equal to `mean`
      and the standard deviation equal to `variance`.

    Example:
      ```mlir
        %size = mo.constant {
          value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
        %mean = mo.constant {
          value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
        %variance = mo.constant {
          value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
        %seed = mo.constant {
          value = #M.dense_array<1> : tensor<1xui64> } : !mo.tensor<[1], ui64>
        %res = rmo.mo.random.normal(%size, %mean, %variance, %seed) :
              (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
              !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
      ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        shape: max._core.Value[max._core.dialects.mo.TensorType],
        mean: max._core.Value[max._core.dialects.mo.TensorType],
        variance: max._core.Value[max._core.dialects.mo.TensorType],
        seed: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def shape(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def mean(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def variance(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def seed(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoRandomUniformOp(max._core.Operation):
    """
    Returns a tensor with shape `shape` populated with random values from a
    uniform distribution on the half-open interval [lowerBound, upperBound).

    Example:
    ```mlir
    %size = mo.constant {
      value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
    %lowerBound = mo.constant {
      value = #M.dense_array<2.0> : tensor<1xf32> } : !mo.tensor<[], f32>
    %upperBound = mo.constant {
      value = #M.dense_array<0.5> : tensor<1xf32> } : !mo.tensor<[], f32>
    %seed = mo.constant {
      value = #M.dense_array<1> : tensor<1xui64> } : !mo.tensor<[1], ui64>
    %res = rmo.mo.random.uniform(%size, %lowerBound, %upperBound, %seed) :
          (!mo.tensor<[4], si64>, !mo.tensor<[], f32>, !mo.tensor<[], f32>,
          !mo.tensor<[1], ui64>) -> !mo.tensor<[1, 1, 7, 8], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        shape: max._core.Value[max._core.dialects.mo.TensorType],
        lower_bound: max._core.Value[max._core.dialects.mo.TensorType],
        upper_bound: max._core.Value[max._core.dialects.mo.TensorType],
        seed: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def shape(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def lower_bound(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def upper_bound(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def seed(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoRangeOp(max._core.Operation):
    """
    Creates a sequence of numbers. The sequence goes from `start` with
    increments of size `step` up to (but not including) `limit`. All arguments
    are mandatory and must have the same element type.

    Note the following restrictions on input values:
    1. `step` must be non-zero
    2. `limit - start` must be zero or have the same sign as `step`

    Example:

    ```mlir
      %limit : !mo.tensor<[], f32>
      %start = mo.constant {
        value = #M.dense_array<0.0> : tensor<f32>} : !mo.tensor<[], f32>
      %step = mo.constant {
        value = #M.dense_array<1.5> : tensor<f32>} : !mo.tensor<[], f32>
      %res = rmo.mo.range(%start, %limit, %step) : (
        !mo.tensor<[], f32>, !mo.tensor<[], f32>, !mo.tensor<[], f32>
      ) -> !mo.tensor<[?], f32>

      %startInt = mo.constant {
        value = #M.dense_array<1> : tensor<si32>} : !mo.tensor<[], si32>
      %stepInt = mo.constant {
        value = #M.dense_array<2> : tensor<si32>} : !mo.tensor<[], si32>
      %limitInt = mo.constant {
        value = #M.dense_array<11> : tensor<si32>} : !mo.tensor<[], si32>
      %oddNumbersBelowTen = rmo.mo.range(%startInt, %limitInt, %stepInt) : (
        !mo.tensor<[], si32>, !mo.tensor<[], si32>, !mo.tensor<[], si32>
      ) -> !mo.tensor<[5], si32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        start: max._core.Value[max._core.dialects.mo.TensorType],
        limit: max._core.Value[max._core.dialects.mo.TensorType],
        step: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def start(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def limit(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def step(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceAddOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their sum, changing that axis's
    dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.add(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceMulOp(max._core.Operation):
    """
    Reduces `input` elements across `axis` to their product, changing that
    axis's dimension to 1 in the output shape.

    The value of `axis` follows numpy semantics, e.g., -1 represents the last
    axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.mul(%arg) {axis = 1 : index} : (
        !mo.tensor<[2, 3], f32>) -> !mo.tensor<[2, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReluOp(max._core.Operation):
    """
    Returns `max(0, x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.relu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReshapeOp(max._core.Operation):
    """Reshape the `input` tensor to the shape represented in `newShape`."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def new_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoResizeBicubicOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the bicubic interpolation algorithm.

    Bicubic interpolation uses a 4x4 pixel neighborhood and cubic polynomials
    to produce smoother results than linear interpolation. This implementation
    uses Keys' cubic convolution with a = -0.5.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 3, 224, 224], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 3, 448, 448> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.resize.bicubic(%input, %size) :
        (!mo.tensor<[1, 3, 224, 224], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 3, 448, 448], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        size: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def size(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoResizeLinearOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the linear algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    When set to true, the antialias attribute causes an antialiasing filter to be applied
    when downscaling.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        size: max._core.Value[max._core.dialects.mo.TensorType],
        coordinate_transform_mode: max._core.dialects.mo.CoordinateTransformModeAttr,
        antialias: max._core.dialects.builtin.BoolAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def size(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def coordinate_transform_mode(
        self,
    ) -> max._core.dialects.mo.CoordinateTransformMode: ...
    @coordinate_transform_mode.setter
    def coordinate_transform_mode(
        self, arg: max._core.dialects.mo.CoordinateTransformModeAttr, /
    ) -> None: ...
    @property
    def antialias(self) -> bool: ...
    @antialias.setter
    def antialias(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoResizeNearestOp(max._core.Operation):
    """
    Resizes a tensor to a new shape using the nearest-neighbor algorithm.

    The coordinate transform mode can be half-pixel, align-corners or asymmetric.

    The values for round mode are:
      - 0: HalfDown
      - 1: HalfUp
      - 2: Floor
      - 3: Ceil

    Round mode is HalfDown (0) by default.

    Example:
    ```mlir
      %input : !mo.tensor<[1, 1, 2, 2], f32>
      %size = mo.constant {
        value = #M.dense_array<1, 1, 7, 8> : tensor<4xsi64>} : !mo.tensor<[4], si64>
      %res = rmo.resize.nearest(%input, %size) {
        coordinate_transform_mode = 0,
        round_mode = 2}:
        (!mo.tensor<[1, 1, 2, 2], f32>, !mo.tensor<[4], si64>) ->
          !mo.tensor<[1, 1, 7, 8], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        size: max._core.Value[max._core.dialects.mo.TensorType],
        coordinate_transform_mode: max._core.dialects.mo.CoordinateTransformModeAttr,
        round_mode: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def size(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def coordinate_transform_mode(
        self,
    ) -> max._core.dialects.mo.CoordinateTransformMode: ...
    @coordinate_transform_mode.setter
    def coordinate_transform_mode(
        self, arg: max._core.dialects.mo.CoordinateTransformModeAttr, /
    ) -> None: ...
    @property
    def round_mode(self) -> int: ...
    @round_mode.setter
    def round_mode(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoRoiAlignOp(max._core.Operation):
    """
    ROI align consumes an input tensor and regions of interest in which to apply pooling.

    Example:
    ```mlir
      %inp: !mo.tensor<[1, 10, 10, 1], f32>
      %rois: !mo.tensor<[1, 5], f32>
      %output_height = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<5> : tensor<1xsi64>} : !mo.tensor<[], si64>
      %spatial_scale = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<1.0> : tensor<1xf32>} : !mo.tensor<[], f32>
      %sampling_ratio = mo.constant {device = #M.device_ref<"cpu", 0>, value = #M.dense_array<2.0> : tensor<1xf32>} : !mo.tensor<[], f32>

      %res = rmo.mo.roi_align(%inp, %rois, %output_height, %output_height, %spatial_scale, %sampling_ratio)
        {aligned = false,  mode = "AVG"}
        : (!mo.tensor<[1, 10, 10, 1], f32>,
          !mo.tensor<[1, 5], f32>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], si64>,
          !mo.tensor<[], f32>,
          !mo.tensor<[], f32>) -> !mo.tensor<[1, 5, 5, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        rois: max._core.Value[max._core.dialects.mo.TensorType],
        output_height: max._core.Value[max._core.dialects.mo.TensorType],
        output_width: max._core.Value[max._core.dialects.mo.TensorType],
        spatial_scale: max._core.Value[max._core.dialects.mo.TensorType],
        sampling_ratio: max._core.Value[max._core.dialects.mo.TensorType],
        aligned: max._core.dialects.builtin.BoolAttr,
        mode: max._core.dialects.builtin.StringAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def rois(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_height(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_width(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def spatial_scale(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def sampling_ratio(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def aligned(self) -> bool: ...
    @aligned.setter
    def aligned(self, arg: max._core.dialects.builtin.BoolAttr, /) -> None: ...
    @property
    def mode(self) -> max._core.dialects.builtin.StringAttr: ...
    @mode.setter
    def mode(self, arg: max._core.dialects.builtin.StringAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoRoundOp(max._core.Operation):
    """
    Returns the elementwise nearest integer, with ties going away from zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.round(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoRsqrtOp(max._core.Operation):
    """
    Returns `1/sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.rsqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterAddOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the sum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via addition.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] += updates[i][j] if axis = 0,
      output[i][indices[i][j]] += updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterMaxOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the maximum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via maximum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = max(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = max(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterMinOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the minimum of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via minimum.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = min(output[indices[i][j]][j], updates[i][j]) if axis = 0,
      output[i][indices[i][j]] = min(output[i][indices[i][j]], updates[i][j]) if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterMulOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices, and it stores the product of elements with duplicate
    indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar axis. The output is a copy of the input, with certain elements
    updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is combined with existing element via multiplication.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] *= updates[i][j] if axis = 0,
      output[i][indices[i][j]] *= updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterNdAddOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the sum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.add(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterNdMaxOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the maximum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.max(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterNdMinOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the minimum of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.min(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterNdMulOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices, and it stores the product of any duplicate indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd.mul(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterNdOp(max._core.Operation):
    """
    Produces an output tensor by scattering slices from updates to input
    according to indices.

    Specifically, it treats the last dimension of indices as a vector of
    integers used to index into a copy of the input, and it replaces that
    resulting slice (or scalar) with corresponding slice (or scalar) from
    the updates tensor.

    Note that the `slice` shows up in case where the index vector length is
    shorter than the rank of input tensor, i.e., the op will slice the leading
    dimensions.

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 2], f32>
      %updates: !mo.tensor<[1, 3, 2], f32>
      %indices: !mo.tensor<[1, 3, 1], si64>
      %res = rmo.mo.scatter_nd(%inputs, %updates, %indices) : (
        !mo.tensor<[4, 2], f32>, !mo.tensor<[1, 3, 2], f32>, !mo.tensor<[1, 3, 1], si64>
      ) -> !mo.tensor<[4, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoScatterOp(max._core.Operation):
    """
    Produces an output tensor by scattering elements from updates to input
    according to indices.

    It takes in `input`, `updates` and `indices` tensors of the same rank, and a
    scalar `axis` compile-time `index` attribute. The output is a copy of the
    input, with certain elements updated based on `updates` and `indices`.

    For each entry in `indices`, the target index for `input` is obtained by
    making a copy of the entry's own index, and then updating the `axis`
    dimension with the value of the `indices` entry. Then the element at this
    target index is updated to the corresponding entry in `updates`.

    For instance, in a 2D tensor case, the update corresponding to the [i][j] entry
    is performed as below:
    ```
      output[indices[i][j]][j] = updates[i][j] if axis = 0,
      output[i][indices[i][j]] = updates[i][j] if axis = 1,
    ```

    Example:

    ```mlir
      %input:   !mo.tensor<[4, 4], f32>
      %updates: !mo.tensor<[2, 3], f32>
      %indices: !mo.tensor<[2, 3], si64>
      %res = rmo.mo.scatter(%inputs, %updates, %indices) {axis = 0 : index} : (
        !mo.tensor<[4, 4], f32>, !mo.tensor<[2, 3], f32>, !mo.tensor<[2, 3], si64>
      ) -> !mo.tensor<[4, 4], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        updates: max._core.Value[max._core.dialects.mo.TensorType],
        indices: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def updates(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def indices(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSelectOp(max._core.Operation):
    """
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[2, 3], f32>
      %y: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        cond: max._core.Value[max._core.dialects.mo.TensorType],
        x: max._core.Value[max._core.dialects.mo.TensorType],
        y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def cond(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoShapeOfOp(max._core.Operation):
    """Calculates the runtime shape of the given tensor as a tensor."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        shape: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSigmoidOp(max._core.Operation):
    """
    Returns the sigmoid activation `1 / (1 + exp(-x))`, where `x` is the input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sigmoid(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSiluOp(max._core.Operation):
    """
    Returns the SiLU (Swish) activation `x * sigmoid(x)`, where `x` is the input
    tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.silu(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSinOp(max._core.Operation):
    """
    Returns `sin(x)`, where `x` is input tensor.

    Example:
    ```mlir
      %arg : !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sin(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSliceOp(max._core.Operation):
    """
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `start`, `stop`, and `step`
    1D index tensors; each index tensor has N elements, one for each dimension
    of the `input` tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `start[i]:stop[i]:step[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `start` and `stop`, e.g., -1
       represents the largest axis.
    3. Out of bound indices in `start` and `stop` will be clamped to
       [-dim, dim], where `dim` is the dimension in the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[?, ?], f32>
      %start: !mo.tensor<[2], si64> // [1, -6]
      %stop: !mo.tensor<[2], si32>  // [-3, 6]
      %step: !mo.tensor<[2], si64>  // [5, 1]
      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = rmo.mo.slice(%input, %start, %stop, %step) : (
        !mo.tensor<[10, 10], f32>,
        !mo.tensor<[2], si64>,
        !mo.tensor<[2], si32>,
        !mo.tensor<[2], si64>
      ) -> !mo.tensor<[?, ?], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        start: max._core.Value[max._core.dialects.mo.TensorType],
        stop: max._core.Value[max._core.dialects.mo.TensorType],
        step: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def start(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def stop(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def step(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoReduceSoftmaxOp(max._core.Operation):
    """
    Returns `exp(input) / sum(exp(input))`, where `x` is input tensor.

    The `sum` reduction is applieed along the last axis.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.reduce.softmax(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSqrtOp(max._core.Operation):
    """
    Returns `sqrt(x)`, where `x` is the input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.sqrt(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSqueezeShapeOp(max._core.Operation):
    """
    Calculates the shape from squeeze like operators. Given an input shape
    vector representing a tensor of rank `N`, and a list of indices of length
    `M`, returns a new shape vector representing a tensor of rank `N - M`. The
    indices represent the 0-based index of dimensions in the original rank `N`
    tensor.

    The indicated indices must represent dimensions of size 1. If
    an index does not point to a dimension to size 1, an error is thrown
    instead.

    This operator supports negative indexing with python-like semantics.
    That is all indices must be in [-N, N), if an index is < 0, it is as if
    we added `N` to it.

    Example:

    ```mlir
      %input_shape : !mo.tensor<[8], si32>
      %indices : !mo.tensor<[4], si32>
      %res = rmo.mo.squeeze_shape(%input_shape, %indices) : (!mo.tensor<[8], si32>, !mo.tensor<[4], si32>) -> !mo.tensor<[4], si32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_shape: max._core.Value[max._core.dialects.mo.TensorType],
        remove_indices: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_shape(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def remove_indices(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoSubOp(max._core.Operation):
    """Does a non-broadcasted elementwise subtraction."""

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoTanhOp(max._core.Operation):
    """
    Computes `tanh(x)`, where `x` is input tensor.

    Example:

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.tanh(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoTileOp(max._core.Operation):
    """
    Returns a new Tensor as the result of copying the input tensor N_i times
    on each dimension, where N_i = tiles[i].

    The i-th dimension of output shape will be the ith dimension of input shape
    multiplied by N_i.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %repeats : !mo.tensor<[2], si64>
      %res = rmo.mo.tile(%input, %repeats) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[?, ?], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        repeats: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def repeats(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoTransposeOp(max._core.Operation):
    """
    Returns a new Tensor as the result of permuting the dimensions of the input
    tensor according to the value of perm.

    Note that `perm` must contain unique values from `[0, input_rank)`.

    Example:

    ```mlir
      %input : !mo.tensor<[2, 3], f32>
      %perm : !mo.tensor<[2], si64>
      %res = rmo.mo.transpose(%input, %perm) : (
        !mo.tensor<[2, 3], f32>, !mo.tensor<[2], si64>) -> !mo.tensor<[3, 2], f32>

      %input : !mo.tensor<[?, 5, ?], f32>
      %perm : !mo.tensor<[3], si32>
      %res = rmo.mo.transpose(%input, %perm) : (
        !mo.tensor<[?, 5, ?], f32>, !mo.tensor<[3], si32>
      ) -> !mo.tensor<[?, ?, 5], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        perm: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def perm(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoTruncOp(max._core.Operation):
    """
    Returns the elementwise integer from truncating the decimal. Also known
    as round-toward-zero.

    ```mlir
      %arg: !mo.tensor<[2, 3], f32>
      %res = rmo.mo.trunc(%arg) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MoXorOp(max._core.Operation):
    """
    Returns `x xor y`, where `x` and `y` are input boolean tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.mo.xor(%lhs, %rhs) : (!mo.tensor<[2, 3], bool>,
                                      !mo.tensor<[2, 3], bool>
                                      ) -> !mo.tensor<[2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_values: Sequence[max._core.Value[max._core.Type]],
        graph_op: max._core.dialects.mo.GraphOp,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AddOp(max._core.Operation):
    """
    A flexible binary elementwise add operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AndOp(max._core.Operation):
    """
    A flexible binary elementwise and comparison operation with implicit broadcasting.

    Returns elementwise `x & y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.and(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class AvgPoolOp(max._core.Operation):
    """
    Computes average pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d average pooling (so input must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    Example:

    ```mlir
    %res = rmo.avg_pool(%input) {
      filter_shape = #mosh<ape<2, 3>> : !mosh.ape
      strides = #mosh<ape<2, 3>> : !mosh.ape
      dilations = #mosh<ape<1, 1>> : !mosh.ape
      paddings = #mosh<ape<0, 0, 0, 0>> : !mosh.ape
      ceil_mode = false
      count_boundary = true
    } : (
      !mo.tensor<[1, 6, 15, 1], f32>,
    ) -> !mo.tensor<[1, 3, 5, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.dialects.mosh.ShapeAttr,
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        ceil_mode: max._core.dialects.builtin.BoolAttr,
        count_boundary: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.dialects.mosh.ShapeAttr,
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        ceil_mode: max._core.dialects.builtin.BoolAttr,
        count_boundary: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.dialects.mosh.ShapeAttr,
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        ceil_mode: bool,
        count_boundary: bool,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter_shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @filter_shape.setter
    def filter_shape(
        self, arg: max._core.dialects.mosh.ShapeAttr, /
    ) -> None: ...
    @property
    def strides(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @strides.setter
    def strides(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def dilations(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @dilations.setter
    def dilations(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def paddings(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @paddings.setter
    def paddings(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def ceil_mode(self) -> bool: ...
    @ceil_mode.setter
    def ceil_mode(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...
    @property
    def count_boundary(self) -> bool: ...
    @count_boundary.setter
    def count_boundary(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class BroadcastLikeOp(max._core.Operation):
    """
    Equivalent to the following:

      ```
      %shape_of = rmo.shape_of(%like)
      %result = rmo.broadcast_to(%broadcast, %shape_of)
      ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        shape_like: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        shape_like: max._core.Value[max._core.dialects.mo.TensorType],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def shape_like(
        self,
    ) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class BroadcastToOp(max._core.Operation):
    """
    Broadcasts the input tensor to the specified shape.

    Shape restrictions:
    1. `newShape` must have known rank.
    2. `newShape` may not contain any dynamic dimensions.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.dialects.mosh.ShapeAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.dialects.mosh.ShapeAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def new_shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @new_shape.setter
    def new_shape(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...

class ConvOp(max._core.Operation):
    """
    Computes the convolution product of the input with the given filter,
    strides, dilations, paddings, and groups.

    The op supports 1D-3D convolution, with the following layout assumptions:
    - input has channel last layout. For 2D, that's NHWC, i.e.,
      (batch_size, height, width, in_channels)
    - filter has layout FCRS, i.e.,
      (out_channels, in_channels / num_groups, height, width)

    The filter layout is determined by the layout attribute on the filter
    tensor type. Supported layouts include FCRS (default), RSCF (legacy),
    and packed variants like FRSCf.

    `strides`, `dilations`, and `padding` are all shape attributes.
    If the input has static rank, and must have have sizes of `input_rank - 2`,
    except padding, which must have size `2 * (input_rank - 2)`. Individual
    elements in the hyperparameters apply to corresponding dimensions of the
    input (after ignoring the batch and channel dimensions), with padding
    representing a before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    Convolution, dim1 here represents H and dim2 represents W. In python like
    syntax, padding a 2x3 spatial `input` with [0, 1, 2, 1] would yield:

    ```python
    input = [
      [1, 2, 3],
      [4, 5, 6]
    ]
    # Shape is 2x3

    padded_input = [
      [0, 0, 1, 2, 3, 0],
      [0, 0, 4, 5, 6, 0]
      [0, 0, 0, 0, 0, 0]
    ]
    # Shape is 3x6
    ```

    The input, output and filter tensors' ranks must match.

    The number of input and output channels must both be divisible by the number
    of groups `num_groups`.

    Example:

    ```mlir
    %input: !mo.tensor<[10, 5, 5, 32], f32>,
    %filter: !mo.tensor<[64, 32, 2, 2], f32>

    %result = rmo.conv(%input, %filter) {
      strides = #mosh<ape[2, 2]> : !mosh.ape,
      paddings = #mosh<ape[0, 0, 0, 0]> : !mosh.ape,
      dilations = #mosh<ape[1, 1]> : !mosh.ape,
      num_groups = 1 : si64
    } : (
      !mo.tensor<[10, 5, 5, 32], f32>, !mo.tensor<[64, 32, 2, 2], f32>
    ) -> !mo.tensor<[10, 4, 4, 64], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        num_groups: max._core.dialects.builtin.IntegerAttr,
        input_layout: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        num_groups: max._core.dialects.builtin.IntegerAttr,
        input_layout: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        num_groups: int,
        input_layout: str = "NHWC",
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @strides.setter
    def strides(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def dilations(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @dilations.setter
    def dilations(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def paddings(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @paddings.setter
    def paddings(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def num_groups(self) -> int: ...
    @num_groups.setter
    def num_groups(
        self, arg: max._core.dialects.builtin.IntegerAttr, /
    ) -> None: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...

class ConvTransposeOp(max._core.Operation):
    """
    This op effectively computes the gradient of a convolution with
    respect to its input (as if the original convolution operation had the same
    filter and hyperparameters as this op). A visualization of the computation
    can be found in https://d2l.ai/chapter_computer-vision/transposed-conv.html.

    The op supports 1D-3D spatial dimensions, with the following layout
    assumptions (note the `out_channel` is w.r.t. the original convolution):
    - input has channel last layout.For 2D, that's NHWC, i.e.,
      (batch_size, height, width, out_channels)
    - filter has layout RSFC, i.e., (height, width, out_channels, in_channels)

    All hyperparameters (i.e. strides, dilations, padding, output_paddings) must
    be of rank 1, or unranked. If the input has static rank, all hyperparameters
    with static shape must have sizes of `input_rank - 2`, except padding, which
    must have size `2 * (input_rank - 2)`. Individual elements in the
    hyperparameters applies to corresponding dimensions of the input (after
    ignoring the batch and channel dimensions), with padding representing a
    before/after pair for each axis.

    The padding values are expected to take the form (pad_dim1_before,
    pad_dim1_after, pad_dim2_before, pad_dim2_after...) and represent padding
    0's before and after the indicated *spatial* dimensions in `input`. In 2D
    ConvTranspose, dim1 here represents H_out and dim2 represents W_out. In
    python like syntax, padding a 2x4 spatial `output` with [0, 1, 2, 1] would
    yield:

    ```python
    output = [
      [1, 2, 3, 4],
      [5, 6, 7, 8]
    ]
    # Shape is 2x4

    padded_input = [
      [3],
    ]
    # Shape is 1x1
    ```

    The `output_paddings` argument is meant to resolve the ambiguity of multiple
    potential output shapes when any stride is greater than 1. Basically,
    we'll add `output_paddings[i]` number of zeros at the end of output's ith
    axis. We only support output_paddings = 0.

    The input, output and filter tensors' ranks must match if statically known.

    Example:

    ```mlir
    %res = rmo.conv_transpose(%input, %filter) {
      strides = #mosh<ape<1, 1>> : !mosh.ape
      dilations = #mosh<ape<1, 1>> : !mosh.ape
      paddings = #mosh<ape<0, 0, 0, 0>> : !mosh.ape
      output_paddings = #mosh<ape<0, 0>> : !mosh.ape
    } : (
      !mo.tensor<[10, 4, 4, 64], f32>, !mo.tensor<[2, 2, 32, 64], f32>,
    ) -> !mo.tensor<[10, 5, 5, 32], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        output_paddings: max._core.dialects.mosh.ShapeAttr,
        input_layout: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        output_paddings: max._core.dialects.mosh.ShapeAttr,
        input_layout: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter: max._core.Value[max._core.dialects.mo.TensorType],
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        output_paddings: max._core.dialects.mosh.ShapeAttr,
        input_layout: str = "NHWC",
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def strides(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @strides.setter
    def strides(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def dilations(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @dilations.setter
    def dilations(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def paddings(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @paddings.setter
    def paddings(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def output_paddings(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @output_paddings.setter
    def output_paddings(
        self, arg: max._core.dialects.mosh.ShapeAttr, /
    ) -> None: ...
    @property
    def input_layout(self) -> str: ...
    @input_layout.setter
    def input_layout(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...

class DivOp(max._core.Operation):
    """
    A flexible binary elementwise div operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class EqualOp(max._core.Operation):
    """
    A flexible binary elementwise equality comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns `x == y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class GreaterEqualOp(max._core.Operation):
    """
    A flexible binary elementwise greater than or equal to comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns `x >= y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.greater_equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class GreaterOp(max._core.Operation):
    """
    A flexible binary elementwise greater than comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns `x > y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.greater(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MatmulOp(max._core.Operation):
    """
    Performs general matrix multiplication with broadcasting.

    If the lhs is 1d, it will be reshaped to `1xD`.
    If the rhs is 1d, it will be reshaped to `Dx1`.
    In both cases, the addition `1` dimensions will be removed from the output shape.

    For the multiplication, the innermost (rightmost) 2 dimensions are treated as a maxtrix.
    The lhs matrix will have the shape `MxK`.
    The rhs matrix will have the shape `KxN`.
    The output will have the shape `MxN`
    The `K` dimensions must be equivalent in both matrices.

    The remaining outer dimensions will be broadcasted.

    Example shapes with outputs:
    [K] @ [K] -> []
    [5, K] @ [K] -> [5]
    [K] @ [K, 6] -> [6]
    [4, K] @ [K, 6] -> [4, 6]
    [8, 10, 4, K] @ [8, 1, K, 6] -> [8, 10, 4, 6]
    [10, 4, K] @ [K, 6] -> [10, 4, 6]
    [10, 4, K] @ [K] -> [10, 4]
    [K] @ [10, 4, K, 7] -> [10, 4, 7]
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...

class MaxOp(max._core.Operation):
    """
    A flexible binary elementwise max operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MaxPoolOp(max._core.Operation):
    """
    Computes max pooling with the given filter shape, strides, and dilations.

    For now the op only supports 2d max pooling (so input must be
    4D), with the following layout assumption:
    - input has layout NHWC, i.e., (batch_size, height, width, in_channels)

    All hyperparameters (i.e. strides, dilations, padding) must be of rank 1, or
    unranked. If the input has static rank, all hyperparameters with static
    shape must have sizes of `input_rank - 2`, except padding, which must have size
    `2 * (input_rank - 2)`. Individual elements in the hyperparameters applies to
    corresponding dimensions of the input (after ignoring the batch and channel dimensions),
    with padding representing a before/after pair for each axis. The padding values
    are expected to take the form (pad_dim1_before, pad_dim1_after, pad_dim2_before,
    pad_dim2_after...). In 2D Convolution, dim1 here represents H and dim2 represents W.

    Example:

    ```mlir
    %res = rmo.max_pool(%input) {
      filter_shape = #mosh<ape<2, 3>> : !mosh.ape
      strides = #mosh<ape<2, 3>> : !mosh.ape
      dilations = #mosh<ape<1, 1>> : !mosh.ape
      paddings = #mosh<ape<0, 0, 0, 0>> : !mosh.ape
      ceil_mode = false
    } : (
      !mo.tensor<[1, 6, 15, 1], f32>,
    ) -> !mo.tensor<[1, 3, 5, 1], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.dialects.mosh.ShapeAttr,
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        ceil_mode: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.dialects.mosh.ShapeAttr,
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        ceil_mode: max._core.dialects.builtin.BoolAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        filter_shape: max._core.dialects.mosh.ShapeAttr,
        strides: max._core.dialects.mosh.ShapeAttr,
        dilations: max._core.dialects.mosh.ShapeAttr,
        paddings: max._core.dialects.mosh.ShapeAttr,
        ceil_mode: bool,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def filter_shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @filter_shape.setter
    def filter_shape(
        self, arg: max._core.dialects.mosh.ShapeAttr, /
    ) -> None: ...
    @property
    def strides(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @strides.setter
    def strides(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def dilations(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @dilations.setter
    def dilations(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def paddings(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @paddings.setter
    def paddings(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def ceil_mode(self) -> bool: ...
    @ceil_mode.setter
    def ceil_mode(
        self, arg: max._core.dialects.builtin.BoolAttr, /
    ) -> None: ...

class MinOp(max._core.Operation):
    """
    A flexible binary elementwise min operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class ModOp(max._core.Operation):
    """
    A flexible binary elementwise mod operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class MulOp(max._core.Operation):
    """
    A flexible binary elementwise mul operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class NotEqualOp(max._core.Operation):
    """
    A flexible binary elementwise not equal comparison operation with implicit broadcasting and implicit dtype promotion.

    Returns elementwise `x != y`, where `x` and `y` are input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], f32>
      %rhs: !mo.tensor<[2, 3], f64>
      %res = rmo.not_equal(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], f32>,
                                        !mo.tensor<[2, 3], f64>
                                        ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class OrOp(max._core.Operation):
    """
    A flexible binary elementwise or comparison operation with implicit broadcasting.

    Returns elementwise `x | y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.or(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class PowOp(max._core.Operation):
    """
    A flexible binary elementwise pow operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class RebindTensorShapeOp(max._core.Operation):
    """
    Unlike `mo.rebind` this also has the semantics of doing a runtime check
    that the given shapes are compatible.

    Right now this is modeled as a side-effect of the operation due to
    limitations of modeling assertions in our stack. In the future this will
    change and this operation may return either an error or a tensor.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        output: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
        message: max._core.dialects.builtin.StringAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.dialects.mosh.ShapeAttr,
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.dialects.mosh.ShapeAttr,
        message: str,
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
    @property
    def message(self) -> str: ...
    @message.setter
    def message(
        self, arg: max._core.dialects.builtin.StringAttr, /
    ) -> None: ...

class ReshapeOp(max._core.Operation):
    """
    Returns a tensor with the same underlying data as `input`, but the shape of `newShape`.

    A static dimension set to `-1` will be automatically computed.

    Shape restrictions:
    1. `newShape` must have known rank.
    2. `newShape` may not contain any dynamic dimensions.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.dialects.mosh.ShapeAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        new_shape: max._core.dialects.mosh.ShapeAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def new_shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @new_shape.setter
    def new_shape(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...

class SelectOp(max._core.Operation):
    """
    Returns `cond ? x : y` (element-wise), where `cond`, `x` and `y` are input
    tensors.

    Broadcasting is handled by first calculating the common shape between `x` and `y`.
    Then, the `cond` shape and common shape are merged to figure out the final shape.
    All inputs are broadcast to the same final shape.

    Dtype casting only applies to the `x` and `y` inputs. `cond` must be a bool.

    Example:

    ```mlir
      %cond: !mo.tensor<[2, 3], bool>
      %x: !mo.tensor<[1, 3], f32>
      %y: !mo.tensor<[2, 3], i8>
      %res = rmo.select(%cond, %x, %y) : !mo.tensor<[2, 3], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        cond: max._core.Value[max._core.dialects.mo.TensorType],
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        cond: max._core.Value[max._core.dialects.mo.TensorType],
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def cond(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...

class ShapeToTensorOp(max._core.Operation):
    """Converts a statically known shape attr into a 1d tensor of the shape."""

    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        shape: max._core.dialects.mosh.ShapeAttr,
    ) -> None: ...
    @property
    def shape(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @shape.setter
    def shape(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...

class SliceOp(max._core.Operation):
    """
    Returns a new tensor with a subset of the elements from an N-dimensional
    `input` tensor. The subset is chosen using the `starts`, `stops`, and
    `steps` shape attributes.
    Each shape attributes has N elements, one for each dimension of the `input`
    tensor.

    The semantics follows the numpy index semantics, such that
    1. For each dimension `i`, `starts[i]:stops[i]:steps[i]` represents the
       "indexing" along that dimension.
    2. Negative indices are supported for `starts` and `stops`, e.g., -1
       represents the largest axis.
    3. Unlike `mo.slice`, Out of bound indices in `starts` and `stops` are not
       allowed and each must be in [-dim, dim], where `dim` is the dimension in
       the corresponding axis.
    4. `step` must contain nonzero elements. Negative steps are supported.

    Note: the order in which negative indices are resolved matches that of
    python for `start:

    1. Normalize negative indices by adding the dimension size.
    2. Apply clipping logic.

    This means the equivalent mo.slice for l[:-1:-1] returns an empty result.
    If we want to reverse the values in `l` we should do l[:-N-1:-1] where
    N is the dimension size. Numbers smaller than -N-1 should also work.

    Example:
    ```mlir
      %input: !mo.tensor<[10, 10], f32>

      // equivalent to this in numpy: `input[1:-3:5, -6:6:1]`
      %res = rmo.slice(%input) {starts = #mosh<ape[1, -6]> : !mosh.ape, steps = #mosh<ape[5, 1]> : !mosh.ape, stops = #mosh<ape[-3, 6]> : !mosh.ape} : (
        !mo.tensor<[10, 10], f32>,
      ) -> !mo.tensor<[2, 2], f32>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        starts: max._core.dialects.mosh.ShapeAttr,
        stops: max._core.dialects.mosh.ShapeAttr,
        steps: max._core.dialects.mosh.ShapeAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        starts: max._core.dialects.mosh.ShapeAttr,
        stops: max._core.dialects.mosh.ShapeAttr,
        steps: max._core.dialects.mosh.ShapeAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        starts: max._core.dialects.mosh.ShapeAttr,
        stops: max._core.dialects.mosh.ShapeAttr,
        steps: max._core.dialects.mosh.ShapeAttr,
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def starts(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @starts.setter
    def starts(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def stops(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @stops.setter
    def stops(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def steps(self) -> max._core.dialects.mosh.ShapeAttr: ...
    @steps.setter
    def steps(self, arg: max._core.dialects.mosh.ShapeAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class SubOp(max._core.Operation):
    """
    A flexible binary elementwise sub operation with implicit broadcasting and implicit dtype promotion.
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class TopKOp(max._core.Operation):
    """
    Computes the largest values and their corresponding indices in a tensor
    along a specified axis.
    Returns values sorted along the axis.

    axis: The axis to compute the largest values over.
      The axis must be in [-rank, rank).
    k: The number of values to compute.

    Example:
    ```mlir
      %input: !mo.tensor<[2, 6], si64>
      %values, %indices = rmo.top_k(%input) {k = 3, axis = 1} :
        (!mo.tensor<[2, 6], si64>) ->
        (!mo.tensor<[2, 3], si64>, !mo.tensor<[2, 3], si64>)
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        values: max._core.dialects.mo.TensorType,
        indices: max._core.dialects.mo.TensorType,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        k: max._core.dialects.builtin.IntegerAttr,
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        k: max._core.dialects.builtin.IntegerAttr,
        axis: max._core.dialects.builtin.IntegerAttr,
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input: max._core.Value[max._core.dialects.mo.TensorType],
        k: int,
        axis: int,
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def k(self) -> int: ...
    @k.setter
    def k(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def axis(self) -> int: ...
    @axis.setter
    def axis(self, arg: max._core.dialects.builtin.IntegerAttr, /) -> None: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...

class XorOp(max._core.Operation):
    """
    A flexible binary elementwise xor comparison operation with implicit broadcasting.

    Returns elementwise `x ^ y`, where `x` and `y` are boolean input tensors.

    Example:

    ```mlir
      %lhs: !mo.tensor<[2, 2, 3], bool>
      %rhs: !mo.tensor<[2, 3], bool>
      %res = rmo.xor(%lhs, %rhs) : (!mo.tensor<[2, 2, 3], bool>,
                                    !mo.tensor<[2, 3], bool>
                                    ) -> !mo.tensor<[2, 2, 3], bool>
    ```
    """

    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        result: max._core.dialects.mo.TensorType,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: max._core.dialects.kgen.ParamDeclArrayAttr,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        input_x: max._core.Value[max._core.dialects.mo.TensorType],
        input_y: max._core.Value[max._core.dialects.mo.TensorType],
        output_param_decls: Sequence[
            max._core.dialects.kgen.ParamDeclAttr
        ] = [],
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @overload
    def __init__(
        self,
        builder: max._core.OpBuilder,
        location: Location,
        operands: Sequence[max._core.Value[max._core.Type]],
        properties: max._core.dialects.builtin.DictionaryAttr = ...,
        discardable_attributes: max._core.dialects.builtin.DictionaryAttr = ...,
    ) -> None: ...
    @property
    def input_x(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def input_y(self) -> max._core.Value[max._core.dialects.mo.TensorType]: ...
    @property
    def output_param_decls(
        self,
    ) -> Sequence[max._core.dialects.kgen.ParamDeclAttr]: ...
    @output_param_decls.setter
    def output_param_decls(
        self, arg: max._core.dialects.kgen.ParamDeclArrayAttr, /
    ) -> None: ...
