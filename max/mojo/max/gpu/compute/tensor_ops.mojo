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
"""This module provides tensor core operations and utilities for GPU computation.

The module includes functions for:
- Tensor core based reductions (tc_reduce) supporting various data types and SIMD widths
- GEVM (General Matrix-Vector Multiplication) reductions using tensor cores
- Efficient warp-level reductions leveraging tensor core operations

The tensor core operations are optimized for NVIDIA GPUs and support different data types
including float32, float16, and bfloat16. The module provides both scalar and vector
variants of reduction operations with different SIMD widths for maximum performance.

Key functions:
- tc_reduce: Main tensor core reduction function supporting various types and widths
- tc_reduce_gevm_8x: 8x GEVM reduction using tensor cores
- tc_reduce_gevm_4x: 4x GEVM reduction using tensor cores

Note:
    Most operations require NVIDIA GPUs with tensor core support.
    Operations are optimized for warp-level execution.
"""

from max.gpu.compute.mma import mma
from std._gpu.primitives.warp import shuffle_down


@always_inline
def tc_reduce_gevm_8x[
    out_type: DType, in_type: DType, simd_width: Int
](val1: SIMD[in_type, simd_width], val2: SIMD[in_type, simd_width]) -> SIMD[
    out_type, simd_width
]:
    """Performs an 8x GEVM reduction using tensor cores.

    Parameters:
        out_type: The output data type for the reduction result (must be float32).
        in_type: The input data type of the vectors to reduce (must be bfloat16).
        simd_width: The width of the SIMD vectors.

    Args:
        val1: First input SIMD vector to reduce.
        val2: Second input SIMD vector to reduce.

    Returns:
        SIMD vector containing the reduced result.

    Note:
        Currently only supports bfloat16 input to float32 output conversion.
        Uses tensor core matrix multiply-accumulate (MMA) operations for reduction.
    """

    comptime assert (
        out_type == .float32 and in_type == .bfloat16
    ), "unsupported input/output type"

    var d_reg = SIMD[out_type, simd_width]()
    var a_reg = SIMD[in_type, simd_width * 2](1)
    mma(d_reg, a_reg, val1, d_reg)

    var c_reg = SIMD[out_type, simd_width]()
    mma(c_reg, a_reg, val2, d_reg)
    return c_reg


@always_inline
def tc_reduce_gevm_4x[
    out_type: DType, in_type: DType, simd_width: Int
](val1: SIMD[in_type, simd_width]) -> SIMD[out_type, simd_width]:
    """Performs a 4x GEVM reduction using tensor cores.

    Parameters:
        out_type: The output data type for the reduction result (must be float32).
        in_type: The input data type of the vector to reduce (must be bfloat16).
        simd_width: The width of the SIMD vector.

    Args:
        val1: Input SIMD vector to reduce.

    Returns:
        SIMD vector containing the reduced result.

    Note:
        Currently only supports bfloat16 input to float32 output conversion.
        Uses tensor core matrix multiply-accumulate (MMA) operations for reduction.
    """

    comptime assert (
        out_type == .float32 and in_type == .bfloat16
    ), "unsupported input/output type"

    var d_reg = SIMD[out_type, simd_width]()
    var a_reg = SIMD[in_type, simd_width * 2](1)
    mma(d_reg, a_reg, val1, d_reg)
    return d_reg


@always_inline
def tc_reduce[
    in_type: DType, simd_width: Int, //, out_type: DType
](val: SIMD[in_type, simd_width]) -> Scalar[out_type]:
    """Performs tensor core based reduction on a SIMD vector.

    Parameters:
        in_type: The input data type of the SIMD vector elements.
        simd_width: The width of the SIMD vector.
        out_type: The output data type for the reduced result.

    Args:
        val: Input SIMD vector to reduce.

    Returns:
        Scalar containing the reduced result.

    Note:
        Dispatches to either scalar or vector reduction implementation based on SIMD width.
        Supports various input/output type combinations using tensor core operations.
    """

    comptime if simd_width == 1:
        return _tc_reduce_scalar[out_type](val._refine[new_size=1]())
    else:
        return _tc_reduce_vector[out_type](val)


@always_inline
def _tc_reduce_vector[
    in_type: DType, simd_width: SIMDLength, //, out_type: DType
](val: SIMD[in_type, simd_width]) -> Scalar[out_type]:
    """Internal vector reduction implementation using tensor cores.

    Parameters:
        in_type: The input data type of the SIMD vector elements.
        simd_width: The width of the SIMD vector.
        out_type: The output data type for the reduced result.

    Args:
        val: Input SIMD vector to reduce.

    Returns:
        Scalar containing the reduced result.

    Note:
        Optimized for different SIMD widths (1,2,4,8,>8) using tensor core operations.
        Currently focused on bfloat16 to float32 conversion.
        Uses matrix multiply-accumulate (MMA) and shuffle operations for efficient reduction.
    """

    comptime assert (
        out_type == .float32 and in_type == .bfloat16
    ), "unsupported input/output type"

    comptime if simd_width == 1:
        return val[0].cast[out_type]()

    elif simd_width == 2:
        var tmp = val.cast[out_type]()
        return tmp[0] + tmp[1]

    elif simd_width == 4:
        # we do m16n8k8 tensor core matmul to get partial results in first row
        var d_reg = SIMD[out_type, 4]()
        var a_reg = SIMD[in_type, 4](1)
        var b_reg = SIMD[in_type, 2]()
        b_reg[0] = val[0]
        b_reg[1] = val[1]
        var c_reg = SIMD[out_type, 4]()

        # do another iteration for the next set of values where
        # result from previous used as C matrix for element wise
        # add by tensor cores
        mma(d_reg, a_reg, b_reg, c_reg)
        b_reg[0] = val[2]
        b_reg[1] = val[3]
        mma(c_reg, a_reg, b_reg, d_reg)

        # a third mma operation needed to sum the 8 elements
        b_reg = SIMD[in_type, 2](1)
        var x_reg = c_reg.cast[in_type]()
        d_reg = SIMD[out_type, 4]()
        mma(d_reg, x_reg, b_reg, d_reg)

        return d_reg[0]

    elif simd_width == 8:
        # matrix A is of tile shape 16x16 and can process partial
        # sums of 256 elements matrix B is all 1s so it can enable the summation
        var d_reg = SIMD[out_type, 4]()
        var b_reg = SIMD[in_type, 4](1)

        # perform a m16n8k16 tensor core matmul to get the first col
        # as resultant partial sums
        mma(d_reg, val, b_reg, d_reg)
        var res = d_reg[0] + d_reg[2]

        # the resultant matrix d from previous operation has its first column
        # containing the values and only threads T0, T4, T8, T16 ... T28
        # contain the results. We need to sum them to get final result with
        # 0x11111111 mask we can restrict the active shuffle threads to every 1 in 4

        res += shuffle_down(0x11111111, res, 16)
        res += shuffle_down(0x11111111, res, 8)
        res += shuffle_down(0x11111111, res, 4)

        return res
    elif simd_width > 8:
        var tmp = SIMD[out_type, simd_width // 8]()

        comptime for i in range(0, simd_width, 8):
            tmp[i // 8] = _tc_reduce_vector[out_type](val.slice[8, offset=i]())

        return tmp.reduce_add()

    else:
        comptime assert False, "unsupported simd_width for BF16"


@always_inline
def _tc_reduce_scalar[
    in_type: DType, //, out_type: DType
](val: Scalar[in_type]) -> Scalar[out_type]:
    """Internal scalar reduction implementation using tensor cores.

    Parameters:
        in_type: Input data type for the reduction. Supported types:
            - DType.float16
            - DType.bfloat16
            - DType.float32.
        out_type: Output data type for the reduction result. Supported types:
            - DType.float32
            - DType.float16 (only when in_type is float16).

    Args:
        val: Input scalar value to reduce.

    Returns:
        Scalar containing the reduced result.

    Note:
        Supports various input/output type combinations:
        - float16 to float32
        - bfloat16 to float32
        - float32 to float32
        - float16 to float16
        Uses matrix multiply-accumulate (MMA) operations for reduction.
    """

    comptime assert out_type == .float32

    comptime if out_type == .float32 and in_type == .float16:
        var d_reg = SIMD[out_type, 2]()
        var a_reg = Scalar[in_type](1)
        var b_reg: Scalar[in_type] = val
        var c_reg = SIMD[out_type, 2]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[in_type]()
        d_reg[0] += d_reg[1]
        x_reg[0] = d_reg[0].cast[in_type]()
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]

    elif out_type == .float32 and in_type == .bfloat16:
        var d_reg = SIMD[out_type, 4]()
        var a_reg = SIMD[in_type, 4](1)
        var b_reg = SIMD[in_type, 2]()
        b_reg[0] = val
        var c_reg = SIMD[out_type, 4]()

        mma(d_reg, a_reg, b_reg, c_reg)
        b_reg = SIMD[in_type, 2](1)
        var x_reg = d_reg.cast[in_type]()
        mma(d_reg, x_reg, b_reg, c_reg)

        return d_reg[0]

    elif out_type == .float32 and in_type == .float32:
        var d_reg = SIMD[out_type, 4]()
        var a_reg = SIMD[in_type, 2](1)
        var b_reg: Scalar[in_type] = val
        var c_reg = SIMD[out_type, 4]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[out_type]()
        x_reg[0] = d_reg[0] + d_reg[1]
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]

    else:
        comptime assert (
            in_type == .float16 and out_type == .float16
        ), "unsupported dtype"
        var d_reg = SIMD[out_type, 2]()
        var a_reg = Scalar[in_type](1)
        var b_reg: Scalar[in_type] = val
        var c_reg = SIMD[out_type, 2]()

        mma(d_reg, a_reg, b_reg, c_reg)
        var x_reg = Scalar[out_type]()
        x_reg[0] = d_reg[0] + d_reg[1]
        mma(d_reg, a_reg, x_reg, c_reg)

        return d_reg[0]
