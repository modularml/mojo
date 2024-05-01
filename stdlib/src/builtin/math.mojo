# ===----------------------------------------------------------------------=== #
# Copyright (c) 2024, Modular Inc. All rights reserved.
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
"""Defines basic math functions for use in the open
source parts of the standard library since the `math`
package is currently closed source and cannot be depended
on in the open source parts of the standard library.
"""

# ===----------------------------------------------------------------------===#
# max
# ===----------------------------------------------------------------------===#


@always_inline
fn max(x: Int, y: Int) -> Int:
    """Gets the maximum of two integers.

    Args:
      x: Integer input to max.
      y: Integer input to max.

    Returns:
      Maximum of x and y.
    """
    return __mlir_op.`index.maxs`(x.value, y.value)


@always_inline
fn max[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Performs elementwise maximum of x and y.

    An element of the result SIMD vector will be the maximum of the
    corresponding elements in x and y.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      x: First SIMD vector.
      y: Second SIMD vector.

    Returns:
      A SIMD vector containing the elementwise maximum of x and y.
    """
    return x.max(y)


# ===----------------------------------------------------------------------===#
# min
# ===----------------------------------------------------------------------===#


@always_inline
fn min(x: Int, y: Int) -> Int:
    """Gets the minimum of two integers.

    Args:
      x: Integer input to max.
      y: Integer input to max.

    Returns:
      Minimum of x and y.
    """
    return __mlir_op.`index.mins`(x.value, y.value)


@always_inline
fn min[
    type: DType, simd_width: Int
](x: SIMD[type, simd_width], y: SIMD[type, simd_width]) -> SIMD[
    type, simd_width
]:
    """Gets the elementwise minimum of x and y.

    An element of the result SIMD vector will be the minimum of the
    corresponding elements in x and y.

    Parameters:
      type: The `dtype` of the input and output SIMD vector.
      simd_width: The width of the input and output SIMD vector.

    Args:
      x: First SIMD vector.
      y: Second SIMD vector.

    Returns:
      A SIMD vector containing the elementwise minimum of x and y.
    """
    return x.min(y)
