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
"""Provides helper functions for comparing values in tests."""

# ===----------------------------------------------------------------------=== #
# utils
# ===----------------------------------------------------------------------=== #


def _minmax[
    dtype: DType, //
](x: Pointer[Scalar[dtype], _], N: Int) -> Tuple[Scalar[dtype], Scalar[dtype]]:
    var max_val = x[unsafe_offset=0]
    var min_val = x[unsafe_offset=0]
    for i in range(1, N):
        if x[unsafe_offset=i] > max_val:
            max_val = x[unsafe_offset=i]
        if x[unsafe_offset=i] < min_val:
            min_val = x[unsafe_offset=i]
    return (min_val, max_val)


def compare[
    dtype: DType, verbose: Bool = True
](
    x: Pointer[Scalar[dtype], _],
    y: Pointer[Scalar[dtype], _],
    num_elements: Int,
    *,
    msg: String = "",
) -> SIMD[dtype, 4]:
    """Compares two arrays and computes absolute and relative error statistics.

    Parameters:
        dtype: The data type of the arrays to compare.
        verbose: Whether to print the error statistics.

    Args:
        x: First array to compare.
        y: Second array to compare.
        num_elements: Number of elements in each array.
        msg: Optional message to print before the statistics.

    Returns:
        A SIMD vector containing [atol_min, atol_max, rtol_min, rtol_max].
    """
    var atol = List(length=num_elements, fill=Scalar[dtype](0))
    var rtol = List(length=num_elements, fill=Scalar[dtype](0))

    for i in range(num_elements):
        var d = abs(x[unsafe_offset=i] - y[unsafe_offset=i])
        var e = abs(d / y[unsafe_offset=i])
        atol[i] = d
        rtol[i] = e

    var atol_minmax = _minmax(atol.unsafe_ptr(), num_elements)
    var rtol_minmax = _minmax(rtol.unsafe_ptr(), num_elements)
    if verbose:
        if msg:
            print(msg)
        print("AbsErr-Min/Max", atol_minmax[0], atol_minmax[1])
        print("RelErr-Min/Max", rtol_minmax[0], rtol_minmax[1])
        print("==========================================================")
    return SIMD[dtype, 4](
        atol_minmax[0], atol_minmax[1], rtol_minmax[0], rtol_minmax[1]
    )
