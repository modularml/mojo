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
"""Provides functions for bit manipulation.

You can import these APIs from the `bit` package. For example:

```mojo
from bit.utils import count_leading_zeros
```
"""

from sys.info import bitwidthof


# ===-----------------------------------------------------------------------===#
# bitmasks
# ===-----------------------------------------------------------------------===#


@always_inline
fn is_negative_bitmask(value: Int) -> Int:
    """Get a bitmask of whether the value is negative.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is negative, filled with `0`
        otherwise.
    """
    return int(is_negative_bitmask(Scalar[DType.index](value)))


@always_inline
fn is_negative_bitmask[D: DType](value: SIMD[D, _]) -> __type_of(value):
    """Get a bitmask of whether the value is negative.

    Parameters:
        D: The DType.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is negative, filled with `0`
        otherwise.
    """
    constrained[D.is_signed(), "This function is for signed types."]()
    return value >> (bitwidthof[D]() - 1)


@always_inline
fn is_true_bitmask[
    D: DType
](value: SIMD[DType.bool, _]) -> SIMD[D, __type_of(value).size]:
    """Get a bitmask of whether the value is `True`.

    Parameters:
        D: The DType.

    Args:
        value: The value to check.

    Returns:
        A bitmask filled with `1` if the value is `True`, filled with `0`
        otherwise.
    """
    return is_negative_bitmask(value.cast[DType.int8]() - 1).cast[D]()


@always_inline
fn are_equal_bitmask(lhs: Int, rhs: Int) -> Int:
    """Get a bitmask of whether the values are equal.

    Args:
        lhs: The value to check.
        rhs: The value to check.

    Returns:
        A bitmask filled with `1` if the values are equal, filled with `0`
        otherwise.
    """
    alias S = Scalar[DType.index]
    return int(are_equal_bitmask(S(lhs), S(rhs)))


@always_inline
fn are_equal_bitmask[
    D: DType
](lhs: SIMD[D, _], rhs: __type_of(lhs)) -> __type_of(lhs):
    """Get a bitmask of whether the values are equal.

    Parameters:
        D: The DType.

    Args:
        lhs: The value to check.
        rhs: The value to check.

    Returns:
        A bitmask filled with `1` if the values are equal, filled with `0`
        otherwise.
    """
    return is_true_bitmask[D](lhs ^ rhs != 0)
