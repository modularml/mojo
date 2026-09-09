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
"""Implements the `arange` operation that fills a tensor with evenly-spaced values over a range."""

from std.math import ceil, iota

from extensibility import *

from std.utils.index import IndexList

# ===-----------------------------------------------------------------------===#
# Arange op
# ===-----------------------------------------------------------------------===#


@always_inline
def arange[
    dtype: DType, simd_width: Int
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
    index: IndexList[1],
) -> SIMD[dtype, simd_width]:
    """Computes a `simd_width`-wide vector of evenly-spaced values for the `arange` operation.

    Returns `start + (iota * step)` where `iota` is seeded from the leading
    index in `index`, producing one SIMD vector of the requested range.

    Parameters:
        dtype: Element type of the range values.
        simd_width: Number of elements per SIMD vector.

    Args:
        start: First value of the range.
        stop: Exclusive upper (or lower) bound of the range.
        step: Spacing between consecutive values.
        index: Per-element offset applied to the iota counter.

    Returns:
        A SIMD vector holding the computed range values for this tile.
    """
    return start + (iota[dtype, simd_width](Scalar[dtype](index[0])) * step)


@always_inline
def arange_shape[
    dtype: DType
](
    start: Scalar[dtype],
    stop: Scalar[dtype],
    step: Scalar[dtype],
) raises -> IndexList[1]:
    """Computes the output shape (number of elements) for the `arange` operation.

    Validates that `step` is non-zero and consistent with the `start`/`stop`
    ordering, then returns the element count as a single-element `IndexList`.

    Parameters:
        dtype: Element type of the range values.

    Args:
        start: First value of the range.
        stop: Exclusive upper (or lower) bound of the range.
        step: Spacing between consecutive values.

    Returns:
        A single-element `IndexList` holding the number of generated values.

    Raises:
        Error: If `step` is zero, or if the `start`/`stop`/`step` ordering is invalid.
    """
    if step == 0:
        raise Error("[range] step must be non-zero")

    comptime if start.dtype.is_integral():
        if step > 0 and stop < start:
            raise Error("[range] requires (start <= stop) for positive step")

        if step < 0 and start < stop:
            raise Error("[range] requires (stop <= start) for negative step")

        return IndexList[1](len(range(Int(start), Int(stop), Int(step))))
    else:
        return IndexList[1](Int(ceil(abs(stop - start) / abs(step))))
