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
"""Implements the SIMD strategy for generating random `SIMD` values in property tests."""

from . import Strategy
from std.testing.prop.random import Rng


__extension SIMD:
    @staticmethod
    def strategy(
        *,
        min: Scalar[dtype] = Scalar[dtype].MIN_FINITE,
        max: Scalar[dtype] = Scalar[dtype].MAX_FINITE,
    ) -> _SIMDStrategy[dtype, length]:
        """Returns a strategy for generating random SIMD values.

        Args:
            min: The minimum value for the SIMD vector.
            max: The maximum value for the SIMD vector.

        Returns:
            A strategy for generating random SIMD values.
        """
        return _SIMDStrategy[dtype, length](min=min, max=max)


struct _SIMDStrategy[dtype: DType, size: Int](Movable, Strategy):
    comptime Value = SIMD[Self.dtype, Self.size]

    var _min: Scalar[Self.dtype]
    var _max: Scalar[Self.dtype]

    def __init__(
        out self,
        *,
        min: Scalar[Self.dtype] = Scalar[Self.dtype].MIN_FINITE,
        max: Scalar[Self.dtype] = Scalar[Self.dtype].MAX_FINITE,
    ):
        self._min = min
        self._max = max

    # TODO: Provide better more consistent "corner case" values
    # e.g. 0, -1, 1, max, min, max-1, min+1, etc...
    def value(mut self, mut rng: Rng) raises -> Self.Value:
        var result = SIMD[Self.dtype, Self.size](0)

        comptime for i in range(Self.size):
            result[i] = rng.rand_scalar[Self.dtype](
                min=self._min, max=self._max
            )
        return result
