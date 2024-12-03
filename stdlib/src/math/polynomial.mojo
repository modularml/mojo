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
"""Provides two implementations for evaluating polynomials.

You can import these APIs from the `math` package. For example:

```mojo
from math.polynomial import polynomial_evaluate
```
"""

from collections import List

# ===-----------------------------------------------------------------------===#
# polynomial_evaluate
# ===-----------------------------------------------------------------------===#


@always_inline
fn polynomial_evaluate[
    dtype: DType,
    simd_width: Int, //,
    coefficients: List[SIMD[dtype, simd_width], *_],
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Evaluates the polynomial.

    Parameters:
        dtype: The dtype of the value.
        simd_width: The simd_width of the computed value.
        coefficients: The coefficients.

    Args:
        x: The value to compute the polynomial with.

    Returns:
        The polynomial evaluation results using the specified value and the
        constant coefficients.
    """
    return _horner_evaluate[coefficients](x)


# ===-----------------------------------------------------------------------===#
# Horner Method
# ===-----------------------------------------------------------------------===#


@always_inline
fn _horner_evaluate[
    dtype: DType,
    simd_width: Int, //,
    coefficients: List[SIMD[dtype, simd_width], *_],
](x: SIMD[dtype, simd_width]) -> SIMD[dtype, simd_width]:
    """Evaluates the polynomial using the passed in value and the specified
    coefficients using the Horner scheme. The Horner scheme evaluates the
    polynomial at point x as `horner(x, coeffs)` where x is a scalar and coeffs
    is a list of coefficients [c0, c1, c2, ..., cn] by:
    ```
    horner(x, coeffs)
        = c0 + x * (c1 + x * (c2 + x * (... + x * cn)))
        = fma(x, horner(x, coeffs[1:]), coeffs[0])
    ```

    Parameters:
        dtype: The dtype of the value.
        simd_width: The simd_width of the computed value.
        coefficients: The coefficients.

    Args:
        x: The value to compute the polynomial with.

    Returns:
        The polynomial specified by the coefficients evaluated at value x.
    """
    alias num_coefficients = len(coefficients)
    alias c_last = coefficients[num_coefficients - 1]
    alias c_second_from_last = coefficients[num_coefficients - 2]

    var result = x.fma(c_last, c_second_from_last)

    @parameter
    for i in reversed(range(num_coefficients - 2)):
        alias c = coefficients[i]
        result = result.fma(x, c)

    return result
