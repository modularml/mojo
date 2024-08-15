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
"""Defines equation solving utilities."""


fn newtons_method[
    func: fn (FloatLiteral) capturing -> FloatLiteral,
    deriv: fn (FloatLiteral) capturing -> FloatLiteral,
    iters: Int = 8,
    atol: FloatLiteral = FloatLiteral.nan,
    epsilon: FloatLiteral = FloatLiteral.nan,
](x0: FloatLiteral, y_offset: FloatLiteral = 0) -> FloatLiteral:
    """Implements newtons method for solving trancendental equations.

    Converges on the roots of the input function `f` using it's derivative `fp`.
    If `tolerance` and `epsilon` are left undefined, only iterations will be used, and may return non-convergent results.

    Parameters:
        func: The function to find the solutions to.
        deriv: The first derivative of f (not calculated automatically).
        iters: The number of iterations to perform.
        atol: If provided, results within the tolerance will be considered solved.
        epsilon: If provided, the calculation will return `nan` for values that explode.

    Args:
        x0: The initial guess of the solution. Determines which solution is found, and how fast it converges.
        y_offset: A vertical offset applied to the input function `f`. Use for solving the inverse of `f`, for values other than 0.

    Returns:
        The converged value, or `nan` if no solution was found.
    """

    var x1: FloatLiteral = x0

    for _ in range(iters):
        var yp: FloatLiteral = deriv(x1)

        @parameter
        if epsilon == epsilon:
            if abs(yp) <= epsilon:
                return FloatLiteral.nan

        var x2: FloatLiteral = x1 - (func(x1) - y_offset) / yp

        @parameter
        if atol == atol:
            if abs(x2 - x1) <= atol:
                return x2

        x1 = x2

    @parameter
    if atol == atol or epsilon == epsilon:
        return FloatLiteral.nan
    return x1
