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

"""Provides QR factorization of matrices using Householder reflections."""

from std.math import copysign, sqrt
from std.os import abort

from layout import Layout, LayoutTensor


def qr_factorization[
    dtype: DType,
    element_layout: Layout,
](
    sigma: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
    A: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    """Performs QR factorization of a matrix `A` using the Householder reflector
    method.

    This function computes the QR factorization of matrix `A` in-place using
    Householder reflections. The result is stored directly in the input matrix
    `A`, with scaling factors in `sigma`. The implementation follows the LAPACK
    algorithm for generating Householder reflectors in-place.

    Parameters:
        dtype: Element type of the matrix `A` and the scaling vector `sigma`.
        element_layout: Memory layout of the `LayoutTensor` inputs.

    Args:
        sigma: Vector of length `n` holding the Householder scaling factor
            `ξ/ν` for each column. Written in-place during factorization.
        A: `m×n` matrix to factorize. Modified in-place: the strictly-lower
            triangle stores the Householder vectors and the upper triangle
            (including the diagonal) stores the `R` factor.

    Algorithm:
        The Householder reflector is defined as:
            U = I - σww^H
        where:
            w = (x + νe₁)/ξ
            σ = ξ/ν
            ξ = x₀ + ν
            ν = sign(x₀)‖x‖₂

        This ensures that U^H x = -νe₁ and U^H U = I.

    References:
        [1] Lehoucq, R. B. (1996). The computation of elementary unitary matrices.
            ACM Transactions on Mathematical Software, 22(4), 393-400.
            https://www.netlib.org/lapack/lawnspdf/lawn72.pdf
            https://library.eecs.utk.edu/files/ut-cs-94-233.pdf

    Note:
        There is a typo in reference [lawn72]. The correct result is U^H x =
        -νe₁.
    """
    var m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    for k in range(n):
        var x_0 = A[k, k]
        var x_norm = SIMD[dtype, A.element_layout.size()](0.0)
        for i in range(m - k):
            x_norm += A[k + i, k] * A[k + i, k]
        x_norm = sqrt(x_norm)
        var nu = copysign(x_norm, x_0)
        A[k, k] = -nu
        var xi = x_0 + nu
        var inv_xi = 1.0 / xi
        for i in range(m - k - 1):
            A[k + i + 1, k] *= inv_xi
        sigma[k] = xi / nu
        # apply reflector to A[k + 1:m, k + 1:n] for each column vector v in A[k
        # :m, k + 1:n], we compute:
        #   (I - \sigma [1; w] [1; w]^T) v = v - \sigma [1; w] ([1; w]^T v)
        # = v - \sigma ([1; w]^T v) [1; w]
        # = v - s [1; w]            where  s = \sigma * (v[0] + w^T v[1:])
        # v[0] -= s
        # v[1:] -= s * w
        for j in range(n - k - 1):
            var dot = A[k, k + j + 1]  # v[0]
            for i in range(m - k - 1):
                var wi = A[k + i + 1, k]  # w[i]
                var vi = A[k + i + 1, k + j + 1]  # v[i + 1]
                dot += wi * vi
            var s = sigma[k] * dot
            A[k, k + j + 1] -= s  # v[0] -= s
            for i in range(m - k - 1):
                A[k + i + 1, k + j + 1] -= (
                    s * A[k + i + 1, k]
                )  # v[i + 1] -= s * w


def apply_q[
    dtype: DType,
    element_layout: Layout,
](
    sigma: LayoutTensor[dtype, element_layout=element_layout, ...],
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    X: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    """Applies the implicit Q factor stored in `A` and `sigma` after calling
    `qr_factorization` to the `X` matrix.

    See `qr_factorization` for more details on the construction of the
    Householder reflector.

    Parameters:
        dtype: Element type of the matrices.
        element_layout: Memory layout of the `LayoutTensor` inputs.

    Args:
        sigma: Vector of length `n` holding the Householder scaling factors
            produced by `qr_factorization`.
        A: `m×n` matrix containing the implicit `Q` factor as produced by
            `qr_factorization`.
        X: `m×q_n` matrix to multiply by `Q`. Must have the same number of
            rows as `A`. Overwritten with `Q·X` in-place.
    """
    var m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    var q_m, q_n = Int(X.runtime_layout.shape[0]), Int(
        X.runtime_layout.shape[1]
    )
    if q_m != m:
        abort("apply_q: X must have the same number of rows as A")
    for k in range(n - 1, -1, -1):
        for j in range(q_n):
            var dot = X[k, j]  # v[0]
            for i in range(m - k - 1):
                var wi = A[k + i + 1, k]  # w[i]
                var vi = X[k + i + 1, j]  # v[i + 1]
                dot += wi * vi
            var s = sigma[k] * dot
            X[k, j] -= s  # v[0] -= s
            for i in range(m - k - 1):
                X[k + i + 1, j] -= s * A[k + i + 1, k]  # v[i + 1] -= s * w


def form_q[
    dtype: DType,
    element_layout: Layout,
](
    sigma: LayoutTensor[dtype, element_layout=element_layout, ...],
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    Q: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    """Forms the Q factor from the implicit Q factor stored in `A` and `sigma`
    after calling `qr_factorization` and stores the result in `Q`.

    Parameters:
        dtype: Element type of the matrices.
        element_layout: Memory layout of the `LayoutTensor` inputs.

    Args:
        sigma: Vector of length `n` holding the Householder scaling factors
            produced by `qr_factorization`.
        A: `m×n` matrix containing the implicit `Q` factor as produced by
            `qr_factorization`.
        Q: `q_m×q_n` output matrix initialized to the identity and
            overwritten with the explicit `Q` factor.
    """
    var q_m, q_n = Int(Q.runtime_layout.shape[0]), Int(
        Q.runtime_layout.shape[1]
    )
    var min_mn = min(q_m, q_n)

    # Q.fill(0.0) doesn't work
    for i in range(q_m):
        for j in range(q_n):
            Q[i, j] = 0.0

    # Set diagonal to 1.0
    for i in range(min_mn):
        Q[i, i] = 1.0

    apply_q[dtype](sigma, A, Q)
