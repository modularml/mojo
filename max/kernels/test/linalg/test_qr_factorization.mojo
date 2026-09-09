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

from std.os import abort
from std.random import rand, seed

import internal_utils
from layout import Layout, LayoutTensor, UNKNOWN_VALUE
from linalg.qr_factorization import form_q, qr_factorization
from std.memory import alloc, unsafe_memcpy
from std.testing import assert_almost_equal


# A is a general matrix, B is a non-unit upper triangular matrix
def trmm[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    B: LayoutTensor[dtype, element_layout=element_layout, ...],
    C: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    var m, k1 = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    var k, n = Int(B.runtime_layout.shape[0]), Int(B.runtime_layout.shape[1])
    var min_kn = min(k, n)
    if k1 < min_kn:
        abort("trmm: A and B must have the at least the same number of columns")
    # C.fill(0.0) doesn't work
    for i in range(m):
        for j in range(n):
            C[i, j] = 0.0
    for i in range(m):
        for j in range(min_kn):
            for p in range(j + 1):
                C[i, j] += A[i, p] * B[p, j]


def a_mul_bt[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    B: LayoutTensor[dtype, element_layout=element_layout, ...],
    C: LayoutTensor[mut=True, dtype, element_layout=element_layout, ...],
):
    var m, k1 = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    var n, k = Int(B.runtime_layout.shape[0]), Int(B.runtime_layout.shape[1])
    if k1 != k:
        abort("a_mul_bt: A and B must have the same number of columns")
    # C.fill(0.0) doesn't work
    for i in range(m):
        for j in range(n):
            C[i, j] = 0.0
    for i in range(m):
        for j in range(n):
            for p in range(k):
                C[i, j] += A[i, p] * B[j, p]


def all_almost_id[
    dtype: DType,
    element_layout: Layout,
](
    A: LayoutTensor[dtype, element_layout=element_layout, ...],
    atol: Float64,
    rtol: Float64,
) raises:
    var m, n = Int(A.runtime_layout.shape[0]), Int(A.runtime_layout.shape[1])
    for i in range(m):
        for j in range(n):
            var reference = SIMD[dtype, A.element_layout.size()](
                1.0 if i == j else 0.0
            )
            assert_almost_equal(A[i, j], reference, atol=atol, rtol=rtol)


def create_vector[
    dtype: DType, layout: Layout
](
    m: Int,
    ptr: MutPointer[Scalar[dtype], _],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m),
        type_of(result.runtime_layout.stride)(1),
    )
    return {ptr, dynamic_layout}


def create_tensor[
    dtype: DType, layout: Layout
](
    m: Int,
    n: Int,
    ptr: MutPointer[Scalar[dtype], _],
    out result: LayoutTensor[dtype, layout, ptr.origin],
):
    var dynamic_layout = type_of(result.runtime_layout)(
        type_of(result.runtime_layout.shape)(m, n),
        type_of(result.runtime_layout.stride)(1, m),
    )
    return {ptr, dynamic_layout}


def main() raises:
    var atol = 1e-5
    var rtol = 1e-3
    var m, n = 80, 50
    var min_mn = min(m, n)
    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, UNKNOWN_VALUE)
    comptime v_layout = Layout(UNKNOWN_VALUE)
    comptime T = Float32
    var a_ptr = alloc[T](m * n)
    var a_ptr_copy = alloc[T](m * n)
    var v_ptr = alloc[T](min_mn)
    seed(123)
    rand[.float32](a_ptr, m * n)
    var a = create_tensor[.float32, a_layout](m, n, a_ptr)
    unsafe_memcpy(dest=a_ptr_copy, src=a_ptr, count=m * n)
    # factorize
    var a_copy = create_tensor[.float32, a_layout](m, n, a_ptr_copy)
    var v = create_vector[.float32, v_layout](min_mn, v_ptr)
    qr_factorization[.float32](v, a)
    # form Q
    var q_ptr = alloc[T](m * m)
    var q = create_tensor[.float32, a_layout](m, m, q_ptr)
    form_q[.float32](v, a, q)
    print("check backward stability")
    var q_mul_r_ptr = alloc[T](m * n)
    var q_mul_r = create_tensor[.float32, a_layout](m, n, q_mul_r_ptr)
    trmm[.float32](q, a, q_mul_r)
    internal_utils.assert_almost_equal(
        q_mul_r.ptr, a_copy.ptr, m * n, atol=atol, rtol=rtol
    )
    print("check orthogonality")
    var q_mul_qt_ptr = alloc[T](m * m)
    var q_mul_qt = create_tensor[.float32, a_layout](m, m, q_mul_qt_ptr)
    a_mul_bt[.float32](q, q, q_mul_qt)
    all_almost_id(q_mul_qt, atol=atol, rtol=rtol)

    a_ptr.free()
    a_ptr_copy.free()
    v_ptr.free()
    q_ptr.free()
    q_mul_r_ptr.free()
    q_mul_qt_ptr.free()
