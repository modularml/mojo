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

"""Figure 15.9 - Optimized matrix multiplication with register blocking."""

comptime tM = 8
comptime tN = 4


@always_inline
def mm_optimized(
    k: Int,
    a: UnsafePointer[Float32, _, address_space=.SHARED],
    lda: Int,
    b: UnsafePointer[Float32, _, address_space=.SHARED],
    ldb: Int,
    c: SIMD[.float32, tM * tN],
) -> SIMD[.float32, tM * tN]:
    """Optimized matrix multiplication with register blocking.

    This version loads strips of A and B into register arrays first,
    then performs outer product updates. This improves register reuse
    and reduces shared memory traffic.

    Args:
        k: Inner dimension size.
        a: Pointer to A tile in shared memory.
        lda: Leading dimension of A.
        b: Pointer to B tile in shared memory.
        ldb: Leading dimension of B.
        c: Accumulator array (register file as SIMD vector).

    Returns:
        Updated accumulator array.
    """
    var result = c

    for i in range(k):
        # Load an m x 1 strip from A into registers
        var a_r = SIMD[.float32, tM](0.0)
        for inner_row in range(tM):
            a_r[inner_row] = Float32(a[inner_row * lda + i])

        # Load a 1 x n strip from B into registers
        var b_r = SIMD[.float32, tN](0.0)
        for inner_col in range(tN):
            b_r[inner_col] = Float32(b[i * ldb + inner_col])

        # Outer product: update result with a_r * b_r^T
        for row in range(tM):
            for col in range(tN):
                result[row * tN + col] += a_r[row] * b_r[col]

    return result


def main():
    print("Figure 15.9 - Optimized mm function with register blocking")
    print(
        "This version improves register reuse and reduces shared memory"
        " traffic."
    )
    print(
        "Uses outer product pattern: loads strips into registers, then updates"
        " accumulator."
    )
    print("SUCCESS: Module loaded correctly")
