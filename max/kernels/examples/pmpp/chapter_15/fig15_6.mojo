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

"""Figure 15.6 - Matrix multiplication compute function."""

comptime tM = 8
comptime tN = 4


@always_inline
def mm(
    k: Int,
    a: UnsafePointer[Float32, _, address_space=.SHARED],
    lda: Int,
    b: UnsafePointer[Float32, _, address_space=.SHARED],
    ldb: Int,
    c: SIMD[.float32, tM * tN],
) -> SIMD[.float32, tM * tN]:
    """Compute matrix multiplication: c += a @ b.

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
    for row in range(tM):
        for col in range(tN):
            for i in range(k):
                result[row * tN + col] += Float32(a[row * lda + i]) * Float32(
                    b[i * ldb + col]
                )
    return result


def main():
    print("Figure 15.6 - Matrix multiplication compute function")
    print("This is a helper function used in matrix multiplication kernels.")
    print(
        "It performs the actual multiplication on tiles stored in shared"
        " memory."
    )
    print("SUCCESS: Module loaded correctly")
