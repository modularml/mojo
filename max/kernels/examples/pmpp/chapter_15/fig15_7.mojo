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

"""Figure 15.7 - Write tile from registers to global memory."""


comptime tM = 8
comptime tN = 4


@always_inline
def writeTile(
    c: UnsafePointer[Float32, MutAnyOrigin],
    ldc: Int,
    maxRow: Int,
    maxCol: Int,
    C_r: SIMD[.float32, tM * tN],
):
    """Write tile from registers to global memory.

    Args:
        c: Pointer to output tile in global memory.
        ldc: Leading dimension of C (stride).
        maxRow: Maximum valid row to write.
        maxCol: Maximum valid column to write.
        C_r: Register array with results (SIMD vector of tM x tN elements).
    """
    for row in range(tM):
        for col in range(tN):
            if row < maxRow and col < maxCol:
                c[row * ldc + col] = C_r[row * tN + col]


def main():
    print("Figure 15.7 - WriteTile function for register to global memory")
    print("This is a helper function used in matrix multiplication kernels.")
    print("It writes computed results from registers to global memory.")
    print("SUCCESS: Module loaded correctly")
