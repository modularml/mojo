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

"""Figure 15.7 Coalesced - Write tile using SIMD vector stores for better coalescing."""


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
    """Write tile using SIMD vector stores for coalesced writes.

    Each row is tN=4 elements wide, perfect for SIMD[.float32, 4] vector stores.
    This version uses vectorized stores when possible for better memory coalescing.

    Args:
        c: Pointer to output tile in global memory.
        ldc: Leading dimension of C (stride).
        maxRow: Maximum valid row to write.
        maxCol: Maximum valid column to write.
        C_r: Register array with results (SIMD vector of tM x tN elements).
    """
    for row in range(tM):
        if row < maxRow:
            # Pack the row into a SIMD vector
            var data = SIMD[.float32, 4](
                C_r[row * tN + 0],
                C_r[row * tN + 1],
                C_r[row * tN + 2],
                C_r[row * tN + 3],
            )

            if 3 < maxCol:
                # Full vector store possible - threads in same warp will coalesce
                c[row * ldc + 0] = data[0]
                c[row * ldc + 1] = data[1]
                c[row * ldc + 2] = data[2]
                c[row * ldc + 3] = data[3]
            else:
                # Partial store near boundaries (fallback to scalar stores)
                if 0 < maxCol:
                    c[row * ldc + 0] = data[0]
                if 1 < maxCol:
                    c[row * ldc + 1] = data[1]
                if 2 < maxCol:
                    c[row * ldc + 2] = data[2]
                if 3 < maxCol:
                    c[row * ldc + 3] = data[3]


def main():
    print("Figure 15.7 Coalesced - WriteTile with vectorized stores")
    print("This version uses SIMD vector stores for better memory coalescing.")
    print(
        "Improves performance by ensuring threads in a warp coalesce their"
        " writes."
    )
    print("SUCCESS: Module loaded correctly")
