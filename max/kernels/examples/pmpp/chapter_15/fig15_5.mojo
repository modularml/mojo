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

"""Figure 15.5 - Load tile from global memory to shared memory."""

from max.gpu import thread_idx

comptime NUM_THREADS_PER_BLOCK = 128


@always_inline
def loadTile(
    T: UnsafePointer[Float32, _],
    lda: Int,
    maxRow: Int,
    maxCol: Int,
    T_s: UnsafePointer[Float32, MutAnyOrigin, address_space=.SHARED],
    ldas: Int,
    height: Int,
    width: Int,
):
    """Load tile from global memory to shared memory.

    Args:
        T: Pointer to global memory tile.
        lda: Leading dimension of T (stride).
        maxRow: Maximum valid row to load.
        maxCol: Maximum valid column to load.
        T_s: Pointer to shared memory tile.
        ldas: Leading dimension of T_s (stride).
        height: Height of tile to load.
        width: Width of tile to load.
    """
    var num_rows_per_tile = NUM_THREADS_PER_BLOCK // width
    var num_subtiles = height // num_rows_per_tile

    var tx = thread_idx.x

    for subtile in range(num_subtiles):
        var row, col = divmod(tx, width)
        row += subtile * num_rows_per_tile

        if row < maxRow and col < maxCol:
            T_s[row * ldas + col] = Float32(T[row * lda + col])
        else:
            T_s[row * ldas + col] = Float32(0.0)


def main():
    print("Figure 15.5 - LoadTile function for shared memory loading")
    print("This is a helper function used in matrix multiplication kernels.")
    print(
        "It efficiently loads tiles from global to shared memory with"
        " coalescing."
    )
    print("SUCCESS: Module loaded correctly")
