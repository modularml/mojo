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

# Figure 2.4: Simple CPU-only vector addition
# Compute vector sum C_h = A_h + B_h


# ========================== KERNEL CODE ==========================


def vec_add(
    a_h: UnsafePointer[Float32, MutUntrackedOrigin],
    b_h: UnsafePointer[Float32, MutUntrackedOrigin],
    c_h: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
):
    """Compute vector sum C_h = A_h + B_h on the CPU.

    Args:
        a_h: Input vector A (host).
        b_h: Input vector B (host).
        c_h: Output vector C (host).
        n: Number of elements in the vectors.
    """
    for i in range(n):
        c_h[i] = a_h[i] + b_h[i]


# ========================== TEST CODE ==========================


def main() raises:
    # Memory allocation for arrays A, B, and C
    var n = 10
    var a = alloc[Float32](n)
    var b = alloc[Float32](n)
    var c = alloc[Float32](n)

    # Initialize input arrays
    for i in range(n):
        a[i] = Float32(i)
        b[i] = Float32(i)

    # Perform vector addition
    vec_add(a, b, c, n)

    # Clean up
    a.free()
    b.free()
    c.free()
