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

# Figure 2.10: GPU kernel for vector addition
# Compute vector sum C = A + B
# Each thread performs pairwise addition

from max.gpu import global_idx

# ========================== KERNEL CODE ==========================


def vec_add(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    c: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
):
    """GPU kernel for vector addition.

    Each thread computes one element of the output vector.

    Args:
        a: Input vector A (device).
        b: Input vector B (device).
        c: Output vector C (device).
        n: Number of elements in the vectors.
    """
    # Calculate global thread index
    var i = global_idx.x

    # Boundary check
    if i < n:
        c[i] = a[i] + b[i]
