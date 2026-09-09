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

# Figure 2.12: Launch configuration calculation
# Demonstrates how to calculate grid and block dimensions

from std.math import ceildiv
from max.gpu.host import DeviceContext

# ========================== KERNEL CODE ==========================


def vec_add(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    c: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Vector addition with launch configuration.

    Demonstrates how to calculate appropriate grid and block dimensions.
    Launch ceil(n/256) blocks of 256 threads each.

    Args:
        a: Input vector A (device).
        b: Input vector B (device).
        c: Output vector C (device).
        n: Number of elements in the vectors.
        ctx: Device context for GPU operations.
    """
    # A_d, B_d, C_d allocations and copies are omitted

    # Calculate launch configuration
    var block_dim = 256
    var _ = ceildiv(n, block_dim)  # grid_dim calculation

    # Note: This is just demonstrating the calculation.
    # The actual kernel launch would look like:
    # var grid_dim = ceildiv(n, block_dim)
    # ctx.enqueue_function[vec_add_kernel](
    #     a, b, c, n,
    #     grid_dim=grid_dim,
    #     block_dim=block_dim,
    # )
