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

# Figure 2.13: Complete host function with memory management and kernel launch
# Demonstrates the complete pattern for GPU execution

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext

# ========================== KERNEL CODE ==========================


def vec_add_kernel(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    c: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
):
    """GPU kernel for vector addition.

    Args:
        a: Input vector A (device).
        b: Input vector B (device).
        c: Output vector C (device).
        n: Number of elements in the vectors.
    """
    var i = global_idx.x
    if i < n:
        c[i] = a[i] + b[i]


# ========================== TEST CODE ==========================


def vec_add(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    c: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Complete vector addition host function.

    This function demonstrates the complete pattern:
    1. Allocate device memory
    2. Copy data to device
    3. Configure and launch kernel
    4. Copy results back to host
    5. Clean up device memory

    Args:
        a: Input vector A (host).
        b: Input vector B (host).
        c: Output vector C (host).
        n: Number of elements in the vectors.
        ctx: Device context for GPU operations.
    """
    comptime dtype = DType.float32

    # Allocate device memory
    var a_d = ctx.enqueue_create_buffer[dtype](n)
    var b_d = ctx.enqueue_create_buffer[dtype](n)
    var c_d = ctx.enqueue_create_buffer[dtype](n)

    # Copy data from host to device
    ctx.enqueue_copy(a_d, a)
    ctx.enqueue_copy(b_d, b)

    # Calculate launch configuration
    var block_dim = 256
    var grid_dim = ceildiv(n, block_dim)

    # Launch kernel
    ctx.enqueue_function[vec_add_kernel](
        a_d,
        b_d,
        c_d,
        n,
        grid_dim=grid_dim,
        block_dim=block_dim,
    )

    # Copy result from device to host
    ctx.enqueue_copy(c, c_d)

    # Note: Device buffers are automatically freed when they go out of scope
