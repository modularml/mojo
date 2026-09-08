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

# Complete vector addition example with kernel, host function, and verification
# This is the complete implementation combining all the concepts from Chapter 2

from std.math import ceildiv
from max.gpu import global_idx
from max.gpu.host import DeviceContext

# ========================== KERNEL CODE ==========================


def vec_add_kernel(
    a_d: UnsafePointer[Float32, MutUntrackedOrigin],
    b_d: UnsafePointer[Float32, MutUntrackedOrigin],
    c_d: UnsafePointer[Float32, MutUntrackedOrigin],
    n_dev: Int32,
):
    """GPU kernel for vector addition.

    Args:
        a_d: Input vector A (device).
        b_d: Input vector B (device).
        c_d: Output vector C (device).
        n_dev: Number of elements in the vectors.
    """
    # Int is not device-passable; widen the fixed-width arg.
    var n = Int(n_dev)
    var i = global_idx.x
    if i < n:
        c_d[i] = a_d[i] + b_d[i]


def vec_add(
    a: UnsafePointer[Float32, MutUntrackedOrigin],
    b: UnsafePointer[Float32, MutUntrackedOrigin],
    c: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Vector addition host function.

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
        Int32(n),
        grid_dim=grid_dim,
        block_dim=block_dim,
    )

    # Copy result from device to host
    ctx.enqueue_copy(c, c_d)

    # Synchronize to ensure completion
    ctx.synchronize()


# ========================== TEST CODE ==========================


def main() raises:
    var n = 512

    # Allocate host memory
    var a = alloc[Float32](n)
    var b = alloc[Float32](n)
    var c = alloc[Float32](n)
    var c_ref = alloc[Float32](n)

    # Initialize input arrays
    for i in range(n):
        a[i] = Float32(i)
        b[i] = Float32(i)

    # Perform vector addition on GPU
    with DeviceContext() as ctx:
        vec_add(a, b, c, n, ctx)

    # Compute reference result on CPU
    for i in range(n):
        c_ref[i] = a[i] + b[i]

    # Verify results
    for i in range(n):
        if c[i] != c_ref[i]:
            print("Error at index", i, ":", c[i], "!=", c_ref[i])
            return

    print("Success")

    # Clean up
    a.free()
    b.free()
    c.free()
    c_ref.free()
