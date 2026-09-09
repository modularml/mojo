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

# Figure 2.8: Memory management - allocate, copy, and free device memory
# This demonstrates the basic pattern for GPU memory management

from max.gpu.host import DeviceContext

# ========================== KERNEL CODE ==========================


def vec_add(
    a_h: UnsafePointer[Float32, MutUntrackedOrigin],
    b_h: UnsafePointer[Float32, MutUntrackedOrigin],
    c_h: UnsafePointer[Float32, MutUntrackedOrigin],
    n: Int,
    ctx: DeviceContext,
) raises:
    """Vector addition with GPU memory management (no kernel execution).

    Args:
        a_h: Input vector A (host).
        b_h: Input vector B (host).
        c_h: Output vector C (host).
        n: Number of elements in the vectors.
        ctx: Device context for GPU operations.
    """
    comptime dtype = DType.float32

    # Allocate device memory
    var a_d = ctx.enqueue_create_buffer[dtype](n)
    var b_d = ctx.enqueue_create_buffer[dtype](n)
    var c_d = ctx.enqueue_create_buffer[dtype](n)

    # Copy data from host to device
    ctx.enqueue_copy(a_d, a_h)
    ctx.enqueue_copy(b_d, b_h)

    # Kernel invocation code would go here

    # Copy result from device to host
    ctx.enqueue_copy(c_h, c_d)

    # Note: Device buffers are automatically freed when they go out of scope
