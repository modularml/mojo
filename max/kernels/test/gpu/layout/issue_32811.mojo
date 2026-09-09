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

from max.gpu.host import DeviceContext
from max.gpu import block_idx, thread_idx
from layout import *


def gpu_kernel(
    dst: MutPointer[Float32, MutAnyOrigin],
    rhs: ImmPointer[Float32, ImmutAnyOrigin],
    lhs: ImmPointer[Float32, ImmutAnyOrigin],
):
    dst[block_idx.x * 4 + thread_idx.x] = (
        rhs[block_idx.x * 4 + thread_idx.x]
        + lhs[block_idx.x * 4 + thread_idx.x]
    )

    _ = LayoutTensor[.float32, Layout(IntTuple(16, 1), IntTuple(1, 1))](dst)


def main() raises:
    with DeviceContext() as ctx:
        var vec_a_ptr = alloc[Float32](16)
        var vec_b_ptr = alloc[Float32](16)
        var vec_c_ptr = alloc[Float32](16)

        var vec_a_dev = ctx.enqueue_create_buffer[.float32](16)
        var vec_b_dev = ctx.enqueue_create_buffer[.float32](16)
        var vec_c_dev = ctx.enqueue_create_buffer[.float32](16)

        for i in range(16):
            vec_a_ptr[i] = Float32(i)
            vec_b_ptr[i] = Float32(i)
            vec_c_ptr[i] = 0

        ctx.enqueue_copy(vec_a_dev, vec_a_ptr)
        ctx.enqueue_copy(vec_b_dev, vec_b_ptr)
        ctx.enqueue_copy(vec_c_dev, vec_c_ptr)

        ctx.enqueue_function[gpu_kernel](
            vec_c_dev,
            vec_a_dev,
            vec_b_dev,
            block_dim=(4),
            grid_dim=(4),
        )

        ctx.enqueue_copy(vec_a_ptr, vec_a_dev)
        ctx.enqueue_copy(vec_b_ptr, vec_b_dev)
        ctx.enqueue_copy(vec_c_ptr, vec_c_dev)

        ctx.synchronize()

        for i in range(16):
            print(vec_a_ptr[i], "+", vec_b_ptr[i], "=", vec_c_ptr[i])
