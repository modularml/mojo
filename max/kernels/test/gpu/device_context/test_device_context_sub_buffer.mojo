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

from max.gpu import global_idx
from max.gpu.host import DeviceContext
from std.testing import assert_equal


def vec_func(
    in0: ImmPointer[Float32, ImmutAnyOrigin],
    in1: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len_dev: Int32,
    supplement_dev: Int32,
):
    var len = Int(len_dev)
    var supplement = Int(supplement_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = in0[tid] + in1[tid] + Float32(supplement)


def test(ctx: DeviceContext) raises:
    comptime length = 1024

    # Allocate the input buffers as sub buffers of a bigger one
    var in_host = ctx.enqueue_create_host_buffer[.float32](2 * length)
    var out_host = ctx.enqueue_create_host_buffer[.float32](length)
    ctx.synchronize()

    for i in range(length):
        in_host[i] = Float32(i)
        in_host[i + length] = 2

    var in_device = ctx.enqueue_create_buffer[.float32](2 * length)
    var in0_device = in_device.create_sub_buffer[.float32](0, length)
    var in1_device = in_device.create_sub_buffer[.float32](length, length)

    var out_device = ctx.enqueue_create_buffer[.float32](length)

    ctx.enqueue_copy(in_device, in_host)

    var block_dim = 32
    var supplement = 5

    ctx.enqueue_function[vec_func](
        in0_device,
        in1_device,
        out_device,
        Int32(length),
        Int32(supplement),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    # Make sure our main input device tensor doesn't disappear
    _ = in_device

    ctx.enqueue_copy(out_host, out_device)

    ctx.synchronize()

    var expected: List[Float32] = [
        7.0,
        8.0,
        9.0,
        10.0,
        11.0,
        12.0,
        13.0,
        14.0,
        15.0,
        16.0,
    ]
    for i in range(10):
        print("at index", i, "the value is", out_host[i])
        assert_equal(out_host[i], expected[i])


def main() raises:
    with DeviceContext() as ctx:
        test(ctx)
