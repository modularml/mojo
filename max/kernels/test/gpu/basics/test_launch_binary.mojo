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
"""This test showcases how one can launch a precompiled device binary from Mojo."""

from max.gpu import global_idx
from std.testing import assert_equal

from max.gpu.host import DeviceContext
from max.gpu.host.device_context import DeviceExternalFunction
from max.gpu.host.compile import _compile_code


def vec_func(
    in0: ImmPointer[Float32, ImmutAnyOrigin],
    in1: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len: Int,
):
    var tid = global_idx.x
    output[tid] = in0[tid] + in1[tid]


def test_vec_add(ctx: DeviceContext) raises:
    comptime length = 1024

    var in0_device = ctx.enqueue_create_buffer[.float32](length)
    var in1_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    with in0_device.map_to_host() as in0_host, in1_device.map_to_host() as in1_host, out_device.map_to_host() as out_host:
        for i in range(length):
            in0_host[i] = Float32(i)
            in1_host[i] = 2

    var block_dim = 32

    var info = DeviceExternalFunction(
        ctx, _compile_code[vec_func, emission_kind="object"]()
    )
    ctx.enqueue_function(
        info,
        in0_device,
        in1_device,
        out_device,
        length,
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    ctx.synchronize()

    with out_device.map_to_host() as out_host:
        for i in range(length):
            assert_equal(
                out_host[i],
                Float32(i + 2),
                msg=String(t"at index{i} the value is{out_host[i]}"),
            )


def main() raises:
    with DeviceContext() as ctx:
        test_vec_add(ctx)
