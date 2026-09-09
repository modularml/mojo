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
from std.testing import assert_equal, assert_true

from std.utils.numerics import inf, isnan, nan, neg_inf


def id(
    input: ImmPointer[Float32, ImmutAnyOrigin],
    output: MutPointer[Float32, MutAnyOrigin],
    len_dev: Int32,
):
    var len = Int(len_dev)
    var tid = global_idx.x
    if tid >= len:
        return
    output[tid] = Float32(BFloat16(input[tid]))


@no_inline
def run_vec_add(ctx: DeviceContext) raises:
    print("== run_vec_add")

    comptime length = 1024

    var in_host = ctx.enqueue_create_host_buffer[.float32](length)

    for i in range(length):
        in_host[i] = Float32(i)

    in_host[4] = nan[.float32]()
    in_host[5] = inf[.float32]()
    in_host[6] = neg_inf[.float32]()
    in_host[7] = -0.0

    var in_device = ctx.enqueue_create_buffer[.float32](length)
    var out_device = ctx.enqueue_create_buffer[.float32](length)

    in_device.enqueue_copy_from(in_host)

    var block_dim = 32

    comptime kernel = id
    ctx.enqueue_function[kernel](
        in_device,
        out_device,
        Int32(length),
        grid_dim=(length // block_dim),
        block_dim=(block_dim),
    )

    var expected: List[Float32] = [
        0.0,
        1.0,
        2.0,
        3.0,
        nan[.float32](),
        inf[.float32](),
        neg_inf[.float32](),
        -0.0,
        8.0,
        9.0,
    ]
    with out_device.map_to_host() as out_host:
        for i in range(10):
            print("at index", i, "the value is", out_host[i])
            if isnan(expected[i]):
                assert_true(isnan(out_host[i]))
            else:
                assert_equal(expected[i], out_host[i])

    _ = in_device


def main() raises:
    with DeviceContext() as ctx:
        run_vec_add(ctx)
