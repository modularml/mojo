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

from max.gpu import block_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.memory import (
    async_copy_commit_group,
    async_copy_wait_group,
)
from layout._fillers import arange
from layout._utils import ManagedLayoutTensor
from layout import Layout, LayoutTensor
from std.testing import assert_true


def test_copy_dram_to_sram_async(ctx: DeviceContext) raises:
    print("== test_copy_dram_to_sram_async")
    comptime tensor_layout = Layout.row_major(4, 16)
    var tensor = ManagedLayoutTensor[.float32, tensor_layout](ctx)
    arange(tensor.tensor())

    var check_state = True

    def copy_to_sram_test_kernel[
        layout: Layout,
    ](
        dram_tensor: LayoutTensor[.float32, layout, ImmutAnyOrigin],
        flag: MutPointer[Scalar[.bool], MutAnyOrigin],
    ):
        var dram_tile = dram_tensor.tile[4, 4](0, block_idx.x)
        var sram_tensor = LayoutTensor[
            .float32,
            Layout.row_major(4, 4),
            MutAnyOrigin,
            address_space=.SHARED,
        ].stack_allocation()
        sram_tensor.copy_from_async(dram_tile)

        async_copy_commit_group()
        async_copy_wait_group(0)

        var col_offset = block_idx.x * 4

        for r in range(4):
            for c in range(4):
                if sram_tensor[r, c] != Float32(r * 16 + col_offset + c):
                    flag[] = False

    comptime kernel = copy_to_sram_test_kernel[tensor_layout]
    var ptr = Pointer(to=check_state).bitcast[Scalar[.bool]]()
    ctx.enqueue_function[kernel](
        tensor.device_tensor(),
        DeviceBuffer[.bool](
            ctx,
            rebind[MutPointer[Scalar[.bool], MutAnyOrigin]](ptr),
            1,
            owning=False,
        ),
        grid_dim=(4),
        block_dim=(1),
    )
    ctx.synchronize()
    assert_true(check_state, "Inconsistent values in shared memory")


def main() raises:
    with DeviceContext() as ctx:
        test_copy_dram_to_sram_async(ctx)
