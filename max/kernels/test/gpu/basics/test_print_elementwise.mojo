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

from std.sys import simd_width_of

from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from layout import IntTuple, Layout, LayoutTensor, RuntimeLayout, UNKNOWN_VALUE
from layout._utils import ManagedLayoutTensor

from std.utils.coord import Coord
from std.utils.index import IndexList


def test_elementwise_print[
    c_type: DType,
    c_layout: Layout,
](c01: LayoutTensor[c_type, c_layout, ...], ctx: DeviceContext) raises:
    var M = c01.dim[0]()
    var N = c01.dim[1]() // 2
    comptime simd_width = simd_width_of[
        c_type, target=get_gpu_target["sm_80"]()
    ]()

    @always_inline
    def binary[simd_width: Int, alignment: Int = 1](idx0: Coord) {var}:
        var m: Int = Int(idx0[0].value())
        var n: Int = Int(idx0[1].value())
        print("print thousands of messages: m=", m, " n=", n, sep="")

    print("about to call elementwise, M=", M, "N=", N)
    elementwise[simd_width, target="gpu"](binary, (M, N), ctx)
    print("called elementwise")
    # Avoid exiting in the middle of the call to the kernel that is printing the test messages.
    ctx.synchronize()
    print("finished elementwise")


def runtime_row_major[
    cols: Int
](
    rows: Int,
    out res: RuntimeLayout[
        Layout(IntTuple(UNKNOWN_VALUE, cols), IntTuple(cols, 1))
    ],
):
    return type_of(res).row_major(IndexList[2]((rows, cols)))


def test_dual_matmul[
    N: Int = 512, K: Int = 512
](ctx: DeviceContext, M: Int = 512) raises:
    comptime dst_type = DType.float32
    var layout_c01 = runtime_row_major[2 * N](M)
    var mat_c01 = ManagedLayoutTensor[dst_type](layout_c01, ctx)
    test_elementwise_print(
        mat_c01.device_tensor(),
        ctx,
    )
    print("returned from test_elementwise_print")


def main() raises:
    with DeviceContext() as ctx:
        test_dual_matmul(ctx)
