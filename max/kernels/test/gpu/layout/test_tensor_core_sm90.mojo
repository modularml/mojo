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
from max.gpu.compute.mma import mma
from layout import Layout, LayoutTensor
from layout._utils import ManagedLayoutTensor
from layout.tensor_core import TensorCore

from std.utils.index import IndexList


def arange(tensor: LayoutTensor[mut=True, ...]):
    comptime for i in range(tensor.shape[0]()):
        comptime for j in range(tensor.shape[1]()):
            tensor[i, j] = Scalar[tensor.dtype](i + j)


def load_and_mma_16x8x32[
    out_type: DType,
    in_type: DType,
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: LayoutTensor[mut=True, out_type, layout_c, MutAnyOrigin],
    mat_a: LayoutTensor[in_type, layout_a, MutAnyOrigin],
    mat_b: LayoutTensor[in_type, layout_b, MutAnyOrigin],
):
    comptime assert (
        in_type == .float8_e4m3fn or in_type == .float8_e5m2
    ), "This kernel only supports E4M3 and E5M2 combinations"

    var mma = TensorCore[
        DType.float32, in_type, IndexList[3](16, 8, 32), False
    ]()
    var a_reg_tile = mma.load_a(mat_a)
    var b_reg_tile = mma.load_b(mat_b)
    var c_reg_tile = mma.load_c(mat_c)
    var d_reg_tile = mma.mma_op(a_reg_tile, b_reg_tile, c_reg_tile)
    mma.store_d(mat_c, d_reg_tile)


# CHECK-LABEL: test_load_and_mma_e4m3_e4m3_f32_16x8x32
# CHECK: 10440.0 10928.0 11385.0 11826.0 12366.0 12858.0 13315.0 13884.0
# CHECK: 10928.0 11464.0 11952.0 12409.0 12978.0 13518.0 14010.0 14595.0
# CHECK: 11385.0 11952.0 12487.0 12974.0 13558.0 14126.0 14665.0 15284.0
# CHECK: 11826.0 12409.0 12974.0 13507.0 14120.0 14702.0 15268.0 15933.0
# CHECK: 12366.0 12978.0 13558.0 14120.0 14794.0 15404.0 15983.0 16690.0
# CHECK: 12858.0 13518.0 14126.0 14702.0 15404.0 16074.0 16680.0 17399.0
# CHECK: 13315.0 14010.0 14665.0 15268.0 15983.0 16680.0 17345.0 18090.0
# CHECK: 13884.0 14595.0 15284.0 15933.0 16690.0 17399.0 18090.0 18909.0
# CHECK: 14420.0 15164.0 15868.0 16550.0 17352.0 18102.0 18804.0 19648.0
# CHECK: 14908.0 15700.0 16436.0 17132.0 17966.0 18760.0 19502.0 20356.0
# CHECK: 15365.0 16188.0 16971.0 17698.0 18545.0 19370.0 20155.0 21048.0
# CHECK: 15806.0 16645.0 17458.0 18231.0 19108.0 19945.0 20760.0 21695.0
# CHECK: 16346.0 17214.0 18042.0 18844.0 19782.0 20648.0 21474.0 22454.0
# CHECK: 16838.0 17754.0 18610.0 19426.0 20392.0 21318.0 22172.0 23162.0
# CHECK: 17295.0 18246.0 19149.0 19992.0 20971.0 21924.0 22837.0 23854.0
# CHECK: 17864.0 18831.0 19768.0 20657.0 21678.0 22643.0 23582.0 24673.0
def test_load_and_mma_e4m3_e4m3_f32_16x8x32(ctx: DeviceContext) raises:
    print("== test_load_and_mma_e4m3_e4m3_f32_16x8x32")
    comptime M = 16
    comptime N = 8
    comptime K = 32
    comptime in_type = DType.float8_e4m3fn
    comptime out_type = DType.float32
    var mat_a = ManagedLayoutTensor[
        in_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(mat_a.tensor())
    var mat_b = ManagedLayoutTensor[
        in_type,
        Layout.row_major(K, N),
    ](ctx)
    arange(mat_b.tensor())

    var mat_c = ManagedLayoutTensor[
        out_type,
        Layout.row_major(M, N),
    ](ctx)
    _ = mat_c.tensor().fill(0)

    comptime load_and_mma_e4m3_e4m3_f32_16x8x32_kernel_fn = load_and_mma_16x8x32[
        out_type,
        in_type,
        mat_c.layout,
        mat_a.layout,
        mat_b.layout,
    ]

    ctx.enqueue_function[load_and_mma_e4m3_e4m3_f32_16x8x32_kernel_fn](
        mat_c.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(32),
    )
    ctx.synchronize()
    print(mat_c.tensor())
    _ = mat_a^
    _ = mat_b^
    _ = mat_c^


# CHECK-LABEL: test_load_and_mma_e5m2_e5m2_f32_16x8x32
# 10500.0 10968.0 11421.0 11866.0 12266.0 12642.0 13239.0 13828.0
# 10968.0 11524.0 11992.0 12445.0 12890.0 13290.0 13922.0 14519.0
# 11421.0 11992.0 12547.0 13014.0 13466.0 13910.0 14565.0 15196.0
# 11866.0 12445.0 13014.0 13567.0 14032.0 14482.0 15180.0 15833.0
# 12266.0 12890.0 13466.0 14032.0 14582.0 15044.0 15747.0 16442.0
# 12642.0 13290.0 13910.0 14482.0 15044.0 15590.0 16304.0 17003.0
# 13239.0 13922.0 14565.0 15180.0 15747.0 16304.0 17165.0 17874.0
# 13828.0 14519.0 15196.0 15833.0 16442.0 17003.0 17874.0 18729.0
# 14368.0 15108.0 15792.0 16462.0 17092.0 17694.0 18568.0 19432.0
# 14852.0 15648.0 16380.0 17056.0 17718.0 18340.0 19254.0 20120.0
# 15320.0 16132.0 16920.0 17644.0 18312.0 18966.0 19900.0 20806.0
# 15750.0 16600.0 17402.0 18180.0 18894.0 19552.0 20516.0 21440.0
# 16148.0 17030.0 17868.0 18658.0 19424.0 20126.0 21092.0 22044.0
# 16778.0 17684.0 18554.0 19380.0 20158.0 20912.0 21986.0 22940.0
# 17392.0 18314.0 19208.0 20066.0 20880.0 21646.0 22772.0 23834.0
# 17964.0 18928.0 19836.0 20716.0 21560.0 22360.0 23496.0 24608.0
def test_load_and_mma_e5m2_e5m2_f32_16x8x32(ctx: DeviceContext) raises:
    print("== test_load_and_mma_e5m2_e5m2_f32_16x8x32")
    comptime M = 16
    comptime N = 8
    comptime K = 32
    comptime in_type = DType.float8_e5m2
    comptime out_type = DType.float32
    var mat_a = ManagedLayoutTensor[
        in_type,
        Layout.row_major(M, K),
    ](ctx)
    arange(mat_a.tensor())
    var mat_b = ManagedLayoutTensor[
        in_type,
        Layout.row_major(K, N),
    ](ctx)
    arange(mat_b.tensor())

    var mat_c = ManagedLayoutTensor[
        out_type,
        Layout.row_major(M, N),
    ](ctx)
    _ = mat_c.tensor().fill(0)

    comptime load_and_mma_e4m3_e4m3_f32_16x8x32_kernel_fn = load_and_mma_16x8x32[
        out_type,
        in_type,
        mat_c.layout,
        mat_a.layout,
        mat_b.layout,
    ]

    ctx.enqueue_function[load_and_mma_e4m3_e4m3_f32_16x8x32_kernel_fn](
        mat_c.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(32),
    )
    ctx.synchronize()
    print(mat_c.tensor())
    _ = mat_a^
    _ = mat_b^
    _ = mat_c^


def main() raises:
    with DeviceContext() as ctx:
        test_load_and_mma_e4m3_e4m3_f32_16x8x32(ctx)
        test_load_and_mma_e5m2_e5m2_f32_16x8x32(ctx)
