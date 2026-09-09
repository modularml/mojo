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

from std.math import ceildiv, isclose
from std.os import abort
from std.random import random_float64
from std.sys import has_amd_gpu_accelerator, has_nvidia_gpu_accelerator

from max.gpu import WARP_SIZE, global_idx
from max.gpu.host import DeviceContext
from layout import *
from layout._utils import ManagedLayoutTensor
from layout.tensor_core import *
from std.testing import *


def mma_layout_tc[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: LayoutTensor[out_type, layout_c, MutAnyOrigin],
    mat_a: LayoutTensor[in_type, layout_a, MutAnyOrigin],
    mat_b: LayoutTensor[in_type, layout_b, MutAnyOrigin],
):
    var tc = TensorCore[out_type, in_type, shape]()
    # alias shapes = TensorCore().get_shapes[out_type, in_type]()
    var a = tc.load_a(mat_a)
    var b = tc.load_b(mat_b)
    var c = tc.load_c(mat_c)
    var d = tc.mma_op(a, b, c)
    tc.store_d(mat_c, d)


def matmul_naive[
    out_type: DType,
    in_type: DType,
    layout_c: Layout,
    layout_a: Layout,
    layout_b: Layout,
](
    mat_c: LayoutTensor[out_type, layout_c, MutAnyOrigin],
    mat_a: LayoutTensor[in_type, layout_a, MutAnyOrigin],
    mat_b: LayoutTensor[in_type, layout_b, MutAnyOrigin],
):
    var x = global_idx.x
    var y = global_idx.y

    if x >= mat_c.shape[0]() or y >= mat_c.shape[1]():
        return

    var accum = mat_c[x, y]
    for i in range(mat_a.shape[1]()):
        accum += mat_a[x, i].cast[out_type]() * mat_b[i, y].cast[out_type]()
    mat_c[x, y] = accum


def test_layout_mma[
    out_type: DType,
    in_type: DType,
    shape: IndexList[3],
    M: Int,
    N: Int,
    K: Int,
](
    ctx: DeviceContext,
    rtol: Float64 = 1e-05,
    rng_width: Float64 = 10.0,
    debug: Bool = False,
) raises:
    print("== run layout mma => ", String(out_type), String(in_type), M, N, K)

    comptime layout_a = Layout(IntTuple(M, K), IntTuple(K, 1))
    comptime layout_b = Layout(IntTuple(K, N), IntTuple(N, 1))
    comptime layout_c = Layout(IntTuple(M, N), IntTuple(N, 1))

    var mat_a = ManagedLayoutTensor[in_type, layout_a](ctx)
    var mat_b = ManagedLayoutTensor[in_type, layout_b](ctx)
    var mat_c = ManagedLayoutTensor[out_type, layout_c](ctx)
    var mat_a_n = ManagedLayoutTensor[in_type, layout_a](ctx)
    var mat_b_n = ManagedLayoutTensor[in_type, layout_b](ctx)
    var mat_c_n = ManagedLayoutTensor[out_type, layout_c](ctx)

    var rand_min = -1 * rng_width
    var rand_max = rng_width
    var mat_a_tensor = mat_a.tensor()
    var mat_b_tensor = mat_b.tensor()
    var mat_c_tensor = mat_c.tensor()

    var mat_a_n_tensor = mat_a_n.tensor()
    var mat_b_n_tensor = mat_b_n.tensor()
    var mat_c_n_tensor = mat_c_n.tensor()

    for i in range(M):
        for j in range(K):
            var val = random_float64(rand_min, rand_max).cast[.float32]()
            mat_a_tensor[i, j] = val.cast[in_type]()
            mat_a_n_tensor[i, j] = mat_a_tensor[i, j]
    for i in range(K):
        for j in range(N):
            var val = random_float64(rand_min, rand_max).cast[.float32]()
            mat_b_tensor[i, j] = val.cast[in_type]()
            mat_b_n_tensor[i, j] = mat_b_tensor[i, j]
    for i in range(M):
        for j in range(N):
            var val = Float32(0)
            mat_c_tensor[i, j] = val.cast[out_type]()
            mat_c_n_tensor[i, j] = mat_c_tensor[i, j]

    comptime kernel = mma_layout_tc[
        out_type, in_type, shape, layout_c, layout_a, layout_b
    ]
    ctx.enqueue_function[kernel](
        mat_c.device_tensor(),
        mat_a.device_tensor(),
        mat_b.device_tensor(),
        grid_dim=(1, 1),
        block_dim=(WARP_SIZE),
    )

    ctx.synchronize()

    comptime warps_per_block = 16
    comptime naive_func = matmul_naive[
        out_type, in_type, layout_c, layout_a, layout_b
    ]
    ctx.enqueue_function[kernel](
        mat_c_n.device_tensor(),
        mat_a_n.device_tensor(),
        mat_b_n.device_tensor(),
        grid_dim=(ceildiv(M, warps_per_block), ceildiv(N, warps_per_block)),
        block_dim=(warps_per_block, warps_per_block),
    )

    ctx.synchronize()

    for i in range(M):
        for j in range(N):
            var out_val = mat_c.tensor()[i, j]
            var out_ref = mat_c_n.tensor()[i, j]
            if debug:
                if not isclose(out_val, out_ref, rtol=rtol):
                    print(i, out_val, out_ref)
            assert_true(isclose(out_val, out_ref, rtol=rtol))


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator():
            comptime shape_884 = IndexList[3](8, 8, 4)
            comptime shape_1684 = IndexList[3](16, 8, 4)
            comptime shape_1688 = IndexList[3](16, 8, 8)
            comptime shape_16816 = IndexList[3](16, 8, 16)

            test_layout_mma[.float64, DType.float64, shape_884, 8, 8, 4](
                ctx, rtol=1e-01
            )

            test_layout_mma[.float64, DType.float64, shape_1684, 16, 8, 4](
                ctx, rtol=1e-01
            )

            test_layout_mma[.float64, DType.float64, shape_1688, 16, 8, 8](
                ctx, rtol=1e-01
            )

            test_layout_mma[
                DType.float64, DType.float64, shape_16816, 16, 8, 16
            ](ctx, rtol=1e-01)

            test_layout_mma[.float32, DType.float32, shape_1684, 16, 8, 4](
                ctx, rtol=1e-01
            )
            test_layout_mma[.float32, DType.float32, shape_1688, 16, 8, 8](
                ctx, rtol=1e-01
            )
            test_layout_mma[
                DType.float32, DType.bfloat16, shape_1688, 16, 8, 8
            ](ctx, rtol=1e-01)
            test_layout_mma[.float32, DType.float16, shape_1688, 16, 8, 8](
                ctx, rtol=1e-01
            )
        elif has_amd_gpu_accelerator():
            comptime shape_161616 = IndexList[3](16, 16, 16)
            comptime shape_16164 = IndexList[3](16, 16, 4)

            test_layout_mma[
                DType.float32, DType.float16, shape_161616, 16, 16, 16
            ](ctx, rtol=1e-01)
            test_layout_mma[
                DType.float32, DType.bfloat16, shape_161616, 16, 16, 16
            ](ctx, rtol=1e-01)
            test_layout_mma[
                DType.float32, DType.float32, shape_16164, 16, 16, 4
            ](ctx, rtol=1e-01)
        else:
            abort("Unknown GPU Accelerator.")
