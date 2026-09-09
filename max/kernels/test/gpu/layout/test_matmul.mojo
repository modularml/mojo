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

from std.sys import has_nvidia_gpu_accelerator

from std.benchmark import Bench
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Layout, LayoutTensor, RuntimeLayout
from layout._fillers import random
from matmul_kernels import (
    run_cublas,
    run_gemm_kernel_1,
    run_gemm_kernel_2,
    run_gemm_kernel_3,
    run_gemm_kernel_4,
    run_gemm_kernel_5,
    run_gemm_kernel_6,
    run_gemm_kernel_tc,
)
from std.testing import assert_almost_equal
from std.utils import IndexList

comptime run_gemm_kernel_type = def(
    mut m: Bench,
    ctx: DeviceContext,
    a: LayoutTensor,
    b: LayoutTensor,
    c: LayoutTensor,
) thin raises -> None


struct test_matmul[
    dtype: DType,
    a_layout: Layout,
    b_layout: Layout,
    c_layout: Layout,
    enable_tc: Bool,
]:
    var ctx: DeviceContext
    var M: Int
    var N: Int
    var K: Int

    var a_device_buffer: DeviceBuffer[Self.dtype]
    var b_device_buffer: DeviceBuffer[Self.dtype]
    var c_device_buffer: DeviceBuffer[Self.dtype]
    var c_device_buffer_ref: DeviceBuffer[Self.dtype]

    def __init__(out self, mut m: Bench, ctx: DeviceContext) raises:
        self.ctx = ctx
        self.M = Self.a_layout.shape[0].value()
        self.N = Self.b_layout.shape[1].value()
        self.K = Self.b_layout.shape[0].value()
        var a_shape = IndexList[2](self.M, self.K)
        var b_shape = IndexList[2](self.K, self.N)
        var c_shape = IndexList[2](self.M, self.N)
        comptime layout_2d = Layout.row_major[2]()

        self.a_device_buffer = ctx.enqueue_create_buffer[Self.dtype](
            a_shape.flattened_length()
        )
        self.b_device_buffer = ctx.enqueue_create_buffer[Self.dtype](
            b_shape.flattened_length()
        )
        self.c_device_buffer = ctx.enqueue_create_buffer[Self.dtype](
            c_shape.flattened_length()
        )
        self.c_device_buffer_ref = ctx.enqueue_create_buffer[Self.dtype](
            c_shape.flattened_length()
        )

        with self.a_device_buffer.map_to_host() as a_host_buffer:
            random(
                LayoutTensor[Self.dtype, layout_2d](
                    a_host_buffer, RuntimeLayout[layout_2d].row_major(a_shape)
                )
            )
        with self.b_device_buffer.map_to_host() as b_host_buffer:
            random(
                LayoutTensor[Self.dtype, layout_2d](
                    b_host_buffer, RuntimeLayout[layout_2d].row_major(b_shape)
                )
            )
        with self.c_device_buffer.map_to_host() as c_host_buffer:
            _ = LayoutTensor[Self.dtype, layout_2d](
                c_host_buffer, RuntimeLayout[layout_2d].row_major(c_shape)
            ).fill(0)
        with self.c_device_buffer_ref.map_to_host() as c_host_ref_buffer:
            _ = LayoutTensor[Self.dtype, layout_2d](
                c_host_ref_buffer, RuntimeLayout[layout_2d].row_major(c_shape)
            ).fill(0)

        run_cublas[Self.dtype, Self.enable_tc](
            m,
            ctx,
            self.M,
            self.N,
            self.K,
            self.a_device_buffer.unsafe_ptr(),
            self.b_device_buffer.unsafe_ptr(),
            self.c_device_buffer_ref.unsafe_ptr(),
        )

    def run_test[gemm: run_gemm_kernel_type](mut self, mut m: Bench) raises:
        print("=== test_matmul")

        var ctx = self.ctx

        def create_tensor[
            layout: Layout
        ](
            m: Int,
            n: Int,
            ptr: Pointer[Scalar[Self.dtype], _],
            out result: LayoutTensor[Self.dtype, layout, ptr.origin],
        ):
            var dynamic_layout = type_of(result.runtime_layout)(
                type_of(result.runtime_layout.shape)(m, n),
                type_of(result.runtime_layout.stride)(n, 1),
            )
            return {ptr, dynamic_layout}

        var a = create_tensor[Self.a_layout](
            self.M, self.K, self.a_device_buffer.unsafe_ptr()
        )
        var b = create_tensor[Self.b_layout](
            self.K, self.N, self.b_device_buffer.unsafe_ptr()
        )
        var c = create_tensor[Self.c_layout](
            self.M, self.N, self.c_device_buffer.unsafe_ptr()
        )

        gemm(m, ctx, a, b, c)

        with self.c_device_buffer_ref.map_to_host() as c_host_ref:
            with self.c_device_buffer.map_to_host() as c_host:
                for i in range(len(c_host_ref)):
                    assert_almost_equal(
                        c_host_ref[i],
                        c_host[i],
                        atol=0.0001,
                        rtol=0.01,
                        msg=String(t"not equal at index: {i}"),
                    )


def main() raises:
    comptime N = 4096
    comptime M = N
    comptime K = M

    var m = Bench()
    with DeviceContext() as ctx:
        comptime a_layout = Layout.row_major(M, K)
        comptime b_layout = Layout.row_major(K, N)
        comptime c_layout = Layout.row_major(M, N)

        var test = test_matmul[
            DType.float32, a_layout, b_layout, c_layout, False
        ](m, ctx)

        var test_tc = test_matmul[
            DType.float32, a_layout, b_layout, c_layout, True
        ](m, ctx)

        comptime k1 = run_gemm_kernel_1[
            DType.float32, a_layout, b_layout, c_layout, 32, 32
        ]

        comptime k2 = run_gemm_kernel_2[
            DType.float32, a_layout, b_layout, c_layout, 32, 32
        ]

        comptime k3 = run_gemm_kernel_3[
            DType.float32, a_layout, b_layout, c_layout, 32, 32, 32
        ]

        comptime k4 = run_gemm_kernel_4[
            DType.float32, a_layout, b_layout, c_layout, 64, 64, 8, 8
        ]

        comptime k5 = run_gemm_kernel_5[
            DType.float32, a_layout, b_layout, c_layout, 128, 128, 8, 8, 8
        ]

        comptime k6 = run_gemm_kernel_6[
            DType.float32, a_layout, b_layout, c_layout, 128, 128, 8, 8, 8
        ]

        comptime MMA_M = 16
        comptime MMA_N = 8 if has_nvidia_gpu_accelerator() else 16
        comptime MMA_K = 8 if has_nvidia_gpu_accelerator() else 4

        comptime k_tc = run_gemm_kernel_tc[
            DType.float32,
            a_layout,
            b_layout,
            c_layout,
            64,  # BM: The block size in the M dimension
            64,  # BN: The block size in the N dimension
            32,  # BK: The block size in the K dimension
            32,  # WM: The warp tile size in the M dimension
            32,  # WN: The warp tile size in the N dimension
            MMA_M,  # MMA_M: Tensor core instruction shape in M dimension
            MMA_N,  # MMA_N: Tensor core instruction shape in N dimension
            MMA_K,  # MMA_K: Tensor core instruction shape in K dimension
        ]

        test.run_test[k1](m)
        test.run_test[k2](m)
        test.run_test[k3](m)
        test.run_test[k4](m)
        test.run_test[k5](m)
        test.run_test[k6](m)
        test_tc.run_test[k_tc](m)

    m.dump_report()
