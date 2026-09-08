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
# mojo build --debug-level=full --mcmodel=medium --large-data-threshold=1048576
# to build this file if running into linking issues with large PTX kernels.

from std.sys import (
    align_of,
    bit_width_of,
    has_nvidia_gpu_accelerator,
    simd_width_of,
    size_of,
)

import linalg.matmul.vendor.blas as vendor_blas
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext, get_gpu_target
from layout import Coord, Idx, TileTensor, row_major, coord_to_index_list
from layout._fillers import arange as arange, random
from linalg.matmul.gpu import _matmul_gpu, multistage_gemm
from linalg.utils_gpu import MatmulConfig
from test_utils import ulp_distance
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.index import Index


comptime epilogue_func_type = def[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](IndexList[2], IndexList[2], SIMD[dtype, width]) capturing -> SIMD[
    dtype, width
]


@__parameter
@always_inline
def epilogue_test_fn[
    dtype: DType, width: SIMDLength, *, alignment: Int = 1
](
    idx: IndexList[2],
    dim_space: IndexList[2],
    val: SIMD[dtype, width],
) -> SIMD[
    dtype, width
]:
    var bias = SIMD[dtype, width](0)

    comptime for i in range(width):
        bias[i] = (
            0.5
            + Float64(idx[0] + idx[1] + i)
            / Float64(dim_space[0] + dim_space[1])
        ).cast[dtype]()

    return val + bias


def select_max_ulp_distance[
    lambda_fn: Optional[epilogue_func_type]
](max_ulp_distance: Optional[Int]) -> Int:
    if max_ulp_distance:
        return max_ulp_distance.value()
    else:
        comptime if lambda_fn:
            return 4
        return 2


def test[
    dtype: DType,
    /,
    *,
    transpose_b: Bool = False,
    arange_a: Bool = False,
    arange_b: Bool = False,
    lambda_fn: Optional[epilogue_func_type] = None,
    config: Optional[MatmulConfig[dtype, dtype, dtype, transpose_b]] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](
    ctx: DeviceContext,
    m: Int,
    n: Int,
    k: Int,
    rtol: Float64 = 1e-3 if dtype == .float32 else 1e-2,
    max_ulp_distance: Optional[Int] = None,
) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k)

    var a_size = m * k
    var b_size = n * k if transpose_b else k * n
    var c_size = m * n

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(m, Idx[K.value()])),
    )
    var b_host = TileTensor(
        b_host_ptr,
        row_major(
            Coord(
                Idx[N.value() if transpose_b else K.value()],
                Idx[K.value() if transpose_b else N.value()],
            )
        ),
    )
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(m, Idx[N.value()])),
    )
    var c_host_ref = TileTensor(
        c_host_ref_ptr,
        row_major(Coord(m, Idx[N.value()])),
    )

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[dtype](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[dtype](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[dtype](c_size)

    var a_device = TileTensor(
        a_device_buffer,
        row_major(Coord(m, Idx[K.value()])),
    )
    var b_device = TileTensor(
        b_device_buffer,
        row_major(
            Coord(
                Idx[N.value() if transpose_b else K.value()],
                Idx[K.value() if transpose_b else N.value()],
            )
        ),
    )
    var c_device = TileTensor(
        c_device_buffer,
        row_major(Coord(m, Idx[N.value()])),
    )
    var c_device_ref = TileTensor(
        c_device_ref_buffer,
        row_major(Coord(m, Idx[N.value()])),
    )

    # Initialize matmul operands
    comptime if arange_a:
        arange(a_host)
    else:
        random(a_host)

    comptime if arange_b:
        arange(b_host)
    else:
        random(b_host)

    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    # Move operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)

    @__parameter
    @always_inline
    @__copy_capture(c_device, m, n)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var update_val: SIMD[_dtype, width] = val

        comptime if lambda_fn:
            comptime func = lambda_fn.value()
            update_val = func(idx, (m, n), update_val)
        c_device.store_linear[alignment=alignment * size_of[dtype]()](
            idx, rebind[SIMD[dtype, width]](update_val)
        )

    comptime if lambda_fn:
        comptime if config:
            multistage_gemm[
                transpose_b=transpose_b,
                config=config.value(),
                elementwise_lambda_fn=epilogue_fn,
            ](
                c_device,
                a_device,
                b_device,
                ctx,
            )
        else:
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
                elementwise_lambda_fn=epilogue_fn,
            ](
                c_device,
                a_device,
                b_device,
                ctx,
            )
    else:
        comptime if config:
            multistage_gemm[
                transpose_b=transpose_b,
                config=config.value(),
            ](
                c_device,
                a_device,
                b_device,
                ctx,
            )
        else:
            _matmul_gpu[
                use_tensor_core=True,
                transpose_b=transpose_b,
            ](
                c_device,
                a_device,
                b_device,
                ctx,
            )

    ctx.synchronize()

    vendor_blas.matmul(
        ctx,
        c_device_ref,
        a_device,
        b_device,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    comptime pack_size = simd_width_of[dtype, target=get_gpu_target()]()

    @always_inline
    def func[simd_width: Int, alignment: Int = 1](idx0: Coord) {var}:
        var val = c_device_ref.load[width=simd_width](idx0)

        var update_val = val

        comptime if lambda_fn:
            comptime element_lambda = lambda_fn.value()
            update_val = element_lambda(
                rebind[IndexList[2]](coord_to_index_list(idx0)), (m, n), val
            )

        c_device_ref.store[alignment=alignment * size_of[dtype]()](
            idx0, update_val
        )

    comptime if lambda_fn:
        elementwise[pack_size, target="gpu"](
            func,
            (m, n),
            ctx,
        )
    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var _max_ulp_distance = select_max_ulp_distance[lambda_fn](max_ulp_distance)
    for mi in range(m):
        for ni in range(n):
            var expect = c_host_ref[mi, ni][0]
            var actual = c_host[mi, ni][0]

            comptime if bit_width_of[dtype]() <= 16:
                var ulp_dist = ulp_distance(actual, expect)
                if ulp_dist <= _max_ulp_distance:
                    continue

            assert_almost_equal(actual, expect, rtol=rtol)

    # Cleanup
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = c_device_ref_buffer^


def main() raises:
    with DeviceContext() as ctx:
        print("===> tfloat32-float32 mma")
        test[
            .float32,
            arange_a=True,
            arange_b=True,
            N=Int(12288),
            K=Int(4096),
        ](ctx, 512, 12288, 4096)
        test[.float32, arange_a=True, N=Int(384), K=Int(128)](
            ctx, 256, 384, 128
        )
        test[.float32, arange_b=True, N=Int(4096), K=Int(4096)](
            ctx, 128, 4096, 4096
        )
        test[
            .float32,
            arange_a=True,
            arange_b=True,
            N=Int(12288),
            K=Int(4096),
        ](ctx, 512, 12288, 4096)
        test[.float32, N=Int(4096), K=Int(11008)](ctx, 23, 4096, 11008)
        test[.float32, N=Int(4096), K=Int(12288)](ctx, 67, 4096, 12288)
        test[.float32, N=Int(4096), K=Int(4096)](ctx, 555, 4096, 4096)

        print("===> bfloat16-float32 mma")
        test[
            .bfloat16,
            arange_a=True,
            transpose_b=True,
            config=MatmulConfig[
                .bfloat16,
                .bfloat16,
                .bfloat16,
                transpose_b=True,
            ](
                block_tile_shape=Index(64, 128, 64),
                warp_tile_shape=Index(16, 128, 64),
                num_pipeline_stages=3,
            ),
            N=Int(128),
            K=Int(128),
        ](ctx, 100, 128, 128)
        test[.bfloat16, arange_b=True, N=Int(12288), K=Int(3072)](
            ctx, 1024, 12288, 3072
        )
        test[
            .bfloat16,
            arange_a=True,
            arange_b=True,
            N=Int(5120),
            K=Int(3072),
        ](ctx, 1024, 5120, 3072)
        test[.bfloat16, N=Int(3072), K=Int(32768)](ctx, 1024, 3072, 32768)
        test[.bfloat16, N=Int(3072), K=Int(3072)](ctx, 1024, 3072, 3072)

        comptime if has_nvidia_gpu_accelerator():
            test[
                .bfloat16,
                transpose_b=True,
                config=MatmulConfig[
                    .bfloat16,
                    .bfloat16,
                    .bfloat16,
                    transpose_b=True,
                ](
                    block_tile_shape=Index(16, 64, 64),
                    warp_tile_shape=Index(16, 64, 32),
                    num_pipeline_stages=3,
                    num_k_partitions=1,
                    num_warp_k_partitions=2,
                ),
                N=Int(4096),
                K=Int(4096),
            ](ctx, 32, 4096, 4096)
            test[
                .bfloat16,
                transpose_b=True,
                config=MatmulConfig[
                    .bfloat16,
                    .bfloat16,
                    .bfloat16,
                    transpose_b=True,
                ](
                    block_tile_shape=Index(32, 64, 32),
                    warp_tile_shape=Index(16, 64, 32),
                    num_pipeline_stages=3,
                    num_k_partitions=1,
                    num_warp_k_partitions=4,
                ),
                N=Int(4096),
                K=Int(4096),
            ](ctx, 32, 4096, 4096)

        print("===> tfloat32-float32 mma with epilogue")
        test[
            .float32,
            lambda_fn=epilogue_test_fn,
            N=Int(3072),
            K=Int(3072),
        ](ctx, 999, 3072, 3072)
        test[
            .float32,
            lambda_fn=epilogue_test_fn,
            N=Int(12288),
            K=Int(2048),
        ](ctx, 777, 12288, 2048)

        print("===> bfloat16-float32 mma with epilogue")
        # Our default split-k reduction precision is output precision. For
        # bfloat16, we need a larger tolerance since the reference may reduce
        # in float32.
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(3072),
            K=Int(12288),
        ](ctx, 14, 3072, 12288, rtol=2e-2)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(12288),
            K=Int(3072),
        ](ctx, 33, 12288, 3072)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(5120),
            K=Int(3072),
        ](ctx, 101, 5120, 3072)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(3072),
            K=Int(32768),
        ](ctx, 400, 3072, 32768, rtol=2e-2)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(3072),
            K=Int(3072),
        ](ctx, 910, 3072, 3072)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(6144),
            K=Int(4096),
        ](ctx, 50, 6144, 4096)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(4096),
            K=Int(4096),
        ](ctx, 22, 4096, 4096)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(28672),
            K=Int(4096),
        ](ctx, 88, 28672, 4096)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(4096),
            K=Int(14336),
        ](ctx, 100, 4096, 14336)
        test[
            .bfloat16,
            transpose_b=True,
            lambda_fn=epilogue_test_fn,
            N=Int(128256),
            K=Int(4096),
        ](ctx, 600, 128256, 4096)

        # Unaligned N: N * size_of(bfloat16) % 16 != 0, so a tile GEMM's TMA
        # output descriptor is misaligned and the dispatcher has to route
        # around it. At m <= 64 the split-K GEMV takes these, odd N included.
        # The three m values land in its three tile_m bands.
        print("===> bfloat16 unaligned N")
        test[.bfloat16, transpose_b=True, N=Int(258), K=Int(4096)](
            ctx, 6, 258, 4096
        )
        test[.bfloat16, transpose_b=True, N=Int(258), K=Int(4096)](
            ctx, 12, 258, 4096
        )
        test[.bfloat16, transpose_b=True, N=Int(258), K=Int(4096)](
            ctx, 64, 258, 4096
        )
        test[.bfloat16, transpose_b=True, N=Int(257), K=Int(4096)](
            ctx, 6, 257, 4096
        )
        test[.bfloat16, transpose_b=True, N=Int(514), K=Int(4096)](
            ctx, 32, 514, 4096
        )
