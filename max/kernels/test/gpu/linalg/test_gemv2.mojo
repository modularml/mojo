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

from std.random import random_si64
from std.math import ceildiv

from max.gpu.host import DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.matmul.gpu import _matmul_gpu, matmul_kernel_naive
from std.utils import IndexList

comptime epilogue_func_type = def[
    type: DType, width: SIMDLength, *, alignment: Int = 1
](IndexList[2], IndexList[2], SIMD[type, width]) capturing -> SIMD[type, width]


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


def test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
    M: Optional[Int],
    N: Optional[Int],
    K: Optional[Int],
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k, "transpose_b", transpose_b)

    var a_size = m * k
    var b_size = k * n
    var c_size = m * n

    # Host buffers
    var a_host = ctx.enqueue_create_host_buffer[in_type](a_size)
    var b_host = ctx.enqueue_create_host_buffer[in_type](b_size)
    var c_host = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_host_ref = ctx.enqueue_create_host_buffer[out_type](c_size)

    comptime rand_min = -100
    comptime rand_max = 100

    for i in range(a_size):
        a_host[i] = random_si64(rand_min, rand_max).cast[in_type]()

    for i in range(b_size):
        b_host[i] = random_si64(rand_min, rand_max).cast[in_type]()

    for i in range(c_size):
        c_host[i] = 0
        c_host_ref[i] = 0

    # Device buffers
    var a_dev = ctx.enqueue_create_buffer[in_type](a_size)
    var b_dev = ctx.enqueue_create_buffer[in_type](b_size)
    var c_dev = ctx.enqueue_create_buffer[out_type](c_size)
    var c_ref_dev = ctx.enqueue_create_buffer[out_type](c_size)

    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(c_dev, c_host)
    ctx.enqueue_copy(c_ref_dev, c_host_ref)

    # TileTensors for _matmul_gpu
    comptime static_b_dim0 = N if transpose_b else K
    comptime static_b_dim1 = K if transpose_b else N
    var a_tensor = TileTensor(
        a_dev,
        row_major(Coord(m, Idx[K.value()])),
    )
    var b_tensor = TileTensor(
        b_dev,
        row_major(
            Coord(
                Idx[static_b_dim0.value()],
                Idx[static_b_dim1.value()],
            )
        ),
    )
    var c_tensor = TileTensor(
        c_dev,
        row_major(Coord(m, Idx[N.value()])),
    )

    _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
        c_tensor,
        a_tensor.as_immut(),
        b_tensor.as_immut(),
        ctx,
    )

    ctx.enqueue_copy(c_host, c_dev)

    var c_tt = TileTensor(
        c_ref_dev,
        row_major(Coord(m, n)),
    )
    var a_tt = TileTensor[mut=False](a_dev, row_major(Coord(m, k)))
    var b_shape_0 = n if transpose_b else k
    var b_shape_1 = k if transpose_b else n
    var b_tt = TileTensor[mut=False](
        b_dev, row_major(Coord(b_shape_0, b_shape_1))
    )

    comptime BLOCK_DIM = 16
    comptime naive_kernel = matmul_kernel_naive[
        out_type,
        in_type,
        in_type,
        type_of(c_tt).LayoutType,
        type_of(a_tt).LayoutType,
        type_of(b_tt).LayoutType,
        BLOCK_DIM,
        transpose_b=transpose_b,
    ]
    ctx.enqueue_function[naive_kernel](
        c_tt,
        a_tt,
        b_tt,
        Int32(m),
        Int32(n),
        Int32(k),
        grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )
    ctx.synchronize()

    ctx.enqueue_copy(c_host_ref, c_ref_dev)
    ctx.synchronize()

    var errors = 0
    for i in range(c_size):
        if c_host[i] != c_host_ref[i]:
            errors += 1

    print("errors", errors)
    if errors > 0:
        raise "GEMV failed: exact mismatch"


def test_gemv_split_k[in_type: DType](ctx: DeviceContext) raises:
    """Test GEMV_SPLIT_K path: M=1, transpose_b=True, K % simd_width == 0."""
    print("=== Testing GEMV_SPLIT_K with", in_type, "===")

    comptime out_type = DType.float32 if in_type == DType.bfloat16 else DType.bfloat16

    # Basic GEMV_SPLIT_K
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(4096),
        K=Int(4096),
    ](ctx, 1, 4096, 4096)

    # N % TILE_N != 0 (bounds checking)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(75837),
        K=Int(5120),
    ](ctx, 1, 75837, 5120)

    # Large K (tests 512-thread / 256-thread dispatch + unroll)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(16384),
    ](ctx, 1, 16384, 16384)

    # Small K (tests 64-thread dispatch)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(1024),
    ](ctx, 1, 16384, 1024)

    # Mid K (tests 128-thread dispatch)
    test[
        in_type=in_type,
        out_type=out_type,
        transpose_b=True,
        M=None,
        N=Int(16384),
        K=Int(2048),
    ](ctx, 1, 16384, 2048)


def main() raises:
    with DeviceContext() as ctx:
        # GEMV_SPLIT_K tests for bf16 and fp8
        test_gemv_split_k[.bfloat16](ctx)
        test_gemv_split_k[.float8_e4m3fn](ctx)

        # BF16-only tests for other GEMV paths (FP8 not supported here)

        # GEMV_KERNEL_VECTOR: M = 1, N wider than the split-K grid limit and
        # K shallow enough that the kernel packs several rows per warp.
        test[
            in_type=.bfloat16,
            out_type=.bfloat16,
            transpose_b=True,
            M=None,
            N=Int(262144),
            K=Int(256),
        ](ctx, 1, 262144, 256)

        # GEMV_KERNEL_VECTOR: N = 1, K % simd_width == 0, transpose_b = False
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=False,
            M=None,
            N=Int(1),
            K=Int(4096),
        ](ctx, 4096, 1, 4096)

        # GEMV_KERNEL_VECTOR: N = 1, K % simd_width == 0, transpose_b = True
        test[
            in_type=.bfloat16,
            out_type=.bfloat16,
            transpose_b=True,
            M=None,
            N=Int(1),
            K=Int(13824),
        ](ctx, 5120, 1, 13824)

        # GEMV_KERNEL: M = 1, K % simd_width !=0, transpose_b = True
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=True,
            M=None,
            N=Int(4096),
            K=Int(4095),
        ](ctx, 1, 4096, 4095)

        # GEMV_KERNEL: N = 1, K % simd_width !=0, transpose_b = False
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=False,
            M=None,
            N=Int(1),
            K=Int(4095),
        ](ctx, 4096, 1, 4095)

        # matmul_naive: M = 1, K % WARP_SIZE != 0, transpose_b = False
        test[
            in_type=.bfloat16,
            out_type=.float32,
            transpose_b=False,
            M=None,
            N=Int(4096),
            K=Int(4095),
        ](ctx, 1, 4096, 4095)
