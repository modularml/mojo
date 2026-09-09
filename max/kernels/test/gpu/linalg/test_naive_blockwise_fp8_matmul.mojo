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

from std.math import ceildiv

from max.gpu.host import DeviceContext
from internal_utils import assert_almost_equal
from std.random import rand
from layout import CoordLike, Coord, Idx, TileTensor, row_major
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul

from std.utils.index import Index, IndexList


def test_naive_blockwise_fp8_matmul[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    input_type: DType,
    block_scales_sizes: IndexList[3],
    transpose_b: Bool = True,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime BLOCK_SCALE_M = block_scales_sizes[0]
    comptime BLOCK_SCALE_N = block_scales_sizes[1]
    comptime BLOCK_SCALE_K = block_scales_sizes[2]

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        "== test_naive_blockwise_fp8_matmul",
        input_type,
        "x",
        M,
        "x",
        N,
        "x",
        K,
        "BLOCK_SCALE_M",
        BLOCK_SCALE_M,
        "BLOCK_SCALE_N",
        BLOCK_SCALE_N,
        "BLOCK_SCALE_K",
        BLOCK_SCALE_K,
        "transpose_b",
        transpose_b,
    )

    var a_shape = Coord(m, k)
    var b_shape = Coord(
        Idx[NType.static_value if transpose_b else KType.static_value],
        Idx[KType.static_value if transpose_b else NType.static_value],
    )
    var c_shape = Coord(m, n)

    var a_scale_shape = Coord(
        ceildiv(K, BLOCK_SCALE_K), ceildiv(M, BLOCK_SCALE_M)
    )
    var b_scale_shape = Coord(
        ceildiv(N, BLOCK_SCALE_N),
        ceildiv(K, BLOCK_SCALE_K),
    ) if transpose_b else Coord(
        ceildiv(K, BLOCK_SCALE_K),
        ceildiv(N, BLOCK_SCALE_N),
    )

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N
    var a_scale_size = ceildiv(K, BLOCK_SCALE_K) * ceildiv(M, BLOCK_SCALE_M)
    var b_scale_size = ceildiv(N, BLOCK_SCALE_N) * ceildiv(K, BLOCK_SCALE_K)

    var a_host_ptr = ctx.enqueue_create_host_buffer[input_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[input_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[.float32](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[.float32](c_size)

    rand(a_host_ptr.unsafe_ptr(), a_size)
    rand(b_host_ptr.unsafe_ptr(), b_size)

    for i in range(c_size):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0

    var a_scale_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        a_scale_size
    )
    var b_scale_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        b_scale_size
    )

    rand(a_scale_host_ptr.unsafe_ptr(), a_scale_size)
    rand(b_scale_host_ptr.unsafe_ptr(), b_scale_size)

    var a_host = TileTensor(a_host_ptr, row_major(a_shape))
    var b_host = TileTensor(b_host_ptr, row_major(b_shape))
    var a_scale_host = TileTensor(a_scale_host_ptr, row_major(a_scale_shape))
    var b_scale_host = TileTensor(b_scale_host_ptr, row_major(b_scale_shape))
    var c_host_ref = TileTensor(c_host_ref_ptr, row_major(c_shape))
    comptime assert c_host_ref.flat_rank == 2
    comptime assert a_host.flat_rank == 2

    # run blockwise CPU as the reference output
    for _m in range(M):
        for _n in range(N):
            var res: Float32 = 0.0
            for _k in range(K):
                var a_scale = a_scale_host[
                    Coord(_k // BLOCK_SCALE_K, _m // BLOCK_SCALE_M)
                ]
                var b_scale = b_scale_host[
                    Coord(_n // BLOCK_SCALE_N, _k // BLOCK_SCALE_K)
                ] if transpose_b else b_scale_host[
                    Coord(_k // BLOCK_SCALE_K, _n // BLOCK_SCALE_N)
                ]
                var b_elem = b_host[Coord(_n, _k)] if transpose_b else b_host[
                    Coord(_k, _n)
                ]
                res += (
                    a_host[_m, _k].cast[.float32]()
                    * b_elem.cast[.float32]()
                    * a_scale
                    * b_scale
                )

            c_host_ref[_m, _n] = res

    var a_device = ctx.enqueue_create_buffer[input_type](a_size)
    var b_device = ctx.enqueue_create_buffer[input_type](b_size)
    var c_device = ctx.enqueue_create_buffer[.float32](c_size)
    var a_scale_device = ctx.enqueue_create_buffer[.float32](a_scale_size)
    var b_scale_device = ctx.enqueue_create_buffer[.float32](b_scale_size)

    ctx.enqueue_copy(a_scale_device, a_scale_host_ptr)
    ctx.enqueue_copy(b_scale_device, b_scale_host_ptr)

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    var c_dev = TileTensor(c_device, row_major(c_shape))
    var a_dev = TileTensor(a_device, row_major(a_shape))
    var b_dev = TileTensor(b_device, row_major(b_shape))
    var a_scale_dev = TileTensor(a_scale_device, row_major(a_scale_shape))
    var b_scale_dev = TileTensor(b_scale_device, row_major(b_scale_shape))

    if (
        M % BLOCK_SCALE_M != 0
        or N % BLOCK_SCALE_N != 0
        or K % BLOCK_SCALE_K != 0
    ):
        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
            scales_granularity_mnk=Index(
                BLOCK_SCALE_M, BLOCK_SCALE_N, BLOCK_SCALE_K
            ),
        ](
            c_dev.to_layout_tensor(),
            a_dev.to_layout_tensor(),
            b_dev.to_layout_tensor(),
            a_scale_dev.to_layout_tensor(),
            b_scale_dev.to_layout_tensor(),
            ctx,
        )
    else:
        naive_blockwise_scaled_fp8_matmul[
            BLOCK_DIM=16,
            transpose_b=transpose_b,
        ](
            c_dev.to_layout_tensor(),
            a_dev.to_layout_tensor(),
            b_dev.to_layout_tensor(),
            a_scale_dev.to_layout_tensor(),
            b_scale_dev.to_layout_tensor(),
            ctx,
        )

    ctx.enqueue_copy(c_host_ptr, c_device)

    ctx.synchronize()

    assert_almost_equal(
        c_host_ptr.unsafe_ptr(),
        c_host_ref_ptr.unsafe_ptr(),
        c_size,
        atol=0.0001,
        rtol=0.0001,
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime for transpose_b in range(0, 2):
            test_naive_blockwise_fp8_matmul[
                .float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b=Bool(transpose_b),
            ](ctx, Idx[128], Idx[128], Idx[128])

            test_naive_blockwise_fp8_matmul[
                .float8_e4m3fn,
                Index(1, 64, 128),
                transpose_b=Bool(transpose_b),
            ](ctx, Idx[128], Idx[256], Idx[128])

            test_naive_blockwise_fp8_matmul[
                .float8_e4m3fn,
                Index(1, 64, 16),
                transpose_b=Bool(transpose_b),
            ](ctx, Idx[128], Idx[128], Idx[128])

            test_naive_blockwise_fp8_matmul[
                .float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b=Bool(transpose_b),
            ](ctx, Idx[120], Idx[128], Idx[128])

            test_naive_blockwise_fp8_matmul[
                .float8_e4m3fn,
                Index(1, 128, 128),
                transpose_b=Bool(transpose_b),
            ](ctx, Idx[120], Idx[129], Idx[128])

            test_naive_blockwise_fp8_matmul[
                .float8_e4m3fn,
                Index(32, 128, 64),
                transpose_b=Bool(transpose_b),
            ](ctx, Idx[120], Idx[129], Idx[129])
