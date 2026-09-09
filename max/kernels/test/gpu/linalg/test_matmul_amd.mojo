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

from std.random import rand, random_si64
from std.math import ceildiv

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceBuffer, DeviceContext
from max.gpu.host.info import MI355X
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from linalg.matmul.gpu import (
    _amdgpu_get_mma_shape,
    _amdgpu_matmul_config_from_block_shape,
    _matmul_gpu,
    matmul_kernel_naive,
    multistage_gemm,
)
from linalg.utils_gpu import MatmulConfig
from std.testing import assert_almost_equal, assert_equal

from std.utils import Index


def test[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    config: Optional[MatmulConfig[a_type, b_type, c_type, transpose_b]] = None,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    comptime assert Bool(N) and Bool(
        K
    ), "This test currently requires static N and K."

    print(m, "x", n, "x", k)

    var a_size = m * k
    var b_size = n * k if transpose_b else k * n
    var c_size = m * n

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)

    comptime b_shape_0 = N if transpose_b else K
    comptime b_shape_1 = K if transpose_b else N
    var a_tensor = TileTensor(
        a_device_buffer,
        row_major(Coord(m, Idx[K.value()])),
    )
    var b_tensor = TileTensor(
        b_device_buffer,
        row_major(Coord(Idx[b_shape_0.value()], Idx[b_shape_1.value()])),
    )
    var c_tensor = TileTensor(
        c_device_buffer,
        row_major(Coord(m, Idx[N.value()])),
    )
    var c_ref_tensor = TileTensor(
        c_device_ref_buffer,
        row_major(Coord(m, Idx[N.value()])),
    )

    comptime if c_type.is_float8():
        rand(a_host_ptr.unsafe_ptr(), m * k, min=-1.0, max=1.0)
        rand(b_host_ptr.unsafe_ptr(), k * n, min=-1.0, max=1.0)
    else:
        comptime rand_min = -100
        comptime rand_max = 100

        for i in range(m * k):
            var val = random_si64(rand_min, rand_max)
            a_host_ptr[i] = val.cast[a_type]()

        for i in range(k * n):
            var val = random_si64(rand_min, rand_max)
            b_host_ptr[i] = val.cast[b_type]()

    # Move operands to the Device
    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)

    comptime if config:
        multistage_gemm[transpose_b=transpose_b, config=config.value()](
            c_tensor,
            a_tensor.as_immut(),
            b_tensor.as_immut(),
            ctx,
        )
    else:
        _matmul_gpu[use_tensor_core=True, transpose_b=transpose_b](
            c_tensor,
            a_tensor.as_immut(),
            b_tensor.as_immut(),
            ctx,
        )

    comptime if c_type.is_float8():
        # The vendor BLAS does not support `BF16 @ BF16 = FP8`, so use the naive
        # kernel as the reference implementation.
        comptime BLOCK_DIM = 16
        comptime gemm_naive = matmul_kernel_naive[
            c_type,
            a_type,
            b_type,
            type_of(c_tensor).LayoutType,
            type_of(a_tensor).LayoutType,
            type_of(b_tensor).LayoutType,
            BLOCK_DIM,
            transpose_b=True,
        ]
        ctx.enqueue_function[gemm_naive](
            c_ref_tensor,
            a_tensor.as_immut(),
            b_tensor.as_immut(),
            Int32(m),
            Int32(n),
            Int32(k),
            grid_dim=(ceildiv(m, BLOCK_DIM), ceildiv(n, BLOCK_DIM)),
            block_dim=(BLOCK_DIM, BLOCK_DIM),
        )
    else:
        vendor_blas.matmul(
            ctx,
            c_ref_tensor,
            a_tensor.as_immut(),
            b_tensor.as_immut(),
            c_row_major=True,
            transpose_b=transpose_b,
        )

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var errors = 0
    for i in range(m * n):
        if c_host_ptr[i].cast[.float32]() != c_host_ref_ptr[i].cast[.float32]():
            errors += 1

    assert_equal(errors, 0)

    # Cleanup
    _ = a_device_buffer^
    _ = b_device_buffer^
    _ = c_device_buffer^
    _ = c_device_ref_buffer^


def test[
    in_type: DType,
    out_type: DType,
    transpose_b: Bool,
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    return test[in_type, in_type, out_type, transpose_b, M=M, N=N, K=K](
        ctx, m, n, k
    )


def test[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    transpose_b: Bool,
    //,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
    M: Optional[Int] = None,
    N: Optional[Int] = None,
    K: Optional[Int] = None,
](ctx: DeviceContext, m: Int, n: Int, k: Int) raises:
    return test[a_type, b_type, c_type, transpose_b, config, M=M, N=N, K=K](
        ctx, m, n, k
    )


def test_bf16(ctx: DeviceContext) raises:
    print("=== test_bf16")

    test[
        in_type=.bfloat16,
        out_type=.float32,
        transpose_b=False,
        N=256,
        K=128,
    ](ctx, 256, 256, 128)
    test[
        in_type=.bfloat16,
        out_type=.float32,
        transpose_b=True,
        N=256,
        K=128,
    ](ctx, 256, 256, 128)
    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=False,
        N=256,
        K=128,
    ](ctx, 256, 256, 128)
    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=256,
        K=128,
    ](ctx, 256, 256, 128)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=False,
        N=256,
        K=128,
    ](ctx, 1024, 256, 128)
    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=False,
        N=256,
        K=256,
    ](ctx, 1024, 256, 256)
    test[
        in_type=.bfloat16,
        out_type=.float32,
        transpose_b=True,
        N=256,
        K=1024,
    ](ctx, 1024, 256, 1024)
    test[
        in_type=.bfloat16,
        out_type=.float32,
        transpose_b=True,
        N=1024,
        K=1024,
    ](ctx, 1024, 1024, 1024)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=284,
        K=256,
    ](ctx, 256, 284, 256)
    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=260,
        K=1024,
    ](ctx, 259, 260, 1024)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=36864,
        K=6144,
    ](ctx, 2, 36864, 6144)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=55296,
        K=6144,
    ](ctx, 2, 55296, 6144)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=6144,
        K=24576,
    ](ctx, 2, 6144, 24576)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=6144,
        K=18432,
    ](ctx, 2, 6144, 18432)

    test[
        in_type=.bfloat16,
        out_type=.bfloat16,
        transpose_b=True,
        N=6144,
        K=6144,
    ](ctx, 2, 6144, 6144)


def test_float8[fp8_type: DType](ctx: DeviceContext) raises:
    print("=== test_float8", fp8_type)

    test[
        in_type=fp8_type,
        out_type=.bfloat16,
        transpose_b=True,
        N=512,
        K=640,
    ](ctx, 480, 512, 640)

    test[
        in_type=.bfloat16,
        out_type=fp8_type,
        transpose_b=True,
        N=384,
        K=128,
    ](ctx, 256, 384, 128)


def test_block_k(ctx: DeviceContext) raises:
    print("=== test_block_k")

    @__parameter
    def test_block_k[
        in_type: DType,
        out_type: DType,
        block_k: Int,
        N: Int,
        K: Int,
    ](m: Int, n: Int, k: Int) raises:
        comptime config = MatmulConfig[in_type, in_type, out_type, True](
            block_tile_shape=Index(64, 64, block_k),
            warp_tile_shape=Index(32, 32, block_k),
        )
        test[config, N=N, K=K](ctx, m, n, k)

    comptime block_ks: List[Int] = [32, 64, 128, 256]

    comptime for i in range(len(block_ks)):
        test_block_k[.bfloat16, .bfloat16, block_ks[i], 1024, 1024](
            192, 1024, 1024
        )


def test_warp_k_partitions(ctx: DeviceContext) raises:
    print("=== test_warp_k_partitions")

    @__parameter
    def test_warp_k_partitions[
        in_type: DType,
        out_type: DType,
        N: Int,
        K: Int,
    ](m: Int, n: Int, k: Int) raises:
        comptime config_type = MatmulConfig[in_type, in_type, out_type, True]
        comptime configs: List[config_type] = [
            # TEST: num_warps=(1, 4, 1).
            config_type(
                block_tile_shape=Index(16, 128, 128),
                warp_tile_shape=Index(16, 32, 128),
            ),
            # TEST: num_warps=(1, 1, 4).
            config_type(
                block_tile_shape=Index(16, 16, 64),
                warp_tile_shape=Index(16, 16, 64),
                num_warp_k_partitions=4,
            ),
            config_type(
                block_tile_shape=Index(16, 16, 128),
                warp_tile_shape=Index(16, 16, 128),
                num_warp_k_partitions=4,
            ),
            # TEST: num_warps=(1, 2, 2).
            config_type(
                block_tile_shape=Index(16, 128, 64),
                warp_tile_shape=Index(16, 64, 64),
                num_warp_k_partitions=2,
            ),
        ]

        comptime for i in range(len(configs)):
            test[configs[i], N=N, K=K](ctx, m, n, k)

    test_warp_k_partitions[.bfloat16, .bfloat16, 2048, 2048](16, 2048, 2048)


def test_float32(ctx: DeviceContext) raises:
    print("=== test_float32")

    test[
        in_type=.float32,
        out_type=.float32,
        transpose_b=False,
        N=256,
        K=128,
    ](ctx, 256, 256, 128)
    test[
        in_type=.float32,
        out_type=.float32,
        transpose_b=True,
        N=256,
        K=128,
    ](ctx, 256, 256, 128)


def test_partial_tile[
    dtype: DType, M: Int, N: Int, K: Int
](ctx: DeviceContext) raises:
    """Checks the C store when a warp tile does not fit inside (M, N).

    The trailing block of a row holds fragments past the last column, whose
    linear offset lands on the next row, and a 4-row store group can run past
    the last row. Bounding those on the offset alone lets them overwrite live
    data. `transpose_b=False` keeps the dispatch on the multistage kernel
    rather than on `AMDMatmul`.

    C carries a guard band past its last element, so a warp tile that straddles
    the last row is caught here rather than by whatever allocation happens to
    follow C.
    """
    comptime WM = 32
    comptime config = MatmulConfig[dtype, dtype, dtype, False](
        block_tile_shape=Index(64, 64, 32),
        warp_tile_shape=Index(WM, 32, 32),
    )

    print("=== test_partial_tile", dtype, M, "x", N, "x", K)

    # A straddling warp tile writes all WM of its rows, so it reaches at most
    # WM * N elements past C's start. The operands below are integers, so the
    # kernel cannot produce a fractional value: only an untouched guard element
    # still holds the fill.
    comptime c_size = M * N + WM * N
    comptime guard_fill = Scalar[dtype](0.5)

    var a_host_ptr = ctx.enqueue_create_host_buffer[dtype](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[dtype](K * N)
    var c_host_ptr = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[dtype](M * N)

    var a_device = ctx.enqueue_create_buffer[dtype](M * K)
    var b_device = ctx.enqueue_create_buffer[dtype](K * N)
    var c_device = ctx.enqueue_create_buffer[dtype](c_size)
    var c_device_ref = ctx.enqueue_create_buffer[dtype](M * N)

    var a_tensor = TileTensor(a_device, row_major(Coord(Idx[M], Idx[K])))
    var b_tensor = TileTensor(b_device, row_major(Coord(Idx[K], Idx[N])))
    var c_tensor = TileTensor(c_device, row_major(Coord(Idx[M], Idx[N])))
    var c_ref_tensor = TileTensor(
        c_device_ref, row_major(Coord(Idx[M], Idx[N]))
    )

    # Small integers keep the accumulation exact in both kernels, so the
    # comparison below can be exact.
    for i in range(M * K):
        a_host_ptr[i] = random_si64(-100, 100).cast[dtype]()
    for i in range(K * N):
        b_host_ptr[i] = random_si64(-100, 100).cast[dtype]()
    for i in range(M * N):
        c_host_ptr[i] = 0
        c_host_ref_ptr[i] = 0
    for i in range(M * N, c_size):
        c_host_ptr[i] = guard_fill

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host_ptr)
    ctx.enqueue_copy(c_device_ref, c_host_ref_ptr)

    multistage_gemm[transpose_b=False, config=config](
        c_tensor, a_tensor.as_immut(), b_tensor.as_immut(), ctx
    )

    comptime BLOCK_DIM = 16
    comptime gemm_naive = matmul_kernel_naive[
        dtype,
        dtype,
        dtype,
        type_of(c_ref_tensor).LayoutType,
        type_of(a_tensor).LayoutType,
        type_of(b_tensor).LayoutType,
        BLOCK_DIM,
        transpose_b=False,
    ]
    ctx.enqueue_function[gemm_naive](
        c_ref_tensor,
        a_tensor.as_immut(),
        b_tensor.as_immut(),
        Int32(M),
        Int32(N),
        Int32(K),
        grid_dim=(ceildiv(M, BLOCK_DIM), ceildiv(N, BLOCK_DIM)),
        block_dim=(BLOCK_DIM, BLOCK_DIM),
    )

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    var errors = 0
    for i in range(M * N):
        if c_host_ptr[i] != c_host_ref_ptr[i]:
            errors += 1

    assert_equal(errors, 0)

    var guard_writes = 0
    for i in range(M * N, c_size):
        if c_host_ptr[i] != guard_fill:
            guard_writes += 1

    assert_equal(guard_writes, 0)

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def _run_fp32_split_k[
    N: Int, K: Int, num_k_partitions: Int
](
    ctx: DeviceContext,
    m: Int,
    a_dev: DeviceBuffer[.float32],
    b_dev: DeviceBuffer[.float32],
    mut c_dev: DeviceBuffer[.float32],
) raises:
    """Runs the skinny-deep matmul with `num_k_partitions` K-splits."""
    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))

    # Mirrors the fp32 split-K band's config; only num_k_partitions differs
    # (1 = single-pass, >1 = split-K).
    comptime config = MatmulConfig[.float32, .float32, .float32, True](
        block_tile_shape=Index(16, 16, 64),
        warp_tile_shape=Index(16, 16, 64),
        mma_shape=_amdgpu_get_mma_shape[.float32, True](),
        num_pipeline_stages=1,
        num_k_partitions=num_k_partitions,
    )
    multistage_gemm[transpose_b=True, config=config](
        c, a.as_immut(), b.as_immut(), config, ctx
    )


def test_fp32_split_k[
    N: Int, K: Int, P: Int
](ctx: DeviceContext, m: Int) raises:
    """Split-K (P) and single-pass (1) must both match an fp32 gold ref."""
    print("  M=", m, " N=", N, " K=", K, " P=", P, sep="")

    var a_dev = ctx.enqueue_create_buffer[.float32](m * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](N * K)
    var c_ref = ctx.enqueue_create_buffer[.float32](m * N)
    var c_splitk = ctx.enqueue_create_buffer[.float32](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)

    ctx.enqueue_memset(c_ref, 0)
    ctx.enqueue_memset(c_splitk, 0)

    _run_fp32_split_k[N, K, 1](ctx, m, a_dev, b_dev, c_ref)
    _run_fp32_split_k[N, K, P](ctx, m, a_dev, b_dev, c_splitk)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb, c_ref.map_to_host() as sp, c_splitk.map_to_host() as sk:
        for row in range(m):
            for col in range(N):
                var acc = Float64(0)
                for kk in range(K):
                    acc += Float64(ha[row * K + kk]) * Float64(hb[col * K + kk])
                var gold = Float32(acc)
                # fp32 accum end-to-end -> reassociation tolerance only.
                assert_almost_equal(
                    sp[row * N + col], gold, rtol=1e-3, atol=1e-2
                )
                assert_almost_equal(
                    sk[row * N + col], gold, rtol=1e-3, atol=1e-2
                )

    _ = a_dev^
    _ = b_dev^
    _ = c_ref^
    _ = c_splitk^


def test_fp32_split_k_dispatch[
    N: Int, K: Int
](ctx: DeviceContext, m: Int) raises:
    """The public dispatch must route the skinny-deep fp32 shape onto the
    split-K band and match an fp32 gold reference."""
    print("  dispatch M=", m, " N=", N, " K=", K, sep="")

    var a_dev = ctx.enqueue_create_buffer[.float32](m * K)
    var b_dev = ctx.enqueue_create_buffer[.float32](N * K)
    var c_dev = ctx.enqueue_create_buffer[.float32](m * N)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb:
        rand(ha.unsafe_ptr(), m * K, min=-1.0, max=1.0)
        rand(hb.unsafe_ptr(), N * K, min=-1.0, max=1.0)
    ctx.enqueue_memset(c_dev, 0)

    var a = TileTensor(a_dev, row_major(Coord(m, Idx[K])))
    var b = TileTensor(b_dev, row_major(Coord(Idx[N], Idx[K])))
    var c = TileTensor(c_dev, row_major(Coord(m, Idx[N])))

    _matmul_gpu[use_tensor_core=True, transpose_b=True](c, a, b, ctx)

    with a_dev.map_to_host() as ha, b_dev.map_to_host() as hb, c_dev.map_to_host() as hc:
        for row in range(m):
            for col in range(N):
                var acc = Float64(0)
                for kk in range(K):
                    acc += Float64(ha[row * K + kk]) * Float64(hb[col * K + kk])
                assert_almost_equal(
                    hc[row * N + col], Float32(acc), rtol=1e-3, atol=1e-2
                )

    _ = a_dev^
    _ = b_dev^
    _ = c_dev^


def test_matmul_config_from_block_shape(ctx: DeviceContext) raises:
    # This test takes too long to execute for CI, but is maintained here as a useful
    # unit test for verifying changes to parts of the matmul dispatcher.
    print("=== test_matmul_config_from_block_shape")

    comptime in_type = DType.bfloat16
    comptime out_type = DType.float32
    comptime transpose_b = True

    # The test is intended to cover partial and complete blocks.
    comptime m_val = 1012
    comptime n_val = 1016

    comptime block_sizes = [16, 32, 64, 96, 128, 160, 192, 224, 256]

    comptime for block_m in block_sizes:
        comptime for block_n in block_sizes:

            @__parameter
            def test_block_shape[block_m: Int, block_n: Int, k: Int]() raises:
                comptime config = _amdgpu_matmul_config_from_block_shape[
                    out_type, in_type, in_type, transpose_b, k
                ](Index(block_m, block_n))
                print(
                    block_m,
                    block_n,
                    config.block_tile_shape,
                    config.warp_tile_shape,
                    config.num_warp_k_partitions,
                )
                test[config, M=m_val, N=n_val, K=k](ctx, m_val, n_val, k)

            comptime if block_m <= 32 and block_n <= 32:
                # Exercise the warp_k partitioning where the number of partitions
                # depends on breaking K into even chunks.
                comptime for k in [256, 384, 512, 768, 1024]:
                    test_block_shape[block_m, block_n, k]()
            else:
                # Exercise the logic where block_k is increased, but only if K is
                # multiple of the increased block size.
                comptime for k in [320, 768]:
                    test_block_shape[block_m, block_n, k]()


def main() raises:
    with DeviceContext() as ctx:
        test_bf16(ctx)

        comptime if ctx.default_device_info == MI355X:
            test_float8[.float8_e4m3fn](ctx)
            test_float8[.float8_e5m2](ctx)
        else:
            test_float8[.float8_e4m3fnuz](ctx)
            test_float8[.float8_e5m2fnuz](ctx)

        test_block_k(ctx)
        test_warp_k_partitions(ctx)
        test_float32(ctx)

        # C tiles that run past (M, N): an N below the block tile, an odd N, a
        # large N with a partial trailing block (GPT-2's lm_head), and an M
        # that does not fill the 4-row store groups.
        comptime for out_type in [DType.float32, DType.bfloat16]:
            test_partial_tile[out_type, M=20, N=17, K=768](ctx)
            test_partial_tile[out_type, M=20, N=100, K=768](ctx)
            test_partial_tile[out_type, M=20, N=257, K=768](ctx)
            test_partial_tile[out_type, M=18, N=128, K=768](ctx)
            test_partial_tile[out_type, M=20, N=50257, K=768](ctx)
            test_partial_tile[out_type, M=20, N=50256, K=768](ctx)

        # Skinny-deep fp32 split-K (e.g. MiniMax-M3 gate: N=128, K=6144).
        print("=== test_fp32_split_k")
        for m in [1, 16, 32]:
            test_fp32_split_k[128, 6144, 4](ctx, m)
            test_fp32_split_k[128, 6144, 8](ctx, m)
            test_fp32_split_k[128, 6144, 16](ctx, m)
            test_fp32_split_k[128, 6144, 32](ctx, m)
        for m in [16, 32]:
            test_fp32_split_k_dispatch[128, 6144](ctx, m)
