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
#
# Test: bf16 matmul (sm100, 1SM) with weight prefetch.
#
# Validates weight prefetching (prefetch_tiles_n > 0) for both swapAB=False
# and swapAB=True across all supported 1SM tile shapes, using PDL overlap.
# Also validates the RMS norm → bf16 matmul pipeline with prefetch.
# ===----------------------------------------------------------------------=== #

from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from max.gpu.primitives.grid_controls import PDLLevel
from linalg.utils import elementwise_epilogue_type

from internal_utils import assert_almost_equal
from std.random import rand
from layout import TileTensor, Coord, CoordLike, row_major, Idx
from linalg.matmul.gpu.sm100_structured.default.matmul import (
    blackwell_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    MatmulConfig,
    choose_config,
)
from nn.normalization import rms_norm_gpu

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def test_blackwell_matmul_with_weight_prefetch[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    pdl_level: PDLLevel = PDLLevel.OVERLAP_AT_END,
    prefetch_tiles_n: Int = 2,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"weight prefetch: dtype={a_type} shape=({M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" cta_group={cta_group} swapAB={swapAB}"
        t" prefetch_tiles_n={prefetch_tiles_n}"
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = (
        Int(n.value())
        * Int(k.value()) if transpose_b else Int(k.value())
        * Int(n.value())
    )
    var c_size = Int(m.value()) * Int(n.value())

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        pdl_level=pdl_level,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        ctx,
    )

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    vendor_blas.matmul(
        ctx,
        c_ref_tensor_lt,
        a_lt,
        b_lt,
        c_row_major=True,
        transpose_b=transpose_b,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )
    print("\n=== TEST PASSED ===\n")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^


def test_rmsnorm_then_matmul[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    pdl_level: PDLLevel = PDLLevel(),
    prefetch_tiles_n: Int = 0,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    print(
        t"rmsnorm->matmul: dtype={a_type} shape=({M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" cta_group={cta_group} swapAB={swapAB}"
        t" prefetch_tiles_n={prefetch_tiles_n}"
    )

    var ak_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = M * K
    var b_size = N * K
    var c_size = M * N

    var a_raw_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var gamma_host_ptr = ctx.enqueue_create_host_buffer[a_type](K)
    var c_vendor_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ours_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)

    rand(a_raw_host_ptr.unsafe_ptr(), a_size)
    rand(b_host_ptr.unsafe_ptr(), b_size)
    for i in range(K):
        gamma_host_ptr[i] = (Float64(i + K) / Float64(K)).cast[a_type]()

    var a_raw_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_raw_tensor = TileTensor(a_raw_device, ak_shape)

    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)

    var gamma_device = ctx.enqueue_create_buffer[a_type](K)
    var gamma_tensor = TileTensor(
        gamma_device, row_major(Idx[KType.static_value])
    )

    var a_normed_vendor_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_normed_vendor_tensor = TileTensor(a_normed_vendor_device, ak_shape)

    var a_normed_ours_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_normed_ours_tensor = TileTensor(a_normed_ours_device, ak_shape)

    var c_vendor_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_vendor_tensor = TileTensor(c_vendor_device, c_shape)

    var c_ours_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ours_tensor = TileTensor(c_ours_device, c_shape)

    ctx.enqueue_copy(a_raw_device, a_raw_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(gamma_device, gamma_host_ptr)

    var epsilon = Float32(0.001)
    var weight_offset = Scalar[a_type](0.0)
    var norm_shape = Index(M, K)

    @always_inline
    @__copy_capture(a_raw_tensor)
    @__parameter
    def input_fn[width: Int](coords: Coord) -> SIMD[a_type, width]:
        return a_raw_tensor.raw_load[width=width](a_raw_tensor.layout(coords))

    @always_inline
    @__copy_capture(a_normed_vendor_tensor)
    @__parameter
    def output_fn_vendor[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[a_type, width]) -> None:
        a_normed_vendor_tensor.raw_store[width=width, alignment=alignment](
            a_normed_vendor_tensor.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, output_fn_vendor, multiply_before_cast=True](
        Coord(norm_shape), gamma_tensor, epsilon, weight_offset, ctx
    )

    vendor_blas.matmul(
        ctx,
        c_vendor_tensor.to_layout_tensor(),
        a_normed_vendor_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        c_row_major=True,
        transpose_b=transpose_b,
    )

    @always_inline
    @__copy_capture(a_normed_ours_tensor)
    @__parameter
    def output_fn_ours[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[a_type, width]) -> None:
        a_normed_ours_tensor.raw_store[width=width, alignment=alignment](
            a_normed_ours_tensor.layout(coords), val
        )

    rms_norm_gpu[2, input_fn, output_fn_ours, multiply_before_cast=True](
        Coord(norm_shape), gamma_tensor, epsilon, weight_offset, ctx
    )

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    @__parameter
    @always_inline
    @__copy_capture(c_ours_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = 1,
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        c_ours_tensor.raw_store[
            width=width, alignment=alignment * size_of[c_type]()
        ](c_ours_tensor.layout(Coord(idx)), rebind[SIMD[c_type, width]](val))

    comptime epi = Optional[elementwise_epilogue_type](epilogue_fn)

    blackwell_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        config=matmul_config,
        elementwise_lambda_fn=epi,
        pdl_level=pdl_level,
    ](
        c_ours_tensor,
        a_normed_ours_tensor,
        b_tensor,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_vendor_host_ptr, c_vendor_device)
    ctx.enqueue_copy(c_ours_host_ptr, c_ours_device)
    ctx.synchronize()

    assert_almost_equal(
        c_ours_host_ptr.unsafe_ptr(),
        c_vendor_host_ptr.unsafe_ptr(),
        c_size,
        atol=0.0001,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    _ = a_raw_device^
    _ = b_device^
    _ = gamma_device^
    _ = a_normed_vendor_device^
    _ = a_normed_ours_device^
    _ = c_vendor_device^
    _ = c_ours_device^


def test_prefetch_depth_fits_the_ring[
    a_type: DType, c_type: DType
](N: Int, K: Int) raises:
    """The prefetch must never outrun the pipeline, or Phase 1 fills the ring
    before any barrier can fire and the kernel hangs."""
    for m in range(1, 512):
        var config = choose_config[a_type, a_type, c_type, True](m, N, K, 1)
        var group_stages = config.num_pipeline_stages // config.k_group_size
        if config.prefetch_tiles_n > group_stages:
            raise Error(
                String(
                    "M=",
                    m,
                    " N=",
                    N,
                    " K=",
                    K,
                    " chose prefetch_tiles_n=",
                    config.prefetch_tiles_n,
                    " but only ",
                    group_stages,
                    " group pipeline stages fit",
                )
            )


def main() raises:
    comptime for nk in [(2624, 6144), (2048, 2048), (256, 128), (8192, 7168)]:
        test_prefetch_depth_fits_the_ring[.bfloat16, .bfloat16](nk[0], nk[1])

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime BK = TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[dtype]()
        comptime MMA_K = 16

        # -----------------------------------------------------------------------
        # Part 1: All 1SM tile shapes with weight prefetch (both swapAB variants)
        # -----------------------------------------------------------------------
        comptime for swizzle in [TensorMapSwizzle.SWIZZLE_128B]:
            comptime BK2 = swizzle.bytes() // size_of[dtype]()

            comptime for mma_m in [64, 128]:
                comptime for mma_n in [
                    8,
                    16,
                    32,
                    48,
                    64,
                    80,
                    88,
                    96,
                    112,
                    128,
                    144,
                    152,
                    184,
                    192,
                    256,
                ]:
                    comptime block_tile_shape = Index(mma_m, mma_n, BK2)
                    comptime umma_shape = Index(mma_m, mma_n, MMA_K)

                    test_blackwell_matmul_with_weight_prefetch[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                        cta_group=1,
                        block_swizzle_size=8,
                        swapAB=False,
                        prefetch_tiles_n=2,
                    ](
                        ctx,
                        Int(1000),
                        Idx[1024],
                        Idx[1024 + 16],
                    )

                    comptime for swapAB in [False, True]:
                        test_blackwell_matmul_with_weight_prefetch[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                            cta_group=1,
                            block_swizzle_size=4,
                            swapAB=swapAB,
                            prefetch_tiles_n=2,
                        ](
                            ctx,
                            Int(512),
                            Idx[4096],
                            Idx[1024 + 16],
                        )

                        test_blackwell_matmul_with_weight_prefetch[
                            dtype,
                            dtype,
                            .bfloat16,
                            block_tile_shape,
                            umma_shape,
                            cluster_shape=StaticTuple[Int32, 3](4, 2, 1),
                            cta_group=1,
                            block_swizzle_size=0,
                            swapAB=swapAB,
                            k_group_size=2,
                            prefetch_tiles_n=2,
                        ](
                            ctx,
                            Int(500),
                            Idx[2048],
                            Idx[4096],
                        )

                    test_blackwell_matmul_with_weight_prefetch[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](8, 2, 1),
                        cta_group=1,
                        block_swizzle_size=2,
                        swapAB=False,
                        prefetch_tiles_n=2,
                    ](
                        ctx,
                        Int(999),
                        Idx[256],
                        Idx[128],
                    )

                    test_blackwell_matmul_with_weight_prefetch[
                        dtype,
                        dtype,
                        .bfloat16,
                        block_tile_shape,
                        umma_shape,
                        cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
                        cta_group=1,
                        block_swizzle_size=1,
                        swapAB=False,
                        prefetch_tiles_n=2,
                    ](
                        ctx,
                        Int(777),
                        Idx[2560],
                        Idx[8192],
                    )

        # -----------------------------------------------------------------------
        # Part 2: RMS norm → matmul with PDL overlap (small-M decode shapes)
        # -----------------------------------------------------------------------
        comptime for m in [1, 2, 4, 8, 16, 32, 48, 63]:
            comptime for swapAB in [False, True]:
                test_rmsnorm_then_matmul[
                    dtype,
                    dtype,
                    .bfloat16,
                    block_tile_shape=Index(64, 128, BK),
                    mma_shape=Index(64, 128, MMA_K),
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                    block_swizzle_size=8,
                    swapAB=swapAB,
                    pdl_level=PDLLevel.OVERLAP_AT_END,
                    prefetch_tiles_n=2,
                ](
                    ctx,
                    Int(m),
                    Idx[4096],
                    Idx[4096],
                )

                test_rmsnorm_then_matmul[
                    dtype,
                    dtype,
                    .bfloat16,
                    block_tile_shape=Index(64, 128, BK),
                    mma_shape=Index(64, 128, MMA_K),
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                    b_swizzle=TensorMapSwizzle.SWIZZLE_128B,
                    block_swizzle_size=8,
                    swapAB=swapAB,
                    pdl_level=PDLLevel.OVERLAP_AT_END,
                    prefetch_tiles_n=2,
                ](
                    ctx,
                    Int(m),
                    Idx[8192],
                    Idx[7168],
                )

        # -----------------------------------------------------------------------
        # Part 3: small-K boundary conditions for weight prefetch
        #
        # Exercises cases where prefetch_tiles_n >= ceildiv(K, BK) so the
        # runtime guard in Phase 1 / Phase 2 must clamp the prefetch loop.
        # BK=64 (SWIZZLE_128B / sizeof[bf16]) for the tile shape used here.
        #
        #   E) K=64,  k_group_size=1, pf=1  → num_iters == pf            (safe)
        #   C) K=128, k_group_size=1, pf=2  → num_iters == pf            (safe)
        #   D) K=256, k_group_size=2, pf=2  → num_group_iters == pf      (safe)
        #   A) K=64,  k_group_size=1, pf=2  → num_iters < pf, guard fires (safe)
        #   B) K=128, k_group_size=2, pf=2  → num_group_iters < pf, guard (safe)
        # -----------------------------------------------------------------------
        comptime for swapAB in [False, True]:
            # E: equality, pf=1
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(64, 128, 16),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                swapAB=swapAB,
                k_group_size=1,
                prefetch_tiles_n=1,
            ](ctx, Int(256), Idx[4096], Idx[64])

            # C: equality, pf=2 k_group=1
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(64, 128, 16),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                swapAB=swapAB,
                k_group_size=1,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[128])

            # D: equality, pf=2 k_group=2
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(64, 128, 16),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                swapAB=swapAB,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[256])

            # A: pf > num_iters, guard fires
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(64, 128, 16),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                swapAB=swapAB,
                k_group_size=1,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[64])

            # B: pf > num_group_iters, guard fires
            test_blackwell_matmul_with_weight_prefetch[
                dtype,
                dtype,
                .bfloat16,
                block_tile_shape=Index(64, 128, 64),
                mma_shape=Index(64, 128, 16),
                cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                cta_group=1,
                swapAB=swapAB,
                k_group_size=2,
                prefetch_tiles_n=2,
            ](ctx, Int(256), Idx[4096], Idx[128])
