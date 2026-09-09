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
# Test: block-scaled FP4 matmul (sm100, small_bn / 2SM) with weight prefetching.
#
# Mirrors test_matmul_sm100_block_scaled_fp4_small_bn_2sm.mojo with
# prefetch_tiles_n=2 added to BlockScaledMatmulConfig.  The raw kernel from
# block_scaled_matmul_small_bn is called directly since the shared testbed
# does not expose the prefetch parameter.
#
# 2SM cooperative MMA: block_tile=(128, mma_n//2, BK), umma=(256, mma_n, MMA_K).
# Tile shapes: mma_n in [16, 32, 48, 64, 96].  NVFP4 only.
# ===----------------------------------------------------------------------=== #
from std.math import align_up, ceildiv
from std.sys import argv, size_of
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.random import rand
from internal_utils import assert_almost_equal
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.matmul.gpu.sm100.block_scaled_matmul_small_bn import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.utils import elementwise_epilogue_type
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from std.random import random_ui64
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def _test_impl[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
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
    k_group_size: Int = 2,
    num_clc_pipeline_stages: Int = 0,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
    normal_epilogue: Bool = False,
    prefetch_tiles_n: Int = 2,
](ctx: DeviceContext, m: MType, n: NType, k: KType,) raises:
    print(
        t"[prefetch-small_bn-2sm]"
        t" dtypes=({a_type},{b_type},{c_type},{scales_dtype})"
        t" shape=({Int(m.value())},{Int(n.value())},{Int(k.value())})"
        t" mma={mma_shape} tile={block_tile_shape}"
        t" cta_group={cta_group}"
        t" cluster=({cluster_shape[0]},{cluster_shape[1]},{cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size}"
        t" SF_VECTOR_SIZE={SF_VECTOR_SIZE}"
        t" prefetch_tiles_n={prefetch_tiles_n}"
    )

    # FP4 data stored packed: 2 FP4 elements per uint8 byte.
    var a_shape = row_major(Coord(m, Idx[KType.static_value // 2]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value // 2])
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * (KType.static_value // 2)
    var b_size = Int(n.value()) * (KType.static_value // 2)
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

    var a_scales_shape = row_major(
        Coord(
            ceildiv(Int(m.value()), SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[ceildiv(NType.static_value, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    var a_lt = a_tensor.to_layout_tensor()
    var b_lt = b_tensor.to_layout_tensor()
    var a_scales_lt = a_scales_tensor.to_layout_tensor()
    var b_scales_lt = b_scales_tensor.to_layout_tensor()
    var c_ref_tensor_lt = c_ref_tensor.to_layout_tensor()

    if simple_init():
        for m in range(Int(m.value())):
            for k in range(KType.static_value // 2):
                comptime assert a_host.flat_rank == 2
                a_host[m, k] = UInt8(m).cast[a_type]()
        for n in range(Int(n.value())):
            for k in range(KType.static_value // 2):
                comptime assert b_host.flat_rank == 2
                b_host[n, k] = UInt8(n).cast[b_type]()
    else:
        rand(a_host._storage, a_host.num_elements(), min=0, max=255)
        rand(b_host._storage, b_host.num_elements(), min=0, max=255)

    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    for idx0 in range(align_up(Int(m.value()), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
            SF_VECTOR_SIZE,
        ):
            if idx0 >= Int(m.value()) or idx1 >= Int(k.value()):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    a_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    for idx0 in range(align_up(Int(n.value()), SF_MN_GROUP_SIZE)):
        for idx1 in range(
            0,
            align_up(Int(k.value()), SF_VECTOR_SIZE * SF_ATOM_K),
            SF_VECTOR_SIZE,
        ):
            if idx0 >= Int(n.value()) or idx1 >= Int(k.value()):
                set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                    b_scales_host, idx0, idx1, Scalar[scales_dtype](0.0)
                )

    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF4NVF4,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    var c_device_lt = c_tensor.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_device_lt)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = 1,
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var scaled = rebind[SIMD[c_type, width]](val) * Scalar[c_type](2)
        c_device_lt.store[store_alignment=alignment * size_of[c_type](),](
            idx, scaled
        )

    comptime epi = Optional[elementwise_epilogue_type](
        epilogue_fn
    ) if normal_epilogue else None

    comptime K_phys = KType.static_value
    blackwell_block_scaled_matmul_tma_umma_warp_specialized[
        transpose_b=transpose_b,
        K=K_phys,
        config=matmul_config,
        elementwise_lambda_fn=epi,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
    )

    vendor_blas.matmul(
        ctx,
        c_ref_tensor_lt.as_unsafe_any_origin(),
        a_lt,
        b_lt,
        a_scales=a_scales_lt.as_imm().as_unsafe_any_origin(),
        b_scales=b_scales_lt.as_imm().as_unsafe_any_origin(),
        transpose_b=transpose_b,
        c_row_major=True,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    comptime if normal_epilogue:
        for i in range(c_host_ref.num_elements()):
            c_host_ref._storage[i] = c_host_ref._storage[i] * Scalar[c_type](2)

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^


def run_matmul_sm100_block_scaled_fp4_small_bn_2sm_prefetch_suite[
    suite_scales_dtype: DType,
    suite_sf_vector_size: Int,
]() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.uint8
        comptime out_dtype = DType.bfloat16
        comptime scales_dtype = suite_scales_dtype
        comptime SF_VECTOR_SIZE = suite_sf_vector_size
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32

        @__parameter
        @always_inline
        def run[
            MType: CoordLike,
            NType: CoordLike,
            KType: CoordLike,
            //,
            a_type: DType,
            b_type: DType,
            c_type: DType,
            _scales_dtype: DType,
            block_tile_shape: IndexList[3],
            mma_shape: IndexList[3],
            cluster_shape: StaticTuple[Int32, 3],
            cta_group: Int,
            transpose_b: Bool = True,
            a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
            b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
            c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
            block_swizzle_size: Int = 0,
            swapAB: Bool = True,
            k_group_size: Int = 2,
            num_clc_pipeline_stages: Int = 0,
            SF_VECTOR_SIZE: Int = suite_sf_vector_size,
            normal_epilogue: Bool = False,
        ](ctx: DeviceContext, m: MType, n: NType, k: KType,) raises:
            _test_impl[
                a_type,
                b_type,
                c_type,
                scales_dtype,
                block_tile_shape,
                mma_shape,
                cluster_shape,
                cta_group,
                transpose_b,
                a_swizzle,
                b_swizzle,
                c_swizzle,
                block_swizzle_size,
                swapAB,
                k_group_size,
                num_clc_pipeline_stages,
                SF_VECTOR_SIZE=suite_sf_vector_size,
                normal_epilogue=normal_epilogue,
                prefetch_tiles_n=2,
            ](ctx, m, n, k)

        # 2SM tests: sweep MMA_N in [16, 32, 48, 64, 96].
        # Note: MMA_N=24 is valid HW but BN=12 breaks TMA tile layout.
        comptime for mma_n in [16, 32, 48, 64, 96]:
            comptime block_tile = Index(128, mma_n // 2, BK)
            comptime umma = Index(256, mma_n, MMA_K)

            # Basic cluster_shape=(2,1,1)
            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Idx[1], Idx[2304], Idx[16384])

            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Idx[1], Idx[16384], Idx[2048])

            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Idx[1], Idx[6656], Idx[16384])

            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Idx[1], Idx[16384], Idx[6656])

            # Larger cluster shapes
            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(4), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=4,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Idx[1], Idx[2304], Idx[16384])

            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                block_tile,
                umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), Int32(2), 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=4,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            ](ctx, Idx[1], Idx[6656], Idx[16384])

        # Kimi-K2.5 shape (k_group_size=1)
        comptime kimi_block_tile = Index(128, 8, BK)
        comptime kimi_umma = Index(256, 16, MMA_K)
        _test_impl[
            dtype,
            dtype,
            out_dtype,
            scales_dtype,
            kimi_block_tile,
            kimi_umma,
            cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
            cta_group=2,
            block_swizzle_size=8,
            swapAB=True,
            k_group_size=1,
            num_clc_pipeline_stages=0,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            prefetch_tiles_n=2,
        ](ctx, Idx[1], Idx[36864], Idx[128 * 28])

        # Epilogue fusion tests (2SM)
        print("\n--- 2SM Epilogue fusion tests ---")
        comptime for mma_n in [16, 32, 48, 64, 96]:
            comptime epi_block_tile = Index(128, mma_n // 2, BK)
            comptime epi_umma = Index(256, mma_n, MMA_K)

            run[
                dtype,
                dtype,
                out_dtype,
                scales_dtype,
                epi_block_tile,
                epi_umma,
                cluster_shape=StaticTuple[Int32, 3](Int32(2), 1, 1),
                cta_group=2,
                a_swizzle=swizzle,
                b_swizzle=swizzle,
                block_swizzle_size=8,
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                normal_epilogue=True,
            ](ctx, Idx[1], Idx[2304], Idx[16384])


def main() raises:
    run_matmul_sm100_block_scaled_fp4_small_bn_2sm_prefetch_suite[
        suite_scales_dtype=NVFP4_SF_DTYPE,
        suite_sf_vector_size=NVFP4_SF_VECTOR_SIZE,
    ]()
