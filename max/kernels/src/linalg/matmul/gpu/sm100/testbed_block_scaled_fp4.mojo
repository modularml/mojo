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
"""Shared test harness for SM100 block-scaled FP4 matmul tests.

Supports both small_bn (1SM / 2SM cooperative) and structured 2SM kernels
via the `is_small_bn` compile-time parameter, and both NVFP4 and MXFP4
scaling formats via the `scales_dtype` parameter.
"""
from std.math import align_up, ceildiv
from std.sys import argv
import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc, bitcast, dealloc, ThinAllocation
from std.memory.alloc import Layout as AllocLayout
from std.random import rand, random_ui64, seed
from internal_utils import assert_almost_equal
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from .block_scaled_matmul_small_bn import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized as _small_bn_kernel,
)
from ..sm100_structured.block_scaled.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized as _structured_kernel,
)
from .config import BlockScaledMatmulConfig
from std.sys import size_of
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.utils import elementwise_epilogue_type
from linalg.fp4_utils import (
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    set_scale_factor,
)
from linalg.block_scaled_quantization import naive_block_scaled_matmul
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def _rand_mxfp4_value() -> Scalar[MXFP4_SF_DTYPE]:
    # E8M0 is all-exponent: valid non-zero, non-NaN values are 0x01..0xFE.
    return bitcast[MXFP4_SF_DTYPE](random_ui64(1, 254).cast[.uint8]())


def _rand_mxfp4[
    scales_dtype: DType
](ptr: UnsafePointer[mut=True, Scalar[scales_dtype], ...], size: Int):
    comptime assert (
        scales_dtype == MXFP4_SF_DTYPE
    ), "_rand_mxfp4 only supports MXFP4 scales dtype"
    for i in range(size):
        ptr[i] = _rand_mxfp4_value().cast[scales_dtype]()


def simple_init() -> Bool:
    """Returns whether the `--simple-init` flag was passed on the command line.

    When enabled, matmul operands are filled with deterministic values derived
    from their indices instead of random data, which simplifies debugging.
    """
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
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
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    num_clc_pipeline_stages: Int = 2,
    SF_VECTOR_SIZE: Int = NVFP4_SF_VECTOR_SIZE,
    is_small_bn: Bool = False,
    normal_epilogue: Bool = False,
](
    ctx: DeviceContext,
    m: MType,
    n: NType,
    k: KType,
    alpha: Float32 = 1.0,
) raises:
    """Runs a block-scaled FP4 matmul on SM100 and checks it against a reference.

    Allocates and initializes the FP4 operands and their scale-factor tensors,
    dispatches either the small-BN or structured 2SM kernel based on
    `is_small_bn`, computes a reference output via `naive_block_scaled_matmul`
    (MXFP4) or a vendor BLAS matmul (NVFP4), and asserts the kernel output
    matches the reference within tolerance.

    Parameters:
        MType: Coord type carrying the M dimension of the matmul (inferred).
        NType: Coord type carrying the N dimension of the matmul (inferred).
        KType: Coord type carrying the K dimension of the matmul (inferred).
        a_type: Element dtype of the left operand (packed FP4 pairs).
        b_type: Element dtype of the right operand (packed FP4 pairs).
        c_type: Element dtype of the output tensor.
        scales_dtype: Scale-factor dtype, selecting NVFP4 or MXFP4 format.
        block_tile_shape: Per-CTA tile shape over (M, N, K) dimensions.
        mma_shape: Hardware MMA shape over (M, N, K) dimensions.
        cluster_shape: Thread-block cluster shape along (X, Y, Z).
        cta_group: Number of CTAs cooperating per output tile.
        transpose_b: Whether the right operand is stored transposed.
        a_swizzle: TMA swizzle pattern for the left operand.
        b_swizzle: TMA swizzle pattern for the right operand.
        c_swizzle: TMA swizzle pattern for the output tensor.
        block_swizzle_size: Block-level swizzle stride, or 0 to disable.
        benchmark: Whether the invocation is being run under a benchmark harness.
        swapAB: Whether to swap the A and B operands before the matmul.
        k_group_size: Number of K tiles grouped together for accumulation.
        num_clc_pipeline_stages: Number of CLC pipeline stages to use.
        SF_VECTOR_SIZE: Number of FP4 elements covered by one scale factor.
        is_small_bn: Selects the small-BN (1SM/2SM cooperative) kernel variant.
        normal_epilogue: Applies a 2x scaling epilogue to verify the lambda runs.

    Args:
        ctx: Device context used for allocation and kernel dispatch.
        m: M dimension of the matmul.
        n: N dimension of the matmul.
        k: K dimension of the matmul.
        alpha: Scalar multiplier applied to the matmul result.
    """
    seed(42)
    print(
        t"in/out dtypes=({a_type}, {b_type}, {c_type}, {scales_dtype})  problem"
        t" shape=({m.value()}, {n.value()}, {k.value()})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape}"
        t" cta_group={cta_group} cluster_shape=({cluster_shape[0]},"
        t" {cluster_shape[1]}, {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size}"
        t" SF_VECTOR_SIZE={SF_VECTOR_SIZE} is_small_bn={is_small_bn}"
        t" alpha={alpha}"
    )

    # Infer scaling_kind from scales_dtype.
    comptime scaling_kind = UMMAKind.KIND_MXF4 if scales_dtype == MXFP4_SF_DTYPE else UMMAKind.KIND_MXF4NVF4

    var a_shape = row_major(Coord(m, Idx[KType.static_value // 2]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value // 2])
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * (KType.static_value // 2)
    var b_size = Int(n.value()) * (KType.static_value // 2)
    var c_size = Int(m.value()) * Int(n.value())

    var a_host_alloc = alloc(
        AllocLayout[Scalar[a_type]](count=a_size)
    ).into_managed()
    var a_host_ptr: UnsafePointer[
        Scalar[a_type], origin_of(a_host_alloc)
    ] = a_host_alloc.unsafe_ptr()
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_alloc = alloc(
        AllocLayout[Scalar[b_type]](count=b_size)
    ).into_managed()
    var b_host_ptr: UnsafePointer[
        Scalar[b_type], origin_of(b_host_alloc)
    ] = b_host_alloc.unsafe_ptr()
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_alloc = alloc(
        AllocLayout[Scalar[c_type]](count=c_size)
    ).into_managed()
    var c_host = TileTensor(c_host_alloc.unsafe_ptr(), c_shape)
    var c_host_ref_alloc = alloc(
        AllocLayout[Scalar[c_type]](count=c_size)
    ).into_managed()
    var c_host_ref_ptr: UnsafePointer[
        Scalar[c_type], origin_of(c_host_ref_alloc)
    ] = c_host_ref_alloc.unsafe_ptr()
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    # This row major layout correlates to this
    # https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#tcgen05-mma-scale-factor-a-layout-4x

    # Dim 0: the scale factors cover batches of 128 rows (4 sets of 32 rows to be specific) so divide to find out how
    # tiles we have over the first mode

    # Dim 1: Assuming NVFP4_SF_VECTOR_SIZE for SF_VECTOR_SIZE, we know each scale factor covers 16 elements. The MMA has K fixed to 64 (32 in fp8),
    # so we divide K by 64 (4 scales) and we get the batch of scales for each mma across that mode.

    # Dim 2: Now in each batch as previously mentioned we have 32 rows
    # Dim 3: each column in the row is actually a subrow there are a total of 4 (32 * 4 gives us 128)
    # Dim 4: each subrow has 4 scale factors.

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
            ceildiv(Int(n.value()), SF_MN_GROUP_SIZE),
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )

    var a_scales_total = a_scales_shape.product()
    var b_scales_total = b_scales_shape.product()

    var a_scales_host_alloc = alloc(
        AllocLayout[Scalar[scales_dtype]](count=a_scales_total)
    ).into_managed()
    var a_scales_host_ptr: UnsafePointer[
        Scalar[scales_dtype], origin_of(a_scales_host_alloc)
    ] = a_scales_host_alloc.unsafe_ptr()
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host_alloc = alloc(
        AllocLayout[Scalar[scales_dtype]](count=b_scales_total)
    ).into_managed()
    var b_scales_host_ptr: UnsafePointer[
        Scalar[scales_dtype], origin_of(b_scales_host_alloc)
    ] = b_scales_host_alloc.unsafe_ptr()
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    var a_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        a_scales_total
    )
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_device = ctx.enqueue_create_buffer[scales_dtype](
        b_scales_total
    )
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    # Initialize matmul operands using TileTensor host views before moving them
    # to the device.
    if simple_init():
        for m in range(Int(m.value())):
            for k in range(Int(k.value()) // 2):
                comptime assert a_host.flat_rank == 2
                a_host[m, k] = UInt8(m).cast[a_type]()
        for n in range(Int(n.value())):
            for k in range(Int(k.value()) // 2):
                comptime assert b_host.flat_rank >= 2
                b_host[n, k] = UInt8(n).cast[b_type]()
    else:
        rand(a_host.ptr, a_host.num_elements(), min=0, max=255)
        rand(b_host.ptr, b_host.num_elements(), min=0, max=255)

    comptime if scales_dtype == NVFP4_SF_DTYPE:
        rand(a_scales_host.ptr, a_scales_host.num_elements())
        rand(b_scales_host.ptr, b_scales_host.num_elements())
    elif scales_dtype == MXFP4_SF_DTYPE:
        _rand_mxfp4(a_scales_host.ptr, a_scales_host.num_elements())
        _rand_mxfp4(b_scales_host.ptr, b_scales_host.num_elements())
    else:
        comptime assert False, "Unsupported scales_dtype in FP4 testbed"

    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
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

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_alloc.unsafe_ptr())
    ctx.enqueue_copy(b_device, b_host_alloc.unsafe_ptr())
    ctx.enqueue_copy(a_scales_device, a_scales_host_alloc.unsafe_ptr())
    ctx.enqueue_copy(b_scales_device, b_scales_host_alloc.unsafe_ptr())

    comptime matmul_config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=scaling_kind,
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=1 if mma_shape[1] in (192, 256) else 2,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        is_small_bn=is_small_bn,
    )

    # Epilogue multiplies output by 2 so we can verify the lambda is actually
    # invoked — if TileWriter skips the lambda the result will be 1x, not 2x,
    # and the comparison against 2x reference will fail.
    @__parameter
    @always_inline
    @__copy_capture(c_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = 1,
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        var scaled = rebind[SIMD[c_type, width]](val) * Scalar[c_type](2)
        c_tensor.store_linear[alignment=alignment * size_of[c_type](),](
            idx, scaled
        )

    comptime epi = Optional[elementwise_epilogue_type](
        epilogue_fn
    ) if normal_epilogue else None

    comptime if is_small_bn:
        comptime K_phys = KType.static_value
        _small_bn_kernel[
            transpose_b=transpose_b,
            config=matmul_config,
            K=K_phys,
            elementwise_lambda_fn=epi,
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            ctx,
            alpha,
        )
    else:
        _structured_kernel[
            transpose_b=transpose_b,
            config=matmul_config,
        ](
            c_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            ctx,
            alpha=alpha,
        )

    comptime if scales_dtype == MXFP4_SF_DTYPE:
        naive_block_scaled_matmul[
            scaling_kind=UMMAKind.KIND_MXF4,
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
        ](
            c_ref_tensor,
            a_tensor,
            b_tensor,
            a_scales_tensor,
            b_scales_tensor,
            ctx,
            alpha,
        )
    else:
        vendor_blas.matmul(
            ctx,
            c_ref_tensor,
            a_tensor,
            b_tensor,
            a_scales=a_scales_tensor,
            b_scales=b_scales_tensor,
            transpose_b=transpose_b,
            c_row_major=True,
            alpha=alpha,
        )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_alloc.unsafe_ptr(), c_device)
    ctx.enqueue_copy(c_host_ref_alloc.unsafe_ptr(), c_device_ref)
    ctx.synchronize()

    # When epilogue multiplies by 2, scale reference to match.
    comptime if normal_epilogue:
        for i in range(c_host_ref.num_elements()):
            c_host_ref.raw_store(i, c_host_ref.raw_load(i) * Scalar[c_type](2))

    assert_almost_equal(
        c_host.ptr,
        c_host_ref.ptr,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    dealloc(a_host_alloc^)
    dealloc(b_host_alloc^)
    dealloc(c_host_alloc^)
    dealloc(c_host_ref_alloc^)
    dealloc(a_scales_host_alloc^)
    dealloc(b_scales_host_alloc^)
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^
