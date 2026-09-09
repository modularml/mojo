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
from std.math import align_up
from std.sys import argv, size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import assert_almost_equal
from linalg.grouped_matmul_sm100_1d1d import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig, GEMMKind
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled_1d1d import (
    grouped_matmul_block_scaled,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig as StructuredBlockScaledMatmulConfig,
)
from std.math import ceildiv, align_up
from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
    set_scale_factor,
)
from std.random import random_ui64, seed, rand
from std.builtin.simd import _convert_f32_to_float8_ue8m0
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind


def simple_init() -> Bool:
    for arg in argv():
        if arg == "--simple-init":
            return True
    return False


def _test_kernel_impl[
    kernel_type: String,  # "old" or "new"
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    num_experts: Int,
    expert_shape: IndexList[2],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = MXFP8_SF_VECTOR_SIZE,
    test_atol: Float64 = 1e-2,
    test_rtol: Float64 = 1e-2,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    seed(1234)
    var total_num_tokens = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]

    var M = total_num_tokens
    var N = expert_shape[0]
    var K = expert_shape[1]

    print(
        t"[{kernel_type} kernel] in/out dtypes=({a_type}, {b_type}, {c_type},"
        t" {scales_dtype})  problem shape=({M}, {N}, {K})"
        t" mma_shape={mma_shape} block_tile_shape={block_tile_shape} cta_group={cta_group} cluster_shape=({cluster_shape[0]},"
        t" {cluster_shape[1]}, {cluster_shape[2]})"
        t" swapAB={swapAB} k_group_size={k_group_size} SF_VECTOR_SIZE={SF_VECTOR_SIZE}"
    )

    var a_shape = row_major(Coord(Int(total_num_tokens), Idx[expert_shape[1]]))
    var b_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[expert_shape[0]],
            Idx[expert_shape[1]],
        )
    )
    var c_shape = row_major(Coord(Int(total_num_tokens), Idx[expert_shape[0]]))

    var a_size = total_num_tokens * K
    var b_size = num_experts * expert_shape[0] * expert_shape[1]
    var c_size = total_num_tokens * expert_shape[0]

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
    var a_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var a_offsets_tensor = TileTensor(
        a_offsets_device,
        row_major(
            Coord(
                Int(num_active_experts + 1),
            )
        ),
    )
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var expert_ids_device = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var expert_ids_tensor = TileTensor(
        expert_ids_device,
        row_major(
            Coord(
                Int(num_active_experts),
            )
        ),
    )
    var a_scale_offsets_device = ctx.enqueue_create_buffer[.uint32](
        num_active_experts
    )
    var a_scale_offsets_tensor = TileTensor(
        a_scale_offsets_device,
        row_major(
            Coord(
                Int(num_active_experts),
            )
        ),
    )
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)
    var expert_scales_device = ctx.enqueue_create_buffer[.float32](num_experts)
    var expert_scales_tensor = TileTensor(
        expert_scales_device,
        row_major(
            Coord(
                Idx[num_experts],
            )
        ),
    )

    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var a_scale_offsets_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )
    var expert_scales_host_ptr = ctx.enqueue_create_host_buffer[.float32](
        num_experts
    )
    for i in range(num_experts):
        expert_scales_host_ptr[i] = 1.0 + Float32(i + 1) / Float32(num_experts)

    var a_scale_dim0 = 0
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_scale_offsets_ptr[i] = UInt32(
            a_scale_dim0
            - Int(a_offsets_host_ptr[i] // UInt32(SF_MN_GROUP_SIZE))
        )
        var local_m = num_tokens_by_expert[i]
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(local_m)
        a_scale_dim0 += ceildiv(local_m, SF_MN_GROUP_SIZE)
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    var a_scales_shape = row_major(
        Coord(
            Int(a_scale_dim0),
            Idx[ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1]],
            Idx[SF_ATOM_K],
        )
    )
    var b_scales_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[ceildiv(expert_shape[0], SF_MN_GROUP_SIZE)],
            Idx[ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)],
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

    # Initialize matmul operands
    if simple_init():
        for m in range(M):
            for k in range(K):
                a_host[m, k] = random_ui64(0, 1).cast[a_type]()
        for e in range(num_experts):
            for n in range(N):
                for k in range(K):
                    b_host[e, n, k] = random_ui64(0, 1).cast[b_type]()
    else:
        rand(a_host._storage, a_host.num_elements())
        rand(b_host._storage, b_host.num_elements())

    var a_scales_tensor_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_tensor_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    for i in range(a_scales_host.num_elements()):
        a_scales_host._storage[i] = Scalar[scales_dtype](0.0)
    # NOTE: It is very important that we set unused scales to 0.0 otherwise we will hit accuracy issues
    var effective_n = expert_shape[0]
    var effective_k = expert_shape[1]

    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        var local_m = end - start
        var actual_start = (
            start // SF_MN_GROUP_SIZE + Int(a_scale_offsets_ptr[i])
        ) * SF_MN_GROUP_SIZE
        var actual_end = actual_start + local_m
        for idx0 in range(actual_start, actual_end):
            for idx1 in range(
                0,
                align_up(effective_k, SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if idx1 < effective_k:
                    var scale_value = _convert_f32_to_float8_ue8m0[
                        scales_dtype
                    ]((1 << random_ui64(0, 2)).cast[.float32]())
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        a_scales_tensor_host, idx0, idx1, scale_value
                    )

    comptime k_groups = ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)
    comptime n_groups = ceildiv(expert_shape[0], SF_MN_GROUP_SIZE)
    for e in range(num_experts):
        var expert_slice_size = (
            Int(b_scales_host.dim(1))
            * Int(b_scales_host.dim(2))
            * Int(b_scales_host.dim(3))
            * Int(b_scales_host.dim(4))
            * Int(b_scales_host.dim(5))
        )
        var b_scales_tensor_expert_slice = TileTensor(
            b_scales_host_ptr.unsafe_ptr() + e * expert_slice_size,
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        for idx0 in range(align_up(effective_n, SF_MN_GROUP_SIZE)):
            for idx1 in range(
                0,
                align_up(effective_k, SF_VECTOR_SIZE * SF_ATOM_K),
                SF_VECTOR_SIZE,
            ):
                if idx0 < effective_n and idx1 < effective_k:
                    var scale_value = _convert_f32_to_float8_ue8m0[
                        scales_dtype
                    ]((1 << random_ui64(0, 2)).cast[.float32]())
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_scales_tensor_expert_slice, idx0, idx1, scale_value
                    )
                else:
                    set_scale_factor[SF_VECTOR_SIZE=SF_VECTOR_SIZE](
                        b_scales_tensor_expert_slice,
                        idx0,
                        idx1,
                        Scalar[scales_dtype](0.0),
                    )

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(a_offsets_device, a_offsets_host_ptr)
    ctx.enqueue_copy(a_scale_offsets_device, a_scale_offsets_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(expert_ids_device, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)
    ctx.enqueue_copy(expert_scales_device, expert_scales_host_ptr)

    # expert_scales_tensor already created above

    # Call appropriate kernel based on kernel_type parameter
    comptime if kernel_type == "old":
        # Old kernel using linalg.grouped_matmul_sm100_1d1d
        comptime matmul_config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=UMMAKind.KIND_MXF8F6F4,
            cluster_shape=Index(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            mma_shape=mma_shape,
            block_swizzle_size=block_swizzle_size,
            cta_group=cta_group,
            AB_swapped=swapAB,
            k_group_size=k_group_size,
            num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
            gemm_kind=GEMMKind.GMM,
        )

        blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=transpose_b,
            config=matmul_config,
        ](
            c_tensor,
            a_tensor,
            a_offsets_tensor,
            a_scale_offsets_tensor,
            b_tensor,
            expert_ids_tensor,
            a_scales_tensor,
            b_scales_tensor,
            expert_scales_tensor,
            num_active_experts,
            ctx,
        )
    elif kernel_type == "new":
        # New structured kernel using grouped_matmul_block_scaled
        comptime new_matmul_config = StructuredBlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=UMMAKind.KIND_MXF8F6F4,
            cluster_shape=Index(
                cluster_shape[0], cluster_shape[1], cluster_shape[2]
            ),
            mma_shape=mma_shape,
            block_swizzle_size=block_swizzle_size,
            cta_group=cta_group,
            AB_swapped=swapAB,
            k_group_size=k_group_size,
            num_accum_pipeline_stages=1 if mma_shape[1] == 256 else 2,
            is_gmm=True,
            gemm_kind=GEMMKind.GMM,
        )

        # Construct scale TileTensors from raw pointers with explicit
        comptime k_groups = ceildiv(expert_shape[1], SF_VECTOR_SIZE * SF_ATOM_K)
        comptime n_groups = ceildiv(expert_shape[0], SF_MN_GROUP_SIZE)
        var a_scales_tt = TileTensor(
            a_scales_device,
            row_major(
                Coord(
                    Int64(a_scale_dim0),
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()
        var b_scales_tt = TileTensor(
            b_scales_device,
            row_major(
                Coord(
                    Idx[num_experts],
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        ).as_unsafe_any_origin()
        var expert_scales_tt = TileTensor(
            expert_scales_device,
            row_major(
                Coord(
                    Int64(num_experts),
                )
            ),
        ).as_unsafe_any_origin()

        grouped_matmul_block_scaled[
            transpose_b=transpose_b,
            config=new_matmul_config,
        ](
            c_tensor,
            a_tensor,
            a_offsets_tensor,
            a_scale_offsets_tensor,
            b_tensor,
            expert_ids_tensor,
            a_scales_tt,
            b_scales_tt,
            expert_scales_tt,
            num_active_experts,
            ctx,
        )
    else:
        comptime assert False, "kernel_type must be 'old' or 'new'"
        pass

    comptime assert a_type != .float8_e4m3fn or transpose_b, (
        "Testing is only supported for transposed_b==True when"
        " a_type==float8_e4m3fn. Add the non-transposed case if needed."
    )

    var c_row_stride = expert_shape[0]
    var a_row_stride = expert_shape[1]
    comptime b_expert_stride = expert_shape[0] * expert_shape[1]
    comptime b_scales_expert_stride = n_groups * k_groups * SF_ATOM_M[
        0
    ] * SF_ATOM_M[1] * SF_ATOM_K
    comptime a_scales_row_stride = k_groups * SF_ATOM_M[0] * SF_ATOM_M[
        1
    ] * SF_ATOM_K

    for i in range(num_active_experts):
        var start = Int(a_offsets_host_ptr[i])
        var end = Int(a_offsets_host_ptr[i + 1])
        var expert_id = expert_ids_host_ptr[i]

        var c_slice = TileTensor(
            c_ref_tensor.ptr + start * c_row_stride,
            row_major((end - start, Idx[expert_shape[0]])),
        )

        var new_a_tensor = TileTensor(
            a_tensor.ptr + start * a_row_stride,
            row_major((end - start, Idx[expert_shape[1]])),
        )

        var new_b_tensor = TileTensor(
            b_tensor.ptr + Int(expert_id) * b_expert_stride,
            row_major((Idx[expert_shape[0]], Idx[expert_shape[1]])),
        )

        var new_b_scales_tensor = TileTensor(
            b_scales_tensor.ptr + Int(expert_id) * b_scales_expert_stride,
            row_major(
                Coord(
                    Idx[n_groups],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )

        var a_scales_start = start // SF_MN_GROUP_SIZE + Int(
            a_scale_offsets_ptr[i]
        )
        var new_a_scales_tensor = TileTensor(
            a_scales_tensor.ptr + a_scales_start * a_scales_row_stride,
            row_major(
                Coord(
                    ceildiv(end - start, SF_MN_GROUP_SIZE),
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )

        var expert_scale = expert_scales_host_ptr[Int(expert_id)]
        vendor_blas.matmul(
            ctx,
            c_slice,
            new_a_tensor,
            new_b_tensor,
            a_scales=new_a_scales_tensor,
            b_scales=new_b_scales_tensor,
            transpose_b=transpose_b,
            c_row_major=True,
            alpha=expert_scale,
        )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=test_atol,
        rtol=test_rtol,
    )
    print("\n=== TEST PASSED ===\n")

    # Cleanup
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^
    _ = a_offsets_device^
    _ = a_scale_offsets_device^
    _ = expert_ids_device^
    _ = expert_scales_device^


# Backward-compatible wrapper that maintains the original function name
def test_blackwell_block_scaled_matmul_tma_umma_warp_specialized[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cluster_shape: StaticTuple[Int32, 3],
    cta_group: Int,
    num_experts: Int,
    expert_shape: IndexList[2],
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    block_swizzle_size: Int = 0,
    benchmark: Bool = False,
    swapAB: Bool = False,
    k_group_size: Int = 1,
    SF_VECTOR_SIZE: Int = MXFP8_SF_VECTOR_SIZE,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    """Test old kernel - backward compatible wrapper."""
    _test_kernel_impl[
        "old",
        a_type,
        b_type,
        c_type,
        scales_dtype,
        block_tile_shape,
        mma_shape,
        cluster_shape,
        cta_group,
        num_experts,
        expert_shape,
        transpose_b,
        a_swizzle,
        b_swizzle,
        c_swizzle,
        block_swizzle_size,
        benchmark,
        swapAB,
        k_group_size,
        SF_VECTOR_SIZE,
    ](num_active_experts, num_tokens_by_expert, expert_ids, ctx)


def main() raises:
    with DeviceContext() as ctx:
        comptime dtype = DType.float8_e4m3fn
        comptime out_dtype = DType.bfloat16
        comptime scale_dtype = MXFP8_SF_DTYPE
        comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime BK = (swizzle.bytes() // size_of[dtype]())
        comptime MMA_K = 32
        comptime bm = 128
        comptime bn = 128
        comptime block_tile_shape = Index(bm, bn, BK)
        comptime umma_shape = Index(bm, bn, MMA_K)

        comptime for structured in [False, True]:
            comptime if structured:
                print("\n========================================")
                print(
                    "Testing NEW kernel (grouped_matmul_block_scaled with"
                    " MXFP8)"
                )
                print("========================================\n")
            else:
                print("\n========================================")
                print(
                    "Testing OLD kernel"
                    " (blackwell_block_scaled_matmul_tma_umma_warp_specialized)"
                )
                print("========================================\n")

            comptime kernel_type = "new" if structured else "old"

            comptime for swapAB in [False, True]:
                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape=Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    3,
                    [128, 512, 1024],
                    [0, 1, 1],
                    ctx,
                )

                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape=Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    3,
                    [64 + 1, 1024 + 3, 128 * 3 + 2],
                    [2, 0, 1],
                    ctx,
                )

                _test_kernel_impl[
                    kernel_type,
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape,
                    cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
                    cta_group=1,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape=Index(2048, 1024),
                    swapAB=swapAB,
                ](
                    4,
                    [512, 1000, 2000, 3000],
                    [0, 3, 2, 4],
                    ctx,
                )

            # 2SM tests (new structured kernel only, swapAB=True required)
            comptime if structured:
                comptime umma_shape_2sm = Index(2 * bm, 2 * bn, MMA_K)

                # 2SM: Aligned token counts
                _test_kernel_impl[
                    "new",
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape_2sm,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape=Index(2048, 1024),
                    swapAB=True,
                ](
                    3,
                    [128, 512, 1024],
                    [0, 1, 1],
                    ctx,
                )

                # 2SM: Unaligned token counts
                _test_kernel_impl[
                    "new",
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape_2sm,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=4,
                    expert_shape=Index(2048, 1024),
                    swapAB=True,
                ](
                    3,
                    [64 + 1, 1024 + 3, 128 * 3 + 2],
                    [2, 0, 1],
                    ctx,
                )

                # 2SM: Large token counts
                _test_kernel_impl[
                    "new",
                    dtype,
                    dtype,
                    out_dtype,
                    scale_dtype,
                    block_tile_shape,
                    umma_shape_2sm,
                    cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
                    cta_group=2,
                    a_swizzle=swizzle,
                    b_swizzle=swizzle,
                    block_swizzle_size=8,
                    num_experts=6,
                    expert_shape=Index(2048, 1024),
                    swapAB=True,
                ](
                    4,
                    [512, 1000, 2000, 3000],
                    [0, 3, 2, 4],
                    ctx,
                )

        # MMA_N=64 tests (new structured kernel only)
        print("\n========================================")
        print("Testing NEW kernel with MMA_N=64 (MXFP8)")
        print("========================================\n")

        comptime umma_shape_n64 = Index(bm, 64, MMA_K)
        comptime block_tile_shape_n64 = Index(bm, 64, BK)

        # MMA_N=64 1SM non-swapped: Aligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n64,
            umma_shape_n64,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=2,
            expert_shape=Index(128, 1024),
            swapAB=False,
        ](
            1,
            [128],
            [0],
            ctx,
        )

        # MMA_N=64 1SM AB_swapped: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n64,
            umma_shape_n64,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=64: Unaligned token counts (non-swapped)
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n64,
            umma_shape_n64,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=64: Small token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n64,
            umma_shape_n64,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
        ](
            3,
            [31, 97, 63],
            [2, 0, 1],
            ctx,
        )

        # --- MMA_N=64 2SM AB_swapped ---
        print("\n========================================")
        print("Testing NEW kernel MMA_N=64 (2SM, AB_swapped, MXFP8)")
        print("========================================\n")

        # mma_n=64, cta_group=2: block_tile_shape auto-derived by config
        comptime umma_shape_2sm_n64 = Index(2 * bm, 64, MMA_K)
        comptime block_tile_shape_2sm_n64 = Index(bm, 64 // 2, BK)

        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_2sm_n64,
            umma_shape_2sm_n64,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=2,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=6,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_2sm_n64,
            umma_shape_2sm_n64,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            cta_group=2,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=8 tests (new structured kernel only, small BN, MXFP8)
        print("\n========================================")
        print("Testing NEW kernel with MMA_N=8 (MXFP8)")
        print("========================================\n")

        comptime umma_shape_n8 = Index(bm, 8, MMA_K)
        comptime block_tile_shape_n8 = Index(bm, 8, BK)

        # MMA_N=8: Aligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            # MMA_N=8 produces slightly different rounding than cuBLAS
            # due to smaller tile accumulation; allow wider tolerance.
            test_rtol=0.08,
        ](
            3,
            [128, 512, 1024],
            [0, 1, 1],
            ctx,
        )

        # MMA_N=8: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            test_rtol=0.08,
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=8: Small token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            test_rtol=0.08,
        ](
            3,
            [31, 97, 63],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=8: cp.async SFB (group_size < 128)
        # Single token per expert
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            test_rtol=0.08,
        ](
            4,
            [1, 2, 3, 4],
            [0, 1, 2, 3],
            ctx,
        )

        # MMA_N=8: Mixed cp.async (gs<128) + TMA (gs>=128)
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            test_rtol=0.08,
        ](
            4,
            [4, 256, 1, 200],
            [0, 1, 2, 3],
            ctx,
        )

        # MMA_N=8: Boundary gs=127/128
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            test_rtol=0.08,
        ](
            4,
            [127, 128, 64, 256],
            [0, 1, 2, 3],
            ctx,
        )

        # MMA_N=16 tests (new structured kernel only, small BN, MXFP8)
        print("\n========================================")
        print("Testing NEW kernel with MMA_N=16 (MXFP8)")
        print("========================================\n")

        comptime umma_shape_n16 = Index(bm, 16, MMA_K)
        comptime block_tile_shape_n16 = Index(bm, 16, BK)

        # MMA_N=16: Large token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n16,
            umma_shape_n16,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=6,
            expert_shape=Index(2048, 1024),
        ](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        # MMA_N=16: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n16,
            umma_shape_n16,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=16: cp.async SFB (group_size < 128)
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n16,
            umma_shape_n16,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
        ](
            4,
            [1, 64, 4, 256],
            [0, 1, 2, 3],
            ctx,
        )

        # MMA_N=32 tests (new structured kernel only, small BN, MXFP8)
        print("\n========================================")
        print("Testing NEW kernel with MMA_N=32 (MXFP8)")
        print("========================================\n")

        comptime umma_shape_n32 = Index(bm, 32, MMA_K)
        comptime block_tile_shape_n32 = Index(bm, 32, BK)

        # MMA_N=32: Large token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n32,
            umma_shape_n32,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=6,
            expert_shape=Index(2048, 1024),
        ](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        # MMA_N=32: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n32,
            umma_shape_n32,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=32: cp.async SFB (group_size < 128)
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n32,
            umma_shape_n32,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
        ](
            4,
            [1, 64, 32, 256],
            [0, 1, 2, 3],
            ctx,
        )

        # --- Small MMA_N AB_swapped tests (MXFP8) ---
        print("\n========================================")
        print("Testing NEW kernel small MMA_N AB_swapped (MXFP8)")
        print("========================================\n")

        # MMA_N=8 AB_swapped: Aligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
            test_rtol=0.08,
        ](
            3,
            [128, 512, 1024],
            [0, 1, 1],
            ctx,
        )

        # MMA_N=8 AB_swapped: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
            test_rtol=0.08,
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=8 AB_swapped: Large token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=6,
            expert_shape=Index(2048, 1024),
            swapAB=True,
            test_rtol=0.08,
        ](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        # MMA_N=8 AB_swapped: cp.async SFB (mixed small/large)
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n8,
            umma_shape_n8,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
            test_rtol=0.08,
        ](
            4,
            [4, 256, 1, 200],
            [0, 1, 2, 3],
            ctx,
        )

        # MMA_N=16 AB_swapped: Large token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n16,
            umma_shape_n16,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=6,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        # MMA_N=16 AB_swapped: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n16,
            umma_shape_n16,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        # MMA_N=32 AB_swapped: Large token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n32,
            umma_shape_n32,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=6,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            4,
            [512, 1000, 2000, 3000],
            [0, 3, 2, 4],
            ctx,
        )

        # MMA_N=32 AB_swapped: Unaligned token counts
        _test_kernel_impl[
            "new",
            dtype,
            dtype,
            out_dtype,
            scale_dtype,
            block_tile_shape_n32,
            umma_shape_n32,
            cluster_shape=StaticTuple[Int32, 3](1, 1, 1),
            cta_group=1,
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            block_swizzle_size=8,
            num_experts=4,
            expert_shape=Index(2048, 1024),
            swapAB=True,
        ](
            3,
            [64 + 1, 1024 + 3, 128 * 3 + 2],
            [2, 0, 1],
            ctx,
        )

        print("\n========================================")
        print("ALL TESTS PASSED!")
        print("========================================")
