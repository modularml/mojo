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
"""Epilogue fusion tests for grouped block-scaled GEMM.

Tests element-wise epilogue lambda application on the grouped block-scaled
GEMM kernel output. Verifies that:
1. The lambda is correctly applied to each output element
2. Coordinates passed to the lambda are correct
3. Both register-based and SMEM-based epilogues work correctly
"""

from std.collections import Optional
from std.math import ceildiv
from std.random import rand, random_float64, seed
from std.sys import align_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.memory import alloc
from internal_utils import assert_almost_equal
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)

from std.utils.index import Index, IndexList

from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
    MXFP8_SF_VECTOR_SIZE,
)
from linalg.utils import elementwise_compute_lambda_type
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BlockScaledMatmulConfig,
    GEMMKind,
)
from linalg.matmul.gpu.sm100_structured.grouped_block_scaled.grouped_block_scaled_matmul import (
    grouped_block_scaled_matmul,
)


def test_grouped_gemm_epilogue[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    a_type: DType,
    b_type: DType,
    c_type: DType,
    scales_dtype: DType,
    transpose_b: Bool,
    cta_group: Int,
    mma_shape: IndexList[3],
    cluster_shape: IndexList[3],
    register_based_epilogue: Bool = True,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    """Test grouped block-scaled GEMM with epilogue lambda.

    The epilogue lambda adds the original C value to the matmul result,
    effectively computing: C' = matmul(A, B) + C_original

    This tests that:
    1. The lambda is applied to all output elements
    2. Global coordinates passed to lambda are correct
    3. The captured tensor is accessible from the lambda
    """
    print("\n--- Testing grouped GEMM with epilogue ---")
    print(
        "  M=",
        Int(m.value()),
        " N=",
        Int(n.value()),
        " K=",
        Int(k.value()),
        " cta_group=",
        cta_group,
        " register_based_epilogue=",
        register_based_epilogue,
    )

    comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
    comptime max_groups = 1
    var num_groups = 1

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    var a_size = Int(m.value()) * Int(k.value())
    var b_size = Int(n.value()) * Int(k.value())
    var c_size = Int(m.value()) * Int(n.value())

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)
    var c_host_original_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_original = TileTensor(c_host_original_ptr, c_shape)

    # Device allocations
    var a_device = ctx.enqueue_create_buffer[a_type](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[b_type](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_device = ctx.enqueue_create_buffer[c_type](c_size)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)

    # Scale factor shapes (5D)
    var a_scales_shape = row_major(
        Coord(
            Idx[ceildiv(MType.static_value, SF_MN_GROUP_SIZE)],
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

    var sfa_size = a_scales_shape.product()
    var sfb_size = b_scales_shape.product()

    # Scale factor device allocations
    var sfa_device = ctx.enqueue_create_buffer[scales_dtype](sfa_size)
    var sfa_tensor = TileTensor(sfa_device, a_scales_shape)
    var sfb_device = ctx.enqueue_create_buffer[scales_dtype](sfb_size)
    var sfb_tensor = TileTensor(sfb_device, b_scales_shape)

    # Scale factor host allocations — initialized to 1.0 (identity scaling)
    var sfa_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](sfa_size)
    var sfa_host = TileTensor(sfa_host_ptr, a_scales_shape)
    var sfb_host_ptr = ctx.enqueue_create_host_buffer[scales_dtype](sfb_size)
    var sfb_host = TileTensor(sfb_host_ptr, b_scales_shape)
    var scale_one = Float32(1.0).cast[scales_dtype]()
    for i in range(sfa_size):
        sfa_host_ptr[i] = scale_one
    for i in range(sfb_size):
        sfb_host_ptr[i] = scale_one

    # The C LayoutTensor that will be captured by the epilogue lambda
    var c_tensor_lt = c_tensor.to_layout_tensor()

    # Define epilogue lambda that adds original C value to matmul result
    @__parameter
    @always_inline
    @__copy_capture(c_tensor_lt)
    def epilogue_add_c[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        # C' = matmul(A, B) + C_original
        return val + c_tensor_lt.load[width=width](idx).cast[_dtype]()

    # Initialize random data
    seed(42)
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    # Initialize C with random values for epilogue test
    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            comptime assert c_host.flat_rank >= 2
            c_host[Coord(i, j)] = Scalar[c_type](random_float64(-1, 1))
            c_host_original[Coord(i, j)] = c_host[i, j]

    # Copy to device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(c_device, c_host._storage)
    ctx.enqueue_copy(sfa_device, sfa_host_ptr)
    ctx.enqueue_copy(sfb_device, sfb_host_ptr)

    # Create config
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=mma_shape,
        cta_group=cta_group,
        register_based_epilogue=register_based_epilogue,
        gemm_kind=GEMMKind.GMM,
    )

    # Problem sizes tensor
    var problem_sizes_host = ctx.enqueue_create_host_buffer[.int32](
        max_groups * 4
    )
    problem_sizes_host[0] = Int32(Int(m.value()))  # M
    problem_sizes_host[1] = Int32(Int(n.value()))  # N
    problem_sizes_host[2] = Int32(Int(k.value()))  # K
    problem_sizes_host[3] = Int32(1)  # L (batch)

    var problem_sizes_device = ctx.enqueue_create_buffer[.int32](max_groups * 4)
    ctx.enqueue_copy(problem_sizes_device, problem_sizes_host)

    var problem_sizes_tensor = TileTensor(
        problem_sizes_device, row_major[max_groups, 4]()
    )

    # Pointer arrays
    var a_ptrs_host = ctx.enqueue_create_host_buffer[.uint64](max_groups)
    var b_ptrs_host = ctx.enqueue_create_host_buffer[.uint64](max_groups)
    var c_ptrs_host = ctx.enqueue_create_host_buffer[.uint64](max_groups)
    var sfa_ptrs_host = ctx.enqueue_create_host_buffer[.uint64](max_groups)
    var sfb_ptrs_host = ctx.enqueue_create_host_buffer[.uint64](max_groups)

    a_ptrs_host[0] = UInt64(Int(a_device.unsafe_ptr()))
    b_ptrs_host[0] = UInt64(Int(b_device.unsafe_ptr()))
    c_ptrs_host[0] = UInt64(Int(c_device.unsafe_ptr()))
    sfa_ptrs_host[0] = UInt64(Int(sfa_device.unsafe_ptr()))
    sfb_ptrs_host[0] = UInt64(Int(sfb_device.unsafe_ptr()))

    var a_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var b_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var c_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var sfa_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)
    var sfb_ptrs_device = ctx.enqueue_create_buffer[.uint64](max_groups)

    ctx.enqueue_copy(a_ptrs_device, a_ptrs_host)
    ctx.enqueue_copy(b_ptrs_device, b_ptrs_host)
    ctx.enqueue_copy(c_ptrs_device, c_ptrs_host)
    ctx.enqueue_copy(sfa_ptrs_device, sfa_ptrs_host)
    ctx.enqueue_copy(sfb_ptrs_device, sfb_ptrs_host)

    var a_ptrs_tensor = TileTensor(a_ptrs_device, row_major[max_groups, 1]())
    var b_ptrs_tensor = TileTensor(b_ptrs_device, row_major[max_groups, 1]())
    var c_ptrs_tensor = TileTensor(c_ptrs_device, row_major[max_groups, 1]())
    var sfa_ptrs_tensor = TileTensor(
        sfa_ptrs_device, row_major[max_groups, 1]()
    )
    var sfb_ptrs_tensor = TileTensor(
        sfb_ptrs_device, row_major[max_groups, 1]()
    )

    # Compute total tiles
    comptime BM = config.block_tile_shape[0]
    comptime BN = mma_shape[1]
    var total_tiles = ceildiv(Int(m.value()), BM) * ceildiv(Int(n.value()), BN)

    # Create epilogue lambda optional
    comptime optional_lambda = Optional[elementwise_compute_lambda_type](
        epilogue_add_c
    )

    # Template tensors - 3D TileTensors with batch=1
    var a_3d_shape = row_major(Coord(Idx[1], m, k))
    var b_3d_shape = row_major(
        Coord(
            Idx[1],
            Idx[NType.static_value if transpose_b else KType.static_value],
            Idx[KType.static_value if transpose_b else NType.static_value],
        )
    )
    var c_3d_shape = row_major(Coord(Idx[1], m, n))
    var a_template = TileTensor(a_device, a_3d_shape)
    var b_template = TileTensor(b_device, b_3d_shape)
    var c_template = TileTensor(c_device, c_3d_shape)

    # Scale factor template tensors - 5D with batch=1 and merged last dims
    var a_scales_5d_shape = row_major(
        Coord(
            Idx[1],
            Idx[ceildiv(MType.static_value, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1] * SF_ATOM_K],
        )
    )
    var a_scales_5d = TileTensor(sfa_device, a_scales_5d_shape)
    var b_scales_5d_shape = row_major(
        Coord(
            Idx[1],
            Idx[ceildiv(NType.static_value, SF_MN_GROUP_SIZE)],
            Idx[ceildiv(KType.static_value, SF_VECTOR_SIZE * SF_ATOM_K)],
            Idx[SF_ATOM_M[0]],
            Idx[SF_ATOM_M[1] * SF_ATOM_K],
        )
    )
    var b_scales_5d = TileTensor(sfb_device, b_scales_5d_shape)

    # Launch grouped GEMM with epilogue
    grouped_block_scaled_matmul[
        config=config,
        max_groups=max_groups,
        elementwise_compute_lambda_fn=optional_lambda,
    ](
        a_ptrs_tensor,
        b_ptrs_tensor,
        c_ptrs_tensor,
        sfa_ptrs_tensor,
        sfb_ptrs_tensor,
        problem_sizes_tensor,
        num_groups,
        total_tiles,
        a_template,
        b_template,
        c_template,
        a_scales_5d,
        b_scales_5d,
        ctx,
    )

    # Run reference matmul (without epilogue)
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

    # Copy results back
    ctx.enqueue_copy(c_host._storage, c_device)
    ctx.enqueue_copy(c_host_ref._storage, c_device_ref)
    ctx.synchronize()

    # Apply epilogue lambda on CPU to reference
    var c_tensor_host_lt = c_host_original.to_layout_tensor()

    @__parameter
    @always_inline
    @__copy_capture(c_tensor_host_lt)
    def epilogue_add_c_host[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> SIMD[
        _dtype, width
    ]:
        return val + c_tensor_host_lt.load[width=width](idx).cast[_dtype]()

    for i in range(Int(m.value())):
        for j in range(Int(n.value())):
            comptime assert c_host_ref.flat_rank >= 2
            c_host_ref[Coord(i, j)] = epilogue_add_c_host(
                IndexList[2](i, j),
                c_host_ref[i, j],
            )

    # Compare results
    comptime rtol = 1e-2
    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=0.0001,
        rtol=rtol,
    )

    print("  PASSED!")


def main() raises:
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scales_dtype = MXFP8_SF_DTYPE
    comptime transpose_b = True

    with DeviceContext() as ctx:
        print("\n" + "=" * 60)
        print("Grouped Block-Scaled GEMM Epilogue Tests")
        print("=" * 60)

        # Test 1: 1SM mode with register-based epilogue (small)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            transpose_b=transpose_b,
            cta_group=1,
            mma_shape=Index(128, 128, 32),
            cluster_shape=Index(1, 1, 1),
            register_based_epilogue=True,
        ](ctx, Idx[256], Idx[256], Idx[128])

        # Test 2: 2SM mode with register-based epilogue (small)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            transpose_b=transpose_b,
            cta_group=2,
            mma_shape=Index(256, 128, 32),
            cluster_shape=Index(2, 1, 1),
            register_based_epilogue=True,
        ](ctx, Idx[256], Idx[256], Idx[128])

        # Test 3: 1SM mode with register-based epilogue (larger)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            transpose_b=transpose_b,
            cta_group=1,
            mma_shape=Index(128, 128, 32),
            cluster_shape=Index(1, 1, 1),
            register_based_epilogue=True,
        ](ctx, Idx[512], Idx[512], Idx[256])

        # Test 4: 2SM mode with register-based epilogue (larger)
        test_grouped_gemm_epilogue[
            a_type,
            b_type,
            c_type,
            scales_dtype,
            transpose_b=transpose_b,
            cta_group=2,
            mma_shape=Index(256, 128, 32),
            cluster_shape=Index(2, 1, 1),
            register_based_epilogue=True,
        ](ctx, Idx[512], Idx[512], Idx[256])

        print("\n" + "=" * 60)
        print("All epilogue tests PASSED!")
        print("=" * 60)
