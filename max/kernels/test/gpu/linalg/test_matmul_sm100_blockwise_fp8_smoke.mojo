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
"""Minimal smoke test for blockwise FP8 - representative configs for fast iteration.

Covers:
- 1SM path (cta_group=1): block_tile_shape == umma_shape
- 2SM path (cta_group=2): umma_shape = 2x block_tile_shape

Target: < 1 minute compile + run for debugging purposes.
"""

from std.math import ceildiv
from std.sys import size_of
from linalg.matmul.gpu.sm100.config import MatmulConfig, GEMMKind
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.memory import alloc
from internal_utils import (
    assert_almost_equal,
    assert_with_measure,
)
from std.random import rand
from internal_utils._measure import relative_difference
from layout import (
    TileTensor,
    Coord,
    CoordLike,
    row_major,
    Idx,
)
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_matmul
from linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)

from std.utils.index import Index, IndexList
from std.utils.numerics import get_accum_type
from std.utils.static_tuple import StaticTuple


def test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
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
    scales_type: DType = .float32,
    transpose_b: Bool = True,
    a_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    b_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
    c_swizzle: TensorMapSwizzle = TensorMapSwizzle.SWIZZLE_128B,
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime BLOCK_SCALE_K = 128
    comptime accum_type = get_accum_type[c_type]()

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())

    if M * size_of[DType.float32]() % 16 != 0:
        raise Error("TMA expects M to be divisible by 16 bytes")

    print(
        "in/out dtypes=(",
        a_type,
        ", ",
        b_type,
        ", ",
        c_type,
        ") ",
        " problem shape=(",
        M,
        ", ",
        N,
        ", ",
        K,
        ") ",
        "mma_shape=",
        mma_shape,
        " block_tile_shape=",
        block_tile_shape,
        " cta_group=",
        cta_group,
        " cluster_shape=(",
        cluster_shape[0],
        ", ",
        cluster_shape[1],
        ", ",
        cluster_shape[2],
        ")",
    )

    # Shapes
    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var c_shape = row_major(Coord(m, Idx[NType.static_value]))

    # Calculate scales dimensions
    var a_scales_shape_k = ceildiv(K, BLOCK_SCALE_K)
    var b_scales_shape_n = ceildiv(N, BLOCK_SCALE_K)
    var b_scales_shape_k = ceildiv(K, BLOCK_SCALE_K)

    var a_scales_shape = row_major(Coord(a_scales_shape_k, m))
    var b_scales_shape = row_major(Coord(b_scales_shape_n, b_scales_shape_k))

    # Allocate host memory
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](M * K)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](N * K)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](M * N)

    var a_host = TileTensor(a_host_ptr, a_shape)
    var b_host = TileTensor(b_host_ptr, b_shape)
    var c_host = TileTensor(c_host_ptr, c_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_shape)

    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_shape_k * M
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_shape_n * b_scales_shape_k
    )

    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_shape)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_shape)

    # Allocate device memory
    var a_device = ctx.enqueue_create_buffer[a_type](M * K)
    var b_device = ctx.enqueue_create_buffer[b_type](N * K)
    var c_device = ctx.enqueue_create_buffer[c_type](M * N)
    var c_device_ref = ctx.enqueue_create_buffer[c_type](M * N)
    var a_scales_device = ctx.enqueue_create_buffer[scales_type](
        a_scales_shape_k * M
    )
    var b_scales_device = ctx.enqueue_create_buffer[scales_type](
        b_scales_shape_n * b_scales_shape_k
    )

    var a_tensor = TileTensor(a_device, a_shape)
    var b_tensor = TileTensor(b_device, b_shape)
    var c_tensor = TileTensor(c_device, c_shape)
    var c_ref_tensor = TileTensor(c_device_ref, c_shape)
    var a_scales_tensor = TileTensor(a_scales_device, a_scales_shape)
    var b_scales_tensor = TileTensor(b_scales_device, b_scales_shape)

    # Initialize with random data
    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    rand(a_scales_host._storage, a_scales_host.num_elements())
    rand(b_scales_host._storage, b_scales_host.num_elements())

    # Move operands to the Device
    ctx.enqueue_copy(a_device, a_host_ptr)
    ctx.enqueue_copy(b_device, b_host_ptr)
    ctx.enqueue_copy(a_scales_device, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device, b_scales_host_ptr)

    comptime matmul_config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(
            cluster_shape[0], cluster_shape[1], cluster_shape[2]
        ),
        mma_shape=mma_shape,
        block_swizzle_size=0,
        cta_group=cta_group,
        gemm_kind=GEMMKind.BLOCK_SCALED_1D2D_FP8,
    )

    blockwise_fp8_matmul[
        transpose_b=transpose_b,
        a_scales_type=scales_type,
        b_scales_type=scales_type,
        config=matmul_config,
    ](
        c_tensor,
        a_tensor,
        b_tensor,
        a_scales_tensor,
        b_scales_tensor,
        ctx,
    )

    # Reference implementation
    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_ref_tensor.to_layout_tensor(),
        a_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        a_scales_tensor.to_layout_tensor(),
        b_scales_tensor.to_layout_tensor(),
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref)
    ctx.synchronize()

    assert_with_measure[relative_difference](
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        threshold=0.001,
    )

    assert_almost_equal(
        c_host._storage,
        c_host_ref._storage,
        c_host.num_elements(),
        atol=1e-2,
        rtol=1e-2,
    )

    # Cleanup
    _ = a_device^
    _ = b_device^
    _ = c_device^
    _ = c_device_ref^
    _ = a_scales_device^
    _ = b_scales_device^

    print("PASSED")


def main() raises:
    with DeviceContext() as ctx:
        comptime swizzle = TensorMapSwizzle.SWIZZLE_128B
        comptime in_dtype = DType.float8_e4m3fn
        comptime BK = (swizzle.bytes() // size_of[in_dtype]())
        comptime MMA_K = 32
        comptime out_dtype = DType.bfloat16

        print("=== SMOKE TEST: 1SM and 2SM Configurations ===")

        # ============================================================
        # 1SM PATH (cta_group=1): block_tile_shape == umma_shape
        # From test_matmul_sm100_1sm_blockwise_fp8.mojo
        # ============================================================
        print("\n--- 1SM Tests (cta_group=1) ---")

        # Config: mma_m_scale=1, mma_n_scale=2
        # block_tile_shape = (64, 16, 128), umma_shape = (64, 16, 32)
        comptime block_tile_1sm = Index(64, 16, BK)
        comptime umma_shape_1sm = Index(64, 16, MMA_K)

        print("block_tile_shape", block_tile_1sm, "umma_shape", umma_shape_1sm)

        # Shape from 1sm test: (1000, 576, 7168)
        _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
            in_dtype,
            in_dtype,
            out_dtype,
            block_tile_1sm,
            umma_shape_1sm,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=1,
        ](
            ctx,
            Int(512),
            Idx[576],
            Idx[512],
        )

        # ============================================================
        # 2SM PATH (cta_group=2): umma_shape = 2x block_tile_shape
        # From test_matmul_sm100_2sm_blockwise_fp8.mojo
        # ============================================================
        print("\n--- 2SM Tests (cta_group=2) ---")

        # Config: mma_m_scale=1, mma_n_scale=2
        # block_tile_shape = (64, 16, 128), umma_shape = (128, 32, 32)
        comptime block_tile_2sm = Index(64, 16, BK)
        comptime umma_shape_2sm = Index(128, 32, MMA_K)

        print("block_tile_shape", block_tile_2sm, "umma_shape", umma_shape_2sm)

        # Shape from 2sm test: (1000, 576, 7168) -> smaller for smoke
        _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
            in_dtype,
            in_dtype,
            out_dtype,
            block_tile_2sm,
            umma_shape_2sm,
            cluster_shape=StaticTuple[Int32, 3](2, 1, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
        ](
            ctx,
            Int(512),
            Idx[576],
            Idx[512],
        )

        # Additional 2SM test with larger cluster (4,4,1) from original tests
        _ = test_blackwell_matmul_tma_umma_warp_specialized_blockwise_fp8[
            in_dtype,
            in_dtype,
            out_dtype,
            block_tile_2sm,
            umma_shape_2sm,
            cluster_shape=StaticTuple[Int32, 3](4, 4, 1),
            a_swizzle=swizzle,
            b_swizzle=swizzle,
            cta_group=2,
        ](
            ctx,
            Int(512),
            Idx[4096],
            Idx[1024],
        )

        print("\n=== ALL SMOKE TESTS PASSED ===")
