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
"""SM100 Fused GEMM+SwiGLU 1SM Tests.

Covers all 1SM (cta_group=1) configurations:
  bm = MMA_M ∈ {64, 128}
  bn = MMA_N ∈ range(8, 256, 8)
  AB_swapped ∈ {False, True}
  register_swiglu ∈ {False, True}  (SMEM→TMA vs direct GMEM)

Subject to:
  - Non-swap SMEM→TMA: bn % 16 == 0 (HalfN ≥ 8 for TMA's 16-byte inner
    dim minimum with SWIZZLE_NONE).
  - The other three paths run for every 8-aligned bn — swap-AB SMEM
    because its inner dim is BM/2 ∈ {32, 64} elements (≥ 64 bytes), and
    both GMEM paths because the C TMA descriptor inner dim is padded to
    ≥ 8 when register_swiglu=True (descriptor is allocated but unused).

Also includes multi-cluster non-swap SMEM tests for cluster shape,
swizzle, k-group-size, and problem-shape coverage.

Reference: cuBLAS matmul (FP32) → SwiGLU in FP32 → cast to BF16.

Usage:
    bazel test //max/kernels/test/gpu/linalg:test_matmul_sm100_swiglu_1sm --config=b200
"""

from std.math import exp, recip
from std.sys import size_of

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from internal_utils import assert_almost_equal
from std.random import rand
from layout import TileTensor, Coord, CoordLike, row_major, Idx

from linalg.matmul.gpu.sm100_structured.fused_swiglu.config import (
    FusedSwiGLUMatmulConfig,
    swiglu_matmul_config,
)
from linalg.matmul.gpu.sm100_structured.fused_swiglu.dispatch import (
    matmul_swiglu_dispatch_sm100,
)

from std.utils.index import Index, IndexList
from std.utils.static_tuple import StaticTuple


def swiglu_reference(
    full_ptr: ImmPointer[Float32, _],
    ref_ptr: MutPointer[BFloat16, MutAnyOrigin],
    M: Int,
    N: Int,
):
    """Compute SwiGLU reference from FP32 matmul output.

    full: [M, N] FP32 where N = 2H, adjacent columns are (gate, up) pairs.
    ref: [M, H] BF16 where ref[m, h] = bf16(silu(full[m, 2h]) * full[m, 2h+1]).
    """
    var H = N // 2
    for m in range(M):
        for h in range(H):
            var gate = full_ptr[m * N + 2 * h]
            var up = full_ptr[m * N + 2 * h + 1]
            var sigmoid = recip(Float32(1.0) + exp(-gate))
            var result = gate * sigmoid * up
            ref_ptr.store(m * H + h, result.cast[.bfloat16]())


def test_swiglu[
    MType: CoordLike,
    NType: CoordLike,
    KType: CoordLike,
    //,
    config: FusedSwiGLUMatmulConfig[.bfloat16, .bfloat16, .bfloat16, True],
](ctx: DeviceContext, m: MType, n: NType, k: KType) raises:
    comptime dtype = DType.bfloat16

    var M = Int(m.value())
    var N = Int(n.value())
    var K = Int(k.value())
    var H = N // 2

    print(
        "  dtypes=(bf16,bf16,bf16) shape=(",
        M,
        ",",
        N,
        ",",
        K,
        ") mma=(",
        config.mma_shape[0],
        ",",
        config.mma_shape[1],
        ") cluster=(",
        config.cluster_shape[0],
        ",",
        config.cluster_shape[1],
        ") swizzle=",
        config.block_swizzle_size,
        " swap_ab=" + ("True" if config.AB_swapped else "False"),
        " path=" + ("gmem" if config.register_swiglu else "smem_tma"),
        sep="",
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var full_shape = row_major(Coord(m, Idx[NType.static_value]))
    var c_shape = row_major(Coord(m, Idx[NType.static_value // 2]))

    var a_size = M * K
    var b_size = N * K
    var full_size = M * N
    var c_size = M * H

    var a_host_buf = ctx.enqueue_create_host_buffer[dtype](a_size)
    var a_host = TileTensor(a_host_buf, a_shape)
    var b_host_buf = ctx.enqueue_create_host_buffer[dtype](b_size)
    var b_host = TileTensor(b_host_buf, b_shape)
    var full_host_buf = ctx.enqueue_create_host_buffer[.float32](full_size)
    var full_host = TileTensor(full_host_buf, full_shape)
    var c_host_buf = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_host = TileTensor(c_host_buf, c_shape)
    var c_ref_buf = ctx.enqueue_create_host_buffer[dtype](c_size)
    var c_ref = TileTensor(c_ref_buf, c_shape)

    var a_device = ctx.enqueue_create_buffer[dtype](a_size)
    var a_tensor = TileTensor(a_device, a_shape)
    var b_device = ctx.enqueue_create_buffer[dtype](b_size)
    var b_tensor = TileTensor(b_device, b_shape)
    var full_device = ctx.enqueue_create_buffer[.float32](full_size)
    var full_tensor = TileTensor(full_device, full_shape)
    var c_device = ctx.enqueue_create_buffer[dtype](c_size)

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_buf)
    ctx.enqueue_copy(b_device, b_host_buf)

    vendor_blas.matmul(
        ctx,
        full_tensor.to_layout_tensor(),
        a_tensor.to_layout_tensor(),
        b_tensor.to_layout_tensor(),
        c_row_major=True,
        transpose_b=True,
    )

    ctx.enqueue_copy(full_host_buf, full_device)
    ctx.synchronize()

    swiglu_reference(
        full_host._storage, c_ref._storage.as_unsafe_any_origin(), M, N
    )

    var c_tensor = TileTensor(c_device, c_shape)
    matmul_swiglu_dispatch_sm100[config](c_tensor, a_tensor, b_tensor, ctx)

    ctx.enqueue_copy(c_host_buf, c_device)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage,
        c_ref._storage,
        c_size,
        atol=1e-2,
        rtol=1e-2,
    )
    print("    PASSED")

    _ = a_device^
    _ = b_device^
    _ = full_device^
    _ = c_device^


def main() raises:
    print("=" * 60)
    print("SM100 FUSED GEMM+SwiGLU 1SM TEST")
    print("=" * 60)

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime MMA_K = 16

        # ============================================================
        # Section 1: Multi-cluster non-swap SMEM sweeps
        # ============================================================
        # Cluster shape, block swizzle size, k_group_size, and varied
        # problem shapes. AB_swapped=False, register_swiglu=False.
        comptime for bm in [64, 128]:
            comptime for bn in [16, 32, 64, 128]:
                comptime mma_shape = Index(bm, bn, MMA_K)

                # cluster=(4,4,1), block_swizzle=8; M=1000,N=1024,K=1040
                comptime config_1 = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(4, 4, 1),
                    mma_shape=mma_shape,
                    block_swizzle_size=8,
                    cta_group=1,
                    AB_swapped=False,
                )
                test_swiglu[config=config_1](
                    ctx, Int(1000), Idx[1024], Idx[1024 + 16]
                )

                # cluster=(4,4,1), block_swizzle=4; M=512,N=4096,K=1040
                comptime config_2 = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(4, 4, 1),
                    mma_shape=mma_shape,
                    block_swizzle_size=4,
                    cta_group=1,
                    AB_swapped=False,
                )
                test_swiglu[config=config_2](
                    ctx, Int(512), Idx[4096], Idx[1024 + 16]
                )

                # cluster=(4,4,1), block_swizzle=0, k_group=2;
                # M=500, N=2048, K=4096
                comptime config_3 = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(4, 4, 1),
                    mma_shape=mma_shape,
                    block_swizzle_size=0,
                    cta_group=1,
                    AB_swapped=False,
                    k_group_size=2,
                )
                test_swiglu[config=config_3](
                    ctx, Int(500), Idx[2048], Idx[4096]
                )

                # cluster=(8,2,1), block_swizzle=2; M=999,N=256,K=128
                comptime config_4 = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(8, 2, 1),
                    mma_shape=mma_shape,
                    block_swizzle_size=2,
                    cta_group=1,
                    AB_swapped=False,
                )
                test_swiglu[config=config_4](ctx, Int(999), Idx[256], Idx[128])

                # cluster=(4,4,1), block_swizzle=1; M=777,N=2560,K=8192
                comptime config_5 = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(4, 4, 1),
                    mma_shape=mma_shape,
                    block_swizzle_size=1,
                    cta_group=1,
                    AB_swapped=False,
                )
                test_swiglu[config=config_5](
                    ctx, Int(777), Idx[2560], Idx[8192]
                )

        # ============================================================
        # Section 2: Comprehensive 1SM matrix
        # ============================================================
        # Every (bm, bn, swap, path) combination, fixed problem shape
        # and cluster=(1,1,1) to isolate the path-level coverage.
        comptime for bm in [64, 128]:
            comptime for bn in range(8, 256, 8):
                comptime mma_shape_1sm = Index(bm, bn, MMA_K)

                # SMEM-TMA, non-swap: needs bn % 16 == 0.
                comptime if bn % 16 == 0:
                    comptime config_smem_ns = swiglu_matmul_config[
                        dtype, dtype, dtype, True
                    ](
                        cluster_shape=Index(1, 1, 1),
                        mma_shape=mma_shape_1sm,
                        block_swizzle_size=0,
                        cta_group=1,
                        AB_swapped=False,
                    )
                    test_swiglu[config=config_smem_ns](
                        ctx, Int(333), Idx[512], Idx[640]
                    )

                # SMEM-TMA, AB_swapped: any 8-aligned bn.
                comptime config_smem_sw = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(1, 1, 1),
                    mma_shape=mma_shape_1sm,
                    block_swizzle_size=0,
                    cta_group=1,
                    AB_swapped=True,
                )
                test_swiglu[config=config_smem_sw](
                    ctx, Int(333), Idx[512], Idx[640]
                )

                # GMEM, non-swap: any 8-aligned bn (padded TMA inner).
                comptime config_gmem_ns = FusedSwiGLUMatmulConfig[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(1, 1, 1),
                    mma_shape=mma_shape_1sm,
                    block_swizzle_size=0,
                    cta_group=1,
                    AB_swapped=False,
                    register_swiglu=True,
                )
                test_swiglu[config=config_gmem_ns](
                    ctx, Int(333), Idx[512], Idx[640]
                )

                # GMEM, AB_swapped: any 8-aligned bn.
                comptime config_gmem_sw = FusedSwiGLUMatmulConfig[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(1, 1, 1),
                    mma_shape=mma_shape_1sm,
                    block_swizzle_size=0,
                    cta_group=1,
                    AB_swapped=True,
                    register_swiglu=True,
                )
                test_swiglu[config=config_gmem_sw](
                    ctx, Int(333), Idx[512], Idx[640]
                )

    print("\nAll SwiGLU 1SM tests passed!")
