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
"""SM100 Fused GEMM+SwiGLU+Bias 2SM Tests.

Mirrors test_matmul_sm100_swiglu_2sm but with a 1D sigma-permuted bias.
Covers all 2SM (cta_group=2) configurations:
  MMA_M ∈ {128, 256}
  BN = MMA_N // 2 ∈ range(8, 128, 8)  ⇒  MMA_N ∈ range(16, 256, 16)
  AB_swapped ∈ {False, True}
  register_swiglu ∈ {False, True}    (SMEM→TMA vs direct GMEM)

Subject to the same SMEM→TMA alignment constraints as the no-bias test.

Reference: cuBLAS matmul (FP32) → add sigma-permuted bias → SwiGLU in FP32
           → cast to BF16.

Bias layout: bias[2h] is added to the gate column (silu branch),
             bias[2h+1] is added to the up column (linear branch).

Usage:
    bazel test //max/kernels/test/gpu/linalg:test_matmul_sm100_swiglu_bias_2sm --config=b200
"""

from std.math import exp, recip
from std.collections import OptionalReg

import linalg.matmul.vendor.blas as vendor_blas
from max.gpu.host import DeviceContext
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

from std.utils.index import Index


def swiglu_bias_reference(
    full_ptr: ImmPointer[Float32, _],
    bias_ptr: ImmPointer[BFloat16, _],
    ref_ptr: MutPointer[BFloat16, MutAnyOrigin],
    M: Int,
    N: Int,
):
    """Compute SwiGLU+bias reference from FP32 matmul output.

    Uses FP32 vendor_blas output to match the kernel's FP32 TMEM accumulation,
    avoiding rounding mismatch that BF16 intermediate would introduce.

    full: [M, N] FP32 from A @ sigma_perm(B).T; adjacent columns are (gate, up).
    bias: [N] BF16, sigma-permuted (bias[2h] = gate bias, bias[2h+1] = up bias).
    ref:  [M, H] BF16; ref[m,h] = bf16(silu(full[m,2h]+bias[2h]) * (full[m,2h+1]+bias[2h+1])).
    """
    var H = N // 2
    for m in range(M):
        for h in range(H):
            var gate = (
                full_ptr[m * N + 2 * h] + bias_ptr[2 * h].cast[.float32]()
            )
            var up = (
                full_ptr[m * N + 2 * h + 1]
                + bias_ptr[2 * h + 1].cast[.float32]()
            )
            var sigmoid = recip(Float32(1.0) + exp(-gate))
            ref_ptr.store(m * H + h, (gate * sigmoid * up).cast[.bfloat16]())


def test_swiglu_bias[
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
        " bias=True",
        sep="",
    )

    var a_shape = row_major(Coord(m, Idx[KType.static_value]))
    var b_shape = row_major(
        Coord(Idx[NType.static_value], Idx[KType.static_value])
    )
    var bias_shape = row_major(Coord(Idx[NType.static_value]))
    var full_shape = row_major(Coord(m, Idx[NType.static_value]))
    var c_shape = row_major(Coord(m, Idx[NType.static_value // 2]))

    var a_size = M * K
    var b_size = N * K
    var bias_size = N
    var full_size = M * N
    var c_size = M * H

    var a_host_buf = ctx.enqueue_create_host_buffer[dtype](a_size)
    var a_host = TileTensor(a_host_buf, a_shape)
    var b_host_buf = ctx.enqueue_create_host_buffer[dtype](b_size)
    var b_host = TileTensor(b_host_buf, b_shape)
    var bias_host_buf = ctx.enqueue_create_host_buffer[dtype](bias_size)
    var bias_host = TileTensor(bias_host_buf, bias_shape)
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
    var bias_device = ctx.enqueue_create_buffer[dtype](bias_size)
    var bias_tensor = TileTensor(bias_device, bias_shape)
    var full_device = ctx.enqueue_create_buffer[.float32](full_size)
    var full_tensor = TileTensor(full_device, full_shape)
    var c_device = ctx.enqueue_create_buffer[dtype](c_size)

    rand(a_host._storage, a_host.num_elements())
    rand(b_host._storage, b_host.num_elements())
    rand(bias_host._storage, bias_host.num_elements())

    ctx.enqueue_copy(a_device, a_host_buf)
    ctx.enqueue_copy(b_device, b_host_buf)
    ctx.enqueue_copy(bias_device, bias_host_buf)

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

    swiglu_bias_reference(
        full_host._storage,
        bias_host._storage,
        c_ref._storage.as_unsafe_any_origin(),
        M,
        N,
    )

    var c_tensor = TileTensor(c_device, c_shape)
    var bias_immut_ptr = rebind[ImmPointer[Scalar[dtype], ImmutAnyOrigin]](
        bias_tensor._storage
    )
    matmul_swiglu_dispatch_sm100[config](
        c_tensor, a_tensor, b_tensor, ctx, OptionalReg(bias_immut_ptr)
    )

    ctx.enqueue_copy(c_host_buf, c_device)
    ctx.synchronize()

    assert_almost_equal(
        c_host._storage, c_ref._storage, c_size, atol=1e-2, rtol=1e-2
    )
    print("    PASSED")

    _ = a_device^
    _ = b_device^
    _ = bias_device^
    _ = full_device^
    _ = c_device^


def main() raises:
    print("=" * 60)
    print("SM100 FUSED GEMM+SwiGLU+BIAS 2SM TEST")
    print("=" * 60)

    with DeviceContext() as ctx:
        comptime dtype = DType.bfloat16
        comptime MMA_K = 16

        # ============================================================
        # Section 1: Comprehensive 2SM matrix with bias
        # ============================================================
        comptime for bn in range(8, 128, 8):
            # ---- MMA_M=256 (BM=128): both paths run for every BN ----
            comptime mma_shape_256 = Index(256, 2 * bn, MMA_K)

            comptime config_256_smem = swiglu_matmul_config[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_256,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=False,
                use_bias=True,
            )
            test_swiglu_bias[config=config_256_smem](
                ctx, Int(333), Idx[512], Idx[640]
            )

            comptime config_256_gmem = FusedSwiGLUMatmulConfig[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_256,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=False,
                use_bias=True,
                register_swiglu=True,
            )
            test_swiglu_bias[config=config_256_gmem](
                ctx, Int(333), Idx[512], Idx[640]
            )

            # ---- MMA_M=256, AB_swapped ----
            comptime config_256_swap_smem = swiglu_matmul_config[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_256,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=True,
                use_bias=True,
            )
            test_swiglu_bias[config=config_256_swap_smem](
                ctx, Int(333), Idx[512], Idx[640]
            )

            comptime config_256_swap_gmem = FusedSwiGLUMatmulConfig[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_256,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=True,
                use_bias=True,
                register_swiglu=True,
            )
            test_swiglu_bias[config=config_256_swap_gmem](
                ctx, Int(333), Idx[512], Idx[640]
            )

            # ---- MMA_M=128 (BM=64, Layout B) ----
            comptime mma_shape_128 = Index(128, 2 * bn, MMA_K)

            # SMEM-TMA only when BN % 16 == 0 (i.e., MMA_N % 32 == 0).
            comptime if bn % 16 == 0:
                comptime config_128_smem = swiglu_matmul_config[
                    dtype, dtype, dtype, True
                ](
                    cluster_shape=Index(2, 1, 1),
                    mma_shape=mma_shape_128,
                    block_swizzle_size=0,
                    cta_group=2,
                    AB_swapped=False,
                    use_bias=True,
                )
                test_swiglu_bias[config=config_128_smem](
                    ctx, Int(333), Idx[512], Idx[640]
                )

            # GMEM path runs for every BN.
            comptime config_128_gmem = FusedSwiGLUMatmulConfig[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_128,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=False,
                use_bias=True,
                register_swiglu=True,
            )
            test_swiglu_bias[config=config_128_gmem](
                ctx, Int(333), Idx[512], Idx[640]
            )

            # ---- MMA_M=128, AB_swapped ----
            comptime config_128_swap_smem = swiglu_matmul_config[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_128,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=True,
                use_bias=True,
            )
            test_swiglu_bias[config=config_128_swap_smem](
                ctx, Int(333), Idx[512], Idx[640]
            )

            comptime config_128_swap_gmem = FusedSwiGLUMatmulConfig[
                dtype, dtype, dtype, True
            ](
                cluster_shape=Index(2, 1, 1),
                mma_shape=mma_shape_128,
                block_swizzle_size=0,
                cta_group=2,
                AB_swapped=True,
                use_bias=True,
                register_swiglu=True,
            )
            test_swiglu_bias[config=config_128_swap_gmem](
                ctx, Int(333), Idx[512], Idx[640]
            )

        # ============================================================
        # Section 2: Multi-cluster non-swap SMEM tests with bias
        # ============================================================

        # --- MMA_M=256, MMA_N=128 ---
        comptime config_256_128 = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(4, 4, 1),
            mma_shape=Index(256, 128, MMA_K),
            block_swizzle_size=4,
            cta_group=2,
            AB_swapped=False,
            use_bias=True,
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(1), Idx[1024], Idx[1024]
        )
        test_swiglu_bias[config=config_256_128](ctx, Int(7), Idx[512], Idx[256])
        test_swiglu_bias[config=config_256_128](
            ctx, Int(63), Idx[2048], Idx[512]
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(129), Idx[1024], Idx[1024]
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(1000), Idx[1024], Idx[1024 + 16]
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(4096), Idx[1024], Idx[512]
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(512), Idx[256], Idx[1024]
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(333), Idx[1024], Idx[8192]
        )
        test_swiglu_bias[config=config_256_128](
            ctx, Int(512), Idx[1024], Idx[128]
        )

        # --- MMA_M=256, MMA_N=128, AB_swapped ---
        comptime config_256_128_swap = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(2, 1, 1),
            mma_shape=Index(256, 128, MMA_K),
            block_swizzle_size=4,
            cta_group=2,
            AB_swapped=True,
            use_bias=True,
        )
        test_swiglu_bias[config=config_256_128_swap](
            ctx, Int(1024), Idx[1024], Idx[1024]
        )
        test_swiglu_bias[config=config_256_128_swap](
            ctx, Int(333), Idx[512], Idx[640]
        )
        test_swiglu_bias[config=config_256_128_swap](
            ctx, Int(63), Idx[2048], Idx[512]
        )

        # --- MMA_M=256, MMA_N=64 ---
        comptime config_256_64 = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(4, 4, 1),
            mma_shape=Index(256, 64, MMA_K),
            block_swizzle_size=4,
            cta_group=2,
            AB_swapped=False,
            use_bias=True,
        )
        test_swiglu_bias[config=config_256_64](
            ctx, Int(63), Idx[1024], Idx[4096]
        )
        test_swiglu_bias[config=config_256_64](
            ctx, Int(999), Idx[4096], Idx[1024 + 16]
        )
        test_swiglu_bias[config=config_256_64](
            ctx, Int(2048), Idx[2048], Idx[512]
        )

        # --- MMA_M=128, MMA_N=128 (BM=64, Layout B) ---
        comptime config_128_128 = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(4, 4, 1),
            mma_shape=Index(128, 128, MMA_K),
            block_swizzle_size=4,
            cta_group=2,
            AB_swapped=False,
            use_bias=True,
        )
        test_swiglu_bias[config=config_128_128](ctx, Int(7), Idx[512], Idx[256])
        test_swiglu_bias[config=config_128_128](
            ctx, Int(65), Idx[1024], Idx[1024]
        )
        test_swiglu_bias[config=config_128_128](
            ctx, Int(1000), Idx[1024], Idx[1024 + 16]
        )
        test_swiglu_bias[config=config_128_128](
            ctx, Int(4096), Idx[1024], Idx[512]
        )
        test_swiglu_bias[config=config_128_128](
            ctx, Int(333), Idx[1024], Idx[8192]
        )

        # --- MMA_M=128, MMA_N=128, AB_swapped ---
        comptime config_128_128_swap = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(2, 1, 1),
            mma_shape=Index(128, 128, MMA_K),
            block_swizzle_size=4,
            cta_group=2,
            AB_swapped=True,
            use_bias=True,
        )
        test_swiglu_bias[config=config_128_128_swap](
            ctx, Int(1024), Idx[1024], Idx[1024]
        )
        test_swiglu_bias[config=config_128_128_swap](
            ctx, Int(333), Idx[512], Idx[640]
        )
        test_swiglu_bias[config=config_128_128_swap](
            ctx, Int(63), Idx[2048], Idx[512]
        )

        # --- MMA_M=128, MMA_N=64 ---
        comptime config_128_64 = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(4, 4, 1),
            mma_shape=Index(128, 64, MMA_K),
            block_swizzle_size=4,
            cta_group=2,
            AB_swapped=False,
            use_bias=True,
        )
        test_swiglu_bias[config=config_128_64](
            ctx, Int(33), Idx[1024], Idx[4096]
        )
        test_swiglu_bias[config=config_128_64](
            ctx, Int(999), Idx[4096], Idx[1024 + 16]
        )

        # --- k_group_size with MMA_M=128/cta_group=2 ---
        comptime config_128_128_kg2 = swiglu_matmul_config[
            dtype, dtype, dtype, True
        ](
            cluster_shape=Index(4, 4, 1),
            mma_shape=Index(128, 128, MMA_K),
            block_swizzle_size=0,
            cta_group=2,
            AB_swapped=False,
            k_group_size=2,
            use_bias=True,
        )
        test_swiglu_bias[config=config_128_128_kg2](
            ctx, Int(500), Idx[2048], Idx[4096]
        )

    print("\nAll SwiGLU+bias 2SM tests passed!")
