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

from std.collections import Optional
from std.sys import align_of

from max.gpu.host import DeviceContext
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.fp8_quantization import naive_blockwise_scaled_fp8_grouped_matmul
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent,
)
from linalg.matmul.gpu.sm100.config import MatmulConfig
from linalg.utils import elementwise_epilogue_type
from std.testing import assert_almost_equal

from std.utils.index import Index, IndexList


def test_grouped_matmul_sm100_blockwise_scaled_fp8[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    umma_shape: IndexList[3] = Index(64, 64, 32),
    use_epilogue: Bool = False,
    scales_type: DType = .float32,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
    ctx: DeviceContext,
) raises:
    comptime BLOCK_SCALE_K = 128
    comptime block_tile_shape = Index(umma_shape[0], umma_shape[1], 128)
    comptime transpose_b = True

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]
    comptime swizzle = TensorMapSwizzle.SWIZZLE_128B

    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        var M = num_tokens_by_expert[i]
        total_num_tokens += M
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, M)

    print(
        "== test_grouped_sm100_blockwise_scaled_fp8_matmul",
        a_type,
        "problem shape: (",
        total_num_tokens,
        "x",
        N,
        "x",
        K,
        ")",
        "block_tile_shape: (",
        block_tile_shape[0],
        "x",
        block_tile_shape[1],
        "x",
        block_tile_shape[2],
        ")",
        "transpose_b:",
        transpose_b,
    )

    assert K % BLOCK_SCALE_K == 0, "K must be divisible by BLOCK_SCALE_K"

    # Define sizes
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N
    var a_scales_size = (K // BLOCK_SCALE_K) * total_num_tokens

    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    # TileTensor shapes for device buffers
    var a_tt_shape = row_major(Coord(Int(total_num_tokens), Idx[K]))
    var b_tt_shape = row_major(Coord(Idx[num_experts], Idx[N], Idx[K]))
    var c_tt_shape = row_major(Coord(Int(total_num_tokens), Idx[N]))
    var a_scales_tt_shape = row_major(
        Coord(Idx[K // BLOCK_SCALE_K], Int(total_num_tokens))
    )
    var b_scales_tt_shape = row_major(
        Coord(
            Idx[num_experts],
            Idx[N // BLOCK_SCALE_K],
            Idx[K // BLOCK_SCALE_K],
        )
    )

    # Host allocations
    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_host_ref_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )
    var a_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        a_scales_size
    )
    var b_scales_host_ptr = ctx.enqueue_create_host_buffer[scales_type](
        b_scales_size
    )

    var a_host = TileTensor(a_host_ptr, a_tt_shape)
    var b_host = TileTensor(b_host_ptr, b_tt_shape)
    var c_host = TileTensor(c_host_ptr, c_tt_shape)
    var c_host_ref = TileTensor(c_host_ref_ptr, c_tt_shape)
    var a_scales_host = TileTensor(a_scales_host_ptr, a_scales_tt_shape)
    var b_scales_host = TileTensor(b_scales_host_ptr, b_scales_tt_shape)

    # Setup offsets and expert ids
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids_list[i])

    # Device allocations
    var a_device_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_device_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_device_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_device_ref_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_device_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_device_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var a_scales_device_buffer = ctx.enqueue_create_buffer[scales_type](
        a_scales_size
    )
    var b_scales_device_buffer = ctx.enqueue_create_buffer[scales_type](
        b_scales_size
    )

    var a_device_tt = TileTensor(a_device_buffer, a_tt_shape)
    var b_device_tt = TileTensor(b_device_buffer, b_tt_shape)
    var c_device_tt = TileTensor(c_device_buffer, c_tt_shape)
    var c_device_ref_tt = TileTensor(c_device_ref_buffer, c_tt_shape)
    var a_offsets_device_tt = TileTensor(
        a_offsets_device_buffer,
        row_major(Coord(Int(num_active_experts + 1))),
    )
    var expert_ids_device_tt = TileTensor(
        expert_ids_device_buffer,
        row_major(Coord(Int(num_active_experts))),
    )
    var a_scales_device_tt = TileTensor(
        a_scales_device_buffer, a_scales_tt_shape
    )
    var b_scales_device_tt = TileTensor(
        b_scales_device_buffer, b_scales_tt_shape
    )

    var c_tensor = c_device_tt

    @__parameter
    @always_inline
    @__copy_capture(c_tensor)
    def epilogue_fn[
        _dtype: DType,
        width: SIMDLength,
        *,
        alignment: Int = align_of[SIMD[_dtype, width]](),
    ](idx: IndexList[2], val: SIMD[_dtype, width]) capturing -> None:
        comptime assert c_tensor.flat_rank >= 2
        c_tensor.store[alignment=alignment](
            Coord(idx[0], idx[1]),
            rebind[SIMD[c_type, width]](val),
        )

    random(a_host)
    random(b_host)
    _ = c_host.fill(0)
    _ = c_host_ref.fill(0)

    random(a_scales_host)
    random(b_scales_host)

    ctx.enqueue_copy(a_device_buffer, a_host_ptr)
    ctx.enqueue_copy(b_device_buffer, b_host_ptr)
    ctx.enqueue_copy(c_device_buffer, c_host_ptr)
    ctx.enqueue_copy(c_device_ref_buffer, c_host_ref_ptr)
    ctx.enqueue_copy(a_offsets_device_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_device_buffer, expert_ids_host_ptr)
    ctx.enqueue_copy(a_scales_device_buffer, a_scales_host_ptr)
    ctx.enqueue_copy(b_scales_device_buffer, b_scales_host_ptr)

    var a = a_device_tt.to_layout_tensor()
    var b = b_device_tt.to_layout_tensor()
    var c_ref = c_device_ref_tt.to_layout_tensor()
    var a_scales = a_scales_device_tt.to_layout_tensor()
    var b_scales = b_scales_device_tt.to_layout_tensor()
    var a_offsets = a_offsets_device_tt.to_layout_tensor()
    var expert_ids = expert_ids_device_tt.to_layout_tensor()

    # Reference first
    naive_blockwise_scaled_fp8_grouped_matmul[
        BLOCK_DIM_M=16,
        BLOCK_DIM_N=16,
        transpose_b=transpose_b,
        scales_granularity_mnk=Index(1, BLOCK_SCALE_K, BLOCK_SCALE_K),
    ](
        c_ref,
        a,
        b,
        a_scales,
        b_scales,
        a_offsets,
        expert_ids,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(1, 1, 1),
        mma_shape=umma_shape,
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
    )

    grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
        config=config,
        elementwise_lambda_fn=Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if use_epilogue else None,
    ](
        c_device_tt,
        a_device_tt,
        b_device_tt,
        a_scales_device_tt,
        b_scales_device_tt,
        a_offsets_device_tt,
        expert_ids_device_tt,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_device_buffer)
    ctx.enqueue_copy(c_host_ref_ptr, c_device_ref_buffer)
    ctx.synchronize()

    var rtol = 1e-2
    var atol = 1e-2
    for mi in range(total_num_tokens):
        for ni in range(N):
            assert_almost_equal(
                c_host_ptr[mi * N + ni],
                c_host_ref_ptr[mi * N + ni],
                msg=String(t"m: {mi} n: {ni}"),
                rtol=rtol,
                atol=atol,
            )

    # Cleanup


def main() raises:
    with DeviceContext() as ctx:
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
            use_epilogue=True,
        ](1, [128], [0], ctx)

        # Single expert, last M-tile partial (100 mod 64 != 0).
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
        ](1, [100], [0], ctx)

        # Qwen3-VL-30B-A3B FP8 gate_up shape (N=1536, K=2048), two experts
        # with per-expert M counts not aligned to ``16 / size_of(scales_type)``
        # (here 4 for fp32 scales). Expert 1 starts at row 707, which makes
        # the scales TMA's strided-coord byte offset 707*4=2828, not a
        # multiple of 16. Before the per-tile alignment check, that faulted
        # with CUDA_ERROR_ILLEGAL_INSTRUCTION in the full-TMA path.
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=2,
            expert_shape=Index(1536, 2048),
        ](2, [707, 709], [0, 1], ctx)

        # Misaligned per-expert offset against the production umma_shape
        # (64, 128, 32) so the BN=128 dispatch hits the partial-tile A copy
        # path. Expert 1 starts at row 5 → byte offset 5*4=20, not 16-aligned.
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=2,
            expert_shape=Index(256, 256),
            umma_shape=Index(64, 128, 32),
        ](2, [5, 11], [0, 1], ctx)

        # total_m stride misalignment: total_num_tokens=7 with fp32 scales
        # gives a K-row stride of 7*4=28, not a multiple of 16, so
        # `create_tma_tile` would reject the descriptor. The host gate in
        # `grouped_matmul_sm100_blockwise_scaled_fp8_persistent` detects this
        # and routes to the naive kernel.
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=2,
            expert_shape=Index(256, 256),
        ](2, [3, 4], [0, 1], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(512, 1024),
        ](1, [256], [0], ctx)

        # Simple expert routing
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](1, [256], [2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(4096, 7168),
        ](2, [128, 256], [0, 2], ctx)

        # Unaligned grouped matmul
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(7168, 2048),
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(1280, 1024),
            use_epilogue=True,
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float32,
            num_experts=4,
            expert_shape=Index(512, 1024),
        ](2, [20, 40], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float32,
            num_experts=1,
            expert_shape=Index(512, 1024),
        ](1, [512], [0], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float32,
            num_experts=6,
            expert_shape=Index(7168, 2048),
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .float32,
            num_experts=6,
            expert_shape=Index(1280, 1024),
            use_epilogue=True,
        ](4, [20, 1500, 300, 28], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(4096, 7168),
        ](2, [8, 64], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(7168, 2048),
        ](4, [20, 4, 4, 40], [0, 3, 2, 4], ctx)

        # bf16 scales tests
        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
            scales_type=.bfloat16,
        ](1, [128], [0], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(512, 1024),
            scales_type=.bfloat16,
        ](1, [256], [0], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(4096, 7168),
            scales_type=.bfloat16,
        ](2, [128, 256], [0, 2], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(7168, 2048),
            scales_type=.bfloat16,
        ](4, [24, 1504, 296, 32], [0, 3, 2, 4], ctx)

        test_grouped_matmul_sm100_blockwise_scaled_fp8[
            .float8_e4m3fn,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(1280, 1024),
            use_epilogue=True,
            scales_type=.bfloat16,
        ](4, [24, 1504, 296, 32], [0, 3, 2, 4], ctx)
