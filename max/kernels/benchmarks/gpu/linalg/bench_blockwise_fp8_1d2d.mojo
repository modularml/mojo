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

"""Benchmark for blockwise FP8 1D2D grouped matmul (structured vs legacy).

Compares:
- Legacy: grouped_matmul_sm100_blockwise_scaled_fp8_persistent (CLC-based)
- Structured: grouped_matmul_dynamic_scaled_fp8_1d2d (GroupedWorkIterator1D1D)

Uses DeepSeek V3-style MoE shapes:
- Gate/Up: num_experts=256, N=2048, K=7168
- Down:    num_experts=256, N=7168, K=2048
- top_k=8

Usage:
    # Run with default DeepSeek V3 shapes
    mojo bench_blockwise_fp8_1d2d.mojo

    # Custom shapes
    mojo bench_blockwise_fp8_1d2d.mojo \
        get_defined_int[N]=7168 get_defined_int[K]=2048 \
        get_defined_int[num_experts]=256
"""

from std.sys import get_defined_int

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major as new_row_major,
)
from linalg.grouped_matmul_sm100_blockwise_fp8 import (
    grouped_matmul_sm100_blockwise_scaled_fp8_persistent,
)
from linalg.matmul.gpu.sm100.config import MatmulConfig
from structured_kernels.tile_types import (
    GMEMLayout1D,
)
from linalg.matmul.gpu.sm100_structured.blockwise_fp8_1d2d import (
    grouped_matmul_dynamic_scaled_fp8_1d2d,
)
from std.utils.index import Index, IndexList


def bench_blockwise_fp8_1d2d[
    num_experts: Int,
    expert_shape: IndexList[2],  # (N, K)
](
    ctx: DeviceContext,
    mut bench: Bench,
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids_list: List[Int],
) raises:
    """Benchmark both legacy and structured blockwise FP8 1D2D kernels."""
    comptime BLOCK_SCALE_K = 128
    comptime transpose_b = True
    comptime a_type = DType.float8_e4m3fn
    comptime b_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    # Compute total tokens and max tokens per expert
    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        var M = num_tokens_by_expert[i]
        total_num_tokens += M
        max_num_tokens_by_expert = max(max_num_tokens_by_expert, M)

    var total_flops = 2 * total_num_tokens * N * K

    # Define layouts
    comptime a_layout = Layout.row_major(UNKNOWN_VALUE, K)
    comptime b_layout = Layout.row_major(num_experts, N, K)
    comptime c_layout = Layout.row_major(UNKNOWN_VALUE, N)
    comptime a_scales_layout = Layout.row_major(
        K // BLOCK_SCALE_K, UNKNOWN_VALUE
    )
    comptime b_scales_layout = Layout.row_major(
        num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )
    comptime expert_scales_layout = Layout.row_major(num_experts)

    # Sizes
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N
    var a_scales_size = (K // BLOCK_SCALE_K) * total_num_tokens
    var b_scales_size = (
        num_experts * (N // BLOCK_SCALE_K) * (K // BLOCK_SCALE_K)
    )

    # Host allocations
    var a_offsets_host_ptr = List(length=num_active_experts + 1, fill=UInt32(0))
    var expert_ids_host_ptr = List(length=num_active_experts, fill=Int32(0))
    var expert_scales_host_ptr = List(length=num_experts, fill=Float32(1.0))

    # Setup offsets, expert ids, scales
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids_list[i])

    # Device allocations
    var a_dev_buf = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buf = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_dev_buf = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev_buf = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )
    var a_scales_dev_buf = ctx.enqueue_create_buffer[.float32](a_scales_size)
    var b_scales_dev_buf = ctx.enqueue_create_buffer[.float32](b_scales_size)
    var expert_scales_dev_buf = ctx.enqueue_create_buffer[.float32](num_experts)

    # Copy offsets and ids to device
    ctx.enqueue_copy(a_offsets_dev_buf, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buf, expert_ids_host_ptr)
    ctx.enqueue_copy(expert_scales_dev_buf, expert_scales_host_ptr)

    # LayoutTensor views for structured kernel
    var dynamic_a_shape = IndexList[2](total_num_tokens, K)
    var dynamic_c_shape = IndexList[2](total_num_tokens, N)
    var dynamic_a_scales_shape = IndexList[2](
        K // BLOCK_SCALE_K, total_num_tokens
    )
    var dynamic_b_scales_shape = IndexList[3](
        num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K
    )

    var a_struct = LayoutTensor[a_type, a_layout](
        a_dev_buf.unsafe_ptr().bitcast[Scalar[a_type]](),
        RuntimeLayout[a_layout].row_major(dynamic_a_shape),
    )
    var b_struct = LayoutTensor[b_type, b_layout](
        b_dev_buf.unsafe_ptr().bitcast[Scalar[b_type]](),
        RuntimeLayout[b_layout].row_major(IndexList[3](num_experts, N, K)),
    )
    var c_struct = LayoutTensor[c_type, c_layout](
        c_dev_buf.unsafe_ptr().bitcast[Scalar[c_type]](),
        RuntimeLayout[c_layout].row_major(dynamic_c_shape),
    )
    var a_scales_struct = LayoutTensor[.float32, a_scales_layout](
        a_scales_dev_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[a_scales_layout].row_major(dynamic_a_scales_shape),
    )
    var b_scales_struct = LayoutTensor[.float32, b_scales_layout](
        b_scales_dev_buf.unsafe_ptr().bitcast[Float32](),
        RuntimeLayout[b_scales_layout].row_major(
            IndexList[3](num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K)
        ),
    )
    var a_offsets_struct = LayoutTensor[
        .uint32, Layout.row_major(UNKNOWN_VALUE)
    ](
        a_offsets_dev_buf.unsafe_ptr().bitcast[UInt32](),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](num_active_experts + 1)
        ),
    )
    var expert_ids_struct = LayoutTensor[
        .int32, Layout.row_major(UNKNOWN_VALUE)
    ](
        expert_ids_dev_buf.unsafe_ptr().bitcast[Int32](),
        RuntimeLayout[Layout.row_major(UNKNOWN_VALUE)].row_major(
            IndexList[1](num_active_experts)
        ),
    )
    var expert_scales_struct = LayoutTensor[.float32, expert_scales_layout](
        expert_scales_dev_buf.unsafe_ptr().bitcast[Float32](),
    )

    # TileTensor versions for the structured kernel
    var a_tt = TileTensor(
        a_dev_buf,
        new_row_major(Coord(Int64(total_num_tokens), Idx[K])),
    )
    var b_tt = TileTensor(
        b_dev_buf,
        new_row_major[num_experts, N, K](),
    )
    var c_tt = TileTensor(
        c_dev_buf,
        new_row_major(Coord(Int64(total_num_tokens), Idx[N])),
    )
    var a_scales_tt = TileTensor(
        a_scales_dev_buf,
        new_row_major(
            Coord(
                Idx[K // BLOCK_SCALE_K],
                Int64(total_num_tokens),
            )
        ),
    )
    var b_scales_tt = TileTensor(
        b_scales_dev_buf,
        new_row_major[num_experts, N // BLOCK_SCALE_K, K // BLOCK_SCALE_K](),
    )
    var a_offsets_tt = TileTensor[.uint32, GMEMLayout1D, MutAnyOrigin](
        a_offsets_dev_buf,
        GMEMLayout1D(
            Coord(Int64(num_active_experts + 1)),
            Coord(Idx[1]),
        ),
    )
    var expert_ids_tt = TileTensor[.int32, GMEMLayout1D, MutAnyOrigin](
        expert_ids_dev_buf,
        GMEMLayout1D(
            Coord(Int64(num_active_experts)),
            Coord(Idx[1]),
        ),
    )
    var expert_scales_tt = TileTensor[.float32, GMEMLayout1D, MutAnyOrigin](
        expert_scales_dev_buf,
        GMEMLayout1D(
            Coord(Int64(num_experts)),
            Coord(Idx[1]),
        ),
    )

    var run_name_prefix = String(
        "blockwise_fp8_1d2d ",
        num_active_experts,
        "x",
        total_num_tokens,
        "x",
        N,
        "x",
        K,
    )

    # ===== Benchmark Legacy Kernel =====
    comptime config = MatmulConfig[a_type, b_type, c_type, transpose_b](
        cluster_shape=Index(1, 1, 1),
        mma_shape=Index(64, 64, 32),
        cta_group=1,
        AB_swapped=False,
        k_group_size=1,
    )

    @always_inline
    def bench_legacy(
        mut bencher: Bencher,
    ) {
        var a_tt,
        var b_tt,
        var c_tt,
        var a_scales_tt,
        var b_scales_tt,
        var a_offsets_tt,
        var expert_ids_tt,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            grouped_matmul_sm100_blockwise_scaled_fp8_persistent[
                config=config,
            ](
                c_tt,
                a_tt,
                b_tt,
                a_scales_tt,
                b_scales_tt,
                a_offsets_tt,
                expert_ids_tt,
                max_num_tokens_by_expert,
                num_active_experts,
                ctx,
            )

        bencher_iter_custom(bencher, kernel_launch, ctx)

    bench.bench_function(
        bench_legacy,
        BenchId(run_name_prefix + " legacy"),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    # ===== Benchmark Structured Kernel =====
    @always_inline
    def bench_structured(
        mut bencher: Bencher,
    ) {
        var a_tt,
        var b_tt,
        var c_tt,
        var a_scales_tt,
        var b_scales_tt,
        var a_offsets_tt,
        var expert_ids_tt,
        var expert_scales_tt,
        imm,
    }:
        @always_inline
        def kernel_launch(ctx: DeviceContext, iteration: Int) raises {imm}:
            grouped_matmul_dynamic_scaled_fp8_1d2d[
                a_scales_type=DType.float32,
                b_scales_type=DType.float32,
                transpose_b=transpose_b,
            ](
                c_tt,
                a_tt,
                b_tt,
                a_scales_tt,
                b_scales_tt,
                a_offsets_tt,
                expert_ids_tt,
                expert_scales_tt,
                num_active_experts,
                ctx,
            )

        bencher_iter_custom(bencher, kernel_launch, ctx)

    bench.bench_function(
        bench_structured,
        BenchId(run_name_prefix + " structured"),
        [ThroughputMeasure(BenchMetric.flops, total_flops)],
    )

    _ = a_dev_buf^
    _ = b_dev_buf^
    _ = c_dev_buf^
    _ = a_offsets_dev_buf^
    _ = expert_ids_dev_buf^
    _ = a_scales_dev_buf^
    _ = b_scales_dev_buf^
    _ = expert_scales_dev_buf^
    _ = expert_scales_host_ptr^
    _ = expert_ids_host_ptr^
    _ = a_offsets_host_ptr^


def main() raises:
    """Benchmark blockwise FP8 1D2D: legacy vs structured.

    Default shapes match DeepSeek V3 MoE dimensions.
    """
    comptime N = get_defined_int["N", 7168]()
    comptime K = get_defined_int["K", 2048]()
    comptime num_experts = get_defined_int["num_experts", 8]()
    comptime expert_shape = IndexList[2](N, K)

    var b = Bench()

    with DeviceContext() as ctx:
        # DeepSeek V3 decode scenario: uniform expert routing
        # 8 experts with 64 tokens each = 512 total tokens
        bench_blockwise_fp8_1d2d[num_experts, expert_shape](
            ctx,
            b,
            num_active_experts=8,
            num_tokens_by_expert=[64, 64, 64, 64, 64, 64, 64, 64],
            expert_ids_list=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        # DeepSeek V3 decode scenario: uneven expert routing
        bench_blockwise_fp8_1d2d[num_experts, expert_shape](
            ctx,
            b,
            num_active_experts=8,
            num_tokens_by_expert=[128, 4, 32, 96, 8, 64, 16, 48],
            expert_ids_list=[0, 1, 2, 3, 4, 5, 6, 7],
        )

        # Single expert (sanity / baseline)
        bench_blockwise_fp8_1d2d[num_experts, expert_shape](
            ctx,
            b,
            num_active_experts=1,
            num_tokens_by_expert=[256],
            expert_ids_list=[0],
        )

    b.dump_report()
