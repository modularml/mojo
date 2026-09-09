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
# REQUIRES: NVIDIA-GPU

# RUN: ./bazelw build @nvshmem_prebuilt//:device
# RUN: BITCODE_PATH=$(./bazelw cquery '@nvshmem_prebuilt//:device' --output=files 2>/dev/null | head -1)
# RUN: mojo build --bitcode-libs $BITCODE_PATH  <path_to>/modular/max/kernels/benchmarks/gpu/comm/bench_ep_dispatch.mojo -o ./test
# RUN: %mpirun-gpu-per-process %t
#
# Alternatively, run manually with:
# NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
# br --run_under="mpirun -n $NUM_GPUS --allow-run-as-root --bind-to none" //max/kernels/benchmarks:gpu/bench_ep_dispatch

from std.random import randint, randn, seed
from std.sys import (
    get_defined_int,
    get_defined_dtype,
    size_of,
)

from max.benchmark import bencher_iter_custom
from std.benchmark import (
    Bench,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
)
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx
from layout.tile_layout import row_major
from std.memory import dealloc
from shmem import *
from shmem.ep_comm import (
    BF16TokenFormat,
    BlockwiseFP8TokenFormat,
    EPLocalSyncCounters,
    TokenFormat,
    dispatch_wait_kernel,
    dispatch_async_kernel,
    combine_wait_kernel,
    combine_async_kernel,
)


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: MutPointer[Int32, _], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids + tok_id * top_k

        # The top-k ids for a token should be unique. If not, we will assign a
        # random id to the duplicate id.
        def is_duplicate() {imm} -> Int:
            for i in range(top_k):
                for j in range(i + 1, top_k):
                    if topk_ids_for_token[i] == topk_ids_for_token[j]:
                        return i
            return -1

        var duplicate_idx = is_duplicate()
        while duplicate_idx != -1:
            randint(topk_ids_for_token + duplicate_idx, 1, 0, n_experts - 1)
            duplicate_idx = is_duplicate()


def bench_dispatch[
    token_dtype: DType,
    scales_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
](ctx: DeviceContext, mut b: Bench, my_rank: Int) raises:
    comptime input_type = token_dtype
    comptime combine_msg_bytes = size_of[input_type]() * hidden_size
    comptime group_size = 128

    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_tokens = n_experts * n_tokens_per_rank

    comptime output_tt_layout = row_major(
        (Idx[max_recv_tokens], Idx[hidden_size])
    )
    comptime output_scales_tt_layout = row_major(
        (hidden_size // group_size, Idx[max_recv_tokens])
    )

    var recv_count = shmem_malloc[.uint64](n_local_experts * n_ranks)
    var recv_count_buf = DeviceBuffer(
        ctx, recv_count, n_local_experts * n_ranks, owning=False
    )
    var atomic_counter = ctx.enqueue_create_buffer[.int32](
        EPLocalSyncCounters[n_experts].total_size()
    )

    ctx.enqueue_memset(recv_count_buf, UInt64.MAX_FINITE)
    ctx.enqueue_memset(atomic_counter, Int32(0))
    var atomic_sync_counters = EPLocalSyncCounters[n_experts](atomic_counter)

    # These host buffers are intentionally leaked (no free in the original
    # code): async `enqueue_copy` reads them, so they must outlive this scope.
    var host_topk_ids = alloc[Int32](
        {count = n_tokens_per_rank * top_k}
    ).into_managed()
    var host_input_tokens = alloc[Scalar[input_type]](
        {count = n_tokens_per_rank * hidden_size}
    ).into_managed()
    var device_topk_buf = ctx.enqueue_create_buffer[.int32](
        n_tokens_per_rank * top_k
    )
    var device_input_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * hidden_size
    )
    var device_output_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * n_ranks * n_local_experts * hidden_size
    )
    var device_output_scales_buf = ctx.enqueue_create_buffer[scales_dtype](
        max_recv_tokens * hidden_size // group_size
    )
    var device_row_offsets_buf = ctx.enqueue_create_buffer[.uint32](
        n_local_experts + 1
    )
    var device_expert_ids_buf = ctx.enqueue_create_buffer[.int32](
        n_local_experts
    )
    var device_src_token_info_buf = ctx.enqueue_create_buffer[.int32](
        n_tokens_per_rank * n_ranks * n_local_experts * 2
    )
    var device_output_2_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * top_k * hidden_size
    )

    var topk_ids_tensor = TileTensor[origin=ImmutAnyOrigin](
        device_topk_buf, row_major(n_tokens_per_rank, Idx[top_k])
    )
    var input_tokens_tensor = TileTensor[origin=ImmutAnyOrigin](
        device_input_buf,
        row_major(n_tokens_per_rank, Idx[hidden_size]),
    )
    var output_tensor = TileTensor[origin=MutAnyOrigin](
        device_output_buf,
        row_major(Idx[max_recv_tokens], Idx[hidden_size]),
    )
    var output_scales_tensor = TileTensor[origin=MutAnyOrigin](
        device_output_scales_buf,
        row_major(hidden_size // group_size, Idx[max_recv_tokens]),
    )
    var row_offsets_tensor = TileTensor[origin=MutAnyOrigin](
        device_row_offsets_buf, row_major[n_local_experts + 1]()
    )
    var expert_ids_tensor = TileTensor[origin=MutAnyOrigin](
        device_expert_ids_buf, row_major[n_local_experts]()
    )
    var src_token_info_tensor = TileTensor[origin=MutAnyOrigin](
        device_src_token_info_buf,
        row_major(Idx[max_recv_tokens], Idx[2]),
    )
    var output_2_tensor = TileTensor[origin=MutAnyOrigin](
        device_output_2_buf,
        row_major(n_tokens_per_rank, Idx[top_k], Idx[hidden_size]),
    )

    comptime hw_info = ctx.default_device_info

    # Initialize the topk ids and input tokens using fixed seed,
    # so that we can reproduce the results later on other ranks.
    seed(Int(my_rank) * n_ranks)
    randint(host_topk_ids.unsafe_span(), 0, n_experts - 1)

    # The topk ids for a token is the expert id it needs to be sent to.
    # Since a token won't be sent to the same expert multiple times, we
    # need to legalize the topk ids to make sure they are unique for
    # each token.
    legalize_topk_ids[n_experts, top_k](
        host_topk_ids.unsafe_ptr(), n_tokens_per_rank
    )

    seed(Int(my_rank) * n_ranks)
    randn(host_input_tokens.unsafe_span())

    ctx.enqueue_copy(device_topk_buf, host_topk_ids.unsafe_span())
    ctx.enqueue_copy(device_input_buf, host_input_tokens.unsafe_span())

    # synchronize before deallocating the host buffers
    ctx.synchronize()
    dealloc(host_topk_ids^)
    dealloc(host_input_tokens^)

    @always_inline
    def clean_up(
        ctx: DeviceContext,
        atomic_counter: DeviceBuffer[.int32],
    ) raises {}:
        ctx.enqueue_memset(atomic_counter, Int32(0))

    @always_inline
    def setup_and_run_benchmark[
        TokenFmtType: TokenFormat,
        FormatHandlerType: TokenFormat,
        ThroughputDtype: DType,
    ](
        ctx: DeviceContext,
        mut b: Bench,
        format_handler: FormatHandlerType,
        throughput_dtype: DType,
        atomic_counter: DeviceBuffer[.int32],
    ) raises {imm}:
        var msg_bytes = TokenFmtType.msg_size()
        var send_buf = shmem_malloc[.uint8](n_tokens_per_rank * msg_bytes)
        var recv_buf = shmem_malloc[.uint8](
            n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes
        )

        comptime dispatch_async = dispatch_async_kernel[
            input_type,
            hw_info.max_thread_block_size,
            input_tokens_tensor.LayoutType,
            topk_ids_tensor.LayoutType,
            hw_info.sm_count,
            n_experts,
            n_ranks,
            n_tokens_per_rank,
            1,  # p2p_world_size
            TokenFmtType,
        ]

        var func = ctx.compile_function[dispatch_async]()
        shmem_module_init(func)

        comptime dispatch_wait = dispatch_wait_kernel[
            hw_info.max_thread_block_size,
            row_offsets_tensor.LayoutType,
            expert_ids_tensor.LayoutType,
            src_token_info_tensor.LayoutType,
            hw_info.sm_count,
            n_experts,
            n_ranks,
            n_tokens_per_rank,
            FormatHandlerType,
        ]

        var func_dispatch_wait = ctx.compile_function[dispatch_wait]()

        comptime combine_async = combine_async_kernel[
            input_type,
            hw_info.max_thread_block_size,
            output_tensor.LayoutType,
            src_token_info_tensor.LayoutType,
            hw_info.sm_count,
            top_k,
            n_experts,
            n_ranks,
            combine_msg_bytes,
            n_tokens_per_rank,
            1,  # p2p_world_size
        ]
        var func_combine_async = ctx.compile_function[combine_async]()
        shmem_module_init(func_combine_async)

        comptime combine_wait = combine_wait_kernel[
            input_type,
            hw_info.max_thread_block_size,
            output_2_tensor.LayoutType,
            hw_info.sm_count,
            top_k,
            n_experts,
            n_ranks,
            combine_msg_bytes,
            n_tokens_per_rank,
        ]
        var func_combine_async_wait = ctx.compile_function[combine_wait]()

        @always_inline
        def run_dispatch_async(
            ctx: DeviceContext,
        ) raises {imm}:
            # the recv_buf ptrs and recv_count ptrs need to be passed in a InlinedArray
            var recv_buf_ptrs: Array[MutPointer[UInt8, MutAnyOrigin], 1] = [
                recv_buf.as_unsafe_any_origin()
            ]
            var recv_count_ptrs: Array[MutPointer[UInt64, MutAnyOrigin], 1] = [
                recv_count.as_unsafe_any_origin()
            ]

            ctx.enqueue_function(
                func,
                input_tokens_tensor,
                topk_ids_tensor,
                send_buf,
                recv_buf_ptrs,
                recv_count_ptrs,
                atomic_sync_counters,
                Int32(my_rank),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

        @always_inline
        def run_dispatch_async_wait(
            ctx: DeviceContext,
        ) raises {imm}:
            ctx.enqueue_function(
                func_dispatch_wait,
                format_handler,
                row_offsets_tensor,
                expert_ids_tensor,
                src_token_info_tensor,
                recv_buf,
                recv_count,
                atomic_sync_counters,
                Int32(my_rank),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

        @always_inline
        def run_dispatch_async_e2e(
            ctx: DeviceContext,
        ) raises {imm}:
            run_dispatch_async(ctx)
            run_dispatch_async_wait(ctx)

        @always_inline
        def run_combine_async(
            ctx: DeviceContext,
        ) raises {imm}:
            # the recv_buf ptrs and recv_count ptrs need to be passed in a InlinedArray
            var combine_recv_buf_ptrs: Array[
                MutPointer[UInt8, MutAnyOrigin], 1
            ] = [send_buf.as_unsafe_any_origin()]
            var combine_recv_count_ptrs: Array[
                MutPointer[UInt64, MutAnyOrigin], 1
            ] = [recv_count.as_unsafe_any_origin()]

            ctx.enqueue_function(
                func_combine_async,
                output_tensor.as_immut(),
                src_token_info_tensor.as_immut(),
                recv_buf,
                combine_recv_buf_ptrs,
                combine_recv_count_ptrs,
                atomic_sync_counters,
                Int32(my_rank),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

        @always_inline
        def run_combine_async_wait(
            ctx: DeviceContext,
        ) raises {imm}:
            ctx.enqueue_function(
                func_combine_async_wait,
                output_2_tensor,
                send_buf,
                recv_count,
                atomic_sync_counters,
                Int32(my_rank),
                grid_dim=hw_info.sm_count,
                block_dim=hw_info.max_thread_block_size,
            )

        @always_inline
        def run_combine_async_e2e(
            ctx: DeviceContext,
        ) raises {imm}:
            run_combine_async(ctx)
            run_combine_async_wait(ctx)

        shmem_barrier_all_on_stream(ctx.stream())

        @always_inline
        def run_dispatch_async_func() raises {imm}:
            run_dispatch_async_e2e(ctx)

        @always_inline
        def run_combine_async_func() raises {imm}:
            run_combine_async_e2e(ctx)

        @always_inline
        def run_clean_up_func() raises {imm}:
            clean_up(ctx, atomic_counter)

        @always_inline
        def dispatch_launch(
            ctx: DeviceContext,
        ) raises {imm}:
            run_dispatch_async_func()

        @always_inline
        def bench_dispatch_func(mut b: Bencher) {imm}:
            bencher_iter_custom(b, dispatch_launch, ctx)

        @always_inline
        def combine_launch(
            ctx: DeviceContext,
        ) raises {imm}:
            run_combine_async_func()

        @always_inline
        def bench_combine_func(mut b: Bencher) {imm}:
            bencher_iter_custom(b, combine_launch, ctx)

        @always_inline
        def clean_up_launch(
            ctx: DeviceContext,
        ) raises {imm}:
            run_clean_up_func()

        @always_inline
        def bench_clean_up_func(mut b: Bencher) {imm}:
            bencher_iter_custom(b, clean_up_launch, ctx)

        var input_id_parts = String(
            "n_tokens_per_rank=",
            n_tokens_per_rank,
            " top_k=",
            top_k,
            " hidden_size=",
            hidden_size,
            " n_experts=",
            n_experts,
            " n_ranks=",
            n_ranks,
            " my_rank=",
            my_rank,
            " token_dtype=",
            String(throughput_dtype),
        )

        b.bench_function(
            bench_dispatch_func,
            BenchId("ep_dispatch", input_id=input_id_parts),
            [
                ThroughputMeasure(
                    BenchMetric.bytes,
                    size_of[ThroughputDtype]()
                    * n_tokens_per_rank
                    * hidden_size,
                )
            ],
            fixed_iterations=1,
        )
        b.bench_function(
            bench_combine_func,
            BenchId("ep_combine", input_id=input_id_parts),
            [
                ThroughputMeasure(
                    BenchMetric.bytes,
                    size_of[ThroughputDtype]()
                    * n_tokens_per_rank
                    * hidden_size,
                )
            ],
            fixed_iterations=1,
        )
        b.bench_function(
            bench_clean_up_func,
            BenchId("ep_clean_up", input_id=input_id_parts),
            [
                ThroughputMeasure(
                    BenchMetric.bytes,
                    size_of[ThroughputDtype]()
                    * n_tokens_per_rank
                    * hidden_size,
                )
            ],
            fixed_iterations=1,
        )

        shmem_free(send_buf)
        shmem_free(recv_buf)

    comptime if token_dtype == .bfloat16:
        comptime token_fmt_type = BF16TokenFormat[
            output_layout=type_of(output_tt_layout), hidden_size, top_k
        ]

        comptime msg_bytes = token_fmt_type.msg_size()

        var send_buf = shmem_malloc[.uint8](n_tokens_per_rank * msg_bytes)
        var recv_buf = shmem_malloc[.uint8](
            n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes
        )

        var bf16_output = output_tensor.bitcast[
            DType.bfloat16
        ]().as_unsafe_any_origin()
        var format_handler = token_fmt_type(bf16_output)

        setup_and_run_benchmark[
            token_fmt_type,
            type_of(format_handler),
            token_dtype,
        ](
            ctx,
            b,
            format_handler,
            token_dtype,
            atomic_counter,
        )

    else:
        comptime token_fmt_type = BlockwiseFP8TokenFormat[
            fp8_dtype=token_dtype,
            scales_dtype=scales_dtype,
            output_layout=type_of(output_tt_layout),
            scales_layout=type_of(output_scales_tt_layout),
            hidden_size,
            top_k,
        ]

        var fp8_output = output_tensor.bitcast[
            token_dtype
        ]().as_unsafe_any_origin()
        var format_handler = token_fmt_type(fp8_output, output_scales_tensor)

        setup_and_run_benchmark[
            token_fmt_type,
            type_of(format_handler),
            token_dtype,
        ](
            ctx,
            b,
            format_handler,
            token_dtype,
            atomic_counter,
        )

    shmem_free(recv_count)


def main() raises:
    comptime hidden_size = get_defined_int["hidden_size", 3584]()
    comptime top_k = get_defined_int["top_k", 8]()
    comptime n_experts = get_defined_int["n_experts", 256]()
    comptime n_ranks = get_defined_int["n_ranks", 8]()
    comptime n_tokens_per_rank = get_defined_int["n_tokens_per_rank", 128]()
    comptime num_gpus = get_defined_int["num_gpus", 8]()
    comptime token_dtype = get_defined_dtype[
        "token_dtype", DType.float8_e4m3fn
    ]()
    comptime scales_dtype = get_defined_dtype["scales_dtype", .float32]()

    var m = Bench()
    var bencher_rank = m.check_mpirun()
    with SHMEMContext() as shmem_ctx:
        var mype_node = shmem_team_my_pe(SHMEM_TEAM_NODE)
        if bencher_rank != Int(mype_node):
            raise Error("bencher_rank does not match mype_node")

        bench_dispatch[
            token_dtype=token_dtype,
            scales_dtype=scales_dtype,
            hidden_size=hidden_size,
            top_k=top_k,
            n_experts=min(num_gpus * 32, n_experts),
            n_ranks=n_ranks,
            n_tokens_per_rank=n_tokens_per_rank,
        ](shmem_ctx.get_device_context(), m, Int(mype_node))

    m.dump_report()
