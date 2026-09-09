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

# RUN: %mojo-build %s -o %t
# RUN: %mpirun-gpu-per-process %t

import std.time
from std.io.io import _printf
from std.math import sqrt
from std.random import randint, randn, seed
from std.sys import argv

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx
from layout.tile_layout import row_major
from std.memory import Pointer
from shmem import *
from shmem.ep_comm import (
    BlockwiseFP8TokenFormat,
    EP_DATA_READY_FLAG,
    EPLocalSyncCounters,
    dispatch_wait_kernel,
    dispatch_async_kernel,
)
from std.testing import assert_almost_equal, assert_equal


def is_benchmark() -> Bool:
    for arg in argv():
        if arg == "--benchmark":
            return True
    return False


def is_pressure_test() -> Bool:
    for arg in argv():
        if arg == "--pressure-test":
            return True
    return False


@always_inline
def welford_update(
    mut mean: Float64, mut m2: Float64, count: Int, new_value: Float64
):
    var delta: Float64
    var delta2: Float64
    delta = new_value - mean
    mean += delta / Float64(count)
    delta2 = new_value - mean
    m2 += delta * delta2


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: Pointer[mut=True, Int32, _], n_tokens: Int):
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


def test_dispatch[
    fp8_dtype: DType,
    scales_dtype: DType,
    hidden_size: Int,
    top_k: Int,
    n_experts: Int,
    n_ranks: Int,
    n_tokens_per_rank: Int,
](ctx: DeviceContext, my_rank: Int) raises:
    comptime input_type = DType.bfloat16
    comptime group_size = 128
    comptime n_local_experts = n_experts // n_ranks
    comptime max_recv_tokens = n_experts * n_tokens_per_rank

    comptime output_tt_layout = row_major(
        (Idx[max_recv_tokens], Idx[hidden_size])
    )
    comptime output_scales_tt_layout = row_major(
        (Idx[hidden_size // group_size], Idx[max_recv_tokens])
    )
    comptime token_fmt_type = BlockwiseFP8TokenFormat[
        fp8_dtype=fp8_dtype,
        scales_dtype=scales_dtype,
        output_layout=type_of(output_tt_layout),
        scales_layout=type_of(output_scales_tt_layout),
        hidden_size,
        top_k,
    ]
    comptime msg_bytes = token_fmt_type.msg_size()

    if my_rank == 0:
        print(
            "Running ep_dispatch test: fp8_dtype:",
            fp8_dtype,
            "scales_dtype:",
            scales_dtype,
            "hidden_size:",
            hidden_size,
            "top_k:",
            top_k,
            "n_experts:",
            n_experts,
            "n_ranks:",
            n_ranks,
            "n_tokens_per_rank:",
            n_tokens_per_rank,
        )

    var send_buf = shmem_malloc[.uint8](n_tokens_per_rank * msg_bytes)
    var recv_buf = shmem_malloc[.uint8](
        n_local_experts * n_ranks * n_tokens_per_rank * msg_bytes
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

    var host_topk_ids = alloc[Int32](n_tokens_per_rank * top_k)
    var host_input_tokens = alloc[Scalar[input_type]](
        n_tokens_per_rank * hidden_size
    )

    var device_topk_buf = ctx.enqueue_create_buffer[.int32](
        n_tokens_per_rank * top_k
    )
    var device_input_buf = ctx.enqueue_create_buffer[input_type](
        n_tokens_per_rank * hidden_size
    )
    var device_output_buf = ctx.enqueue_create_buffer[fp8_dtype](
        max_recv_tokens * hidden_size
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
        max_recv_tokens * 2
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
        row_major(Idx[hidden_size // group_size], Idx[max_recv_tokens]),
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

    var format_handler = token_fmt_type(output_tensor, output_scales_tensor)

    comptime hw_info = ctx.default_device_info

    comptime dispatch_async = dispatch_async_kernel[
        input_type,
        hw_info.max_thread_block_size,
        input_tokens_tensor.LayoutType,
        topk_ids_tensor.LayoutType,
        hw_info.sm_count,
        n_experts,
        n_ranks,
        n_tokens_per_rank,
        1,  # p2p world size
        token_fmt_type,
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
        type_of(format_handler),
    ]

    var func_wait = ctx.compile_function[dispatch_wait]()

    var num_iters: Int = 100 if is_benchmark() or is_pressure_test() else 3
    var dispatch_async_stat_m: Float64 = 0
    var dispatch_async_stat_m2: Float64 = 0
    var dispatch_wait_stat_m: Float64 = 0
    var dispatch_wait_stat_m2: Float64 = 0
    var e2e_stat_m: Float64 = 0
    var e2e_stat_m2: Float64 = 0

    @always_inline
    def run_dispatch_async(ctx: DeviceContext) raises {var atomic_counter, imm}:
        # the recv_buf ptrs and recv_count ptrs need to be passed in a InlinedArray
        var recv_buf_ptrs: Array[Pointer[UInt8, MutAnyOrigin], 1] = [recv_buf]
        var recv_count_ptrs: Array[Pointer[UInt64, MutAnyOrigin], 1] = [
            recv_count
        ]

        ctx.enqueue_function(
            func,
            input_tokens_tensor,
            topk_ids_tensor,
            send_buf,
            recv_buf_ptrs,
            recv_count_ptrs,
            EPLocalSyncCounters[n_experts](atomic_counter),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    def run_dispatch_async_wait(
        ctx: DeviceContext,
    ) raises {var atomic_counter, imm}:
        ctx.enqueue_function(
            func_wait,
            format_handler,
            row_offsets_tensor,
            expert_ids_tensor,
            src_token_info_tensor,
            recv_buf,
            recv_count,
            EPLocalSyncCounters[n_experts](atomic_counter),
            Int32(my_rank),
            grid_dim=hw_info.sm_count,
            block_dim=hw_info.max_thread_block_size,
        )

    @always_inline
    def run_e2e(ctx: DeviceContext) raises {imm}:
        run_dispatch_async(ctx)
        run_dispatch_async_wait(ctx)

    @always_inline
    @__parameter
    def clean_up(ctx: DeviceContext) raises:
        ctx.enqueue_memset(atomic_counter, Int32(0))

    for i in range(num_iters):
        # Initialize the topk ids and input tokens using fixed seed,
        # so that we can reproduce the results later on other ranks.
        seed(Int(my_rank) + i * n_ranks)
        randint(host_topk_ids, n_tokens_per_rank * top_k, 0, n_experts - 1)

        # The topk ids for a token is the expert id it needs to be sent to.
        # Since a token won't be sent to the same expert multiple times, we
        # need to legalize the topk ids to make sure they are unique for
        # each token.
        legalize_topk_ids[n_experts, top_k](host_topk_ids, n_tokens_per_rank)

        seed(Int(my_rank) + i * n_ranks)
        randn(host_input_tokens, n_tokens_per_rank * hidden_size)

        ctx.enqueue_copy(device_topk_buf, host_topk_ids)
        ctx.enqueue_copy(device_input_buf, host_input_tokens)

        # warm-up
        shmem_barrier_all_on_stream(ctx.stream())
        run_e2e(ctx)
        clean_up(ctx)

        shmem_barrier_all_on_stream(ctx.stream())

        var new_value: Float64

        # First, bench kernel overhead
        new_value = Float64(ctx.execution_time(run_dispatch_async, 1)) * 1e-3
        welford_update(
            dispatch_async_stat_m, dispatch_async_stat_m2, i + 1, new_value
        )

        # sleep 10 ms to make sure transfer is finished
        std.time.sleep(1e-2)

        new_value = (
            Float64(ctx.execution_time(run_dispatch_async_wait, 1)) * 1e-3
        )
        welford_update(
            dispatch_wait_stat_m, dispatch_wait_stat_m2, i + 1, new_value
        )
        clean_up(ctx)

        # run one more time to measure bandwidth
        shmem_barrier_all_on_stream(ctx.stream())
        new_value = Float64(ctx.execution_time(run_e2e, 1)) * 1e-3
        welford_update(e2e_stat_m, e2e_stat_m2, i + 1, new_value)
        # this time we do the clean up after we verify the results

        if not is_benchmark():
            var host_output = alloc[Scalar[fp8_dtype]](
                max_recv_tokens * hidden_size
            )
            var host_output_scales = alloc[Scalar[scales_dtype]](
                max_recv_tokens * hidden_size // group_size
            )
            ctx.enqueue_copy(host_output, device_output_buf)
            ctx.enqueue_copy(host_output_scales, device_output_scales_buf)

            var host_row_offsets = alloc[UInt32](n_local_experts + 1)
            ctx.enqueue_copy(host_row_offsets, device_row_offsets_buf)

            var host_expert_ids = alloc[Int32](n_tokens_per_rank)
            ctx.enqueue_copy(host_expert_ids, device_expert_ids_buf)

            var host_src_token_info = alloc[Int32](max_recv_tokens * 2)
            ctx.enqueue_copy(host_src_token_info, device_src_token_info_buf)

            var host_atomic_counter = alloc[Int32](
                EPLocalSyncCounters[n_experts].total_size()
            )
            ctx.enqueue_copy(host_atomic_counter, atomic_counter)
            var host_dispatch_wait_counter = EPLocalSyncCounters[n_experts](
                host_atomic_counter
            ).get_dispatch_wait_ptr()

            ctx.synchronize()

            # Check the results

            # First, reproduce the input tokens and topk ids
            var all_ranks_input_tokens = alloc[Scalar[input_type]](
                n_tokens_per_rank * n_ranks * hidden_size
            )
            var all_ranks_topk_ids = alloc[Int32](
                n_tokens_per_rank * n_ranks * top_k
            )

            for rank in range(n_ranks):
                seed(Int(rank) + i * n_ranks)
                randn(
                    all_ranks_input_tokens
                    + rank * n_tokens_per_rank * hidden_size,
                    n_tokens_per_rank * hidden_size,
                )
                seed(Int(rank) + i * n_ranks)
                randint(
                    all_ranks_topk_ids + rank * n_tokens_per_rank * top_k,
                    n_tokens_per_rank * top_k,
                    0,
                    n_experts - 1,
                )
                legalize_topk_ids[n_experts, top_k](
                    all_ranks_topk_ids + rank * n_tokens_per_rank * top_k,
                    n_tokens_per_rank,
                )

            # Check if we have received the correct number of tokens
            var expert_start_idx = n_local_experts * my_rank
            var expert_end_idx = expert_start_idx + n_local_experts
            var expected_tokens = 0
            var received_tokens = 0

            # Count expected tokens from all ranks for this slot
            for i in range(n_tokens_per_rank * n_ranks * top_k):
                if (
                    expert_start_idx
                    <= Int(all_ranks_topk_ids[i])
                    < expert_end_idx
                ):
                    expected_tokens += 1

            # Then, check the output
            for expert_idx in range(n_local_experts):
                var curr_local_expert = host_expert_ids[expert_idx]
                var curr_expert = (
                    Int32(n_local_experts * my_rank) + curr_local_expert
                )

                for remote_rank in range(n_ranks):
                    var expert_rank_offset = curr_local_expert * Int32(
                        n_ranks
                    ) + Int32(remote_rank)
                    var token_end = (
                        host_dispatch_wait_counter[2 * expert_rank_offset]
                        - EP_DATA_READY_FLAG
                    )
                    var num_tokens = host_dispatch_wait_counter[
                        2 * expert_rank_offset + 1
                    ]
                    var token_start = token_end - num_tokens
                    received_tokens += Int(num_tokens)

                    for token_idx in range(token_start, token_end):
                        var remote_loc = host_src_token_info[2 * token_idx]
                        var remote_topk_id = host_src_token_info[
                            2 * token_idx + 1
                        ]

                        var remote_rank_top_k_ids = (
                            all_ranks_topk_ids
                            + remote_rank * n_tokens_per_rank * top_k
                        )

                        assert_equal(
                            remote_rank_top_k_ids[
                                remote_loc * Int32(top_k) + remote_topk_id
                            ],
                            Int32(curr_expert),
                        )

                        var remote_rank_input_tokens = (
                            all_ranks_input_tokens
                            + remote_rank * n_tokens_per_rank * hidden_size
                        )
                        for i in range(hidden_size):
                            var remote_token_val = remote_rank_input_tokens[
                                remote_loc * Int32(hidden_size) + Int32(i)
                            ]
                            var curr_fp8_val = host_output[
                                token_idx * Int32(hidden_size) + Int32(i)
                            ]
                            var curr_token_scale = host_output_scales[
                                Int32((i // group_size) * max_recv_tokens)
                                + token_idx
                            ]
                            var curr_token_val = (
                                curr_fp8_val.cast[scales_dtype]()
                                * curr_token_scale
                            )
                            assert_almost_equal(
                                remote_token_val,
                                curr_token_val.cast[input_type](),
                                String(token_idx) + ", " + String(i),
                                rtol=1e-1,
                                atol=1e-1,
                            )

            assert_equal(received_tokens, expected_tokens)
        clean_up(ctx)

    _printf[
        "Rank #%d:  Dispatch_async latency: %4.2fus ± %1.2fus  Dispatch_wait"
        " latency: %4.2fus ± %1.2fus  E2E latency: %4.2fus ± %1.2fus\n"
    ](
        my_rank,
        dispatch_async_stat_m,
        sqrt(dispatch_async_stat_m2 / Float64(num_iters)),
        dispatch_wait_stat_m,
        sqrt(dispatch_wait_stat_m2 / Float64(num_iters)),
        e2e_stat_m,
        sqrt(e2e_stat_m2 / Float64(num_iters)),
    )

    shmem_free(send_buf)
    shmem_free(recv_buf)
    shmem_free(recv_count)


def main() raises:
    comptime test_gpu_counts = (2, 4, 8)

    comptime for gpu_idx in range(len(test_gpu_counts)):
        comptime num_gpus = rebind[Int](test_gpu_counts[gpu_idx])
        if DeviceContext.number_of_devices() != num_gpus:
            continue

        with SHMEMContext() as shmem_ctx:
            var mype_node = shmem_team_my_pe(SHMEM_TEAM_NODE)
            test_dispatch[
                fp8_dtype=DType.float8_e4m3fn,
                scales_dtype=DType.float32,
                hidden_size=7168,
                top_k=8,
                n_experts=min(num_gpus * 32, 256),
                n_ranks=num_gpus,
                n_tokens_per_rank=128,
            ](shmem_ctx.get_device_context(), Int(mype_node))
