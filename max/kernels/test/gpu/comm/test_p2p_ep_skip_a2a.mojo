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

"""Single-GPU test for EP fused dispatch + combine in skip_a2a mode.

Validates that the skip_a2a path correctly routes only local-expert tokens
and that the combine kernel produces the expected weighted reduction:
  output[i] = input[i] * sum(w_k for local k)
  output[i] = input[i] * (1 + sum(w_k for local k))  [fused_shared_expert]
"""

from std.random import randint, randn, seed
from std.sys import has_nvidia_gpu_accelerator, has_amd_gpu_accelerator, size_of

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import TileTensor, Idx, row_major
from shmem.ep import (
    ep_fused_dispatch_kernel_api,
    ep_fused_combine_kernel_api,
)
from shmem.ep_comm import (
    BF16TokenFormat,
    EPLocalSyncCounters,
    router_weights_wrapper_type,
)
from std.testing import assert_almost_equal


def legalize_topk_ids[
    n_experts: Int, top_k: Int
](topk_ids: MutPointer[Int32, _], n_tokens: Int):
    for tok_id in range(n_tokens):
        var topk_ids_for_token = topk_ids + tok_id * top_k

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


def test_skip_a2a[
    hidden_size: Int,
    top_k: Int,
    n_local_experts: Int,
    n_total_experts: Int,
    n_tokens: Int,
    fused_shared_expert: Bool,
](ctx: DeviceContext) raises:
    comptime input_type = DType.bfloat16
    comptime n_gpus_per_node = 1
    comptime n_nodes = 1
    comptime max_recv_num_tokens = n_tokens * n_local_experts
    comptime combine_msg_bytes = hidden_size * size_of[Scalar[input_type]]()
    comptime shared_expert_offset = 1 if fused_shared_expert else 0

    comptime output_layout = row_major(
        (Idx[max_recv_num_tokens], Idx[hidden_size])
    )
    comptime token_fmt_type = BF16TokenFormat[
        output_layout=type_of(output_layout), hidden_size, top_k
    ]
    comptime msg_bytes = token_fmt_type.msg_size()

    var my_rank = Int(ctx.id())

    print(
        "Running skip_a2a test: hidden_size:",
        hidden_size,
        "top_k:",
        top_k,
        "n_local_experts:",
        n_local_experts,
        "n_total_experts:",
        n_total_experts,
        "n_tokens:",
        n_tokens,
        "fused_shared_expert:",
        fused_shared_expert,
        "my_rank:",
        my_rank,
    )

    # fmt: off
    # --- Device buffers ---
    var dispatch_send_buf = ctx.enqueue_create_buffer[.uint8](n_tokens * msg_bytes)
    var dispatch_recv_buf = ctx.enqueue_create_buffer[.uint8](max_recv_num_tokens * msg_bytes)
    var dispatch_recv_count_buf = ctx.enqueue_create_buffer[.uint64](n_local_experts)
    ctx.enqueue_memset(dispatch_recv_count_buf, UInt64.MAX_FINITE)

    var combine_send_buf = ctx.enqueue_create_buffer[.uint8](max_recv_num_tokens * combine_msg_bytes)
    var combine_recv_buf = ctx.enqueue_create_buffer[.uint8](n_tokens * top_k * combine_msg_bytes)
    var combine_recv_count_buf = ctx.enqueue_create_buffer[.uint64](n_local_experts)
    ctx.enqueue_memset(combine_recv_count_buf, UInt64.MAX_FINITE)

    var atomic_counters_buf = ctx.enqueue_create_buffer[.int32](EPLocalSyncCounters[n_local_experts].total_size())
    ctx.enqueue_memset(atomic_counters_buf, Int32(0))

    var device_input_buf = ctx.enqueue_create_buffer[input_type](n_tokens * hidden_size)
    var device_topk_buf = ctx.enqueue_create_buffer[.int32](n_tokens * top_k)
    var device_router_weights_buf = ctx.enqueue_create_buffer[.float32](n_tokens * top_k)
    var device_output_buf = ctx.enqueue_create_buffer[input_type](max_recv_num_tokens * hidden_size)
    var device_combine_output_buf = ctx.enqueue_create_buffer[input_type](n_tokens * hidden_size)
    var device_row_offsets_buf = ctx.enqueue_create_buffer[.uint32](n_local_experts + 1 + shared_expert_offset)
    var device_expert_ids_buf = ctx.enqueue_create_buffer[.int32](n_local_experts + shared_expert_offset)
    var device_src_info_buf = ctx.enqueue_create_buffer[.int32](max_recv_num_tokens * 2)

    # --- Host buffers ---
    var host_topk_ids = alloc[Int32](n_tokens * top_k)
    var host_input_tokens = alloc[Scalar[input_type]](n_tokens * hidden_size)
    var host_router_weights = alloc[Float32](n_tokens * top_k)

    # Initialize random data
    seed(42)
    randint(host_topk_ids, n_tokens * top_k, 0, n_total_experts - 1)
    legalize_topk_ids[n_total_experts, top_k](host_topk_ids, n_tokens)
    randn(host_input_tokens, n_tokens * hidden_size)
    randn(host_router_weights, n_tokens * top_k)

    ctx.enqueue_copy(device_topk_buf, host_topk_ids)
    ctx.enqueue_copy(device_input_buf, host_input_tokens)
    ctx.enqueue_copy(device_router_weights_buf, host_router_weights)

    # --- Pointer TileTensors ---
    # The ep.mojo API reads device buffer addresses from host-side uint64 arrays.
    comptime N_DEVICES = 2
    var ptrs_layout = row_major[N_DEVICES]()

    var dispatch_send_ptrs = alloc[UInt64](N_DEVICES)
    dispatch_send_ptrs[my_rank] = UInt64(Int(dispatch_send_buf.unsafe_ptr()))
    var dispatch_recv_ptrs = alloc[UInt64](N_DEVICES)
    dispatch_recv_ptrs[my_rank] = UInt64(Int(dispatch_recv_buf.unsafe_ptr()))
    var dispatch_recv_count_ptrs = alloc[UInt64](N_DEVICES)
    dispatch_recv_count_ptrs[my_rank] = UInt64(Int(dispatch_recv_count_buf.unsafe_ptr()))

    var combine_send_ptrs = alloc[UInt64](N_DEVICES)
    combine_send_ptrs[my_rank] = UInt64(Int(combine_send_buf.unsafe_ptr()))
    var combine_recv_ptrs = alloc[UInt64](N_DEVICES)
    combine_recv_ptrs[my_rank] = UInt64(Int(combine_recv_buf.unsafe_ptr()))
    var combine_recv_count_ptrs = alloc[UInt64](N_DEVICES)
    combine_recv_count_ptrs[my_rank] = UInt64(Int(combine_recv_count_buf.unsafe_ptr()))
    # fmt: on

    # --- Layouts ---
    var topk_ids_layout = row_major(n_tokens, Idx[top_k])
    var input_tokens_layout = row_major((n_tokens, Idx[hidden_size]))
    var row_offsets_layout = row_major[
        n_local_experts + 1 + shared_expert_offset
    ]()
    var expert_ids_layout = row_major[n_local_experts + shared_expert_offset]()
    var src_info_layout = row_major((Idx[max_recv_num_tokens], Idx[2]))
    var combine_output_layout = row_major((n_tokens, Idx[hidden_size]))
    comptime counters_size = EPLocalSyncCounters[n_local_experts].total_size()
    var counters_layout = row_major[counters_size]()

    # --- Build TileTensors ---
    # fmt: off
    var input_tokens_tt = TileTensor(device_input_buf, input_tokens_layout)
    var topk_ids_tt = TileTensor(device_topk_buf, topk_ids_layout)
    var row_offsets_tt = TileTensor(device_row_offsets_buf, row_offsets_layout)
    var expert_ids_tt = TileTensor(device_expert_ids_buf, expert_ids_layout)
    var src_info_tt = TileTensor(device_src_info_buf, src_info_layout)
    var atomic_counters_tt = TileTensor(atomic_counters_buf, counters_layout)
    var dispatch_send_ptrs_tt = TileTensor(dispatch_send_ptrs, ptrs_layout)
    var dispatch_recv_ptrs_tt = TileTensor(dispatch_recv_ptrs, ptrs_layout)
    var dispatch_recv_count_ptrs_tt = TileTensor(dispatch_recv_count_ptrs, ptrs_layout)
    var combine_send_ptrs_tt = TileTensor(combine_send_ptrs, ptrs_layout)
    var combine_recv_ptrs_tt = TileTensor(combine_recv_ptrs, ptrs_layout)
    var combine_recv_count_ptrs_tt = TileTensor(combine_recv_count_ptrs, ptrs_layout)
    # fmt: on

    var dispatch_output_tt = TileTensor(device_output_buf, output_layout)
    var token_handler = token_fmt_type(dispatch_output_tt)

    var combine_output_tt = TileTensor(
        device_combine_output_buf, combine_output_layout
    )

    # --- Router weights wrapper (captures TileTensor for GPU-side use) ---
    var router_weights_layout = row_major((n_tokens, Idx[top_k]))
    var router_weights_tt = TileTensor(
        device_router_weights_buf, router_weights_layout
    )

    @always_inline
    @__parameter
    @__copy_capture(router_weights_tt)
    def router_weights_fn[
        width: Int
    ](token_idx: Int, topk_id: Int) -> SIMD[.float32, width]:
        var w = router_weights_tt.load[width=1]((token_idx, topk_id))
        return SIMD[.float32, width](w)

    # --- Run fused dispatch ---
    ep_fused_dispatch_kernel_api[
        n_local_experts,
        n_tokens,
        n_gpus_per_node,
        n_nodes,
        fused_shared_expert,
        "gpu",
        skip_a2a=True,
        allreduce_world_size=N_DEVICES,
    ](
        token_handler,
        row_offsets_tt,
        expert_ids_tt,
        src_info_tt,
        atomic_counters_tt,
        input_tokens_tt,
        topk_ids_tt,
        dispatch_send_ptrs_tt,
        dispatch_recv_ptrs_tt,
        dispatch_recv_count_ptrs_tt,
        ctx,
    )

    # --- Run fused combine ---
    var dispatch_output_immut = TileTensor(device_output_buf, output_layout)
    var src_info_immut = TileTensor(device_src_info_buf, src_info_layout)

    var topk_ids_immut_ptr = (
        device_topk_buf.unsafe_ptr()
        .as_imm()
        .unsafe_origin_cast[ImmUntrackedOrigin]()
    )

    ep_fused_combine_kernel_api[
        hidden_size,
        top_k,
        n_local_experts,
        n_tokens,
        n_gpus_per_node,
        n_nodes,
        "gpu",
        router_weights_wrapper=router_weights_fn,
        fused_shared_expert=fused_shared_expert,
        skip_a2a=True,
        allreduce_world_size=N_DEVICES,
    ](
        combine_output_tt,
        atomic_counters_tt,
        dispatch_output_immut,
        src_info_immut,
        combine_send_ptrs_tt,
        combine_recv_ptrs_tt,
        combine_recv_count_ptrs_tt,
        ctx,
        topk_ids_p=topk_ids_immut_ptr,
    )

    ctx.synchronize()

    # --- Verify results ---
    print("Verifying results...")
    var host_output = alloc[Scalar[input_type]](n_tokens * hidden_size)
    ctx.enqueue_copy(host_output, device_combine_output_buf)
    ctx.synchronize()

    for token_idx in range(n_tokens):
        for h in range(hidden_size):
            var input_val = host_input_tokens[token_idx * hidden_size + h].cast[
                DType.float32
            ]()

            # Accumulate per-element exactly as the kernel does: for each
            # local top-k selection, accum += weight * token_value_f32.
            var accum = Float32(0)
            for k in range(top_k):
                var expert_id = host_topk_ids[token_idx * top_k + k]
                if (
                    my_rank * n_local_experts
                    <= Int(expert_id)
                    < (my_rank + 1) * n_local_experts
                ):
                    var w = host_router_weights[token_idx * top_k + k]
                    accum += w * input_val

            # With _allreduce_world_size=N_DEVICES, each rank handles a
            # slice of the shared expert tokens. rank_start..rank_start+count
            # is this rank's slice.
            if fused_shared_expert:
                var ar_world = N_DEVICES
                var rank_start = my_rank * (n_tokens // ar_world) + min(
                    my_rank, n_tokens % ar_world
                )
                var rank_count = (n_tokens + ar_world - my_rank - 1) // ar_world
                if rank_start <= token_idx < rank_start + rank_count:
                    accum += input_val

            var actual = host_output[token_idx * hidden_size + h]

            assert_almost_equal(
                actual,
                accum.cast[input_type](),
                "Mismatch at token "
                + String(token_idx)
                + " hidden "
                + String(h),
                rtol=5e-2,
                atol=5e-3,
            )

    print("All results verified successfully!")

    # --- Cleanup ---
    host_topk_ids.free()
    host_input_tokens.free()
    host_router_weights.free()
    host_output.free()
    dispatch_send_ptrs.free()
    dispatch_recv_ptrs.free()
    dispatch_recv_count_ptrs.free()
    combine_send_ptrs.free()
    combine_recv_ptrs.free()
    combine_recv_count_ptrs.free()

    # Extend the lifetime of the device buffers, as we directly use the pointers
    _ = dispatch_send_buf^
    _ = dispatch_recv_buf^
    _ = dispatch_recv_count_buf^
    _ = combine_send_buf^
    _ = combine_recv_buf^
    _ = combine_recv_count_buf^
    _ = device_topk_buf^
    _ = device_router_weights_buf^


def main() raises:
    comptime assert (
        has_nvidia_gpu_accelerator() or has_amd_gpu_accelerator()
    ), "Only NVIDIA and AMD GPUs are supported"

    var ctx_0 = DeviceContext(device_id=0)
    var ctx_1 = DeviceContext(device_id=1)

    test_skip_a2a[
        hidden_size=7168,
        top_k=8,
        n_local_experts=48,
        n_total_experts=384,
        n_tokens=32,
        fused_shared_expert=False,
    ](ctx_0)

    test_skip_a2a[
        hidden_size=7168,
        top_k=8,
        n_local_experts=48,
        n_total_experts=384,
        n_tokens=32,
        fused_shared_expert=True,
    ](ctx_0)

    test_skip_a2a[
        hidden_size=7168,
        top_k=8,
        n_local_experts=48,
        n_total_experts=384,
        n_tokens=32,
        fused_shared_expert=False,
    ](ctx_1)

    test_skip_a2a[
        hidden_size=7168,
        top_k=8,
        n_local_experts=48,
        n_total_experts=384,
        n_tokens=32,
        fused_shared_expert=True,
    ](ctx_1)
