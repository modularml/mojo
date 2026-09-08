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
"""Implements Mixture-of-Experts (MoE) routing, token dispatch, and expert computation kernels."""

from std.collections import OptionalReg

from std.math import align_up, ceildiv, exp, log1p
from std.math.uutils import umod
from std.memory import unsafe_stack_allocation

from std.atomic import Atomic, Ordering
from shmem.ep_comm import BLOCK_SCOPE
from std.sys import has_apple_gpu_accelerator
from std.sys.info import is_amd_gpu, is_nvidia_gpu

import max.gpu.primitives.warp as warp
import max.gpu.primitives.block as block
from std.bit import pop_count, log2_floor
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    warp_id,
    lane_id,
    thread_idx,
)
from max.gpu.sync import barrier
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from max.gpu.host.info import is_gpu
from layout import (
    Coord,
    Idx,
    TensorLayout,
    TileTensor,
    row_major,
    stack_allocation as tensor_alloc,
)
from max.runtime.tracing import Trace, TraceLevel

from std.utils.index import IndexList, StaticTuple

from nn.activations import sigmoid
from nn.topk import TopK_2


from max.gpu.memory import (
    async_copy,
    async_copy_commit_group,
    async_copy_wait_all,
)


# Experts are visited in chunks of one block width, carrying the running token
# and aligned-scale offsets across chunks, so an expert count above the block
# width costs another pass over the tokens instead of another kernel. Every
# expert count in use today fits in a single chunk.
comptime _BLOCK_THREADS = 512


@always_inline
def _cta_atomic_scope() -> StaticString:
    """Returns the narrowest atomic scope the target has for CTA-local counters.

    Returns:
        The block-scope syncscope on NVIDIA and AMD, and the default (system)
        scope elsewhere: Metal exposes no block scope, and system scope is
        still correct for shared memory, only slower.
    """
    comptime if is_nvidia_gpu() or is_amd_gpu():
        return BLOCK_SCOPE
    else:
        return ""


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(_BLOCK_THREADS))
)
@__name(t"moe_create_indices_{input_type}")
def moe_create_indices_kernel[
    input_type: DType,
    TokenExpertOrderLayoutType: TensorLayout,
    ExpertStartIndicesLayoutType: TensorLayout,
    RestoreTokenOrderLayoutType: TensorLayout,
    ExpertIdsLayoutType: TensorLayout,
    ExpertUsageStatsLayoutType: TensorLayout,
    TopkIdsLayoutType: TensorLayout,
    _scale_alignment: UInt32 = 128,
](
    token_expert_order: TileTensor[
        mut=True, .uint32, TokenExpertOrderLayoutType, MutAnyOrigin
    ],
    expert_start_indices: TileTensor[
        mut=True, .uint32, ExpertStartIndicesLayoutType, MutAnyOrigin
    ],
    restore_token_order: TileTensor[
        mut=True, .uint32, RestoreTokenOrderLayoutType, MutAnyOrigin
    ],
    expert_ids: TileTensor[mut=True, .int32, ExpertIdsLayoutType, MutAnyOrigin],
    expert_usage_stats: TileTensor[
        mut=True, .uint32, ExpertUsageStatsLayoutType, MutAnyOrigin
    ],
    topk_ids: TileTensor[input_type, TopkIdsLayoutType, ImmutAnyOrigin],
    scales_offset_p: Optional[UnsafePointer[UInt32, MutAnyOrigin]],
):
    """Builds MoE routing indices in one CTA using a block-wide scan.

    Groups tokens by their assigned expert so downstream kernels such as
    grouped matmul can process each expert's tokens contiguously. Every expert
    in `[0, num_experts)` gets a slot, including ones with zero tokens;
    `expert_usage_stats` holds the largest per-expert token count and
    `num_experts`.

    Slot order is ascending expert id; token order within a slot follows the
    shared cursor. Neither is load-bearing: consumers pair
    `expert_start_indices[g]` and `[g + 1]` with `expert_ids[g]`, and invert
    through `restore_token_order`.
    """
    comptime assert token_expert_order.flat_rank == 1
    comptime assert expert_start_indices.flat_rank == 1
    comptime assert restore_token_order.flat_rank == 1
    comptime assert expert_ids.flat_rank == 1
    comptime assert expert_usage_stats.flat_rank == 1
    comptime assert topk_ids.flat_rank == 1

    comptime atomic_scope = _cta_atomic_scope()

    var total_tokens = Int(topk_ids.dim(0))
    var num_experts = Int(expert_ids.dim(0))
    var tid = Int(thread_idx.x)

    var counts = tensor_alloc[.uint32, address_space=.SHARED](
        row_major[_BLOCK_THREADS]()
    )

    # Tokens, and aligned scale slots, placed by the preceding expert chunks.
    var token_carry = UInt32(0)
    var scale_carry = UInt32(0)
    var max_count = UInt32(0)

    for base in range(0, num_experts, _BLOCK_THREADS):
        var chunk_experts = min(_BLOCK_THREADS, num_experts - base)
        var expert = base + tid

        counts[Coord(tid)] = 0
        barrier()

        # Counts commute, so increment order does not matter.
        # Block scope and relaxed are load-bearing: the default is system
        # scope and sequentially consistent, which fences every increment
        # and costs about 5x. The barrier below is what publishes these
        # counts to their readers. That holds on NVIDIA, where the barrier
        # fences shared memory; on AMD it rests on the backend emitting
        # s_waitcnt lgkmcnt(0) ahead of s_barrier.
        for i in range(tid, total_tokens, _BLOCK_THREADS):
            var e = Int(topk_ids[i]) - base
            if e < 0 or e >= chunk_experts:
                continue
            _ = Atomic[UInt32, scope=atomic_scope].fetch_add[
                ordering=Ordering.RELAXED
            ](counts.ptr + e, UInt32(1))
        barrier()

        var count_val = counts[Coord(tid)]
        var aligned_val = align_up(count_val, _scale_alignment)

        # These collectives do not leave the block synchronized, so each needs
        # a barrier before the next reuses its scratch.
        var g_offset = token_carry + block.prefix_sum[
            block_size=_BLOCK_THREADS, exclusive=True
        ](count_val)
        barrier()

        var aligned_g_offset = scale_carry + block.prefix_sum[
            block_size=_BLOCK_THREADS, exclusive=True
        ](aligned_val)
        barrier()

        # broadcast=False leaves the reduced value in warp 0 lane 0 only,
        # which is why expert_usage_stats[0] is written from tid 0 below. That
        # output sizes downstream grouped-matmul work, so if the primitive ever
        # changes which thread holds the result, that write has to follow.
        max_count = max(
            max_count,
            block.max[block_size=_BLOCK_THREADS, broadcast=False](count_val),
        )

        if expert < num_experts:
            expert_ids[Coord(expert)] = Int32(expert)
            expert_start_indices[Coord(expert)] = g_offset
            if scales_offset_p:
                var _ptr = scales_offset_p.value()
                _ptr[expert] = (
                    aligned_g_offset // _scale_alignment
                    - g_offset // _scale_alignment
                )

        if expert == num_experts - 1:
            # Last expert's exclusive prefix plus its own count is the grand
            # total: the final `expert_start_indices` entry.
            expert_start_indices[Coord(num_experts)] = g_offset + count_val

        # The last thread's exclusive prefix plus its own count is the chunk
        # total. Only another chunk needs it, so a single chunk pays nothing.
        # A block collective does not leave the block synchronized and its
        # scratch is private to the primitive, so every pair of them gets a
        # barrier between. That is what the three here are for.
        if base + _BLOCK_THREADS < num_experts:
            barrier()
            token_carry = block.broadcast[block_size=_BLOCK_THREADS](
                g_offset + count_val, src_thread=_BLOCK_THREADS - 1
            )
            barrier()
            scale_carry = block.broadcast[block_size=_BLOCK_THREADS](
                aligned_g_offset + aligned_val, src_thread=_BLOCK_THREADS - 1
            )
            barrier()

        # Reuse `counts` as the scatter cursor: seed each slot with its own
        # start offset, then atomically claim positions from it.
        counts[Coord(tid)] = g_offset
        barrier()

        for i in range(tid, total_tokens, _BLOCK_THREADS):
            var e = Int(topk_ids[i]) - base
            if e < 0 or e >= chunk_experts:
                continue
            var pos = Atomic[UInt32, scope=atomic_scope].fetch_add[
                ordering=Ordering.RELAXED
            ](counts.ptr + e, UInt32(1))
            token_expert_order[Coord(pos)] = UInt32(i)
            restore_token_order[Coord(i)] = pos
        barrier()

    if tid == 0:
        expert_usage_stats[Coord(0)] = max_count
        expert_usage_stats[Coord(1)] = UInt32(num_experts)


@always_inline
def moe_create_indices[
    input_type: DType,
    //,
    target: StaticString,
](
    token_expert_order: TileTensor[mut=True, .uint32, ...],
    expert_start_indices: TileTensor[mut=True, .uint32, ...],
    restore_token_order: TileTensor[mut=True, .uint32, ...],
    expert_ids: TileTensor[mut=True, .int32, ...],
    expert_usage_stats: TileTensor[mut=True, .uint32, ...],
    topk_ids: TileTensor[mut=False, input_type, ...],
    context: DeviceContext,
    scales_offset_p: Optional[UnsafePointer[UInt32, MutAnyOrigin]] = None,
) raises:
    """Launches the MoE index creation kernel on GPU.

    Groups tokens by their assigned expert so that downstream kernels such as
    grouped matmul can process each expert's tokens contiguously. One CTA
    histograms the expert ids in shared memory, scans the histogram into CSR
    offsets, and scatters the tokens.

    Parameters:
        input_type: DType of the topk_ids tensor.
        target: The target device to run the kernel on.

    Args:
        token_expert_order: Output 1D tensor of token indices grouped by expert.
        expert_start_indices: Output 1D tensor of CSR-style start offsets for
            each expert in token_expert_order.
        restore_token_order: Output 1D tensor mapping each token to its new
            position in token_expert_order.
        expert_ids: Output 1D tensor of the expert IDs in output order, one
            slot per expert in ascending ID order.
        expert_usage_stats: Output 1D tensor holding the maximum tokens
            assigned to any expert and the expert count.
        topk_ids: Input 1D tensor of expert IDs, one per token.
        context: The device context.
        scales_offset_p: Optional pointer receiving the aligned scale offsets
            for FP8/block-scaled grouped matmul.
    """
    comptime assert is_gpu[
        target
    ](), "Creating MoE indices is only supported on GPU"

    comptime if has_apple_gpu_accelerator():
        # Above one chunk the kernel repeats its block collectives in a loop,
        # which desynchronizes warps on Metal (see the Apple path in
        # `nn/sampling/topk_fi.mojo`). Apple rejected these expert counts
        # before this kernel existed too, so nothing that worked is lost.
        if expert_ids.dim(0) > _BLOCK_THREADS:
            raise Error(
                t"Apple MoE: num_experts={expert_ids.dim(0)} exceeds the"
                t" 512-expert single-chunk cap"
            )

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.create_indices", task_id=Int(context.id())
    ):
        comptime kernel = moe_create_indices_kernel[
            input_type,
            token_expert_order.LayoutType,
            expert_start_indices.LayoutType,
            restore_token_order.LayoutType,
            expert_ids.LayoutType,
            expert_usage_stats.LayoutType,
            topk_ids.LayoutType,
        ]

        context.enqueue_function[kernel](
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            topk_ids,
            scales_offset_p,
            grid_dim=1,
            block_dim=_BLOCK_THREADS,
        )


# Function to perform warp-level sorting
@always_inline
@__parameter
def _warp_bitonic_sort[
    T: DType,
    num_lanes: Int = WARP_SIZE,
    descending: Bool = True,
](_val: TopK_2[T]) -> TopK_2[T]:
    """
    Performs warp-level bitonic sort to sort TopK_2 elements.

    Parameters:
        T: DType - Data type of the values being compared.
        num_lanes: Int - Number of lanes that participate in the reduction.
        descending: Bool - Whether to sort in descending order.

    Arguments:
        _val: TopK_2[T] - TopK_2 value from each thread to be sorted.

    Returns:
        TopK_2[T] - Sorted TopK_2 value across the warp.
    """

    comptime assert num_lanes.is_power_of_two(), "num_lanes must be power of 2"

    @always_inline
    def bitonic_sort_step(
        v: TopK_2[T],
        step: UInt32,
        stage: UInt32,
        i: UInt32,
    ) -> TopK_2[T]:
        var partner = TopK_2[T](
            u=warp.shuffle_xor(v.u, step),  # u is the value
            p=Int(warp.shuffle_xor(Int32(v.p), step)),  # p is the index
        )

        var cmp_val = (v.u < partner.u) ^ descending
        if v.u == partner.u:
            cmp_val = v.p > partner.p

        var merge_direction = pop_count(i & (stage | step)) == 1

        if cmp_val == merge_direction:
            return partner
        else:
            return v

    var val = _val
    # Use modulo so merge direction is consistent across all lane groups
    var i = UInt32(umod(lane_id(), num_lanes))

    comptime for stage_i in range(1, log2_floor(num_lanes) + 1):
        var stage = 1 << stage_i

        comptime for step_i in reversed(range(stage_i)):
            var step = 1 << step_i
            val = bitonic_sort_step(val, UInt32(step), UInt32(stage), i)

    return val


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"group_limited_router_{scores_type}_{bias_type}_t{num_threads}",
)
def group_limited_router_kernel[
    scores_type: DType,
    bias_type: DType,
    ExpertIndicesLayoutType: TensorLayout,
    ExpertWeightsLayoutType: TensorLayout,
    ExpertScoresLayoutType: TensorLayout,
    ExpertBiasLayoutType: TensorLayout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
    norm_weights: Bool,
    num_threads: Int,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[
        mut=True, .int32, ExpertIndicesLayoutType, MutAnyOrigin
    ],
    expert_weights: TileTensor[
        mut=True, scores_type, ExpertWeightsLayoutType, MutAnyOrigin
    ],
    expert_scores: TileTensor[
        scores_type, ExpertScoresLayoutType, ImmutAnyOrigin
    ],
    expert_bias: TileTensor[bias_type, ExpertBiasLayoutType, ImmutAnyOrigin],
    routed_scaling_factor: Float32,
):
    """A manually fused MoE router with the group-limited strategy. It divides all
    the experts into `n_groups` groups and then finds the top `topk_group`
    groups with the highest scores. The final experts for each token are
    selected from the experts in the selected groups. The bias will be applied
    to the scores during the selection process, but the final weights will not
    include the bias.

    Parameters:
        scores_type: DType of the routing scores and the output expert
            weights.
        bias_type: DType of the per-expert bias added to scores during
            selection.
        ExpertIndicesLayoutType: `TensorLayout` of the `expert_indices`
            output tensor.
        ExpertWeightsLayoutType: `TensorLayout` of the `expert_weights`
            output tensor.
        ExpertScoresLayoutType: `TensorLayout` of the `expert_scores`
            input tensor.
        ExpertBiasLayoutType: `TensorLayout` of the `expert_bias` input
            tensor.
        n_routed_experts: Total number of routed experts scored per token.
            Also equals the thread count per block.
        n_experts_per_tok: Number of experts selected per token (the top-k
            value).
        n_groups: Number of groups the routed experts are partitioned into.
            `n_routed_experts` must be divisible by this.
        topk_group: Number of highest-scoring groups from which the final
            experts are selected.
        norm_weights: Whether to normalize the selected weights to sum to
            one before applying the scaling factor.
        num_threads: Threads per block; must equal `n_routed_experts` so
            each thread scores one expert.
        scores_input_fn: Optional lambda that loads scores given a
            `(token, expert)` index; when `None`, scores load from
            `expert_scores`.

    Args:
        expert_indices: Output tensor holding the selected expert index per
            token. Shape `[num_tokens, n_experts_per_tok]`.
        expert_weights: Output tensor holding the routing weight per
            selected expert per token, excluding the bias. Shape
            `[num_tokens, n_experts_per_tok]`.
        expert_scores: Input tensor of routing scores for every expert per
            token. Shape `[num_tokens, n_routed_experts]`.
        expert_bias: Input tensor of per-expert bias added to scores during
            selection but excluded from the final weights. Shape
            `[n_routed_experts]`.
        routed_scaling_factor: Factor multiplied into the final, optionally
            normalized, expert weights.
    """
    comptime assert expert_indices.flat_rank == 2
    comptime assert expert_weights.flat_rank == 2
    comptime assert expert_scores.flat_rank == 2
    comptime assert expert_bias.flat_rank == 1
    comptime assert expert_bias.flat_rank >= 1
    comptime assert expert_scores.flat_rank >= 2
    comptime assert expert_indices.flat_rank >= 2

    comptime assert (
        expert_scores.static_shape[1] == n_routed_experts
    ), "expert_scores.static_shape[1] must be equal to n_routed_experts"

    comptime assert (
        expert_indices.static_shape[1] == n_experts_per_tok
    ), "expert_indices.static_shape[1] must be equal to n_experts_per_tok"
    comptime assert (
        expert_weights.static_shape[1] == n_experts_per_tok
    ), "expert_weights.static_shape[1] must be equal to n_experts_per_tok"

    comptime group_size = n_routed_experts // n_groups
    comptime assert (
        WARP_SIZE % group_size == 0
    ), "WARP_SIZE must be divisible by group_size"
    comptime n_groups_per_warp = WARP_SIZE // group_size
    comptime assert (
        topk_group * n_experts_per_tok <= WARP_SIZE
    ), "topk_group * n_experts_per_tok must be less than or equal to WARP_SIZE"

    comptime assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    var token_idx = block_idx.x
    var tid = thread_idx.x
    var warp_id = tid // WARP_SIZE

    var num_tokens = expert_scores.dim(0)

    var shared_mem = unsafe_stack_allocation[
        topk_group * n_experts_per_tok,
        TopK_2[scores_type],
        address_space=.SHARED,
    ]()
    var selected_group = unsafe_stack_allocation[
        topk_group, DType.int32, address_space=.SHARED
    ]()
    var thread_group_id, tid_in_group = divmod(tid, group_size)

    var thread_expert_bias = expert_bias.load[width=1](Coord(tid)).cast[
        scores_type
    ]()

    with PDL():
        var thread_expert_score: Scalar[scores_type]

        comptime if scores_input_fn:
            comptime scores_fn = scores_input_fn.value()
            thread_expert_score = scores_fn[width=1]((token_idx, tid))
        else:
            thread_expert_score = expert_scores.load[width=1]((token_idx, tid))

        thread_expert_score += thread_expert_bias
        var thd_topk2 = TopK_2(u=thread_expert_score, p=tid)
        var sorted_group = _warp_bitonic_sort[num_lanes=group_size](thd_topk2)

        # In each group, the sum of the first two highest scores is the
        # score for the group. Store the two scores in shared memory.

        if tid_in_group == 0 or tid_in_group == 1:
            shared_mem[2 * thread_group_id + tid_in_group] = TopK_2(
                u=sorted_group.u, p=thread_group_id
            )
        barrier()

        # The first warp finds the `topk_group` groups with the highest scores.
        if warp_id == 0:
            if tid < n_groups:
                var group_scores = (
                    shared_mem[2 * tid].u + shared_mem[2 * tid + 1].u
                )
                thd_topk2 = TopK_2(u=group_scores, p=tid)
            else:
                thd_topk2 = TopK_2[scores_type]()

            var sorted_group_id = _warp_bitonic_sort[num_lanes=n_groups](
                thd_topk2
            )

            if tid < topk_group:
                selected_group[tid] = Int32(sorted_group_id.p)

        # Check if this group is selected
        barrier()
        var selected_group_smem_offset: Int32 = -1

        comptime for i in range(topk_group):
            if selected_group[i] == Int32(thread_group_id):
                selected_group_smem_offset = Int32(i * n_experts_per_tok)

        if selected_group_smem_offset >= 0:
            # Store the selected group's top `n_experts_per_tok` experts in
            # shared memory.
            if tid_in_group < n_experts_per_tok:
                shared_mem[
                    selected_group_smem_offset + Int32(tid_in_group)
                ] = sorted_group

        # Now, we use the first warp to find the global top `n_experts_per_tok` experts.
        barrier()
        if warp_id == 0:
            if tid < topk_group * n_experts_per_tok:
                thd_topk2 = shared_mem[tid]
            else:
                thd_topk2 = TopK_2[scores_type]()

            var global_topk_result = _warp_bitonic_sort[
                num_lanes=topk_group * n_experts_per_tok
            ](thd_topk2)

            var weights_sum: Scalar[scores_type] = 0
            var original_weight: Scalar[scores_type] = 0

            if tid < n_experts_per_tok:
                # We need to subtract the expert bias from the weight to get the original score.
                # This global load shouldn't be a problem since the expert bias is likely to be cached in L1.
                original_weight = (
                    global_topk_result.u
                    - expert_bias.load[width=1](
                        Coord(global_topk_result.p)
                    ).cast[scores_type]()
                )

            weights_sum = warp.lane_group_sum[num_lanes=n_experts_per_tok](
                original_weight
            )

            comptime if norm_weights:
                original_weight /= weights_sum

            original_weight *= Scalar[scores_type](routed_scaling_factor)

            if tid < n_experts_per_tok:
                expert_indices.store(
                    (token_idx, tid), Int32(global_topk_result.p)
                )
                expert_weights[token_idx, tid] = original_weight


@always_inline
def router_group_limited[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_groups: Int,
    topk_group: Int,
    norm_weights: Bool,
    target: StaticString,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[mut=True, .int32, ...],
    expert_weights: TileTensor[mut=True, scores_type, ...],
    expert_scores: TileTensor[mut=False, scores_type, ...],
    expert_bias: TileTensor[mut=False, bias_type, ...],
    routed_scaling_factor: Float32,
    context: DeviceContext,
) raises:
    """
    A manually fused MoE router with the group-limited strategy.

    Reference: https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py#L566.

    Parameters:
        scores_type: The data type of the scores and the output weights.
        bias_type: The data type of the expert bias.
        n_routed_experts: The number of experts to route to.
        n_experts_per_tok: The number of experts to be selected per token.
        n_groups: The number of expert groups.
        topk_group: The number of expert groups to be selected per token.
        norm_weights: Whether to normalize the selected weights.
        target: The target device to run the kernel on.
        scores_input_fn: Input lambda function to load the scores.

    Inputs:
        expert_indices: The indices of the routed experts for each token.
            Shape: [num_tokens, num_experts_per_tok].
        expert_weights: The weights of the routed experts for each token.
            Shape: [num_tokens, num_experts_per_tok].
        expert_scores: The scores for each expert for each token. Shape:
            [num_tokens, n_routed_experts].
        expert_bias: The bias for each expert. Shape: [n_routed_experts].
        routed_scaling_factor: The scaling factor for the routed expert weights.
        context: The device context.
    """
    comptime assert is_gpu[
        target
    ](), "Group limited MoE router is only supported on GPU"

    if expert_scores.dim(0) == 0:
        return

    var gpu_ctx = context

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.router_group_limited", task_id=Int(gpu_ctx.id())
    ):
        comptime num_threads = n_routed_experts
        comptime hw_info = gpu_ctx.default_device_info
        comptime blocks_per_sm = hw_info.threads_per_multiprocessor // num_threads

        comptime num_sms = hw_info.sm_count

        comptime kernel = group_limited_router_kernel[
            scores_type,
            bias_type,
            expert_indices.LayoutType,
            expert_weights.LayoutType,
            expert_scores.LayoutType,
            expert_bias.LayoutType,
            n_routed_experts,
            n_experts_per_tok,
            n_groups,
            topk_group,
            norm_weights,
            num_threads,
            scores_input_fn=scores_input_fn,
        ]

        gpu_ctx.enqueue_function[kernel](
            expert_indices,
            expert_weights,
            expert_scores,
            expert_bias,
            routed_scaling_factor,
            grid_dim=expert_scores.dim(0),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@always_inline
def _block_top_k[
    scores_type: DType,
    //,
    n_experts_per_tok: Int,
    num_threads: Int,
](biased_score: Scalar[scores_type]) -> TopK_2[scores_type]:
    """Selects a block's top `n_experts_per_tok` scores by warp-bitonic sort.

    One score per thread, `num_threads` per block. Runs in 2 or 3 phases
    depending on WARP_SIZE: each warp sorts its own lanes and keeps its top
    `n_experts_per_tok` (phase 1), the survivors are re-sorted down to one
    warp's worth (phase 2, eliminated at compile time when phase 1 already
    fits in one warp, as it does on AMD's 64-lane wavefronts), and warp 0
    sorts those to the global top k (phase 3).

    All threads must call this: it barriers between phases. Ties break by
    lower index, per `_warp_bitonic_sort`.

    Parameters:
        scores_type: DType of the scores being ranked.
        n_experts_per_tok: Number of winners to select.
        num_threads: Threads per block; also the number of scores ranked.

    Args:
        biased_score: This thread's score.

    Returns:
        In warp 0, lane `i < n_experts_per_tok` holds the `i`th-largest score
        and its originating thread index. Every other lane and warp gets an
        unspecified value.
    """
    comptime assert (
        num_threads % WARP_SIZE == 0
    ), "num_threads must be a whole number of warps"

    # Phase 1 produces num_warps × n_experts_per_tok survivors:
    comptime num_warps = num_threads // WARP_SIZE
    comptime phase1_candidates = num_warps * n_experts_per_tok

    # Phase 2 spreads the phase-1 survivors over ceil(ph1/WARP_SIZE) warps,
    # each sorting a full warp:
    comptime num_phase2_warps = ceildiv(phase1_candidates, WARP_SIZE)
    comptime phase2_candidates = num_phase2_warps * n_experts_per_tok

    # A 64-lane wavefront fits more survivors per warp, so phase 1 can
    # already leave one warp's worth and phase 2 drops out entirely.
    comptime skip_phase2 = (num_phase2_warps == 1)
    # Phase 3 takes ph2_candidates padded up to WARP_SIZE.
    comptime assert (
        phase2_candidates <= WARP_SIZE
    ), "phase2_candidates must be less than or equal to WARP_SIZE"

    comptime if skip_phase2:
        # When skipping phase 2, warp 0 reads directly from smem_phase1.
        # Requires phase1_candidates to fit within one warp.
        comptime assert (
            phase1_candidates <= WARP_SIZE
        ), "phase1_candidates exceeds WARP_SIZE, cannot skip phase 2"

    comptime total_smem = phase1_candidates if skip_phase2 else (
        phase1_candidates + phase2_candidates
    )

    var tid = Int(thread_idx.x)
    var warp_id = warp_id()
    var lane_id = lane_id()

    var shared_mem = unsafe_stack_allocation[
        total_smem,
        TopK_2[scores_type],
        address_space=.SHARED,
    ]()

    var shared_mem_phase1 = shared_mem
    var shared_mem_phase2 = shared_mem + phase1_candidates

    var val = TopK_2(u=biased_score, p=tid)
    var sorted_val = _warp_bitonic_sort[num_lanes=WARP_SIZE](val)

    if lane_id < n_experts_per_tok:
        shared_mem_phase1[warp_id * n_experts_per_tok + lane_id] = sorted_val

    barrier()

    comptime if not skip_phase2:
        var val2: TopK_2[scores_type]
        # Sequential read: warp W reads the contiguous WARP_SIZE-wide slice
        # of smem_phase1 starting at W*WARP_SIZE, so no bank conflicts. The
        # `tid < phase1_candidates` guard matters whenever phase1_candidates
        # is not a whole number of warps -- 160 routed experts with k=8
        # leaves 40 survivors across 2 phase-2 warps -- since without it the
        # trailing lanes would read past shared_mem_phase1 into
        # shared_mem_phase2's backing memory, which nothing has written yet.
        if warp_id < num_phase2_warps and tid < phase1_candidates:
            val2 = shared_mem_phase1[tid]
        else:
            # Inactive warps: dead-value cannot corrupt the sort because
            # _warp_bitonic_sort is fully intra-warp.
            val2 = TopK_2[scores_type]()

        var sorted_val2 = _warp_bitonic_sort[num_lanes=WARP_SIZE](val2)

        if warp_id < num_phase2_warps and lane_id < n_experts_per_tok:
            shared_mem_phase2[
                warp_id * n_experts_per_tok + lane_id
            ] = sorted_val2

        barrier()

    var winners = TopK_2[scores_type]()
    if warp_id == 0:
        var val3: TopK_2[scores_type]

        comptime if skip_phase2:
            # Wide-wavefront path: warp 0 reads the phase-1 survivors.
            if lane_id < phase1_candidates:
                val3 = shared_mem_phase1[lane_id]
            else:
                val3 = TopK_2[scores_type]()  # padding: -inf
        else:
            # Narrow-warp path: warp 0 reads the phase-2 survivors.
            if lane_id < phase2_candidates:
                val3 = shared_mem_phase2[lane_id]
            else:
                val3 = TopK_2[scores_type]()  # padding: -inf

        winners = _warp_bitonic_sort[num_lanes=WARP_SIZE](val3)

    return winners


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"single_group_router_{scores_type}_{bias_type}_t{num_threads}")
def single_group_router_kernel[
    scores_type: DType,
    bias_type: DType,
    ExpertIndicesLayoutType: TensorLayout,
    ExpertWeightsLayoutType: TensorLayout,
    ExpertScoresLayoutType: TensorLayout,
    ExpertBiasLayoutType: TensorLayout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    norm_weights: Bool,
    num_threads: Int,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[
        mut=True, .int32, ExpertIndicesLayoutType, MutAnyOrigin
    ],
    expert_weights: TileTensor[
        mut=True, scores_type, ExpertWeightsLayoutType, MutAnyOrigin
    ],
    expert_scores: TileTensor[
        scores_type, ExpertScoresLayoutType, ImmutAnyOrigin
    ],
    expert_bias: TileTensor[bias_type, ExpertBiasLayoutType, ImmutAnyOrigin],
    routed_scaling_factor: Float32,
):
    """Single-group MoE router kernel. One block per token, one thread per expert.

    Fuses: corrected = scores + bias → top-k selection (`_block_top_k`) →
    weight = corrected - bias → optional normalize → scale.
    """

    comptime assert expert_indices.flat_rank == 2
    comptime assert expert_weights.flat_rank == 2
    comptime assert expert_scores.flat_rank == 2
    comptime assert expert_bias.flat_rank == 1

    comptime assert (
        expert_scores.static_shape[1] == n_routed_experts
    ), "expert_scores.static_shape[1] must be equal to n_routed_experts"

    comptime assert (
        expert_weights.static_shape[1] == n_experts_per_tok
    ), "expert_weights.static_shape[1] must be equal to n_experts_per_tok"

    comptime assert (
        expert_indices.static_shape[1] == n_experts_per_tok
    ), "expert_indices.static_shape[1] must be equal to n_experts_per_tok"

    comptime assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    # The weight reduction below is a lane_group_sum over n_experts_per_tok
    # lanes, which must be a power of two.
    comptime assert (
        n_experts_per_tok.is_power_of_two()
    ), "n_experts_per_tok must be a power of two"

    var token_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var warp_id = warp_id()
    var lane_id = lane_id()

    with PDL():
        var thread_expert_bias = expert_bias.load[width=1](Coord(tid)).cast[
            scores_type
        ]()

        var thread_expert_score: Scalar[scores_type]
        comptime if scores_input_fn:
            comptime scores_fn = scores_input_fn.value()
            thread_expert_score = scores_fn[width=1]((token_idx, tid))
        else:
            thread_expert_score = expert_scores.load[width=1]((token_idx, tid))
        var biased_score = thread_expert_score + thread_expert_bias

        var sorted_val3 = _block_top_k[n_experts_per_tok, num_threads](
            biased_score
        )

        # WARP 0 ONLY gives top n_experts_per_tok
        if warp_id == 0:
            # get the original weights and normalize them
            var original_weight: Scalar[scores_type] = 0
            if lane_id < n_experts_per_tok:
                comptime if scores_input_fn:
                    comptime d_fn = scores_input_fn.value()
                    original_weight = d_fn[width=1]((token_idx, sorted_val3.p))
                else:
                    original_weight = expert_scores.load[width=1](
                        (token_idx, sorted_val3.p)
                    )

            var weights_sum = warp.lane_group_sum[num_lanes=n_experts_per_tok](
                original_weight
            )

            comptime if norm_weights:
                original_weight /= weights_sum

            original_weight *= Scalar[scores_type](routed_scaling_factor)

            # Write expert index and weight for this token.
            if lane_id < n_experts_per_tok:
                expert_indices.store((token_idx, lane_id), Int32(sorted_val3.p))
                expert_weights[token_idx, lane_id] = original_weight


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(
    t"single_group_router_eplb_{scores_type}_{bias_type}_t{num_threads}_n{num_log}_r{max_replicas}_h{Int(hash_decorrelate)}",
)
def single_group_router_eplb_kernel[
    scores_type: DType,
    bias_type: DType,
    ExpertIndicesLayoutType: TensorLayout,  # phy ids out
    ExpertIndicesLogLayoutType: TensorLayout,  # log ids out (for histogram)
    ExpertWeightsLayoutType: TensorLayout,
    ExpertScoresLayoutType: TensorLayout,
    ExpertBiasLayoutType: TensorLayout,
    LogcntLayoutType: TensorLayout,
    Log2phyLayoutType: TensorLayout,
    LayerIdxLayoutType: TensorLayout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    norm_weights: Bool,
    num_threads: Int,
    num_log: Int,
    max_replicas: Int,
    hash_decorrelate: Bool,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[
        mut=True, .int32, ExpertIndicesLayoutType, MutAnyOrigin
    ],  # phy ids
    expert_indices_log: TileTensor[
        mut=True, .int32, ExpertIndicesLogLayoutType, MutAnyOrigin
    ],  # log ids (for EPLB histogram)
    expert_weights: TileTensor[
        mut=True, scores_type, ExpertWeightsLayoutType, MutAnyOrigin
    ],
    expert_scores: TileTensor[
        scores_type, ExpertScoresLayoutType, ImmutAnyOrigin
    ],
    expert_bias: TileTensor[bias_type, ExpertBiasLayoutType, ImmutAnyOrigin],
    logcnt: TileTensor[.int32, LogcntLayoutType, ImmutAnyOrigin],
    log2phy: TileTensor[.int32, Log2phyLayoutType, ImmutAnyOrigin],
    layer_idx: TileTensor[.int32, LayerIdxLayoutType, ImmutAnyOrigin],
    routed_scaling_factor: Float32,
):
    """Single-group MoE router fused with EPLB log->phy remap.

    Selects with the same `_block_top_k` as `single_group_router_kernel`, then
    at the K writers performs the EPLB lookup using a per-block SMEM cache of
    the current layer's logcnt/log2phy slice.

    Backend specialization:
      - NVIDIA: cp.async issues the table fetch up front; sort hides the latency.
      - AMD/Apple: plain ld_global into registers up front, ds_write later.
    """

    comptime assert expert_indices.flat_rank == 2
    comptime assert expert_indices_log.flat_rank == 2
    comptime assert expert_weights.flat_rank == 2
    comptime assert expert_scores.flat_rank == 2
    comptime assert expert_bias.flat_rank == 1
    comptime assert logcnt.flat_rank == 2
    comptime assert log2phy.flat_rank == 3
    comptime assert layer_idx.flat_rank == 1

    comptime assert (
        expert_scores.static_shape[1] == n_routed_experts
    ), "expert_scores.static_shape[1] must be equal to n_routed_experts"
    comptime assert (
        expert_indices.static_shape[1] == n_experts_per_tok
    ), "expert_indices.static_shape[1] must be equal to n_experts_per_tok"
    comptime assert (
        expert_indices_log.static_shape[1] == n_experts_per_tok
    ), "expert_indices_log.static_shape[1] must be equal to n_experts_per_tok"
    comptime assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    # The weight reduction below is a lane_group_sum over n_experts_per_tok
    # lanes, which must be a power of two.
    comptime assert (
        n_experts_per_tok.is_power_of_two()
    ), "n_experts_per_tok must be a power of two"

    comptime assert (
        logcnt.static_shape[1] == num_log
    ), "logcnt.static_shape[1] must equal num_log"
    comptime assert (
        log2phy.static_shape[1] == num_log
    ), "log2phy.static_shape[1] must equal num_log"
    comptime assert (
        log2phy.static_shape[2] == max_replicas
    ), "log2phy.static_shape[2] must equal max_replicas"

    var token_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var w_id = warp_id()
    var l_id = lane_id()

    with PDL():
        var Lidx = Int(layer_idx.load[width=1](Coord(Idx[0]))[0])
        var thread_expert_bias = expert_bias.load[width=1](Coord(tid)).cast[
            scores_type
        ]()

        var thread_expert_score: Scalar[scores_type]
        comptime if scores_input_fn:
            comptime scores_fn = scores_input_fn.value()
            thread_expert_score = scores_fn[width=1]((token_idx, tid))
        else:
            thread_expert_score = expert_scores.load[width=1]((token_idx, tid))
        var biased_score = thread_expert_score + thread_expert_bias

        var sorted_val3 = _block_top_k[n_experts_per_tok, num_threads](
            biased_score
        )

        # ============================================================
        # REMAP + STORE (warp 0 only)
        # ============================================================
        if w_id == 0:
            comptime assert (
                max_replicas == 1
                or max_replicas == 2
                or max_replicas == 4
                or max_replicas == 8
                or max_replicas == 16
            ), "max_replicas must be a SIMD-loadable width (1,2,4,8,16)"

            # Hoisted out of the `if l_id < K` so non-writer lanes still
            # have valid registers for the reduction's uniform shuffles.
            var log: Int = 0
            var original_weight: Scalar[scores_type] = 0
            var cnt: Int = 1
            var phy_all = SIMD[.int32, max_replicas](0)

            if l_id < n_experts_per_tok:
                log = Int(sorted_val3.p)

                # Burst load #1 — original_weight (existing).
                comptime if scores_input_fn:
                    comptime d_fn = scores_input_fn.value()
                    original_weight = d_fn[width=1]((token_idx, log))
                else:
                    original_weight = expert_scores.load[width=1](
                        (token_idx, log)
                    )

                # Burst load #2 — full log2phy[Lidx, log, :] slice.
                # One 4*max_replicas-byte HBM transaction (16B for mr=4,
                # 32B for mr=8). Replica selection moves into registers.
                phy_all = log2phy.load[width=max_replicas]((Lidx, log, Idx[0]))

                # Burst load #3 — cnt (only when mr > 1).
                comptime if max_replicas > 1:
                    cnt = Int(logcnt.load[width=1]((Lidx, log))[0])

            # ---------- Weight reduction (loads above are in flight) ----
            var weights_sum = warp.lane_group_sum[num_lanes=n_experts_per_tok](
                original_weight
            )
            comptime if norm_weights:
                original_weight /= weights_sum
            original_weight *= Scalar[scores_type](routed_scaling_factor)

            # ---------- Replica pick + store (all register ops) ---------
            if l_id < n_experts_per_tok:
                var r = _pick_replica[
                    max_replicas, hash_decorrelate, n_experts_per_tok
                ](log, cnt, token_idx, Int(l_id))

                var phy: Int32
                comptime if max_replicas == 1:
                    phy = phy_all[0]
                else:
                    # Comptime-unrolled select chain. Keeps phy_all in
                    # registers — dynamic SIMD indexing can otherwise
                    # spill to local memory on some lowerings.
                    phy = phy_all[0]
                    comptime for ri in range(1, max_replicas):
                        if r == ri:
                            phy = phy_all[ri]

                expert_indices.store((token_idx, l_id), phy)
                expert_indices_log.store((token_idx, l_id), Int32(log))
                expert_weights[token_idx, l_id] = original_weight


@always_inline
def single_group_router[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    norm_weights: Bool,
    target: StaticString,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[mut=True, .int32, ...],
    expert_weight: TileTensor[mut=True, scores_type, ...],
    expert_scores: TileTensor[mut=False, scores_type, ...],
    expert_bias: TileTensor[mut=False, bias_type, ...],
    routed_scaling_factor: Float32,
    context: DeviceContext,
) raises:
    """Launch the single-group MoE router on GPU.

    One block per token, one thread per expert. Selects top n_experts_per_tok
    experts using warp-bitonic sort with 2 or 3 reduction phases depending on
    hardware warp size (AMD skips phase 2 at compile time).

    Parameters:
        scores_type: DType of routing scores and output weights.
        bias_type: DType of the expert correction bias.
        n_routed_experts: Total number of experts (e.g. 384 for Kimi K2.5).
        n_experts_per_tok: Experts selected per token, must be a power of 2
            (e.g. 8 for Kimi K2.5).
        norm_weights: If True, normalize selected weights to sum to 1 before
            applying routed_scaling_factor.
        target: The target device to run the kernel on.
        scores_input_fn: Optional fused input lambda to load scores. If None,
            scores are loaded directly from expert_scores.

    Inputs:
        expert_indices: Output expert indices. Shape: [num_tokens, n_experts_per_tok].
        expert_weights: Output expert weights. Shape: [num_tokens, n_experts_per_tok].
        expert_scores: Input routing scores. Shape: [num_tokens, n_routed_experts].
        expert_bias: Per-expert correction bias used for selection only.
        routed_scaling_factor: Scalar multiplied into every output weight.
        context: The device context.
    """
    comptime assert is_gpu[
        target
    ](), "Single group router is only supported on GPU"

    if expert_scores.dim(0) == 0:
        return

    var gpu_ctx = context

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.router_single_group", task_id=Int(gpu_ctx.id())
    ):
        # comptime num_tokens = Int(expert_scores.dim(0))
        comptime num_threads = n_routed_experts
        comptime hw_info = gpu_ctx.default_device_info
        comptime blocks_per_sm = hw_info.threads_per_multiprocessor // num_threads

        comptime num_sms = hw_info.sm_count

        comptime kernel = single_group_router_kernel[
            scores_type,
            bias_type,
            expert_indices.LayoutType,
            expert_weight.LayoutType,
            expert_scores.LayoutType,
            expert_bias.LayoutType,
            n_routed_experts,
            n_experts_per_tok,
            norm_weights,
            num_threads,
            scores_input_fn=scores_input_fn,
        ]

        # launch the kernle using gpu_ctx
        gpu_ctx.enqueue_function[kernel](
            expert_indices,
            expert_weight,
            expert_scores,
            expert_bias,
            routed_scaling_factor,
            grid_dim=expert_scores.dim(0),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(num_threads))
)
@__name(t"sink_gate_router_{scores_type}_{bias_type}_t{num_threads}")
def sink_gate_router_kernel[
    scores_type: DType,
    bias_type: DType,
    ExpertIndicesLayoutType: TensorLayout,
    ExpertWeightsLayoutType: TensorLayout,
    SinkWeightsLayoutType: TensorLayout,
    LogitsLayoutType: TensorLayout,
    ExpertBiasLayoutType: TensorLayout,
    GlobalScaleLayoutType: TensorLayout,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_shared_experts: Int,
    num_threads: Int,
](
    expert_indices: TileTensor[
        mut=True, .int32, ExpertIndicesLayoutType, MutAnyOrigin
    ],
    expert_weights: TileTensor[
        mut=True, scores_type, ExpertWeightsLayoutType, MutAnyOrigin
    ],
    sink_weights: TileTensor[
        mut=True, scores_type, SinkWeightsLayoutType, MutAnyOrigin
    ],
    logits: TileTensor[scores_type, LogitsLayoutType, ImmutAnyOrigin],
    expert_bias: TileTensor[bias_type, ExpertBiasLayoutType, ImmutAnyOrigin],
    global_scale: TileTensor[
        scores_type, GlobalScaleLayoutType, ImmutAnyOrigin
    ],
    route_scale: Float32,
):
    """Fused sigmoid-gate MoE router with always-on sink (shared-expert) lanes.

    Sink lanes are gated shared experts, not attention sinks.

    One block per token, one thread per routed expert. Fuses: sigmoid(logit) +
    bias -> top-k selection (`_block_top_k`) -> softmax over the log-sigmoid of
    the selected experts' raw (unbiased) logits concatenated with
    `n_shared_experts` always-selected sink logits -> scale by
    `route_scale * global_scale`.

    Softmax over log-sigmoids equals `sigmoid(z_i) / sum_j sigmoid(z_j)`,
    computed in log space so it stays finite where the sigmoids themselves
    would underflow.

    Expert bucketing stays in `moe_create_indices`: it needs every token's
    assignment before it can build the per-expert CSR, which this
    per-token-block kernel cannot provide without a grid-wide sync.

    Parameters:
        scores_type: DType of the logits and the output weights.
        bias_type: DType of the per-routed-expert selection bias.
        ExpertIndicesLayoutType: `TensorLayout` of the `expert_indices`
            output tensor.
        ExpertWeightsLayoutType: `TensorLayout` of the `expert_weights`
            output tensor.
        SinkWeightsLayoutType: `TensorLayout` of the `sink_weights` output
            tensor.
        LogitsLayoutType: `TensorLayout` of the `logits` input tensor.
        ExpertBiasLayoutType: `TensorLayout` of the `expert_bias` input
            tensor.
        GlobalScaleLayoutType: `TensorLayout` of the `global_scale` input
            tensor.
        n_routed_experts: Total number of routed experts scored per token.
            Also equals the thread count per block.
        n_experts_per_tok: Number of routed experts selected per token.
        n_shared_experts: Number of always-selected sink experts. Together
            with n_experts_per_tok, must sum to a power of two no greater
            than the warp size (the two are jointly softmax-normalized by a
            single warp-level reduction).
        num_threads: Threads per block; must equal n_routed_experts.

    Args:
        expert_indices: Output selected routed-expert index per token. Shape
            [num_tokens, n_experts_per_tok].
        expert_weights: Output routing weight per selected routed expert.
            Shape [num_tokens, n_experts_per_tok].
        sink_weights: Output routing weight per sink expert. Shape
            [num_tokens, n_shared_experts].
        logits: Input raw (pre-sigmoid) gate logits, routed experts followed
            by sink experts. Shape [num_tokens, n_routed_experts + n_shared_experts].
        expert_bias: Per-routed-expert bias added during selection only.
            Shape [n_routed_experts].
        global_scale: Single scalar multiplied into every weight. Shape [1].
        route_scale: Compile-time-known-per-model scalar multiplied into
            every weight alongside global_scale.
    """
    # is_floating_point is what proves exp/log1p below well-formed; the
    # float32 bound is narrower, and is all the joint softmax's reduce and
    # divide have been validated at.
    comptime assert (
        scores_type.is_floating_point()
    ), "scores_type must be floating point"
    comptime assert scores_type == .float32, "scores_type must be float32"
    comptime assert expert_indices.flat_rank == 2
    comptime assert expert_weights.flat_rank == 2
    comptime assert sink_weights.flat_rank == 2
    comptime assert logits.flat_rank == 2
    comptime assert expert_bias.flat_rank == 1
    comptime assert global_scale.flat_rank == 1

    comptime assert (
        logits.static_shape[1] == n_routed_experts + n_shared_experts
    ), "logits.static_shape[1] must be n_routed_experts + n_shared_experts"
    comptime assert (
        expert_weights.static_shape[1] == n_experts_per_tok
    ), "expert_weights.static_shape[1] must be equal to n_experts_per_tok"
    comptime assert (
        expert_indices.static_shape[1] == n_experts_per_tok
    ), "expert_indices.static_shape[1] must be equal to n_experts_per_tok"
    comptime assert (
        sink_weights.static_shape[1] == n_shared_experts
    ), "sink_weights.static_shape[1] must be equal to n_shared_experts"
    comptime assert (
        expert_bias.static_shape[0] == n_routed_experts
    ), "expert_bias.static_shape[0] must be equal to n_routed_experts"

    comptime assert (
        num_threads == n_routed_experts
    ), "num_threads must be equal to n_routed_experts"

    comptime k_total = n_experts_per_tok + n_shared_experts
    comptime assert k_total.is_power_of_two(), (
        "n_experts_per_tok + n_shared_experts must be a power of two (the joint"
        " softmax is one warp-level reduction)"
    )
    comptime assert (
        k_total <= WARP_SIZE
    ), "n_experts_per_tok + n_shared_experts must fit in one warp"

    var token_idx = Int(block_idx.x)
    var tid = Int(thread_idx.x)
    var warp_id = warp_id()
    var lane_id = lane_id()

    with PDL():
        var thread_bias = expert_bias.load[width=1](Coord(tid)).cast[
            scores_type
        ]()
        var thread_logit = logits.load[width=1]((token_idx, tid))
        var biased_score = sigmoid(thread_logit) + thread_bias

        var sorted_val3 = _block_top_k[n_experts_per_tok, num_threads](
            biased_score
        )

        # WARP 0 ONLY: compute the log-sigmoid softmax weights over the
        # selected experts plus the n_shared_experts always-on sink lanes.
        if warp_id == 0:
            # sorted_val3.u is the biased selection score; the softmax needs
            # the raw logit, so winner lanes reload it by the winning index
            # and sink lanes read their fixed columns. Lanes >= k_total sit
            # outside this reduction's warp segment and never get read.
            var raw_val: Scalar[scores_type] = 0
            if lane_id < n_experts_per_tok:
                raw_val = logits.load[width=1]((token_idx, sorted_val3.p))
            elif lane_id < k_total:
                var sink_idx = n_routed_experts + (
                    Int(lane_id) - n_experts_per_tok
                )
                raw_val = logits.load[width=1]((token_idx, sink_idx))

            # log_sigmoid(x) = min(x, 0) - log1p(exp(-abs(x))), stable where
            # sigmoid(x) itself would underflow.
            var zero = Scalar[scores_type](0)
            var log_score = min(raw_val, zero) - log1p(exp(-abs(raw_val)))

            var shift = warp.lane_group_max[num_lanes=k_total](log_score)
            var score = exp(log_score - shift)
            var sum_score = warp.lane_group_sum[num_lanes=k_total](score)

            var global_scale_val = global_scale.load[width=1](Coord(0)).cast[
                scores_type
            ]()
            var factor = (
                Scalar[scores_type](route_scale) * global_scale_val
            ) / sum_score
            var weight = score * factor

            if lane_id < n_experts_per_tok:
                expert_indices.store((token_idx, lane_id), Int32(sorted_val3.p))
                expert_weights[token_idx, lane_id] = weight
            elif lane_id < k_total:
                sink_weights[
                    token_idx, Int(lane_id) - n_experts_per_tok
                ] = weight


@always_inline
def sink_gate_router[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    n_shared_experts: Int,
    target: StaticString,
](
    expert_indices: TileTensor[mut=True, .int32, ...],
    expert_weights: TileTensor[mut=True, scores_type, ...],
    sink_weights: TileTensor[mut=True, scores_type, ...],
    logits: TileTensor[mut=False, scores_type, ...],
    expert_bias: TileTensor[mut=False, bias_type, ...],
    global_scale: TileTensor[mut=False, scores_type, ...],
    route_scale: Float32,
    context: DeviceContext,
) raises:
    """Launch the fused sink-gate MoE router on GPU.

    See `sink_gate_router_kernel` for the fused computation. One block per
    token, one thread per routed expert.

    Parameters:
        scores_type: DType of logits and output weights.
        bias_type: DType of the expert selection bias.
        n_routed_experts: Total number of routed experts (e.g. 256 for
            Inkling-Small).
        n_experts_per_tok: Routed experts selected per token (e.g. 6 for
            Inkling-Small).
        n_shared_experts: Always-selected sink experts (e.g. 2 for
            Inkling-Small).
        target: The target device to run the kernel on.

    Inputs:
        expert_indices: Output selected expert indices. Shape:
            [num_tokens, n_experts_per_tok].
        expert_weights: Output selected-expert weights. Shape:
            [num_tokens, n_experts_per_tok].
        sink_weights: Output sink-expert weights. Shape:
            [num_tokens, n_shared_experts].
        logits: Input raw gate logits (routed then sink columns). Shape:
            [num_tokens, n_routed_experts + n_shared_experts].
        expert_bias: Per-routed-expert selection bias.
        global_scale: Scalar output-scaling weight.
        route_scale: Scalar output-scaling factor.
        context: The device context.
    """
    comptime assert is_gpu[
        target
    ](), "sink_gate_router is only supported on GPU"

    if logits.dim(0) == 0:
        return

    var gpu_ctx = context

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.router_sink_gate", task_id=Int(gpu_ctx.id())
    ):
        comptime num_threads = n_routed_experts

        comptime kernel = sink_gate_router_kernel[
            scores_type,
            bias_type,
            expert_indices.LayoutType,
            expert_weights.LayoutType,
            sink_weights.LayoutType,
            logits.LayoutType,
            expert_bias.LayoutType,
            global_scale.LayoutType,
            n_routed_experts,
            n_experts_per_tok,
            n_shared_experts,
            num_threads,
        ]

        gpu_ctx.enqueue_function[kernel](
            expert_indices,
            expert_weights,
            sink_weights,
            logits,
            expert_bias,
            global_scale,
            route_scale,
            grid_dim=logits.dim(0),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )


# EPLB remap (log2hy id) kernel
@always_inline
def single_group_router_eplb[
    scores_type: DType,
    bias_type: DType,
    //,
    n_routed_experts: Int,
    n_experts_per_tok: Int,
    norm_weights: Bool,
    num_log: Int,
    max_replicas: Int,
    hash_decorrelate: Bool,
    target: StaticString,
    scores_input_fn: OptionalReg[
        def[width: Int](IndexList[2]) capturing -> SIMD[scores_type, width]
    ] = None,
](
    expert_indices: TileTensor[mut=True, .int32, ...],
    expert_indices_log: TileTensor[mut=True, .int32, ...],
    expert_weights: TileTensor[mut=True, scores_type, ...],
    expert_scores: TileTensor[scores_type, ...],
    expert_bias: TileTensor[bias_type, ...],
    logcnt: TileTensor[.int32, ...],
    log2phy: TileTensor[.int32, ...],
    layer_idx: TileTensor[.int32, ...],
    routed_scaling_factor: Float32,
    context: DeviceContext,
) raises:
    """Launches the single-group MoE router with EPLB log->phy remap on GPU.

    Selects the top n_experts_per_tok experts per token using warp-bitonic sort
    (2 or 3 phases depending on warp size), then remaps each selected logical
    expert ID to a physical expert ID via the per-layer logcnt and log2phy
    tables. One block is launched per token.

    Parameters:
        scores_type: DType of the routing scores and output weights.
        bias_type: DType of the expert correction bias.
        n_routed_experts: Total number of routed experts.
        n_experts_per_tok: Experts selected per token, must be a power of two.
        norm_weights: If True, normalize selected weights to sum to 1 before
            applying routed_scaling_factor.
        num_log: Number of logical experts per layer.
        max_replicas: Maximum number of physical replicas per logical expert.
        hash_decorrelate: If True, xor-hash the flat position with a Knuth
            multiplicative hash before the modulo to break structured-position
            bias in replica selection.
        target: The target device to run the kernel on.
        scores_input_fn: Optional fused input lambda to load scores. If None,
            scores are loaded directly from expert_scores.

    Inputs:
        expert_indices: Output physical expert IDs.
            Shape: [num_tokens, n_experts_per_tok].
        expert_indices_log: Output logical expert IDs for EPLB histogram.
            Shape: [num_tokens, n_experts_per_tok].
        expert_weights: Output expert weights.
            Shape: [num_tokens, n_experts_per_tok].
        expert_scores: Input routing scores.
            Shape: [num_tokens, n_routed_experts].
        expert_bias: Per-expert correction bias used for selection only.
        logcnt: Per-(layer, logical) replica count.
            Shape: [num_layers, num_log].
        log2phy: Per-(layer, logical, replica) physical-ID table.
            Shape: [num_layers, num_log, max_replicas].
        layer_idx: Rank-1 scalar tensor carrying the current MoE layer index.
        routed_scaling_factor: Scalar multiplied into every output weight.
        context: The device context.
    """
    comptime assert is_gpu[
        target
    ](), "Single group router (EPLB) is only supported on GPU"

    if expert_scores.dim(0) == 0:
        return

    var gpu_ctx = context

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.single.group.router.eplb", task_id=Int(gpu_ctx.id())
    ):
        comptime num_threads = n_routed_experts

        comptime kernel = single_group_router_eplb_kernel[
            scores_type,
            bias_type,
            expert_indices.LayoutType,
            expert_indices_log.LayoutType,
            expert_weights.LayoutType,
            expert_scores.LayoutType,
            expert_bias.LayoutType,
            logcnt.LayoutType,
            log2phy.LayoutType,
            layer_idx.LayoutType,
            n_routed_experts,
            n_experts_per_tok,
            norm_weights,
            num_threads,
            num_log,
            max_replicas,
            hash_decorrelate,
            scores_input_fn=scores_input_fn,
        ]

        gpu_ctx.enqueue_function[kernel](
            expert_indices,
            expert_indices_log,
            expert_weights,
            expert_scores,
            expert_bias,
            logcnt,
            log2phy,
            layer_idx,
            routed_scaling_factor,
            grid_dim=expert_scores.dim(0),
            block_dim=num_threads,
            attributes=pdl_launch_attributes(PDLLevel(1)),
        )


@always_inline
def _pick_replica[
    max_replicas: Int,
    hash_decorrelate: Bool,
    K: Int,
](log: Int, cnt: Int, n: Int, k: Int,) -> Int:
    """Deterministic replica picker. cnt is ignored when max_replicas == 1."""
    comptime if max_replicas == 1:
        return 0
    else:
        comptime HASH_C = UInt32(2654435761)  # Knuth golden ratio
        var pos: UInt32 = UInt32(n) * UInt32(K) + UInt32(k)
        comptime if hash_decorrelate:
            pos = pos ^ (UInt32(log) * HASH_C)
        return Int(pos % UInt32(cnt))


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(tile_tokens * K))
)
@__name(
    t"eplb_remap_kernel_n{num_log}_r{max_replicas}_k{K}_t{tile_tokens}_h{Int(hash_decorrelate)}",
)
def eplb_remap_kernel[
    PhyIdxLayoutType: TensorLayout,
    RouterIdxLayoutType: TensorLayout,
    LogcntLayoutType: TensorLayout,
    Log2phyLayoutType: TensorLayout,
    LayerIdxLayoutType: TensorLayout,
    num_log: Int,
    max_replicas: Int,
    K: Int,  # topK experts per token
    tile_tokens: Int,  # rows of router_idx per block; threads/block = tile_tokens * K
    hash_decorrelate: Bool,
](
    phy_idx: TileTensor[mut=True, .int32, PhyIdxLayoutType, MutAnyOrigin],
    router_idx: TileTensor[.int32, RouterIdxLayoutType, ImmutAnyOrigin],
    logcnt: TileTensor[.int32, LogcntLayoutType, ImmutAnyOrigin],
    log2phy: TileTensor[.int32, Log2phyLayoutType, ImmutAnyOrigin],
    layer_idx: TileTensor[.int32, LayerIdxLayoutType, ImmutAnyOrigin],
):
    """Fused EPLB per tile_token rows of router idx; one thread per (n,k) element.
    Each block cooperatively caces the current layer's logcnt and log2phy slices in
    SMEM, then every thread does: HBM-load logical id -> SMEM-looup cnt -> int mod -> SMEM-Lookup phy id
    -> HBM-store.

    Portable across all hardwares.

    Optimality of choosing : hash_decorrelate=True xor-hashes the flat position with a
    Knuth multiplicative hash of the logical id before the modulo, breaking
    structured position-vs-cnt alignment without warp ops.

    Parameters:
        PhyIdxLayoutType: `TensorLayout` of the `phy_idx` output tensor.
        RouterIdxLayoutType: `TensorLayout` of the `router_idx` input tensor.
        LogcntLayoutType: `TensorLayout` of the `logcnt` input tensor.
        Log2phyLayoutType: `TensorLayout` of the `log2phy` input tensor.
        LayerIdxLayoutType: `TensorLayout` of the `layer_idx` input tensor.
        num_log: Number of logical experts per MoE layer.
        max_replicas: Maximum number of physical replicas per logical expert.
        K: Number of top-K experts selected per token. Must be a power of
            two so `tid % K` is a bitmask.
        tile_tokens: Number of `router_idx` rows processed per block. Block
            size is `tile_tokens * K` threads.
        hash_decorrelate: If `True`, xor-hash the flat position with a Knuth
            multiplicative hash of the logical id before the modulo to break
            structured-position bias. If `False`, use plain `pos % cnt`.

    Args:
        phy_idx: Output `[num_tokens, K]` tensor of physical expert IDs after
            EPLB remap.
        router_idx: Input `[num_tokens, K]` tensor of logical expert IDs from
            the gate.
        logcnt: Input `[num_moe_layers, num_log]` tensor of replica counts per
            (layer, logical expert).
        log2phy: Input `[num_moe_layers, num_log, max_replicas]` tensor of
            physical-id lookup table entries.
        layer_idx: Input rank-1 `[1]` scalar tensor carrying the current MoE
            layer index.
    """

    comptime assert phy_idx.flat_rank == 2
    comptime assert router_idx.flat_rank == 2
    comptime assert logcnt.flat_rank == 2
    comptime assert log2phy.flat_rank == 3
    comptime assert layer_idx.flat_rank == 1

    comptime assert (
        router_idx.static_shape[1] == K
    ), "router_idx.static_shape[1] must equal K"
    comptime assert (
        phy_idx.static_shape[1] == K
    ), "phy_idx.static_shape[1] must equal K"
    comptime assert (
        logcnt.static_shape[1] == num_log
    ), "logcnt.static_shape[1] must equal num_log"
    comptime assert (
        log2phy.static_shape[1] == num_log
    ), "log2phy.static_shape[1] must equal num_log"
    comptime assert (
        log2phy.static_shape[2] == max_replicas
    ), "log2phy.static_shape[2] must equal max_replicas"
    comptime assert (
        K.is_power_of_two()
    ), "K must be a power of two so (tid % K) is a bitmask"

    comptime BLOCK_THREADS = tile_tokens * K
    comptime HASH_C = UInt32(
        2654435761
    )  # Knuth golden-ratio multiplicative hash

    var tid = Int(thread_idx.x)

    var smem_cnt = unsafe_stack_allocation[
        num_log,
        DType.int32,
        address_space=.SHARED,
    ]()

    var smem_phy = unsafe_stack_allocation[
        num_log * max_replicas,
        DType.int32,
        address_space=.SHARED,
    ]()

    with PDL():
        # Broadcast scalar layer index. Every thread reads the same address →
        # one HBM transaction, hot in L1 for the rest of the block.
        var Lidx = Int(layer_idx.load[width=1](Coord(Idx[0]))[0])

        # Cooperative SMEM load of (logcnt, log2phy) slice for layer Lidx.
        # BLOCK_THREADS threads cover num_log entries; unrolled at comptime.
        comptime for off in range(ceildiv(num_log, BLOCK_THREADS)):
            var i = tid + off * BLOCK_THREADS
            if i < num_log:
                # logcnt only matters for the round-robin path.
                comptime if max_replicas > 1:
                    smem_cnt[i] = Int32(logcnt.load[width=1]((Lidx, i))[0])
                comptime for r in range(max_replicas):
                    smem_phy[i * max_replicas + r] = Int32(
                        log2phy.load[width=1]((Lidx, i, r))[0]
                    )
        barrier()

        # Per element remap, One thread = one (n,k)
        var token_in_block = tid // K
        var k = tid % K
        var n = Int(block_idx.x) * tile_tokens + token_in_block
        var N = Int(phy_idx.dim(0))

        if n < N:
            var log = Int(router_idx.load[width=1]((n, k))[0])

            comptime if max_replicas == 1:
                # Pure permutation: cnt is always 1, r is always 0.
                # No cnt lookup, no modulo, no hash.
                var phy = Int32(smem_phy[log])
                phy_idx.store((n, k), phy)
            else:
                # Permutation + round-robin replica picker.
                var cnt = Int(smem_cnt[log])
                var pos: UInt32 = UInt32(n) * UInt32(K) + UInt32(k)

                comptime if hash_decorrelate:
                    # XOR-hash to break structured-position bias against `cnt`.
                    pos = pos ^ (UInt32(log) * HASH_C)

                var r = Int(pos % UInt32(cnt))
                var phy = Int32(smem_phy[log * max_replicas + r])
                phy_idx.store((n, k), phy)


@always_inline
def eplb_remap[
    num_log: Int,
    max_replicas: Int,
    K: Int,
    hash_decorrelate: Bool,
    target: StaticString,
](
    phy_idx: TileTensor[mut=True, .int32, ...],  # [N, K] output
    router_idx: TileTensor[.int32, ...],  # [N, K] logical ids
    logcnt: TileTensor[.int32, ...],  # [L, num_log]
    log2phy: TileTensor[.int32, ...],  # [L, num_log, max_replicas]
    layer_idx: TileTensor[.int32, ...],  # rank-1 [1] scalar
    context: DeviceContext,
) raises:
    """Launch the fused EPLB log->phy remap on GPU.

    One block per tile_tokens rows of router_idx; one thread per
    (n, k) element.

    Parameters:
        num_log: Number of logical experts per layer.
        max_replicas: Maximum physical replicas per logical expert.
        K: Top-K experts per token. Must be a power of two.
        hash_decorrelate: If True, xor-hash the position before the modulo
            to break structured-position bias in replica selection. If False,
            preserves the exact pos % cnt semantics of the legacy chain.
        target: The target device to run the kernel on.

    Args:
        phy_idx: Output physical expert ids. Shape: [num_tokens, K].
        router_idx: Input logical expert ids from the gate.
            Shape: [num_tokens, K].
        logcnt: Per-(layer, logical) replica count.
            Shape: [num_moe_layers, num_log].
        log2phy: Per-(layer, logical, replica) physical-id table.
            Shape: [num_moe_layers, num_log, max_replicas].
        layer_idx: Rank-1 scalar tensor of shape [1] carrying the current
            MoE layer index. Sits on the same device as router_idx.
        context: DeviceContext.
    """
    comptime assert is_gpu[
        target
    ](), "EPLB remap kernel is only supported on GPU"

    if router_idx.dim(0) == 0:
        return

    var gpu_ctx = context

    with Trace[TraceLevel.OP, target=target](
        "mo.moe.eplb.remap", task_id=Int(gpu_ctx.id())
    ):
        # Target ~128 threads/block. Divides cleanly into NVIDIA warp=32,
        # AMD wave=64, and Apple SIMD=32 so no lanes idle from divisibility.
        # tile_tokens scales with K so block_dim stays ≈128 regardless of model.
        comptime tile_tokens = 128 // K if K <= 128 else 1

        comptime kernel = eplb_remap_kernel[
            phy_idx.LayoutType,
            router_idx.LayoutType,
            logcnt.LayoutType,
            log2phy.LayoutType,
            layer_idx.LayoutType,
            num_log,
            max_replicas,
            K,
            tile_tokens,
            hash_decorrelate,
        ]

        gpu_ctx.enqueue_function[kernel](
            phy_idx,
            router_idx,
            logcnt,
            log2phy,
            layer_idx,
            grid_dim=ceildiv(Int(router_idx.dim(0)), tile_tokens),
            block_dim=tile_tokens * K,
            attributes=pdl_launch_attributes(PDLLevel(1)),
        )
