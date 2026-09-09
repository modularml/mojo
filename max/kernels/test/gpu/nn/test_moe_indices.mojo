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


from max.gpu.host import DeviceContext, HostBuffer
from std.collections import Set
from std.math import align_up
from layout import Idx, TileTensor, row_major
from layout._fillers import random
from nn.moe import moe_create_indices
from std.testing import assert_equal, assert_true


def get_expert_dictionary(
    topk_ids: HostBuffer[.uint32], num_tokens: Int
) -> Dict[UInt32, UInt32]:
    var expert_dictionary = Dict[UInt32, UInt32]()

    for i in range(num_tokens):
        var expert_id = topk_ids[i]
        var current_value = expert_dictionary.get(expert_id, 0)
        current_value += 1
        expert_dictionary[expert_id] = current_value

    return expert_dictionary^


def check_token_expert_order(
    token_expert_order: HostBuffer[.uint32],
    topk_ids: HostBuffer[.uint32],
    num_tokens: Int,
) raises:
    """
    This function asserts all tokens of the same expert are together in the token_expert_order.
    """

    var expert_dictionary = get_expert_dictionary(topk_ids, num_tokens)
    var current_expert_id = topk_ids[Int(token_expert_order[0])]
    var token_count = expert_dictionary.get(current_expert_id, 0) - 1

    for i in range(1, num_tokens):
        var expert_id = topk_ids[Int(token_expert_order[i])]

        if expert_id != current_expert_id:
            assert_equal(token_count, 0, "tokens are grouped incorrectly")
            expert_dictionary[current_expert_id] = 0
            current_expert_id = expert_id

            token_count = expert_dictionary.get(current_expert_id, 0) - 1
        else:
            token_count -= 1

    assert_equal(token_count, 0, "tokens are grouped incorrectly")
    expert_dictionary[current_expert_id] = 0

    for k_v in expert_dictionary.take_items():
        assert_equal(k_v.value, 0, "tokens are grouped incorrectly")


def check_expert_stats(
    expert_usage_stats: HostBuffer[.uint32],
    topk_ids: HostBuffer[.uint32],
    num_tokens: Int,
    num_experts: Int,
) raises:
    """
    Checks if the most frequent expert is accurate, and if the number of experts is correct.
    """

    var expert_dictionary = get_expert_dictionary(topk_ids, num_tokens)

    var mx_value: UInt32 = 0

    for k_v in expert_dictionary.take_items():
        if k_v.value > mx_value:
            mx_value = k_v.value

    assert_equal(
        expert_usage_stats[0], mx_value, "most frequent expert is incorrect"
    )
    assert_equal(
        expert_usage_stats[1],
        UInt32(num_experts),
        "expert count is incorrect",
    )


def check_expert_indices(
    expert_start_indices: HostBuffer[.uint32],
    expert_ids: HostBuffer[.int32],
    token_expert_order: HostBuffer[.uint32],
    expert_usage_stats: HostBuffer[.uint32],
    topk_ids: HostBuffer[.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if the provided start indices are correct.
    """

    for i in range(expert_usage_stats[1]):
        var start_idx = expert_start_indices[Int(i)]
        var end_idx = expert_start_indices[Int(i + 1)]
        var expert_id = expert_ids[Int(i)]

        for j in range(start_idx, end_idx):
            var current_expert_id = topk_ids[Int(token_expert_order[Int(j)])]
            assert_equal(
                expert_id,
                Int32(current_expert_id),
                "expert range in start indices is incorrect",
            )


def check_total_token_count(
    expert_start_indices: HostBuffer[.uint32],
    num_experts: Int,
    num_tokens: Int,
) raises:
    """
    Checks that the final CSR boundary accounts for every token.
    """

    assert_equal(
        Int(expert_start_indices[num_experts]),
        num_tokens,
        "expert_start_indices[num_experts] does not cover every token",
    )


def check_expert_ids_permutation(
    expert_ids: HostBuffer[.int32],
    num_experts: Int,
) raises:
    """
    Checks that expert_ids holds every expert in [0, num_experts) exactly
    once, regardless of which order a kernel assigns them to slots.
    """

    var seen = Set[Int]()
    for i in range(num_experts):
        var expert_id = Int(expert_ids[i])
        assert_true(
            expert_id >= 0 and expert_id < num_experts,
            "expert_ids["
            + String(i)
            + "] is out of range: "
            + String(expert_id),
        )
        assert_true(
            not (expert_id in seen),
            "expert id appears more than once: " + String(expert_id),
        )
        seen.add(expert_id)


def check_restore_token_order(
    restore_token_order: HostBuffer[.uint32],
    token_expert_order: HostBuffer[.uint32],
    num_tokens: Int,
) raises:
    """
    Checks if original export order can be restored.
    """

    for i in range(num_tokens):
        assert_equal(
            i,
            Int(token_expert_order[Int(restore_token_order[i])]),
            "restore token order is incorrect",
        )


def check_scales_offset[
    scale_alignment: Int = 128,
](
    scales_offset: HostBuffer[.uint32],
    expert_start_indices: HostBuffer[.uint32],
    expert_usage_stats: HostBuffer[.uint32],
) raises:
    """Validates scales_offset values against expert_start_indices.

    For each expert i (in ascending expert id), scales_offset[i] should
    equal the difference between the cumulative aligned block count and the
    cumulative actual block count up to that expert.
    """
    var num_experts = Int(expert_usage_stats[1])
    var cumulative_actual: UInt32 = 0
    var cumulative_aligned: UInt32 = 0

    for i in range(num_experts):
        var expected = cumulative_aligned // UInt32(
            scale_alignment
        ) - cumulative_actual // UInt32(scale_alignment)
        assert_equal(
            scales_offset[i],
            expected,
            "scales_offset mismatch at expert index " + String(i),
        )
        var token_count = expert_start_indices[i + 1] - expert_start_indices[i]
        cumulative_actual += token_count
        cumulative_aligned += align_up(token_count, UInt32(scale_alignment))


def fill_skewed(top_k_buffer_host: HostBuffer[.uint32], num_tokens: Int):
    """
    Routes every token to expert 0, the extreme case of a skewed expert
    distribution the uniform random fill never produces.
    """

    for i in range(num_tokens):
        top_k_buffer_host[i] = 0


def test_moe_create_indices[
    num_experts: Int = 256,
    test_scales_offset: Bool = False,
    skewed: Bool = False,
](token_expert_order_length: Int, ctx: DeviceContext) raises:
    var token_expert_order_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](token_expert_order_length)
    var top_k_buffer_host = ctx.enqueue_create_host_buffer[.uint32](
        token_expert_order_length
    )
    var restore_token_order_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](token_expert_order_length)
    var expert_usage_stats_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](2)
    var expert_start_indices_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](num_experts + 1)
    var expert_ids_buffer_host = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )

    var token_expert_order_buffer_device = ctx.enqueue_create_buffer[
        DType.uint32
    ](token_expert_order_length)
    var expert_start_indices_buffer = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var restore_token_order_buffer = ctx.enqueue_create_buffer[.uint32](
        token_expert_order_length
    )
    var expert_ids_buffer = ctx.enqueue_create_buffer[.int32](num_experts)
    var expert_usage_stats_buffer = ctx.enqueue_create_buffer[.uint32](2)
    var top_k_buffer_device = ctx.enqueue_create_buffer[.uint32](
        token_expert_order_length
    )

    var token_expert_order = TileTensor(
        token_expert_order_buffer_device,
        row_major(token_expert_order_length),
    )

    var expert_start_indices = TileTensor(
        expert_start_indices_buffer,
        row_major(num_experts + 1),
    )

    var restore_token_order = TileTensor(
        restore_token_order_buffer,
        row_major(token_expert_order_length),
    )

    var expert_ids = TileTensor(
        expert_ids_buffer,
        row_major(num_experts),
    )

    var expert_usage_stats = TileTensor(
        expert_usage_stats_buffer,
        row_major(Idx[2]),
    )

    var top_k = TileTensor(
        top_k_buffer_device,
        row_major(token_expert_order_length),
    )

    var top_k_host = TileTensor(
        top_k_buffer_host,
        row_major(token_expert_order_length),
    )

    ctx.synchronize()

    # Fill top_k_host with random expert IDs, or route every token to one
    # expert to exercise a skewed (non-uniform) distribution.
    comptime if skewed:
        fill_skewed(top_k_buffer_host, token_expert_order_length)
    else:
        random(top_k_host, min=0, max=UInt32(num_experts))
    ctx.enqueue_copy(top_k_buffer_device, top_k_buffer_host)

    var scales_offset_buffer_host = ctx.enqueue_create_host_buffer[
        DType.uint32
    ](num_experts)
    var scales_offset_buffer = ctx.enqueue_create_buffer[.uint32](num_experts)

    comptime if test_scales_offset:
        moe_create_indices["gpu"](
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            top_k,
            ctx,
            scales_offset_p=scales_offset_buffer.unsafe_ptr().as_unsafe_any_origin(),
        )
    else:
        moe_create_indices["gpu"](
            token_expert_order,
            expert_start_indices,
            restore_token_order,
            expert_ids,
            expert_usage_stats,
            top_k,
            ctx,
        )

    ctx.enqueue_copy(
        token_expert_order_buffer_host, token_expert_order_buffer_device
    )
    ctx.enqueue_copy(
        restore_token_order_buffer_host, restore_token_order_buffer
    )
    ctx.enqueue_copy(expert_usage_stats_buffer_host, expert_usage_stats_buffer)
    ctx.enqueue_copy(expert_ids_buffer_host, expert_ids_buffer)
    ctx.enqueue_copy(
        expert_start_indices_buffer_host, expert_start_indices_buffer
    )
    ctx.enqueue_copy(scales_offset_buffer_host, scales_offset_buffer)
    ctx.synchronize()

    comptime if test_scales_offset:
        check_scales_offset(
            scales_offset_buffer_host,
            expert_start_indices_buffer_host,
            expert_usage_stats_buffer_host,
        )

    check_token_expert_order(
        token_expert_order_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )
    check_expert_stats(
        expert_usage_stats_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
        num_experts,
    )
    check_expert_indices(
        expert_start_indices_buffer_host,
        expert_ids_buffer_host,
        token_expert_order_buffer_host,
        expert_usage_stats_buffer_host,
        top_k_buffer_host,
        token_expert_order_length,
    )
    check_total_token_count(
        expert_start_indices_buffer_host,
        num_experts,
        token_expert_order_length,
    )
    check_expert_ids_permutation(expert_ids_buffer_host, num_experts)

    check_restore_token_order(
        restore_token_order_buffer_host,
        token_expert_order_buffer_host,
        token_expert_order_length,
    )


def main() raises:
    with DeviceContext() as ctx:
        test_moe_create_indices(197, ctx)
        test_moe_create_indices(2500, ctx)
        test_moe_create_indices(11, ctx)
        test_moe_create_indices(1, ctx)
        test_moe_create_indices(20660, ctx)
        test_moe_create_indices(100_000, ctx)

        test_moe_create_indices[test_scales_offset=True](197, ctx)
        test_moe_create_indices[test_scales_offset=True](2500, ctx)
        test_moe_create_indices[test_scales_offset=True](11, ctx)
        test_moe_create_indices[test_scales_offset=True](1, ctx)
        test_moe_create_indices[test_scales_offset=True](20660, ctx)
        test_moe_create_indices[test_scales_offset=True](100_000, ctx)

        # The kernel walks the experts in chunks of one block width (512),
        # carrying the running offsets across chunks: one exact chunk, a
        # partial second chunk, and a third chunk holding a single expert.
        test_moe_create_indices[num_experts=512](100, ctx)
        test_moe_create_indices[num_experts=513](2500, ctx)
        test_moe_create_indices[num_experts=1025](2500, ctx)
        test_moe_create_indices[num_experts=513, test_scales_offset=True](
            2500, ctx
        )

        # Skewed distribution: every token on expert 0, so one group holds
        # them all and the rest are empty.
        test_moe_create_indices[skewed=True](1024, ctx)
        test_moe_create_indices[num_experts=513, skewed=True](100_000, ctx)
