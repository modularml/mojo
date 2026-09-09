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

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.grouped_matmul import grouped_matmul_vendor, naive_grouped_matmul
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.index import Index


def test_vendor[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
        "Testing vendor grouped matmul:",
        num_active_experts,
        "active of",
        num_experts,
        "experts of shape",
        expert_shape,
    )
    print("tokens:", end="")
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print("expert ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    # Total and max number of tokens
    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # Define shapes
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N

    # Host allocations
    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var c_ref_host_ptr = alloc[Scalar[c_type]](c_size)
    var a_offsets_host_ptr = alloc[UInt32](num_active_experts + 1)
    var expert_ids_host_ptr = alloc[Int32](num_active_experts)

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_num_tokens, Idx[K])),
    )
    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
    )
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(total_num_tokens, Idx[N])),
    )
    var c_ref_host = TileTensor(
        c_ref_host_ptr,
        row_major(Coord(total_num_tokens, Idx[N])),
    )

    # Create host TileTensors for offsets and expert_ids
    var a_offsets_host = TileTensor(
        a_offsets_host_ptr.as_unsafe_any_origin(),
        row_major(Coord(num_active_experts + 1)),
    )
    var expert_ids_host = TileTensor(
        expert_ids_host_ptr.as_unsafe_any_origin(),
        row_major(Coord(num_active_experts)),
    )

    # Setup offsets and expert ids
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    # Initialize matmul inputs
    random(a_host)
    random(b_host)

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )

    var a_dev = TileTensor(
        a_dev_buffer,
        row_major(Coord(total_num_tokens, K)),
    )
    var b_dev = TileTensor(
        b_dev_buffer,
        row_major[num_experts, N, K](),
    )
    var c_dev = TileTensor(
        c_dev_buffer,
        row_major(Coord(total_num_tokens, N)),
    )
    var c_ref_dev = TileTensor(
        c_ref_dev_buffer,
        row_major(Coord(total_num_tokens, N)),
    )
    var a_offsets_dev = TileTensor(
        a_offsets_dev_buffer,
        row_major(Coord(num_active_experts + 1)),
    )
    var expert_ids_dev = TileTensor(
        expert_ids_dev_buffer,
        row_major(Coord(num_active_experts)),
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)

    # Run reference implementation
    naive_grouped_matmul(
        c_ref_dev,
        a_dev,
        b_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    # Run vendor implementation
    grouped_matmul_vendor(
        c_dev,
        a_dev,
        b_dev,
        a_offsets_host,
        expert_ids_host,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    # Copy results back to host
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev_buffer)
    ctx.enqueue_copy(c_host_ptr, c_dev_buffer)
    ctx.synchronize()

    # Verify results
    var rtol = 1e-2
    for m in range(total_num_tokens):
        for n in range(N):
            var expect = c_ref_host[m, n][0]
            var actual = c_host[m, n][0]
            assert_almost_equal(
                actual, expect, msg=String(t"m: {m} n: {n}"), rtol=rtol
            )

    print("✓ Vendor grouped matmul test passed")

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    c_ref_host_ptr.free()
    a_offsets_host_ptr.free()
    expert_ids_host_ptr.free()
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = c_ref_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def test_negative_lora_id_vendor[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
        "Testing vendor negative lora_id behavior:",
        num_active_experts,
        "active of",
        num_experts,
        "experts of shape",
        expert_shape,
    )
    print("tokens:", end="")
    for i in range(len(num_tokens_by_expert)):
        print(num_tokens_by_expert[i], end=" ")
    print("expert ids:", end="")
    for i in range(len(expert_ids)):
        print(expert_ids[i], end=" ")
    print()

    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    # Total and max number of tokens
    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    # Define shapes
    var a_size = total_num_tokens * K
    var b_size = num_experts * N * K
    var c_size = total_num_tokens * N

    # Host allocations
    var a_host_ptr = alloc[Scalar[a_type]](a_size)
    var b_host_ptr = alloc[Scalar[b_type]](b_size)
    var c_host_ptr = alloc[Scalar[c_type]](c_size)
    var a_offsets_host_ptr = alloc[UInt32](num_active_experts + 1)
    var expert_ids_host_ptr = alloc[Int32](num_active_experts)

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_num_tokens, Idx[K])),
    )
    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
    )
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(total_num_tokens, Idx[N])),
    )

    # Create host TileTensors for offsets and expert_ids
    var a_offsets_host = TileTensor(
        a_offsets_host_ptr.as_unsafe_any_origin(),
        row_major(Coord(num_active_experts + 1)),
    )
    var expert_ids_host = TileTensor(
        expert_ids_host_ptr.as_unsafe_any_origin(),
        row_major(Coord(num_active_experts)),
    )

    # Setup offsets and expert ids
    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    # Initialize matmul inputs with non-zero values
    random(a_host)
    random(b_host)

    # Device allocations
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )

    var a_dev = TileTensor(
        a_dev_buffer,
        row_major(Coord(total_num_tokens, K)),
    )
    var b_dev = TileTensor(
        b_dev_buffer,
        row_major[num_experts, N, K](),
    )
    var c_dev = TileTensor(
        c_dev_buffer,
        row_major(Coord(total_num_tokens, N)),
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)

    # Run vendor grouped matmul
    grouped_matmul_vendor(
        c_dev,
        a_dev,
        b_dev,
        a_offsets_host,
        expert_ids_host,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    # Copy result back to host
    ctx.enqueue_copy(c_host_ptr, c_dev_buffer)
    ctx.synchronize()

    # Verify results
    var current_offset = 0

    for expert_idx in range(num_active_experts):
        var expert_id = expert_ids[expert_idx]
        var num_tokens = num_tokens_by_expert[expert_idx]
        var should_be_zero = expert_id < 0

        print(
            "Expert",
            expert_idx,
            "has id",
            expert_id,
            "with",
            num_tokens,
            "tokens",
        )

        # Check each token for this expert
        for token_idx in range(num_tokens):
            var global_token_idx = current_offset + token_idx
            var has_non_zero = False

            # Check if any output value is non-zero for this token
            for n in range(N):
                var value = c_host[global_token_idx, n][0]
                if value != 0:
                    has_non_zero = True
                    break

            if should_be_zero and has_non_zero:
                print(
                    "ERROR: Found non-zero value for expert_id -1 at token",
                    global_token_idx,
                )
                print("Values:", end="")
                for n in range(min(5, N)):  # Print first 5 values
                    print(c_host[global_token_idx, n][0], end=" ")
                print()
                raise Error("Expected zero values for expert_id -1")
            elif not should_be_zero and not has_non_zero:
                print(
                    "WARNING: All values are zero for valid expert_id",
                    expert_id,
                    "at token",
                    global_token_idx,
                )

        current_offset += num_tokens

    print(
        "✓ Vendor negative lora_id test passed - expert_id -1 produces zero"
        " outputs"
    )

    # Cleanup
    a_host_ptr.free()
    b_host_ptr.free()
    c_host_ptr.free()
    a_offsets_host_ptr.free()
    expert_ids_host_ptr.free()
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def main() raises:
    with DeviceContext() as ctx:
        # Single matmul
        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
        ](1, [128], [0], ctx)

        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(512, 1024),
        ](1, [256], [0], ctx)

        # Multiple matmuls selecting part of experts
        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(768, 1024),
        ](2, [128, 256], [0, 2], ctx)

        # Multiple matmuls selecting part of experts
        # num_tokens not multiple of tile size
        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(1280, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Multiple matmuls selecting part of experts
        # num_tokens not multiple of tile size
        # expert N dimension not multiple of 256
        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(192, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Test that expert id of -1 results in 0s in the output
        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=2,
            expert_shape=Index(256, 512),
        ](2, [64, 128], [0, -1], ctx)

        # Test negative lora_id behavior with vendor matmul
        test_negative_lora_id_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=2,
            expert_shape=Index(256, 512),
        ](2, [64, 128], [0, -1], ctx)

        # Additional test cases for different data types
        test_vendor[
            .float32,
            .float32,
            num_experts=3,
            expert_shape=Index(384, 768),
        ](2, [100, 200], [1, 2], ctx)

        # Test with mixed valid and invalid expert ids
        test_vendor[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(512, 512),
        ](3, [50, 100, 75], [0, -1, 2], ctx)

        print("\n✅ All vendor grouped matmul tests passed!")
