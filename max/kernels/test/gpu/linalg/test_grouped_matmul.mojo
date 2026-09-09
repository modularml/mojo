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

from max.gpu.host import DeviceContext
from max.gpu.host.info import B200, H100, _is_sm10x_gpu
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from linalg.utils import elementwise_epilogue_type
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.index import Index
import std.itertools


@always_inline
def test_epilogue[
    dtype: DType
](m: Int, n: Int, val: Scalar[dtype]) -> Scalar[dtype]:
    return val + 4 * (Scalar[dtype]((m + n) % 21 - 10))


def test[
    in_type: DType,
    out_type: DType,
    num_experts: Int,
    expert_shape: IndexList[2],
    has_epilogue: Bool = False,
    qkv_perm_dim: Bool = False,
](
    num_active_experts: Int,
    num_tokens_by_expert: List[Int],
    expert_ids: List[Int],
    ctx: DeviceContext,
) raises:
    print(
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

    # Create host A C buffers
    var a_size = total_num_tokens * K

    comptime actual_N = 3 * N if qkv_perm_dim else N
    var c_size = total_num_tokens * actual_N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_experts + 1
    )
    for i in range(num_experts + 1):
        a_offsets_host_ptr[i] = 0

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_num_tokens, Idx[K])),
    )
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(total_num_tokens, Idx[actual_N])),
    )
    var c_ref_host = TileTensor(
        c_ref_host_ptr,
        row_major(Coord(total_num_tokens, Idx[actual_N])),
    )

    # Create host B buffers
    var b_size = num_experts * (3 * N if qkv_perm_dim else N) * K
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )
    for i in range(num_experts):
        expert_ids_host_ptr[i] = 0

    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, 3 * N if qkv_perm_dim else N, K](),
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

    # Create device buffers
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var c_ref_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[.int32](num_experts)

    var a_dev = TileTensor[a_type](
        a_dev_buffer,
        row_major(Coord(total_num_tokens, Idx[K])),
    )
    var c_dev = TileTensor[c_type](
        c_dev_buffer,
        row_major(Coord(total_num_tokens, Idx[actual_N])),
    )
    var c_ref_dev = TileTensor[c_type](
        c_ref_dev_buffer,
        row_major(Coord(total_num_tokens, Idx[actual_N])),
    )
    var b_dev = TileTensor[b_type](
        b_dev_buffer,
        row_major[num_experts, 3 * N if qkv_perm_dim else N, K](),
    )
    var a_offsets_dev = TileTensor[.uint32](
        a_offsets_dev_buffer,
        row_major(Coord(num_experts + 1)),
    )
    var expert_ids_dev = TileTensor[.int32](
        expert_ids_dev_buffer,
        row_major(Coord(Idx[num_experts])),
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)
    ctx.synchronize()

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
    ctx.synchronize()

    var c_dev_tile = c_dev

    comptime assert not (
        qkv_perm_dim and has_epilogue
    ), "qkv_perm_dim and has_epilogue cannot be True at the same time"

    @always_inline
    @__copy_capture(c_dev_tile)
    @__parameter
    def epilogue_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var new_val = val

        comptime for i in range(width):
            new_val[i] = test_epilogue(idx[0], idx[1] + i, val[i])

        var ptr = (
            c_dev_tile._storage.bitcast[Scalar[out_type]]()
            + idx[0] * N
            + idx[1]
        )

        ptr.store[width=width, alignment=alignment](new_val.cast[out_type]())

    @always_inline
    @__copy_capture(c_dev_tile, total_num_tokens)
    @__parameter
    def perm_dim_fn[
        dtype: DType, width: SIMDLength, *, alignment: Int = 1
    ](idx: IndexList[2], val: SIMD[dtype, width]) -> None:
        var new_val = val
        var i = idx[0]
        var j = idx[1]
        var new_j, new_k = divmod(j, N)
        comptime assert N % width == 0, "N must be divisible by width"
        # The current index is [i, new_j, new_k] in the M x 3 x N row major
        # tensor.
        # The permdim tensor has the shape 3 x M x N, so the index is then
        # [new_j, i, new_k].
        var ptr = (
            c_dev_tile._storage.bitcast[Scalar[out_type]]()
            + new_j * total_num_tokens * N
            + i * N
            + new_k
        )
        ptr.store[width=width, alignment=alignment](new_val.cast[out_type]())

    comptime elementwise_lambda_fn = Optional[elementwise_epilogue_type](
        perm_dim_fn
    ) if qkv_perm_dim else (
        Optional[elementwise_epilogue_type](
            epilogue_fn
        ) if has_epilogue else None
    )
    grouped_matmul[elementwise_lambda_fn=elementwise_lambda_fn](
        c_dev,
        a_dev,
        b_dev,
        a_offsets_dev,
        expert_ids_dev,
        max_num_tokens_by_expert,
        num_active_experts,
        ctx,
    )

    ctx.synchronize()
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev_buffer)
    ctx.enqueue_copy(c_host_ptr, c_dev_buffer)
    ctx.synchronize()

    var rtol = 1e-2

    comptime if qkv_perm_dim:
        for qkv_idx, m, n in std.itertools.product(
            range(3), range(total_num_tokens), range(N)
        ):
            var expect = c_ref_host[m, qkv_idx * N + n][0]

            var actual = c_host_ptr[qkv_idx * total_num_tokens * N + m * N + n]
            assert_almost_equal(
                actual,
                expect,
                msg=String(
                    t"qkv_idx: {qkv_idx} m: {m} n: {n} ref: {expect} actual:"
                    t" {actual}"
                ),
                rtol=rtol,
            )
    else:
        for m, n in std.itertools.product(range(total_num_tokens), range(N)):
            var expect: Scalar[out_type]

            comptime if has_epilogue:
                expect = test_epilogue(m, n, c_ref_host[m, n][0])
            else:
                expect = c_ref_host[m, n][0]

            var actual = c_host[m, n][0]
            assert_almost_equal(
                actual, expect, msg=String(t"m: {m} n: {n}"), rtol=rtol
            )

    # Cleanup
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = c_ref_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def test_negative_lora_id[
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
        "Testing negative lora_id behavior:",
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

    # Create host A C buffers
    var a_size = total_num_tokens * K

    var c_size = total_num_tokens * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_active_experts + 1
    )

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_num_tokens, Idx[K])),
    )

    # Create host B buffers
    var b_size = num_experts * N * K
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_active_experts
    )

    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
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

    # Create device buffers
    var a_dev_buffer = ctx.enqueue_create_buffer[a_type](a_size)
    var c_dev_buffer = ctx.enqueue_create_buffer[c_type](c_size)
    var b_dev_buffer = ctx.enqueue_create_buffer[b_type](b_size)
    var a_offsets_dev_buffer = ctx.enqueue_create_buffer[.uint32](
        num_active_experts + 1
    )
    var expert_ids_dev_buffer = ctx.enqueue_create_buffer[.int32](
        num_active_experts
    )

    var a_dev = TileTensor[a_type](
        a_dev_buffer,
        row_major(Coord(total_num_tokens, Idx[K])),
    )
    var c_dev = TileTensor[c_type](
        c_dev_buffer,
        row_major(Coord(total_num_tokens, Idx[N])),
    )
    var b_dev = TileTensor[b_type](
        b_dev_buffer,
        row_major[num_experts, N, K](),
    )
    var a_offsets_dev = TileTensor[.uint32](
        a_offsets_dev_buffer,
        row_major(Coord(num_active_experts + 1)),
    )
    var expert_ids_dev = TileTensor[.int32](
        expert_ids_dev_buffer,
        row_major(Coord(num_active_experts)),
    )

    # Move inputs to device
    ctx.enqueue_copy(a_dev_buffer, a_host_ptr)
    ctx.enqueue_copy(b_dev_buffer, b_host_ptr)
    ctx.enqueue_copy(a_offsets_dev_buffer, a_offsets_host_ptr)
    ctx.enqueue_copy(expert_ids_dev_buffer, expert_ids_host_ptr)

    # Run naive grouped matmul
    naive_grouped_matmul(
        c_dev,
        a_dev,
        b_dev,
        a_offsets_dev,
        expert_ids_dev,
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
        var has_negative_id = expert_id == -1

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
                var value = c_host_ptr[global_token_idx * N + n]
                if value != 0:
                    has_non_zero = True
                    break

            if has_negative_id == has_non_zero:
                print(
                    "ERROR: Found non-zero value for expert_id -1 at token",
                    global_token_idx,
                )
                print("Values:", end="")
                for n in range(min(5, N)):  # Print first 5 values
                    print(c_host_ptr[global_token_idx * N + n], end=" ")
                print()
                raise Error("Expected zero values for expert_id -1")
            else:
                # For valid expert_id, should have mostly non-zero values
                if not has_non_zero:
                    print(
                        "WARNING: All values are zero for valid expert_id",
                        expert_id,
                        "at token",
                        global_token_idx,
                    )

        current_offset += num_tokens

    print("✓ Negative lora_id test passed - expert_id -1 produces zero outputs")

    # Cleanup
    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def test_step3p5_moe_dims[
    in_type: DType,
    out_type: DType,
    expert_shape: IndexList[2],
    num_experts: Int,
](num_active: Int, tokens_per_expert: Int, ctx: DeviceContext,) raises:
    """Correctness test for Step-3.5-Flash MoE dimensions on B200.

    Tests grouped_matmul against naive_grouped_matmul with 288 experts
    and a >4 GiB weight buffer.
    """
    var total_tokens = num_active * tokens_per_expert
    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    print(
        num_active,
        "active of",
        num_experts,
        "experts of shape",
        expert_shape,
        "total_tokens",
        total_tokens,
    )

    # ---- A (activations): [total_tokens, K] ----
    var a_size = total_tokens * K
    var a_host_ptr = ctx.enqueue_create_host_buffer[in_type](a_size)
    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_tokens, Idx[K])),
    )
    random(a_host)

    # ---- B (expert weights): [num_experts, N, K] ----
    # Use runtime Idx so `random` uses a runtime loop instead of
    # hitting the compile-time element cap.
    var b_size = num_experts * N * K
    var b_host_ptr = ctx.enqueue_create_host_buffer[in_type](b_size)
    var b_host = TileTensor(
        b_host_ptr,
        row_major(Coord(num_experts, Idx[N], Idx[K])),
    )
    random(b_host)

    # ---- C (output): [total_tokens, N] ----
    var c_size = total_tokens * N
    var c_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[out_type](c_size)
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(total_tokens, Idx[N])),
    )
    var c_ref_host = TileTensor(
        c_ref_host_ptr,
        row_major(Coord(total_tokens, Idx[N])),
    )

    # ---- offsets & expert ids ----
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_experts + 1
    )
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )
    for i in range(num_experts + 1):
        a_offsets_host_ptr[i] = 0
    for i in range(num_experts):
        expert_ids_host_ptr[i] = 0

    a_offsets_host_ptr[0] = 0
    for i in range(num_active):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            tokens_per_expert
        )
        # Use the LAST experts to hit the highest byte offsets.
        expert_ids_host_ptr[i] = Int32(num_experts - num_active + i)

    # ---- device buffers ----
    var a_dev_buf = ctx.enqueue_create_buffer[in_type](a_size)
    var b_dev_buf = ctx.enqueue_create_buffer[in_type](b_size)
    var c_dev_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var c_ref_dev_buf = ctx.enqueue_create_buffer[out_type](c_size)
    var off_dev_buf = ctx.enqueue_create_buffer[.uint32](num_experts + 1)
    var eid_dev_buf = ctx.enqueue_create_buffer[.int32](num_experts)

    var a_dev = TileTensor[in_type](
        a_dev_buf, row_major(Coord(total_tokens, Idx[K]))
    )
    var b_dev = TileTensor[in_type](
        b_dev_buf,
        row_major[num_experts, N, K](),
    )
    var c_dev = TileTensor[out_type](
        c_dev_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var c_ref_dev = TileTensor[out_type](
        c_ref_dev_buf, row_major(Coord(total_tokens, Idx[N]))
    )
    var off_dev = TileTensor[.uint32](
        off_dev_buf, row_major(Coord(num_experts + 1))
    )
    var eid_dev = TileTensor[.int32](
        eid_dev_buf, row_major(Coord(Idx[num_experts]))
    )

    ctx.enqueue_copy(a_dev_buf, a_host_ptr)
    ctx.enqueue_copy(b_dev_buf, b_host_ptr)
    ctx.enqueue_copy(off_dev_buf, a_offsets_host_ptr)
    ctx.enqueue_copy(eid_dev_buf, expert_ids_host_ptr)
    ctx.synchronize()

    naive_grouped_matmul(
        c_ref_dev,
        a_dev,
        b_dev,
        off_dev,
        eid_dev,
        tokens_per_expert,
        num_active,
        ctx,
    )
    ctx.synchronize()

    grouped_matmul(
        c_dev,
        a_dev,
        b_dev,
        off_dev,
        eid_dev,
        tokens_per_expert,
        num_active,
        ctx,
    )
    ctx.synchronize()

    ctx.enqueue_copy(c_host_ptr, c_dev_buf)
    ctx.enqueue_copy(c_ref_host_ptr, c_ref_dev_buf)
    ctx.synchronize()

    var rtol = 1e-2
    for m, n in std.itertools.product(range(total_tokens), range(N)):
        var expect = c_ref_host[m, n][0]
        var actual = c_host[m, n][0]
        assert_almost_equal(
            actual, expect, msg=String(t"m: {m} n: {n}"), rtol=rtol
        )

    _ = a_dev_buf^
    _ = b_dev_buf^
    _ = c_dev_buf^
    _ = c_ref_dev_buf^
    _ = off_dev_buf^
    _ = eid_dev_buf^


def main() raises:
    with DeviceContext() as ctx:
        # Single matmul
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 256),
        ](1, [128], [0], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(16, 256),
        ](1, [128], [0], ctx)

        # unaligned matmul
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(1024, 256),
        ](1, [200], [0], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(512, 1024),
        ](1, [256], [0], ctx)

        # simple expert routing
        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(256, 64),
        ](1, [128], [2], ctx)

        # simple aligned group routing
        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(256, 64),
        ](3, [32, 32 * 3, 32 * 7], [2, 0, 1], ctx)

        # simple unaligned group routing
        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(256, 64),
        ](2, [10, 60], [2, 0], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(2880, 512),
        ](2, [10, 60], [2, 0], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(5760, 512),
        ](2, [10, 60], [2, 0], ctx)

        # Multiple matmuls selecting part of experts
        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(768, 1024),
        ](2, [128, 256], [0, 2], ctx)

        # Multiple matmuls selecting part of experts
        # num_tokesn not multiple of tile size
        test[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(1280, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Multiple matmuls selecting part of experts
        # num_tokesn not multiple of tile size
        # expert N dimension not multiple of 256
        test[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(192, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        comptime if _is_sm10x_gpu(ctx.default_device_info):
            test[
                .bfloat16,
                .bfloat16,
                num_experts=6,
                expert_shape=Index(1280, 16),
            ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(16, 1024),
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Multiple matmuls selecting part of experts with epilogue
        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(768, 1024),
            has_epilogue=True,
        ](2, [128, 256], [0, 2], ctx)

        comptime ns = [16, 256]
        comptime ms = [16, 512]

        comptime for n_idx in range(len(ns)):
            comptime for m_idx in range(len(ms)):
                comptime n = ns[n_idx]
                comptime m = ms[m_idx]

                comptime if m == 16 or n == 16:
                    comptime if ctx.default_device_info != B200:
                        continue
                # Test that expert id of -1 results in 0s in the output
                test[
                    .bfloat16,
                    .bfloat16,
                    num_experts=2,
                    expert_shape=Index(n, m),
                ](2, [64, 128], [0, -1], ctx)

                # Test negative lora_id behavior with naive matmul
                test_negative_lora_id[
                    .bfloat16,
                    .bfloat16,
                    num_experts=2,
                    expert_shape=Index(n, m),
                ](2, [64, 128], [0, -1], ctx)

        # QKV perm dim test
        test[
            .bfloat16,
            .bfloat16,
            num_experts=6,
            expert_shape=Index(192, 1024),
            qkv_perm_dim=True,
        ](4, [27, 1500, 300, 150], [0, 3, 2, 4], ctx)

        # Step-3.5-Flash MoE dimensions (288 experts, hidden=4096,
        # moe_dim=1280, top_k=8, bfloat16).
        # The model crashes with CUDA_ERROR_ILLEGAL_ADDRESS on B200
        # in grouped_matmul_sm100 (blackwell_tma_umma_warp_specialized_kernel)
        # when processing >=1024 input tokens (8192 token-expert rows).
        comptime if _is_sm10x_gpu(ctx.default_device_info):
            # Reproduces CUDA_ERROR_ILLEGAL_ADDRESS on B200 with the
            # Step-3.5-Flash MoE dimensions: 288 experts, 8 active,
            # expert_shape=(2560, 4096).  Total B buffer is ~5.6 GiB;
            # the crash appears related to >4 GiB weight buffer
            # addressing in the sm100 grouped matmul kernel.
            test_step3p5_moe_dims[.bfloat16, .bfloat16, Index(2560, 4096), 288](
                8, 256, ctx
            )

        # FP8 grouped matmul (H100 only).
        comptime if ctx.default_device_info == H100:
            test[
                .float8_e4m3fn,
                .bfloat16,
                num_experts=4,
                expert_shape=Index(256, 256),
            ](2, [32, 64], [0, 2], ctx)

            test[
                .float8_e4m3fn,
                .bfloat16,
                num_experts=4,
                expert_shape=Index(256, 128),
            ](3, [10, 60, 30], [2, 0, 1], ctx)

            # Non-128-aligned K dimension.
            test[
                .float8_e4m3fn,
                .bfloat16,
                num_experts=4,
                expert_shape=Index(256, 192),
            ](2, [25, 40], [1, 3], ctx)

        # float32 C output. The SM90/SM100 TMA kernels cannot emit a float32
        # C (their store path is 16-bit and they hard-assert
        # `c_type != float32`); the `grouped_matmul` dispatch must route
        # float32 output to the naive path. These shapes mirror the LoRA
        # `o_proj` grouped matmuls that crashed an un-quantized (float32)
        # model serving with LoRA on B200: the LoRA-A "shrink"
        # (`[M, hidden] @ [G, rank, hidden]`, expert_shape=(rank, hidden)) and
        # the LoRA-B "expand" (`[M, rank] @ [G, hidden, rank]`,
        # expert_shape=(hidden, rank)). hidden=576, rank=16 = SmolLM-135M.
        # See SERVOPT-1478.
        test[
            .float32,
            .float32,
            num_experts=4,
            expert_shape=Index(16, 576),  # LoRA shrink (contracts hidden)
        ](2, [128, 64], [0, 2], ctx)

        test[
            .float32,
            .float32,
            num_experts=4,
            expert_shape=Index(576, 16),  # LoRA expand (contracts rank)
        ](2, [128, 64], [0, 2], ctx)

        # float32 output with an elementwise epilogue (mirrors the LoRA
        # shrink's permute epilogue threading through the naive path).
        test[
            .float32,
            .float32,
            num_experts=4,
            expert_shape=Index(16, 576),
            has_epilogue=True,
        ](2, [128, 64], [0, 2], ctx)

        # float32 QKV-permute epilogue: the exact epilogue shape the LoRA
        # shrink (`shrink_qkv_permute_3mn_sm100`) uses.
        test[
            .float32,
            .float32,
            num_experts=4,
            expert_shape=Index(16, 576),
            qkv_perm_dim=True,
        ](2, [128, 64], [0, 2], ctx)
