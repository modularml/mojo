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
"""Regression test for SM100 grouped matmul with small K.

When K is small enough that the SM100 UMMA minimum K (16 for bf16) is
not met, the kernel pads BK and uses CUDA core copies (instead of TMA)
to load tiles into shared memory with zero-fill.  When the TMA stride
constraint (K * sizeof >= 16 bytes) is satisfied but K < UMMA minimum,
TMA OOB zero-fill handles the padding.  This test exercises K=4 (CUDA
core path) and K=8 (TMA with OOB zero-fill).
"""


from max.gpu.host import DeviceContext
from layout import (
    Coord,
    Idx,
    TileTensor,
    row_major,
)
from layout._fillers import random
from linalg.grouped_matmul import grouped_matmul, naive_grouped_matmul
from std.testing import assert_almost_equal

from std.utils import IndexList
from std.utils.index import Index
import std.itertools


def test[
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
    comptime a_type = in_type
    comptime b_type = in_type
    comptime c_type = out_type

    comptime N = expert_shape[0]
    comptime K = expert_shape[1]

    print(
        "Testing: experts=",
        num_experts,
        " N=",
        N,
        " K=",
        K,
        " active=",
        num_active_experts,
    )

    var total_num_tokens = 0
    var max_num_tokens_by_expert = 0
    for i in range(len(num_tokens_by_expert)):
        total_num_tokens += num_tokens_by_expert[i]
        max_num_tokens_by_expert = max(
            max_num_tokens_by_expert, num_tokens_by_expert[i]
        )

    var a_size = total_num_tokens * K
    var c_size = total_num_tokens * N

    var a_host_ptr = ctx.enqueue_create_host_buffer[a_type](a_size)
    var c_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var c_ref_host_ptr = ctx.enqueue_create_host_buffer[c_type](c_size)
    var a_offsets_host_ptr = ctx.enqueue_create_host_buffer[.uint32](
        num_experts + 1
    )

    var a_host = TileTensor(
        a_host_ptr,
        row_major(Coord(total_num_tokens, Idx[K])),
    )
    var c_host = TileTensor(
        c_host_ptr,
        row_major(Coord(total_num_tokens, Idx[N])),
    )
    var c_ref_host = TileTensor(
        c_ref_host_ptr,
        row_major(Coord(total_num_tokens, Idx[N])),
    )

    var b_size = num_experts * N * K
    var b_host_ptr = ctx.enqueue_create_host_buffer[b_type](b_size)
    var expert_ids_host_ptr = ctx.enqueue_create_host_buffer[.int32](
        num_experts
    )

    var b_host = TileTensor(
        b_host_ptr,
        row_major[num_experts, N, K](),
    )

    a_offsets_host_ptr[0] = 0
    for i in range(num_active_experts):
        a_offsets_host_ptr[i + 1] = a_offsets_host_ptr[i] + UInt32(
            num_tokens_by_expert[i]
        )
        expert_ids_host_ptr[i] = Int32(expert_ids[i])

    random(a_host)
    random(b_host)

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
        row_major(Coord(total_num_tokens, Idx[N])),
    )
    var c_ref_dev = TileTensor[c_type](
        c_ref_dev_buffer,
        row_major(Coord(total_num_tokens, Idx[N])),
    )
    var b_dev = TileTensor[b_type](
        b_dev_buffer,
        row_major[num_experts, N, K](),
    )
    var a_offsets_dev = TileTensor[.uint32](
        a_offsets_dev_buffer,
        row_major(Coord(num_experts + 1)),
    )
    var expert_ids_dev = TileTensor[.int32](
        expert_ids_dev_buffer,
        row_major(Coord(Idx[num_experts])),
    )

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

    grouped_matmul(
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

    for m, n in std.itertools.product(range(total_num_tokens), range(N)):
        var expect = c_ref_host[m, n][0]
        var actual = c_host[m, n][0]
        assert_almost_equal(
            actual, expect, msg=String(t"m: {m} n: {n}"), rtol=1e-2
        )

    print("  PASS")

    _ = a_dev_buffer^
    _ = b_dev_buffer^
    _ = c_dev_buffer^
    _ = c_ref_dev_buffer^
    _ = a_offsets_dev_buffer^
    _ = expert_ids_dev_buffer^


def main() raises:
    with DeviceContext() as ctx:
        # K=8: TMA path (stride=16 bytes, meets TMA alignment).
        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 8),
        ](1, [46], [0], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=1,
            expert_shape=Index(256, 8),
        ](1, [128], [0], ctx)

        test[
            .bfloat16,
            .bfloat16,
            num_experts=4,
            expert_shape=Index(256, 8),
        ](3, [16, 32, 24], [0, 2, 1], ctx)

        # K=1..7: CUDA core path (stride < 16 bytes).
        comptime for K in range(1, 8):
            test[
                .bfloat16,
                .bfloat16,
                num_experts=1,
                expert_shape=Index(128, K),
            ](1, [64], [0], ctx)

        print("All small-K grouped matmul tests passed!")

        print("All small-K grouped matmul tests passed!")
