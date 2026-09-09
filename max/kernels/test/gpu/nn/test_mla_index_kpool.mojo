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
"""Tests for the DSA indexer's k-pool compression kernel."""

from std.math import exp
from std.random import rand
from std.testing import assert_almost_equal, assert_equal, assert_true

from layout import TileTensor, row_major
from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer

from nn.attention.gpu.mla_index_kpool import (
    kpool_compress_kernel,
    kpool_seed_tail_kernel,
    kpool_expand_topk_kernel,
    kpool_tail_update_kernel,
)


def test_kpool_compress[
    head_dim: Int, kpool: Int
](
    seq_lens: List[Int],
    ctx: DeviceContext,
    cache_lens: List[Int] = List[Int](),
    extra_blocks: Int = 0,
) raises:
    """Compress `seq_lens` requests' keys and check every pooled channel.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.

    Args:
        seq_lens: Tokens per request in this call.
        ctx: Device context.
        cache_lens: Cached-prefix length per request; empty means all zero.
        extra_blocks: Surplus blocks to launch beyond the real pool count.
    """
    var batch_size = len(seq_lens)
    var clens = List[Int]()
    for i in range(batch_size):
        clens.append(cache_lens[i] if len(cache_lens) > 0 else 0)
    var total_tokens = 0
    for i in range(batch_size):
        total_tokens += seq_lens[i]

    # A cached prefix ending mid-pool costs the call the pool it opens in.
    var aligns = List[Int]()
    var pools_per = List[Int]()
    var total_pools = 0
    for i in range(batch_size):
        var align = (kpool - clens[i] % kpool) % kpool
        var n = (seq_lens[i] - align) // kpool if seq_lens[i] > align else 0
        aligns.append(align)
        pools_per.append(n)
        total_pools += n

    print(
        "kpool_compress: head_dim=",
        head_dim,
        " kpool=",
        kpool,
        " tokens=",
        total_tokens,
        " pools=",
        total_pools,
    )
    assert_true(total_pools > 0, "test shape selects no complete pool")

    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var pro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var clen_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](
        (total_pools + extra_blocks) * head_dim
    )
    ctx.synchronize()

    rand(k_host.unsafe_ptr(), total_tokens * head_dim)
    rand(gate_host.unsafe_ptr(), total_tokens * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)

    # Spread the gate scores, or the member weights come out nearly equal and
    # a wrong member mapping would not show.
    for i in range(total_tokens * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 4.0

    var tok_off = 0
    var pool_off = 0
    for b in range(batch_size):
        iro_host[b] = UInt32(tok_off)
        pro_host[b] = UInt32(pool_off)
        tok_off += seq_lens[b]
        pool_off += pools_per[b]
    iro_host[batch_size] = UInt32(tok_off)
    pro_host[batch_size] = UInt32(pool_off)

    for b in range(batch_size):
        clen_host[b] = UInt32(clens[b])
    var clen_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(clen_dev, clen_host)
    var k_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var gate_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var pro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](
        (total_pools + extra_blocks) * head_dim
    )
    ctx.enqueue_memset(out_dev, 0)

    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(gate_dev, gate_host)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(pro_dev, pro_host)
    ctx.synchronize()

    var out_tile = TileTensor(
        out_dev, row_major(total_pools + extra_blocks, head_dim)
    )
    var k_tile = TileTensor(k_dev, row_major(total_tokens, head_dim))
    var gate_tile = TileTensor(gate_dev, row_major(total_tokens, head_dim))
    var ape_tile = TileTensor(ape_dev, row_major(kpool, head_dim))
    var iro_tile = TileTensor(iro_dev, row_major(batch_size + 1))
    var pro_tile = TileTensor(pro_dev, row_major(batch_size + 1))
    var clen_tile = TileTensor(clen_dev, row_major(batch_size))

    comptime kernel = kpool_compress_kernel[
        .bfloat16,
        type_of(k_tile.as_immut()).LayoutType,
        ImmOrigin(k_tile.origin),
        type_of(gate_tile.as_immut()).LayoutType,
        ImmOrigin(gate_tile.origin),
        type_of(ape_tile.as_immut()).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(iro_tile.as_immut()).LayoutType,
        ImmOrigin(iro_tile.origin),
        type_of(pro_tile.as_immut()).LayoutType,
        ImmOrigin(pro_tile.origin),
        type_of(clen_tile.as_immut()).LayoutType,
        out_tile.LayoutType,
        out_tile.origin,
        head_dim,
        kpool,
    ]
    ctx.enqueue_function[kernel](
        out_tile,
        k_tile.as_immut(),
        gate_tile.as_immut(),
        ape_tile.as_immut(),
        iro_tile.as_immut(),
        pro_tile.as_immut(),
        clen_tile.as_immut(),
        # Over-sized on purpose: the surplus blocks must write nothing.
        grid_dim=(total_pools + extra_blocks, 1, 1),
        block_dim=(head_dim, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for b in range(batch_size):
        for p in range(pools_per[b]):
            var pool_row = Int(pro_host[b]) + p
            var first_row = Int(iro_host[b]) + aligns[b] + p * kpool
            for c in range(head_dim):
                # No max-subtraction here, so a match also says the kernel's
                # stabilization was neutral.
                var denom = Float64(0)
                var acc = Float64(0)
                for m in range(kpool):
                    var logit = Float64(
                        gate_host[(first_row + m) * head_dim + c].cast[
                            .float32
                        ]()
                    ) + Float64(ape_host[m * head_dim + c])
                    var w = exp(logit)
                    denom += w
                    acc += w * Float64(
                        k_host[(first_row + m) * head_dim + c].cast[.float32]()
                    )
                var want = acc / denom
                var got = Float64(
                    out_host[pool_row * head_dim + c].cast[.float32]()
                )
                # The tolerance comes from bf16's 8-bit mantissa.
                assert_almost_equal(
                    got,
                    want,
                    atol=4e-3,
                    rtol=1e-2,
                    msg=String("pool ", pool_row, " channel ", c, " mismatch"),
                )

    # Surplus blocks must have written nothing.
    for i in range(
        total_pools * head_dim, (total_pools + extra_blocks) * head_dim
    ):
        assert_equal(
            out_host[i],
            Scalar[.bfloat16](0),
            String("a surplus block wrote output element ", i),
        )

    _ = k_dev
    _ = gate_dev
    _ = ape_dev
    _ = iro_dev
    _ = pro_dev
    _ = clen_dev
    _ = out_dev


def test_kpool_one_is_identity[head_dim: Int](ctx: DeviceContext) raises:
    """At kpool=1 a pooled key must be its single member, bit for bit.

    A one-member softmax is exactly 1.0 whatever the gate and position
    embedding hold, so this pins that neither reaches the output.
    """
    comptime kpool = 1
    var seq_lens = [5, 3]
    var batch_size = len(seq_lens)
    var total_tokens = 8

    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var pro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var clen_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var out_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    ctx.synchronize()

    rand(k_host.unsafe_ptr(), total_tokens * head_dim)
    rand(gate_host.unsafe_ptr(), total_tokens * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)
    # At kpool=1 neither the gate nor the position embedding may reach the
    # output, so make both large and uneven.
    for i in range(total_tokens * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 20.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 20.0

    iro_host[0] = 0
    iro_host[1] = UInt32(seq_lens[0])
    iro_host[2] = UInt32(total_tokens)
    pro_host[0] = 0
    pro_host[1] = UInt32(seq_lens[0])
    pro_host[2] = UInt32(total_tokens)

    for b in range(batch_size):
        clen_host[b] = 0
    var clen_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(clen_dev, clen_host)
    var k_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var gate_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var pro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var out_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)

    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(gate_dev, gate_host)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(pro_dev, pro_host)
    ctx.synchronize()

    var out_tile = TileTensor(out_dev, row_major(total_tokens, head_dim))
    var k_tile = TileTensor(k_dev, row_major(total_tokens, head_dim))
    var gate_tile = TileTensor(gate_dev, row_major(total_tokens, head_dim))
    var ape_tile = TileTensor(ape_dev, row_major(kpool, head_dim))
    var iro_tile = TileTensor(iro_dev, row_major(batch_size + 1))
    var pro_tile = TileTensor(pro_dev, row_major(batch_size + 1))
    var clen_tile = TileTensor(clen_dev, row_major(batch_size))

    comptime kernel = kpool_compress_kernel[
        .bfloat16,
        type_of(k_tile.as_immut()).LayoutType,
        ImmOrigin(k_tile.origin),
        type_of(gate_tile.as_immut()).LayoutType,
        ImmOrigin(gate_tile.origin),
        type_of(ape_tile.as_immut()).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(iro_tile.as_immut()).LayoutType,
        ImmOrigin(iro_tile.origin),
        type_of(pro_tile.as_immut()).LayoutType,
        ImmOrigin(pro_tile.origin),
        type_of(clen_tile.as_immut()).LayoutType,
        out_tile.LayoutType,
        out_tile.origin,
        head_dim,
        kpool,
    ]
    ctx.enqueue_function[kernel](
        out_tile,
        k_tile.as_immut(),
        gate_tile.as_immut(),
        ape_tile.as_immut(),
        iro_tile.as_immut(),
        pro_tile.as_immut(),
        clen_tile.as_immut(),
        grid_dim=(total_tokens, 1, 1),
        block_dim=(head_dim, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for i in range(total_tokens * head_dim):
        assert_equal(
            out_host[i],
            k_host[i],
            String("kpool=1 changed element ", i),
        )
    print("kpool=1 identity holds over", total_tokens * head_dim, "elements")

    _ = k_dev
    _ = gate_dev
    _ = ape_dev
    _ = iro_dev
    _ = pro_dev
    _ = clen_dev
    _ = out_dev


def test_decode_writer_matches_prefill_writer[
    head_dim: Int, kpool: Int
](num_requests: Int, seq_len: Int, ctx: DeviceContext) raises:
    """Feeding tokens one at a time must build the same pools as one batch.

    Prefill sees a pool's members together; decode sees them on `kpool`
    separate steps and has to carry the earlier ones in its ring. The pooled
    keys have to come out identical, and identical here means bit for bit --
    both writers run the same shared math, so anything less is a bug rather
    than a rounding difference.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.

    Args:
        num_requests: Requests decoded in lockstep.
        seq_len: Tokens fed to each request.
        ctx: Device context.
    """
    var total_tokens = num_requests * seq_len
    var pools_per_req = seq_len // kpool
    var total_pools = num_requests * pools_per_req
    print(
        "decode-vs-prefill: requests=",
        num_requests,
        " seq_len=",
        seq_len,
        " pools=",
        total_pools,
    )
    assert_true(total_pools > 0, "test shape closes no pool")

    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_tokens * head_dim
    )
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    ctx.synchronize()
    rand(k_host.unsafe_ptr(), total_tokens * head_dim)
    rand(gate_host.unsafe_ptr(), total_tokens * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)
    for i in range(total_tokens * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 4.0

    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    ctx.enqueue_copy(ape_dev, ape_host)
    var ape_tile = TileTensor(ape_dev, row_major(kpool, head_dim))

    # Arm A: one prefill pass.
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](num_requests + 1)
    var pro_host = ctx.enqueue_create_host_buffer[.uint32](num_requests + 1)
    ctx.synchronize()
    for b in range(num_requests):
        iro_host[b] = UInt32(b * seq_len)
        pro_host[b] = UInt32(b * pools_per_req)
    iro_host[num_requests] = UInt32(total_tokens)
    pro_host[num_requests] = UInt32(total_pools)

    var k_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var gate_dev = ctx.enqueue_create_buffer[.bfloat16](total_tokens * head_dim)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](num_requests + 1)
    var pro_dev = ctx.enqueue_create_buffer[.uint32](num_requests + 1)
    var prefill_dev = ctx.enqueue_create_buffer[.bfloat16](
        total_pools * head_dim
    )
    ctx.enqueue_copy(k_dev, k_host)
    ctx.enqueue_copy(gate_dev, gate_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(pro_dev, pro_host)
    ctx.synchronize()

    var prefill_tile = TileTensor(prefill_dev, row_major(total_pools, head_dim))
    var k_tile = TileTensor(k_dev, row_major(total_tokens, head_dim))
    var gate_tile = TileTensor(gate_dev, row_major(total_tokens, head_dim))
    var iro_tile = TileTensor(iro_dev, row_major(num_requests + 1))
    var pro_tile = TileTensor(pro_dev, row_major(num_requests + 1))
    var clen_dev2 = ctx.enqueue_create_buffer[.uint32](num_requests)
    ctx.enqueue_memset(clen_dev2, 0)
    var clen_tile = TileTensor(clen_dev2, row_major(num_requests))

    comptime prefill_kernel = kpool_compress_kernel[
        .bfloat16,
        type_of(k_tile.as_immut()).LayoutType,
        ImmOrigin(k_tile.origin),
        type_of(gate_tile.as_immut()).LayoutType,
        ImmOrigin(gate_tile.origin),
        type_of(ape_tile.as_immut()).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(iro_tile.as_immut()).LayoutType,
        ImmOrigin(iro_tile.origin),
        type_of(pro_tile.as_immut()).LayoutType,
        ImmOrigin(pro_tile.origin),
        type_of(clen_tile.as_immut()).LayoutType,
        prefill_tile.LayoutType,
        prefill_tile.origin,
        head_dim,
        kpool,
    ]
    ctx.enqueue_function[prefill_kernel](
        prefill_tile,
        k_tile.as_immut(),
        gate_tile.as_immut(),
        ape_tile.as_immut(),
        iro_tile.as_immut(),
        pro_tile.as_immut(),
        clen_tile.as_immut(),
        grid_dim=(total_pools, 1, 1),
        block_dim=(head_dim, 1, 1),
    )

    # Arm B: one token per step, through the ring.
    var tail_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * 2 * kpool * head_dim
    )
    ctx.enqueue_memset(tail_dev, 0)
    var step_k_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * head_dim
    )
    var step_gate_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * head_dim
    )
    var pos_dev = ctx.enqueue_create_buffer[.int32](num_requests)
    var slot_dev = ctx.enqueue_create_buffer[.uint32](num_requests)
    var step_pooled_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * head_dim
    )
    var closed_dev = ctx.enqueue_create_buffer[.int32](num_requests)

    var step_k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var step_gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var pos_host = ctx.enqueue_create_host_buffer[.int32](num_requests)
    var step_pooled_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var closed_host = ctx.enqueue_create_host_buffer[.int32](num_requests)
    var decode_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_pools * head_dim
    )
    var prefill_host = ctx.enqueue_create_host_buffer[.bfloat16](
        total_pools * head_dim
    )
    ctx.synchronize()

    var tail_tile = TileTensor(
        tail_dev, row_major(num_requests, 2 * kpool, head_dim)
    )
    var step_k_tile = TileTensor(step_k_dev, row_major(num_requests, head_dim))
    var step_gate_tile = TileTensor(
        step_gate_dev, row_major(num_requests, head_dim)
    )
    var pos_tile = TileTensor(pos_dev, row_major(num_requests))
    var slot_host = ctx.enqueue_create_host_buffer[.uint32](num_requests)
    ctx.synchronize()
    for r in range(num_requests):
        slot_host[r] = UInt32(r)
    ctx.enqueue_copy(slot_dev, slot_host)
    var slot_tile = TileTensor(slot_dev, row_major(num_requests))
    var step_pooled_tile = TileTensor(
        step_pooled_dev, row_major(num_requests, head_dim)
    )
    var closed_tile = TileTensor(closed_dev, row_major(num_requests))

    comptime tail_kernel = kpool_tail_update_kernel[
        .bfloat16,
        tail_tile.LayoutType,
        tail_tile.origin,
        step_pooled_tile.LayoutType,
        step_pooled_tile.origin,
        closed_tile.LayoutType,
        closed_tile.origin,
        type_of(step_k_tile.as_immut()).LayoutType,
        ImmOrigin(step_k_tile.origin),
        type_of(step_gate_tile.as_immut()).LayoutType,
        ImmOrigin(step_gate_tile.origin),
        type_of(ape_tile.as_immut()).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(pos_tile.as_immut()).LayoutType,
        ImmOrigin(pos_tile.origin),
        type_of(slot_tile.as_immut()).LayoutType,
        ImmOrigin(slot_tile.origin),
        head_dim,
        kpool,
    ]

    var closes_seen = 0
    for t in range(seq_len):
        for r in range(num_requests):
            pos_host[r] = Int32(t)
            for c in range(head_dim):
                step_k_host[r * head_dim + c] = k_host[
                    (r * seq_len + t) * head_dim + c
                ]
                step_gate_host[r * head_dim + c] = gate_host[
                    (r * seq_len + t) * head_dim + c
                ]
        ctx.enqueue_copy(step_k_dev, step_k_host)
        ctx.enqueue_copy(step_gate_dev, step_gate_host)
        ctx.enqueue_copy(pos_dev, pos_host)

        ctx.enqueue_function[tail_kernel](
            tail_tile,
            step_pooled_tile,
            closed_tile,
            step_k_tile.as_immut(),
            step_gate_tile.as_immut(),
            ape_tile.as_immut(),
            pos_tile.as_immut(),
            slot_tile.as_immut(),
            Int32(num_requests),
            grid_dim=(num_requests, 1, 1),
            block_dim=(head_dim, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(closed_host, closed_dev)
        ctx.enqueue_copy(step_pooled_host, step_pooled_dev)
        ctx.synchronize()

        var want_closed = t % kpool == kpool - 1
        for r in range(num_requests):
            var cp = Int(closed_host[r])
            if want_closed:
                assert_equal(
                    cp,
                    t // kpool,
                    String("step ", t, " request ", r, " closed wrong pool"),
                )
                closes_seen += 1
                var dst = (r * pools_per_req + cp) * head_dim
                for c in range(head_dim):
                    decode_host[dst + c] = step_pooled_host[r * head_dim + c]
            else:
                assert_equal(
                    cp,
                    -1,
                    String("step ", t, " request ", r, " closed early"),
                )

    assert_equal(closes_seen, total_pools, "wrong number of pools closed")

    ctx.enqueue_copy(prefill_host, prefill_dev)
    ctx.synchronize()

    for i in range(total_pools * head_dim):
        assert_equal(
            decode_host[i],
            prefill_host[i],
            String("decode and prefill disagree at element ", i),
        )
    print("  decode matches prefill exactly over", total_pools, "pools")

    _ = k_dev
    _ = gate_dev
    _ = iro_dev
    _ = pro_dev
    _ = prefill_dev
    _ = ape_dev
    _ = tail_dev
    _ = step_k_dev
    _ = step_gate_dev
    _ = pos_dev
    _ = slot_dev
    _ = clen_dev2
    _ = step_pooled_dev
    _ = closed_dev


def _decode_pools_for_request_zero[
    head_dim: Int, kpool: Int
](
    k_host: HostBuffer[.bfloat16],
    gate_host: HostBuffer[.bfloat16],
    ape_tile: TileTensor[mut=False, .float32, ...],
    num_requests: Int,
    seq_len: Int,
    start_offsets: List[Int],
    out_host: HostBuffer[.bfloat16],
    ctx: DeviceContext,
) raises -> Int:
    """Decode `num_requests` requests, keeping only request 0's pooled keys.

    Request `r` starts at absolute position `start_offsets[r]`, so requests
    close their pools on different steps. Returns how many pools request 0
    closed, and writes them to `out_host`.
    """
    var tail_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * 2 * kpool * head_dim
    )
    ctx.enqueue_memset(tail_dev, 0)
    var step_k_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * head_dim
    )
    var step_gate_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * head_dim
    )
    var pos_dev = ctx.enqueue_create_buffer[.int32](num_requests)
    var slot_dev = ctx.enqueue_create_buffer[.uint32](num_requests)
    var pooled_dev = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * head_dim
    )
    var closed_dev = ctx.enqueue_create_buffer[.int32](num_requests)

    var step_k_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var step_gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var pos_host = ctx.enqueue_create_host_buffer[.int32](num_requests)
    var pooled_host = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var closed_host = ctx.enqueue_create_host_buffer[.int32](num_requests)
    ctx.synchronize()

    var tail_tile = TileTensor(
        tail_dev, row_major(num_requests, 2 * kpool, head_dim)
    )
    var step_k_tile = TileTensor(step_k_dev, row_major(num_requests, head_dim))
    var step_gate_tile = TileTensor(
        step_gate_dev, row_major(num_requests, head_dim)
    )
    var pos_tile = TileTensor(pos_dev, row_major(num_requests))
    var slot_host = ctx.enqueue_create_host_buffer[.uint32](num_requests)
    ctx.synchronize()
    for r in range(num_requests):
        slot_host[r] = UInt32(r)
    ctx.enqueue_copy(slot_dev, slot_host)
    var slot_tile = TileTensor(slot_dev, row_major(num_requests))
    var pooled_tile = TileTensor(pooled_dev, row_major(num_requests, head_dim))
    var closed_tile = TileTensor(closed_dev, row_major(num_requests))

    comptime tail_kernel = kpool_tail_update_kernel[
        .bfloat16,
        tail_tile.LayoutType,
        tail_tile.origin,
        pooled_tile.LayoutType,
        pooled_tile.origin,
        closed_tile.LayoutType,
        closed_tile.origin,
        type_of(step_k_tile.as_immut()).LayoutType,
        ImmOrigin(step_k_tile.origin),
        type_of(step_gate_tile.as_immut()).LayoutType,
        ImmOrigin(step_gate_tile.origin),
        type_of(ape_tile).LayoutType,
        ImmOrigin(ape_tile.origin),
        type_of(pos_tile.as_immut()).LayoutType,
        ImmOrigin(pos_tile.origin),
        type_of(slot_tile.as_immut()).LayoutType,
        ImmOrigin(slot_tile.origin),
        head_dim,
        kpool,
    ]

    var kept = 0
    for t in range(seq_len):
        for r in range(num_requests):
            pos_host[r] = Int32(start_offsets[r] + t)
            for c in range(head_dim):
                # Every request reads the same tokens, so a leak between
                # rings still produces a plausible-looking key.
                step_k_host[r * head_dim + c] = k_host[t * head_dim + c]
                step_gate_host[r * head_dim + c] = gate_host[t * head_dim + c]
        ctx.enqueue_copy(step_k_dev, step_k_host)
        ctx.enqueue_copy(step_gate_dev, step_gate_host)
        ctx.enqueue_copy(pos_dev, pos_host)
        ctx.enqueue_function[tail_kernel](
            tail_tile,
            pooled_tile,
            closed_tile,
            step_k_tile.as_immut(),
            step_gate_tile.as_immut(),
            ape_tile,
            pos_tile.as_immut(),
            slot_tile.as_immut(),
            Int32(num_requests),
            grid_dim=(num_requests, 1, 1),
            block_dim=(head_dim, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(closed_host, closed_dev)
        ctx.enqueue_copy(pooled_host, pooled_dev)
        ctx.synchronize()
        if Int(closed_host[0]) >= 0:
            for c in range(head_dim):
                out_host[kept * head_dim + c] = pooled_host[c]
            kept += 1

    _ = tail_dev
    _ = step_k_dev
    _ = step_gate_dev
    _ = pos_dev
    _ = slot_dev
    _ = pooled_dev
    _ = closed_dev
    return kept


def test_rings_are_per_request[
    head_dim: Int, kpool: Int
](ctx: DeviceContext) raises:
    """A request's pooled keys must not depend on who else is decoding.

    The neighbours here start at staggered positions, so they close their pools
    on different steps from request 0 and from each other -- the arrangement
    that catches a ring shared between requests, where a neighbour's stash
    lands in request 0's in-progress pool. Every request is fed identical
    tokens, so a leak produces a plausible-looking key rather than garbage.
    """
    comptime seq_len = 12
    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](seq_len * head_dim)
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](
        seq_len * head_dim
    )
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    var alone_host = ctx.enqueue_create_host_buffer[.bfloat16](
        seq_len * head_dim
    )
    var crowded_host = ctx.enqueue_create_host_buffer[.bfloat16](
        seq_len * head_dim
    )
    ctx.synchronize()
    rand(k_host.unsafe_ptr(), seq_len * head_dim)
    rand(gate_host.unsafe_ptr(), seq_len * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)
    for i in range(seq_len * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 4.0

    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.synchronize()
    var ape_tile = TileTensor(ape_dev, row_major(kpool, head_dim)).as_immut()

    var alone = _decode_pools_for_request_zero[head_dim, kpool](
        k_host, gate_host, ape_tile, 1, seq_len, [0], alone_host, ctx
    )
    var crowded = _decode_pools_for_request_zero[head_dim, kpool](
        k_host,
        gate_host,
        ape_tile,
        4,
        seq_len,
        [0, 1, 2, 3],
        crowded_host,
        ctx,
    )
    assert_equal(alone, crowded, "pool count changed with neighbours present")
    assert_true(alone > 0, "request 0 closed no pool")
    for i in range(alone * head_dim):
        assert_equal(
            alone_host[i],
            crowded_host[i],
            String("neighbours perturbed request 0 at element ", i),
        )
    print(
        "  request 0 unchanged by 3 staggered neighbours over", alone, "pools"
    )
    _ = ape_dev


def _drive_two_requests[
    head_dim: Int, kpool: Int
](
    k_host: HostBuffer[.bfloat16],
    gate_host: HostBuffer[.bfloat16],
    ape_t: TileTensor[mut=False, .float32, ...],
    seq_len: Int,
    swap_odd: Bool,
    out_host: HostBuffer[.bfloat16],
    ctx: DeviceContext,
) raises:
    """Decodes two requests, optionally swapping their rows on odd steps.

    `slot_idx` follows the request rather than the row, so both schedules must
    leave the same pooled keys. Output is indexed by (request, pool id).
    """
    comptime num_requests = 2
    var tail_d = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * 2 * kpool * head_dim
    )
    ctx.enqueue_memset(tail_d, 0)
    var k_d = ctx.enqueue_create_buffer[.bfloat16](num_requests * head_dim)
    var g_d = ctx.enqueue_create_buffer[.bfloat16](num_requests * head_dim)
    var pos_d = ctx.enqueue_create_buffer[.int32](num_requests)
    var slot_d = ctx.enqueue_create_buffer[.uint32](num_requests)
    var pooled_d = ctx.enqueue_create_buffer[.bfloat16](num_requests * head_dim)
    var closed_d = ctx.enqueue_create_buffer[.int32](num_requests)

    var k_s = ctx.enqueue_create_host_buffer[.bfloat16](num_requests * head_dim)
    var g_s = ctx.enqueue_create_host_buffer[.bfloat16](num_requests * head_dim)
    var pos_s = ctx.enqueue_create_host_buffer[.int32](num_requests)
    var slot_s = ctx.enqueue_create_host_buffer[.uint32](num_requests)
    var pooled_s = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * head_dim
    )
    var closed_s = ctx.enqueue_create_host_buffer[.int32](num_requests)
    ctx.synchronize()

    var tail_t = TileTensor(
        tail_d, row_major(num_requests, 2 * kpool, head_dim)
    )
    var k_t = TileTensor(k_d, row_major(num_requests, head_dim))
    var g_t = TileTensor(g_d, row_major(num_requests, head_dim))
    var pos_t = TileTensor(pos_d, row_major(num_requests))
    var slot_t = TileTensor(slot_d, row_major(num_requests))
    var pooled_t = TileTensor(pooled_d, row_major(num_requests, head_dim))
    var closed_t = TileTensor(closed_d, row_major(num_requests))

    comptime kern = kpool_tail_update_kernel[
        .bfloat16,
        tail_t.LayoutType,
        tail_t.origin,
        pooled_t.LayoutType,
        pooled_t.origin,
        closed_t.LayoutType,
        closed_t.origin,
        type_of(k_t.as_immut()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(g_t.as_immut()).LayoutType,
        ImmOrigin(g_t.origin),
        type_of(ape_t).LayoutType,
        ImmOrigin(ape_t.origin),
        type_of(pos_t.as_immut()).LayoutType,
        ImmOrigin(pos_t.origin),
        type_of(slot_t.as_immut()).LayoutType,
        ImmOrigin(slot_t.origin),
        head_dim,
        kpool,
    ]

    for t in range(seq_len):
        var swapped = swap_odd and (t % 2 == 1)
        for row in range(num_requests):
            # The request sitting in this row on this step.
            var req = (num_requests - 1 - row) if swapped else row
            pos_s[row] = Int32(t)
            slot_s[row] = UInt32(req)
            var src = (req * seq_len + t) * head_dim
            for c in range(head_dim):
                k_s[row * head_dim + c] = k_host[src + c]
                g_s[row * head_dim + c] = gate_host[src + c]
        ctx.enqueue_copy(k_d, k_s)
        ctx.enqueue_copy(g_d, g_s)
        ctx.enqueue_copy(pos_d, pos_s)
        ctx.enqueue_copy(slot_d, slot_s)
        ctx.enqueue_function[kern](
            tail_t,
            pooled_t,
            closed_t,
            k_t.as_immut(),
            g_t.as_immut(),
            ape_t,
            pos_t.as_immut(),
            slot_t.as_immut(),
            Int32(num_requests),
            grid_dim=(num_requests, 1, 1),
            block_dim=(head_dim, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(closed_s, closed_d)
        ctx.enqueue_copy(pooled_s, pooled_d)
        ctx.synchronize()
        for row in range(num_requests):
            var cid = Int(closed_s[row])
            if cid < 0:
                continue
            var req = (num_requests - 1 - row) if swapped else row
            var dst = (req * (seq_len // kpool) + cid) * head_dim
            for c in range(head_dim):
                out_host[dst + c] = pooled_s[row * head_dim + c]

    _ = tail_d
    _ = k_d
    _ = g_d
    _ = pos_d
    _ = slot_d
    _ = pooled_d
    _ = closed_d


def test_ring_survives_a_batch_reorder[
    head_dim: Int, kpool: Int
](seq_len: Int, ctx: DeviceContext) raises:
    """A request's pooled keys must not change when the batch reorders.

    A serving batch reorders between steps, so the request in row `r` at one
    step need not be the one that held row `r` at the last. The ring is
    addressed by `slot_idx[r]` for that reason. Decoding two requests with
    their rows swapped on every odd step must give the same pooled keys as
    decoding them in a fixed order.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.

    Args:
        seq_len: Tokens per request; a multiple of `kpool`.
        ctx: Device context.
    """
    comptime num_requests = 2
    var total = num_requests * seq_len
    print("batch reorder: seq_len=", seq_len, " kpool=", kpool)

    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](total * head_dim)
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](total * head_dim)
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    ctx.synchronize()
    rand(k_host.unsafe_ptr(), total * head_dim)
    rand(gate_host.unsafe_ptr(), total * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)
    # The requests need different tokens, or a read from the wrong slot would
    # still look right.
    for i in range(seq_len * head_dim, total * head_dim):
        k_host[i] = (k_host[i] - 0.5) * 4.0
    for i in range(total * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 4.0

    var ape_d = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    ctx.enqueue_copy(ape_d, ape_host)
    ctx.synchronize()
    var ape_t = TileTensor(ape_d, row_major(kpool, head_dim)).as_immut()

    var pools = total // kpool
    var fixed_out = ctx.enqueue_create_host_buffer[.bfloat16](pools * head_dim)
    var swapped_out = ctx.enqueue_create_host_buffer[.bfloat16](
        pools * head_dim
    )
    ctx.synchronize()

    _drive_two_requests[head_dim, kpool](
        k_host, gate_host, ape_t, seq_len, False, fixed_out, ctx
    )
    _drive_two_requests[head_dim, kpool](
        k_host, gate_host, ape_t, seq_len, True, swapped_out, ctx
    )

    for i in range(pools * head_dim):
        assert_equal(
            swapped_out[i],
            fixed_out[i],
            String("reordering the batch changed pooled element ", i),
        )
    print("  ", pools, "pools unchanged by a reordering batch")
    _ = ape_d


def test_expand_topk[
    kpool: Int, pool_topk: Int, always_select_tail: Bool
](seq_lens: List[Int], cache_lens: List[Int], ctx: DeviceContext) raises:
    """Expand selected pools to token positions and check every column.

    Parameters:
        kpool: Tokens per pool.
        pool_topk: Selected pools per token.
        always_select_tail: Whether the incomplete trailing pool is appended.

    Args:
        seq_lens: New tokens per request.
        cache_lens: Cached prefix length per request.
        ctx: Device context.
    """
    comptime tail_width = (kpool - 1) if always_select_tail else 0
    comptime out_width = pool_topk * kpool + tail_width

    var batch_size = len(seq_lens)
    var total_seq_len = 0
    for i in range(batch_size):
        total_seq_len += seq_lens[i]
    print(
        "expand_topk: kpool=",
        kpool,
        " pool_topk=",
        pool_topk,
        " tail=",
        tail_width,
        " tokens=",
        total_seq_len,
    )

    var pool_host = ctx.enqueue_create_host_buffer[.int32](
        total_seq_len * pool_topk
    )
    var iro_host = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var clen_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var out_host = ctx.enqueue_create_host_buffer[.int32](
        total_seq_len * out_width
    )
    ctx.synchronize()

    var off = 0
    for b in range(batch_size):
        iro_host[b] = UInt32(off)
        clen_host[b] = UInt32(cache_lens[b])
        off += seq_lens[b]
    iro_host[batch_size] = UInt32(off)

    # A -1 in every row, so the sentinel path runs on every token.
    for t in range(total_seq_len):
        for j in range(pool_topk):
            if j == pool_topk - 1:
                pool_host[t * pool_topk + j] = Int32(-1)
            else:
                pool_host[t * pool_topk + j] = Int32((t * 7 + j * 3) % 32)

    var pool_dev = ctx.enqueue_create_buffer[.int32](total_seq_len * pool_topk)
    var iro_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var clen_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    var out_dev = ctx.enqueue_create_buffer[.int32](total_seq_len * out_width)
    ctx.enqueue_copy(pool_dev, pool_host)
    ctx.enqueue_copy(iro_dev, iro_host)
    ctx.enqueue_copy(clen_dev, clen_host)
    ctx.synchronize()

    var out_tile = TileTensor(out_dev, row_major(total_seq_len, out_width))
    var pool_tile = TileTensor(pool_dev, row_major(total_seq_len, pool_topk))
    var iro_tile = TileTensor(iro_dev, row_major(batch_size + 1))
    var clen_tile = TileTensor(clen_dev, row_major(batch_size))

    comptime kernel = kpool_expand_topk_kernel[
        out_tile.LayoutType,
        out_tile.origin,
        type_of(pool_tile.as_immut()).LayoutType,
        ImmOrigin(pool_tile.origin),
        type_of(iro_tile.as_immut()).LayoutType,
        ImmOrigin(iro_tile.origin),
        type_of(clen_tile.as_immut()).LayoutType,
        kpool,
        pool_topk,
        always_select_tail,
    ]
    ctx.enqueue_function[kernel](
        out_tile,
        pool_tile.as_immut(),
        iro_tile.as_immut(),
        clen_tile.as_immut(),
        Int32(total_seq_len),
        grid_dim=(total_seq_len, 1, 1),
        block_dim=(128, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(out_host, out_dev)
    ctx.synchronize()

    for b in range(batch_size):
        for local in range(seq_lens[b]):
            var t = Int(iro_host[b]) + local
            var visible = cache_lens[b] + local + 1
            var tail_count = visible % kpool
            var tail_start = visible - tail_count

            for col in range(out_width):
                var got = Int(out_host[t * out_width + col])
                var want: Int
                if col < pool_topk * kpool:
                    var pid = Int(pool_host[t * pool_topk + col // kpool])
                    # An unselected pool stays unselected in all of its slots.
                    want = -1 if pid < 0 else pid * kpool + col % kpool
                else:
                    var tail_idx = col - pool_topk * kpool
                    want = (
                        tail_start + tail_idx if tail_idx < tail_count else -1
                    )
                assert_equal(
                    got,
                    want,
                    String("token ", t, " column ", col, " wrong"),
                )

            # The tail must never reach past the query's own position.
            comptime if always_select_tail:
                for tail_idx in range(tail_width):
                    var v = Int(
                        out_host[t * out_width + pool_topk * kpool + tail_idx]
                    )
                    assert_true(
                        v < visible,
                        String("token ", t, " tail ", v, " past position"),
                    )
                assert_equal(
                    tail_count,
                    visible - tail_start,
                    "tail arithmetic disagrees with itself",
                )

    print("  every column matches over", total_seq_len, "tokens")
    _ = pool_dev
    _ = iro_dev
    _ = clen_dev
    _ = out_dev


def _run_ring[
    head_dim: Int, kpool: Int, next_n: Int
](
    k_host: HostBuffer[.bfloat16],
    gate_host: HostBuffer[.bfloat16],
    ape_t: TileTensor[mut=False, .float32, ...],
    num_requests: Int,
    seq_len: Int,
    out_host: HostBuffer[.bfloat16],
    ids_host: HostBuffer[.int32],
    ctx: DeviceContext,
) raises -> Int:
    """Drives the ring `seq_len // next_n` times, collecting the closed pools.
    """
    comptime max_closed = (next_n + kpool - 1) // kpool
    var steps = seq_len // next_n

    var tail_d = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * 2 * kpool * head_dim
    )
    ctx.enqueue_memset(tail_d, 0)
    var k_d = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * next_n * head_dim
    )
    var g_d = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * next_n * head_dim
    )
    var pos_d = ctx.enqueue_create_buffer[.int32](num_requests * next_n)
    var slot_d = ctx.enqueue_create_buffer[.uint32](num_requests)
    var pooled_d = ctx.enqueue_create_buffer[.bfloat16](
        num_requests * max_closed * head_dim
    )
    var closed_d = ctx.enqueue_create_buffer[.int32](num_requests * max_closed)

    var k_s = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * next_n * head_dim
    )
    var g_s = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * next_n * head_dim
    )
    var pos_s = ctx.enqueue_create_host_buffer[.int32](num_requests * next_n)
    var slot_s = ctx.enqueue_create_host_buffer[.uint32](num_requests)
    var pooled_s = ctx.enqueue_create_host_buffer[.bfloat16](
        num_requests * max_closed * head_dim
    )
    var closed_s = ctx.enqueue_create_host_buffer[.int32](
        num_requests * max_closed
    )
    ctx.synchronize()

    var tail_t = TileTensor(
        tail_d, row_major(num_requests, 2 * kpool, head_dim)
    )
    var k_t = TileTensor(k_d, row_major(num_requests * next_n, head_dim))
    var g_t = TileTensor(g_d, row_major(num_requests * next_n, head_dim))
    var pos_t = TileTensor(pos_d, row_major(num_requests * next_n))
    for r in range(num_requests):
        slot_s[r] = UInt32(r)
    ctx.enqueue_copy(slot_d, slot_s)
    var slot_t = TileTensor(slot_d, row_major(num_requests))
    var pooled_t = TileTensor(
        pooled_d, row_major(num_requests * max_closed, head_dim)
    )
    var closed_t = TileTensor(closed_d, row_major(num_requests * max_closed))

    comptime kern = kpool_tail_update_kernel[
        .bfloat16,
        tail_t.LayoutType,
        tail_t.origin,
        pooled_t.LayoutType,
        pooled_t.origin,
        closed_t.LayoutType,
        closed_t.origin,
        type_of(k_t.as_immut()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(g_t.as_immut()).LayoutType,
        ImmOrigin(g_t.origin),
        type_of(ape_t).LayoutType,
        ImmOrigin(ape_t.origin),
        type_of(pos_t.as_immut()).LayoutType,
        ImmOrigin(pos_t.origin),
        type_of(slot_t.as_immut()).LayoutType,
        ImmOrigin(slot_t.origin),
        head_dim,
        kpool,
        next_n,
    ]

    var kept = 0
    for step in range(steps):
        for r in range(num_requests):
            for t in range(next_n):
                var pos = step * next_n + t
                pos_s[r * next_n + t] = Int32(pos)
                var src = (r * seq_len + pos) * head_dim
                var dst = (r * next_n + t) * head_dim
                for c in range(head_dim):
                    k_s[dst + c] = k_host[src + c]
                    g_s[dst + c] = gate_host[src + c]
        ctx.enqueue_copy(k_d, k_s)
        ctx.enqueue_copy(g_d, g_s)
        ctx.enqueue_copy(pos_d, pos_s)
        ctx.enqueue_function[kern](
            tail_t,
            pooled_t,
            closed_t,
            k_t.as_immut(),
            g_t.as_immut(),
            ape_t,
            pos_t.as_immut(),
            slot_t.as_immut(),
            Int32(num_requests),
            grid_dim=(num_requests, 1, 1),
            block_dim=(head_dim, 1, 1),
        )
        ctx.synchronize()
        ctx.enqueue_copy(closed_s, closed_d)
        ctx.enqueue_copy(pooled_s, pooled_d)
        ctx.synchronize()
        for r in range(num_requests):
            for j in range(max_closed):
                var cid = Int(closed_s[r * max_closed + j])
                if cid < 0:
                    continue
                # Index by request and pool id. Arrival order differs between
                # the arms and says nothing about the pooled keys.
                var dst_row = r * (seq_len // kpool) + cid
                ids_host[dst_row] = Int32(cid)
                var src = (r * max_closed + j) * head_dim
                for c in range(head_dim):
                    out_host[dst_row * head_dim + c] = pooled_s[src + c]
                kept += 1

    _ = tail_d
    _ = k_d
    _ = g_d
    _ = pos_d
    _ = slot_d
    _ = pooled_d
    _ = closed_d
    return kept


def test_spec_step_matches_single_steps[
    head_dim: Int, kpool: Int, next_n: Int
](num_requests: Int, seq_len: Int, ctx: DeviceContext) raises:
    """A speculative step must equal the same tokens fed one at a time.

    A speculative step appends `next_n` tokens per call instead of one, so
    several pools can complete in a single invocation and the ring has to be
    walked in position order. Feeding the identical token stream both ways must
    produce identical pooled keys and identical closed pool ids, bit for bit,
    since the same arithmetic runs either way.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.
        next_n: Tokens appended per request per call.

    Args:
        num_requests: Requests decoded together.
        seq_len: Total tokens per request; must divide by `next_n`.
        ctx: Device context.
    """
    var steps = seq_len // next_n
    assert_equal(steps * next_n, seq_len, "seq_len must divide by next_n")
    print(
        "spec-vs-single: requests=",
        num_requests,
        " next_n=",
        next_n,
        " seq_len=",
        seq_len,
    )

    var total = num_requests * seq_len
    var k_host = ctx.enqueue_create_host_buffer[.bfloat16](total * head_dim)
    var gate_host = ctx.enqueue_create_host_buffer[.bfloat16](total * head_dim)
    var ape_host = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    ctx.synchronize()
    rand(k_host.unsafe_ptr(), total * head_dim)
    rand(gate_host.unsafe_ptr(), total * head_dim)
    rand(ape_host.unsafe_ptr(), kpool * head_dim)
    for i in range(total * head_dim):
        gate_host[i] = (gate_host[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_host[i] = (ape_host[i] - 0.5) * 4.0

    var ape_dev = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    ctx.enqueue_copy(ape_dev, ape_host)
    ctx.synchronize()
    var ape_t = TileTensor(ape_dev, row_major(kpool, head_dim)).as_immut()

    var wide_out = ctx.enqueue_create_host_buffer[.bfloat16](total * head_dim)
    var single_out = ctx.enqueue_create_host_buffer[.bfloat16](total * head_dim)
    var wide_ids = ctx.enqueue_create_host_buffer[.int32](total)
    var single_ids = ctx.enqueue_create_host_buffer[.int32](total)
    ctx.synchronize()

    var wide_n = _run_ring[head_dim, kpool, next_n](
        k_host, gate_host, ape_t, num_requests, seq_len, wide_out, wide_ids, ctx
    )
    var single_n = _run_ring[head_dim, kpool, 1](
        k_host,
        gate_host,
        ape_t,
        num_requests,
        seq_len,
        single_out,
        single_ids,
        ctx,
    )

    assert_equal(
        wide_n, single_n, "the two arms closed a different number of pools"
    )
    assert_true(wide_n > 0, "no pool closed; the shape proves nothing")
    var slots = num_requests * (seq_len // kpool)
    assert_equal(wide_n, slots, "not every pool of every request closed")
    for i in range(slots):
        assert_equal(
            wide_ids[i],
            single_ids[i],
            String("closed pool id at slot ", i, " differs between arms"),
        )
    for i in range(slots * head_dim):
        assert_equal(
            wide_out[i],
            single_out[i],
            String("pooled key element ", i, " differs between arms"),
        )
    print("  ", wide_n, "pools identical between a wide step and single steps")
    _ = ape_dev


def _ring_after[
    head_dim: Int, kpool: Int, next_n: Int
](
    k_host: HostBuffer[.bfloat16],
    gate_host: HostBuffer[.bfloat16],
    ape_t: TileTensor[mut=False, .float32, ...],
    n_valid: Int,
    ring_out: HostBuffer[.bfloat16],
    ctx: DeviceContext,
) raises:
    """Runs one call over `next_n` slots and copies the resulting ring out.

    The first `n_valid` slots carry real positions; the rest carry -1, which is
    what a padded entry in a speculative batch holds.
    """
    comptime max_closed = (next_n + kpool - 1) // kpool
    var ring_elems = 2 * kpool * head_dim
    var ring_d = ctx.enqueue_create_buffer[.bfloat16](ring_elems)
    ctx.enqueue_memset(ring_d, 0)
    var k_d = ctx.enqueue_create_buffer[.bfloat16](next_n * head_dim)
    var g_d = ctx.enqueue_create_buffer[.bfloat16](next_n * head_dim)
    var pos_d = ctx.enqueue_create_buffer[.int32](next_n)
    var slot_d = ctx.enqueue_create_buffer[.uint32](1)
    var pooled_d = ctx.enqueue_create_buffer[.bfloat16](max_closed * head_dim)
    var closed_d = ctx.enqueue_create_buffer[.int32](max_closed)
    var pos_h = ctx.enqueue_create_host_buffer[.int32](next_n)
    var slot_h = ctx.enqueue_create_host_buffer[.uint32](1)
    ctx.synchronize()

    for t in range(next_n):
        pos_h[t] = Int32(t) if t < n_valid else Int32(-1)
    slot_h[0] = UInt32(0)
    # The callers' host buffers cover `next_n` slots; the HostBuffer-based
    # enqueue_copy is source-sized, so a shorter helper call over a wider
    # caller's buffer would write past `k_d`/`g_d`. Copy through the
    # destination-sized pointer overload instead.
    ctx.enqueue_copy(k_d, k_host.unsafe_ptr())
    ctx.enqueue_copy(g_d, gate_host.unsafe_ptr())
    ctx.enqueue_copy(pos_d, pos_h)
    ctx.enqueue_copy(slot_d, slot_h)

    var ring_t = TileTensor(ring_d, row_major(1, 2 * kpool, head_dim))
    var k_t = TileTensor(k_d, row_major(next_n, head_dim))
    var g_t = TileTensor(g_d, row_major(next_n, head_dim))
    var pos_t = TileTensor(pos_d, row_major(next_n))
    var slot_t = TileTensor(slot_d, row_major(1))
    var pooled_t = TileTensor(pooled_d, row_major(max_closed, head_dim))
    var closed_t = TileTensor(closed_d, row_major(max_closed))

    comptime kern = kpool_tail_update_kernel[
        .bfloat16,
        ring_t.LayoutType,
        ring_t.origin,
        pooled_t.LayoutType,
        pooled_t.origin,
        closed_t.LayoutType,
        closed_t.origin,
        type_of(k_t.as_immut()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(g_t.as_immut()).LayoutType,
        ImmOrigin(g_t.origin),
        type_of(ape_t).LayoutType,
        ImmOrigin(ape_t.origin),
        type_of(pos_t.as_immut()).LayoutType,
        ImmOrigin(pos_t.origin),
        type_of(slot_t.as_immut()).LayoutType,
        ImmOrigin(slot_t.origin),
        head_dim,
        kpool,
        next_n,
    ]
    ctx.enqueue_function[kern](
        ring_t,
        pooled_t,
        closed_t,
        k_t.as_immut(),
        g_t.as_immut(),
        ape_t,
        pos_t.as_immut(),
        slot_t.as_immut(),
        Int32(1),
        grid_dim=(1, 1, 1),
        block_dim=(head_dim, 1, 1),
    )
    ctx.synchronize()
    ctx.enqueue_copy(ring_out, ring_d)
    ctx.synchronize()

    _ = ring_d
    _ = k_d
    _ = g_d
    _ = pos_d
    _ = slot_d
    _ = pooled_d
    _ = closed_d


def test_padded_slot_leaves_no_trace[
    head_dim: Int, kpool: Int, next_n: Int
](ctx: DeviceContext) raises:
    """A padded slot must leave the ring exactly where a shorter call does.

    A speculative batch pads its unused slots with a negative position. The
    kernel skips those, so a call over `next_n` slots whose last one is padded
    has to leave the ring byte-identical to a call over just the real tokens.
    Skipping the slot only partway -- stashing it, or folding it into a
    neighbour -- would show up here and nowhere else.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.
        next_n: Slots the wide call covers.

    Args:
        ctx: Device context.
    """
    comptime real = next_n - 1
    print("padded slot: next_n=", next_n, " real=", real, " kpool=", kpool)

    var k_h = ctx.enqueue_create_host_buffer[.bfloat16](next_n * head_dim)
    var g_h = ctx.enqueue_create_host_buffer[.bfloat16](next_n * head_dim)
    var ape_h = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    ctx.synchronize()
    rand(k_h.unsafe_ptr(), next_n * head_dim)
    rand(g_h.unsafe_ptr(), next_n * head_dim)
    rand(ape_h.unsafe_ptr(), kpool * head_dim)
    for i in range(next_n * head_dim):
        g_h[i] = (g_h[i] - 0.5) * 8.0
    for i in range(kpool * head_dim):
        ape_h[i] = (ape_h[i] - 0.5) * 4.0

    var ape_d = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    ctx.enqueue_copy(ape_d, ape_h)
    ctx.synchronize()
    var ape_t = TileTensor(ape_d, row_major(kpool, head_dim)).as_immut()

    var ring_elems = 2 * kpool * head_dim
    var padded = ctx.enqueue_create_host_buffer[.bfloat16](ring_elems)
    var shorter = ctx.enqueue_create_host_buffer[.bfloat16](ring_elems)
    ctx.synchronize()

    _ring_after[head_dim, kpool, next_n](k_h, g_h, ape_t, real, padded, ctx)
    _ring_after[head_dim, kpool, real](k_h, g_h, ape_t, real, shorter, ctx)

    for i in range(ring_elems):
        assert_equal(
            padded[i],
            shorter[i],
            String("a padded slot changed ring element ", i),
        )
    print("  the padded slot left the ring untouched")
    _ = ape_d


def _seed_chunk[
    head_dim: Int, kpool: Int
](
    k_d: DeviceBuffer[.bfloat16],
    g_d: DeviceBuffer[.bfloat16],
    ring_t: TileTensor[mut=True, .bfloat16, ...],
    row_start: Int,
    n: Int,
    cache_len: Int,
    ctx: DeviceContext,
) raises:
    """Seeds one prefill chunk's trailing tokens into slot 0 of `ring_t`."""
    var iro_d = ctx.enqueue_create_buffer[.uint32](2)
    var clen_d = ctx.enqueue_create_buffer[.uint32](1)
    var slot_d = ctx.enqueue_create_buffer[.uint32](1)
    var iro_h = ctx.enqueue_create_host_buffer[.uint32](2)
    var clen_h = ctx.enqueue_create_host_buffer[.uint32](1)
    var slot_h = ctx.enqueue_create_host_buffer[.uint32](1)
    ctx.synchronize()
    iro_h[0] = UInt32(row_start)
    iro_h[1] = UInt32(row_start + n)
    clen_h[0] = UInt32(cache_len)
    slot_h[0] = 0
    ctx.enqueue_copy(iro_d, iro_h)
    ctx.enqueue_copy(clen_d, clen_h)
    ctx.enqueue_copy(slot_d, slot_h)

    var k_t = TileTensor(k_d, row_major(row_start + n, head_dim))
    var g_t = TileTensor(g_d, row_major(row_start + n, head_dim))
    var iro_t = TileTensor(iro_d, row_major(2))
    var clen_t = TileTensor(clen_d, row_major(1))
    var slot_t = TileTensor(slot_d, row_major(1))

    comptime kern = kpool_seed_tail_kernel[
        .bfloat16,
        ring_t.LayoutType,
        ring_t.origin,
        type_of(k_t.as_immut()).LayoutType,
        ImmOrigin(k_t.origin),
        type_of(g_t.as_immut()).LayoutType,
        ImmOrigin(g_t.origin),
        type_of(iro_t.as_immut()).LayoutType,
        ImmOrigin(iro_t.origin),
        type_of(clen_t.as_immut()).LayoutType,
        type_of(slot_t.as_immut()).LayoutType,
        ImmOrigin(slot_t.origin),
        head_dim,
        kpool,
    ]
    ctx.enqueue_function[kern](
        ring_t,
        k_t.as_immut(),
        g_t.as_immut(),
        iro_t.as_immut(),
        clen_t.as_immut(),
        slot_t.as_immut(),
        Int32(1),
        grid_dim=(1, 1, 1),
        block_dim=(head_dim, 1, 1),
    )
    ctx.synchronize()
    _ = iro_d
    _ = clen_d
    _ = slot_d


def test_seed_matches_feeding_the_prefix[
    head_dim: Int, kpool: Int, prefill_len: Int
](ctx: DeviceContext) raises:
    """Seeding a prefill chunk must leave the ring where decoding it would.

    Compression writes only whole pools, so the trailing tokens reach the ring
    through the seed kernel instead. The ring it leaves has to match what the
    decode path produces from the same tokens, or the first pool to close
    during decode pools against whatever the ring happened to hold. The second
    arm also splits the prefill into two chunks, which must land in the same
    place as one.

    Parameters:
        head_dim: Channels per key.
        kpool: Tokens per pool.
        prefill_len: Tokens in the prefill; not a multiple of `kpool`.

    Args:
        ctx: Device context.
    """
    print("seed vs decode: prefill_len=", prefill_len, " kpool=", kpool)
    var ring_elems = 2 * kpool * head_dim

    var k_h = ctx.enqueue_create_host_buffer[.bfloat16](prefill_len * head_dim)
    var g_h = ctx.enqueue_create_host_buffer[.bfloat16](prefill_len * head_dim)
    var ape_h = ctx.enqueue_create_host_buffer[.float32](kpool * head_dim)
    ctx.synchronize()
    rand(k_h.unsafe_ptr(), prefill_len * head_dim)
    rand(g_h.unsafe_ptr(), prefill_len * head_dim)
    rand(ape_h.unsafe_ptr(), kpool * head_dim)
    for i in range(kpool * head_dim):
        ape_h[i] = (ape_h[i] - 0.5) * 4.0

    var ape_d = ctx.enqueue_create_buffer[.float32](kpool * head_dim)
    var k_d = ctx.enqueue_create_buffer[.bfloat16](prefill_len * head_dim)
    var g_d = ctx.enqueue_create_buffer[.bfloat16](prefill_len * head_dim)
    ctx.enqueue_copy(ape_d, ape_h)
    ctx.enqueue_copy(k_d, k_h)
    ctx.enqueue_copy(g_d, g_h)
    ctx.synchronize()
    var ape_t = TileTensor(ape_d, row_major(kpool, head_dim)).as_immut()

    var seeded_d = ctx.enqueue_create_buffer[.bfloat16](ring_elems)
    var chunked_d = ctx.enqueue_create_buffer[.bfloat16](ring_elems)
    ctx.enqueue_memset(seeded_d, 0)
    ctx.enqueue_memset(chunked_d, 0)
    var seeded_h = ctx.enqueue_create_host_buffer[.bfloat16](ring_elems)
    var chunked_h = ctx.enqueue_create_host_buffer[.bfloat16](ring_elems)
    var decoded_h = ctx.enqueue_create_host_buffer[.bfloat16](ring_elems)
    ctx.synchronize()
    var seeded_t = TileTensor(seeded_d, row_major(1, 2 * kpool, head_dim))
    var chunked_t = TileTensor(chunked_d, row_major(1, 2 * kpool, head_dim))

    # One chunk.
    _seed_chunk[head_dim, kpool](k_d, g_d, seeded_t, 0, prefill_len, 0, ctx)
    ctx.enqueue_copy(seeded_h, seeded_d)

    # The same prefill, split in two.
    comptime first = kpool
    _seed_chunk[head_dim, kpool](k_d, g_d, chunked_t, 0, first, 0, ctx)
    _seed_chunk[head_dim, kpool](
        k_d, g_d, chunked_t, first, prefill_len - first, first, ctx
    )
    ctx.enqueue_copy(chunked_h, chunked_d)

    # The decode path over the same tokens.
    _ring_after[head_dim, kpool, prefill_len](
        k_h, g_h, ape_t, prefill_len, decoded_h, ctx
    )
    ctx.synchronize()

    # Only the in-progress pool's members that have already arrived need to be
    # in the ring. Decode overwrites the remaining slots before the pool
    # closes.
    var live_slots = prefill_len % kpool
    for slot in range(live_slots):
        for c in range(head_dim):
            var key = slot * head_dim + c
            var gate_el = (kpool + slot) * head_dim + c
            assert_equal(
                seeded_h[key],
                decoded_h[key],
                String("seeded key differs at slot ", slot, " channel ", c),
            )
            assert_equal(
                seeded_h[gate_el],
                decoded_h[gate_el],
                String("seeded gate differs at slot ", slot, " channel ", c),
            )
    assert_true(live_slots > 0, "prefill_len must not be a multiple of kpool")

    # Two chunks must land exactly where one does.
    for i in range(ring_elems):
        assert_equal(
            chunked_h[i],
            seeded_h[i],
            String("chunked seeding differs at element ", i),
        )
    print("  the in-progress pool's", live_slots, "seeded slots match")
    _ = ape_d
    _ = k_d
    _ = g_d
    _ = seeded_d
    _ = chunked_d


def main() raises:
    with DeviceContext() as ctx:
        test_kpool_one_is_identity[head_dim=128](ctx)

        # GLM-5.3-Flash geometry.
        test_kpool_compress[head_dim=128, kpool=4](seq_lens=[8, 4, 12], ctx=ctx)
        # Lengths not divisible by kpool: the trailing tokens must be ignored
        # rather than folded into a short pool.
        test_kpool_compress[head_dim=128, kpool=4](
            seq_lens=[7, 5, 3, 1], ctx=ctx
        )
        # Cached prefixes at every phase: a request whose prefix ends
        # mid-pool opens this call partway through a pool it cannot build.
        test_kpool_compress[head_dim=128, kpool=4](
            seq_lens=[9, 9, 9, 9], ctx=ctx, cache_lens=[0, 1, 2, 3]
        )
        # An over-sized grid: the kernel bounds itself from the offsets, so
        # the surplus blocks must not write.
        test_kpool_compress[head_dim=128, kpool=4](
            seq_lens=[8, 4], ctx=ctx, extra_blocks=3
        )
        # A single request, and a pool count above one block per request.
        test_kpool_compress[head_dim=128, kpool=4](seq_lens=[64], ctx=ctx)

        # Several requests decoding in lockstep: a request closing its pool
        # must not disturb any other request's ring.
        test_decode_writer_matches_prefill_writer[head_dim=128, kpool=4](
            num_requests=3, seq_len=8, ctx=ctx
        )
        # A length that leaves a partial pool open at the end.
        test_decode_writer_matches_prefill_writer[head_dim=128, kpool=4](
            num_requests=2, seq_len=10, ctx=ctx
        )

        test_rings_are_per_request[head_dim=128, kpool=4](ctx)

        # kpool=1 closes a pool on every token, so the ring never carries a
        # member between steps and each pooled key is its single member.
        test_decode_writer_matches_prefill_writer[head_dim=128, kpool=1](
            num_requests=2, seq_len=5, ctx=ctx
        )

        # A serving batch reorders between steps; the ring must follow the
        # request, not the row.
        test_ring_survives_a_batch_reorder[head_dim=128, kpool=4](
            seq_len=8, ctx=ctx
        )
        # Ragged, with cached prefixes that put each request's tail at a
        # different phase: cache_len % kpool of 0, 1, 2 and 3.
        test_expand_topk[kpool=4, pool_topk=8, always_select_tail=True](
            seq_lens=[3, 5, 1, 4], cache_lens=[0, 9, 6, 7], ctx=ctx
        )
        # Without the tail the width is exactly the expanded selection.
        test_expand_topk[kpool=4, pool_topk=8, always_select_tail=False](
            seq_lens=[2, 6], cache_lens=[4, 11], ctx=ctx
        )
        # kpool=1 leaves the selection untouched and adds no tail columns.
        test_expand_topk[kpool=1, pool_topk=16, always_select_tail=True](
            seq_lens=[4, 2], cache_lens=[0, 5], ctx=ctx
        )
        # GLM-5.3-Flash's real width: 512 pools expand to 2048 positions plus
        # a 3-wide tail, so each row takes several strided passes.
        test_expand_topk[kpool=4, pool_topk=512, always_select_tail=True](
            seq_lens=[3, 2], cache_lens=[8000, 4097], ctx=ctx
        )

        # Speculative steps: several tokens per request per call.
        test_spec_step_matches_single_steps[head_dim=128, kpool=4, next_n=2](
            num_requests=2, seq_len=8, ctx=ctx
        )
        # `next_n` above `kpool`, so more than one pool closes in one call.
        test_spec_step_matches_single_steps[head_dim=128, kpool=4, next_n=8](
            num_requests=2, seq_len=16, ctx=ctx
        )

        # A speculative batch pads unused slots with a negative position.
        test_padded_slot_leaves_no_trace[head_dim=128, kpool=4, next_n=4](ctx)

        # Prefill's trailing tokens must reach the ring for decode to pool
        # against them.
        test_seed_matches_feeding_the_prefix[
            head_dim=128, kpool=4, prefill_len=6
        ](ctx)

        print("\nAll tests passed!")
