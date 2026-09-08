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
"""K-pool compression for the DSA indexer.

A k-pooled indexer stores one candidate key per `kpool` consecutive tokens
instead of one per token. A pooled key is a weighted average of its members,
where the weights come from a softmax over a gate score plus a learned
within-pool position embedding:

    logits[m, c]    = gate[member m, c] + ape[m, c]
    weights[:, c]   = softmax over m         # independently per channel c
    pooled[p, c]    = sum_m weights[m, c] * k[member m, c]

The softmax runs per channel, not per member. One weight per member would be a
different function.

Pool `p` covers absolute positions `[p * kpool, (p + 1) * kpool)` of one
request. Only pools whose members all arrive in the same call are written.
"""

from std.math import exp, max

from layout import TensorLayout, TileTensor

from max.gpu import block_dim, block_idx, thread_idx


@always_inline
def pool_channel[
    kpool: Int
](logits: Array[Float32, kpool], vals: Array[Float32, kpool]) -> Float32:
    """One pooled channel: softmax over `logits`, weighted sum of `vals`.

    Shared by the prefill and decode writers so the two cannot drift.
    """
    var m_max = -Float32.MAX
    for m in range(kpool):
        m_max = max(m_max, logits[m])

    var acc = Float32(0)
    var denom = Float32(0)
    for m in range(kpool):
        var w = exp(logits[m] - m_max)
        denom += w
        acc += w * vals[m]
    return acc / denom


@always_inline
def _batch_of_pool_row(
    pool_row_offsets: TileTensor[mut=False, .uint32, ...],
    batch_size: Int,
    pool_row: Int,
) -> Int:
    """Batch owning output row `pool_row`."""
    for b in range(batch_size):
        if pool_row < Int(pool_row_offsets.raw_load(b + 1)):
            return b
    return batch_size - 1


@__name(t"mla_kpool_compress_{kpool}_{head_dim}")
def kpool_compress_kernel[
    dtype: DType,
    KLayoutType: TensorLayout,
    k_origin: ImmOrigin,
    GateLayoutType: TensorLayout,
    gate_origin: ImmOrigin,
    ApeLayoutType: TensorLayout,
    ape_origin: ImmOrigin,
    IROLayoutType: TensorLayout,
    iro_origin: ImmOrigin,
    PROLayoutType: TensorLayout,
    pro_origin: ImmOrigin,
    CacheLenLayoutType: TensorLayout,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    head_dim: Int,
    kpool: Int,
](
    pooled: TileTensor[dtype, OutLayoutType, out_origin],
    k: TileTensor[mut=False, dtype, KLayoutType, k_origin],
    gate: TileTensor[mut=False, dtype, GateLayoutType, gate_origin],
    ape: TileTensor[mut=False, .float32, ApeLayoutType, ape_origin],
    input_row_offsets: TileTensor[
        mut=False, .uint32, IROLayoutType, iro_origin
    ],
    pool_row_offsets: TileTensor[mut=False, .uint32, PROLayoutType, pro_origin],
    cache_lengths: TileTensor[
        mut=False, .uint32, CacheLenLayoutType, ImmutAnyOrigin
    ],
):
    """Builds one pooled key per block; one thread per channel.

    Parameters:
        dtype: Element type of `k`, `gate` and `pooled`.
        KLayoutType: Layout of `k`.
        k_origin: Origin of `k`.
        GateLayoutType: Layout of `gate`.
        gate_origin: Origin of `gate`.
        ApeLayoutType: Layout of `ape`.
        ape_origin: Origin of `ape`.
        IROLayoutType: Layout of `input_row_offsets`.
        iro_origin: Origin of `input_row_offsets`.
        PROLayoutType: Layout of `pool_row_offsets`.
        pro_origin: Origin of `pool_row_offsets`.
        CacheLenLayoutType: Layout of `cache_lengths`.
        OutLayoutType: Layout of `pooled`.
        out_origin: Origin of `pooled`.
        head_dim: Channels per key; also the block width.
        kpool: Tokens per pool.

    Args:
        pooled: Output `[total_pools, head_dim]`, where `total_pools` is the
            last entry of `pool_row_offsets`.
        k: Layer-normed indexer keys, `[total_tokens, head_dim]`.
        gate: Per-token gate scores, `[total_tokens, head_dim]`.
        ape: Within-pool position embedding, `[kpool, head_dim]`, f32.
        input_row_offsets: Token row offsets per request, `[batch_size + 1]`.
        pool_row_offsets: Output row offsets per request, `[batch_size + 1]`.
            Request `b` owns output rows `[pool_row_offsets[b],
            pool_row_offsets[b + 1])`, one per pool built here.
        cache_lengths: Cached-prefix length per request, `[batch_size]`. A
            pool covers absolute positions, so this is what places the call's
            tokens on the pool grid.
    """
    var pool_row = block_idx.x
    # The last offset is the exact pool count, so an over-sized grid is
    # harmless.
    var batch_size = Int(pool_row_offsets.dim[0]()) - 1
    if pool_row >= Int(pool_row_offsets.raw_load(batch_size)):
        return

    var c = thread_idx.x
    if c >= head_dim:
        return

    var b = _batch_of_pool_row(pool_row_offsets, batch_size, pool_row)

    var local_pool = pool_row - Int(pool_row_offsets.raw_load(b))
    # A cached prefix ending mid-pool means this call opens partway through a
    # pool it cannot build. Skip those leading tokens; the tail ring carries
    # that pool.
    var cache_len = Int(cache_lengths[b])
    var align = (kpool - cache_len % kpool) % kpool
    var first_row = (
        Int(input_row_offsets.raw_load(b)) + align + local_pool * kpool
    )

    # Every member of a written pool exists, so nothing needs masking.
    var logits = Array[Float32, kpool]()
    var vals = Array[Float32, kpool]()
    for m in range(kpool):
        logits[m] = gate.raw_load((first_row + m) * head_dim + c).cast[
            .float32
        ]() + ape.raw_load(m * head_dim + c)
        vals[m] = k.raw_load((first_row + m) * head_dim + c).cast[.float32]()

    pooled.raw_store(
        pool_row * head_dim + c, pool_channel[kpool](logits, vals).cast[dtype]()
    )


@__name(t"mla_kpool_tail_update_{kpool}_{head_dim}")
def kpool_tail_update_kernel[
    dtype: DType,
    TailLayoutType: TensorLayout,
    tail_origin: MutOrigin,
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    ClosedLayoutType: TensorLayout,
    closed_origin: MutOrigin,
    KLayoutType: TensorLayout,
    k_origin: ImmOrigin,
    GateLayoutType: TensorLayout,
    gate_origin: ImmOrigin,
    ApeLayoutType: TensorLayout,
    ape_origin: ImmOrigin,
    PosLayoutType: TensorLayout,
    pos_origin: ImmOrigin,
    SlotLayoutType: TensorLayout,
    slot_origin: ImmOrigin,
    head_dim: Int,
    kpool: Int,
    next_n: Int = 1,
](
    tail: TileTensor[dtype, TailLayoutType, tail_origin],
    pooled: TileTensor[dtype, OutLayoutType, out_origin],
    closed_pool: TileTensor[.int32, ClosedLayoutType, closed_origin],
    k: TileTensor[mut=False, dtype, KLayoutType, k_origin],
    gate: TileTensor[mut=False, dtype, GateLayoutType, gate_origin],
    ape: TileTensor[mut=False, .float32, ApeLayoutType, ape_origin],
    positions: TileTensor[mut=False, .int32, PosLayoutType, pos_origin],
    slot_idx: TileTensor[mut=False, .uint32, SlotLayoutType, slot_origin],
    num_requests: Int32,
):
    """Stashes a request's new tokens, and closes each pool as it fills.

    A decoded token cannot be pooled on arrival, because its pool-mates arrived
    on earlier steps and have left the batch. Each request keeps its
    in-progress pool in `tail`, a ring of `kpool` slots addressed by
    `position % kpool`.

    A speculative step appends `next_n` tokens at once, so several pools can
    close in one call.

    The ring is indexed by `slot_idx[r]`, not by `r`. A batch reorders between
    steps, so row `r` is not always the same request.

    Every real token stashes, whether or not it closes a pool.

    Rejected speculative tokens are the caller's problem. The ring holds no
    pointer to rewind, so a rejected token that has already stashed stays.

    Parameters:
        dtype: Element type of `tail`, `k`, `gate` and `pooled`.
        TailLayoutType: Layout of `tail`.
        tail_origin: Origin of `tail`.
        OutLayoutType: Layout of `pooled`.
        out_origin: Origin of `pooled`.
        ClosedLayoutType: Layout of `closed_pool`.
        closed_origin: Origin of `closed_pool`.
        KLayoutType: Layout of `k`.
        k_origin: Origin of `k`.
        GateLayoutType: Layout of `gate`.
        gate_origin: Origin of `gate`.
        ApeLayoutType: Layout of `ape`.
        ape_origin: Origin of `ape`.
        PosLayoutType: Layout of `positions`.
        pos_origin: Origin of `positions`.
        SlotLayoutType: Layout of `slot_idx`.
        slot_origin: Origin of `slot_idx`.
        head_dim: Channels per key; also the block width.
        kpool: Tokens per pool.
        next_n: Tokens appended per request per call.

    Args:
        tail: Per-slot ring, `[max_slots, 2, kpool, head_dim]`, sized by the
            engine's concurrent-request capacity. Index 0 holds keys, index 1
            holds gate scores. Only slot `slot_idx[r]` is touched for row `r`.
            Persists across steps.
        pooled: Output `[num_requests, ceil(next_n / kpool), head_dim]`,
            meaningful only where `closed_pool` is non-negative.
        closed_pool: Output `[num_requests, ceil(next_n / kpool)]`. The pool ids
            this call completed, in order, padded with -1.
        k: This step's layer-normed keys, `[num_requests, next_n, head_dim]`.
        gate: This step's gate scores, `[num_requests, next_n, head_dim]`.
        ape: Within-pool position embedding, `[kpool, head_dim]`, f32.
        positions: Absolute position of each new token,
            `[num_requests, next_n]`. Negative marks a padded entry.
        slot_idx: Ring slot owned by each batch row, `[num_requests]`, `uint32`.
        num_requests: Requests actually present.
    """
    var r = block_idx.x
    if r >= Int(num_requests):
        return

    var c = thread_idx.x
    if c >= head_dim:
        return

    comptime max_closed = (next_n + kpool - 1) // kpool
    var tail_base = Int(slot_idx.raw_load(r)) * 2 * kpool * head_dim
    var n_closed = 0

    # Walk in position order: a pool closing here reads slots that earlier
    # tokens stashed in this same call. No barrier is needed, because each
    # stash is read back by the same thread.
    for t in range(next_n):
        var pos = Int(positions.raw_load(r * next_n + t))
        # A padded entry in a speculative batch.
        if pos < 0:
            continue

        var slot = pos % kpool
        tail.raw_store(
            tail_base + slot * head_dim + c,
            k.raw_load((r * next_n + t) * head_dim + c),
        )
        tail.raw_store(
            tail_base + (kpool + slot) * head_dim + c,
            gate.raw_load((r * next_n + t) * head_dim + c),
        )

        if slot != kpool - 1:
            continue

        # The pool closing here runs from `pos - kpool + 1` to `pos`, so
        # member `m` sits at slot `m` and pairs with `ape[m]`.
        var logits = Array[Float32, kpool]()
        var vals = Array[Float32, kpool]()
        for m in range(kpool):
            logits[m] = tail.raw_load(
                tail_base + (kpool + m) * head_dim + c
            ).cast[.float32]() + ape.raw_load(m * head_dim + c)
            vals[m] = tail.raw_load(tail_base + m * head_dim + c).cast[
                .float32
            ]()

        pooled.raw_store(
            (r * max_closed + n_closed) * head_dim + c,
            pool_channel[kpool](logits, vals).cast[dtype](),
        )
        if c == 0:
            closed_pool.raw_store(
                r * max_closed + n_closed, Int32(pos // kpool)
            )
        n_closed += 1

    if c == 0:
        for j in range(n_closed, max_closed):
            closed_pool.raw_store(r * max_closed + j, Int32(-1))


@__name(t"mla_kpool_seed_tail_{kpool}_{head_dim}")
def kpool_seed_tail_kernel[
    dtype: DType,
    TailLayoutType: TensorLayout,
    tail_origin: MutOrigin,
    KLayoutType: TensorLayout,
    k_origin: ImmOrigin,
    GateLayoutType: TensorLayout,
    gate_origin: ImmOrigin,
    IROLayoutType: TensorLayout,
    iro_origin: ImmOrigin,
    CacheLenLayoutType: TensorLayout,
    SlotLayoutType: TensorLayout,
    slot_origin: ImmOrigin,
    head_dim: Int,
    kpool: Int,
](
    tail: TileTensor[dtype, TailLayoutType, tail_origin],
    k: TileTensor[mut=False, dtype, KLayoutType, k_origin],
    gate: TileTensor[mut=False, dtype, GateLayoutType, gate_origin],
    input_row_offsets: TileTensor[
        mut=False, .uint32, IROLayoutType, iro_origin
    ],
    cache_lengths: TileTensor[
        mut=False, .uint32, CacheLenLayoutType, ImmutAnyOrigin
    ],
    slot_idx: TileTensor[mut=False, .uint32, SlotLayoutType, slot_origin],
    num_requests: Int32,
):
    """Stashes a prefill chunk's trailing tokens into the tail ring.

    Compression writes only whole pools, so the tokens after the last complete
    pool have nowhere else to go. They are the members of the request's
    in-progress pool, and decode reads them back from the ring.

    Only the trailing tokens this call owns are written, so a chunked prefill
    lands where a single one does.

    Parameters:
        dtype: Element type of `tail`, `k` and `gate`.
        TailLayoutType: Layout of `tail`.
        tail_origin: Origin of `tail`.
        KLayoutType: Layout of `k`.
        k_origin: Origin of `k`.
        GateLayoutType: Layout of `gate`.
        gate_origin: Origin of `gate`.
        IROLayoutType: Layout of `input_row_offsets`.
        iro_origin: Origin of `input_row_offsets`.
        CacheLenLayoutType: Layout of `cache_lengths`.
        SlotLayoutType: Layout of `slot_idx`.
        slot_origin: Origin of `slot_idx`.
        head_dim: Channels per key; also the block width.
        kpool: Tokens per pool.

    Args:
        tail: Per-slot ring, `[max_slots, 2, kpool, head_dim]`. Index 0 holds
            keys, index 1 holds gate scores.
        k: Layer-normed indexer keys, `[total_tokens, head_dim]`.
        gate: Per-token gate scores, `[total_tokens, head_dim]`.
        input_row_offsets: Token row offsets per request, `[batch_size + 1]`.
        cache_lengths: Cached-prefix length per request, `[batch_size]`.
        slot_idx: Ring slot owned by each batch row, `[num_requests]`, `uint32`.
        num_requests: Requests actually present.
    """
    var r = block_idx.x
    if r >= Int(num_requests):
        return

    var c = thread_idx.x
    if c >= head_dim:
        return

    var row_start = Int(input_row_offsets.raw_load(r))
    var n = Int(input_row_offsets.raw_load(r + 1)) - row_start
    var cache_len = Int(cache_lengths[r])
    var end = cache_len + n
    # Positions after the last complete pool, clipped to this call.
    var seed_start = max((end // kpool) * kpool, cache_len)
    var tail_base = Int(slot_idx.raw_load(r)) * 2 * kpool * head_dim

    for pos in range(seed_start, end):
        var slot = pos % kpool
        var row = row_start + (pos - cache_len)
        tail.raw_store(
            tail_base + slot * head_dim + c, k.raw_load(row * head_dim + c)
        )
        tail.raw_store(
            tail_base + (kpool + slot) * head_dim + c,
            gate.raw_load(row * head_dim + c),
        )


@__name(t"mla_kpool_expand_topk_{kpool}_{pool_topk}_{always_select_tail}")
def kpool_expand_topk_kernel[
    OutLayoutType: TensorLayout,
    out_origin: MutOrigin,
    PoolLayoutType: TensorLayout,
    pool_origin: ImmOrigin,
    IROLayoutType: TensorLayout,
    iro_origin: ImmOrigin,
    CacheLenLayoutType: TensorLayout,
    kpool: Int,
    pool_topk: Int,
    always_select_tail: Bool,
](
    out_indices: TileTensor[.int32, OutLayoutType, out_origin],
    pool_ids: TileTensor[mut=False, .int32, PoolLayoutType, pool_origin],
    input_row_offsets: TileTensor[
        mut=False, .uint32, IROLayoutType, iro_origin
    ],
    cache_lengths: TileTensor[
        mut=False, .uint32, CacheLenLayoutType, ImmutAnyOrigin
    ],
    total_seq_len: Int32,
):
    """Turns selected pool ids back into the token positions they cover.

    The indexer selects pools; attention reads tokens. Each selected pool
    expands to the `kpool` consecutive positions it covers.

    An unselected slot expands to `-1` in every one of its positions, never to
    a clamped valid one, which would point attention at a token the indexer did
    not choose.

    With `always_select_tail` the output carries `kpool - 1` further columns
    holding the query's most recent positions, the ones no complete pool covers
    yet. Their location comes from the query's visible count, so it tracks the
    pool currently being filled.

    Parameters:
        OutLayoutType: Layout of `out_indices`.
        out_origin: Origin of `out_indices`.
        PoolLayoutType: Layout of `pool_ids`.
        pool_origin: Origin of `pool_ids`.
        IROLayoutType: Layout of `input_row_offsets`.
        iro_origin: Origin of `input_row_offsets`.
        CacheLenLayoutType: Layout of `cache_lengths`.
        kpool: Tokens per pool.
        pool_topk: Selected pools per token, `index_topk // kpool`.
        always_select_tail: Whether to append the incomplete trailing pool.

    Args:
        out_indices: Output `[total_seq_len, pool_topk * kpool + tail]`, where
            `tail` is `kpool - 1` when `always_select_tail` and 0 otherwise.
        pool_ids: Selected pool ids, `[total_seq_len, pool_topk]`, `-1` where
            fewer than `pool_topk` pools were available.
        input_row_offsets: Token row offsets per request, `[batch_size + 1]`.
        cache_lengths: Cached-prefix length per request, `[batch_size]`.
        total_seq_len: Number of token rows.
    """
    comptime tail_width = (kpool - 1) if always_select_tail else 0
    comptime out_width = pool_topk * kpool + tail_width

    var token_idx = block_idx.x
    if token_idx >= Int(total_seq_len):
        return

    var batch_size = Int(input_row_offsets.dim[0]()) - 1
    var b = 0
    for i in range(batch_size):
        if token_idx < Int(input_row_offsets.raw_load(i + 1)):
            b = i
            break

    var local = token_idx - Int(input_row_offsets.raw_load(b))
    # Causal: a query sees its own position and everything before it.
    var visible = Int(cache_lengths[b]) + local + 1

    var tail_count = visible % kpool
    var tail_start = visible - tail_count

    var col = thread_idx.x
    while col < out_width:
        var value = Int32(-1)
        if col < pool_topk * kpool:
            var pid = Int(
                pool_ids.raw_load(token_idx * pool_topk + col // kpool)
            )
            if pid >= 0:
                value = Int32(pid * kpool + col % kpool)
        else:
            var tail_idx = col - pool_topk * kpool
            if tail_idx < tail_count:
                value = Int32(tail_start + tail_idx)
        out_indices.raw_store(token_idx * out_width + col, value)
        col += block_dim.x
