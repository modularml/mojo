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
"""Builds persistent-scheduling work-partition metadata for MLA-prefill on the host.

Faithful port of the reference's `kn_generate_ps_metadata`
(`v1_2_host.cuh:61-241`) + `WorkInfo`/`QTile` (`ps.h`). Copies the reference
exactly; does not invent.

Even-division-of-split-units load balancer (the prefill-ps batch-flatness
lever): query tiles (qlen_granularity tokens x all heads) are built per batch
with causal effective-KV, counted in kvlen_granularity units, and total units
are split evenly across `available_tgs` persistent workgroups, splitting a
tile's KV across TGs when one TG's quota is exceeded (-> partials + reduce).

This is a pure host/CPU module (no GPU): the persistent kernel (S2) builds
the work partition on the host, uploads `work_indptr`/`work_info` to device
buffers, and the persistent grid consumes them. Tested standalone by
`test/gpu/structured_kernels/test_ps_metadata.mojo`.
"""

from std.collections import List
from std.math import ceildiv


def pack_dword(low_part: Int32, high_part: Int32) -> Int32:
    """Packs two 16-bit halves into one 32-bit word as `(high<<16) | (low & 0xFFFF)`.

    Mirrors `ps.h:10-14` from the reference implementation.

    Args:
        low_part: Value occupying the low 16 bits of the packed word; masked
            with `0xFFFF` before packing.
        high_part: Value occupying the high 16 bits of the packed word;
            shifted left by 16 before packing.
    """
    # ps.h:10-14 -- (high<<16) | (low & 0xFFFF)
    return (high_part << 16) | (low_part & 0xFFFF)


@fieldwise_init
struct QTile(Copyable, Movable):
    """Per ps.h:69-86; qo_start/qo_end are global TOKEN offsets."""

    var batch_idx: Int32
    var qo_start: Int32
    var qo_end: Int32
    var num_blocks: Int32
    var effective_kv_length: Int32


# WorkInfo (ps.h:24-51) is 8 int32: batch_idx, partial_o_loc, qo_start, qo_end,
# kv_start, kv_end, kv_offset, q_head_range. Stored flat (8 per work item).
comptime WORKINFO_DW = 8


struct PsMetadata(Copyable, Movable):
    """Holds the persistent-scheduling work partition built on the host.

    Stores `work_indptr` (prefix-sum of work counts per thread-group, length
    `available_tgs+1`) and `work_info` (flat `WorkInfo` records, 8 int32 each)
    that the persistent GPU kernel (S2) consumes to drive its grid.
    """

    var work_indptr: List[Int32]  # [available_tgs+1], prefix-sum of work counts
    var work_info: List[Int32]  # [num_works * 8], flat WorkInfo
    var num_works: Int

    def __init__(out self):
        self.work_indptr = List[Int32]()
        self.work_info = List[Int32]()
        self.num_works = 0


def _append_work(
    mut wi: List[Int32],
    batch_idx: Int32,
    partial_o_loc: Int32,
    qo_start: Int32,
    qo_end: Int32,
    kv_start: Int32,
    kv_end: Int32,
    kv_offset: Int32,
    q_head_range: Int32,
):
    wi.append(batch_idx)
    wi.append(partial_o_loc)
    wi.append(qo_start)
    wi.append(qo_end)
    wi.append(kv_start)
    wi.append(kv_end)
    wi.append(kv_offset)
    wi.append(q_head_range)


def kn_generate_ps_metadata(
    seqlens_qo_indptr: List[Int32],  # [batch+1] token indptr
    pages_kv_indptr: List[Int32],  # [batch+1] block indptr
    context_lens: List[Int32],  # [batch] kv token length
    num_heads_k: Int32,
    qhead_granularity: Int32,
    qlen_granularity: Int32,
    kvlen_granularity: Int32,
    block_size: Int32,
    is_causal: Bool,
    available_tgs: Int32,
    cluster_id: Int32,
) -> PsMetadata:
    """Faithful port of v1_2_host.cuh:61-241 (single-cluster: cluster_id=0,
    current_work_idx=0 -- the nkv=1 prefill case).

    Args:
        seqlens_qo_indptr: Prefix-sum array of length `batch+1`; entry `i`
            is the global query-token offset where batch `i` begins.
        pages_kv_indptr: Prefix-sum of KV block counts per batch, length
            `batch+1`; entry `i` is the block offset where batch `i`'s KV
            pages begin.
        context_lens: Per-batch KV token lengths, length `batch`; the number
            of cached KV tokens for each sequence.
        num_heads_k: Number of KV heads assigned to this cluster; each head
            receives an identical work split.
        qhead_granularity: Number of query heads packed per work-item; sets
            the stride used to build `q_head_range`.
        qlen_granularity: Query tile width in tokens; one work-item covers
            this many tokens of a single head.
        kvlen_granularity: KV split unit width in tokens; must be a multiple
            of `block_size`. Drives the even split across thread-groups.
        block_size: Paged-attention block size in tokens; the granularity of
            `pages_kv_indptr` and the KV block count.
        is_causal: When true, each query tile's effective KV length is masked
            to the causal boundary; when false, the full KV length is used.
        available_tgs: Number of persistent thread-groups to split the total
            KV units across evenly.
        cluster_id: Index of this cluster; offsets the KV head index used to
            build `q_head_range`.
    """
    var batch_size = len(seqlens_qo_indptr) - 1
    debug_assert(
        kvlen_granularity % block_size == 0,
        "kvlen_granularity must be a multiple of block_size",
    )
    var blocks_per_unit = kvlen_granularity // block_size

    # --- Step 1: build query tiles (ping-pong order) + count split units. ---
    var query_tiles = List[QTile]()
    var total_units: Int32 = 0
    for batch_idx in range(batch_size):
        var qo_length = (
            seqlens_qo_indptr[batch_idx + 1] - seqlens_qo_indptr[batch_idx]
        )
        var kv_length = context_lens[batch_idx]

        var range_start = List[Int32]()
        var range_end = List[Int32]()
        var q_offset: Int32 = 0
        while q_offset < qo_length:
            range_start.append(q_offset)
            range_end.append(min(q_offset + qlen_granularity, qo_length))
            q_offset += qlen_granularity
        var n = len(range_start)

        for i in range(n):
            # ping-pong head<->tail: 0, n-1, 1, n-2, ... (v1_2:104)
            var idx = (i // 2) if (i % 2 == 0) else (n - 1 - i // 2)
            var local_qo_start = range_start[idx]
            var local_qo_end = range_end[idx]

            var effective_kv_length: Int32
            if is_causal:
                effective_kv_length = min(
                    kv_length - qo_length + local_qo_end, kv_length
                )
            else:
                effective_kv_length = kv_length
            var num_units = ceildiv(effective_kv_length, kvlen_granularity)
            query_tiles.append(
                QTile(
                    Int32(batch_idx),
                    local_qo_start + seqlens_qo_indptr[batch_idx],
                    local_qo_end + seqlens_qo_indptr[batch_idx],
                    num_units * blocks_per_unit,
                    effective_kv_length,
                )
            )
            total_units += num_units

    # --- Step 2: distribute split units evenly across TGs. ---
    var average = total_units // available_tgs
    var remainder = total_units % available_tgs

    var result = PsMetadata()
    result.work_indptr.append(0)  # work_indptr[0] = 0

    var current_tile_idx = 0
    var current_block_idx: Int32 = 0
    var partial_tile_idx: Int32 = 0
    var current_work_idx: Int32 = 0
    var num_tiles = len(query_tiles)

    for tg_idx in range(Int(available_tgs)):
        for k_head_offset in range(Int(num_heads_k)):
            var k_head_idx = cluster_id * num_heads_k + Int32(k_head_offset)
            var qhead_range = pack_dword(
                k_head_idx * qhead_granularity,
                (k_head_idx + 1) * qhead_granularity,
            )
            var saved_tile_idx = current_tile_idx
            var saved_block_idx = current_block_idx
            var saved_partial_tile_idx = partial_tile_idx

            var blocks_capacity = (
                (average + 1) if (Int32(tg_idx) < remainder) else average
            ) * blocks_per_unit

            while current_tile_idx < num_tiles and blocks_capacity > 0:
                var ct = query_tiles[current_tile_idx].copy()
                var remaining_blocks = ct.num_blocks - current_block_idx
                var remaining_kv_len = (
                    ct.effective_kv_length - current_block_idx * block_size
                )
                var kv_start = (
                    current_block_idx + pages_kv_indptr[Int(ct.batch_idx)]
                )

                var consuming_blocks: Int32
                if remaining_kv_len <= blocks_capacity * block_size:
                    # Fits: this TG finishes the tile to the causal boundary.
                    consuming_blocks = remaining_blocks
                    var partial_o_loc: Int32 = -1
                    if current_block_idx != 0:
                        partial_o_loc = qlen_granularity * partial_tile_idx
                        partial_tile_idx += 1
                    var kv_end = min(
                        kv_start + consuming_blocks,
                        pages_kv_indptr[Int(ct.batch_idx) + 1],
                    )
                    _append_work(
                        result.work_info,
                        ct.batch_idx,
                        partial_o_loc,
                        ct.qo_start,
                        ct.qo_end,
                        kv_start,
                        kv_end,
                        0,
                        qhead_range,
                    )
                    current_work_idx += 1
                    current_tile_idx += 1
                    current_block_idx = 0
                else:
                    # Split: consume this TG's quota, leave the rest for later.
                    consuming_blocks = blocks_capacity
                    var partial_o_loc = qlen_granularity * partial_tile_idx
                    partial_tile_idx += 1
                    var kv_end = min(
                        kv_start + consuming_blocks,
                        pages_kv_indptr[Int(ct.batch_idx) + 1],
                    )
                    var kv_length = context_lens[Int(ct.batch_idx)]
                    var kv_offset = (
                        kv_length
                        - (kv_end - pages_kv_indptr[Int(ct.batch_idx)])
                        * block_size
                    )
                    _append_work(
                        result.work_info,
                        ct.batch_idx,
                        partial_o_loc,
                        ct.qo_start,
                        ct.qo_end,
                        kv_start,
                        kv_end,
                        kv_offset,
                        qhead_range,
                    )
                    current_work_idx += 1
                    current_block_idx += consuming_blocks

                blocks_capacity -= consuming_blocks

            # Per-k-head cursor rewind (each k-head gets identical split).
            if k_head_offset != Int(num_heads_k) - 1:
                current_tile_idx = saved_tile_idx
                current_block_idx = saved_block_idx
                partial_tile_idx = saved_partial_tile_idx

        result.work_indptr.append(current_work_idx)  # [cluster*tgs + tg + 1]

    result.num_works = Int(current_work_idx)
    return result^


def _gcd(a: Int32, b: Int32) -> Int32:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def build_ps_metadata(
    seqlens_qo_indptr: List[Int32],
    pages_kv_indptr: List[Int32],
    context_lens: List[Int32],
    num_heads_k: Int32,
    gqa_ratio: Int32,
    tile_q: Int32,
    tile_kv: Int32,
    block_size: Int32,
    is_causal: Bool,
    available_tgs: Int32,
) -> PsMetadata:
    """Port of `get_ps_metadata_v1_2_host` (v1_2_host.cuh:265-314): the host
    wrapper that GCD-clusters heads across TGs, then calls the per-cluster
    `kn_generate_ps_metadata` and concatenates.

    For MLA-prefill this is MHA (`gqa_ratio==1`, one head per work-item): the
    work-item Q tile is `qlen_granularity = tile_q // gqa_ratio` TOKENS of ONE
    head (token-major; the 256 MMA rows are 256 tokens, NOT 16 tok x 16 head),
    and `q_head_range`'s low 16 bits = the head index (= cluster_id).

    Args:
        seqlens_qo_indptr: Prefix-sum array of length `batch+1`; entry `i`
            is the global query-token offset where batch `i` begins.
        pages_kv_indptr: Prefix-sum of KV block counts per batch, length
            `batch+1`; entry `i` is the block offset where batch `i`'s KV
            pages begin.
        context_lens: Per-batch KV token lengths, length `batch`; the number
            of cached KV tokens for each sequence.
        num_heads_k: Total number of KV heads; GCD-clustered across
            `available_tgs` thread-groups.
        gqa_ratio: Ratio of query heads to KV heads; for MLA-prefill MHA
            this is 1. Sets `qhead_granularity` and divides `tile_q` for
            `qlen_granularity`.
        tile_q: Query tile size in tokens before head division; one work-item
            covers `tile_q // gqa_ratio` tokens of one head.
        tile_kv: KV tile size in blocks; sets `kvlen_granularity` via
            `max(tile_kv, block_size)`.
        block_size: Paged-attention block size in tokens; the granularity of
            `pages_kv_indptr`.
        is_causal: When true, each query tile's effective KV length is masked
            to the causal boundary.
        available_tgs: Number of persistent thread-groups; GCD-clustered with
            `num_heads_k` to form clusters.
    """
    var qhead_granularity = gqa_ratio
    var qlen_granularity = tile_q // gqa_ratio
    var kvlen_granularity = max(tile_kv, block_size)

    var num_clusters = _gcd(num_heads_k, available_tgs)
    var kheads_per_cluster = num_heads_k // num_clusters
    var tgs_per_cluster = available_tgs // num_clusters

    var result = PsMetadata()
    result.work_indptr.append(0)
    var work_base: Int32 = 0
    for cluster_id in range(Int(num_clusters)):
        var md_c = kn_generate_ps_metadata(
            seqlens_qo_indptr,
            pages_kv_indptr,
            context_lens,
            kheads_per_cluster,
            qhead_granularity,
            qlen_granularity,
            kvlen_granularity,
            block_size,
            is_causal,
            tgs_per_cluster,
            Int32(cluster_id),
        )
        for i in range(len(md_c.work_info)):
            result.work_info.append(md_c.work_info[i])
        # md_c.work_indptr has tgs_per_cluster+1 entries starting with 0; offset
        # by the running work-item total and append all but the leading 0.
        for i in range(1, len(md_c.work_indptr)):
            result.work_indptr.append(work_base + md_c.work_indptr[i])
        work_base += Int32(md_c.num_works)

    result.num_works = Int(work_base)
    return result^


def build_uniform(
    batch: Int,
    seq: Int32,
    num_q_heads: Int32 = 16,
    available_tgs: Int32 = 256,
) -> PsMetadata:
    """Build metadata for uniform-seqlen self-attention, FP8 causal MLA-prefill.

    The bench/test shape: every sequence is `seq` tokens, causal self-attention,
    one latent KV head, `num_q_heads` query heads (MHA, gqa=1). One work-item is
    `tile_q=256` TOKENS of ONE head (token-major BM=256); the KV split unit is
    `tile_kv=128` blocks (= KV_BLOCK).

    `available_tgs` = number of persistent thread-groups (= grid_dim.x = device
    CU count, 256 on MI355X). Set it to `num_q_heads` for a split-free partition
    at any seq (1 TG per head, useful for correctness gating).

    Args:
        batch: Number of sequences in the batch; all sequences have length
            `seq`.
        seq: Token length of every sequence in the batch.
        num_q_heads: Number of query heads (MHA, gqa=1) (defaults to 16).
        available_tgs: Number of persistent thread-groups; set to
            `num_q_heads` for a split-free partition (defaults to 256).
    """
    var qo_indptr = List[Int32]()
    var kv_indptr = List[Int32]()
    var ctx_lens = List[Int32]()
    for b in range(batch + 1):
        qo_indptr.append(Int32(b) * seq)
        kv_indptr.append(Int32(b) * seq)
    for _ in range(batch):
        ctx_lens.append(seq)
    return build_ps_metadata(
        qo_indptr,
        kv_indptr,
        ctx_lens,
        num_q_heads,  # num_heads_k = num KV heads = num q-heads (MHA, gqa=1)
        1,  # gqa_ratio
        256,  # tile_q
        128,  # tile_kv
        1,  # block_size
        True,  # is_causal
        available_tgs,
    )
