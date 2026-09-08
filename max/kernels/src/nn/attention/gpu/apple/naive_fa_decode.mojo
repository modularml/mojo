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
"""Apple (Metal) split-K naive flash-attention DECODE kernels.

Apple silicon GPU (Metal), decode-only (one query token per sequence), paged KV
cache via the `MHAOperand` contract, BF16 storage / FP32 accumulation.

Warp-centric producer: one simdgroup (32 lanes) owns one split of the KV range
for one `(batch, head)`. Lane `L` owns the contiguous head-dim chunk
`[L*EPL, L*EPL+EPL)` where `EPL = head_dim // WARP_SIZE`; the query and running
output stay in registers, `Q.K^T` is reduced across lanes with one `air.simd_sum`
per key, and `P.V` is reduction-free. The inner loop has **no `barrier()` and no
    threadgroup memory** (the two levers Apple silicon is most sensitive to).

Two kernels:
  * `naive_fa_decode_apple_core`: producer. Grid `(num_partitions,
    batch_size, num_heads)`, block = `WARP_SIZE` (one simdgroup). Each block
    writes per-partition partials `(o_partial, m_partial, l_partial)` via online
    softmax over `BN`-wide KV tiles.
  * `naive_fa_decode_apple_stitch`: stitch. Grid `(num_heads, batch_size)`,
    block `depth`. One thread per depth element; combines the contiguous
    per-partition partials into the final `output` with a log-sum-exp (LSE)
    reduction.

The host launcher `naive_fa_decode_apple` allocates the partials and enqueues
both kernels; `flash_attention_dispatch` selects it for Apple decode by default
(set `MODULAR_ENABLE_APPLE_NAIVE_FA_DECODE=0` to opt out). The launcher
dispatches the runtime `depth` to a compile-time `Depth` specialization;
`naive_fa_decode_apple_supports_depth` is the single definition of which head
dims those are, and the dispatcher sends the rest to `mha_gpu_naive`.

Partial-buffer layout (partition-last / contiguous):
  * `ml_idx(b, head, split) = (b*num_heads + head)*num_partitions + split`
  * `o_idx(b, head, d, split) = ((b*num_heads + head)*depth + d)*num_partitions
    + split`
"""

from std.collections import OptionalReg
from max.gpu import WARP_SIZE, block_idx, lane_id, thread_idx
from max.gpu.host import DeviceContext
from std.math import ceildiv, exp
from std.sys import llvm_intrinsic
from std.utils.index import Index
from std.utils.numerics import get_accum_type

from layout import (
    UNKNOWN_VALUE,
    Idx,
    Layout,
    LayoutTensor,
    DefaultEngine,
    TensorEngine,
    TileTensor,
)
from layout.coord import Coord
from layout.tile_layout import (
    TensorLayout,
    row_major,
)

from nn.attention.mha_mask import MHAMask
from nn.attention.mha_operand import MHAOperand

comptime BN = 16  # KV keys per producer tile step
comptime NEG_INF = Float32(-3.0e38)

# Dispatcher gate: larger dims (and non-multiples of WARP_SIZE) fall back to
# mha_gpu_naive.
comptime NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM = 256


@always_inline
def naive_fa_decode_apple_supports_depth(depth: Int) -> Bool:
    """Whether this kernel has a `Depth` specialization for `depth`.

    Splitting the head dim across lanes needs `depth % WARP_SIZE == 0`, but a
    lane's fragment is an `EPL = depth // WARP_SIZE` wide SIMD and a SIMD length
    must be a power of two — so `EPL` is constrained too, which rules out head
    dims like 96 and 160. The launcher specializes exactly this set, and
    `flash_attention_dispatch` sends everything else to `mha_gpu_naive`; both
    ask here so the two cannot disagree and drop a launch on the floor.

    Args:
        depth: The head dimension to check.

    Returns:
        Whether `depth` is dispatchable to this kernel.
    """
    return (
        depth % WARP_SIZE == 0
        and depth <= NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM
        and Bool((depth // WARP_SIZE).is_power_of_two())
    )


@always_inline
def _apple_simd_sum(val: Float32) -> Float32:
    """Sum `val` across the simdgroup (broadcast to all lanes).

    One hardware instruction vs. a 5-shuffle butterfly tree.
    """
    return llvm_intrinsic["llvm.air.simd_sum", Float32](val)


@always_inline
def _ml_idx(
    b: Int, head: Int, split: Int, num_heads: Int, num_partitions: Int
) -> Int:
    """Partition-last index for the m/l partials."""
    return (b * num_heads + head) * num_partitions + split


@always_inline
def _o_idx(
    b: Int,
    head: Int,
    d: Int,
    split: Int,
    num_heads: Int,
    depth: Int,
    num_partitions: Int,
) -> Int:
    """Partition-last index for the o partials."""
    return ((b * num_heads + head) * depth + d) * num_partitions + split


# ===-------------------------------------------------------------------=== #
# Producer: warp-centric split-K online-softmax.
# Grid (num_partitions, batch, head); block = WARP_SIZE (one simdgroup).
# ===-------------------------------------------------------------------=== #
def naive_fa_decode_apple_core[
    q_type: DType,
    # `output_type` is unused; the parameter list mirrors `mha_gpu_naive` for
    # dispatch uniformity.
    output_type: DType,
    p_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    p_layout: TensorLayout,
    q_layout: TensorLayout,
    valid_length_layout: TensorLayout,
    sink_layout: TensorLayout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    Depth: Int,
    SplitSize: Int,
    OPartialEngine: TensorEngine = DefaultEngine[element_width=1],
    MPartialEngine: TensorEngine = DefaultEngine[element_width=1],
    LPartialEngine: TensorEngine = DefaultEngine[element_width=1],
    QEngine: TensorEngine = DefaultEngine[element_width=1],
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
    SinkEngine: TensorEngine = DefaultEngine[element_width=1],
](
    o_partial: TileTensor[
        p_type, p_layout, MutAnyOrigin, Engine=OPartialEngine
    ],
    m_partial: TileTensor[
        p_type, p_layout, MutAnyOrigin, Engine=MPartialEngine
    ],
    l_partial: TileTensor[
        p_type, p_layout, MutAnyOrigin, Engine=LPartialEngine
    ],
    q: TileTensor[q_type, q_layout, ImmutAnyOrigin, Engine=QEngine],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        .uint32, valid_length_layout, ImmutAnyOrigin, Engine=VLEngine
    ],
    sink_weights: OptionalReg[
        TileTensor[q_type, sink_layout, ImmutAnyOrigin, Engine=SinkEngine]
    ],
    scale: Float32,
    batch_size: Int32,
    max_prompt_len: Int32,
    # Full key count for the dense decode path (the K tensor's seq dim); the
    # KVCache/ragged paths derive their key count from `cache_length` +
    # `cur_query_len` instead. See the `cur_cache_len` branch below.
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    group: Int32,
    num_partitions: Int32,
):
    """Warp-centric split-K online-softmax producer for Apple decode attention.

    Lane `L` of the simdgroup owns the contiguous head-dim chunk
    `[L*EPL, L*EPL+EPL)` where `EPL = Depth // WARP_SIZE`. q and the running
    output stay in registers; `Q.K^T` is reduced across lanes with one
    `air.simd_sum` per key; `P.V` is reduction-free. No barriers, no shared
    memory.

    Parameters:
        q_type: Element type of the query tensor (inferred).
        output_type: Unused; mirrors `mha_gpu_naive` for dispatch
            uniformity (inferred).
        p_type: Accumulation and partials element type (inferred).
        k_t: `MHAOperand` type of the key cache operand (inferred).
        v_t: `MHAOperand` type of the value cache operand (inferred).
        mask_t: `MHAMask` functor type applied to attention scores
            (inferred).
        p_layout: `TensorLayout` of the partials buffers (inferred).
        q_layout: `TensorLayout` of the query tensor (inferred).
        valid_length_layout: `TensorLayout` of the `valid_length` tensor
            (inferred).
        sink_layout: `TensorLayout` of the sink weights tensor (inferred).
        ragged: Whether sequences are ragged with variable lengths and
            row offsets in `valid_length` (defaults to `False`).
        sink: Whether attention sink is enabled, pre-seeding split 0
            with per-head sink weights (defaults to `False`).
        _use_valid_length: Whether to use `valid_length` for KVCache
            decode as per-sequence query lengths (defaults to `False`).
        _is_cache_length_accurate: Whether the cache length equals the
            query length, so no new-token KV is added (defaults to
            `False`).
        Depth: Compile-time head dimension; must be a multiple of
            `WARP_SIZE`.
        SplitSize: Per-partition KV span in keys.
        OPartialEngine: Engine of the `o_partial` tile.
        MPartialEngine: Engine of the `m_partial` tile.
        LPartialEngine: Engine of the `l_partial` tile.
        QEngine: Engine of the `q` tile.
        VLEngine: Engine of the `valid_length` tile.
        SinkEngine: Engine of the `sink_weights` tile.

    Args:
        o_partial: Flat 1D partial output buffer; one accumulator per
            `(batch, head, depth, split)`.
        m_partial: Flat 1D partial row-max buffer; one running max per
            `(batch, head, split)`.
        l_partial: Flat 1D partial row-sum buffer; one running
            denominator per `(batch, head, split)`.
        q: Flat 1D query tensor; one token per sequence (decode).
        k: Key cache operand implementing the `MHAOperand` contract.
        v: Value cache operand implementing the `MHAOperand` contract.
        mask_functor: Mask functor applied to each attention score.
        valid_length: Per-sequence row offsets or query lengths
            (`uint32`); meaning depends on `ragged` and
            `_use_valid_length`.
        sink_weights: Optional per-head attention sink weights; when
            `sink` is enabled, pre-seeds split 0 before KV attention.
        scale: Softmax scale factor applied to `Q.K^T` scores.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum prompt length; the dense decode path's
            query length.
        max_cache_size: Full key count for the dense decode path (the K
            tensor's seq dim).
        num_heads: Number of query attention heads.
        depth: Runtime head dimension; must equal the compile-time
            `Depth`.
        group: Number of query heads per KV head (GQA group size).
        num_partitions: Number of KV splits; the grid X dimension.

    Constraints:
        `Depth % WARP_SIZE == 0`: the head dim must split evenly across lanes.
    """
    var _batch_size = Int(batch_size)
    var _max_prompt_len = Int(max_prompt_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _group = Int(group)
    var _num_partitions = Int(num_partitions)
    comptime assert (
        Depth % WARP_SIZE == 0
    ), "naive_fa_decode_apple_core requires Depth % WARP_SIZE == 0"
    comptime EPL = Depth // WARP_SIZE
    debug_assert(_depth == Depth, "runtime _depth must match comptime Depth")

    var split_id = Int(block_idx.x)
    var batch_id = Int(block_idx.y)
    var head_id = Int(block_idx.z)
    var kv_head = head_id // _group
    var lane = Int(lane_id())

    # Decode offset math — mirror `_bmm0_bs` (mha.mojo:5560-5589). The
    # `cur_cache_len` (number of keys to attend) is set PER BRANCH because the
    # dense (`else`) path takes it from `_max_cache_size` (the K tensor's full
    # seq dim), NOT from `cur_query_len` / `cache_length` — exactly as the naive
    # fallback `_bmm0_bs` and the Apple prefill producer (`fa_prefill.mojo`) do.
    # The prior shared `cur_cache_len = cur_query_len` (under
    # `_is_cache_length_accurate`) silently attended only 1 key on the dense
    # decode path (`_use_valid_length=False, _is_cache_length_accurate=True` --
    # the `flash_attention` dense overload's ABI), so a dense decode dropped all
    # but the first key. Only the KVCache decode path (`_use_valid_length=True`)
    # was ever exercised.
    var seq_start: Int
    var cur_query_len: Int
    var q_offset: Int
    var cur_cache_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        q_offset = _depth * (seq_start * _num_heads + head_id)
        # The new token's own KV sits at index `cache_length`, so an inaccurate
        # cache length must include it. Mirror `_bmm0_bs` (mha.mojo:5567-5575).
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    elif _use_valid_length:
        # KVCache decode: valid_length holds per-sequence query lengths, not row
        # offsets. Mirror `_bmm0_bs` (mha.mojo:5576-5582).
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        q_offset = _depth * (head_id + _num_heads * _max_prompt_len * batch_id)
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    else:
        # Dense decode: all sequences share one length and cache length; the
        # full key count is `_max_cache_size` (the K tensor's seq dim). Mirror
        # `_bmm0_bs` (mha.mojo:5585-5589).
        seq_start = batch_id
        cur_query_len = _max_prompt_len
        q_offset = _depth * (head_id + _num_heads * _max_prompt_len * batch_id)
        cur_cache_len = _max_cache_size
    var seq_len = cur_cache_len

    var start = split_id * SplitSize
    if start >= seq_len:
        return
    var end = min(start + SplitSize, seq_len)

    # Decode token's score-matrix row (== cache_length). Mirror mha.mojo:5324.
    var score_row = cur_cache_len - cur_query_len

    # Q is a flat 1D TileTensor over the whole buffer; this lane owns the
    # head-dim chunk [lane*EPL, lane*EPL+EPL) at `q_offset`. Vectorized load
    # through the tile (no raw pointer arithmetic).
    var q_frag = q.load[width=EPL](Coord(q_offset + lane * EPL)).cast[
        DType.float32
    ]()

    # KV sub-tile layout: a 1D (_depth,) contiguous view of one token's K/V for
    # `kv_head`, reused for every key in this split. `block_paged_tile` infers
    # the type from this value; each lane loads its `EPL` chunk.
    var kv_token_layout = row_major(Coord(_depth))

    # Replicated on every lane, so the running softmax needs no cross-lane comms.
    var m = NEG_INF
    var l = Float32(0.0)
    var o_frag = SIMD[.float32, EPL](0.0)

    # Attention sink as init-state: pre-seed (m, l) with a virtual "key -1" of
    # raw score `sink_weight`, contributing `exp(sink - m) = 1` to the running
    # denominator (plain `exp`, no log2e — Apple, like the prefill kernel and
    # nn/softmax.mojo, compares the UNSCALED sink weight against the post-scale
    # row max). Seed ONLY split 0: this is split-K, so the stitch kernel does a
    # cross-split LSE combine; seeding every split would count the sink
    # `_num_partitions` times. Split 0 always exists (start=0 < seq_len), so the
    # sink is counted exactly once. Mirrors AppleSoftmax.seed_sink in
    # fa_prefill.mojo and amd-attention-sink-as-init-state.
    comptime if sink:
        if split_id == 0:
            # Per-head sink weight from the nullable `OptionalReg[TileTensor]`
            # (NOT a dangling pointer -- KB `unsafepointer-is-non-nullable`).
            # The deref is comptime-gated on `sink`, so None is never reached.
            m = rebind[Scalar[q_type]](sink_weights.value()[head_id]).cast[
                DType.float32
            ]()
            l = Float32(1.0)

    for kv0 in range(start, end, BN):
        var partials = SIMD[.float32, BN](0.0)

        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var k_tile = k.block_paged_tile[1](
                    UInt32(batch_id),
                    UInt32(j),
                    UInt32(kv_head),
                    kv_token_layout,
                )
                var kvec = k_tile.load[width=EPL](Coord(lane * EPL)).cast[
                    DType.float32
                ]()
                partials[kk] = (q_frag * kvec).reduce_add()

        # `air.simd_sum` is a warp collective; the `j < end` guard is
        # lane-independent, so all lanes enter it together.
        var scores = SIMD[.float32, BN](NEG_INF)

        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var s = _apple_simd_sum(partials[kk]) * scale
                scores[kk] = mask_functor.mask(
                    Index(batch_id, head_id, score_row, j), s
                )

        var m_tile = scores.reduce_max()
        var m_new = max(m, m_tile)
        var alpha = exp(m - m_new)
        var p = exp(scores - m_new)  # OOB keys are NEG_INF -> exp == 0
        l = l * alpha + p.reduce_add()
        m = m_new
        o_frag = o_frag * alpha

        # No cross-lane reduction: each lane accumulates its own output chunk.
        comptime for kk in range(BN):
            var j = kv0 + kk
            if j < end:
                var v_tile = v.block_paged_tile[1](
                    UInt32(batch_id),
                    UInt32(j),
                    UInt32(kv_head),
                    kv_token_layout,
                )
                var vvec = v_tile.load[width=EPL](Coord(lane * EPL)).cast[
                    DType.float32
                ]()
                o_frag = o_frag + p[kk] * vvec

    comptime assert (
        o_partial.flat_rank == 1 and m_partial.flat_rank == 1
    ), "partials are flat 1D TileTensors"
    comptime for i in range(EPL):
        var d = lane * EPL + i
        var oi = _o_idx(
            batch_id, head_id, d, split_id, _num_heads, Depth, _num_partitions
        )
        o_partial[oi] = rebind[o_partial.ElementType](
            SIMD[p_type, 1](o_frag[i].cast[p_type]())
        )
    if lane == 0:
        var idx = _ml_idx(
            batch_id, head_id, split_id, _num_heads, _num_partitions
        )
        l_partial[idx] = rebind[l_partial.ElementType](
            SIMD[p_type, 1](l.cast[p_type]())
        )
        m_partial[idx] = rebind[m_partial.ElementType](
            SIMD[p_type, 1](m.cast[p_type]())
        )


# ===-------------------------------------------------------------------=== #
# Stitch: LSE-combine the per-partition partials. Grid (_num_heads, batch),
# block `_depth`.
# ===-------------------------------------------------------------------=== #
def naive_fa_decode_apple_stitch[
    output_type: DType,
    p_type: DType,
    k_t: MHAOperand,
    # `v_t` and `mask_t` are unused; the parameter list mirrors `mha_gpu_naive`
    # for dispatch uniformity.
    v_t: MHAOperand,
    mask_t: MHAMask,
    output_layout: TensorLayout,
    p_layout: TensorLayout,
    valid_length_layout: TensorLayout,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    SplitSize: Int,
    OutEngine: TensorEngine = DefaultEngine[element_width=1],
    OPartialEngine: TensorEngine = DefaultEngine[element_width=1],
    MPartialEngine: TensorEngine = DefaultEngine[element_width=1],
    LPartialEngine: TensorEngine = DefaultEngine[element_width=1],
    VLEngine: TensorEngine = DefaultEngine[element_width=1],
](
    output: TileTensor[
        output_type, output_layout, MutAnyOrigin, Engine=OutEngine
    ],
    o_partial: TileTensor[
        p_type, p_layout, ImmutAnyOrigin, Engine=OPartialEngine
    ],
    m_partial: TileTensor[
        p_type, p_layout, ImmutAnyOrigin, Engine=MPartialEngine
    ],
    l_partial: TileTensor[
        p_type, p_layout, ImmutAnyOrigin, Engine=LPartialEngine
    ],
    k: k_t,
    valid_length: TileTensor[
        .uint32, valid_length_layout, ImmutAnyOrigin, Engine=VLEngine
    ],
    max_prompt_len: Int32,
    # Full key count for the dense decode path; mirrors the producer so the
    # combine's `active_splits` matches the splits the producer actually wrote.
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    num_partitions: Int32,
):
    var _max_prompt_len = Int(max_prompt_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _num_partitions = Int(num_partitions)
    comptime assert (
        o_partial.flat_rank == 1
        and m_partial.flat_rank == 1
        and output.flat_rank == 1
    ), "partials and output are flat 1D TileTensors"
    var head_id = Int(block_idx.x)
    var batch_id = Int(block_idx.y)
    var d = Int(thread_idx.x)

    if d >= _depth:
        return

    # Output offset — mirror mha.mojo:5390. `cur_cache_len` (the attend span)
    # is set PER BRANCH and MUST match the producer's exactly, so the combine
    # reads precisely the splits the producer wrote (the dense path takes it
    # from `_max_cache_size`, not `cur_query_len`).
    var seq_start: Int
    var cur_query_len: Int
    var cur_cache_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    elif _use_valid_length:
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    else:
        # Dense decode: full key count is `_max_cache_size`.
        seq_start = batch_id
        cur_query_len = _max_prompt_len
        cur_cache_len = _max_cache_size

    # Split count must mirror the producer's attend span (`cur_cache_len`), not
    # the bare cache length, so we read exactly the partials that were written.
    var active_splits = ceildiv(cur_cache_len, SplitSize)

    # Combine in Float32 (partials cast in on read), matching the producer.
    var m = NEG_INF
    var l = Float32(0.0)
    var acc = Float32(0.0)

    for split in range(active_splits):
        var ml = _ml_idx(batch_id, head_id, split, _num_heads, _num_partitions)
        var m_s = rebind[Scalar[p_type]](m_partial[ml]).cast[.float32]()
        var m_new = max(m, m_s)
        var corr = exp(m - m_new)
        # `p` must use the same exp base as the producer for an exact combine.
        var p = exp(m_s - m_new)
        var l_s = rebind[Scalar[p_type]](l_partial[ml]).cast[.float32]()
        l = l * corr + p * l_s
        var oi = _o_idx(
            batch_id, head_id, d, split, _num_heads, _depth, _num_partitions
        )
        var o_s = rebind[Scalar[p_type]](o_partial[oi]).cast[.float32]()
        acc = acc * corr + p * o_s
        m = m_new

    var o_off = (seq_start * _num_heads + head_id) * _depth
    output[o_off + d] = rebind[output.ElementType](
        SIMD[output_type, 1]((acc / l).cast[output_type]())
    )


# ===-------------------------------------------------------------------=== #
# Host launcher. Mirrors `mha_gpu_naive` (MHAOperand overload, mha.mojo:5066)
# signature; enqueues the producer/stitch pair. Dispatches the runtime `_depth`
# to a compile-time `Depth` specialization over multiples of WARP_SIZE.
# ===-------------------------------------------------------------------=== #
def naive_fa_decode_apple[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
](
    q: LayoutTensor[mut=False, address_space=.GENERIC, ...],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    output: LayoutTensor[mut=True, output_type, address_space=.GENERIC, ...],
    valid_length: LayoutTensor[mut=False, .uint32, address_space=.GENERIC, ...],
    scale: Float32,
    batch_size: Int,
    max_prompt_len: Int,
    max_cache_size: Int,
    num_heads: Int,
    depth: Int,
    group: Int,
    ctx: DeviceContext,
    sink_weights: OptionalReg[
        LayoutTensor[
            mut=False, q.dtype, Layout.row_major(UNKNOWN_VALUE), ImmutAnyOrigin
        ]
    ] = None,
) raises:
    """Host launcher for the Apple split-K decode attention pair (decode-only).

    Parameters:
        output_type: The element type of the `output` tensor (inferred).
            Unused by the kernels; mirrors `mha_gpu_naive` for dispatch
            uniformity.
        k_t: The `MHAOperand` type of the key cache operand (inferred).
        v_t: The `MHAOperand` type of the value cache operand (inferred).
        mask_t: The `MHAMask` functor type applied to attention scores
            (inferred).
        ragged: Whether sequences are ragged with variable lengths and
            row offsets in `valid_length` (defaults to `False`).
        sink: Whether attention sink is enabled, pre-seeding split 0
            with per-head sink weights (defaults to `False`).
        _use_valid_length: Whether to use `valid_length` for KVCache
            decode as per-sequence query lengths (defaults to `False`).
        _is_cache_length_accurate: Whether the cache length equals the
            query length, so no new-token KV is added (defaults to
            `False`).

    Args:
        q: The query tensor; one token per sequence (decode).
        k: The key cache operand implementing the `MHAOperand` contract.
        v: The value cache operand implementing the `MHAOperand`
            contract.
        mask_functor: The mask functor applied to each attention score.
        output: The output tensor; written by the stitch kernel with
            the normalized attention output.
        valid_length: Per-sequence row offsets or query lengths
            (`uint32`); meaning depends on `ragged` and
            `_use_valid_length`.
        scale: The softmax scale factor applied to `Q.K^T` scores.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum prompt length; the dense decode path's
            query length.
        max_cache_size: Full key count for the dense decode path (the K
            tensor's seq dim).
        num_heads: Number of query attention heads.
        depth: Head dimension; must be a multiple of `WARP_SIZE` and
            at most `NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM`.
        group: Number of query heads per KV head (GQA group size).
        ctx: The device context used to enqueue kernels and allocate
            partial buffers.
        sink_weights: Per-head sink weights (shape `[num_heads]`); read
            only when `sink` is `True` (defaults to `None`).
    """
    # No `is_apple_gpu()` assert here — this launcher compiles for the host
    # target, where that target-query is always False. The Apple gate is the
    # caller's (`has_apple_gpu_accelerator()` in dispatch).
    comptime q_type = q.dtype

    var num_keys = max_cache_size

    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    debug_assert(
        naive_fa_decode_apple_supports_depth(depth),
        (
            "naive_fa_decode_apple requires a depth that"
            " naive_fa_decode_apple_supports_depth accepts; the dispatcher must"
            " gate unsupported head dims to mha_gpu_naive"
        ),
    )

    comptime p_type = get_accum_type[q_type]()

    comptime SplitSize = 32  # per-partition KV span

    # Uniform (not per-sequence) alloc avoids a device->host cache_lengths sync
    # each step. The `+ max_prompt_len` covers the producer's `cur_cache_len`
    # span — without it the alloc is one partition short when the cache length
    # is an exact multiple of SplitSize.
    var partition_keys: Int
    comptime if _is_cache_length_accurate:
        partition_keys = max_cache_size
    else:
        partition_keys = max_cache_size + max_prompt_len
    var num_partitions = ceildiv(partition_keys, SplitSize)

    var o_partial_n = batch_size * num_heads * depth * num_partitions
    var ml_partial_n = batch_size * num_heads * num_partitions
    var o_partial_dev = ctx.enqueue_create_buffer[p_type](o_partial_n)
    var m_partial_dev = ctx.enqueue_create_buffer[p_type](ml_partial_n)
    var l_partial_dev = ctx.enqueue_create_buffer[p_type](ml_partial_n)

    # Flat 1D TileTensor views over the q/output/partial buffers. The kernels
    # bake the per-(batch, head, split, depth) offset into a linear index
    # (`_o_idx`/`_ml_idx`) and the BSHD/ragged q/out offset, so the flat views
    # just carry the storage handles with TileTensor typing (no raw pointers /
    # DeviceBuffer-as-pointer inside the kernels).
    var q_flat = TileTensor(
        q.ptr.as_imm().as_unsafe_any_origin(),
        row_major(Coord(Int(q.size()))),
    )
    var output_flat = TileTensor(
        output.ptr.as_unsafe_any_origin(),
        row_major(Coord(Int(output.size()))),
    )
    var valid_length_flat = TileTensor(
        valid_length.ptr.as_imm().as_unsafe_any_origin(),
        row_major(Coord(Int(valid_length.size()))),
    )
    var o_partial_t = TileTensor(o_partial_dev, row_major(Coord(o_partial_n)))
    var m_partial_t = TileTensor(m_partial_dev, row_major(Coord(ml_partial_n)))
    var l_partial_t = TileTensor(l_partial_dev, row_major(Coord(ml_partial_n)))
    var o_partial_imm = o_partial_t.as_immut()
    var m_partial_imm = m_partial_t.as_immut()
    var l_partial_imm = l_partial_t.as_immut()

    # Sink weights: a nullable `OptionalReg[TileTensor]` passed by value (NOT a
    # dangling `UnsafePointer` -- KB `unsafepointer-is-non-nullable`). When
    # sink=False this is None and never read (the seed is comptime-gated on
    # `sink` in the producer). The per-head [num_heads] tensor is converted to
    # a TileTensor so the kernel stays TileTensor-only.
    var sink_layout_val = row_major(Coord(num_heads))
    comptime SinkTile = TileTensor[
        q_type, type_of(sink_layout_val), ImmutAnyOrigin
    ]
    var sink_tile: OptionalReg[SinkTile]
    comptime if sink:
        var sw = sink_weights.value()
        sink_tile = OptionalReg[SinkTile](
            SinkTile(
                sw.ptr.as_imm().as_unsafe_any_origin(),
                sink_layout_val,
            )
        )
    else:
        sink_tile = None

    # The producer needs the head dim at compile time (EPL = Depth //
    # WARP_SIZE), so specialize one kernel per supported `depth` and select at
    # runtime. The dispatcher guarantees `depth` matches one branch.
    comptime MAX_D_STEPS = NAIVE_FA_DECODE_APPLE_MAX_HEAD_DIM // WARP_SIZE
    comptime for di in range(1, MAX_D_STEPS + 1):
        comptime D = di * WARP_SIZE
        # gate non-power-of-two `di` instantiating an invalid vec type
        comptime if naive_fa_decode_apple_supports_depth(D):
            if depth == D:
                comptime core_kernel = naive_fa_decode_apple_core[
                    q_type,
                    output_type,
                    p_type,
                    k_t,
                    v_t,
                    mask_t,
                    type_of(o_partial_t).LayoutType,
                    type_of(q_flat).LayoutType,
                    type_of(valid_length_flat).LayoutType,
                    type_of(sink_layout_val),
                    ragged=ragged,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    Depth=D,
                    SplitSize=SplitSize,
                    OPartialEngine=o_partial_t.Engine,
                    MPartialEngine=m_partial_t.Engine,
                    LPartialEngine=l_partial_t.Engine,
                    QEngine=q_flat.Engine,
                    VLEngine=valid_length_flat.Engine,
                    SinkEngine=SinkTile.Engine,
                ]
                ctx.enqueue_function[core_kernel](
                    o_partial_t,
                    m_partial_t,
                    l_partial_t,
                    q_flat,
                    k,
                    v,
                    mask_functor,
                    valid_length_flat,
                    sink_tile,
                    scale,
                    Int32(batch_size),
                    Int32(max_prompt_len),
                    Int32(max_cache_size),
                    Int32(num_heads),
                    Int32(depth),
                    Int32(group),
                    Int32(num_partitions),
                    grid_dim=(num_partitions, batch_size, num_heads),
                    block_dim=WARP_SIZE,
                )

    comptime stitch_kernel = naive_fa_decode_apple_stitch[
        output_type,
        p_type,
        k_t,
        v_t,
        mask_t,
        type_of(output_flat).LayoutType,
        type_of(o_partial_imm).LayoutType,
        type_of(valid_length_flat).LayoutType,
        ragged=ragged,
        sink=sink,
        _use_valid_length=_use_valid_length,
        _is_cache_length_accurate=_is_cache_length_accurate,
        SplitSize=SplitSize,
        OutEngine=output_flat.Engine,
        OPartialEngine=o_partial_imm.Engine,
        MPartialEngine=m_partial_imm.Engine,
        LPartialEngine=l_partial_imm.Engine,
        VLEngine=valid_length_flat.Engine,
    ]
    ctx.enqueue_function[stitch_kernel](
        output_flat,
        o_partial_imm,
        m_partial_imm,
        l_partial_imm,
        k,
        valid_length_flat,
        Int32(max_prompt_len),
        Int32(max_cache_size),
        Int32(num_heads),
        Int32(depth),
        Int32(num_partitions),
        grid_dim=(num_heads, batch_size),
        block_dim=depth,
    )

    # Keep the partial buffers alive until both kernels have been enqueued.
    _ = o_partial_dev^
    _ = m_partial_dev^
    _ = l_partial_dev^
