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
"""Apple (Metal) M5 MMA-based flash-attention (FA2) prefill kernel.

Two `MmaOpApple` 16x16 simdgroup GEMMs (Q.K^T and P.V) sandwiching an online
softmax, for Apple silicon prefill (`compute_capability == 5`), paged KV via
`MHAOperand`, BF16/FP16 storage with FP32 accumulation.

4 simdgroups share one threadgroup with no threadgroup memory and no
`barrier()`; they are co-resident only to share KV reads through the L2, which
beat both a single-simdgroup launch and an SMEM-staged variant. Each simdgroup
independently owns an `Sq x depth` query-row tile for one `(batch, head)` and
streams the KV range online in `Sk`-wide tiles, computing QK one 16-key
column-block at a time. Each column-block resolves its own page-aligned KV
sub-tile, so KV need only satisfy `page_size == 0` or `page_size % 16 == 0`;
other page sizes fall back to `mha_gpu_naive` via the dispatcher.

The per-row max/sum over a score tile whose rows are scattered across the MMA
fragment uses an XOR-butterfly reduction; `air.simd_sum` (the decode reduction)
would reduce all 32 lanes and mix rows.

P.V keeps P register-resident: the QK score fragment, cast to `q_type`, is fed
directly as the P.V A-operand -- the C-output and A-input fragment layouts are
the identical `_apple_frag_layout` bit-scatter, so P never round-trips memory.
Q, K, and V load from memory each tile.

`flash_attention_dispatch` selects `fa_prefill_apple` for Apple prefill by
default; set `MODULAR_ENABLE_APPLE_FA_PREFILL=0` to fall back to `mha_gpu_naive`.
"""

from std.collections import OptionalReg
from max.gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    warp_id,
)
from max.gpu.sync import syncwarp
from max.gpu.compute.arch.mma_apple import (
    _apple_frag_layout,
    _mma_apple_transposable,
)
from max.gpu.host import DeviceContext
from max.gpu.primitives.warp import shuffle_xor
from std.math import ceildiv, exp2
from std.math.constants import log2e
from std.os.env import getenv
from std.sys.info import align_of, size_of
from std.sys.intrinsics import llvm_intrinsic
from std.utils.index import Index
from std.utils.numerics import get_accum_type


from layout import (
    UNKNOWN_VALUE,
    Idx,
    Layout,
    LayoutTensor,
    TensorEngine,
    TileTensor,
)
from layout.coord import Coord
from layout.tile_layout import (
    Layout as TileLayout,
    RowMajorLayout,
    TensorLayout,
    row_major,
)

from linalg.arch.apple.mma import MmaOpApple

from nn.attention.mha_mask import CausalMask, MHAMask, TileMaskStatus
from nn.attention.mha_operand import MHAOperand

comptime NEG_INF = Float32(-3.0e38)


@always_inline
def _threadgroup_barrier_mem_none():
    """Threadgroup execution barrier with no memory fence.

    `max.gpu.barrier()` hardcodes `mem_threadgroup` (a full LDS fence); this
    emits `llvm.air.wg.barrier` with the fence cleared. Apple/M5-only.

    Do NOT delete as vestigial: though a runtime no-op for this no-SMEM,
    independent-per-simdgroup kernel, empirically it prevents some instruction
    reordering and removing it regresses ~40%. See `_prevent_inst_reorder`, which
    keeps the fence alive in codegen without executing it.
    """
    llvm_intrinsic["llvm.air.wg.barrier", NoneType](Int32(0), Int32(1))


@always_inline
def _prevent_inst_reorder[warp_scope: Bool = False](opaque_len: Int):
    """Never-executing scheduler fence -- no runtime cost, no kernel arg.

    Emits a barrier inside an always-false branch (`opaque_len < 0`) so it stays
    in codegen but never runs. Empirically this prevents some instruction
    reordering and avoids a ~40% regression; removing it, or changing its scope
    or placement, brings the regression back.

    Parameters:
        warp_scope: If True, a simdgroup (warp) barrier for points with divergent
            per-simdgroup control flow; if False, the stronger threadgroup
            barrier for uniform points.

    Args:
        opaque_len: Runtime length used as the always-false guard condition.
    """
    if opaque_len < 0:
        comptime if warp_scope:
            syncwarp()
        else:
            _threadgroup_barrier_mem_none()


# Max head dim (multiple of 16) the prefill kernel handles; larger head dims
# fall back to mha_gpu_naive via the dispatcher.
comptime FA_PREFILL_APPLE_MAX_HEAD_DIM = 256

# Per-simdgroup tile: Sq = NUM_M_MMAS*16 query rows, Sk = NUM_N_MMAS*16 KV cols.
# With many simdgroups co-resident, the per-simdgroup register footprint
# (NUM_M_MMAS*((depth/16) + NUM_N_MMAS) fp32 fragments) binds occupancy: a narrow
# 1x2 tile keeps them resident while a wider score tile collapses occupancy.
# NUM_N_MMAS=2 measured fastest.
comptime DEFAULT_NUM_M_MMAS = 1
comptime DEFAULT_NUM_N_MMAS = 2
comptime MMA_DIM = 16


# ===-------------------------------------------------------------------=== #
# Online softmax over the MmaOpApple accumulator (register-resident).
# ===-------------------------------------------------------------------=== #
# The M5 16x16 simdgroup fragment owns TWO rows per lane (`rb` and `rb+8`), so
# every per-row quantity is width-2 and the cross-lane reduction is an XOR
# butterfly over the row-sharing lanes. The `(m, l)` state lives in kernel-local
# `Array[Float32, _SOFTMAX_FRAG_ROWS]`s that the `_softmax_*` free functions
# take and mutate.

comptime _SOFTMAX_FRAG_ROWS = 2  # M5 lane owns rows {rb, rb+8} of each 16x16 subtile


@always_inline
def _softmax_seed_sink(
    mut sm_m: Array[Float32, _SOFTMAX_FRAG_ROWS],
    mut sm_l: Array[Float32, _SOFTMAX_FRAG_ROWS],
    sink_weight: Float32,
):
    """Pre-seed `(m, l)` with a sink token's contribution (init-state trick).

    Equivalent to a virtual "tile -1" holding one score `sink_weight`, entering
    `1` into the running sum so the hot loop stays sink-agnostic. `sink_weight`
    arrives already in base-2 units and unscaled (the naive reference compares
    the raw sink weight against the post-scale row max). Both owned rows
    {rb, rb+8} get the same seed.
    """
    sm_m[0] = sink_weight
    sm_m[1] = sink_weight
    sm_l[0] = Float32(1)
    sm_l[1] = Float32(1)


@always_inline
def _softmax_row_max[
    num_n_mmas: Int
](
    scores: MmaOpApple[.float32, DType.float32, 1, num_n_mmas].AccumType,
) -> Array[Float32, _SOFTMAX_FRAG_ROWS]:
    """Full-row max over the single score row-block (across all `ni` cols).

    Returns `[rb_max, rb8_max]` for the two rows this lane owns. The per-fragment
    max is combined across the `num_n_mmas` column fragments in registers before a
    single XOR butterfly reduces the row-sharing lanes.
    """
    var r = Array[Float32, _SOFTMAX_FRAG_ROWS](fill=NEG_INF)
    comptime for ni in range(num_n_mmas):
        var frag = scores[ni]
        var r0 = max(max(frag[0], frag[1]), max(frag[2], frag[3]))
        var r1 = max(max(frag[4], frag[5]), max(frag[6], frag[7]))
        r[0] = max(r[0], r0)
        r[1] = max(r[1], r1)
    # One butterfly over the 4 row-sharing lanes {1, 8}, per row.
    r[0] = max(r[0], shuffle_xor(r[0], UInt32(1)))
    r[0] = max(r[0], shuffle_xor(r[0], UInt32(8)))
    r[1] = max(r[1], shuffle_xor(r[1], UInt32(1)))
    r[1] = max(r[1], shuffle_xor(r[1], UInt32(8)))
    return r^


@always_inline
def _softmax_update[
    num_n_mmas: Int, out_num_n_mmas: Int
](
    mut sm_m: Array[Float32, _SOFTMAX_FRAG_ROWS],
    mut sm_l: Array[Float32, _SOFTMAX_FRAG_ROWS],
    mut scores: MmaOpApple[
        DType.float32, DType.float32, 1, num_n_mmas
    ].AccumType,
    mut output: MmaOpApple[
        DType.float32, DType.float32, 1, out_num_n_mmas
    ].AccumType,
):
    """One online-softmax step over the masked score tile `scores`.

    In place: `scores` becomes the (unnormalized) probabilities `P`, `output` is
    rescaled by the running-max correction, and `m`/`l` advance. The output's
    column count (`out_num_n_mmas = depth / 16`) differs from the score's
    (`num_n_mmas = Sk / 16`); both share the single 16-row block and the same
    per-lane `(rb, rb+8)` row ownership.

    Base-2 throughout (`exp2`); scores carry the log2e-folded scale:
      1. row max `m_tile`  -> `m_new = max(m, m_tile)`
      2. correction `alpha = exp2(m - m_new)`
      3. `P = exp2(scores - m_new)` (OOB/masked entries are NEG_INF -> 0)
      4. `l = l*alpha + rowsum(P)`, `m = m_new`
      5. `output *= alpha`
    """
    var m_tile = _softmax_row_max[num_n_mmas](scores)
    var m_new = Array[_, _SOFTMAX_FRAG_ROWS](
        fill_with_unrolled=lambda [i: Int]() -> Float32: max(sm_m[i], m_tile[i])
    )
    # A still-fully-masked row keeps its running max at the finite NEG_INF floor
    # (finite, so the subtraction never NaNs), and resolves once its first real
    # key arrives in a later tile.
    var alpha = Array[_, _SOFTMAX_FRAG_ROWS](
        fill_with_unrolled=lambda [i: Int]() -> Float32: exp2(
            sm_m[i] - m_new[i]
        )
    )

    # Accumulate `l` from each P fragment while it is still register-live (vs a
    # second pass re-reading the written-back scores), shortening the softmax
    # dependency chain. One butterfly after the loop does the cross-lane reduction.
    var l_acc = Array[Float32, _SOFTMAX_FRAG_ROWS](fill=Float32(0))
    comptime for ni in range(num_n_mmas):
        var p = scores[ni]
        var p_lo = exp2(p.slice[4, offset=0]() - SIMD[.float32, 4](m_new[0]))
        var p_hi = exp2(p.slice[4, offset=4]() - SIMD[.float32, 4](m_new[1]))
        scores[ni] = p_lo.join(p_hi)
        l_acc[0] = l_acc[0] + (p_lo[0] + p_lo[1] + p_lo[2] + p_lo[3])
        l_acc[1] = l_acc[1] + (p_hi[0] + p_hi[1] + p_hi[2] + p_hi[3])
    l_acc[0] = l_acc[0] + shuffle_xor(l_acc[0], UInt32(1))
    l_acc[0] = l_acc[0] + shuffle_xor(l_acc[0], UInt32(8))
    l_acc[1] = l_acc[1] + shuffle_xor(l_acc[1], UInt32(1))
    l_acc[1] = l_acc[1] + shuffle_xor(l_acc[1], UInt32(8))
    sm_l[0] = sm_l[0] * alpha[0] + l_acc[0]
    sm_l[1] = sm_l[1] * alpha[1] + l_acc[1]
    sm_m[0] = m_new[0]
    sm_m[1] = m_new[1]

    comptime for ni in range(out_num_n_mmas):
        var o = output[ni]
        var o_lo = o.slice[4, offset=0]() * SIMD[.float32, 4](alpha[0])
        var o_hi = o.slice[4, offset=4]() * SIMD[.float32, 4](alpha[1])
        output[ni] = o_lo.join(o_hi)


@always_inline
def _softmax_normalize[
    out_num_n_mmas: Int
](
    sm_l: Array[Float32, _SOFTMAX_FRAG_ROWS],
    mut output: MmaOpApple[
        DType.float32, DType.float32, 1, out_num_n_mmas
    ].AccumType,
):
    """Final epilogue: divide each output row by its running denominator `l`.

    `sm_l[0]` normalizes the `rb` row, `sm_l[1]` the `rb+8` row.

    A fully-masked row has `l == 0` and `O == 0`, so `O/l` is `0/0 = NaN`; per the
    FlashAttention convention we clamp `inv` to 0 there and emit 0. Reachable with
    a sliding-window / non-causal mask when a query row falls outside both its
    window and the key range; causal always attends its own position and the sink
    seed keeps `l >= 1`, so the guard is a no-op there.
    """
    var inv = Array[_, _SOFTMAX_FRAG_ROWS](
        fill_with_unrolled=lambda [i: Int]() -> Float32: (
            Float32(1) / sm_l[i] if sm_l[i] > Float32(0) else Float32(0)
        )
    )
    comptime for ni in range(out_num_n_mmas):
        var o = output[ni]
        var o_lo = o.slice[4, offset=0]() * SIMD[.float32, 4](inv[0])
        var o_hi = o.slice[4, offset=4]() * SIMD[.float32, 4](inv[1])
        output[ni] = o_lo.join(o_hi)


# ===-------------------------------------------------------------------=== #
# Prefill kernel: one simdgroup per (q_tile, head, batch).
# Grid (num_q_tiles, num_heads, batch); block = WARP_SIZE.
# ===-------------------------------------------------------------------=== #
def fa_prefill_apple_core[
    q_type: DType,
    output_type: DType,
    p_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    q_layout: TensorLayout,
    output_layout: TensorLayout,
    valid_length_layout: TensorLayout,
    sink_layout: TensorLayout,
    output_engine: TensorEngine,
    q_engine: TensorEngine,
    valid_length_engine: TensorEngine,
    sink_engine: TensorEngine,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    *,
    Depth: Int,
    NumNMmas: Int,
    NumSimdgroups: Int = 1,
](
    output: TileTensor[
        output_type, output_layout, MutAnyOrigin, Engine=output_engine
    ],
    q: TileTensor[q_type, q_layout, ImmutAnyOrigin, Engine=q_engine],
    k: k_t,
    v: v_t,
    mask_functor: mask_t,
    valid_length: TileTensor[
        .uint32, valid_length_layout, ImmutAnyOrigin, Engine=valid_length_engine
    ],
    sink_weights: OptionalReg[
        TileTensor[q_type, sink_layout, ImmutAnyOrigin, Engine=sink_engine]
    ],
    scale: Float32,
    batch_size: Int32,
    max_prompt_len: Int32,
    max_cache_size: Int32,
    num_heads: Int32,
    depth: Int32,
    group: Int32,
):
    """Per-simdgroup core of the prefill kernel.

    Each simdgroup owns an `Sq x Depth` query-row tile (`Sq = MMA_DIM = 16`, one
    MMA tile) for one `(batch, head)` and streams the KV range online (see the
    module docstring for the geometry).
    P.V runs on the M5 16x16 simdgroup MMA via `_mma_apple_transposable`, with the
    QK score fragment fed directly as the PV A-operand (register-resident P).

    Parameters:
        q_type: The dtype of the query tensor `q`, also used as the
            P-fragment cast type in the PV matmul.
        output_type: The dtype of the output tensor `output`, applied as
            a final cast on store.
        p_type: The accumulation type for the softmax probabilities,
            derived from `get_accum_type[q_type]` (`Float32`).
        k_t: The `MHAOperand` type of the paged key cache, determining
            key dtype and page size.
        v_t: The `MHAOperand` type of the paged value cache, determining
            value dtype.
        mask_t: The `MHAMask` type of the attention mask functor,
            selecting causal vs non-causal masking at compile time.
        q_layout: The `TensorLayout` of the flattened query `TileTensor`.
        output_layout: The `TensorLayout` of the flattened output
            `TileTensor`.
        valid_length_layout: The `TensorLayout` of the flattened
            `valid_length` `TileTensor`.
        sink_layout: The `TensorLayout` of the sink weights `TileTensor`.
        output_engine: The `TensorEngine` of the `output` `TileTensor`.
        q_engine: The `TensorEngine` of the `q` `TileTensor`.
        valid_length_engine: The `TensorEngine` of the `valid_length`
            `TileTensor`.
        sink_engine: The `TensorEngine` of the sink weights `TileTensor`.
        ragged: If True, `valid_length` is a cumulative offset buffer
            over variable-length sequences in the batch (defaults to
            False).
        sink: If True, pre-seed the softmax state from per-head sink
            weights (defaults to False).
        _use_valid_length: If True, read each batch's query length from
            `valid_length[batch_id]` (defaults to False).
        _is_cache_length_accurate: If True, the total attention length
            equals the query length (no cached prefix); if False, add the
            key operand's per-batch cache length (defaults to False).
        Depth: The compile-time head dimension; must be a multiple of 16
            and at most `FA_PREFILL_APPLE_MAX_HEAD_DIM`.
        NumNMmas: Number of 16-key column-blocks per KV tile, setting the
            tile width `Sk = NumNMmas * 16`.
        NumSimdgroups: Number of simdgroups co-resident per threadgroup,
            controlling per-launch occupancy (defaults to 1).

    Args:
        output: The output tensor, written in the same flattened BSHD
            layout as `q`.
        q: The query tensor in flattened BSHD layout (batch, seq, head,
            depth).
        k: The paged key cache operand.
        v: The paged value cache operand.
        mask_functor: The attention mask functor applied to each score
            tile.
        valid_length: Per-batch `uint32` sequence lengths or cumulative
            offsets, depending on `ragged` and `_use_valid_length`.
        sink_weights: Optional per-head sink weights indexed by head id;
            dereferenced only when `sink` is True.
        scale: The softmax scale factor applied to QK score products.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum query length across the batch, in tokens.
        max_cache_size: Maximum KV cache length across the batch, in
            tokens.
        num_heads: Number of query attention heads.
        depth: Head dimension; must match the compile-time `Depth`
            parameter.
        group: Number of query heads per KV head (GQA group size).

    Constraints:
        `Depth % 16 == 0`, and either `k.page_size == 0` or `k.page_size % 16 == 0`
        so a 16-key sub-tile never crosses a page boundary. Other page sizes are
        gated to `mha_gpu_naive` by the dispatcher.
    """
    var _batch_size = Int(batch_size)
    var _max_prompt_len = Int(max_prompt_len)
    var _max_cache_size = Int(max_cache_size)
    var _num_heads = Int(num_heads)
    var _depth = Int(depth)
    var _group = Int(group)
    comptime assert Depth % MMA_DIM == 0, "Depth must be a multiple of 16"
    comptime assert k_t.page_size == 0 or k_t.page_size % MMA_DIM == 0, (
        "fa_prefill_apple_core requires contiguous KV (page_size == 0) or a"
        " page size that is a multiple of MMA_DIM=16 (so a 16-row sub-tile"
        " never crosses a page boundary)"
    )
    comptime SQ = MMA_DIM
    comptime SK = NumNMmas * MMA_DIM
    comptime DEPTH_MMAS = Depth // MMA_DIM
    debug_assert(_depth == Depth, "runtime _depth must match comptime Depth")
    # 3D grid (num_q_tiles, num_heads, batch) read directly from block_idx. The x
    # reversal schedules the highest q-tile (most causal KV work) first so the
    # longest poles don't form the makespan tail; light low-row tiles backfill the
    # grid drain.
    var num_q_tiles = ceildiv(_max_prompt_len, NumSimdgroups * SQ)
    var q_tile_id = (num_q_tiles - 1) - Int(block_idx.x)
    var head_id = Int(block_idx.y)
    var batch_id = Int(block_idx.z)
    var kv_head = head_id // _group
    var sg = Int(warp_id())  # this simdgroup's slot in the threadgroup
    var lane = Int(lane_id())
    var rb_cb = _apple_frag_layout(lane)
    var rb = rb_cb[0]
    var cb = rb_cb[1]

    # Offset math — mirror `_bmm0_bs` / `_bmm1_bs` (mha.mojo).
    var seq_start: Int
    var cur_query_len: Int
    var q_offset: Int
    var out_offset: Int
    var cur_cache_len: Int
    comptime if ragged:
        seq_start = Int(valid_length[batch_id])
        var seq_end = Int(valid_length[batch_id + 1])
        cur_query_len = seq_end - seq_start
        q_offset = _depth * (seq_start * _num_heads + head_id)
        out_offset = (seq_start * _num_heads + head_id) * _depth
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    elif _use_valid_length:
        seq_start = batch_id
        cur_query_len = Int(valid_length[batch_id])
        q_offset = _depth * (head_id + _num_heads * _max_prompt_len * batch_id)
        out_offset = q_offset
        comptime if _is_cache_length_accurate:
            cur_cache_len = cur_query_len
        else:
            cur_cache_len = k.cache_length(batch_id) + cur_query_len
    else:
        seq_start = batch_id
        cur_query_len = _max_prompt_len
        q_offset = _depth * (head_id + _num_heads * _max_prompt_len * batch_id)
        out_offset = q_offset
        cur_cache_len = _max_cache_size

    # Each simdgroup owns its own `SQ` contiguous rows within the threadgroup's
    # `NumSimdgroups*SQ`-row span. A simdgroup whose rows are entirely past the
    # sequence must NOT early-return: the threadgroup barriers require every
    # simdgroup to reach them equally, so an inactive one flows through the loop
    # with a clamped q-load and NEG_INF-masked scores, contributing nothing.
    var q_row0 = q_tile_id * (NumSimdgroups * SQ) + sg * SQ
    # QK reads `q_tile` unconditionally; an `sg_active` branch kept the QK path
    # conditional and inflated spill (~9% latency). Inactive simdgroups clamp to
    # row 0.
    var q_load_row0 = q_row0 if q_row0 < cur_query_len else 0

    # Q/output row stride is _num_heads*_depth (the BSHD / ragged head stride);
    # _depth is contiguous. Build this simdgroup's SQ-row TileTensor view by
    # baking the ragged/BSHD offset + the q_row0 row offset into the tile base,
    # then descending to 16x16 sub-tiles with `.tile[16, 16]()`. The score MMA
    # reads Q from this view; no raw pointer arithmetic in the QK loop.
    # Q tile shape (SQ, Depth) -- extents comptime, the row (token) stride a
    # runtime `Int` (_num_heads*_depth) and the inner _depth stride a comptime 1.
    # Build the layout value, then derive the TileTensor type via `type_of`
    # (the matmul idiom for a runtime-stride layout).
    var q_row_stride = _num_heads * _depth
    var q_layout_val = TileLayout(
        Coord(Idx[SQ], Idx[Depth]), Coord(q_row_stride, Idx[1])
    )
    var q_tile = TileTensor[q_type, type_of(q_layout_val), ImmutAnyOrigin](
        ptr=q.ptr + q_offset + q_load_row0 * q_row_stride,
        layout=q_layout_val,
    )

    # KV token stride (within a page): difference of two consecutive key
    # pointers. Robust to the ragged-3D vs dense-4D head layout, and for paged
    # KV it is the IN-PAGE token stride: tokens 0 and 1 always share a page
    # (page_size >= 16 > 1), and the paged blocks layout
    # [total_blocks, page_size, _num_heads, head_size] makes consecutive in-page
    # tokens contiguous with a constant stride (_num_heads*head_size). We resolve
    # the page per 16-row sub-tile below, so this in-page stride is the only
    # stride a sub-tile load ever needs. The stride feeds the sub-tile
    # TileTensor's layout (`kv_layout_val`), so `MmaOpApple` reads it via
    # `_row_stride` -- no manual `ptr + row*stride` arithmetic in the load.
    var k_base0 = k.block_paged_ptr[1](
        UInt32(batch_id), UInt32(0), UInt32(kv_head), 0
    )
    var k_base1 = k.block_paged_ptr[1](
        UInt32(batch_id), UInt32(1), UInt32(kv_head), 0
    )
    # `Int(ptr)` is a byte address; convert the byte diff to an element stride.
    comptime kv_elt_size = size_of[Scalar[k_t.dtype]]()
    var kv_row_stride = (Int(k_base1) - Int(k_base0)) // kv_elt_size

    # KV sub-tile layout (16 keys x Depth), reused for every K/V sub-tile.
    var kv_layout_val = TileLayout(
        Coord(Idx[MMA_DIM], Idx[Depth]), Coord(kv_row_stride, Idx[1])
    )

    # ScoreMma loads K in its native dtype; q and k share the model dtype in
    # attention, so `k_t.dtype == q_type` in practice.
    comptime ScoreMma = MmaOpApple[
        DType.float32,
        q_type,
        1,
        1,
        b_type=k_t.dtype,
        transpose_b=True,
    ]
    # Full Sk-wide score storage (NumNMmas col-blocks), filled one page-resolved
    # 16-key column-block at a time.
    comptime ScoreAccum = MmaOpApple[
        DType.float32,
        q_type,
        1,
        NumNMmas,
        b_type=k_t.dtype,
        transpose_b=True,
    ]
    comptime OutMma = MmaOpApple[.float32, q_type, 1, DEPTH_MMAS]

    # QK MMA op (lane-offset setup is loop-invariant).
    var score_mma = ScoreMma()
    var out_mma = OutMma()
    var output_accum = OutMma.zero_accum()
    # Online-softmax state, seeded m = NEG_INF, l = 0 (see `_softmax_*`).
    var sm_m = Array[Float32, _SOFTMAX_FRAG_ROWS](fill=NEG_INF)
    var sm_l = Array[Float32, _SOFTMAX_FRAG_ROWS](fill=Float32(0))

    comptime if sink:
        # Pre-seed (m, l) from the per-head sink weight (indexed by head_id). The
        # naive reference compares the raw, unscaled sink weight against the
        # post-scale row max, so seed the raw weight (no `scale`) in base-2 units
        # (`* log2e`) to match the log2e-folded scores. The deref is comptime-gated
        # on `sink`, so None is never reached.
        var sw = rebind[Scalar[q_type]](sink_weights.value()[head_id])
        _softmax_seed_sink(sm_m, sm_l, sw.cast[.float32]() * Float32(log2e))

    # Fold log2e into the scale so per-element scaling lands scores directly in
    # the units `exp2` consumes. Exact here: the supported masks only pass-through
    # or set NEG_INF, adding no score-space bias.
    var scale_l2 = scale * Float32(log2e)

    var score_row_base = q_row0 + cur_cache_len - cur_query_len

    # Threadgroup-uniform skip/break base: the tile skip below must be identical
    # across simdgroups so they reach the barriers in lockstep, so it is decided
    # over the whole `NumSimdgroups*SQ`-row span. Tiles fully masked for only some
    # simdgroups are still processed; those rows get NEG_INF per-element.
    var tg_score_row_base = (
        q_tile_id * (NumSimdgroups * SQ) + cur_cache_len - cur_query_len
    )

    var valid_q = min(SQ, cur_query_len - q_row0)
    # Whole-tile query-row validity, loop-invariant (q_row0 fixed): true except on
    # the last partial Q block, letting the mask/scale and store hot paths drop the
    # per-element query-row bound.
    var q_tile_full = valid_q == SQ

    # Causal (monotonic-in-k) mask: once a KV tile is fully masked every higher
    # `kv0` is too, so `break`. Non-monotonic masks (chunked, sliding-window)
    # full-mask both ends, so they must `continue` on a FULL_MASK tile. Gated
    # comptime so NullMask compiles the check away and only CausalMask early-exits.
    comptime causal_monotonic = mask_t == CausalMask

    # Stream the KV range in SK-wide tiles.
    for kv0 in range(0, cur_cache_len, SK):
        # A FULL_MASK tile is a no-op on the softmax state (all scores NEG_INF:
        # m, l, and O all unchanged), so skipping it is equivalent to processing
        # it. Decided over the whole threadgroup row span so the skip/break is
        # identical across simdgroups (barrier lockstep).
        var tg_tile_status = mask_functor.status[element_type=DType.uint32](
            UInt32(batch_id),
            Index[dtype=DType.uint32](tg_score_row_base, kv0),
            Index[dtype=DType.uint32](NumSimdgroups * SQ, SK),
        )
        if tg_tile_status == TileMaskStatus.FULL_MASK:
            comptime if causal_monotonic:
                break
            else:
                continue

        # Per-simdgroup status drives this simdgroup's fast-path / mask decision.
        var tile_status = mask_functor.status[element_type=DType.uint32](
            UInt32(batch_id),
            Index[dtype=DType.uint32](score_row_base, kv0),
            Index[dtype=DType.uint32](SQ, SK),
        )

        # Whole-tile KV validity: every key in [kv0, kv0+SK) is < cur_cache_len.
        # Only the last KV tile is partial.
        var sk_tile_full = kv0 + SK <= cur_cache_len

        # --- QK: scores[Sq, Sk] = Q @ K^T (transpose_b), one 16-key column-block
        # at a time. Each block resolves its own page-aligned KV sub-tile (never
        # crossing a page boundary, per the page_size % 16 assert), so the in-page
        # `kv_row_stride` is valid. QK runs unconditionally; an inactive
        # simdgroup's clamped row-0 reads produce garbage scores that are masked to
        # NEG_INF below and never stored.
        var scores = ScoreAccum.zero_accum()
        comptime for ni in range(NumNMmas):
            var k_tile = k.block_paged_tile[1](
                UInt32(batch_id),
                UInt32(kv0 + ni * MMA_DIM),
                UInt32(kv_head),
                kv_layout_val,
            )
            var score_col = ScoreMma.zero_accum()
            # Width-8 Q+K load per BK=32 strip (no preload, so nothing is held
            # across the loop), split low4/high4 -- halves load taps vs width-4.
            comptime q_align = align_of[Scalar[q_type]]()
            comptime k_align = align_of[Scalar[k_t.dtype]]()
            comptime if DEPTH_MMAS % 2 == 0:
                comptime for sp in range(DEPTH_MMAS // 2):
                    var qlo = (
                        q_tile.ptr + rb * q_row_stride + 32 * sp + 2 * cb
                    ).load[width=8, alignment=q_align]()
                    var qhi = (
                        q_tile.ptr + (rb + 8) * q_row_stride + 32 * sp + 2 * cb
                    ).load[width=8, alignment=q_align]()
                    var klo = (
                        k_tile.ptr + rb * kv_row_stride + 32 * sp + 2 * cb
                    ).load[width=8, alignment=k_align]()
                    var khi = (
                        k_tile.ptr + (rb + 8) * kv_row_stride + 32 * sp + 2 * cb
                    ).load[width=8, alignment=k_align]()
                    _mma_apple_transposable(
                        score_col[0],
                        qlo.slice[4, offset=0]().join(qhi.slice[4, offset=0]()),
                        klo.slice[4, offset=0]().join(khi.slice[4, offset=0]()),
                        score_col[0],
                        False,
                        True,
                    )
                    _mma_apple_transposable(
                        score_col[0],
                        qlo.slice[4, offset=4]().join(qhi.slice[4, offset=4]()),
                        klo.slice[4, offset=4]().join(khi.slice[4, offset=4]()),
                        score_col[0],
                        False,
                        True,
                    )
            else:
                comptime for ki in range(DEPTH_MMAS):
                    var qf = score_mma.load_fragment[q_type](
                        q_tile.tile[MMA_DIM, MMA_DIM](0, ki)
                    )
                    var kf = score_mma.load_fragment[k_t.dtype](
                        k_tile.tile[MMA_DIM, MMA_DIM](0, ki)
                    )
                    _mma_apple_transposable(
                        score_col[0], qf, kf, score_col[0], False, True
                    )
            scores[ni] = rebind[type_of(scores[ni])](score_col[0])

        # --- Mask + scale. The mask sets OOB keys/rows to NEG_INF, which is why
        # QK can read K rows past num_keys unguarded (the garbage score is
        # overwritten here); PV cannot rely on this (0 * V_oob poisons the accum)
        # so it bounds its V load instead. A fully in-bounds, unmasked interior
        # tile skips masking entirely (`tile_fast`) -- the common case for causal
        # prefill's lower triangle.
        var tile_fast = (
            q_tile_full
            and sk_tile_full
            and tile_status == TileMaskStatus.NO_MASK
        )
        # Scale first, unconditionally; the mask paths below only select
        # visible-vs-NEG_INF. A NO_MASK interior tile (`tile_fast`) is then done.
        comptime for idx in range(NumNMmas):
            scores[idx] = scores[idx] * scale_l2

        if not tile_fast:
            # Edge and PARTIAL_MASK tiles still need masking. The split is comptime
            # (not a runtime branch) so exactly one arm compiles per mask type --
            # backend DCE of the unused arm is unreliable on Apple.
            comptime if causal_monotonic:
                # Vectorized causal mask: the lane owns two runs of 4 keys sharing
                # one q_idx, and `CausalMask.mask` evaluates `q >= iota(k)`
                # width-wise, so apply it once per row-half over a width-4 slice
                # instead of 8 scalar `mask()` calls. `CausalMask.mask` is pure, so
                # it is safe on OOB keys, overridden by the bounds select.
                comptime for ni in range(NumNMmas):
                    var kbase = kv0 + ni * 16 + cb
                    # Per-key in-bounds mask for the 4 consecutive keys.
                    var keys = SIMD[.int32, 4](Int32(kbase)) + SIMD[
                        DType.int32, 4
                    ](0, 1, 2, 3)
                    var k_ok = keys.lt(Int32(cur_cache_len))
                    var neg = SIMD[.float32, 4](NEG_INF)
                    var lrow_lo = rb
                    var lrow_hi = lrow_lo + 8
                    var s_lo = scores[ni].slice[4, offset=0]()
                    var s_hi = scores[ni].slice[4, offset=4]()
                    var m_lo = mask_functor.mask(
                        Index(
                            batch_id, head_id, score_row_base + lrow_lo, kbase
                        ),
                        s_lo,
                    )
                    var m_hi = mask_functor.mask(
                        Index(
                            batch_id, head_id, score_row_base + lrow_hi, kbase
                        ),
                        s_hi,
                    )
                    var ib_lo = (
                        SIMD[.bool, 4](fill=q_row0 + lrow_lo < cur_query_len)
                        & k_ok
                    )
                    var ib_hi = (
                        SIMD[.bool, 4](fill=q_row0 + lrow_hi < cur_query_len)
                        & k_ok
                    )
                    scores[ni] = ib_lo.select(m_lo, neg).join(
                        ib_hi.select(m_hi, neg)
                    )
            else:
                comptime for ni in range(NumNMmas):
                    var frag = scores[ni]
                    comptime for el in range(8):
                        var lrow = rb + (8 if el > 3 else 0)
                        var lcol = ni * 16 + cb + (el & 3)
                        var key = kv0 + lcol
                        var qrow = q_row0 + lrow
                        if qrow < cur_query_len and key < cur_cache_len:
                            frag[el] = mask_functor.mask(
                                Index(
                                    batch_id,
                                    head_id,
                                    score_row_base + lrow,
                                    key,
                                ),
                                frag[el],
                            )
                        else:
                            frag[el] = NEG_INF
                    scores[ni] = frag

        _softmax_update[NumNMmas, DEPTH_MMAS](sm_m, sm_l, scores, output_accum)

        # Scheduler fence between softmax and P.V (warp-scope, safe under the
        # independent per-simdgroup control flow; see `_prevent_inst_reorder`).
        _prevent_inst_reorder[warp_scope=True](cur_cache_len)

        # --- PV: output += P @ V. P (the QK C-fragment) is fed directly as the
        # MMA A-operand -- same layout, no memory round-trip. Only the V load
        # differs full-vs-partial: a partial last KV tile bounds the load so OOB
        # key-rows zero-fill (else 0 * V_oob poisons the accumulator).
        comptime for ki in range(NumNMmas):
            var p_frag = scores[ki].cast[q_type]()
            var v_tile = v.block_paged_tile[1](
                UInt32(batch_id),
                UInt32(kv0 + ki * MMA_DIM),
                UInt32(kv_head),
                kv_layout_val,
            )
            # Prefetch this key-block's depth-wide V span: issue all DEPTH_MMAS
            # fragment loads before the MMAs consume them so the strided reads
            # overlap (~2x read BW) and the PV MMAs run back-to-back. Batching
            # wider than DEPTH_MMAS regressed (over-reserved registers).
            var v_frags = Array[SIMD[q_type, 8], DEPTH_MMAS](uninitialized=True)
            if sk_tile_full:
                comptime for ni in range(DEPTH_MMAS):
                    v_frags[ni] = out_mma.load_fragment[v_t.dtype](
                        v_tile.tile[MMA_DIM, MMA_DIM](0, ni)
                    ).cast[q_type]()
            else:
                # Partial last KV tile: valid V rows are [kv0+ki*16, cur_cache_len);
                # bound the load so OOB rows zero-fill.
                var v_valid_rows = cur_cache_len - (kv0 + ki * MMA_DIM)
                comptime for ni in range(DEPTH_MMAS):
                    var v_sub = v_tile.tile[MMA_DIM, MMA_DIM](0, ni)
                    if v_valid_rows >= MMA_DIM:
                        v_frags[ni] = out_mma.load_fragment[v_t.dtype](
                            v_sub
                        ).cast[q_type]()
                    else:
                        v_frags[ni] = out_mma.load_fragment[
                            v_t.dtype, bounded=True
                        ](v_sub, v_valid_rows).cast[q_type]()
            comptime for ni in range(DEPTH_MMAS):
                _mma_apple_transposable(
                    output_accum[ni],
                    p_frag,
                    v_frags[ni],
                    output_accum[ni],
                    False,
                    False,
                )
            # Mid-P.V scheduler fence after the first ki block. Reached by all
            # simdgroups (PV is ungated), so no deadlock.
            comptime if ki == 0:
                _prevent_inst_reorder(cur_cache_len)

    # Epilogue: normalize, cast, masked store.
    _softmax_normalize[DEPTH_MMAS](sm_l, output_accum)

    # Output view over this tile's SQ x Depth span. The fp32 accumulator must be
    # cast to `output_type` on store, so the fp32-only `store_bounded` fast path
    # is unavailable; scatter each fragment via `out_mma`'s `rb`/`cb` lane map
    # with a per-element cast and a query-row bounds check.
    var out_layout_val = TileLayout(
        Coord(Idx[SQ], Idx[Depth]), Coord(q_row_stride, Idx[1])
    )
    var out_tile = TileTensor[
        output_type, type_of(out_layout_val), MutAnyOrigin
    ](
        ptr=output.ptr + out_offset + q_row0 * q_row_stride,
        layout=out_layout_val,
    )
    # Vectorized store: the lane owns two contiguous width-4 depth runs (rows `rb`
    # and `rb+8`), so store each row-half in one `width=4` cast-and-store instead
    # of 8 scalar stores. The depth bound always holds (`depth == Depth`); only the
    # loop-invariant query-row guards clamp, folding to True on a full Q block.
    var row_lo = rb
    var lo_ok = row_lo < valid_q
    var hi_ok = row_lo + 8 < valid_q
    var off_lo = rb * q_row_stride + cb
    var off_hi = (rb + 8) * q_row_stride + cb
    comptime for ni in range(DEPTH_MMAS):
        var o_sub = out_tile.tile[MMA_DIM, MMA_DIM](0, ni)
        comptime assert o_sub.flat_rank == 2, "output sub-tile must be 2D"
        var frag = output_accum[ni]
        if lo_ok:
            (o_sub.ptr + off_lo).store(
                frag.slice[4, offset=0]().cast[output_type]()
            )
        if hi_ok:
            (o_sub.ptr + off_hi).store(
                frag.slice[4, offset=4]().cast[output_type]()
            )


# ===-------------------------------------------------------------------=== #
# Host launcher. Mirrors `mha_gpu_naive` / `naive_fa_decode_apple` (MHAOperand
# overload) signature; enqueues the prefill kernel. Dispatches the runtime
# `_depth` to a compile-time `Depth` specialization over multiples of 16.
# ===-------------------------------------------------------------------=== #
def fa_prefill_apple[
    output_type: DType,
    k_t: MHAOperand,
    v_t: MHAOperand,
    mask_t: MHAMask,
    //,
    ragged: Bool = False,
    sink: Bool = False,
    _use_valid_length: Bool = False,
    _is_cache_length_accurate: Bool = False,
    num_simdgroups: Int = 4,
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
    """Host launcher for the Apple M5 flash-attention prefill kernel.

    Mirrors `mha_gpu_naive`'s `MHAOperand` overload so `flash_attention_dispatch`
    routes to it like the fallback, and specializes one kernel per supported
    `depth` (a multiple of 16 up to `FA_PREFILL_APPLE_MAX_HEAD_DIM`). The external
    `LayoutTensor` ABI is converted to `TileTensor` at the enqueue boundary so the
    kernel is TileTensor-only.

    Parameters:
        output_type: The dtype of the output tensor (inferred).
        k_t: The `MHAOperand` type of the paged key cache (inferred).
        v_t: The `MHAOperand` type of the paged value cache (inferred).
        mask_t: The `MHAMask` type of the attention mask functor (inferred).
        ragged: If True, `valid_length` is a cumulative offset buffer over
            variable-length sequences in the batch (defaults to False).
        sink: If True, pre-seed the softmax state from per-head sink weights
            (defaults to False).
        _use_valid_length: If True, read each batch's query length from
            `valid_length[batch_id]` (defaults to False).
        _is_cache_length_accurate: If True, the total attention length equals
            the query length (no cached prefix); if False, add the key
            operand's per-batch cache length (defaults to False).
        num_simdgroups: Number of simdgroups co-resident per threadgroup,
            controlling per-launch occupancy (defaults to 4).

    Args:
        q: The query tensor in flattened BSHD layout (batch, seq, head,
            depth).
        k: The paged key cache operand.
        v: The paged value cache operand.
        mask_functor: The attention mask functor applied to each score tile.
        output: The output tensor, written in the same BSHD layout as `q`.
        valid_length: Per-batch `uint32` sequence lengths or cumulative
            offsets, depending on `ragged` and `_use_valid_length`.
        scale: The softmax scale factor applied to QK score products.
        batch_size: Number of sequences in the batch.
        max_prompt_len: Maximum query length across the batch, in tokens.
        max_cache_size: Maximum KV cache length across the batch, in tokens.
        num_heads: Number of query attention heads.
        depth: Head dimension; must be a multiple of 16 and at most
            `FA_PREFILL_APPLE_MAX_HEAD_DIM`.
        group: Number of query heads per KV head (GQA group size).
        ctx: The device context used to enqueue the kernel.
        sink_weights: Optional per-head sink weights indexed by head id
            (defaults to None).
    """
    # No `is_apple_gpu()` assert: this launcher compiles for the host target where
    # that query is always False; the Apple gate is the caller's in dispatch.
    comptime q_type = q.dtype
    comptime p_type = get_accum_type[q_type]()

    var num_keys = max_cache_size
    if batch_size == 0 or num_keys == 0 or max_prompt_len == 0:
        return

    debug_assert(
        depth % MMA_DIM == 0 and depth <= FA_PREFILL_APPLE_MAX_HEAD_DIM,
        (
            "fa_prefill_apple requires depth %% 16 == 0 and depth <="
            " FA_PREFILL_APPLE_MAX_HEAD_DIM; the dispatcher must gate"
            " unsupported head dims to mha_gpu_naive"
        ),
    )

    comptime NumNMmas = DEFAULT_NUM_N_MMAS
    comptime SQ = MMA_DIM

    # Flatten the LayoutTensor ABI to 1D TileTensors; the kernel bakes the
    # ragged/BSHD + q_row0 offset into each per-simdgroup tile base.
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

    # Sink weights as a nullable `OptionalReg[TileTensor]` (not a dangling
    # pointer). None when sink=False; converted to a TileTensor so the kernel
    # stays TileTensor-only and indexes by head_id.
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

    # MODULAR_APPLE_FA_PREFILL_NUM_SIMDGROUPS={4,8,16,32} overrides the
    # simdgroups-per-threadgroup at runtime; otherwise the `num_simdgroups`
    # parameter.
    var sg_env = getenv("MODULAR_APPLE_FA_PREFILL_NUM_SIMDGROUPS", "")

    comptime MAX_D_STEPS = FA_PREFILL_APPLE_MAX_HEAD_DIM // MMA_DIM
    comptime for di in range(1, MAX_D_STEPS + 1):
        comptime D = di * MMA_DIM
        if depth == D:

            @__parameter
            def _enqueue[sg: Int]() raises:
                comptime core_kernel = fa_prefill_apple_core[
                    q_type,
                    output_type,
                    p_type,
                    k_t,
                    v_t,
                    mask_t,
                    type_of(q_flat).LayoutType,
                    type_of(output_flat).LayoutType,
                    type_of(valid_length_flat).LayoutType,
                    type_of(sink_layout_val),
                    type_of(output_flat).Engine,
                    type_of(q_flat).Engine,
                    type_of(valid_length_flat).Engine,
                    SinkTile.Engine,
                    ragged=ragged,
                    sink=sink,
                    _use_valid_length=_use_valid_length,
                    _is_cache_length_accurate=_is_cache_length_accurate,
                    Depth=D,
                    NumNMmas=NumNMmas,
                    NumSimdgroups=sg,
                ]
                var num_q_tiles = ceildiv(max_prompt_len, sg * SQ)
                ctx.enqueue_function[core_kernel](
                    output_flat,
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
                    grid_dim=(num_q_tiles, num_heads, batch_size),
                    block_dim=sg * WARP_SIZE,
                )

            @__parameter
            def _dispatch[sg: Int]() raises:
                _enqueue[sg]()

            if sg_env == "4":
                _dispatch[4]()
            elif sg_env == "8":
                _dispatch[8]()
            elif sg_env == "16":
                _dispatch[16]()
            elif sg_env == "32":
                _dispatch[32]()
            else:
                _dispatch[num_simdgroups]()
