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
"""Block-wide bitonic sort top-k for the MLA/MSA indexer.

Replaces `topk_gpu`, which is pathological when `k ≈ N` (e.g. the
`k = N = 2048` DeepSeek-V3 / MiniMax-M3 indexer config). For `N > 2048` a
streaming variant folds `TILE`-wide tiles into a running top-`TILE`
champion, selecting `K ≤ TILE` out of arbitrarily large `N`.
"""

from std.sys import align_of, is_nvidia_gpu, size_of
from std.sys._assembly import inlined_assembly

from std.bit import count_leading_zeros, log2_floor

from std.atomic import Atomic, Ordering
from shmem.ep_comm import BLOCK_SCOPE
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
    warp_id,
)
from max.gpu.memory import external_memory
from max.gpu.sync import barrier
import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp
from max.gpu.host import DeviceAttribute, DeviceContext, FuncAttribute
from std.memory import bitcast, unsafe_stack_allocation
from std.math import ceildiv
from std.time import global_perf_counter_ns
from std.utils.numerics import min_or_neg_inf
from std.utils.static_tuple import StaticTuple
from structured_kernels.trace_buf import GmemTrace, NullTrace, TraceBuf

# ===----------------------------------------------------------------------=== #
# Compile-time constants
# ===----------------------------------------------------------------------=== #

# Locked together: _PTOPK_BLOCK * _PTOPK_ITEMS == _PTOPK_TOTAL,
# _PTOPK_LOG2 == log2(_PTOPK_TOTAL), and
# PERSISTENT_TOPK_MAX_N == _PTOPK_TOTAL. Changing the top-k capacity
# means changing all of them.
comptime PERSISTENT_TOPK_MAX_N: Int = 2048

comptime _PTOPK_BLOCK: Int = 512
comptime _PTOPK_ITEMS: Int = 4
comptime _PTOPK_TOTAL: Int = _PTOPK_BLOCK * _PTOPK_ITEMS
comptime _PTOPK_LOG2: Int = 11
comptime _TILE: Int = _PTOPK_TOTAL

# Each thread owns the `_PTOPK_ITEMS` (=4) contiguous canonical elements
# `e0..e3 = tid*4 + {0,1,2,3}`, so those byte offsets are a multiple of 16.
# `_V4_ALIGN` (= 16 B) is the alignment of a width-4 f32/i32 SIMD; the `alignment`
# argument on the width-4 loads/stores below is REQUIRED — the default (element
# alignment) makes LLVM legalize the vector op back into 4 scalar accesses.
comptime _V4_ALIGN: Int = align_of[SIMD[.float32, _PTOPK_ITEMS]]()


@always_inline
def _load4_scores(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    base: Int,
    index_base: Int,
    local0: Int,
    count: Int,
) -> Tuple[SIMD[.float32, _PTOPK_ITEMS], SIMD[.int32, _PTOPK_ITEMS]]:
    """Load the 4 contiguous scores a thread owns, plus their column indices.

    Reads `in_scores[base + index_base + local0 + j]` for `j` in `0..3`, storing
    the row-global column `index_base + local0 + j` as the index so a merged
    partial still carries row-global indices. Positions past `count` pad with
    (-inf, -1).

    Emits a single 128-bit vector load on the fast path — taken when the 4
    elements are fully in-bounds and the address is 16B-aligned. `base`,
    `index_base` and the tile offset folded into `local0` are uniform across a
    block and `tid*4` is a multiple of 4, so the alignment test is uniform: it
    only varies with `base` (the row/slice origin). Odd `N` (e.g. decode-long
    `N = 32769`) makes odd rows' bases non-16B-aligned and the boundary tile
    partially out of bounds; both fall to the scalar path, which a width-4
    aligned load would fault on.
    """
    var col0 = index_base + local0
    var off = base + col0
    if local0 + _PTOPK_ITEMS <= count and (off & (_PTOPK_ITEMS - 1)) == 0:
        var v = in_scores.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](off)
        var idx = SIMD[.int32, _PTOPK_ITEMS](Int32(col0)) + SIMD[
            DType.int32, _PTOPK_ITEMS
        ](0, 1, 2, 3)
        return (v, idx)

    var vv = SIMD[.float32, _PTOPK_ITEMS](min_or_neg_inf[.float32]())
    var ii = SIMD[.int32, _PTOPK_ITEMS](Int32(-1))
    comptime for j in range(_PTOPK_ITEMS):
        if local0 + j < count:
            vv[j] = in_scores[off + j]
            ii[j] = Int32(col0 + j)
    return (vv, ii)


@always_inline
def _halfclean4[
    cv_origin: MutOrigin,
    sv_origin: MutOrigin,
](
    champ_v: UnsafePointer[Float32, cv_origin, address_space=.SHARED],
    champ_i: UnsafePointer[Int32, cv_origin, address_space=.SHARED],
    scratch_v: UnsafePointer[Float32, sv_origin, address_space=.SHARED],
    scratch_i: UnsafePointer[Int32, sv_origin, address_space=.SHARED],
    tid: Int,
) -> Tuple[SIMD[.float32, _PTOPK_ITEMS], SIMD[.int32, _PTOPK_ITEMS]]:
    """Batcher half-cleaner: element-wise max of the champion and reversed tile.

    Reads the 4 canonical champion slots `champ[e0..e3]` and the mirrored
    partner run `scratch[_TILE-1-e0 .. _TILE-1-e3]`. That partner run is the
    contiguous descending block `scratch[_TILE-4-e0 .. _TILE-1-e0]`, so it loads
    as one 16B vector and reverses in registers (lane `p` = `scratch[_TILE-1-e_p]`).
    Bases are 16B-aligned (`_V4_ALIGN`-aligned allocations, `tid*4` a multiple of
    4), so both champion and scratch reads are single `LDS.128`s.
    """
    var e0 = tid * _PTOPK_ITEMS
    var cv = champ_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    var ci = champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    var rbase = _TILE - _PTOPK_ITEMS - e0
    var bv = scratch_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        rbase
    ).reversed()
    var bi = scratch_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        rbase
    ).reversed()
    var take = bv.gt(cv)  # element-wise mask; `>` is scalar-only on SIMD
    return (take.select(bv, cv), take.select(bi, ci))


@always_inline
def _ranks_below[
    tiebreak: Bool
](v0: Float32, i0: Int32, v1: Float32, i1: Int32,) -> Bool:
    """Whether `(v0, i0)` belongs after `(v1, i1)` in the sorted output.

    With `tiebreak` the order is `(descending value, ascending index)`, a total
    order on a row (indices are distinct), so the network's output is a function
    of its input alone. Without it, equal values come out in an order that
    depends on the compare-exchange schedule.
    """
    comptime if tiebreak:
        return v0 < v1 or (v0 == v1 and i0 > i1)
    else:
        return v0 < v1


@always_inline
def _select_lane_after_xor[
    tiebreak: Bool = False
](
    v: Float32,
    i: Int32,
    pv: Float32,
    pi: Int32,
    want_d: Bool,
    is_lo: Bool,
) -> Tuple[Float32, Int32]:
    var do_swap: Bool
    if is_lo:
        do_swap = _ranks_below[tiebreak](v, i, pv, pi) == want_d
    else:
        do_swap = _ranks_below[tiebreak](pv, pi, v, i) == want_d
    if do_swap:
        return (pv, pi)
    return (v, i)


@always_inline
def _swap_pair_if[
    tiebreak: Bool = False
](
    v0: Float32,
    i0: Int32,
    v1: Float32,
    i1: Int32,
    want_d: Bool,
) -> Tuple[
    Float32,
    Int32,
    Float32,
    Int32,
]:
    if _ranks_below[tiebreak](v0, i0, v1, i1) == want_d:
        return (v1, i1, v0, i0)
    return (v0, i0, v1, i1)


# ===----------------------------------------------------------------------=== #
# Bitonic sort core (register + warp-shuffle + swizzled-SMEM)
# ===----------------------------------------------------------------------=== #


@always_inline
def _bitonic_sort_desc[
    sv_origin: MutOrigin,
    si_origin: MutOrigin,
    tiebreak: Bool = False,
](
    mut v0: Float32,
    mut v1: Float32,
    mut v2: Float32,
    mut v3: Float32,
    mut i0: Int32,
    mut i1: Int32,
    mut i2: Int32,
    mut i3: Int32,
    smem_v: UnsafePointer[Float32, sv_origin, address_space=.SHARED],
    smem_i: UnsafePointer[Int32, si_origin, address_space=.SHARED],
    tid: Int,
):
    """Full descending bitonic sort of `_PTOPK_TOTAL` elements in place.

    Thread `t` owns canonical elements `e0=4t..e3=4t+3` in registers on
    entry and holds them sorted-descending on exit (position 0 = largest).
    `smem_v`/`smem_i` must be `_PTOPK_TOTAL`-wide; they are scratch
    (contents undefined on return).

    `tiebreak` breaks equal values by ascending index (see `_ranks_below`).
    """
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    comptime for s in range(1, 8):  # stages 1..7
        comptime for h in reversed(range(1, s + 1)):  # h = s..1
            comptime stride = 1 << (h - 1)
            var want_d = ((e0 >> s) & 1) == 0

            comptime if stride == 1:
                var want_d0 = ((e0 >> s) & 1) == 0
                var want_d2 = ((e2 >> s) & 1) == 0
                v0, i0, v1, i1 = _swap_pair_if[tiebreak](
                    v0, i0, v1, i1, want_d0
                )
                v2, i2, v3, i3 = _swap_pair_if[tiebreak](
                    v2, i2, v3, i3, want_d2
                )

            elif stride == 2:
                v0, i0, v2, i2 = _swap_pair_if[tiebreak](v0, i0, v2, i2, want_d)
                v1, i1, v3, i3 = _swap_pair_if[tiebreak](v1, i1, v3, i3, want_d)

            else:
                comptime ts = stride >> 2
                var pv0 = warp.shuffle_xor(v0, UInt32(ts))
                var pi0 = warp.shuffle_xor(i0, UInt32(ts))
                var pv1 = warp.shuffle_xor(v1, UInt32(ts))
                var pi1 = warp.shuffle_xor(i1, UInt32(ts))
                var pv2 = warp.shuffle_xor(v2, UInt32(ts))
                var pi2 = warp.shuffle_xor(i2, UInt32(ts))
                var pv3 = warp.shuffle_xor(v3, UInt32(ts))
                var pi3 = warp.shuffle_xor(i3, UInt32(ts))
                var is_lo = (tid & ts) == 0
                v0, i0 = _select_lane_after_xor[tiebreak](
                    v0, i0, pv0, pi0, want_d, is_lo
                )
                v1, i1 = _select_lane_after_xor[tiebreak](
                    v1, i1, pv1, pi1, want_d, is_lo
                )
                v2, i2 = _select_lane_after_xor[tiebreak](
                    v2, i2, pv2, pi2, want_d, is_lo
                )
                v3, i3 = _select_lane_after_xor[tiebreak](
                    v3, i3, pv3, pi3, want_d, is_lo
                )

    # XOR bank-swizzle (bijection) to keep the SMEM stages bank-conflict-free.
    var sw0 = (e0 & ~31) | ((e0 ^ (e0 >> 5)) & 31)
    var sw1 = (e1 & ~31) | ((e1 ^ (e1 >> 5)) & 31)
    var sw2 = (e2 & ~31) | ((e2 ^ (e2 >> 5)) & 31)
    var sw3 = (e3 & ~31) | ((e3 ^ (e3 >> 5)) & 31)

    comptime for s in range(8, _PTOPK_LOG2 + 1):  # stages 8..11
        smem_v[sw0] = v0
        smem_i[sw0] = i0
        smem_v[sw1] = v1
        smem_i[sw1] = i1
        smem_v[sw2] = v2
        smem_i[sw2] = i2
        smem_v[sw3] = v3
        smem_i[sw3] = i3
        barrier()

        comptime for h in reversed(range(8, s + 1)):  # h = s..8
            comptime stride = 1 << (h - 1)  # S = 128, 256, 512, 1024
            var want_d = ((e0 >> s) & 1) == 0
            comptime for item in range(4):
                var ei = e0 + item
                var ej = ei ^ stride
                if (ei & stride) == 0:
                    var si = (ei & ~31) | ((ei ^ (ei >> 5)) & 31)
                    var sj = (ej & ~31) | ((ej ^ (ej >> 5)) & 31)
                    var vi = smem_v[si]
                    var vj = smem_v[sj]
                    var ii = smem_i[si]
                    var ij = smem_i[sj]
                    if _ranks_below[tiebreak](vi, ii, vj, ij) == want_d:
                        smem_v[si] = vj
                        smem_v[sj] = vi
                        smem_i[si] = ij
                        smem_i[sj] = ii
            barrier()

        v0 = smem_v[sw0]
        i0 = smem_i[sw0]
        v1 = smem_v[sw1]
        i1 = smem_i[sw1]
        v2 = smem_v[sw2]
        i2 = smem_i[sw2]
        v3 = smem_v[sw3]
        i3 = smem_i[sw3]

        comptime for h in reversed(range(3, 8)):  # h = 7..3
            comptime stride = 1 << (h - 1)
            comptime ts = stride >> 2  # thread-stride ∈ [1..16]
            var want_d = ((e0 >> s) & 1) == 0
            var pv0 = warp.shuffle_xor(v0, UInt32(ts))
            var pi0 = warp.shuffle_xor(i0, UInt32(ts))
            var pv1 = warp.shuffle_xor(v1, UInt32(ts))
            var pi1 = warp.shuffle_xor(i1, UInt32(ts))
            var pv2 = warp.shuffle_xor(v2, UInt32(ts))
            var pi2 = warp.shuffle_xor(i2, UInt32(ts))
            var pv3 = warp.shuffle_xor(v3, UInt32(ts))
            var pi3 = warp.shuffle_xor(i3, UInt32(ts))
            var is_lo = (tid & ts) == 0
            v0, i0 = _select_lane_after_xor[tiebreak](
                v0, i0, pv0, pi0, want_d, is_lo
            )
            v1, i1 = _select_lane_after_xor[tiebreak](
                v1, i1, pv1, pi1, want_d, is_lo
            )
            v2, i2 = _select_lane_after_xor[tiebreak](
                v2, i2, pv2, pi2, want_d, is_lo
            )
            v3, i3 = _select_lane_after_xor[tiebreak](
                v3, i3, pv3, pi3, want_d, is_lo
            )

        var want_d2 = ((e0 >> s) & 1) == 0
        v0, i0, v2, i2 = _swap_pair_if[tiebreak](v0, i0, v2, i2, want_d2)
        v1, i1, v3, i3 = _swap_pair_if[tiebreak](v1, i1, v3, i3, want_d2)

        var want_d1 = ((e0 >> s) & 1) == 0
        v0, i0, v1, i1 = _swap_pair_if[tiebreak](v0, i0, v1, i1, want_d1)
        v2, i2, v3, i3 = _swap_pair_if[tiebreak](v2, i2, v3, i3, want_d1)


# ===----------------------------------------------------------------------=== #
# The packed 64-bit ordering key
# ===----------------------------------------------------------------------=== #
#
# The radix select's ordering contract is the pair `(phi(score), ~column)`
# compared lexicographically. Packing it into one unsigned 64-bit word makes the
# whole order a single `<`, which is what both the select's threshold test and
# the final rank cost.


@always_inline
def _pack_key(phi: UInt32, rcol: UInt32) -> UInt64:
    """The select's key as one word: `phi` above, `~column` below.

    `phi` is a monotone image of the score (see `_phi`) and `~column` descends
    with the column, so a single descending sort on this word is descending
    score, ascending column -- the output contract. Neither field can carry into
    the other, so the packing is order-preserving rather than approximate.
    """
    return (UInt64(phi) << UInt64(32)) | UInt64(rcol)


@always_inline
def _key_column(key: UInt64) -> Int32:
    """The column a packed key came from."""
    return Int32(~UInt32(key & UInt64(0xFFFFFFFF)))


@always_inline
def _bitonic_merge_desc[
    sv_origin: MutOrigin,
    si_origin: MutOrigin,
](
    mut v0: Float32,
    mut v1: Float32,
    mut v2: Float32,
    mut v3: Float32,
    mut i0: Int32,
    mut i1: Int32,
    mut i2: Int32,
    mut i3: Int32,
    smem_v: UnsafePointer[Float32, sv_origin, address_space=.SHARED],
    smem_i: UnsafePointer[Int32, si_origin, address_space=.SHARED],
    tid: Int,
):
    """Descending bitonic *merge* of a `_PTOPK_TOTAL`-element bitonic sequence.

    Precondition: the block-wide sequence held in registers (thread `t` owns
    canonical elements `e0=4t..e3=4t+3`) is already bitonic. On exit it is
    sorted descending in place. This is the final merge pass of
    `_bitonic_sort_desc` (stage `s = _PTOPK_LOG2`, where the direction is
    uniformly descending) factored out: a caller that has produced a bitonic
    sequence — e.g. a half-cleaner of two sorted runs — finishes in this one
    pass (strides 1024..1, ~5 block barriers) instead of a full sort
    (~14 barriers). `smem_v`/`smem_i` must be `_PTOPK_TOTAL`-wide scratch
    (contents undefined on return).
    """
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var sw0 = (e0 & ~31) | ((e0 ^ (e0 >> 5)) & 31)
    var sw1 = (e1 & ~31) | ((e1 ^ (e1 >> 5)) & 31)
    var sw2 = (e2 & ~31) | ((e2 ^ (e2 >> 5)) & 31)
    var sw3 = (e3 & ~31) | ((e3 ^ (e3 >> 5)) & 31)

    smem_v[sw0] = v0
    smem_i[sw0] = i0
    smem_v[sw1] = v1
    smem_i[sw1] = i1
    smem_v[sw2] = v2
    smem_i[sw2] = i2
    smem_v[sw3] = v3
    smem_i[sw3] = i3
    barrier()

    # Cross-warp substages (stride >= 128) go through swizzled SMEM; direction
    # is uniformly descending so the largest value moves to the lower index.
    comptime for h in reversed(range(8, _PTOPK_LOG2 + 1)):  # strides 1024..128
        comptime stride = 1 << (h - 1)
        comptime for item in range(4):
            var ei = e0 + item
            var ej = ei ^ stride
            if (ei & stride) == 0:
                var si = (ei & ~31) | ((ei ^ (ei >> 5)) & 31)
                var sj = (ej & ~31) | ((ej ^ (ej >> 5)) & 31)
                var vi = smem_v[si]
                var vj = smem_v[sj]
                if vi < vj:
                    smem_v[si] = vj
                    smem_v[sj] = vi
                    smem_i[si], smem_i[sj] = smem_i[sj], smem_i[si]
        barrier()

    v0 = smem_v[sw0]
    i0 = smem_i[sw0]
    v1 = smem_v[sw1]
    i1 = smem_i[sw1]
    v2 = smem_v[sw2]
    i2 = smem_i[sw2]
    v3 = smem_v[sw3]
    i3 = smem_i[sw3]

    # Sub-warp substages (stride 64..4) via register shuffles, no barrier.
    comptime for h in reversed(range(3, 8)):
        comptime ts = (1 << (h - 1)) >> 2
        var pv0 = warp.shuffle_xor(v0, UInt32(ts))
        var pi0 = warp.shuffle_xor(i0, UInt32(ts))
        var pv1 = warp.shuffle_xor(v1, UInt32(ts))
        var pi1 = warp.shuffle_xor(i1, UInt32(ts))
        var pv2 = warp.shuffle_xor(v2, UInt32(ts))
        var pi2 = warp.shuffle_xor(i2, UInt32(ts))
        var pv3 = warp.shuffle_xor(v3, UInt32(ts))
        var pi3 = warp.shuffle_xor(i3, UInt32(ts))
        var is_lo = (tid & ts) == 0
        v0, i0 = _select_lane_after_xor(v0, i0, pv0, pi0, True, is_lo)
        v1, i1 = _select_lane_after_xor(v1, i1, pv1, pi1, True, is_lo)
        v2, i2 = _select_lane_after_xor(v2, i2, pv2, pi2, True, is_lo)
        v3, i3 = _select_lane_after_xor(v3, i3, pv3, pi3, True, is_lo)

    # Register substages (stride 2 then 1).
    if v0 < v2:
        v0, v2 = v2, v0
        i0, i2 = i2, i0
    if v1 < v3:
        v1, v3 = v3, v1
        i1, i3 = i3, i1
    if v0 < v1:
        v0, v1 = v1, v0
        i0, i1 = i1, i0
    if v2 < v3:
        v2, v3 = v3, v2
        i2, i3 = i3, i2


# ===----------------------------------------------------------------------=== #
# GPU kernels
# ===----------------------------------------------------------------------=== #


@__name(t"persistent_topk_2048")
def _persistent_topk_2048_kernel(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
):
    """Block-wide bitonic top-k for `N <= _PTOPK_TOTAL` (one block per row)."""
    _persistent_topk_2048_impl(in_scores, out_idxs, N, K, Int(N))


@__name(t"persistent_topk_2048_bounded")
def _persistent_topk_2048_bounded_kernel(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    row_bounds: UnsafePointer[Int32, ImmutAnyOrigin],
):
    """`_persistent_topk_2048_kernel` with a per-row live-column bound.

    Row `r` reads only `in_scores[r * N .. r * N + row_bounds[r])`; columns
    past the bound load as `(-inf, -1)`, so they are never selected and pad
    the output with `-1`. Scan cost tracks each row's real length rather than
    the row stride `N`.
    """
    var bound = Int(row_bounds[Int(block_idx.x)])
    _persistent_topk_2048_impl(
        in_scores, out_idxs, N, K, min(Int(N), max(0, bound))
    )


@always_inline
def _persistent_topk_2048_impl(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    count: Int,
):
    var tid = thread_idx.x
    var token = block_idx.x

    var smem_v = unsafe_stack_allocation[
        _PTOPK_TOTAL,
        Float32,
        address_space=.SHARED,
    ]()
    var smem_i = unsafe_stack_allocation[
        _PTOPK_TOTAL, Int32, address_space=.SHARED
    ]()

    var _N = Int(N)
    var _K = Int(K)
    var row = token * _N
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var v0: Float32
    var v1: Float32
    var v2: Float32
    var v3: Float32
    var i0: Int32
    var i1: Int32
    var i2: Int32
    var i3: Int32

    var lv, li = _load4_scores(in_scores, row, 0, e0, count)
    v0 = lv[0]
    v1 = lv[1]
    v2 = lv[2]
    v3 = lv[3]
    i0 = li[0]
    i1 = li[1]
    i2 = li[2]
    i3 = li[3]

    _bitonic_sort_desc(v0, v1, v2, v3, i0, i1, i2, i3, smem_v, smem_i, tid)

    var base = token * _K
    if e0 < _K:
        out_idxs[base + e0] = i0
    if e1 < _K:
        out_idxs[base + e1] = i1
    if e2 < _K:
        out_idxs[base + e2] = i2
    if e3 < _K:
        out_idxs[base + e3] = i3


@__name(t"streaming_topk")
def _streaming_topk_kernel(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
):
    """Streaming top-K for `N > _PTOPK_TOTAL`, `K <= _TILE` (one block per row).

    Folds each `_TILE`-wide tile of the row into a running sorted
    top-`_TILE` champion in SMEM (Batcher half-cleaner merge), then
    writes the top-`K` indices.
    """
    var tid = thread_idx.x
    var token = block_idx.x

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var champ_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()

    var _N = Int(N)
    var _K = Int(K)
    var row = token * _N
    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var neg_inf = min_or_neg_inf[.float32]()
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[.float32, _PTOPK_ITEMS](neg_inf)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[.int32, _PTOPK_ITEMS](Int32(-1))
    )
    barrier()

    var v0: Float32
    var v1: Float32
    var v2: Float32
    var v3: Float32
    var i0: Int32
    var i1: Int32
    var i2: Int32
    var i3: Int32

    var num_tiles = ceildiv(_N, _TILE)
    for t in range(num_tiles):
        var g = t * _TILE
        var lv, li = _load4_scores(in_scores, row, 0, g + e0, _N)
        v0 = lv[0]
        v1 = lv[1]
        v2 = lv[2]
        v3 = lv[3]
        i0 = li[0]
        i1 = li[1]
        i2 = li[2]
        i3 = li[3]

        _bitonic_sort_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        # Barrier before reusing `scratch` as the stash: the sort's
        # final swizzled reads alias the canonical write indices (WAR hazard).
        barrier()

        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

        # Each thread touches only its own champion slots e0..e3, so the
        # merge needs no barrier until the write-back below.
        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        # Barrier: finish all champion/scratch reads before the re-sort
        # below overwrites them.
        barrier()

        # The half-cleaner above already made `v` a bitonic sequence (max of a
        # descending champion and a reversed-descending tile), so a bitonic
        # merge finishes it — no need to pay for a full sort.
        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var base = token * _K
    var out_i = champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    if e0 < _K:
        out_idxs[base + e0] = out_i[0]
    if e1 < _K:
        out_idxs[base + e1] = out_i[1]
    if e2 < _K:
        out_idxs[base + e2] = out_i[2]
    if e3 < _K:
        out_idxs[base + e3] = out_i[3]


# Digit widths of the radix select, per round of a key half: 12 bits, then 10
# and 10. The two are set by opposing pressures.
#
# Round 0 histograms every column of the row, so its digit wants to be wide --
# a narrow one piles the whole row onto few bins and serializes the atomics. 12
# bits also fixes the histogram at 16 KB of SMEM.
#
# Every later round histograms only the bracket's columns, so a wide digit no
# longer buys atomic spread -- while a round pays to zero and suffix-sum its bins
# whatever it counts, which the trace prices at ~1.0 us per split at 4096 bins.
# Narrowing rounds 1 and 2 to 10 bits quarters their bins and still resolves the
# same 32 bits in the same three rounds, so nothing is traded against it.
# Narrowing them costs something too, and only the decode side can afford it: a
# wider bracket after round 1 means the "the bracket holds exactly what we need"
# early exit fires less often, so more rows run a third round. That is free where
# rows under-fill the GPU and measured +1.1% on the prefill row count, so the
# tail width is a per-instantiation choice with today's 12 as the default.
comptime _HSEL_BITS: Int = 12
comptime _HSEL_TAIL_BITS: Int = 10

# Digit width of the final rank, narrower than the select's 12: the rank scans
# its bins for at most `_PTOPK_TOTAL` keys, so a bin per key is enough and the
# scan costs half as much.
comptime _HSEL_RANK_BITS: Int = 11
comptime _HSEL_RANK_BINS: Int = 1 << _HSEL_RANK_BITS
comptime _HSEL_BINS: Int = 1 << _HSEL_BITS

# Keys the select may park for the rank, against a `K` of at most
# `_PTOPK_TOTAL`. The slack above `K` is what lets a round stop early: the rank
# orders any set of distinct keys exactly, so the rounds only have to *bound* the
# candidate set, not narrow it until the bracket holds the K-th largest -- the
# top `K` of a superset of the top `K` is the top `K`. Overflowing the cap costs
# one more round rather than a dropped tie.
comptime _HSEL_SEL_CAP: Int = 2 * _PTOPK_TOTAL


@always_inline
def _hsel_half_rounds[tail_bits: Int]() -> Int:
    """Rounds needed to resolve a 32-bit key half at these digit widths."""
    return 1 + ceildiv(32 - _HSEL_BITS, tail_bits)


@always_inline
def _hsel_w_in[tail_bits: Int, r: Int]() -> Int:
    """Bits of the key half still unresolved when round `r` starts."""
    return 32 - _HSEL_BITS * min(r, 1) - tail_bits * (r - min(r, 1))


@always_inline
def _hsel_w_out[tail_bits: Int, r: Int]() -> Int:
    """Bits left unresolved after round `r` takes its digit."""
    return max(
        0,
        _hsel_w_in[tail_bits, r]() - (_HSEL_BITS if r == 0 else tail_bits),
    )


# Columns a thread loads per row-scan step. The scan is latency-bound rather
# than issue-bound, so more loads in flight per thread pays -- up to the point
# where holding the group costs occupancy, which is why this is 8 and not 16.
comptime _HSEL_SCAN_ITEMS: Int = 8
comptime _HSEL_SCAN_STEP: Int = _PTOPK_BLOCK * _HSEL_SCAN_ITEMS

# Columns at or above the first round's bracket are parked here, so later rounds
# refine in SMEM instead of re-reading the row. A region per warp is the only
# granularity without a defect: one cursor for the block serializes every parked
# column on one SMEM address, one per thread leaves each thread a different
# count and runs later rounds at the widest lane's. Overflowing a region falls
# back to refining from the row, which is exact and only slower.
comptime _HSEL_WARPS: Int = _PTOPK_BLOCK // WARP_SIZE
comptime _HSEL_WARP_CAP: Int = 1024
comptime _HSEL_CAND_CAP: Int = _HSEL_WARP_CAP * _HSEL_WARPS
comptime _HSEL_SMEM_BYTES: Int = _HSEL_CAND_CAP * size_of[Int32]()

# Every row past the single-block tier goes to the select, including the rows
# just above it where the select is the slower of the two on a tie-dense
# distribution. The fold is not merely slower there, it is wrong: for `K` close
# to `N` it returns the right top-k set in the wrong order, which no gate caught
# because none had covered 2048 < N < 4096 at K = 2048.
comptime _HSEL_MIN_N: Int = PERSISTENT_TOPK_MAX_N + 1

# Geometry of the register-resident select: 1024 threads holding `_HSEL_RES_VECS`
# `float4` of the row each, so it serves rows up to `_HSEL_RES_MAX` columns.
# The width is a register budget, and a resident column costs more than the one
# register it occupies because the per-column test and atomic keep addresses live
# alongside the payload. Eight lands under the limit with no spill; sixteen
# spills, and fewer threads holding more columns loses to re-testing what round 1
# would have discarded.
comptime _HSEL_RES_BLOCK: Int = 1024
comptime _HSEL_RES_VECS: Int = 2
comptime _HSEL_RES_MAX: Int = _HSEL_RES_BLOCK * _HSEL_RES_VECS * _PTOPK_ITEMS
# A wider payload, for rows past what the narrow one holds. The streaming select
# re-reads such a row once per round, and a distribution whose threshold bin
# overflows `cand` makes that every round; holding the row instead costs only
# registers, and where the rows under-fill the GPU those are free.
comptime _HSEL_RES_VECS_WIDE: Int = 4
comptime _HSEL_RES_MAX_WIDE: Int = (
    _HSEL_RES_BLOCK * _HSEL_RES_VECS_WIDE * _PTOPK_ITEMS
)

# Trace slots per row: [0] block entry, [1 + 2r] end of round `r`'s pass,
# [2 + 2r] end of its split, [13] thread 0 done appending, [16] every warp done
# appending, [17] the sort's inputs in registers, [14] end of the sort, [15]
# block exit. Rounds a row never runs leave their slots at zero, so the round
# count is read off the trace rather than reported separately.
#
# 13 and 16 straddle the append barrier on purpose: the gap between them is the
# slowest warp's overhang on a ragged candidate list, otherwise charged to the
# phase after it. 18..22 split the rank open: digit chosen, histogram built,
# suffix scan and cursor done, keys grouped, run scan done; [14] closes it.
#
# [23 + r] is not a timestamp but round `r`'s shape,
# `(bracket << 40) | (atOrAbove << 20) | need`, because a phase table says how long
# a round took and never why it ran: whether a later round is needed is a question
# about the bracket the previous one left, and `atOrAbove == need` is exactly the
# exact-fit exit. [29] closes the cheap tail's per-group scans, and [30] and [31]
# cut a split into its suffix scan, its walk to the bin, and whatever the contract
# then does with the bracket -- unrelated costs that one phase boundary cannot
# tell apart.
comptime HSEL_TRACE_EVENTS: Int = 32


@always_inline
def _phi(v: Float32) -> UInt32:
    """The monotone float-to-uint32 bijection: flip the sign bit of a positive,
    every bit of a negative. Comparing the results as unsigned orders the
    scores, so a radix digit of one is a range of scores.
    """
    var bits = bitcast[.uint32, 1](v)
    return bits ^ ((-(bits >> 31)) | UInt32(0x80000000))


@always_inline
def _phi_group(
    v: SIMD[.float32, _HSEL_SCAN_ITEMS]
) -> SIMD[.uint32, _HSEL_SCAN_ITEMS]:
    """`_phi` over a thread's whole scan group."""
    var bits = bitcast[.uint32, _HSEL_SCAN_ITEMS](v)
    return bits ^ ((-(bits >> 31)) | UInt32(0x80000000))


@always_inline
def _load_scan_group(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    off: Int,
    local0: Int,
    count: Int,
) -> SIMD[.float32, _HSEL_SCAN_ITEMS]:
    """The contiguous scores a thread owns, as back-to-back 128-bit loads.

    Positions past `count` pad with the bit pattern whose `_phi` image is 0,
    which is below every non-NaN score's (`-inf` maps to `0x007FFFFF`). That is
    what lets the select count the padding as an ordinary candidate: no
    threshold it produces can reach 0, so padding is never selected.

    See `_load4_scores` for why the alignment test is uniform and required.
    """
    if local0 + _HSEL_SCAN_ITEMS <= count and (off & (_PTOPK_ITEMS - 1)) == 0:
        return in_scores.load[width=_HSEL_SCAN_ITEMS, alignment=_V4_ALIGN](off)

    var bits = SIMD[.uint32, _HSEL_SCAN_ITEMS](UInt32(0xFFFFFFFF))
    comptime for j in range(_HSEL_SCAN_ITEMS):
        if local0 + j < count:
            bits[j] = bitcast[.uint32, 1](in_scores[off + j])
    return bitcast[.float32, _HSEL_SCAN_ITEMS](bits)


@always_inline
def _phi4(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    off: Int,
    local0: Int,
    count: Int,
) -> SIMD[.uint32, _PTOPK_ITEMS]:
    """`_phi` of the four contiguous scores a resident group owns.

    Converting at the load is what keeps the resident payload at one register
    per column: `phi` is all the select ever reads and is the same width as the
    score, so the score itself never has to be kept.

    Positions past `count` pad exactly as `_load_scan_group` does -- see there
    for why padding is safe to histogram. A group entirely past the row end
    takes the third branch and touches no memory at all, which is what makes a
    payload wider than the row cost only its registers.
    """
    var bits = SIMD[.uint32, _PTOPK_ITEMS](UInt32(0xFFFFFFFF))
    if local0 + _PTOPK_ITEMS <= count and (off & (_PTOPK_ITEMS - 1)) == 0:
        bits = bitcast[.uint32, _PTOPK_ITEMS](
            in_scores.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](off)
        )
    elif local0 < count:
        comptime for j in range(_PTOPK_ITEMS):
            if local0 + j < count:
                bits[j] = bitcast[.uint32, 1](in_scores[off + j])
    return bits ^ ((-(bits >> 31)) | UInt32(0x80000000))


@always_inline
def _warp_sum_u32(val: UInt32) -> UInt32:
    """Warp-wide sum of `val`, broadcast to every lane.

    `redux` reduces a warp in a single instruction on recent NVIDIA parts, but the
    stdlib wires it only for float32 max/min, so the integer form is spelled out
    here the way it spells that one. The shuffle butterfly is the portable
    fallback and takes a step per halving to reach the same answer.
    """
    comptime if is_nvidia_gpu():
        return inlined_assembly[
            "redux.sync.add.u32 $0, $1, $2;",
            UInt32,
            constraints="=r,r,i",
            has_side_effect=True,
        ](val, Int32(-1))
    return warp.sum(val)


@always_inline
def _hsel_block_scan[
    block_size: Int, mut_origin: MutOrigin
](
    val: UInt32,
    wsum: UnsafePointer[UInt32, mut_origin, address_space=.SHARED],
) -> UInt32:
    """Block-wide inclusive prefix sum across `block_size` threads, one barrier.

    `block.prefix_sum` needs two: it stores each warp's total, waits, has warp 0
    scan those totals, and waits again so the rest can read the result. Having
    every warp reduce the totals itself instead removes the second wait, and a
    warp reduction is cheap enough that nothing replaces the barrier it drops.
    `wsum` must hold `block_size / WARP_SIZE` entries and its previous contents
    are not read.

    The caller must already be past any barrier that protects `wsum`'s prior use.
    """
    comptime n_warps = block_size // WARP_SIZE
    var lane = Int(lane_id())
    var wid = Int(warp_id())
    var inc = warp.prefix_sum(val)
    if lane == WARP_SIZE - 1:
        wsum[wid] = inc
    barrier()
    # Every warp sums the totals of the warps before it. A warp shares one `wid`,
    # so masking by it leaves each warp reducing exactly its own prefix.
    var mine = UInt32(0)
    if lane < n_warps:
        mine = wsum[lane]
    return inc + _warp_sum_u32(mine if lane < wid else UInt32(0))


@always_inline
def _hsel_split_scan[
    block_size: Int, one_barrier: Bool, mut_origin: MutOrigin
](
    val: UInt32,
    wsum: UnsafePointer[UInt32, mut_origin, address_space=.SHARED],
) -> UInt32:
    """The round split's block-wide inclusive prefix sum, whichever is faster here.

    Which one that is depends on the instantiation rather than on the shape, so the
    choice is a parameter: the one-barrier scan wins everywhere except the ordered
    per-bin digit, where it measured slower than the stdlib's. `wsum` is scratch for
    that scan alone, so a caller which does not select it may pass a single element.

    The value is returned rather than written into a variable the caller declares
    ahead of the branch, which keeps that declaration a live initialization.

    Neither arm leaves the block synchronized, so consecutive calls need a barrier
    between them for the scratch each one reuses. What both do guarantee is that
    every thread's input has been consumed before any thread returns, so a caller
    may overwrite the slots its own input came from without one.
    """
    comptime if one_barrier:
        return _hsel_block_scan[block_size](val, wsum)
    return block.prefix_sum[block_size=block_size](val)


@always_inline
def _hsel_contested[
    half: Int
](phi: UInt32, rcol: UInt32, lo: UInt32, hi: UInt32, t_phi: UInt32) -> Bool:
    """Whether a column's key is still in the bracket around the K-th largest.

    The key is `(phi, ~column)` compared lexicographically. While `half` 0 is
    resolving the score, the bracket only constrains `phi`; by `half` 1 the
    score is pinned to `t_phi` and the bracket constrains the column.
    """
    comptime if half == 0:
        return phi >= lo and phi <= hi
    else:
        return phi == t_phi and rcol >= lo and rcol <= hi


@always_inline
def _hsel_digit[
    half: Int, w_out: Int, nbins: Int
](phi: UInt32, rcol: UInt32) -> Int:
    """The radix digit this round splits on, from whichever key half it owns."""
    comptime if half == 0:
        return Int((phi >> UInt32(w_out)) & UInt32(nbins - 1))
    else:
        return Int((rcol >> UInt32(w_out)) & UInt32(nbins - 1))


@always_inline
def _hsel_selected(
    phi: UInt32, rcol: UInt32, t_phi: UInt32, t_rcol: UInt32
) -> Bool:
    """Whether a column is one of the K, given the threshold key."""
    return phi > t_phi or (phi == t_phi and rcol >= t_rcol)


@__name(t"histsel_topk")
def _histsel_topk_kernel[
    TraceBufT: TraceBuf,
    enable_trace: Bool = False,
    prefetch: Bool = False,
    tail_bits: Int = _HSEL_BITS,
    rank_bits: Int = _HSEL_RANK_BITS - 1,
    rank_slots: Bool = False,
    sel_cap: Int = _PTOPK_TOTAL,
    ordered: Bool = True,
    deterministic: Bool = True,
](
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    trace_buf: TraceBufT,
):
    """`_histsel_topk_impl` over the full row width `N` — see there."""
    _histsel_topk_impl[
        TraceBufT,
        enable_trace,
        prefetch,
        tail_bits,
        rank_bits,
        rank_slots,
        sel_cap,
        ordered,
        deterministic,
    ](in_scores, out_idxs, N, K, Int(N), trace_buf)


@__name(t"histsel_topk_bounded")
def _histsel_topk_bounded_kernel[
    TraceBufT: TraceBuf,
    enable_trace: Bool = False,
    prefetch: Bool = False,
    tail_bits: Int = _HSEL_BITS,
    rank_bits: Int = _HSEL_RANK_BITS - 1,
    rank_slots: Bool = False,
    sel_cap: Int = _PTOPK_TOTAL,
    ordered: Bool = True,
    deterministic: Bool = True,
](
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    row_bounds: UnsafePointer[Int32, ImmutAnyOrigin],
    trace_buf: TraceBufT,
):
    """`_histsel_topk_impl` with a per-row live-column bound.

    Row `r` scans only `[0, row_bounds[r])` of its `N`-wide stripe; columns
    past the bound are never selected and pad the output with `-1` (the same
    result as the unbounded kernel over a `-inf`-padded row), while the row
    traffic tracks the real length. A bound at or below `K` selects every
    live column, in descending order, and pads the rest of the output row
    with `-1`.
    """
    var bound = Int(row_bounds[Int(block_idx.x)])
    _histsel_topk_impl[
        TraceBufT,
        enable_trace,
        prefetch,
        tail_bits,
        rank_bits,
        rank_slots,
        sel_cap,
        ordered,
        deterministic,
    ](in_scores, out_idxs, N, K, min(Int(N), max(0, bound)), trace_buf)


@always_inline
def _histsel_topk_impl[
    TraceBufT: TraceBuf,
    enable_trace: Bool = False,
    prefetch: Bool = False,
    tail_bits: Int = _HSEL_BITS,
    rank_bits: Int = _HSEL_RANK_BITS - 1,
    rank_slots: Bool = False,
    sel_cap: Int = _PTOPK_TOTAL,
    ordered: Bool = True,
    deterministic: Bool = True,
](
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    count: Int,
    trace_buf: TraceBufT,
):
    """Top-K by radix select on `(score, column)`, one block per row.

    A round histograms one digit of the key over the columns whose higher digits
    bracket the K-th largest, then splits that histogram at the K-th element;
    the bin holding it becomes the next round's bracket. Three rounds resolve
    the score half of the key and three more the column half, so this terminates
    on any input -- a row of equal scores included, where the column half does
    the selecting. Splitting the key into two 32-bit halves rather than
    bracketing one 64-bit key keeps the row scan in 32-bit ALU ops.

    Only the first two rounds read the row. The second parks the bracket's
    columns in `cand` and appends the columns already above it, so every later
    round refines over `cand` in SMEM. That caps the row traffic at two passes
    no matter how many rounds the score distribution needs -- the difference
    between a distribution with few ties and one that is mostly plateaus.

    What the rounds produce is a threshold key with at most `sel_cap` columns at
    or above it -- exactly `K` when `sel_cap` is `_PTOPK_TOTAL`, and possibly a
    superset when the caller allows slack (see `_HSEL_SEL_CAP`). One pass appends
    those columns and one counting rank puts them in order and drops the slack.

    `prefetch` runs the row scan one group ahead. A thread consumes its group the
    instruction after it lands, so without it the scan holds one 128-bit load in
    flight per thread and settles at ~2.3 TB/s -- Little's law, not bandwidth,
    since B200 peaks near 8. Carrying the next group costs eight registers,
    which takes the kernel from 54 to 80 and with it from two resident blocks per
    SM to one. That is free when the rows do not fill the GPU, where every block
    is resident either way, and costs close to half the throughput when they do,
    so the caller picks by row count instead of this being unconditional.
    """
    var tid = Int(thread_idx.x)
    var token = Int(block_idx.x)

    var hist = unsafe_stack_allocation[
        _HSEL_BINS + 1, UInt32, address_space=.SHARED
    ]()
    var sel_k = unsafe_stack_allocation[
        sel_cap,
        UInt64,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    # [0] split bin, [1] keys strictly above it, [2] keys at or above it,
    # [3] append cursor into `sel`, [4] collect cursor into `cand`.
    var ctl = unsafe_stack_allocation[8, UInt32, address_space=.SHARED]()
    var wcur = unsafe_stack_allocation[
        _HSEL_WARPS, UInt32, address_space=.SHARED
    ]()
    var wor = unsafe_stack_allocation[
        2 * _HSEL_WARPS + 2,
        UInt64,
        address_space=.SHARED,
    ]()
    var cand = external_memory[
        Int32,
        address_space=.SHARED,
        alignment=align_of[Int32](),
    ]()

    var _N = Int(N)
    var _K = Int(K)
    var row = token * _N
    # Fewer live columns than selections: every live column is in the top K,
    # so the threshold rounds have nothing to resolve (their counts could never
    # reach `need`). Skip them and have the append take every live column; the
    # rank orders those and the output tail past `count` is written `-1`.
    var select_all = count < _K

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 0,
                UInt64(global_perf_counter_ns()),
            )

    if tid == 0:
        ctl[3] = 0
        ctl[4] = 0
    if tid < _HSEL_WARPS:
        wcur[tid] = 0

    # A column is one of the K when `phi > t_phi`, or `phi == t_phi` and its
    # `~column` is at least `t_rcol` -- descending score, ascending column.
    var t_phi = UInt32(0)
    var t_rcol = UInt32(0)
    # `[lo, hi]` brackets the K-th largest value of the half being resolved and
    # `need` is how many of the top K that bracket still has to supply. Every
    # thread derives them from the same broadcast counts, so they stay
    # block-uniform without a reduction.
    var lo = UInt32(0)
    var hi = UInt32.MAX
    var need = _K
    # Columns the append will park. Exactly `_K` unless a round hands over early.
    var m_sel = count if select_all else _K
    var found = select_all
    var parked = False
    var lane = Int(lane_id())
    var wbase = Int(warp_id()) * _HSEL_WARP_CAP
    var wcur_p = wcur + Int(warp_id())
    var wn = 0

    comptime half_rounds = _hsel_half_rounds[tail_bits]()
    comptime for half in range(2):
        comptime for r in range(half_rounds):
            comptime rr = half * half_rounds + r
            if not found:
                comptime w_in = _hsel_w_in[tail_bits, r]()
                comptime w_out = _hsel_w_out[tail_bits, r]()
                comptime nbins = 1 << (w_in - w_out)

                # The column half's first round is arithmetic, not a histogram:
                # every column is below `N`, so every `~column` shares its top
                # `32 - bits(N)` bits and the digit is the same for all of them --
                # there is nothing to count, and counting it anyway costs a
                # 4096-bin histogram, a block scan and four barriers. The test is
                # block-uniform, so the block skips together.
                var resolved = False
                comptime if half == 1 and r == 0:
                    resolved = _N <= (1 << w_out)
                if resolved:
                    lo = (UInt32.MAX >> UInt32(w_out)) << UInt32(w_out)
                    hi = lo + ((UInt32(1) << UInt32(w_out)) - 1)
                else:
                    var z = tid
                    while z < nbins:
                        hist[z] = 0
                        z += _PTOPK_BLOCK
                    barrier()

                    if parked:
                        # `cand` -> column -> score is two dependent loads whose
                        # result is consumed immediately, so a refine round would
                        # otherwise run at one outstanding load per lane. The next
                        # index is clamped rather than branched on: past the end it
                        # re-reads the last candidate, a valid address and an L1
                        # hit, for less than a divergent branch would cost.
                        var i = lane
                        var col = 0
                        var sc = Float32(0)
                        if i < wn:
                            col = Int(cand[wbase + i])
                            sc = in_scores[row + col]
                        while i < wn:
                            var ni = i + WARP_SIZE
                            var ncol = Int(cand[wbase + min(ni, wn - 1)])
                            var nsc = in_scores[row + ncol]
                            var ph = _phi(sc)
                            var rc = ~UInt32(col)
                            if _hsel_contested[half](ph, rc, lo, hi, t_phi):
                                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                                    ordering=Ordering.RELAXED
                                ](
                                    hist
                                    + _hsel_digit[half, w_out, nbins](ph, rc),
                                    UInt32(1),
                                )
                            col = ncol
                            sc = nsc
                            i = ni
                    else:
                        var base = tid * _HSEL_SCAN_ITEMS
                        var cur = SIMD[.float32, _HSEL_SCAN_ITEMS](0)
                        comptime if prefetch:
                            cur = _load_scan_group(
                                in_scores, row + base, base, count
                            )
                        while base < count:
                            var nb = base + _HSEL_SCAN_STEP
                            var lv = cur
                            comptime if prefetch:
                                # Issue the next group before histogramming this
                                # one. A step past the row end reads nothing --
                                # `_load_scan_group` returns padding without
                                # touching memory -- so priming and over-running
                                # need no guard.
                                cur = _load_scan_group(
                                    in_scores, row + nb, nb, count
                                )
                            else:
                                lv = _load_scan_group(
                                    in_scores, row + base, base, count
                                )
                            var ph = _phi_group(lv)
                            comptime if rr == 1:
                                # Fused: columns already above the bracket are
                                # final, so append them now; the bracket's own
                                # columns get parked for the later rounds. Most
                                # groups hold neither, and one vector compare says
                                # so far cheaper than eight scalar branches.
                                comptime for j in range(_HSEL_SCAN_ITEMS):
                                    if ph[j] >= lo:
                                        var slot = Int(
                                            Atomic[
                                                UInt32, scope=BLOCK_SCOPE
                                            ].fetch_add[
                                                ordering=Ordering.RELAXED
                                            ](
                                                wcur_p, UInt32(1)
                                            )
                                        )
                                        # Wrap instead of testing the cap: a warp
                                        # that overruns its region turns `parked`
                                        # off for the whole block below, so nothing
                                        # reads `cand` afterwards and what the
                                        # wrapped slots hold cannot matter.
                                        cand[
                                            wbase
                                            + (slot & (_HSEL_WARP_CAP - 1))
                                        ] = Int32(base + j)
                                        if ph[j] <= hi:
                                            _ = Atomic[
                                                UInt32, scope=BLOCK_SCOPE
                                            ].fetch_add[
                                                ordering=Ordering.RELAXED
                                            ](
                                                hist
                                                + _hsel_digit[
                                                    half, w_out, nbins
                                                ](ph[j], ~UInt32(base + j)),
                                                UInt32(1),
                                            )
                            else:
                                comptime for j in range(_HSEL_SCAN_ITEMS):
                                    var rc = ~UInt32(base + j)
                                    if _hsel_contested[half](
                                        ph[j], rc, lo, hi, t_phi
                                    ):
                                        _ = Atomic[
                                            UInt32, scope=BLOCK_SCOPE
                                        ].fetch_add[ordering=Ordering.RELAXED](
                                            hist
                                            + _hsel_digit[half, w_out, nbins](
                                                ph[j], rc
                                            ),
                                            UInt32(1),
                                        )
                            base = nb
                    barrier()

                    comptime if enable_trace:
                        if tid == 0:
                            trace_buf.store(
                                token * HSEL_TRACE_EVENTS + 1 + 2 * rr,
                                UInt64(global_perf_counter_ns()),
                            )

                    # Suffix-sum the bins: thread `t` owns the descending run ending
                    # at `nbins-1 - t*run`, so an ordinary prefix sum over threads
                    # is a suffix sum over bins and the split lands in exactly one
                    # thread's run. `nbins < _PTOPK_BLOCK` leaves the high threads
                    # empty, which the same test excludes.
                    comptime run = max(1, nbins // _PTOPK_BLOCK)
                    var top = nbins - 1 - tid * run
                    var local = UInt32(0)
                    if top >= 0:
                        comptime for j in range(run):
                            local += hist[top - j]
                    var inclusive = block.prefix_sum[block_size=_PTOPK_BLOCK](
                        local
                    )
                    var exclusive = inclusive - local
                    var uneed = UInt32(need)
                    if exclusive < uneed and uneed <= inclusive:
                        var acc = exclusive
                        var split = top
                        while True:
                            acc += hist[split]
                            if acc >= uneed:
                                break
                            split -= 1
                        ctl[0] = UInt32(split)
                        ctl[1] = acc - hist[split]
                        ctl[2] = acc
                    barrier()

                    lo += UInt32(Int(ctl[0])) << UInt32(w_out)
                    hi = lo + ((UInt32(1) << UInt32(w_out)) - 1)
                    # Hand the rest to the rank once the columns at or above the
                    # bracket fit `sel_k`; that sum is exactly what the append's
                    # threshold test selects. Not from round 0, even when it would
                    # fit -- the append walks the list round 1 parks, so handing
                    # over before it exists costs a second pass over the row.
                    var msel = _K
                    var fits = Int(ctl[2]) == need
                    comptime if sel_cap > _PTOPK_TOTAL and rr >= 1:
                        msel = _K - need + Int(ctl[2])
                        fits = msel <= sel_cap
                    if fits:
                        comptime if half == 0:
                            t_phi = lo
                        else:
                            t_rcol = lo
                        m_sel = msel
                        found = True
                    else:
                        need -= Int(ctl[1])
                    comptime if rr == 1:
                        wn = Int(wcur[Int(warp_id())])
                        if wn > _HSEL_WARP_CAP:
                            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                                ordering=Ordering.RELAXED
                            ](ctl + 4, UInt32(1))
                    # The next round's zeroing must not race the reads above.
                    barrier()
                    comptime if rr == 1:
                        parked = ctl[4] == 0

                    comptime if enable_trace:
                        if tid == 0:
                            trace_buf.store(
                                token * HSEL_TRACE_EVENTS + 2 + 2 * rr,
                                UInt64(global_perf_counter_ns()),
                            )

        comptime if half == 0:
            # Carry the resolved score into the column half. Reaching here means
            # `need` columns share it, so `lo == hi` is that score.
            if not found:
                t_phi = lo
                lo = UInt32(0)
                hi = UInt32.MAX

    # Append the K. Parked columns already had everything above the round-1
    # bracket appended; without them the whole row is rescanned, so the cursor
    # has to go back to zero first.
    if parked:
        # One candidate's score ahead, as in the refine rounds above.
        var i = lane
        var col = 0
        var sc = Float32(0)
        if i < wn:
            col = Int(cand[wbase + i])
            sc = in_scores[row + col]
        while i < wn:
            var ni = i + WARP_SIZE
            var ncol = Int(cand[wbase + min(ni, wn - 1)])
            var nsc = in_scores[row + ncol]
            var ph = _phi(sc)
            var rc = ~UInt32(col)
            if _hsel_selected(ph, rc, t_phi, t_rcol):
                var pos = Int(
                    Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                        ordering=Ordering.RELAXED
                    ](ctl + 3, UInt32(1))
                )
                if pos < sel_cap:
                    sel_k[pos] = _pack_key(ph, rc)
            col = ncol
            sc = nsc
            i = ni
    else:
        var base = tid * _HSEL_SCAN_ITEMS
        while base < count:
            var lv = _load_scan_group(in_scores, row + base, base, count)
            var ph = _phi_group(lv)
            comptime for j in range(_HSEL_SCAN_ITEMS):
                var rc = ~UInt32(base + j)
                # Select-all (count < K): take every live column and none of the
                # partial group's padding, whose phi of 0 the threshold test
                # would otherwise admit against the all-zero threshold.
                var take: Bool
                if select_all:
                    take = base + j < count
                else:
                    take = _hsel_selected(ph[j], rc, t_phi, t_rcol)
                if take:
                    # Exactly `m_sel` columns clear the threshold; the bound only
                    # keeps a NaN score (out of contract) from writing past the
                    # buffer.
                    var pos = Int(
                        Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                            ordering=Ordering.RELAXED
                        ](ctl + 3, UInt32(1))
                    )
                    if pos < sel_cap:
                        sel_k[pos] = _pack_key(ph[j], rc)
            base += _HSEL_SCAN_STEP

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 13,
                UInt64(global_perf_counter_ns()),
            )

    # The append's atomics have to be visible to the rank's reads.
    barrier()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 16,
                UInt64(global_perf_counter_ns()),
            )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 17,
                UInt64(global_perf_counter_ns()),
            )

    # The cheap contract is the append's own output read back out. `sel_cap` of
    # `_PTOPK_TOTAL` makes the rounds land on `K` exactly, so `sel_k` already
    # holds the `K` columns and the only thing the rank adds is their order.
    #
    # `deterministic` is what decides whether that is affordable here, and unlike
    # the resident kernel it is not a free choice: the append claims slots with an
    # atomic, so `sel_k`'s arrangement is arrival-ordered and varies between
    # launches. The rank is what erases that, which makes it the *determinism*
    # mechanism on this path and not only an ordering one. Placing deterministically
    # instead would need each thread's slot base up front, and the counts that give
    # it are only known after a pass -- a second pass over a row this long costs
    # more than the rank it would save. So the reproducible contract keeps the rank
    # and only `deterministic=False` skips it.
    comptime if (not ordered) and (not deterministic):
        var q = tid
        while q < _K:
            out_idxs[token * _K + q] = _key_column(sel_k[q])
            q += _PTOPK_BLOCK
        comptime if enable_trace:
            if tid == 0:
                trace_buf.store(
                    token * HSEL_TRACE_EVENTS + 14,
                    UInt64(global_perf_counter_ns()),
                )
                trace_buf.store(
                    token * HSEL_TRACE_EVENTS + 15,
                    UInt64(global_perf_counter_ns()),
                )
        return

    # `cand` is dead once the append is done, so the rank's per-bin cursor takes
    # its first 16 KB rather than costing the block more shared memory -- which
    # at the prefill row count is what decides whether two blocks fit an SM.
    _hsel_rank_write[
        TraceBufT,
        _PTOPK_BLOCK,
        enable_trace,
        rank_bits,
        rank_slots,
        sel_cap,
        False,
    ](
        sel_k,
        hist,
        cand.bitcast[UInt32](),
        wor,
        hist,
        sel_k,
        out_idxs,
        token * _K,
        m_sel,
        _K,
        0,
        tid,
        trace_buf,
        token,
    )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 14,
                UInt64(global_perf_counter_ns()),
            )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 15,
                UInt64(global_perf_counter_ns()),
            )


# ===----------------------------------------------------------------------=== #
# Ordering the selected K by counting rank
# ===----------------------------------------------------------------------=== #
#
# The select leaves exactly `K` packed keys unordered and the contract is to write
# them descending, which a counting rank does in one histogram pass instead of a
# bitonic network's fixed log-squared cost, because a key's output slot *is* the
# number of keys above it:
#
#     rank(k) = (keys in a strictly higher bin) + (keys in k's bin above k)
#
# the first term falling out of a suffix sum over the bins. Keys are distinct --
# columns are -- so rank is a bijection onto `[0, K)`.
#
# It turns on the digit, and the top bits of `phi` will not do: the selected keys
# are the *largest* K, so they share sign and exponent by construction and a
# 12-bit slice of an fp32 key carries 3 mantissa bits, collapsing K keys into a
# handful of bins. Take the most significant bits on which the keys actually
# *differ* instead. The keys agree on every bit outside `vary`, so gathering its
# top set bits in significance order is order-preserving while spending every bit
# on information -- and it degrades gracefully where a fixed digit does not, a row
# of one repeated score putting every key in its own bin on the column's bits.


@always_inline
def _hsel_vary_positions[rank_bits: Int](vary: UInt64) -> SIMD[.uint64, 2]:
    """The `rank_bits` most significant set bits of `vary`, as positions.

    Packed six bits each so the block shares them through a shared-memory
    broadcast: a position vector would be sixteen registers, and re-deriving it
    per thread was measured at 2.8 us of a 5.9 us rank. A position needs six
    bits, so ten fill a word and the eleventh starts a second one -- packing all
    eleven into one word truncates the last position, which the gate caught as
    seven wrong configurations while the timings looked fine.

    Positions past the last set bit stay at 0, which adds the same constant to
    every key's digit and so cannot change the order.
    """
    var packed = SIMD[.uint64, 2](0)
    var m = vary
    comptime for i in range(rank_bits):
        if m != 0:
            var b = 63 - Int(count_leading_zeros(m))
            packed[i // 10] |= UInt64(b) << UInt64(6 * (i % 10))
            m &= ~(UInt64(1) << UInt64(b))
    return packed


# Where the select's first digit sits in the packed key: `phi` is the high half
# and that round takes its top `_HSEL_BITS`, so two keys share a coarse bin exactly
# when they agree above this bit.
comptime _HSEL_COARSE_SHIFT: Int = 64 - _HSEL_BITS


@always_inline
def _hsel_coarse_bin(key: UInt64) -> Int:
    """The select's round-0 digit, recovered from the packed key."""
    return Int((key >> UInt64(_HSEL_COARSE_SHIFT)) & UInt64(_HSEL_BINS - 1))


@always_inline
def _hsel_bin_shifts(nocc: Int, nbins: Int, rank_bits: Int) -> Tuple[Int, Int]:
    """How `nocc` dense bin numbers fit into `nbins` rank bins.

    Returns `(left, right)`. `left` is how many bits the bin number leaves over
    for the key's own varying bits; `right` is how far to merge neighbouring bin
    numbers when there are already more of them than rank bins. At most one is
    nonzero and both preserve order.
    """
    var l = 0
    var r = 0
    if nocc <= nbins:
        while l < rank_bits and (nocc << (l + 1)) <= nbins:
            l += 1
    else:
        while ((nocc - 1) >> r) >= nbins:
            r += 1
    return (l, r)


@always_inline
def _hsel_rank_digit[
    rank_bits: Int
](key: UInt64, packed: SIMD[.uint64, 2]) -> Int:
    """Gather the key's varying bits, most significant first."""
    var d = UInt32(0)
    comptime for i in range(rank_bits):
        comptime w = i // 10
        comptime sh = 6 * (i % 10)
        d |= UInt32(
            (key >> ((packed[w] >> UInt64(sh)) & UInt64(63))) & UInt64(1)
        ) << UInt32(rank_bits - 1 - i)
    return Int(d)


@always_inline
def _hsel_digit_of[
    rank_bits: Int, bin_digit: Bool, d_origin: MutOrigin
](
    key: UInt64,
    packed: SIMD[.uint64, 2],
    dmap: UnsafePointer[UInt32, d_origin, address_space=.SHARED],
    left: Int,
    right: Int,
    use_bin: Bool,
) -> Int:
    """The rank's digit.

    Without `bin_digit`, the key's bits at the most significant varying positions.
    With it, the coarse bin's dense number carrying the high bits and however many
    varying positions that leaves room for in the low ones -- which is what lets
    the low bits come from *within-bin* variation without breaking order across
    bins.
    """
    var e = _hsel_rank_digit[rank_bits](key, packed)
    comptime if not bin_digit:
        return e
    if not use_bin:
        return e
    var d = (dmap[_hsel_coarse_bin(key)] << UInt32(left)) | (
        UInt32(e) >> UInt32(rank_bits - left)
    )
    return Int(d >> UInt32(right))


@always_inline
def _hsel_rank_write[
    TraceBufT: TraceBuf,
    nthreads: Int,
    enable_trace: Bool,
    rank_bits: Int,
    rank_slots: Bool,
    slots: Int,
    bin_digit: Bool,
    sk_origin: MutOrigin,
    h_origin: MutOrigin,
    c_origin: MutOrigin,
    d_origin: MutOrigin,
](
    sel_k: UnsafePointer[UInt64, sk_origin, address_space=.SHARED],
    hist: UnsafePointer[UInt32, h_origin, address_space=.SHARED],
    cur: UnsafePointer[UInt32, c_origin, address_space=.SHARED],
    wor: UnsafePointer[UInt64, sk_origin, address_space=.SHARED],
    dmap: UnsafePointer[UInt32, d_origin, address_space=.SHARED],
    banchor: UnsafePointer[UInt64, d_origin, address_space=.SHARED],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    out_base: Int,
    M: Int,
    K: Int,
    nocc: Int,
    tid: Int,
    trace_buf: TraceBufT,
    token: Int,
):
    """Write the largest `K` of the `M` keys in `sel_k` to `out_idxs`, descending.

    `M` may exceed `K` when `slots` does: a rank is a bijection onto `[0, M)`, so
    the keys the caller wants are exactly the ranks below `K` and the rest are
    dropped by not being written. That is what lets the select hand over a
    superset. A caller with `slots == _PTOPK_TOTAL` must pass `M == K`, unless a
    per-row bound left fewer than `K` live columns: `M < K` writes all `M` keys
    descending and fills output slots `[M, K)` with `-1`.

    `hist` must hold `nbins + 1` counters and `cur` `nbins + _PTOPK_TOTAL`; both
    are scratch. `wor` needs two slots per warp plus two. `sel_k` is rewritten,
    and only its first `M` entries are ever read.

    `bin_digit` measures the keys' variation against a representative of their own
    coarse bin rather than one global key, and folds the bin's dense number
    (`dmap`, `nocc` of them) into the digit's high bits. That is what stops one
    outlying group of keys from claiming every varying position -- see
    `_hsel_digit_of`. Callers without it pass `dmap`/`banchor` unused and
    `nocc = 0`.
    """
    comptime nbins = 1 << rank_bits
    comptime items = slots // nthreads
    comptime nwarps = nthreads // WARP_SIZE
    comptime run = max(1, nbins // nthreads)
    comptime assert rank_bits <= _HSEL_RANK_BITS, "cur is sized for the widest"
    comptime assert items * nthreads == slots, "items must divide evenly"
    comptime assert run * nthreads >= nbins, "every rank bin needs an owner"

    var p0 = tid * items
    var mine = SIMD[.uint64, items](0)
    comptime for i in range(items):
        if p0 + i < M:
            mine[i] = sel_k[p0 + i]

    # Which bits the keys disagree on, measured against slot 0 -- always a real
    # key, since `M >= K >= 1` -- or, under `bin_digit`, against a representative
    # of each key's own coarse bin. The latter exists because a group differing
    # from every other group high up claims all of the digit's positions while
    # being constant inside its own bin, where the digit has to separate. Any key
    # in the bin serves as representative, so the store below races harmlessly:
    # the XOR depends on which positions are non-constant, not on which key won.
    # It needs no mask either, two keys in one bin agreeing above
    # `_HSEL_COARSE_SHIFT` by definition.
    comptime if bin_digit:
        comptime for i in range(items):
            if p0 + i < M:
                banchor[_hsel_coarse_bin(mine[i])] = mine[i]
        barrier()

    var anchor = sel_k[0]
    var vary = UInt64(0)
    var vphi = UInt32(0)
    comptime for i in range(items):
        if p0 + i < M:
            vary |= mine[i] ^ anchor
            comptime if bin_digit:
                vphi |= UInt32(
                    (mine[i] ^ banchor[_hsel_coarse_bin(mine[i])]) >> UInt64(32)
                )
    # A step per lane-halving, so lane 0 below holds the whole warp's mask. A
    # narrower one leaves out positions the keys differ at, and the digit built
    # from it then stops preserving order.
    comptime for sp in range(log2_floor(WARP_SIZE)):
        vary |= warp.shuffle_xor(vary, UInt32(1 << sp))
        comptime if bin_digit:
            vphi |= warp.shuffle_xor(vphi, UInt32(1 << sp))
    if Int(lane_id()) == 0:
        wor[Int(warp_id())] = vary
        comptime if bin_digit:
            wor[nwarps + 2 + Int(warp_id())] = UInt64(vphi)
    barrier()
    # Which mask the digit's positions come from. The per-bin mask separates keys
    # inside a bin, but it only wins where the plain mask would have spent its
    # positions elsewhere, and the one bit that tells them apart is whether `phi`
    # varies within a bin: if it does, the plain mask's top positions are already
    # score bits that split the set and the per-bin machinery is overhead.
    var use_bin = False
    comptime if bin_digit:
        if tid == 0:
            # `nocc == 0` here means the bin numbering never ran (a
            # `select_all` row): `dmap` is uninitialized, so the per-bin
            # digit must stay off.
            var vp = UInt64(0)
            comptime for w in range(nwarps):
                vp |= wor[nwarps + 2 + w]
            cur[0] = (
                UInt32(1) if nocc > 0
                and vp == 0
                and _hsel_bin_shifts(nocc, nbins, rank_bits)[0]
                > 0 else UInt32(0)
            )
        barrier()
        use_bin = cur[0] == 1

    # Only now, and only if the digit will use it, is the per-bin mask worth its
    # own reduction and the bin numbering worth its scan.
    if use_bin:
        var vbin = UInt64(0)
        comptime for i in range(items):
            if p0 + i < M:
                vbin |= mine[i] ^ banchor[_hsel_coarse_bin(mine[i])]
        comptime for sp in range(log2_floor(WARP_SIZE)):
            vbin |= warp.shuffle_xor(vbin, UInt32(1 << sp))
        if Int(lane_id()) == 0:
            wor[Int(warp_id())] = vbin
        # Turn round 0's occupancy flags into dense numbers in place.
        comptime brun = _HSEL_BINS // nthreads
        var b0 = tid * brun
        var flag = SIMD[.uint32, brun](0)
        var nloc = UInt32(0)
        comptime for j in range(brun):
            flag[j] = dmap[b0 + j]
            nloc += flag[j]
        var binc = block.prefix_sum[block_size=nthreads](nloc)
        var g = binc - nloc
        comptime for j in range(brun):
            dmap[b0 + j] = g
            g += flag[j]
    barrier()

    if tid == 0:
        var v = UInt64(0)
        comptime for w in range(nwarps):
            v |= wor[w]
        var pk = _hsel_vary_positions[rank_bits](v)
        wor[nwarps] = pk[0]
        wor[nwarps + 1] = pk[1]
    barrier()

    var packed = SIMD[.uint64, 2](wor[nwarps], wor[nwarps + 1])
    var sh = (0, 0)
    comptime if bin_digit:
        sh = _hsel_bin_shifts(nocc, nbins, rank_bits)
    var dig = SIMD[.int32, items](0)
    comptime for i in range(items):
        dig[i] = Int32(
            _hsel_digit_of[rank_bits, bin_digit](
                mine[i], packed, dmap, sh[0], sh[1], use_bin
            )
        )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 18,
                UInt64(global_perf_counter_ns()),
            )

    var z = tid
    while z < nbins + 1:
        hist[z] = 0
        z += nthreads
    barrier()

    comptime for i in range(items):
        if p0 + i < M:
            _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                ordering=Ordering.RELAXED
            ](hist + Int(dig[i]), UInt32(1))
    barrier()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 19,
                UInt64(global_perf_counter_ns()),
            )

    # Suffix sum: thread `t` owns the descending run ending at
    # `_HSEL_BINS-1 - t*run`, so a prefix sum over threads is a suffix sum over
    # bins. Afterwards `hist[b]` counts the keys in bins at or above `b`, so bin
    # `b` owns output slots `[hist[b+1], hist[b])`.
    var top = nbins - 1 - tid * run
    var cnt = SIMD[.uint32, run](0)
    var local = UInt32(0)
    if top >= 0:
        comptime for j in range(run):
            cnt[j] = hist[top - j]
            local += cnt[j]
    var inclusive = block.prefix_sum[block_size=nthreads](local)
    barrier()
    # `acc` before the add is the count of keys in strictly higher bins, which is
    # both bin `b`'s first output slot and its cursor's initial value -- so the
    # cursor costs no second pass over the bins and no extra barrier.
    var acc = inclusive - local
    if top >= 0:
        comptime for j in range(run):
            cur[top - j] = acc
            acc += cnt[j]
            hist[top - j] = acc
    barrier()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 20,
                UInt64(global_perf_counter_ns()),
            )

    comptime for i in range(items):
        if p0 + i < M:
            var p = Int(
                Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](cur + Int(dig[i]), UInt32(1))
            )
            sel_k[p] = mine[i]
    barrier()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 21,
                UInt64(global_perf_counter_ns()),
            )

    # The array is now bin-ordered, each bin's run unordered inside, so a key's
    # rank is its run's start plus the members of the run above it.
    #
    # `rank_slots` picks which key a thread ranks, and it decides this phase
    # wherever runs are wide. Ranking the key in the thread's *slot* puts a warp's
    # lanes in one run, making `sel_k[j]` a broadcast read; ranking the key the
    # thread already holds saves recomputing the digit but turns that into 32
    # scattered reads. Neither dominates at every shape, so the caller chooses.
    #
    # A rank is a scattered address, so ranked columns land in shared memory and
    # leave in one coalesced sweep.
    #
    # Slots are taken strided rather than adjacent, because the phase costs the
    # *widest* run and adjacent slots share one. Striding spreads a thread's keys
    # across runs while keeping a warp's lanes together.
    var stage = cur + nbins
    comptime for i in range(items):
        var key: UInt64
        var b: Int
        var live: Bool
        var slot = tid + i * nthreads
        comptime if rank_slots:
            # Inside the test, not before it: `items` is sized by the buffer's
            # capacity, so a caller with slack has tail iterations that own no key
            # and would otherwise pay for a slot read and an 11-position digit.
            live = slot < M
            key = 0
            b = 0
            if live:
                key = sel_k[slot]
                b = _hsel_digit_of[rank_bits, bin_digit](
                    key, packed, dmap, sh[0], sh[1], use_bin
                )
        else:
            live = p0 + i < M
            key = mine[i]
            b = Int(dig[i])
        if live:
            var lo_p = Int(hist[b + 1])
            var hi_p = Int(hist[b])
            var r = lo_p
            # A warp's lanes hold consecutive slots, so on a run wider than the
            # warp they share `j` and this read is one broadcast serving 32
            # compares -- which is why unrolling it to put four members in flight
            # measured *worse* everywhere: the loads were never the serial part.
            # The phase's cost is the run width, and only a finer digit moves it.
            for j in range(lo_p, hi_p):
                if sel_k[j] > key:
                    r += 1
            # Ranks at or past `K` belong to the keys the select handed over as
            # slack; leaving them unwritten is what discards them. A caller
            # without slack passes `M == K`, so the test is comptime-dead there
            # and the instantiation keeps its old register count.
            comptime if slots > _PTOPK_TOTAL:
                if r < K:
                    stage[r] = UInt32(Int(_key_column(key)))
            else:
                stage[r] = UInt32(Int(_key_column(key)))

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 22,
                UInt64(global_perf_counter_ns()),
            )
    barrier()

    var q = tid
    while q < K:
        # `M < K` only under a per-row bound with fewer live columns than
        # selections; ranks exist for the `M` live keys and the tail is `-1`.
        if q < M:
            out_idxs[out_base + q] = Int32(Int(stage[q]))
        else:
            out_idxs[out_base + q] = Int32(-1)
        q += nthreads


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_HSEL_RES_BLOCK)
    ),
)
@__name(t"histsel_resident_topk")
def _histsel_resident_kernel[
    TraceBufT: TraceBuf,
    enable_trace: Bool = False,
    sel_cap: Int = _HSEL_SEL_CAP,
    bin_digit: Bool = False,
    ordered: Bool = True,
    deterministic: Bool = True,
    res_vecs: Int = _HSEL_RES_VECS,
](
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    trace_buf: TraceBufT,
):
    """`_histsel_resident_impl` over the full row width `N` — see there."""
    _histsel_resident_impl[
        TraceBufT,
        enable_trace,
        sel_cap,
        bin_digit,
        ordered,
        deterministic,
        res_vecs,
    ](in_scores, out_idxs, N, K, Int(N), trace_buf)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_HSEL_RES_BLOCK)
    ),
)
@__name(t"histsel_resident_topk_bounded")
def _histsel_resident_bounded_kernel[
    TraceBufT: TraceBuf,
    enable_trace: Bool = False,
    sel_cap: Int = _HSEL_SEL_CAP,
    bin_digit: Bool = False,
    ordered: Bool = True,
    deterministic: Bool = True,
    res_vecs: Int = _HSEL_RES_VECS,
](
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    row_bounds: UnsafePointer[Int32, ImmutAnyOrigin],
    trace_buf: TraceBufT,
):
    """`_histsel_resident_impl` with a per-row live-column bound.

    Same contract as `_histsel_topk_bounded_kernel`: row `r` reads only
    `[0, row_bounds[r])` of its stripe, and a bound at or below `K` yields all
    live columns descending with a `-1` output tail.
    """
    var bound = Int(row_bounds[Int(block_idx.x)])
    _histsel_resident_impl[
        TraceBufT,
        enable_trace,
        sel_cap,
        bin_digit,
        ordered,
        deterministic,
        res_vecs,
    ](in_scores, out_idxs, N, K, min(Int(N), max(0, bound)), trace_buf)


@always_inline
def _histsel_resident_impl[
    TraceBufT: TraceBuf,
    enable_trace: Bool = False,
    sel_cap: Int = _HSEL_SEL_CAP,
    bin_digit: Bool = False,
    ordered: Bool = True,
    deterministic: Bool = True,
    res_vecs: Int = _HSEL_RES_VECS,
](
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int32,
    K: Int32,
    count: Int,
    trace_buf: TraceBufT,
):
    """`_histsel_topk_kernel` with the row held in registers, for short rows.

    Same select, same key, same sort, so the output is byte-identical; only
    where the scores live changes. A row is read once into at most `res_vecs`
    `float4` per thread, and every round then re-tests those registers.
    That removes both of the streaming path's costs at these lengths: the
    second row scan, and the parked candidate list every later round has to
    gather through (`cand` -> column -> score, dependent global loads per
    candidate). It is also why the candidate list is absent here -- with the row
    in registers there is nothing to park -- which is most of the block's shared
    memory saved. `res_vecs` is therefore a width the dispatch chooses per row
    length, not a constant: wider holds a longer row at the cost of registers,
    and registers are what decide how many blocks an SM keeps resident.

    `ordered` is the output contract. `True` emits the `K` columns ranked by
    descending score and ascending column; `False` emits the same `K` columns in
    ascending column order and skips the rank entirely, which is most of this
    kernel's work. The *set* is identical either way -- the tie-break lives inside
    the packed key the select already compares, so the rank only ever ordered a
    set the select had already fixed, and `deterministic` then decides whether
    even that set is promised. A caller can take the cheap contract when its
    consumer reads a slot's index rather than the slot's position: the sparse-MLA
    gather is one, since `SparseCausalLogical` decides causality by looking a
    slot's key position up instead of counting slots.

    Trace slots are `_histsel_topk_kernel`'s, so one analyzer reads both.
    """
    comptime items = res_vecs * _PTOPK_ITEMS
    comptime res_step = _HSEL_RES_BLOCK * _PTOPK_ITEMS
    var tid = Int(thread_idx.x)
    var token = Int(block_idx.x)

    var hist = unsafe_stack_allocation[
        _HSEL_BINS + 1, UInt32, address_space=.SHARED
    ]()
    # Everything the rank needs is dead weight under the cheap contract, which
    # compacts straight from registers to the output and never packs a key.
    comptime selk_n = sel_cap if ordered else 1
    comptime cur_n = (_HSEL_RANK_BINS + _PTOPK_TOTAL) if ordered else 1
    comptime wor_n = (2 * (_HSEL_RES_BLOCK // WARP_SIZE) + 2) if ordered else 1
    var sel_k = unsafe_stack_allocation[
        selk_n,
        UInt64,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    # The leading slots are `_histsel_topk_kernel`'s: [0] split bin, [1] keys
    # strictly above it, [2] keys at or above it, [3] append cursor. Past those
    # the plateau fold reduces the bracket's extremes into a pair of its own, and
    # the deterministic unordered tail takes one slot per column group beyond
    # everything the select uses -- described by role rather than by index
    # because the count of groups is a width the dispatch chooses, and a tail
    # slot that collided with a live one would corrupt the row silently.
    comptime ctl_tail: Int = 8
    var ctl = unsafe_stack_allocation[
        ctl_tail + res_vecs,
        UInt32,
        address_space=.SHARED,
    ]()
    # The rank's per-bin cursor and its staged output. The streaming path folds
    # these into its dead candidate list; with no candidate list there is nothing
    # to fold into.
    var cur = unsafe_stack_allocation[
        cur_n,
        UInt32,
        address_space=.SHARED,
    ]()
    var wor = unsafe_stack_allocation[
        wor_n,
        UInt64,
        address_space=.SHARED,
    ]()
    # The one-barrier block scan is taken wherever it measured faster, which is
    # every instantiation except the ordered per-bin digit -- there it loses to the
    # stdlib's, so that one keeps the stdlib scan and needs no scratch at all.
    comptime one_barrier_scan = not (ordered and bin_digit)
    # Per-warp totals for that scan. One region per column group, because the
    # relaxed tail runs their scans back to back with no barrier between them and a
    # shared region would let a fast warp's store race a slow warp's read of the
    # scan before it; only that tail needs more than one region.
    comptime wscan_warps = _HSEL_RES_BLOCK // WARP_SIZE
    comptime wscan_n = res_vecs * wscan_warps if (
        (not ordered) and deterministic
    ) else (wscan_warps if one_barrier_scan else 1)
    var wscan = unsafe_stack_allocation[
        wscan_n,
        UInt32,
        address_space=.SHARED,
    ]()
    # Round 0's occupied bins, numbered densely, and one representative key per
    # bin -- both only for the instantiation whose digit uses them.
    comptime nbin_aux = _HSEL_BINS if (bin_digit and ordered) else 1
    var dmap = unsafe_stack_allocation[
        nbin_aux, UInt32, address_space=.SHARED
    ]()
    var banchor = unsafe_stack_allocation[
        nbin_aux, UInt64, address_space=.SHARED
    ]()

    var _N = Int(N)
    var _K = Int(K)
    var row = token * _N
    # As in `_histsel_topk_impl`: with fewer live columns than selections the
    # rounds have nothing to resolve, so the append takes every live column and
    # the rank writes a `-1` tail.
    var select_all = count < _K

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 0,
                UInt64(global_perf_counter_ns()),
            )

    if tid == 0:
        ctl[3] = 0
    # Only a select-all row needs this: it skips every round below, and with them
    # the barriers that would otherwise order this store ahead of the claims.
    # `select_all` is block-uniform, so the barrier is not divergent.
    if select_all:
        barrier()

    # Thread `t` owns columns `4*t + v*res_step .. +3` for each `v`, so
    # a warp's 32 lanes read 32 adjacent `float4` -- one fully coalesced 512-byte
    # transaction per group -- and all `vecs` groups are in flight at once. The
    # streaming path reaches two outstanding loads per thread by carrying one
    # group ahead; here the whole row is the prefetch.
    var ph = SIMD[.uint32, items](0)
    comptime for v in range(res_vecs):
        var c0 = tid * _PTOPK_ITEMS + v * res_step
        var g = _phi4(in_scores, row + c0, c0, count)
        comptime for j in range(_PTOPK_ITEMS):
            ph[v * _PTOPK_ITEMS + j] = g[j]

    var t_phi = UInt32(0)
    var t_rcol = UInt32(0)
    var lo = UInt32(0)
    var hi = UInt32.MAX
    var need = _K
    var m_sel = count if select_all else _K
    var nocc = 0
    var found = select_all

    comptime half_rounds = _hsel_half_rounds[_HSEL_TAIL_BITS]()
    # The column half resolves *which* members of the tied plateau at `t_phi` are
    # in, by narrowing `~column` until the bracket holds exactly `K`. The cheap
    # contract does not need it: the tie-break is already the lowest column, and
    # the compaction below walks columns in order, so it can take the plateau's
    # first `need` members by counting instead. That is what makes a tie-dense
    # row no more expensive here than a distinct one.
    comptime nhalves = 2 if ordered else 1
    comptime for half in range(nhalves):
        comptime for r in range(half_rounds):
            comptime rr = half * half_rounds + r
            if not found:
                comptime w_in = _hsel_w_in[_HSEL_TAIL_BITS, r]()
                comptime w_out = _hsel_w_out[_HSEL_TAIL_BITS, r]()
                comptime nbins = 1 << (w_in - w_out)

                # The column half's first round is arithmetic, not a histogram:
                # every column is below `N`, so every `~column` shares its top
                # `32 - bits(N)` bits and the digit is the same for all of them --
                # there is nothing to count, and counting it anyway costs a
                # 4096-bin histogram, a block scan and four barriers. The test is
                # block-uniform, so the block skips together.
                var resolved = False
                comptime if half == 1 and r == 0:
                    resolved = _N <= (1 << w_out)
                if resolved:
                    lo = (UInt32.MAX >> UInt32(w_out)) << UInt32(w_out)
                    hi = lo + ((UInt32(1) << UInt32(w_out)) - 1)
                else:
                    var z = tid
                    while z < nbins:
                        hist[z] = 0
                        z += _HSEL_RES_BLOCK
                    # The cheap contract's single-valued question, asked of the
                    # bracket this round is about to split. It rides the histogram's
                    # own sweep, which already visits exactly the bracket's columns
                    # and already ends in a barrier -- so the answer rides along on
                    # registers the sweep has loaded anyway, where asking it
                    # separately cost a sweep of its own.
                    #
                    # Asking it here rather than of the previous round's leftovers
                    # is also what makes it unconditional: whether a bracket is a
                    # plateau is a property of the bracket, and no property of the
                    # row predicts it. A row of continuous scores still ends in a
                    # plateau wherever the float grid is coarser than the
                    # distribution it was sampled from, which at `K` close to `N`
                    # is exactly where the threshold lands.
                    comptime fold_bracket = (
                        not ordered
                    ) and half == 0 and r > 0
                    comptime if fold_bracket:
                        if tid == 0:
                            ctl[6] = UInt32(0)
                            ctl[7] = UInt32.MAX
                    barrier()

                    var pmax = UInt32(0)
                    var pmin = UInt32.MAX
                    comptime for v in range(res_vecs):
                        var c0 = tid * _PTOPK_ITEMS + v * res_step
                        comptime for j in range(_PTOPK_ITEMS):
                            var p = ph[v * _PTOPK_ITEMS + j]
                            var rc = ~UInt32(c0 + j)
                            if _hsel_contested[half](p, rc, lo, hi, t_phi):
                                comptime if fold_bracket:
                                    if p > pmax:
                                        pmax = p
                                    if p < pmin:
                                        pmin = p
                                _ = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                                    ordering=Ordering.RELAXED
                                ](
                                    hist
                                    + _hsel_digit[half, w_out, nbins](p, rc),
                                    UInt32(1),
                                )
                    comptime if fold_bracket:
                        # One atomic per warp, not per thread: a whole block on one
                        # shared address serializes into as many read-modify-writes.
                        var wmax = warp.max(pmax)
                        var wmin = warp.min(pmin)
                        if lane_id() == 0:
                            Atomic[UInt32, scope=BLOCK_SCOPE].max[
                                ordering=Ordering.RELAXED
                            ](ctl + 6, wmax)
                            Atomic[UInt32, scope=BLOCK_SCOPE].min[
                                ordering=Ordering.RELAXED
                            ](ctl + 7, wmin)
                    barrier()

                    var folded = False
                    comptime if fold_bracket:
                        if ctl[6] == ctl[7]:
                            t_phi = ctl[6]
                            found = True
                            folded = True

                    # A plateau bracket has already answered the round, so the
                    # scan, the split and their barriers are skipped rather than
                    # run and discarded. Block-uniform: the verdict comes out of
                    # shared memory.
                    if not folded:
                        comptime if enable_trace:
                            if tid == 0:
                                trace_buf.store(
                                    token * HSEL_TRACE_EVENTS + 1 + 2 * rr,
                                    UInt64(global_perf_counter_ns()),
                                )

                        # Suffix sum and split, as in `_histsel_topk_kernel`. 1024
                        # threads halve the bins per thread, so the scan is half the
                        # work as well.
                        comptime run = max(1, nbins // _HSEL_RES_BLOCK)
                        var top = nbins - 1 - tid * run
                        var local = UInt32(0)
                        if top >= 0:
                            comptime for j in range(run):
                                local += hist[top - j]
                        var inclusive = _hsel_split_scan[
                            _HSEL_RES_BLOCK, one_barrier_scan
                        ](local, wscan)
                        var exclusive = inclusive - local
                        comptime if enable_trace:
                            comptime if rr == 0:
                                if tid == 0:
                                    trace_buf.store(
                                        token * HSEL_TRACE_EVENTS + 30,
                                        UInt64(global_perf_counter_ns()),
                                    )
                        var uneed = UInt32(need)
                        if exclusive < uneed and uneed <= inclusive:
                            var acc = exclusive
                            var split = top
                            while True:
                                acc += hist[split]
                                if acc >= uneed:
                                    break
                                split -= 1
                            ctl[0] = UInt32(split)
                            ctl[1] = acc - hist[split]
                            ctl[2] = acc
                        barrier()

                        comptime if enable_trace:
                            comptime if rr == 0:
                                if tid == 0:
                                    trace_buf.store(
                                        token * HSEL_TRACE_EVENTS + 31,
                                        UInt64(global_perf_counter_ns()),
                                    )

                        lo += UInt32(Int(ctl[0])) << UInt32(w_out)
                        hi = lo + ((UInt32(1) << UInt32(w_out)) - 1)
                        # Hand over as soon as the columns at or above the bracket fit
                        # `sel_k`, and from round 0 unlike `_histsel_topk_kernel`: the
                        # append re-tests registers here, so there is no parked list to
                        # wait for.
                        #
                        # `hist` still holds this round's *raw* per-bin counts -- the
                        # split's scan reads them and writes nothing back, unlike the
                        # rank's -- so a bin is occupied exactly when its count is
                        # nonzero. The numbering has to be taken here because round 0
                        # is the only round whose histogram covers every column.
                        comptime if bin_digit and half == 0 and r == 0:
                            comptime orun = nbins // _HSEL_RES_BLOCK
                            var b0 = tid * orun
                            var occ = SIMD[.uint32, orun](0)
                            var nloc = UInt32(0)
                            comptime for j in range(orun):
                                if hist[b0 + j] > 0:
                                    occ[j] = 1
                                    nloc += 1
                            # Flags only: turning them into dense numbers is a block
                            # scan, and whether the digit will read them is not knowable
                            # until the keys are parked, so the rank does that itself.
                            comptime for j in range(orun):
                                dmap[b0 + j] = occ[j]
                            nocc = Int(
                                block.sum[
                                    block_size=_HSEL_RES_BLOCK, broadcast=True
                                ](nloc)
                            )

                        # A caller that gave no slack gets the old exact-fit test and
                        # the old code -- see the dispatch for when that is chosen.
                        var msel = _K
                        var fits = Int(ctl[2]) == need
                        comptime if sel_cap > _PTOPK_TOTAL:
                            msel = _K - need + Int(ctl[2])
                            fits = msel <= sel_cap
                        if fits:
                            comptime if half == 0:
                                t_phi = lo
                            else:
                                t_rcol = lo
                            m_sel = msel
                            found = True
                        else:
                            need -= Int(ctl[1])

                        comptime if enable_trace:
                            if tid == 0:
                                trace_buf.store(
                                    token * HSEL_TRACE_EVENTS + 23 + rr,
                                    (
                                        UInt64(Int(ctl[2]) - Int(ctl[1]))
                                        << UInt64(40)
                                    )
                                    | (UInt64(ctl[2]) << UInt64(20))
                                    | UInt64(UInt32(need)),
                                )

                        # The next round's zeroing must not race the reads above.
                        barrier()
                    comptime if enable_trace:
                        if tid == 0:
                            trace_buf.store(
                                token * HSEL_TRACE_EVENTS + 2 + 2 * rr,
                                UInt64(global_perf_counter_ns()),
                            )

        comptime if half == 0:
            if not found:
                t_phi = lo
                lo = UInt32(0)
                hi = UInt32.MAX

    comptime if not ordered:
        comptime if enable_trace:
            if tid == 0:
                trace_buf.store(
                    token * HSEL_TRACE_EVENTS + 13,
                    UInt64(global_perf_counter_ns()),
                )

        # `t_phi` brackets the K-th largest score, so `phi > t_phi` is in and the
        # plateau at `phi == t_phi` contributes its lowest `K - count(>)` columns
        # -- the same tie-break the packed key gives the ordered path, reached by
        # counting instead of by the column half's rounds.
        #
        # Thread `tid` owns columns `_PTOPK_ITEMS*tid + v*res_step` upward, so
        # within a group the thread index already *is* the column order, and the
        # groups are in column order too: a key's rank among its class is its scan
        # prefix, plus every earlier group's total.
        #
        # Both class counts ride in one word, greaters high and equals low.
        # Neither total can exceed the row width, so at 16 bits apiece the low
        # field cannot carry into the high one.
        #
        # An atomic cursor is cheaper still, and `deterministic=False` is what
        # takes it: see below for what that costs and what it buys.
        comptime if not deterministic:
            # Warp-aggregated cursor claims and no block scan at all. What the
            # block scans buy is an ordering, and this contract reads none: every
            # column above the
            # plateau is selected whatever slot it lands in, and the plateau is
            # cut by arrival rather than by column, so which of its members
            # survive varies between identical runs.
            #
            # That is a real weakening -- a different set of keys is attended, not
            # merely a different arrangement -- and it is why the default is the
            # other way. It exists because it is the contract this kernel is
            # compared against: an implementation that ranks a wide tie plateau on
            # the score alone, and places the survivors by atomic, cannot promise
            # more than this, so matching it is what makes such a comparison
            # like-for-like.
            #
            # The cut needs the count above the plateau, which no thread knows
            # until every greater column has claimed, so the plateau is written
            # after the barrier rather than in the same sweep. One cursor serves
            # both: continuing it past `n_above` numbers the plateau's slots, and
            # a claim at or beyond `K` is simply dropped -- exactly `K - n_above`
            # survive, which is the cut.
            var obase = token * _K
            var emask = SIMD[.uint32, res_vecs](0)
            var ec_all = UInt32(0)
            comptime for v in range(res_vecs):
                var c0 = tid * _PTOPK_ITEMS + v * res_step
                var gmask = UInt32(0)
                var gc = UInt32(0)
                comptime for j in range(_PTOPK_ITEMS):
                    var p = ph[v * _PTOPK_ITEMS + j]
                    if p > t_phi:
                        gmask |= UInt32(1) << UInt32(j)
                        gc += 1
                    elif p == t_phi:
                        emask[v] |= UInt32(1) << UInt32(j)
                        ec_all += 1
                var pre = warp.prefix_sum[exclusive=True](gc)
                var wb = UInt32(0)
                if lane_id() == WARP_SIZE - 1:
                    wb = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                        ordering=Ordering.RELAXED
                    ](ctl + 3, pre + gc)
                var gp = Int(warp.shuffle_idx(wb, UInt32(WARP_SIZE - 1)) + pre)
                comptime for j in range(_PTOPK_ITEMS):
                    if (gmask >> UInt32(j)) & UInt32(1) != 0:
                        out_idxs[obase + gp] = Int32(c0 + j)
                        gp += 1

            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        token * HSEL_TRACE_EVENTS + 29,
                        UInt64(global_perf_counter_ns()),
                    )
            barrier()
            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        token * HSEL_TRACE_EVENTS + 14,
                        UInt64(global_perf_counter_ns()),
                    )

            var epre = warp.prefix_sum[exclusive=True](ec_all)
            var ewb = UInt32(0)
            if lane_id() == WARP_SIZE - 1:
                ewb = Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                    ordering=Ordering.RELAXED
                ](ctl + 3, epre + ec_all)
            var ep = Int(warp.shuffle_idx(ewb, UInt32(WARP_SIZE - 1)) + epre)
            comptime for v in range(res_vecs):
                var c0 = tid * _PTOPK_ITEMS + v * res_step
                comptime for j in range(_PTOPK_ITEMS):
                    if (emask[v] >> UInt32(j)) & UInt32(1) != 0:
                        if ep < _K:
                            out_idxs[obase + ep] = Int32(c0 + j)
                        ep += 1

            comptime if enable_trace:
                if tid == 0:
                    trace_buf.store(
                        token * HSEL_TRACE_EVENTS + 15,
                        UInt64(global_perf_counter_ns()),
                    )
            return

        var gmask = SIMD[.uint32, res_vecs](0)
        var emask = SIMD[.uint32, res_vecs](0)
        var pref = SIMD[.uint32, res_vecs](0)
        comptime for v in range(res_vecs):
            var gc = UInt32(0)
            var ec = UInt32(0)
            comptime for j in range(_PTOPK_ITEMS):
                var p = ph[v * _PTOPK_ITEMS + j]
                if p > t_phi:
                    gmask[v] |= UInt32(1) << UInt32(j)
                    gc += 1
                elif p == t_phi:
                    emask[v] |= UInt32(1) << UInt32(j)
                    ec += 1
            var local = (gc << UInt32(16)) | ec
            var inclusive = _hsel_block_scan[_HSEL_RES_BLOCK](
                local, wscan + v * wscan_warps
            )
            pref[v] = inclusive - local
            if tid == _HSEL_RES_BLOCK - 1:
                ctl[ctl_tail + v] = inclusive

        comptime if enable_trace:
            if tid == 0:
                trace_buf.store(
                    token * HSEL_TRACE_EVENTS + 29,
                    UInt64(global_perf_counter_ns()),
                )
        barrier()

        # Every key above the plateau is in, so their count is where the plateau's
        # slots start. Reading it off the scans rather than accumulating it over
        # the rounds is what keeps an early exact fit correct: that exit leaves the
        # bracket possibly holding keys above `t_phi` which no round counted.
        #
        # Each group scanned only itself, so a group's slots start after every
        # earlier group's totals. Accumulating them is what keeps that right at any
        # payload width; getting it wrong overlays an earlier group's slots and
        # writes past the row.
        var g_before = SIMD[.uint32, res_vecs](0)
        var e_before = SIMD[.uint32, res_vecs](0)
        var g_acc = UInt32(0)
        var e_acc = UInt32(0)
        comptime for v in range(res_vecs):
            g_before[v] = g_acc
            e_before[v] = e_acc
            var tot = ctl[ctl_tail + v]
            g_acc += tot >> UInt32(16)
            e_acc += tot & UInt32(0xFFFF)
        var n_above = Int(g_acc)
        var need_eq = _K - n_above

        comptime if enable_trace:
            if tid == 0:
                trace_buf.store(
                    token * HSEL_TRACE_EVENTS + 14,
                    UInt64(global_perf_counter_ns()),
                )

        # Straight to `out_idxs`: a thread's own keys land on consecutive slots,
        # and staging them in shared memory to copy out by slot measured slower
        # than the scatter it avoids.
        var obase = token * _K
        comptime for v in range(res_vecs):
            var c0 = tid * _PTOPK_ITEMS + v * res_step
            var gp = Int(pref[v] >> UInt32(16)) + Int(g_before[v])
            var ep = Int(pref[v] & UInt32(0xFFFF)) + Int(e_before[v])
            comptime for j in range(_PTOPK_ITEMS):
                if (gmask[v] >> UInt32(j)) & UInt32(1) != 0:
                    out_idxs[obase + gp] = Int32(c0 + j)
                    gp += 1
                elif (emask[v] >> UInt32(j)) & UInt32(1) != 0:
                    if ep < need_eq:
                        out_idxs[obase + n_above + ep] = Int32(c0 + j)
                    ep += 1

        comptime if enable_trace:
            if tid == 0:
                trace_buf.store(
                    token * HSEL_TRACE_EVENTS + 15,
                    UInt64(global_perf_counter_ns()),
                )
        return

    comptime for v in range(res_vecs):
        var c0 = tid * _PTOPK_ITEMS + v * res_step
        comptime for j in range(_PTOPK_ITEMS):
            var p = ph[v * _PTOPK_ITEMS + j]
            var rc = ~UInt32(c0 + j)
            # Select-all: as in `_histsel_topk_impl`'s append.
            var take: Bool
            if select_all:
                take = c0 + j < count
            else:
                take = _hsel_selected(p, rc, t_phi, t_rcol)
            if take:
                var pos = Int(
                    Atomic[UInt32, scope=BLOCK_SCOPE].fetch_add[
                        ordering=Ordering.RELAXED
                    ](ctl + 3, UInt32(1))
                )
                if pos < sel_cap:
                    sel_k[pos] = _pack_key(p, rc)

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 13,
                UInt64(global_perf_counter_ns()),
            )

    # The append's atomics have to be visible to the rank's reads.
    barrier()

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 16,
                UInt64(global_perf_counter_ns()),
            )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 17,
                UInt64(global_perf_counter_ns()),
            )

    _hsel_rank_write[
        TraceBufT,
        _HSEL_RES_BLOCK,
        enable_trace,
        _HSEL_RANK_BITS,
        True,
        sel_cap,
        bin_digit,
    ](
        sel_k,
        hist,
        cur,
        wor,
        dmap,
        banchor,
        out_idxs,
        token * _K,
        m_sel,
        _K,
        nocc,
        tid,
        trace_buf,
        token,
    )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 14,
                UInt64(global_perf_counter_ns()),
            )

    comptime if enable_trace:
        if tid == 0:
            trace_buf.store(
                token * HSEL_TRACE_EVENTS + 15,
                UInt64(global_perf_counter_ns()),
            )


@__name(t"split_partial_topk")
def _split_partial_kernel(
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    part_v: UnsafePointer[Float32, MutAnyOrigin],
    part_i: UnsafePointer[Int32, MutAnyOrigin],
    N_dev: Int32,
    slice_len_dev: Int32,
    S_dev: Int32,
):
    """Phase 1 of the split streaming top-k.

    Grid is `rows * S` blocks. Block `b` folds slice `s = b % S` of row
    `r = b // S` (columns `[s*slice_len, min((s+1)*slice_len, N))`) into a
    sorted top-`_TILE` partial written to `part_v`/`part_i` at
    `(r*S + s)*_TILE`. The fold is identical to `_streaming_topk_kernel` but
    restricted to the slice and emitting the full champion (values + indices)
    so phase 2 can merge partials by value.
    """
    var N = Int(N_dev)
    var slice_len = Int(slice_len_dev)
    var S = Int(S_dev)

    var tid = thread_idx.x
    var block = block_idx.x
    var row = block // S
    var slce = block % S

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var champ_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()

    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var neg_inf = min_or_neg_inf[.float32]()
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[.float32, _PTOPK_ITEMS](neg_inf)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, SIMD[.int32, _PTOPK_ITEMS](Int32(-1))
    )
    barrier()

    var row_base = row * N
    var slice_base = slce * slice_len
    var slice_count = 0
    if slice_base < N:
        slice_count = min(slice_len, N - slice_base)

    var v0: Float32
    var v1: Float32
    var v2: Float32
    var v3: Float32
    var i0: Int32
    var i1: Int32
    var i2: Int32
    var i3: Int32

    var num_tiles = ceildiv(slice_count, _TILE)
    for t in range(num_tiles):
        var g = t * _TILE
        var lv, li = _load4_scores(
            in_scores, row_base, slice_base, g + e0, slice_count
        )
        v0 = lv[0]
        v1 = lv[1]
        v2 = lv[2]
        v3 = lv[3]
        i0 = li[0]
        i1 = li[1]
        i2 = li[2]
        i3 = li[3]

        _bitonic_sort_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        # Barrier before reusing `scratch` as the stash: the sort's final
        # swizzled reads alias the canonical write indices (WAR hazard).
        barrier()

        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        barrier()

        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var out_base = (row * S + slce) * _TILE
    part_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )
    part_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )


@__name(t"reduce_partials_topk")
def _reduce_partials_kernel(
    in_v: UnsafePointer[Float32, ImmutAnyOrigin],
    in_i: UnsafePointer[Int32, ImmutAnyOrigin],
    out_v: UnsafePointer[Float32, MutAnyOrigin],
    out_i: UnsafePointer[Int32, MutAnyOrigin],
    count_in_dev: Int32,
    count_out_dev: Int32,
    g_dev: Int32,
):
    """Tree phase-2 round: merge groups of `g` partials in parallel.

    Grid is `rows * count_out` blocks. Block `b` merges the input partials
    `[grp*g, min(grp*g + g, count_in))` of row `r` -- where `r = b // count_out`,
    `grp = b % count_out` -- into a single sorted top-`_TILE` partial written to
    `out_v`/`out_i` at `(r*count_out + grp)*_TILE`. Each input partial is
    already sorted descending, so folding one in is a half-cleaner plus a single
    bitonic merge. Fanning the `S`-way reduction across `rows * count_out` blocks
    (vs one block per row) is what unblocks the low-row decode regime.
    """
    var count_in = Int(count_in_dev)
    var count_out = Int(count_out_dev)
    var g = Int(g_dev)

    var tid = thread_idx.x
    var block = block_idx.x
    var row = block // count_out
    var grp = block % count_out

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var champ_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()

    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    var first = grp * g
    var n_parts = min(g, count_in - first)

    # Seed the champion with the group's first partial.
    var base0 = (row * count_in + first) * _TILE
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, in_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, in_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    barrier()

    # Seeded to keep definite-assignment happy; the half-cleaner overwrites all
    # four before the merge reads them.
    var neg_inf = min_or_neg_inf[.float32]()
    var v0 = neg_inf
    var v1 = neg_inf
    var v2 = neg_inf
    var v3 = neg_inf
    var i0 = Int32(-1)
    var i1 = Int32(-1)
    var i2 = Int32(-1)
    var i3 = Int32(-1)

    for s in range(1, n_parts):
        var bs = (row * count_in + first + s) * _TILE
        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, in_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, in_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        barrier()

        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        # Finish champion/scratch reads before the merge overwrites scratch.
        barrier()

        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var out_base = (row * count_out + grp) * _TILE
    out_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )
    out_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        out_base + e0,
        champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0),
    )


@__name(t"merge_partials_topk")
def _merge_partials_kernel(
    part_v: UnsafePointer[Float32, ImmutAnyOrigin],
    part_i: UnsafePointer[Int32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    count_dev: Int32,
    K_dev: Int32,
):
    """Final round of the split streaming top-k (one block per row).

    Merges the `count` sorted top-`_TILE` partials of row `block_idx.x` into the
    final top-`K`. Each partial is already sorted descending, so folding one in
    is a half-cleaner (element-wise max against the reversed champion) followed
    by a single bitonic merge — no per-partial full sort. `count` is either the
    phase-1 split factor `S` (no tree reduction) or the residual partial count
    after `_reduce_partials_kernel` rounds.
    """
    var count = Int(count_dev)
    var K = Int(K_dev)

    var tid = thread_idx.x
    var row = block_idx.x

    # 16B-aligned so the canonical `e0..e3` accesses below are single 128-bit
    # LDS/STS (the swizzled sort/merge accesses stay scalar).
    var champ_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var champ_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_v = unsafe_stack_allocation[
        _TILE,
        Float32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()
    var scratch_i = unsafe_stack_allocation[
        _TILE,
        Int32,
        alignment=_V4_ALIGN,
        address_space=.SHARED,
    ]()

    var e0 = tid * 4
    var e1 = tid * 4 + 1
    var e2 = tid * 4 + 2
    var e3 = tid * 4 + 3

    # Seed the champion with partial 0.
    var base0 = (row * count) * _TILE
    champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, part_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
        e0, part_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](base0 + e0)
    )
    barrier()

    # Seeded to keep definite-assignment happy; the half-cleaner below
    # overwrites all four before the merge reads them.
    var neg_inf = min_or_neg_inf[.float32]()
    var v0 = neg_inf
    var v1 = neg_inf
    var v2 = neg_inf
    var v3 = neg_inf
    var i0 = Int32(-1)
    var i1 = Int32(-1)
    var i2 = Int32(-1)
    var i3 = Int32(-1)

    for s in range(1, count):
        var bs = (row * count + s) * _TILE
        scratch_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, part_v.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        scratch_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, part_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](bs + e0)
        )
        barrier()

        var mv, mi = _halfclean4(champ_v, champ_i, scratch_v, scratch_i, tid)
        v0 = mv[0]
        v1 = mv[1]
        v2 = mv[2]
        v3 = mv[3]
        i0 = mi[0]
        i1 = mi[1]
        i2 = mi[2]
        i3 = mi[3]
        # Finish champion/scratch reads before the merge overwrites scratch.
        barrier()

        _bitonic_merge_desc(
            v0, v1, v2, v3, i0, i1, i2, i3, scratch_v, scratch_i, tid
        )

        champ_v.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.float32, _PTOPK_ITEMS](v0, v1, v2, v3)
        )
        champ_i.store[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](
            e0, SIMD[.int32, _PTOPK_ITEMS](i0, i1, i2, i3)
        )
        barrier()

    var _K = Int(K)
    var base = row * _K
    var out_i = champ_i.load[width=_PTOPK_ITEMS, alignment=_V4_ALIGN](e0)
    if e0 < _K:
        out_idxs[base + e0] = out_i[0]
    if e1 < _K:
        out_idxs[base + e1] = out_i[1]
    if e2 < _K:
        out_idxs[base + e2] = out_i[2]
    if e3 < _K:
        out_idxs[base + e3] = out_i[3]


# ===----------------------------------------------------------------------=== #
# Host launcher
# ===----------------------------------------------------------------------=== #


def persistent_topk_block(
    ctx: DeviceContext,
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    total_seq_len: Int,
    row_bounds: Optional[UnsafePointer[Int32, ImmutAnyOrigin]] = None,
) raises:
    """Launch block-wide bitonic top-k for `total_seq_len` score rows.

    For `N ≤ PERSISTENT_TOPK_MAX_N` (= 2048) a single block sorts the whole row.
    For `N > PERSISTENT_TOPK_MAX_N` a streaming variant folds `_TILE`-wide tiles
    into a running top-`_TILE` champion; this requires `K ≤ PERSISTENT_TOPK_MAX_N`
    (the champion width).  Call sites needing `K > PERSISTENT_TOPK_MAX_N` must
    use `topk_gpu`.

    Each row of `N` float32 scores yields the `K` highest-scoring column indices
    (as int32) in descending score order in `out_idxs`.

    Args:
        ctx: Device context.
        in_scores: Flat score buffer `[total_seq_len × N]` row-major.
        out_idxs: Output buffer `[total_seq_len × K]` row-major (int32).
        N: Score columns per token (the row stride).
        K: Top-k count per token (≤ N, and ≤ PERSISTENT_TOPK_MAX_N when N > 2048).
        total_seq_len: Number of rows (one block per row).
        row_bounds: Optional `[total_seq_len]` int32 per-row live-column counts.
            When set, row `r` reads only its first `row_bounds[r]` columns;
            columns past the bound are never selected, never read (they may be
            uninitialized), and pad the output with `-1`. Scan cost tracks the
            real row lengths. Only supported for `N ≤ PERSISTENT_TOPK_MAX_N`
            here; wider rows go through `persistent_topk_block_split`.
    """
    if N <= PERSISTENT_TOPK_MAX_N:
        if row_bounds:
            ctx.enqueue_function[_persistent_topk_2048_bounded_kernel](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                row_bounds.value(),
                grid_dim=total_seq_len,
                block_dim=_PTOPK_BLOCK,
            )
        else:
            ctx.enqueue_function[_persistent_topk_2048_kernel](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                grid_dim=total_seq_len,
                block_dim=_PTOPK_BLOCK,
            )
    else:
        if row_bounds:
            raise Error(
                "row_bounds is not supported on the streaming top-k path;"
                " use persistent_topk_block_split for N > "
                + String(PERSISTENT_TOPK_MAX_N)
            )
        ctx.enqueue_function[_streaming_topk_kernel](
            in_scores,
            out_idxs,
            Int32(N),
            Int32(K),
            grid_dim=total_seq_len,
            block_dim=_PTOPK_BLOCK,
        )


@always_inline
def _choose_split_factor(rows: Int, num_tiles: Int, sm_count: Int) -> Int:
    """Pick the N-split factor `S` for the streaming top-k.

    Returns 1 (no split) when the rows already fill the GPU (splitting would
    only add merge overhead without new parallelism) or there is a single tile
    to fold. Otherwise splits as finely as the block budget allows: each of the
    `rows*S` phase-1 blocks then folds ~one tile (the cheapest phase 1), and the
    tree phase-2 reduces the resulting `S` partials in parallel so a large `S`
    is affordable. `S` is capped so `rows*S` stays within ~a couple of waves
    (`2*sm_count`); beyond that phase-1 blocks just serialize on the SMs.
    """
    if num_tiles <= 1 or rows >= sm_count:
        return 1
    var S = min(num_tiles, max(2, ceildiv(2 * sm_count, rows)))
    if S < 2:
        return 1
    return S


def persistent_topk_block_split[
    ordered: Bool = False, deterministic: Bool = False
](
    ctx: DeviceContext,
    in_scores: UnsafePointer[Float32, ImmutAnyOrigin],
    out_idxs: UnsafePointer[Int32, MutAnyOrigin],
    N: Int,
    K: Int,
    total_seq_len: Int,
    row_bounds: Optional[UnsafePointer[Int32, ImmutAnyOrigin]] = None,
) raises:
    """Launch bitonic top-k, splitting the N dimension when rows under-fill GPU.

    Same contract and output as `persistent_topk_block`. When the row count is
    small relative to the SM count and `N` spans many tiles (the long-context
    decode regime — a handful of blocks would otherwise each fold the whole row
    serially), the streaming fold is split across `rows * S` blocks (phase 1),
    each producing a sorted top-`_TILE` partial, then merged per row (phase 2).
    All other shapes fall back to `persistent_topk_block` unchanged.

    Parameters:
        ordered: Whether slot `q` must hold the `q`-th largest score. `True` is the
            historical contract: descending score, ties by ascending column.
            `False` promises the same `K` columns, deterministically, in an
            unspecified order -- which lets the short-row path skip its ranking
            pass, most of its work. The set does not depend on this, because the
            tie-break lives in the key the select already compares. Only shapes
            that have a cheaper path take one; the rest stay ordered, which
            satisfies the weaker promise too.
        deterministic: Whether one input must always give one output. `True`, the
            default, is the guarantee above: the same `K` columns every run.
            `False` drops it, and drops the *set* guarantee with it -- when more
            columns tie at the threshold than there are slots left, which of them
            survive is decided by the order the block's warps reach a cursor. In
            exchange the cheap contract's tail needs no ordering scan at all.
            Only meaningful together with `ordered=False`; the ordered path is
            deterministic by construction.

    Args:
        ctx: Device context.
        in_scores: Flat score buffer `[total_seq_len × N]` row-major.
        out_idxs: Output buffer `[total_seq_len × K]` row-major (int32).
        N: Score columns per token (the row stride).
        K: Top-k count per token (≤ N, and ≤ PERSISTENT_TOPK_MAX_N when N > 2048).
        total_seq_len: Number of rows.
        row_bounds: Optional `[total_seq_len]` int32 per-row live-column counts;
            see `persistent_topk_block`. Row `r` scans only `[0, row_bounds[r])`
            of its `N`-wide stripe, so under capture-frozen metadata (where `N`
            is a worst-case bound) the scan cost tracks each row's real length.
            Supported on every path this launcher selects for `N > 2048` (the
            histogram-select family) and on the 2048 single-block path.
    """
    if N <= PERSISTENT_TOPK_MAX_N:
        persistent_topk_block(
            ctx, in_scores, out_idxs, N, K, total_seq_len, row_bounds
        )
        return

    if N >= _HSEL_MIN_N:
        # Both selects below want the same question answered: do the rows fill
        # the GPU?
        var hsel_fills_gpu = total_seq_len >= ctx.get_attribute(
            DeviceAttribute.MULTIPROCESSOR_COUNT
        )

        # Short enough to hold in registers: read the row once and let every later
        # round re-test registers instead of gathering a parked candidate's score.
        # Rows must also under-fill the GPU, because this block's register count
        # leaves one block per SM where the streaming path's fits two -- and the
        # wider the payload the more that is true, which is why the width below is
        # chosen by `N` rather than fixed at the widest that fits.
        @__parameter
        @always_inline
        def launch_resident[res_vecs: Int]() raises:
            comptime if not ordered:
                # With no rank to feed there is no reason to hand over a superset
                # and no reason for the per-bin digit -- both exist only to make
                # ranking cheap. `sel_cap` of `_PTOPK_TOTAL` is what asks the
                # rounds to land exactly on `K` instead.
                if row_bounds:
                    ctx.enqueue_function[
                        _histsel_resident_bounded_kernel[
                            NullTrace,
                            sel_cap=_PTOPK_TOTAL,
                            ordered=False,
                            deterministic=deterministic,
                            res_vecs=res_vecs,
                        ]
                    ](
                        in_scores,
                        out_idxs,
                        Int32(N),
                        Int32(K),
                        row_bounds.value(),
                        NullTrace(),
                        grid_dim=total_seq_len,
                        block_dim=_HSEL_RES_BLOCK,
                    )
                    return
                ctx.enqueue_function[
                    _histsel_resident_kernel[
                        NullTrace,
                        sel_cap=_PTOPK_TOTAL,
                        ordered=False,
                        deterministic=deterministic,
                        res_vecs=res_vecs,
                    ]
                ](
                    in_scores,
                    out_idxs,
                    Int32(N),
                    Int32(K),
                    NullTrace(),
                    grid_dim=total_seq_len,
                    block_dim=_HSEL_RES_BLOCK,
                )
                return

            # Whether the rounds may hand a superset to the rank has to be a
            # property of the instantiation, because the slack costs something
            # even when it is never taken: the rank's slot loop is sized by the
            # capacity, so a wider `sel_k` walks every thread over twice the slots
            # to find the same live keys. It pays where the rank's runs are narrow,
            # and they are widest as `K` approaches `N` -- such a row selects every
            # score level including the masked tail, whose `phi` takes the digit's
            # top varying positions and leaves it unable to split within a level.
            if N >= K + K // 2:
                if row_bounds:
                    ctx.enqueue_function[
                        _histsel_resident_bounded_kernel[
                            NullTrace, res_vecs=res_vecs
                        ]
                    ](
                        in_scores,
                        out_idxs,
                        Int32(N),
                        Int32(K),
                        row_bounds.value(),
                        NullTrace(),
                        grid_dim=total_seq_len,
                        block_dim=_HSEL_RES_BLOCK,
                    )
                    return
                ctx.enqueue_function[
                    _histsel_resident_kernel[NullTrace, res_vecs=res_vecs]
                ](
                    in_scores,
                    out_idxs,
                    Int32(N),
                    Int32(K),
                    NullTrace(),
                    grid_dim=total_seq_len,
                    block_dim=_HSEL_RES_BLOCK,
                )
                return
            # Below the crossover the rank's runs would go wide, so this
            # instantiation takes the per-bin digit -- and only this one, since it
            # costs a scan and shared memory the shapes above do not need. With the
            # runs narrow again the hand-over is affordable here too.
            if row_bounds:
                ctx.enqueue_function[
                    _histsel_resident_bounded_kernel[
                        NullTrace, bin_digit=True, res_vecs=res_vecs
                    ]
                ](
                    in_scores,
                    out_idxs,
                    Int32(N),
                    Int32(K),
                    row_bounds.value(),
                    NullTrace(),
                    grid_dim=total_seq_len,
                    block_dim=_HSEL_RES_BLOCK,
                )
                return
            ctx.enqueue_function[
                _histsel_resident_kernel[
                    NullTrace, bin_digit=True, res_vecs=res_vecs
                ]
            ](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                NullTrace(),
                grid_dim=total_seq_len,
                block_dim=_HSEL_RES_BLOCK,
            )

        if not hsel_fills_gpu:
            if N <= _HSEL_RES_MAX:
                launch_resident[_HSEL_RES_VECS]()
                return
            # The wider payload serves the rank-free contracts only. Holding the
            # row saves the streaming select a re-read per round, which is what
            # pays for the registers; with a rank to run as well it measured
            # slower than streaming, so the ordered contract keeps streaming here
            # and is unchanged by this.
            comptime if not ordered:
                if N <= _HSEL_RES_MAX_WIDE:
                    launch_resident[_HSEL_RES_VECS_WIDE]()
                    return

        # Rows below the SM count leave every block resident whatever its
        # register count, so the scan can buy memory parallelism with registers
        # (see `prefetch`). Above it, occupancy is what throughput is made of.
        #
        # The rank-free contract takes the exact-fit `sel_cap` either way: the
        # slack exists to let a round stop early and hand the rank a superset, and
        # there is no rank here to drop the slack again.
        if not hsel_fills_gpu:
            comptime if (not ordered) and (not deterministic):
                if row_bounds:
                    ctx.enqueue_function[
                        _histsel_topk_bounded_kernel[
                            NullTrace,
                            prefetch=True,
                            tail_bits=_HSEL_TAIL_BITS,
                            ordered=False,
                            deterministic=False,
                        ]
                    ](
                        in_scores,
                        out_idxs,
                        Int32(N),
                        Int32(K),
                        row_bounds.value(),
                        NullTrace(),
                        grid_dim=total_seq_len,
                        block_dim=_PTOPK_BLOCK,
                        shared_mem_bytes=_HSEL_SMEM_BYTES,
                        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                            UInt32(_HSEL_SMEM_BYTES)
                        ),
                    )
                    return
                ctx.enqueue_function[
                    _histsel_topk_kernel[
                        NullTrace,
                        prefetch=True,
                        tail_bits=_HSEL_TAIL_BITS,
                        ordered=False,
                        deterministic=False,
                    ]
                ](
                    in_scores,
                    out_idxs,
                    Int32(N),
                    Int32(K),
                    NullTrace(),
                    grid_dim=total_seq_len,
                    block_dim=_PTOPK_BLOCK,
                    shared_mem_bytes=_HSEL_SMEM_BYTES,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(_HSEL_SMEM_BYTES)
                    ),
                )
                return
            if row_bounds:
                ctx.enqueue_function[
                    _histsel_topk_bounded_kernel[
                        NullTrace,
                        prefetch=True,
                        tail_bits=_HSEL_TAIL_BITS,
                        rank_bits=_HSEL_RANK_BITS,
                        rank_slots=True,
                        sel_cap=_HSEL_SEL_CAP,
                    ]
                ](
                    in_scores,
                    out_idxs,
                    Int32(N),
                    Int32(K),
                    row_bounds.value(),
                    NullTrace(),
                    grid_dim=total_seq_len,
                    block_dim=_PTOPK_BLOCK,
                    shared_mem_bytes=_HSEL_SMEM_BYTES,
                    func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                        UInt32(_HSEL_SMEM_BYTES)
                    ),
                )
                return
            ctx.enqueue_function[
                _histsel_topk_kernel[
                    NullTrace,
                    prefetch=True,
                    tail_bits=_HSEL_TAIL_BITS,
                    rank_bits=_HSEL_RANK_BITS,
                    rank_slots=True,
                    sel_cap=_HSEL_SEL_CAP,
                ]
            ](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                NullTrace(),
                grid_dim=total_seq_len,
                block_dim=_PTOPK_BLOCK,
                shared_mem_bytes=_HSEL_SMEM_BYTES,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(_HSEL_SMEM_BYTES)
                ),
            )
            return

        if row_bounds:
            ctx.enqueue_function[
                _histsel_topk_bounded_kernel[
                    NullTrace, ordered=ordered, deterministic=deterministic
                ]
            ](
                in_scores,
                out_idxs,
                Int32(N),
                Int32(K),
                row_bounds.value(),
                NullTrace(),
                grid_dim=total_seq_len,
                block_dim=_PTOPK_BLOCK,
                shared_mem_bytes=_HSEL_SMEM_BYTES,
                func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                    UInt32(_HSEL_SMEM_BYTES)
                ),
            )
            return
        ctx.enqueue_function[
            _histsel_topk_kernel[
                NullTrace, ordered=ordered, deterministic=deterministic
            ]
        ](
            in_scores,
            out_idxs,
            Int32(N),
            Int32(K),
            NullTrace(),
            grid_dim=total_seq_len,
            block_dim=_PTOPK_BLOCK,
            shared_mem_bytes=_HSEL_SMEM_BYTES,
            func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
                UInt32(_HSEL_SMEM_BYTES)
            ),
        )
        return

    if row_bounds:
        # Unreachable today: every N > PERSISTENT_TOPK_MAX_N takes the
        # histogram-select paths above (_HSEL_MIN_N adjoins the 2048 cutoff).
        raise Error(
            "row_bounds is not supported on the streaming split top-k path"
        )

    var _N = Int(N)
    var num_tiles = ceildiv(_N, _TILE)
    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)
    var S = _choose_split_factor(total_seq_len, num_tiles, sm_count)
    if S <= 1:
        persistent_topk_block(ctx, in_scores, out_idxs, N, K, total_seq_len)
        return

    var slice_len = ceildiv(N, S)
    var part_count = total_seq_len * S * _TILE

    # Phase 1: fan the streaming fold across `rows * S` blocks into buffer A.
    var buf_a_v = ctx.enqueue_create_buffer[.float32](part_count)
    var buf_a_i = ctx.enqueue_create_buffer[.int32](part_count)
    var a_v = rebind[UnsafePointer[Float32, MutAnyOrigin]](buf_a_v.unsafe_ptr())
    var a_i = rebind[UnsafePointer[Int32, MutAnyOrigin]](buf_a_i.unsafe_ptr())

    ctx.enqueue_function[_split_partial_kernel](
        in_scores,
        a_v,
        a_i,
        Int32(N),
        Int32(slice_len),
        Int32(S),
        grid_dim=total_seq_len * S,
        block_dim=_PTOPK_BLOCK,
    )

    # Phase 2: reduce the `S` partials per row toward `_MERGE_FANIN` via parallel
    # group merges (each round fanned across `rows * count_out` blocks),
    # ping-ponging A<->B, then a single per-row final merge to top-K. A serial
    # per-row merge would leave only `rows` blocks busy (8 SMs of 148 in the
    # long-context decode case), so the reduction is what fills the GPU.
    # Fan-in 3 is a deliberate latency trade: long-context decode -18.7% for
    # +1.5% on the MTP-decode shape (whose S = 5 then takes one reduce round),
    # accepted by the GLM serving owner; fan-in 5 keeps S = 5 reduce-free if
    # that trade is ever reversed.
    comptime _MERGE_FANIN = 3
    var count = S
    var src_v = a_v
    var src_i = a_i
    var do_reduce = S > _MERGE_FANIN
    var buf_b_v = ctx.enqueue_create_buffer[.float32](
        part_count if do_reduce else 1
    )
    var buf_b_i = ctx.enqueue_create_buffer[.int32](
        part_count if do_reduce else 1
    )
    var dst_v = rebind[UnsafePointer[Float32, MutAnyOrigin]](
        buf_b_v.unsafe_ptr()
    )
    var dst_i = rebind[UnsafePointer[Int32, MutAnyOrigin]](buf_b_i.unsafe_ptr())

    while count > _MERGE_FANIN:
        var count_out = ceildiv(count, _MERGE_FANIN)
        ctx.enqueue_function[_reduce_partials_kernel](
            rebind[UnsafePointer[Float32, ImmutAnyOrigin]](src_v),
            rebind[UnsafePointer[Int32, ImmutAnyOrigin]](src_i),
            dst_v,
            dst_i,
            Int32(count),
            Int32(count_out),
            Int32(_MERGE_FANIN),
            grid_dim=total_seq_len * count_out,
            block_dim=_PTOPK_BLOCK,
        )
        var tv = src_v
        src_v = dst_v
        dst_v = tv
        var ti = src_i
        src_i = dst_i
        dst_i = ti
        count = count_out

    ctx.enqueue_function[_merge_partials_kernel](
        rebind[UnsafePointer[Float32, ImmutAnyOrigin]](src_v),
        rebind[UnsafePointer[Int32, ImmutAnyOrigin]](src_i),
        out_idxs,
        Int32(count),
        Int32(K),
        grid_dim=total_seq_len,
        block_dim=_PTOPK_BLOCK,
    )

    _ = buf_a_v
    _ = buf_a_i
    _ = buf_b_v
    _ = buf_b_i
