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
"""Isolated correctness tests for `block_select_topk` (MSA indexer top-k).

Each case launches one thread block per score row and checks the selected
indices for:
  - the top-k invariant: every selected score >= every non-selected score;
  - distinctness, in-range, and the `-1` sentinel tail when `k > num_blocks`;
  - on tie-free random data, exact-set equality with a host top-k reference.

Cases cover random scores, all-equal scores (stresses winner eviction), a
boundary tie, a forced large-value block, the 512K-context row (4096 blocks),
`k > num_blocks`, a single-block row, and degenerate rows with no selectable
value (all `-inf` or all `NaN`), which must emit all `-1` without writing out
of bounds.

A final group asserts the output does not depend on `block_dim`. That is not a
property of this kernel alone -- the MSA prefill launcher SIZES its CTA to the
work (`sparse_indexer_prefill.mojo`'s `clamp(align_up(ceildiv(max_num_blocks,
16), WARP_SIZE), WARP_SIZE, 128)`), which is only sound because the winner is
the globally smallest index among tied maxima at every width. The argument is
that `insert` keeps the lowest index it saw and the block reduction breaks ties
the same way; `_run_block_dim_invariance` turns that argument into evidence.
"""

from std.collections import Set
from max.gpu import WARP_SIZE, block_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from std.math import min
from std.random import rand
from std.testing import assert_equal, assert_true
from std.utils.index import Index
from std.utils.numerics import min_or_neg_inf, nan

from layout import TensorLayout, TileTensor, row_major

from nn.attention.gpu.sparse_indexer_common import block_select_topk

# Fill modes for the score rows.
comptime MODE_RANDOM = 0  # tie-free random -> exact set cross-check is valid
comptime MODE_ALL_EQUAL = 1  # every score equal -> stresses winner eviction
comptime MODE_BOUNDARY_TIE = 2  # distinct highs + a tie for the last slot
comptime MODE_ALL_DEAD = 3  # all -inf/dead -> no selectable winner
comptime MODE_ALL_NAN = 4  # all NaN -> no selectable winner


def _select_test_kernel[
    ScoresLT: TensorLayout,
    OutLT: TensorLayout,
](
    scores: TileTensor[.float32, ScoresLT, MutAnyOrigin],
    out_idxs: TileTensor[.int32, OutLT, MutAnyOrigin],
    num_blocks_dev: Int32,
    k_dev: Int32,
):
    var num_blocks = Int(num_blocks_dev)
    var k = Int(k_dev)
    comptime assert scores.flat_rank == 2 and out_idxs.flat_rank == 2
    var row = block_idx.x
    var s_lt = scores.to_layout_tensor()
    var o_lt = out_idxs.to_layout_tensor()
    var scores_row = rebind[MutPointer[Float32, MutAnyOrigin]](
        s_lt.ptr_at_offset(Index(row, 0))
    )
    var out_row = rebind[MutPointer[Int32, MutAnyOrigin]](
        o_lt.ptr_at_offset(Index(row, 0))
    )
    block_select_topk[.float32, DType.int32](scores_row, num_blocks, k, out_row)


def _host_topk_set(
    scores: MutPointer[Float32, MutAnyOrigin],
    num_blocks: Int,
    k: Int,
) -> Set[Int]:
    """Reference top-k-by-value (ties: smaller index) returned as an index set.
    """
    var k_batch = min(k, num_blocks)
    var picked = Set[Int]()
    for _pick in range(k_batch):
        var best_idx = -1
        var best_val = Float32(0)
        for i in range(num_blocks):
            if i in picked:
                continue
            var v = scores[i]
            if best_idx == -1 or v > best_val:
                best_val = v
                best_idx = i
        picked.add(best_idx)
    return picked^


def _fill_row(
    row_ptr: MutPointer[Float32, _],
    num_blocks: Int,
    k: Int,
    mode: Int,
):
    if mode == MODE_ALL_EQUAL:
        for j in range(num_blocks):
            row_ptr[j] = Float32(0.5)
    elif mode == MODE_BOUNDARY_TIE:
        # All tied at 50, except the first (k-1) blocks set to distinct highs.
        # Top-k must therefore be {0 .. k-2} plus exactly one tied block.
        for j in range(num_blocks):
            row_ptr[j] = Float32(50.0)
        for j in range(k - 1):
            row_ptr[j] = Float32(1000.0 + Float32(k - 1 - j))
    elif mode == MODE_ALL_DEAD:
        for j in range(num_blocks):
            row_ptr[j] = min_or_neg_inf[.float32]()
    elif mode == MODE_ALL_NAN:
        for j in range(num_blocks):
            row_ptr[j] = nan[.float32]()


def _run_case(
    num_rows: Int,
    num_blocks: Int,
    k: Int,
    force_idx: Int,
    mode: Int,
    ctx: DeviceContext,
) raises:
    print(
        "  case: num_rows=",
        num_rows,
        " num_blocks=",
        num_blocks,
        " k=",
        k,
        " force_idx=",
        force_idx,
        " mode=",
        mode,
    )
    var n_scores = num_rows * num_blocks
    var n_out = num_rows * k

    var scores_host = ctx.enqueue_create_host_buffer[.float32](n_scores)
    var out_host = ctx.enqueue_create_host_buffer[.int32](n_out)

    if mode == MODE_RANDOM:
        rand(scores_host.as_span())
    else:
        for r in range(num_rows):
            _fill_row(
                scores_host.unsafe_ptr() + r * num_blocks, num_blocks, k, mode
            )
    # Force a known block to a large value in every row; it must be selected.
    if force_idx >= 0:
        for r in range(num_rows):
            scores_host[r * num_blocks + force_idx] = Float32(1.0e30)
    ctx.synchronize()

    # Leading guard element at index 0; row data lives at [1, n_scores]. A
    # write to scores[-1] for row 0 (the pre-guard OOB) would land on this
    # canary, so the readback below detects it.
    var canary = Float32(13371337.0)
    var scores_dev = ctx.enqueue_create_buffer[.float32](n_scores + 1)
    var stage = ctx.enqueue_create_host_buffer[.float32](n_scores + 1)
    stage[0] = canary
    for j in range(n_scores):
        stage[1 + j] = scores_host[j]
    ctx.enqueue_copy(dst_buf=scores_dev, src_buf=stage)

    var out_dev = ctx.enqueue_create_buffer[.int32](n_out)
    out_dev.enqueue_fill(Int32(-2))  # poison: must be overwritten

    var data_buf = DeviceBuffer[.float32](
        ctx, scores_dev.unsafe_ptr() + 1, n_scores, owning=False
    )
    var scores_t = TileTensor(data_buf, row_major((num_rows, num_blocks)))
    var out_t = TileTensor(out_dev, row_major((num_rows, k)))

    comptime kernel = _select_test_kernel[
        type_of(scores_t).LayoutType, type_of(out_t).LayoutType
    ]
    ctx.enqueue_function[kernel](
        scores_t,
        out_t,
        Int32(num_blocks),
        Int32(k),
        grid_dim=num_rows,
        block_dim=128,
    )
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)

    # Read back the buffer (with its leading canary) and confirm the guard
    # element was not overwritten -- i.e. no `scores[-1]` write occurred.
    var full_host = ctx.enqueue_create_host_buffer[.float32](n_scores + 1)
    ctx.enqueue_copy(dst_buf=full_host, src_buf=scores_dev)
    ctx.synchronize()
    assert_equal(full_host[0], canary, "OOB write to scores[-1] before row 0")

    var host_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scores_host.unsafe_ptr()
    )
    var no_winner = mode == MODE_ALL_DEAD or mode == MODE_ALL_NAN
    var k_batch = min(k, num_blocks)
    for r in range(num_rows):
        var base = r * num_blocks

        # No selectable value in the row: the kernel must not write out of
        # bounds and must emit all `-1` (reaching here at all proves no OOB
        # write / no barrier deadlock occurred).
        if no_winner:
            for j in range(k):
                assert_equal(
                    Int(out_host[r * k + j]), -1, "no-winner row must be all -1"
                )
            continue

        var got = Set[Int]()
        for j in range(k_batch):
            var idx = Int(out_host[r * k + j])
            assert_true(
                idx >= 0 and idx < num_blocks, "selected index out of range"
            )
            got.add(idx)
        assert_equal(len(got), k_batch, "duplicate or missing selected index")

        # Sentinel tail must be -1.
        for j in range(k_batch, k):
            assert_equal(Int(out_host[r * k + j]), -1, "tail must be -1")

        # Top-k invariant: every selected score >= every non-selected score.
        var sel_min = Float32(0)
        var have_sel = False
        for idx in got:
            var v = host_ptr[base + idx]
            if not have_sel or v < sel_min:
                sel_min = v
                have_sel = True
        var non_max = Float32(0)
        var have_non = False
        for i in range(num_blocks):
            if i in got:
                continue
            var v = host_ptr[base + i]
            if not have_non or v > non_max:
                non_max = v
                have_non = True
        if have_sel and have_non:
            assert_true(
                sel_min >= non_max,
                "selected element ranked below an excluded one",
            )

        # Forced block must be selected.
        if force_idx >= 0:
            assert_true(force_idx in got, "forced block not selected")

        if mode == MODE_RANDOM:
            # Tie-free: exact set must match the independent host reference.
            var expected = _host_topk_set(host_ptr + base, num_blocks, k)
            for e in expected:
                assert_true(e in got, "kernel missed an expected top-k index")
        elif mode == MODE_BOUNDARY_TIE:
            # The distinct clear-highs (indices 0..k-2) must all be selected.
            for j in range(k - 1):
                assert_true(j in got, "clear-high block not selected")
    _ = scores_dev
    _ = out_dev


def _launch_select(
    num_rows: Int,
    num_blocks: Int,
    k: Int,
    scores_host: MutPointer[Float32, MutAnyOrigin],
    block_dim: Int,
    ctx: DeviceContext,
) raises -> List[Int32]:
    """Select over a FRESH device copy of `scores_host` at one CTA width.

    The copy is per call, not per case: `block_select_topk` evicts each winner
    in place, so a second launch over the same device buffer would select from
    an already-consumed row and agree with the first for the wrong reason.
    """
    var n_scores = num_rows * num_blocks
    var n_out = num_rows * k

    var stage = ctx.enqueue_create_host_buffer[.float32](n_scores)
    for j in range(n_scores):
        stage[j] = scores_host[j]
    var scores_dev = ctx.enqueue_create_buffer[.float32](n_scores)
    ctx.enqueue_copy(dst_buf=scores_dev, src_buf=stage)

    var out_dev = ctx.enqueue_create_buffer[.int32](n_out)
    out_dev.enqueue_fill(Int32(-2))  # poison: must be overwritten

    var scores_t = TileTensor(scores_dev, row_major((num_rows, num_blocks)))
    var out_t = TileTensor(out_dev, row_major((num_rows, k)))
    comptime kernel = _select_test_kernel[
        type_of(scores_t).LayoutType, type_of(out_t).LayoutType
    ]
    ctx.enqueue_function[kernel](
        scores_t,
        out_t,
        Int32(num_blocks),
        Int32(k),
        grid_dim=num_rows,
        block_dim=block_dim,
    )

    var out_host = ctx.enqueue_create_host_buffer[.int32](n_out)
    ctx.enqueue_copy(dst_buf=out_host, src_buf=out_dev)
    ctx.synchronize()
    var out = List[Int32](capacity=n_out)
    for i in range(n_out):
        out.append(out_host[i])
    _ = scores_dev
    _ = out_dev
    return out^


def _run_block_dim_invariance(
    num_rows: Int,
    num_blocks: Int,
    k: Int,
    mode: Int,
    ctx: DeviceContext,
) raises:
    """Every legal CTA width must emit a bit-identical `out_idxs`.

    The ladder is derived from `WARP_SIZE` rather than hard-coded, because the
    contract is "a positive multiple of the warp size, at most 1024" and a
    wave-64 arch has no legal 32-wide block. `3 * WARP_SIZE` is deliberate: the
    contract admits non-power-of-two widths, and the prefill launcher's
    `align_up(..., WARP_SIZE)` sizing rule can land on one.

    Tie-heavy modes carry this test. On tie-free data any correct selector
    agrees at every width, so `MODE_ALL_EQUAL` / `MODE_BOUNDARY_TIE` are what
    would actually catch a width-dependent tiebreak.
    """
    print(
        "  block_dim invariance: num_rows=",
        num_rows,
        " num_blocks=",
        num_blocks,
        " k=",
        k,
        " mode=",
        mode,
    )
    var n_scores = num_rows * num_blocks
    var scores_host = ctx.enqueue_create_host_buffer[.float32](n_scores)
    if mode == MODE_RANDOM:
        rand(scores_host.as_span())
    else:
        for r in range(num_rows):
            _fill_row(
                scores_host.unsafe_ptr() + r * num_blocks, num_blocks, k, mode
            )
    ctx.synchronize()
    var host_ptr = rebind[MutPointer[Float32, MutAnyOrigin]](
        scores_host.unsafe_ptr()
    )

    var bdims = List[Int]()
    for mult in [1, 2, 3, 4, 8, 16, 32]:
        if mult * WARP_SIZE <= 1024:
            bdims.append(mult * WARP_SIZE)

    var reference = _launch_select(
        num_rows, num_blocks, k, host_ptr, bdims[0], ctx
    )
    for bdim in bdims:
        var got = _launch_select(num_rows, num_blocks, k, host_ptr, bdim, ctx)
        for i in range(len(reference)):
            assert_equal(
                got[i],
                reference[i],
                String(
                    "block_dim ",
                    bdim,
                    " differs from ",
                    bdims[0],
                    " at slot ",
                    i,
                    ": ",
                    got[i],
                    " vs ",
                    reference[i],
                ),
            )


def main() raises:
    with DeviceContext() as ctx:
        # Tie-free random (exact set cross-check). k <= num_blocks.
        _run_case(4, 64, 16, -1, MODE_RANDOM, ctx)
        _run_case(4, 256, 16, 7, MODE_RANDOM, ctx)
        # Headline long-context row: 512K ctx / 128 block = 4096 blocks.
        _run_case(4, 4096, 16, 4095, MODE_RANDOM, ctx)
        # topk > num_blocks: k_batch = num_blocks, rest -1.
        _run_case(4, 8, 16, 0, MODE_RANDOM, ctx)
        # Single block row.
        _run_case(2, 1, 16, 0, MODE_RANDOM, ctx)
        # All-equal scores (stresses winner eviction).
        _run_case(4, 64, 16, -1, MODE_ALL_EQUAL, ctx)
        _run_case(4, 4096, 16, -1, MODE_ALL_EQUAL, ctx)
        # Boundary tie: distinct highs + a tie for the last selected slot.
        _run_case(4, 64, 16, -1, MODE_BOUNDARY_TIE, ctx)
        _run_case(4, 256, 16, -1, MODE_BOUNDARY_TIE, ctx)
        # No selectable winner (all -inf / all NaN): must not write OOB; all -1.
        _run_case(4, 64, 16, -1, MODE_ALL_DEAD, ctx)
        _run_case(2, 4096, 16, -1, MODE_ALL_NAN, ctx)

        # Width invariance, which is what makes the prefill launcher's dynamic
        # CTA sizing sound. 1024 blocks spans every width in the ladder; the
        # narrow rows cover the widths that exceed the row itself (most threads
        # contribute nothing to the reduction) and the no-winner early-out.
        _run_block_dim_invariance(8, 1024, 16, MODE_ALL_EQUAL, ctx)
        _run_block_dim_invariance(8, 1024, 16, MODE_BOUNDARY_TIE, ctx)
        _run_block_dim_invariance(8, 1024, 16, MODE_RANDOM, ctx)
        _run_block_dim_invariance(8, 8, 16, MODE_RANDOM, ctx)
        _run_block_dim_invariance(8, 8, 16, MODE_ALL_NAN, ctx)
        print("all block_select_topk cases passed")
