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
"""Shared device functions for the MiniMax-M3 sparse-attention (MSA) indexer.

The MSA indexer selects, per query and per index head, the top-k key *blocks*
to attend to. This module holds the block-selection primitive shared by the
prefill and decode indexer kernels; the per-block scoring (QK -> block-max ->
local-block forcing) lives in the phase-specific kernels that produce the score
buffer this primitive consumes.

`block_select_topk` is a cooperative, single-CTA-per-row top-k: one thread block
selects the top-k block indices from one row of block scores, reusing the
`TopK_2` / `_block_reduce_topk` primitives from `nn.topk` via the same iterative
max-extract as `topk_gpu`'s stage 2. It is a straight-line re-scan with uniform
control flow (so the block barriers are safe) and allocates no global scratch,
which keeps it usable inside a CUDA-graph capture region. A register-heap fast
path is a possible future optimization.
"""

from max.gpu import block_dim, thread_idx
from max.gpu.sync import barrier
from std.math import min
from std.memory import unsafe_stack_allocation

from nn.topk import TopK_2, _block_reduce_topk, _topk_dead_val


@always_inline
def block_select_topk[
    T: DType,
    out_idx_type: DType,
    largest: Bool = True,
](
    scores: UnsafePointer[Scalar[T], MutAnyOrigin],
    num_blocks: Int,
    k: Int,
    out_idxs: UnsafePointer[Scalar[out_idx_type], MutAnyOrigin],
):
    """Select the top-k block indices from one row of block scores.

    Cooperative across the whole thread block: on each of `k_batch = min(k,
    num_blocks)` iterations, every thread finds the best score over its strided
    slice of `scores[0:num_blocks]`, a block-wide reduction picks the global
    winner, thread 0 records the winner's index and evicts it (writes the dead
    sentinel so it cannot be reselected), and a barrier makes the eviction
    visible before the next iteration. Output positions `[k_batch, k)` -- or
    earlier, if the row runs out of selectable (finite, for `largest`) values --
    are filled with the `-1` sentinel.

    Forcing (e.g. always-keep the local block) must already be baked into
    `scores` by the caller (write a large sentinel value into that block before
    calling); selection is purely by value, so a forced block wins a slot.

    Parameters:
        T: Element dtype of the scores (float32 expected for MSA).
        out_idx_type: Output index dtype (int32 for MSA).
        largest: Select largest (True) or smallest values.

    Args:
        scores: Pointer to this row's block scores, length `num_blocks`. Mutated
            in place during extraction (the caller must treat it as scratch).
        num_blocks: Number of valid block scores in the row.
        k: Number of indices to emit (output length).
        out_idxs: Pointer to this row's output indices, length `k`.

    Note:
        One thread block per call (`grid_dim` = one block per row). `block_dim.x`
        must be a multiple of the warp size (required by `_block_reduce_topk`),
        and all threads must reach every iteration uniformly -- guaranteed here
        because `k_batch` depends only on `k` and `num_blocks`, identical across
        the block.
    """
    var tid = thread_idx.x
    var bsize = block_dim.x
    var k_batch = min(k, num_blocks)

    # 1-element shared scratch to broadcast the winner index so every thread
    # reaches the same control-flow decision (and thus the same barriers).
    var winner_sram = unsafe_stack_allocation[1, Int, address_space=.SHARED]()

    var n_written = k_batch
    for kk in range(k_batch):
        # Every thread's best over its strided slice of the row.
        var partial = TopK_2[T, largest]()
        for i in range(tid, num_blocks, bsize):
            partial.insert(scores[i], i)

        # Block-wide best (valid at thread 0); ties broken by smallest index.
        var total = _block_reduce_topk[ascending=largest](partial)
        if tid == 0:
            # Write -1 when no selectable value remains (u == dead_val covers
            # both all-evicted and all-NaN rows, since NaN never beats dead_val
            # in insert() and p stays at its 0 default).
            if total.u == _topk_dead_val[T, largest]():
                winner_sram[0] = -1
            else:
                winner_sram[0] = total.p
        barrier()

        # `p < 0` means no selectable value remains in the row (all evicted, or
        # the live values are non-finite and never beat the dead sentinel).
        # Stop uniformly across the block and `-1`-pad the remaining outputs.
        var winner_p = winner_sram[0]
        if winner_p < 0:
            n_written = kk
            break

        if tid == 0:
            out_idxs[kk] = Int(winner_p).cast[out_idx_type]()
            # Evict the winner so the next iteration cannot reselect it.
            scores[winner_p] = _topk_dead_val[T, largest]()
        # Make the eviction visible before any thread re-reads `scores`.
        barrier()

    # Sentinel-fill the unused tail [n_written, k) with -1, in parallel.
    for rem in range(n_written + tid, k, bsize):
        out_idxs[rem] = Scalar[out_idx_type](-1)
