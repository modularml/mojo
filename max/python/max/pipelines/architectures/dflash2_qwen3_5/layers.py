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
"""The two modules DFlash2 adds on top of the DFlash v1 draft body.

Both are transcribed from the authors' own vLLM fork
(``vllm/model_executor/models/qwen3_dflash2.py``): ``_grouped_conv`` /
``DFlashGroupedConv`` and ``_score_edges`` / ``CandidateSelector``.
"""

from __future__ import annotations

from max.dtype import DType
from max.graph import DeviceRef, TensorValue, Weight, ops
from max.nn.layer import Module
from max.nn.linear import Linear


class DFlash2GroupedConv(Module):
    """One sublayer's pair of dynamic grouped depthwise convolutions.

    Applied along the *block* axis, once before the sublayer and once after
    it::

        out[t] = sum_tap (base_kernel[side, tap] + delta[t, tap, group(c)])
                 * x[t - tap]

    ``base_kernel`` is per-channel at full resolution; ``delta`` is per group
    of ``group_size`` channels and broadcast across the group. ``t`` is
    block-local, so the ``t - tap`` taps are zero for the first ``tap``
    positions of every block and the convolution never reads across a block
    boundary.

    ``kernel_projection`` runs *once*, on the sublayer's input, and emits the
    deltas for both sides at once — hence its ``2 * taps * num_groups``
    output width. :meth:`prepare` applies side 0 and hands back side 1's
    coefficients for :meth:`finish` to reuse after the sublayer.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        taps: int,
        group_size: int,
        block_size: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        if hidden_size % group_size:
            raise ValueError(
                f"conv_group_size={group_size} must divide"
                f" hidden_size={hidden_size}."
            )
        if taps < 1:
            raise ValueError(f"conv_kernel_size must be >= 1, got {taps}.")
        if taps > block_size:
            raise ValueError(
                f"conv_kernel_size={taps} exceeds block_size={block_size};"
                " the convolution never reads across a block boundary."
            )
        self.hidden_size = hidden_size
        self.taps = taps
        self.group_size = group_size
        self.block_size = block_size
        self.num_groups = hidden_size // group_size

        self.base_kernel = Weight(
            "base_kernel", dtype, [2, taps, hidden_size], device=device
        )
        self.kernel_projection = Linear(
            in_dim=hidden_size,
            out_dim=2 * taps * self.num_groups,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def prepare(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        """Applies the pre-sublayer convolution.

        Args:
            hidden_states: ``[total_tokens, hidden_size]``, one dense
                ``block_size``-row block per sequence.

        Returns:
            The convolved states and the ``[total_tokens, taps, num_groups]``
            coefficients :meth:`finish` must reuse.
        """
        coefficients = self.kernel_projection(hidden_states).reshape(
            (-1, 2, self.taps, self.num_groups)
        )
        return (
            self._convolve(hidden_states, coefficients[:, 0], side=0),
            coefficients[:, 1],
        )

    def finish(
        self, hidden_states: TensorValue, coefficients: TensorValue
    ) -> TensorValue:
        """Applies the post-sublayer convolution with :meth:`prepare`'s
        coefficients."""
        return self._convolve(hidden_states, coefficients, side=1)

    def _convolve(
        self, hidden_states: TensorValue, delta: TensorValue, *, side: int
    ) -> TensorValue:
        block_size = self.block_size
        # The block axis is explicit here rather than derived from a modulo
        # over a flat position, which is what makes the boundary zeroing
        # structural: the shift below pads with zeros inside each block.
        blocks = hidden_states.reshape(
            (-1, block_size, self.num_groups, self.group_size)
        )
        base = (
            self.base_kernel[side]
            .cast(blocks.dtype)
            .reshape((1, 1, self.taps, self.num_groups, self.group_size))
        )
        coefficients = base + ops.unsqueeze(
            delta.reshape((-1, block_size, self.taps, self.num_groups)), -1
        )

        out = coefficients[:, :, 0] * blocks
        for tap in range(1, self.taps):
            zeros = ops.broadcast_to(
                ops.constant(0, blocks.dtype, device=blocks.device),
                [blocks.shape[0], tap, self.num_groups, self.group_size],
            )
            shifted = ops.concat([zeros, blocks[:, :-tap]], axis=1)
            out = out + coefficients[:, :, tap] * shifted
        return out.reshape((-1, self.hidden_size))

    def __call__(
        self, hidden_states: TensorValue
    ) -> tuple[TensorValue, TensorValue]:
        # Alias for prepare to satisfy the Module ABC.
        return self.prepare(hidden_states)


class DFlash2CandidateSelector(Module):
    """Low-rank bilinear scorer over adjacent candidate tokens, plus its walk.

    ``S[l, p, c] = unary[l, c] + <pred_book[pred_id[l, p]] * proj(h_l),
    succ_book[cand_id[l, c]]>``, where ``pred_id`` at slot ``l`` is slot
    ``l - 1``'s candidate list and, at ``l == 0``, the anchor (the last
    verified token) broadcast across the predecessor axis.

    The predecessor/successor role assignment is read from ``_score_edges``
    in the authors' fork, not inferred from the tensor names.
    """

    def __init__(
        self,
        hidden_size: int,
        *,
        vocab_size: int,
        rank: int,
        top_k: int,
        dtype: DType,
        device: DeviceRef,
    ) -> None:
        super().__init__()
        self.top_k = top_k
        self.rank = rank
        self.vocab_size = vocab_size
        self.predecessor_codebook = Weight(
            "predecessor_codebook", dtype, [vocab_size, rank], device=device
        )
        self.successor_codebook = Weight(
            "successor_codebook", dtype, [vocab_size, rank], device=device
        )
        self.hidden_projection = Linear(
            in_dim=hidden_size,
            out_dim=rank,
            dtype=dtype,
            device=device,
            has_bias=False,
        )

    def score_edges(
        self,
        candidate_ids: TensorValue,
        unary_logits: TensorValue,
        hidden_states: TensorValue,
        anchor_token_ids: TensorValue,
    ) -> TensorValue:
        """Scores every adjacent candidate pair in the block.

        Args:
            candidate_ids: ``[batch, steps, top_k]`` top-k ids per mask slot.
            unary_logits: ``[batch, steps, top_k]`` their draft-head logits.
            hidden_states: ``[batch, steps, hidden]`` drafter output per slot.
            anchor_token_ids: ``[batch]`` id of the last verified token.

        Returns:
            ``[batch, steps, top_k, top_k]`` indexed ``[b, l, p, c]``.
        """
        top_k = self.top_k
        hidden = self.hidden_projection(hidden_states)
        successors = ops.gather(self.successor_codebook, candidate_ids, axis=0)
        anchor_row = ops.broadcast_to(
            anchor_token_ids.reshape((-1, 1, 1)),
            [candidate_ids.shape[0], 1, top_k],
        )
        predecessor_ids = ops.concat(
            [anchor_row, candidate_ids[:, :-1]], axis=1
        )
        predecessors = ops.gather(
            self.predecessor_codebook, predecessor_ids, axis=0
        )
        edges = ops.matmul(
            predecessors * ops.unsqueeze(hidden, 2),
            ops.transpose(successors, -1, -2),
        )
        return ops.unsqueeze(unary_logits, 2) + edges

    def select_path(
        self, scores: TensorValue, candidate_ids: TensorValue
    ) -> TensorValue:
        """Walks the score tensor left to right, following the best successor.

        A single greedy pass, unrolled at graph-build time over the static
        slot axis: no beam and no Viterbi. Slot 0's predecessor row is the
        anchor broadcast across ``top_k``, so seeding the walk at index 0
        selects the same row as any other index.

        Args:
            scores: ``[batch, steps, top_k, top_k]`` from
                :meth:`score_edges`.
            candidate_ids: ``[batch, steps, top_k]``.

        Returns:
            ``[batch, steps]`` chosen token ids.
        """
        steps = int(scores.shape[1])
        previous = ops.broadcast_to(
            ops.constant(0, DType.int64, device=scores.device),
            [scores.shape[0], 1],
        )
        tokens: list[TensorValue] = []
        for step in range(steps):
            row = ops.gather_nd(scores[:, step], previous, batch_dims=1)
            previous = ops.argmax(row, axis=-1).cast(DType.int64)
            tokens.append(
                ops.gather_nd(candidate_ids[:, step], previous, batch_dims=1)
            )
        return ops.stack(tokens, axis=1)

    def __call__(
        self,
        candidate_ids: TensorValue,
        unary_logits: TensorValue,
        hidden_states: TensorValue,
        anchor_token_ids: TensorValue,
    ) -> TensorValue:
        # Alias for score_edges to satisfy the Module ABC.
        return self.score_edges(
            candidate_ids, unary_logits, hidden_states, anchor_token_ids
        )
