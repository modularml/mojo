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
"""Qwen3.5 fused with a DFlash2 block drafter: verify, roll back, draft.

Two precedents meet here and each supplies the half it already solved.

From :mod:`..unified_mtp_qwen3_5` comes everything the hybrid target needs:
verifying a block advances 48 Gated DeltaNet recurrences that no length
pointer can rewind, so the verify runs on a shadow copy of the state pools and
the accepted prefix is replayed into the live ones. That machinery is
parameterized in the accepted length, not in ``K``, so it carries to a block of
8 unchanged -- see :mod:`..unified_mtp_qwen3_5.state_rollback`.

From :mod:`..unified_dflash_gemma4_31b` comes the drafter half: the target's
tapped hidden states become the drafter's context K/V, and one non-causal
forward over ``block_size`` query rows proposes the whole next block. What
differs from that precedent is the drafter body (DFlash2 adds a dynamic
convolution and a candidate selector, both in :mod:`..dflash2_qwen3_5`) and
the draft KV leaf, which is windowed here rather than full-length.

Acceptance is stochastic, never the greedy fast path, for the same reason the
MTP graph gives: only the stochastic sampler applies the grammar bitmask, and
this checkpoint's ``lm_head`` carries 243 live padding rows past
``sampleable_vocab_size`` that nothing else excludes.
"""

from __future__ import annotations

from dataclasses import replace

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import AcceptanceSampler
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)
from max.pipelines.speculative.unified_graph_ops import (
    apply_overlap_bitmask,
    merge_tokens_and_host_offsets,
)

from ..dflash2_qwen3_5 import DFlash2Qwen3_5
from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs
from ..qwen3_5.qwen3_5 import Qwen3_5, Qwen3_5LinearAttentionBlock
from ..unified_mtp_qwen3_5.state_rollback import (
    accepted_row_plan,
    replay_state_pools,
    snapshot_state_pools,
)
from .model_config import UnifiedDflash2Qwen3_5Config


def _block_dispatch_metadata(meta: TensorValue | None, k: int) -> TensorValue:
    """Rebuilds the MHA dispatch metadata at the draft block's query width.

    The 4-int CPU buffer is ``[batch_size, q_max_seq_len, num_partitions,
    max_cache_valid_length]``. Neither manager-supplied buffer fits the block:
    the leaf's own metadata carries the *verify* query width, which equals the
    block only on a decode batch and is far larger on a prefill batch, where
    the oversized query bound drives the block's layer-0 flash attention to
    NaN. ``num_partitions`` is zeroed so the kernel recomputes the split-K
    count for the drafter's own head geometry instead of reusing the target's.
    """
    assert meta is not None
    cpu = DeviceRef.CPU()
    return ops.concat(
        [
            meta[0:1],
            ops.constant(k, DType.int64, device=cpu).reshape((1,)),
            ops.constant(0, DType.int64, device=cpu).reshape((1,)),
            meta[3:4],
        ],
        axis=0,
    )


class UnifiedDflash2Qwen3_5(Module):
    """Fused module: merge, verify, accept, roll back, materialize, draft."""

    def __init__(
        self,
        config: UnifiedDflash2Qwen3_5Config,
        enable_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_structured_output = enable_structured_output
        self.block_size = config.block_size
        # The anchor slot carries the committed/bonus token and never predicts,
        # so a block of B proposes B - 1 tokens.
        self.num_speculative_tokens = config.num_speculative_tokens
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = config.mask_token_id

        speculative_config = config.speculative_config
        if speculative_config.use_greedy_acceptance:
            raise ValueError(
                "DFlash2 on Qwen3.5 requires stochastic acceptance: the greedy"
                " path ignores token_bitmasks, and this checkpoint's lm_head"
                " padding rows are only excluded through the bitmask."
            )
        if speculative_config.synthetic_acceptance_rate is not None:
            raise ValueError(
                "synthetic acceptance would bypass the state rollback's"
                " accepted-length plan"
            )
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if speculative_config.use_relaxed_acceptance_for_thinking:
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta
        self.acceptance_sampler = AcceptanceSampler(
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )

        self.target = Qwen3_5(config.target)
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
        self.draft = DFlash2Qwen3_5(
            config.draft,
            num_context_features=len(self.target_layer_ids),
            block_size=self.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
            selector_rank=config.selector_rank,
            selector_top_k=config.selector_top_k,
            layer_types=config.layer_types or None,
        )
        self.merger = RaggedTokenMerger(config.target.devices[0])
        self.num_linear_layers = len(self.target.linear_layer_indices)

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        target_kv: list[PagedCacheValues],
        draft_kv: list[PagedCacheValues],
        return_n_logits: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        in_thinking_phase: TensorValue,
        slot_idx: list[TensorValue],
        live_conv_pools: list[list[BufferValue]],
        live_recurrent_pools: list[list[BufferValue]],
        shadow_conv_pools: list[list[BufferValue]],
        shadow_recurrent_pools: list[list[BufferValue]],
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        del data_parallel_splits  # DFlash2 is single-replica; see input_types.
        devices = self.config.target.devices
        n_devs = len(devices)
        device0 = devices[0]

        # Pre-step committed length. Only full-attention layers hold KV, so the
        # target leaf's cache_lengths is the logical sequence length.
        pre_cache_lengths = ops.rebind(
            target_kv[0].cache_lengths, ["batch_size"]
        )

        merged_tokens, merged_offsets, _host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )

        # -- Snapshot: the verify runs on the shadow pools, so the live ones
        # still hold the pre-verify state when the accepted length is known.
        batch_scalar = ops.shape_to_tensor([slot_idx[0].shape[0]])[0]
        snapshot_state_pools(
            live_conv_pools, shadow_conv_pools, slot_idx, batch_scalar
        )
        snapshot_state_pools(
            live_recurrent_pools,
            shadow_recurrent_pools,
            slot_idx,
            batch_scalar,
        )
        shadow_slot_idx = [
            ops.range(
                start=0,
                stop=slot_idx[i].shape[0],
                out_dim="batch_size",
                device=devices[i],
                dtype=DType.uint32,
            )
            for i in range(n_devs)
        ]

        # -- Target verify over the merged window, on the shadow pools.
        captures: list[list[GatedDeltaReplayInputs]] = [
            [] for _ in range(n_devs)
        ]
        for layer_idx in self.target.linear_layer_indices:
            layer = self.target.layers[layer_idx]
            assert isinstance(layer, Qwen3_5LinearAttentionBlock)
            layer.replay_capture = captures

        target_outputs = self.target(
            merged_tokens,
            target_kv,
            return_n_logits,
            merged_offsets,
            signal_buffers,
            shadow_slot_idx,
            shadow_conv_pools,
            shadow_recurrent_pools,
        )
        for layer_idx in self.target.linear_layer_indices:
            layer = self.target.layers[layer_idx]
            assert isinstance(layer, Qwen3_5LinearAttentionBlock)
            layer.replay_capture = None

        # VARIABLE logits + SELECTED_LAYERS ->
        # (last_logits, logits, offsets, tap_0..tap_{n-1}), device-major.
        logits = target_outputs[1]
        target_hs_concat = fuse_captured_hidden_states(
            captures_by_device(target_outputs[3:], n_devs)
        )[0]

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=device0,
        )
        # The shared ``accept_and_pick_next_tokens`` picks the committed token
        # straight from the sampler's accepted count. This graph zeroes that
        # count first (prefill and seeded all-magic rows accepted nothing), so
        # the pick has to happen after, or the committed token comes from a
        # position the row never accepted.
        num_accepted, recovered, bonus = self.acceptance_sampler(
            draft_tokens,
            logits,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            in_thinking_phase=in_thinking_phase,
            token_bitmasks=effective_bitmasks,
        )

        num_steps_u32 = _shape_to_scalar(
            draft_tokens.shape[1], device0, dtype=DType.uint32
        )
        zero = ops.constant(0, DType.uint32, device=device0)
        is_prefill = (num_steps_u32 == zero).broadcast_to(["batch_size"])
        magic_token = ops.constant(
            MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device0
        )
        num_magic_tokens = ops.squeeze(
            ops.sum(
                (draft_tokens == magic_token)
                .cast(DType.int32)
                .rebind(["batch_size", "num_steps"]),
                axis=-1,
            ),
            axis=-1,
        )
        num_steps_i32 = _shape_to_scalar(
            draft_tokens.shape[1], device0, dtype=DType.int32
        )
        is_dummy_draft = num_magic_tokens == num_steps_i32.broadcast_to(
            ["batch_size"]
        )
        # A seeded all-magic row proposed nothing, so anything the sampler
        # "accepted" there is noise that would be reported as acceptance and
        # would move the block's anchor position.
        num_accepted = ops.where(
            is_prefill | is_dummy_draft,
            ops.constant(0, num_accepted.dtype, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )

        # -- Roll the live pools forward over exactly the accepted prefix.
        # ``num_draft_tokens`` is the runtime width of this step's draft
        # tensor, not the compiled block: it is 0 on prefill, where the plan
        # then covers the whole prompt with no phase branch. It runs after
        # the settling above so the replay covers the accepted prefix and
        # not a count the row never accepted.
        row_indices, replay_offsets = accepted_row_plan(
            merged_offsets,
            num_accepted,
            _shape_to_scalar(draft_tokens.shape[1], device0),
            merged_tokens.shape[0],
            device0,
        )
        replay_state_pools(
            captures,
            live_conv_pools,
            live_recurrent_pools,
            slot_idx,
            row_indices,
            replay_offsets,
            signal_buffers,
        )

        # The committed token, from a table wide enough for any index the
        # sampler can produce. ``ops.gather_nd`` is not bounds-checked, and a
        # prefill row's table would otherwise be a single column: an index
        # past it reads off the end of the row and returns whatever is there,
        # which then becomes the drafter's block anchor and is broadcast into
        # a gather of the selector's ``[248320, 256]`` predecessor codebook --
        # where an out-of-range id is a device assert, not a wrong answer.
        #
        # Padding with ``block_size`` copies of the bonus makes the table at
        # least ``block_size`` wide whatever the draft width is, and the
        # padding carries the right answer rather than filler: an index at or
        # past the draft width means every draft was accepted, whose committed
        # token is the bonus. On a prefill row that is the only case, and the
        # bonus is the target's own token at the last prompt position.
        next_tokens = ops.gather_nd(
            ops.concat(
                [
                    recovered,
                    ops.broadcast_to(bonus, ["batch_size", self.block_size]),
                ],
                axis=1,
            ),
            ops.unsqueeze(num_accepted.cast(DType.int64), axis=-1),
            batch_dims=1,
        )
        prompt_lens = (input_row_offsets[1:] - input_row_offsets[:-1]).rebind(
            ["batch_size"]
        )
        commit_lengths = ops.where(
            is_prefill,
            prompt_lens,
            (num_accepted + 1).cast(DType.uint32),
        )

        next_draft_tokens = self.draft_next_block(
            target_hs_concat=target_hs_concat,
            merged_offsets=merged_offsets,
            pre_cache_lengths=pre_cache_lengths,
            commit_lengths=commit_lengths,
            anchor_tokens=next_tokens,
            draft_kv=draft_kv[0],
            input_row_offsets=input_row_offsets,
            signal_buffers=signal_buffers,
        )

        return (num_accepted, next_tokens, next_draft_tokens)

    def draft_next_block(
        self,
        *,
        target_hs_concat: TensorValue,
        merged_offsets: TensorValue,
        pre_cache_lengths: TensorValue,
        commit_lengths: TensorValue,
        anchor_tokens: TensorValue,
        draft_kv: PagedCacheValues,
        input_row_offsets: TensorValue,
        signal_buffers: list[BufferValue],
    ) -> TensorValue:
        """Proposes the next block from the verify pass's tapped hidden states.

        Split out of :meth:`__call__` so the half that the reference fixture
        covers -- everything from the tap projection to the selector's walk --
        is one callable a harness can drive with the fixture's own tensors,
        rather than a transcription of it.

        Args:
            target_hs_concat: ``[merged_seq_len, taps * hidden]`` tapped target
                states, ordered by ascending layer index.
            merged_offsets: Ragged offsets over the verify window.
            pre_cache_lengths: ``[batch]`` committed length before this step.
            commit_lengths: ``[batch]`` tokens committed by this step.
            anchor_tokens: ``[batch]`` the committed token, block slot 0.
            draft_kv: The drafter's own KV leaf.
            input_row_offsets: The pre-merge offsets; only its length (batch +
                1) is read, to lay the block out as dense segments.
            signal_buffers: For the target's collective embedding and head.

        Returns:
            ``[batch, block_size - 1]`` proposed tokens.
        """
        device0 = self.config.target.devices[0]
        block = self.block_size

        ctx_hidden = self.draft.project_target_hidden(target_hs_concat)

        # The draft leaf carries its own blocks / lookup table / dispatch
        # metadata; only cache_lengths is overridden, with the pre-step value,
        # so row i of the verify window lands at its own absolute position.
        #
        # Every merged row is written, including a long prefill's rows that
        # already fell out of the drafter's 2048-position window. That is safe
        # and deliberate: a windowed leaf's lookup table is absolute, with the
        # slots below the window pointing at the null page, so those writes are
        # discarded exactly as the reference's input-prep kernel discards them.
        # Trimming the write instead would be actively wrong -- the RoPE-store
        # kernel grids over every row of its input and derives the row's batch
        # from the offsets, so rows past the last offset would still be
        # written, at positions past the end of the sequence.
        draft_kv_collection = replace(draft_kv, cache_lengths=pre_cache_lengths)
        self.draft.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=merged_offsets,
            kv_collection=draft_kv_collection,
        )

        # The block sits at the post-commit position: slot 0 is the committed
        # token and slot i lands at absolute position committed + i.
        block_kv_collection = replace(
            draft_kv_collection,
            cache_lengths=pre_cache_lengths + commit_lengths,
            attention_dispatch_metadata=_block_dispatch_metadata(
                draft_kv_collection.attention_dispatch_metadata, block
            ),
            max_prompt_length=ops.constant(
                block, DType.uint32, device=DeviceRef.CPU()
            ).broadcast_to([1]),
        )

        mask_tail = ops.constant(
            self.mask_token_id, DType.int64, device=device0
        ).broadcast_to(["batch_size", block - 1])
        block_ids = ops.concat(
            [ops.unsqueeze(anchor_tokens, axis=1), mask_tail], axis=1
        )
        # The drafter borrows the target's embedding table; Qwen3.5 applies no
        # embedding scale, which is what the drafter was trained against
        # (``dflash_config`` declares no ``input_embedding_scale``).
        block_embeds = self.target.embed_tokens(
            block_ids.reshape((-1,)), signal_buffers
        )[0]

        # A dense ``block``-row segment per sequence: the drafter's block axis
        # is a reshape of the token axis, not a function of these offsets.
        block_indices = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device0,
            dtype=DType.uint32,
        )
        block_hs = self.draft.forward_block(
            input_embeds=block_embeds,
            kv_collection=block_kv_collection,
            input_row_offsets=block_indices
            * ops.constant(block, DType.uint32, device=device0),
        )

        # Anchor drop: slot 0 holds the committed token and is untrained, so
        # only the mask slots produce drafts.
        block_hs_3d = block_hs.reshape(
            ("batch_size", block, self.config.draft.hidden_size)
        )
        mask_hidden = block_hs_3d[:, 1:, :]

        # The drafter has no head of its own. Bind the target's for the
        # duration of the candidate projection, the way the verify binds the
        # linear blocks' replay capture, so the borrowed head stays out of the
        # drafter's own weight namespace.
        self.draft.lm_head = lambda x: self.target.lm_head([x], signal_buffers)[
            0
        ]
        try:
            candidate_ids, unary_logits = self.draft.compute_candidates(
                mask_hidden
            )
        finally:
            self.draft.lm_head = None

        scores = self.draft.candidate_selector.score_edges(
            candidate_ids, unary_logits, mask_hidden, anchor_tokens
        )
        return self.draft.candidate_selector.select_path(
            scores, candidate_ids
        ).rebind(["batch_size", block - 1])

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Canonical spec-decode signature plus the Qwen state-pool tail.

        Byte-for-byte the Qwen3.5 MTP graph's signature: the tail is
        ``slot_idx``, then the live conv and recurrent pools, then their
        shadows, every block device-major. Only the draft KV leaf's shapes
        differ (five drafter layers of 8 x 128 rather than one target-shaped
        layer), so Mach's Qwen slot layout carries over unchanged.
        """
        config = self.config.target
        devices = config.devices
        spec_types = build_spec_decode_input_types(
            SpecDecodeInputTypeSpec(
                distributed=True,
                data_parallel_degree=1,
                include_in_thinking_phase=True,
                enable_structured_output=self.enable_structured_output,
            ),
            devices=devices,
            kv_params=kv_params,
        )

        num_devices = len(devices)
        conv_dim = self.target._conv_dim // num_devices
        num_v_heads = self.target._num_v_heads // num_devices
        conv_span = self.target._conv_kernel_size - 1
        recurrent_shape = [
            num_v_heads,
            self.target._key_head_dim,
            self.target._value_head_dim,
        ]
        state_dtype = config.state_dtype

        tail: list[TensorType | BufferType] = [
            TensorType(DType.uint32, shape=["batch_size"], device=device)
            for device in devices
        ]
        for slots in ("max_slots", "max_shadow_slots"):
            tail.extend(
                BufferType(
                    state_dtype,
                    shape=[slots, conv_dim, conv_span],
                    device=device,
                )
                for device in devices
                for _ in range(self.num_linear_layers)
            )
            tail.extend(
                BufferType(
                    state_dtype,
                    shape=[slots, *recurrent_shape],
                    device=device,
                )
                for device in devices
                for _ in range(self.num_linear_layers)
            )

        return (*spec_types, *tail)
