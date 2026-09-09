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
"""Unified DSpark Gemma4 nn.Module: target + KV materialize + draft block.

Structurally the unified DFlash graph (merge -> target -> reject ->
materialize -> block forward), with the DSpark differences: ALL
``block_size`` positions produce drafts (DFlash drops position 0), the base
draft logits are soft-capped before correction, and the drafts are sampled
through the sequential markov-head chain seeded by the anchor token.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import (
    KVCacheInputs,
    MultiKVCacheInputs,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import AcceptanceSampler
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

from ..gemma4.gemma4 import Gemma4TextModel
from .dspark_gemma4 import DSparkGemma4
from .model_config import UnifiedDSparkGemma4_12BConfig


@dataclass
class UnifiedDSparkGemma4_12BValues:
    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    return_n_logits: TensorValue
    signal_buffers: list[BufferValue]
    sliding_kv_collection: PagedCacheValues
    global_kv_collection: PagedCacheValues
    draft_kv_collection: PagedCacheValues
    seed: TensorValue
    temperature: TensorValue
    top_k: TensorValue
    max_k: TensorValue
    top_p: TensorValue
    min_top_p: TensorValue


class UnifiedDSparkGemma4_12B(Module):
    """Fused module: merge → target → reject → materialize → draft block."""

    def __init__(self, config: UnifiedDSparkGemma4_12BConfig) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.resolve_block_size()
        # DSpark drafts at ALL block positions: the anchor position predicts
        # draft 1 (unlike DFlash, which drops position 0).
        self.num_speculative_tokens = self.block_size
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                config.speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
        )

        self.target = Gemma4TextModel(config.target)
        self.draft = DSparkGemma4(
            config.draft,
            kv_params=config.draft_kv_params,
            devices=list(config.target.devices),
            dtype=config.target.unquantized_dtype,
        )
        self.merger = RaggedTokenMerger(config.target.devices[0])

    def _unified_kv_params(self) -> MultiKVCacheParams:
        return MultiKVCacheParams.from_params(
            {
                "target": self.config.target.kv_params,
                "draft": self.config.draft_kv_params,
            }
        )

    def _unflatten_graph_inputs(
        self,
        inputs: Sequence[Value[Any]],
    ) -> UnifiedDSparkGemma4_12BValues:
        it = iter(inputs)
        tokens = next(it)
        input_row_offsets = next(it)
        return_n_logits = next(it)
        signal_buffers = [
            next(it).buffer for _ in range(len(self.config.target.devices))
        ]
        kv_tree = self._unified_kv_params().unflatten_kv_inputs(it)
        assert isinstance(kv_tree, MultiKVCacheInputs)
        target_tree = kv_tree.children["target"]
        assert isinstance(target_tree, MultiKVCacheInputs)
        sliding_leaf = target_tree.children["sliding_attention"]
        global_leaf = target_tree.children["full_attention"]
        draft_leaf = kv_tree.children["draft"]
        assert isinstance(sliding_leaf, KVCacheInputs)
        assert isinstance(global_leaf, KVCacheInputs)
        assert isinstance(draft_leaf, KVCacheInputs)
        draft_tokens = next(it)
        seed = next(it)
        temperature = next(it)
        top_k = next(it)
        max_k = next(it)
        top_p = next(it)
        min_top_p = next(it)

        return UnifiedDSparkGemma4_12BValues(
            tokens=tokens.tensor,
            input_row_offsets=input_row_offsets.tensor,
            draft_tokens=draft_tokens.tensor,
            return_n_logits=return_n_logits.tensor,
            signal_buffers=signal_buffers,
            sliding_kv_collection=sliding_leaf.inputs[0],
            global_kv_collection=global_leaf.inputs[0],
            draft_kv_collection=draft_leaf.inputs[0],
            seed=seed.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Single-device DSpark graph. See
        :func:`build_spec_decode_input_types` for the canonical ordering.
        Signal buffers are declared even though the graph is single-device:
        Gemma4's embedding/lm_head layers use collectives unconditionally.
        """
        return build_spec_decode_input_types(
            SpecDecodeInputTypeSpec(
                distributed=False, include_signal_buffers=True
            ),
            devices=self.config.target.devices,
            kv_params=self._unified_kv_params(),
        )

    def _empty_vision_inputs(self) -> tuple[TensorValue, TensorValue]:
        """Zero-row image embeddings + scatter indices for the text-only
        target forward (the vision merge scatter is a no-op)."""
        device = self.config.target.devices[0]
        hidden = self.config.target.text_config.hidden_size
        empty_embeds = ops.constant(
            np.zeros((0, hidden), dtype=np.float32),
            DType.float32,
            device=device,
        ).cast(self.config.target.unquantized_dtype)
        empty_indices = ops.range(
            0, 0, 1, out_dim=0, dtype=DType.int32, device=device
        )
        return empty_embeds, empty_indices

    def __call__(
        self,
        inputs: UnifiedDSparkGemma4_12BValues,
    ) -> tuple[TensorValue, ...]:
        device = inputs.tokens.device
        K = self.block_size
        signal_buffers = inputs.signal_buffers
        # Pre-step committed length; the global (full-attention) leaf tracks
        # the logical sequence length.
        pre_cache_lengths = ops.rebind(
            inputs.global_kv_collection.cache_lengths, ["batch_size"]
        )

        merged_tokens, merged_offsets = self.merger(
            inputs.tokens,
            inputs.input_row_offsets,
            inputs.draft_tokens,
        )
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["input_row_offsets_len"])

        empty_embeds, empty_indices = self._empty_vision_inputs()
        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            [inputs.sliding_kv_collection],
            [inputs.global_kv_collection],
            inputs.return_n_logits,
            [merged_offsets],
            [empty_embeds],
            [empty_indices],
        )
        target_logits = target_outputs[1]
        target_hs_concat = fuse_captured_hidden_states(
            captures_by_device(target_outputs[3:], 1)
        )[0]

        seed_scalar = inputs.seed[0]
        num_accepted, recovered, bonus = self.acceptance_sampler(
            inputs.draft_tokens,
            target_logits,
            seed=seed_scalar,
            temperature=inputs.temperature,
            top_k=inputs.top_k,
            max_k=inputs.max_k,
            top_p=inputs.top_p,
            min_top_p=inputs.min_top_p,
        )

        num_steps_u32 = _shape_to_scalar(
            inputs.draft_tokens.shape[1], device, dtype=DType.uint32
        )
        zero = ops.constant(0, DType.uint32, device=device)
        is_prefill = (num_steps_u32 == zero).broadcast_to(["batch_size"])
        magic_token = ops.constant(
            MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device
        )
        num_magic_tokens = ops.squeeze(
            ops.sum(
                (inputs.draft_tokens == magic_token)
                .cast(DType.int32)
                .rebind(["batch_size", "num_steps"]),
                axis=-1,
            ),
            axis=-1,
        )
        num_steps = _shape_to_scalar(
            inputs.draft_tokens.shape[1], device, dtype=DType.int32
        )
        is_dummy_draft = num_magic_tokens == num_steps.broadcast_to(
            ["batch_size"]
        )
        num_accepted = ops.where(
            is_prefill | is_dummy_draft,
            ops.constant(0, num_accepted.dtype, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )
        prompt_lens = (
            inputs.input_row_offsets[1:] - inputs.input_row_offsets[:-1]
        ).rebind(["batch_size"])
        decode_commit = (num_accepted + 1).cast(DType.uint32)
        commit_lengths = ops.where(is_prefill, prompt_lens, decode_commit)

        target_tokens = ops.concat([recovered, bonus], axis=1)
        gather_idx = ops.where(
            is_prefill,
            ops.constant(0, DType.int64, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted.cast(DType.int64),
        )
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(gather_idx, axis=-1),
            batch_dims=1,
        )

        ctx_hidden = self.draft.project_target_hidden(target_hs_concat)

        # The draft leaf already carries draft blocks + lookup_table /
        # max_prompt_length / max_cache_length / dispatch metadata (mirroring
        # the target leaves); only cache_lengths is overridden with the
        # pre-step value.
        draft_kv_collection = replace(
            inputs.draft_kv_collection,
            cache_lengths=pre_cache_lengths,
        )

        self.draft.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=merged_offsets,
            kv_collection=draft_kv_collection,
        )

        # DSpark runs the draft as a full block (the accepted token plus the
        # mask-token tail), not as EAGLE-style single-token draft steps. The
        # KV manager's draft metadata is sized with ``q_max_seq_len=1`` for
        # EAGLE, so keep the target metadata, which is sized for the merged
        # verify block and is safe for this block forward.
        bumped_cache_lengths = pre_cache_lengths + commit_lengths
        block_kv_collection = replace(
            draft_kv_collection,
            cache_lengths=bumped_cache_lengths,
        )

        next_tokens_2d = ops.unsqueeze(next_tokens, axis=1)
        mask_const = ops.constant(
            self.mask_token_id, DType.int64, device=device
        )
        mask_tail = mask_const.broadcast_to(["batch_size", K - 1])
        block_ids = ops.concat([next_tokens_2d, mask_tail], axis=1)
        block_ids_flat = block_ids.reshape((-1,))

        # ScaledWordEmbedding applies the sqrt(hidden) embedding scale.
        block_embeds = self.target.embed_tokens(block_ids_flat, signal_buffers)

        block_indices = ops.range(
            start=0,
            stop=inputs.input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device,
            dtype=DType.uint32,
        )
        draft_block_offsets = block_indices * ops.constant(
            K, DType.uint32, device=device
        )

        block_hs = self.draft.forward_block(
            input_embeds=block_embeds[0],
            kv_collection=block_kv_collection,
            input_row_offsets=draft_block_offsets,
        )

        # ALL K positions produce drafts (no position-0 drop).
        block_hs_2d = block_hs.reshape(
            ("batch_size", K, self.config.draft.hidden_size)
        )
        base_logits = self.target.lm_head([block_hs_2d], signal_buffers)[0]
        softcap = self.config.draft.final_logit_softcapping
        if softcap is not None:
            base_logits = ops.tanh(base_logits / softcap) * softcap

        # Sequential markov correction seeded by the anchor token, greedy.
        assert self.draft.markov_head is not None
        next_draft_tokens = self.draft.markov_head(base_logits, next_tokens)

        # Force num_accepted=0 in the prefill output even if the sampler
        # happened to "accept" garbage drafts, so downstream metrics don't
        # report bogus acceptances.
        num_accepted_out = ops.where(
            is_prefill,
            ops.constant(0, num_accepted.dtype, device=device).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )

        return (num_accepted_out, next_tokens, next_draft_tokens)
