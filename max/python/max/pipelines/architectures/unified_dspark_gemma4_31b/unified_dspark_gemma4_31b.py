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
"""Unified speculators-DSpark Gemma4 nn.Module: target + KV materialize +
draft block.

Structurally the unified dense-DSpark graph (merge -> target -> reject ->
materialize -> block forward), with the speculators-format differences (the
vLLM ``qwen3_dspark`` runtime semantics):

- ``sample_from_anchor: false`` — the anchor slot never predicts, so the
  model drafts ``block_size - 1`` tokens per step and the anchor slot's
  hidden state is dropped before the head (the DFlash position-0 drop).
- The base draft logits come from the draft's OWN pruned-vocab lm_head with
  no softcapping, not the target's full-vocab head.
- The markov chain runs over the pruned draft vocab and applies the ``d2t``
  offset map in-chain, emitting TARGET-vocab draft ids.
- Block slots embed RAW ``embed_tokens`` rows — never the target's
  sqrt(hidden)-scaled embedding path.
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
    DeviceRef,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.embedding import Embedding
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
from max.pipelines.speculative.unified_graph_ops import apply_overlap_bitmask

from ..dspark_draft.dspark_speculators_draft import DSparkSpeculatorsDraft
from ..gemma4.gemma4 import Gemma4TextModel
from .model_config import UnifiedDSparkGemma4_31BConfig


def _block_dispatch_metadata(meta: TensorValue | None, k: int) -> TensorValue:
    """Rebuilds the MHA dispatch metadata at the draft block's query width.

    The 4-int CPU buffer is ``[batch_size, q_max_seq_len, num_partitions,
    max_cache_valid_length]``. ``q_max_seq_len`` becomes the block width
    ``k`` and ``num_partitions`` is zeroed so the decode kernel recomputes
    the split-K count for the draft's own head geometry instead of reusing
    the target's.

    Args:
        meta: The leaf's verify-width dispatch metadata buffer.
        k: The draft block width (anchor slot plus drafted tokens).

    Returns:
        The rebuilt dispatch metadata buffer.
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


@dataclass
class UnifiedDSparkGemma4_31BValues:
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
    in_thinking_phase: TensorValue
    pinned_bitmask: TensorValue | None = None
    wait_payload: BufferValue | None = None
    device_bitmask_scratch: BufferValue | None = None


class UnifiedDSparkGemma4_31B(Module):
    """Fused module: merge → target → reject → materialize → draft block."""

    def __init__(
        self,
        config: UnifiedDSparkGemma4_31BConfig,
        enable_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_structured_output = enable_structured_output
        # The effective block width (anchor slot + resolved drafts): the
        # draft is causal, so a block narrower than the trained one is
        # prefix-stable and stays trained-equivalent; a wider one runs the
        # extra slots as extrapolation.
        self.block_size = config.effective_block_size
        # sample_from_anchor=false: the anchor slot is the committed/bonus
        # token and never predicts, so drafts = block_size - 1 (unlike the
        # dense DSpark drafter, which drafts at every block position).
        self.num_speculative_tokens = self.block_size - 1
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if config.speculative_config.use_relaxed_acceptance_for_thinking:
            relaxed_topk = config.speculative_config.relaxed_topk
            relaxed_delta = config.speculative_config.relaxed_delta
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                config.speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )

        self.target = Gemma4TextModel(config.target)
        self.draft = DSparkSpeculatorsDraft(
            hidden_size=config.draft.hidden_size,
            num_hidden_layers=config.draft.num_hidden_layers,
            num_attention_heads=config.draft.num_attention_heads,
            num_key_value_heads=config.draft.num_key_value_heads,
            head_dim=config.draft.head_dim,
            intermediate_size=config.draft.intermediate_size,
            rms_norm_eps=config.draft.rms_norm_eps,
            rope_theta=config.draft.rope_theta,
            sliding_window=config.draft.sliding_window,
            layer_causal=[config.draft.causal] * config.draft.num_hidden_layers,
            vocab_size=config.draft.vocab_size,
            draft_vocab_size=config.draft.draft_vocab_size,
            markov_rank=config.draft.markov_rank,
            block_size=config.draft.block_size,
            sample_from_anchor=config.draft.sample_from_anchor,
            mask_token_id=config.draft.mask_token_id,
            num_context_features=config.draft.num_context_features,
            max_seq_len=config.draft.max_seq_len,
            kv_params=config.draft_kv_params,
            devices=list(config.target.devices),
            dtype=config.target.unquantized_dtype,
        )
        # The block stream embeds RAW rows (runtime-semantics contract: no
        # gemma sqrt(hidden) scale on the speculators draft path), so the
        # draft owns a plain Embedding loaded verbatim from the checkpoint's
        # frozen full-vocab copy rather than routing through the target's
        # ScaledWordEmbedding.
        self.draft.embed_tokens = Embedding(
            vocab_size=config.draft.vocab_size,
            hidden_dim=config.draft.hidden_size,
            dtype=config.target.unquantized_dtype,
            device=config.target.devices[0],
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
    ) -> UnifiedDSparkGemma4_31BValues:
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
        in_thinking_phase = next(it)
        pinned_bitmask: TensorValue | None = None
        wait_payload: BufferValue | None = None
        device_bitmask_scratch: BufferValue | None = None
        if self.enable_structured_output:
            pinned_bitmask = next(it).tensor
            wait_payload = next(it).buffer
            device_bitmask_scratch = next(it).buffer
        # Every declared input must be consumed: a declared-but-unconsumed
        # trailing input keeps the arity valid at execute while the feature
        # it carries is silently dead.
        sentinel = object()
        assert next(it, sentinel) is sentinel, (
            "input_types() and _unflatten_graph_inputs disagree: unconsumed"
            " graph inputs remain"
        )

        return UnifiedDSparkGemma4_31BValues(
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
            in_thinking_phase=in_thinking_phase.tensor,
            pinned_bitmask=pinned_bitmask,
            wait_payload=wait_payload,
            device_bitmask_scratch=device_bitmask_scratch,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Single-device DSpark graph. See
        :func:`build_spec_decode_input_types` for the canonical ordering.
        Signal buffers are declared even though the graph is single-device:
        Gemma4's embedding/lm_head layers use collectives unconditionally.
        """
        return build_spec_decode_input_types(
            SpecDecodeInputTypeSpec(
                distributed=False,
                include_signal_buffers=True,
                include_in_thinking_phase=True,
                enable_structured_output=self.enable_structured_output,
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
        inputs: UnifiedDSparkGemma4_31BValues,
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
        # Grammar constraining is target/verify-side only: every K+1 verify
        # row is masked before acceptance, so a grammar-violating draft is
        # deterministically rejected and its replacement sampled from the
        # masked residual. The draft chain itself stays unconstrained.
        effective_bitmasks = apply_overlap_bitmask(
            inputs.pinned_bitmask,
            inputs.wait_payload,
            inputs.device_bitmask_scratch,
            num_steps=inputs.draft_tokens.shape[1],
            device=device,
        )
        num_accepted, recovered, bonus = self.acceptance_sampler(
            inputs.draft_tokens,
            target_logits,
            seed=seed_scalar,
            temperature=inputs.temperature,
            top_k=inputs.top_k,
            max_k=inputs.max_k,
            top_p=inputs.top_p,
            min_top_p=inputs.min_top_p,
            in_thinking_phase=inputs.in_thinking_phase,
            token_bitmasks=effective_bitmasks,
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
        # mask-token tail): K query rows per sequence on every batch type.
        # Neither manager-provided metadata fits that geometry: the leaf's
        # attention_dispatch_metadata is the TARGET/verify key, whose
        # q_max_seq_len is the batch's max prompt length — equal to the block
        # width only on decode batches, and e.g. 129 on a prefill batch,
        # where the oversized query bound drives the block's layer-0 flash
        # attention to produce NaN. The manager's
        # draft_attention_dispatch_metadata is resolved one row narrow
        # (num_draft_tokens_per_step = K - 1) with the target's partition
        # count. Rebuild the dispatch buffer at the block's true width.
        bumped_cache_lengths = pre_cache_lengths + commit_lengths
        block_kv_collection = replace(
            draft_kv_collection,
            cache_lengths=bumped_cache_lengths,
            attention_dispatch_metadata=_block_dispatch_metadata(
                draft_kv_collection.attention_dispatch_metadata, K
            ),
            max_prompt_length=ops.constant(
                K, DType.uint32, device=DeviceRef.CPU()
            ).broadcast_to([1]),
        )

        next_tokens_2d = ops.unsqueeze(next_tokens, axis=1)
        mask_const = ops.constant(
            self.mask_token_id, DType.int64, device=device
        )
        mask_tail = mask_const.broadcast_to(["batch_size", K - 1])
        block_ids = ops.concat([next_tokens_2d, mask_tail], axis=1)
        block_ids_flat = block_ids.reshape((-1,))

        # RAW embedding rows: slot 0 = the anchor/bonus token, slots
        # 1..K-1 = the mask token, at positions p..p+K-1 where p is the
        # post-commit sequence length (bumped_cache_lengths above).
        assert self.draft.embed_tokens is not None
        block_embeds = self.draft.embed_tokens(block_ids_flat)

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
            input_embeds=block_embeds,
            kv_collection=block_kv_collection,
            input_row_offsets=draft_block_offsets,
        )

        # Anchor-slot drop (sample_from_anchor=false): slot 0's output is
        # untrained; only the K-1 mask slots produce drafts.
        block_hs_2d = block_hs.reshape(
            ("batch_size", K, self.config.draft.hidden_size)
        )
        base_logits = self.draft.lm_head(block_hs_2d[:, 1:, :])

        # Sequential markov correction seeded by the anchor token, greedy,
        # with the in-chain d2t map; emits TARGET-vocab draft ids.
        next_draft_tokens = self.draft.sample_draft_tokens(
            base_logits, next_tokens
        )

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
