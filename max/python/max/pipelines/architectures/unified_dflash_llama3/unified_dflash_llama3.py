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
"""Unified DFlash Llama3 nn.Module: target + KV materialize + draft block."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

from max.dtype import DType
from max.graph import BufferType, TensorType, TensorValue, Value, ops
from max.nn.kv_cache import MultiKVCacheParams, PagedCacheValues
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

from ..dflash_llama3 import DFlashLlama3
from ..llama3.llama3 import Llama3
from .model_config import UnifiedDflashLlama3Config


@dataclass
class UnifiedDflashLlama3Values:
    tokens: TensorValue
    input_row_offsets: TensorValue
    draft_tokens: TensorValue
    return_n_logits: TensorValue
    kv_collection: PagedCacheValues
    draft_kv_collection: PagedCacheValues
    seed: TensorValue
    temperature: TensorValue
    top_k: TensorValue
    max_k: TensorValue
    top_p: TensorValue
    min_top_p: TensorValue


class UnifiedDflashLlama3(Module):
    """Fused module: merge → target → reject → materialize → draft block."""

    def __init__(self, config: UnifiedDflashLlama3Config) -> None:
        super().__init__()
        self.config = config
        self.block_size = config.resolve_block_size()
        self.num_speculative_tokens = self.block_size - 1
        self.target_layer_ids = list(config.target_layer_ids)
        self.mask_token_id = int(config.mask_token_id)
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                config.speculative_config.synthetic_acceptance_rate
            ),
            num_draft_steps=self.num_speculative_tokens,
            use_stochastic=True,
        )

        self.target = Llama3(config.target)
        self.draft = DFlashLlama3(
            config.draft,
            num_context_features=len(self.target_layer_ids),
        )
        self.merger = RaggedTokenMerger(config.target.devices[0])

    def _unflatten_graph_inputs(
        self,
        inputs: Sequence[Value[Any]],
    ) -> UnifiedDflashLlama3Values:
        it = iter(inputs)
        tokens = next(it)
        input_row_offsets = next(it)
        return_n_logits = next(it)
        kv_params = MultiKVCacheParams.from_params(
            {
                "target": self.config.target.kv_params,
                "draft": self.config.draft.kv_params,
            }
        )
        target_kv_collections, draft_kv_collections = (
            kv_params.unflatten_basic_kv_tree(it)
        )
        target_kv_collection = target_kv_collections[0]
        draft_kv_collection = draft_kv_collections[0]
        draft_tokens = next(it)
        seed = next(it)
        temperature = next(it)
        top_k = next(it)
        max_k = next(it)
        top_p = next(it)
        min_top_p = next(it)

        return UnifiedDflashLlama3Values(
            tokens=tokens.tensor,
            input_row_offsets=input_row_offsets.tensor,
            draft_tokens=draft_tokens.tensor,
            return_n_logits=return_n_logits.tensor,
            kv_collection=target_kv_collection,
            draft_kv_collection=draft_kv_collection,
            seed=seed.tensor,
            temperature=temperature.tensor,
            top_k=top_k.tensor,
            max_k=max_k.tensor,
            top_p=top_p.tensor,
            min_top_p=min_top_p.tensor,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Single-device dflash graph. See
        :func:`build_spec_decode_input_types` for the canonical ordering.
        """
        return build_spec_decode_input_types(
            SpecDecodeInputTypeSpec(distributed=False),
            devices=self.config.target.devices,
            kv_params=MultiKVCacheParams.from_params(
                {
                    "target": self.config.target.kv_params,
                    "draft": self.config.draft.kv_params,
                }
            ),
        )

    def __call__(
        self,
        inputs: UnifiedDflashLlama3Values,
    ) -> tuple[TensorValue, ...]:
        device = inputs.tokens.device
        K = self.block_size
        pre_cache_lengths = ops.rebind(
            inputs.kv_collection.cache_lengths, ["batch_size"]
        )

        merged_tokens, merged_offsets = self.merger(
            inputs.tokens,
            inputs.input_row_offsets,
            inputs.draft_tokens,
        )
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["input_row_offsets_len"])

        target_outputs = self.target(
            merged_tokens,
            inputs.kv_collection,
            inputs.return_n_logits,
            merged_offsets,
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
        # the target leaf); only cache_lengths is overridden with the pre-step
        # value.
        draft_kv_collection = replace(
            inputs.draft_kv_collection,
            cache_lengths=pre_cache_lengths,
        )

        self.draft.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=merged_offsets,
            kv_collection=draft_kv_collection,
        )

        # DFlash runs the draft as a full block (the accepted token plus
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

        block_embeds = self.target.embed_tokens(block_ids_flat)
        if self.target.embedding_multiplier != 1.0:
            block_embeds = block_embeds * ops.constant(
                self.target.embedding_multiplier,
                block_embeds.dtype,
                device=device,
            )

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

        block_hs_2d = block_hs.reshape(
            ("batch_size", K, self.config.draft.hidden_size)
        )
        draft_logits = self.target.lm_head(block_hs_2d[:, 1:, :])
        next_draft_tokens = ops.argmax(draft_logits, axis=-1).reshape(
            ("batch_size", K - 1)
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
