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
"""Unified EAGLE nn.Module: merge + target forward + rejection + shift + draft."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Any

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
from max.nn import ReturnHiddenStates, ReturnLogits
from max.nn.kv_cache import MultiKVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)
from max.pipelines.speculative.unified_graph_ops import (
    accept_and_pick_next_tokens,
    apply_overlap_bitmask,
    shift_corrected_tokens,
)

from ..eagle_llama3.eagle_llama3 import EagleLlama3
from ..llama3.llama3 import Llama3
from .model_config import UnifiedEagleLlama3Config


@dataclass
class UnifiedEagleLlama3Values:
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
    pinned_bitmask: TensorValue | None = None
    """Pinned-host bitmask for constrained decoding.

    Shape: ``[batch_size, num_speculative_tokens + 1, vocab_size]``.
    Position i contains the valid-token mask given the FSM state after
    consuming draft[0:i-1]; position ``num_speculative_tokens`` is for
    the bonus token. Read by an in-graph H2D into
    :attr:`device_bitmask_scratch` after the ``mo.wait_host_value_with_dep``
    op observes the host callback's release-store of the completion
    flag. ``None`` when structured output is disabled (graph compiled
    without the bitmask triple).
    """

    wait_payload: BufferValue | None = None
    """CPU ``int64[2]`` payload consumed by
    ``mo.wait_host_value_with_dep`` (``[CompletionFlag._unsafe_ptr,
    1]``). Owned by :class:`StructuredOutputOverlapState`."""

    device_bitmask_scratch: BufferValue | None = None
    """Device scratch buffer that receives the in-graph H2D from
    :attr:`pinned_bitmask`; the acceptance sampler reads from it."""


class UnifiedEagleLlama3(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift + draft."""

    def __init__(self, config: UnifiedEagleLlama3Config) -> None:
        super().__init__()

        self.config = config
        num_draft_steps = config.speculative_config.num_speculative_tokens
        # Unset resolves to the eagle default at SpeculativeConfig
        # construction.
        assert num_draft_steps is not None
        self.num_draft_steps = num_draft_steps
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=config.speculative_config.synthetic_acceptance_rate,
            num_draft_steps=self.num_draft_steps,
            use_stochastic=True,
        )
        self.num_devices = 1

        # TODO: support distributed llama3 model
        if len(config.target.devices) != 1:
            raise ValueError("UnifiedEagleLlama3 only supports a single device")

        self.target = Llama3(config.target)
        self.draft = EagleLlama3(config.draft)
        self.merger = RaggedTokenMerger(config.target.devices[0])

    def _unflatten_graph_inputs(
        self,
        inputs: Sequence[Value[Any]],
    ) -> UnifiedEagleLlama3Values:
        # Use an iterator to consume inputs sequentially, avoiding a hardcoded
        # count that would break if the kv cache structure changes.
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
        # draft model inputs
        draft_tokens = next(it)
        # stochastic acceptance seed (uint64 [batch_size] on the primary device)
        seed = next(it)
        # sampling params for stochastic acceptance
        temperature = next(it)
        top_k = next(it)
        max_k = next(it)
        top_p = next(it)
        min_top_p = next(it)
        # Optional constrained-decoding bitmask triple (appended when
        # structured output is enabled). The triple is bound by the
        # OverlapTextGenerationPipeline from
        # :class:`StructuredOutputOverlapState`.
        pinned_bitmask_in: TensorValue | None = None
        wait_payload_in: BufferValue | None = None
        device_bitmask_scratch_in: BufferValue | None = None
        if self.config.enable_structured_output:
            pinned_bitmask_in = next(it).tensor
            wait_payload_in = next(it).buffer
            device_bitmask_scratch_in = next(it).buffer

        return UnifiedEagleLlama3Values(
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
            pinned_bitmask=pinned_bitmask_in,
            wait_payload=wait_payload_in,
            device_bitmask_scratch=device_bitmask_scratch_in,
        )

    def input_types(self) -> tuple[TensorType | BufferType, ...]:
        """Input types for the unified graph.

        Single-device eagle graph that appends the structured-output bitmask
        triple when ``config.enable_structured_output`` is set. See
        :func:`build_spec_decode_input_types` for the canonical ordering.
        """
        return build_spec_decode_input_types(
            SpecDecodeInputTypeSpec(
                distributed=False,
                enable_structured_output=self.config.enable_structured_output,
            ),
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
        inputs: UnifiedEagleLlama3Values,
    ) -> tuple[TensorValue, ...]:
        # Notation:
        #   B   = batch_size
        #   S   = total_seq_len   (ragged sum of prompt lengths)
        #   K   = num_draft_tokens_to_verify (K is either 0 or num_speculative_tokens)
        #   V   = vocab_size
        #   H   = hidden_size
        #
        # inputs.tokens           : [S]
        # inputs.input_row_offsets: [B+1]
        # inputs.draft_tokens     : [B, K]
        # inputs.return_n_logits  : [1] (CPU)
        tokens = inputs.tokens
        input_row_offsets = inputs.input_row_offsets
        draft_tokens = inputs.draft_tokens
        return_n_logits = inputs.return_n_logits
        kv_collection = inputs.kv_collection
        draft_kv_collection = inputs.draft_kv_collection

        device = tokens.device

        # merged_tokens : [S+B*K]
        # merged_offsets: [B+1]
        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        # Rebind to clean symbolic dims so downstream reshapes in
        # Llama3's attention layers can simplify element counts.
        merged_tokens = merged_tokens.rebind(["merged_seq_len"])
        merged_offsets = merged_offsets.rebind(["input_row_offsets_len"])

        # --- Target step ---
        target_outputs = self.target(
            merged_tokens,
            kv_collection,
            return_n_logits,
            merged_offsets,
        )
        # logits       : [B*(K+1), V] (K+1 logits per request)
        # hidden_states: [S+B*K, H]. ``extract_hs`` flattens per-device
        # hs into positional tuple elements; single-device here so the
        # hs is at index 3.
        logits = target_outputs[1]
        hidden_states = target_outputs[3]

        hidden_dim = hidden_states.shape[1]

        effective_bitmasks = apply_overlap_bitmask(
            inputs.pinned_bitmask,
            inputs.wait_payload,
            inputs.device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=device,
        )

        # num_accepted_draft_tokens: [B]     (index of first rejected step, 0..K)
        # recovered                : [B, K]  (target argmax at each draft position)
        # bonus                    : [B, 1]  (target argmax at the +1 position)
        seed_scalar = inputs.seed[0]
        num_accepted_draft_tokens, recovered, bonus, next_tokens = (
            accept_and_pick_next_tokens(
                self.acceptance_sampler,
                draft_tokens,
                logits,
                seed=seed_scalar,
                temperature=inputs.temperature,
                top_k=inputs.top_k,
                max_k=inputs.max_k,
                top_p=inputs.top_p,
                min_top_p=inputs.min_top_p,
                token_bitmasks=effective_bitmasks,
            )
        )

        num_draft_sentinel_gpu = _shape_to_scalar(draft_tokens.shape[1], device)

        shifted_corrected = shift_corrected_tokens(
            self.merger, tokens, input_row_offsets, recovered, bonus
        )

        # --- Draft step 0 ---
        # Hack the return_hidden_states, return_logits and reset it to match
        # the target model.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,  # [S+B*K]
            draft_kv_collection,
            return_n_logits,
            merged_offsets,  # [B+1]
            hidden_states,  # [S+B*K, H]
        )
        self.draft.return_hidden_states = ReturnHiddenStates.LAST
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        # logits       : [B*(K+1), V] (K+1 logits per request)
        # hidden_states: [S+B*K, H] (single-device, single TensorValue).
        logits = draft_outputs[1]
        hs = draft_outputs[3]

        last_idx = merged_offsets[1:] - 1
        last_accepted_idx = (
            ops.rebind(last_idx, ["batch_size"])
            - num_draft_sentinel_gpu.broadcast_to(["batch_size"])
            + num_accepted_draft_tokens
        )
        draft_hs = ops.gather(hs, last_accepted_idx, axis=0)

        # Sample the first draft token
        logits = _reshape_target_logits(logits)
        # tokens: [B, K+1]
        tokens = ops.squeeze(ops.argmax(logits, axis=-1), axis=-1)

        next_draft_tokens = ops.gather_nd(
            tokens,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        )

        # Compute the new kv cache collection
        prev_cache_lengths = ops.rebind(
            draft_kv_collection.cache_lengths, ["batch_size"]
        )
        input_lengths = input_row_offsets[1:] - input_row_offsets[:-1]
        cache_lengths = (
            prev_cache_lengths
            + ops.rebind(input_lengths, ["batch_size"])
            + num_accepted_draft_tokens.cast(DType.uint32)
        )

        # Prepare the new input_row_offsets (all reqs have 1 token)
        input_row_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device,
            dtype=DType.uint32,
        )

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])

        # Set up the max cache length for the next step.
        # Assume that all tokens are accepted in this calculation.
        # Confusingly max_cache_length != max(cache_lengths). Instead max_cache_length
        # is more like max_total_seq_len including cached and input tokens.
        orig_max_cache_length = draft_kv_collection.max_cache_length
        max_cache_length = orig_max_cache_length + 1

        # draft_return_n_logits: [1] (CPU)
        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        draft_kv_collection = replace(
            draft_kv_collection,
            max_prompt_length=one,
            max_cache_length=orig_max_cache_length,
            attention_dispatch_metadata=draft_kv_collection.draft_attention_dispatch_metadata,
        )

        # --- Draft steps 1..N-1 ---
        all_draft_tokens = [next_draft_tokens]
        for _ in range(1, self.num_draft_steps):
            next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
            draft_hs = draft_hs.rebind(["batch_size", hidden_dim])

            draft_kv_collection = replace(
                draft_kv_collection, cache_lengths=cache_lengths
            )

            draft_outputs = self.draft(
                next_draft_tokens,
                draft_kv_collection,
                draft_return_n_logits,
                input_row_offsets,
                draft_hs,
            )
            logits = draft_outputs[0]
            draft_hs = draft_outputs[1]

            next_draft_tokens = ops.argmax(logits, axis=-1).reshape([-1])

            # Store the new tokens for this step
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

            # Increment cache length for the next step
            cache_lengths = cache_lengths + 1
            max_cache_length = max_cache_length + 1

        # next_draft_tokens: [B, num_draft_steps]
        if len(all_draft_tokens) > 1:
            next_draft_tokens = ops.stack(all_draft_tokens, axis=-1)
        else:
            next_draft_tokens = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            num_accepted_draft_tokens,  # [B]
            next_tokens,  # [B]
            next_draft_tokens,  # [B, num_draft_steps]
        )
