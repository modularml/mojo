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
"""Gemma4 with MTP nn.Module: merge + target forward + rejection + shift."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

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
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.ragged_token_merger import RaggedTokenMerger
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)
from max.pipelines.speculative.unified_graph_ops import (
    accept_and_pick_next_tokens,
    apply_overlap_bitmask,
    gather_accepted_hidden_states,
    merge_tokens_and_host_offsets,
    shift_corrected_tokens,
)

from ..gemma4.gemma4 import Gemma4TextModel
from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4_assistant.model_config import Gemma4AssistantConfig


class UnifiedMTPGemma4(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift.

    Composes RaggedTokenMerger, Gemma4TextModel (target), Gemma4Assistant
    (draft), AcceptanceSampler, and eagle_prefill_shift_tokens into a single
    graph-buildable module.
    """

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        draft_config: Gemma4AssistantConfig,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        use_greedy_acceptance: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_structured_output = enable_structured_output
        self.num_draft_steps = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            and speculative_config.num_speculative_tokens is not None
            else 1
        )
        # Greedy acceptance (argmax) has no mid-graph allocation, so the fused
        # spec graph is CUDA-graph-capturable; the stochastic path is not.
        # Greedy serving only (temp 0, top_k 1); relaxed/synthetic are rejected.
        if use_greedy_acceptance and speculative_config is not None:
            if speculative_config.use_relaxed_acceptance_for_thinking:
                raise ValueError(
                    "use_greedy_acceptance is incompatible with "
                    "use_relaxed_acceptance_for_thinking"
                )
            if speculative_config.synthetic_acceptance_rate is not None:
                raise ValueError(
                    "use_greedy_acceptance is incompatible with "
                    "synthetic_acceptance_rate"
                )
        self.use_greedy_acceptance = use_greedy_acceptance
        relaxed_topk: int | None = None
        relaxed_delta: float | None = None
        if (
            speculative_config is not None
            and speculative_config.use_relaxed_acceptance_for_thinking
        ):
            relaxed_topk = speculative_config.relaxed_topk
            relaxed_delta = speculative_config.relaxed_delta
        self.acceptance_sampler = AcceptanceSampler(
            synthetic_acceptance_rate=(
                speculative_config.synthetic_acceptance_rate
                if speculative_config
                else None
            ),
            num_draft_steps=self.num_draft_steps,
            use_stochastic=not use_greedy_acceptance,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )
        self.target = Gemma4TextModel(config)
        self.merger = RaggedTokenMerger(config.devices[0])

        self.draft: Any = None
        self._draft_config = draft_config

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        image_embeddings: list[TensorValue],
        image_token_indices: list[TensorValue],
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        sliding_kv_collections: list[PagedCacheValues],
        global_kv_collections: list[PagedCacheValues],
        return_n_logits: TensorValue,
        host_input_row_offsets: TensorValue,
        data_parallel_splits: TensorValue,
        batch_context_lengths: list[TensorValue],
        seed: TensorValue,
        temperature: TensorValue,
        top_k: TensorValue,
        max_k: TensorValue,
        top_p: TensorValue,
        min_top_p: TensorValue,
        in_thinking_phase: TensorValue,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        # -- 1. Merge tokens + draft tokens --
        merged_tokens, merged_offsets, host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )

        # -- 2. Broadcast merged offsets to all devices --
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )

        devices = self.config.devices
        n_devs = len(devices)
        device0 = devices[0]

        # -- 3. Target forward --
        # Vision embeddings are scattered into the target's token embeddings at
        # image_token_indices. Images are only present during prefill, where
        # draft_tokens is [batch, 0] (K=0), so merged_tokens == prompt tokens
        # and merged_offsets_per_dev == input_row_offsets: the indices computed
        # against the active window map directly onto merged_tokens. During
        # decode both lists are empty (zero-row), so the scatter is a no-op.
        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            sliding_kv_collections,
            global_kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            image_embeddings,
            image_token_indices,
        )

        logits = target_outputs[1]
        hidden_states = list(target_outputs[3 : 3 + n_devs])

        # -- 4. Rejection sampling --
        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=self.config.devices[0],
        )
        num_accepted_draft_tokens, recovered, bonus, next_tokens = (
            accept_and_pick_next_tokens(
                self.acceptance_sampler,
                draft_tokens,
                logits,
                # Per-row seeds: each row's sampling is keyed off its own
                # seed, never coupled to co-residents' draws.
                seed=seed,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                in_thinking_phase=in_thinking_phase,
                token_bitmasks=effective_bitmasks,
            )
        )

        # -- 5. Compute corrected merged sequence and shift --
        shifted_corrected = shift_corrected_tokens(
            self.merger, tokens, input_row_offsets, recovered, bonus
        )

        # Compute q_max_seq_len for cross-attention kernel (uint32 scalar on CPU)
        host_seq_lens = host_merged_offsets[1:] - host_merged_offsets[:-1]
        q_max_seq_len_prefill = (
            ops.max(host_seq_lens, axis=0).cast(DType.uint32).broadcast_to([1])
        )

        # -- 6. Draft step 0 (prefill): pass target hidden states to draft --
        assert self.draft is not None
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            tokens=shifted_corrected,
            hidden_states=hidden_states,
            signal_buffers=signal_buffers,
            target_sliding_kv=sliding_kv_collections,
            target_global_kv=global_kv_collections,
            return_n_logits=return_n_logits,
            input_row_offsets=merged_offsets_per_dev,
            kv_input_row_offsets=merged_offsets_per_dev,
            q_max_seq_len=q_max_seq_len_prefill,
        )

        # Steps 1..K use LAST_PER_DEVICE for hidden states.
        self.draft.return_hidden_states = ReturnHiddenStates.LAST_PER_DEVICE
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_variable_logits = draft_outputs[1]
        all_hs = list(draft_outputs[3 : 3 + n_devs])

        draft_logits_3d = _reshape_target_logits(draft_variable_logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        hidden_dim = self._draft_config.backbone_hidden_size

        draft_hs = gather_accepted_hidden_states(
            all_hs,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
            num_accepted=num_accepted_draft_tokens,
            num_draft_tokens=draft_tokens.shape[1],
            data_parallel_degree=1,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=signal_buffers,
            device=device0,
            split_prefix="mtp",
        )

        # -- 7. Draft steps 1..K (decode) --
        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted_draft_tokens.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(devices) > 1
        # The draft shares the target's KV; its logical cache length equals
        # the target sliding cache's. (Used only for the draft's RoPE
        # position, not for any cache write.)
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            data_parallel_splits,
            [kv.cache_lengths for kv in sliding_kv_collections],
            signal_buffers if use_comm else None,
        )

        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])

        decode_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device0,
            dtype=DType.uint32,
        )
        decode_offsets_per_dev = ops.distributed_broadcast(
            decode_offsets, signal_buffers
        )

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        q_max_seq_len_decode = ops.constant(
            1, DType.uint32, DeviceRef.CPU()
        ).broadcast_to([1])

        # Create decode-step KV collections with max_prompt_length = 1.
        # Without this, cross-attention sees max_prompt_length() > 1 (the
        # merged prefill length), is_token_generation stays False, and the
        # depth512 pair-CTA prefill kernel is invoked for a 1-token query —
        # a degenerate case that deadlocks under high KV-cache pressure.
        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])
        decode_sliding_kv = [
            replace(
                kv,
                max_prompt_length=one,
                max_cache_length=kv.max_cache_length,
            )
            for kv in sliding_kv_collections
        ]
        decode_global_kv = [
            replace(
                kv,
                max_prompt_length=one,
                max_cache_length=kv.max_cache_length,
            )
            for kv in global_kv_collections
        ]

        for step in range(1, self.num_draft_steps):
            draft_hs = [
                draft_hs[i].rebind([f"mtp_step{step}_batch", hidden_dim])
                for i in range(n_devs)
            ]

            step_outputs = self.draft(
                tokens=next_draft_tokens,
                hidden_states=draft_hs,
                signal_buffers=signal_buffers,
                target_sliding_kv=decode_sliding_kv,
                target_global_kv=decode_global_kv,
                return_n_logits=draft_return_n_logits,
                input_row_offsets=decode_offsets_per_dev,
                kv_input_row_offsets=merged_offsets_per_dev,
                q_max_seq_len=q_max_seq_len_decode,
                rope_cache_lengths=cache_lengths_per_dev,
            )

            logits_step = step_outputs[0]
            draft_hs_full = list(step_outputs[1 : 1 + n_devs])
            # TP / single-device: each device already holds a full
            # replica, no per-device slicing needed.
            draft_hs = list(draft_hs_full)

            next_draft_tokens = ops.argmax(logits_step, axis=-1).reshape([-1])
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            num_accepted_draft_tokens,
            next_tokens,
            new_token,
        )

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the unified MTP Gemma4 graph.

        Distributed (signals + DP splits) graph with the per-row
        ``in_thinking_phase`` flag, appending the structured-output bitmask
        triple when enabled. See :func:`build_spec_decode_input_types` for the
        canonical ordering.
        """
        return build_spec_decode_input_types(
            SpecDecodeInputTypeSpec(
                distributed=True,
                data_parallel_degree=1,
                enable_vision=True,
                vision_hidden_size=self.config.text_config.hidden_size,
                include_in_thinking_phase=True,
                enable_structured_output=self.enable_structured_output,
            ),
            devices=self.config.devices,
            kv_params=kv_params,
        )
