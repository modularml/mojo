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
"""Eagle3 + Kimi K2.5 fused nn.Module: merge + target + rejection + shift."""

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
    Value,
    ops,
)
from max.nn.kv_cache import MultiKVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.transformer import captures_by_device
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
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

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..deepseekV3.model_config import DeepseekV3Config
from .eagle3_kimi_k25 import Eagle3KimiK25


class Eagle3KimiK25Unified(Module):
    """Fused nn.Module: merge + target forward + greedy rejection + shift.

    The target model returns concatenated hidden states from 3 intermediate
    layers (first, middle, last). The draft model fuses these via ``fc`` and
    generates the next speculative token.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: DeepseekV3Config | None = None,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        enable_vision: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_structured_output = enable_structured_output
        # ``enable_vision`` controls whether the unified graph accepts
        # per-device image embeddings + scatter indices and scatters them
        # into the merged token embedding before the target forward.
        # Only Kimi-style targets that carry a vision encoder set this;
        # the bare DeepseekV3 + Eagle pipeline leaves it False.
        self.enable_vision = enable_vision
        self.num_draft_steps = (
            speculative_config.num_speculative_tokens
            if speculative_config is not None
            and speculative_config.num_speculative_tokens is not None
            else 1
        )
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
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )
        self.target = DeepseekV3(config)
        self.merger = RaggedTokenMerger(config.devices[0])

        self.draft: Eagle3KimiK25 | None = None
        if draft_config is not None:
            self.draft = Eagle3KimiK25(draft_config)

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
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
        image_embeddings: list[TensorValue] | None = None,
        image_token_indices: list[TensorValue] | None = None,
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        merged_tokens, merged_offsets, host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )

        assert self.draft is not None
        devices = self.config.devices
        n_devs = len(devices)
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )
        if self.enable_vision:
            # Embed merged tokens, scatter image embeddings into the
            # merged sequence at ``image_token_indices``, then run the
            # rest of the target stack on the resulting hidden states.
            # Mirrors ``KimiK2_5MoEDecoder.__call__`` but on the merged
            # sequence.
            #
            # During prefill the merger inserts zero draft tokens per row
            # (K=0), so ``image_token_indices`` remain valid for the
            # merged sequence without remapping. During decode no new
            # images are introduced, so ``image_token_indices`` is empty.
            assert image_embeddings is not None
            assert image_token_indices is not None
            h_per_dev = self.target.embed_tokens(merged_tokens, signal_buffers)
            h_per_dev = [
                merge_multimodal_embeddings(
                    inputs_embeds=h_d,
                    multimodal_embeddings=img_emb_d,
                    image_token_indices=img_idx_d,
                )
                for h_d, img_emb_d, img_idx_d in zip(
                    h_per_dev,
                    image_embeddings,
                    image_token_indices,
                    strict=True,
                )
            ]
            target_outputs = self.target._process_hidden_states(
                h_per_dev,
                signal_buffers,
                kv_collections,
                return_n_logits,
                list(merged_offsets_per_dev),
                host_merged_offsets,
                data_parallel_splits,
                batch_context_lengths,
                ep_inputs,
            )
        else:
            target_outputs = self.target(
                merged_tokens,
                signal_buffers,
                kv_collections,
                return_n_logits,
                merged_offsets_per_dev,
                host_merged_offsets,
                data_parallel_splits,
                batch_context_lengths,
                ep_inputs,
            )
        logits = target_outputs[1]
        hidden_states = captures_by_device(target_outputs[3:], n_devs)

        # ``seed`` is the per-batch ``[batch_size]`` uint64 device buffer
        # that feeds ``topk_fused_sampling`` (recovered + bonus tokens) per
        # row. The rejection-decision RNG below is a Bernoulli coin flip —
        # its marginal accept distribution is unchanged whether each row
        # gets its own Philox stream or all rows share one with offset-
        # based diversity, so we collapse to ``seed[0]`` here. The
        # token-sampling path keeps the full tensor.
        seed_scalar = seed[0]

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=devices[0],
        )

        first_rejected, recovered, bonus, next_tokens = (
            accept_and_pick_next_tokens(
                self.acceptance_sampler,
                draft_tokens,
                logits,
                seed=seed_scalar,
                temperature=temperature,
                top_k=top_k,
                max_k=max_k,
                top_p=top_p,
                min_top_p=min_top_p,
                in_thinking_phase=in_thinking_phase,
                token_bitmasks=effective_bitmasks,
            )
        )

        shifted_corrected = shift_corrected_tokens(
            self.merger, tokens, input_row_offsets, recovered, bonus
        )

        assert draft_kv_collections is not None

        # Step 0 always uses ALL hidden states (for per-batch-element gather
        # at accepted positions) + VARIABLE logits (for draft argmax).
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,
            draft_kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
        )
        # Steps 1..K run in decode mode (one token per batch element). In
        # decode mode, ALL-hs == LAST-hs — we use ALL so the draft returns
        # per-device hidden states directly (avoiding the LAST path's
        # allgather). step_outputs[1] feeds straight back as the next draft
        # step's fused_target_hs.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.LAST_TOKEN

        draft_variable_logits = draft_outputs[1]
        all_hs = list(draft_outputs[3 : 3 + n_devs])

        draft_logits_3d = _reshape_target_logits(draft_variable_logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(first_rejected, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        device0 = devices[0]
        hidden_dim = self.draft.config.hidden_size

        draft_hs = gather_accepted_hidden_states(
            all_hs,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
            num_accepted=first_rejected,
            num_draft_tokens=draft_tokens.shape[1],
            data_parallel_degree=self.config.data_parallel_degree,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=signal_buffers,
            device=device0,
            split_prefix="eagle3",
        )

        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + first_rejected.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(devices) > 1
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            data_parallel_splits,
            [kv.cache_lengths for kv in draft_kv_collections],
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
        # Broadcast once so the draft can skip its own broadcast for every
        # step of the multi-step loop.
        decode_offsets_per_dev = ops.distributed_broadcast(
            decode_offsets, signal_buffers
        )
        host_decode_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=DeviceRef.CPU(),
            dtype=DType.uint32,
        )

        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])

        draft_kv_collections = [
            replace(
                kv,
                max_prompt_length=one,
                max_cache_length=kv.max_cache_length,
                attention_dispatch_metadata=kv.draft_attention_dispatch_metadata,
                mla_num_partitions=kv.draft_mla_num_partitions,
            )
            for kv in draft_kv_collections
        ]

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        for step in range(1, self.num_draft_steps):
            # Per-device shapes differ across DP replicas; use per-device
            # dim names. The draft internally rebinds to
            # `{split_prefix}_seq_dev_{i}` once inside __call__.
            draft_hs = [
                draft_hs[i].rebind(
                    [f"draft_step{step}_batch_dev_{i}", hidden_dim]
                )
                for i in range(n_devs)
            ]

            step_kv: list[PagedCacheValues] = [
                replace(kv, cache_lengths=cl)
                for kv, cl in zip(
                    draft_kv_collections, cache_lengths_per_dev, strict=True
                )
            ]

            step_outputs = self.draft(
                next_draft_tokens,
                draft_hs,
                signal_buffers,
                step_kv,
                draft_return_n_logits,
                decode_offsets_per_dev,
                host_decode_offsets,
                data_parallel_splits,
                batch_context_lengths,
                split_prefix=f"eagle3_draft_step{step}",
            )

            logits = step_outputs[0]
            draft_hs = list(step_outputs[1 : 1 + n_devs])

            next_draft_tokens = ops.argmax(logits, axis=-1).reshape([-1])
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

            cache_lengths_per_dev = [cl + 1 for cl in cache_lengths_per_dev]
            batch_context_lengths = [bcl + 1 for bcl in batch_context_lengths]

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        return (
            first_rejected,  # num_accepted_draft_tokens [B]
            next_tokens,  # next_tokens [B]
            new_token,  # next_draft_tokens [B, num_draft_steps]
        )

    def input_types(
        self, kv_params: MultiKVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the Eagle3 unified graph.

        Distributed (DP + signals + EP) MLA-draft graph that optionally
        prepends per-device vision inputs and carries the per-row
        ``in_thinking_phase`` flag plus the structured-output bitmask triple.
        See :func:`build_spec_decode_input_types` for the canonical ordering.
        """
        spec = SpecDecodeInputTypeSpec(
            data_parallel_degree=self.config.data_parallel_degree,
            distributed=True,
            enable_vision=self.enable_vision,
            vision_hidden_size=self.config.hidden_size,
            include_in_thinking_phase=True,
            enable_structured_output=self.enable_structured_output,
        )
        ep_input_types = (
            self.target.ep_manager.input_types()
            if self.target.ep_manager is not None
            else ()
        )
        return build_spec_decode_input_types(
            spec,
            devices=self.config.devices,
            kv_params=kv_params,
            ep_input_types=ep_input_types,
        )
