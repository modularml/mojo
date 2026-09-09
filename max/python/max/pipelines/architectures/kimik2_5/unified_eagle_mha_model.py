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
"""Eagle3 MHA-draft + Kimi K2.5 (MLA target) fused nn.Module.

Mirrors :class:`Eagle3KimiK25Unified` but:

- carries an MHA draft (:class:`Eagle3MHADraft`) whose KV cache geometry
  is independent of the target's MLA cache;
- declares an independent set of per-device draft KV inputs (kv_blocks,
  cache_lengths, lookup_table, max_prompt_length, max_cache_length,
  attention_dispatch_metadata) instead of borrowing the target's slots.
"""

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
from ..eagle_common.eagle_mha_draft import (
    Eagle3MHADraft,
    Eagle3MHADraftConfig,
)


class Eagle3MHAKimiK25Unified(Module):
    """Fused: merge + target (MLA) forward + rejection + shift + MHA draft.

    The target forwards as a standard distributed DeepseekV3 (MLA). The
    draft consumes its **own** per-device ``PagedCacheValues`` (with MHA
    layout), allowing independent KV geometry and independent dispatch
    metadata.
    """

    def __init__(
        self,
        config: DeepseekV3Config,
        draft_config: Eagle3MHADraftConfig,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
        enable_vision: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.draft_config = draft_config
        self.enable_structured_output = enable_structured_output
        # ``enable_vision`` controls whether the unified graph accepts
        # per-device image embeddings + scatter indices and scatters them
        # into the merged token embedding before the target forward.
        # Only Kimi-style targets that carry a vision encoder set this;
        # the bare DeepseekV3 + MHA-draft pipeline leaves it False.
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

        self.draft: Eagle3MHADraft | None = None
        if draft_config is not None:
            self.draft = Eagle3MHADraft(draft_config)
            aux_layer_ids = config.eagle_aux_hidden_state_layer_ids
            assert aux_layer_ids is not None
            assert len(aux_layer_ids) == draft_config.fc_input_multiplier, (
                f"the target captures {len(aux_layer_ids)} aux hidden states "
                f"but the draft's fc fuses {draft_config.fc_input_multiplier}"
            )

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
            # Mirror ``KimiK2_5MoEDecoder.__call__``: embed the merged
            # tokens, scatter image embeddings into the merged sequence at
            # ``image_token_indices``, then run the rest of the target
            # stack on the resulting hidden states.
            #
            # During prefill K=0 (no draft tokens to merge), so the
            # indices remain valid for the merged sequence with no
            # remapping. During decode no new images are introduced so
            # ``image_token_indices`` is empty.
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

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=devices[0],
        )

        seed_scalar = seed[0]
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

        draft0_kv_collections = [
            _patch_draft0_kv_cache(kv) for kv in draft_kv_collections
        ]

        # Step 0: ALL hidden states + VARIABLE logits.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,
            draft0_kv_collections,
            return_n_logits,
            merged_offsets_per_dev,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
        )
        # Steps 1..K: LAST + ALL (per-device hs feeds next step's
        # fused_target_hs directly).
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
            split_prefix="eagle3_mha",
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

        # The pipeline_model already swapped attention_dispatch_metadata to
        # the draft slot at graph-init; carry that through to the step-1+
        # collections along with the updated max prompt / cache lengths.
        draft_kv_collections = [
            replace(
                kv,
                max_prompt_length=one,
                max_cache_length=kv.max_cache_length,
                attention_dispatch_metadata=kv.draft_attention_dispatch_metadata,
            )
            for kv in draft_kv_collections
        ]

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        for step in range(1, self.num_draft_steps):
            draft_hs = [
                draft_hs[i].rebind(
                    [f"draft_mha_step{step}_batch_dev_{i}", hidden_dim]
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
                split_prefix=f"eagle3_mha_draft_step{step}",
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
            first_rejected,
            next_tokens,
            new_token,
        )

    def input_types(
        self,
        kv_params: MultiKVCacheParams,
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types for the unified MHA-draft graph.

        ``kv_params`` is the unified ``{"target", "draft"}`` tree. The target
        leaf is MLA; the draft leaf is MHA and carries its own per-device
        blocks, cache lengths, lookup table, and dispatch metadata (the qN
        verify slot plus the q1 decode slot), so the graph-capture branch can
        populate MHA geometry for the draft independently of the target's MLA
        geometry. Optionally prepends per-device vision inputs and carries the
        per-row ``in_thinking_phase`` flag plus the structured-output bitmask
        triple. See :func:`build_spec_decode_input_types` for the canonical
        ordering.
        """
        spec = SpecDecodeInputTypeSpec(
            distributed=True,
            data_parallel_degree=self.config.data_parallel_degree,
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


# TODO(SERVOPT-1437): This is a temporary patch, until we have a proper way
# to pass the draft0 attention dispatch metadata.
def _patch_draft0_kv_cache(kv: PagedCacheValues) -> PagedCacheValues:
    """Returns ``kv`` with ``attention_dispatch_metadata`` re-sized for the
    unified-eagle draft's step-0 prefill.

    The unified-eagle graph calls the draft twice per spec-decode step:
      1. First on a merged prefill sequence whose per-ctx q is larger than 1,
      2. Then on single-token decode steps where q == 1.

    The ``cache_manager`` resolves ``draft_attention_dispatch_metadata`` with
    ``max_prompt_length == 1``, which is the correct width for the
    decode steps, but not for prefill. This function patches the metadata
    to the correct width for the prefill step.
    """
    decode_md = kv.draft_attention_dispatch_metadata
    assert decode_md is not None
    step0_max_prompt = kv.max_prompt_length.cast(DType.int64).reshape([1]) + 1
    step0_md = ops.concat(
        [decode_md[0:1], step0_max_prompt, decode_md[2:4]],
        axis=0,
    )
    return replace(kv, attention_dispatch_metadata=step0_md)
