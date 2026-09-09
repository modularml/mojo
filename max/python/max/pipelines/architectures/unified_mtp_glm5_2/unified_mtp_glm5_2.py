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
"""GLM-5.2 (DeepSeek-V3.2 sparse) with MTP nn.Module.

Composes the DeepSeek-V3.2 target, a single-layer sparse NextN draft, greedy
rejection sampling, and the prefill shift into one graph-buildable module —
the V3.2 analogue of the UnifiedMTPDeepseekV3 module.
Two structural differences:

- The target and draft both use *sparse* MLA (lightning indexer), so each
  carries a paired ``{mla, indexer}`` KV cache rather than a single MLA cache.
- ``index_share_for_mtp_iteration``: the draft computes its lightning-indexer
  top-k selection at step 0 and reuses it (gathered at the accepted positions)
  for every subsequent draft step.
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
from max.nn.kv_cache import KVCacheParamInterface, PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import forward_sharded_layers
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

from ..deepseekV3_2.deepseekV3_2 import DeepseekV3_2
from ..deepseekV3_2.model_config import DeepseekV3_2Config
from ..deepseekV3_2_nextn.deepseekV3_2_nextn import DeepseekV3_2NextN
from ..deepseekV3_2_nextn.model_config import DeepseekV3_2NextNConfig


class UnifiedMTPGlm5_2(Module):
    """Fused nn.Module: merge + V3.2 target + greedy rejection + sparse draft."""

    def __init__(
        self,
        config: DeepseekV3_2Config,
        draft_config: DeepseekV3_2NextNConfig | None = None,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
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
        self.target = DeepseekV3_2(config)
        self.target.emit_last_token_logits = False
        self.merger = RaggedTokenMerger(config.devices[0])

        assert draft_config is not None
        self.draft = DeepseekV3_2NextN(draft_config)

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        draft_tokens: TensorValue,
        signal_buffers: list[BufferValue],
        target_mla_kv: list[PagedCacheValues],
        target_indexer_kv: list[PagedCacheValues],
        draft_mla_kv: list[PagedCacheValues],
        draft_indexer_kv: list[PagedCacheValues],
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
        ep_inputs: list[Value[Any]] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        devices = self.config.devices
        n_devs = len(devices)
        device0 = devices[0]

        merged_tokens, merged_offsets, host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )

        # ``DeepseekV3_2.__call__`` broadcasts ``input_row_offsets`` internally,
        # so the target takes the single merged offsets tensor.
        target_outputs = self.target(
            merged_tokens,
            signal_buffers,
            target_mla_kv,
            target_indexer_kv,
            return_n_logits,
            merged_offsets,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
        )

        # VARIABLE logits + ALL_NORMALIZED hidden states with last_logits
        # suppressed (emit_last_token_logits=False) ->
        # (logits, offsets, hs_0..hs_{n-1}).
        logits = target_outputs[0]
        hidden_states = list(target_outputs[2 : 2 + n_devs])

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=device0,
        )

        num_accepted_draft_tokens, recovered, bonus, next_tokens = (
            accept_and_pick_next_tokens(
                self.acceptance_sampler,
                draft_tokens,
                logits,
                seed=seed[0],
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

        # The draft offsets are needed per-device (the draft consumes a
        # per-device ``input_row_offsets`` list).
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )

        # Step 0: ALL hidden states (per-batch gather at accepted positions) +
        # VARIABLE logits (draft argmax). The draft computes its own lightning
        # indexer top-k here and returns it for reuse in later steps.
        # last_logits (index 0) is unused here, so suppress its vocab-sized
        # lm_head projection.
        self.draft.return_hidden_states = ReturnHiddenStates.ALL
        self.draft.return_logits = ReturnLogits.VARIABLE
        self.draft.emit_last_token_logits = False
        draft_outputs = self.draft(
            shifted_corrected,
            hidden_states,
            signal_buffers,
            draft_mla_kv,
            draft_indexer_kv,
            return_n_logits,
            merged_offsets_per_dev,
            host_merged_offsets,
            data_parallel_splits,
            batch_context_lengths,
            ep_inputs,
            prev_topk_indices=None,
            reuse_prev_topk=False,
        )
        # Steps 1..K reuse the LAST_PER_DEVICE path (its internal allgather
        # fences successive draft invocations) and read last_logits at index 0,
        # so re-enable it.
        self.draft.return_hidden_states = ReturnHiddenStates.LAST_PER_DEVICE
        self.draft.return_logits = ReturnLogits.LAST_TOKEN
        self.draft.emit_last_token_logits = True

        # Step-0 layout with last_logits suppressed:
        # (logits, offsets, hs[n], topk[n]).
        draft_variable_logits = draft_outputs[0]
        all_hs = list(draft_outputs[2 : 2 + n_devs])
        step0_topk = list(draft_outputs[2 + n_devs : 2 + 2 * n_devs])

        draft_logits_3d = _reshape_target_logits(draft_variable_logits)
        draft_argmax = ops.squeeze(
            ops.argmax(draft_logits_3d, axis=-1), axis=-1
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        hidden_dim = self.draft.config.hidden_size
        index_topk = self.draft.config.index_topk

        draft_hs = gather_accepted_hidden_states(
            all_hs,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
            num_accepted=num_accepted_draft_tokens,
            num_draft_tokens=draft_tokens.shape[1],
            data_parallel_degree=self.config.data_parallel_degree,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=signal_buffers,
            device=device0,
            split_prefix="mtp",
        )
        # The draft's ``hnorm`` expects the post-final-norm hidden state -- the
        # same convention the target hands over at step 0 (ALL_NORMALIZED).
        # ``shared_head_norm`` is row-wise, so normalizing after the gather is
        # equivalent to normalizing the whole sequence and far cheaper.
        draft_hs = forward_sharded_layers(
            self.draft.shared_head_norm_shards, draft_hs
        )

        # Gather the step-0 lightning-indexer top-k at the same accepted
        # positions so later decode steps reuse the step-0 selection
        # (index_share_for_mtp_iteration). Each entry is [batch, index_topk].
        reuse_topk = gather_accepted_hidden_states(
            step0_topk,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
            num_accepted=num_accepted_draft_tokens,
            num_draft_tokens=draft_tokens.shape[1],
            data_parallel_degree=self.config.data_parallel_degree,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=signal_buffers,
            device=device0,
            split_prefix="mtp_topk",
        )

        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted_draft_tokens.cast(DType.int64)
        ).rebind(["batch_size"])

        use_comm = len(devices) > 1
        # Only the MLA cache participates in the per-step decode; the indexer is
        # skipped on reused-topk steps, so its cache lengths are left untouched.
        mla_cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            data_parallel_splits,
            [kv.cache_lengths for kv in draft_mla_kv],
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

        # Switch the draft MLA cache to its decode dispatch metadata for the
        # single-token steps.
        draft_mla_kv = [
            replace(
                kv,
                max_prompt_length=one,
                max_cache_length=kv.max_cache_length,
                attention_dispatch_metadata=kv.draft_attention_dispatch_metadata,
                mla_num_partitions=kv.draft_mla_num_partitions,
            )
            for kv in draft_mla_kv
        ]

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]

        for step in range(1, self.num_draft_steps):
            draft_hs = [
                draft_hs[i].rebind(
                    [f"mtp_step{step}_batch_dev_{i}", hidden_dim]
                )
                for i in range(n_devs)
            ]
            reuse_topk = [
                reuse_topk[i].rebind(
                    [f"mtp_step{step}_batch_dev_{i}", index_topk]
                )
                for i in range(n_devs)
            ]

            step_mla_kv: list[PagedCacheValues] = [
                replace(kv, cache_lengths=cl)
                for kv, cl in zip(
                    draft_mla_kv, mla_cache_lengths_per_dev, strict=True
                )
            ]

            step_outputs = self.draft(
                next_draft_tokens,
                draft_hs,
                signal_buffers,
                step_mla_kv,
                draft_indexer_kv,
                draft_return_n_logits,
                decode_offsets_per_dev,
                host_decode_offsets,
                data_parallel_splits,
                batch_context_lengths,
                ep_inputs,
                prev_topk_indices=reuse_topk,
                reuse_prev_topk=True,
                split_prefix=f"mtp_draft_step{step}",
            )

            # LAST_PER_DEVICE layout: (last_logits, hs[n], topk[n]).
            logits = step_outputs[0]
            draft_hs_full = list(step_outputs[1 : 1 + n_devs])
            next_step = step + 1
            if self.config.data_parallel_degree > 1:
                draft_hs = [
                    ops.slice_tensor(
                        draft_hs_full[i],
                        [
                            (
                                slice(
                                    data_parallel_splits[i],
                                    data_parallel_splits[i + 1],
                                ),
                                f"mtp_step{next_step}_batch_dev_{i}",
                            ),
                        ],
                    )
                    for i in range(n_devs)
                ]
            else:
                draft_hs = list(draft_hs_full)
            draft_hs = forward_sharded_layers(
                self.draft.shared_head_norm_shards, draft_hs
            )

            next_draft_tokens = ops.argmax(logits, axis=-1).reshape([-1])
            all_draft_tokens.append(
                ops.rebind(next_draft_tokens, ["batch_size"])
            )

            mla_cache_lengths_per_dev = [
                cl + 1 for cl in mla_cache_lengths_per_dev
            ]
            batch_context_lengths = [bcl + 1 for bcl in batch_context_lengths]

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
        """Input types for the GLM-5.2 with-MTP graph.

        ``kv_params`` is the nested ``{target: {mla, indexer}, draft: {mla,
        indexer}}`` tree; its flattened inputs carry all four caches. See
        :func:`build_spec_decode_input_types` for the canonical ordering.
        """
        spec = SpecDecodeInputTypeSpec(
            distributed=True,
            data_parallel_degree=self.config.data_parallel_degree,
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
