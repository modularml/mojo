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
"""Qwen3.5 with its MTP head: merge, verify, roll back, and draft in one graph.

The hybrid target makes this graph differ from the three existing unified MTP
graphs in one structural way: verifying K draft tokens advances 48 Gated
DeltaNet recurrences that no length pointer can rewind. The verify therefore
runs on a shadow copy of the state pools and the accepted prefix is replayed
into the live ones -- see :mod:`.state_rollback`.

Two smaller Qwen-specific choices:

- Acceptance is **stochastic**, never the greedy fast path, because only the
  stochastic sampler applies the grammar bitmask. This checkpoint's
  ``lm_head`` carries 243 live padding rows, so an unmasked argmax can emit an
  undecodable id -- including at the prefill's position 0, which is the one
  the design flagged.
- The draft's output projection is the target's NVFP4 ``lm_head``, applied at
  one position per request rather than over the whole verified window.
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
from max.pipelines.kv_cache.paged_kv_cache.increment_cache_lengths import (
    increment_cache_lengths_from_counts,
)
from max.pipelines.speculative.config import SpeculativeConfig
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
    gather_accepted_hidden_states,
    merge_tokens_and_host_offsets,
    shift_corrected_tokens,
)

from ..qwen3_5.layers.gated_deltanet import GatedDeltaReplayInputs
from ..qwen3_5.model_config import Qwen3_5Config
from ..qwen3_5.mtp import Qwen3_5MTP
from ..qwen3_5.qwen3_5 import Qwen3_5, Qwen3_5LinearAttentionBlock
from .state_rollback import (
    accepted_row_plan,
    replay_state_pools,
    snapshot_state_pools,
)


class UnifiedMTPQwen3_5(Module):
    """Fused module: merge + target verify + state rollback + draft chain."""

    def __init__(
        self,
        config: Qwen3_5Config,
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
        if speculative_config is not None:
            if speculative_config.use_greedy_acceptance:
                raise ValueError(
                    "Qwen3.5 MTP requires stochastic acceptance: this"
                    " checkpoint's state rollback and lm_head padding"
                    " exclusion are only validated through the stochastic"
                    " path."
                )
            if speculative_config.synthetic_acceptance_rate is not None:
                raise ValueError(
                    "synthetic acceptance would bypass the state rollback's"
                    " accepted-length plan"
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
            num_draft_steps=self.num_draft_steps,
            use_stochastic=True,
            relaxed_topk=relaxed_topk,
            relaxed_delta=relaxed_delta,
        )

        self.target = Qwen3_5(config)
        self.target.return_logits = ReturnLogits.VARIABLE
        self.target.return_hidden_states = ReturnHiddenStates.ALL_NORMALIZED
        self.merger = RaggedTokenMerger(config.devices[0])

        # The draft owns a single-layer KV group, so its layer index is 0. It
        # shares the target's embedding table and rotary cache by reference,
        # which is what makes `mtp_use_dedicated_embeddings: false` structural
        # rather than a load-time convention.
        self.draft = Qwen3_5MTP(
            config=config,
            embed_tokens=self.target.embed_tokens,
            rope=self.target.rope,
            create_norm=self.target.create_norm,
            kv_layer_idx=0,
        )

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
        devices = self.config.devices
        n_devs = len(devices)
        device0 = devices[0]

        merged_tokens, merged_offsets, _host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )
        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
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

        # -- Target verify over the merged window.
        captures: list[list[GatedDeltaReplayInputs]] = [
            [] for _ in range(n_devs)
        ]
        for layer_idx in self.target.linear_layer_indices:
            block = self.target.layers[layer_idx]
            assert isinstance(block, Qwen3_5LinearAttentionBlock)
            block.replay_capture = captures

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
            block = self.target.layers[layer_idx]
            assert isinstance(block, Qwen3_5LinearAttentionBlock)
            block.replay_capture = None

        # VARIABLE logits + ALL_NORMALIZED hidden states ->
        # (last_logits, logits, offsets, hs_0..hs_{n-1}).
        logits = target_outputs[1]
        hidden_states = list(target_outputs[3 : 3 + n_devs])

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=device0,
        )
        num_accepted, recovered, bonus, next_tokens = (
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

        # -- Roll the live pools forward over exactly the accepted prefix.
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

        # -- Draft step 0 over the corrected window, then K-1 decode steps.
        shifted_corrected = shift_corrected_tokens(
            self.merger, tokens, input_row_offsets, recovered, bonus
        )
        draft_hs_all = self.draft(
            tokens=shifted_corrected,
            hidden_states=hidden_states,
            signal_buffers=signal_buffers,
            kv_collections=draft_kv,
            input_row_offsets=merged_offsets_per_dev,
        )
        draft_hs = gather_accepted_hidden_states(
            draft_hs_all,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
            num_accepted=num_accepted,
            num_draft_tokens=draft_tokens.shape[1],
            data_parallel_degree=1,
            data_parallel_splits=data_parallel_splits,
            signal_buffers=signal_buffers,
            device=device0,
            split_prefix="mtp",
        )

        hidden_dim = self.config.hidden_size
        input_lengths = ops.rebind(
            (input_row_offsets[1:] - input_row_offsets[:-1]).cast(DType.int64),
            ["batch_size"],
        )
        accepted_lengths = (
            input_lengths + num_accepted.cast(DType.int64)
        ).rebind(["batch_size"])
        cache_lengths_per_dev = increment_cache_lengths_from_counts(
            accepted_lengths,
            data_parallel_splits,
            [kv.cache_lengths for kv in draft_kv],
            signal_buffers if n_devs > 1 else None,
        )

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
        one = ops.constant(1, DType.uint32, DeviceRef.CPU()).broadcast_to([1])
        # A single-token draft step must not inherit the merged window's
        # prompt length, or cross-attention picks the prefill kernel for a
        # one-row query. Qwen3.5 declares no separate draft dispatch metadata
        # (its `construct_kv_params` leaves `speculative_method` unset), so
        # this length is the only thing distinguishing the two shapes.
        draft_decode_kv = [
            replace(kv, max_prompt_length=one) for kv in draft_kv
        ]

        all_draft_tokens = [
            self._draft_argmax(draft_hs, signal_buffers).rebind(["batch_size"])
        ]
        for _ in range(1, self.num_draft_steps):
            # The gather that produced these left the row count opaque; it is
            # the batch, and the concat inside the draft needs to see that.
            draft_hs = [
                draft_hs[i].rebind(["batch_size", hidden_dim])
                for i in range(n_devs)
            ]
            step_kv = [
                replace(kv, cache_lengths=cl)
                for kv, cl in zip(
                    draft_decode_kv, cache_lengths_per_dev, strict=True
                )
            ]
            draft_hs = self.draft(
                tokens=all_draft_tokens[-1],
                hidden_states=draft_hs,
                signal_buffers=signal_buffers,
                kv_collections=step_kv,
                input_row_offsets=decode_offsets_per_dev,
            )
            all_draft_tokens.append(
                self._draft_argmax(draft_hs, signal_buffers).rebind(
                    ["batch_size"]
                )
            )
            cache_lengths_per_dev = [cl + 1 for cl in cache_lengths_per_dev]

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)

        return (num_accepted, next_tokens, new_token)

    def _draft_argmax(
        self,
        draft_hs: list[TensorValue],
        signal_buffers: list[BufferValue],
    ) -> TensorValue:
        """Projects one hidden state per request through the shared lm_head."""
        logits = self.target.lm_head(draft_hs, signal_buffers)[0]
        return ops.argmax(logits, axis=-1).reshape([-1])

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Canonical spec-decode signature plus the Qwen state-pool tail.

        The tail is ``slot_idx``, then the live conv and recurrent pools, then
        their shadows -- every block device-major, matching the base graph's
        ordering so the Mach slot layout stays a superset of it.
        """
        config = self.config
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
        # The same property the base graph and the serving-side state cache
        # read: in spec mode the engine wires ONE pool allocation into both
        # MEFs, so the dtype is a single global choice — bf16 by default,
        # fp32 via the state_pool_dtype knob (the rollback is bit-exact at
        # either; what bf16 gives up is spec-on bit-matching spec-off).
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
