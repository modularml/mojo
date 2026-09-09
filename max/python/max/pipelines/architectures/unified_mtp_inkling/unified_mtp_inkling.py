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
"""Inkling with MTP nn.Module: merge + target + reject + chained draft depths."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    DeviceRef,
    Dim,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.comm import Signals
from max.nn.kernels import merge_ragged_tensors
from max.nn.kv_cache import (
    KVCacheParamInterface,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import (
    AcceptanceSampler,
    _reshape_target_logits,
)
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.nn.transformer.distributed_transformer import (
    distributed_logits_postprocess,
)
from max.nn.transformer.transformer import logits_postprocess
from max.pipelines.speculative.config import SpeculativeConfig
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
)
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    spec_decode_tail_input_types,
)
from max.pipelines.speculative.unified_graph_ops import (
    accept_and_pick_next_tokens,
    apply_overlap_bitmask,
    gather_accepted_hidden_states,
    merge_tokens_and_host_offsets,
    shift_corrected_tokens,
)

from ..inkling.inkling import Inkling
from ..inkling.model_config import InklingConfig
from .inkling_mtp import InklingMultiTokenPredictor


def _zero_conv_pools(pools: Sequence[Sequence[BufferValue]]) -> None:
    """Clears draft conv slots so a re-prefill does not replay old windows."""
    for rank_pools in pools:
        for pool in rank_pools:
            ops.buffer_store(
                pool,
                ops.broadcast_to(
                    ops.constant(0.0, pool.dtype, device=pool.device),
                    pool.shape,
                ),
            )


class UnifiedMTPInkling(Module):
    """Fused nn.Module: merge + Inkling target + rejection + chained MTP depths."""

    def __init__(
        self,
        config: InklingConfig,
        draft: InklingMultiTokenPredictor,
        speculative_config: SpeculativeConfig | None = None,
        enable_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_structured_output = enable_structured_output
        self.num_draft_steps = draft.n_depths
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
        target_config = config
        kv = config.kv_params
        if "target" in kv.children:
            target_kv = kv.children["target"]
            assert isinstance(target_kv, MultiKVCacheParams)
            target_config = replace(config, kv_params=target_kv)
        self.target = Inkling(
            target_config,
            return_logits=ReturnLogits.VARIABLE,
            return_hidden_states=ReturnHiddenStates.ALL_NORMALIZED,
        )
        self.draft = draft
        self.merger = RaggedTokenMerger(config.devices[0])

    def __call__(
        self,
        tokens: TensorValue,
        input_row_offsets: TensorValue,
        positions: TensorValue,
        draft_tokens: TensorValue,
        image_embeddings: TensorValue,
        image_indices: TensorValue,
        signal_buffers: list[BufferValue],
        target_kv: dict[str, list[PagedCacheValues]],
        draft_kv: dict[str, list[PagedCacheValues]],
        return_n_logits: TensorValue,
        host_input_row_offsets: TensorValue,
        slot_idx: list[TensorValue],
        target_conv_pools: list[list[BufferValue]],
        draft_conv_pools: list[list[BufferValue]],
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
        devices = self.config.devices
        n_devs = len(devices)
        device0 = devices[0]
        use_comm = n_devs > 1

        merged_tokens, merged_offsets, host_merged_offsets = (
            merge_tokens_and_host_offsets(
                self.merger,
                tokens,
                input_row_offsets,
                draft_tokens,
                host_input_row_offsets,
            )
        )
        del host_merged_offsets
        merged_pos = _merge_positions(
            positions, input_row_offsets, draft_tokens
        )
        if use_comm:
            merged_offsets_per_dev = ops.distributed_broadcast(
                merged_offsets, signal_buffers
            )
        else:
            merged_offsets_per_dev = [merged_offsets]

        target_outputs = self.target(
            merged_tokens,
            merged_offsets,
            merged_pos,
            return_n_logits,
            image_embeddings,
            image_indices,
            signal_buffers,
            target_kv,
            slot_idx,
            target_conv_pools,
        )
        logits = target_outputs[1]
        hidden_states = list(target_outputs[3 : 3 + n_devs])

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
        _zero_conv_pools(draft_conv_pools)

        token_embeds = self.draft.embed_tokens(
            shifted_corrected, signal_buffers
        )
        hs = self.draft.forward_depth(
            0,
            token_embeds,
            hidden_states,
            draft_kv,
            merged_offsets_per_dev,
            merged_pos,
            draft_conv_pools,
            slot_idx,
            signal_buffers,
        )
        draft_logits, draft_argmax = self._depth_logits(
            hs,
            merged_offsets,
            merged_offsets_per_dev,
            return_n_logits,
            True,
            signal_buffers,
        )
        next_draft_tokens = ops.gather_nd(
            draft_argmax,
            ops.unsqueeze(num_accepted_draft_tokens, axis=-1),
            batch_dims=1,
        ).reshape([-1])

        hidden_dim = self.config.text_config.hidden_size
        draft_hs = _gather_accepted(
            hidden_states=hs,
            merged_offsets=merged_offsets,
            merged_offsets_per_dev=merged_offsets_per_dev,
            num_accepted=num_accepted_draft_tokens,
            num_draft_tokens=draft_tokens.shape[1],
            signal_buffers=signal_buffers,
            device=device0,
            n_devs=n_devs,
        )
        del draft_logits

        decode_offsets = ops.range(
            start=0,
            stop=input_row_offsets.shape[0],
            out_dim="input_row_offsets_len",
            device=device0,
            dtype=DType.uint32,
        )
        decode_offsets_per_dev = (
            ops.distributed_broadcast(decode_offsets, signal_buffers)
            if use_comm
            else [decode_offsets]
        )
        last_accepted_idx = _last_accepted_idx(
            merged_offsets,
            num_accepted_draft_tokens,
            draft_tokens.shape[1],
            device0,
        )
        decode_pos = (
            ops.gather(merged_pos, last_accepted_idx, axis=0) + 1
        ).rebind(["batch_size"])

        next_draft_tokens = next_draft_tokens.rebind(["batch_size"])
        all_draft_tokens = [next_draft_tokens]
        draft_return_n_logits = ops.constant(
            1, DType.int64, DeviceRef.CPU()
        ).broadcast_to([1])
        decode_kv = _kv_with_zero_cache_lengths(draft_kv)

        for step in range(1, self.num_draft_steps):
            step_batch = f"mtp_step{step}_batch"
            draft_hs = [h.rebind([step_batch, hidden_dim]) for h in draft_hs]
            step_embeds = [
                e.rebind([step_batch, hidden_dim])
                for e in self.draft.embed_tokens(
                    next_draft_tokens, signal_buffers
                )
            ]
            draft_hs = self.draft.forward_depth(
                step,
                step_embeds,
                draft_hs,
                decode_kv,
                decode_offsets_per_dev,
                decode_pos.rebind([step_batch]),
                draft_conv_pools,
                slot_idx,
                signal_buffers,
            )
            _, step_argmax = self._depth_logits(
                draft_hs,
                decode_offsets,
                decode_offsets_per_dev,
                draft_return_n_logits,
                False,
                signal_buffers,
            )
            next_draft_tokens = step_argmax.reshape([-1]).rebind(["batch_size"])
            all_draft_tokens.append(next_draft_tokens)
            decode_pos = (decode_pos + 1).rebind(["batch_size"])

        if len(all_draft_tokens) > 1:
            new_token = ops.stack(all_draft_tokens, axis=-1)
        else:
            new_token = ops.unsqueeze(all_draft_tokens[0], -1)
        return (num_accepted_draft_tokens, next_tokens, new_token)

    def _depth_logits(
        self,
        hs: list[TensorValue],
        offsets: TensorValue,
        offsets_per_dev: list[TensorValue],
        return_n_logits: TensorValue,
        variable: bool,
        signal_buffers: list[BufferValue],
    ) -> tuple[TensorValue, TensorValue]:
        """LM-head over a depth's hidden states; argmax is always returned."""
        return_logits = (
            ReturnLogits.VARIABLE if variable else ReturnLogits.LAST_TOKEN
        )
        if len(self.config.devices) > 1:
            outputs = distributed_logits_postprocess(
                hs,
                offsets_per_dev,
                return_n_logits,
                lm_head=self.target._distributed_lm_head,
                signal_buffers=signal_buffers,
                return_logits=return_logits,
                device=self.config.devices[0],
                norm_shards=self.target.norm_shards,
                logits_scaling=self.config.text_config.logits_mup_width_multiplier,
            )
        else:
            outputs = logits_postprocess(
                hs[0],
                offsets,
                return_n_logits,
                self.target.norm,
                self.target._lm_head,
                return_logits,
                logits_scaling=self.config.text_config.logits_mup_width_multiplier,
            )
        if variable:
            logits = outputs[1]
            argmax = ops.squeeze(
                ops.argmax(_reshape_target_logits(logits), axis=-1), axis=-1
            )
            return logits, argmax
        logits = outputs[0]
        return logits, ops.argmax(logits, axis=-1)

    def input_types(
        self, kv_params: KVCacheParamInterface
    ) -> tuple[TensorType | BufferType, ...]:
        """Inkling extras (positions, slots, conv pools) plus the spec-decode tail."""
        devices = self.config.devices
        device = devices[0]
        n_devs = len(devices)
        signals = Signals(devices).input_types() if n_devs > 1 else []

        types: list[TensorType | BufferType] = [
            TensorType(DType.int64, shape=["total_seq_len"], device=device),
            TensorType(
                DType.uint32, shape=["input_row_offsets_len"], device=device
            ),
            TensorType(DType.uint32, shape=["total_seq_len"], device=device),
            TensorType(
                DType.uint32,
                shape=["input_row_offsets_len"],
                device=DeviceRef.CPU(),
            ),
            TensorType(
                DType.int64, shape=["return_n_logits"], device=DeviceRef.CPU()
            ),
            TensorType(
                self.config.dtype,
                shape=[
                    "total_image_tokens",
                    self.config.text_config.hidden_size,
                ],
                device=device,
            ),
            TensorType(
                DType.int32, shape=["total_image_tokens"], device=device
            ),
            *signals,
            *kv_params.flattened_kv_inputs(),
            *(
                TensorType(DType.uint32, shape=["batch_size"], device=dev)
                for dev in devices
            ),
            *self.target.conv_layout.buffer_types(devices),
            *self.draft.conv_layout.buffer_types(devices),
        ]
        types.extend(
            spec_decode_tail_input_types(
                SpecDecodeInputTypeSpec(
                    distributed=n_devs > 1,
                    include_in_thinking_phase=True,
                    enable_structured_output=self.enable_structured_output,
                ),
                device,
            )
        )
        return tuple(types)


def _merge_positions(
    positions: TensorValue,
    input_row_offsets: TensorValue,
    draft_tokens: TensorValue,
) -> TensorValue:
    """Extends each sequence's position ramp across the appended draft tokens."""
    device = positions.device
    k = _shape_to_scalar(draft_tokens.shape[1], device, dtype=DType.uint32)
    last_pos = ops.gather(positions, input_row_offsets[1:] - 1, axis=0)
    batch_plus_1 = ops.shape_to_tensor([input_row_offsets.shape[0]])[0]
    indices = ops.range(
        start=0,
        stop=batch_plus_1,
        out_dim=input_row_offsets.shape[0],
        device=device,
        dtype=DType.uint32,
    )
    draft_offsets = indices * k
    k_range = ops.range(
        start=0,
        stop=ops.shape_to_tensor([draft_tokens.shape[1]])[0],
        out_dim=draft_tokens.shape[1],
        device=device,
        dtype=DType.uint32,
    )
    draft_pos = ops.reshape(
        ops.unsqueeze(last_pos, 1) + 1 + ops.unsqueeze(k_range, 0),
        [-1],
    )
    merged, _ = merge_ragged_tensors(
        positions, input_row_offsets, draft_pos, draft_offsets
    )
    return ops.rebind(merged, ["merged_seq_len"])


def _last_accepted_idx(
    merged_offsets: TensorValue,
    num_accepted: TensorValue,
    num_draft_tokens: Dim,
    device: DeviceRef,
) -> TensorValue:
    last_idx = merged_offsets[1:] - 1
    k = _shape_to_scalar(num_draft_tokens, device)
    return (
        ops.rebind(last_idx, ["batch_size"])
        - k.broadcast_to(["batch_size"])
        + num_accepted.cast(last_idx.dtype)
    )


def _kv_with_zero_cache_lengths(
    kv: dict[str, list[PagedCacheValues]],
) -> dict[str, list[PagedCacheValues]]:
    """Draft depths after step 0 write one token and have no shared past."""
    return {
        key: [
            replace(
                collection,
                cache_lengths=ops.broadcast_to(
                    ops.constant(
                        0,
                        collection.cache_lengths.dtype,
                        collection.cache_lengths.device,
                    ),
                    collection.cache_lengths.shape,
                ),
            )
            for collection in per_dev
        ]
        for key, per_dev in kv.items()
    }


def _gather_accepted(
    *,
    hidden_states: list[TensorValue],
    merged_offsets: TensorValue,
    merged_offsets_per_dev: list[TensorValue],
    num_accepted: TensorValue,
    num_draft_tokens: Dim,
    signal_buffers: list[BufferValue],
    device: DeviceRef,
    n_devs: int,
) -> list[TensorValue]:
    if n_devs == 1:
        idx = _last_accepted_idx(
            merged_offsets, num_accepted, num_draft_tokens, device
        ).cast(DType.int64)
        return [ops.gather(hidden_states[0], idx, axis=0)]
    dummy_splits = ops.concat(
        [
            ops.constant(0, DType.int64, DeviceRef.CPU()).reshape([1]),
            ops.shape_to_tensor([num_accepted.shape[0]]).reshape([1]),
        ]
    )
    return gather_accepted_hidden_states(
        hidden_states,
        merged_offsets=merged_offsets,
        merged_offsets_per_dev=merged_offsets_per_dev,
        num_accepted=num_accepted,
        num_draft_tokens=num_draft_tokens,
        data_parallel_degree=1,
        data_parallel_splits=dummy_splits,
        signal_buffers=signal_buffers,
        device=device,
        split_prefix="mtp",
    )
