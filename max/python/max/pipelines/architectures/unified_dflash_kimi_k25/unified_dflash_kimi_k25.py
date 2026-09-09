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
"""Unified DFlash Kimi K2.5 nn.Module: MLA target + DFlash MHA/GQA draft."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferType,
    BufferValue,
    TensorType,
    TensorValue,
    Value,
    ops,
)
from max.nn.kv_cache import MultiKVCacheParams, PagedCacheValues
from max.nn.layer import Module
from max.nn.sampling.rejection_sampler import AcceptanceSampler
from max.nn.transformer.transformer import (
    captures_by_device,
    fuse_captured_hidden_states,
)
from max.pipelines.lib.vlm_utils import merge_multimodal_embeddings
from max.pipelines.speculative.config import MAGIC_DRAFT_TOKEN_ID
from max.pipelines.speculative.ragged_token_merger import (
    RaggedTokenMerger,
    _shape_to_scalar,
    compute_host_merged_offsets,
)
from max.pipelines.speculative.spec_input_types import (
    SpecDecodeInputTypeSpec,
    build_spec_decode_input_types,
)
from max.pipelines.speculative.unified_graph_ops import apply_overlap_bitmask

from ..deepseekV3.deepseekV3 import DeepseekV3
from ..dflash_kimi_k25 import DFlashKimiK25
from .model_config import UnifiedDflashKimiK25Config


class UnifiedDflashKimiK25(Module):
    """Fused: merge -> target (MLA) -> reject -> materialize -> draft block."""

    def __init__(
        self,
        config: UnifiedDflashKimiK25Config,
        enable_structured_output: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.enable_structured_output = enable_structured_output
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

        self.target = DeepseekV3(config.target)
        self.draft = DFlashKimiK25(config.draft)
        self.merger = RaggedTokenMerger(config.target.devices[0])

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
        image_embeddings: list[TensorValue],
        image_token_indices: list[TensorValue],
        ep_inputs: list[Value[Any]] | None = None,
        draft_kv_collections: list[PagedCacheValues] | None = None,
        pinned_bitmask: TensorValue | None = None,
        wait_payload: BufferValue | None = None,
        device_bitmask_scratch: BufferValue | None = None,
    ) -> tuple[TensorValue, ...]:
        assert draft_kv_collections is not None

        devices = self.config.target.devices
        n_devs = len(devices)
        device0 = devices[0]
        K = self.block_size
        dp_degree = self.config.target.data_parallel_degree
        is_dp = dp_degree > 1

        pre_cache_lengths = [kv.cache_lengths for kv in draft_kv_collections]

        merged_tokens, merged_offsets = self.merger(
            tokens, input_row_offsets, draft_tokens
        )
        merged_tokens = ops.rebind(merged_tokens, ["merged_seq_len"])
        merged_offsets = ops.rebind(merged_offsets, ["input_row_offsets_len"])

        host_merged_offsets = compute_host_merged_offsets(
            host_input_row_offsets, draft_tokens
        )

        merged_offsets_per_dev = ops.distributed_broadcast(
            merged_offsets, signal_buffers
        )

        h_per_dev = [
            merge_multimodal_embeddings(
                inputs_embeds=h_d,
                multimodal_embeddings=img_emb_d,
                image_token_indices=img_idx_d,
            )
            for h_d, img_emb_d, img_idx_d in zip(
                self.target.embed_tokens(merged_tokens, signal_buffers),
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
        target_logits = target_outputs[1]
        target_hs_per_dev = fuse_captured_hidden_states(
            captures_by_device(target_outputs[3:], n_devs)
        )

        effective_bitmasks = apply_overlap_bitmask(
            pinned_bitmask,
            wait_payload,
            device_bitmask_scratch,
            num_steps=draft_tokens.shape[1],
            device=device0,
        )

        seed_scalar = seed[0]
        num_accepted, recovered, bonus = self.acceptance_sampler(
            draft_tokens,
            target_logits,
            seed=seed_scalar,
            temperature=temperature,
            top_k=top_k,
            max_k=max_k,
            top_p=top_p,
            min_top_p=min_top_p,
            token_bitmasks=effective_bitmasks,
        )

        num_steps_scalar_i32 = _shape_to_scalar(
            draft_tokens.shape[1], device0, dtype=DType.int32
        )
        num_steps_scalar_u32 = _shape_to_scalar(
            draft_tokens.shape[1], device0, dtype=DType.uint32
        )
        zero_u32 = ops.constant(0, DType.uint32, device=device0)
        is_prefill = (num_steps_scalar_u32 == zero_u32).broadcast_to(
            ["batch_size"]
        )
        magic_token = ops.constant(
            MAGIC_DRAFT_TOKEN_ID, DType.int64, device=device0
        )
        num_magic_tokens = ops.squeeze(
            ops.sum(
                (draft_tokens == magic_token)
                .cast(DType.int32)
                .rebind(["batch_size", "num_steps"]),
                axis=-1,
            ),
            axis=-1,
        )
        is_dummy_draft = num_magic_tokens == num_steps_scalar_i32.broadcast_to(
            ["batch_size"]
        )
        num_accepted = ops.where(
            is_prefill | is_dummy_draft,
            ops.constant(0, num_accepted.dtype, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )
        prompt_lens = (input_row_offsets[1:] - input_row_offsets[:-1]).rebind(
            ["batch_size"]
        )
        decode_commit = (num_accepted + 1).cast(DType.uint32)
        commit_lengths = ops.where(is_prefill, prompt_lens, decode_commit)
        devices_per_replica = n_devs // dp_degree

        def _replica_sym(i: int) -> str:
            return f"replica_{i // devices_per_replica}_batch_size"

        commit_lengths_per_dev_raw = ops.distributed_broadcast(
            commit_lengths, signal_buffers
        )
        if is_dp:
            commit_lengths_per_dev = []
            for i in range(n_devs):
                start = data_parallel_splits[i]
                end = data_parallel_splits[i + 1]
                local = ops.slice_tensor(
                    commit_lengths_per_dev_raw[i],
                    [(slice(start, end), f"dflash_commit_split_{i}")],
                )
                local = ops.rebind(local, [_replica_sym(i)])
                commit_lengths_per_dev.append(local)
        else:
            commit_lengths_per_dev = [
                ops.rebind(c, [_replica_sym(i)])
                for i, c in enumerate(commit_lengths_per_dev_raw)
            ]

        target_tokens = ops.concat([recovered, bonus], axis=1)
        gather_idx = ops.where(
            is_prefill,
            ops.constant(0, DType.int64, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted.cast(DType.int64),
        )
        next_tokens = ops.gather_nd(
            target_tokens,
            ops.unsqueeze(gather_idx, axis=-1),
            batch_dims=1,
        )

        ctx_hidden = self.draft.project_target_hidden(target_hs_per_dev)

        ctx_kv_collections: list[PagedCacheValues] = [
            replace(kv, cache_lengths=pre)
            for kv, pre in zip(
                draft_kv_collections, pre_cache_lengths, strict=True
            )
        ]
        bumped = [
            pre + commit_lengths_per_dev[i]
            for i, pre in enumerate(pre_cache_lengths)
        ]
        block_kv_collections: list[PagedCacheValues] = [
            replace(kv, cache_lengths=b)
            for kv, b in zip(draft_kv_collections, bumped, strict=True)
        ]

        next_tokens_2d = ops.unsqueeze(next_tokens, axis=1)
        mask_const = ops.constant(
            self.mask_token_id, DType.int64, device=device0
        )
        mask_tail = mask_const.broadcast_to(["batch_size", K - 1])
        block_ids = ops.concat([next_tokens_2d, mask_tail], axis=1)
        block_ids_flat = block_ids.reshape((-1,))

        block_embeds_per_dev = self.target.embed_tokens(
            block_ids_flat, signal_buffers
        )

        block_offsets_per_dev = [
            ops.range(
                start=0,
                stop=input_row_offsets.shape[0],
                out_dim="input_row_offsets_len",
                device=dev,
                dtype=DType.uint32,
            )
            * ops.constant(K, DType.uint32, device=dev)
            for dev in devices
        ]
        if is_dp:
            ctx_input_row_offsets_per_dev: list[TensorValue] = []
            block_offsets_per_dev_list: list[TensorValue] = []
            block_embeds_per_dev_local: list[TensorValue] = []
            for i in range(n_devs):
                start = data_parallel_splits[i]
                end = data_parallel_splits[i + 1]
                start_offset = merged_offsets_per_dev[i][start]
                ctx_offsets_local = (
                    ops.slice_tensor(
                        merged_offsets_per_dev[i],
                        [
                            (
                                slice(start, end + 1),
                                f"dflash_ctx_offset_split_{i}",
                            )
                        ],
                    )
                    - start_offset
                )
                ctx_input_row_offsets_per_dev.append(ctx_offsets_local)

                block_offsets_dev = block_offsets_per_dev[i]
                block_start_offset = block_offsets_dev[start]
                block_offsets_local = (
                    ops.slice_tensor(
                        block_offsets_dev,
                        [
                            (
                                slice(start, end + 1),
                                f"dflash_block_offset_split_{i}",
                            )
                        ],
                    )
                    - block_start_offset
                )
                block_offsets_per_dev_list.append(block_offsets_local)

                token_start = start * K
                token_end = end * K
                block_embeds_local = ops.slice_tensor(
                    block_embeds_per_dev[i],
                    [
                        (
                            slice(token_start, token_end),
                            f"dflash_block_embed_split_{i}",
                        )
                    ],
                )
                block_embeds_per_dev_local.append(block_embeds_local)

            ctx_input_row_offsets_arg = ctx_input_row_offsets_per_dev
            block_input_row_offsets_arg = block_offsets_per_dev_list
            block_embeds_arg = block_embeds_per_dev_local
        else:
            ctx_input_row_offsets_arg = merged_offsets_per_dev
            block_input_row_offsets_arg = block_offsets_per_dev
            block_embeds_arg = block_embeds_per_dev

        block_hs_per_dev = self.draft(
            block_embeds=block_embeds_arg,
            ctx_hidden=ctx_hidden,
            signal_buffers=signal_buffers,
            ctx_kv_collections=ctx_kv_collections,
            block_kv_collections=block_kv_collections,
            ctx_input_row_offsets=ctx_input_row_offsets_arg,
            block_input_row_offsets=block_input_row_offsets_arg,
        )

        hidden_size = self.config.draft.hidden_size
        block_hs_2d_per_dev: list[TensorValue] = []
        for hs in block_hs_per_dev:
            n = hs.shape[0]
            hs_rebound = hs.rebind([(n // K) * K, hidden_size])
            hs_3d = hs_rebound.reshape([n // K, K, hidden_size])
            block_hs_2d_per_dev.append(hs_3d[:, 1:, :])
        if is_dp:
            block_hs_2d_per_dev = ops.allgather(
                block_hs_2d_per_dev, signal_buffers, axis=0
            )
        draft_logits = self.target.lm_head(block_hs_2d_per_dev, signal_buffers)[
            0
        ]
        argmax_out = ops.argmax(draft_logits, axis=-1)
        argmax_out = argmax_out.rebind(["batch_size", K - 1, 1])
        next_draft_tokens = argmax_out.reshape(("batch_size", K - 1))

        num_accepted_out = ops.where(
            is_prefill,
            ops.constant(0, num_accepted.dtype, device=device0).broadcast_to(
                ["batch_size"]
            ),
            num_accepted,
        )

        return (num_accepted_out, next_tokens, next_draft_tokens)

    def input_types(
        self, kv_params: MultiKVCacheParams
    ) -> tuple[TensorType | BufferType, ...]:
        """Input types mirror :class:`Eagle3MHAKimiK25Unified.input_types`.

        ``kv_params`` is the unified ``{"target", "draft"}`` tree; the target
        leaf is MLA and the draft leaf is MHA, each carrying its own blocks
        and dispatch metadata. Distributed (DP + signals + EP) MHA-draft graph
        with vision (no in-thinking-phase) that appends the structured-output
        bitmask triple when ``enable_structured_output`` is set. See
        :func:`build_spec_decode_input_types` for the canonical ordering.
        """
        spec = SpecDecodeInputTypeSpec(
            distributed=True,
            data_parallel_degree=self.config.target.data_parallel_degree,
            enable_vision=True,
            vision_hidden_size=self.config.target.hidden_size,
            enable_structured_output=self.enable_structured_output,
        )
        ep_input_types = (
            self.target.ep_manager.input_types()
            if self.target.ep_manager is not None
            else ()
        )
        return build_spec_decode_input_types(
            spec,
            devices=self.config.target.devices,
            kv_params=kv_params,
            ep_input_types=ep_input_types,
        )
