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

"""Qwen3.5 multi-token-prediction (MTP) draft head.

The checkpoint's 15 ``mtp.*`` tensors are a DeepSeek-NextN-shaped draft head:
two pre-norms, a fusion projection, one **full-attention** decoder layer, and
a final norm. There is no ``linear_attn`` and therefore no recurrent state --
the draft needs an ordinary KV slot and nothing else.

Two layout facts silently produce a low-acceptance draft rather than a crash
if they are reversed, so both are asserted by construction here:

- ``fc.weight`` is ``[hidden, 2 * hidden]`` and is **embedding-first**:
  columns ``[0:hidden]`` multiply the normed embedding and
  ``[hidden:2*hidden]`` the normed target hidden state.
- Every ``mtp.*`` norm uses the ``(1 + w)`` convention, like the rest of the
  language model (``linear_attn.norm`` is the sole exception in Qwen3.5, and
  the draft has none).

The embedding table and ``lm_head`` are shared with the target
(``mtp_use_dedicated_embeddings: false``; the checkpoint ships neither
``mtp.embed_tokens`` nor ``mtp.lm_head``). ``embed_tokens`` is injected by the
caller so the two graphs reference one weight; ``lm_head`` stays with the
caller because the draft's output projection is the target's **NVFP4** head
while the draft body is BF16.
"""

from __future__ import annotations

from collections.abc import Callable

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.embedding import VocabParallelEmbedding
from max.nn.kv_cache import PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.transformer.transformer import forward_sharded_layers

from .model_config import Qwen3_5Config
from .qwen3_5 import Qwen3_5FullAttentionBlock


class Qwen3_5MTP(Module):
    """The single-layer NextN draft head that ships in the checkpoint.

    ``__call__`` returns the draft's final *normed* hidden state per device,
    not logits: the output projection is the target's shared ``lm_head``, and
    the fused speculative graph applies it only at the positions it needs
    rather than over the whole draft prefill.
    """

    def __init__(
        self,
        config: Qwen3_5Config,
        embed_tokens: VocabParallelEmbedding,
        rope: Llama3RotaryEmbedding,
        create_norm: Callable[..., RMSNorm],
        kv_layer_idx: int,
    ) -> None:
        """Builds the draft head against an already-built target.

        Args:
            config: The target's config; the draft reuses its hidden size,
                head geometry and ``compute_dtype``.
            embed_tokens: The target's embedding table, shared not copied.
            rope: The target's rotary embedding, shared not copied.
            create_norm: The target's norm factory, which carries the
                ``(1 + w)`` offset and the norm dtype.
            kv_layer_idx: The draft layer's index in the KV cache it writes.
                The draft owns its own single-layer cache group, so this is 0
                unless the caller appends the draft to the target's cache.
        """
        super().__init__()
        self.config = config
        self.devices = config.devices
        num_devices = len(config.devices)
        replicate = ShardingStrategy.replicate(num_devices)
        compute_dtype = config.compute_dtype
        self.kv_layer_idx = kv_layer_idx

        self.embed_tokens = embed_tokens
        self.rope = rope

        self.pre_fc_norm_embedding = create_norm()
        self.pre_fc_norm_embedding.sharding_strategy = replicate
        self.pre_fc_norm_embedding_shards = self.pre_fc_norm_embedding.shard(
            config.devices
        )

        self.pre_fc_norm_hidden = create_norm()
        self.pre_fc_norm_hidden.sharding_strategy = replicate
        self.pre_fc_norm_hidden_shards = self.pre_fc_norm_hidden.shard(
            config.devices
        )

        # BF16 in the checkpoint even when the MLPs are NVFP4, so this takes
        # `compute_dtype` and no quant config. `config.dtype` is `uint8` for
        # an NVFP4 export and would misread every byte.
        self.fc = Linear(
            in_dim=2 * config.hidden_size,
            out_dim=config.hidden_size,
            dtype=compute_dtype,
            device=config.devices[0],
            has_bias=False,
        )
        self.fc.sharding_strategy = replicate
        self.fc_shards = self.fc.shard(config.devices)

        # The draft body is unquantized here, so it takes neither config.
        self.layers = LayerList(
            [
                Qwen3_5FullAttentionBlock(
                    config=config,
                    layer_idx=0,
                    rope=rope,
                    create_norm=create_norm,
                    linear_cls=Linear,
                    attn_quant_config=None,
                    mlp_quant_config=None,
                )
            ]
        )

        self.norm = create_norm()
        self.norm.sharding_strategy = replicate
        self.norm_shards = self.norm.shard(config.devices)

    def __call__(
        self,
        tokens: TensorValue,
        hidden_states: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        """Runs the draft head over a ragged batch.

        Args:
            tokens: ``[total_seq_len]`` int64 next-token ids -- for a draft
                step these are the tokens shifted one position left of the
                target's, so position ``t`` predicts ``t + 2``.
            hidden_states: Per-device target hidden states,
                ``[total_seq_len, hidden_size]``, aligned with ``tokens``.
            signal_buffers: Allreduce signal buffers.
            kv_collections: Per-device KV handles for the draft's own cache.
            input_row_offsets: Per-device ragged row offsets.

        Returns:
            Per-device ``[total_seq_len, hidden_size]`` final normed hidden
            states, ready for the shared ``lm_head``.
        """
        embeds = self.embed_tokens(tokens, signal_buffers)

        normed_embeds = forward_sharded_layers(
            self.pre_fc_norm_embedding_shards, embeds
        )
        normed_hidden = forward_sharded_layers(
            self.pre_fc_norm_hidden_shards, list(hidden_states)
        )

        # Embedding first: `fc.weight[:, :hidden]` multiplies the embedding.
        # Reversing this compiles, runs, and drafts badly.
        fused = [
            ops.concat([embed, hidden], axis=-1)
            for embed, hidden in zip(normed_embeds, normed_hidden, strict=True)
        ]
        hs = forward_sharded_layers(self.fc_shards, fused)

        freqs_cis = [self.rope.freqs_cis.to(device) for device in self.devices]
        layer_idx = ops.constant(
            self.kv_layer_idx, DType.uint32, device=DeviceRef.CPU()
        )
        for kv in kv_collections:
            assert kv.attention_dispatch_metadata is not None

        hs = self.layers[0](
            hs,
            layer_idx,
            signal_buffers,
            [kv.kv_blocks for kv in kv_collections],
            [kv.cache_lengths for kv in kv_collections],
            [kv.lookup_table for kv in kv_collections],
            [kv.max_prompt_length for kv in kv_collections],
            [kv.max_cache_length for kv in kv_collections],
            [kv.attention_dispatch_metadata for kv in kv_collections],
            freqs_cis,
            input_row_offsets,
        )

        return forward_sharded_layers(self.norm_shards, hs)
