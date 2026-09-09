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

"""Implements the Gemma4 Assistant (MTP draft) model."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.nn.attention import MHAMaskVariant, num_heads_for_device
from max.nn.comm.allreduce import Allreduce
from max.nn.kernels import cross_attention_ragged, rope_ragged
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module, Shardable
from max.nn.linear import MLP, ColumnParallelLinear, Linear
from max.nn.rotary_embedding import Llama3RotaryEmbedding, RotaryEmbedding
from max.nn.transformer import ReturnHiddenStates
from max.nn.transformer.distributed_transformer import (
    distributed_logits_postprocess,
    forward_sharded_layers,
)
from max.nn.transformer.transformer import extract_hs
from max.pipelines.architectures.gemma3.layers.scaled_word_embedding import (
    ScaledWordEmbedding,
)
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalRotaryEmbedding,
)

from .model_config import Gemma4AssistantConfig

# Map from layer type string to the index in the kv_by_type list.
_LAYER_TYPE_TO_KV_INDEX = {
    "sliding_attention": 0,
    "full_attention": 1,
}


class Gemma4AssistantAttention(Module, Shardable):
    """Q-only attention with cross-attention against the target model's KV cache.

    The assistant model has no K/V projection weights. It projects Q from its
    own hidden states, applies RoPE, then performs cross-attention against the
    target model's KV cache.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        head_dim: int,
        target_layer_idx_in_cache: int,
        is_sliding: bool,
        dtype: DType = DType.bfloat16,
        devices: list[DeviceRef],
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.head_dim = head_dim
        self.kv_params = kv_params
        self.target_layer_idx_in_cache = target_layer_idx_in_cache
        self.is_sliding = is_sliding
        self.devices = devices
        self.scale = 1.0
        self.local_window_size = local_window_size
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        self.q_weight_dim = head_dim * num_attention_heads

        self.q_proj = Linear(
            in_dim=hidden_size,
            out_dim=self.q_weight_dim,
            dtype=dtype,
            device=devices[0],
        )

        self.q_norm = Gemma4RMSNorm(head_dim, DType.bfloat16, qk_norm_eps)

        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
        )

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs: object,
    ) -> TensorValue:
        total_seq_len = x.shape[0]

        layer_idx = ops.constant(
            self.target_layer_idx_in_cache, DType.uint32, device=DeviceRef.CPU()
        )

        # Q projection
        x_q = self.q_proj(x)

        # Per-head Q norm
        x_q = self.q_norm(
            x_q.reshape((-1, self.n_heads, self.head_dim))
        ).reshape((-1, self.q_weight_dim))

        # Apply RoPE to Q using rope_ragged.
        # rope_ragged expects: input [total_seq, n_heads, head_dim],
        # input_row_offsets [batch+1], start_pos [batch] (cache_lengths).
        freqs_cis = ops.cast(self.rope.freqs_cis, x_q.dtype).to(x_q.device)
        xq = x_q.reshape((-1, self.n_heads, self.head_dim))

        input_row_offsets = kwargs["input_row_offsets"]
        assert isinstance(input_row_offsets, TensorValue)
        kv_input_row_offsets = kwargs["kv_input_row_offsets"]
        assert isinstance(kv_input_row_offsets, TensorValue)
        q_max_seq_len = kwargs["q_max_seq_len"]
        assert isinstance(q_max_seq_len, TensorValue)

        rope_cache_lengths = kwargs.get("rope_cache_lengths")
        if isinstance(rope_cache_lengths, TensorValue):
            rope_positions = rope_cache_lengths
        else:
            rope_positions = kv_collection.cache_lengths
        xq = rope_ragged(
            xq,
            input_row_offsets,
            rope_positions,
            freqs_cis,
            interleaved=self.rope.interleaved,
            output_dtype=self.kv_params.dtype,
        )

        # Cross-attention against target's KV cache.
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.is_sliding
            else MHAMaskVariant.CAUSAL_MASK
        )
        attn_out = cross_attention_ragged(
            self.kv_params,
            input=xq,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            mask_variant=mask_variant,
            kv_input_row_offsets=kv_input_row_offsets,
            q_max_seq_len=q_max_seq_len,
            scale=self.scale,
            local_window_size=self.local_window_size if self.is_sliding else -1,
            output_dtype=DType.bfloat16,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        if sharding_strategy.is_replicate:
            self.q_norm.sharding_strategy = sharding_strategy
            self.q_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy
        elif sharding_strategy.is_tensor_parallel:
            num_devices = sharding_strategy.num_devices
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.head_dim
                )
            )
        else:
            raise ValueError(
                "Gemma4AssistantAttention only supports tensor parallel "
                "and replicate sharding strategy"
            )
        self._sharding_strategy = sharding_strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4AssistantAttention]:
        if not self.sharding_strategy:
            raise ValueError(
                "Gemma4AssistantAttention layer cannot be sharded because "
                "no sharding strategy was provided."
            )

        q_proj_shards = self.q_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)
        q_norm_weight_shards = self.q_norm.weight.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            sharded_num_heads = num_heads_for_device(
                num_heads=self.n_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )

            sharded = Gemma4AssistantAttention(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                hidden_size=self.q_weight_dim,
                kv_params=self.kv_params,
                head_dim=self.head_dim,
                target_layer_idx_in_cache=self.target_layer_idx_in_cache,
                is_sliding=self.is_sliding,
                dtype=self.o_proj.weight.dtype,
                devices=[device],
                qk_norm_eps=self.qk_norm_eps,
                local_window_size=self.local_window_size,
            )

            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm.weight = q_norm_weight_shards[shard_idx]

            shards.append(sharded)

        return shards


class Gemma4AssistantDecoderLayer(Module):
    """Gemma4 assistant decoder layer: Attention + FeedForward with pre/post norms.

    Same structure as Gemma4TextDecoderLayer but uses Gemma4AssistantAttention
    (Q-only cross-attention) instead of full self-attention.
    """

    def __init__(
        self,
        attention: Gemma4AssistantAttention,
        mlp: MLP,
        hidden_size: int,
        rms_norm_eps: float,
        devices: list[DeviceRef],
        unquantized_dtype: DType = DType.bfloat16,
    ) -> None:
        super().__init__()

        self.self_attn = attention
        self.self_attn.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.self_attn_shards = attention.shard(devices)

        self.mlp = mlp
        self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
            len(devices)
        )
        self.mlp_shards = mlp.shard(devices)

        self.input_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.input_layernorm.sharding_strategy = ShardingStrategy.replicate(
            len(devices)
        )
        self.input_layernorm_shards = self.input_layernorm.shard(devices)

        self.post_attention_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.post_attention_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(devices)
        )

        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.pre_feedforward_layernorm_shards = (
            self.pre_feedforward_layernorm.shard(devices)
        )

        self.post_feedforward_layernorm = Gemma4RMSNorm(
            hidden_size, unquantized_dtype, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm.sharding_strategy = (
            ShardingStrategy.replicate(len(devices))
        )
        self.post_feedforward_layernorm_shards = (
            self.post_feedforward_layernorm.shard(devices)
        )

        self.devices = devices
        self.allreduce = Allreduce(num_accelerators=len(devices))

        self.layer_scalar = Weight(
            "layer_scalar", unquantized_dtype, shape=[1], device=DeviceRef.CPU()
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        xs: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        input_row_offsets: list[TensorValue],
        kv_input_row_offsets: list[TensorValue],
        q_max_seq_len: TensorValue,
        **kwargs: object,
    ) -> list[TensorValue]:
        residual = xs
        norm_xs = forward_sharded_layers(self.input_layernorm_shards, xs)
        rope_cache_lengths_list = kwargs.get("rope_cache_lengths")
        attn_out = [
            shard(
                norm_xs[i],
                kv_collections[i],
                input_row_offsets=input_row_offsets[i],
                kv_input_row_offsets=kv_input_row_offsets[i],
                q_max_seq_len=q_max_seq_len,
                **(
                    {"rope_cache_lengths": rope_cache_lengths_list[i]}  # type: ignore
                    if rope_cache_lengths_list is not None
                    else {}
                ),
            )
            for i, shard in enumerate(self.self_attn_shards)
        ]
        attn_out = self.allreduce(attn_out, signal_buffers)

        hidden_states = forward_sharded_layers(
            self.post_attention_layernorm_shards, attn_out
        )
        hidden_states = [
            residual[i] + hidden_states[i] for i in range(len(hidden_states))
        ]

        residual = hidden_states

        norm_xs = forward_sharded_layers(
            self.pre_feedforward_layernorm_shards, hidden_states
        )
        hidden_states = forward_sharded_layers(self.mlp_shards, norm_xs)
        hidden_states = self.allreduce(hidden_states, signal_buffers)

        hidden_states = forward_sharded_layers(
            self.post_feedforward_layernorm_shards, hidden_states
        )
        hidden_states = [
            residual[i] + hidden_states[i] for i in range(len(hidden_states))
        ]

        hidden_states = [
            h * self.layer_scalar.to(h.device) for h in hidden_states
        ]

        return hidden_states


class Gemma4Assistant(Module):
    """The Gemma4 Assistant (MTP draft) model.

    Takes token IDs and hidden states from the backbone model, projects them
    into the assistant's hidden dimension, runs through a small stack of decoder
    layers with cross-attention against the target's KV cache, then projects
    back to the backbone dimension for logits computation.
    """

    def __init__(
        self,
        config: Gemma4AssistantConfig,
        target_layer_types: list[str],
        target_sliding_kv_params: KVCacheParams,
        target_global_kv_params: KVCacheParams,
    ) -> None:
        super().__init__()
        self.devices = config.devices

        # Build per-layer-type rotary embeddings.
        rope_sliding = Llama3RotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.sliding_window_rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.head_dim,
            interleaved=False,
            scaling_params=None,
        )

        rope_global = ProportionalRotaryEmbedding(
            dim=config.hidden_size,
            n_heads=config.num_attention_heads,
            theta=config.global_rope_theta,
            max_seq_len=config.max_position_embeddings,
            head_dim=config.global_head_dim,
            interleaved=False,
            scaling_params=config.global_rope_scaling,
        )

        # Target embedding (backbone_hidden_size) for the concat input.
        # Will be aliased to the target model's embed_tokens at load time.
        self.embed_tokens = ScaledWordEmbedding(
            config.vocab_size,
            config.backbone_hidden_size,
            config.dtype,
            config.devices,
            embed_scale=config.backbone_hidden_size**0.5,
        )

        # Assistant's own embedding (hidden_size=1024) for tied lm_head.
        # Loaded from the assistant checkpoint (model.embed_tokens.weight).
        self.draft_embed_tokens = ScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.dtype,
            config.devices,
            embed_scale=config.hidden_size**0.5,
        )

        # lm_head tied to the assistant's 1024-dim embedding (NOT the
        # target's 5376-dim embedding).  HF applies lm_head to the
        # 1024-dim norm output, not the 5376-dim post_projection output.
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            dtype=config.dtype,
            devices=config.devices,
            tied_weight=self.draft_embed_tokens.weight,
        )

        # Pre-projection: concat(embed, hidden_states) -> assistant dim.
        self.pre_projection = Linear(
            in_dim=config.backbone_hidden_size * 2,
            out_dim=config.hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )
        self.pre_projection.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.pre_projection_shards = self.pre_projection.shard(config.devices)

        # Post-projection: assistant dim -> backbone dim.
        self.post_projection = Linear(
            in_dim=config.hidden_size,
            out_dim=config.backbone_hidden_size,
            dtype=config.dtype,
            device=config.devices[0],
        )
        self.post_projection.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self.post_projection_shards = self.post_projection.shard(config.devices)

        # Compute per-layer mapping from assistant layer type to the target's
        # KV cache layer index within that cache type.
        # For each unique layer_type in the TARGET, we need to know which
        # target layer index (within the sub-cache) to cross-attend against.
        # The assistant's num_kv_shared_layers layers reuse the LAST
        # num_kv_shared_layers target layers' KV caches.
        target_layer_type_counts: dict[str, int] = {
            "sliding_attention": 0,
            "full_attention": 0,
        }
        for lt in target_layer_types:
            target_layer_type_counts[lt] += 1

        # The assistant reuses the last `num_kv_shared_layers` target KV layers.
        # Compute the target cache indices to cross-attend against.
        # For the assistant model with layer_types
        # [sliding, sliding, sliding, full], we map:
        #   assistant layer 0 (sliding) -> last sliding target layer
        #   assistant layer 1 (sliding) -> last sliding target layer
        #   assistant layer 2 (sliding) -> last sliding target layer
        #   assistant layer 3 (full) -> last full target layer
        # (All assistant layers of the same type share the same target KV layer.)
        self._target_kv_layer_idx: dict[str, int] = {}
        for lt, count in target_layer_type_counts.items():
            # Use the last target layer of this type.
            self._target_kv_layer_idx[lt] = count - 1

        # Per-layer KV params from target, indexed by layer type.
        kv_params_by_type: dict[str, KVCacheParams] = {
            "sliding_attention": target_sliding_kv_params,
            "full_attention": target_global_kv_params,
        }

        # Build decoder layers.
        layers = []
        self._layer_kv_index = []
        for i in range(config.num_hidden_layers):
            layer_type = config.layer_types[i]
            is_sliding = layer_type == "sliding_attention"
            kv_params = kv_params_by_type[layer_type]
            head_dim = config.head_dim if is_sliding else config.global_head_dim

            layers.append(
                Gemma4AssistantDecoderLayer(
                    attention=Gemma4AssistantAttention(
                        rope=rope_sliding if is_sliding else rope_global,
                        num_attention_heads=config.num_attention_heads,
                        hidden_size=config.hidden_size,
                        kv_params=kv_params,
                        head_dim=head_dim,
                        target_layer_idx_in_cache=(
                            self._target_kv_layer_idx[layer_type]
                        ),
                        is_sliding=is_sliding,
                        dtype=config.dtype,
                        devices=config.devices,
                        qk_norm_eps=config.rms_norm_eps,
                        local_window_size=config.sliding_window,
                    ),
                    mlp=MLP(
                        dtype=config.dtype,
                        quantization_encoding=None,
                        hidden_dim=config.hidden_size,
                        feed_forward_length=config.intermediate_size,
                        devices=config.devices,
                        activation_function=config.hidden_activation,
                    ),
                    hidden_size=config.hidden_size,
                    rms_norm_eps=config.rms_norm_eps,
                    devices=config.devices,
                    unquantized_dtype=config.dtype,
                )
            )
            self._layer_kv_index.append(_LAYER_TYPE_TO_KV_INDEX[layer_type])

        self.layers = LayerList(layers)

        # Final norm on assistant hidden dim (applied before post_projection).
        self.norm = Gemma4RMSNorm(
            config.hidden_size, config.dtype, config.rms_norm_eps
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(
            len(config.devices)
        )
        self._norm_shards = self.norm.shard(config.devices)

        self.return_logits = config.return_logits
        self.return_hidden_states = config.return_hidden_states

    def __call__(
        self,
        tokens: TensorValue,
        hidden_states: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        target_sliding_kv: Sequence[PagedCacheValues],
        target_global_kv: Sequence[PagedCacheValues],
        return_n_logits: TensorValue,
        input_row_offsets: Sequence[TensorValue],
        kv_input_row_offsets: Sequence[TensorValue],
        q_max_seq_len: TensorValue,
        rope_cache_lengths: Sequence[TensorValue] | None = None,
    ) -> tuple[TensorValue, ...]:
        n_devs = len(self.devices)

        kv_by_type: list[Sequence[PagedCacheValues]] = [
            target_sliding_kv,
            target_global_kv,
        ]

        # Embed tokens (backbone dim).
        h_embed = self.embed_tokens(tokens, signal_buffers)

        # Align batch dims: token embed has 'batch_size' but
        # hidden_states may have a per-step dim from the draft loop.
        h_concat = [
            ops.concat(
                [h_embed[i].rebind(hidden_states[i].shape), hidden_states[i]],
                axis=-1,
            )
            for i in range(n_devs)
        ]

        # Project to assistant hidden dim.
        h = forward_sharded_layers(self.pre_projection_shards, h_concat)

        # Run through decoder layers with cross-attention.
        for idx, layer in enumerate(self.layers):
            layer_idx_tensor = ops.constant(
                idx, DType.uint32, device=self.devices[0]
            )
            kv_collections = list(kv_by_type[self._layer_kv_index[idx]])
            h = layer(
                layer_idx_tensor,
                list(h),
                list(signal_buffers),
                kv_collections,
                input_row_offsets=list(input_row_offsets),
                kv_input_row_offsets=list(kv_input_row_offsets),
                q_max_seq_len=q_max_seq_len,
                rope_cache_lengths=list(rope_cache_lengths)
                if rope_cache_lengths is not None
                else None,
            )

        # Apply assistant-dim final norm.
        h = forward_sharded_layers(self._norm_shards, h)

        # HF applies lm_head to the 1024-dim norm output (before
        # post_projection), using the assistant's own tied embedding.
        # The 5376-dim post_projection output is returned as hidden
        # states for the next MTP step, NOT used for logits.
        h_out = forward_sharded_layers(self.post_projection_shards, h)

        # Compute logits from 1024-dim h.
        logits_result = distributed_logits_postprocess(
            h,
            input_row_offsets,
            return_n_logits,
            lm_head=self.lm_head,
            signal_buffers=signal_buffers,
            return_logits=self.return_logits,
            device=self.devices[0],
            return_hidden_states=ReturnHiddenStates.NONE,
        )

        # Append 5376-dim hidden states (from post_projection) to the
        # output tuple so callers can feed them to the next MTP step.
        hs_tuple = extract_hs(
            return_hidden_states=self.return_hidden_states,
            last_token_hs_distributed=[
                ops.gather(h_out_dev, offsets[1:] - 1, axis=0)
                for h_out_dev, offsets in zip(
                    h_out, input_row_offsets, strict=True
                )
            ],
            all_hs_distributed=h_out,
        )

        return logits_result + hs_tuple
