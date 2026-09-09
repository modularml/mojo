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
"""DFlash draft module for a Kimi K2.5 (MLA) target."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    ops,
)
from max.nn.attention.attention_with_rope import (
    AttentionWithRope,
    DataParallelAttentionWithRope,
    TensorParallelAttentionWithRope,
)
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import LayerList, Module
from max.nn.linear import MLP, Linear
from max.nn.norm import RMSNorm
from max.nn.rotary_embedding import (
    DeepseekYarnRopeScalingParams,
    DeepseekYarnRotaryEmbedding,
)
from max.nn.transformer.distributed_transformer import forward_sharded_layers


@dataclass(kw_only=True)
class DFlashKimiK25DraftConfig:
    """Minimal config for a DFlash draft over a Kimi K2.5 (MLA) target.

    Held separate from ``DeepseekV3Config`` because the DFlash draft is
    MHA/GQA with its own KV geometry; the target's MLA-specific fields
    (``kv_lora_rank``, ``v_head_dim``, etc.) don't apply.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    devices: list[DeviceRef]
    data_parallel_degree: int
    dtype: DType
    norm_dtype: DType
    kv_params: KVCacheParams
    rope_scaling: dict[str, Any]
    target_layer_ids: list[int]

    residual_multiplier: float = 1.0
    attention_multiplier: float | None = None
    attention_bias: bool = False
    stacked_qkv: bool = False
    clip_qkv: float | None = None
    sliding_window: int | None = None
    use_qk_norm: bool = True


class _DFlashKimiK25Layer(Module):
    """One DFlash draft decoder layer with per-device sharded components."""

    def __init__(
        self,
        config: DFlashKimiK25DraftConfig,
        *,
        rope: DeepseekYarnRotaryEmbedding,
        use_tensor_parallel: bool,
        use_data_parallel_attention: bool,
    ) -> None:
        super().__init__()
        self._use_tensor_parallel = use_tensor_parallel
        self._use_data_parallel_attention = use_data_parallel_attention

        devices = config.devices
        num_devices = len(devices)
        dtype = config.dtype
        norm_dtype = config.norm_dtype

        attn_kwargs: dict[str, Any] = dict(
            rope=rope,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            hidden_size=config.hidden_size,
            kv_params=config.kv_params,
            devices=devices,
            dtype=dtype,
            stacked_qkv=config.stacked_qkv,
            scale=config.attention_multiplier,
            has_bias=config.attention_bias,
            clip_qkv=config.clip_qkv,
            use_qk_norm=config.use_qk_norm,
            rms_norm_eps=config.rms_norm_eps,
            mask_variant=(
                MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK
                if config.sliding_window is not None
                else MHAMaskVariant.NULL_MASK
            ),
            sliding_window=config.sliding_window,
        )
        self.self_attn: AttentionWithRope
        if use_data_parallel_attention:
            self.self_attn = DataParallelAttentionWithRope(**attn_kwargs)
        elif use_tensor_parallel:
            self.self_attn = TensorParallelAttentionWithRope(**attn_kwargs)
        else:
            self.self_attn = AttentionWithRope(**attn_kwargs)

        def _norm() -> RMSNorm:
            n = RMSNorm(
                config.hidden_size,
                norm_dtype,
                config.rms_norm_eps,
                multiply_before_cast=True,
            )
            n.sharding_strategy = ShardingStrategy.replicate(num_devices)
            return n

        self.input_layernorm = _norm()
        self.input_layernorm_shards = self.input_layernorm.shard(devices)

        self.post_attention_layernorm = _norm()
        self.post_attention_layernorm_shards = (
            self.post_attention_layernorm.shard(devices)
        )

        self.mlp = MLP(
            dtype=dtype,
            quantization_encoding=None,
            hidden_dim=config.hidden_size,
            feed_forward_length=config.intermediate_size,
            devices=devices,
            quant_config=None,
        )
        if use_tensor_parallel:
            self.mlp.sharding_strategy = ShardingStrategy.tensor_parallel(
                num_devices
            )
        else:
            self.mlp.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.mlp_shards = list(self.mlp.shard(devices))

    def materialize_kv(
        self,
        layer_idx: TensorValue,
        ctx_hidden: list[TensorValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
    ) -> None:
        """Write K/V derived from external ctx hidden states into the KV cache."""
        if self._use_data_parallel_attention:
            assert isinstance(self.self_attn, DataParallelAttentionWithRope)
            self.self_attn.materialize_kv_from_hidden(
                layer_idx=layer_idx,
                hiddens=ctx_hidden,
                kv_collections=kv_collections,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        elif self._use_tensor_parallel:
            assert isinstance(self.self_attn, TensorParallelAttentionWithRope)
            self.self_attn.materialize_kv_from_hidden(
                layer_idx=layer_idx,
                hiddens=ctx_hidden,
                kv_collections=kv_collections,
                freqs_cis=freqs_cis,
                input_row_offsets=input_row_offsets,
            )
        else:
            self.self_attn.materialize_kv_from_hidden(
                layer_idx=layer_idx,
                hidden=ctx_hidden[0],
                kv_collection=kv_collections[0],
                freqs_cis=freqs_cis[0],
                input_row_offsets=input_row_offsets[0],
            )

    def __call__(
        self,
        layer_idx: TensorValue,
        h: list[TensorValue],
        signal_buffers: list[BufferValue],
        kv_collections: list[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: list[TensorValue],
    ) -> list[TensorValue]:
        h_pre_attn = forward_sharded_layers(self.input_layernorm_shards, h)
        attn_outs: list[TensorValue]
        if self._use_data_parallel_attention:
            assert isinstance(self.self_attn, DataParallelAttentionWithRope)
            attn_outs = self.self_attn(
                layer_idx,
                h_pre_attn,
                kv_collections,
                freqs_cis,
                input_row_offsets,
            )
        elif self._use_tensor_parallel:
            assert isinstance(self.self_attn, TensorParallelAttentionWithRope)
            attn_outs = self.self_attn(
                layer_idx,
                h_pre_attn,
                signal_buffers,
                kv_collections,
                freqs_cis,
                input_row_offsets,
            )
        else:
            single_out = self.self_attn(
                layer_idx,
                h_pre_attn[0],
                kv_collections[0],
                freqs_cis[0],
                input_row_offsets[0],
            )
            attn_outs = [single_out]

        h = [hd + ao for hd, ao in zip(h, attn_outs, strict=True)]

        h_pre_mlp = forward_sharded_layers(
            self.post_attention_layernorm_shards, h
        )
        mlp_outs = forward_sharded_layers(self.mlp_shards, h_pre_mlp)
        if self._use_tensor_parallel:
            mlp_outs = ops.allreduce.sum(mlp_outs, signal_buffers)
        out = [hd + mo for hd, mo in zip(h, mlp_outs, strict=True)]
        return out


class DFlashKimiK25(Module):
    """DFlash draft transformer for a Kimi K2.5 (MLA) target."""

    def __init__(self, config: DFlashKimiK25DraftConfig) -> None:
        super().__init__()
        if config.num_hidden_layers <= 0:
            raise ValueError(
                "DFlashKimiK25 requires positive num_hidden_layers, got "
                f"{config.num_hidden_layers}."
            )
        if len(config.target_layer_ids) != config.num_hidden_layers:
            raise ValueError(
                "DFlashKimiK25 invariant: len(target_layer_ids) must equal "
                "num_hidden_layers (one captured target layer per draft "
                f"layer). Got target_layer_ids={config.target_layer_ids} "
                f"num_hidden_layers={config.num_hidden_layers}."
            )

        self.config = config
        devices = config.devices
        num_devices = len(devices)
        device0 = devices[0]
        dtype = config.dtype
        norm_dtype = config.norm_dtype

        self.use_tensor_parallel = (
            config.data_parallel_degree == 1 and num_devices > 1
        )
        self.use_data_parallel_attention = (
            num_devices > 1 and config.data_parallel_degree == num_devices
        )

        scaling_params = DeepseekYarnRopeScalingParams(
            scaling_factor=config.rope_scaling["factor"],
            original_max_position_embeddings=config.rope_scaling[
                "original_max_position_embeddings"
            ],
            beta_fast=config.rope_scaling["beta_fast"],
            beta_slow=config.rope_scaling["beta_slow"],
            mscale=config.rope_scaling["mscale"],
            mscale_all_dim=config.rope_scaling["mscale_all_dim"],
        )
        self.rope = DeepseekYarnRotaryEmbedding(
            config.head_dim,
            n_heads=config.num_attention_heads,
            theta=config.rope_theta,
            max_seq_len=config.max_position_embeddings,
            scaling_params=scaling_params,
            interleaved=False,
        )

        self.fc = Linear(
            in_dim=config.hidden_size * len(config.target_layer_ids),
            out_dim=config.hidden_size,
            dtype=dtype,
            device=device0,
            quantization_encoding=None,
            has_bias=False,
        )
        self.fc.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.fc_shards = self.fc.shard(devices)

        self.hidden_norm = RMSNorm(
            config.hidden_size,
            norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=True,
        )
        self.hidden_norm.sharding_strategy = ShardingStrategy.replicate(
            num_devices
        )
        self.hidden_norm_shards = self.hidden_norm.shard(devices)

        layers: list[_DFlashKimiK25Layer] = []
        for _ in range(config.num_hidden_layers):
            layers.append(
                _DFlashKimiK25Layer(
                    config,
                    rope=self.rope,
                    use_tensor_parallel=self.use_tensor_parallel,
                    use_data_parallel_attention=self.use_data_parallel_attention,
                )
            )
        self.layers = LayerList(layers)

        self.norm = RMSNorm(
            config.hidden_size,
            norm_dtype,
            config.rms_norm_eps,
            multiply_before_cast=True,
        )
        self.norm.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.norm_shards = self.norm.shard(devices)

        self.embed_tokens: Any = None
        self.lm_head: Any = None

    def project_target_hidden(
        self, target_hs_concat: Sequence[TensorValue]
    ) -> list[TensorValue]:
        if len(target_hs_concat) != len(self.config.devices):
            raise ValueError(
                f"Expected {len(self.config.devices)} per-device target "
                f"hidden tensors, got {len(target_hs_concat)}."
            )
        fc_outs = forward_sharded_layers(self.fc_shards, list(target_hs_concat))
        return forward_sharded_layers(self.hidden_norm_shards, fc_outs)

    def _freqs_cis_per_device(self) -> list[TensorValue]:
        return [
            self.rope.freqs_cis.to(device) for device in self.config.devices
        ]

    def materialize_kv(
        self,
        ctx_hidden: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        kv_collections: Sequence[PagedCacheValues],
    ) -> None:
        num_devices = len(self.config.devices)
        if not all(
            len(seq) == num_devices
            for seq in (ctx_hidden, input_row_offsets, kv_collections)
        ):
            raise ValueError(
                "All per-device inputs to materialize_kv must have length "
                f"equal to the number of devices ({num_devices})."
            )
        freqs_cis = self._freqs_cis_per_device()
        ctx_hidden_list = list(ctx_hidden)
        kv_list = list(kv_collections)
        offsets_list = list(input_row_offsets)
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, _DFlashKimiK25Layer)
            layer_idx_const = ops.constant(
                layer_idx, DType.uint32, device=DeviceRef.CPU()
            )
            layer.materialize_kv(
                layer_idx_const,
                ctx_hidden_list,
                kv_list,
                freqs_cis,
                offsets_list,
            )

    def forward_block(
        self,
        input_embeds: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        num_devices = len(self.config.devices)
        if not all(
            len(seq) == num_devices
            for seq in (input_embeds, kv_collections, input_row_offsets)
        ):
            raise ValueError(
                "All per-device inputs to forward_block must have length "
                f"equal to the number of devices ({num_devices})."
            )
        freqs_cis = self._freqs_cis_per_device()
        signal_buffers_list = list(signal_buffers)
        kv_list = list(kv_collections)
        offsets_list = list(input_row_offsets)
        h: list[TensorValue] = list(input_embeds)
        for layer_idx, layer in enumerate(self.layers):
            assert isinstance(layer, _DFlashKimiK25Layer)
            layer_idx_const = ops.constant(
                layer_idx, DType.uint32, device=DeviceRef.CPU()
            )
            h = layer(
                layer_idx_const,
                h,
                signal_buffers_list,
                kv_list,
                freqs_cis,
                offsets_list,
            )
        out = forward_sharded_layers(self.norm_shards, h)
        return out

    def __call__(
        self,
        block_embeds: Sequence[TensorValue],
        ctx_hidden: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        ctx_kv_collections: Sequence[PagedCacheValues],
        block_kv_collections: Sequence[PagedCacheValues],
        ctx_input_row_offsets: Sequence[TensorValue],
        block_input_row_offsets: Sequence[TensorValue],
    ) -> list[TensorValue]:
        self.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=ctx_input_row_offsets,
            kv_collections=ctx_kv_collections,
        )
        return self.forward_block(
            input_embeds=block_embeds,
            signal_buffers=signal_buffers,
            kv_collections=block_kv_collections,
            input_row_offsets=block_input_row_offsets,
        )
