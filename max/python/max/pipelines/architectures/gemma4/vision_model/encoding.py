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
from __future__ import annotations

from collections.abc import Iterable

from max.graph import DeviceRef, ShardingStrategy, TensorValue
from max.nn import MLP
from max.nn.layer import LayerList, Module

from ..layers.rms_norm import Gemma4RMSNorm
from ..model_config import Gemma4ForConditionalGenerationConfig
from .attention import Gemma4VisionAttention


class Gemma4VisionEncoderLayer(Module):
    """One transformer block of the Gemma4 SigLIP vision encoder.

    This is functionally the same as Gemma4TextDecoderLayer, but:
    - only full attention layers.
    - no layer scalar.
    - no MOE block.
    """

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        vision_config = config.vision_config
        assert vision_config is not None
        vision_dtype = config.unquantized_dtype

        self.device = device if device is not None else config.devices[0]
        self.hidden_size = vision_config.hidden_size
        self.layer_idx = layer_idx

        self.input_layernorm = Gemma4RMSNorm(
            self.hidden_size, vision_dtype, eps=vision_config.rms_norm_eps
        )
        self.self_attn = Gemma4VisionAttention(
            config=config,
            layer_idx=layer_idx,
            device=self.device,
        )
        self.post_attention_layernorm = Gemma4RMSNorm(
            self.hidden_size, vision_dtype, eps=vision_config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = Gemma4RMSNorm(
            self.hidden_size, vision_dtype, eps=vision_config.rms_norm_eps
        )
        self.mlp = MLP(
            dtype=vision_dtype,
            quantization_encoding=None,
            hidden_dim=self.hidden_size,
            feed_forward_length=vision_config.intermediate_size,
            devices=[self.device],
            activation_function=vision_config.hidden_activation,
        )
        self.post_feedforward_layernorm = Gemma4RMSNorm(
            self.hidden_size, vision_dtype, eps=vision_config.rms_norm_eps
        )

        self.hidden_size_per_layer_input = getattr(
            config, "hidden_size_per_layer_input", 0
        )
        self.enable_moe_block = getattr(config, "enable_moe_block", False)
        if self.hidden_size_per_layer_input > 0:
            raise NotImplementedError(
                "hidden_size_per_layer_input is not supported"
            )
        if self.enable_moe_block:
            raise NotImplementedError("enable_moe_block is not supported")

    def __call__(
        self,
        hidden_states: TensorValue,
        freqs_cis: TensorValue,
        cu_seqlens: TensorValue,
        max_seq_len: TensorValue,
    ) -> TensorValue:
        """Process packed patch embeddings through attention and MLP.

        Args:
            hidden_states: Packed patch embeddings,
                shape ``[total_patches, hidden_size]``.
            freqs_cis: Pre-computed vision RoPE frequencies,
                shape ``[total_patches, head_dim // 2, 2]``.
            cu_seqlens: Cumulative sequence lengths (image boundaries),
                shape ``[num_images + 1]``, dtype uint32.
            max_seq_len: Maximum patches per image (scalar uint32, on CPU).

        Returns:
            Output embeddings, shape ``[total_patches, hidden_size]``.
        """
        # Attention sub-block with residual.
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states, freqs_cis, cu_seqlens, max_seq_len
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP sub-block with residual.
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.self_attn.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        self.input_layernorm.sharding_strategy = strategy
        self.self_attn.sharding_strategy = strategy
        self.post_attention_layernorm.sharding_strategy = strategy
        self.pre_feedforward_layernorm.sharding_strategy = strategy
        self.mlp.sharding_strategy = strategy
        self.post_feedforward_layernorm.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4VisionEncoderLayer]:
        assert self.sharding_strategy

        self_attn_shards = self.self_attn.shard(devices)
        mlp_shards = self.mlp.shard(devices)

        input_layernorm_shards = self.input_layernorm.shard(devices)
        post_attention_layernorm_shards = self.post_attention_layernorm.shard(
            devices
        )
        pre_feedforward_layernorm_shards = self.pre_feedforward_layernorm.shard(
            devices
        )
        post_feedforward_layernorm_shards = (
            self.post_feedforward_layernorm.shard(devices)
        )

        shards = []
        for (
            device,
            attn_shard,
            mlp_shard,
            input_norm_shard,
            post_attn_norm_shard,
            pre_ffn_norm_shard,
            post_ffn_norm_shard,
        ) in zip(
            devices,
            self_attn_shards,
            mlp_shards,
            input_layernorm_shards,
            post_attention_layernorm_shards,
            pre_feedforward_layernorm_shards,
            post_feedforward_layernorm_shards,
            strict=True,
        ):
            sharded = Gemma4VisionEncoderLayer(
                self.config, self.layer_idx, device
            )

            sharded.self_attn = attn_shard
            sharded.mlp = mlp_shard
            sharded.input_layernorm = input_norm_shard
            sharded.post_attention_layernorm = post_attn_norm_shard
            sharded.pre_feedforward_layernorm = pre_ffn_norm_shard
            sharded.post_feedforward_layernorm = post_ffn_norm_shard

            shards.append(sharded)

        return shards


class Gemma4VisionEncoder(Module):
    """Stack of ``Gemma4VisionEncoderLayer`` blocks for the SigLIP encoder."""

    def __init__(self, config: Gemma4ForConditionalGenerationConfig) -> None:
        super().__init__()
        self.config = config
        assert config.vision_config is not None
        self._sharding_strategy: ShardingStrategy | None = None
        encoder_layers = [
            Gemma4VisionEncoderLayer(config, layer_idx)
            for layer_idx in range(config.vision_config.num_hidden_layers)
        ]

        self.layers = LayerList(encoder_layers)

    def __call__(
        self,
        hidden_state: TensorValue,
        freqs_cis: TensorValue,
        cu_seqlen: TensorValue,
        max_seq_len: TensorValue,
    ) -> TensorValue:
        """Process hidden states through all encoder layers.

        Args:
            hidden_states: Packed patch embeddings, shape
                ``[total_patches, hidden_size]``.
            freqs_cis: Pre-computed vision RoPE frequencies, shape
                ``[total_patches, head_dim // 2, 2]``.
            cu_seqlens: Cumulative sequence lengths, shape
                ``[num_images + 1]``, dtype uint32.
            max_seq_len: Maximum patches per image (scalar uint32, on CPU).

        Returns:
            Encoded patch embeddings with the same shape as ``hidden_states``.
        """
        for layer in self.layers:
            hidden_state = layer(
                hidden_state, freqs_cis, cu_seqlen, max_seq_len
            )
        return hidden_state

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        for layer in self.layers:
            layer.sharding_strategy = strategy

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Gemma4VisionEncoder]:
        assert self.sharding_strategy

        layer_shards = [layer.shard(devices) for layer in self.layers]

        shards = []
        for shard_idx, _ in enumerate(devices):
            sharded = Gemma4VisionEncoder(self.config)
            sharded.layers = LayerList(
                [
                    layer_shards[layer_idx][shard_idx]
                    for layer_idx in range(len(self.layers))
                ]
            )

            shards.append(sharded)

        return shards
