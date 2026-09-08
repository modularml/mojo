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

"""Gemma3 Attention Layer."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.attention import MHAMaskVariant, num_heads_for_device
from max.nn.kernels import (
    flash_attention_ragged,
    rope_split_store_ragged,
)
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    PagedCacheValues,
)
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import Llama3RotaryEmbedding
from max.nn.stacked_linear import StackedLinear
from max.pipelines.architectures.gemma3.layers.rms_norm import Gemma3RMSNorm


class Gemma3Attention(Module, Shardable):
    """Implementation of the attention layer for the Gemma3 text model."""

    def __init__(
        self,
        *,
        rope_global: Llama3RotaryEmbedding,
        rope_local: Llama3RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        is_sliding: bool,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
        quant_config: QuantConfig | None = None,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope_global: Rotary embedding used for global (non-sliding window)
                attention layers.
            rope_local: Rotary embedding used for sliding window attention
                layers.
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            layer_idx: The layer number associated with this Attention block.
            dtype: DType of the attention inputs and weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
            quant_config: Scaled quantization configuration. Defaults to None.
        """

        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.is_sliding = is_sliding
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self._sharding_strategy: ShardingStrategy | None = None
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / self.kv_params.head_dim)
        )
        self.local_window_size = local_window_size
        self.qk_norm_eps = qk_norm_eps
        self.quant_config = quant_config

        self.q_norm = Gemma3RMSNorm(
            self.kv_params.head_dim, DType.bfloat16, self.qk_norm_eps
        )
        self.k_norm = Gemma3RMSNorm(
            self.kv_params.head_dim, DType.bfloat16, self.qk_norm_eps
        )
        self.q_weight_dim = self.kv_params.head_dim * num_attention_heads
        self.kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.qkv_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=[
                self.q_weight_dim,
                self.kv_weight_dim,
                self.kv_weight_dim,
            ],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=dtype,
            device=devices[0],
            stacked=False,
            has_bias=has_bias,
            linear_cls=linear_cls,
            quant_config=quant_config,
        )

        self.o_proj = linear_cls(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            quant_config=quant_config,
        )

    def __call__(
        self,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        **kwargs,
    ) -> TensorValue:
        # Get attributes from input.
        total_seq_len = x.shape[0]

        layer_idx = ops.constant(
            self.layer_idx, DType.uint32, device=DeviceRef.CPU()
        )

        head_dim = self.kv_params.head_dim
        q_dim = self.q_weight_dim
        kv_dim = self.kv_weight_dim
        num_kv_heads = kv_dim // head_dim

        # QKV projection
        qkv = self.qkv_proj(x)

        # Split into Q, K, V
        x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)

        # Per-head QK norm
        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )

        # Re-concat and apply RoPE + KV cache store
        qkv = ops.concat((x_q, x_k, x_v), axis=-1)

        use_local = self.is_sliding
        rope = self.rope_local if use_local else self.rope_global

        freqs_cis = ops.cast(rope.freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            self.kv_params,
            qkv,
            kwargs["input_row_offsets"],
            freqs_cis,
            kv_collection,
            layer_idx,
            n_heads=self.n_heads,
            interleaved=rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Calculate Flash Attention.
        #
        # For multimodal models, global (non-sliding) attention layers use
        # bidirectional masking (NULL_MASK) when image tokens are present
        # during prefill.  This matches the HuggingFace reference where
        # image tokens attend to each other without causal constraints.
        # For a single decode token NULL_MASK and CAUSAL_MASK are
        # equivalent, so this is safe to apply unconditionally at runtime
        # via ops.cond.
        image_token_indices = kwargs.get("image_token_indices")

        if use_local:
            # Local (sliding-window) layers always use the sliding mask.
            attn_out = flash_attention_ragged(
                self.kv_params,
                input=xq,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=kwargs["input_row_offsets"],
                mask_variant=MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK,
                scale=self.scale,
                local_window_size=self.local_window_size,
            )
        elif image_token_indices is not None:
            # Global layer in a multimodal model: switch mask at runtime.
            # shape_to_tensor gives us the runtime dimension as a tensor so
            # we can branch on it with ops.cond.
            num_image_tokens = ops.shape_to_tensor(image_token_indices.shape)
            has_images = num_image_tokens[0] > ops.constant(
                0, DType.int64, device=DeviceRef.CPU()
            )

            out_type = TensorType(
                dtype=xq.dtype,
                shape=xq.shape,
                device=xq.device,
            )

            def _bidirectional_attn() -> TensorValue:
                return flash_attention_ragged(
                    self.kv_params,
                    input=xq,
                    kv_collection=kv_collection,
                    layer_idx=layer_idx,
                    input_row_offsets=kwargs["input_row_offsets"],
                    mask_variant=MHAMaskVariant.NULL_MASK,
                    scale=self.scale,
                    local_window_size=self.local_window_size,
                )

            def _causal_attn() -> TensorValue:
                return flash_attention_ragged(
                    self.kv_params,
                    input=xq,
                    kv_collection=kv_collection,
                    layer_idx=layer_idx,
                    input_row_offsets=kwargs["input_row_offsets"],
                    mask_variant=MHAMaskVariant.CAUSAL_MASK,
                    scale=self.scale,
                    local_window_size=self.local_window_size,
                )

            attn_out = ops.cond(
                has_images,
                [out_type],
                _bidirectional_attn,
                _causal_attn,
            )[0].tensor
        else:
            # Text-only model: standard causal mask.
            attn_out = flash_attention_ragged(
                self.kv_params,
                input=xq,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=kwargs["input_row_offsets"],
                mask_variant=MHAMaskVariant.CAUSAL_MASK,
                scale=self.scale,
                local_window_size=self.local_window_size,
            )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        ret = self.o_proj(attn_out)
        return ret

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        num_devices = sharding_strategy.num_devices

        if sharding_strategy.is_replicate:
            self.q_norm.sharding_strategy = sharding_strategy
            self.k_norm.sharding_strategy = sharding_strategy
            self.qkv_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )

            self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )

        else:
            raise ValueError(
                "Gemma3Attention only supports tensor parallel and replicate sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Gemma3Attention]:
        """Creates sharded views of this attention layer across multiple devices.

        Overrides the parent method to handle QK normalization layers.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3Attention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Gemma3Attention layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        qkv_proj_shards = self.qkv_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        # Shard QK normalization weights
        q_norm_weight_shards = self.q_norm.weight.shard(devices)
        k_norm_weight_shards = self.k_norm.weight.shard(devices)

        shards = []
        for shard_idx, device in enumerate(devices):
            # Calculate sharded dimensions - handle uneven head distribution
            sharded_num_heads = num_heads_for_device(
                num_heads=self.n_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )
            assert isinstance(self.kv_params, MHAKVCacheParams)
            sharded_num_kv_heads = num_heads_for_device(
                num_heads=self.kv_params.n_kv_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )

            # Create new attention instance with sharded configuration
            sharded = Gemma3Attention(
                rope_global=self.rope_global,
                rope_local=self.rope_local,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.q_weight_dim + self.kv_weight_dim * 2,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                is_sliding=self.is_sliding,
                dtype=self.o_proj.weight.dtype,
                devices=[device],
                linear_cls=self.o_proj.__class__,
                scale=self.scale,
                has_bias=self.has_bias,
                qk_norm_eps=self.qk_norm_eps,
                local_window_size=self.local_window_size,
                quant_config=self.quant_config,
            )

            # Assign sharded weights
            sharded.qkv_proj = qkv_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Assign QK normalization weights
            sharded.q_norm.weight = q_norm_weight_shards[shard_idx]
            sharded.k_norm.weight = k_norm_weight_shards[shard_idx]

            shards.append(sharded)

        return shards
