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

"""GptOss Attention Layer."""

from __future__ import annotations

import math
from collections.abc import Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
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
from max.nn.rotary_embedding import YarnRotaryEmbedding
from max.nn.stacked_linear import StackedLinear


class GptOssAttention(Module, Shardable):
    """Implementation of the distributed attention layer for the GptOss text model.

    Depending on the layer type, the attention layer can be either a full attention
    layer or a sliding window attention layer. This layer generates the attention mask
    based on the layer type.

    This layer also supports sink attention, which is a technique to improve the
    attention mechanism by adding an extra logit column that acts as an attention
    sink.
    """

    def __init__(
        self,
        *,
        rope: YarnRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        layer_type: str = "full_attention",
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        scale: float | None = None,
        has_bias: bool = False,
        local_window_size: int = 1024,
    ) -> None:
        """Initializes the attention layer.

        Args:
            rope: Rotary embedding used for all attention layers (full + sliding window).
            num_attention_heads: The number of attention heads.
            num_key_value_heads: The number of key/value heads.
            hidden_size: The dimension of the hidden states.
            kv_params: KV Cache Params, including the number of kv heads, the
                head dim, and data type.
            layer_idx: Index of this layer within its KV group (not the
                global decoder index).
            dtype: DType of the attention inputs and weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            scale: Value used to scale the results of the attention output.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
        """

        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx
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
        self.layer_type = layer_type

        # Initialize sinks parameter for each attention head
        self.sinks = Weight(
            name="sinks",
            dtype=dtype,
            shape=[num_attention_heads],
            device=devices[0],
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
        )

        self.o_proj = Linear(
            in_dim=self.q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=self.has_bias,
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

        # QKV matmul.
        qkv = self.qkv_proj(x)

        # Fused rope + split + KV store.
        rope = self.rope
        freqs_cis = ops.cast(rope.freqs_cis, qkv.dtype).to(qkv.device)
        xq = rope_split_store_ragged(
            kv_params=self.kv_params,
            qkv=qkv,
            input_row_offsets=kwargs["input_row_offsets"],
            freqs_cis=freqs_cis,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
            interleaved=rope.interleaved,
        )
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Calculate Flash Attention with sinks.
        mask_variant = (
            MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
            if self.layer_type == "sliding_attention"
            else MHAMaskVariant.CAUSAL_MASK
        )
        # The sinks parameter modifies the attention computation by adding an extra
        # logit column that acts as an attention sink.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size,
            sink_weights=self.sinks,
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
            self.qkv_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
            self.sinks.sharding_strategy = ShardingStrategy.rowwise(num_devices)
        else:
            raise ValueError(
                "GptOssAttention only supports tensor parallel and replicate sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[GptOssAttention]:
        """Creates sharded views of this attention layer across multiple devices.

        Overrides the parent method to handle QK normalization layers.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded GptOssAttention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "GptOssAttention layer cannot be sharded because no sharding strategy was provided."
            )

        # Get sharded weights
        qkv_proj_shards = self.qkv_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)

        # Shard sinks parameter
        sinks_shards = self.sinks.shard(devices)

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
            sharded = GptOssAttention(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                layer_type=self.layer_type,
                dtype=o_proj_shards[0].weight.dtype,
                devices=[device],
                scale=self.scale,
                has_bias=self.has_bias,
                local_window_size=self.local_window_size,
            )

            # Assign sharded weights
            sharded.qkv_proj = qkv_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Assign sinks parameter
            sharded.sinks = sinks_shards[shard_idx]

            shards.append(sharded)

        return shards
