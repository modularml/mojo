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

"""Gemma4 Attention Layer."""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import (
    DeviceRef,
    ShardingStrategy,
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
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.stacked_linear import StackedLinear
from max.pipelines.architectures.gemma4.layers.rms_norm import Gemma4RMSNorm


class Gemma4Attention(Module, Shardable):
    """Implementation of the attention layer for the Gemma3 text model."""

    # Flash-attention mask override for subclasses (e.g. the DSpark draft's
    # non-causal block attention). ``None`` selects the standard causal /
    # sliding-window-causal mask by layer type.
    mask_variant: MHAMaskVariant | None = None

    def __init__(
        self,
        *,
        rope_global: RotaryEmbedding,
        rope_local: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_global_key_value_heads: int,
        attention_k_eq_v: bool,
        hidden_size: int,
        kv_params: KVCacheParams,
        global_head_dim: int,
        layer_idx: int,
        layer_idx_in_cache: int,
        is_sliding: bool,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        has_bias: bool = False,
        qk_norm_eps: float = 1e-6,
        local_window_size: int = 1024,
        quant_config: QuantConfig | None = None,
        fused_qkv: bool = False,
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
            layer_idx_in_cache: The 0-based index of this layer within its
                sub-cache (sliding or global).
            is_sliding: Whether this layer uses sliding window attention.
            dtype: DType of the attention inputs and weights.
            devices: Device to place the weights and run the computation. If
                multiple are provided, the first device is used. Use
                `TensorParallelAttentionWithRope` to use all devices during
                attention computation.
            linear_cls: Linear class to use for the outputs dense layer.
            has_bias: Whether to use an attention bias. Defaults to False.
            qk_norm_eps: Value to use for numerical stability. Defaults to 1e-6.
            quant_config: Scaled quantization configuration. Defaults to None.
            fused_qkv: When True, the qkv/qk projection uses a single stacked
                weight (``StackedLinear(stacked=True)``) loaded pre-fused from
                the checkpoint instead of concatenating per-projection weights
                in-graph (DISTINF-194). Defaults to False.
        """

        super().__init__()
        self.rope_global = rope_global
        self.rope_local = rope_local
        self.n_heads = num_attention_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self._sharding_strategy: ShardingStrategy | None = None
        self.dtype = dtype
        self.scale = 1.0
        self.local_window_size = local_window_size
        self.qk_norm_eps = qk_norm_eps
        self.quant_config = quant_config

        self.num_global_key_value_heads = num_global_key_value_heads
        self.global_head_dim = global_head_dim
        self.attention_k_eq_v = attention_k_eq_v

        self.use_local = is_sliding
        self.layer_idx_in_cache = layer_idx_in_cache

        self.head_dim = (
            self.kv_params.head_dim
        )  # MultiKVCacheParams sets head dim to either local or global

        self.q_norm = Gemma4RMSNorm(self.head_dim, dtype, self.qk_norm_eps)
        self.k_norm = Gemma4RMSNorm(self.head_dim, dtype, self.qk_norm_eps)
        self.v_norm = Gemma4RMSNorm(
            self.head_dim,
            dtype,
            self.qk_norm_eps,
            with_weight=False,
        )
        num_key_value_heads = (
            num_key_value_heads
            if self.use_local
            else num_global_key_value_heads
        )
        self.num_key_value_heads = num_key_value_heads
        self.q_weight_dim = self.head_dim * num_attention_heads
        self.kv_weight_dim = self.head_dim * num_key_value_heads

        # When attention_k_eq_v and not sliding, V reuses K weights so we
        # only need q and k projections.  Otherwise we need all three.
        self._has_v_proj = not (attention_k_eq_v and not self.use_local)
        if self._has_v_proj:
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
                stacked=fused_qkv,
                has_bias=has_bias,
                linear_cls=linear_cls,
                quant_config=quant_config,
            )
        else:
            self.qk_proj = StackedLinear(
                in_dim=hidden_size,
                out_dims=[self.q_weight_dim, self.kv_weight_dim],
                names=["q_proj", "k_proj"],
                dtype=dtype,
                device=devices[0],
                stacked=fused_qkv,
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
            self.layer_idx_in_cache, DType.uint32, device=DeviceRef.CPU()
        )

        head_dim = self.head_dim
        q_dim = self.q_weight_dim
        kv_dim = self.kv_weight_dim
        num_kv_heads = kv_dim // head_dim

        # QKV projection
        if self._has_v_proj:
            qkv = self.qkv_proj(x)
            x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
        else:
            qk = self.qk_proj(x)
            x_q, x_k = ops.split(qk, [q_dim, kv_dim], axis=-1)
            x_v = x_k

        # Per-head QKV norm
        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )
        x_v = self.v_norm(x_v.reshape((-1, num_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )

        # Re-concat and apply RoPE + KV cache store
        qkv = ops.concat((x_q, x_k, x_v), axis=-1)

        rope = self.rope_local if self.use_local else self.rope_global

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
            q_out_dtype=self.kv_params.dtype,
        )
        xq = xq.reshape((-1, self.n_heads, self.head_dim))

        # Calculate Flash Attention.
        mask_variant = self.mask_variant
        if mask_variant is None:
            mask_variant = (
                MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
                if self.use_local
                else MHAMaskVariant.CAUSAL_MASK
            )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=kwargs["input_row_offsets"],
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=self.local_window_size if self.use_local else -1,
            output_dtype=self.dtype,
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
            if self._has_v_proj:
                self.qkv_proj.sharding_strategy = sharding_strategy
            else:
                self.qk_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )

            if self._has_v_proj:
                self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
            else:
                self.qk_proj.sharding_strategy = ShardingStrategy.rowwise(
                    num_devices
                )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )

        else:
            raise ValueError(
                "Gemma3Attention only supports tensor parallel and replicate"
                " sharding strategy"
            )

        self._sharding_strategy = sharding_strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[Gemma4Attention]:
        """Creates sharded views of this attention layer across multiple devices.

        Overrides the parent method to handle QK normalization layers.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded Gemma3Attention instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "Gemma3Attention layer cannot be sharded because no sharding"
                " strategy was provided."
            )

        # Get sharded weights
        if self._has_v_proj:
            qkv_proj_shards = self.qkv_proj.shard(devices)
        else:
            qk_proj_shards = self.qk_proj.shard(devices)
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
            sharded_num_global_kv_heads = num_heads_for_device(
                num_heads=self.num_global_key_value_heads,
                device_idx=shard_idx,
                num_devices=self.sharding_strategy.num_devices,
            )

            # Create new attention instance with sharded configuration.
            # Construct via type(self) so subclasses (e.g. a noncausal-mask
            # decoder variant) shard into their own type rather than the base.
            sharded = type(self)(
                rope_global=self.rope_global,
                rope_local=self.rope_local,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                num_global_key_value_heads=sharded_num_global_kv_heads,
                attention_k_eq_v=self.attention_k_eq_v,
                hidden_size=self.q_weight_dim + self.kv_weight_dim * 2,
                kv_params=self.kv_params,
                global_head_dim=self.global_head_dim,
                layer_idx=self.layer_idx,
                layer_idx_in_cache=self.layer_idx_in_cache,
                is_sliding=self.use_local,
                dtype=self.o_proj.weight.dtype,
                devices=[device],
                linear_cls=self.o_proj.__class__,
                has_bias=self.has_bias,
                qk_norm_eps=self.qk_norm_eps,
                local_window_size=self.local_window_size,
                quant_config=self.quant_config,
            )

            # Assign sharded weights
            if self._has_v_proj:
                sharded.qkv_proj = qkv_proj_shards[shard_idx]
            else:
                sharded.qk_proj = qk_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]

            # Assign QK normalization weights
            sharded.q_norm.weight = q_norm_weight_shards[shard_idx]
            sharded.k_norm.weight = k_norm_weight_shards[shard_idx]

            shards.append(sharded)

        return shards
