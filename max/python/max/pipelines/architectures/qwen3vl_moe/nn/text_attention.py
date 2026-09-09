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
"""Qwen3VL-MoE attention layer for the language model decoder (text)."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention import MHAMaskVariant
from max.nn.attention.attention_with_rope import _compute_shard_range
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
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.stacked_linear import StackedLinear

from .text_rotary import Qwen3VLTextRotaryEmbedding


class Qwen3VLMoEDecoderAttentionWithRope(Module, Shardable):
    """Qwen3VLMoE-style attention with RoPE and per-head Q/K RMSNorm (text)."""

    rope: Qwen3VLTextRotaryEmbedding

    def __init__(
        self,
        *,
        rope: Qwen3VLTextRotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        devices: Sequence[DeviceRef] | None = None,
        dtype: DType = DType.float32,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        has_bias: bool = True,
        rms_norm_eps: float = 1e-6,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()

        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.scale = (
            scale
            if scale is not None
            else math.sqrt(1.0 / float(self.kv_params.head_dim))
        )

        self.devices = devices or [DeviceRef.CPU()]
        self._sharding_strategy: ShardingStrategy | None = None
        self.quant_config = quant_config

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        self.qkv_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=[q_weight_dim, kv_weight_dim, kv_weight_dim],
            names=["q_proj", "k_proj", "v_proj"],
            dtype=dtype,
            device=self.devices[0],
            stacked=False,
            has_bias=has_bias,
            linear_cls=linear_cls,
            quant_config=quant_config,
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=self.devices[0],
            has_bias=has_bias,
            quant_config=quant_config,
        )

        # Per-head RMSNorm for Q and K.
        self.q_norm = RMSNorm(
            dim=self.kv_params.head_dim,
            dtype=kv_params.dtype,
            eps=rms_norm_eps,
            multiply_before_cast=True,
        )
        self.k_norm = RMSNorm(
            dim=self.kv_params.head_dim,
            dtype=kv_params.dtype,
            eps=rms_norm_eps,
            multiply_before_cast=True,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass over a batch of tokens in ragged form.

        Args:
            layer_idx: Layer index for KV cache.
            x: Flattened input [T, H] for all sequences in the batch, where
                T = sum_i L_i over all sequences.
            kv_collection: KV cache handle.
            freqs_cis: Per-token MRoPE frequency table of shape
                ``[total_seq_len, head_dim]``. Row ``i`` corresponds to
                token ``i`` (positions are pre-baked by
                :class:`Qwen3VLTextRotaryEmbedding`).
            input_row_offsets: Ragged offsets [0, L0, L0+L1, ...]. For a single
                contiguous sequence of length L this is simply [0, L].
        """
        total_seq_len = x.shape[0]
        head_dim = self.kv_params.head_dim
        # Use per-module head counts so tensor-parallel shards match StackedLinear
        # output widths (kv_params.n_kv_heads stays global for the KV cache).
        n_kv_heads = self.num_key_value_heads
        q_dim = head_dim * self.n_heads
        kv_dim = head_dim * n_kv_heads

        # QKV projection through StackedLinear. This handles FP8/BF16/NVFP4
        # uniformly and returns a BF16 buffer.
        qkv = self.qkv_proj(x)

        # Per-head RMSNorm on Q and K before RoPE (Qwen3VL reference order).
        x_q, x_k, x_v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
        x_q = self.q_norm(x_q.reshape((-1, self.n_heads, head_dim))).reshape(
            (-1, q_dim)
        )
        x_k = self.k_norm(x_k.reshape((-1, n_kv_heads, head_dim))).reshape(
            (-1, kv_dim)
        )
        qkv = ops.concat((x_q, x_k, x_v), axis=-1)

        # Fused RoPE + split + KV cache store. `freqs_cis` is per-token
        # (row i = token i) so we pass explicit position_ids = [0..T-1] to
        # override the kernel's default cache_length + token_idx indexing.
        freqs_cis = ops.cast(freqs_cis, qkv.dtype).to(qkv.device)
        position_ids = ops.unsqueeze(
            ops.range(
                0, total_seq_len, 1, device=qkv.device, dtype=DType.uint32
            ),
            0,
        )
        xq = rope_split_store_ragged(
            self.kv_params,
            qkv,
            input_row_offsets,
            freqs_cis,
            kv_collection,
            layer_idx,
            n_heads=self.n_heads,
            interleaved=self.rope.interleaved,
            position_ids=position_ids,
        )
        xq = xq.reshape((-1, self.n_heads, head_dim))

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, sharding_strategy: ShardingStrategy) -> None:
        num_devices = sharding_strategy.num_devices

        if sharding_strategy.is_replicate:
            self.qkv_proj.sharding_strategy = sharding_strategy
            self.o_proj.sharding_strategy = sharding_strategy
            self.q_norm.sharding_strategy = sharding_strategy
            self.k_norm.sharding_strategy = sharding_strategy

        elif sharding_strategy.is_tensor_parallel:
            self.qkv_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        else:
            raise ValueError(
                "Qwen3VLMoEDecoderAttentionWithRope only supports tensor parallel "
                "and replicate sharding strategies."
            )

        self._sharding_strategy = sharding_strategy

    def shard(
        self,
        devices: Iterable[DeviceRef],
    ) -> list[Qwen3VLMoEDecoderAttentionWithRope]:
        if not self.sharding_strategy:
            raise ValueError(
                "Qwen3VLMoEDecoderAttentionWithRope layer cannot be sharded "
                "because no sharding strategy was provided."
            )

        devices_list = list(devices)

        qkv_proj_shards = self.qkv_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)

        shards: list[Qwen3VLMoEDecoderAttentionWithRope] = []
        num_shards = len(devices_list)

        for shard_idx, device in enumerate(devices_list):
            head_start, head_end = _compute_shard_range(
                self.n_heads, shard_idx, num_shards
            )
            sharded_num_heads = head_end - head_start

            assert isinstance(self.kv_params, MHAKVCacheParams)
            kv_head_start, kv_head_end = _compute_shard_range(
                self.kv_params.n_kv_heads, shard_idx, num_shards
            )
            sharded_num_kv_heads = kv_head_end - kv_head_start

            sharded = Qwen3VLMoEDecoderAttentionWithRope(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                dtype=self.qkv_proj._child("q_proj").weight.dtype,
                devices=[device],
                linear_cls=self.qkv_proj._child("q_proj").__class__,
                scale=self.scale,
                has_bias=self.has_bias,
                rms_norm_eps=self.rms_norm_eps,
                quant_config=self.quant_config,
            )

            sharded.qkv_proj = qkv_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]
            shards.append(sharded)

        return shards
