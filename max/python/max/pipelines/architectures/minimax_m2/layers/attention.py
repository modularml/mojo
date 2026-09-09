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

"""MiniMax-M2 Attention Layer.

MiniMax-M2 applies per-projection RMSNorm on Q/K after projection and before
RoPE. The checkpoint stores:
  q_norm.weight: [num_attention_heads * head_dim]
  k_norm.weight: [num_key_value_heads * head_dim]

Under data parallelism each device sees all heads, so RMS is computed over
the full projection (all heads together) for both Q and K. For K, we use
rms_norm_key_cache(per_head_norm=False) which applies RMS over the full
[num_key_value_heads * head_dim] gamma.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention import MHAMaskVariant
from max.nn.attention.attention_with_rope import _compute_shard_range
from max.nn.kernels import (
    apply_qk_rms_norm,
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
    row_mean_of_squares_qk,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.quant_config import QuantConfig
from max.nn.quant_ops import quantized_fused_qkv_matmul
from max.nn.rotary_embedding import RotaryEmbedding


class MiniMaxM2Attention(Module, Shardable):
    """Attention layer for MiniMax-M2 with per-layer QK norm.

    Uses raw Weight for q_norm gamma to avoid RMSNorm sharding limitations,
    and rms_norm_key_cache(per_head_norm=False) for K norm.
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float,
        qk_norm_eps: float = 1e-6,
        norm_dtype: DType,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.norm_dtype = norm_dtype
        self.quant_config = quant_config
        self.linear_cls = linear_cls
        self.scale = scale
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        q_weight_dim = self.kv_params.head_dim * num_attention_heads
        kv_weight_dim = self.kv_params.head_dim * num_key_value_heads

        # Q norm gamma as raw Weight [n_heads * head_dim]
        # We reshape to [n_heads, head_dim] in __call__ for per-head norm
        self.q_norm = Weight(
            "q_norm.weight",
            shape=[q_weight_dim],
            dtype=self.norm_dtype,
            device=devices[0],
        )
        # K norm gamma as raw Weight [n_kv_heads * head_dim]
        # Used directly by rms_norm_key_cache
        self.k_norm = Weight(
            "k_norm.weight",
            shape=[kv_weight_dim],
            dtype=self.norm_dtype,
            device=devices[0],
        )

        self.q_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=q_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.k_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.v_proj = linear_cls(
            in_dim=hidden_size,
            out_dim=kv_weight_dim,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )
        self.o_proj = linear_cls(
            in_dim=q_weight_dim,
            out_dim=hidden_size,
            dtype=dtype,
            device=devices[0],
            has_bias=False,
        )

    @property
    def wqkv(self) -> TensorValue:
        wq: TensorValue = self.q_proj.weight
        wk: TensorValue = self.k_proj.weight
        wv: TensorValue = self.v_proj.weight
        return ops.concat((wq, wk, wv)).to(self.devices[0])

    def _qkv_weight_scale(self) -> TensorValue:
        assert self.q_proj.weight_scale is not None
        assert self.k_proj.weight_scale is not None
        assert self.v_proj.weight_scale is not None
        q_scale_w = self.q_proj.weight_scale
        k_scale_w = self.k_proj.weight_scale
        v_scale_w = self.v_proj.weight_scale
        q_scale: TensorValue | Weight = (
            q_scale_w.reshape((1,)) if len(q_scale_w.shape) == 0 else q_scale_w
        )
        k_scale: TensorValue | Weight = (
            k_scale_w.reshape((1,)) if len(k_scale_w.shape) == 0 else k_scale_w
        )
        v_scale: TensorValue | Weight = (
            v_scale_w.reshape((1,)) if len(v_scale_w.shape) == 0 else v_scale_w
        )
        return ops.concat((q_scale, k_scale, v_scale))

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if strategy.is_replicate:
            self.q_proj.sharding_strategy = strategy
            self.k_proj.sharding_strategy = strategy
            self.v_proj.sharding_strategy = strategy
            self.o_proj.sharding_strategy = strategy
            self.q_norm.sharding_strategy = strategy
            self.k_norm.sharding_strategy = strategy
        elif strategy.is_tensor_parallel:
            num_devices = strategy.num_devices
            # Q/K/V: column-shard by head groups (output dim split across devices)
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            # O proj: row-shard (input dim = partial head outputs, output summed
            # via allreduce in the containing block)
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, self.kv_params.head_dim
                )
            )
            # QK norms: shard head-dim slice matching Q/K head split
            self.q_norm.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
        else:
            raise ValueError(
                "MiniMaxM2Attention supports replicate (DP+EP) or "
                f"tensor_parallel sharding only. Got: {strategy}"
            )
        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[MiniMaxM2Attention]:
        if not self._sharding_strategy:
            raise ValueError(
                "MiniMaxM2Attention layer cannot be sharded because no "
                "sharding strategy was provided."
            )

        devices_list = list(devices)
        num_devices = len(devices_list)
        is_replicate = self._sharding_strategy.is_replicate

        q_proj_shards = self.q_proj.shard(devices_list)
        k_proj_shards = self.k_proj.shard(devices_list)
        v_proj_shards = self.v_proj.shard(devices_list)
        o_proj_shards = self.o_proj.shard(devices_list)
        q_norm_shards = self.q_norm.shard(devices_list)
        k_norm_shards = self.k_norm.shard(devices_list)

        shards: list[MiniMaxM2Attention] = []

        for shard_idx, device in enumerate(devices_list):
            if is_replicate:
                sharded_num_heads = self.n_heads
                sharded_num_kv_heads = self.num_key_value_heads
            else:
                head_start, head_end = _compute_shard_range(
                    self.n_heads, shard_idx, num_devices
                )
                sharded_num_heads = head_end - head_start

                kv_head_start, kv_head_end = _compute_shard_range(
                    self.num_key_value_heads, shard_idx, num_devices
                )
                sharded_num_kv_heads = kv_head_end - kv_head_start

            sharded = MiniMaxM2Attention(
                rope=self.rope,
                num_attention_heads=sharded_num_heads,
                num_key_value_heads=sharded_num_kv_heads,
                hidden_size=self.hidden_size,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                qk_norm_eps=self.qk_norm_eps,
                norm_dtype=self.norm_dtype,
                quant_config=self.quant_config,
            )

            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]

            shards.append(sharded)

        return shards

    def _apply_q_rms_norm(self, xq: TensorValue) -> TensorValue:
        """Apply per-projection RMSNorm to Q matching the HF reference.

        The HF reference applies RMSNorm to the flat Q projection output
        [seq, n_heads * head_dim] before reshaping to heads. RMS is
        computed across ALL heads together (6144 dims), not per-head (128).
        """
        q_dim = self.n_heads * self.kv_params.head_dim
        xq_flat = ops.reshape(xq, shape=[-1, q_dim])
        xq_normed = ops.rms_norm(
            xq_flat,
            weight=self.q_norm.to(xq.device),
            epsilon=self.qk_norm_eps,
        )
        return xq_normed.reshape((-1, self.n_heads, self.kv_params.head_dim))

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """Forward pass through the attention layer."""
        total_seq_len = x.shape[0]
        wqkv = self.wqkv

        if self.quant_config is not None:
            xq = quantized_fused_qkv_matmul(
                kv_params=self.kv_params,
                x=x,
                wqkv=wqkv,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                input_row_offsets=input_row_offsets,
                n_heads=self.n_heads,
                quant_config=self.quant_config,
                weight_scale=self._qkv_weight_scale(),
                bias=None,
            )
        else:
            xq = fused_qkv_ragged_matmul(
                kv_params=self.kv_params,
                input=x,
                input_row_offsets=input_row_offsets,
                wqkv=wqkv,
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
            )

        # xq: [total_seq_len, n_heads, head_dim]
        xq = xq.reshape((-1, self.n_heads, self.kv_params.head_dim))

        # Apply per-layer Q RMSNorm (across all heads, matching HF reference)
        xq = self._apply_q_rms_norm(xq)

        # Apply K norm in-place on the KV cache
        # per_head_norm=False: gamma shape [n_kv_heads * head_dim], cross-head norm
        rms_norm_key_cache(
            self.kv_params,
            kv_collection=kv_collection,
            gamma=self.k_norm.cast(self.kv_params.dtype).to(self.devices[0]),
            epsilon=self.qk_norm_eps,
            layer_idx=layer_idx,
            total_seq_len=total_seq_len,
            input_row_offsets=input_row_offsets,
            weight_offset=0.0,
            multiply_before_cast=False,
            per_head_norm=False,
        )

        # Apply rotary embedding
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)
        xq = fused_qk_ragged_rope(
            self.kv_params,
            xq,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        # Flash Attention
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=xq,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        # Output projection
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)

    def tp_project(
        self,
        x: TensorValue,
        input_row_offsets: TensorValue,
    ) -> tuple[TensorValue, TensorValue, TensorValue, TensorValue]:
        """TP phase 1: project Q/K/V and compute local q/k norm statistics.

        Under tensor parallelism the per-layer QK RMSNorm is a cross-head norm
        over the *full* Q (n_heads * head_dim) and K (n_kv_heads * head_dim)
        projections, but each device only holds a slice of the heads. This
        returns the unnormalized Q/K/V (head slices for this device) along with
        the per-token mean of squares over the *local* head slice. The caller
        all-reduces these statistics across TP ranks so the global RMS can be
        applied in :meth:`tp_finish`.

        Returns:
            Tuple of (q, k, v, qk_var_local) where q/k/v are flat projections
            [total_seq_len, local_dim] and qk_var_local is [total_seq_len, 2]
            holding (mean(q^2), mean(k^2)) over local dims in float32.
        """
        if self.quant_config is None:
            # Fuse Q/K/V into one matmul + split (2 GEMMs/layer instead of 4).
            head_dim = self.kv_params.head_dim
            q_dim = self.n_heads * head_dim
            kv_dim = self.num_key_value_heads * head_dim
            qkv = x @ self.wqkv.T
            q, k, v = ops.split(qkv, [q_dim, kv_dim, kv_dim], axis=-1)
        else:
            # Quantized attention projections use a different fused kernel in
            # the DP path; keep the separate-matmul form here for correctness.
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)

        # Per-row mean of squares (float32 accum) via a warp-tiled reduction
        # kernel instead of the generic ops.mean(x*x) reduce, which is heavily
        # over-provisioned at decode-time small M. Fused into a single launch
        # that emits [total_seq_len, 2] directly (col 0 = mean(q^2), col 1 =
        # mean(k^2)), avoiding two launches plus a concat.
        qk_var_local = row_mean_of_squares_qk(q, k)  # [total_seq_len, 2] f32
        return q, k, v, qk_var_local

    def tp_finish(
        self,
        layer_idx: TensorValue,
        q: TensorValue,
        k: TensorValue,
        v: TensorValue,
        qk_var_global: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        """TP phase 2: apply cross-rank QK norm, then attention and o_proj.

        Args:
            qk_var_global: [total_seq_len, 2] holding the globally-reduced mean
                of squares for Q and K (already averaged across TP ranks).
        """
        head_dim = self.kv_params.head_dim

        # Fused cross-head QK-RMSNorm apply: a single kernel does the rsqrt
        # scale (from the all-reduced [M, 2] statistics), the per-column gamma
        # multiply, and the downcast for both Q and K. This replaces the ~7
        # small slice/cast/rsqrt/mul/View kernels the unfused form lowers to at
        # decode-time small M. The gamma casts fold to constants at compile
        # time (q_norm/k_norm are weights).
        gamma_q = ops.cast(self.q_norm.to(q.device), DType.float32)
        gamma_k = ops.cast(self.k_norm.to(k.device), DType.float32)
        q, k = apply_qk_rms_norm(
            q,
            k,
            qk_var_global,
            gamma_q,
            gamma_k,
            self.qk_norm_eps,
        )

        total_seq_len = q.shape[0]
        q = ops.reshape(q, shape=[-1, self.n_heads, head_dim])
        k = ops.reshape(k, shape=[-1, self.num_key_value_heads, head_dim])
        v = ops.reshape(v, shape=[-1, self.num_key_value_heads, head_dim])

        # Store normed (un-roped) K and V; rope rotates K in-cache below.
        store_k_cache_ragged(kv_collection, k, input_row_offsets, layer_idx)
        store_v_cache_ragged(kv_collection, v, input_row_offsets, layer_idx)

        freqs_cis = ops.cast(freqs_cis, q.dtype).to(q.device)
        q = fused_qk_ragged_rope(
            self.kv_params,
            q,
            input_row_offsets,
            kv_collection,
            freqs_cis,
            layer_idx,
            interleaved=self.rope.interleaved,
        )

        attn_out = flash_attention_ragged(
            self.kv_params,
            input=q,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
        )

        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])
        return self.o_proj(attn_out)
