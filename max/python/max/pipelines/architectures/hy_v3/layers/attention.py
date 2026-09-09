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
"""Hy3 GQA attention: per-head_dim Q/K RMSNorm, split-half RoPE, causal
flash attention. ``q_norm`` / ``k_norm`` gammas are ``[head_dim]`` and
broadcast over the heads dim.

Sharding is tensor-parallel (TP), modeled on MAX's native
``AttentionWithRope`` / ``TensorParallelAttentionWithRope`` and the Qwen3
GQA-with-QK-norm port (``Qwen3Attention``):

* q_proj / k_proj / v_proj are sharded rowwise (axis 0 = output channels =
  heads): device ``d`` gets ``num_attention_heads // TP`` Q heads and
  ``num_key_value_heads // TP`` KV heads.
* o_proj is sharded head-aware columnwise (axis 1 = input channels = heads),
  so each device consumes only its local Q-head slice and produces a partial
  output. The caller (``HYV3TransformerBlock``) allreduces the per-device
  o_proj outputs before the residual add, exactly as
  ``TensorParallelAttentionWithRope.__call__`` does.
* q_norm / k_norm gammas are ``[head_dim]`` and shared across heads, so they
  are replicated (not split) across devices.

TP makes each device's local Q/KV head count match the TP-sharded paged KV
cache (``n_kv_heads_per_device = num_key_value_heads // n_devices``, set by
``KVCacheParams.__post_init__`` when ``data_parallel_degree == 1`` and
``n_devices > 1``). TP must divide ``num_key_value_heads`` (8) so each device
holds a whole number of KV heads; valid TP for Hy3 is 1, 2, 4, or 8. The MoE
stays expert-parallel; attention-TP and MoE-EP compose under the same
N-device mesh, matching the native qwen3_moe / mixtral wiring.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, Weight, ops
from max.nn.attention import MHAMaskVariant
from max.nn.attention.attention_with_rope import _compute_shard_range
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    fused_qkv_ragged_matmul,
    rms_norm_key_cache,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.rotary_embedding import RotaryEmbedding


class HYV3Attention(Module, Shardable):
    """GQA attention with per-head QK-norm and split-half RoPE.

    Supports tensor-parallel (TP) and replicate (single-device) sharding.
    Under TP, each sharded instance owns ``num_attention_heads // TP`` Q
    heads and ``num_key_value_heads // TP`` KV heads and returns a PARTIAL
    o_proj output; the caller allreduces the per-device outputs. See the
    module docstring for the full TP wiring and the head-count constraint
    (TP must divide ``num_key_value_heads``).
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
        self.linear_cls = linear_cls
        self.scale = scale
        self.qk_norm_eps = qk_norm_eps
        self._sharding_strategy: ShardingStrategy | None = None

        head_dim = self.kv_params.head_dim
        q_weight_dim = head_dim * num_attention_heads
        kv_weight_dim = head_dim * num_key_value_heads

        self.q_norm = Weight(
            "q_norm.weight",
            shape=[head_dim],
            dtype=self.norm_dtype,
            device=devices[0],
        )
        self.k_norm = Weight(
            "k_norm.weight",
            shape=[head_dim],
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

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the module sharding strategy and propagate it to the weights.

        Tensor parallel (the default for multi-GPU Hy3):
          * q_proj / k_proj / v_proj -> ROWWISE (axis 0 = output channels =
            heads), so each device gets ``n_heads // TP`` Q heads and
            ``num_key_value_heads // TP`` KV heads.
          * o_proj -> HEAD_AWARE_COLUMNWISE (axis 1 = input channels =
            heads), so each device consumes its local Q-head slice and emits
            a partial sum (allreduced by the caller).
          * q_norm / k_norm -> REPLICATE (per-head ``[head_dim]`` gamma,
            shared across heads).

        Replicate (single-device): every weight is copied to each device.

        This mirrors ``AttentionWithRope.sharding_strategy`` /
        ``Qwen3Attention.sharding_strategy`` in MAX.
        """
        num_devices = strategy.num_devices

        if strategy.is_tensor_parallel:
            # Constraint: TP must divide num_key_value_heads so every device
            # holds a whole number of KV heads, matching the TP-sharded KV
            # cache (n_kv_heads_per_device = num_key_value_heads // TP).
            if self.num_key_value_heads % num_devices != 0:
                raise ValueError(
                    "HYV3Attention tensor-parallel degree "
                    f"({num_devices}) must divide num_key_value_heads "
                    f"({self.num_key_value_heads}); valid TP for Hy3 is "
                    "1, 2, 4, or 8."
                )
            head_dim = self.kv_params.head_dim
            # Rowwise split by output heads for each projection.
            self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.k_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.v_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            # Head-aware columnwise split of o_proj input (the Q heads).
            self.o_proj.sharding_strategy = (
                ShardingStrategy.head_aware_columnwise(
                    num_devices, self.n_heads, head_dim
                )
            )
            # Per-head QK-norm gammas are shared across heads -> replicate.
            self.q_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
            self.k_norm.sharding_strategy = ShardingStrategy.replicate(
                num_devices
            )
        elif strategy.is_replicate:
            self.q_proj.sharding_strategy = strategy
            self.k_proj.sharding_strategy = strategy
            self.v_proj.sharding_strategy = strategy
            self.o_proj.sharding_strategy = strategy
            self.q_norm.sharding_strategy = strategy
            self.k_norm.sharding_strategy = strategy
        else:
            raise ValueError(
                "HYV3Attention only supports tensor-parallel and replicate "
                "sharding strategies."
            )

        self._sharding_strategy = strategy

    def shard(self, devices: Iterable[DeviceRef]) -> list[HYV3Attention]:
        if not self._sharding_strategy:
            raise ValueError(
                "HYV3Attention layer cannot be sharded: no sharding "
                "strategy was provided."
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

        shards: list[HYV3Attention] = []
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

            sharded = HYV3Attention(
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
            )
            sharded.q_proj = q_proj_shards[shard_idx]
            sharded.k_proj = k_proj_shards[shard_idx]
            sharded.v_proj = v_proj_shards[shard_idx]
            sharded.o_proj = o_proj_shards[shard_idx]
            sharded.q_norm = q_norm_shards[shard_idx]
            sharded.k_norm = k_norm_shards[shard_idx]
            shards.append(sharded)
        return shards

    def _apply_q_per_head_rms_norm(self, xq: TensorValue) -> TensorValue:
        return ops.rms_norm(
            xq,
            weight=self.q_norm.to(xq.device),
            epsilon=self.qk_norm_eps,
        )

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
    ) -> TensorValue:
        total_seq_len = x.shape[0]
        head_dim = self.kv_params.head_dim

        xq = fused_qkv_ragged_matmul(
            kv_params=self.kv_params,
            input=x,
            input_row_offsets=input_row_offsets,
            wqkv=self.wqkv,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            n_heads=self.n_heads,
        )

        xq = xq.reshape((-1, self.n_heads, head_dim))
        xq = self._apply_q_per_head_rms_norm(xq)

        # Per-head Q/K RMSNorm: Q is normed in-graph above; K is normed
        # in-place in the paged cache so the cached K slots already carry
        # the post-norm values that flash attention reads.
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
            per_head_norm=True,
        )

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
