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
"""Inkling attention: qkvr projection, K/V short convolutions, relative bias.

Reference: ``nvidia/ops/qkvr_prep.py`` and ``nvidia/attention.py`` at vLLM
v0.26.0."""

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
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import MHAKVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.stacked_linear import StackedLinear

from ..model_config import InklingTextConfig
from .short_convolution import ShortConvolution

_TAU_DTYPE = DType.float32


def log_scaling_tau(
    positions: TensorValue, *, alpha: float, n_floor: int
) -> TensorValue:
    """``tau = 1 + alpha * log(max(1, (position + 1) / n_floor))``, float32."""
    effective = (ops.cast(positions, _TAU_DTYPE) + 1.0) / float(n_floor)
    return 1.0 + alpha * ops.log(ops.max(effective, 1.0))


class InklingAttention(Module, Shardable):
    """One decoder layer's attention block, unfused."""

    def __init__(
        self,
        *,
        text_config: InklingTextConfig,
        layer_idx: int,
        kv_params: MHAKVCacheParams,
        dtype: DType,
        devices: list[DeviceRef],
        tp_size: int = 1,
        is_sharding: bool = False,
        is_local: bool | None = None,
    ) -> None:
        super().__init__()
        device = devices[0]
        self.text_config = text_config
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.dtype = dtype
        if is_local is None:
            is_local = text_config.is_local_attention(layer_idx)
        self._is_local = is_local
        self._sharding_strategy: ShardingStrategy | None = None
        self.num_heads = text_config.num_heads(is_local) // tp_size
        self.num_kv_heads = text_config.num_kv_heads(is_local) // tp_size
        self.head_dim = text_config.head_dim_for(is_local)
        self.local_window_size = text_config.attention_window(is_local)
        self.applies_log_scaling = text_config.applies_log_scaling(is_local)
        # muP scales attention logits by 1/d, not the usual 1/sqrt(d).
        self.scale = 1.0 / self.head_dim
        self.out_dims = tuple(
            dim // tp_size for dim in text_config.qkvr_out_dims(is_local)
        )

        if tp_size > 1 and kv_params.n_kv_heads_per_device != self.num_kv_heads:
            raise ValueError(
                f"layer {layer_idx} is sharded {tp_size} ways into "
                f"{self.num_kv_heads} KV heads per rank, but its cache holds "
                f"{kv_params.n_kv_heads_per_device} per device; the cache and "
                "the projection must be split the same way"
            )

        self.qkvr_proj = StackedLinear(
            in_dim=text_config.hidden_size,
            out_dims=self.out_dims,
            names=["wq_du", "wk_dv", "wv_dv", "wr_du"],
            dtype=dtype,
            device=device,
            has_bias=text_config.q_bias,
            _is_sharding=is_sharding,
        )
        kv_conv_dim = text_config.kv_conv_dim(is_local) // tp_size
        if not is_sharding:
            self.k_sconv = ShortConvolution(
                channels=kv_conv_dim,
                kernel_size=text_config.sconv_kernel_size,
                dtype=dtype,
                device=device,
            )
            self.v_sconv = ShortConvolution(
                channels=kv_conv_dim,
                kernel_size=text_config.sconv_kernel_size,
                dtype=dtype,
                device=device,
            )
        self.q_norm = RMSNorm(
            self.head_dim, dtype, eps=text_config.rms_norm_eps
        )
        self.k_norm = RMSNorm(
            self.head_dim, dtype, eps=text_config.rms_norm_eps
        )
        # A plain weight rather than a submodule so its name can carry the
        # checkpoint's own rel_logits_proj.proj spelling.
        self.d_rel = text_config.d_rel
        self.rel_extent = text_config.rel_extent_for(is_local)
        if not is_sharding:
            self.rel_logits_proj = Weight(
                "rel_logits_proj.proj",
                dtype,
                [self.d_rel, self.rel_extent],
                device=device,
            )
        self.wo_ud = Linear(
            in_dim=self.num_heads * self.head_dim,
            out_dim=text_config.hidden_size,
            dtype=dtype,
            device=device,
            has_bias=text_config.o_bias,
            is_sharding=is_sharding,
        )

    def __call__(
        self,
        x: TensorValue,
        *,
        kv_collection: PagedCacheValues,
        input_row_offsets: TensorValue,
        log_scaling: TensorValue,
        k_conv_pool: BufferValue,
        v_conv_pool: BufferValue,
        slot_idx: TensorValue,
        cache_layer_idx: TensorValue,
    ) -> TensorValue:
        """Runs the block over one ragged batch. ``cache_layer_idx`` is an
        operand, not a folded constant, so every layer of one attention flavor
        can share a compiled subgraph."""
        total_tokens = x.shape[0]
        q_dim, k_dim, v_dim, r_dim = self.out_dims

        qkvr = self.qkvr_proj(x)
        q, k, v, r = ops.split(qkvr, [q_dim, k_dim, v_dim, r_dim], axis=-1)

        k = self.k_sconv(k, k_conv_pool, slot_idx, input_row_offsets)
        v = self.v_sconv(v, v_conv_pool, slot_idx, input_row_offsets)

        q = self.q_norm(
            q.reshape([total_tokens, self.num_heads, self.head_dim])
        )
        k = self.k_norm(
            k.reshape([total_tokens, self.num_kv_heads, self.head_dim])
        )
        bias = self._relative_bias(r)
        if self.applies_log_scaling:
            q = _scale_rows(q, log_scaling)
            bias = _scale_rows(bias, log_scaling)

        store_k_cache_ragged(
            kv_collection, k, input_row_offsets, cache_layer_idx
        )
        store_v_cache_ragged(
            kv_collection,
            v.reshape([total_tokens, self.num_kv_heads, self.head_dim]),
            input_row_offsets,
            cache_layer_idx,
        )
        mask_variant = (
            MHAMaskVariant.CAUSAL_MASK
            if self.local_window_size is None
            else MHAMaskVariant.SLIDING_WINDOW_CAUSAL_MASK
        )
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=q,
            input_row_offsets=input_row_offsets,
            kv_collection=kv_collection,
            layer_idx=cache_layer_idx,
            mask_variant=mask_variant,
            scale=self.scale,
            local_window_size=(
                -1 if self.local_window_size is None else self.local_window_size
            ),
            rel_logits=bias,
        )
        return self.wo_ud(attn_out.reshape([total_tokens, q_dim]))

    def _relative_bias(self, r: TensorValue) -> TensorValue:
        """Maps the head-major relative branch to
        ``[total_tokens, num_heads, extent]``, unscaled."""
        total_tokens = r.shape[0]
        rows = r.reshape([total_tokens * self.num_heads, self.d_rel])
        return (rows @ self.rel_logits_proj).reshape(
            [total_tokens, self.num_heads, self.rel_extent]
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the heads across devices: qkvr column-parallel, ``wo_ud``
        row-parallel, norms and relative-logits projection replicated.
        Column/row-parallel are the Megatron terms; in MAX terms,
        ``ShardingStrategy.rowwise`` splits the stored ``[out, in]``
        weight's rows, so it is what implements column-parallel."""
        num_devices = strategy.num_devices
        if strategy.is_tensor_parallel:
            if self.num_heads % num_devices or self.num_kv_heads % num_devices:
                raise ValueError(
                    f"layer {self.layer_idx} has {self.num_heads} query heads "
                    f"and {self.num_kv_heads} KV heads, which do not both "
                    f"divide over {num_devices} devices"
                )
            self.qkvr_proj.sharding_strategy = ShardingStrategy.rowwise(
                num_devices
            )
            self.wo_ud.sharding_strategy = ShardingStrategy.columnwise(
                num_devices
            )
        else:
            raise ValueError(
                "InklingAttention only supports the tensor parallel sharding "
                "strategy."
            )
        self._sharding_strategy = strategy
        replicate = ShardingStrategy.replicate(num_devices)
        self.k_sconv.sharding_strategy = strategy
        self.v_sconv.sharding_strategy = strategy
        self.q_norm.sharding_strategy = replicate
        self.k_norm.sharding_strategy = replicate
        self.rel_logits_proj.sharding_strategy = replicate

    def shard(self, devices: Iterable[DeviceRef]) -> Sequence[InklingAttention]:
        """Creates one per-device view of this attention block."""
        if self._sharding_strategy is None:
            raise ValueError(
                "InklingAttention cannot be sharded: no sharding strategy."
            )
        devices = list(devices)
        part_shards = {
            name: getattr(self, name).shard(devices)
            for name in (
                "qkvr_proj",
                "k_sconv",
                "v_sconv",
                "q_norm",
                "k_norm",
                "rel_logits_proj",
                "wo_ud",
            )
        }
        shards = []
        for shard_idx, device in enumerate(devices):
            sharded = InklingAttention(
                text_config=self.text_config,
                layer_idx=self.layer_idx,
                kv_params=self.kv_params,
                dtype=self.dtype,
                devices=[device],
                tp_size=len(devices),
                is_sharding=True,
                is_local=self._is_local,
            )
            for name, per_device in part_shards.items():
                setattr(sharded, name, per_device[shard_idx])
            shards.append(sharded)
        return shards


def _scale_rows(x: TensorValue, factor: TensorValue) -> TensorValue:
    """Scales each token's rows in float32 with a round-trip cast, matching
    the reference's precision on a bf16 tensor."""
    rows = ops.reshape(factor, [factor.shape[0]] + [1] * (x.rank - 1))
    return ops.cast(ops.cast(x, _TAU_DTYPE) * rows, x.dtype)
