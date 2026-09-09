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
"""An opaque KV Cache optimized attention mechanism with Rope."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)

from ..comm import Allreduce
from ..kernels import (
    flare_mla_prefill_plan,
    mla_decode_graph,
    mla_prefill_decode_graph,
    mla_prefill_graph,
)
from ..kv_cache import KVCacheParams, PagedCacheValues
from ..layer import Module, Shardable
from ..linear import Linear
from ..norm import RMSNorm
from ..quant_config import QuantConfig
from ..rotary_embedding import RotaryEmbedding
from .mask_config import MHAMaskVariant


@dataclass
class MLAPrefillMetadata:
    """Dataclass to hold MLA prefill metadata."""

    buffer_row_offsets: TensorValue
    cache_offsets: TensorValue
    buffer_lengths: TensorValue


class LatentAttentionWithRope(Module, Shardable):
    """Implementation of Latent Attention with Rope.

    Args:
        rope: The rope layer to borrow the freqs_cis value from.
        num_attention_heads: The number of attention heads.
        num_key_value_heads: Number of key/value heads.
        hidden_size: The dimension of the hidden states.
        kv_params: KV Cache Params, including the number of kv heads, the
            head dim, and data type.
        dtype: DType of the weights, currently only bfloat16 is supported.
        devices: Device to place the weights and run the computation. If
            multiple are provided, the first device is used.
        linear_cls: Linear class to use for the outputs dense layer.
        o_proj_dtype: Optional dtype override for the output projection.
        o_proj_quant_config: Optional quantization config for the output projection.
        scale: Value used to scale the results of the attention output.
        q_lora_rank: Optional LoRA rank for Q projection.
        kv_lora_rank: LoRA rank for KV projections.
        qk_nope_head_dim: Head dimension for non-positional encoding part.
        qk_rope_head_dim: Head dimension for rope part.
        v_head_dim: Head dimension for value.
        buffer_size: Buffer size for storing the temporal results during
            prefill, in unit of tokens.
        graph_mode: Pipeline role to use for the attention layer. Should be
            "prefill", "decode", or "auto".
    """

    # TODO: This will be replaced with a generic Yarn Rope implementation for Deepseek-V2-lite.
    rope: RotaryEmbedding

    _sharding_strategy: ShardingStrategy | None = None
    """The sharding strategy for the module."""

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        dtype: DType,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        o_proj_dtype: DType | None = None,
        o_proj_quant_config: QuantConfig | None = None,
        scale: float | None = None,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
        norm_dtype: DType | None = None,
    ) -> None:
        super().__init__()

        _role = graph_mode or "auto"
        if _role not in ("prefill", "decode", "auto"):
            raise ValueError(
                f"Invalid graph_mode '{_role}'. Use 'prefill', 'decode', or 'auto'."
            )
        self.graph_mode = _role

        if dtype != DType.bfloat16:
            raise ValueError(
                f"Latent Attention with Rope only supports bfloat16 dtype weights but got {dtype}"
            )

        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.norm_dtype = norm_dtype or dtype
        self.linear_cls = linear_cls

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_head_dim = kv_lora_rank + qk_rope_head_dim

        self.BUFFER_TOK_SIZE = buffer_size

        self._scale = (
            scale if scale is not None else math.sqrt(1.0 / self.qk_head_dim)
        )
        self.scale = self.rope.compute_scale(self._scale)
        self.devices = devices or [DeviceRef.CPU()]

        if self.q_lora_rank is not None:
            self.q_a_proj = Weight(
                name="q_a_proj.weight",
                dtype=dtype,
                shape=(self.q_lora_rank, self.hidden_size),
                device=self.devices[0],
            )
            self.q_a_layernorm = RMSNorm(
                dim=self.q_lora_rank,
                dtype=self.norm_dtype,
                eps=1e-6,
                multiply_before_cast=False,
            )
            self.q_b_proj = Weight(
                name="q_b_proj.weight",
                dtype=dtype,
                shape=(self.n_heads * self.qk_head_dim, self.q_lora_rank),
                device=self.devices[0],
            )
        else:
            self.q_proj = Weight(
                name="q_proj.weight",
                dtype=dtype,
                shape=(self.n_heads * self.qk_head_dim, self.hidden_size),
                device=self.devices[0],
            )

        self.kv_a_proj_layernorm = Weight(
            name="kv_a_layernorm.weight",
            dtype=self.norm_dtype,
            shape=(self.kv_lora_rank,),
            device=self.devices[0],
        )
        self.kv_a_proj_with_mqa = Weight(
            name="kv_a_proj_with_mqa.weight",
            dtype=dtype,
            shape=(self.cache_head_dim, self.hidden_size),
            device=self.devices[0],
        )
        self.kv_b_proj = Weight(
            name="kv_b_proj.weight",
            dtype=dtype,
            shape=(
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ),
            device=self.devices[0],
        )
        proj_dtype = o_proj_dtype if o_proj_dtype is not None else dtype
        self._o_proj_dtype = proj_dtype
        self._o_proj_quant_config = o_proj_quant_config
        self.o_proj = linear_cls(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=self.hidden_size,
            dtype=proj_dtype,
            device=self.devices[0],
            quant_config=o_proj_quant_config,
        )

    def create_mla_prefill_metadata(
        self, input_row_offsets: TensorValue, kv_collection: PagedCacheValues
    ) -> MLAPrefillMetadata:
        """Creates the prefill planning metadata required by MLA prefill kernels.

        Args:
            input_row_offsets: Ragged row offsets tensor describing the token
                boundaries for each sequence in the batch.
            kv_collection: Paged KV cache values for the current device.

        Returns:
            An :class:`MLAPrefillMetadata` instance containing buffer row
            offsets, cache offsets, and buffer lengths for the prefill step.
        """
        (buffer_row_offsets, cache_offsets, buffer_lengths) = (
            flare_mla_prefill_plan(
                self.kv_params,
                input_row_offsets,
                kv_collection,
                ops.constant(0, DType.uint32, device=DeviceRef.CPU()),
                self.BUFFER_TOK_SIZE,
                max_chunks=1,  # we only do one-shot prefill now.
            )
        )

        return MLAPrefillMetadata(
            buffer_row_offsets=buffer_row_offsets,
            cache_offsets=cache_offsets,
            buffer_lengths=buffer_lengths,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the Module sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Set the Module sharding strategy.

        Args:
            strategy: The strategy describing the Module sharding.
        """
        if strategy.is_tensor_parallel:
            self._sharding_strategy = strategy

            if self.n_heads % strategy.num_devices != 0:
                raise ValueError(
                    f"Number of attention heads ({self.n_heads}) must be"
                    f" divisible by the number of devices ({strategy.num_devices})."
                )

            # Tensor parallelism: shard/replicate weights appropriately
            if self.q_lora_rank is not None:
                self.q_a_proj.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )
                self.q_a_layernorm.weight.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
                self.q_b_proj.sharding_strategy = ShardingStrategy.rowwise(
                    strategy.num_devices
                )
            else:
                self.q_proj.sharding_strategy = ShardingStrategy.rowwise(
                    strategy.num_devices
                )

            self.kv_a_proj_layernorm.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
            self.kv_a_proj_with_mqa.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
            self.kv_b_proj.sharding_strategy = ShardingStrategy.rowwise(
                strategy.num_devices
            )
            self.o_proj.sharding_strategy = ShardingStrategy.columnwise(
                strategy.num_devices
            )
        elif strategy.is_replicate:
            # Data parallelism: replicate the entire module's weights to each device.
            self._sharding_strategy = strategy

            if self.q_lora_rank is not None:
                self.q_a_proj.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )
                self.q_a_layernorm.weight.sharding_strategy = (
                    ShardingStrategy.replicate(strategy.num_devices)
                )
                self.q_b_proj.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )
            else:
                self.q_proj.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )

            self.kv_a_proj_layernorm.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
            self.kv_a_proj_with_mqa.sharding_strategy = (
                ShardingStrategy.replicate(strategy.num_devices)
            )
            self.kv_b_proj.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
            self.o_proj.sharding_strategy = ShardingStrategy.replicate(
                strategy.num_devices
            )
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[LatentAttentionWithRope]:
        """Creates sharded views of this Module across multiple devices.

        Args:
            devices: Iterable of devices to place the shards on.

        Returns:
            List of sharded LatentAttentionWithRope instances, one for each device.
        """
        if not self.sharding_strategy:
            raise ValueError(
                "LatentAttentionWithRope layer cannot be sharded because no sharding strategy was provided."
            )

        if self.sharding_strategy.is_tensor_parallel:
            # Shard weights once for all devices
            if self.q_lora_rank is not None:
                q_a_proj_shards = self.q_a_proj.shard(devices)
                q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                    devices
                )
                q_b_proj_shards = self.q_b_proj.shard(devices)
            else:
                q_proj_shards = self.q_proj.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            o_proj_shards = self.o_proj.shard(devices)

            shards = []
            for shard_idx, device in enumerate(devices):
                sharded = LatentAttentionWithRope(
                    rope=self.rope,
                    num_attention_heads=self.n_heads
                    // self.sharding_strategy.num_devices,
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    dtype=self.dtype,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    o_proj_dtype=self._o_proj_dtype,
                    o_proj_quant_config=self._o_proj_quant_config,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                    norm_dtype=self.norm_dtype,
                )

                # Replace the weights with sharded versions.
                if self.q_lora_rank is not None:
                    sharded.q_a_proj = q_a_proj_shards[shard_idx]
                    sharded.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                        shard_idx
                    ]
                    sharded.q_b_proj = q_b_proj_shards[shard_idx]
                else:
                    sharded.q_proj = q_proj_shards[shard_idx]

                sharded.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                sharded.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                sharded.kv_b_proj = kv_b_proj_shards[shard_idx]
                sharded.o_proj = o_proj_shards[shard_idx]

                shards.append(sharded)

            return shards
        elif self.sharding_strategy.is_replicate:
            # Replicate full weights to each device (no head split).
            if self.q_lora_rank is not None:
                q_a_proj_shards = self.q_a_proj.shard(devices)
                q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                    devices
                )
                q_b_proj_shards = self.q_b_proj.shard(devices)
            else:
                q_proj_shards = self.q_proj.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            o_proj_shards = self.o_proj.shard(devices)

            replicas: list[LatentAttentionWithRope] = []
            for shard_idx, device in enumerate(devices):
                replica = LatentAttentionWithRope(
                    rope=self.rope,
                    num_attention_heads=self.n_heads,  # DP keeps full heads
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    dtype=self.dtype,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    o_proj_dtype=self._o_proj_dtype,
                    o_proj_quant_config=self._o_proj_quant_config,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                    norm_dtype=self.norm_dtype,
                )

                if self.q_lora_rank is not None:
                    replica.q_a_proj = q_a_proj_shards[shard_idx]
                    replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                        shard_idx
                    ]
                    replica.q_b_proj = q_b_proj_shards[shard_idx]
                else:
                    replica.q_proj = q_proj_shards[shard_idx]

                replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                replica.kv_b_proj = kv_b_proj_shards[shard_idx]
                replica.o_proj = o_proj_shards[shard_idx]

                replicas.append(replica)

            return replicas
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    @property
    def _kv_b_proj_weight(self) -> TensorValue:
        """Returns `kv_b_proj` reshaped for per-head projection slicing."""
        kv_b_proj_weight: TensorValue = self.kv_b_proj.transpose(0, 1)
        kv_b_proj_weight = kv_b_proj_weight.reshape(
            (self.kv_lora_rank, self.n_heads, -1)
        )
        return kv_b_proj_weight

    @property
    def w_uk(self) -> TensorValue:
        """Returns decode K-projection weights with shape [H, qk_nope_dim, kv_rank]."""
        kv_b_proj_weight = self._kv_b_proj_weight
        w_uk_base = kv_b_proj_weight[..., : self.qk_nope_head_dim]
        return w_uk_base.transpose(0, 1)

    @property
    def w_uv(self) -> TensorValue:
        """Returns decode V-projection weights with shape [H, kv_rank, v_dim]."""
        kv_b_proj_weight = self._kv_b_proj_weight
        w_uv = kv_b_proj_weight[..., self.qk_nope_head_dim :]
        return w_uv.permute([1, 2, 0])

    @property
    def w_k(self) -> TensorValue:
        """Returns prefill K-projection weights with shape [H*qk_nope_dim, kv_rank]."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        w_k = w_uk_base.permute([1, 2, 0]).reshape((-1, self.kv_lora_rank))
        return w_k

    @property
    def wqkv(self) -> TensorValue:
        """Returns the concatenation of q_a_proj and kv_a_proj_with_mqa weight vectors."""
        if self.q_lora_rank is not None:
            return ops.concat((self.q_a_proj, self.kv_a_proj_with_mqa))
        else:
            return ops.concat((self.q_proj, self.kv_a_proj_with_mqa))

    def _mla_impl(
        self,
        xq: TensorValue,
        kv: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_norm_gamma: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        epsilon: float = 1e-6,
    ) -> TensorValue:
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_norm_gamma,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": epsilon,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
        }

        if self.graph_mode in ["prefill", "auto"]:
            mla_prefill_metadata: MLAPrefillMetadata
            if _mla_prefill_metadata is None:
                mla_prefill_metadata = self.create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            else:
                mla_prefill_metadata = _mla_prefill_metadata

            attn_kwargs["buffer_row_offsets"] = (
                mla_prefill_metadata.buffer_row_offsets
            )
            attn_kwargs["cache_offsets"] = mla_prefill_metadata.cache_offsets
            attn_kwargs["buffer_length"] = (
                mla_prefill_metadata.buffer_lengths.to(DeviceRef.CPU())
            )
            attn_kwargs["w_k"] = self.w_k
            attn_kwargs["w_uv"] = self.w_uv

        if self.graph_mode in ["decode", "auto"]:
            attn_kwargs["w_uk"] = self.w_uk
            attn_kwargs["w_uv"] = self.w_uv
            assert kv_collection.attention_dispatch_metadata is not None
            attn_kwargs["scalar_args"] = (
                kv_collection.attention_dispatch_metadata
            )
            assert kv_collection.mla_num_partitions is not None
            attn_kwargs["num_partitions_scalar"] = (
                kv_collection.mla_num_partitions
            )

        if self.graph_mode == "prefill":
            result = mla_prefill_graph(**attn_kwargs)
        elif self.graph_mode == "decode":
            result = mla_decode_graph(**attn_kwargs)
        else:
            result = mla_prefill_decode_graph(**attn_kwargs)

        result = ops.reshape(
            result, shape=[result.shape[0], self.n_heads * self.v_head_dim]
        )

        return result

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> TensorValue:
        qkv = x @ self.wqkv.T
        xq, kv = ops.split(
            qkv,
            [
                self.q_lora_rank or self.n_heads * self.qk_head_dim,
                self.cache_head_dim,
            ],
            axis=1,
        )

        if self.q_lora_rank is not None:
            xq = self.q_a_layernorm(xq) @ self.q_b_proj.T

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))
        freqs_cis = ops.cast(freqs_cis, xq.dtype).to(xq.device)

        attn_out = self._mla_impl(
            xq,
            kv,
            kv_collection,
            layer_idx,
            input_row_offsets,
            freqs_cis,
            self.kv_a_proj_layernorm,
            mla_prefill_metadata,
            epsilon=1e-6,
        )

        return self.o_proj(attn_out)


class TensorParallelLatentAttentionWithRope(LatentAttentionWithRope):
    """Distributed tensor parallel implementation of the Latent Attention with
    Rope. Note that using tensor parallelism for MLA will cause the KV-cache to
    be duplicated across all devices, which is not efficient.

    When ``skip_allreduce`` is True, the final all-reduce is skipped. This is
    intended for mixed TP-attention + EP-MoE configurations, where the
    communication is handled explicitly by the caller.
    """

    def __init__(self, *, skip_allreduce: bool = False, **kwargs) -> None:
        super().__init__(**kwargs)
        num_devices = len(self.devices)
        self.skip_allreduce = skip_allreduce
        self.sharding_strategy = ShardingStrategy.tensor_parallel(num_devices)
        self.allreduce = Allreduce(num_devices)

        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        """Creates per-device MLA prefill metadata for tensor-parallel execution.

        Args:
            input_row_offsets_: Per-device ragged row offset tensors.
            kv_collections: Per-device paged KV cache values.

        Returns:
            A list of :class:`MLAPrefillMetadata` instances, one per device.
        """
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: Sequence[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")
        if len(input_row_offsets) != len(self.devices):
            raise ValueError(
                f"Expected {len(self.devices)} input_row_offsets, got {len(input_row_offsets)}"
            )
        if not all(isinstance(x, TensorValue) for x in input_row_offsets):
            raise TypeError(
                "All elements in input_row_offsets must be TensorValue instances"
            )

        n = len(self.devices)
        inputs: list[TensorValue] = []
        for i in range(n):
            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                mla_prefill_metadata_i = None
            inputs.append(
                self.list_of_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                    mla_prefill_metadata=mla_prefill_metadata_i,
                )
            )

        if self.skip_allreduce:
            return inputs

        return self.allreduce(
            inputs=inputs,
            signal_buffers=signal_buffers,
        )


class DataParallelLatentAttentionWithRope(LatentAttentionWithRope):
    """Data-parallel implementation of Latent Attention with RoPE.

    This replicates the attention module across devices and runs each replica on
    its local inputs (x, kv, freqs_cis, input_row_offsets). No collective ops
    are required; KV-cache remains local to each device.

    Notes:
      - `signal_buffers` is accepted for interface parity with the distributed
        implementation but is not used here.
      - Assumes the caller has already distributed `xs`, `kv_collections`,
        `freqs_cis`, and `input_row_offsets` so that index i corresponds to
        device i, with `input_row_offsets[i]` rebased to start at 0.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        num_devices = len(self.devices)
        self.sharding_strategy = ShardingStrategy.replicate(num_devices)
        self.list_of_attentions = self.shard(self.devices)

    def create_mla_prefill_metadata(  # type: ignore[override]
        self,
        input_row_offsets_: list[TensorValue],
        kv_collections: list[PagedCacheValues],
    ) -> list[MLAPrefillMetadata]:
        """Creates per-device MLA prefill metadata for data-parallel execution.

        Args:
            input_row_offsets_: Per-device ragged row offset tensors.
            kv_collections: Per-device paged KV cache values.

        Returns:
            A list of :class:`MLAPrefillMetadata` instances, one per device.
        """
        multi_mla_prefill_metadata: list[MLAPrefillMetadata] = []

        for input_row_offsets, kv_collection in zip(
            input_row_offsets_, kv_collections, strict=True
        ):
            multi_mla_prefill_metadata.append(
                super().create_mla_prefill_metadata(
                    input_row_offsets, kv_collection
                )
            )

        return multi_mla_prefill_metadata

    def __call__(  # type: ignore[override]
        self,
        layer_idx: TensorValue,
        xs: Sequence[TensorValue],
        signal_buffers: Sequence[BufferValue],
        kv_collections: Sequence[PagedCacheValues],
        freqs_cis: list[TensorValue],
        input_row_offsets: Sequence[TensorValue],
        mla_prefill_metadata: list[MLAPrefillMetadata] | None = None,
    ) -> list[TensorValue]:
        if not self.devices:
            raise ValueError("devices cannot be None or empty")

        n = len(self.devices)
        if not (
            len(xs)
            == len(kv_collections)
            == len(freqs_cis)
            == len(input_row_offsets)
            == n
        ):
            raise ValueError(
                "xs, kv_collections, freqs_cis, and input_row_offsets must all have "
                f"length equal to number of devices ({n})"
            )

        outs: list[TensorValue] = []
        for i in range(n):
            mla_prefill_metadata_i: MLAPrefillMetadata | None
            if (
                mla_prefill_metadata is not None
                and len(mla_prefill_metadata) == n
            ):
                mla_prefill_metadata_i = mla_prefill_metadata[i]
            else:
                assert (
                    mla_prefill_metadata is None
                    or len(mla_prefill_metadata) == 0
                )
                mla_prefill_metadata_i = None
            outs.append(
                self.list_of_attentions[i](
                    layer_idx,
                    xs[i],
                    kv_collections[i],
                    freqs_cis=freqs_cis[i],
                    input_row_offsets=input_row_offsets[i],
                    mla_prefill_metadata=mla_prefill_metadata_i,
                )
            )
        return outs
