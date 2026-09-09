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
from typing import Any

import numpy as np
from max.dtype import DType
from max.graph import (
    BufferValue,
    DeviceRef,
    ShardingStrategy,
    TensorValue,
    Weight,
    ops,
)
from max.support.math import ceildiv

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
from ..quant_config import QuantConfig, fp4_packed_k
from ..quant_ops import quantized_matmul
from ..rotary_embedding import RotaryEmbedding
from .mask_config import MHAMaskVariant
from .multi_latent_attention import MLAPrefillMetadata


class LatentAttentionWithRopeFp8(Module, Shardable):
    """Implementation of Latent Attention with Rope with FP8 weights."""

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
        quant_config: QuantConfig,
        devices: list[DeviceRef] | None = None,
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        q_lora_rank: int = 1536,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
        norm_dtype: DType = DType.bfloat16,
    ) -> None:
        """Initializes the latent attention layer.

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
            norm_dtype: DType of the weights for normalization layers.
        """
        super().__init__()

        _role = graph_mode or "auto"
        if _role not in ("prefill", "decode", "auto"):
            raise ValueError(
                f"Invalid graph_mode '{_role}'. Use 'prefill', 'decode', or 'auto'."
            )
        if (
            not quant_config.weight_scale.is_block
            or not quant_config.input_scale.is_block
        ):
            raise ValueError(
                "Weight scale and input scale must be block-wise for LatentAttentionWithRopeFp8"
            )

        self.graph_mode = _role
        self.quant_config = quant_config
        self.norm_dtype = norm_dtype

        self.rope = rope
        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
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
        assert quant_config.weight_scale.block_size is not None
        assert quant_config.input_scale.block_size is not None
        self.weight_block_size = quant_config.weight_scale.block_size
        input_k_block = quant_config.input_scale.block_size[1]

        # Granularity at which every per-head K-chunk fits in a single
        # on-disk scale block. Equals `weight_block_size[0]` when the
        # per-head row count is a multiple of it; otherwise the GCD of
        # the residue and the block (e.g. 64 when (Dn + Dv) % 128 != 0).
        block_m = int(self.weight_block_size[0])
        per_head = self.qk_nope_head_dim + self.v_head_dim
        residue = per_head % block_m
        if residue == 0 and self.qk_nope_head_dim % block_m == 0:
            self._b_scale_granularity = block_m
        else:
            self._b_scale_granularity = math.gcd(residue, block_m)

        proj_dtype = DType.float8_e4m3fn
        self.q_a_proj = Weight(
            name="q_a_proj.weight",
            dtype=proj_dtype,
            shape=(self.q_lora_rank, self.hidden_size),
            device=self.devices[0],
        )
        self.q_a_proj_scale = Weight(
            name="q_a_proj.weight_scale",
            dtype=quant_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.q_a_proj.shape[0]),
                    quant_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.q_a_proj.shape[1]),
                    input_k_block,
                ),
            ),
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
            dtype=proj_dtype,
            shape=(self.n_heads * self.qk_head_dim, self.q_lora_rank),
            device=self.devices[0],
        )
        self.q_b_proj_scale = Weight(
            name="q_b_proj.weight_scale",
            dtype=quant_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.q_b_proj.shape[0]),
                    quant_config.weight_scale.block_size[0],
                ),
                ceildiv(
                    int(self.q_b_proj.shape[1]),
                    input_k_block,
                ),
            ),
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
            dtype=proj_dtype,
            shape=(self.cache_head_dim, self.hidden_size),
            device=self.devices[0],
        )
        self.kv_a_proj_with_mqa_scale = Weight(
            name="kv_a_proj_with_mqa.weight_scale",
            dtype=quant_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.kv_a_proj_with_mqa.shape[0]),
                    self.weight_block_size[0],
                ),
                ceildiv(
                    int(self.kv_a_proj_with_mqa.shape[1]),
                    input_k_block,
                ),
            ),
            device=self.devices[0],
        )

        self.kv_b_proj = Weight(
            name="kv_b_proj.weight",
            dtype=proj_dtype,
            shape=(
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ),
            device=self.devices[0],
        )
        self.kv_b_proj_scale = Weight(
            name="kv_b_proj.weight_scale",
            dtype=quant_config.weight_scale.dtype,
            shape=(
                ceildiv(
                    int(self.kv_b_proj.shape[0]),
                    self.weight_block_size[0],
                ),
                ceildiv(
                    int(self.kv_b_proj.shape[1]),
                    input_k_block,
                ),
            ),
            device=self.devices[0],
        )

        o_proj_quant_config = quant_config
        o_proj_in_dim = fp4_packed_k(
            self.n_heads * self.v_head_dim, quant_config
        )
        self.o_proj = linear_cls(
            in_dim=o_proj_in_dim,
            out_dim=self.hidden_size,
            dtype=proj_dtype,
            device=self.devices[0],
            quant_config=o_proj_quant_config,
        )

    def create_mla_prefill_metadata(
        self, input_row_offsets: TensorValue, kv_collection: PagedCacheValues
    ) -> MLAPrefillMetadata:
        """Creates the prefill planning metadata required by FP8 MLA prefill kernels.

        Args:
            input_row_offsets: Ragged row offsets tensor describing token
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
            # Tensor parallelism: split the attention heads across devices.
            #
            #   fused_qkv_a_proj (q_a_proj + kv_a_proj_with_mqa)  replicated
            #   q_b_proj                       column-parallel (rowwise on
            #                                  [N*D_qk, Lq]) -> [N/n*D_qk, Lq]
            #   kv_b_proj                      column-parallel (rowwise on
            #                                  [N*(Dn+Dv), Lkv])
            #   o_proj                         row-parallel (columnwise on
            #                                  [H, N*Dv]) -> all-reduce over TP
            #
            # The per-head row layout of q_b_proj / kv_b_proj is preserved
            # because n_heads is divisible by num_devices, so an even rowwise
            # split lands on head boundaries. Each FP8 block-wise scale is
            # sharded along the same axis as its weight.
            self._sharding_strategy = strategy

            if self.n_heads % strategy.num_devices != 0:
                raise ValueError(
                    f"Number of attention heads ({self.n_heads}) must be"
                    f" divisible by the number of devices ({strategy.num_devices})."
                )

            n = strategy.num_devices

            # q_a path (fused with kv_a) is replicated: it projects the full
            # hidden state down to the LoRA rank before the head split.
            self.q_a_proj.sharding_strategy = ShardingStrategy.replicate(n)
            self.q_a_proj_scale.sharding_strategy = ShardingStrategy.replicate(
                n
            )
            self.q_a_layernorm.weight.sharding_strategy = (
                ShardingStrategy.replicate(n)
            )

            # q_b projects the LoRA rank up to per-head query dims: shard the
            # output rows (column-parallel).
            self.q_b_proj.sharding_strategy = ShardingStrategy.rowwise(n)
            self.q_b_proj_scale.sharding_strategy = ShardingStrategy.rowwise(n)

            # kv_a path is replicated (shared MQA latent + rope).
            self.kv_a_proj_layernorm.sharding_strategy = (
                ShardingStrategy.replicate(n)
            )
            self.kv_a_proj_with_mqa.sharding_strategy = (
                ShardingStrategy.replicate(n)
            )
            self.kv_a_proj_with_mqa_scale.sharding_strategy = (
                ShardingStrategy.replicate(n)
            )

            # kv_b projects the KV latent up to per-head K/V dims: shard the
            # output rows (column-parallel).
            self.kv_b_proj.sharding_strategy = ShardingStrategy.rowwise(n)
            self.kv_b_proj_scale.sharding_strategy = ShardingStrategy.rowwise(n)

            # o_proj contracts the per-head value dim back to hidden: shard the
            # input columns (row-parallel). The Linear handles its block-wise
            # weight scale, which is sharded along the same (K) axis.
            self.o_proj.sharding_strategy = ShardingStrategy.columnwise(n)
        elif strategy.is_replicate:
            # Data parallelism: replicate the entire module's weights to each device.
            self._sharding_strategy = strategy

            weights = [
                self.q_a_proj,
                self.q_a_proj_scale,
                self.q_a_layernorm.weight,
                self.q_b_proj,
                self.q_b_proj_scale,
                self.kv_a_proj_layernorm,
                self.kv_a_proj_with_mqa,
                self.kv_a_proj_with_mqa_scale,
                self.kv_b_proj,
                self.kv_b_proj_scale,
                self.o_proj.weight,
            ]

            if self.o_proj.input_scale is not None:
                weights.append(self.o_proj.input_scale)
            if self.o_proj.weight_scale is not None:
                weights.append(self.o_proj.weight_scale)

            for weight in weights:
                weight.sharding_strategy = ShardingStrategy.replicate(
                    strategy.num_devices
                )
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are "
                "supported for LatentAttentionWithRopeFp8"
            )

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[LatentAttentionWithRopeFp8]:
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
            q_a_proj_shards = self.q_a_proj.shard(devices)
            q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
            q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                devices
            )
            q_b_proj_shards = self.q_b_proj.shard(devices)
            q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_a_proj_with_mqa_scale_shards = (
                self.kv_a_proj_with_mqa_scale.shard(devices)
            )
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)

            o_proj_weight_shards = self.o_proj.weight.shard(devices)
            if self.o_proj.input_scale is not None:
                o_proj_scale_shards = self.o_proj.input_scale.shard(devices)
            if self.o_proj.weight_scale is not None:
                o_proj_weight_scale_shards = self.o_proj.weight_scale.shard(
                    devices
                )

            shards = []
            for shard_idx, device in enumerate(devices):
                sharded = LatentAttentionWithRopeFp8(
                    rope=self.rope,
                    num_attention_heads=self.n_heads
                    // self.sharding_strategy.num_devices,
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    quant_config=self.quant_config,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                    norm_dtype=self.norm_dtype,
                )

                sharded.q_a_proj = q_a_proj_shards[shard_idx]
                sharded.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
                sharded.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                    shard_idx
                ]
                sharded.q_b_proj = q_b_proj_shards[shard_idx]
                sharded.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]

                sharded.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                sharded.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                sharded.kv_a_proj_with_mqa_scale = (
                    kv_a_proj_with_mqa_scale_shards[shard_idx]
                )
                sharded.kv_b_proj = kv_b_proj_shards[shard_idx]
                sharded.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]

                sharded.o_proj.weight = o_proj_weight_shards[shard_idx]
                if self.o_proj.input_scale is not None:
                    sharded.o_proj.input_scale = o_proj_scale_shards[shard_idx]
                if self.o_proj.weight_scale is not None:
                    sharded.o_proj.weight_scale = o_proj_weight_scale_shards[
                        shard_idx
                    ]

                shards.append(sharded)

            return shards
        elif self.sharding_strategy.is_replicate:
            # Replicate full weights to each device (no head split).
            q_a_proj_shards = self.q_a_proj.shard(devices)
            q_a_proj_scale_shards = self.q_a_proj_scale.shard(devices)
            q_a_layernorm_weight_shards = self.q_a_layernorm.weight.shard(
                devices
            )
            q_b_proj_shards = self.q_b_proj.shard(devices)
            q_b_proj_scale_shards = self.q_b_proj_scale.shard(devices)

            kv_a_proj_layernorm_shards = self.kv_a_proj_layernorm.shard(devices)
            kv_a_proj_with_mqa_shards = self.kv_a_proj_with_mqa.shard(devices)
            kv_a_proj_with_mqa_scale_shards = (
                self.kv_a_proj_with_mqa_scale.shard(devices)
            )
            kv_b_proj_shards = self.kv_b_proj.shard(devices)
            kv_b_proj_scale_shards = self.kv_b_proj_scale.shard(devices)
            o_proj_weight_shards = self.o_proj.weight.shard(devices)

            if self.o_proj.input_scale is not None:
                o_proj_scale_shards = self.o_proj.input_scale.shard(devices)
            if self.o_proj.weight_scale is not None:
                o_proj_weight_scale_shards = self.o_proj.weight_scale.shard(
                    devices
                )

            replicas: list[LatentAttentionWithRopeFp8] = []
            for shard_idx, device in enumerate(devices):
                replica = LatentAttentionWithRopeFp8(
                    rope=self.rope,
                    num_attention_heads=self.n_heads,  # DP keeps full heads
                    num_key_value_heads=self.num_key_value_heads,
                    hidden_size=self.hidden_size,
                    kv_params=self.kv_params,
                    quant_config=self.quant_config,
                    devices=[device],
                    graph_mode=self.graph_mode,
                    linear_cls=self.linear_cls,
                    scale=self._scale,
                    q_lora_rank=self.q_lora_rank,
                    kv_lora_rank=self.kv_lora_rank,
                    qk_nope_head_dim=self.qk_nope_head_dim,
                    qk_rope_head_dim=self.qk_rope_head_dim,
                    v_head_dim=self.v_head_dim,
                    buffer_size=self.BUFFER_TOK_SIZE,
                    norm_dtype=self.norm_dtype,
                )

                replica.q_a_proj = q_a_proj_shards[shard_idx]
                replica.q_a_proj_scale = q_a_proj_scale_shards[shard_idx]
                replica.q_a_layernorm.weight = q_a_layernorm_weight_shards[
                    shard_idx
                ]
                replica.q_b_proj = q_b_proj_shards[shard_idx]
                replica.q_b_proj_scale = q_b_proj_scale_shards[shard_idx]

                replica.kv_a_proj_layernorm = kv_a_proj_layernorm_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa = kv_a_proj_with_mqa_shards[
                    shard_idx
                ]
                replica.kv_a_proj_with_mqa_scale = (
                    kv_a_proj_with_mqa_scale_shards[shard_idx]
                )
                replica.kv_b_proj = kv_b_proj_shards[shard_idx]
                replica.kv_b_proj_scale = kv_b_proj_scale_shards[shard_idx]
                replica.o_proj.weight = o_proj_weight_shards[shard_idx]
                if self.o_proj.input_scale is not None:
                    replica.o_proj.input_scale = o_proj_scale_shards[shard_idx]
                if self.o_proj.weight_scale is not None:
                    replica.o_proj.weight_scale = o_proj_weight_scale_shards[
                        shard_idx
                    ]

                replicas.append(replica)

            return replicas
        else:
            raise ValueError(
                "Only tensor parallel or replicate sharding strategies are supported for LatentAttentionWithRope"
            )

    @property
    def wqkv(self) -> tuple[TensorValue, TensorValue]:
        """The concatenation of q_a_proj and kv_a_proj_with_mqa weight vectors."""
        wqkv = ops.concat((self.q_a_proj, self.kv_a_proj_with_mqa))
        wqkv_scale = ops.concat(
            (self.q_a_proj_scale, self.kv_a_proj_with_mqa_scale)
        )

        return (wqkv, wqkv_scale)

    @property
    def _kv_b_proj_weight(self) -> TensorValue:
        """Returns `kv_b_proj` reshaped for per-head projection slicing."""
        kv_b_proj_weight: TensorValue = self.kv_b_proj.transpose(0, 1)
        kv_b_proj_weight = kv_b_proj_weight.reshape(
            (self.kv_lora_rank, self.n_heads, -1)
        )
        return kv_b_proj_weight

    def _gather_per_head_scale(
        self, start_row_offset: int, n_rows: int
    ) -> TensorValue:
        """Gathers per-head B-scale chunks from the flat on-disk
        `kv_b_proj_scale` for all heads in a single op.

        For each head `h` and chunk `k` of `g` rows, the on-disk
        scale row is `(h * per_head_row + start_row_offset + k * g) //
        block_m`. When `block_k > g` (finer-than-on-disk N
        granularity), gathered cols are replicated by `block_k // g`
        so the kernel — called with `n/k_scale_granularity = g` —
        sees a matching block count along the on-disk K axis.

        Returns a `[H, n_chunks, n_cols_kernel]` tensor.
        """
        g = self._b_scale_granularity
        block_m = int(self.weight_block_size[0])
        block_k = int(self.weight_block_size[1])
        per_head_row = self.qk_nope_head_dim + self.v_head_dim
        n_chunks = ceildiv(n_rows, g)
        heads = np.arange(self.n_heads, dtype=np.int32)
        chunks = np.arange(n_chunks, dtype=np.int32)
        row_indices = (
            (
                heads[:, None] * per_head_row
                + start_row_offset
                + chunks[None, :] * g
            )
            // block_m
        ).reshape(-1)
        gathered = ops.gather(
            self.kv_b_proj_scale,
            ops.constant(
                row_indices,
                DType.int32,
                device=self.kv_b_proj_scale.device,
            ),
            axis=0,
        )
        n_cols_on_disk = int(self.kv_b_proj_scale.shape[1])
        gathered = gathered.reshape((self.n_heads, n_chunks, n_cols_on_disk))
        col_repeat = block_k // g
        if col_repeat == 1:
            return gathered
        col_indices = np.repeat(
            np.arange(n_cols_on_disk, dtype=np.int32), col_repeat
        )
        return ops.gather(
            gathered,
            ops.constant(
                col_indices,
                DType.int32,
                device=self.kv_b_proj_scale.device,
            ),
            axis=2,
        )

    @property
    def w_uk(self) -> tuple[TensorValue, TensorValue]:
        """Decode K-up projection: weight `[H, R, Dn]`, scale
        `[H, R/block_k_kernel, ceildiv(Dn, g)]` after transpose.

        The batched FP8 matmul reads `b_scales` as `[H, N_blk,
        K_blk]`. For `Q @ w_uk` the matmul has `N = R` and
        `K = Dn`, so the gather returns `[H, K_blk, N_blk]` and we
        transpose the trailing two axes to match the kernel layout.
        """
        w_uk = self._kv_b_proj_weight[..., : self.qk_nope_head_dim].transpose(
            0, 1
        )
        w_uk_scale = self._gather_per_head_scale(
            start_row_offset=0, n_rows=self.qk_nope_head_dim
        ).transpose(1, 2)
        return (w_uk, w_uk_scale)

    @property
    def w_uv(self) -> tuple[TensorValue, TensorValue]:
        """Decode V-up projection: weight `[H, Dv, R]`, scale
        `[H, ceildiv(Dv, g), R/block_k_kernel]`.

        For `raw_out @ w_uv`: `N_matmul = Dv`, `K_matmul = R`.
        The gather chunks `Dv` at granularity `g`, so the rows axis
        is already `N_blk_matmul` and no transpose is needed.
        """
        w_uv = self._kv_b_proj_weight[..., self.qk_nope_head_dim :].permute(
            [1, 2, 0]
        )
        w_uv_scale = self._gather_per_head_scale(
            start_row_offset=self.qk_nope_head_dim, n_rows=self.v_head_dim
        )
        return (w_uv, w_uv_scale)

    @property
    def w_k(self) -> tuple[TensorValue, TensorValue]:
        """Prefill K-up projection: weight `[H*Dn, R]`, scale
        `[H*ceildiv(Dn, g), R/block_k_kernel]`."""
        w_k = (
            self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
            .permute([1, 2, 0])
            .reshape((-1, self.kv_lora_rank))
        )
        block_k_kernel = self._b_scale_granularity
        w_k_scale = self._gather_per_head_scale(
            start_row_offset=0, n_rows=self.qk_nope_head_dim
        ).reshape((-1, self.kv_lora_rank // block_k_kernel))
        return (w_k, w_k_scale)

    def _mla_impl(
        self,
        xq: TensorValue,
        kv: TensorValue,
        kv_collection: PagedCacheValues,
        layer_idx: TensorValue,
        input_row_offsets: TensorValue,
        freqs_cis: TensorValue,
        kv_a_proj_layernorm: TensorValue,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> TensorValue:
        # Prepare the inputs and weights for the prefill and decode branches.
        attn_kwargs: dict[str, Any] = {
            "q": xq,
            "kv": kv,
            "input_row_offsets": input_row_offsets,
            "freqs_cis": freqs_cis,
            "kv_norm_gamma": kv_a_proj_layernorm,
            "kv_params": self.kv_params,
            "kv_collection": kv_collection,
            "layer_idx": layer_idx,
            "epsilon": 1e-6,
            "mask_variant": MHAMaskVariant.CAUSAL_MASK,
            "scale": self.scale,
            "v_head_dim": self.v_head_dim,
            "quant_config": self.quant_config,
            # When the per-head row count straddles the on-disk block,
            # the kernel must use the finer granularity at which every
            # per-head K-chunk fits in a single on-disk scale block
            # instead of the on-disk `weight_scale.block_size`.
            "scale_granularity_override": self._b_scale_granularity,
        }

        w_k, w_k_scale = self.w_k
        w_uk, w_uk_scale = self.w_uk
        w_uv, w_uv_scale = self.w_uv
        if self.graph_mode in ["prefill", "auto"]:
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
            attn_kwargs["w_k"] = w_k
            attn_kwargs["w_k_scale"] = w_k_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode in ["decode", "auto"]:
            attn_kwargs["w_uk"] = w_uk
            attn_kwargs["w_uk_scale"] = w_uk_scale
            attn_kwargs["w_uv"] = w_uv
            attn_kwargs["w_uv_scale"] = w_uv_scale
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

        return result.reshape((-1, self.n_heads * self.v_head_dim))

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> TensorValue:
        # First FP8 matmul: x @ q_a_proj.T, fused with x @ kv_a_proj_with_mqa.T
        wqkv, wqkv_scale = self.wqkv
        qkv = quantized_matmul(
            x=x,
            weight=wqkv,
            weight_scale=wqkv_scale,
            input_scale=None,  # Dynamic scaling
            quant_config=self.quant_config,
        )

        q_a_out, kv = ops.split(
            qkv, [self.q_lora_rank, self.cache_head_dim], axis=1
        )

        # Apply layer norm
        q_a_normed = self.q_a_layernorm(q_a_out)

        # Second FP8 matmul: q_a_normed @ q_b_proj.T
        xq = quantized_matmul(
            x=q_a_normed,
            weight=self.q_b_proj,
            weight_scale=self.q_b_proj_scale,
            input_scale=None,  # Dynamic scaling
            quant_config=self.quant_config,
        )

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

        # QK RoPE and RMSNorm of K cache are handled inside the MLA kernel.
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
        )

        return self.o_proj(attn_out)


class TensorParallelLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
    """Distributed tensor parallel implementation of FP8 Latent Attention with
    Rope.

    Attention heads are split across devices: ``q_b_proj`` / ``kv_b_proj`` are
    column-parallel (each device owns ``num_heads / num_devices`` heads) and
    ``o_proj`` is row-parallel, so a final all-reduce sums the per-device
    partial outputs. ``q_a_proj`` / ``kv_a_proj_with_mqa`` and the layernorm
    weights are replicated. Note that, as with the bf16 variant, tensor
    parallelism duplicates the KV-cache across all devices.

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
        """Creates per-device FP8 MLA prefill metadata for tensor-parallel execution.

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


class DataParallelLatentAttentionWithRopeFp8(LatentAttentionWithRopeFp8):
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
        """Creates per-device FP8 MLA prefill metadata for data-parallel execution.

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
            if xs[i].shape[0] == 0:
                outs.append(xs[i])
                continue

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
