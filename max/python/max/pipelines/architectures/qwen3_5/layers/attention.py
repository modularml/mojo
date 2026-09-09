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

"""Qwen3.5 full attention layer.

Differences from Qwen3 attention:
- q_proj outputs 2x width: [hidden_size -> num_heads * head_dim * 2], laid out
  per-head interleaved as [head0 Q | head0 gate | head1 Q | head1 gate | ...].
  A block split into [all Q | all gate] compiles, runs, and silently corrupts.
- Partial RoPE: only partial_rotary_factor * head_dim dimensions get rotation.
- RMSNorm on Q/K uses (1 + weight) offset (weight_offset=1.0).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import numpy as np
from max.dtype import DType
from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.graph.weight import Segment
from max.nn.attention import MHAMaskVariant
from max.nn.kernels import (
    flash_attention_ragged,
    fused_qk_ragged_rope,
    rope_split_store_ragged,
    store_k_cache_ragged,
    store_v_cache_ragged,
)
from max.nn.kv_cache import KVCacheParams, PagedCacheValues
from max.nn.layer import Module, Shardable
from max.nn.linear import Linear
from max.nn.norm import RMSNorm
from max.nn.quant_config import QuantConfig
from max.nn.rotary_embedding import RotaryEmbedding
from max.nn.stacked_linear import StackedLinear


def _projection(stack: StackedLinear, name: str) -> Linear:
    """Returns one projection of an unfused stack.

    The children are set as dynamically named attributes, so this narrows
    what would otherwise be an untyped attribute read.
    """
    child = getattr(stack, name)
    assert isinstance(child, Linear)
    return child


class Qwen3_5Attention(Module, Shardable):
    """Full attention layer for Qwen3.5 with gated output and partial RoPE.

    This attention layer differs from standard GQA in several ways:
    1. q_proj produces 2x output width, interleaved per head as
       [head0 Q | head0 gate | ...]; each head's gate is applied by sigmoid to
       that head's attention output before the output projection.
    2. Only partial_rotary_factor (25%) of head_dim gets rotary embedding.
    3. QK RMSNorm uses (1 + weight) scaling (weight_offset=1.0).
    """

    def __init__(
        self,
        *,
        rope: RotaryEmbedding,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        head_dim: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        dtype: DType = DType.float32,
        devices: list[DeviceRef],
        linear_cls: Callable[..., Linear] = Linear,
        scale: float | None = None,
        partial_rotary_factor: float = 0.25,
        has_bias: bool = False,
        norm_dtype: DType | None = None,
        norm_eps: float = 1e-6,
        quant_config: QuantConfig | None = None,
    ) -> None:
        super().__init__()
        self.rope = rope
        self.n_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.layer_idx = layer_idx
        self.kv_params = kv_params
        self.has_bias = has_bias
        self.devices = devices
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.partial_rotary_factor = partial_rotary_factor
        self.rotary_dim = int(head_dim * partial_rotary_factor)
        self.norm_dtype = norm_dtype if norm_dtype is not None else dtype
        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / self.head_dim)
        )
        self.norm_eps = norm_eps
        self.quant_config = quant_config
        self.linear_cls = linear_cls
        self._sharding_strategy: ShardingStrategy | None = None

        # QK norm with (1 + weight) offset
        self.q_norm = RMSNorm(
            self.head_dim,
            dtype=self.norm_dtype,
            eps=norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )
        self.k_norm = RMSNorm(
            self.head_dim,
            dtype=self.norm_dtype,
            eps=norm_eps,
            weight_offset=1.0,
            multiply_before_cast=False,
        )

        self.q_weight_dim = head_dim * num_attention_heads
        self.kv_weight_dim = head_dim * num_key_value_heads

        # q_proj outputs 2x width when gated (query + gate)
        self.qkv_proj = StackedLinear(
            in_dim=hidden_size,
            out_dims=[
                self.q_weight_dim * 2,
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
            has_bias=has_bias,
            quant_config=quant_config,
        )

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        """Get the layer's sharding strategy."""
        return self._sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        """Splits the layer by attention head, and propagates that to weights.

        Args:
            strategy: Must be tensor-parallel; there is no data-parallel path.

        Raises:
            ValueError: If the strategy is not tensor-parallel, or if the
                device count does not divide either head count.
        """
        if not strategy.is_tensor_parallel:
            raise ValueError(
                "Qwen3_5Attention supports only tensor-parallel sharding, got "
                f"{strategy}"
            )
        num_devices = strategy.num_devices
        for count, name in (
            (self.n_heads, "num_attention_heads"),
            (self.num_key_value_heads, "num_key_value_heads"),
        ):
            if count % num_devices:
                raise ValueError(
                    f"Qwen3_5Attention {name} ({count}) must be divisible by "
                    f"the device count ({num_devices})"
                )

        self._sharding_strategy = strategy

        # `q_proj` packs each head as one `[query | gate]` block of
        # `2 * head_dim` consecutive rows, so a head-aware split of blocks
        # that wide keeps every gate with the query it multiplies. Treating
        # the weight as `[all queries | all gates]` compiles and silently
        # pairs each head's output with another head's gate.
        _projection(
            self.qkv_proj, "q_proj"
        ).sharding_strategy = ShardingStrategy.segmented(
            num_devices,
            axis=0,
            segments=(Segment.head_aware(self.n_heads, self.head_dim * 2),),
        )
        kv_segments = (
            Segment.head_aware(self.num_key_value_heads, self.head_dim),
        )
        for name in ("k_proj", "v_proj"):
            _projection(
                self.qkv_proj, name
            ).sharding_strategy = ShardingStrategy.segmented(
                num_devices, axis=0, segments=kv_segments
            )

        # Q/K norm gamma is per head-dim element, shared by every head.
        replicate = ShardingStrategy.replicate(num_devices)
        self.q_norm.sharding_strategy = replicate
        self.k_norm.sharding_strategy = replicate

        self.o_proj.sharding_strategy = ShardingStrategy.head_aware_columnwise(
            num_devices, self.n_heads, self.head_dim
        )

    def shard(self, devices: Iterable[DeviceRef]) -> list[Qwen3_5Attention]:
        """Creates one per-device view of this layer, split by head.

        Args:
            devices: Devices to place the shards on.

        Returns:
            One :class:`Qwen3_5Attention` per device, each dimensioned for
            its own head slice.

        Raises:
            ValueError: If no sharding strategy has been set.
        """
        if self._sharding_strategy is None:
            raise ValueError(
                "Qwen3_5Attention cannot be sharded because no sharding "
                "strategy was provided."
            )
        devices = list(devices)
        num_devices = len(devices)

        qkv_shards = self.qkv_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)
        q_norm_shards = self.q_norm.shard(devices)
        k_norm_shards = self.k_norm.shard(devices)

        shards: list[Qwen3_5Attention] = []
        for i, device in enumerate(devices):
            shard = Qwen3_5Attention(
                rope=self.rope,
                num_attention_heads=self.n_heads // num_devices,
                num_key_value_heads=self.num_key_value_heads // num_devices,
                hidden_size=self.hidden_size,
                head_dim=self.head_dim,
                kv_params=self.kv_params,
                layer_idx=self.layer_idx,
                dtype=self.dtype,
                devices=[device],
                linear_cls=self.linear_cls,
                scale=self.scale,
                partial_rotary_factor=self.partial_rotary_factor,
                has_bias=self.has_bias,
                norm_dtype=self.norm_dtype,
                norm_eps=self.norm_eps,
                quant_config=self.quant_config,
            )
            shard.qkv_proj = qkv_shards[i]
            shard.o_proj = o_proj_shards[i]
            shard.q_norm = q_norm_shards[i]
            shard.k_norm = k_norm_shards[i]
            shards.append(shard)
        return shards

    def _full_width_freqs(
        self, freqs_cis: TensorValue, dtype: DType
    ) -> TensorValue:
        """Widens the partial-rotary table to ``head_dim`` for the fused store.

        ``rope_split_store`` rotates every one of ``head_dim`` lanes against a
        ``[positions, head_dim]`` table, while Qwen3.5 rotates only
        ``rotary_dim`` of them. Q and K are already rearranged to
        ``[NoPE | RoPE]``, so prepending the identity rotation ``(cos, sin) =
        (1, 0)`` over the NoPE lanes reproduces partial rotary exactly. The
        table is a graph constant, so the widening costs build time, not
        runtime.
        """
        freqs_cis = ops.cast(freqs_cis, dtype).to(self.devices[0])
        nope_dim = self.head_dim - self.rotary_dim
        if nope_dim == 0:
            return freqs_cis
        identity = ops.cast(
            ops.constant(
                np.tile([1.0, 0.0], nope_dim // 2).astype(np.float32),
                DType.float32,
                device=self.devices[0],
            ),
            dtype,
        )
        identity = ops.broadcast_to(
            ops.unsqueeze(identity, 0), [freqs_cis.shape[0], nope_dim]
        )
        return ops.concat((identity, freqs_cis), axis=-1)

    def __call__(
        self,
        layer_idx: TensorValue,
        x: TensorValue,
        kv_collection: PagedCacheValues,
        freqs_cis: TensorValue,
        input_row_offsets: TensorValue,
        freq_row_ids: TensorValue | None = None,
    ) -> TensorValue:
        """Forward pass through the gated full attention layer.

        Args:
            layer_idx: Layer index for KV cache.
            x: Input hidden states [total_seq_len, hidden_size].
            kv_collection: KV cache handle.
            freqs_cis: RoPE frequency table.
            input_row_offsets: Ragged offsets for batched sequences.
            freq_row_ids: ``[1, total_seq_len]`` row of ``freqs_cis`` to use
                for each token. Set when the table is per token (M-RoPE);
                ``None`` leaves the kernels on their default
                ``cache_length + token_idx`` lookup, which is only right when
                the table is indexed by absolute position.

        Returns:
            Output hidden states [total_seq_len, hidden_size].
        """
        total_seq_len = x.shape[0]

        qkv = self.qkv_proj(x)
        q_out, key, value = ops.split(
            qkv,
            [
                self.q_weight_dim * 2,
                self.kv_weight_dim,
                self.kv_weight_dim,
            ],
            axis=-1,
        )

        # Split into query and gate.
        # The weight layout is interleaved per head:
        # [head0_query, head0_gate, head1_query, head1_gate, ...]
        # Reshape to [total_seq_len, n_heads, head_dim * 2], then split on last axis.
        q_out_reshaped = ops.reshape(
            q_out, shape=[-1, self.n_heads, self.head_dim * 2]
        )
        # query: [total_seq_len, n_heads, head_dim]
        query = ops.slice_tensor(
            q_out_reshaped,
            [slice(None), slice(None), slice(0, self.head_dim)],
        )
        # gate: [total_seq_len, n_heads, head_dim] -> flatten to [total_seq_len, n_heads * head_dim]
        gate = ops.reshape(
            ops.slice_tensor(
                q_out_reshaped,
                [
                    slice(None),
                    slice(None),
                    slice(self.head_dim, self.head_dim * 2),
                ],
            ),
            shape=[-1, self.n_heads * self.head_dim],
        )

        # query is already [total_seq_len, n_heads, head_dim] from the reshape above

        # Apply Q/K norms in original dim ordering (weight elements align
        # with the HF convention where RoPE dims come first).
        query = self.q_norm(query)

        # Reshape K for per-head norm: [total_seq_len, n_kv_heads, head_dim]
        key = ops.reshape(
            key, shape=[-1, self.num_key_value_heads, self.head_dim]
        )
        key = self.k_norm(key)

        # Reshape V: [total_seq_len, n_kv_heads, head_dim]
        value = ops.reshape(
            value, shape=[-1, self.num_key_value_heads, self.head_dim]
        )

        # Rearrange Q and K head dims for the fused RoPE kernel.
        #
        # Two transformations needed:
        # 1. NoPE/RoPE swap: HF puts RoPE dims first [RoPE_64 | NoPE_192],
        #    but the kernel rotates the LAST dims: [NoPE_192 | RoPE_64].
        # 2. Interleave RoPE dims: HF uses rotate_half (non-interleaved)
        #    which pairs (dim_i, dim_{i+D/2}), but the kernel with
        #    interleaved=True pairs consecutive dims (dim_{2i}, dim_{2i+1}).
        #    Rearrange [x0,..,x31,x32,..,x63] → [x0,x32,x1,x33,..,x31,x63]
        #    so the kernel's consecutive-pair rotation matches HF's halves.
        #
        # The interleaving is equivalent to: reshape [rd] → [2, half_rd],
        # transpose to [half_rd, 2], flatten back to [rd]. This uses fewer
        # graph ops than the slice+concat approach (5 vs 9 per Q/K).
        rd = self.rotary_dim  # 64
        half_rd = rd // 2  # 32
        q_rope = ops.slice_tensor(
            query, [slice(None), slice(None), slice(0, rd)]
        )
        q_pass = ops.slice_tensor(
            query, [slice(None), slice(None), slice(rd, self.head_dim)]
        )
        q_rope_interleaved = ops.reshape(
            ops.transpose(
                ops.reshape(q_rope, [-1, self.n_heads, 2, half_rd]),
                -1,
                -2,
            ),
            [-1, self.n_heads, rd],
        )
        query = ops.concat([q_pass, q_rope_interleaved], axis=-1)

        k_rope = ops.slice_tensor(key, [slice(None), slice(None), slice(0, rd)])
        k_pass = ops.slice_tensor(
            key, [slice(None), slice(None), slice(rd, self.head_dim)]
        )
        k_rope_interleaved = ops.reshape(
            ops.transpose(
                ops.reshape(k_rope, [-1, self.num_key_value_heads, 2, half_rd]),
                -1,
                -2,
            ),
            [-1, self.num_key_value_heads, rd],
        )
        key = ops.concat([k_pass, k_rope_interleaved], axis=-1)

        if self.kv_params.is_fp8_kv_dtype:
            # The bf16 branch below stores raw K and ropes it in place inside
            # the cache, which on FP8 would round twice around the rope. The
            # fused rope+store ropes in registers and casts once, so it is the
            # right topology here with or without M-RoPE position ids. Q is
            # emitted in the cache dtype for flash attention's dtype guard.
            qkv = ops.concat(
                (
                    ops.reshape(query, [total_seq_len, -1]),
                    ops.reshape(key, [total_seq_len, -1]),
                    ops.reshape(value, [total_seq_len, -1]),
                ),
                axis=-1,
            )
            query = rope_split_store_ragged(
                kv_params=self.kv_params,
                qkv=qkv,
                input_row_offsets=input_row_offsets,
                freqs_cis=self._full_width_freqs(freqs_cis, qkv.dtype),
                kv_collection=kv_collection,
                layer_idx=layer_idx,
                n_heads=self.n_heads,
                interleaved=self.rope.interleaved,
                position_ids=freq_row_ids,
                q_out_dtype=self.kv_params.dtype,
            )
            query = ops.reshape(
                query, [total_seq_len, self.n_heads, self.head_dim]
            )
        else:
            # Write rearranged, normed K and V to cache.
            store_k_cache_ragged(
                kv_collection, key, input_row_offsets, layer_idx
            )
            store_v_cache_ragged(
                kv_collection, value, input_row_offsets, layer_idx
            )

            # Apply RoPE (kernel rotates last rotary_dim dims of Q and K in cache)
            freqs_cis = ops.cast(freqs_cis, query.dtype).to(query.device)
            query = fused_qk_ragged_rope(
                self.kv_params,
                query,
                input_row_offsets,
                kv_collection,
                freqs_cis,
                layer_idx,
                interleaved=self.rope.interleaved,
                position_ids=freq_row_ids,
            )

        # Flash attention. `output_dtype` is pinned to the activation dtype so
        # an FP8 query still yields a bf16 attention output for the gate and
        # o_proj; it is a no-op on the bf16 path.
        attn_out = flash_attention_ragged(
            self.kv_params,
            input=query,
            kv_collection=kv_collection,
            layer_idx=layer_idx,
            input_row_offsets=input_row_offsets,
            mask_variant=MHAMaskVariant.CAUSAL_MASK,
            scale=self.scale,
            output_dtype=x.dtype,
        )

        # Reshape attention output: [total_seq_len, n_heads * head_dim]
        attn_out = ops.reshape(attn_out, shape=[total_seq_len, -1])

        # Apply sigmoid gate
        gate_sigmoid = ops.sigmoid(gate)
        attn_out = attn_out * gate_sigmoid

        # Output projection
        return self.o_proj(attn_out)
