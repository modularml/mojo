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
from __future__ import annotations

from collections.abc import Iterable

from max.graph import DeviceRef, ShardingStrategy, TensorValue, ops
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kernels import flash_attention_ragged_gpu
from max.nn.layer import Module
from max.nn.linear import Linear
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    apply_multidimensional_rope,
)

from ..layers.rms_norm import Gemma4RMSNorm
from ..model_config import Gemma4ForConditionalGenerationConfig


class Gemma4VisionAttention(Module):
    """Ragged multi-head self-attention for the SigLIP vision encoder.

    Differences from the text-side attention:
    - Operates on packed (ragged) sequences: input is ``[total_patches, hidden]``
      rather than ``[batch, seq, hidden]``.
    - Uses ``flash_attention_ragged_gpu`` with ``NULL_MASK`` (bidirectional).
    - Applies per-head RMS norms to Q and K (Gemma4-style).
    - Applies a scale-free RMS norm to V (no learnable weight).
    - Applies 2-D multidimensional RoPE to Q and K.
    - Attention scale is fixed to 1.0 (HuggingFace convention for this model).

    Weight names match the checkpoint under
    ``model.vision_tower.encoder.layers.N.self_attn.*``:
    ``q_proj``, ``k_proj``, ``v_proj``, ``o_proj``, ``q_norm``, ``k_norm``.
    """

    def __init__(
        self,
        config: Gemma4ForConditionalGenerationConfig,
        layer_idx: int,
        device: DeviceRef | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        vision_cfg = config.vision_config
        assert vision_cfg is not None
        vision_dtype = config.unquantized_dtype

        self.layer_idx = layer_idx
        self.device = device if device is not None else config.devices[0]
        self.head_dim = vision_cfg.head_dim
        self.num_attention_heads = vision_cfg.num_attention_heads
        self.num_key_value_heads = vision_cfg.num_key_value_heads
        self.v_norm_eps = vision_cfg.rms_norm_eps

        self.q_proj = Linear(
            vision_cfg.hidden_size,
            self.num_attention_heads * self.head_dim,
            has_bias=vision_cfg.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )
        self.k_proj = Linear(
            vision_cfg.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=vision_cfg.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )
        self.v_proj = Linear(
            vision_cfg.hidden_size,
            self.num_key_value_heads * self.head_dim,
            has_bias=vision_cfg.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )
        self.o_proj = Linear(
            self.num_attention_heads * self.head_dim,
            vision_cfg.hidden_size,
            has_bias=vision_cfg.attention_bias,
            dtype=vision_dtype,
            device=self.device,
        )

        # Per-head RMS norms on Q and K (have learnable weights).
        self.q_norm = Gemma4RMSNorm(
            self.head_dim, vision_dtype, eps=vision_cfg.rms_norm_eps
        )
        self.k_norm = Gemma4RMSNorm(
            self.head_dim, vision_dtype, eps=vision_cfg.rms_norm_eps
        )
        self.v_norm = Gemma4RMSNorm(
            self.head_dim,
            vision_dtype,
            eps=vision_cfg.rms_norm_eps,
            with_weight=False,
        )

    @property
    def wqkv(self) -> TensorValue:
        return ops.concat(
            (self.q_proj.weight, self.k_proj.weight, self.v_proj.weight)
        )

    def __call__(
        self,
        hidden_states: TensorValue,
        freqs_cis: TensorValue,
        cu_seqlens: TensorValue,
        max_seq_len: TensorValue,
    ) -> TensorValue:
        """Compute bidirectional packed self-attention over flat patch tokens.

        Args:
            hidden_states: Packed patch embeddings,
                shape ``[total_patches, hidden_size]``.
            freqs_cis: Pre-computed vision RoPE frequencies,
                shape ``[total_patches, head_dim // 2, 2]``.
            cu_seqlens: Cumulative sequence lengths (image boundaries),
                shape ``[num_images + 1]``, dtype uint32.
            max_seq_len: Maximum patches per image (scalar uint32, on CPU).

        Returns:
            Output embeddings, shape ``[total_patches, hidden_size]``.
        """
        total_patches = hidden_states.shape[0]

        # Project to Q, K, V.
        x = hidden_states @ self.wqkv.T
        xq, xk, xv = ops.split(
            x,
            [
                self.num_attention_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
                self.num_key_value_heads * self.head_dim,
            ],
            axis=-1,
        )

        # Reshape to [total_patches, num_heads, head_dim].
        xq = ops.reshape(
            xq, [total_patches, self.num_attention_heads, self.head_dim]
        )
        xk = ops.reshape(
            xk, [total_patches, self.num_key_value_heads, self.head_dim]
        )
        xv = ops.reshape(
            xv, [total_patches, self.num_key_value_heads, self.head_dim]
        )

        # Apply per-head RMS norms.
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)
        xv = self.v_norm(xv)

        # Add num_heads broadcast dim: [p, head_dim//2, 2] → [p, 1, head_dim//2, 2].
        # Use total_patches (symbolic Dim) to preserve si64 strides.
        freqs_cis_bcast = ops.reshape(
            freqs_cis, [total_patches, 1, self.head_dim // 2, 2]
        )
        # Apply 2-D multidimensional RoPE to Q and K.
        xq = apply_multidimensional_rope(xq, freqs_cis_bcast, ndim=2)
        xk = apply_multidimensional_rope(xk, freqs_cis_bcast, ndim=2)

        # Bidirectional packed flash attention (no causal mask, scale=1.0).
        output = flash_attention_ragged_gpu(
            xq,
            xk,
            xv,
            input_row_offsets=cu_seqlens,
            max_seq_len=max_seq_len,
            mask_variant=MHAMaskVariant.NULL_MASK,
            scale=1.0,
        )
        # output: [total_patches, num_heads, head_dim]

        output = ops.reshape(output, [total_patches, -1])
        return self.o_proj(output)

    @property
    def sharding_strategy(self) -> ShardingStrategy | None:
        return self.q_proj.sharding_strategy

    @sharding_strategy.setter
    def sharding_strategy(self, strategy: ShardingStrategy) -> None:
        if not strategy.is_replicate:
            raise ValueError(
                "Only replicate is currently supported for Gemma4VisionAttention"
            )
        self.q_proj.sharding_strategy = strategy
        self.k_proj.sharding_strategy = strategy
        self.v_proj.sharding_strategy = strategy
        self.o_proj.sharding_strategy = strategy
        self.q_norm.weight.sharding_strategy = strategy
        self.k_norm.weight.sharding_strategy = strategy

    def shard(
        self, devices: Iterable[DeviceRef]
    ) -> list[Gemma4VisionAttention]:
        assert self.sharding_strategy

        q_proj_shards = self.q_proj.shard(devices)
        k_proj_shards = self.k_proj.shard(devices)
        v_proj_shards = self.v_proj.shard(devices)
        o_proj_shards = self.o_proj.shard(devices)
        q_norm_weight_shards = self.q_norm.weight.shard(devices)
        k_norm_weight_shards = self.k_norm.weight.shard(devices)

        shards = []
        for device, q_s, k_s, v_s, o_s, qn_s, kn_s in zip(
            devices,
            q_proj_shards,
            k_proj_shards,
            v_proj_shards,
            o_proj_shards,
            q_norm_weight_shards,
            k_norm_weight_shards,
            strict=True,
        ):
            sharded = Gemma4VisionAttention(self.config, self.layer_idx, device)
            sharded.q_proj = q_s
            sharded.k_proj = k_s
            sharded.v_proj = v_s
            sharded.o_proj = o_s
            sharded.q_norm.weight = qn_s
            sharded.k_norm.weight = kn_s
            shards.append(sharded)

        return shards
