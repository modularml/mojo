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

"""Quantize-aware Multi-Latent Attention with RoPE."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

from max.driver import CPU, Device
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Module
from max.experimental.nn.common_layers.functional_kernels import (
    flare_mla_prefill_plan,
    mla_decode_graph,
    mla_prefill_decode_graph,
    mla_prefill_graph,
)
from max.experimental.nn.common_layers.kv_cache import PagedCacheValues
from max.experimental.nn.common_layers.multi_latent_attention import (
    MLAPrefillMetadata,
    assign_columnwise_mapping,
    assign_replicated_mapping,
    assign_rowwise_mapping,
)
from max.experimental.nn.norm import RMSNorm
from max.experimental.sharding import DeviceMapping, DeviceMesh
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams
from max.nn.quant_config import QuantConfig
from typing_extensions import Self

from . import quant_ops
from .quant_linear import QuantizedLinear
from .quant_tensor import FP8BlockTensor, NVFP4Tensor, QuantAwareTensor


def _data(weight: QuantAwareTensor) -> Tensor:
    """Underlying data tensor of a weight (``.data`` when quantized, else itself)."""
    if isinstance(weight, (FP8BlockTensor, NVFP4Tensor)):
        return weight.data
    assert isinstance(weight, Tensor)
    return weight


def _scale(weight: QuantAwareTensor) -> Tensor | None:
    """Per-block weight scales, or ``None`` for bf16 weights."""
    if isinstance(weight, FP8BlockTensor):
        return weight.weight_scale_inv
    if isinstance(weight, NVFP4Tensor):
        return weight.weight_scale
    return None


class QuantizedLatentAttentionWithRope(Module[..., Tensor]):
    """Latent Attention with RoPE and quantize-aware projection weights."""

    def __init__(
        self,
        *,
        num_attention_heads: int,
        num_key_value_heads: int,
        hidden_size: int,
        kv_params: KVCacheParams,
        layer_idx: int,
        scale: float | None = None,
        q_lora_rank: int | None = None,
        kv_lora_rank: int = 512,
        qk_nope_head_dim: int = 128,
        qk_rope_head_dim: int = 64,
        v_head_dim: int = 128,
        buffer_size: int = 16384,
        graph_mode: str | None = None,
        quant_config: QuantConfig | None = None,
        quantize_o_proj: bool = True,
    ) -> None:
        super().__init__()
        _role = graph_mode or "auto"
        if _role not in ("prefill", "decode", "auto"):
            raise ValueError(
                f"Invalid graph_mode '{_role}'. Use 'prefill', 'decode', or"
                " 'auto'."
            )

        self.graph_mode = _role

        self.n_heads = num_attention_heads
        self.kv_params = kv_params
        self.num_key_value_heads = num_key_value_heads
        self.hidden_size = hidden_size
        self.layer_idx = layer_idx

        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_nope_head_dim + qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.cache_head_dim = kv_lora_rank + qk_rope_head_dim

        self.BUFFER_TOK_SIZE = buffer_size

        self.scale = (
            scale if scale is not None else math.sqrt(1.0 / self.qk_head_dim)
        )

        # In NVFP4 quantized checkpoints, ``q``/``kv`` projections stay bf16
        # (only ``o_proj`` may be quantized)
        self.quant_config = quant_config
        self.quantized = quant_ops.is_fp8_block_quantized(quant_config)
        if self.quantized:
            assert quant_config is not None
            if not quant_config.input_scale.is_block:
                raise ValueError(
                    "Input scale must be block-wise for FP8 "
                    "QuantizedLatentAttentionWithRope"
                )
            assert quant_config.weight_scale.block_size is not None
            self.weight_block_size: tuple[int, int] | None = (
                quant_config.weight_scale.block_size
            )
        else:
            self.weight_block_size = None

        weight_quant_config = quant_config if self.quantized else None

        def _weight(out_dim: int, in_dim: int) -> QuantAwareTensor:
            return quant_ops.quantized_weight(
                out_dim, in_dim, weight_quant_config
            )

        if self.q_lora_rank is not None:
            self.q_a_proj = _weight(self.q_lora_rank, self.hidden_size)
            self.q_a_layernorm = RMSNorm(dim=self.q_lora_rank, eps=1e-6)
            self.q_b_proj = _weight(
                self.n_heads * self.qk_head_dim, self.q_lora_rank
            )
        else:
            self.q_proj = _weight(
                self.n_heads * self.qk_head_dim, self.hidden_size
            )

        # ``kv_a_proj_layernorm`` is a plain RMSNorm gamma (not quantized).
        self.kv_a_proj_layernorm = Tensor.ones((self.kv_lora_rank,))
        self.kv_a_proj_with_mqa = _weight(self.cache_head_dim, self.hidden_size)

        self.kv_b_proj = _weight(
            self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
            self.kv_lora_rank,
        )

        self.o_proj = QuantizedLinear(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=self.hidden_size,
            bias=False,
            quant_config=quant_config if quantize_o_proj else None,
        )

    def to(self, target: Device | DeviceMesh | DeviceMapping) -> Self:
        super().to(target)

        self.kv_a_proj_with_mqa = self.kv_a_proj_with_mqa.to(target)
        self.kv_b_proj = self.kv_b_proj.to(target)
        if self.q_lora_rank is not None:
            self.q_a_proj = self.q_a_proj.to(target)
            self.q_b_proj = self.q_b_proj.to(target)
        else:
            self.q_proj = self.q_proj.to(target)
        return self

    @property
    def _kv_b_proj_weight(self) -> Tensor:
        """``kv_b_proj`` data reshaped to ``[kv_rank, n_heads, qk_nope+v]``."""
        return _data(self.kv_b_proj).T.reshape(
            (self.kv_lora_rank, self.n_heads, -1)
        )

    @property
    def _kv_b_proj_weight_scale(self) -> Tensor:
        """``kv_b_proj`` scales reshaped to align with the data layout.

        Only valid for FP8 block-scaled weights.
        """
        assert self.weight_block_size is not None
        scale = _scale(self.kv_b_proj)
        assert scale is not None
        block_m, _ = self.weight_block_size
        return scale.T.reshape((self.kv_lora_rank // block_m, self.n_heads, -1))

    @property
    def _qk_nope_head_scale_dim(self) -> int:
        assert self.weight_block_size is not None
        return self.qk_nope_head_dim // self.weight_block_size[0]

    @property
    def w_uk(self) -> tuple[Tensor, Tensor | None]:
        """Decode K-projection ``(data, weight_scale_inv|None)``."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        w_uk = w_uk_base.transpose(0, 1)
        if not self.quantized:
            return w_uk, None
        w_uk_scale = self._kv_b_proj_weight_scale[
            ..., : self._qk_nope_head_scale_dim
        ].transpose(0, 1)
        return w_uk, w_uk_scale

    @property
    def w_uv(self) -> tuple[Tensor, Tensor | None]:
        """Decode V-projection ``(data, weight_scale_inv|None)``."""
        w_uv = self._kv_b_proj_weight[..., self.qk_nope_head_dim :].permute(
            [1, 2, 0]
        )
        if not self.quantized:
            return w_uv, None
        w_uv_scale = self._kv_b_proj_weight_scale[
            ..., self._qk_nope_head_scale_dim :
        ].permute([1, 2, 0])
        return w_uv, w_uv_scale

    @property
    def w_k(self) -> tuple[Tensor, Tensor | None]:
        """Prefill K-projection ``(data, weight_scale_inv|None)``."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        w_k = w_uk_base.permute([1, 2, 0]).reshape((-1, self.kv_lora_rank))
        if not self.quantized:
            return w_k, None
        assert self.weight_block_size is not None
        w_k_scale = (
            self._kv_b_proj_weight_scale[..., : self._qk_nope_head_scale_dim]
            .permute([1, 2, 0])
            .reshape((-1, self.kv_lora_rank // self.weight_block_size[0]))
        )
        return w_k, w_k_scale

    @property
    def wqkv(self) -> QuantAwareTensor:
        """Fused ``q_a_proj || kv_a_proj_with_mqa`` (bf16 or FP8)."""
        q = self.q_a_proj if self.q_lora_rank is not None else self.q_proj
        return quant_ops.concat_weights(q, self.kv_a_proj_with_mqa, axis=0)

    def create_mla_prefill_metadata(
        self, input_row_offsets: Tensor, kv_collection: PagedCacheValues
    ) -> MLAPrefillMetadata:
        layer_idx = F.constant(0, DType.uint32, device=CPU())
        buffer_row_offsets, cache_offsets, buffer_lengths = (
            flare_mla_prefill_plan(
                self.kv_params,
                input_row_offsets,
                kv_collection,
                layer_idx,
                self.BUFFER_TOK_SIZE,
                max_chunks=1,
            )
        )
        return MLAPrefillMetadata(
            buffer_row_offsets=buffer_row_offsets,
            cache_offsets=cache_offsets,
            buffer_lengths=buffer_lengths,
        )

    def _mla_impl(
        self,
        xq: Tensor,
        kv: Tensor,
        kv_collection: PagedCacheValues,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        freqs_cis: Tensor,
        kv_norm_gamma: Tensor,
        _mla_prefill_metadata: MLAPrefillMetadata | None = None,
        epsilon: float = 1e-6,
    ) -> Tensor:
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
        # Only the FP8 block-scaled kernels consume ``quant_config``; bf16
        # leaves it at the kernel default (None).
        if self.quantized:
            attn_kwargs["quant_config"] = self.quant_config

        if self.graph_mode in ("prefill", "auto"):
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
            buffer_lengths = mla_prefill_metadata.buffer_lengths
            buffer_lengths = buffer_lengths.to(CPU())
            attn_kwargs["buffer_length"] = buffer_lengths
            w_k, w_k_scale = self.w_k
            w_uv, w_uv_scale = self.w_uv
            attn_kwargs["w_k"] = w_k
            attn_kwargs["w_uv"] = w_uv
            if self.quantized:
                attn_kwargs["w_k_scale"] = w_k_scale
                attn_kwargs["w_uv_scale"] = w_uv_scale

        if self.graph_mode in ("decode", "auto"):
            w_uk, w_uk_scale = self.w_uk
            w_uv, w_uv_scale = self.w_uv
            attn_kwargs["w_uk"] = w_uk
            attn_kwargs["w_uv"] = w_uv
            if self.quantized:
                attn_kwargs["w_uk_scale"] = w_uk_scale
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

        return result.reshape([result.shape[0], self.n_heads * self.v_head_dim])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        layer_idx: Tensor,
        input_row_offsets: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> Tensor:
        if self.q_lora_rank is not None:
            qkv = quant_ops.matmul(x, self.wqkv)
            xq, kv = qkv.split([self.q_lora_rank, self.cache_head_dim], axis=1)
            xq = quant_ops.matmul(self.q_a_layernorm(xq), self.q_b_proj)
        else:
            xq = quant_ops.matmul(x, self.q_proj)
            kv = quant_ops.matmul(x, self.kv_a_proj_with_mqa)

        # Explicit token dim: the shape rule cannot prove a ``-1`` leading
        # dim against a per-shard (data-parallel) token count.
        xq = xq.reshape((xq.shape[0], self.n_heads, self.qk_head_dim))

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


def _assign_quant_aware(
    weight: QuantAwareTensor, assign: Callable[[Tensor], None]
) -> None:
    """Apply a placement-assignment to a (possibly quantized) weight."""
    if isinstance(weight, FP8BlockTensor):
        assign(weight.data)
        assign(weight.weight_scale_inv)
    elif isinstance(weight, NVFP4Tensor):
        assign(weight.data)
        assign(weight.weight_scale)
        assign_replicated_mapping(weight.weight_scale_2)
        assign_replicated_mapping(weight.input_scale)
    else:
        assert isinstance(weight, Tensor)
        assign(weight)


def tensor_parallel_latent_attention_with_rope(
    layer: QuantizedLatentAttentionWithRope,
) -> QuantizedLatentAttentionWithRope:
    """Modifies latent attention layer to be tensor parallel along the TP axis."""
    # Replicated weights: q_a_proj, q_a_layernorm
    if layer.q_lora_rank is not None:
        assert isinstance(layer.q_a_layernorm.weight, Tensor)
        _assign_quant_aware(layer.q_a_proj, assign_replicated_mapping)
        assign_replicated_mapping(layer.q_a_layernorm.weight)
        _assign_quant_aware(layer.q_b_proj, assign_rowwise_mapping)
    else:
        _assign_quant_aware(layer.q_proj, assign_rowwise_mapping)

    assert isinstance(layer.kv_a_proj_layernorm, Tensor)
    assign_replicated_mapping(layer.kv_a_proj_layernorm)
    _assign_quant_aware(layer.kv_a_proj_with_mqa, assign_replicated_mapping)
    _assign_quant_aware(layer.kv_b_proj, assign_rowwise_mapping)
    _assign_quant_aware(layer.o_proj.weight, assign_columnwise_mapping)

    return layer
