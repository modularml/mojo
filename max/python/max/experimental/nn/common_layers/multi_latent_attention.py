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
"""Multi-Latent Attention with RoPE (ModuleV3, single-GPU)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from max.driver import CPU
from max.dtype import DType
from max.experimental import functional as F
from max.experimental.nn import Linear, Module
from max.experimental.nn.common_layers.functional_kernels import (
    flare_mla_prefill_plan,
    mla_decode_graph,
    mla_prefill_decode_graph,
    mla_prefill_graph,
)
from max.experimental.nn.common_layers.mesh_axis import TP
from max.experimental.nn.norm import RMSNorm
from max.experimental.sharding import NamedMapping
from max.experimental.tensor import Tensor
from max.nn.attention import MHAMaskVariant
from max.nn.kv_cache import KVCacheParams, PagedCacheValues


@dataclass
class MLAPrefillMetadata:
    """Dataclass to hold MLA prefill metadata."""

    buffer_row_offsets: Tensor
    cache_offsets: Tensor
    buffer_lengths: Tensor

    def __tree_flatten__(self) -> tuple[tuple[Tensor, ...], None]:
        return (
            self.buffer_row_offsets,
            self.cache_offsets,
            self.buffer_lengths,
        ), None

    @classmethod
    def __tree_unflatten__(
        cls, aux: None, children: Sequence[Tensor]
    ) -> MLAPrefillMetadata:
        """Rebuilds an :class:`MLAPrefillMetadata` from flattened leaves."""
        del aux
        return cls(*children)


class LatentAttentionWithRope(Module[..., Tensor]):
    """Implementation of Latent Attention with Rope (ModuleV3, single-GPU).

    Args:
        rope: The rope layer to borrow the freqs_cis value from.
        num_attention_heads: The number of attention heads.
        num_key_value_heads: Number of key/value heads.
        hidden_size: The dimension of the hidden states.
        kv_params: KV Cache Params, including the number of kv heads, the
            head dim, and data type.
        layer_idx: The layer number associated with this attention block.
        scale: Value used to scale the results of the attention output.
        q_lora_rank: Optional LoRA rank for Q projection.
        kv_lora_rank: LoRA rank for KV projections.
        qk_nope_head_dim: Head dimension for non-positional encoding part.
        qk_rope_head_dim: Head dimension for rope part.
        v_head_dim: Head dimension for value.
        buffer_size: Buffer size for storing temporal results during prefill,
            in units of tokens.
        graph_mode: Pipeline role to use for the attention layer. Should be
            ``"prefill"``, ``"decode"``, or ``"auto"``.
    """

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
    ) -> None:
        super().__init__()

        _role = graph_mode or "auto"
        if _role not in ("prefill", "decode", "auto"):
            raise ValueError(
                f"Invalid graph_mode '{_role}'. Use 'prefill', 'decode', or 'auto'."
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

        if self.q_lora_rank is not None:
            self.q_a_proj = Tensor.zeros([self.q_lora_rank, self.hidden_size])
            self.q_a_layernorm = RMSNorm(dim=self.q_lora_rank, eps=1e-6)
            self.q_b_proj = Tensor.zeros(
                [self.n_heads * self.qk_head_dim, self.q_lora_rank]
            )
        else:
            self.q_proj = Tensor.zeros(
                [self.n_heads * self.qk_head_dim, self.hidden_size]
            )

        self.kv_a_proj_layernorm = Tensor.ones([self.kv_lora_rank])
        self.kv_a_proj_with_mqa = Tensor.zeros(
            [self.cache_head_dim, self.hidden_size]
        )
        self.kv_b_proj = Tensor.zeros(
            [
                self.n_heads * (self.qk_nope_head_dim + self.v_head_dim),
                self.kv_lora_rank,
            ]
        )
        self.o_proj = Linear(
            in_dim=self.n_heads * self.v_head_dim,
            out_dim=self.hidden_size,
            bias=False,
        )

    def create_mla_prefill_metadata(
        self, input_row_offsets: Tensor, kv_collection: PagedCacheValues
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

    @property
    def _kv_b_proj_weight(self) -> Tensor:
        """Returns ``kv_b_proj`` reshaped for per-head projection slicing."""
        return self.kv_b_proj.T.reshape(
            [
                self.kv_lora_rank,
                self.n_heads,
                self.qk_nope_head_dim + self.v_head_dim,
            ]
        )

    @property
    def w_uk(self) -> Tensor:
        """Returns decode K-projection weights with shape [H, qk_nope_dim, kv_rank]."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        return w_uk_base.transpose(0, 1)

    @property
    def w_uv(self) -> Tensor:
        """Returns decode V-projection weights with shape [H, kv_rank, v_dim]."""
        w_uv = self._kv_b_proj_weight[..., self.qk_nope_head_dim :]
        return w_uv.permute([1, 2, 0])

    @property
    def w_k(self) -> Tensor:
        """Returns prefill K-projection weights with shape [H*qk_nope_dim, kv_rank]."""
        w_uk_base = self._kv_b_proj_weight[..., : self.qk_nope_head_dim]
        return w_uk_base.permute([1, 2, 0]).reshape([-1, self.kv_lora_rank])

    @property
    def wqkv(self) -> Tensor:
        """Returns the concatenation of q_a_proj/q_proj and kv_a_proj_with_mqa."""
        if self.q_lora_rank is not None:
            return F.concat([self.q_a_proj, self.kv_a_proj_with_mqa])
        return F.concat([self.q_proj, self.kv_a_proj_with_mqa])

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
                mla_prefill_metadata.buffer_lengths.to(CPU())
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

        return result.reshape([result.shape[0], self.n_heads * self.v_head_dim])

    def forward(
        self,
        x: Tensor,
        kv_collection: PagedCacheValues,
        freqs_cis: Tensor,
        input_row_offsets: Tensor,
        mla_prefill_metadata: MLAPrefillMetadata | None = None,
    ) -> Tensor:
        """Forward pass for the LatentAttentionWithRope."""
        layer_idx = F.constant(self.layer_idx, DType.uint32, device=CPU())

        if self.q_lora_rank is not None:
            # Use fused matmul because weights in `wqkv` (`q_proj` and
            # `kv_a_proj_with_mqa`) have the same placement.
            qkv = x @ self.wqkv.T
            xq, kv = qkv.split([self.q_lora_rank, self.cache_head_dim], axis=1)
            xq = self.q_a_layernorm(xq) @ self.q_b_proj.T
        else:
            xq = x @ self.q_proj.T
            kv = x @ self.kv_a_proj_with_mqa.T

        xq = xq.reshape((-1, self.n_heads, self.qk_head_dim))

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


def assign_replicated_mapping(weight: Tensor) -> None:
    """Assigns a replicated mapping to the weight."""
    replicated = (None,) * len(weight.shape)
    weight._mapping = NamedMapping(weight.mesh, replicated)


def assign_rowwise_mapping(weight: Tensor) -> None:
    """Assigns a rowwise mapping to the weight."""
    rowwise = (TP,) + (None,) * (len(weight.shape) - 1)
    weight._mapping = NamedMapping(weight.mesh, rowwise)


def assign_columnwise_mapping(weight: Tensor) -> None:
    """Assigns a columnwise mapping to the weight."""
    columnwise = (None, TP) + (None,) * (len(weight.shape) - 2)
    weight._mapping = NamedMapping(weight.mesh, columnwise)


def tensor_parallel_latent_attention_with_rope(
    layer: LatentAttentionWithRope,
) -> LatentAttentionWithRope:
    """Modifies latent attention layer to be tensor parallel along the TP axis."""
    # Replicated weights: q_a_proj, q_a_layernorm
    if layer.q_lora_rank is not None:
        assign_replicated_mapping(layer.q_a_proj)
        assign_replicated_mapping(layer.q_a_layernorm.weight)
        assign_rowwise_mapping(layer.q_b_proj)
    else:
        assign_rowwise_mapping(layer.q_proj)

    assign_replicated_mapping(layer.kv_a_proj_layernorm)
    assign_replicated_mapping(layer.kv_a_proj_with_mqa)
    assign_rowwise_mapping(layer.kv_b_proj)
    assign_columnwise_mapping(layer.o_proj.weight)

    return layer
