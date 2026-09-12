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
"""Graph construction for ModernBERT embedding models."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
from max import nn
from max.driver import DLPackArray
from max.dtype import DType
from max.graph import DeviceRef, Graph, TensorType, TensorValue, ops
from max.graph.dim import DimLike
from max.graph.weights import WeightData
from transformers import AutoConfig

from .model_config import ModernBertConfig


def _resolve_norm_eps(hf_config: AutoConfig) -> float:
    return float(
        getattr(
            hf_config, "norm_eps", getattr(hf_config, "layer_norm_eps", 1e-5)
        )
    )


def _resolve_norm_bias(hf_config: AutoConfig) -> bool:
    return bool(getattr(hf_config, "norm_bias", False))


def _resolve_attention_bias(hf_config: AutoConfig) -> bool:
    return bool(getattr(hf_config, "attention_bias", False))


def _resolve_mlp_bias(hf_config: AutoConfig) -> bool:
    return bool(getattr(hf_config, "mlp_bias", False))


def _resolve_rope_thetas(hf_config: AutoConfig) -> tuple[float, float]:
    rope_parameters = getattr(hf_config, "rope_parameters", None)
    if rope_parameters is not None:
        full_rope = rope_parameters.get("full_attention", {})
        sliding_rope = rope_parameters.get("sliding_attention", {})
        if "rope_theta" in full_rope and "rope_theta" in sliding_rope:
            return float(full_rope["rope_theta"]), float(
                sliding_rope["rope_theta"]
            )

    global_theta = getattr(hf_config, "global_rope_theta", None)
    local_theta = getattr(hf_config, "local_rope_theta", None)
    if global_theta is not None and local_theta is not None:
        return float(global_theta), float(local_theta)

    default_theta = float(getattr(hf_config, "rope_theta", 10000.0))
    return default_theta, default_theta


def _resolve_local_window(hf_config: AutoConfig) -> int:
    if hasattr(hf_config, "sliding_window"):
        return int(hf_config.sliding_window)
    return int(getattr(hf_config, "local_attention", 128) // 2)


class _IdentityNorm(nn.Module):
    def __call__(self, x: TensorValue) -> TensorValue:
        return x


class ModernBertEmbeddings(nn.Module):
    """Token embedding plus embedding layer norm."""

    def __init__(self, config: ModernBertConfig) -> None:
        super().__init__()
        hf = config.huggingface_config

        self.tok_embeddings = nn.Embedding(
            hf.vocab_size,
            hf.hidden_size,
            config.dtype,
            config.device,
        )
        self.norm = nn.LayerNorm(
            hf.hidden_size,
            [config.device],
            DType.float32,
            eps=_resolve_norm_eps(hf),
            use_bias=_resolve_norm_bias(hf),
        )

    def __call__(self, input_ids: TensorValue) -> TensorValue:
        return self.norm(self.tok_embeddings(input_ids))


class ModernBertMLP(nn.Module):
    """ModernBERT GeGLU MLP block."""

    def __init__(self, config: ModernBertConfig) -> None:
        super().__init__()
        hf = config.huggingface_config
        self.intermediate_size = hf.intermediate_size

        self.Wi = nn.Linear(
            hf.hidden_size,
            2 * hf.intermediate_size,
            config.dtype,
            config.device,
            has_bias=_resolve_mlp_bias(hf),
        )
        self.Wo = nn.Linear(
            hf.intermediate_size,
            hf.hidden_size,
            config.dtype,
            config.device,
            has_bias=_resolve_mlp_bias(hf),
        )

    def __call__(self, hidden_states: TensorValue) -> TensorValue:
        fused = self.Wi(hidden_states)
        gate, up = ops.split(
            fused,
            [self.intermediate_size, self.intermediate_size],
            axis=-1,
        )
        return self.Wo(ops.gelu(gate) * up)


class ModernBertAttention(nn.Module):
    """Fused-QKV self-attention with optional local band masking."""

    def __init__(self, config: ModernBertConfig) -> None:
        super().__init__()
        hf = config.huggingface_config

        self.hidden_size = hf.hidden_size
        self.num_heads = hf.num_attention_heads
        self.head_dim = hf.hidden_size // hf.num_attention_heads

        self.Wqkv = nn.Linear(
            hf.hidden_size,
            3 * hf.hidden_size,
            config.dtype,
            config.device,
            has_bias=_resolve_attention_bias(hf),
        )
        self.Wo = nn.Linear(
            hf.hidden_size,
            hf.hidden_size,
            config.dtype,
            config.device,
            has_bias=_resolve_attention_bias(hf),
        )

    @staticmethod
    def _local_band_mask(
        seq_len: DimLike,
        window: int,
        device: DeviceRef,
    ) -> TensorValue:
        ones = ops.broadcast_to(
            ops.constant(1.0, DType.float32, device=device),
            (seq_len, seq_len),
        )
        band = ops.band_part(ones, num_lower=window, num_upper=window)
        neg_inf = ops.constant(
            float(np.finfo(np.float32).min),
            DType.float32,
            device=device,
        )
        mask = (
            ops.constant(1.0, DType.float32, device=device) - band
        ) * neg_inf
        return ops.reshape(mask, (1, 1, seq_len, seq_len))

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
        rope: nn.RotaryEmbedding,
        is_global: bool,
        local_window: int,
    ) -> TensorValue:
        batch_size, seq_len, hidden_size = hidden_states.shape

        qkv = self.Wqkv(hidden_states)
        q, k, v = ops.split(
            qkv, [hidden_size, hidden_size, hidden_size], axis=-1
        )

        q = ops.reshape(q, (batch_size, seq_len, self.num_heads, self.head_dim))
        k = ops.reshape(k, (batch_size, seq_len, self.num_heads, self.head_dim))
        v = ops.reshape(v, (batch_size, seq_len, self.num_heads, self.head_dim))

        q = rope(q)
        k = rope(k)

        q = ops.permute(q, [0, 2, 1, 3])
        k = ops.permute(k, [0, 2, 1, 3])
        v = ops.permute(v, [0, 2, 1, 3])

        scores = q @ ops.permute(k, [0, 1, 3, 2])
        scores = scores * ops.constant(
            1.0 / math.sqrt(self.head_dim),
            DType.float32,
            device=hidden_states.device,
        )
        scores = scores + attention_mask

        if not is_global:
            scores = scores + self._local_band_mask(
                seq_len,
                local_window,
                hidden_states.device,
            )

        probs = ops.softmax(scores, axis=-1)
        ctx = probs @ v
        ctx = ops.permute(ctx, [0, 2, 1, 3])
        ctx = ops.reshape(ctx, (batch_size, seq_len, hidden_size))
        return self.Wo(ctx)


class ModernBertLayer(nn.Module):
    """Single pre-norm ModernBERT transformer block."""

    def __init__(self, config: ModernBertConfig, layer_idx: int) -> None:
        super().__init__()
        hf = config.huggingface_config
        self.is_global = layer_idx % hf.global_attn_every_n_layers == 0
        global_rope_theta, local_rope_theta = _resolve_rope_thetas(hf)
        rope_theta = global_rope_theta if self.is_global else local_rope_theta
        self.rope = nn.RotaryEmbedding(
            dim=hf.hidden_size,
            n_heads=hf.num_attention_heads,
            theta=rope_theta,
            max_seq_len=hf.max_position_embeddings,
            interleaved=False,
        )
        attn_norm: nn.Module
        if layer_idx == 0:
            attn_norm = _IdentityNorm()
        else:
            attn_norm = nn.LayerNorm(
                hf.hidden_size,
                [config.device],
                DType.float32,
                eps=_resolve_norm_eps(hf),
                use_bias=_resolve_norm_bias(hf),
            )
        self.attn_norm = attn_norm
        self.attn = ModernBertAttention(config)
        self.mlp_norm = nn.LayerNorm(
            hf.hidden_size,
            [config.device],
            DType.float32,
            eps=_resolve_norm_eps(hf),
            use_bias=_resolve_norm_bias(hf),
        )
        self.mlp = ModernBertMLP(config)
        self.local_window = _resolve_local_window(hf)

    def __call__(
        self,
        hidden_states: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        attn_out = self.attn(
            self.attn_norm(hidden_states),
            attention_mask,
            self.rope,
            self.is_global,
            self.local_window,
        )
        hidden_states = hidden_states + attn_out
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class ModernBertModel(nn.Module):
    """ModernBERT encoder backbone with optional masked mean pooling."""

    def __init__(self, config: ModernBertConfig) -> None:
        super().__init__()
        hf = config.huggingface_config
        self.embeddings = ModernBertEmbeddings(config)
        # Use a LayerList so weights register at top-level "layers.{i}.*",
        # matching the names produced by the safetensor weight adapter.
        self.layers = nn.LayerList(
            [ModernBertLayer(config, i) for i in range(hf.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(
            hf.hidden_size,
            [config.device],
            DType.float32,
            eps=_resolve_norm_eps(hf),
            use_bias=_resolve_norm_bias(hf),
        )
        self.pool_outputs = config.pool_embeddings

    def __call__(
        self,
        input_ids: TensorValue,
        attention_mask: TensorValue,
    ) -> TensorValue:
        hidden_states = self.embeddings(input_ids)

        batch_size, seq_len = attention_mask.shape
        extended_mask = ops.reshape(attention_mask, (batch_size, 1, 1, seq_len))
        neg_inf = ops.constant(
            float(np.finfo(np.float32).min),
            DType.float32,
            device=attention_mask.device,
        )
        extended_mask = (
            ops.constant(1.0, DType.float32, device=attention_mask.device)
            - extended_mask
        ) * neg_inf

        for layer in self.layers:
            hidden_states = layer(hidden_states, extended_mask)
        hidden_states = self.final_norm(hidden_states)

        if not self.pool_outputs:
            return hidden_states

        # Masked mean pooling.
        # Compute per-sample token count from the original 2D mask [B, S]
        # before any transpose/broadcast, to avoid shape ambiguity.
        token_counts = ops.max(
            ops.sum(attention_mask),
            ops.constant(1e-9, DType.float32, device=hidden_states.device),
        )  # [B, 1]

        # ``ops.sum`` reduces the innermost axis (keeping rank), so
        # transpose hidden states to put seq_len last.
        encoded = hidden_states.transpose(1, 2)  # [B, H, S]
        mask_expanded = ops.broadcast_to(
            ops.unsqueeze(attention_mask, 1),
            ("batch_size", encoded.shape[1], "seq_len"),
        )
        summed = ops.sum(encoded * mask_expanded)  # [B, H, 1]

        lengths = ops.unsqueeze(token_counts, 1)  # [B, 1, 1]
        pooled = summed / lengths  # [B, H, 1]
        return ops.squeeze(pooled, 2)  # [B, H]


def build_graph(
    config: ModernBertConfig,
    state_dict: Mapping[str, DLPackArray | WeightData],
) -> Graph:
    """Build a ModernBERT graph for embeddings inference."""
    input_ids_type = TensorType(
        DType.int64,
        shape=["batch_size", "seq_len"],
        device=config.device,
    )
    attention_mask_type = TensorType(
        DType.float32,
        shape=["batch_size", "seq_len"],
        device=config.device,
    )

    with Graph(
        "modernbert",
        input_types=[input_ids_type, attention_mask_type],
    ) as graph:
        model = ModernBertModel(config)
        model.load_state_dict(state_dict)
        input_ids = graph.inputs[0].tensor
        attention_mask = graph.inputs[1].tensor
        graph.output(model(input_ids, attention_mask))

    return graph
