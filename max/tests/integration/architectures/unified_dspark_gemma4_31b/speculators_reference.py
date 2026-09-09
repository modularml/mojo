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
"""PyTorch reference for the speculators-format DSpark draft.

Faithful port of the vLLM speculators-DSpark runtime (``qwen3_dflash.py``
DFlashQwen3Attention/DecoderLayer, ``qwen3_dspark.py`` heads,
``dspark/speculator.py`` chain). Training-only code (t2d, confidence head)
is dropped; the math of the inference path is kept verbatim. Shared by the
draft-module logit-verify test and the unified-graph step-parity test.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# Training-only tensors, skipped at load exactly like vLLM; embed_tokens is
# consumed test-side to build the raw block embeddings, and the markov/d2t
# weights load into the reference chain helper instead of the backbone.
REF_SKIP_KEYS = ("t2d", "d2t")
REF_SKIP_PREFIXES = ("confidence_head.", "markov_head.", "embed_tokens.")


def reference_state_dict(
    checkpoint_weights: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        k: v
        for k, v in checkpoint_weights.items()
        if k not in REF_SKIP_KEYS and not k.startswith(REF_SKIP_PREFIXES)
    }


def _require_int(mapping: Mapping[str, object], key: str) -> int:
    value = mapping[key]
    assert isinstance(value, int), f"{key}: expected int, got {type(value)}"
    return value


def _require_float(mapping: Mapping[str, object], key: str) -> float:
    value = mapping[key]
    assert isinstance(value, (int, float)), (
        f"{key}: expected number, got {type(value)}"
    )
    return float(value)


@dataclass(frozen=True)
class DraftCheckpointConfig:
    """The subset of the speculators checkpoint config the reference needs.

    Parsed directly from the raw ``config.json`` (top-level dspark fields
    plus the nested ``transformer_layer_config``); the typed model-config
    class is out of scope for the reference harness.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    sliding_window: int | None
    layer_causal: tuple[bool, ...]
    vocab_size: int
    draft_vocab_size: int
    markov_rank: int
    block_size: int
    sample_from_anchor: bool
    mask_token_id: int
    num_context_features: int

    @classmethod
    def from_config_json(
        cls, cfg: Mapping[str, object]
    ) -> DraftCheckpointConfig:
        tl_obj = cfg["transformer_layer_config"]
        assert isinstance(tl_obj, dict)
        tl: dict[str, object] = tl_obj
        layer_types = tl["layer_types"]
        assert isinstance(layer_types, list)
        # Causality rule from the vLLM reference (_dflash_layer_causal):
        # causal iff the layer type is sliding_attention.
        layer_causal = tuple(lt == "sliding_attention" for lt in layer_types)
        rope_obj = tl["rope_parameters"]
        assert isinstance(rope_obj, dict)
        rope_parameters: dict[str, object] = rope_obj
        aux_ids = cfg["aux_hidden_state_layer_ids"]
        assert isinstance(aux_ids, list)
        sliding_window = (
            _require_int(tl, "sliding_window")
            if tl.get("sliding_window") is not None
            else None
        )
        return cls(
            hidden_size=_require_int(tl, "hidden_size"),
            num_hidden_layers=_require_int(tl, "num_hidden_layers"),
            num_attention_heads=_require_int(tl, "num_attention_heads"),
            num_key_value_heads=_require_int(tl, "num_key_value_heads"),
            head_dim=_require_int(tl, "head_dim"),
            intermediate_size=_require_int(tl, "intermediate_size"),
            rms_norm_eps=_require_float(tl, "rms_norm_eps"),
            rope_theta=_require_float(rope_parameters, "rope_theta"),
            sliding_window=sliding_window,
            layer_causal=layer_causal,
            vocab_size=_require_int(tl, "vocab_size"),
            draft_vocab_size=_require_int(cfg, "draft_vocab_size"),
            markov_rank=_require_int(cfg, "markov_rank"),
            block_size=_require_int(cfg, "block_size"),
            sample_from_anchor=bool(cfg["sample_from_anchor"]),
            mask_token_id=_require_int(cfg, "mask_token_id"),
            num_context_features=len(aux_ids),
        )


class RefRMSNorm(nn.Module):
    """Qwen3/llama-style RMSNorm: fp32 rms, cast back, then scale."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.float()
        variance = hidden.pow(2).mean(-1, keepdim=True)
        hidden = hidden * torch.rsqrt(variance + self.eps)
        return self.weight * hidden.type_as(x)


class RefMLP(nn.Module):
    """Qwen3MLP: silu-gated MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Full-rotary neox RoPE tables for the given absolute positions."""
    inv_freq = 1.0 / (
        theta
        ** (torch.arange(0, head_dim, 2, dtype=torch.int64).float() / head_dim)
    )
    inv_freq = inv_freq.to(position_ids.device)
    freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    # x: [bsz, heads, seq, head_dim]; cos/sin: [bsz, seq, head_dim].
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (x * cos) + (_rotate_half(x) * sin)


class RefDSparkSpeculatorsAttention(nn.Module):
    """DFlashQwen3Attention with the precomputed-context-KV split.

    Q comes from the (normed) block stream; K/V are the concat of context
    K/V — computed from the shared fc/hidden_norm output, NOT the block
    stream — and block K/V. K gets per-head k_norm + rope on both halves at
    absolute positions; V is un-normed and un-roped.
    """

    def __init__(self, cfg: DraftCheckpointConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.num_key_value_groups = (
            cfg.num_attention_heads // cfg.num_key_value_heads
        )
        self.scaling = cfg.head_dim**-0.5
        q_dim = cfg.num_attention_heads * cfg.head_dim
        kv_dim = cfg.num_key_value_heads * cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, q_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, kv_dim, bias=False)
        self.o_proj = nn.Linear(q_dim, cfg.hidden_size, bias=False)
        self.q_norm = RefRMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RefRMSNorm(cfg.head_dim, eps=cfg.rms_norm_eps)

    def _project_kv(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        bsz, seq_len = hidden.shape[:-1]
        cfg = self.cfg
        k = self.k_proj(hidden).view(
            bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = (
            self.v_proj(hidden)
            .view(bsz, seq_len, cfg.num_key_value_heads, cfg.head_dim)
            .transpose(1, 2)
        )
        return k, v

    def forward(
        self,
        block_hidden: torch.Tensor,
        ctx_hidden: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        bsz, q_len = block_hidden.shape[:-1]
        ctx_len = ctx_hidden.shape[1]

        q = self.q_proj(block_hidden).view(
            bsz, q_len, cfg.num_attention_heads, cfg.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k_ctx, v_ctx = self._project_kv(ctx_hidden)
        k_blk, v_blk = self._project_kv(block_hidden)

        positions = torch.arange(
            ctx_len + q_len, device=block_hidden.device
        ).unsqueeze(0)
        cos, sin = _rope_cos_sin(
            positions, cfg.head_dim, cfg.rope_theta, block_hidden.dtype
        )
        q = _apply_rope(q, cos[:, ctx_len:, :], sin[:, ctx_len:, :])
        k = torch.cat([k_ctx, k_blk], dim=2)
        k = _apply_rope(k, cos, sin)
        v = torch.cat([v_ctx, v_blk], dim=2)

        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        return self.o_proj(attn_output.reshape(bsz, q_len, -1))


def build_block_attn_mask(
    ctx_len: int,
    q_len: int,
    causal: bool,
    sliding_window: int | None,
    device: torch.device,
) -> torch.Tensor:
    """Boolean [q_len, ctx_len + q_len] mask over absolute positions.

    Query i sits at absolute position ``ctx_len + i``; keys 0..ctx_len-1 are
    the context, then the block. Causal: ``k_pos <= q_pos``; sliding window
    W keeps ``k_pos >= q_pos - W + 1`` (vLLM FA window ``(W-1, 0)``, same as
    the MAX SlidingWindowCausalMask).
    """
    q_pos = torch.arange(ctx_len, ctx_len + q_len, device=device).unsqueeze(1)
    k_pos = torch.arange(ctx_len + q_len, device=device).unsqueeze(0)
    allowed = torch.ones(
        (q_len, ctx_len + q_len), dtype=torch.bool, device=device
    )
    if causal:
        allowed &= k_pos <= q_pos
    if sliding_window is not None:
        allowed &= k_pos >= q_pos - sliding_window + 1
    return allowed


class RefDSparkSpeculatorsDecoderLayer(nn.Module):
    """Pre-norm 2-sandwich: input_ln -> attn -> +res; post_attn_ln -> mlp
    -> +res."""

    def __init__(self, cfg: DraftCheckpointConfig) -> None:
        super().__init__()
        self.self_attn = RefDSparkSpeculatorsAttention(cfg)
        self.mlp = RefMLP(cfg.hidden_size, cfg.intermediate_size)
        self.input_layernorm = RefRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RefRMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        ctx_hidden: torch.Tensor,
        attn_mask: torch.Tensor,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            block_hidden=hidden_states,
            ctx_hidden=ctx_hidden,
            attn_mask=attn_mask,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class RefDSparkSpeculatorsBackbone(nn.Module):
    """fc + hidden_norm conditioning, the draft decoder stack, the final
    norm, and the 32k draft lm_head (no softcap)."""

    def __init__(self, cfg: DraftCheckpointConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                RefDSparkSpeculatorsDecoderLayer(cfg)
                for _ in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RefRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(
            cfg.num_context_features * cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
        )
        self.hidden_norm = RefRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.lm_head = nn.Linear(
            cfg.hidden_size, cfg.draft_vocab_size, bias=False
        )

    def project_target_hidden(self, taps: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(taps))

    def forward_backbone(
        self,
        block_embedding: torch.Tensor,
        target_taps: torch.Tensor,
    ) -> torch.Tensor:
        cfg = self.cfg
        ctx_hidden = self.project_target_hidden(target_taps)
        ctx_len = ctx_hidden.shape[1]
        q_len = block_embedding.shape[1]
        hidden_states = block_embedding
        for layer_idx, layer in enumerate(self.layers):
            attn_mask = build_block_attn_mask(
                ctx_len,
                q_len,
                causal=cfg.layer_causal[layer_idx],
                sliding_window=cfg.sliding_window,
                device=block_embedding.device,
            )
            hidden_states = layer(
                hidden_states=hidden_states,
                ctx_hidden=ctx_hidden,
                attn_mask=attn_mask,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)


class RefMarkovD2T(nn.Module):
    """DSparkMarkovHead + map_draft_to_target: asymmetric-vocab bias and the
    post-argmax d2t offset map."""

    def __init__(
        self, vocab_size: int, draft_vocab_size: int, markov_rank: int
    ) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, draft_vocab_size, bias=False)
        self.register_buffer(
            "d2t", torch.zeros(draft_vocab_size, dtype=torch.long)
        )

    def compute_step_bias(self, prev_target_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(prev_target_ids.long()))

    def map_draft_to_target(self, draft_ids: torch.Tensor) -> torch.Tensor:
        return draft_ids + self.d2t[draft_ids]

    def sample_block_tokens_greedy(
        self,
        base_logits: torch.Tensor,
        anchor_target_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Temperature-0 sequential chain: bias, argmax, in-chain d2t."""
        num_slots = base_logits.shape[1]
        sampled: list[torch.Tensor] = []
        prev = anchor_target_ids.long()
        for k in range(num_slots):
            step_logits = base_logits[:, k, :] + self.compute_step_bias(prev)
            draft_ids = torch.argmax(step_logits, dim=-1)
            prev = self.map_draft_to_target(draft_ids)
            sampled.append(prev)
        return torch.stack(sampled, dim=1)
