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
"""Logit verification for the DSpark Gemma4 draft module.

Loads the real ``deepseek-ai/dspark_gemma4_12b_block7`` checkpoint and
compares the MAX draft module against a faithful PyTorch port of the
DeepSpec reference (``deepspec/modeling/dspark/gemma4/modeling.py`` and
``markov_head.py``). Both implementations receive identical random target
hidden-state context features and an identical ``[anchor, mask x 6]``
block, and are compared on:

1. the projected context hidden states (``hidden_norm(fc(ctx))``),
2. the block hidden states after the 5-layer draft backbone,
3. the base draft logits (``softcap(lm_head(h))``), and
4. the markov-corrected greedy token sequence at temperature 0.
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
from types import SimpleNamespace
from typing import NamedTuple

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession, Model
from max.graph import DeviceRef, Graph, TensorType, ops
from max.nn.kv_cache import MHAKVCacheParams
from max.nn.layer import Module
from max.nn.linear import Linear
from max.pipelines.architectures.unified_dspark_gemma4_12b import (
    DSparkGemma4,
    DSparkGemma4DraftConfig,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from safetensors.torch import load_file
from test_common.context_utils import create_text_context
from torch import nn
from torch.utils.dlpack import from_dlpack

HF_REPO_ID = "deepseek-ai/dspark_gemma4_12b_block7"
TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

MAX_SEQ_LEN = 4096
CTX_LENGTHS = [32, 128]

# Repo logit-verify conventions (see max/CLAUDE.md) for the logits and the
# fc projection.
COS_DIST_THRESHOLD = 1e-3
# Intermediate block hidden states and KL are gated relative to a float32
# oracle: MAX must stay within a small factor of the torch reference's OWN
# bf16 rounding error (flash-attention vs SDPA reduction order legitimately
# differs at the bf16 floor; measured max/ref ratios are 0.6-1.8x). A truly
# wrong semantic (e.g. a causal instead of non-causal mask) diverges by
# orders of magnitude, far beyond the margin and the absolute backstops.
_ORACLE_MARGIN = 2.5
HS_COS_DIST_BACKSTOP = 1e-2
# Sub-bf16-resolution top-2 gap under which a greedy argmax flip between two
# implementations is a numerical tie, not a semantic difference (softcapped
# logits span ±30; bf16 resolution near 30 is ~0.125).
GREEDY_TIE_GAP_EPS = 0.25
KL_DIV_BACKSTOP = 0.05


# ---------------------------------------------------------------------------
# PyTorch reference, ported from the DeepSpec DSpark Gemma4 drafter
# (deepspec/modeling/dspark/gemma4/modeling.py @ 005e03b81ce). Training-only
# code (anchor sampling, flex-attention, confidence head) is dropped; the
# math of the inference path is kept verbatim.
# ---------------------------------------------------------------------------


class RefGemma4RMSNorm(nn.Module):
    """HF ``Gemma4RMSNorm``: fp32 rms-norm, optional scale, no +1 offset."""

    def __init__(self, dim: int, eps: float, with_scale: bool = True) -> None:
        super().__init__()
        self.eps = eps
        self.with_scale = with_scale
        if with_scale:
            self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = x.float()
        mean_squared = hidden.pow(2).mean(-1, keepdim=True) + self.eps
        normed = hidden * torch.pow(mean_squared, -0.5)
        if self.with_scale:
            normed = normed * self.weight.float()
        return normed.type_as(x)


class RefGemma4MLP(nn.Module):
    """HF ``Gemma4TextMLP``: gelu_pytorch_tanh gated MLP."""

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(
            F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x)
        )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim: int = 1,
) -> torch.Tensor:
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (_rotate_half(x) * sin)


def _proportional_rope_cos_sin(
    position_ids: torch.Tensor,
    head_dim: int,
    theta: float,
    partial_rotary_factor: float,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """HF proportional RoPE for ``layer_type="full_attention"``.

    Only the first ``partial_rotary_factor * head_dim // 2`` frequencies are
    non-zero; the rest are zero (identity rotation).
    """
    rope_angles = int(partial_rotary_factor * head_dim // 2)
    nope_angles = head_dim // 2 - rope_angles
    inv_freq = 1.0 / (
        theta
        ** (
            torch.arange(0, 2 * rope_angles, 2, dtype=torch.int64).float()
            / head_dim
        )
    )
    if nope_angles > 0:
        inv_freq = torch.cat([inv_freq, torch.zeros(nope_angles)])
    inv_freq = inv_freq.to(position_ids.device)
    freqs = position_ids[:, :, None].float() * inv_freq[None, None, :]
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)


class RefDSparkAttention(nn.Module):
    """``Gemma4DSparkAttention`` (k_eq_v inference path, SDPA)."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_attention_heads // num_key_value_heads
        self.head_dim = head_dim
        self.scaling = 1.0
        self.q_proj = nn.Linear(
            hidden_size, num_attention_heads * head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            hidden_size, num_key_value_heads * head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            num_attention_heads * head_dim, hidden_size, bias=False
        )
        self.q_norm = RefGemma4RMSNorm(head_dim, eps=rms_norm_eps)
        self.k_norm = RefGemma4RMSNorm(head_dim, eps=rms_norm_eps)
        self.v_norm = RefGemma4RMSNorm(
            head_dim, eps=rms_norm_eps, with_scale=False
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden_states.shape[1]
        q = self.q_proj(hidden_states).view(
            bsz, q_len, self.num_attention_heads, self.head_dim
        )
        q = self.q_norm(q).transpose(1, 2)
        k_ctx = self.k_proj(target_hidden_states)
        k_noise = self.k_proj(hidden_states)
        # attention_k_eq_v: V reuses the *pre-norm* K projection.
        v_ctx = k_ctx
        v_noise = k_noise
        k = torch.cat([k_ctx, k_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        v = torch.cat([v_ctx, v_noise], dim=1).view(
            bsz, ctx_len + q_len, self.num_key_value_heads, self.head_dim
        )
        k = self.k_norm(k).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)
        cos, sin = position_embeddings
        q = _apply_rotary_pos_emb(
            q, cos[:, -q_len:, :], sin[:, -q_len:, :], unsqueeze_dim=1
        )
        # K is roped over ctx + block positions; V gets no rope.
        k = _apply_rotary_pos_emb(k, cos, sin, unsqueeze_dim=1)
        k = k.repeat_interleave(self.num_key_value_groups, dim=1)
        v = v.repeat_interleave(self.num_key_value_groups, dim=1)
        attn_output = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            scale=self.scaling,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, -1)
        return self.o_proj(attn_output)


class RefDSparkDecoderLayer(nn.Module):
    """``Gemma4DSparkDecoderLayer``: 4-norm sandwich + layer_scalar."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        head_dim: int,
        rms_norm_eps: float,
    ) -> None:
        super().__init__()
        self.self_attn = RefDSparkAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
        )
        self.mlp = RefGemma4MLP(hidden_size, intermediate_size)
        self.input_layernorm = RefGemma4RMSNorm(hidden_size, eps=rms_norm_eps)
        self.post_attention_layernorm = RefGemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.pre_feedforward_layernorm = RefGemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.post_feedforward_layernorm = RefGemma4RMSNorm(
            hidden_size, eps=rms_norm_eps
        )
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            position_embeddings=position_embeddings,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states * self.layer_scalar


class RefDSparkBackbone(nn.Module):
    """``Gemma4DSparkModel`` backbone: fc + hidden_norm conditioning, the
    draft decoder stack, the final norm, and the soft-capped lm_head."""

    def __init__(self, cfg: DSparkGemma4DraftConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList(
            [
                RefDSparkDecoderLayer(
                    hidden_size=cfg.hidden_size,
                    intermediate_size=cfg.intermediate_size,
                    num_attention_heads=cfg.num_attention_heads,
                    num_key_value_heads=cfg.num_key_value_heads,
                    head_dim=cfg.head_dim,
                    rms_norm_eps=cfg.rms_norm_eps,
                )
                for _ in range(cfg.num_hidden_layers)
            ]
        )
        self.norm = RefGemma4RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.fc = nn.Linear(
            cfg.num_context_features * cfg.hidden_size,
            cfg.hidden_size,
            bias=False,
        )
        self.hidden_norm = RefGemma4RMSNorm(
            cfg.hidden_size, eps=cfg.rms_norm_eps
        )
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)

    def project_target_hidden(self, taps: torch.Tensor) -> torch.Tensor:
        return self.hidden_norm(self.fc(taps))

    def forward_backbone(
        self,
        noise_embedding: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = noise_embedding
        target_hidden_states = self.project_target_hidden(target_hidden_states)
        position_embeddings = _proportional_rope_cos_sin(
            position_ids,
            head_dim=self.cfg.head_dim,
            theta=self.cfg.rope_theta,
            partial_rotary_factor=self.cfg.partial_rotary_factor,
            dtype=hidden_states.dtype,
        )
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_states,
                position_embeddings=position_embeddings,
            )
        return self.norm(hidden_states)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states)
        softcap = self.cfg.final_logit_softcapping
        if softcap is not None:
            logits = torch.tanh(logits / softcap) * softcap
        return logits


class RefVanillaMarkov(nn.Module):
    """``VanillaMarkov`` (markov_head.py): bias = w2(w1[prev_token])."""

    def __init__(self, vocab_size: int, markov_rank: int) -> None:
        super().__init__()
        self.markov_w1 = nn.Embedding(vocab_size, markov_rank)
        self.markov_w2 = nn.Linear(markov_rank, vocab_size, bias=False)

    def compute_step_bias(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.markov_w2(self.markov_w1(token_ids.long()))

    def sample_block_tokens_greedy(
        self,
        base_logits: torch.Tensor,
        first_prev_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Temperature-0 ``sample_block_tokens``: sequential bias + argmax."""
        proposal_len = base_logits.shape[1]
        sampled_tokens = []
        prev_token_ids = first_prev_token_ids.long()
        for step_idx in range(proposal_len):
            step_logits = base_logits[:, step_idx, :] + self.compute_step_bias(
                prev_token_ids
            )
            next_token_ids = torch.argmax(step_logits, dim=-1)
            sampled_tokens.append(next_token_ids)
            prev_token_ids = next_token_ids
        return torch.stack(sampled_tokens, dim=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def draft_config() -> DSparkGemma4DraftConfig:
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        pytest.skip("HF Hub offline mode is enabled")
    config_path = hf_hub_download(HF_REPO_ID, "config.json")
    with open(config_path) as f:
        config_json = json.load(f)
    # Attribute-style access mirrors the transformers PretrainedConfig the
    # serving path passes; nested dicts (rope_parameters) stay dicts.
    return DSparkGemma4DraftConfig.from_huggingface_config(
        SimpleNamespace(**config_json), max_seq_len=MAX_SEQ_LEN
    )


@pytest.fixture(scope="module")
def checkpoint_weights() -> dict[str, torch.Tensor]:
    path = hf_hub_download(HF_REPO_ID, "model.safetensors")
    return load_file(path)


# The confidence head is out of scope for M1; the markov head is applied
# test-side; embeddings are fed as inputs (aliased to the target's in the
# unified pipeline).
_REF_SKIP_PREFIXES = ("confidence_head.", "markov_head.", "embed_tokens.")
# The MAX draft module owns the markov head but aliases embed_tokens and
# lm_head rather than owning them.
_DRAFT_SKIP_PREFIXES = ("confidence_head.", "embed_tokens.", "lm_head.")


@pytest.fixture(scope="module")
def torch_reference(
    draft_config: DSparkGemma4DraftConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> RefDSparkBackbone:
    model = RefDSparkBackbone(draft_config)
    backbone_sd = {
        k: v
        for k, v in checkpoint_weights.items()
        if not k.startswith(_REF_SKIP_PREFIXES)
    }
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    assert not missing, f"Reference is missing checkpoint keys: {missing}"
    assert not unexpected, f"Unexpected checkpoint keys: {unexpected}"
    return model.to(TORCH_DTYPE).to("cuda").eval()


@pytest.fixture(scope="module")
def torch_reference_fp32(
    draft_config: DSparkGemma4DraftConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> RefDSparkBackbone:
    """Float32 CPU oracle used to attribute bf16 divergence.

    Both bf16 implementations (MAX and the torch reference) are compared
    against this oracle: MAX must not be meaningfully less accurate than
    the reference's own bf16 rounding.
    """
    model = RefDSparkBackbone(draft_config)
    backbone_sd = {
        k: v
        for k, v in checkpoint_weights.items()
        if not k.startswith(_REF_SKIP_PREFIXES)
    }
    missing, unexpected = model.load_state_dict(backbone_sd, strict=False)
    assert not missing and not unexpected
    return model.to(torch.float32).eval()


@pytest.fixture(scope="module")
def torch_markov(
    draft_config: DSparkGemma4DraftConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> RefVanillaMarkov:
    markov = RefVanillaMarkov(draft_config.vocab_size, draft_config.markov_rank)
    markov.load_state_dict(
        {
            "markov_w1.weight": checkpoint_weights[
                "markov_head.markov_w1.weight"
            ],
            "markov_w2.weight": checkpoint_weights[
                "markov_head.markov_w2.weight"
            ],
        }
    )
    return markov.to(TORCH_DTYPE).to("cuda").eval()


class _DSparkHarness(Module):
    """DSpark draft module plus the checkpoint's frozen lm_head copy.

    In the unified pipeline the lm_head is aliased to the target's; the
    checkpoint ships an identical frozen copy which stands in for it here.
    """

    def __init__(
        self,
        config: DSparkGemma4DraftConfig,
        kv_params: MHAKVCacheParams,
        device_ref: DeviceRef,
    ) -> None:
        super().__init__()
        self.draft = DSparkGemma4(
            config,
            kv_params=kv_params,
            devices=[device_ref],
            dtype=MAX_DTYPE,
        )
        self.lm_head = Linear(
            in_dim=config.hidden_size,
            out_dim=config.vocab_size,
            dtype=MAX_DTYPE,
            device=device_ref,
            has_bias=False,
        )

    def __call__(self) -> None:
        raise NotImplementedError(
            "Weight container only; the test graph drives the children."
        )


class CompiledDraft(NamedTuple):
    compiled: Model
    kv_manager: PagedKVCacheManager


@pytest.fixture(scope="module")
def compiled_draft(
    session: InferenceSession,
    draft_config: DSparkGemma4DraftConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> CompiledDraft:
    device_ref = DeviceRef.GPU()
    kv_params = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        devices=[device_ref],
        n_kv_heads=draft_config.num_key_value_heads,
        head_dim=draft_config.head_dim,
        num_layers=draft_config.num_hidden_layers,
        page_size=128,
    )
    harness = _DSparkHarness(draft_config, kv_params, device_ref)

    state_dict = {
        f"draft.{k}": v.cpu()
        for k, v in checkpoint_weights.items()
        if not k.startswith(_DRAFT_SKIP_PREFIXES)
    }
    state_dict["lm_head.weight"] = checkpoint_weights["lm_head.weight"].cpu()
    harness.load_state_dict(state_dict, strict=True)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=32,
        session=session,
        max_batch_size=8,
    )

    num_ctx_features = draft_config.num_context_features
    ctx_features_type = TensorType(
        MAX_DTYPE,
        ["total_ctx_len", num_ctx_features * draft_config.hidden_size],
        device=device_ref,
    )
    block_embeds_type = TensorType(
        MAX_DTYPE,
        ["total_block_len", draft_config.hidden_size],
        device=device_ref,
    )
    offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    flattened_kv_types = kv_params.flattened_kv_inputs()

    with Graph(
        "dspark_gemma4_draft",
        input_types=(
            ctx_features_type,
            block_embeds_type,
            offsets_type,
            offsets_type,
            *flattened_kv_types,
        ),
    ) as graph:
        (
            ctx_features,
            block_embeds,
            ctx_offsets,
            block_offsets,
            *kv_inputs,
        ) = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_inputs)).inputs[0]

        ctx_hidden = harness.draft.project_target_hidden(ctx_features.tensor)
        harness.draft.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=ctx_offsets.tensor,
            kv_collection=kv_collection,
        )

        # The block forward attends over [materialized ctx ; block]: bump
        # cache_lengths by the per-sequence ctx length, mirroring the
        # unified DFlash graph.
        pre_cache_lengths = ops.rebind(
            kv_collection.cache_lengths, ["batch_size"]
        )
        ctx_offsets_t = ctx_offsets.tensor.rebind(["input_row_offsets_len"])
        ctx_lens = (ctx_offsets_t[1:] - ctx_offsets_t[:-1]).rebind(
            ["batch_size"]
        )
        block_kv_collection = replace(
            kv_collection,
            cache_lengths=pre_cache_lengths + ctx_lens,
        )

        block_hs = harness.draft.forward_block(
            input_embeds=block_embeds.tensor,
            kv_collection=block_kv_collection,
            input_row_offsets=block_offsets.tensor,
        )

        logits = harness.lm_head(block_hs)
        softcap = draft_config.final_logit_softcapping
        assert softcap is not None
        logits = ops.tanh(logits / softcap) * softcap

        graph.output(ctx_hidden, block_hs, logits)

    compiled = session.load(graph, weights_registry=harness.state_dict())
    return CompiledDraft(compiled=compiled, kv_manager=kv_manager)


def _execute_max_draft(
    compiled_draft: CompiledDraft,
    device: Device,
    ctx_features: torch.Tensor,
    block_embeds: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx_len = ctx_features.shape[0]
    block_len = block_embeds.shape[0]
    kv_manager = compiled_draft.kv_manager

    context = create_text_context(np.zeros(ctx_len + block_len))
    kv_manager.claim(context)
    try:
        kv_manager.alloc(context)
        kv_inputs = kv_manager.runtime_inputs(
            [[context]], max_cache_length=ctx_len + block_len
        )
        outputs = compiled_draft.compiled.execute(
            Buffer.from_dlpack(ctx_features.contiguous()).to(device),
            Buffer.from_dlpack(block_embeds.contiguous()).to(device),
            Buffer.from_numpy(np.array([0, ctx_len], dtype=np.uint32)).to(
                device
            ),
            Buffer.from_numpy(np.array([0, block_len], dtype=np.uint32)).to(
                device
            ),
            *kv_inputs.flatten(),
        )
    finally:
        kv_manager.release(context)
    ctx_hidden, block_hs, logits = (from_dlpack(out) for out in outputs)
    return ctx_hidden, block_hs, logits


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    cos = torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))
    return float(1.0 - cos)


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.to(torch.float32) - b.to(torch.float32)).abs().max())


def _mean_kl_div(ref_logits: torch.Tensor, max_logits: torch.Tensor) -> float:
    """Mean per-position KL(ref || max) over fp32 softmax distributions."""
    ref_logp = F.log_softmax(ref_logits.to(torch.float32), dim=-1)
    max_logp = F.log_softmax(max_logits.to(torch.float32), dim=-1)
    kl = (ref_logp.exp() * (ref_logp - max_logp)).sum(dim=-1)
    return float(kl.mean())


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctx_len", CTX_LENGTHS)
def test_dspark_draft_matches_reference(
    ctx_len: int,
    device: Device,
    draft_config: DSparkGemma4DraftConfig,
    checkpoint_weights: dict[str, torch.Tensor],
    torch_reference: RefDSparkBackbone,
    torch_reference_fp32: RefDSparkBackbone,
    torch_markov: RefVanillaMarkov,
    compiled_draft: CompiledDraft,
) -> None:
    cfg = draft_config
    block_size = cfg.block_size
    torch.manual_seed(20260728 + ctx_len)

    # Random context features standing in for the target's multi-layer
    # hidden-state taps, at a residual-stream-like magnitude.
    ctx_features = (
        torch.randn(ctx_len, cfg.num_context_features * cfg.hidden_size).to(
            TORCH_DTYPE
        )
        * 4.0
    ).to("cuda")

    anchor_token = int(
        torch.randint(low=8, high=cfg.vocab_size, size=(1,)).item()
    )
    block_ids = torch.full(
        (1, block_size), cfg.mask_token_id, dtype=torch.long, device="cuda"
    )
    block_ids[0, 0] = anchor_token

    # Both sides receive the identical scaled block embedding, computed with
    # the checkpoint's frozen embed_tokens copy (aliased to the target's in
    # the unified pipeline).
    embed_weight = checkpoint_weights["embed_tokens.weight"].to("cuda")
    embed_scale = torch.tensor(cfg.hidden_size**0.5).to(TORCH_DTYPE)
    block_embeds = (embed_weight[block_ids[0]] * embed_scale).to(TORCH_DTYPE)

    # Reference forward: ctx at positions [0, ctx_len), block at
    # [ctx_len, ctx_len + block_size).
    position_ids = torch.arange(ctx_len + block_size, device="cuda").unsqueeze(
        0
    )
    with torch.no_grad():
        ref_ctx_hidden = torch_reference.project_target_hidden(
            ctx_features.unsqueeze(0)
        )
        ref_block_hs = torch_reference.forward_backbone(
            noise_embedding=block_embeds.unsqueeze(0),
            target_hidden_states=ctx_features.unsqueeze(0),
            position_ids=position_ids,
        )
        ref_logits = torch_reference.compute_logits(ref_block_hs)

    max_ctx_hidden, max_block_hs, max_logits = _execute_max_draft(
        compiled_draft, device, ctx_features, block_embeds
    )

    # Float32 CPU oracle on the identical inputs: attributes bf16 divergence
    # between the two bf16 implementations to rounding rather than semantics.
    with torch.no_grad():
        oracle_block_hs = torch_reference_fp32.forward_backbone(
            noise_embedding=block_embeds.unsqueeze(0).cpu().float(),
            target_hidden_states=ctx_features.unsqueeze(0).cpu().float(),
            position_ids=position_ids.cpu(),
        )
        oracle_logits = torch_reference_fp32.compute_logits(oracle_block_hs)

    ctx_cos = _cos_dist(ref_ctx_hidden[0], max_ctx_hidden)
    ctx_max_abs = _max_abs_diff(ref_ctx_hidden[0], max_ctx_hidden)
    hs_cos = _cos_dist(ref_block_hs[0], max_block_hs)
    hs_max_abs = _max_abs_diff(ref_block_hs[0], max_block_hs)
    logits_cos = _cos_dist(ref_logits[0], max_logits)
    logits_max_abs = _max_abs_diff(ref_logits[0], max_logits)
    kl = _mean_kl_div(ref_logits[0], max_logits.to("cuda"))

    oracle_hs = oracle_block_hs[0]
    oracle_lg = oracle_logits[0]
    hs_cos_max_vs_oracle = _cos_dist(oracle_hs, max_block_hs.cpu())
    hs_cos_ref_vs_oracle = _cos_dist(oracle_hs, ref_block_hs[0].cpu())
    logits_cos_max_vs_oracle = _cos_dist(oracle_lg, max_logits.cpu())
    logits_cos_ref_vs_oracle = _cos_dist(oracle_lg, ref_logits[0].cpu())
    kl_max_vs_oracle = _mean_kl_div(oracle_lg, max_logits.cpu())
    kl_ref_vs_oracle = _mean_kl_div(oracle_lg, ref_logits[0].cpu())

    # Teacher-forced greedy comparison: both sides consume the REF chain's
    # prev token at every position, so one borderline argmax flip cannot
    # cascade through the rest of the block. A mismatch is excused only when
    # the ref's own top-2 corrected-logit gap at that position is inside
    # bf16-tie territory — a semantic bug (e.g. the causal-mask RED probe)
    # mismatches at positions with gaps orders of magnitude wider.
    with torch.no_grad():
        first_prev = torch.tensor([anchor_token], device="cuda")
        ref_tokens = torch_markov.sample_block_tokens_greedy(
            ref_logits, first_prev_token_ids=first_prev
        )[0]
        max_logits_cuda = max_logits.to("cuda")
        mismatches = []
        tie_excused = []
        prev = first_prev
        for k in range(block_size):
            bias = torch_markov.compute_step_bias(prev)
            ref_row = ref_logits[0, k, :] + bias[0]
            max_row = max_logits_cuda[k, :] + bias[0]
            ref_tok = int(torch.argmax(ref_row))
            max_tok = int(torch.argmax(max_row))
            if max_tok != ref_tok:
                top2 = torch.topk(ref_row.float(), 2).values
                gap = float(top2[0] - top2[1])
                if gap < GREEDY_TIE_GAP_EPS:
                    tie_excused.append((k, ref_tok, max_tok, round(gap, 4)))
                else:
                    mismatches.append((k, ref_tok, max_tok, round(gap, 4)))
            prev = ref_tokens[k : k + 1]

    print(
        f"[dspark ctx={ctx_len}] ctx_hidden cos_dist={ctx_cos:.3e}"
        f" max_abs={ctx_max_abs:.3e} | block_hs cos_dist={hs_cos:.3e}"
        f" max_abs={hs_max_abs:.3e} | logits cos_dist={logits_cos:.3e}"
        f" max_abs={logits_max_abs:.3e} kl={kl:.3e}"
        f" | greedy match {block_size - len(mismatches) - len(tie_excused)}"
        f"/{block_size} (tie-excused: {tie_excused})"
        f" (mismatches: {mismatches})"
    )
    print(
        f"[dspark ctx={ctx_len}] vs fp32 oracle:"
        f" block_hs cos_dist max={hs_cos_max_vs_oracle:.3e}"
        f" ref_bf16={hs_cos_ref_vs_oracle:.3e}"
        f" | logits cos_dist max={logits_cos_max_vs_oracle:.3e}"
        f" ref_bf16={logits_cos_ref_vs_oracle:.3e}"
        f" | kl max={kl_max_vs_oracle:.3e} ref_bf16={kl_ref_vs_oracle:.3e}"
    )

    assert ctx_cos < COS_DIST_THRESHOLD, (
        f"ctx_hidden cosine distance {ctx_cos:.3e} exceeds {COS_DIST_THRESHOLD}"
    )
    assert logits_cos < COS_DIST_THRESHOLD, (
        f"base-logits cosine distance {logits_cos:.3e} exceeds"
        f" {COS_DIST_THRESHOLD}"
    )
    # Intermediates and KL are gated against the fp32 oracle: MAX must be
    # about as close to the true function as the torch reference's own bf16
    # rounding (bf16-vs-bf16 distance alone conflates the two rounding
    # paths). The absolute backstops catch correlated-but-wrong outputs.
    assert (
        hs_cos_max_vs_oracle < _ORACLE_MARGIN * hs_cos_ref_vs_oracle + 1e-4
    ), (
        f"block hidden states: MAX-vs-fp32 {hs_cos_max_vs_oracle:.3e} is"
        f" worse than {_ORACLE_MARGIN}x the reference's own bf16 error"
        f" {hs_cos_ref_vs_oracle:.3e}"
    )
    assert hs_cos < HS_COS_DIST_BACKSTOP, (
        f"block hidden-state cosine distance {hs_cos:.3e} exceeds"
        f" {HS_COS_DIST_BACKSTOP}"
    )
    assert kl_max_vs_oracle < _ORACLE_MARGIN * kl_ref_vs_oracle + 1e-3, (
        f"base-logits KL: MAX-vs-fp32 {kl_max_vs_oracle:.3e} is worse than"
        f" {_ORACLE_MARGIN}x the reference's own bf16 KL"
        f" {kl_ref_vs_oracle:.3e}"
    )
    assert kl < KL_DIV_BACKSTOP, (
        f"base-logits bf16-vs-bf16 KL {kl:.3e} exceeds {KL_DIV_BACKSTOP}"
    )
    assert not mismatches, (
        f"markov-corrected greedy tokens diverged beyond bf16-tie gaps at"
        f" {mismatches} (tie-excused: {tie_excused})"
    )
    assert len(tie_excused) <= 2, (
        f"too many tie-excused greedy flips: {tie_excused}"
    )
