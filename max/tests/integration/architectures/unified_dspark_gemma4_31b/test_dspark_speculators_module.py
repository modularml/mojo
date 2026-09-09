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
"""Logit verification for the speculators-format DSpark draft module.

Loads the real ``RedHatAI/gemma-4-31B-it-speculator.dspark`` checkpoint and
compares the MAX draft module against a faithful PyTorch port of the vLLM
runtime reference (``qwen3_dflash.py`` / ``qwen3_dspark.py`` /
``dspark/speculator.py``). Both implementations receive identical random
target-tap context features and an identical ``[anchor, mask x 7]`` block,
and are compared on:

1. the projected context hidden states (``hidden_norm(fc(ctx))``),
2. the block hidden states after the 5-layer causal-SWA draft backbone
   (all 8 slots),
3. the base draft logits from the draft-owned 32k lm_head (no softcap), and
4. the markov-corrected greedy TARGET-vocab draft ids (in-chain d2t) for the
   7 drafted slots (anchor slot 0 dropped, ``sample_from_anchor: false``).

Context lengths straddle the 2048 sliding window (128 and 4096) and include
the exact window/page boundary (2048).
"""

from __future__ import annotations

import json
import os
from dataclasses import replace
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
from max.pipelines.architectures.dspark_draft import (
    dspark_speculators_draft,
)
from max.pipelines.kv_cache import PagedKVCacheManager
from safetensors.torch import load_file
from speculators_reference import (
    DraftCheckpointConfig,
    RefDSparkSpeculatorsBackbone,
    RefMarkovD2T,
    reference_state_dict,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack

HF_REPO_ID = "RedHatAI/gemma-4-31B-it-speculator.dspark"
TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16

MAX_SEQ_LEN = 8192
# 128: window inactive; 2048: first block query sits exactly at the window
# (and KV page) boundary; 4096: window fully active for every query.
CTX_LENGTHS = [128, 2048, 4096]

# Repo logit-verify conventions (see max/CLAUDE.md) for the logits and the
# fc projection.
COS_DIST_THRESHOLD = 1e-3
# Intermediate block hidden states and KL are gated relative to a float32
# oracle: MAX must stay within a small factor of the torch reference's OWN
# bf16 rounding error (flash-attention vs SDPA reduction order legitimately
# differs at the bf16 floor). A truly wrong semantic (e.g. a non-causal
# instead of causal-SWA mask) diverges by orders of magnitude, far beyond
# the margin and the absolute backstops.
_ORACLE_MARGIN = 2.5
HS_COS_DIST_BACKSTOP = 1e-2
# Sub-bf16-resolution top-2 gap under which a greedy argmax flip between two
# implementations is a numerical tie, not a semantic difference.
GREEDY_TIE_GAP_EPS = 0.25
KL_DIV_BACKSTOP = 0.05


# The PyTorch reference (vLLM speculators-DSpark runtime port) lives in
# speculators_reference.py, shared with the unified-graph step-parity test.

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
def draft_config() -> DraftCheckpointConfig:
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        pytest.skip("HF Hub offline mode is enabled")
    config_path = hf_hub_download(HF_REPO_ID, "config.json")
    with open(config_path) as f:
        config_json = json.load(f)
    return DraftCheckpointConfig.from_config_json(config_json)


@pytest.fixture(scope="module")
def checkpoint_weights() -> dict[str, torch.Tensor]:
    path = hf_hub_download(HF_REPO_ID, "model.safetensors")
    return load_file(path)


# The MAX module owns lm_head, markov head, and d2t; embed_tokens is fed as
# an input (aliased to the target's raw rows in the unified pipeline).
_DRAFT_SKIP_KEYS = ("t2d",)
_DRAFT_SKIP_PREFIXES = ("confidence_head.", "embed_tokens.")


@pytest.fixture(scope="module")
def torch_reference(
    draft_config: DraftCheckpointConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> RefDSparkSpeculatorsBackbone:
    model = RefDSparkSpeculatorsBackbone(draft_config)
    missing, unexpected = model.load_state_dict(
        reference_state_dict(checkpoint_weights), strict=False
    )
    assert not missing, f"Reference is missing checkpoint keys: {missing}"
    assert not unexpected, f"Unexpected checkpoint keys: {unexpected}"
    return model.to(TORCH_DTYPE).to("cuda").eval()


@pytest.fixture(scope="module")
def torch_reference_fp32(
    draft_config: DraftCheckpointConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> RefDSparkSpeculatorsBackbone:
    """Float32 CPU oracle used to attribute bf16 divergence.

    Both bf16 implementations (MAX and the torch reference) are compared
    against this oracle: MAX must not be meaningfully less accurate than
    the reference's own bf16 rounding.
    """
    model = RefDSparkSpeculatorsBackbone(draft_config)
    missing, unexpected = model.load_state_dict(
        reference_state_dict(checkpoint_weights), strict=False
    )
    assert not missing and not unexpected
    return model.to(torch.float32).eval()


@pytest.fixture(scope="module")
def torch_markov(
    draft_config: DraftCheckpointConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> RefMarkovD2T:
    markov = RefMarkovD2T(
        draft_config.vocab_size,
        draft_config.draft_vocab_size,
        draft_config.markov_rank,
    )
    markov.load_state_dict(
        {
            "markov_w1.weight": checkpoint_weights[
                "markov_head.markov_w1.weight"
            ],
            "markov_w2.weight": checkpoint_weights[
                "markov_head.markov_w2.weight"
            ],
            "d2t": checkpoint_weights["d2t"],
        }
    )
    markov = markov.to(TORCH_DTYPE).to("cuda").eval()
    # d2t must stay integral; .to(dtype) only touches floating-point params.
    assert markov.d2t.dtype == torch.long
    return markov


class CompiledDraft(NamedTuple):
    compiled: Model
    kv_manager: PagedKVCacheManager


@pytest.fixture(scope="module")
def compiled_draft(
    session: InferenceSession,
    draft_config: DraftCheckpointConfig,
    checkpoint_weights: dict[str, torch.Tensor],
) -> CompiledDraft:
    cfg = draft_config
    device_ref = DeviceRef.GPU()
    kv_params = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        devices=[device_ref],
        n_kv_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        num_layers=cfg.num_hidden_layers,
        page_size=128,
    )
    draft = dspark_speculators_draft.DSparkSpeculatorsDraft(
        hidden_size=cfg.hidden_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        num_key_value_heads=cfg.num_key_value_heads,
        head_dim=cfg.head_dim,
        intermediate_size=cfg.intermediate_size,
        rms_norm_eps=cfg.rms_norm_eps,
        rope_theta=cfg.rope_theta,
        sliding_window=cfg.sliding_window,
        layer_causal=list(cfg.layer_causal),
        vocab_size=cfg.vocab_size,
        draft_vocab_size=cfg.draft_vocab_size,
        markov_rank=cfg.markov_rank,
        block_size=cfg.block_size,
        sample_from_anchor=cfg.sample_from_anchor,
        mask_token_id=cfg.mask_token_id,
        num_context_features=cfg.num_context_features,
        max_seq_len=MAX_SEQ_LEN,
        kv_params=kv_params,
        devices=[device_ref],
        dtype=MAX_DTYPE,
    )

    # Checkpoint keys load verbatim (StackedLinear exposes q/k/v_proj under
    # their checkpoint names; markov_head/lm_head/d2t match by construction).
    state_dict = {
        k: v.cpu()
        for k, v in checkpoint_weights.items()
        if k not in _DRAFT_SKIP_KEYS and not k.startswith(_DRAFT_SKIP_PREFIXES)
    }
    draft.load_state_dict(state_dict, strict=True)

    kv_manager = PagedKVCacheManager(
        params=kv_params,
        total_num_pages=64,
        session=session,
        max_batch_size=8,
    )

    ctx_features_type = TensorType(
        MAX_DTYPE,
        ["total_ctx_len", cfg.num_context_features * cfg.hidden_size],
        device=device_ref,
    )
    # 3D so the post-lm_head reshape back to [batch, block, vocab] is a
    # provable product split (a flat free "total_block_len" dim is not
    # symbolically divisible by block_size).
    block_embeds_type = TensorType(
        MAX_DTYPE,
        ["batch_size", cfg.block_size, cfg.hidden_size],
        device=device_ref,
    )
    offsets_type = TensorType(
        DType.uint32, shape=["input_row_offsets_len"], device=device_ref
    )
    anchor_type = TensorType(DType.int64, ["batch_size"], device=device_ref)
    flattened_kv_types = kv_params.flattened_kv_inputs()

    with Graph(
        "dspark_speculators_draft",
        input_types=(
            ctx_features_type,
            block_embeds_type,
            offsets_type,
            offsets_type,
            anchor_type,
            *flattened_kv_types,
        ),
    ) as graph:
        (
            ctx_features,
            block_embeds,
            ctx_offsets,
            block_offsets,
            anchor_tokens,
            *kv_inputs,
        ) = graph.inputs
        kv_collection = kv_params.unflatten_kv_inputs(iter(kv_inputs)).inputs[0]

        ctx_hidden = draft.project_target_hidden(ctx_features.tensor)
        draft.materialize_kv(
            ctx_hidden=ctx_hidden,
            input_row_offsets=ctx_offsets.tensor,
            kv_collection=kv_collection,
        )

        # The block forward attends over [materialized ctx ; block]: bump
        # cache_lengths by the per-sequence ctx length, mirroring the
        # unified graph.
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

        block_embeds_flat = block_embeds.tensor.reshape((-1, cfg.hidden_size))
        block_hs = draft.forward_block(
            input_embeds=block_embeds_flat,
            kv_collection=block_kv_collection,
            input_row_offsets=block_offsets.tensor,
        )

        logits = draft.lm_head(block_hs)

        # sample_from_anchor=false: the anchor slot's logits are dropped and
        # the chain runs over the block_size-1 drafted slots, exactly as the
        # unified graph will consume the module.
        logits_3d = logits.reshape(
            ("batch_size", cfg.block_size, cfg.draft_vocab_size)
        )
        draft_target_ids = draft.sample_draft_tokens(
            logits_3d[:, 1:, :], anchor_tokens.tensor
        )

        graph.output(ctx_hidden, block_hs, logits, draft_target_ids)

    compiled = session.load(graph, weights_registry=draft.state_dict())
    return CompiledDraft(compiled=compiled, kv_manager=kv_manager)


def _execute_max_draft(
    compiled_draft: CompiledDraft,
    device: Device,
    ctx_features: torch.Tensor,
    block_embeds: torch.Tensor,
    anchor_token: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    ctx_len = ctx_features.shape[0]
    block_len = block_embeds.shape[0]
    kv_manager = compiled_draft.kv_manager

    context = create_text_context(
        np.zeros(ctx_len + block_len), max_length=MAX_SEQ_LEN
    )
    kv_manager.claim(context)
    try:
        kv_manager.alloc(context)
        kv_inputs = kv_manager.runtime_inputs(
            [[context]], max_cache_length=ctx_len + block_len
        )
        outputs = compiled_draft.compiled.execute(
            Buffer.from_dlpack(ctx_features.contiguous()).to(device),
            Buffer.from_dlpack(block_embeds.unsqueeze(0).contiguous()).to(
                device
            ),
            Buffer.from_numpy(np.array([0, ctx_len], dtype=np.uint32)).to(
                device
            ),
            Buffer.from_numpy(np.array([0, block_len], dtype=np.uint32)).to(
                device
            ),
            Buffer.from_numpy(np.array([anchor_token], dtype=np.int64)).to(
                device
            ),
            *kv_inputs.flatten(),
        )
    finally:
        kv_manager.release(context)
    ctx_hidden, block_hs, logits, draft_ids = (
        from_dlpack(out) for out in outputs
    )
    return ctx_hidden, block_hs, logits, draft_ids


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
def test_dspark_speculators_draft_matches_reference(
    ctx_len: int,
    device: Device,
    draft_config: DraftCheckpointConfig,
    checkpoint_weights: dict[str, torch.Tensor],
    torch_reference: RefDSparkSpeculatorsBackbone,
    torch_reference_fp32: RefDSparkSpeculatorsBackbone,
    torch_markov: RefMarkovD2T,
    compiled_draft: CompiledDraft,
) -> None:
    cfg = draft_config
    block_size = cfg.block_size
    num_drafts = block_size - 1
    torch.manual_seed(20260801 + ctx_len)

    # Random context features standing in for the target's multi-layer
    # hidden-state taps, at a residual-stream-like magnitude. Real tap dumps
    # are the tap-parity slice's business, not this module test's.
    ctx_features = (
        torch.randn(ctx_len, cfg.num_context_features * cfg.hidden_size).to(
            TORCH_DTYPE
        )
        * 4.0
    ).to("cuda")

    anchor_token = int(
        torch.randint(low=8, high=cfg.vocab_size, size=(1,)).item()
    )
    block_ids = torch.full((block_size,), cfg.mask_token_id, dtype=torch.long)
    block_ids[0] = anchor_token

    # Both sides receive the identical RAW block embedding rows — NO gemma
    # sqrt(hidden) scaling on the speculators draft path.
    embed_weight = checkpoint_weights["embed_tokens.weight"]
    block_embeds = embed_weight[block_ids].to(TORCH_DTYPE).to("cuda")

    with torch.no_grad():
        ref_ctx_hidden = torch_reference.project_target_hidden(
            ctx_features.unsqueeze(0)
        )
        ref_block_hs = torch_reference.forward_backbone(
            block_embedding=block_embeds.unsqueeze(0),
            target_taps=ctx_features.unsqueeze(0),
        )
        ref_logits = torch_reference.compute_logits(ref_block_hs)

    max_ctx_hidden, max_block_hs, max_logits, max_chain_ids = (
        _execute_max_draft(
            compiled_draft, device, ctx_features, block_embeds, anchor_token
        )
    )

    # Float32 CPU oracle on the identical inputs: attributes bf16 divergence
    # between the two bf16 implementations to rounding rather than semantics.
    with torch.no_grad():
        oracle_block_hs = torch_reference_fp32.forward_backbone(
            block_embedding=block_embeds.unsqueeze(0).cpu().float(),
            target_taps=ctx_features.unsqueeze(0).cpu().float(),
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
    kl_max_vs_oracle = _mean_kl_div(oracle_lg, max_logits.cpu())
    kl_ref_vs_oracle = _mean_kl_div(oracle_lg, ref_logits[0].cpu())

    # Teacher-forced greedy comparison over the 7 drafted slots: both sides
    # consume the REF chain's prev token at every position, so one borderline
    # argmax flip cannot cascade through the rest of the block. A mismatch is
    # excused only when the ref's own top-2 corrected-logit gap at that
    # position is inside bf16-tie territory — a semantic bug (e.g. a wrong
    # mask) mismatches at positions with gaps orders of magnitude wider.
    # Slot 0 (anchor) is dropped; chain step k corresponds to block slot k+1.
    with torch.no_grad():
        first_prev = torch.tensor([anchor_token], device="cuda")
        ref_chain = torch_markov.sample_block_tokens_greedy(
            ref_logits[:, 1:, :], anchor_target_ids=first_prev
        )[0]
        max_logits_cuda = max_logits.to("cuda")
        mismatches = []
        tie_excused = []
        prev = first_prev
        for k in range(num_drafts):
            bias = torch_markov.compute_step_bias(prev)
            ref_row = ref_logits[0, k + 1, :] + bias[0]
            max_row = max_logits_cuda[k + 1, :] + bias[0]
            ref_tok = int(
                torch_markov.map_draft_to_target(torch.argmax(ref_row))
            )
            max_tok = int(
                torch_markov.map_draft_to_target(torch.argmax(max_row))
            )
            if max_tok != ref_tok:
                top2 = torch.topk(ref_row.float(), 2).values
                gap = float(top2[0] - top2[1])
                if gap < GREEDY_TIE_GAP_EPS:
                    tie_excused.append((k, ref_tok, max_tok, round(gap, 4)))
                else:
                    mismatches.append((k, ref_tok, max_tok, round(gap, 4)))
            prev = ref_chain[k : k + 1]

        # In-graph chain wiring check: rerun the exact chain math in torch on
        # MAX's own base logits; the graph's unrolled chain (bias, argmax,
        # in-chain d2t, prev threading) must reproduce it. Comparison stops
        # at the first bf16-tie flip, after which the chains legitimately
        # diverge.
        torch_chain_on_max = torch_markov.sample_block_tokens_greedy(
            max_logits_cuda.unsqueeze(0)[:, 1:, :],
            anchor_target_ids=first_prev,
        )[0]
        chain_mismatch = None
        chain_compared = num_drafts
        prev = first_prev
        for k in range(num_drafts):
            expected = int(torch_chain_on_max[k])
            actual = int(max_chain_ids[0, k])
            if actual != expected:
                bias = torch_markov.compute_step_bias(prev)[0]
                row = max_logits_cuda[k + 1, :] + bias
                top2 = torch.topk(row.float(), 2).values
                gap = float(top2[0] - top2[1])
                if gap < GREEDY_TIE_GAP_EPS:
                    chain_compared = k
                    break
                chain_mismatch = (k, expected, actual, round(gap, 4))
                break
            prev = torch_chain_on_max[k : k + 1]

    # Every emitted id must be a d2t-mapped target id.
    valid_target_ids = (
        torch.arange(cfg.draft_vocab_size) + torch_markov.d2t.cpu()
    )
    assert bool(
        torch.isin(max_chain_ids.cpu().flatten(), valid_target_ids).all()
    ), f"in-graph chain emitted ids outside the d2t image: {max_chain_ids}"

    print(
        f"[dspark-spec ctx={ctx_len}] ctx_hidden cos_dist={ctx_cos:.3e}"
        f" max_abs={ctx_max_abs:.3e} | block_hs cos_dist={hs_cos:.3e}"
        f" max_abs={hs_max_abs:.3e} | logits cos_dist={logits_cos:.3e}"
        f" max_abs={logits_max_abs:.3e} kl={kl:.3e}"
        f" | greedy match {num_drafts - len(mismatches) - len(tie_excused)}"
        f"/{num_drafts} (tie-excused: {tie_excused})"
        f" (mismatches: {mismatches})"
        f" | in-graph chain compared {chain_compared}/{num_drafts}"
    )
    print(
        f"[dspark-spec ctx={ctx_len}] vs fp32 oracle:"
        f" block_hs cos_dist max={hs_cos_max_vs_oracle:.3e}"
        f" ref_bf16={hs_cos_ref_vs_oracle:.3e}"
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
        f" {_ORACLE_MARGIN}x the reference's own bf16 KL {kl_ref_vs_oracle:.3e}"
    )
    assert kl < KL_DIV_BACKSTOP, (
        f"base-logits bf16-vs-bf16 KL {kl:.3e} exceeds {KL_DIV_BACKSTOP}"
    )
    assert not mismatches, (
        f"markov-corrected greedy target ids diverged beyond bf16-tie gaps at"
        f" {mismatches} (tie-excused: {tie_excused})"
    )
    assert len(tie_excused) <= 2, (
        f"too many tie-excused greedy flips: {tie_excused}"
    )
    assert chain_mismatch is None, (
        f"in-graph markov chain diverged from the chain math on MAX's own"
        f" logits at {chain_mismatch}"
    )
    assert chain_compared >= 1, (
        "in-graph chain comparison ended at slot 0 on a tie; no slots verified"
    )
