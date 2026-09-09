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
"""Single-step parity for the unified speculators-DSpark Gemma4 graph.

Compiles the production unified graph at real dims with the REAL
``google/gemma-4-31B-it`` target and ``RedHatAI/gemma-4-31B-it-
speculator.dspark`` draft weights, runs a prefill step and one full
speculative decode step on a fixed prompt, and checks against the shared
torch reference (``speculators_reference.py``, the vLLM runtime port):

1. accepted-count and the committed next token reproduce greedy rejection
   sampling on MAX's own verify logits (temperature 0: accept iff draft ==
   target argmax), tie-aware;
2. the drafted TARGET-vocab ids match the reference markov chain run on
   MAX's own target taps (identical inputs isolate the in-graph draft
   path), tie-aware, and land inside the d2t image;
3. the block inputs are the RAW embedding rows — slot 0 the anchor/bonus
   token and slots 1..7 ``embed_tokens[mask_token_id]`` — bit-exact, at
   positions ``p+1..p+7`` (the block's KV write offset ``p`` = the
   post-commit sequence length, asserted via the debug cache lengths; the
   reference mask places its queries at the same absolute positions).

The graph's build-time debug values are added as extra outputs here; the
production 3-output ABI is asserted unchanged.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from max.driver import Accelerator, Buffer, Device
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import DeviceRef, Graph, TensorValue
from max.graph.weights import load_weights
from max.nn.comm import Signals
from max.nn.kv_cache import (
    BatchCharacteristics,
    MHAKVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.gemma4.weight_adapters import (
    convert_safetensor_language_state_dict,
)
from max.pipelines.architectures.speculators_common import (
    DSparkSpeculatorsDraftConfig,
    merge_unified_state_dict,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model_config import (
    UnifiedDSparkGemma4_31BConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.unified_dspark_gemma4_31b import (
    UnifiedDSparkGemma4_31B,
)
from max.pipelines.context import TextContext
from max.pipelines.kv_cache import PagedKVCacheManager
from max.pipelines.lib.config import SpeculativeConfig
from safetensors.torch import load_file
from speculators_reference import (
    DraftCheckpointConfig,
    RefDSparkSpeculatorsBackbone,
    RefMarkovD2T,
    reference_state_dict,
)
from test_common.context_utils import create_text_context
from torch.utils.dlpack import from_dlpack
from transformers import AutoTokenizer

TARGET_REPO_ID = "google/gemma-4-31B-it"
DRAFT_REPO_ID = "RedHatAI/gemma-4-31B-it-speculator.dspark"
MAX_DTYPE = DType.bfloat16
TORCH_DTYPE = torch.bfloat16
MAX_SEQ_LEN = 4096

BLOCK_SIZE = 8
NUM_DRAFTS = BLOCK_SIZE - 1
MASK_TOKEN_ID = 4

# Sub-bf16-resolution top-2 gap under which a greedy argmax flip between two
# implementations is a numerical tie, not a semantic difference (module-test
# convention).
GREEDY_TIE_GAP_EPS = 0.25
# Base-draft-logits agreement on identical inputs (repo logit-verify
# convention, same gate as the module test).
COS_DIST_THRESHOLD = 1e-3

PROMPT = (
    "Attention mechanisms let a transformer weigh every token in the"
    " context against every other token, so the residual stream at each"
    " layer accumulates increasingly abstract features of the sequence."
    " Speculative decoding exploits the fact that a small draft model can"
    " often predict the next few tokens of a much larger model."
)


def _cos_dist(a: torch.Tensor, b: torch.Tensor) -> float:
    af = a.to(torch.float32).flatten()
    bf = b.to(torch.float32).flatten()
    cos = torch.dot(af, bf) / (torch.linalg.norm(af) * torch.linalg.norm(bf))
    return float(1.0 - cos)


def _build_target_config(
    config_json: dict[str, Any], devices: list[DeviceRef]
) -> Gemma4ForConditionalGenerationConfig:
    """Real-dims MAX target config from google/gemma-4-31B-it's config.json.

    Mirrors the tap-parity builder; the SELECTED_LAYERS capture wiring is
    applied by ``UnifiedDSparkGemma4_31BConfig.__post_init__``.
    """
    tc = config_json["text_config"]
    layer_types: list[str] = tc["layer_types"]
    sliding_layers = sum(1 for t in layer_types if t == "sliding_attention")
    global_layers = sum(1 for t in layer_types if t == "full_attention")
    global_rope = tc["rope_parameters"]["full_attention"]
    sliding_rope = tc["rope_parameters"]["sliding_attention"]
    assert global_rope["rope_type"] == "proportional"
    assert sliding_rope["rope_type"] == "default"
    activation_map = {"gelu_pytorch_tanh": "gelu_tanh"}

    sliding_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=tc["num_global_key_value_heads"],
        head_dim=tc["global_head_dim"],
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )
    text_kv = MHAKVCacheParams(
        dtype=MAX_DTYPE,
        n_kv_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        num_layers=tc["num_hidden_layers"],
        devices=devices,
        page_size=128,
    )
    text_config = Gemma4TextConfig(
        vocab_size=tc["vocab_size"],
        hidden_size=tc["hidden_size"],
        intermediate_size=tc["intermediate_size"],
        num_hidden_layers=tc["num_hidden_layers"],
        num_attention_heads=tc["num_attention_heads"],
        num_key_value_heads=tc["num_key_value_heads"],
        head_dim=tc["head_dim"],
        hidden_activation=activation_map.get(
            tc["hidden_activation"], tc["hidden_activation"]
        ),
        max_position_embeddings=tc["max_position_embeddings"],
        max_seq_len=MAX_SEQ_LEN,
        rms_norm_eps=tc["rms_norm_eps"],
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=tc["attention_bias"],
        sliding_window=tc["sliding_window"],
        final_logit_softcapping=tc["final_logit_softcapping"],
        attn_logit_softcapping=None,
        rope_local_base_freq=sliding_rope["rope_theta"],
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=tc["num_global_key_value_heads"],
        global_head_dim=tc["global_head_dim"],
        attention_k_eq_v=tc["attention_k_eq_v"],
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=global_rope["partial_rotary_factor"]
        ),
        global_rope_theta=global_rope["rope_theta"],
        sliding_window_rope_theta=sliding_rope["rope_theta"],
        layer_types=layer_types,
    )
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        image_token_index=config_json["image_token_id"],
        text_config=text_config,
        vision_config=None,
        tie_word_embeddings=config_json["tie_word_embeddings"],
    )


class _RecordingUnified(UnifiedDSparkGemma4_31B):
    """Records the graph-build intermediates the parity gates check.

    The production module keeps no test hooks, so this subclass intercepts
    the sub-module boundaries each intermediate crosses during the build:
    ``project_target_hidden`` receives the fused target-tap concat, the
    acceptance sampler receives the target logits, ``forward_block``
    receives the block embeddings and the bumped block cache lengths, and
    ``sample_draft_tokens`` receives the pre-markov base draft logits.
    """

    def __init__(self, config: UnifiedDSparkGemma4_31BConfig) -> None:
        super().__init__(config)
        self.recorded: dict[str, TensorValue] = {}
        draft = self.draft
        orig_project = draft.project_target_hidden
        orig_forward_block = draft.forward_block
        orig_sample_draft_tokens = draft.sample_draft_tokens
        orig_sampler = self.acceptance_sampler

        def project_target_hidden(
            target_hs_concat: TensorValue,
        ) -> TensorValue:
            self.recorded["target_hs_concat"] = target_hs_concat
            return orig_project(target_hs_concat)

        def forward_block(
            *,
            input_embeds: TensorValue,
            kv_collection: PagedCacheValues,
            input_row_offsets: TensorValue,
        ) -> TensorValue:
            self.recorded["block_embeds"] = input_embeds
            self.recorded["block_cache_lengths"] = kv_collection.cache_lengths
            return orig_forward_block(
                input_embeds=input_embeds,
                kv_collection=kv_collection,
                input_row_offsets=input_row_offsets,
            )

        def sample_draft_tokens(
            base_logits: TensorValue, anchor_tokens: TensorValue
        ) -> TensorValue:
            self.recorded["base_logits"] = base_logits
            return orig_sample_draft_tokens(base_logits, anchor_tokens)

        def acceptance_sampler(
            draft_tokens: TensorValue,
            target_logits: TensorValue,
            **kwargs: TensorValue | None,
        ) -> tuple[TensorValue, TensorValue, TensorValue]:
            self.recorded["target_logits"] = target_logits
            return orig_sampler(draft_tokens, target_logits, **kwargs)

        # setattr keeps mypy happy: these are deliberate instance-level
        # wrappers over bound methods, applied before the single graph build.
        setattr(draft, "project_target_hidden", project_target_hidden)  # noqa: B010
        setattr(draft, "forward_block", forward_block)  # noqa: B010
        setattr(draft, "sample_draft_tokens", sample_draft_tokens)  # noqa: B010
        setattr(self, "acceptance_sampler", acceptance_sampler)  # noqa: B010


class _StepHarness:
    """Compiled unified graph plus the buffers to drive it step by step."""

    def __init__(self, session: InferenceSession, device: Device) -> None:
        self.device = device
        device_ref = DeviceRef.GPU()

        draft_config_path = hf_hub_download(DRAFT_REPO_ID, "config.json")
        with open(draft_config_path) as f:
            draft_config_json = json.load(f)
        self.ref_config = DraftCheckpointConfig.from_config_json(
            draft_config_json
        )
        draft_config = DSparkSpeculatorsDraftConfig.from_huggingface_config(
            draft_config_json, max_seq_len=8192
        )

        snapshot_dir = Path(
            snapshot_download(
                TARGET_REPO_ID,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )
        with open(snapshot_dir / "config.json") as f:
            target_config_json = json.load(f)
        target_config = _build_target_config(target_config_json, [device_ref])

        draft_kv_params = MHAKVCacheParams(
            dtype=MAX_DTYPE,
            n_kv_heads=draft_config.num_key_value_heads,
            head_dim=draft_config.head_dim,
            num_layers=draft_config.num_hidden_layers,
            devices=[device_ref],
            page_size=128,
        )
        config = UnifiedDSparkGemma4_31BConfig(
            target=target_config,
            draft=draft_config,
            draft_kv_params=draft_kv_params,
            speculative_config=SpeculativeConfig(
                speculative_method="dflash",
                num_speculative_tokens=NUM_DRAFTS,
            ),
            target_layer_ids=list(draft_config.target_layer_ids),
            mask_token_id=draft_config.mask_token_id,
            block_size=draft_config.block_size,
        )
        config.validate_dspark_fields()
        assert config.block_size == BLOCK_SIZE
        assert config.mask_token_id == MASK_TOKEN_ID

        nn_model = _RecordingUnified(config)

        target_weights = load_weights(
            sorted(snapshot_dir.glob("*.safetensors"))
        )
        target_state_dict = convert_safetensor_language_state_dict(
            dict(target_weights.items())
        )
        draft_path = hf_hub_download(DRAFT_REPO_ID, "model.safetensors")
        self.checkpoint_weights = load_file(draft_path)
        unified_state_dict = merge_unified_state_dict(
            target_state_dict,
            {k: v.cpu() for k, v in self.checkpoint_weights.items()},
        )
        # strict=False for the tied target lm_head only (audited by the
        # pipeline model in production; here the S1-proven draft names and
        # the gemma4 adapter cover the rest).
        nn_model.load_state_dict(
            unified_state_dict, weight_alignment=1, strict=False
        )

        with Graph(
            "unified_dspark_gemma4_31b_step_parity",
            input_types=nn_model.input_types(),
        ) as graph:
            values = nn_model._unflatten_graph_inputs(graph.inputs)
            outputs = nn_model(values)
            assert len(outputs) == 3
            recorded = nn_model.recorded
            graph.output(
                *outputs,
                recorded["target_hs_concat"],
                recorded["target_logits"],
                recorded["block_embeds"],
                recorded["block_cache_lengths"],
                recorded["base_logits"],
            )
        self.compiled = session.load(
            graph, weights_registry=nn_model.state_dict()
        )

        self.kv_manager = PagedKVCacheManager(
            params=MultiKVCacheParams.from_params(
                {"target": target_config.kv_params, "draft": draft_kv_params}
            ),
            total_num_pages=16,
            session=session,
            max_batch_size=1,
        )

    def execute_step(
        self,
        context: TextContext,
        draft_tokens: np.ndarray,
    ) -> dict[str, torch.Tensor]:
        """Runs one graph step for the context's active tokens + drafts."""
        device = self.device
        active = np.asarray(context.tokens.active, dtype=np.int64)
        offsets = np.array([0, len(active)], dtype=np.uint32)
        num_steps = draft_tokens.shape[1]
        return_n_logits = np.array([num_steps + 1], dtype=np.int64)

        self.kv_manager.alloc(context)
        total_len = context.tokens.processed_length + len(active)
        # The manager derives the dispatch query width from the context's
        # spec-decoding state, which production populates with the pending
        # drafts; this harness feeds drafts as a plain graph input instead, so
        # pass the merged verify width (and the block forward's slots)
        # explicitly or the decode step resolves a q=1 attention dispatch key
        # for a q=8 forward.
        kv_inputs = self.kv_manager.runtime_inputs(
            [[context]],
            max_cache_length=total_len + 2 * BLOCK_SIZE,
            batch_characteristics=BatchCharacteristics(
                batch_size=1,
                max_prompt_length=max(len(active) + num_steps, BLOCK_SIZE),
                max_cache_valid_length=total_len + 2 * BLOCK_SIZE,
            ),
        )
        # Greedy sampling params: temperature 0 routes the stochastic
        # acceptance sampler onto its target-argmax path.
        outs = self.compiled.execute(
            Buffer.from_numpy(active).to(device),
            Buffer.from_numpy(offsets).to(device),
            Buffer.from_numpy(return_n_logits),
            Buffer.zeros(
                shape=(Signals.NUM_BYTES,), dtype=DType.uint8, device=device
            ),
            *kv_inputs.flatten(),
            Buffer.from_numpy(draft_tokens).to(device),
            Buffer.from_numpy(np.array([1], dtype=np.uint64)).to(device),
            Buffer.from_numpy(np.array([0.0], dtype=np.float32)).to(device),
            Buffer.from_numpy(np.array([1], dtype=np.int64)).to(device),
            Buffer.from_numpy(np.array(1, dtype=np.int64)),
            Buffer.from_numpy(np.array([1.0], dtype=np.float32)).to(device),
            Buffer.from_numpy(np.array(1.0, dtype=np.float32)),
            Buffer.from_numpy(np.zeros(1, dtype=np.bool_)).to(device),
        )
        names = (
            "num_accepted",
            "next_tokens",
            "next_draft_tokens",
            "target_hs_concat",
            "target_logits",
            "block_embeds",
            "block_cache_lengths",
            "base_logits",
        )
        assert len(outs) == len(names)
        return {
            name: from_dlpack(out).to("cpu")
            for name, out in zip(names, outs, strict=False)
        }


@pytest.fixture(scope="module")
def device() -> Device:
    return Accelerator()


@pytest.fixture(scope="module")
def session(device: Device) -> InferenceSession:
    return InferenceSession(devices=[device])


@pytest.fixture(scope="module")
def harness(session: InferenceSession, device: Device) -> _StepHarness:
    if os.environ.get("HF_HUB_OFFLINE", "0") == "1":
        pytest.skip("HF Hub offline mode is enabled")
    return _StepHarness(session, device)


@pytest.fixture(scope="module")
def torch_reference(harness: _StepHarness) -> RefDSparkSpeculatorsBackbone:
    model = RefDSparkSpeculatorsBackbone(harness.ref_config)
    missing, unexpected = model.load_state_dict(
        reference_state_dict(harness.checkpoint_weights), strict=False
    )
    assert not missing and not unexpected
    return model.to(TORCH_DTYPE).to("cuda").eval()


@pytest.fixture(scope="module")
def torch_markov(harness: _StepHarness) -> RefMarkovD2T:
    cfg = harness.ref_config
    markov = RefMarkovD2T(cfg.vocab_size, cfg.draft_vocab_size, cfg.markov_rank)
    markov.load_state_dict(
        {
            "markov_w1.weight": harness.checkpoint_weights[
                "markov_head.markov_w1.weight"
            ],
            "markov_w2.weight": harness.checkpoint_weights[
                "markov_head.markov_w2.weight"
            ],
            "d2t": harness.checkpoint_weights["d2t"],
        }
    )
    markov = markov.to(TORCH_DTYPE).to("cuda").eval()
    assert markov.d2t.dtype == torch.long
    return markov


def _assert_greedy_token(
    logits_row: torch.Tensor, actual: int, label: str
) -> None:
    """Asserts ``actual`` is the argmax of ``logits_row``, excusing
    bf16-tie flips."""
    expected = int(torch.argmax(logits_row))
    if actual == expected:
        return
    top2 = torch.topk(logits_row.float(), 2).values
    gap = float(top2[0] - top2[1])
    assert gap < GREEDY_TIE_GAP_EPS, (
        f"{label}: token {actual} != greedy {expected} (top-2 gap {gap:.4f})"
    )


def _assert_block_inputs(
    step: dict[str, torch.Tensor],
    embed_weight: torch.Tensor,
    expected_block_offset: int,
) -> None:
    """The block must embed [anchor, mask x 7] as RAW rows (bit-exact
    against the checkpoint's embedding table), and its KV write offset p
    must be the post-commit length, placing the mask slots at positions
    p+1..p+7."""
    anchor = int(step["next_tokens"][0])
    block_embeds = step["block_embeds"]
    assert block_embeds.shape[0] == BLOCK_SIZE
    assert torch.equal(
        block_embeds[0], embed_weight[anchor].to(block_embeds.dtype)
    ), "block slot 0 is not the raw anchor embedding row"
    for slot in range(1, BLOCK_SIZE):
        assert torch.equal(
            block_embeds[slot],
            embed_weight[MASK_TOKEN_ID].to(block_embeds.dtype),
        ), f"block slot {slot} is not the raw embed_tokens[{MASK_TOKEN_ID}] row"
    assert int(step["block_cache_lengths"][0]) == expected_block_offset


def _compare_chain(
    ref_base_logits: torch.Tensor,
    torch_markov: RefMarkovD2T,
    anchor: int,
    max_ids: torch.Tensor,
    label: str,
) -> int:
    """Walks the reference markov chain slot by slot against MAX's drafted
    target ids.

    Comparison stops at the first bf16-tie flip (after which the two chains
    legitimately diverge); a mismatch with a wide top-2 gap is a semantic
    failure. Returns the number of slots compared.
    """
    prev = torch.tensor([anchor], device="cuda")
    for k in range(NUM_DRAFTS):
        row = ref_base_logits[k] + torch_markov.compute_step_bias(prev)[0]
        ref_tok = int(torch_markov.map_draft_to_target(torch.argmax(row)))
        actual = int(max_ids[k])
        if actual != ref_tok:
            top2 = torch.topk(row.float(), 2).values
            gap = float(top2[0] - top2[1])
            assert gap < GREEDY_TIE_GAP_EPS, (
                f"{label}: drafted id at slot {k} is {actual}, reference"
                f" chain says {ref_tok} (top-2 gap {gap:.4f})"
            )
            return k
        prev = torch.tensor([ref_tok], device="cuda")
    return NUM_DRAFTS


def _reference_drafts(
    torch_reference: RefDSparkSpeculatorsBackbone,
    embed_weight: torch.Tensor,
    ctx_features: torch.Tensor,
    anchor: int,
) -> torch.Tensor:
    """Reference base draft logits for the 7 mask slots, given the context
    taps and the anchor token (block at absolute positions
    ctx_len..ctx_len+7)."""
    block_ids = torch.full((BLOCK_SIZE,), MASK_TOKEN_ID, dtype=torch.long)
    block_ids[0] = anchor
    block_embeds = embed_weight[block_ids].to(TORCH_DTYPE).to("cuda")
    with torch.no_grad():
        block_hs = torch_reference.forward_backbone(
            block_embedding=block_embeds.unsqueeze(0),
            target_taps=ctx_features.unsqueeze(0),
        )
        logits = torch_reference.compute_logits(block_hs)
    return logits[0, 1:, :]


def test_unified_single_step_parity(
    harness: _StepHarness,
    torch_reference: RefDSparkSpeculatorsBackbone,
    torch_markov: RefMarkovD2T,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(TARGET_REPO_ID)
    prompt_ids = np.asarray(tokenizer(PROMPT).input_ids, dtype=np.int64)
    prompt_len = len(prompt_ids)
    embed_weight = harness.checkpoint_weights["embed_tokens.weight"]
    valid_target_ids = (
        torch.arange(harness.ref_config.draft_vocab_size)
        + torch_markov.d2t.cpu()
    )

    context = create_text_context(prompt_ids, max_length=MAX_SEQ_LEN)
    harness.kv_manager.claim(context)
    try:
        # Step 1 — prefill: empty draft tokens, commit the prompt, draft
        # the first block.
        prefill = harness.execute_step(
            context, np.zeros((1, 0), dtype=np.int64)
        )
        assert int(prefill["num_accepted"][0]) == 0
        assert prefill["target_hs_concat"].shape[0] == prompt_len
        bonus = int(prefill["next_tokens"][0])
        _assert_greedy_token(
            prefill["target_logits"][-1].float(), bonus, "prefill bonus"
        )
        _assert_block_inputs(prefill, embed_weight, prompt_len)

        prefill_drafts = prefill["next_draft_tokens"][0]
        assert prefill_drafts.shape == (NUM_DRAFTS,)
        assert bool(torch.isin(prefill_drafts, valid_target_ids).all()), (
            "prefill drafts left the d2t image"
        )

        prefill_taps = prefill["target_hs_concat"].to("cuda")
        ref_prefill_logits = _reference_drafts(
            torch_reference, embed_weight, prefill_taps, bonus
        )
        prefill_cos = _cos_dist(
            ref_prefill_logits.cpu(), prefill["base_logits"][0]
        )
        print(
            f"[step-parity] prefill base-logits cos_dist={prefill_cos:.3e}",
            flush=True,
        )
        assert prefill_cos < COS_DIST_THRESHOLD
        prefill_compared = _compare_chain(
            ref_prefill_logits,
            torch_markov,
            bonus,
            prefill_drafts,
            "prefill drafts",
        )
        assert prefill_compared >= 1, "prefill chain ended at slot 0 on a tie"

        # Commit the bonus token; the decode step verifies it plus the 7
        # drafts as a q=8 block.
        context.update(bonus)

        # Step 2 — one full speculative step.
        draft_tokens = (
            prefill_drafts.numpy().reshape(1, NUM_DRAFTS).astype(np.int64)
        )
        decode = harness.execute_step(context, draft_tokens)
        target_logits = decode["target_logits"].float()
        assert target_logits.shape[0] == NUM_DRAFTS + 1
        assert decode["target_hs_concat"].shape[0] == NUM_DRAFTS + 1

        # Accepted-count reference: greedy rejection sampling accepts the
        # longest prefix of drafts matching the target argmax.
        ref_accepted = 0
        for i in range(NUM_DRAFTS):
            if int(torch.argmax(target_logits[i])) != int(draft_tokens[0, i]):
                break
            ref_accepted += 1
        num_accepted = int(decode["num_accepted"][0])
        assert 0 <= num_accepted <= NUM_DRAFTS
        if num_accepted != ref_accepted:
            # A bf16 tie at the disagreement boundary flips both the
            # verdict and everything after it; excuse exactly that.
            boundary = min(num_accepted, ref_accepted)
            top2 = torch.topk(target_logits[boundary], 2).values
            gap = float(top2[0] - top2[1])
            assert gap < GREEDY_TIE_GAP_EPS, (
                f"accepted-count {num_accepted} != reference {ref_accepted}"
                f" (boundary top-2 gap {gap:.4f})"
            )
            # The chains diverge from the boundary; stop the step checks
            # that depend on the accepted count.
            print(
                f"[step-parity] tie-excused accepted-count mismatch:"
                f" max={num_accepted} ref={ref_accepted} gap={gap:.4f}"
            )
            return

        # Committed token = target argmax at the first rejected position
        # (or the bonus row when everything was accepted).
        next_token = int(decode["next_tokens"][0])
        _assert_greedy_token(
            target_logits[num_accepted], next_token, "decode next token"
        )

        # Block geometry: p = pre-step cache length (the prompt) + committed
        # tokens this step (the verified bonus + accepted drafts).
        expected_p = prompt_len + num_accepted + 1
        _assert_block_inputs(decode, embed_weight, expected_p)

        # Drafted target ids: the reference consumes MAX's own taps for the
        # committed context — the prefill's prompt taps plus this step's
        # taps at the accepted positions (the [bonus, accepted drafts]
        # prefix of the merged block).
        decode_taps = decode["target_hs_concat"].to("cuda")
        ctx_features = torch.cat(
            [prefill_taps, decode_taps[: num_accepted + 1]], dim=0
        )
        assert ctx_features.shape[0] == expected_p
        ref_decode_logits = _reference_drafts(
            torch_reference, embed_weight, ctx_features, next_token
        )
        decode_cos = _cos_dist(
            ref_decode_logits.cpu(), decode["base_logits"][0]
        )

        decode_drafts = decode["next_draft_tokens"][0]
        assert bool(torch.isin(decode_drafts, valid_target_ids).all()), (
            "decode drafts left the d2t image"
        )
        decode_compared = _compare_chain(
            ref_decode_logits,
            torch_markov,
            next_token,
            decode_drafts,
            "decode drafts",
        )

        print(
            f"[step-parity] prompt_len={prompt_len} bonus={bonus}"
            f" accepted={num_accepted} next={next_token}"
            f" | base-logits cos_dist prefill={prefill_cos:.3e}"
            f" decode={decode_cos:.3e}"
            f" | chain compared prefill={prefill_compared}/{NUM_DRAFTS}"
            f" decode={decode_compared}/{NUM_DRAFTS}"
        )
        assert decode_cos < COS_DIST_THRESHOLD
        assert decode_compared >= 1, "decode chain ended at slot 0 on a tie"
    finally:
        harness.kv_manager.release(context)
