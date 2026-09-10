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
"""Parsing and validating the DFlash2 drafter's structural constants.

Every value read here is one whose mismatch produces wrong drafts and no
error: the block width sets the convolution's period, the tap ids choose which
target layers ``fc`` sees, and the selector geometry sizes a walk that would
otherwise index a differently shaped tensor. The real ``z-lab/Qwen3.8-27B-
DFlash2`` config is committed under ``testdata/`` so this is parsing the
shipped file, not a paraphrase of it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from max.dtype import DType
from max.graph import DeviceRef
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.qwen3_5.model_config import Qwen3_5Config
from max.pipelines.architectures.unified_dflash2_qwen3_5.model_config import (
    UnifiedDflash2Qwen3_5Config,
    dflash2_draft_width,
    parse_dflash2_draft_hf_config,
    resolve_dflash2_num_speculative_tokens,
)
from max.pipelines.lib.config.config import (
    _apply_speculative_target_architecture,
    _construct_from_user_fields,
)
from max.pipelines.speculative.config import (
    SpeculativeConfig,
    SpeculativeMethod,
)

TESTDATA = Path(__file__).parent / "testdata" / "zlab_dflash2_config.json"


def _draft_hf_config(**overrides: Any) -> SimpleNamespace:
    raw = json.loads(TESTDATA.read_text())
    dflash = dict(raw["dflash_config"])
    dflash.update(overrides.pop("dflash_config", {}))
    raw.update(overrides)
    raw["dflash_config"] = dflash
    return SimpleNamespace(**raw)


def test_the_shipped_config_parses_to_the_drafters_geometry() -> None:
    parsed = parse_dflash2_draft_hf_config(_draft_hf_config())
    assert parsed.block_size == 8
    assert parsed.mask_token_id == 248070
    assert parsed.target_layer_ids == [5, 19, 33, 47, 61]
    assert parsed.num_target_layers == 64
    # The four keys v1's parser ignores. Each one sizes a tensor: the conv's
    # taps and group width, and the selector's rank and candidate count.
    assert parsed.conv_kernel_size == 2
    assert parsed.conv_group_size == 16
    assert parsed.selector_rank == 256
    assert parsed.selector_top_k == 16


@pytest.mark.parametrize(
    "missing",
    ["conv_kernel_size", "conv_group_size", "selector_rank", "selector_top_k"],
)
def test_a_missing_selector_or_conv_key_is_refused(missing: str) -> None:
    raw = json.loads(TESTDATA.read_text())
    dflash = {k: v for k, v in raw["dflash_config"].items() if k != missing}
    raw["dflash_config"] = dflash
    with pytest.raises(ValueError, match=missing):
        parse_dflash2_draft_hf_config(SimpleNamespace(**raw))


def test_an_embedding_or_logit_rescale_is_refused() -> None:
    """The reference applies both; this graph applies neither.

    A checkpoint that set one would draft from a differently scaled space and
    produce fluent, low-acceptance drafts rather than an error.
    """
    with pytest.raises(ValueError, match="input_embedding_scale"):
        parse_dflash2_draft_hf_config(
            _draft_hf_config(dflash_config={"input_embedding_scale": 2.0})
        )
    with pytest.raises(ValueError, match="output_multiplier"):
        parse_dflash2_draft_hf_config(
            _draft_hf_config(dflash_config={"output_multiplier": 0.5})
        )


def _pipeline_config_stub(num_speculative_tokens: int | None) -> Any:
    return SimpleNamespace(
        speculative=SimpleNamespace(
            num_speculative_tokens=num_speculative_tokens
        ),
        draft_model=SimpleNamespace(huggingface_config=_draft_hf_config()),
    )


def test_the_draft_width_comes_from_the_checkpoints_block_size() -> None:
    assert (
        resolve_dflash2_num_speculative_tokens(_pipeline_config_stub(None)) == 7
    )
    assert resolve_dflash2_num_speculative_tokens(_pipeline_config_stub(7)) == 7


def test_a_disagreeing_draft_width_is_a_hard_error() -> None:
    """v1 warns and overrides; DFlash2 refuses.

    The export records the width beside the MEF, which the engine checks its
    own ``--num-speculative-tokens`` against at boot, so an override would
    compile a block-8 graph and declare a different width -- and the engine
    feeds the graph at the width it was told.
    """
    with pytest.raises(ValueError, match="block_size=8"):
        resolve_dflash2_num_speculative_tokens(_pipeline_config_stub(3))


def test_an_omitted_width_resolves_the_way_pipeline_config_resolves_it() -> (
    None
):
    """``from_args`` fills an omitted width only via ``checkpoint_draft_width``.

    Nothing else resolves it, so without the registered callback the pipeline
    config keeps ``None`` while the graph is built at seven, and whatever reads
    the config instead of the graph sees zero. Building a real
    ``PipelineConfig`` needs the checkpoints, so this drives the two steps
    ``from_args`` runs: the registry callback, then the override applied to the
    user-set fields.
    """
    unset = SpeculativeConfig(speculative_method="dflash2")
    assert unset.num_speculative_tokens is None

    width = dflash2_draft_width(unset, None, _draft_hf_config())
    assert width == 7
    resolved = _construct_from_user_fields(
        unset, **({"num_speculative_tokens": width} if width else {})
    )
    assert resolved.num_speculative_tokens == 7
    assert resolved.draft_width == 7


def test_the_registry_callback_refuses_a_mismatch_and_a_missing_draft() -> None:
    explicit = SpeculativeConfig(
        speculative_method="dflash2", num_speculative_tokens=3
    )
    with pytest.raises(ValueError, match="block_size=8"):
        dflash2_draft_width(explicit, None, _draft_hf_config())
    with pytest.raises(ValueError, match="requires a draft model"):
        dflash2_draft_width(
            SpeculativeConfig(speculative_method="dflash2"), None, None
        )


def _config(**overrides: Any) -> UnifiedDflash2Qwen3_5Config:
    device = DeviceRef.CPU()
    kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        devices=[device],
        n_kv_heads=1,
        head_dim=16,
        num_layers=1,
        page_size=32,
    )
    target = Qwen3_5Config(
        hidden_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=4,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=128,
        interleaved_rope_weights=True,
        vocab_size=128,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv,
        norm_dtype=DType.bfloat16,
        rms_norm_eps=1e-6,
        attention_multiplier=0.25,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        layer_types=["linear_attention", "full_attention"] * 2,
    )
    draft = Llama3Config(
        hidden_size=overrides.pop("draft_hidden_size", 64),
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        rope_theta=1e7,
        rope_scaling_params=None,
        max_seq_len=128,
        intermediate_size=128,
        interleaved_rope_weights=False,
        vocab_size=overrides.pop("draft_vocab_size", 128),
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=kv,
        rms_norm_eps=1e-6,
        attention_multiplier=0.25,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=[device],
        clip_qkv=None,
        sliding_window=2048,
    )
    fields: dict[str, Any] = {
        "target_layer_ids": [0, 1, 2, 3],
        "layer_types": ["sliding_attention"] * 2,
        "mask_token_id": 127,
        "block_size": 8,
        "conv_kernel_size": 2,
        "conv_group_size": 8,
        "selector_rank": 4,
        "selector_top_k": 3,
    }
    fields.update(overrides)
    return UnifiedDflash2Qwen3_5Config(
        target=target,
        draft=draft,
        draft_kv_params=kv,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash2", num_speculative_tokens=7
        ),
        **fields,
    )


def test_a_valid_pairing_passes_validation() -> None:
    config = _config()
    config.validate_dflash2_fields()
    assert config.num_speculative_tokens == 7


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"target_layer_ids": [0, 1, 1, 2]}, "distinct"),
        ({"target_layer_ids": [0, 1, 2, 99]}, "index the target's layers"),
        ({"num_target_layers": 8}, "trained against a target of"),
        ({"target_layer_ids": []}, "non-empty"),
        ({"draft_hidden_size": 32}, "hidden_size must match"),
        ({"draft_vocab_size": 64}, "vocab must match"),
        ({"mask_token_id": 999}, "mask_token_id"),
        ({"block_size": 1}, "block_size must be at least 2"),
        ({"layer_types": ["sliding_attention"]}, "one entry per draft layer"),
    ],
)
def test_validation_refuses_a_mismatched_pairing(
    overrides: dict[str, Any], match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        _config(**overrides).validate_dflash2_fields()


def test_the_declared_target_depth_is_checked_but_stays_optional() -> None:
    """The tap ids cannot catch a depth mismatch on their own.

    They only check that each id is in range, so a drafter trained against a
    deeper target passes them against any target that holds its deepest tap
    while reading a differently trained residual stream.
    """
    _config(num_target_layers=4).validate_dflash2_fields()
    # Checkpoints that omit the optional field stay usable.
    _config(num_target_layers=None).validate_dflash2_fields()


def test_the_target_is_put_into_tap_mode_and_out_of_subgraph_mode() -> None:
    """Two settings the fused graph cannot work without.

    ``SELECTED_LAYERS`` plus the tap ids is what makes the drafter's context
    exist at all; ``use_subgraphs = False`` is what lets the rollback read the
    verify pass's per-layer state-kernel inputs, which cannot cross a subgraph
    boundary.
    """
    config = _config()
    assert config.target.target_layer_ids == [0, 1, 2, 3]
    assert config.target.return_hidden_states.name == "SELECTED_LAYERS"
    assert config.target.return_logits.name == "VARIABLE"
    assert config.target.use_subgraphs is False
    assert config.target.vision_config is None


def test_multi_device_is_refused_rather_than_silently_single_device() -> None:
    device = DeviceRef.CPU()
    with pytest.raises(ValueError, match="single device"):
        config = _config()
        UnifiedDflash2Qwen3_5Config(
            target=replace(config.target, devices=[device, device]),
            draft=config.draft,
            draft_kv_params=config.draft_kv_params,
            speculative_config=config.speculative_config,
            target_layer_ids=list(config.target_layer_ids),
            layer_types=list(config.layer_types),
            mask_token_id=config.mask_token_id,
            block_size=config.block_size,
            conv_kernel_size=config.conv_kernel_size,
            conv_group_size=config.conv_group_size,
            selector_rank=config.selector_rank,
            selector_top_k=config.selector_top_k,
        )


class TestArchitectureRewrite:
    """``_apply_speculative_target_architecture``'s new DFlash2 arm.

    The rewrite is the only place a base architecture becomes a fused one, and
    it is wrapped in a bare ``except Exception`` that logs at debug, so a
    branch that fails to fire degrades to "no rewrite" and surfaces much later
    as the wrong architecture.
    """

    @staticmethod
    def _rewrite(
        *,
        method: SpeculativeMethod,
        draft_arch: str | None,
        has_mtp: bool = True,
    ) -> str:
        text_config = SimpleNamespace(mtp_num_hidden_layers=1 if has_mtp else 0)
        models: dict[str, Any] = {
            "main": SimpleNamespace(
                huggingface_config=SimpleNamespace(
                    architectures=["Qwen3_5ForConditionalGeneration"],
                    text_config=text_config,
                )
            )
        }
        if draft_arch is not None:
            models["draft"] = SimpleNamespace(
                huggingface_config=SimpleNamespace(architectures=[draft_arch])
            )
        _apply_speculative_target_architecture(
            SpeculativeConfig(speculative_method=method),
            models,
        )
        return models["main"].huggingface_config.architectures[0]

    def test_dflash2_draft_selects_the_fused_dflash2_arch(self) -> None:
        assert (
            self._rewrite(method="dflash2", draft_arch="DFlash2DraftModel")
            == "UnifiedDflash2Qwen3_5ForConditionalGeneration"
        )

    def test_the_mtp_arm_is_unaffected(self) -> None:
        assert (
            self._rewrite(method="mtp", draft_arch=None)
            == "UnifiedMTPQwen3_5ForConditionalGeneration"
        )

    def test_dflash2_without_a_draft_model_does_not_rewrite(self) -> None:
        assert (
            self._rewrite(method="dflash2", draft_arch=None)
            == "Qwen3_5ForConditionalGeneration"
        )

    def test_a_v1_dflash_draft_does_not_select_the_v2_arch(self) -> None:
        """v1 and v2 share a fused-graph shape but not a drafter body."""
        assert (
            self._rewrite(method="dflash", draft_arch="DFlashDraftModel")
            == "Qwen3_5ForConditionalGeneration"
        )


def test_is_dflash_is_the_family_predicate_and_is_dflash2_is_exact() -> None:
    """``is_dflash()`` gates generic machinery (pipeline-class selection, the
    draft-arch rewrite) that wants both drafters; ``is_dflash2()`` gates the
    places where the two differ."""
    v1 = SpeculativeConfig(speculative_method="dflash")
    v2 = SpeculativeConfig(speculative_method="dflash2")
    assert v1.is_dflash() and v2.is_dflash()
    assert not v1.is_dflash2() and v2.is_dflash2()
    assert not SpeculativeConfig(speculative_method="mtp").is_dflash()
