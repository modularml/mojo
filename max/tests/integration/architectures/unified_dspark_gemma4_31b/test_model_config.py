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
"""Speculators-format DSpark config parsing and its guards.

``testdata/redhat_speculator_config.json`` is the real (unmodified)
``config.json`` of ``RedHatAI/gemma-4-31B-it-speculator.dspark`` at revision
``0026c7d1899651ca3c45ede471712f04849723ac``. It has no top-level
``model_type``, so loading it goes through the ``_hf_config.py`` raw-JSON
fallback. Guard-negative tests mutate a copy of the real dict, one field at
a time.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Iterator
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest
from max.driver import (
    CPU,
    Buffer,
    set_virtual_device_api,
    set_virtual_device_count,
    set_virtual_device_target_arch,
)
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import BufferType, DeviceRef, Graph, TensorType
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.speculators_common import (
    DSparkSpeculatorsDraftArchConfig,
    DSparkSpeculatorsDraftConfig,
    construct_draft_kv_params,
)
from max.pipelines.architectures.speculators_common.draft_config import (
    speculators_dspark_width,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model import (
    UnifiedDSparkGemma4_31BInputs,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.model_config import (
    UnifiedDSparkGemma4_31BConfig,
)
from max.pipelines.architectures.unified_dspark_gemma4_31b.unified_dspark_gemma4_31b import (
    UnifiedDSparkGemma4_31B,
    _block_dispatch_metadata,
)
from max.pipelines.kv_cache import KVCacheConfig, cache_dtype_for_encoding
from max.pipelines.lib import MAXModelConfig
from max.pipelines.lib._hf_config import load_huggingface_config
from max.pipelines.lib.config import (
    PipelineConfig,
    SpeculativeConfig,
)
from max.pipelines.weights.hf_utils import HuggingFaceRepo
from max.pipelines.weights.quant import parse_quant_config
from transformers import PretrainedConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).parent / "testdata" / "redhat_speculator_config.json"
)


def _raw() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return cast(dict[str, Any], json.load(f))


def _parse(raw: dict[str, Any]) -> DSparkSpeculatorsDraftConfig:
    return DSparkSpeculatorsDraftConfig.from_huggingface_config(
        PretrainedConfig.from_dict(raw)
    )


def _assert_redhat_fields(draft: DSparkSpeculatorsDraftConfig) -> None:
    assert draft.hidden_size == 5376
    assert draft.intermediate_size == 21504
    assert draft.num_hidden_layers == 5
    assert draft.num_attention_heads == 32
    assert draft.num_key_value_heads == 16
    assert draft.head_dim == 256
    assert draft.rms_norm_eps == 1e-6
    assert draft.vocab_size == 262144
    assert draft.draft_vocab_size == 32000
    assert draft.hidden_activation == "silu"
    assert draft.rope_theta == 10000.0
    # max_position_embeddings is NESTED under transformer_layer_config.
    assert draft.max_seq_len == 262144
    assert draft.sliding_window == 2048
    assert draft.causal is True
    assert draft.block_size == 8
    assert draft.sample_from_anchor is False
    assert draft.mask_token_id == 4
    assert draft.aux_hidden_state_layer_ids == (1, 17, 29, 47, 58)
    assert draft.markov_rank == 256
    assert draft.markov_head_type == "vanilla"
    # block_size counts the anchor slot: 7 drafts per step, not 8.
    assert draft.num_speculative_tokens == 7
    # MAX captures layer OUTPUTS; vLLM aux id j is the INPUT of layer j.
    assert draft.target_layer_ids == (0, 16, 28, 46, 57)
    assert draft.num_context_features == 5


def test_parse_real_config() -> None:
    _assert_redhat_fields(_parse(_raw()))


def test_load_huggingface_config_raw_json_fallback(
    tmp_path: pathlib.Path,
) -> None:
    """End-to-end through the ``_hf_config.py`` raw-JSON fallback path."""
    raw = _raw()
    # The fallback only fires (and re-raising is suppressed) because the
    # speculators config has no top-level model_type.
    assert "model_type" not in raw
    (tmp_path / "config.json").write_text(json.dumps(raw))

    hf_config = load_huggingface_config(HuggingFaceRepo(str(tmp_path)))
    assert isinstance(hf_config, PretrainedConfig)
    _assert_redhat_fields(
        DSparkSpeculatorsDraftConfig.from_huggingface_config(hf_config)
    )


def test_wrong_speculators_model_type_rejected() -> None:
    raw = _raw()
    raw["speculators_model_type"] = "eagle3"
    with pytest.raises(ValueError, match="speculators_model_type"):
        _parse(raw)
    del raw["speculators_model_type"]
    with pytest.raises(ValueError, match="speculators_model_type"):
        _parse(raw)


def test_missing_transformer_layer_config_rejected() -> None:
    raw = _raw()
    del raw["transformer_layer_config"]
    with pytest.raises(ValueError, match="transformer_layer_config"):
        _parse(raw)


def test_block_size_lower_bound() -> None:
    raw = _raw()
    raw["block_size"] = 1
    with pytest.raises(ValueError, match="block_size must be >= 2"):
        _parse(raw)


def test_mask_token_id_bounds() -> None:
    raw = _raw()
    raw["mask_token_id"] = -1
    with pytest.raises(ValueError, match="mask_token_id"):
        _parse(raw)
    raw["mask_token_id"] = 262144
    with pytest.raises(ValueError, match="mask_token_id"):
        _parse(raw)


def test_draft_vocab_size_bounds() -> None:
    raw = _raw()
    raw["draft_vocab_size"] = 0
    with pytest.raises(ValueError, match="draft_vocab_size"):
        _parse(raw)
    raw["draft_vocab_size"] = 262145
    with pytest.raises(ValueError, match="draft_vocab_size"):
        _parse(raw)


def test_aux_layer_ids_guards() -> None:
    raw = _raw()
    raw["aux_hidden_state_layer_ids"] = []
    with pytest.raises(ValueError, match="aux_hidden_state_layer_ids"):
        _parse(raw)
    # 0 is invalid under the vLLM eagle convention (input of layer j, j >= 1).
    raw["aux_hidden_state_layer_ids"] = [0, 17, 29, 47, 58]
    with pytest.raises(ValueError, match="vLLM eagle convention"):
        _parse(raw)
    raw["aux_hidden_state_layer_ids"] = [1, 29, 17, 47, 58]
    with pytest.raises(ValueError, match="strictly"):
        _parse(raw)


def test_layer_types_guards() -> None:
    raw = _raw()
    raw["transformer_layer_config"]["layer_types"] = ["sliding_attention"] * 4
    with pytest.raises(ValueError, match="every layer"):
        _parse(raw)
    raw["transformer_layer_config"]["layer_types"] = ["chunked_attention"] * 5
    with pytest.raises(ValueError, match="unsupported layer_types"):
        _parse(raw)
    raw["transformer_layer_config"]["layer_types"] = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]
    with pytest.raises(ValueError, match="mixed layer_types"):
        _parse(raw)


def test_causality_derivation() -> None:
    # All full_attention (the makora/GLM convention) parses as non-causal
    # and does not require a sliding window.
    raw = _raw()
    raw["transformer_layer_config"]["layer_types"] = ["full_attention"] * 5
    raw["transformer_layer_config"]["sliding_window"] = None
    draft = _parse(raw)
    assert draft.causal is False

    # An explicit top-level `causal` field overrides the layer_types rule
    # (the vLLM _dflash_layer_causal precedence).
    raw = _raw()
    raw["causal"] = False
    assert _parse(raw).causal is False


def test_sliding_layers_require_window() -> None:
    raw = _raw()
    raw["transformer_layer_config"]["sliding_window"] = 0
    with pytest.raises(ValueError, match="sliding_window > 0"):
        _parse(raw)


def test_sliding_window_non_causal_rejected() -> None:
    raw = _raw()
    raw["sliding_window_non_causal"] = True
    with pytest.raises(ValueError, match="sliding_window_non_causal"):
        _parse(raw)


def test_proposal_speculative_tokens_cross_check() -> None:
    raw = _raw()
    raw["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 6
    with pytest.raises(ValueError, match="speculative_tokens=6"):
        _parse(raw)


def test_sample_from_anchor_shifts_expected_tokens() -> None:
    # With sample_from_anchor the anchor slot also predicts: block_size 8
    # means 8 drafts (the GLM dspark convention), so the RedHat proposal
    # value of 7 must now be rejected...
    raw = _raw()
    raw["sample_from_anchor"] = True
    with pytest.raises(ValueError, match="sample_from_anchor=True"):
        _parse(raw)
    # ...and 8 accepted.
    raw["speculators_config"]["proposal_methods"][0]["speculative_tokens"] = 8
    draft = _parse(raw)
    assert draft.sample_from_anchor is True
    assert draft.num_speculative_tokens == 8


def test_markov_head_guards() -> None:
    raw = _raw()
    raw["markov_head_type"] = "dense"
    with pytest.raises(ValueError, match="markov_head_type"):
        _parse(raw)
    raw = _raw()
    raw["markov_rank"] = 0
    with pytest.raises(ValueError, match="markov_rank"):
        _parse(raw)


def test_rope_guards() -> None:
    raw = _raw()
    raw["transformer_layer_config"]["rope_parameters"]["rope_type"] = "yarn"
    with pytest.raises(ValueError, match="rope_type"):
        _parse(raw)
    raw = _raw()
    del raw["transformer_layer_config"]["rope_parameters"]["rope_theta"]
    with pytest.raises(ValueError, match="rope_theta"):
        _parse(raw)


def test_missing_nested_max_position_embeddings_rejected() -> None:
    raw = _raw()
    del raw["transformer_layer_config"]["max_position_embeddings"]
    with pytest.raises(ValueError, match="max_position_embeddings"):
        _parse(raw)


def test_draft_arch_config_reads_nested_max_position_embeddings() -> None:
    assert (
        DSparkSpeculatorsDraftArchConfig.calculate_max_seq_len(
            PretrainedConfig.from_dict(_raw()),
            cast(MAXModelConfig, None),
        )
        == 262144
    )

    raw = _raw()
    del raw["transformer_layer_config"]["max_position_embeddings"]
    with pytest.raises(ValueError, match="max_position_embeddings"):
        DSparkSpeculatorsDraftArchConfig.calculate_max_seq_len(
            PretrainedConfig.from_dict(raw),
            cast(MAXModelConfig, None),
        )


def _make_target(
    *,
    num_hidden_layers: int = 60,
    hidden_size: int = 5376,
    vocab_size: int = 262144,
    n_devices: int = 1,
) -> SimpleNamespace:
    """Stand-in exposing only the target attributes the config touches."""
    return SimpleNamespace(
        text_config=SimpleNamespace(
            num_hidden_layers=num_hidden_layers,
            hidden_size=hidden_size,
            vocab_size=vocab_size,
            return_logits=None,
            return_hidden_states=None,
            target_layer_ids=[],
        ),
        devices=[DeviceRef.CPU()] * n_devices,
    )


def _make_unified(
    draft: DSparkSpeculatorsDraftConfig,
    *,
    target: SimpleNamespace | None = None,
    num_speculative_tokens: int | None = 7,
) -> UnifiedDSparkGemma4_31BConfig:
    target = target if target is not None else _make_target()
    # None means unset: a config that never mentions num_speculative_tokens.
    speculative_config = SpeculativeConfig(
        speculative_method="dflash",
        num_speculative_tokens=num_speculative_tokens,
    )
    return UnifiedDSparkGemma4_31BConfig(
        target=cast(Gemma4ForConditionalGenerationConfig, target),
        draft=draft,
        draft_kv_params=cast(KVCacheParams, None),
        speculative_config=speculative_config,
        target_layer_ids=list(draft.target_layer_ids),
        mask_token_id=draft.mask_token_id,
        block_size=draft.block_size,
    )


def test_unified_config_wires_target_capture() -> None:
    config = _make_unified(_parse(_raw()))
    text_config = config.target.text_config
    assert text_config.return_logits == ReturnLogits.VARIABLE
    assert (
        text_config.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
    )
    assert text_config.target_layer_ids == [0, 16, 28, 46, 57]
    config.validate_dspark_fields()


def test_unified_config_single_device_only() -> None:
    with pytest.raises(ValueError, match="single device"):
        _make_unified(_parse(_raw()), target=_make_target(n_devices=2))


def test_validate_rejects_aux_ids_beyond_target_depth() -> None:
    config = _make_unified(
        _parse(_raw()), target=_make_target(num_hidden_layers=40)
    )
    with pytest.raises(ValueError, match="40-layer target"):
        config.validate_dspark_fields()


def test_validate_rejects_hidden_size_mismatch() -> None:
    config = _make_unified(
        _parse(_raw()), target=_make_target(hidden_size=3840)
    )
    with pytest.raises(ValueError, match="hidden_size"):
        config.validate_dspark_fields()


def test_validate_rejects_vocab_mismatch() -> None:
    config = _make_unified(
        _parse(_raw()), target=_make_target(vocab_size=32000)
    )
    with pytest.raises(ValueError, match="vocab"):
        config.validate_dspark_fields()


def test_construction_honors_user_num_speculative_tokens() -> None:
    # K=4 below the trained width (8 - 1 = 7) is honored: the draft block
    # is causal, so truncating trailing mask slots is prefix-stable. KV
    # headroom follows as the effective anchor+drafts width.
    config = _make_unified(_parse(_raw()), num_speculative_tokens=4)
    assert config.resolved_num_speculative_tokens == 4
    assert config.effective_block_size == 5


def test_construction_warns_num_speculative_tokens_beyond_trained(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # K=9 above the trained width (8 - 1 = 7) is honored too: the block is
    # width-generic and the extra positions run as extrapolation, so a
    # warning names the trained geometry. KV headroom still follows the
    # effective anchor+drafts width.
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        config = _make_unified(_parse(_raw()), num_speculative_tokens=9)
    assert config.resolved_num_speculative_tokens == 9
    assert config.effective_block_size == 10
    assert any(
        "trained at block_size=8" in record.message for record in caplog.records
    )


def test_construction_rejects_non_positive_num_speculative_tokens() -> None:
    with pytest.raises(ValueError, match="num_speculative_tokens=0"):
        _make_unified(_parse(_raw()), num_speculative_tokens=0)


def test_construction_defaults_num_speculative_tokens_to_trained() -> None:
    # Unset keeps the drafter's trained width (block_size 8 -> 7 drafts).
    config = _make_unified(_parse(_raw()), num_speculative_tokens=None)
    assert config.resolved_num_speculative_tokens == 7
    assert config.effective_block_size == 8
    assert config.speculative_config.num_speculative_tokens is None


def _kv_pipeline_stand_in(kv_cache: KVCacheConfig) -> PipelineConfig:
    """Stand-in exposing only what the KV-param constructors touch."""
    return cast(
        PipelineConfig,
        SimpleNamespace(
            model=SimpleNamespace(kv_cache=kv_cache, data_parallel_degree=1),
            speculative=SpeculativeConfig(
                speculative_method="dflash", num_speculative_tokens=7
            ),
        ),
    )


def _target_hf_geometry() -> SimpleNamespace:
    """Tiny gemma4-shaped attention geometry (one leaf per attention type)."""
    return SimpleNamespace(
        text_config=SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            num_key_value_heads=8,
            num_global_key_value_heads=4,
            head_dim=128,
            global_head_dim=128,
            # Matches the real gemma-4-31B-it config's sliding window.
            sliding_window=1024,
        )
    )


@pytest.mark.parametrize("kv_cache_format", [None, "float8_e4m3fn"])
def test_kv_cache_format_reaches_target_leaves_draft_stays_bfloat16(
    kv_cache_format: str | None,
) -> None:
    """``kv_cache_format`` governs the target tree; the draft leaf is pinned.

    Mirrors ``gemma4_31b_dspark_nvfp4.yaml``'s ``float8_e4m3fn`` override
    through the same helpers ``model.py``'s KV re-derivation uses. The draft
    leaf stays bfloat16 whatever the target's cache format: the draft block
    runs the generic ``AttentionWithRope`` path, which feeds a bfloat16 Q
    into ``flash_attention_ragged`` (no fp8-Q accommodation like the gemma4
    target attention's ``q_out_dtype``), and the drafter was trained against
    bfloat16. The mixed-dtype ``{target: fp8, draft: bf16}`` tree must pass
    ``MultiKVCacheParams`` validation — dtype is per-leaf, page geometry
    must stay uniform.
    """
    kv_cache_config = KVCacheConfig(
        kv_cache_format=kv_cache_format, kv_cache_page_size=128
    )
    pipeline_config = _kv_pipeline_stand_in(kv_cache_config)
    devices = [DeviceRef.CPU()]

    # The recipe's resolution: NVFP4-encoded target, optional format override.
    cache_dtype = cache_dtype_for_encoding(
        "float4_e2m1fnx2", kv_cache_config.kv_cache_format
    )
    expected_target_dtype = (
        DType.float8_e4m3fn if kv_cache_format else DType.bfloat16
    )
    assert cache_dtype == expected_target_dtype

    target_kv = Gemma4ForConditionalGenerationConfig.construct_kv_params(
        cast(Any, _target_hf_geometry()),
        pipeline_config,
        devices,
        kv_cache_config,
        cache_dtype,
    )
    draft_kv = construct_draft_kv_params(
        pipeline_config, _parse(_raw()), devices, num_draft_tokens=8
    )

    corrected_children: dict[str, KVCacheParams] = {}
    for leaf_name in ("sliding_attention", "full_attention"):
        leaf = target_kv.children[leaf_name]
        assert isinstance(leaf, KVCacheParams)
        assert leaf.dtype == expected_target_dtype
        assert leaf.page_size == 128
        # model.py's re-derivation aligns every leaf on the effective block
        # width before assembling the tree (the Multi uniformity rule).
        corrected_children[leaf_name] = replace(leaf, num_draft_tokens=8)
    assert draft_kv.dtype == DType.bfloat16
    assert draft_kv.page_size == 128

    tree = MultiKVCacheParams.from_params(
        {
            "target": MultiKVCacheParams.from_params(corrected_children),
            "draft": draft_kv,
        }
    )
    draft_leaf = tree.children["draft"]
    assert isinstance(draft_leaf, KVCacheParams)
    assert draft_leaf.dtype == DType.bfloat16


def _nvfp4_quant_config(num_hidden_layers: int = 2) -> QuantConfig:
    """The parsed modelopt/NVFP4 config of an NVFP4 Gemma4 target.

    Mirrors ``nvidia/Gemma-4-31B-IT-NVFP4``'s ``quantization_config`` (every
    ``self_attn`` subtree plus ``lm_head`` ignore-listed, ``Linear`` targets
    otherwise) at two layers, and passes the same name prefixes
    ``Gemma4ForConditionalGenerationConfig.finalize`` does, so the per-layer
    classification comes from the production parser rather than a hand-built
    :class:`QuantConfig`.
    """
    hf_config = PretrainedConfig.from_dict(
        {
            "quantization_config": {
                "quant_method": "modelopt",
                "quant_algo": "NVFP4",
                "config_groups": {
                    "group_0": {
                        "input_activations": {
                            "dynamic": False,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16,
                        },
                        "weights": {
                            "dynamic": False,
                            "num_bits": 4,
                            "type": "float",
                            "group_size": 16,
                        },
                        "targets": ["Linear"],
                    }
                },
                "ignore": ["lm_head", "model.embed_vision*"]
                + [
                    f"model.language_model.layers.{i}.self_attn*"
                    for i in range(num_hidden_layers)
                ],
            },
        }
    )
    # Gemma 4 nests the language tower's depth under text_config, which is
    # what the parser reads to enumerate layers.
    hf_config.text_config = PretrainedConfig.from_dict(
        {"num_hidden_layers": num_hidden_layers}
    )
    quant_config = parse_quant_config(
        hf_config,
        {},
        DType.uint8,
        state_dict_name_prefix="model.language_model.",
        ignored_modules_prefix="model.language_model.",
    )
    assert quant_config is not None
    assert quant_config.is_nvfp4
    assert quant_config.mlp_quantized_layers == set(range(num_hidden_layers))
    assert quant_config.attn_quantized_layers == set()
    return quant_config


def _make_real_target(
    devices: list[DeviceRef],
    quant_config: QuantConfig | None = None,
) -> Gemma4ForConditionalGenerationConfig:
    """Real (non-stand-in) target config at tiny dims.

    The graph-signature tests below construct the actual module, whose ctor
    builds ``Gemma4TextModel`` — the ``SimpleNamespace`` stand-in used by the
    config tests is not enough. Weight VALUES never materialize (module
    construction creates metadata only), so tiny dims keep this a CPU test.
    """
    layer_types = ["sliding_attention", "full_attention"]
    sliding_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=32,
        num_layers=1,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=32,
        num_layers=1,
        devices=devices,
        page_size=128,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )
    text_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=32,
        num_layers=2,
        devices=devices,
        page_size=128,
    )
    text_config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        hidden_activation="gelu_tanh",
        max_position_embeddings=512,
        max_seq_len=256,
        rms_norm_eps=1e-6,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=64,
        final_logit_softcapping=30.0,
        attn_logit_softcapping=None,
        rope_local_base_freq=10000.0,
        sliding_window_pattern=-1,
        dtype=DType.bfloat16,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=1,
        global_head_dim=32,
        attention_k_eq_v=True,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=0.25
        ),
        global_rope_theta=1_000_000.0,
        sliding_window_rope_theta=10000.0,
        layer_types=layer_types,
        quant_config=quant_config,
    )
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        # An NVFP4 target runs with the packed-uint8 weight dtype and keeps
        # bfloat16 for everything the checkpoint leaves unquantized (norms,
        # attention, the drafter's whole tower).
        dtype=DType.uint8 if quant_config else DType.bfloat16,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=None,
        image_token_index=200,
        tie_word_embeddings=True,
    )


def _make_unified_real(
    draft: DSparkSpeculatorsDraftConfig,
    quant_config: QuantConfig | None = None,
) -> UnifiedDSparkGemma4_31BConfig:
    devices = [DeviceRef.GPU()]
    draft_kv_params = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=draft.num_key_value_heads,
        head_dim=draft.head_dim,
        num_layers=draft.num_hidden_layers,
        devices=devices,
        page_size=128,
    )
    return UnifiedDSparkGemma4_31BConfig(
        target=_make_real_target(devices, quant_config),
        draft=draft,
        draft_kv_params=draft_kv_params,
        speculative_config=SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=7
        ),
        target_layer_ids=[0, 1],
        mask_token_id=draft.mask_token_id,
        block_size=draft.block_size,
    )


def _signature(
    types: tuple[TensorType | BufferType, ...],
) -> list[tuple[str, DType, str]]:
    return [(type(t).__name__, t.dtype, str(t.shape)) for t in types]


@pytest.fixture
def virtual_gpu() -> Iterator[None]:
    """Module construction eagerly creates driver ``Accelerator`` handles
    (``Allreduce`` in ``VocabParallelEmbedding.__init__``), which fails on
    the CPU-only presubmit workers. Virtual-device mode satisfies device
    creation without hardware; these tests only read graph metadata.
    """
    set_virtual_device_api("cuda")
    set_virtual_device_target_arch("sm_80")
    set_virtual_device_count(1)
    try:
        yield
    finally:
        set_virtual_device_count(0)


def test_graph_signature_binds_thinking_and_structured_output(
    virtual_gpu: None,
) -> None:
    """Structured output off: the tail ends at ``in_thinking_phase``; on:
    exactly the packed bitmask triple is appended, prefix unchanged.

    Guards the lockstep between the module's ``SpecDecodeInputTypeSpec``
    flags and ``_spec_decode_tail_buffers`` in ``model.py`` — flipping one
    side alone is an input-arity failure at execute, or worse a silently
    dead feature (the historical bug class in this exact area).
    """
    base_types = UnifiedDSparkGemma4_31B(
        _make_unified_real(_parse(_raw()))
    ).input_types()
    so_types = UnifiedDSparkGemma4_31B(
        _make_unified_real(_parse(_raw())), enable_structured_output=True
    ).input_types()

    thinking = base_types[-1]
    assert isinstance(thinking, TensorType)
    assert thinking.dtype == DType.bool

    assert _signature(so_types[: len(base_types)]) == _signature(base_types)
    assert len(so_types) == len(base_types) + 3
    pinned, wait, scratch = so_types[-3:]
    assert isinstance(pinned, TensorType)
    assert pinned.dtype == DType.int32
    assert isinstance(wait, BufferType)
    assert wait.dtype == DType.int64
    assert isinstance(scratch, BufferType)
    assert scratch.dtype == DType.int32


def _make_placeholder_inputs(
    *, structured_output: bool
) -> UnifiedDSparkGemma4_31BInputs:
    b = Buffer.from_numpy(np.zeros(1, dtype=np.int64))
    if structured_output:
        return UnifiedDSparkGemma4_31BInputs(
            tokens=b,
            input_row_offsets=b,
            return_n_logits=b,
            signal_buffers=[b],
            kv_cache_inputs=None,
            draft_tokens=b,
            seed=b,
            temperature=b,
            top_k=b,
            max_k=b,
            top_p=b,
            min_top_p=b,
            in_thinking_phase=b,
            structured_output=True,
            pinned_bitmask=b,
            wait_payload=b,
            device_bitmask_scratch=b,
        )
    return UnifiedDSparkGemma4_31BInputs(
        tokens=b,
        input_row_offsets=b,
        return_n_logits=b,
        signal_buffers=[b],
        kv_cache_inputs=None,
        draft_tokens=b,
        seed=b,
        temperature=b,
        top_k=b,
        max_k=b,
        top_p=b,
        min_top_p=b,
        in_thinking_phase=b,
        structured_output=False,
    )


def test_inputs_buffer_tail_matches_graph_signature(
    virtual_gpu: None,
) -> None:
    """The host-side buffer packing (``model.py`` tail flags) must track the
    module's graph signature in both structured-output states.

    The graph-signature test above covers only the module side; a one-sided
    flip of the ``_spec_decode_tail_buffers`` flags would keep it green and
    fail at execute with an input-arity error. ``kv_cache_inputs=None``
    drops exactly the KV-tree inputs from the packed tuple, so parity is
    signature length minus the flattened KV inputs.
    """
    for structured_output in (False, True):
        module = UnifiedDSparkGemma4_31B(
            _make_unified_real(_parse(_raw())),
            enable_structured_output=structured_output,
        )
        n_kv = len(module._unified_kv_params().flattened_kv_inputs())
        inputs = _make_placeholder_inputs(structured_output=structured_output)
        assert len(inputs.buffers) == len(module.input_types()) - n_kv


def test_nvfp4_target_quantizes_mlp_and_leaves_draft_bfloat16(
    virtual_gpu: None,
) -> None:
    """An NVFP4 target quantizes exactly the checkpoint's non-ignored Linears.

    The weight names and dtypes here are the load-time contract: the pipeline
    model audits the module's expected names against the merged state dict, so
    a scale the graph declares but the checkpoint lacks (or vice versa) fails
    the load. The drafter must stay bfloat16 whatever the target runs as — it
    is loaded from its own bfloat16 checkpoint and its ``fc`` consumes the
    target's (unquantized) hidden-state taps.
    """
    weights = UnifiedDSparkGemma4_31B(
        _make_unified_real(_parse(_raw()), _nvfp4_quant_config())
    ).raw_state_dict()

    for layer in range(2):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            prefix = f"target.layers.{layer}.mlp.{proj}"
            assert weights[f"{prefix}.weight"].dtype == DType.uint8
            assert (
                weights[f"{prefix}.weight_scale"].dtype == DType.float8_e4m3fn
            )
            assert weights[f"{prefix}.weight_scale_2"].dtype == DType.float32
            assert weights[f"{prefix}.input_scale"].dtype == DType.float32

    # attention is ignore-listed in the first-party NVFP4 checkpoints: bf16
    # weights and no scale tensors at all.
    for name in ("q_proj", "k_proj", "o_proj"):
        matches = [n for n in weights if n.endswith(f"self_attn.{name}.weight")]
        assert matches
        for n in matches:
            assert weights[n].dtype == DType.bfloat16
    assert not [n for n in weights if "self_attn" in n and "_scale" in n]

    draft_names = [n for n in weights if n.startswith("draft.")]
    assert {
        weights[n].dtype for n in draft_names if weights[n].dtype.is_float()
    } == {DType.bfloat16}
    # d2t is an integer draft-to-target vocab offset table, not a tensor any
    # weight encoding applies to.
    assert [n for n in draft_names if not weights[n].dtype.is_float()] == [
        "draft.d2t"
    ]


def test_nvfp4_target_keeps_graph_signature(virtual_gpu: None) -> None:
    """Quantizing the target must not perturb the spec-decode input contract.

    The graph inputs are activations, KV pages and sampling scalars — none of
    which the weight encoding touches — so a signature change here would mean
    the NVFP4 path diverged from the bfloat16 one somewhere it should not.
    """
    bf16_types = UnifiedDSparkGemma4_31B(
        _make_unified_real(_parse(_raw()))
    ).input_types()
    nvfp4_types = UnifiedDSparkGemma4_31B(
        _make_unified_real(_parse(_raw()), _nvfp4_quant_config())
    ).input_types()
    assert _signature(nvfp4_types) == _signature(bf16_types)


def test_relaxed_acceptance_threads_into_sampler(virtual_gpu: None) -> None:
    """``use_relaxed_acceptance_for_thinking`` wires the config's relaxed
    params into the acceptance sampler; off leaves the strict rule."""
    strict = UnifiedDSparkGemma4_31B(_make_unified_real(_parse(_raw())))
    assert strict.acceptance_sampler._relaxed_topk is None
    assert strict.acceptance_sampler._relaxed_delta is None

    config = _make_unified_real(_parse(_raw()))
    config.speculative_config = SpeculativeConfig(
        speculative_method="dflash",
        num_speculative_tokens=7,
        use_relaxed_acceptance_for_thinking=True,
        relaxed_topk=5,
        relaxed_delta=0.1,
    )
    relaxed = UnifiedDSparkGemma4_31B(config)
    assert relaxed.acceptance_sampler._relaxed_topk == 5
    assert relaxed.acceptance_sampler._relaxed_delta == 0.1


def _tiny_draft() -> DSparkSpeculatorsDraftConfig:
    """Draft config at dims consistent with ``_make_real_target`` (hidden 64,
    vocab 256, 2 target layers), so the unified graph can actually STAGE —
    the real RedHat draft (hidden 5376) only supports metadata-level tests."""
    return DSparkSpeculatorsDraftConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        rms_norm_eps=1e-6,
        vocab_size=256,
        draft_vocab_size=128,
        hidden_activation="silu",
        rope_theta=10000.0,
        max_seq_len=512,
        sliding_window=64,
        causal=True,
        block_size=8,
        sample_from_anchor=False,
        mask_token_id=4,
        aux_hidden_state_layer_ids=(1, 2),
        markov_rank=16,
        markov_head_type="vanilla",
    )


def test_block_forward_rebuilds_attention_dispatch_metadata(
    virtual_gpu: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The draft block forward must not inherit the leaf's verify-width
    ``attention_dispatch_metadata``.

    The leaf carries the TARGET key, whose ``q_max_seq_len`` is the batch's
    max prompt length — equal to the block width only on decode batches. On
    prefill (CE) batches the inherited oversized query bound drives the
    block's layer-0 flash attention to produce NaN, so the block forward has
    to run on metadata rebuilt in-graph at the block width (content pinned
    by ``test_block_dispatch_metadata_rebuild_content`` below).
    """
    nn_model = UnifiedDSparkGemma4_31B(_make_unified_real(_tiny_draft()))
    for name, weight in nn_model.raw_state_dict().items():
        weight.name = name

    captured: list[PagedCacheValues] = []
    real_forward_block = nn_model.draft.forward_block

    def spy_forward_block(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["kv_collection"])
        return real_forward_block(*args, **kwargs)

    monkeypatch.setattr(nn_model.draft, "forward_block", spy_forward_block)

    with Graph(
        "unified_dspark_gemma4_31b_block_meta_test",
        input_types=nn_model.input_types(),
    ) as graph:
        values = nn_model._unflatten_graph_inputs(graph.inputs)
        graph.output(*nn_model(values))

    assert len(captured) == 1
    block_kv = captured[0]
    block_meta = block_kv.attention_dispatch_metadata
    leaf_meta = values.draft_kv_collection.attention_dispatch_metadata
    assert block_meta is not None
    assert leaf_meta is not None
    assert block_meta._mlir_value != leaf_meta._mlir_value
    # Rebuilt in-graph, not any leaf's metadata input passed through.
    assert all(block_meta._mlir_value != v._mlir_value for v in graph.inputs)
    # The paired query-width bound must be block-width too.
    assert (
        block_kv.max_prompt_length._mlir_value
        != values.draft_kv_collection.max_prompt_length._mlir_value
    )


def test_block_dispatch_metadata_rebuild_content() -> None:
    """The rebuilt 4-int buffer keeps batch and cache bound, sets
    ``q_max_seq_len`` to the block width, and zeroes ``num_partitions`` so
    the decode kernel recomputes the split-K count for the draft's own head
    geometry."""
    with Graph(
        "block_dispatch_metadata",
        input_types=(TensorType(DType.int64, [4], DeviceRef.CPU()),),
    ) as graph:
        (meta,) = graph.inputs
        graph.output(_block_dispatch_metadata(meta.tensor, 8))

    session = InferenceSession(devices=[CPU()])
    compiled = session.load(graph)
    (out,) = compiled.execute(
        Buffer.from_numpy(np.array([3, 129, 5, 4321], dtype=np.int64))
    )
    assert isinstance(out, Buffer)
    np.testing.assert_array_equal(out.to_numpy(), [3, 8, 0, 4321])


def test_width_honors_explicit() -> None:
    draft_hf = _raw()
    unset = SpeculativeConfig(speculative_method="dflash")
    assert speculators_dspark_width(unset, None, draft_hf) == 7
    explicit = SpeculativeConfig(
        speculative_method="dflash", num_speculative_tokens=4
    )
    assert speculators_dspark_width(explicit, None, draft_hf) == 4
