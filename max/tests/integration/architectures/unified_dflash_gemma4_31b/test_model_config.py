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
"""DFlash Gemma4 config parsing, draft geometry, and graph signature.

``testdata/zlab_dflash_config.json`` is the real (unmodified) ``config.json``
of ``z-lab/gemma-4-31B-it-DFlash`` at revision
``eabd648301ce28583cc14757912e5e0f84e152e1``.
"""

from __future__ import annotations

import json
import logging
import pathlib
from collections.abc import Iterator
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
from max.nn.attention import AttentionWithRope
from max.nn.attention.mask_config import MHAMaskVariant
from max.nn.kv_cache import (
    KVCacheParams,
    MHAKVCacheParams,
    MultiKVCacheParams,
    PagedCacheValues,
)
from max.nn.quant_config import QuantConfig
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.dflash_llama3 import DFlashLlama3
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
)
from max.pipelines.architectures.llama3.model_config import Llama3Config
from max.pipelines.architectures.unified_dflash_gemma4_31b.model import (
    UnifiedDflashGemma4_31BInputs,
    UnifiedDflashGemma4_31BModel,
)
from max.pipelines.architectures.unified_dflash_gemma4_31b.model_config import (
    UnifiedDflashGemma4_31BConfig,
    parse_dflash_draft_hf_config,
    resolve_dflash_num_speculative_tokens,
)
from max.pipelines.architectures.unified_dflash_gemma4_31b.unified_dflash_gemma4_31b import (
    UnifiedDflashGemma4_31B,
    _block_dispatch_metadata,
)
from max.pipelines.lib import (
    KVCacheConfig,
    MAXModelConfig,
    PipelineConfig,
)
from max.pipelines.lib.config import SpeculativeConfig
from max.pipelines.lib.model_manifest import ModelManifest
from max.pipelines.weights.quant import parse_quant_config
from transformers import PretrainedConfig

_CONFIG_PATH = (
    pathlib.Path(__file__).parent / "testdata" / "zlab_dflash_config.json"
)

_DRAFT_HIDDEN = 5376
_DRAFT_LAYERS = 5
_TARGET_LAYERS = 60
_VOCAB = 262144


def _raw() -> dict[str, Any]:
    with open(_CONFIG_PATH) as f:
        return cast(dict[str, Any], json.load(f))


def test_parses_the_real_zlab_config() -> None:
    parsed = parse_dflash_draft_hf_config(PretrainedConfig.from_dict(_raw()))
    assert parsed.mask_token_id == 4
    assert parsed.block_size == 16
    assert parsed.num_target_layers == _TARGET_LAYERS
    # Six taps into five draft layers: the tap count is fc's in-features
    # (5376 x 32256), NOT the draft's layer count. Passed straight through:
    # MAX captures layer OUTPUTS and the native DFlash config already indexes
    # outputs (vLLM shifts by one only for the speculators aux-id form).
    assert parsed.target_layer_ids == [1, 12, 23, 35, 46, 57]


def _make_pipeline_config(
    num_speculative_tokens: int | None,
) -> PipelineConfig:
    """Builds a minimal two-model PipelineConfig; the draft carries the real
    z-lab checkpoint config, so the trained width resolves to 15."""
    model_config = MAXModelConfig.model_construct(model_path="fake/target")
    draft_config = MAXModelConfig.model_construct(model_path="fake/draft")
    draft_config._huggingface_config = PretrainedConfig.from_dict(_raw())
    return PipelineConfig.model_construct(
        models=ModelManifest({"main": model_config, "draft": draft_config}),
        speculative=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=num_speculative_tokens,
        ),
    )


def test_trained_width_resolves_to_fifteen_when_unset() -> None:
    """block_size 16 minus the anchor slot; the caller's config is never
    written back to."""
    pipeline_config = _make_pipeline_config(None)
    assert resolve_dflash_num_speculative_tokens(pipeline_config) == 15
    assert pipeline_config.speculative is not None
    assert pipeline_config.speculative.num_speculative_tokens is None


def test_mismatched_width_is_overridden_to_fifteen(
    caplog: pytest.LogCaptureFixture,
) -> None:
    pipeline_config = _make_pipeline_config(4)
    with caplog.at_level(logging.WARNING, logger="max.pipelines"):
        resolved = resolve_dflash_num_speculative_tokens(pipeline_config)
    assert resolved == 15
    assert "overridden from 4 to 15" in caplog.text


def test_width_bakes_into_every_kv_leaf() -> None:
    """Both Gemma4 leaves bake the width the config was built with."""
    pipeline_config = _make_pipeline_config(15)
    huggingface_config = SimpleNamespace(
        text_config=SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            num_key_value_heads=2,
            head_dim=16,
            global_head_dim=32,
            # Matches the real gemma-4-31B-it config's sliding window.
            sliding_window=1024,
        )
    )
    kv_params = UnifiedDflashGemma4_31BModel.get_kv_params(
        huggingface_config,
        pipeline_config,
        [DeviceRef.CPU()],
        KVCacheConfig(),
        DType.bfloat16,
    )
    assert set(kv_params.children) == {"sliding_attention", "full_attention"}
    for leaf in kv_params.children.values():
        assert isinstance(leaf, KVCacheParams)
        assert leaf.num_draft_tokens == 15


def _make_target_stub(
    *,
    num_hidden_layers: int = _TARGET_LAYERS,
    hidden_size: int = _DRAFT_HIDDEN,
    vocab_size: int = _VOCAB,
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


def _make_draft_stub(
    *,
    hidden_size: int = _DRAFT_HIDDEN,
    vocab_size: int = _VOCAB,
    num_hidden_layers: int = _DRAFT_LAYERS,
) -> SimpleNamespace:
    return SimpleNamespace(
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        num_hidden_layers=num_hidden_layers,
        return_hidden_states=None,
    )


def _make_unified(
    *,
    target: SimpleNamespace | None = None,
    draft: SimpleNamespace | None = None,
    num_speculative_tokens: int | None = 15,
) -> UnifiedDflashGemma4_31BConfig:
    parsed = parse_dflash_draft_hf_config(PretrainedConfig.from_dict(_raw()))
    return UnifiedDflashGemma4_31BConfig(
        target=cast(
            Gemma4ForConditionalGenerationConfig,
            target if target is not None else _make_target_stub(),
        ),
        draft=cast(
            Llama3Config, draft if draft is not None else _make_draft_stub()
        ),
        draft_kv_params=cast(Any, None),
        speculative_config=SpeculativeConfig(
            speculative_method="dflash",
            num_speculative_tokens=num_speculative_tokens,
        ),
        target_layer_ids=list(parsed.target_layer_ids),
        layer_types=list(_raw()["layer_types"]),
        mask_token_id=parsed.mask_token_id,
        block_size=parsed.block_size or 0,
    )


def test_unified_config_wires_target_capture() -> None:
    config = _make_unified()
    text_config = config.target.text_config
    assert text_config.return_logits == ReturnLogits.VARIABLE
    assert (
        text_config.return_hidden_states == ReturnHiddenStates.SELECTED_LAYERS
    )
    assert text_config.target_layer_ids == [1, 12, 23, 35, 46, 57]
    # Six taps against five draft layers is legal for DFlash Gemma4 — unlike
    # the Llama3 and Kimi pipelines, which require one tap per draft layer.
    config.validate_dflash_fields()


def test_effective_block_size_counts_the_anchor_slot() -> None:
    assert _make_unified().effective_block_size == 16


def test_unified_config_single_device_only() -> None:
    with pytest.raises(ValueError, match="single device"):
        _make_unified(target=_make_target_stub(n_devices=2))


def test_validate_rejects_taps_beyond_target_depth() -> None:
    config = _make_unified(target=_make_target_stub(num_hidden_layers=40))
    with pytest.raises(ValueError, match="target_layer_ids"):
        config.validate_dflash_fields()


def test_validate_rejects_hidden_size_mismatch() -> None:
    config = _make_unified(target=_make_target_stub(hidden_size=3840))
    with pytest.raises(ValueError, match="hidden_size"):
        config.validate_dflash_fields()


def test_validate_rejects_vocab_mismatch() -> None:
    """The draft has no head of its own, so a vocab split from the target
    would silently index the wrong embedding rows."""
    config = _make_unified(draft=_make_draft_stub(vocab_size=32000))
    with pytest.raises(ValueError, match="vocab"):
        config.validate_dflash_fields()


def _tiny_draft_config(
    devices: list[DeviceRef],
    *,
    hidden_size: int = 64,
    num_attention_heads: int = 4,
    head_dim: int = 32,
    num_hidden_layers: int = _DRAFT_LAYERS,
    sliding_window: int | None = 2048,
) -> Llama3Config:
    """Draft config at tiny dims with the checkpoint's head geometry.

    ``head_dim * num_attention_heads != hidden_size`` on purpose: that is the
    real drafter's shape (64 heads x 128 over a 5376 hidden) and the case
    that broke a hidden-size-derived RoPE.
    """
    return Llama3Config(
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=1,
        num_hidden_layers=num_hidden_layers,
        rope_theta=1_000_000.0,
        rope_scaling_params=None,
        max_seq_len=256,
        intermediate_size=128,
        interleaved_rope_weights=False,
        vocab_size=256,
        dtype=DType.bfloat16,
        model_quantization_encoding=None,
        quantization_config=None,
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=1,
            head_dim=head_dim,
            num_layers=num_hidden_layers,
            devices=devices,
            page_size=128,
        ),
        rms_norm_eps=1e-6,
        attention_multiplier=1.0,
        embedding_multiplier=1.0,
        residual_multiplier=1.0,
        devices=devices,
        clip_qkv=None,
        sliding_window=sliding_window,
    )


def test_draft_rope_spans_heads_times_head_dim(virtual_gpu: None) -> None:
    draft = DFlashLlama3(
        _tiny_draft_config([DeviceRef.GPU()]), num_context_features=6
    )
    assert draft.rope.head_dim == 32


def test_draft_layer_types_select_per_layer_masks(virtual_gpu: None) -> None:
    """The checkpoint's four sliding + one full layer must produce four
    windowed masks and one unmasked one; a uniform window would silently
    truncate the last layer's context."""
    layer_types = _raw()["layer_types"]
    draft = DFlashLlama3(
        _tiny_draft_config([DeviceRef.GPU()]),
        num_context_features=6,
        layer_types=layer_types,
    )
    variants = []
    windows = []
    for layer in draft.layers:
        attn = layer.self_attn
        assert isinstance(attn, AttentionWithRope)
        variants.append(attn.mask_variant)
        windows.append(attn.sliding_window)
    assert variants == [MHAMaskVariant.SLIDING_WINDOW_NONCAUSAL_MASK] * 4 + [
        MHAMaskVariant.NULL_MASK
    ]
    assert windows == [2048] * 4 + [None]


def test_draft_rejects_layer_types_length_mismatch(virtual_gpu: None) -> None:
    with pytest.raises(ValueError, match="one entry per draft layer"):
        DFlashLlama3(
            _tiny_draft_config([DeviceRef.GPU()]),
            num_context_features=6,
            layer_types=["full_attention"],
        )


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

    The graph-signature tests construct the actual module, whose ctor builds
    ``Gemma4TextModel``. Weight VALUES never materialize (module construction
    creates metadata only), so tiny dims keep this a CPU test.
    """
    kv_leaf = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=1,
        head_dim=32,
        num_layers=1,
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
        kv_params=MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=1,
            head_dim=32,
            num_layers=2,
            devices=devices,
            page_size=128,
        ),
        num_global_key_value_heads=1,
        global_head_dim=32,
        attention_k_eq_v=True,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=0.25
        ),
        global_rope_theta=1_000_000.0,
        sliding_window_rope_theta=10000.0,
        layer_types=["sliding_attention", "full_attention"],
        quant_config=quant_config,
    )
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        # An NVFP4 target runs with the packed-uint8 weight dtype and keeps
        # bfloat16 for everything the checkpoint leaves unquantized (norms,
        # attention, the drafter's whole tower).
        dtype=DType.uint8 if quant_config else DType.bfloat16,
        kv_params=MultiKVCacheParams.from_params(
            {"sliding_attention": kv_leaf, "full_attention": kv_leaf}
        ),
        text_config=text_config,
        vision_config=None,
        image_token_index=200,
        tie_word_embeddings=True,
    )


def _make_unified_real(
    quant_config: QuantConfig | None = None,
) -> UnifiedDflashGemma4_31BConfig:
    devices = [DeviceRef.GPU()]
    draft = _tiny_draft_config(devices, num_hidden_layers=2)
    return UnifiedDflashGemma4_31BConfig(
        target=_make_real_target(devices, quant_config),
        draft=draft,
        draft_kv_params=cast(MHAKVCacheParams, draft.kv_params),
        speculative_config=SpeculativeConfig(
            speculative_method="dflash", num_speculative_tokens=15
        ),
        target_layer_ids=[0, 1],
        layer_types=["sliding_attention", "full_attention"],
        mask_token_id=4,
        block_size=16,
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
    exactly the packed bitmask triple is appended, prefix unchanged."""
    base_types = UnifiedDflashGemma4_31B(_make_unified_real()).input_types()
    so_types = UnifiedDflashGemma4_31B(
        _make_unified_real(), enable_structured_output=True
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


@pytest.mark.parametrize("structured_output", [False, True])
def test_graph_stages_end_to_end(
    virtual_gpu: None, structured_output: bool
) -> None:
    """The whole merge -> target -> reject -> materialize -> block graph
    stages, and the drafted output is ``[batch, block_size - 1]``.

    Staging-only (no compile, no weights): this is the cheap gate on op
    wiring that would otherwise surface only after a multi-minute weight
    load on the GPU. The Gemma4-specific hops it covers are the target's
    collective embedding and tied head being driven from the draft's block
    stream.
    """
    config = _make_unified_real()
    nn_model = UnifiedDflashGemma4_31B(
        config, enable_structured_output=structured_output
    )
    # Assign the hierarchical weight names that ``load_state_dict`` would,
    # so sibling ``weight`` leaves don't collide when the graph registers
    # them.
    for name, weight in nn_model.raw_state_dict().items():
        weight.name = name

    with Graph(
        "unified_dflash_gemma4_31b_test", input_types=nn_model.input_types()
    ) as graph:
        values = nn_model._unflatten_graph_inputs(graph.inputs)
        assert (values.pinned_bitmask is not None) == structured_output
        outputs = nn_model(values)
        assert len(outputs) == 3
        next_draft_tokens = outputs[2]
        assert int(next_draft_tokens.shape[1]) == config.block_size - 1
        assert next_draft_tokens.dtype == DType.int64
        graph.output(*outputs)


def test_structured_output_rejects_synthetic_acceptance(
    virtual_gpu: None,
) -> None:
    """The synthetic acceptance path ignores token bitmasks, so combining it
    with structured output would silently stop enforcing grammars."""
    config = _make_unified_real()
    config.speculative_config = SpeculativeConfig(
        speculative_method="dflash",
        num_speculative_tokens=15,
        synthetic_acceptance_rate=0.8,
    )
    with pytest.raises(ValueError, match="synthetic"):
        UnifiedDflashGemma4_31B(config, enable_structured_output=True)


def _make_placeholder_inputs(
    *, structured_output: bool
) -> UnifiedDflashGemma4_31BInputs:
    b = Buffer.from_numpy(np.zeros(1, dtype=np.int64))
    common: dict[str, Any] = dict(
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
    )
    if structured_output:
        return UnifiedDflashGemma4_31BInputs(
            **common,
            structured_output=True,
            pinned_bitmask=b,
            wait_payload=b,
            device_bitmask_scratch=b,
        )
    return UnifiedDflashGemma4_31BInputs(**common, structured_output=False)


def test_inputs_buffer_tail_matches_graph_signature(
    virtual_gpu: None,
) -> None:
    """The host-side buffer packing (``model.py`` tail flags) must track the
    module's graph signature in both structured-output states.

    A one-sided flip of the ``_spec_decode_tail_buffers`` flags would keep
    the signature test above green and fail at execute with an input-arity
    error. ``kv_cache_inputs=None`` drops exactly the KV-tree inputs from the
    packed tuple, so parity is signature length minus the flattened KV
    inputs.
    """
    for structured_output in (False, True):
        module = UnifiedDflashGemma4_31B(
            _make_unified_real(),
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
    weights = UnifiedDflashGemma4_31B(
        _make_unified_real(_nvfp4_quant_config())
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
    assert draft_names
    assert {weights[n].dtype for n in draft_names} == {DType.bfloat16}


def test_nvfp4_target_keeps_graph_signature(virtual_gpu: None) -> None:
    """Quantizing the target must not perturb the spec-decode input contract.

    The graph inputs are activations, KV pages and sampling scalars — none of
    which the weight encoding touches — so a signature change here would mean
    the NVFP4 path diverged from the bfloat16 one somewhere it should not.
    """
    bf16_types = UnifiedDflashGemma4_31B(_make_unified_real()).input_types()
    nvfp4_types = UnifiedDflashGemma4_31B(
        _make_unified_real(_nvfp4_quant_config())
    ).input_types()
    assert _signature(nvfp4_types) == _signature(bf16_types)


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
    nn_model = UnifiedDflashGemma4_31B(_make_unified_real())
    for name, weight in nn_model.raw_state_dict().items():
        weight.name = name

    captured: list[PagedCacheValues] = []
    real_forward_block = nn_model.draft.forward_block

    def spy_forward_block(*args: Any, **kwargs: Any) -> Any:
        captured.append(kwargs["kv_collection"])
        return real_forward_block(*args, **kwargs)

    monkeypatch.setattr(nn_model.draft, "forward_block", spy_forward_block)

    with Graph(
        "unified_dflash_gemma4_31b_block_meta_test",
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
        graph.output(_block_dispatch_metadata(meta.tensor, 16))

    session = InferenceSession(devices=[CPU()])
    compiled = session.load(graph)
    (out,) = compiled.execute(
        Buffer.from_numpy(np.array([3, 129, 5, 4321], dtype=np.int64))
    )
    assert isinstance(out, Buffer)
    np.testing.assert_array_equal(out.to_numpy(), [3, 16, 0, 4321])
