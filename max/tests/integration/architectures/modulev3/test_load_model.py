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
"""load_model() integration tests for ModuleV3 architectures.

Models tested here use small synthetic configs and zero weights, so they
exercise module init (config parsing, weight adaptation, graph tracing)
without needing real checkpoints.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
import torch
from max.driver import Buffer, load_devices, scan_available_devices
from max.engine import InferenceSession, Model
from max.experimental import functional as F
from max.graph.weights import SafetensorWeights
from max.pipelines.architectures.gemma4_modulev3.model import Gemma4Model
from max.pipelines.architectures.gemma4_modulev3.vision_model.vision_model import (
    Gemma4VisionModel,
)
from max.pipelines.architectures.gemma4_modulev3.weight_adapters import (
    convert_safetensor_state_dict as convert_gemma4_state_dict,
)
from max.pipelines.architectures.gemma4_modulev3.weight_adapters import (
    convert_vision_state_dict_for_module,
)
from max.pipelines.architectures.llama3_modulev3.model import Llama3Model
from max.pipelines.architectures.llama3_modulev3.weight_adapters import (
    convert_safetensor_state_dict,
)
from max.pipelines.architectures.olmo_modulev3.model import OlmoModel
from max.pipelines.architectures.phi3_modulev3.model import Phi3Model
from max.pipelines.lib import MemoryPlan
from test_common.load_model_helpers import (
    assert_load_model_succeeds,
    make_pipeline_config_factory,
    make_small_llama_config,
    make_zero_weights,
)
from transformers import PretrainedConfig


@pytest.mark.parametrize(
    "model_cls,repo_id",
    [
        (Llama3Model, "meta-llama/Llama-3.1-8B-Instruct"),
        (Phi3Model, "microsoft/phi-4"),
        (OlmoModel, "allenai/OLMo-1B-hf"),
        (Llama3Model, "ibm-granite/granite-3.1-8b-instruct"),
    ],
    ids=["llama3", "phi3", "olmo", "granite"],
)
def test_load_model(model_cls: type, repo_id: str) -> None:
    hf_config = make_small_llama_config()
    weights = make_zero_weights(hf_config)
    make_pipeline_config = make_pipeline_config_factory(hf_config, repo_id)
    assert_load_model_succeeds(
        model_cls, make_pipeline_config, weights, convert_safetensor_state_dict
    )


def make_small_gemma4_config() -> PretrainedConfig:
    """Synthetic 6-layer gemma4 config: one full 5:1 sliding:full period."""
    layer_types = ["sliding_attention"] * 5 + ["full_attention"]
    text_config = PretrainedConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=6,
        # Dims are chosen so no two geometries collide: sliding kv width
        # (4*16=64) != global kv width (1*32=32), and both q widths
        # (8*16=128 sliding, 8*32=256 global) differ from hidden_size (64).
        # num_attention_heads stays a multiple of both kv head counts.
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=16,
        num_global_key_value_heads=1,
        global_head_dim=32,
        hidden_activation="gelu_pytorch_tanh",
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        attention_bias=False,
        sliding_window=128,
        final_logit_softcapping=30.0,
        attention_k_eq_v=True,
        num_kv_shared_layers=0,
        enable_moe_block=False,
        use_double_wide_mlp=False,
        num_experts=0,
        top_k_experts=0,
        moe_intermediate_size=0,
        vocab_size_per_layer_input=256,
        hidden_size_per_layer_input=0,
        layer_types=layer_types,
        rope_parameters={
            "full_attention": {
                "rope_type": "proportional",
                "partial_rotary_factor": 0.25,
                "rope_theta": 1_000_000.0,
            },
            "sliding_attention": {
                "rope_type": "default",
                "rope_theta": 10_000.0,
            },
        },
    )
    hf_config = PretrainedConfig(
        text_config=text_config,
        vision_config={},  # non-None sentinel; unified model_type skips it
        # Read by make_pipeline_config_factory to size max_length.
        max_position_embeddings=text_config.max_position_embeddings,
        image_token_id=255,
        tie_word_embeddings=True,
        architectures=["Gemma4UnifiedForConditionalGeneration"],
    )
    hf_config.model_type = "gemma4_unified"
    return hf_config


def make_small_gemma4_vision_config() -> PretrainedConfig:
    """Small gemma4 config WITH a vision tower (model_type ``gemma4``).

    ``Gemma4VisionConfig.initialize_from_config`` reads every field set here;
    ``hidden_activation`` stays in its HuggingFace spelling because that
    classmethod applies ``_HIDDEN_ACTIVATION_MAP``.
    """
    hf_config = make_small_gemma4_config()
    hf_config.model_type = "gemma4"
    hf_config.architectures = ["Gemma4ForConditionalGeneration"]
    hf_config.vision_config = PretrainedConfig(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        # head_dim // (2 * ndim) must be integral for the 2-D vision rope.
        head_dim=16,
        patch_size=4,
        position_embedding_size=16,
        pooling_kernel_size=2,
        rms_norm_eps=1e-6,
        max_position_embeddings=4096,
        attention_bias=False,
        hidden_activation="gelu_pytorch_tanh",
        standardize=True,
        rope_parameters={"rope_theta": 10_000.0},
    )
    return hf_config


def _gemma4_vision_zero_weights(
    wm: dict[str, torch.Tensor], hf_config: PretrainedConfig
) -> None:
    """Add checkpoint-shaped vision keys to ``wm``.

    Names are the inverse of ``GEMMA4_VISION_SAFETENSOR_MAP`` in
    ``gemma4/weight_adapters.py``: the tower lives under
    ``model.vision_tower.`` (stripped) and the projector under
    ``model.embed_vision.`` (kept as ``embed_vision.``).
    """
    v = hf_config.vision_config

    def z(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=torch.bfloat16)

    prefix = "model.vision_tower."
    wm[prefix + "patch_embedder.input_proj.weight"] = z(
        v.hidden_size, 3 * v.patch_size**2
    )
    wm[prefix + "patch_embedder.position_embedding_table"] = z(
        2, v.position_embedding_size, v.hidden_size
    )
    for i in range(v.num_hidden_layers):
        lp = f"{prefix}encoder.layers.{i}."
        wm[lp + "self_attn.q_proj.weight"] = z(
            v.num_attention_heads * v.head_dim, v.hidden_size
        )
        wm[lp + "self_attn.k_proj.weight"] = z(
            v.num_key_value_heads * v.head_dim, v.hidden_size
        )
        wm[lp + "self_attn.v_proj.weight"] = z(
            v.num_key_value_heads * v.head_dim, v.hidden_size
        )
        wm[lp + "self_attn.o_proj.weight"] = z(
            v.hidden_size, v.num_attention_heads * v.head_dim
        )
        wm[lp + "self_attn.q_norm.weight"] = z(v.head_dim)
        wm[lp + "self_attn.k_norm.weight"] = z(v.head_dim)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            wm[lp + norm + ".weight"] = z(v.hidden_size)
        wm[lp + "mlp.gate_proj.weight"] = z(v.intermediate_size, v.hidden_size)
        wm[lp + "mlp.up_proj.weight"] = z(v.intermediate_size, v.hidden_size)
        wm[lp + "mlp.down_proj.weight"] = z(v.hidden_size, v.intermediate_size)
    wm[prefix + "std_bias"] = z(v.hidden_size)
    wm[prefix + "std_scale"] = z(v.hidden_size)
    wm["model.embed_vision.embedding_projection.weight"] = z(
        hf_config.text_config.hidden_size, v.hidden_size
    )


def _gemma4_language_zero_weights(
    hf_config: PretrainedConfig,
) -> dict[str, torch.Tensor]:
    t = hf_config.text_config

    def z(*shape: int) -> torch.Tensor:
        return torch.zeros(*shape, dtype=torch.bfloat16)

    wm: dict[str, torch.Tensor] = {}
    prefix = "model.language_model."
    wm[prefix + "embed_tokens.weight"] = z(t.vocab_size, t.hidden_size)
    wm[prefix + "norm.weight"] = z(t.hidden_size)
    for i, layer_type in enumerate(t.layer_types):
        lp = f"{prefix}layers.{i}."
        sliding = layer_type == "sliding_attention"
        hd = t.head_dim if sliding else t.global_head_dim
        n_kv = (
            t.num_key_value_heads if sliding else t.num_global_key_value_heads
        )
        wm[lp + "self_attn.q_proj.weight"] = z(
            t.num_attention_heads * hd, t.hidden_size
        )
        wm[lp + "self_attn.k_proj.weight"] = z(n_kv * hd, t.hidden_size)
        if sliding:
            wm[lp + "self_attn.v_proj.weight"] = z(n_kv * hd, t.hidden_size)
        wm[lp + "self_attn.o_proj.weight"] = z(
            t.hidden_size, t.num_attention_heads * hd
        )
        wm[lp + "self_attn.q_norm.weight"] = z(hd)
        wm[lp + "self_attn.k_norm.weight"] = z(hd)
        for norm in (
            "input_layernorm",
            "post_attention_layernorm",
            "pre_feedforward_layernorm",
            "post_feedforward_layernorm",
        ):
            wm[lp + norm + ".weight"] = z(t.hidden_size)
        wm[lp + "mlp.gate_proj.weight"] = z(t.intermediate_size, t.hidden_size)
        wm[lp + "mlp.up_proj.weight"] = z(t.intermediate_size, t.hidden_size)
        wm[lp + "mlp.down_proj.weight"] = z(t.hidden_size, t.intermediate_size)
        wm[lp + "layer_scalar"] = z(1)
    return wm


def _as_safetensor_weights(wm: dict[str, torch.Tensor]) -> SafetensorWeights:
    """Wrap the torch tensors as MAX buffers.

    ``SafetensorWeights._st_weight_map`` is a ``dict[str, Buffer]`` and
    ``WeightData.dtype`` is taken straight off the entry, so raw torch tensors
    would leak a ``torch.dtype`` into the model config (gemma4 derives
    ``unquantized_dtype`` from the checkpoint, and the vision tower builds
    ``TensorType``s from it).
    """
    return SafetensorWeights(
        [],
        tensors=set(wm.keys()),
        tensors_to_file_idx={},
        _st_weight_map={
            name: Buffer.from_dlpack(tensor) for name, tensor in wm.items()
        },
    )


def make_gemma4_zero_weights(hf_config: PretrainedConfig) -> SafetensorWeights:
    return _as_safetensor_weights(_gemma4_language_zero_weights(hf_config))


def make_gemma4_vision_zero_weights(
    hf_config: PretrainedConfig,
) -> SafetensorWeights:
    wm = _gemma4_language_zero_weights(hf_config)
    _gemma4_vision_zero_weights(wm, hf_config)
    return _as_safetensor_weights(wm)


def test_load_model_gemma4_modulev3() -> None:
    """Text-only ``gemma4_unified`` checkpoint: the ``vision_config is None``
    path through the multi-graph base (no vision tower is compiled)."""
    hf_config = make_small_gemma4_config()
    weights = make_gemma4_zero_weights(hf_config)
    make_pipeline_config = make_pipeline_config_factory(
        hf_config, "google/gemma-4-31B-it"
    )
    assert_load_model_succeeds(
        Gemma4Model, make_pipeline_config, weights, convert_gemma4_state_dict
    )


def _build_gemma4_model(
    hf_config: PretrainedConfig, weights: SafetensorWeights
) -> Gemma4Model:
    """Construct a ``Gemma4Model`` the way ``assert_load_model_succeeds`` does.

    Returns the model so callers can assert on which towers were compiled;
    only ``InferenceSession`` is mocked, so config parsing, weight adaptation
    and module construction all run for real.
    """
    device_specs = scan_available_devices()[:1]
    pipeline_config = make_pipeline_config_factory(
        hf_config, "google/gemma-4-31B-it"
    )(device_specs)

    mock_session = MagicMock(spec=InferenceSession)
    mock_session.load.return_value = MagicMock(spec=Model, input_metadata=[])
    return Gemma4Model(
        pipeline_config=pipeline_config,
        session=mock_session,
        devices=load_devices(device_specs),
        kv_cache_config=pipeline_config.model.kv_cache,
        weights=weights,
        adapter=convert_gemma4_state_dict,
        # Mirrors assert_load_model_succeeds: the factory pins max_length
        # to the checkpoint bound, so the planned value equals what the
        # arch policies derived before plans became required.
        memory_plan=MemoryPlan(
            planned_max_batch_size=1,
            footprint=0,
            planned_max_length=pipeline_config.model.max_length,
        ),
    )


def test_load_model_gemma4_modulev3_vision() -> None:
    """Full ``gemma4`` checkpoint: vision tower + language tower both compile."""
    hf_config = make_small_gemma4_vision_config()
    model = _build_gemma4_model(
        hf_config, make_gemma4_vision_zero_weights(hf_config)
    )
    assert model.vision_model is not None
    assert model.language_model is not None


def test_gemma4_modulev3_vision_weight_keys_match_module_tree() -> None:
    """The vision converter's output keys must exactly cover the eager tower.

    ``Module.compile(weights=...)`` silently ignores unmatched entries, so
    without this the load test would pass on a misnamed checkpoint key and the
    tower would serve its zero-initialized defaults.
    """
    hf_config = make_small_gemma4_vision_config()
    weights = make_gemma4_vision_zero_weights(hf_config)
    model = _build_gemma4_model(hf_config, weights)

    with F.lazy():
        tower = Gemma4VisionModel(model.config)

    converted = set(convert_vision_state_dict_for_module(dict(weights.items())))
    assert converted == set(dict(tower.parameters))


def test_load_model_gemma4_modulev3_text_only_skips_vision() -> None:
    """A ``gemma4_unified`` checkpoint has ``vision_config is None``, so the
    multi-graph base must compile the language tower only."""
    hf_config = make_small_gemma4_config()
    model = _build_gemma4_model(hf_config, make_gemma4_zero_weights(hf_config))
    assert model.vision_model is None
    assert model.language_model is not None
