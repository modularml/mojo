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
"""Tests for Gemma4TextModel."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from conftest import (  # type: ignore[import-not-found]
    TEXT_ATTENTION_K_EQ_V,
    TEXT_FINAL_LOGIT_SOFTCAPPING,
    TEXT_GLOBAL_HEAD_DIM,
    TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
    TEXT_GLOBAL_ROPE_THETA,
    TEXT_HEAD_DIM,
    TEXT_HIDDEN_ACTIVATION,
    TEXT_HIDDEN_SIZE,
    TEXT_INTERMEDIATE_SIZE,
    TEXT_LAYER_TYPES,
    TEXT_NUM_ATTENTION_HEADS,
    TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
    TEXT_NUM_HIDDEN_LAYERS,
    TEXT_NUM_KEY_VALUE_HEADS,
    TEXT_RMS_NORM_EPS,
    TEXT_SLIDING_WINDOW,
    TEXT_SLIDING_WINDOW_ROPE_THETA,
    TEXT_TIE_WORD_EMBEDDINGS,
    TEXT_VOCAB_SIZE,
    TorchGemma4TextModel,
)
from max.driver import Accelerator, Buffer
from max.dtype import DType
from max.engine import InferenceSession
from max.graph import (
    BufferValue,
    DeviceRef,
    Graph,
    TensorType,
    TensorValue,
    ops,
)
from max.nn.comm.allreduce import Signals
from max.nn.kv_cache import MHAKVCacheParams, MultiKVCacheParams
from max.nn.transformer import ReturnHiddenStates, ReturnLogits
from max.pipelines.architectures.gemma4.batch_vision_inputs import (
    create_empty_embeddings,
    create_empty_indices,
)
from max.pipelines.architectures.gemma4.gemma4 import Gemma4TextModel
from max.pipelines.architectures.gemma4.layers.rotary_embedding import (
    ProportionalScalingParams,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
    Gemma4TextConfig,
    Gemma4VisionConfig,
)

TORCH_DTYPE = torch.bfloat16
MAX_DTYPE = DType.bfloat16


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_kv_params(
    devices: list[DeviceRef],
) -> MultiKVCacheParams:
    """Build MultiKVCacheParams matching the gemma-4-31B-it config."""
    sliding_layers = sum(
        1 for t in TEXT_LAYER_TYPES if t == "sliding_attention"
    )
    global_layers = sum(1 for t in TEXT_LAYER_TYPES if t == "full_attention")

    sliding_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
        head_dim=TEXT_GLOBAL_HEAD_DIM,
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    return MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )


def _make_text_config(
    devices: list[DeviceRef],
    num_hidden_layers: int = TEXT_NUM_HIDDEN_LAYERS,
    layer_types: list[str] | None = None,
) -> Gemma4TextConfig:
    """Build a Gemma4TextConfig matching the canonical config.

    The ``kv_params`` field on Gemma4TextConfig (inherited from Gemma3Config)
    holds a plain KVCacheParams for backward compat. The per-layer-type
    MultiKVCacheParams lives on the top-level config.
    """
    if layer_types is None:
        layer_types = TEXT_LAYER_TYPES[:num_hidden_layers]

    # Text config uses the sliding-window KVCacheParams (inherited field).
    text_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        num_layers=num_hidden_layers,
        devices=devices,
        page_size=128,
    )

    return Gemma4TextConfig(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=TEXT_NUM_ATTENTION_HEADS,
        num_key_value_heads=TEXT_NUM_KEY_VALUE_HEADS,
        head_dim=TEXT_HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=131072,
        max_seq_len=131072,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=TEXT_HEAD_DIM,
        sliding_window=TEXT_SLIDING_WINDOW,
        final_logit_softcapping=TEXT_FINAL_LOGIT_SOFTCAPPING,
        attn_logit_softcapping=None,
        rope_local_base_freq=TEXT_SLIDING_WINDOW_ROPE_THETA,
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=TEXT_NUM_GLOBAL_KEY_VALUE_HEADS,
        global_head_dim=TEXT_GLOBAL_HEAD_DIM,
        attention_k_eq_v=TEXT_ATTENTION_K_EQ_V,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
        ),
        global_rope_theta=TEXT_GLOBAL_ROPE_THETA,
        sliding_window_rope_theta=TEXT_SLIDING_WINDOW_ROPE_THETA,
        layer_types=layer_types,
    )


def _make_vision_config() -> Gemma4VisionConfig:
    """Build a minimal Gemma4VisionConfig for constructing the top-level config."""
    return Gemma4VisionConfig(
        hidden_size=1152,
        intermediate_size=4304,
        num_hidden_layers=27,
        num_attention_heads=16,
        num_key_value_heads=16,
        head_dim=72,
        hidden_activation="gelu_tanh",
        rms_norm_eps=1e-6,
        max_position_embeddings=131072,
        patch_size=16,
        position_embedding_size=10240,
        pooling_kernel_size=3,
    )


def _make_model_config(
    devices: list[DeviceRef],
    num_hidden_layers: int = TEXT_NUM_HIDDEN_LAYERS,
    layer_types: list[str] | None = None,
) -> Gemma4ForConditionalGenerationConfig:
    """Build a Gemma4ForConditionalGenerationConfig for model construction."""
    kv_params = _make_kv_params(devices)
    text_config = _make_text_config(devices, num_hidden_layers, layer_types)
    # Finalize requires return_logits to be set.
    text_config.return_logits = ReturnLogits.LAST_TOKEN
    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=_make_vision_config(),
        image_token_index=258880,
        tie_word_embeddings=TEXT_TIE_WORD_EMBEDDINGS,
    )


# ---------------------------------------------------------------------------
# Tests: State dict structure
# ---------------------------------------------------------------------------


def test_state_dict_has_embed_tokens() -> None:
    """Verify the model state dict contains embedding weights."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    embed_keys = [k for k in state_dict if k.startswith("embed_tokens.")]
    assert len(embed_keys) > 0, "Expected embed_tokens weights in state dict"


def test_state_dict_has_norm() -> None:
    """Verify the model state dict contains the final norm weight."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    assert "norm.weight" in state_dict


def test_state_dict_has_lm_head_when_untied() -> None:
    """Verify the model state dict contains lm_head weights when not tied."""
    config = _make_model_config([DeviceRef.GPU()])
    config.tie_word_embeddings = False
    # Rebuild model without tied weights.
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    lm_head_keys = [k for k in state_dict if k.startswith("lm_head.")]
    assert len(lm_head_keys) > 0, "Expected lm_head weights in state dict"


def test_state_dict_tied_embedding_omits_lm_head_weight() -> None:
    """When tie_word_embeddings=True, lm_head.weight should not appear in raw_state_dict.

    ColumnParallelLinear with tied_weight uses the embedding weight directly,
    so no separate lm_head.weight is registered.
    """
    config = _make_model_config([DeviceRef.GPU()])
    assert config.tie_word_embeddings
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    assert "embed_tokens.weight" in state_dict
    # With tied weights, lm_head.weight should NOT appear separately.
    assert "lm_head.weight" not in state_dict


def test_state_dict_layer_count() -> None:
    """Verify the correct number of layers are present in state dict."""
    num_layers = 6
    layer_types = TEXT_LAYER_TYPES[:num_layers]
    config = _make_model_config(
        [DeviceRef.GPU()],
        num_hidden_layers=num_layers,
        layer_types=layer_types,
    )
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    layer_indices = {
        int(k.split(".")[1]) for k in state_dict if k.startswith("layers.")
    }
    assert layer_indices == set(range(num_layers))


def test_state_dict_all_layers_have_layer_scalar() -> None:
    """Verify all decoder layers have a layer_scalar weight (text model)."""
    num_layers = 6
    layer_types = TEXT_LAYER_TYPES[:num_layers]
    config = _make_model_config(
        [DeviceRef.GPU()],
        num_hidden_layers=num_layers,
        layer_types=layer_types,
    )
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    for i in range(num_layers):
        key = f"layers.{i}.layer_scalar"
        assert key in state_dict, f"Missing {key}"


def test_state_dict_has_four_norms_per_layer() -> None:
    """Verify each decoder layer has all four norm weights."""
    num_layers = 6
    layer_types = TEXT_LAYER_TYPES[:num_layers]
    config = _make_model_config(
        [DeviceRef.GPU()],
        num_hidden_layers=num_layers,
        layer_types=layer_types,
    )
    model = Gemma4TextModel(config)
    state_dict = model.raw_state_dict()
    norm_names = [
        "input_layernorm",
        "post_attention_layernorm",
        "pre_feedforward_layernorm",
        "post_feedforward_layernorm",
    ]
    for i in range(num_layers):
        for norm_name in norm_names:
            key = f"layers.{i}.{norm_name}.weight"
            assert key in state_dict, f"Missing {key}"


# ---------------------------------------------------------------------------
# Tests: Layer-to-KV-index routing
# ---------------------------------------------------------------------------


def test_layer_kv_index_length() -> None:
    """_layer_kv_key should have one entry per layer."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    assert len(model._layer_kv_key) == TEXT_NUM_HIDDEN_LAYERS


@pytest.mark.parametrize(
    "layer_idx",
    [0, 1, 4],
    ids=["layer_0_sliding", "layer_1_sliding", "layer_4_sliding"],
)
def test_sliding_layers_map_to_kv_index_0(layer_idx: int) -> None:
    """Sliding attention layers should map to the sliding cache key."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    assert TEXT_LAYER_TYPES[layer_idx] == "sliding_attention"
    assert model._layer_kv_key[layer_idx] == "sliding_attention"


@pytest.mark.parametrize(
    "layer_idx",
    [5, 11, 59],
    ids=["layer_5_full", "layer_11_full", "layer_59_full"],
)
def test_full_attention_layers_map_to_kv_index_1(layer_idx: int) -> None:
    """Full attention layers should map to the full-attention cache key."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    assert TEXT_LAYER_TYPES[layer_idx] == "full_attention"
    assert model._layer_kv_key[layer_idx] == "full_attention"


def test_layer_kv_index_matches_layer_types() -> None:
    """Every _layer_kv_key entry should match its layer type."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    for i, (kv_key, layer_type) in enumerate(
        zip(model._layer_kv_key, TEXT_LAYER_TYPES, strict=True)
    ):
        assert kv_key == layer_type, (
            f"Layer {i}: expected KV key {layer_type}, got {kv_key}"
        )


def test_kv_index_counts_match_layer_type_counts() -> None:
    """The number of sliding/global KV keys should match the layer type counts."""
    config = _make_model_config([DeviceRef.GPU()])
    model = Gemma4TextModel(config)
    sliding_count = sum(
        1 for k in model._layer_kv_key if k == "sliding_attention"
    )
    global_count = sum(1 for k in model._layer_kv_key if k == "full_attention")

    expected_sliding = sum(
        1 for t in TEXT_LAYER_TYPES if t == "sliding_attention"
    )
    expected_global = sum(1 for t in TEXT_LAYER_TYPES if t == "full_attention")

    assert sliding_count == expected_sliding
    assert global_count == expected_global


# ---------------------------------------------------------------------------
# Tests: Model structure mirrors torch reference
# ---------------------------------------------------------------------------


def _torch_identity_attn_factory(**kwargs: Any) -> torch.nn.Module:
    """Factory that creates identity attention stubs."""

    class _Identity(torch.nn.Module):
        def forward(self, x: torch.Tensor, **kw: Any) -> torch.Tensor:
            return x

    return _Identity()


def test_torch_reference_layer_count_matches() -> None:
    """Verify the torch and MAX models have the same number of layers."""
    num_layers = 6
    layer_types = TEXT_LAYER_TYPES[:num_layers]
    config = _make_model_config(
        [DeviceRef.GPU()],
        num_hidden_layers=num_layers,
        layer_types=layer_types,
    )
    max_model = Gemma4TextModel(config)

    torch_model = TorchGemma4TextModel(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_hidden_layers=num_layers,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=layer_types,
        attn_factory=_torch_identity_attn_factory,
    )

    assert len(max_model.layers) == len(torch_model.layers) == num_layers


def test_embed_scale_matches_torch() -> None:
    """Verify the embedding scale matches the torch reference."""
    config = _make_model_config([DeviceRef.GPU()])
    max_model = Gemma4TextModel(config)
    expected_scale = TEXT_HIDDEN_SIZE**0.5
    assert max_model.embed_tokens.embed_scale == pytest.approx(expected_scale)


def test_state_dict_keys_structurally_match_torch() -> None:
    """Verify the MAX model has the same top-level weight key prefixes as torch.

    The MAX model adds lm_head (which lives in Gemma4ForCausalLM in torch),
    so we check the structural subset that should match.
    """
    num_layers = 2
    layer_types = TEXT_LAYER_TYPES[:num_layers]

    # Build torch model
    torch_model = TorchGemma4TextModel(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_hidden_layers=num_layers,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=layer_types,
        attn_factory=_torch_identity_attn_factory,
    )
    torch_prefixes = {k.split(".")[0] for k in torch_model.state_dict()}

    # Build MAX model
    config = _make_model_config(
        [DeviceRef.GPU()],
        num_hidden_layers=num_layers,
        layer_types=layer_types,
    )
    max_model = Gemma4TextModel(config)
    max_prefixes = {k.split(".")[0] for k in max_model.raw_state_dict()}

    # The torch model has: embed_tokens, layers, norm
    # The MAX model has those plus: lm_head
    # So torch prefixes should be a subset of MAX prefixes.
    assert torch_prefixes.issubset(max_prefixes), (
        f"Torch prefixes {torch_prefixes} not a subset of MAX prefixes "
        f"{max_prefixes}"
    )


def test_per_layer_norm_keys_match_torch() -> None:
    """Verify each layer's norm weight keys match the torch reference."""
    num_layers = 2
    layer_types = TEXT_LAYER_TYPES[:num_layers]

    torch_model = TorchGemma4TextModel(
        vocab_size=TEXT_VOCAB_SIZE,
        hidden_size=TEXT_HIDDEN_SIZE,
        num_hidden_layers=num_layers,
        intermediate_size=TEXT_INTERMEDIATE_SIZE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=layer_types,
        attn_factory=_torch_identity_attn_factory,
    )

    config = _make_model_config(
        [DeviceRef.GPU()],
        num_hidden_layers=num_layers,
        layer_types=layer_types,
    )
    max_model = Gemma4TextModel(config)

    for i in range(num_layers):
        # Extract layer-level keys (strip layer prefix)
        torch_layer_keys = {
            k.replace(f"layers.{i}.", "")
            for k in torch_model.state_dict()
            if k.startswith(f"layers.{i}.")
        }
        max_layer_keys = {
            k.replace(f"layers.{i}.", "")
            for k in max_model.raw_state_dict()
            if k.startswith(f"layers.{i}.")
        }

        # Both should have the four norm weights
        norm_keys = {
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "post_feedforward_layernorm.weight",
        }
        assert norm_keys.issubset(torch_layer_keys), (
            f"Torch layer {i} missing norm keys: {norm_keys - torch_layer_keys}"
        )
        assert norm_keys.issubset(max_layer_keys), (
            f"MAX layer {i} missing norm keys: {norm_keys - max_layer_keys}"
        )


# ---------------------------------------------------------------------------
# Tests: Execution (embed → layers → norm → lm_head)
# ---------------------------------------------------------------------------

# Reduced dimensions for execution tests.  These are intentionally smaller
# than the 31B reference to keep memory and compile time low.  All other
# config values (eps, activation, rope thetas, etc.) reuse the 31B constants
# from conftest so the test mirrors the real model structure.
_EXEC_HIDDEN = 64
_EXEC_INTERMEDIATE = 128
_EXEC_VOCAB = 256
_EXEC_NUM_LAYERS = 6  # 5 sliding + 1 full (preserves the 5:1 ratio)
_EXEC_LAYER_TYPES = [
    "sliding_attention" if (i + 1) % 6 else "full_attention"
    for i in range(_EXEC_NUM_LAYERS)
]
_EXEC_HEAD_DIM = 32
_EXEC_GLOBAL_HEAD_DIM = 64
_EXEC_N_HEADS = 2
_EXEC_N_KV_HEADS = 2
_EXEC_N_GLOBAL_KV_HEADS = 1


def _make_small_model_config(
    devices: list[DeviceRef],
) -> Gemma4ForConditionalGenerationConfig:
    """Build a small Gemma4 config suitable for execution tests.

    Uses reduced dimensions (_EXEC_*) but mirrors the 31B reference for
    all non-dimension config values (eps, activation, rope, etc.).
    """
    sliding_layers = sum(
        1 for t in _EXEC_LAYER_TYPES if t == "sliding_attention"
    )
    global_layers = sum(1 for t in _EXEC_LAYER_TYPES if t == "full_attention")

    sliding_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_EXEC_N_KV_HEADS,
        head_dim=_EXEC_HEAD_DIM,
        num_layers=sliding_layers,
        devices=devices,
        page_size=128,
    )
    global_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_EXEC_N_GLOBAL_KV_HEADS,
        head_dim=_EXEC_GLOBAL_HEAD_DIM,
        num_layers=global_layers,
        devices=devices,
        page_size=128,
    )
    kv_params = MultiKVCacheParams.from_params(
        {"sliding_attention": sliding_kv, "full_attention": global_kv}
    )

    text_kv = MHAKVCacheParams(
        dtype=DType.bfloat16,
        n_kv_heads=_EXEC_N_KV_HEADS,
        head_dim=_EXEC_HEAD_DIM,
        num_layers=_EXEC_NUM_LAYERS,
        devices=devices,
        page_size=128,
    )

    text_config = Gemma4TextConfig(
        vocab_size=_EXEC_VOCAB,
        hidden_size=_EXEC_HIDDEN,
        intermediate_size=_EXEC_INTERMEDIATE,
        num_hidden_layers=_EXEC_NUM_LAYERS,
        num_attention_heads=_EXEC_N_HEADS,
        num_key_value_heads=_EXEC_N_KV_HEADS,
        head_dim=_EXEC_HEAD_DIM,
        hidden_activation="gelu_tanh",
        max_position_embeddings=1024,
        max_seq_len=1024,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        rope_theta=-1,
        rope_scaling=None,
        attention_bias=False,
        query_pre_attn_scalar=_EXEC_HEAD_DIM,
        sliding_window=TEXT_SLIDING_WINDOW,
        final_logit_softcapping=TEXT_FINAL_LOGIT_SOFTCAPPING,
        attn_logit_softcapping=None,
        rope_local_base_freq=TEXT_SLIDING_WINDOW_ROPE_THETA,
        sliding_window_pattern=-1,
        dtype=MAX_DTYPE,
        devices=devices,
        interleaved_rope_weights=False,
        kv_params=text_kv,
        num_global_key_value_heads=_EXEC_N_GLOBAL_KV_HEADS,
        global_head_dim=_EXEC_GLOBAL_HEAD_DIM,
        attention_k_eq_v=TEXT_ATTENTION_K_EQ_V,
        global_rope_scaling=ProportionalScalingParams(
            partial_rotary_factor=TEXT_GLOBAL_PARTIAL_ROTARY_FACTOR,
        ),
        global_rope_theta=TEXT_GLOBAL_ROPE_THETA,
        sliding_window_rope_theta=TEXT_SLIDING_WINDOW_ROPE_THETA,
        layer_types=_EXEC_LAYER_TYPES,
    )
    text_config.return_logits = ReturnLogits.LAST_TOKEN

    return Gemma4ForConditionalGenerationConfig(
        devices=devices,
        dtype=MAX_DTYPE,
        kv_params=kv_params,
        text_config=text_config,
        vision_config=_make_vision_config(),
        image_token_index=0,
        tie_word_embeddings=TEXT_TIE_WORD_EMBEDDINGS,
    )


def _stub_attention_shards(model: Gemma4TextModel) -> None:
    """Replace every decoder layer's attention shards with identity functions.

    This avoids the need for real KV caches or flash-attention kernels while
    keeping the rest of the forward pass (norms, MLP, residuals, layer_scalar,
    embedding, lm_head) intact.
    """

    def _identity(
        x: TensorValue, kv_collection: Any = None, **kwargs: Any
    ) -> TensorValue:
        return x

    for layer in model.layers:
        layer.self_attn_shards = [_identity]


def _build_shared_weights(
    max_model: Gemma4TextModel,
) -> dict[str, torch.Tensor]:
    """Generate random weights for both MAX and torch models.

    Uses the MAX model's ``raw_state_dict()`` to discover all required
    weight keys (including attention weights that won't actually be used
    due to identity stubs), then generates matching random tensors.
    The non-attention weights are shared between MAX and torch.
    """
    torch.manual_seed(42)
    w: dict[str, torch.Tensor] = {}

    for key, weight_obj in max_model.raw_state_dict().items():
        shape = tuple(int(d) for d in weight_obj.shape)
        w[key] = torch.randn(shape, dtype=TORCH_DTYPE)

    # Set layer scalars to a non-trivial value.
    for key in w:
        if key.endswith(".layer_scalar"):
            w[key] = torch.tensor([2.0], dtype=TORCH_DTYPE)

    return w


@pytest.mark.parametrize(
    "seq_len",
    [1],
    ids=["single_token"],
)
def test_text_model_execution_matches_torch(seq_len: int) -> None:
    """Execute the full text model graph and compare against torch reference.

    Uses identity attention stubs so the test exercises: scaled embedding,
    4-norm decoder layers, gelu_tanh MLP, layer_scalar, final norm, and
    tied lm_head.  Both sliding and full attention layer types are present.

    Uses ``ReturnLogits.ALL`` so that logits for every token position are
    compared, not just the last token.
    """
    device = Accelerator(0)
    device_ref = DeviceRef.GPU()

    # Build MAX model with identity attention.
    config = _make_small_model_config([device_ref])
    config.text_config.return_logits = ReturnLogits.ALL
    max_model = Gemma4TextModel(config)
    _stub_attention_shards(max_model)

    # Build torch reference with identity attention.
    torch_model = TorchGemma4TextModel(
        vocab_size=_EXEC_VOCAB,
        hidden_size=_EXEC_HIDDEN,
        num_hidden_layers=_EXEC_NUM_LAYERS,
        intermediate_size=_EXEC_INTERMEDIATE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=_EXEC_LAYER_TYPES,
        attn_factory=_torch_identity_attn_factory,
    )

    shared_weights = _build_shared_weights(max_model)
    max_model.load_state_dict(shared_weights)
    torch_model.load_state_dict(shared_weights, strict=False)

    # Input tokens (valid indices into the small vocab).
    torch.manual_seed(99)
    tokens = torch.randint(0, _EXEC_VOCAB, (seq_len,), dtype=torch.int64)

    # -- Run torch reference (embed → layers → norm → lm_head) --
    # TorchGemma4TextModel.forward returns hidden states after norm.
    # Apply lm_head (tied weight = embed_tokens.weight) to get logits.
    torch_model = torch_model.to(TORCH_DTYPE)
    with torch.no_grad():
        torch_hidden = torch_model(tokens)
    embed_w = shared_weights["embed_tokens.weight"].to(TORCH_DTYPE)
    torch_logits = torch_hidden @ embed_w.T
    # Match the model's final logit softcapping (cap * tanh(logits / cap)),
    # applied in the compute dtype as DistributedLogitsPostprocessMixin does.
    cap = TEXT_FINAL_LOGIT_SOFTCAPPING
    torch_logits = (torch.tanh(torch_logits / cap) * cap).float()

    # -- Build and run MAX graph --
    signals = Signals([device_ref])
    session = InferenceSession(devices=[device])
    with Graph(
        "test_text_model_exec",
        input_types=[
            TensorType(DType.int64, [seq_len], device=device_ref),
            TensorType(DType.uint32, [2], device=device_ref),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            *signals.input_types(),
        ],
    ) as graph:
        tokens_in, row_offsets_in, return_n_logits_in, signal_buf = graph.inputs
        assert isinstance(tokens_in, TensorValue)
        assert isinstance(row_offsets_in, TensorValue)
        assert isinstance(return_n_logits_in, TensorValue)
        assert isinstance(signal_buf, BufferValue)

        results = max_model(
            tokens_in,
            signal_buffers=[signal_buf],
            sliding_kv_collections=[None],  # type: ignore[list-item]
            global_kv_collections=[None],  # type: ignore[list-item]
            return_n_logits=return_n_logits_in,
            input_row_offsets=[row_offsets_in],
            image_embeddings=[
                ops.constant(
                    create_empty_embeddings(
                        [device], _EXEC_HIDDEN, DType.bfloat16
                    )[0]
                )
            ],
            image_token_indices=[
                ops.constant(create_empty_indices([device])[0])
            ],
        )
        # ReturnLogits.ALL returns (last_logits, all_logits, offsets).
        graph.output(results[1])

    compiled = session.load(graph, weights_registry=max_model.state_dict())

    tokens_gpu = Buffer.from_dlpack(tokens).to(device)
    row_offsets = torch.tensor([0, seq_len], dtype=torch.uint32)
    row_offsets_gpu = Buffer.from_dlpack(row_offsets).to(device)
    # Unused by ReturnLogits.ALL but required as a graph input.
    return_n_logits = torch.tensor([0], dtype=torch.uint32)
    (result_buf,) = compiled.execute(
        tokens_gpu, row_offsets_gpu, return_n_logits, *signals.buffers()
    )
    assert isinstance(result_buf, Buffer)
    max_logits = torch.from_dlpack(result_buf).cpu().float()

    # Compare all token logits.
    torch.testing.assert_close(
        torch_logits,
        max_logits,
        rtol=0.02,
        atol=0.07,
    )


def test_text_model_selected_layer_hidden_states() -> None:
    """SELECTED_LAYERS returns per-layer post-block taps at target_layer_ids.

    Spec-decode drafters (DFlash/DSpark) condition on the target's hidden
    states at a fixed set of layers, fusing the per-layer taps themselves
    (see ``fuse_captured_hidden_states``). Verifies both the per-tap shape
    ``[seq, hidden]`` and, against the torch reference with per-layer
    forward hooks, that the captures come from the right layers in order.
    """
    device = Accelerator(0)
    device_ref = DeviceRef.GPU()
    seq_len = 4
    target_layer_ids = [1, 3, 5]

    config = _make_small_model_config([device_ref])
    config.text_config.return_hidden_states = ReturnHiddenStates.SELECTED_LAYERS
    config.text_config.target_layer_ids = target_layer_ids
    max_model = Gemma4TextModel(config)
    _stub_attention_shards(max_model)

    torch_model = TorchGemma4TextModel(
        vocab_size=_EXEC_VOCAB,
        hidden_size=_EXEC_HIDDEN,
        num_hidden_layers=_EXEC_NUM_LAYERS,
        intermediate_size=_EXEC_INTERMEDIATE,
        hidden_activation=TEXT_HIDDEN_ACTIVATION,
        rms_norm_eps=TEXT_RMS_NORM_EPS,
        layer_types=_EXEC_LAYER_TYPES,
        attn_factory=_torch_identity_attn_factory,
    )

    shared_weights = _build_shared_weights(max_model)
    max_model.load_state_dict(shared_weights)
    torch_model.load_state_dict(shared_weights, strict=False)

    torch.manual_seed(99)
    tokens = torch.randint(0, _EXEC_VOCAB, (seq_len,), dtype=torch.int64)

    # Capture each tapped layer's post-block output (torch layers return a
    # tuple whose first element is the hidden state).
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_id: int) -> Any:
        def _hook(module: torch.nn.Module, args: Any, output: Any) -> None:
            captured[layer_id] = output[0].detach()

        return _hook

    for layer_id in target_layer_ids:
        torch_model.layers[layer_id].register_forward_hook(_make_hook(layer_id))
    torch_model = torch_model.to(TORCH_DTYPE)
    with torch.no_grad():
        torch_model(tokens)
    torch_taps = torch.cat(
        [captured[i] for i in target_layer_ids], dim=-1
    ).float()

    signals = Signals([device_ref])
    session = InferenceSession(devices=[device])
    with Graph(
        "test_text_model_selected_layers",
        input_types=[
            TensorType(DType.int64, [seq_len], device=device_ref),
            TensorType(DType.uint32, [2], device=device_ref),
            TensorType(DType.uint32, [1], device=DeviceRef.CPU()),
            *signals.input_types(),
        ],
    ) as graph:
        tokens_in, row_offsets_in, return_n_logits_in, signal_buf = graph.inputs
        assert isinstance(tokens_in, TensorValue)
        assert isinstance(row_offsets_in, TensorValue)
        assert isinstance(return_n_logits_in, TensorValue)
        assert isinstance(signal_buf, BufferValue)

        results = max_model(
            tokens_in,
            signal_buffers=[signal_buf],
            sliding_kv_collections=[None],  # type: ignore[list-item]
            global_kv_collections=[None],  # type: ignore[list-item]
            return_n_logits=return_n_logits_in,
            input_row_offsets=[row_offsets_in],
            image_embeddings=[
                ops.constant(
                    create_empty_embeddings(
                        [device], _EXEC_HIDDEN, DType.bfloat16
                    )[0]
                )
            ],
            image_token_indices=[
                ops.constant(create_empty_indices([device])[0])
            ],
        )
        # LAST_TOKEN logits + SELECTED_LAYERS returns
        # (last_logits, *per_layer_taps), one tap per target layer.
        graph.output(*results[1:])

    compiled = session.load(graph, weights_registry=max_model.state_dict())

    tokens_gpu = Buffer.from_dlpack(tokens).to(device)
    row_offsets = torch.tensor([0, seq_len], dtype=torch.uint32)
    row_offsets_gpu = Buffer.from_dlpack(row_offsets).to(device)
    return_n_logits = torch.tensor([1], dtype=torch.uint32)
    result_bufs = compiled.execute(
        tokens_gpu, row_offsets_gpu, return_n_logits, *signals.buffers()
    )
    assert len(result_bufs) == len(target_layer_ids), (
        f"Expected {len(target_layer_ids)} taps, got {len(result_bufs)}"
    )
    tap_tensors: list[torch.Tensor] = []
    for result_buf in result_bufs:
        assert isinstance(result_buf, Buffer)
        tap = torch.from_dlpack(result_buf).cpu().float()
        assert tap.shape == (seq_len, _EXEC_HIDDEN), (
            f"Unexpected tap shape {tuple(tap.shape)}"
        )
        tap_tensors.append(tap)
    max_taps = torch.cat(tap_tensors, dim=-1)
    torch.testing.assert_close(torch_taps, max_taps, rtol=0.02, atol=0.5)
