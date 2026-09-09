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

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import max.tests.integration.tools.hf_config_overrides as hf_overrides
import pytest
import transformers
from max.experimental.nn import Module as ModuleV3
from max.nn.layer import Module
from max.pipelines.lib.config.model_config import MAXModelConfig


def test_set_config_overrides() -> None:
    """Test that set_config_overrides works for config objects."""
    config = transformers.AutoConfig.for_model("gpt2")
    original_n_head = config.n_head

    hf_overrides.set_config_overrides(
        config, {"n_embd": 2048, "n_layer": 24}, "config"
    )

    assert config.n_embd == 2048
    assert config.n_layer == 24
    assert config.n_head == original_n_head  # unchanged

    config2 = transformers.AutoConfig.for_model("gpt2")
    with pytest.raises(ValueError, match=r"Invalid override key"):
        hf_overrides.set_config_overrides(
            config2, {"invalid_key": 999}, "config"
        )


def _make_max_model_config_with_hf_config(
    model_path: str, hf_config: Any
) -> MAXModelConfig:
    """Construct a MAXModelConfig with a pre-seeded HuggingFace config.

    The custom __init__ that accepts _huggingface_config is hidden from type
    checkers (guarded by ``if not TYPE_CHECKING``), so we set the private
    attribute directly after construction to keep mypy happy.
    """
    # ``__init__`` eagerly builds the HuggingFace repo handles; keep that
    # offline for the placeholder repo (CI runs under HF_HUB_OFFLINE).
    with (
        patch("max.pipelines.lib.config.model_config.validate_hf_repo_access"),
        patch("max.pipelines.weights.hf_utils.validate_hf_repo_access"),
        patch(
            "max.pipelines.weights.hf_utils.generate_local_model_path",
            side_effect=lambda repo_id, revision=None: f"/fake/cache/{repo_id}",
        ),
    ):
        cfg = MAXModelConfig(model_path=model_path)
    cfg._huggingface_config = hf_config
    return cfg


def test_apply_hf_config_override() -> None:
    """Test apply_hf_config_override context manager."""
    base_cfg = transformers.AutoConfig.for_model("gpt2")
    orig_prop = MAXModelConfig.huggingface_config

    real_cfg = _make_max_model_config_with_hf_config("dummy/text", base_cfg)

    original_n_layer = base_cfg.n_layer

    with hf_overrides.apply_hf_config_override({"n_embd": 2048}):
        assert MAXModelConfig.huggingface_config is not orig_prop
        cfg_in_ctx = real_cfg.huggingface_config
        assert cfg_in_ctx is not None
        assert cfg_in_ctx.n_embd == 2048
        assert cfg_in_ctx.n_layer == original_n_layer

    assert MAXModelConfig.huggingface_config is orig_prop

    fresh_cfg = transformers.AutoConfig.for_model("gpt2")
    real_cfg2 = _make_max_model_config_with_hf_config("dummy/text", fresh_cfg)
    cfg_after = real_cfg2.huggingface_config
    assert cfg_after is not None
    assert cfg_after.n_embd == fresh_cfg.n_embd

    real_cfg3 = _make_max_model_config_with_hf_config(
        "dummy/text", transformers.AutoConfig.for_model("gpt2")
    )
    with pytest.raises(ValueError, match=r"Invalid override key"):
        with hf_overrides.apply_hf_config_override({"not_a_key": 1}):
            _ = real_cfg3.huggingface_config


def test_apply_non_strict_load() -> None:
    """Test apply_non_strict_load context manager."""
    orig_module_load = Module.load_state_dict
    orig_module_v3_load = getattr(ModuleV3, "load_state_dict", None)

    strict_values: list[bool | None] = []

    def mock_load_state_dict(self: Any, *args: Any, **kwargs: Any) -> Any:
        strict_values.append(kwargs.get("strict"))
        return None

    with patch.object(Module, "load_state_dict", mock_load_state_dict):
        with hf_overrides.apply_non_strict_load():
            module_mock = Mock(spec=Module)
            Module.load_state_dict(module_mock, {})
            assert strict_values[-1] is False

            Module.load_state_dict(module_mock, {}, strict=True)
            assert strict_values[-1] is False

        assert Module.load_state_dict == mock_load_state_dict

    assert Module.load_state_dict == orig_module_load
    if orig_module_v3_load is not None:
        assert ModuleV3.load_state_dict == orig_module_v3_load


def test_create_layer_overrides_with_all() -> None:
    """Test that create_layer_overrides returns empty dict when 'all' is passed."""
    result = hf_overrides.create_layer_overrides("all", "gpt2")
    assert result == {}


def test_create_layer_overrides_with_standard_key() -> None:
    """Test create_layer_overrides with a model that has standard layer key."""
    # gpt2 has n_layer as its layer count key
    mock_config = Mock()
    mock_config.to_dict.return_value = {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    }

    with patch(
        "transformers.AutoConfig.from_pretrained", return_value=mock_config
    ):
        result = hf_overrides.create_layer_overrides("3", "gpt2")
        assert result == {"n_layer": 3}
        assert isinstance(result["n_layer"], int)


def test_create_layer_overrides_with_integer() -> None:
    """Test create_layer_overrides accepts integer input."""
    mock_config = Mock()
    mock_config.to_dict.return_value = {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    }

    with patch(
        "transformers.AutoConfig.from_pretrained", return_value=mock_config
    ):
        result = hf_overrides.create_layer_overrides(5, "gpt2")
        assert result == {"n_layer": 5}


def test_create_layer_overrides_no_matching_keys() -> None:
    """Test that create_layer_overrides raises error when no layer keys found."""
    # Mock a config with no matching layer keys
    mock_config = Mock()
    mock_config.to_dict.return_value = {"hidden_size": 768, "vocab_size": 50257}

    with patch(
        "transformers.AutoConfig.from_pretrained", return_value=mock_config
    ):
        with pytest.raises(ValueError, match="No common layer keys found"):
            hf_overrides.create_layer_overrides("1", "fake-model")


def test_detect_layer_keys_in_config() -> None:
    """Test detecting layer keys in a config."""
    mock_config = Mock()
    mock_config.to_dict.return_value = {
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
    }

    found = hf_overrides.detect_layer_keys_in_config(mock_config)

    # gpt2 should have n_layer
    assert "n_layer" in found
    assert isinstance(found, list)


def test_detect_layer_keys_in_config_none_found() -> None:
    """Test when no layer keys are found in config."""
    mock_config = Mock()
    mock_config.to_dict.return_value = {"hidden_size": 768, "vocab_size": 50257}

    found = hf_overrides.detect_layer_keys_in_config(mock_config)
    assert found == []
