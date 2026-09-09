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

"""Utilities for applying HuggingFace config overrides."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from transformers import AutoConfig, PretrainedConfig

# Common keys used in HuggingFace configs to specify number of hidden layers
COMMON_LAYER_KEYS = [
    # Standard keys
    "num_hidden_layers",  # Most common (BERT, RoBERTa, LLaMA, Mistral, etc.)
    "n_layer",  # GPT-2, GPT-J, GPT-NeoX
    "num_layers",  # Some T5 variants and other models
    "n_layers",  # Some custom models
    "depth",  # Vision Transformers (ViT variants)
    # Nested config keys for multimodal/composite models
    "text_config.num_hidden_layers",  # CLIP, LLaVA variants
    "vision_config.num_hidden_layers",  # CLIP, multimodal models
    "language_config.num_hidden_layers",  # Some multimodal models
    "llm_config.num_hidden_layers",  # Some custom multimodal architectures
    "encoder_config.num_hidden_layers",  # Some encoder-decoder models
    "decoder_config.num_hidden_layers",  # Some encoder-decoder models
    # Encoder-Decoder specific
    "encoder_layers",  # T5, BART, and other encoder-decoder models
    "decoder_layers",  # T5, BART, and other encoder-decoder models
    "num_encoder_layers",  # Some encoder-decoder variants
    "num_decoder_layers",  # Some encoder-decoder variants
]


def _nested_key_exists(config_dict: dict[str, Any], key: str) -> bool:
    """Check if a potentially nested key exists in a config dictionary.

    Args:
        config_dict: Dictionary to search in.
        key: Key to search for (may use dot notation for nesting).

    Returns:
        True if the key exists, False otherwise.
    """
    parts = key.split(".")
    current = config_dict
    for part in parts:
        if isinstance(current, dict) and part in current:
            current = current[part]
        else:
            return False
    return True


def _set_nested_value(obj: Any, key: str, value: Any) -> None:
    """Set a potentially nested attribute on an object.

    Args:
        obj: Object to set attribute on.
        key: Attribute name (may use dot notation for nesting).
        value: Value to set.
    """
    parts = key.split(".")
    current = obj
    for part in parts[:-1]:
        current = getattr(current, part)
    setattr(current, parts[-1], value)


def set_config_overrides(
    config: PretrainedConfig,
    overrides: dict[str, Any],
    config_type: str = "config",
) -> None:
    """Set overrides on a HuggingFace config object with validation.

    Args:
        config: The HuggingFace config object to modify.
        overrides: Dictionary of key-value pairs to override in the config.
                  Supports nested keys using dot notation (e.g., "vision_model.num_hidden_layers").
        config_type: Description of config type for error messages (e.g., "AutoConfig").

    Raises:
        ValueError: If any override keys are not valid config attributes.
    """
    config_dict = config.to_dict()

    # Validate all keys before applying any overrides
    invalid_keys = [
        k for k in overrides if not _nested_key_exists(config_dict, k)
    ]
    if invalid_keys:
        valid_lines = "\n  - ".join(sorted(config_dict.keys()))
        invalid_lines = "\n  - ".join(invalid_keys)
        raise ValueError(
            f"Invalid override key(s):\n  - {invalid_lines}\n\n"
            f"Allowed {config_type} keys that can be overridden:\n  - {valid_lines}"
        )

    # Apply all overrides
    for key, value in overrides.items():
        _set_nested_value(config, key, value)


def detect_layer_keys_in_config(config: PretrainedConfig) -> list[str]:
    """Detect which common layer keys exist in a HuggingFace config.

    Args:
        config: The HuggingFace config object to inspect.

    Returns:
        List of found layer keys that exist in the config.
    """
    config_dict = config.to_dict()

    return [
        key for key in COMMON_LAYER_KEYS if _nested_key_exists(config_dict, key)
    ]


def create_layer_overrides(
    num_hidden_layers: str | int, pipeline_name: str
) -> dict[str, Any]:
    """Create config overrides for limiting the number of hidden layers.

    Args:
        num_hidden_layers: Number of layers to use, or 'all' for no override.
        pipeline_name: Name of the pipeline being debugged.

    Returns:
        Dictionary of config overrides to apply.

    Raises:
        ValueError: If no common layer keys are found in the config.
    """
    if num_hidden_layers == "all":
        return {}

    try:
        config = AutoConfig.from_pretrained(
            pipeline_name, trust_remote_code=True
        )
    except Exception as e:
        raise ValueError(
            f"Failed to load config for {pipeline_name}: {e}"
        ) from e

    found_keys = detect_layer_keys_in_config(config)

    if not found_keys:
        script_path = Path(__file__)
        raise ValueError(
            f"No common layer keys found in config for '{pipeline_name}'.\n\n"
            f"Options:\n"
            f"  1. Pass --num-hidden-layers=all to run all layers\n"
            f"  2. Pass --hf-config-overrides='{{\"<correct_key>\":{num_hidden_layers}}}' "
            f"to manually set the layer count\n"
            f"  3. Add the correct key to COMMON_LAYER_KEYS in {script_path}\n\n"
            f"Current common layer keys: {', '.join(COMMON_LAYER_KEYS)}"
        )

    # Create overrides for all found layer keys
    return {key: int(num_hidden_layers) for key in found_keys}


@contextmanager
def apply_hf_config_override(
    hf_config_overrides: dict[str, Any],
) -> Iterator[None]:
    """Apply overrides to HuggingFace config property.

    TODO (MODELS-792): This patch is a temporary workaround to allow overriding
    the HuggingFace config. In a future version of the MAXModelConfig class,
    we should be able to edit the object directly.
    """
    from max.pipelines.lib.config.model_config import MAXModelConfig

    orig_hf_prop = MAXModelConfig.huggingface_config
    if not isinstance(orig_hf_prop, property) or orig_hf_prop.fget is None:
        raise RuntimeError(
            "Expected MAXModelConfig.huggingface_config to be a @property."
        )
    original_getter = orig_hf_prop.fget

    def _patched_getter(self: Any) -> Any:
        cfg = original_getter(self)
        set_config_overrides(cfg, hf_config_overrides, "AutoConfig")
        return cfg

    MAXModelConfig.huggingface_config = property(_patched_getter)
    try:
        yield
    finally:
        MAXModelConfig.huggingface_config = orig_hf_prop


@contextmanager
def apply_non_strict_load() -> Iterator[None]:
    """Wrap load_state_dict methods to use strict=False."""
    from max.experimental.nn import Module as ModuleV3
    from max.nn.layer import Module

    def _wrap_non_strict(original_fn: Any) -> Any:
        def _wrapped(self: Any, *args: Any, **kwargs: Any) -> Any:
            kwargs["strict"] = False
            return original_fn(self, *args, **kwargs)

        return _wrapped

    orig_max_load = Module.load_state_dict
    cast(Any, Module).load_state_dict = _wrap_non_strict(orig_max_load)

    orig_max_v3_load = getattr(ModuleV3, "load_state_dict", None)
    if orig_max_v3_load is not None:
        cast(Any, ModuleV3).load_state_dict = _wrap_non_strict(orig_max_v3_load)

    try:
        yield
    finally:
        cast(Any, Module).load_state_dict = orig_max_load
        if orig_max_v3_load is not None:
            cast(Any, ModuleV3).load_state_dict = orig_max_v3_load
