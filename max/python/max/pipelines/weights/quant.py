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

"""Scaled quantization configuration parsing utilities for Hugging Face models."""

from __future__ import annotations

import fnmatch
import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any

import huggingface_hub
from max.driver import accelerator_api
from max.dtype import DType
from max.graph.quantization import QuantizationConfig
from max.graph.weights import WeightData
from max.nn.float8_scale_stacking import can_use_fused_mlp
from max.nn.quant_config import (
    InputScaleSpec,
    QuantConfig,
    QuantFormat,
    ScaleGranularity,
    ScaleOrigin,
    WeightScaleSpec,
)
from max.pipelines.modeling.config_enums import SupportedEncoding
from transformers import AutoConfig


def gptq_quant_config(
    quantization_encoding: SupportedEncoding | None,
    huggingface_config: AutoConfig,
) -> QuantizationConfig | None:
    """Builds the GPTQ ``QuantizationConfig`` for a ``"gptq"`` encoding.

    Computed on demand from the resolved ``quantization_encoding`` and the
    checkpoint's Hugging Face ``quantization_config``; returns ``None`` for any
    non-GPTQ encoding.

    Args:
        quantization_encoding: The resolved weight encoding.
        huggingface_config: The checkpoint's Hugging Face config.

    Returns:
        The GPTQ ``QuantizationConfig``, or ``None`` when the encoding is not
        GPTQ.

    Raises:
        ValueError: If the GPTQ scales are not float16.
    """
    if quantization_encoding != "gptq":
        return None

    hf_quant_config = huggingface_config.quantization_config
    # Alert users to a GPTQ scale format we don't support yet, rather than
    # running the GPTQ pipeline on it and emitting gibberish.
    if str(huggingface_config.torch_dtype) not in [
        "float16",
        "torch.float16",
    ]:
        raise ValueError(
            f"{huggingface_config.torch_dtype} scales are not supported for GPTQ-quantized models."
        )
    return QuantizationConfig(
        quant_method=hf_quant_config["quant_method"],
        bits=hf_quant_config["bits"],
        group_size=hf_quant_config["group_size"],
        desc_act=hf_quant_config["desc_act"],
        sym=hf_quant_config["sym"],
    )


def _get_num_hidden_layers(huggingface_config: AutoConfig) -> int:
    """Retrieve the number of hidden layers from the HuggingFace configuration.

    If the model is multi-modal, the number of hidden layers is taken from the text_config.
    """
    if hasattr(huggingface_config, "text_config"):
        return huggingface_config.text_config.num_hidden_layers
    else:
        return huggingface_config.num_hidden_layers


def _quantized_layers_and_embedding_dtype(
    huggingface_config: AutoConfig,
    ignored_modules: set[str],
    state_dict: Mapping[str, WeightData],
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> tuple[set[int], set[int], DType | None]:
    """Helper to determine quantized MLP/Attention layers and embedding output dtype.

    # TODO: For llama3, the layer name re-mapping is not applied to the `ignore`
    # list in quantization config, hence the two prefixes are needed here.
    """
    num_hidden_layers = _get_num_hidden_layers(huggingface_config)
    mlp_quantized_layers: set[int] = set()
    attn_quantized_layers: set[int] = set()

    for i in range(num_hidden_layers):
        # Check MLP components (gate_proj, up_proj, down_proj).
        not_converted_mlp_modules = [
            f"{ignored_modules_prefix}layers.{i}.mlp.{proj}" in ignored_modules
            for proj in ["gate_proj", "up_proj", "down_proj"]
        ]
        is_mlp_not_converted = any(not_converted_mlp_modules)
        if not is_mlp_not_converted:
            mlp_quantized_layers.add(i)
        elif not all(not_converted_mlp_modules):
            raise ValueError(
                "float8 quantization currently assumes uniform quantization for MLPs"
            )

        # Check Attention QKV components (q_proj, k_proj, v_proj, o_proj)
        not_converted_attn_qkv_modules = [
            f"{ignored_modules_prefix}layers.{i}.self_attn.{proj}"
            in ignored_modules
            for proj in ["q_proj", "k_proj", "v_proj", "o_proj"]
        ]
        is_attn_qkv_not_converted = any(not_converted_attn_qkv_modules)
        if not is_attn_qkv_not_converted:
            attn_quantized_layers.add(i)
        elif not all(not_converted_attn_qkv_modules):
            raise ValueError(
                "float8 quantization currently assumes uniform quantization for attention QKV and output projections"
            )

    # Determine embedding_output_dtype
    embedding_output_dtype: DType | None
    if f"{state_dict_name_prefix}embed_tokens.weight" in state_dict:
        # Check embed_tokens first since it's the actual embedding layer dtype
        embedding_output_dtype = state_dict[
            f"{state_dict_name_prefix}embed_tokens.weight"
        ].dtype
    elif f"{state_dict_name_prefix}lm_head.weight" in state_dict:
        # Fall back to lm_head dtype (for tied embeddings or when embed_tokens is missing)
        embedding_output_dtype = state_dict[
            f"{state_dict_name_prefix}lm_head.weight"
        ].dtype
    elif f"{state_dict_name_prefix}lm_head" in ignored_modules:
        # If `lm_head` is in ignored_modules, but its weight isn't in the
        # checkpoint, and neither are the embedding weights, consider that a
        # buggy checkpoint.
        raise ValueError("cannot determine original type from checkpoint")
    else:
        # Default to `lm_head` being quantized to float8.
        embedding_output_dtype = DType.float8_e4m3fn

    return mlp_quantized_layers, attn_quantized_layers, embedding_output_dtype


def _normalize_modelopt_ignore_pattern(pattern: str) -> str:
    """Normalize modelopt ``ignore`` globs to canonical module paths.

    VL checkpoints often list ``model.language_model.layers.*`` while the
    weight adapter strips those wrappers to ``layers.*``.
    """
    normalized = pattern.removeprefix("model.language_model.").removeprefix(
        "language_model."
    )
    if normalized.startswith("model.layers."):
        normalized = normalized.removeprefix("model.")
    # NVFP4 checkpoints map ``block_sparse_moe.*`` to ``mlp.*`` (including
    # ``shared_experts`` after #89385 fused shared-expert MoE).
    normalized = normalized.replace("block_sparse_moe", "mlp")
    return normalized


def _modelopt_ignore_patterns(
    hf_quant_config: Mapping[str, Any] | Any,
) -> list[str]:
    """Return modelopt ``ignore`` entries (glob patterns) from a HF quant config."""
    ignore = hf_quant_config.get("ignore", [])
    if not isinstance(ignore, Sequence) or isinstance(ignore, str):
        return []
    return [_normalize_modelopt_ignore_pattern(str(entry)) for entry in ignore]


def _modelopt_layer_subtree_ignored(
    layer_idx: int,
    subtree: str,
    ignore_patterns: Sequence[str],
    *,
    modules_prefix: str = "model.",
) -> bool:
    """Return whether an entire layer subtree (``mlp`` or ``self_attn``) is ignored.

    Matches modelopt-style globs such as ``model.layers.3.self_attn*`` from
    https://huggingface.co/lukealonso/GLM-5.1-NVFP4/blob/main/config.json.
    """
    layered = f"layers.{layer_idx}.{subtree}"
    prefix_path = f"{modules_prefix}{layered}"
    candidate_paths = (
        (prefix_path, f"{prefix_path}*"),
        (layered, f"{layered}*"),
    )
    for pattern in ignore_patterns:
        for module_path, wildcard_path in candidate_paths:
            if pattern in (module_path, wildcard_path):
                return True
            if fnmatch.fnmatch(module_path, pattern):
                return True
            # Only glob patterns (containing ``*``) ignore an entire subtree.
            # Exact paths such as ``layers.0.mlp.gate_up_proj`` skip one module
            # but leave the rest of the MLP quantized (e.g. ``down_proj``).
            if "*" in pattern and fnmatch.fnmatch(wildcard_path, pattern):
                return True
    return False


def _quantized_layers_from_modelopt_ignore(
    huggingface_config: AutoConfig,
    ignore_patterns: Sequence[str],
    *,
    modules_prefix: str = "model.",
) -> tuple[set[int], set[int]]:
    """Derive per-layer MLP/attention quantization from modelopt ``ignore`` globs."""
    num_hidden_layers = _get_num_hidden_layers(huggingface_config)
    mlp_quantized_layers: set[int] = set()
    attn_quantized_layers: set[int] = set()

    for layer_idx in range(num_hidden_layers):
        if not _modelopt_layer_subtree_ignored(
            layer_idx, "mlp", ignore_patterns, modules_prefix=modules_prefix
        ):
            mlp_quantized_layers.add(layer_idx)
        if not _modelopt_layer_subtree_ignored(
            layer_idx,
            "self_attn",
            ignore_patterns,
            modules_prefix=modules_prefix,
        ):
            attn_quantized_layers.add(layer_idx)

    return mlp_quantized_layers, attn_quantized_layers


def _modelopt_shared_experts_quantized_dtype(
    ignore_patterns: Sequence[str],
) -> DType | None:
    """Return quant dtype if MoE shared experts are quantized (not in ``ignore``)."""
    # modelopt leaves unquantized modules in ``ignore`` as per-layer globs like
    # ``layers.3.mlp.shared_experts*``, so any ignore entry naming shared_experts
    # means they stay bf16. A substring test sidesteps the ``model.`` prefix that
    # ``_normalize_modelopt_ignore_pattern`` strips from ``layers.*`` globs.
    for pattern in ignore_patterns:
        if "shared_experts" in pattern:
            return DType.bfloat16
    return None


def _embedding_output_dtype_from_state_dict(
    state_dict: Mapping[str, WeightData],
    ignore_patterns: Sequence[str],
    *,
    state_dict_name_prefix: str = "",
    modules_prefix: str = "model.",
) -> DType:
    """Embedding output dtype for modelopt NVFP4, respecting ``lm_head`` ignore."""
    lm_head_ignored = any(
        fnmatch.fnmatch(f"{modules_prefix}lm_head", pattern)
        or fnmatch.fnmatch(f"{modules_prefix}lm_head*", pattern)
        for pattern in ignore_patterns
    )
    if f"{state_dict_name_prefix}embed_tokens.weight" in state_dict:
        return state_dict[f"{state_dict_name_prefix}embed_tokens.weight"].dtype
    if f"{state_dict_name_prefix}lm_head.weight" in state_dict:
        return state_dict[f"{state_dict_name_prefix}lm_head.weight"].dtype
    if lm_head_ignored:
        raise ValueError("cannot determine original type from checkpoint")
    return DType.bfloat16


def _parse_compressed_tensors_fp8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> QuantConfig:
    """Parses a QuantConfig in the compressed-tensors format."""
    # This function specifically handles "compressed-tensors" style.
    # It assumes hf_quant_config and its structure are present.

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    # Verification should done by the caller.
    assert hf_quant_config and (
        hf_quant_config.get("quant_method") == "compressed-tensors"
        # compressed-tensors might have a missing quant_method if it's the default.
        or not hf_quant_config.get("quant_method")
    )

    # Extract group config.
    # Assume only one group 'group_0' matters for now.
    group_config = hf_quant_config["config_groups"]["group_0"]
    input_act_config = group_config["input_activations"]
    weight_config = group_config["weights"]

    # Parse input scaling spec.
    input_origin = (
        ScaleOrigin.DYNAMIC
        if input_act_config["dynamic"]
        else ScaleOrigin.STATIC
    )
    input_strategy_str = input_act_config["strategy"]
    if input_strategy_str == "tensor":
        input_granularity = ScaleGranularity.TENSOR
    elif input_strategy_str == "channel":
        input_granularity = ScaleGranularity.ROWWISE
    elif input_strategy_str == "token":
        input_granularity = ScaleGranularity.COLWISE
    else:
        raise ValueError(
            f"unsupported FP8 input activation strategy: {input_strategy_str}"
        )

    input_scale_name = (
        f"{state_dict_name_prefix}layers.0.mlp.down_proj.input_scale"
    )
    has_input_scale = input_scale_name in state_dict
    input_spec = InputScaleSpec(
        granularity=input_granularity,
        origin=input_origin,
        # Set reasonable defaults if the static input scale isn't present.
        dtype=state_dict[input_scale_name].dtype if has_input_scale else dtype,
        activation_scale_ub=None,
    )

    # Parse weight spec.
    weight_strategy_str = weight_config["strategy"]
    if weight_strategy_str == "tensor":
        weight_granularity = ScaleGranularity.TENSOR
    elif weight_strategy_str == "channel":
        weight_granularity = ScaleGranularity.ROWWISE
    elif weight_strategy_str == "token":
        weight_granularity = ScaleGranularity.COLWISE
    else:
        raise ValueError(
            f"unsupported FP8 weight strategy: {weight_strategy_str}"
        )

    # Validate weight config, which shouldn't dynamically quantize.
    if weight_config["dynamic"]:
        # This method uses static weight scaling according to the examples provided.
        raise ValueError(
            "dynamic weight scaling is not supported for compressed-tensors FP8 method"
        )

    weight_scale = state_dict[
        f"{state_dict_name_prefix}layers.0.mlp.down_proj.weight_scale"
    ]
    weight_spec = WeightScaleSpec(
        granularity=weight_granularity, dtype=weight_scale.dtype
    )

    # Determine which layers have MLP and QKV in float8.
    # Modules listed in `ignore` are not converted to float8.
    ignore_modules = set(hf_quant_config.get("ignore", []))
    mlp_quantized_layers, attn_quantized_layers, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config,
            ignore_modules,
            state_dict,
            state_dict_name_prefix=state_dict_name_prefix,
            ignored_modules_prefix=ignored_modules_prefix,
        )
    )

    bias_dtype = _bias_dtype(state_dict)

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=mlp_quantized_layers,
        attn_quantized_layers=attn_quantized_layers,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
        format=QuantFormat.COMPRESSED_TENSORS_FP8,
    )


def _weight_scale_dtype(state_dict: Mapping[str, WeightData]) -> DType:
    """Determines the weight scale dtype from the state dict.

    Verifies the expected weight scale quantization along the way:
    - row-wise,
    - uniform weight scale dtype.
    """
    weight_scale_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if "weight_scale" not in weight_name:
            continue

        if (
            (len(weight.shape) != 2)
            or (weight.shape[1] != 1)
            or (weight_scale_dtype and (weight.dtype != weight_scale_dtype))
        ):
            raise ValueError(
                "only row-wise weight quantization with uniform weight scale "
                "dtype is supported for FBGEMM FP8"
            )

        weight_scale_dtype = weight.dtype
    if not weight_scale_dtype:
        raise ValueError(
            "could not find weight scale dtype for FBGEMM FP8 quantized weights"
        )

    return weight_scale_dtype


def _bias_dtype(state_dict: Mapping[str, WeightData]) -> DType | None:
    """Determines the bias dtype from the state dict.

    Looks for any weight ending with `.bias` and returns its dtype.
    Assumes all bias weights have the same dtype.

    Args:
        state_dict: The state dictionary containing weights.

    Returns:
        The dtype of bias weights if found, None otherwise.
    """
    bias_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if weight_name.endswith(".bias"):
            if bias_dtype is None:
                bias_dtype = weight.dtype
            elif weight.dtype != bias_dtype:
                raise ValueError(
                    f"Inconsistent bias dtypes found: {bias_dtype} vs {weight.dtype} "
                    f"in {weight_name}"
                )
    return bias_dtype


def _parse_fbgemm_fp8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> QuantConfig:
    """Parses a QuantConfig in the FBGEMM FP8 format."""
    if dtype != DType.float8_e4m3fn:
        raise TypeError("`_parse_fbgemm_fp8_config` only supports float8 dtype")

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    assert (
        hf_quant_config and hf_quant_config.get("quant_method") == "fbgemm_fp8"
    )

    # Get the original Hugging Face module names.
    modules_to_not_convert_hf = set(
        hf_quant_config.get("modules_to_not_convert", [])
    )
    activation_scale_ub = hf_quant_config.get("activation_scale_ub")

    # For fbgemm_fp8, assume input is dynamic and column-wise.
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.COLWISE,
        origin=ScaleOrigin.DYNAMIC,
        dtype=dtype,
        activation_scale_ub=activation_scale_ub,
    )

    # For fbgemm_fp8, weight is static, row-wise.
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.ROWWISE,
        dtype=_weight_scale_dtype(state_dict),
    )

    # Determine which layers have MLP and QKV in float8.
    # Modules listed in `modules_to_not_convert` are not converted to float8.
    mlp_quantized_layers, attn_quantized_layers, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config, modules_to_not_convert_hf, state_dict
        )
    )

    bias_dtype = _bias_dtype(state_dict)

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=mlp_quantized_layers,
        attn_quantized_layers=attn_quantized_layers,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
        format=QuantFormat.FBGEMM_FP8,
    )


def _parse_tensorwise_fp8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> QuantConfig:
    """Parses a QuantConfig for tensorwise (per-tensor) FP8 models.

    These models have quant_method="fp8" and per-tensor weight scales. Supports both static and dynamic activation
    schemes.
    """
    if dtype != DType.float8_e4m3fn:
        raise TypeError(
            "`_parse_tensorwise_fp8_config` only supports float8 dtype"
        )

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    assert hf_quant_config and hf_quant_config.get("quant_method") == "fp8"

    activation_scheme = hf_quant_config.get("activation_scheme")
    if activation_scheme == "dynamic":
        input_origin = ScaleOrigin.DYNAMIC
    elif activation_scheme == "static":
        input_origin = ScaleOrigin.STATIC
    else:
        raise ValueError(
            f"Unsupported activation_scheme for tensorwise FP8: "
            f"'{activation_scheme}'. Expected 'dynamic' or 'static'."
        )

    # Determine weight scale dtype from checkpoint tensors.
    # Accept scalar () or (1,) shapes — no rowwise shape check.
    weight_scale_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if "weight_scale" not in weight_name:
            continue
        weight_scale_dtype = weight.dtype
    if not weight_scale_dtype:
        raise ValueError(
            "could not find weight scale dtype for tensorwise FP8 quantized"
            " weights"
        )

    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.TENSOR,
        origin=input_origin,
        dtype=weight_scale_dtype,
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.TENSOR,
        dtype=weight_scale_dtype,
    )

    modules_to_not_convert = set(
        hf_quant_config.get("modules_to_not_convert", [])
    )

    mlp_quantized_layers, attn_quantized_layers, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config, modules_to_not_convert, state_dict
        )
    )

    bias_dtype = _bias_dtype(state_dict)

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=mlp_quantized_layers,
        attn_quantized_layers=attn_quantized_layers,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
        format=QuantFormat.FBGEMM_FP8,
    )


def _parse_blockscaled_fp8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> QuantConfig:
    """Parses a QuantConfig when quant_method="fp8"."""
    if dtype != DType.float8_e4m3fn:
        raise TypeError(
            "`_parse_blockscaled_fp8_config` only supports float8 dtype"
        )

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)

    # The quant method is checked by the caller.
    assert hf_quant_config and hf_quant_config.get("quant_method") == "fp8"

    if "activation_scheme" not in hf_quant_config:
        raise ValueError("activation_scheme must be specified")
    if hf_quant_config["activation_scheme"] != "dynamic":
        raise ValueError("activation_scheme must be dynamic")
    if "weight_block_size" not in hf_quant_config:
        raise ValueError("weight_block_size must be specified")
    if hf_quant_config["weight_block_size"] != [128, 128]:
        raise ValueError("weight_block_size must be [128, 128]")

    weight_scale_dtype: DType | None = None
    for weight_name, weight in state_dict.items():
        if "weight_scale" not in weight_name:
            continue
        weight_scale_dtype = weight.dtype
    if not weight_scale_dtype:
        raise ValueError(
            "could not find weight scale dtype for FP8 quantized weights"
        )

    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=weight_scale_dtype,
        block_size=(1, 128),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=weight_scale_dtype,
        block_size=(128, 128),
    )

    bias_dtype = _bias_dtype(state_dict)

    # All layers use FP8 in this format (e.g. Qwen3-30B-A3B FP8, DeepSeekV3).
    # modules_to_not_convert only lists layernorms and router gate, not linear layers.
    num_hidden_layers = _get_num_hidden_layers(huggingface_config)
    all_layers = set(range(num_hidden_layers))

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=all_layers,
        attn_quantized_layers=all_layers,
        embedding_output_dtype=DType.bfloat16,
        bias_dtype=bias_dtype,
        format=QuantFormat.BLOCKSCALED_FP8,
    )


def _parse_mxfp8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> QuantConfig:
    """Parse QuantConfig for MXFP8 (quant_method="mxfp8", weight_block_size=[1,32]).

    MXFP8 stores float8_e4m3fn weights with per-row E8M0 block scales
    (one scale per 32 weight columns).  The weight adapter converts E8M0 bytes
    to float32 before loading, so we declare DType.float32 for weight scales
    here.  Activation scales are computed dynamically at runtime.
    """
    text_config = getattr(huggingface_config, "text_config", huggingface_config)
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if hf_quant_config is None:
        hf_quant_config = getattr(text_config, "quantization_config", {})

    ignored_layers: list[str] = list(hf_quant_config.get("ignored_layers", []))
    ignored_modules: set[str] = set(ignored_layers)

    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, 32),
    )
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, 32),
    )

    mlp_quantized_layers, attn_quantized_layers, embedding_output_dtype = (
        _quantized_layers_and_embedding_dtype(
            huggingface_config,
            ignored_modules,
            state_dict,
        )
    )

    bias_dtype = _bias_dtype(state_dict)

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=mlp_quantized_layers,
        attn_quantized_layers=attn_quantized_layers,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
        format=QuantFormat.MXFP8,
    )


def _parse_fp8_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> QuantConfig | None:
    """Parses QuantConfig from Hugging Face config (if exists) by dispatching to format-specific parsers.

    Dispatches to the appropriate format-specific parser based on the
    quantization method in the Hugging Face config. Returns None if the dtype is not float8_e4m3fn.
    """
    if dtype != DType.float8_e4m3fn:
        return None

    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if not hf_quant_config:
        raise ValueError(
            "expected a `quantization_config` field in Hugging Face config when "
            "the dtype is float8"
        )

    quant_method = hf_quant_config.get("quant_method")

    if quant_method == "compressed-tensors":
        return _parse_compressed_tensors_fp8_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix=state_dict_name_prefix,
            ignored_modules_prefix=ignored_modules_prefix,
        )
    elif quant_method == "fbgemm_fp8":
        return _parse_fbgemm_fp8_config(huggingface_config, state_dict, dtype)
    elif quant_method == "fp8":
        if hf_quant_config.get("weight_block_size"):
            return _parse_blockscaled_fp8_config(
                huggingface_config, state_dict, dtype
            )
        else:
            # Tensorwise scaling is the fallback.
            # Blockwise and rowwise (fbgemm_fp8) both ruled out above.
            return _parse_tensorwise_fp8_config(
                huggingface_config, state_dict, dtype
            )
    elif quant_method == "mxfp8":
        return _parse_mxfp8_config(huggingface_config, state_dict, dtype)
    elif quant_method == "modelopt":
        quant_algo = hf_quant_config.get("quant_algo")
        if quant_algo == "MIXED_PRECISION":
            return _parse_mxfp8_config(huggingface_config, state_dict, dtype)

    raise ValueError(
        "FP8 dtype specified, but an unsupported or incompatible 'quantization_config' "
        f"was found. Quant method: '{quant_method}'. "
        "Supported methods are 'compressed-tensors', 'fbgemm_fp8', and 'fp8'."
    )


_logger = logging.getLogger("max.pipelines")


_FP4_DTYPES = (DType.uint8, DType.float4_e2m1fn)


def _load_standalone_quant_config(
    huggingface_config: AutoConfig,
) -> dict[str, Any] | None:
    """Try to load quantization config from a standalone hf_quant_config.json file.

    Some models (e.g., nvidia/Llama-3.1-405B-Instruct-NVFP4) store their quantization
    config in a separate hf_quant_config.json file rather than in the main config.json.

    The standalone file has a different structure:
    {
        "producer": {"name": "modelopt", "version": "..."},
        "quantization": {"quant_algo": "NVFP4", ...}
    }

    This function maps it to the standard format:
    {"quant_method": "modelopt", "quant_algo": "NVFP4", ...}

    Returns:
        dict with quant_method and quant_algo if found, None otherwise.
    """
    model_path = getattr(huggingface_config, "_name_or_path", None)
    if not model_path:
        return None

    quant_config_filename = "hf_quant_config.json"

    try:
        # Try local path first
        if os.path.isdir(model_path):
            local_path = os.path.join(model_path, quant_config_filename)
            if os.path.exists(local_path):
                with open(local_path) as f:
                    standalone_config = json.load(f)
            else:
                return None
        else:
            # Try to download from HuggingFace
            try:
                downloaded_path = huggingface_hub.hf_hub_download(
                    repo_id=model_path,
                    filename=quant_config_filename,
                    revision=getattr(huggingface_config, "_commit_hash", None),
                    local_files_only=huggingface_hub.constants.HF_HUB_OFFLINE,
                )
                with open(downloaded_path) as f:
                    standalone_config = json.load(f)
            except Exception:
                # File doesn't exist or can't be downloaded
                return None

        # Map the standalone format to the standard format
        producer = standalone_config.get("producer", {})
        quantization = standalone_config.get("quantization", {})

        quant_method = producer.get("name")
        quant_algo = quantization.get("quant_algo")

        if quant_method and quant_algo:
            mapped: dict[str, Any] = {
                "quant_method": quant_method,
                "quant_algo": quant_algo,
            }
            if "ignore" in quantization:
                mapped["ignore"] = quantization["ignore"]
            # modelopt records per-module algorithms here for mixed-precision
            # checkpoints; without it a `MIXED_PRECISION` quant_algo is
            # uninterpretable.
            if "quantized_layers" in quantization:
                mapped["quantized_layers"] = quantization["quantized_layers"]
            return mapped
        return None

    except Exception as e:
        _logger.debug(f"Failed to load standalone quant config: {e}")
        return None


def resolve_hf_quant_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
) -> dict[str, Any] | None:
    """Resolves a checkpoint's Hugging Face quantization config.

    Tries, in order: the config's own ``quantization_config`` field, a
    standalone ``hf_quant_config.json`` beside the weights, and finally a sniff
    of the state dict for modelopt NVFP4's ``weight_scale_2`` tensors.

    Args:
        huggingface_config: The config to read ``quantization_config`` from.
            For a nested multimodal config this must be the top-level one --
            the sub-config does not carry the field.
        state_dict: The checkpoint weights, used only by the final sniff.

    Returns:
        The resolved quantization config, or ``None`` when the checkpoint
        declares no quantization.
    """
    if hf_quant_config := getattr(
        huggingface_config, "quantization_config", None
    ):
        return hf_quant_config

    standalone_config = _load_standalone_quant_config(huggingface_config)
    if standalone_config:
        return standalone_config

    if any("weight_scale_2" in name for name in state_dict):
        return {"quant_method": "modelopt", "quant_algo": "NVFP4"}

    return None


def _parse_modelopt_float4_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    *,
    quant_method_override: str | None = None,
    quant_algo_override: str | None = None,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> QuantConfig | None:
    resolved_quant_config = resolve_hf_quant_config(
        huggingface_config, state_dict
    )
    quant_method = quant_method_override
    quant_algo = quant_algo_override
    if resolved_quant_config:
        quant_method = resolved_quant_config.get("quant_method", quant_method)
        quant_algo = resolved_quant_config.get("quant_algo", quant_algo)
    if not quant_method or not quant_algo:
        return None
    if quant_algo != "NVFP4":
        # The builder below hard-codes the NVFP4 two-level scheme. Returning it
        # for another algorithm silently reinterprets the payload: a
        # MIXED_PRECISION checkpoint's FP8 bases would be read as packed FP4.
        raise ValueError(
            f"modelopt quant_algo {quant_algo!r} cannot be loaded as NVFP4. "
            "Only 'NVFP4' is uniform enough for a single QuantConfig; a "
            "mixed-precision checkpoint needs per-module handling in its "
            "architecture (see Qwen3.5's `_parse_quant_config`, or MiniMax-M3's "
            "for one built on `build_modelopt_nvfp4_config`)."
        )

    return build_modelopt_nvfp4_config(
        huggingface_config,
        state_dict,
        resolved_quant_config,
        state_dict_name_prefix=state_dict_name_prefix,
        ignored_modules_prefix=ignored_modules_prefix,
    )


def build_modelopt_nvfp4_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    resolved_quant_config: Mapping[str, Any] | None,
    *,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> QuantConfig:
    """Builds the QuantConfig for modelopt's NVFP4 two-level scheme.

    Skips the whole-checkpoint ``quant_algo == "NVFP4"`` check that a
    ``MIXED_PRECISION`` export cannot pass, so the caller must have established
    which modules are NVFP4: handing this config to one quantized some other
    way reads its payload as packed FP4 and yields garbage.

    Args:
        huggingface_config: The config the layer count is read from.
        state_dict: The checkpoint weights, read for the bias and embedding
            dtypes.
        resolved_quant_config: The resolved quantization config, from
            :func:`resolve_hf_quant_config`. Only its ``ignore`` globs are read;
            ``None`` means nothing is ignored.
        state_dict_name_prefix: Optional prefix on the ``state_dict`` keys.
        ignored_modules_prefix: Prefix the ``ignore`` globs are written against.

    Returns:
        The NVFP4 config, with the fused-kernel flags left at their defaults
        for :func:`apply_fused_kernel_flags` to set.
    """
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.STATIC,
        dtype=DType.float32,
        block_size=(1, 16),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e4m3fn,
        block_size=(1, 16),
    )

    bias_dtype = _bias_dtype(state_dict)
    ignore_patterns = (
        _modelopt_ignore_patterns(resolved_quant_config)
        if resolved_quant_config
        else []
    )
    mlp_quantized_layers, attn_quantized_layers = (
        _quantized_layers_from_modelopt_ignore(
            huggingface_config,
            ignore_patterns,
            modules_prefix=ignored_modules_prefix,
        )
    )
    embedding_output_dtype = _embedding_output_dtype_from_state_dict(
        state_dict,
        ignore_patterns,
        state_dict_name_prefix=state_dict_name_prefix,
        modules_prefix=ignored_modules_prefix,
    )

    shared_experts_weight_dtype = _modelopt_shared_experts_quantized_dtype(
        ignore_patterns
    )

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=mlp_quantized_layers,
        attn_quantized_layers=attn_quantized_layers,
        embedding_output_dtype=embedding_output_dtype,
        bias_dtype=bias_dtype,
        shared_experts_weight_dtype=shared_experts_weight_dtype,
        format=QuantFormat.NVFP4,
    )


def _is_mxfp4_config(hf_quant_config: dict[str, Any]) -> bool:
    """Checks whether a HuggingFace quantization config describes MXFP4."""
    quant_method = hf_quant_config.get("quant_method", "")
    # https://huggingface.co/openai/gpt-oss-120b/blob/main/config.json
    if quant_method.lower() == "mxfp4":
        return True
    # https://huggingface.co/amd/Kimi-K2.5-MXFP4/blob/main/config.json
    if quant_method.lower() == "quark":
        weight_cfg = hf_quant_config.get("global_quant_config", {}).get(
            "weight", {}
        )
        return (
            weight_cfg.get("scale_format") == "e8m0"
            and weight_cfg.get("dtype") == "fp4"
        )
    # compressed-tensors carries the scheme in `format` rather than in
    # `quant_method`, which it shares with FP8 and INT4 checkpoints, so the
    # format string is the only thing that distinguishes MXFP4 here.
    if quant_method.lower() == "compressed-tensors":
        return "mxfp4" in str(hf_quant_config.get("format", "")).lower()
    return False


def _is_mxfp6_config(hf_quant_config: dict[str, Any]) -> bool:
    """Checks whether a HuggingFace quantization config describes MXFP6."""
    quant_method = hf_quant_config.get("quant_method", "").lower()
    if quant_method == "mxfp6":
        return True
    # Quark-shaped config, as `amd/*-MXFP4` uses for FP4.
    if quant_method == "quark":
        weight_cfg = hf_quant_config.get("global_quant_config", {}).get(
            "weight", {}
        )
        return weight_cfg.get("scale_format") == "e8m0" and weight_cfg.get(
            "dtype"
        ) in ("fp6", "fp6_e2m3", "fp6_e3m2")
    return False


def _mxfp6_element_format(hf_quant_config: dict[str, Any]) -> str:
    """Extracts the FP6 element encoding from a quantization config.

    Both encodings occupy 6 bits, so the packed tensor shape cannot tell them
    apart -- the config is the only record of which one the bytes hold, and
    decoding with the wrong one silently produces garbage. Defaults to E2M3,
    which is what the requantizer writes unless asked otherwise.
    """
    declared = hf_quant_config.get("fp6_format")
    if declared is None:
        weight_cfg = hf_quant_config.get("global_quant_config", {}).get(
            "weight", {}
        )
        dtype = weight_cfg.get("dtype", "")
        declared = dtype.removeprefix("fp6_") if "_" in dtype else None
    if declared is None:
        return "e2m3"
    if declared.lower() not in ("e2m3", "e3m2"):
        raise ValueError(
            f"Unrecognized MXFP6 element format {declared!r}; expected "
            f"'e2m3' or 'e3m2'."
        )
    return declared.lower()


def _parse_mxfp6_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> QuantConfig:
    """Parses a QuantConfig for MXFP6 quantization.

    Structurally identical to MXFP4 -- float8_e8m0fnu scales over 32-element
    blocks, no input scale in the checkpoint because activations are quantized
    on the fly -- differing only in how many bytes the packed weight occupies
    and in which element encoding those bytes use.

    The caller must verify :func:`_is_mxfp6_config` before calling this.
    """
    hf_quant_config = getattr(huggingface_config, "quantization_config", {})

    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, 32),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, 32),
    )

    num_hidden_layers = _get_num_hidden_layers(huggingface_config)

    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=set(range(num_hidden_layers)),
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        bias_dtype=_bias_dtype(state_dict),
        format=QuantFormat.MXFP6,
        _mxfp6_element_format=_mxfp6_element_format(hf_quant_config),
        can_use_fused_mlp=can_use_fused_mlp(state_dict),
    )


def is_mxfp4_checkpoint(huggingface_config: AutoConfig) -> bool:
    """Reports whether a checkpoint's quantization config describes MXFP4.

    Unlike :func:`parse_quant_config` this reads the HuggingFace config alone,
    so a caller that must decide before the weights are loaded can still ask.
    Sizing an expert-parallel dispatch is the motivating case: the dispatch
    dtype has to be fixed while the ``EPConfig`` is built, which happens before
    the state dict exists.

    Args:
        huggingface_config: The config carrying ``quantization_config``. Pass
            the text config for multimodal checkpoints that nest it there.

    Returns:
        ``True`` if the checkpoint stores MXFP4 weights.
    """
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    # `isinstance`, not a truthiness check: the attribute is untyped, and a
    # config that carries something other than a mapping would otherwise reach
    # `_is_mxfp4_config` and fail on `.get`.
    if not isinstance(hf_quant_config, dict):
        return False
    return _is_mxfp4_config(hf_quant_config)


def _parse_mxfp4_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
) -> QuantConfig:
    """Parses a QuantConfig for MXFP4 quantization.

    MXFP4 uses float8_e8m0fnu scales (one per 32 elements) with packed uint8 weights.
    No input scale is needed; activations stay in bfloat16 and are cast to FP8 on-the-fly.

    The caller must verify :func:`_is_mxfp4_config` before calling this function.
    """
    # MXFP4: block-scaled with 32-element blocks, E8M0 scales
    input_spec = InputScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        origin=ScaleOrigin.DYNAMIC,
        dtype=DType.float32,
        block_size=(1, 32),
    )
    weight_spec = WeightScaleSpec(
        granularity=ScaleGranularity.BLOCK,
        dtype=DType.float8_e8m0fnu,
        block_size=(1, 32),
    )

    bias_dtype = _bias_dtype(state_dict)

    num_hidden_layers = _get_num_hidden_layers(huggingface_config)
    all_layers = set(range(num_hidden_layers))

    # `mlp_quantized_layers` controls which layers carry a QuantConfig
    # (StackedMoE checks it for the MXFP4 path).
    return QuantConfig(
        input_scale=input_spec,
        weight_scale=weight_spec,
        mlp_quantized_layers=all_layers,
        attn_quantized_layers=set(),
        embedding_output_dtype=DType.bfloat16,
        bias_dtype=bias_dtype,
        format=QuantFormat.MXFP4,
        can_use_fused_mlp=can_use_fused_mlp(state_dict),
    )


def _parse_float4_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> QuantConfig | None:
    # Accept both uint8 (fp4-e2m1fnX2 format) and float4_e2m1fn
    if dtype not in _FP4_DTYPES:
        return None

    quant_config = resolve_hf_quant_config(huggingface_config, state_dict)
    if not quant_config:
        return None

    quant_method = quant_config.get("quant_method")
    if quant_method == "modelopt":
        return _parse_modelopt_float4_config(
            huggingface_config,
            state_dict,
            dtype,
            quant_method_override=quant_method,
            quant_algo_override=quant_config.get("quant_algo"),
            state_dict_name_prefix=state_dict_name_prefix,
            ignored_modules_prefix=ignored_modules_prefix,
        )

    _logger.debug(
        "Skipping FP4 parsing for unsupported quant method: %s",
        quant_method,
    )
    return None


def parse_quant_config(
    huggingface_config: AutoConfig,
    state_dict: Mapping[str, WeightData],
    dtype: DType,
    state_dict_name_prefix: str = "",
    ignored_modules_prefix: str = "model.",
) -> QuantConfig | None:
    """Parses scaled quantization config from HuggingFace config and state dict.

    Args:
        huggingface_config: HuggingFace model configuration.
        state_dict: Weight state dict to inspect for scales.
        dtype: Target dtype (e.g. float8_e4m3fn or packed fp4).
        state_dict_name_prefix: Optional prefix for state dict keys.
        ignored_modules_prefix: Prefix of modules to ignore when parsing.

    Returns:
        QuantConfig if supported, otherwise None.
    """
    # Check for MXFP4 quantization (can appear with bfloat16 model dtype
    # since only MoE weights are quantized; attention/embedding stay bf16).
    hf_quant_config = getattr(huggingface_config, "quantization_config", None)
    if hf_quant_config:
        # MXFP6 is checked first: a Quark config declares both through the same
        # fields, and only the weight `dtype` tells them apart.
        if _is_mxfp6_config(hf_quant_config):
            return _parse_mxfp6_config(huggingface_config, state_dict, dtype)

        if _is_mxfp4_config(hf_quant_config):
            return _parse_mxfp4_config(huggingface_config, state_dict, dtype)

        quant_method = hf_quant_config.get("quant_method", "")
        if quant_method and quant_method not in (
            "compressed-tensors",
            "fbgemm_fp8",
            "fp8",
            "gptq",
            "modelopt",
            "mxfp4",
            "mxfp6",
            "mxfp8",
            "quark",
        ):
            raise ValueError(
                f"Unrecognized quantization config: the `quant_method` "
                f"field ({quant_method!r}) is provided but not recognized."
            )

    # uint8 is packed fp4 (float4_e2m1fnx2) in NVFP4 checkpoints.
    if dtype in _FP4_DTYPES:
        config = _parse_float4_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix=state_dict_name_prefix,
            ignored_modules_prefix=ignored_modules_prefix,
        )
    elif dtype == DType.float8_e4m3fn:
        config = _parse_fp8_config(
            huggingface_config,
            state_dict,
            dtype,
            state_dict_name_prefix,
            ignored_modules_prefix,
        )
    else:
        config = None

    if config is not None:
        apply_fused_kernel_flags(config, state_dict)

    return config


def apply_fused_kernel_flags(
    config: QuantConfig, state_dict: Mapping[str, WeightData]
) -> QuantConfig:
    """Sets the fused-kernel eligibility flags on a freshly built QuantConfig.

    The last step of :func:`parse_quant_config`, split out for callers that
    build a config through one of the format-specific builders. Skipping it
    leaves both flags off, silently dropping the fused MoE path.

    Args:
        config: The config to finish, mutated in place.
        state_dict: The checkpoint weights, read to decide MLP fusion.

    Returns:
        ``config``, for chaining.
    """
    config.can_use_fused_mlp = can_use_fused_mlp(
        state_dict,
        tensor_wise=(
            config.weight_scale.is_tensor
            and config.input_scale.is_tensor
            and not config.is_static
        ),
    )
    if not config.can_use_fused_mlp:
        _logger.warning(
            "Fused MLP is not supported for this model. "
            "This may impact performance."
        )
    # Default-on for NVFP4: the SM100 fused SwiGLU+NVFP4 grouped matmul
    # kernel folds the MoE gate/up matmul + SwiGLU + NVFP4 quant into a
    # single launch, saving one BF16 HBM round trip per MoE layer.
    # Process-time kill-switch: ``MAX_DISABLE_FUSED_SWIGLU_NVFP4=1``.
    # The kill-switch flips the QuantConfig flag so the model's
    # ``gate_up_proj`` and ``gate_up_proj_scales`` properties (which
    # gate the sigma-permutation on this flag) stay byte-equal to the
    # historical chained-kernel path.
    # NOTE: `accelerator_api()` probes the LOCAL machine, so this flag --
    # and with it the sigma-permutation applied to `gate_up_proj` -- is a
    # function of the build host, not the target device. Parsing the same
    # checkpoint on a CPU-only host yields a different weight layout. The
    # assumption is build-host == inference-target; fixing it properly
    # means threading the target device spec in here.
    # MXFP8 is cuda-only here: the fused kernel is SM100, and on AMD the
    # MoE gate/up SwiGLU is fused by `fused_silu_mx_kernel` off the
    # chained path instead (as MXFP4 already does). Gating the flag rather
    # than the call site keeps the sigma-permutation and the kernel choice
    # consistent.
    config.can_use_fused_swiglu = (
        config.is_nvfp4 or (config.is_mxfp8 and accelerator_api() == "cuda")
    ) and (os.environ.get("MAX_DISABLE_FUSED_SWIGLU_NVFP4") != "1")

    return config
