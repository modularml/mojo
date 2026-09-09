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

import numpy as np
from max.dtype import DType
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import MAXModelConfig, PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_dtype_cast,
    _select_quantization_encoding,
)
from max.pipelines.modeling.config_enums import (
    supported_encoding_dtype,
    supported_encoding_quantization,
)
from transformers import LlamaConfig

from .model_config import Llama3Config

# Maps from Safetensor to MAX weight names.
LLAMA_SAFETENSOR_MAPPING = {
    "model.": "",  # Removes the "model" prefix.
    "g_idx": "perm_idx",  # Specific to Llama GPT-Q weights.
}


def _convert_safetensor_with_model_config(
    state_dict: dict[str, Weights],
    huggingface_config: LlamaConfig,
    model_config: MAXModelConfig,
) -> dict[str, WeightData]:
    """Helper function to convert safetensor state dict using a specific model config.

    This allows the same conversion logic to be used for both target and draft models.
    """
    # TODO(MXF-517): this should be resolved by the ArchConfig, not the adapter.
    resolved_encoding = _select_quantization_encoding(
        model_config, Llama3Config.DEFAULT_ENCODING
    )
    cast_from, cast_to = _select_dtype_cast(
        model_config, Llama3Config.DEFAULT_ENCODING
    )
    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in LLAMA_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()
    if resolved_encoding == "gptq":
        # hack: argsort the perm_idx array
        for key, weight_data in new_state_dict.items():
            np_array = np.from_dlpack(weight_data.data)
            if key.endswith("perm_idx"):
                new_state_dict[key] = WeightData.from_numpy(
                    np.argsort(np_array).astype(np.int32), key
                )
    if resolved_encoding == "gptq":
        for key, weight_data in new_state_dict.items():
            # TODO(E2EOPT-243): gptq models actually have a dtype of float16
            # not bfloat16. Sadly, MMA does not support float16 currently, so
            # we must use bfloat16 for now.
            # That said, leave scale and bias in float16 (apparently that is
            # needed for correctness). The rest must be converted to bfloat16.
            if weight_data.dtype == DType.float16 and not (
                key.endswith(("bias", "scales"))
            ):
                new_state_dict[key] = weight_data.astype(DType.bfloat16)

    if cast_from:
        assert cast_to, (
            "Invalid configuration: cast_to is not set but cast_from is set. "
            "This should not happen."
        )
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == supported_encoding_dtype(cast_from):
                new_state_dict[key] = weight_data.astype(
                    supported_encoding_dtype(cast_to)
                )

    # The GPTQ algorithm only use a subset of its keys based on the specific
    # configuration, while the unused keys remain present in the state dict
    # but are filled with dummy values for compatibility reasons. We have
    # `strict=True`, so we need to remove those unused keys when we're running
    # a GPTQ Llama model.
    if hasattr(huggingface_config, "quantization_config"):
        UNUSED_KEYS = [".bias", ".qzeros"]
        if huggingface_config.quantization_config.get("desc_act") is True:
            UNUSED_KEYS.extend(["v_proj.perm_idx", "k_proj.perm_idx"])
        else:
            UNUSED_KEYS.append("perm_idx")
        keys_to_remove = [
            key
            for key in new_state_dict
            if any(key.endswith(suffix) for suffix in UNUSED_KEYS)
        ]
        for key in keys_to_remove:
            new_state_dict.pop(key, None)

    return new_state_dict


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: LlamaConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    return _convert_safetensor_with_model_config(
        state_dict, huggingface_config, pipeline_config.model
    )


# Maps from GGUF to MAX weight names.
LLAMA_GGUF_UNFUSED_QKV_MAPPING = {
    "token_embd": "embed_tokens",
    "blk": "layers",
    "ffn_up": "mlp.up_proj",
    "ffn_down": "mlp.down_proj",
    "ffn_gate": "mlp.gate_proj",
    "ffn_norm": "post_attention_layernorm",
    "attn_norm": "input_layernorm",
    "attn_q": "self_attn.q_proj",
    "attn_v": "self_attn.v_proj",
    "attn_k": "self_attn.k_proj",
    "attn_output": "self_attn.o_proj",
    "output.weight": "lm_head.weight",
    "output_norm": "norm",
}

LLAMA_GGUF_QUANTIZED_MAPPING = LLAMA_GGUF_UNFUSED_QKV_MAPPING


def convert_gguf_state_dict(
    state_dict: dict[str, Weights],
    pipeline_config: PipelineConfig | None = None,
    **unused_kwargs,
) -> dict[str, WeightData]:
    gguf_mapping = LLAMA_GGUF_UNFUSED_QKV_MAPPING
    if pipeline_config is not None:
        # TODO(MXF-517): this should be resolved by the ArchConfig, not the adapter.
        resolved_encoding = _select_quantization_encoding(
            pipeline_config.model, Llama3Config.DEFAULT_ENCODING
        )
        if supported_encoding_quantization(resolved_encoding) is not None:
            gguf_mapping = LLAMA_GGUF_QUANTIZED_MAPPING

    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for gguf_name, value in state_dict.items():
        max_name = gguf_name
        for before, after in gguf_mapping.items():
            max_name = max_name.replace(before, after)
        new_state_dict[max_name] = value.data()

    # GGUF bakes `rope_freqs` into the weights file, because `Llama.cpp` expects
    # to just load and use it as a precomputed tensor. On the other hand, in
    # the HuggingFace ecosystem, rotary positional encodings are not parameters;
    # they are deterministic and computed from the config at runtime. So, we
    # need to remove `rope_freqs.weight` from the state dict.
    new_state_dict.pop("rope_freqs.weight", None)

    return new_state_dict
