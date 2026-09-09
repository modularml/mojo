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


"""Weight adapters for GritLM safetensors."""

from __future__ import annotations

from typing import Any

from max.dtype import DType
from max.graph.weights import WeightData, Weights
from transformers import AutoConfig

GRITLM_SAFETENSOR_MAPPING: dict[str, str] = {
    "model.": "language_model.",
}

# Pooling head weights — only used for embedding mode, not CausalLM serving
_DROP_PREFIXES = ("gritlm_pooling", "pooling")


def _target_dtype(pipeline_config: Any) -> DType | None:
    try:
        from max.pipelines.modeling.config_enums import supported_encoding_dtype

        enc = pipeline_config.model.quantization_encoding
        if enc is not None:
            return supported_encoding_dtype(enc)
    except Exception:
        pass
    return None


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig | None = None,
    pipeline_config: Any = None,
    **kwargs,
) -> dict[str, WeightData]:
    """Remap GritLM safetensor weight names to MAX internal convention.

    Applies the following transformations:
        - ``model.*``        → ``language_model.*``
        - ``lm_head.*``      → ``language_model.lm_head.*``
        - ``gritlm_pooling`` → dropped (embedding path, not CausalLM)

    Args:
        state_dict: Raw GritLM checkpoint weights keyed by safetensor name.
        huggingface_config: HuggingFace model configuration (unused, kept for
            adapter interface compatibility).
        pipeline_config: Pipeline configuration used to resolve the target
            dtype from the quantization encoding.
        **kwargs: Additional keyword arguments (ignored).

    Returns:
        Transformed weight dict with MAX-internal key names and, if a
        quantization encoding is active, weights cast to the target dtype.
    """
    target_dtype = _target_dtype(pipeline_config) if pipeline_config else None

    new_state_dict: dict[str, WeightData] = {}
    for name, value in state_dict.items():
        # Drop pooling head weights
        if any(name.startswith(p) for p in _DROP_PREFIXES):
            continue

        max_name = name
        for before, after in GRITLM_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)

        # lm_head sits at top-level (no model. prefix) — add language_model. prefix
        if max_name == "lm_head.weight":
            max_name = "language_model.lm_head.weight"

        wd: WeightData = value.data()
        if target_dtype is not None and wd.dtype != target_dtype:
            wd = wd.astype(target_dtype)
        new_state_dict[max_name] = wd

    return new_state_dict
