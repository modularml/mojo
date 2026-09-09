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
from max.graph.weights import WeightData, Weights
from max.pipelines.lib import PipelineConfig
from max.pipelines.lib.config.model_config import (
    _select_dtype_cast,
)
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from transformers import AutoConfig

from .model_config import MambaConfig

# Maps from Safetensor to MAX weight names.
# Note: Replacements are applied in order, so order matters for overlapping patterns
MAMBA_SAFETENSOR_MAPPING = {
    "backbone.": "",  # Removes the "backbone" prefix if present.
    "model.": "",  # Removes the "model" prefix.
    # HuggingFace uses "embeddings" (plural), our model uses "embedding" (singular)
    "embeddings.weight": "embedding.weight",
    # HuggingFace uses "norm_f" for final norm, our model uses "norm"
    "norm_f.weight": "norm.weight",
    # Note: HuggingFace Mamba uses tied embeddings (no separate lm_head.weight in safetensor)
    # Our model also uses tied embeddings via set_shared_weight(), so no mapping needed.
    # HuggingFace uses "conv1d.weight" and "conv1d.bias" (dot notation)
    # Our model uses "conv1d_weight" and "conv1d_bias" (underscore notation)
    "conv1d.weight": "conv1d_weight",
    "conv1d.bias": "conv1d_bias",
    # Mamba-specific weight name mappings
    # HuggingFace format: layers.{i}.mixer.A_log -> MAX format: layers.{i}.mixer.A_log
    # HuggingFace format: layers.{i}.mixer.D -> MAX format: layers.{i}.mixer.D
    # These should match automatically, but we handle any prefix differences here.
}


def convert_safetensor_state_dict(
    state_dict: dict[str, Weights],
    huggingface_config: AutoConfig,
    pipeline_config: PipelineConfig,
    **unused_kwargs,
) -> dict[str, WeightData]:
    """Convert safetensor state dict to MAX format.

    This function handles weight name mapping from HuggingFace Mamba format to MAX format.
    Key weight names that are handled:
    - layers.{i}.mixer.A_log: State transition matrix in log space (intermediate_size, d_state)
    - layers.{i}.mixer.D: Skip connection parameter (intermediate_size,)
    - layers.{i}.mixer.conv1d.weight: Causal 1D convolution weight (intermediate_size, conv_width)
    - layers.{i}.mixer.conv1d.bias: Causal 1D convolution bias (intermediate_size,)
    - layers.{i}.mixer.in_proj.weight: Input projection weight
    - layers.{i}.mixer.x_proj.weight: State space parameter projection weight
    - layers.{i}.mixer.dt_proj.weight: Delta projection weight
    - layers.{i}.mixer.dt_proj.bias: Delta projection bias (dt_bias)
    - layers.{i}.mixer.out_proj.weight: Output projection weight

    Args:
        state_dict: Dictionary mapping weight names to Weights objects from safetensors.
        huggingface_config: HuggingFace model configuration.
        pipeline_config: MAX pipeline configuration.
        **unused_kwargs: Additional unused keyword arguments.

    Returns:
        Dictionary mapping MAX weight names to WeightData objects.
    """
    new_state_dict: dict[str, WeightData] = {}
    # Map the weight names.
    for safetensor_name, value in state_dict.items():
        max_name = safetensor_name
        for before, after in MAMBA_SAFETENSOR_MAPPING.items():
            max_name = max_name.replace(before, after)

        weight_data = value.data()

        # Handle conv1d weight shape conversion
        # HuggingFace: [out_channels, 1, kernel_size] -> MAX: [out_channels, kernel_size]
        if "conv1d_weight" in max_name:
            shape = tuple(int(d) for d in weight_data.shape)
            if len(shape) == 3 and shape[1] == 1:
                # Squeeze the middle dimension (in_channels/groups = 1)
                # Convert via DLPack protocol to numpy
                arr = np.from_dlpack(weight_data)
                arr = arr.reshape(
                    shape[0], shape[2]
                ).copy()  # copy to ensure contiguity
                weight_data = WeightData.from_numpy(arr, max_name)

        new_state_dict[max_name] = weight_data

    # TODO(MXF-517): this should be resolved by the ArchConfig, not the adapter.
    cast_from, cast_to = _select_dtype_cast(
        pipeline_config.model, MambaConfig.DEFAULT_ENCODING
    )

    if cast_from:
        assert cast_to, (
            "Invalid configuration: applied_dtype_cast_to is not set but applied_dtype_cast_from is set. "
            "This should not happen."
        )
        for key, weight_data in new_state_dict.items():
            if weight_data.dtype == supported_encoding_dtype(cast_from):
                new_state_dict[key] = weight_data.astype(
                    supported_encoding_dtype(cast_to)
                )

    return new_state_dict
