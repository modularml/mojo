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

"""Configuration for Mistral3 text encoder."""

from __future__ import annotations

import math
from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from pydantic import Field
from typing_extensions import Self

# Mapping from HuggingFace config keys to our config keys
_HF_KEY_MAP = {
    "max_position_embeddings": "max_seq_len",
}


class Mistral3TextEncoderConfig(MAXModelConfigBase):
    """Configuration for Mistral3 text encoder."""

    hidden_size: int = 5120
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    num_hidden_layers: int = 40
    head_dim: int = 128
    vocab_size: int = 131072
    intermediate_size: int = 14336
    rope_theta: float = 1000000.0
    max_seq_len: int = 32768
    rms_norm_eps: float = 1e-5
    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)
    hidden_state_layers: list[int] = Field(default_factory=lambda: [10, 20, 30])
    output_seq_len: int | None = None
    """When set, the forward output is left-padded with zeros to this many
    tokens.  Required when the downstream transformer expects a static
    text sequence length (e.g. FLUX.2 trained the joint-attention with
    a padded ``512``-token text stream)."""

    @property
    def attention_multiplier(self) -> float:
        """Compute attention scale factor."""
        return math.sqrt(1.0 / self.head_dim)

    @classmethod
    def initialize_from_config(
        cls,
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Self:
        """Initialize configuration from a dictionary.

        Args:
            config_dict: Configuration dictionary (may contain text_config nested).
            encoding: Encoding configuration for dtype.
            devices: List of devices.

        Returns:
            Initialized Mistral3 text encoder configuration.
        """
        text_config = config_dict.get("text_config", config_dict)

        init_dict = {}
        for key, value in text_config.items():
            mapped_key = _HF_KEY_MAP.get(key, key)
            if mapped_key in cls.model_fields:
                init_dict[mapped_key] = value

        # transformers v5 moved `rope_theta` from top-level into a nested
        # `rope_parameters` dict. So if rope_theta isn't set by the above, scan
        # for it in the sub-dict.
        if "rope_theta" not in init_dict:
            rope_params = text_config.get("rope_parameters")
            if rope_params and "rope_theta" in rope_params:
                init_dict["rope_theta"] = rope_params["rope_theta"]

        assert "rope_theta" in init_dict, (
            "Failed to parse rope_theta from text_config"
        )

        # Compute head_dim if not provided
        if "head_dim" not in init_dict:
            hidden_size = init_dict.get("hidden_size", 5120)
            num_attention_heads = init_dict.get("num_attention_heads", 32)
            init_dict["head_dim"] = hidden_size // num_attention_heads

        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )

        return cls(**init_dict)
