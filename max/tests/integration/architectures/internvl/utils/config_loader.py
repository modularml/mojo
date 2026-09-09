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

"""Configuration loader for InternVL tests."""

import json
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from internvl_impl.configuration_intern_vit import (
    InternVisionConfig as HFInternVLConfig,
)
from max.dtype import DType
from max.graph import DeviceRef
from max.graph.weights.weights import WeightData
from max.nn import ReturnLogits
from max.nn.kv_cache import MHAKVCacheParams
from max.pipelines.architectures.internvl.model_config import (
    InternVLConfig,
    VisionConfig,
)
from max.pipelines.architectures.llama3.model_config import (
    Llama3Config,
)


class ConfigNames(Enum):
    """Config name."""

    INTERNVL_2B = "internvl-2B"
    INTERNVL_8B = "internvl-8B"
    INTERNVL_38B = "internvl-38B"


class ConfigLoader:
    """Loader for HuggingFace InternVL configs."""

    def __init__(self, config_dir: Path):
        """Initialize with config directory path."""
        self.config_dir = config_dir
        self._cache: dict[str, dict[str, Any]] = {}

    def load_hf_config(self, config_name: ConfigNames) -> dict[str, Any]:
        """Load HuggingFace config from JSON file."""
        if config_name.value not in self._cache:
            config_path = self.config_dir / f"{config_name.value}.json"
            if not config_path.exists():
                raise FileNotFoundError(f"Config file not found: {config_path}")

            with open(config_path) as f:
                self._cache[config_name.value] = json.load(f)

        return self._cache[config_name.value]

    def load_hf_vision_config(
        self, config_name: ConfigNames
    ) -> HFInternVLConfig:
        """Load HuggingFace InternVLVisionConfig object from JSON file."""
        # Load the full config first
        full_config = self.load_hf_config(config_name)

        # Extract the vision_config section
        if "vision_config" not in full_config:
            raise ValueError(f"Vision config not found in {config_name} config")

        vision_config_dict = full_config["vision_config"]

        # Create InternVisionConfig from the vision config dictionary
        return HFInternVLConfig(**vision_config_dict)

    def create_vision_config(self, config_name: ConfigNames) -> VisionConfig:
        """Create MAX VisionConfig from HuggingFace config."""
        hf_config = self.load_hf_vision_config(config_name)

        vision_config = VisionConfig.initialize_from_config(hf_config)
        vision_config.finalize(
            DType.bfloat16,
            {
                "vision_model.encoder.layers.0.attn.o_proj.bias": WeightData.from_numpy(
                    np.array([0.0]),
                    "vision_model.encoder.layers.0.attn.o_proj.bias",
                )
            },
        )
        return vision_config

    def create_llm_config(self, config_name: ConfigNames) -> Llama3Config:
        """Create MAX Llama3Config from HuggingFace config."""
        hf_config = self.load_hf_config(config_name)
        llm_config = hf_config["llm_config"]

        # Create minimal KV cache params
        kv_params = MHAKVCacheParams(
            dtype=DType.bfloat16,
            n_kv_heads=llm_config["num_key_value_heads"],
            head_dim=llm_config["hidden_size"]
            // llm_config["num_attention_heads"],
            num_layers=llm_config["num_hidden_layers"],
            page_size=16,
            enable_prefix_caching=False,
            devices=[DeviceRef.GPU()],
        )

        return Llama3Config(
            num_attention_heads=llm_config["num_attention_heads"],
            num_key_value_heads=llm_config["num_key_value_heads"],
            hidden_size=llm_config["hidden_size"],
            num_hidden_layers=llm_config["num_hidden_layers"],
            intermediate_size=llm_config["intermediate_size"],
            vocab_size=llm_config["vocab_size"],
            rope_theta=llm_config.get("rope_theta", 10000.0),
            rope_scaling_params=None,
            max_seq_len=llm_config.get("max_position_embeddings", 2048),
            interleaved_rope_weights=True,
            dtype=DType.bfloat16,
            model_quantization_encoding=None,
            quantization_config=None,
            kv_params=kv_params,
            return_logits=ReturnLogits.LAST_TOKEN,
            norm_method="rms_norm",
            norm_dtype=None,
            attention_bias=True,
            rms_norm_eps=llm_config.get("rms_norm_eps", 1e-6),
            tie_word_embeddings=llm_config.get("tie_word_embeddings", False),
            stacked_mlp=False,
            stacked_qkv=False,
            attention_multiplier=1.0,
            embedding_multiplier=1.0,
            residual_multiplier=1.0,
            devices=[DeviceRef.GPU()],
            clip_qkv=None,
            quant_config=None,
        )

    def create_internvl_config(
        self, config_name: ConfigNames
    ) -> InternVLConfig:
        """Create MAX InternVLConfig from HuggingFace config."""
        hf_config = self.load_hf_config(config_name)

        vision_config = self.create_vision_config(config_name)
        llm_config = self.create_llm_config(config_name)

        return InternVLConfig(
            devices=[DeviceRef.GPU()],
            downsample_ratio=hf_config.get("downsample_ratio", 0.5),
            num_image_token=256,  # Default value
            vision_config=vision_config,
            llm_config=llm_config,
        )

    def get_model_dimensions(self, config_name: ConfigNames) -> dict[str, int]:
        """Get key model dimensions for weight generation."""
        hf_config = self.load_hf_config(config_name)
        vision_config = hf_config["vision_config"]
        llm_config = hf_config["llm_config"]

        return {
            "vision_hidden_size": vision_config["hidden_size"],
            "vision_intermediate_size": vision_config["intermediate_size"],
            "vision_num_layers": vision_config["num_hidden_layers"],
            "vision_num_heads": vision_config["num_attention_heads"],
            "vision_image_size": vision_config["image_size"],
            "vision_patch_size": vision_config["patch_size"],
            "llm_hidden_size": llm_config["hidden_size"],
            "llm_intermediate_size": llm_config["intermediate_size"],
            "llm_num_layers": llm_config["num_hidden_layers"],
            "downsample_ratio": hf_config.get("downsample_ratio", 0.5),
        }


# Global instance
_config_loader = None


def get_config_loader() -> ConfigLoader:
    """Get global config loader instance."""
    global _config_loader
    if _config_loader is None:
        config_dir = Path(__file__).parent.parent / "configs"
        _config_loader = ConfigLoader(config_dir)
    return _config_loader
