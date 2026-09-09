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
"""Configuration for the Ideogram 4 flow-matching transformer (DiT)."""

from typing import Any

from max.driver import Device
from max.dtype import DType
from max.graph import DeviceRef
from max.pipelines.lib import MAXModelConfigBase, SupportedEncoding
from max.pipelines.modeling.config_enums import supported_encoding_dtype
from pydantic import Field
from typing_extensions import Self

# Qwen3-VL layers whose hidden states are concatenated and fed to the
# transformer as conditioning. Must match the reference
# (ideogram4.constants.QWEN3_VL_ACTIVATION_LAYERS).
QWEN3_VL_ACTIVATION_LAYERS: tuple[int, ...] = (
    0,
    3,
    6,
    9,
    12,
    15,
    18,
    21,
    24,
    27,
    30,
    33,
    35,
)

# Per-token role indicators (ideogram4.constants).
OUTPUT_IMAGE_INDICATOR = 2
LLM_TOKEN_INDICATOR = 3
# Image grid coordinates start at this offset so they never collide with text
# token positions in the shared MRoPE space.
IMAGE_POSITION_OFFSET = 65536


class Ideogram4Config(MAXModelConfigBase):
    """Architecture parameters for ``Ideogram4Transformer2DModel``.

    Defaults mirror ``ideogram-ai/ideogram-4-fp8`` ``transformer/config.json``
    and the reference ``ideogram4.modeling_ideogram4.Ideogram4Config``.
    """

    # Transformer backbone.
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adaln_dim: int = 512
    norm_eps: float = 1e-5

    # Latent dim after 2x2 patchification: ae_channels (32) * patch**2 (4) = 128.
    in_channels: int = 128

    # Qwen3-VL hidden size (4096) * number of extracted layers (13) = 53248.
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)

    # 3D MRoPE.
    rope_theta: float = 5_000_000.0
    mrope_section: tuple[int, ...] = (24, 20, 20)

    dtype: DType = DType.bfloat16
    device: DeviceRef = Field(default_factory=DeviceRef.GPU)

    @property
    def head_dim(self) -> int:
        return self.emb_dim // self.num_heads

    @classmethod
    def initialize_from_config(
        cls,
        config_dict: dict[str, Any],
        encoding: SupportedEncoding,
        devices: list[Device],
    ) -> Self:
        # Map the HuggingFace transformer/config.json field names onto our
        # canonical names, tolerating either spelling.
        alias = {
            "num_attention_heads": "num_heads",
        }
        init_dict: dict[str, Any] = {}
        for key, value in config_dict.items():
            target = alias.get(key, key)
            if target in cls.model_fields:
                init_dict[target] = value

        # transformer/config.json carries num_attention_heads + attention_head_dim
        # rather than emb_dim; derive the model width from their product.
        num_heads = config_dict.get(
            "num_attention_heads", init_dict.get("num_heads")
        )
        head_dim = config_dict.get("attention_head_dim")
        if num_heads is not None and head_dim is not None:
            init_dict["emb_dim"] = int(num_heads) * int(head_dim)

        if "mrope_section" in init_dict and isinstance(
            init_dict["mrope_section"], list
        ):
            init_dict["mrope_section"] = tuple(init_dict["mrope_section"])

        init_dict.update(
            {
                "dtype": supported_encoding_dtype(encoding),
                "device": DeviceRef.from_device(devices[0]),
            }
        )
        return cls(**init_dict)
