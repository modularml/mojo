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

from dataclasses import dataclass
from typing import ClassVar

from max.graph.weights import WeightsFormat
from max.pipelines.context import validate_wan_max_pixel_area
from max.pipelines.diffusion.config import TaylorSeerDefaults
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig
from typing_extensions import Self

from .context import WanContext
from .tokenizer import WanTokenizer
from .wan_executor import WanExecutor


@dataclass(kw_only=True)
class WanArchConfig(ArchConfig):
    """Pipeline-level config for Wan (implements ArchConfig; no KV cache)."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float32",
        "float8_e4m3fn",
    }

    pipeline_config: PipelineConfig
    quantization_encoding: SupportedEncoding | None = None

    def get_max_seq_len(self) -> int:
        # Tokenizer padding length — matches diffusers __call__ default.
        return 512

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del huggingface_config, model_config
        return 512

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        if model_config is None:
            model_config = pipeline_config.models.get("transformer")
        if model_config is None and "main" in pipeline_config.models:
            model_config = pipeline_config.model
        if model_config is None:
            raise ValueError(
                "Wan requires a 'transformer' model component in "
                "pipeline_config.models."
            )
        if len(model_config.device_specs) != 1:
            raise ValueError("Wan is only supported on a single device")
        return cls(pipeline_config=pipeline_config)


# TaylorSeer defaults (from https://github.com/Shenyi-Z/TaylorSeer).
_WAN_TAYLORSEER_DEFAULTS = TaylorSeerDefaults(
    cache_interval=5, warmup_steps=4, max_order=1
)

wan_arch = SupportedArchitecture(
    name="WanPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding=WanArchConfig.DEFAULT_ENCODING,
    supported_encodings=WanArchConfig.SUPPORTED_ENCODINGS,
    example_repo_ids=[
        "Wan-AI/Wan2.2-T2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-T2V-14B-Diffusers",
        "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
        "yetter-ai/Wan2.2-TI2V-5B-Turbo-Diffusers",
    ],
    pipeline_model=WanExecutor,
    context_type=WanContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=WanTokenizer,
    config=WanArchConfig,
    context_validators=[validate_wan_max_pixel_area],
    denoising_cache_defaults=_WAN_TAYLORSEER_DEFAULTS,
)

wan_i2v_arch = SupportedArchitecture(
    name="WanImageToVideoPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding=WanArchConfig.DEFAULT_ENCODING,
    supported_encodings=WanArchConfig.SUPPORTED_ENCODINGS,
    example_repo_ids=[
        "Wan-AI/Wan2.2-I2V-A14B-Diffusers",
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",
    ],
    pipeline_model=WanExecutor,
    context_type=WanContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=WanTokenizer,
    config=WanArchConfig,
    context_validators=[validate_wan_max_pixel_area],
    denoising_cache_defaults=_WAN_TAYLORSEER_DEFAULTS,
)
