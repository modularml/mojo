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
from max.pipelines.context import PixelContext
from max.pipelines.diffusion.config import GENERIC_TAYLORSEER_DEFAULTS
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.config import MAXModelConfig, PipelineConfig
from max.pipelines.lib.interfaces import ArchConfig
from max.pipelines.modeling.config_enums import SupportedEncoding
from max.pipelines.modeling.types import PipelineTask
from transformers import AutoConfig
from typing_extensions import Self

from .pipeline_z_image import ZImagePipeline
from .tokenizer import ZImageTokenizer


@dataclass(kw_only=True)
class ZImageArchConfig(ArchConfig):
    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {"bfloat16"}

    pipeline_config: PipelineConfig
    quantization_encoding: SupportedEncoding | None = None

    def get_max_seq_len(self) -> int:
        return 0

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del huggingface_config, model_config
        return 0

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        model_config = model_config or pipeline_config.model
        if len(model_config.device_specs) != 1:
            raise ValueError("Z-Image is only supported on a single device")
        return cls(pipeline_config=pipeline_config)


z_image_arch = SupportedArchitecture(
    name="ZImagePipeline",
    task=PipelineTask.PIXEL_GENERATION,
    default_encoding=ZImageArchConfig.DEFAULT_ENCODING,
    supported_encodings=ZImageArchConfig.SUPPORTED_ENCODINGS,
    example_repo_ids=[
        "Tongyi-MAI/Z-Image",
        "Zyphra/Z-Image",
    ],
    pipeline_model=ZImagePipeline,  # type: ignore[arg-type]
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=ZImageTokenizer,
    config=ZImageArchConfig,
    denoising_cache_defaults=GENERIC_TAYLORSEER_DEFAULTS,
)
