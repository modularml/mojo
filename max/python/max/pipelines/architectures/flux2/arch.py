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
from max.pipelines.modeling.types import InputModality, PipelineTask
from transformers import AutoConfig
from typing_extensions import Self

# Text sequence length for FLUX.2 pipelines.
# Note: Text embeddings sent to the denoiser must be padded to this length even
# when the given prompt(s) are shorter. The denoiser uses unmasked non-causal
# attention and the padding tokens function as "registers" or "scratch pad".
FLUX2_TEXT_SEQ_LEN: int = 512

from .flux2_executor import Flux2Executor
from .flux2_klein_executor import Flux2KleinExecutor
from .tokenizer import Flux2Tokenizer


@dataclass(kw_only=True)
class Flux2ArchConfig(ArchConfig):
    """Pipeline-level config for Flux2 (implements ArchConfig; no KV cache)."""

    DEFAULT_ENCODING: ClassVar[SupportedEncoding] = "bfloat16"
    SUPPORTED_ENCODINGS: ClassVar[set[SupportedEncoding]] = {
        "bfloat16",
        "float4_e2m1fnx2",
    }

    pipeline_config: PipelineConfig
    quantization_encoding: SupportedEncoding | None = None

    def get_max_seq_len(self) -> int:
        return FLUX2_TEXT_SEQ_LEN

    @classmethod
    def calculate_max_seq_len(
        cls,
        huggingface_config: AutoConfig,
        model_config: MAXModelConfig,
    ) -> int:
        del huggingface_config, model_config
        return FLUX2_TEXT_SEQ_LEN

    @classmethod
    def initialize(
        cls,
        pipeline_config: PipelineConfig,
        model_config: MAXModelConfig | None = None,
        *,
        max_seq_len: int,
    ) -> Self:
        return cls(pipeline_config=pipeline_config)


flux2_arch = SupportedArchitecture(
    name="Flux2Pipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=Flux2ArchConfig.DEFAULT_ENCODING,
    supported_encodings=Flux2ArchConfig.SUPPORTED_ENCODINGS,
    example_repo_ids=[
        "black-forest-labs/FLUX.2-dev",
        "black-forest-labs/FLUX.2-dev-NVFP4",
    ],
    pipeline_model=Flux2Executor,
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=Flux2Tokenizer,
    config=Flux2ArchConfig,
    denoising_cache_defaults=GENERIC_TAYLORSEER_DEFAULTS,
)

flux2_klein_arch = SupportedArchitecture(
    name="Flux2KleinPipeline",
    task=PipelineTask.PIXEL_GENERATION,
    input_modalities={InputModality.TEXT, InputModality.IMAGE},
    default_encoding=Flux2ArchConfig.DEFAULT_ENCODING,
    supported_encodings=Flux2ArchConfig.SUPPORTED_ENCODINGS,
    example_repo_ids=[
        "black-forest-labs/FLUX.2-klein-4B",
        "black-forest-labs/FLUX.2-klein-9B",
        "black-forest-labs/FLUX.2-klein-base-4B",
        "black-forest-labs/FLUX.2-klein-base-9B",
        "black-forest-labs/FLUX.2-klein-4b-nvfp4",
        "black-forest-labs/FLUX.2-klein-9b-nvfp4",
    ],
    pipeline_model=Flux2KleinExecutor,
    context_type=PixelContext,
    default_weights_format=WeightsFormat.safetensors,
    tokenizer=Flux2Tokenizer,
    config=Flux2ArchConfig,
    denoising_cache_defaults=GENERIC_TAYLORSEER_DEFAULTS,
)
