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
"""Registration for the DiffusionGemma block-diffusion architecture.

Reuses the Gemma 4 tokenizer, request context, memory planner, and
tool/reasoning parsers. The block-diffusion generation loop is supplied via
``pipeline_cls`` because it cannot be expressed as one-token-per-step
decoding. Input modality is text-only for now: the vision tower is built and
weighted but the image batching path is not yet validated for this model.
"""

from max.graph.weights import WeightsFormat
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.gemma4.memory_planner import (
    Gemma4MemoryPlanner,
)
from max.pipelines.architectures.gemma4.tokenizer import Gemma4Tokenizer
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.lib.pipeline_variants.block_diffusion_text_generation import (
    BlockDiffusionTextGenerationPipeline,
)
from max.pipelines.modeling.types import InputModality, PipelineTask

from . import weight_adapters
from .model import DiffusionGemmaForBlockDiffusionModel
from .model_config import DiffusionGemmaForBlockDiffusionConfig

diffusion_gemma_arch = SupportedArchitecture(
    name="DiffusionGemmaForBlockDiffusion",
    example_repo_ids=[
        "nvidia/diffusiongemma-26B-A4B-it-NVFP4",
        "google/diffusiongemma-26B-A4B-it",
    ],
    default_encoding=DiffusionGemmaForBlockDiffusionConfig.DEFAULT_ENCODING,
    supported_encodings=DiffusionGemmaForBlockDiffusionConfig.SUPPORTED_ENCODINGS,
    pipeline_model=DiffusionGemmaForBlockDiffusionModel,
    pipeline_cls=BlockDiffusionTextGenerationPipeline,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=Gemma4Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    input_modalities={InputModality.TEXT},
    context_type=Gemma4Context,
    config=DiffusionGemmaForBlockDiffusionConfig,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    default_structured_output_backend="xgrammar",
    memory_planner=Gemma4MemoryPlanner,
    # The decoder writes canvas K/V into uncommitted cache slots and reads
    # them back within a step; prefix-cache block reuse across requests is
    # not yet audited against that pattern.
    required_arguments={"enable_prefix_caching": False},
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
)
