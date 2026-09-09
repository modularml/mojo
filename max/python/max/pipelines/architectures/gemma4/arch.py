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

import dataclasses

from max.graph.weights import WeightsFormat
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from .batch_processor import Gemma4BatchProcessor
from .context import Gemma4Context
from .memory_planner import Gemma4MemoryPlanner
from .model import Gemma3_MultiModalModel
from .model_config import Gemma4ForConditionalGenerationConfig
from .tokenizer import Gemma4Tokenizer

example_repo_ids = [
    # it = Instruction tuned (recommended).
    # pt = Pre-trained.
    "google/gemma-4-31B-it",
    # "google/gemma-4-26B-A4B-it"
    "nvidia/Gemma-4-31B-IT-NVFP4",
    # "nvidia/Gemma-4-26B-A4B-NVFP4"
]

gemma4_arch = SupportedArchitecture(
    name="Gemma4ForConditionalGeneration",
    example_repo_ids=example_repo_ids,
    default_encoding=Gemma4ForConditionalGenerationConfig.DEFAULT_ENCODING,
    supported_encodings=Gemma4ForConditionalGenerationConfig.SUPPORTED_ENCODINGS,
    pipeline_model=Gemma3_MultiModalModel,
    batching=Gemma4BatchProcessor,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=Gemma4Tokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=True,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    context_type=Gemma4Context,
    config=Gemma4ForConditionalGenerationConfig,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    default_structured_output_backend="xgrammar",
    memory_planner=Gemma4MemoryPlanner,
    supports_device_graph_capture=False,
)


# The public "unified" checkpoints (google/gemma-4-12B-it and derived quants,
# model_type "gemma4_unified") ship the regular Gemma 4 layout with no bundled
# MTP draft weights, so they are served by this architecture under their own
# architectures[0] name.
gemma4_unified_arch = dataclasses.replace(
    gemma4_arch,
    name="Gemma4UnifiedForConditionalGeneration",
    example_repo_ids=["google/gemma-4-12B-it"],
    # Served text-only: the unified vision embedder is not implemented.
    input_modalities={InputModality.TEXT},
    supports_device_graph_capture=True,
)
