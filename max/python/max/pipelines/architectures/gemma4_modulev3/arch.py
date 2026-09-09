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
from max.pipelines.architectures.gemma4.context import Gemma4Context
from max.pipelines.architectures.gemma4.memory_planner import (
    Gemma4MemoryPlanner,
)
from max.pipelines.architectures.gemma4.model_config import (
    Gemma4ForConditionalGenerationConfig,
)
from max.pipelines.architectures.gemma4.tokenizer import Gemma4Tokenizer
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from . import weight_adapters
from .batch_processor import Gemma4ModuleV3BatchProcessor
from .model import Gemma4Model

gemma4_modulev3_arch = SupportedArchitecture(
    name="Gemma4ForConditionalGeneration_ModuleV3",
    example_repo_ids=["google/gemma-4-31B-it"],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=Gemma4Model,
    task=PipelineTask.TEXT_GENERATION,
    tokenizer=Gemma4Tokenizer,
    context_type=Gemma4Context,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    config=Gemma4ForConditionalGenerationConfig,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    default_structured_output_backend="xgrammar",
    batching=Gemma4ModuleV3BatchProcessor,
    memory_planner=Gemma4MemoryPlanner,
    supports_overlap_scheduler=False,
    supports_device_graph_capture=False,
)

# Text-only model_type "gemma4_unified" line (e.g. google/gemma-4-12B-it):
# same model, different HF architecture string -- mirrors the graph side's
# dataclasses.replace. Served text-only: the unified
# vision embedder is not implemented.
gemma4_unified_modulev3_arch = dataclasses.replace(
    gemma4_modulev3_arch,
    name="Gemma4UnifiedForConditionalGeneration_ModuleV3",
    example_repo_ids=["google/gemma-4-12B-it"],
    input_modalities={InputModality.TEXT},
)
