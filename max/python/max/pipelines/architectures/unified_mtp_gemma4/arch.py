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

from max.graph.weights import WeightsFormat
from max.pipelines.lib import SupportedArchitecture
from max.pipelines.modeling.types import InputModality, PipelineTask

from ..gemma4.context import Gemma4Context
from ..gemma4.memory_planner import Gemma4MemoryPlanner
from ..gemma4.model_config import Gemma4ForConditionalGenerationConfig
from ..gemma4.tokenizer import Gemma4Tokenizer
from .batch_processor import UnifiedMTPGemma4BatchProcessor
from .model import UnifiedMTPGemma4Model
from .weight_adapters import convert_safetensor_state_dict

unified_mtp_gemma4_arch = SupportedArchitecture(
    name="UnifiedMTPGemma4ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "nvidia/Gemma-4-31B-IT-NVFP4",
        "google/gemma-4-31B-it",
        "nvidia/Gemma-4-26B-A4B-NVFP4",
        "google/gemma-4-26B-A4B-it",
    ],
    default_encoding="float4_e2m1fnx2",
    supported_encodings={"bfloat16", "float4_e2m1fnx2"},
    pipeline_model=UnifiedMTPGemma4Model,
    tokenizer=Gemma4Tokenizer,
    context_type=Gemma4Context,
    input_modalities={
        InputModality.TEXT,
        InputModality.IMAGE,
        InputModality.VIDEO,
    },
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={WeightsFormat.safetensors: convert_safetensor_state_dict},
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    multi_gpu_supported=True,
    config=Gemma4ForConditionalGenerationConfig,
    memory_planner=Gemma4MemoryPlanner,
    tool_parser="gemma4",
    reasoning_parser="gemma4",
    batching=UnifiedMTPGemma4BatchProcessor,
    # The vision encoder runs eagerly during prefill (outside the captured
    # graph) and produces variable-shape image embeddings, so this path does
    # not auto-enable device graph capture (mirrors the non-MTP gemma4 arch).
    supports_device_graph_capture=False,
    default_structured_output_backend="xgrammar",
)
