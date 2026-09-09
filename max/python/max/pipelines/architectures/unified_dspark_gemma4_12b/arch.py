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
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..gemma4.memory_planner import Gemma4MemoryPlanner
from .batch_processor import UnifiedDSparkGemma4_12BBatchProcessor
from .model import UnifiedDSparkGemma4_12BModel
from .model_config import (
    Gemma4DSparkDraftArchConfig,
    UnifiedDSparkGemma4_12BConfig,
    gemma4_dspark_12b_width,
)
from .weight_adapters import convert_safetensor_state_dict

unified_dspark_gemma4_12b_arch = SupportedArchitecture(
    name="UnifiedDSparkGemma4_12BForCausalLM",
    example_repo_ids=[
        "google/gemma-4-12B-it",
    ],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=UnifiedDSparkGemma4_12BModel,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    weight_adapters={
        WeightsFormat.safetensors: convert_safetensor_state_dict,
    },
    task=PipelineTask.TEXT_GENERATION,
    config=UnifiedDSparkGemma4_12BConfig,
    memory_planner=Gemma4MemoryPlanner,
    supports_device_graph_capture=True,
    batching=UnifiedDSparkGemma4_12BBatchProcessor,
)

# The DSpark draft checkpoint declares architectures ["Gemma4DSparkModel"].
# It is only served as a --draft-model: the config resolution rewrites the
# (gemma4_unified target, DSpark draft) pair to the unified architecture
# above, so this registration exists for draft-side name lookup, field
# validation, and the max-sequence-length clamp. The pipeline_model /
# tokenizer / context_type placeholders are never constructed for a draft.
gemma4_dspark_draft_arch = SupportedArchitecture(
    name="Gemma4DSparkModel",
    example_repo_ids=[
        "deepseek-ai/dspark_gemma4_12b_block7",
    ],
    default_encoding="bfloat16",
    supported_encodings={"bfloat16"},
    pipeline_model=UnifiedDSparkGemma4_12BModel,
    context_type=TextContext,
    tokenizer=TextTokenizer,
    default_weights_format=WeightsFormat.safetensors,
    multi_gpu_supported=False,
    task=PipelineTask.TEXT_GENERATION,
    config=Gemma4DSparkDraftArchConfig,
    checkpoint_draft_width=gemma4_dspark_12b_width,
)
