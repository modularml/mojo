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

from max.experimental.cascade.pipelines.common_textgen import (
    CommonTextGenPipeline,
)
from max.graph.weights import WeightsFormat
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from . import weight_adapters
from .batch_processor import DeepseekV3BatchProcessor
from .memory_planner import DeepseekV3MemoryPlanner
from .model import DeepseekV3Model
from .model_config import DeepseekV3Config
from .tool_parser import resolve_deepseekv3_tool_parser

deepseekV3_arch = SupportedArchitecture(
    name="DeepseekV3ForCausalLM",
    task=PipelineTask.TEXT_GENERATION,
    example_repo_ids=[
        "deepseek-ai/DeepSeek-V3",
    ],
    default_encoding=DeepseekV3Config.DEFAULT_ENCODING,
    supported_encodings=DeepseekV3Config.SUPPORTED_ENCODINGS,
    multi_gpu_supported=True,
    pipeline_model=DeepseekV3Model,
    tokenizer=TextTokenizer,
    context_type=TextContext,
    default_weights_format=WeightsFormat.safetensors,
    weight_adapters={
        WeightsFormat.safetensors: weight_adapters.convert_safetensor_state_dict,
    },
    batching=DeepseekV3BatchProcessor,
    supports_empty_batches=True,
    requires_max_batch_context_length=True,
    config=DeepseekV3Config,
    tool_parser=resolve_deepseekv3_tool_parser,
    memory_planner=DeepseekV3MemoryPlanner,
    cascade_pipeline_factory=CommonTextGenPipeline,
)
