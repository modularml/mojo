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
from max.pipelines.architectures.deepseekV3.tool_parser import (
    resolve_deepseekv3_tool_parser,
)
from max.pipelines.context import TextContext
from max.pipelines.lib import SupportedArchitecture, TextTokenizer
from max.pipelines.modeling.types import PipelineTask

from ..deepseekV3.memory_planner import DeepseekV3MemoryPlanner
from . import weight_adapters
from .batch_processor import DeepseekV3ModuleV3BatchProcessor
from .model import DeepseekV3Model
from .model_config import DeepseekV3Config

deepseekV3_modulev3_arch = SupportedArchitecture(
    name="DeepseekV3ForCausalLM_ModuleV3",
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
    requires_max_batch_context_length=True,
    config=DeepseekV3Config,
    batching=DeepseekV3ModuleV3BatchProcessor,
    tool_parser=resolve_deepseekv3_tool_parser,
    memory_planner=DeepseekV3MemoryPlanner,
)
